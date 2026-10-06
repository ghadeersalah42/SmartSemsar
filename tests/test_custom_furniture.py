import base64
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import trimesh

from backend.schema.plan import load_plan
from backend.schema.staging import load_catalog
from backend import config
from backend.services import colab_service
from backend.services.colab_service import ColabError, generate_model, health
from backend.services.custom_furniture import (add_furniture_from_photo, add_furniture_model,
                                               fit_model, read_glb, with_custom_item)
from backend.services.furniture_placer import stage_plan
from backend.services.staging_validator import validate_staging
from backend.services.walkthrough import build_walkthrough

KEY = "secret-key"


def raw_model(path, extents=(0.5, 0.2, 1.4), offset=(3.0, -1.0, 2.0)):
    """Stand-in for a generated model: unit-less, off-centre, long side along z (facing sideways)."""
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(offset)
    box.visual.vertex_colors = [180, 40, 40, 255]
    box.export(path)
    return path


# ---------- a fake Colab server (same contract as cv_service/colab_server.ipynb) ----------
class FakeColab(BaseHTTPRequestHandler):
    glb = b""
    seen = {}

    def log_message(self, *args):
        pass

    def _reply(self, code, body, kind="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("X-API-Key") != KEY:
            return self._reply(401, json.dumps({"detail": "Wrong or missing X-API-Key."}).encode())
        self._reply(200, json.dumps({"status": "ok", "model": "TripoSR", "gpu": "fake"}).encode())

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.headers.get("X-API-Key") != KEY:
            return self._reply(401, json.dumps({"detail": "Wrong or missing X-API-Key."}).encode())
        FakeColab.seen = {"path": self.path, "type": self.headers["Content-Type"], "body": body}
        if b"not-furniture" in body:
            return self._reply(500, json.dumps({"detail": "UnidentifiedImageError: cannot identify image"}).encode())
        self._reply(200, FakeColab.glb, "model/gltf-binary")


@pytest.fixture
def colab(tmp_path, monkeypatch):
    FakeColab.glb = raw_model(tmp_path / "generated.glb").read_bytes()
    server = HTTPServer(("127.0.0.1", 0), FakeColab)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv(colab_service.URL_ENV, url)
    monkeypatch.setenv(colab_service.KEY_ENV, KEY)
    yield url
    server.shutdown()


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "my sofa.jpg"
    path.write_bytes(b"\xff\xd8\xff fake jpeg bytes")
    return path


# ---------- client ----------
def test_health(colab):
    assert health()["model"] == "TripoSR"


def test_generate_sends_the_photo_and_saves_the_glb(colab, photo, tmp_path):
    out = generate_model(photo, tmp_path / "out" / "raw.glb", mc_resolution=128)
    assert out.read_bytes() == FakeColab.glb
    seen = FakeColab.seen
    assert seen["path"] == "/generate" and seen["type"].startswith("multipart/form-data; boundary=")
    assert b'name="image"; filename="my sofa.jpg"' in seen["body"] and photo.read_bytes() in seen["body"]
    assert b'name="mc_resolution"\r\n\r\n128' in seen["body"]


def test_wrong_key_is_a_clear_error(colab, photo, tmp_path):
    with pytest.raises(ColabError, match="401.*X-API-Key"):
        generate_model(photo, tmp_path / "raw.glb", api_key="nope")


def test_server_failure_is_reported_with_its_reason(colab, tmp_path):
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"not-furniture")
    with pytest.raises(ColabError, match="500.*cannot identify image"):
        generate_model(bad, tmp_path / "raw.glb")
    assert not (tmp_path / "raw.glb").exists()


def test_missing_address_says_what_to_do(monkeypatch, photo, tmp_path):
    monkeypatch.delenv(colab_service.URL_ENV, raising=False)
    monkeypatch.setattr(config, "ENV_FILES", [])
    with pytest.raises(ColabError, match="colab_server.ipynb"):
        generate_model(photo, tmp_path / "raw.glb")


def test_address_and_key_can_come_from_a_dotenv_file(colab, photo, tmp_path, monkeypatch):
    monkeypatch.delenv(colab_service.URL_ENV)
    monkeypatch.delenv(colab_service.KEY_ENV)
    env_file = tmp_path / ".env"
    lines = ["# colab", f"{colab_service.URL_ENV}={colab}", f"{colab_service.KEY_ENV}='{KEY}'"]
    env_file.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILES", [tmp_path / "missing.env", env_file])
    assert health()["status"] == "ok"


def test_unreachable_server(photo, tmp_path):
    with pytest.raises(ColabError, match="not reachable"):
        generate_model(photo, tmp_path / "raw.glb", base_url="http://127.0.0.1:9", api_key=KEY, timeout=3)


# ---------- fit ----------
def test_fit_scales_centres_and_stands_the_model(tmp_path):
    raw = raw_model(tmp_path / "raw.glb")                       # 0.5 wide, 1.4 deep, off-centre
    size = fit_model(raw, tmp_path / "fit.glb", width_m=1.0)
    assert size == (1.0, 2.8, 0.4)                              # uniform scale x2, proportions kept
    lo, hi = trimesh.load(tmp_path / "fit.glb").bounds
    assert lo == pytest.approx([-0.5, 0.0, -1.4], abs=1e-4) and hi == pytest.approx([0.5, 0.4, 1.4], abs=1e-4)


def test_fit_can_turn_a_sideways_model(tmp_path):
    raw = raw_model(tmp_path / "raw.glb")
    size = fit_model(raw, tmp_path / "fit.glb", width_m=2.1, yaw_deg=90)    # long side becomes the width
    assert size == (2.1, 0.75, 0.3)
    lo, hi = trimesh.load(tmp_path / "fit.glb").bounds
    assert hi - lo == pytest.approx([2.1, 0.3, 0.75], abs=1e-3)


def test_fit_keeps_the_models_data(tmp_path):
    raw = raw_model(tmp_path / "raw.glb")
    fit_model(raw, tmp_path / "fit.glb", width_m=1.0)
    (before, bin_before), (after, bin_after) = read_glb(raw), read_glb(tmp_path / "fit.glb")
    assert bin_after == bin_before and after["meshes"] == before["meshes"]      # only a root node was added
    assert len(after["nodes"]) == len(before["nodes"]) + 1


def test_fit_rejects_an_absurd_size(tmp_path):
    raw = raw_model(tmp_path / "raw.glb")
    with pytest.raises(ValueError, match="depth is 5.88 m"):
        fit_model(raw, tmp_path / "fit.glb", width_m=2.1)       # sideways sofa: 2.1 wide would be 5.9 deep


# ---------- catalog ----------
def test_custom_piece_replaces_the_stock_item_of_its_kind(tmp_path):
    stock = load_catalog()
    catalog, item = with_custom_item(stock, "sofa", tmp_path / "s.glb", (2.0, 0.9, 0.8))
    assert item.id == "sofa_3_seat" and item.name == "Your sofa"            # same id: rules still apply
    assert (item.width_m, item.depth_m, item.height_m) == (2.0, 0.9, 0.8)
    assert catalog.get("sofa_3_seat") == item and catalog.get("sofa_2_seat") == stock.get("sofa_2_seat")
    assert stock.get("sofa_3_seat").width_m != 2.0 or stock.get("sofa_3_seat").mesh != item.mesh   # stock untouched
    with pytest.raises(KeyError, match="Known kinds"):
        with_custom_item(stock, "piano", tmp_path / "p.glb", (1, 1, 1))


def test_width_defaults_to_the_stock_items_width(tmp_path):
    stock = load_catalog()
    raw = raw_model(tmp_path / "raw.glb", extents=(1.0, 0.4, 0.45))
    _, item = add_furniture_model(raw, "sofa", tmp_path / "user1", stock)
    assert item.width_m == stock.get("sofa_3_seat").width_m
    assert (tmp_path / "user1" / "sofa_3_seat.glb").exists()


# ---------- photo -> walkthrough ----------
def test_photo_to_walkthrough(colab, photo, tmp_path):
    plan = load_plan("data/plans/PROP_1002.json")
    catalog, item = add_furniture_from_photo(photo, "sofa", tmp_path / "user1", width_m=2.0, yaw_deg=90,
                                             generator="colab")
    assert (item.width_m, item.depth_m) == (2.0, 0.71)
    assert (tmp_path / "user1" / "sofa_3_seat_raw.glb").exists()            # kept for a later re-fit

    staging = stage_plan(plan, catalog)
    sofa = next(i for i in staging.items if i.catalog_id == "sofa_3_seat")
    assert (sofa.width_m, sofa.depth_m) == (2.0, 0.71)                      # placed at the custom size
    assert validate_staging(plan, staging, catalog)["approved"]
    assert not validate_staging(plan, staging)["approved"]                  # against the stock catalog the size is wrong

    out = build_walkthrough("data/plans/PROP_1002.json", tmp_path / "w.html", staging=staging, catalog=catalog)
    embedded = json.loads(re.search(r'id="staging-data">(.*?)</script>', out.read_text(encoding="utf-8"), re.S).group(1))
    fitted = (tmp_path / "user1" / "sofa_3_seat.glb").read_bytes()
    assert base64.b64decode(embedded["models"]["sofa_3_seat"]) == fitted    # the user's model is in the page
    assert next(i for i in embedded["items"] if i["catalog_id"] == "sofa_3_seat")["name"] == "Your sofa"

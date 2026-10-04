import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from backend import config
from backend.schema.design import DesignPreferences, FurniturePhoto
from backend.schema.staging import load_catalog
from backend.services import vision_service
from backend.services.vision_service import VisionError, analyze_photo

KEY = "test-key"
KINDS = sorted({i.kind for i in load_catalog().items})

ARMCHAIR = {"scene": "single_item", "fully_visible": True, "kind": "armchair", "name": "round swivel armchair",
            "width_m": 1.3, "style": "modern", "colors": ["cream", "black"], "materials": ["fabric"],
            "description": "A round cream fabric swivel chair with dark accent cushions."}
ROOM = {"scene": "room", "fully_visible": True, "kind": "sofa", "name": "living room", "width_m": 5.0,
        "style": "modern", "colors": ["red", "grey", "white"], "materials": ["boucle fabric", "glass"],
        "description": "A bright modern living room with red and grey modular sofas."}


# ---------- a fake Gemini endpoint (same request / response shape as generateContent) ----------
class FakeGemini(BaseHTTPRequestHandler):
    reply_text = "{}"
    status = 200
    seen = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeGemini.seen = {"path": self.path, "key": self.headers.get("x-goog-api-key"), "body": body}
        if FakeGemini.status != 200:
            payload = {"error": {"code": FakeGemini.status, "message": "Resource has been exhausted."}}
        elif FakeGemini.reply_text is None:
            payload = {"promptFeedback": {"blockReason": "SAFETY"}}
        else:
            payload = {"candidates": [{"content": {"parts": [{"text": FakeGemini.reply_text}]}}]}
        data = json.dumps(payload).encode()
        self.send_response(FakeGemini.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def gemini(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeGemini)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(vision_service.URL_ENV, f"http://127.0.0.1:{server.server_port}/v1beta")
    monkeypatch.setenv(vision_service.KEY_ENV, KEY)
    monkeypatch.delenv(vision_service.MODEL_ENV, raising=False)
    FakeGemini.status = 200

    def answer(value):
        FakeGemini.reply_text = value if isinstance(value, str) or value is None else json.dumps(value)
    yield answer
    server.shutdown()


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "chair.png"
    Image.new("RGBA", (2000, 1500), (240, 235, 220, 0)).save(path)     # large, with transparency
    return path


# ---------- the request ----------
def test_request_carries_the_photo_the_kinds_and_asks_for_json(gemini, photo):
    gemini(ARMCHAIR)
    analyze_photo(photo)
    seen = FakeGemini.seen
    assert seen["path"] == f"/v1beta/models/{vision_service.DEFAULT_MODEL}:generateContent" and seen["key"] == KEY
    text, image = seen["body"]["contents"][0]["parts"]
    assert all(kind in text["text"] for kind in KINDS)                 # closed vocabulary is in the prompt
    assert seen["body"]["generationConfig"] == {"temperature": 0, "responseMimeType": "application/json"}
    sent = Image.open(io.BytesIO(base64.b64decode(image["inline_data"]["data"])))
    assert image["inline_data"]["mime_type"] == "image/jpeg" and max(sent.size) == vision_service.MAX_IMAGE_PX


def test_model_name_can_be_overridden(gemini, photo, monkeypatch):
    gemini(ARMCHAIR)
    monkeypatch.setenv(vision_service.MODEL_ENV, "gemini-next")
    analyze_photo(photo)
    assert FakeGemini.seen["path"].endswith("/models/gemini-next:generateContent")


# ---------- reading the answer ----------
def test_single_piece_is_usable(gemini, photo):
    gemini(ARMCHAIR)
    seen = analyze_photo(photo)
    assert seen.usable and seen.reason is None
    assert (seen.kind, seen.width_m, seen.name) == ("armchair", 1.3, "round swivel armchair")


def test_answer_wrapped_in_a_code_fence_is_still_read(gemini, photo):
    gemini("```json\n" + json.dumps(ARMCHAIR) + "\n```")
    assert analyze_photo(photo).kind == "armchair"


def test_room_photo_is_not_sent_to_3d_but_gives_the_style(gemini, photo):
    gemini(ROOM)
    seen = analyze_photo(photo)
    assert not seen.usable and "whole room" in seen.reason
    assert seen.kind is None and seen.width_m is None           # the model's "sofa" / 5.0 are not used
    prefs = DesignPreferences.from_loose(seen.style_preferences())
    assert prefs.style == "modern" and prefs.palette == ["red", "grey", "white"]
    assert prefs.style_brief.startswith("A bright modern living room")


@pytest.mark.parametrize("change, expected", [
    ({"scene": "several_items"}, "one piece at a time"),
    ({"scene": "not_furniture"}, "No furniture"),
    ({"scene": "a nice chair"}, "No furniture"),                # unknown scene value -> safest reading
    ({"kind": "other", "name": "grand piano"}, "no place in the plan for this piece (grand piano)"),
    ({"kind": "piano"}, "no place in the plan"),                # not a catalog kind
    ({"fully_visible": False}, "cut off"),
])
def test_unusable_photos_get_a_reason(change, expected):
    seen = FurniturePhoto.from_model_answer({**ARMCHAIR, **change}, KINDS)
    assert not seen.usable and expected in seen.reason


@pytest.mark.parametrize("width", [0.05, 12, "1.3 m", None, True])
def test_absurd_width_is_dropped_not_trusted(width):
    seen = FurniturePhoto.from_model_answer({**ARMCHAIR, "width_m": width}, KINDS)
    assert seen.usable and seen.width_m is None                 # still usable: the stock width is used instead


def test_junk_fields_fall_back_to_defaults():
    seen = FurniturePhoto.from_model_answer({"scene": "single_item", "kind": "sofa", "style": "cozy",
                                             "colors": "red", "materials": ["a", "b", "c", "d"], "name": 7}, KINDS)
    assert seen.usable and seen.style == "unknown" and seen.colors == [] and seen.materials == ["a", "b", "c"]
    assert seen.name is None and seen.style_preferences() == {"materials": ["a", "b", "c"]}


# ---------- failures ----------
def test_missing_key_says_where_to_get_one(photo, monkeypatch):
    monkeypatch.delenv(vision_service.KEY_ENV, raising=False)
    monkeypatch.setattr(config, "ENV_FILES", [])
    assert not vision_service.is_configured()
    with pytest.raises(VisionError, match="aistudio.google.com"):
        analyze_photo(photo)


def test_quota_error_is_explained(gemini, photo):
    FakeGemini.status = 429
    with pytest.raises(VisionError, match="429.*exhausted.*try again later"):
        analyze_photo(photo)


def test_blocked_or_empty_answer(gemini, photo):
    gemini(None)
    with pytest.raises(VisionError, match="no answer"):
        analyze_photo(photo)


def test_answer_that_is_not_json(gemini, photo):
    gemini("It looks like a lovely chair!")
    with pytest.raises(VisionError, match="not JSON"):
        analyze_photo(photo)


# ---------- vision -> image-to-3D -> plan, as one run ----------
def fake_tools(monkeypatch, answer):
    import trimesh

    from backend.services import colab_service

    def generate(image_path, out_glb, **options):       # stands in for the Colab server
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        trimesh.creation.box(extents=(1.0, 0.8, 0.9)).export(out_glb)
        return out_glb
    monkeypatch.setattr(vision_service, "is_configured", lambda: True)
    monkeypatch.setattr(vision_service, "analyze_photo",
                        lambda photo, catalog=None: FurniturePhoto.from_model_answer(answer, KINDS))
    monkeypatch.setattr(colab_service, "generate_model", generate)


def test_run_takes_kind_and_width_from_the_photo(monkeypatch, photo, tmp_path):
    from inference import run
    fake_tools(monkeypatch, ARMCHAIR)
    lines = []
    html = run("PROP_1007", photo=photo, out_dir=tmp_path, say=lines.append)
    report = "\n".join(lines)
    assert html is not None and html.exists()
    assert "round swivel armchair: kind armchair, about 1.3 m wide" in report
    assert "Your armchair: 1.3 x" in report and "(replaces armchair) from photo" in report


def test_run_stops_on_a_room_photo(monkeypatch, photo, tmp_path):
    from inference import PipelineError, run
    fake_tools(monkeypatch, ROOM)
    with pytest.raises(PipelineError, match="whole room"):
        run("PROP_1002", photo=photo, out_dir=tmp_path, say=lambda line: None)


def test_users_own_answers_win_over_the_photo(monkeypatch, photo, tmp_path):
    from inference import run
    fake_tools(monkeypatch, {**ARMCHAIR, "kind": "other"})        # model could not name it; the user did
    lines = []
    run("PROP_1007", photo=photo, kind="armchair", width_m=1.0, out_dir=tmp_path, say=lines.append)
    assert "Your armchair: 1.0 x" in "\n".join(lines)

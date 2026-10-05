"""inference.run: what happens when Gemini is busy, contradictory wishes, the user's own piece."""
import io
import json
import urllib.error

import pytest

import inference
from backend.schema.design import DesignPreferences
from backend.schema.staging import load_staging
from backend.services import vision_service
from backend.services.custom_furniture import add_furniture_model

STOCK_ARMCHAIR = "data/catalog/models/armchair.glb"


def fake_photo_to_3d(photo, kind, folder, catalog=None, width_m=None, yaw_deg=0.0, name=None, **kw):
    """Stands in for TRELLIS / TripoSR: uses a model from the catalog instead of a photo."""
    return add_furniture_model(STOCK_ARMCHAIR, kind, folder, catalog, width_m, yaw_deg, name)


@pytest.fixture
def gemini_busy(monkeypatch):
    monkeypatch.setattr(vision_service, "is_configured", lambda: True)

    def busy(*a, **k):
        raise vision_service.VisionError("Gemini answered 503: This model is currently experiencing high demand.")
    monkeypatch.setattr(vision_service, "analyze_photo", busy)
    monkeypatch.setattr(inference, "add_furniture_from_photo", fake_photo_to_3d)


def test_busy_gemini_does_not_stop_a_run_when_the_kind_is_chosen(tmp_path, gemini_busy):
    lines = []
    html = inference.run("PROP_1002", photo="chair.jpg", kind="armchair",      # width not given: Gemini is asked
                         out_dir=tmp_path, say=lines.append, generator="trellis")
    text = "\n".join(lines)
    assert html is not None and "could not be read" in text and "using your choice: armchair" in text
    staging = load_staging(tmp_path / "PROP_1002" / "staging.json")
    assert any(i.catalog_id == "armchair" for i in staging.items)          # the user's piece is placed


def test_busy_gemini_without_a_kind_says_what_to_do(tmp_path, gemini_busy):
    with pytest.raises(inference.PipelineError, match="Choose what it is yourself"):
        inference.run("PROP_1002", photo="chair.jpg", out_dir=tmp_path, say=lambda s: None)


def test_must_have_wins_over_leave_out_and_own_piece_is_never_left_out(tmp_path, gemini_busy):
    lines = []
    prefs = DesignPreferences(must_have=["armchair"], exclude=["armchair", "tv"])
    inference.run("PROP_1002", prefs, photo="chair.jpg", kind="armchair", out_dir=tmp_path, say=lines.append)
    text = "\n".join(lines)
    assert "kept it as must have" in text
    staging = load_staging(tmp_path / "PROP_1002" / "staging.json")
    ids = {i.catalog_id for i in staging.items}
    assert "armchair" in ids and "tv_unit" not in ids


def test_gemini_is_retried_when_busy(monkeypatch):
    answers = [503, 503, {"candidates": [{"content": {"parts": [{"text": json.dumps({"scene": "room"})}]}}]}]
    monkeypatch.setattr(vision_service.time, "sleep", lambda s: None)

    def urlopen(request, timeout):
        a = answers.pop(0)
        if isinstance(a, int):
            raise urllib.error.HTTPError(request.full_url, a, "busy", {}, io.BytesIO(b'{"error":{"message":"busy"}}'))
        return io.BytesIO(json.dumps(a).encode())
    monkeypatch.setattr(vision_service.urllib.request, "urlopen", urlopen)
    data = vision_service._ask_gemini("p", "data/matched_images/4770.png", "key", "m", "https://x", 5)
    assert data == {"scene": "room"} and not answers                         # 3rd try answered

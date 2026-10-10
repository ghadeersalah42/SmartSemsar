"""Keys pasted with line breaks, tokens hidden in errors, Groq taking over when Gemini is busy."""
import pytest

from backend.config import mask_secrets, setting
from backend.schema.plan import load_plan
from backend.schema.staging import load_catalog
from backend.services import llm_furnisher, vision_service
from backend.services.llm_furnisher import furnish


def test_pasted_token_with_line_break_is_cleaned(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abcDEF123\r\nghiJKL456  ")
    monkeypatch.setenv("SMARTSEMSAR_COLAB_URL", "https://x.ngrok-free.app ")
    assert setting("HF_TOKEN") == "hf_abcDEF123ghiJKL456"
    assert setting("SMARTSEMSAR_COLAB_URL") == "https://x.ngrok-free.app"     # not a secret: only stripped


def test_tokens_never_shown_in_errors():
    text = mask_secrets("Illegal header value b'Bearer hf_FAKEtoken123456789' and gsk_abcdefghijkl and AIzaSyABCDEFGHIJK")
    assert "FAKEtoken" not in text and "abcdefgh" not in text and "SyABCDEF" not in text


class Busy:
    name = "gemini/test"

    def chat_json(self, messages):
        raise RuntimeError("503 This model is currently experiencing high demand")


class Works:
    name = "groq/test"

    def __init__(self, plan):
        bed = next(r for r in plan.rooms() if r.type == "bedroom" and r.area_sqm > 20)
        self.room = bed.id

    def chat_json(self, messages):
        return {"rooms": {self.room: [{"item": "nightstand", "place": "free"}]}, "summary": "ok"}


def test_planner_switches_to_groq_when_gemini_is_busy(monkeypatch):
    plan = load_plan("data/plans/PROP_1002.json")
    groq = Works(plan)
    monkeypatch.setattr(llm_furnisher, "detect_llms", lambda: [Busy(), groq])
    st, report = furnish(plan, None, load_catalog())
    assert "Switched to groq/test." in report.warnings and report.planner.endswith("groq/test")
    assert any(i.room_id == groq.room for i in st.items)


def test_photo_is_read_by_groq_when_gemini_is_busy(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

    def busy(*a, **k):
        raise vision_service.VisionError("Gemini answered 503: high demand.")
    monkeypatch.setattr(vision_service, "_ask_gemini", busy)
    monkeypatch.setattr(vision_service, "_ask_groq", lambda *a, **k: {
        "scene": "single_item", "fully_visible": True, "kind": "armchair", "name": "lounge chair",
        "width_m": 0.75, "style": "modern", "colors": ["beige"], "materials": ["walnut"]})
    seen = vision_service.analyze_photo("data/matched_images/4770.png")
    assert seen.usable and seen.kind == "armchair" and seen.width_m == 0.75


def test_no_groq_key_keeps_the_gemini_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(vision_service, "_ask_gemini",
                        lambda *a, **k: (_ for _ in ()).throw(vision_service.VisionError("Gemini answered 503")))
    with pytest.raises(vision_service.VisionError, match="503"):
        vision_service.analyze_photo("data/matched_images/4770.png")


def test_photo_tries_the_lighter_gemini_before_groq(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv(vision_service.MODEL_ENV, raising=False)
    monkeypatch.delenv(vision_service.BACKUP_MODEL_ENV, raising=False)
    tried = []

    def gemini(prompt, path, key, model, url, timeout):
        tried.append(model)
        if model == vision_service.DEFAULT_MODEL:
            raise vision_service.VisionError("Gemini answered 503: high demand.")
        return {"scene": "single_item", "fully_visible": True, "kind": "sofa", "width_m": 2.1}
    monkeypatch.setattr(vision_service, "_ask_gemini", gemini)
    seen = vision_service.analyze_photo("data/matched_images/4770.png")
    assert tried == [vision_service.DEFAULT_MODEL, vision_service.BACKUP_MODEL] and seen.kind == "sofa"


def test_planner_order_gemini_groq_then_lighter_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    for name in ("SMARTSEMSAR_GEMINI_MODEL", "SMARTSEMSAR_GROQ_MODEL", "SMARTSEMSAR_GEMINI_BACKUP_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9/v1")      # nothing listens there
    names = [c.name for c in llm_furnisher.detect_llms()]
    assert names == ["gemini/gemini-flash-latest", f"groq/{llm_furnisher.GROQ_DEFAULT_MODEL}",
                     "gemini/gemini-flash-lite-latest"]


def test_model_check_keeps_the_first_model_that_answers(monkeypatch):
    from backend.services import model_check
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    for name in ("SMARTSEMSAR_GEMINI_MODEL", "SMARTSEMSAR_GROQ_MODEL", "SMARTSEMSAR_GROQ_VISION_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def gemini(prompt, path, key, model, url, timeout):
        if model == "gemini-flash-latest":
            raise vision_service.VisionError("Gemini answered 503")
    monkeypatch.setattr(vision_service, "_ask_gemini", gemini)

    def groq_photo(prompt, path, key, model, timeout):
        if model != "qwen/new-vision":
            raise vision_service.VisionError("Groq answered 400: does not support images")
    monkeypatch.setattr(vision_service, "_ask_groq", groq_photo)
    monkeypatch.setattr(model_check, "groq_models", lambda key: ["openai/gpt-oss-120b", "qwen/new-vision",
                                                                 "whisper-large-v3"])

    def chat(self, messages):
        if self.model in ("gemini-flash-latest", "gemini-flash-lite-latest"):
            raise RuntimeError("503")
        return {"ok": True}
    monkeypatch.setattr(llm_furnisher.LLMClient, "chat_json", chat)

    chosen = model_check.pick_working_models(say=lambda s: None)
    assert chosen == {"gemini_photo": "gemini-flash-lite-latest", "gemini_text": "gemini-3.8-flash",
                      "groq_photo": "qwen/new-vision", "groq_text": "openai/gpt-oss-120b"}
    assert setting("SMARTSEMSAR_GEMINI_MODEL") == "gemini-flash-lite-latest"
    assert setting("SMARTSEMSAR_GROQ_VISION_MODEL") == "qwen/new-vision"
    assert setting("SMARTSEMSAR_GROQ_MODEL") == "openai/gpt-oss-120b"

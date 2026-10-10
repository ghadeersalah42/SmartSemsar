"""
Which Gemini and Groq models answer right now? Each one is really called; the first that works is used.
بيجرب موديلات Gemini و Groq بجد (سؤال نصي + صورة) ويختار أول واحد بيرد.

Google and Groq rename, retire and overload models often (llama-4-scout and llama-3.3-70b were
removed from Groq, gemini-2.5-flash closed to new users, gemini-flash-latest often answers 503).
This finds the working ones and sets the environment variables the services read:

    SMARTSEMSAR_GEMINI_MODEL       Gemini for photos and for arranging furniture
    SMARTSEMSAR_GROQ_VISION_MODEL  Groq for photos (vision_service backup)
    SMARTSEMSAR_GROQ_MODEL         Groq for arranging furniture (llm_furnisher backup)

    chosen = pick_working_models()          # prints one line per model tried

Usage:
    python -m backend.services.model_check
"""
import os
from pathlib import Path
from typing import Callable, Optional

import requests

from backend.config import mask_secrets, setting
from backend.services import llm_furnisher, vision_service

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_PHOTO = REPO_ROOT / "OIP.jpg"          # a small furniture photo that ships with the repo
ASK_PHOTO = 'Look at the photo and answer only JSON: {"kind": "<what furniture it is>"}'
ASK_TEXT = 'Answer only this JSON: {"ok": true}'
GEMINI_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-2.5-flash"]
GROQ_TEXT_FIRST = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
GROQ_VISION_WORDS = ("qwen", "llama-4", "vision", "-vl", "scout", "maverick")
GROQ_NOT_CHAT = ("whisper", "tts", "guard", "compound", "playai", "orpheus", "allam", "safeguard")
TIMEOUT_S = 40


def _short(e: Exception) -> str:
    return f"{type(e).__name__}: {mask_secrets(e)[:110]}"


def _try(test: Callable[[], object]) -> str:
    try:
        test()
        return "OK"
    except Exception as e:      # busy, gone, no access, bad JSON: all just mean "not this one"
        return _short(e)


def _first(models: list[str], test: Callable[[str], str], say: Callable[[str], None]) -> Optional[str]:
    for m in models:
        result = test(m)
        say(f"   {m:40s} {result}")
        if result == "OK":
            return m
    return None


def groq_models(key: str) -> list[str]:
    r = requests.get("https://api.groq.com/openai/v1/models", timeout=20, headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return sorted(m["id"] for m in r.json()["data"])


def pick_working_models(say: Callable[[str], None] = print, set_env: bool = True) -> dict[str, Optional[str]]:
    """{job: model or None}. With set_env, the environment points the services at the working ones."""
    chosen: dict[str, Optional[str]] = {}
    gemini_key, groq_key = setting("GEMINI_API_KEY"), setting("GROQ_API_KEY")
    photo = str(TEST_PHOTO)

    if gemini_key:
        say("Gemini, reading a photo:")
        chosen["gemini_photo"] = _first(GEMINI_MODELS, lambda m: _try(lambda: vision_service._ask_gemini(
            ASK_PHOTO, photo, gemini_key, m, vision_service.DEFAULT_URL, TIMEOUT_S)), say)
        say("Gemini, arranging furniture (text):")
        chosen["gemini_text"] = _first(GEMINI_MODELS, lambda m: _try(lambda: llm_furnisher.LLMClient(
            "gemini", llm_furnisher.GEMINI_URL, m, gemini_key, TIMEOUT_S).chat_json(
            [{"role": "user", "content": ASK_TEXT}])), say)
        best = chosen["gemini_photo"] or chosen["gemini_text"]
        if set_env and best:
            os.environ["SMARTSEMSAR_GEMINI_MODEL"] = best
    else:
        say("GEMINI_API_KEY not set")

    if groq_key:
        try:
            ids = groq_models(groq_key)
        except Exception as e:
            ids = []
            say(f"Could not list Groq's models: {_short(e)}")
        chat = [i for i in ids if not any(w in i.lower() for w in GROQ_NOT_CHAT)]
        vision_first = [i for i in chat if any(w in i.lower() for w in GROQ_VISION_WORDS)]
        say("Groq, reading a photo:")
        chosen["groq_photo"] = _first(vision_first + [i for i in chat if i not in vision_first],
                                      lambda m: _try(lambda: vision_service._ask_groq(
                                          ASK_PHOTO, photo, groq_key, m, TIMEOUT_S)), say)
        say("Groq, arranging furniture (text):")
        text_first = [m for m in GROQ_TEXT_FIRST if m in chat]
        chosen["groq_text"] = _first(text_first + [i for i in chat if i not in text_first],
                                     lambda m: _try(lambda: llm_furnisher.LLMClient(
                                         "groq", "https://api.groq.com/openai/v1", m, groq_key, TIMEOUT_S).chat_json(
                                         [{"role": "user", "content": ASK_TEXT}])), say)
        if set_env and chosen["groq_photo"]:
            os.environ["SMARTSEMSAR_GROQ_VISION_MODEL"] = chosen["groq_photo"]
        if set_env and chosen["groq_text"]:
            os.environ["SMARTSEMSAR_GROQ_MODEL"] = chosen["groq_text"]
    else:
        say("GROQ_API_KEY not set")

    say("Used by the app: " + ", ".join(f"{job} = {m or 'NONE WORKS'}" for job, m in chosen.items()))
    return chosen


if __name__ == "__main__":
    pick_working_models()

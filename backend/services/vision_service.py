"""
Vision tool: look at an uploaded photo and say what it is (Gemini).
أداة الـ VLM: بتبص على صورة المستخدم وتقول هي إيه، قبل ما تتبعت لموديل الـ 3D.

The VLM only reads the picture and answers in JSON. It does not build the 3D model
(that is colab_service) and it does not place anything (that is furniture_placer).

    photo = analyze_photo("chair.jpg")
    photo.usable      -> True: one whole piece we have a place for; send it to image-to-3D
    photo.kind        -> "armchair"          (always a catalog kind, or None)
    photo.width_m     -> 1.3                 (estimate of the real width)
    photo.reason      -> why not, if not usable (a room, several pieces, not furniture ...)
    photo.style_preferences() -> style / palette / materials / style_brief for DesignPreferences

The model's answer is never trusted as it is: kind is checked against the catalog, the width
against a sane range, and `usable` is decided here, not by the model.

Key: GEMINI_API_KEY in the environment or .env  (https://aistudio.google.com/apikey)

Usage:
    python -m backend.services.vision_service <photo>
"""
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

from backend.config import mask_secrets, setting
from backend.schema.design import FurniturePhoto
from backend.schema.staging import Catalog, load_catalog

KEY_ENV, MODEL_ENV, URL_ENV = "GEMINI_API_KEY", "SMARTSEMSAR_GEMINI_MODEL", "SMARTSEMSAR_GEMINI_URL"
# backup when Gemini is busy or not set: Groq's vision model (OpenAI-compatible API)
GROQ_KEY_ENV, GROQ_MODEL_ENV = "GROQ_API_KEY", "SMARTSEMSAR_GROQ_VISION_MODEL"
# Groq retired llama-4-scout; this one read furniture photos in October 2026.
# backend/services/model_check.py finds a working one when this goes too.
GROQ_DEFAULT_MODEL = "qwen/qwen3.8-27b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "gemini-flash-latest"   # alias Google keeps pointed at the current Flash model
# tried when the main model stays busy (503) or is gone (404): a lighter model, usually free of the rush
BACKUP_MODEL_ENV, BACKUP_MODEL = "SMARTSEMSAR_GEMINI_BACKUP_MODEL", "gemini-flash-lite-latest"
DEFAULT_URL = "https://generativelanguage.googleapis.com/v1beta"
MAX_IMAGE_PX = 1024
BUSY_CODES = {500, 502, 503, 504}      # "high demand" and other temporary errors
BUSY_RETRY_WAITS_S = (2, 5)            # wait before the 2nd and 3rd try
WIDTH_RANGE_M = (0.3, 3.5)      # an estimate outside this is ignored

PROMPT = """You are helping a home-staging app. Look at the photo and answer with ONE JSON object, nothing else.

{{
  "scene": one of ["single_item", "several_items", "room", "not_furniture"],
  "fully_visible": true or false,
  "kind": one of {kinds} or "other",
  "name": short plain name of the piece, e.g. "round swivel armchair",
  "width_m": your estimate of its real width in meters, as a number,
  "style": one of ["modern", "classic", "minimal", "industrial", "unknown"],
  "colors": up to 3 main colours as plain words,
  "materials": up to 3 main materials as plain words,
  "description": one sentence describing the look, usable as a style brief
}}

Rules:
- "single_item": exactly one piece of furniture is the subject. Cushions on a sofa are part of it.
- "several_items": two or more separate pieces. "room": an interior scene. "not_furniture": anything else.
- "fully_visible": false if the piece is cut off by the frame or mostly hidden.
- "kind": choose the closest from the list; use "other" only if none fits. A wide one-person seat is an "armchair".
- For "room" or "several_items", set kind to "other" and width_m to null, and still fill style, colors,
  materials and description for the whole picture.
- Never invent details you cannot see."""


class VisionError(RuntimeError):
    """The vision model is not configured, not reachable, or gave an unusable answer."""


def _image_part(image_path) -> dict:
    image = Image.open(image_path)
    if image.mode != "RGB":
        background = Image.new("RGB", image.size, "white")      # transparent PNGs: white behind
        background.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[-1])
        image = background
    image.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX))
    out = io.BytesIO()
    image.save(out, "JPEG", quality=88)
    return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(out.getvalue()).decode("ascii")}}


def _ask_gemini(prompt: str, image_path, api_key: str, model: str, base_url: str, timeout: float) -> dict:
    body = {"contents": [{"parts": [{"text": prompt}, _image_part(image_path)]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models/{model}:generateContent", data=json.dumps(body).encode("utf-8"),
        method="POST", headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    for wait in (*BUSY_RETRY_WAITS_S, None):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                answer = json.loads(response.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in BUSY_CODES and wait is not None:      # Google is overloaded for a moment: retry
                time.sleep(wait)
                continue
            detail = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            hint = {400: " Check GEMINI_API_KEY.", 403: " Check GEMINI_API_KEY.",
                    404: f" The model name '{model}' may be out of date; set {MODEL_ENV}.",
                    429: " The free quota is used up for now; try again later.",
                    503: " Google's servers are busy; this is temporary."}.get(e.code, "")
            raise VisionError(f"Gemini answered {e.code}: {detail}{hint}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise VisionError(f"Gemini not reachable ({e}).") from e

    try:
        text = answer["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise VisionError(f"Gemini gave no answer for this photo ({answer.get('promptFeedback', 'no details')}).")
    text = text.strip()
    if text.startswith("```"):                      # some models still wrap JSON in a code fence
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except ValueError as e:
        raise VisionError(f"Gemini's answer was not JSON: {text[:120]!r}") from e
    if not isinstance(data, dict):
        raise VisionError("Gemini's answer was not a JSON object.")
    return data


def _ask_groq(prompt: str, image_path, api_key: str, model: str, timeout: float) -> dict:
    """Same question to Groq's vision model (OpenAI-compatible chat API)."""
    image = _image_part(image_path)["inline_data"]
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{image['mime_type']};base64,{image['data']}"}}]}]}
    request = urllib.request.Request(GROQ_URL, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
                                              "User-Agent": "SmartSemsar"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.loads(response.read())
        text = answer["choices"][0]["message"]["content"]
        data = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except urllib.error.HTTPError as e:
        raise VisionError(f"Groq answered {e.code}: {mask_secrets(e.read().decode('utf-8', 'replace'))[:160]}") from e
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError) as e:
        raise VisionError(f"Groq gave no usable answer ({mask_secrets(e)[:120]}).") from e
    if not isinstance(data, dict):
        raise VisionError("Groq's answer was not a JSON object.")
    return data


def analyze_photo(image_path, catalog: Optional[Catalog] = None, api_key: Optional[str] = None,
                  model: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 60) -> FurniturePhoto:
    """What is in this photo, and can it go to the image-to-3D model?
    Gemini first; if it is busy or not set and GROQ_API_KEY is set, Groq's vision model answers instead."""
    api_key = api_key or setting(KEY_ENV)
    groq_key = setting(GROQ_KEY_ENV)
    if not api_key and not groq_key:
        raise VisionError(f"No vision key. Get one at https://aistudio.google.com/apikey and set {KEY_ENV} in .env.")
    kinds = sorted({i.kind for i in (catalog or load_catalog()).items if i.kind})
    prompt = PROMPT.format(kinds=json.dumps(kinds))
    gemini_error = None
    if api_key:
        main = model or setting(MODEL_ENV, DEFAULT_MODEL)
        backup = None if model else setting(BACKUP_MODEL_ENV, BACKUP_MODEL)    # a model asked for by name is not swapped
        for name in [main] + ([backup] if backup and backup != main else []):
            try:
                data = _ask_gemini(prompt, Path(image_path), api_key, name,
                                   base_url or setting(URL_ENV, DEFAULT_URL), timeout)
                return FurniturePhoto.from_model_answer(data, kinds, WIDTH_RANGE_M)
            except VisionError as e:
                gemini_error = gemini_error or e         # the main model's error is the one worth showing
        if not groq_key:
            raise gemini_error
    try:
        data = _ask_groq(prompt, Path(image_path), groq_key, setting(GROQ_MODEL_ENV, GROQ_DEFAULT_MODEL), timeout)
    except VisionError as e:
        raise VisionError(f"{gemini_error} Then: {e}" if gemini_error else str(e)) from e
    return FurniturePhoto.from_model_answer(data, kinds, WIDTH_RANGE_M)


def is_configured() -> bool:
    return bool(setting(KEY_ENV) or setting(GROQ_KEY_ENV))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m backend.services.vision_service <photo>")
    try:
        result = analyze_photo(sys.argv[1])
    except VisionError as e:
        sys.exit(f"error: {e}")
    print(result.model_dump_json(indent=1))

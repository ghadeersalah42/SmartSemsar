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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

from backend.config import setting
from backend.schema.design import FurniturePhoto
from backend.schema.staging import Catalog, load_catalog

KEY_ENV, MODEL_ENV, URL_ENV = "GEMINI_API_KEY", "SMARTSEMSAR_GEMINI_MODEL", "SMARTSEMSAR_GEMINI_URL"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_URL = "https://generativelanguage.googleapis.com/v1beta"
MAX_IMAGE_PX = 1024
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
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        hint = {400: " Check GEMINI_API_KEY.", 403: " Check GEMINI_API_KEY.",
                404: f" The model name '{model}' may be out of date; set {MODEL_ENV}.",
                429: " The free quota is used up for now; try again later."}.get(e.code, "")
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


def analyze_photo(image_path, catalog: Optional[Catalog] = None, api_key: Optional[str] = None,
                  model: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 60) -> FurniturePhoto:
    """What is in this photo, and can it go to the image-to-3D model?"""
    api_key = api_key or setting(KEY_ENV)
    if not api_key:
        raise VisionError(f"No vision key. Get one at https://aistudio.google.com/apikey and set {KEY_ENV} in .env.")
    kinds = sorted({i.kind for i in (catalog or load_catalog()).items if i.kind})
    data = _ask_gemini(PROMPT.format(kinds=json.dumps(kinds)), Path(image_path), api_key,
                       model or setting(MODEL_ENV, DEFAULT_MODEL), base_url or setting(URL_ENV, DEFAULT_URL), timeout)
    return FurniturePhoto.from_model_answer(data, kinds, WIDTH_RANGE_M)


def is_configured() -> bool:
    return bool(setting(KEY_ENV))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m backend.services.vision_service <photo>")
    try:
        result = analyze_photo(sys.argv[1])
    except VisionError as e:
        sys.exit(f"error: {e}")
    print(result.model_dump_json(indent=1))

"""
Photo -> 3D model with Microsoft TRELLIS.2, through its free Hugging Face Space (ZeroGPU).
صورة قطعة فرش -> موديل GLB بخامات، من غير GPU عندنا (بيستخدم الـ quota المجانية لحساب Hugging Face).

Same contract as colab_service.generate_model(): one photo of one piece in, a GLB out. The GLB has
no real size and may face any way; custom_furniture.fit_model() fixes both afterwards.

Steps on the Space: start_session -> preprocess_image (removes the background) -> image_to_3d -> extract_glb.
Measured on PROP_1002's test chair: ~40 s at resolution 512, GLB ~8 MB with 200k faces.

Key: HF_TOKEN in the environment or .env (a Read token is enough). Without it the Space still works
but with the much smaller anonymous GPU quota.

    glb = generate_model("chair.jpg", "out/chair_raw.glb")

Usage:
    python -m backend.services.trellis_service <photo> <out.glb>
"""
import shutil
import sys
from pathlib import Path
from typing import Optional

from backend.config import setting

SPACE_ENV, TOKEN_ENV = "SMARTSEMSAR_TRELLIS_SPACE", "HF_TOKEN"
DEFAULT_SPACE = "microsoft/TRELLIS.2"
RESOLUTIONS = ("512", "1024", "1536")


class TrellisError(RuntimeError):
    """The Space is not reachable, out of GPU quota, or returned no model."""


def _client(space: str, token: Optional[str]):
    try:
        from gradio_client import Client
    except ImportError as e:
        raise TrellisError("gradio_client is not installed (pip install gradio_client).") from e
    try:
        return Client(space, token=token or None, verbose=False)
    except Exception as e:      # network, Space asleep or renamed
        raise TrellisError(f"Could not connect to the Space {space} ({e}).") from e


def _path(result) -> str:
    """gradio returns a file path or a {'path': ...} dict depending on the version."""
    if isinstance(result, (list, tuple)):
        result = result[0]
    return result["path"] if isinstance(result, dict) else str(result)


def generate_model(image_path, out_glb, resolution: str = "512", faces: int = 200_000,
                   texture_size: int = 1024, seed: Optional[int] = None, space: Optional[str] = None,
                   token: Optional[str] = None, client=None) -> Path:
    """One photo of one piece of furniture -> textured GLB saved at out_glb.
    resolution 512 is fastest and uses the least quota; faces / texture_size set the GLB size."""
    if str(resolution) not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
    from gradio_client import handle_file

    image_path, out_glb = Path(image_path), Path(out_glb)
    c = client or _client(space or setting(SPACE_ENV, DEFAULT_SPACE), token or setting(TOKEN_ENV))
    try:
        c.predict(api_name="/start_session")
        clean = c.predict(handle_file(str(image_path)), api_name="/preprocess_image")
        seed_value = c.predict(seed is None, seed or 0, api_name="/get_seed")
        c.predict(handle_file(_path(clean)), seed_value, str(resolution), api_name="/image_to_3d")
        glb, _ = c.predict(faces, texture_size, api_name="/extract_glb")
    except Exception as e:
        text = str(e)
        hint = " The free GPU quota may be used up; try again later or log in with HF_TOKEN." \
            if "quota" in text.lower() or "zerogpu" in text.lower() else ""
        raise TrellisError(f"TRELLIS.2 failed: {text[:200]}{hint}") from e
    src = Path(_path(glb))
    if not src.exists() or src.read_bytes()[:4] != b"glTF":
        raise TrellisError("TRELLIS.2 did not return a GLB model.")
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out_glb)
    return out_glb


def is_configured() -> bool:
    return bool(setting(TOKEN_ENV))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m backend.services.trellis_service <photo> <out.glb>")
    try:
        out = generate_model(sys.argv[1], sys.argv[2])
    except TrellisError as e:
        sys.exit(f"error: {e}")
    print(f"{out} ({out.stat().st_size // 1024} KB)")

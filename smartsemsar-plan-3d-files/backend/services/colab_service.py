"""
Client for the Colab furniture-3D server (cv_service/colab_server.ipynb).
صورة قطعة فرش -> موديل GLB، عن طريق سيرفر Colab المفتوح بـ ngrok.

The address and key come from the environment, or from a .env file in this folder's
root or the repo root (the notebook prints both lines):
    SMARTSEMSAR_COLAB_URL=https://....ngrok-free.app
    SMARTSEMSAR_COLAB_KEY=<the SMARTSEMSAR_KEY secret set in the notebook>

    glb_path = generate_model("sofa.jpg", "out/sofa_raw.glb")

The GLB that comes back has no real size and may face any way.
custom_furniture.fit_model() turns it into a catalog-ready model.

Usage:
    python -m backend.services.colab_service health
    python -m backend.services.colab_service generate <photo> <out.glb>
"""
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from backend.config import setting

URL_ENV, KEY_ENV = "SMARTSEMSAR_COLAB_URL", "SMARTSEMSAR_COLAB_KEY"
GENERATE_TIMEOUT_S = 300        # a cold GPU can take a while on the first photo


class ColabError(RuntimeError):
    """The Colab server is not configured, not reachable, or refused the request."""


def _settings(base_url: Optional[str], api_key: Optional[str]) -> tuple[str, str]:
    base_url = (base_url or setting(URL_ENV)).rstrip("/")
    if not base_url:
        raise ColabError(f"No Colab address. Run cv_service/colab_server.ipynb and set {URL_ENV}.")
    return base_url, api_key or setting(KEY_ENV)


def _call(request: urllib.request.Request, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except ValueError:
            pass
        raise ColabError(f"Colab server answered {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ColabError(f"Colab server not reachable at {request.full_url} ({e}). "
                         "Is the notebook still running? Its address changes on every restart.") from e


def _headers(api_key: str) -> dict:
    # ngrok's free tier shows a browser warning page unless this header is sent
    return {"X-API-Key": api_key, "ngrok-skip-browser-warning": "1", "User-Agent": "SmartSemsar"}


def health(base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 15) -> dict:
    base_url, api_key = _settings(base_url, api_key)
    request = urllib.request.Request(base_url + "/health", headers=_headers(api_key))
    return json.loads(_call(request, timeout))


def _multipart(fields: dict[str, str], file_field: str, file_name: str, file_bytes: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    kind = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
                 f'filename="{file_name}"\r\nContent-Type: {kind}\r\n\r\n'.encode() + file_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def generate_model(image_path, out_glb, base_url: Optional[str] = None, api_key: Optional[str] = None,
                   mc_resolution: int = 192, strip_background: bool = True,
                   timeout: float = GENERATE_TIMEOUT_S) -> Path:
    """Send one photo of one piece of furniture; save the GLB that comes back."""
    base_url, api_key = _settings(base_url, api_key)
    image_path, out_glb = Path(image_path), Path(out_glb)
    body, content_type = _multipart(
        {"mc_resolution": str(mc_resolution), "strip_background": "true" if strip_background else "false"},
        "image", image_path.name, image_path.read_bytes())
    request = urllib.request.Request(base_url + "/generate", data=body, method="POST",
                                     headers={**_headers(api_key), "Content-Type": content_type})
    glb = _call(request, timeout)
    if glb[:4] != b"glTF":
        raise ColabError("Colab server did not return a GLB model.")
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out_glb.write_bytes(glb)
    return out_glb


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["health"]:
            print(health())
        elif sys.argv[1:2] == ["generate"] and len(sys.argv) == 4:
            out = generate_model(sys.argv[2], sys.argv[3])
            print(f"{out} ({out.stat().st_size // 1024} KB)")
        else:
            sys.exit("usage: python -m backend.services.colab_service health | generate <photo> <out.glb>")
    except ColabError as e:
        sys.exit(f"error: {e}")

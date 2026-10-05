"""
Settings: API keys and service addresses.
قراءة المفاتيح والعناوين من الـ environment أو من ملف .env (مش بيترفع على git).

    SMARTSEMSAR_COLAB_URL / SMARTSEMSAR_COLAB_KEY   image-to-3D server (cv_service/colab_server.ipynb)
    GEMINI_API_KEY                                  vision model (https://aistudio.google.com/apikey)
    SMARTSEMSAR_GEMINI_MODEL                        optional, overrides the default model name
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILES = [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]


SECRET_SUFFIXES = ("_KEY", "_TOKEN")


def setting(name: str, default: str = "") -> str:
    """Environment first, then the first .env file that has the name.
    API keys and tokens (names ending in _KEY / _TOKEN) never contain whitespace, so spaces and
    line breaks picked up when pasting them (e.g. into Colab Secrets) are removed."""
    value = os.environ.get(name, "").strip()
    if not value:
        for env_file in ENV_FILES:
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    key, _, v = line.partition("=")
                    if key.strip() == name and v.strip():
                        value = v.strip().strip("\"'")
                        break
            if value:
                break
    if not value:
        return default
    return "".join(value.split()) if name.endswith(SECRET_SUFFIXES) else value


def mask_secrets(text: str) -> str:
    """Hide anything that looks like a token or API key before showing an error to a user."""
    import re
    text = re.sub(r"hf_[A-Za-z0-9]{6,}", "hf_***", str(text))
    text = re.sub(r"(gsk_|AIza)[A-Za-z0-9_\-]{8,}", r"\1***", text)
    return re.sub(r"(Bearer\s+)[^\s'\"]+", r"\1***", text)

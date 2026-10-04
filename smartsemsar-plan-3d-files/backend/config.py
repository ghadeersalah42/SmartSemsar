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


def setting(name: str, default: str = "") -> str:
    """Environment first, then the first .env file that has the name."""
    if os.environ.get(name):
        return os.environ[name]
    for env_file in ENV_FILES:
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == name and value.strip():
                    return value.strip().strip("\"'")
    return default

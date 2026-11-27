"""
AI configuration loader (Round 4, R4.5).

Every prompt, model ID, embedding model and AI threshold lives in ONE versioned file:
config/ai_config.yaml. Code reads it only through this module:

    from backend.ai_config import model, prompt, prompt_version, threshold, config_version

    model("intent")                          -> "openai/gpt-oss-20b"   (env override if set)
    prompt("orchestrator")                   -> the prompt text
    prompt_version("intent")                 -> "v1.1.0"
    threshold("matching.area_tolerance")     -> 0.10
    config_version()                         -> "1.0.0+3f2a9c1d"  (SemVer + content hash)
    trace_metadata("intent")                 -> tags for a LangSmith trace (R4.1)

Point SMARTSEMSAR_AI_CONFIG at another file to try a candidate config (used by the R4.5 gate).
Only the standard library and PyYAML are needed, so standalone scripts can import it too.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "ai_config.yaml"
CONFIG_PATH_ENV = "SMARTSEMSAR_AI_CONFIG"


class AIConfigError(KeyError):
    """A name that is not in config/ai_config.yaml."""


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH)


@lru_cache(maxsize=None)
def _load(path: str) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    for section in ("version", "models", "prompts", "thresholds"):
        if section not in data:
            raise AIConfigError(f"{path}: missing section '{section}'")
    return data, hashlib.sha256(raw).hexdigest()[:8]


def load() -> dict:
    """The whole config as a dict."""
    return _load(str(config_path()))[0]


def reload() -> dict:
    """Drop the cache (after editing the file, or in tests)."""
    _load.cache_clear()
    return load()


def config_version() -> str:
    """SemVer from the file + a hash of its bytes: changes on every edit, even without a bump."""
    data, digest = _load(str(config_path()))
    return f"{data['version']}+{digest}"


def _env_override(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:  # same .env lookup the rest of the backend uses
        from backend.config import setting
        return setting(name)
    except Exception:
        return ""


def model_entry(name: str) -> dict:
    try:
        return load()["models"][name]
    except KeyError:
        raise AIConfigError(f"model '{name}' is not in {config_path().name}") from None


def model(name: str) -> str:
    """Model ID for a role. The entry's `env` variable, when set, overrides the default."""
    entry = model_entry(name)
    env = entry.get("env")
    return (_env_override(env) if env else "") or str(entry["id"])


def model_default(name: str) -> str:
    """The configured ID with no env override (for code that applies its own override later)."""
    return str(model_entry(name)["id"])


def prompt_entry(name: str) -> dict:
    try:
        return load()["prompts"][name]
    except KeyError:
        raise AIConfigError(f"prompt '{name}' is not in {config_path().name}") from None


def prompt(name: str) -> str:
    return prompt_entry(name)["text"]


def prompt_version(name: str) -> str:
    return str(prompt_entry(name)["version"])


def threshold(dotted: str) -> Any:
    """threshold("matching.area_tolerance")"""
    node: Any = load()["thresholds"]
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise AIConfigError(f"threshold '{dotted}' is not in {config_path().name}")
        node = node[part]
    return node


def trace_metadata(prompt_name: str, model_name: str | None = None, stage: str | None = None) -> dict:
    """What every trace step is tagged with (R4.1): stage, prompt version, model, config version."""
    return {
        "stage": stage or prompt_name,
        "prompt_name": prompt_name,
        "prompt_version": prompt_version(prompt_name),
        "model_name": model(model_name or prompt_name),
        "config_version": config_version(),
    }

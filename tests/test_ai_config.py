"""R4.5 - every prompt, model ID, embedding model and AI threshold sits in one versioned config."""
import re
from pathlib import Path

import pytest
import yaml

from backend import ai_config
from backend.ai_config import AIConfigError

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "ai_config.yaml"
SEMVER = re.compile(r"^v?\d+\.\d+\.\d+$")


@pytest.fixture(autouse=True)
def _fresh_config(monkeypatch):
    monkeypatch.delenv(ai_config.CONFIG_PATH_ENV, raising=False)
    ai_config.reload()
    yield
    ai_config.reload()


def test_config_has_a_semver_and_a_content_hash():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert SEMVER.match(str(data["version"]))
    version = ai_config.config_version()
    assert re.match(r"^\d+\.\d+\.\d+\+[0-9a-f]{8}$", version)
    assert version.startswith(f"{data['version']}+")


def test_every_prompt_has_a_version_and_text():
    prompts = ai_config.load()["prompts"]
    assert len(prompts) >= 9
    for name, entry in prompts.items():
        assert SEMVER.match(str(entry["version"])), name
        assert ai_config.prompt(name).strip(), name


def test_every_model_has_an_id():
    for name, entry in ai_config.load()["models"].items():
        assert str(entry.get("id", "")).strip(), name
        assert ai_config.model(name), name


def test_retrieval_models_are_in_the_config():
    assert ai_config.model("embedding") == "intfloat/multilingual-e5-small"
    assert ai_config.model("reranker") == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_thresholds_are_read_by_dotted_name():
    assert ai_config.threshold("matching.area_tolerance") == pytest.approx(0.10)
    assert ai_config.threshold("evaluation.max_allowed_accuracy_drop") == pytest.approx(0.02)
    assert ai_config.threshold("furnisher.max_rounds") <= 3      # R4.1.3: at most 3 repair rounds


def test_unknown_names_fail_loudly():
    with pytest.raises(AIConfigError):
        ai_config.model("no_such_model")
    with pytest.raises(AIConfigError):
        ai_config.prompt("no_such_prompt")
    with pytest.raises(AIConfigError):
        ai_config.threshold("matching.no_such_threshold")


def test_env_variable_still_overrides_a_model(monkeypatch):
    monkeypatch.setenv("SMARTSEMSAR_GUARD_MODEL", "some/other-model")
    assert ai_config.model("input_guard") == "some/other-model"
    assert ai_config.model_default("input_guard") == "openai/gpt-oss-20b"


def test_a_candidate_config_file_can_be_loaded(tmp_path, monkeypatch):
    """The R4.5 gate evaluates a proposed config before it replaces the real one."""
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["version"] = "9.9.9"
    data["prompts"]["intent"]["text"] = "a weaker prompt"
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    real_version = ai_config.config_version()
    monkeypatch.setenv(ai_config.CONFIG_PATH_ENV, str(candidate))
    ai_config.reload()
    assert ai_config.prompt("intent") == "a weaker prompt"
    assert ai_config.config_version().startswith("9.9.9+")
    assert ai_config.config_version() != real_version


def test_trace_metadata_carries_prompt_model_and_config_versions():
    meta = ai_config.trace_metadata("intent", stage="detect_micro_intent")
    assert meta == {
        "stage": "detect_micro_intent",
        "prompt_name": "intent",
        "prompt_version": ai_config.prompt_version("intent"),
        "model_name": ai_config.model("intent"),
        "config_version": ai_config.config_version(),
    }


def test_code_reads_prompts_from_the_config():
    from backend.guardrails import input_guard
    from backend.services import llm_furnisher, vision_service

    assert input_guard.LLM_PROMPT == ai_config.prompt("input_guard")
    assert llm_furnisher.SYSTEM_PROMPT == ai_config.prompt("furniture_plan")
    assert vision_service.PROMPT == ai_config.prompt("furniture_vision")
    assert vision_service.DEFAULT_MODEL == ai_config.model_default("vision_gemini")
    assert llm_furnisher.MAX_ROUNDS == ai_config.threshold("furnisher.max_rounds")


# Lists of candidates that model_check.py probes, a price table and benchmark sample data:
# these name models on purpose and are not settings.
ALLOWED_MODEL_ID_FILES = {
    "backend/ai_config.py",
    "backend/services/model_check.py",
    "langsmith_ops/tracer.py",
    "langsmith_ops/latency_benchmark.py",
}


def _code_part(line: str) -> str:
    return line.split("#", 1)[0]


def test_no_model_id_is_hard_coded_outside_the_config():
    ids = {str(m["id"]) for m in ai_config.load()["models"].values()}
    ids = {i for i in ids if "/" in i or "-" in i}          # skip short ids like "small" / "qwen2.5"
    offenders = []
    for folder in ("backend", "langsmith_ops"):
        for path in (REPO / folder).rglob("*.py"):
            rel = path.relative_to(REPO).as_posix()
            if rel in ALLOWED_MODEL_ID_FILES or " copy" in path.name:
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                code = _code_part(line)
                if any(f'"{i}"' in code or f"'{i}'" in code for i in ids):
                    offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, "model IDs belong in config/ai_config.yaml:\n" + "\n".join(offenders)


def test_no_prompt_text_is_duplicated_in_code():
    """A prompt copied back into a .py file would silently bypass the versioned config."""
    offenders = []
    sources = {p: p.read_text(encoding="utf-8") for f in ("backend", "langsmith_ops")
               for p in (REPO / f).rglob("*.py") if " copy" not in p.name}
    for name, entry in ai_config.load()["prompts"].items():
        lines = [l.strip() for l in entry["text"].splitlines() if len(l.strip()) > 60]
        if not lines:
            continue
        probe = max(lines, key=len)
        for path, text in sources.items():
            if probe in text:
                offenders.append(f"{name}: {path.relative_to(REPO).as_posix()}")
    assert not offenders, "prompt text found in code:\n" + "\n".join(offenders)

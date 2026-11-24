"""
Fallback Chain Alerting & Logging Monitor for Smart Semsar
Logs structured alerts whenever a model in the fallback chain fails, times out, or is retired.
"""

import logging
import time
import json
from pathlib import Path
from typing import Optional

from .config import TRACES_DIR

ALERTS_LOG_FILE = TRACES_DIR / "fallback_alerts.log"

_logger = logging.getLogger("smart_semsar.fallback")
if not _logger.handlers:
    _logger.setLevel(logging.WARNING)
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _fh = logging.FileHandler(ALERTS_LOG_FILE, encoding="utf-8")
    _fh.setFormatter(_fmt)
    _logger.addHandler(_sh)
    _logger.addHandler(_fh)


RETIREMENT_INDICATORS = ("404", "retired", "deprecated", "decommissioned", "not found", "does not exist", "closed")
TRANSIENT_INDICATORS = ("429", "503", "502", "timeout", "overloaded", "rate limit", "busy")


def classify_model_failure(error: Exception | str) -> str:
    msg = str(error).lower()
    if any(ind in msg for ind in RETIREMENT_INDICATORS):
        return "RETIRED_OR_MISSING"
    if any(ind in msg for ind in TRANSIENT_INDICATORS):
        return "OVERLOADED_OR_TIMEOUT"
    return "MODEL_FAILURE"


def log_fallback_alert(
    stage_name: str,
    failed_model: str,
    error: Exception | str,
    next_model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Emits a structured warning/alert log line when a model in the fallback chain fails or is retired.
    """
    reason_code = classify_model_failure(error)
    err_preview = str(error).replace("\n", " ")[:180]
    next_target = next_model or "RULE_BASED_FALLBACK"

    alert_line = (
        f"[FALLBACK_ALERT] stage={stage_name} | status={reason_code} | "
        f"failed_model={failed_model} | fallback_to={next_target} | "
        f"session={session_id or 'n/a'} | error={err_preview}"
    )
    _logger.warning(alert_line)

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage_name": stage_name,
        "status": reason_code,
        "failed_model": failed_model,
        "fallback_to": next_target,
        "session_id": session_id,
        "error": err_preview,
    }
    try:
        with open(TRACES_DIR / "fallback_alerts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return record

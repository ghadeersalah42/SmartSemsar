"""
Furnishing and 3D metrics: one JSON line per event, summed up for the Round 2 report.
بيسجل كل قطعة 3D (وقتها، وهل جت من الكاش) وكل فرش اتعمل (عدى الـ validator ولا لأ وليه).

    record("model_3d", generator="colab", cache_hit=False, seconds=212.4, ok=True)
    record("layout", plan_id="PROP_1002", planner="llm", approved=True, failed_checks=[])
    summary = summarize(load())     # -> {"model_3d": {...}, "layout": {...}}

The log is data/metrics/furnishing.jsonl (not committed). SMARTSEMSAR_METRICS_FILE points it
elsewhere, "off" turns it off. Under pytest nothing is written unless that variable is set.

Usage:
    python -m backend.services.furnishing_metrics          # summary of the log so far
"""
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "data" / "metrics" / "furnishing.jsonl"
PATH_ENV = "SMARTSEMSAR_METRICS_FILE"


def log_path() -> Optional[Path]:
    value = os.environ.get(PATH_ENV, "").strip()
    if value.lower() == "off" or (not value and "PYTEST_CURRENT_TEST" in os.environ):
        return None
    return Path(value) if value else DEFAULT_PATH


def record(event: str, **fields) -> None:
    """Append one event. Never raises: a metrics problem must not break furnishing."""
    path = log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, "ts": round(time.time(), 2), **fields}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def load(path=None) -> list[dict]:
    path = Path(path) if path else log_path() or DEFAULT_PATH
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def percentile(values: list[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (q in 0..100); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(q / 100 * len(ordered) + 0.5) - 1))
    return ordered[k]


def summarize_3d(events: Iterable[dict]) -> dict:
    """Time per generated piece (cache misses that succeeded) and cache hit rate, per generator."""
    out = {}
    calls = [e for e in events if e.get("event") == "model_3d"]
    for gen in sorted({e.get("generator", "?") for e in calls}):
        mine = [e for e in calls if e.get("generator", "?") == gen]
        made = [e["seconds"] for e in mine if not e.get("cache_hit") and e.get("ok")]
        hits = sum(bool(e.get("cache_hit")) for e in mine)
        out[gen] = {
            "calls": len(mine),
            "cache_hits": hits,
            "cache_hit_rate": round(hits / len(mine), 3) if mine else None,
            "generated": len(made),
            "failed": sum(not e.get("ok") for e in mine),
            "seconds_mean": round(statistics.mean(made), 1) if made else None,
            "seconds_median": round(statistics.median(made), 1) if made else None,
            "seconds_p95": percentile(made, 95),
        }
    return out


def summarize_layouts(events: Iterable[dict]) -> dict:
    layouts = [e for e in events if e.get("event") == "layout"]
    reasons = Counter(c for e in layouts if not e.get("approved") for c in e.get("failed_checks", []))
    return {"layouts": len(layouts), "approved": sum(bool(e.get("approved")) for e in layouts),
            "rejection_reasons": dict(reasons.most_common())}


def summarize(events: list[dict]) -> dict:
    return {"model_3d": summarize_3d(events), "layout": summarize_layouts(events)}


if __name__ == "__main__":
    print(json.dumps(summarize(load(sys.argv[1] if len(sys.argv) > 1 else None)), indent=1))

"""
Layout benchmark: how many furnished layouts pass the validator on the first try, and why the rest fail.
بيفرش كل المخططات بكذا طلب، ويعد كام فرش عدى الـ validator من أول مرة، وأكتر أسباب الرفض.

"First try" means the planner's first arrangement, before any repair:
  rules: furniture_placer drops items when a door is cut off (keep_paths_open); a room where it had
         to drop something was rejected on the first try (reason "paths"). The final staging is
         then checked by staging_validator.
  llm:   llm_furnisher round 1. Every instruction is checked as it is applied; one that fails is a
         first-try rejection (reasons from its error text). Rounds 2-3 repair, the rules fill gaps.

Usage:
    python -m backend.eval.layout_benchmark                      # rules, all plans x all profiles
    python -m backend.eval.layout_benchmark --llm --limit 20     # also the LLM on 20 plans (uses quota)
    -> reports/furnishing/layout_benchmark.json and .md
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from backend.schema.plan import load_plan
from backend.schema.staging import REPO_ROOT, load_catalog
from backend.services.furniture_placer import stage_plan
from backend.services.staging_validator import validate_staging

PLANS_DIR = REPO_ROOT / "data" / "plans"
REPORT_DIR = REPO_ROOT / "reports" / "furnishing"

# the customer wishes every plan is furnished with (each is one layout per plan)
PROFILES = {
    "default": {},
    "modern_full": {"style": "modern", "density": "full"},
    "minimal": {"density": "minimal"},
    "classic_office": {"style": "classic", "must_have": ["desk"]},
    "dining_6_no_tv": {"dining_seats": 6, "exclude": ["tv"]},
}

# error text from furniture_actions -> short reason; first match wins, so specific ones come first
REASON_PATTERNS = [
    (r"goes outside the room or into a wall", "inside_room"),
    (r"overlaps a built-in", "fixtures_clear"),
    (r"blocks a door", "doors_clear"),
    (r"cuts off the way to a door", "paths"),
    (r"too tall to stand in front of a window", "window"),
    (r"no free space in front of it", "front_clearance"),
    (r"overlaps the ", "no_overlaps"),
    (r"does not fit on wall", "no_free_wall"),
    (r"does not fit in the middle|needs open floor", "no_open_floor"),
    (r"is open \(no wall\)|does not exist in", "bad_wall"),
    (r"nothing called", "bad_reference"),
    (r"is not in the catalog", "unknown_item"),
    (r"is not meant for a", "wrong_room_type"),
    (r"unknown place", "bad_instruction"),
    (r"is not in this plan", "unknown_room"),
]


def reasons_of(error: str) -> list[str]:
    """One error line from apply_actions -> the reason(s) it was rejected."""
    m = re.search(r"validator failed (.+)$", error)
    if m:
        return [r.strip() for r in m.group(1).split(",") if r.strip()]
    for pattern, reason in REASON_PATTERNS:
        if re.search(pattern, error):
            return [reason]
    return ["other"]


def run_rules(plan, catalog, profile: dict) -> dict:
    drops: dict[str, int] = {}
    started = time.perf_counter()
    staging = stage_plan(plan, catalog, preferences=profile, drops=drops)
    result = validate_staging(plan, staging, catalog)
    failed = [c["name"] for c in result["checks"] if not c["passed"]]
    reasons = ["paths"] * len(drops) + failed
    return {"first_try": not drops and result["approved"], "final": result["approved"],
            "reasons": reasons, "rooms_repaired": len(drops), "items_dropped": sum(drops.values()),
            "items": len(staging.items), "unplaced": len(staging.unplaced),
            "seconds": round(time.perf_counter() - started, 3)}


def run_llm(plan, catalog, profile: dict) -> dict:
    from backend.services.llm_furnisher import furnish
    started = time.perf_counter()
    staging, report = furnish(plan, profile, catalog)
    result = validate_staging(plan, staging, catalog)
    reasons = [r for errs in report.first_round_errors.values() for e in errs for r in reasons_of(e)]
    return {"first_try": report.first_try_approved, "final": result["approved"], "reasons": reasons,
            "rounds": report.rounds, "planner": report.planner, "rooms_by_rules": len(report.rule_rooms),
            "instructions_failed_round1": sum(len(v) for v in report.first_round_errors.values()),
            "items": len(staging.items), "seconds": round(time.perf_counter() - started, 1),
            "warnings": report.warnings}


def summarize(runs: list[dict]) -> dict:
    scored = [r for r in runs if r["first_try"] is not None]     # LLM runs where no model answered are left out
    first = sum(bool(r["first_try"]) for r in scored)
    final = sum(bool(r["final"]) for r in runs)
    reasons = Counter(x for r in runs for x in r["reasons"])
    total = sum(reasons.values())
    return {
        "layouts": len(runs),
        "scored": len(scored),
        "first_try_pass": first,
        "first_try_rate": round(first / len(scored), 3) if scored else None,
        "final_pass": final,
        "final_rate": round(final / len(runs), 3) if runs else None,
        "rejection_reasons": [{"reason": k, "count": n, "share": round(n / total, 3)}
                              for k, n in reasons.most_common()],
        "seconds_mean": round(sum(r["seconds"] for r in runs) / len(runs), 3) if runs else None,
    }


def to_markdown(summary: dict) -> str:
    lines = ["# Furnishing: layouts vs. the validator", ""]
    for planner, s in summary.items():
        if planner.startswith("_"):
            continue
        lines += [f"## {planner}", "",
                  f"- Layouts: {s['layouts']} ({summary['_meta']['plans']} plans x "
                  f"{len(summary['_meta']['profiles'])} customer profiles)" if planner == "rules"
                  else f"- Layouts: {s['layouts']} ({s['scored']} where a model answered)",
                  f"- **Pass on the first try: {s['first_try_pass']} / {s['scored']}"
                  f" ({(s['first_try_rate'] or 0) * 100:.1f} %)**",
                  f"- Pass after repair (what the user sees): {s['final_pass']} / {s['layouts']}"
                  f" ({(s['final_rate'] or 0) * 100:.1f} %)",
                  f"- Mean time per layout: {s['seconds_mean']} s", "",
                  "| Rejection reason | Count | Share |", "|---|---:|---:|"]
        lines += [f"| {r['reason']} | {r['count']} | {r['share'] * 100:.1f} % |" for r in s["rejection_reasons"]]
        if not s["rejection_reasons"]:
            lines.append("| (none) | 0 | - |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--llm", action="store_true", help="also run the LLM planner (uses API quota)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N plans")
    ap.add_argument("--profiles", default=",".join(PROFILES), help="comma-separated profile names")
    ap.add_argument("--out", default=str(REPORT_DIR))
    args = ap.parse_args(argv)

    catalog = load_catalog()
    plan_files = sorted(PLANS_DIR.glob("*.json"))[:args.limit]
    profiles = {k: PROFILES[k] for k in args.profiles.split(",") if k}
    planners = {"rules": run_rules, **({"llm": run_llm} if args.llm else {})}
    runs: dict[str, list[dict]] = {p: [] for p in planners}
    for path in plan_files:
        plan = load_plan(path)
        for pname, profile in profiles.items():
            for planner, run in planners.items():
                try:
                    r = run(plan, catalog, profile)
                except Exception as e:      # a crash is a rejection too, and must show up in the report
                    r = {"first_try": False, "final": False, "reasons": [f"crash:{type(e).__name__}"], "seconds": 0.0}
                runs[planner].append({"plan": path.stem, "profile": pname, **r})
        print(f"{path.stem}: " + "  ".join(
            f"{p} {sum(bool(r['first_try']) for r in runs[p][-len(profiles):])}/{len(profiles)}" for p in planners))

    summary = {p: summarize(rs) for p, rs in runs.items()}
    summary["_meta"] = {"plans": len(plan_files), "profiles": profiles, "date": time.strftime("%Y-%m-%d %H:%M")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "layout_benchmark.json").write_text(
        json.dumps({"summary": summary, "runs": runs}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "layout_benchmark.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))
    return summary


if __name__ == "__main__":
    main()

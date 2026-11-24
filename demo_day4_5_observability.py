"""
Smart Semsar — Day 4-5 Observability, Latency, Cost & Fallback Alerting Demo
"""

import sys
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langsmith_ops.latency_benchmark import run_50_session_benchmark
from langsmith_ops.fallback_monitor import log_fallback_alert, ALERTS_LOG_FILE
from langsmith_ops.merged_router import run_merged_router_streaming
from langsmith_ops.tracer import trace_full_session


def main():
    print("=" * 90)
    print("DAY 4-5: OBSERVABILITY, LATENCY (50 RUNS), COST IN EGP & IMPROVEMENTS")
    print("=" * 90)

    report = run_50_session_benchmark()

    # 1. Single Full Session Trace Tree
    sess = report["sample_full_session_trace"]
    print(f"\n[1] SINGLE FULL SESSION TRACE (trace_id={sess['trace_id'][:8]}..., session={sess['session_id']})")
    print(f"    User Query : {sess['user_query']}")
    print(f"    Total Cost : EGP {sess['total_cost_egp']:.4f} | Total Tokens: {sess['total_tokens']}")
    tree_rows = []
    for st in sess["stages"]:
        tree_rows.append([
            st["stage_name"],
            st["prompt_version"],
            st["model_answered"],
            f"{st['latency_ms']:.1f} ms",
            f"{st['ttft_ms']:.1f} ms" if st["ttft_ms"] else "-",
            st["total_tokens"],
            f"EGP {st['cost_egp']:.5f}",
        ])
    print(tabulate(tree_rows, headers=["Stage Name", "Prompt Ver", "Model Answered", "Latency", "TTFT", "Tokens", "Cost (EGP)"], tablefmt="fancy_grid"))

    # 2. Per-Stage P50, P95, P99 & TTFT (50 Runs)
    print("\n[2] PER-STAGE LATENCY & TTFT ACROSS 50 RUNS")
    stage_rows = []
    for sname, metrics in report["per_stage_latency"].items():
        stage_rows.append([
            sname,
            f"{metrics['p50_ms']:.1f} ms",
            f"{metrics['p95_ms']:.1f} ms",
            f"{metrics['p99_ms']:.1f} ms",
            f"{metrics['ttft_p50_ms']:.1f} ms" if metrics["ttft_p50_ms"] else "-",
            f"{metrics['ttft_p95_ms']:.1f} ms" if metrics["ttft_p95_ms"] else "-",
        ])
    print(tabulate(stage_rows, headers=["Stage", "P50", "P95", "P99", "TTFT P50", "TTFT P95"], tablefmt="fancy_grid"))

    # 3. 10.5s P99 Explanation
    p99_info = report["p99_explanation"]
    print(f"\n[3] 10.5s P99 ROOT-CAUSE ANALYSIS (Observed Session P99: {p99_info['observed_p99_s']}s)")
    print(f"    Culprit Stage : {p99_info['culprit_stage']}")
    print(f"    Explanation   : {p99_info['reason']}")

    # 4. Cost per Session in EGP vs Deck EGP 5.2
    cost = report["cost_analysis_egp"]
    print("\n[4] REAL TOKEN COST PER SESSION (EGP) VS DECK (EGP 5.20)")
    print(f"    Avg Tokens / Session : {cost['avg_tokens_per_session']}")
    print(f"    Actual Avg Cost      : EGP {cost['avg_cost_egp']:.4f}  (P95: EGP {cost['p95_cost_egp']:.4f})")
    print(f"    Deck Benchmark       : EGP {cost['deck_assumed_cost_egp']:.2f}")
    print(f"    Savings vs Deck      : {cost['savings_vs_deck_pct']}% (EGP {cost['difference_egp']:.4f} cheaper per session)")

    # 5. Improvement: Streaming + Merged Supervisor/Intent
    imp = report["improvement_comparison"]
    print("\n[5] IMPROVEMENT: STREAMING + MERGED SUPERVISOR & INTENT CALL")
    comp_rows = [
        ["Before (Separate + Non-Streaming)", f"{imp['router_before_separate_ms']['p50']} ms", f"{imp['router_before_separate_ms']['p95']} ms", f"{imp['router_before_separate_ms']['ttft_p50']} ms"],
        ["After (Merged + Streaming)", f"{imp['router_after_merged_streaming_ms']['p50']} ms", f"{imp['router_after_merged_streaming_ms']['p95']} ms", f"{imp['router_after_merged_streaming_ms']['ttft_p50']} ms"],
    ]
    print(tabulate(comp_rows, headers=["Configuration", "Router P50", "Router P95", "TTFT P50"], tablefmt="fancy_grid"))
    print(f"    Recommendation: {imp['architectural_recommendation']}")

    # 6. Fallback Alert Demo
    print("\n[6] FALLBACK CHAIN ALERT LOGGING (RETIRED / FAILED MODEL)")
    log_fallback_alert(
        stage_name="model_check",
        failed_model="llama-3.3-70b-versatile",
        error="404 Not Found: model has been retired/decommissioned",
        next_model="openai/gpt-oss-120b",
        session_id="demo_session",
    )
    print(f"    Alert log written to: {ALERTS_LOG_FILE}")
    print("\n[✓] Full report saved to: langsmith_ops/reports/day4_5_observability_report.json")


if __name__ == "__main__":
    main()

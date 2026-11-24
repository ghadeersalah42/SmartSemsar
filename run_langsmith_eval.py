"""
CLI Runner for Smart Semsar LangSmith LLMOps
Fulfills the core requirements:
1. Version every prompt and tag each trace with the prompt version and the model that answered.
2. Run an eval set automatically on every prompt change.
"""

import sys
import argparse
import json
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langsmith_ops.config import TRACES_DIR
from langsmith_ops.eval_pipeline import LangSmithEvalRunner
from langsmith_ops.prompt_hub import get_prompt_hub


def run_cli():
    parser = argparse.ArgumentParser(description="Smart Semsar LangSmith Evaluation Runner")
    parser.add_argument("--agent", type=str, default=None, help="Agent to evaluate (e.g., 'intent_detector')")
    parser.add_argument("--version", type=str, default="v1.1.0", help="Prompt version to evaluate")
    parser.add_argument("--baseline", type=str, default="v1.0.0", help="Baseline version for regression check")
    parser.add_argument("--all", action="store_true", help="Run evaluations on all agents")
    parser.add_argument("--traces", action="store_true", help="Show recent LangSmith execution traces")
    args = parser.parse_args()

    runner = LangSmithEvalRunner()
    hub = get_prompt_hub()

    if args.traces:
        print("\n" + "=" * 90)
        print("LANGSMITH OBSERVABILITY: RECENT TRACES (Prompt Version & Model Tags)")
        print("=" * 90)
        traces_file = TRACES_DIR / "traces.jsonl"
        if not traces_file.exists():
            print("No traces logged yet.")
            return

        traces = []
        with open(traces_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        traces.append(json.loads(line))
                    except Exception:
                        pass

        table = []
        for t in traces[-15:]:
            table.append([
                t.get("trace_id", "")[:8],
                t.get("name", ""),
                t.get("prompt_version", ""),
                t.get("model_name", ""),
                f"{t.get('latency_ms', 0):.1f} ms",
                t.get("status", ""),
                str(t.get("input", ""))[:35]
            ])
        headers = ["Trace ID", "Agent / Run", "Prompt Ver", "Model Answered", "Latency", "Status", "Input Preview"]
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
        print(f"Total Traces Logged: {len(traces)}")
        return

    print("\n" + "=" * 90)
    print("SMART SEMSAR - LANGSMITH AUTOMATED PROMPT EVALUATION RUNNER")
    print("Requirement: Run an eval set automatically on every prompt change")
    print("=" * 90)

    if args.agent:
        agents = [args.agent]
    else:
        agents = ["intent_detector", "orchestrator_router", "customer_requirements"]

    summary_rows = []
    for ag in agents:
        try:
            ver = args.version if ag == "intent_detector" else "v1.0.0"
            base = args.baseline if ag == "intent_detector" else None
            rep = runner.run_agent_eval(ag, prompt_version=ver, baseline_version=base)
            delta = rep.get("accuracy_delta") or "-"
            summary_rows.append([
                rep["agent_name"],
                rep["prompt_version"],
                rep["model_name"],
                rep["dataset_name"],
                rep["accuracy_pct"],
                delta,
                "✓ PASS" if rep["status"] == "PASS" else "✗ FAIL"
            ])
        except Exception as e:
            summary_rows.append([ag, args.version, "-", "-", "ERROR", "-", f"Failed: {e}"])

    headers = [
        "Agent Name",
        "Prompt Version",
        "Model Evaluated",
        "LangSmith Dataset",
        "Eval Accuracy",
        "Baseline Delta",
        "Status"
    ]
    print("\n" + tabulate(summary_rows, headers=headers, tablefmt="fancy_grid"))
    print("\n[✓] Reports saved to: langsmith_ops/reports/")
    print("[✓] Traces streamed to: langsmith_ops/traces/traces.jsonl (and LangSmith Cloud Dashboard)\n")


if __name__ == "__main__":
    run_cli()

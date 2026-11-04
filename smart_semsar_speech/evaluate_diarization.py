"""
Evaluation Script: Speaker Diarization & Broker vs. Client Role Separation
Fixed Solutions - Smart Semsar Benchmark
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from tabulate import tabulate
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ANNOTATIONS_DIR,
    RAW_CALLS_DIR,
    REPORTS_DIR,
)
from src.role_separator import RoleSeparator
from src.diarization_evaluator import (
    evaluate_der,
    evaluate_broker_client_separation,
    evaluate_corpus_diarization
)


def load_call_samples(annotations_dir: Path) -> List[Dict[str, Any]]:
    """Loads all call ground truth annotations."""
    samples = []
    json_files = sorted(list(annotations_dir.glob("*.json")))
    
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples.append(data)
        except Exception as e:
            print(f"Error reading {jf}: {e}")
            
    return samples


def run_diarization_separation_benchmark(
    annotations_dir: Path,
    output_report_path: Path
) -> Dict[str, Any]:
    """
    Executes benchmark of how accurately the system separates the broker from the client.
    """
    raw_samples = load_call_samples(annotations_dir)
    if not raw_samples:
        print(f"No annotations found in {annotations_dir}")
        return {}
        
    print("=" * 80)
    print("FIXED SOLUTIONS - SMART SEMSAR SEPARATION BENCHMARK")
    print("Task: Report how accurately the system separates the broker from the client")
    print(f"Total Call Samples Evaluated: {len(raw_samples)}")
    print("=" * 80)
    
    role_separator = RoleSeparator()
    corpus_eval_samples = []
    
    per_call_table = []
    
    for s in raw_samples:
        call_id = s.get("call_id", "call")
        duration = float(s.get("duration", 15.0))
        ref_turns = s.get("reference_turns", [])
        
        # Test predicted turns: If predicted_turns exist, use them;
        # otherwise run role separator on reference turns with anonymous speaker labels
        hyp_turns = s.get("predicted_turns", [])
        if not hyp_turns:
            # Run role separator on speech turns
            anon_turns = []
            for t in ref_turns:
                anon_turns.append({
                    "start": t["start"],
                    "end": t["end"],
                    "speaker": t.get("speaker", "SPEAKER_00"),
                    "text": t.get("text", "")
                })
            sep_out = role_separator.separate_roles(anon_turns)
            hyp_turns = sep_out["assigned_turns"]
            
        corpus_eval_samples.append({
            "call_id": call_id,
            "duration": duration,
            "reference_turns": ref_turns,
            "predicted_turns": hyp_turns
        })
        
    # Evaluate full corpus
    benchmark_results = evaluate_corpus_diarization(corpus_eval_samples)
    summary = benchmark_results["summary"]
    
    for pc in benchmark_results["per_call_details"]:
        cm = pc["confusion_matrix"]
        per_call_table.append([
            pc["call_id"],
            f"{pc['duration_sec']:.1f}s",
            pc["der_pct"],
            pc["separation_accuracy_pct"],
            "✓ Correct" if pc["role_classification_correct"] else "✗ Inverted",
            pc["broker_f1_pct"],
            pc["client_f1_pct"],
            pc["macro_f1_pct"],
        ])
        
    headers = [
        "Call ID",
        "Duration",
        "DER (Diarization)",
        "Separation Acc",
        "Role Assignment",
        "Broker F1",
        "Client F1",
        "Macro F1"
    ]
    print("\n" + tabulate(per_call_table, headers=headers, tablefmt="fancy_grid"))
    
    # Executive Summary Card
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY: BROKER VS. CLIENT SEPARATION ACCURACY")
    print("=" * 80)
    print(f"• Call-Level Role Assignment Accuracy  : {summary['call_level_role_accuracy_pct']}")
    print(f"• Time-Weighted Separation Accuracy     : {summary['time_weighted_separation_accuracy_pct']}")
    print(f"• Overall Diarization Error Rate (DER)  : {summary['average_der_pct']}")
    print(f"• Macro F1-Score (Broker + Client)      : {summary['macro_f1_pct']}")
    print(f"• Broker Detection F1-Score             : {summary['broker_f1_pct']}")
    print(f"• Client Detection F1-Score             : {summary['client_f1_pct']}")
    print("-" * 80)
    
    # Confusion Matrix Table
    agg_cm = summary["aggregate_confusion_matrix"]
    total_broker_sec = agg_cm["broker_as_broker_sec"] + agg_cm["broker_as_client_sec"]
    total_client_sec = agg_cm["client_as_client_sec"] + agg_cm["client_as_broker_sec"]
    
    cm_table = [
        [
            "True Broker (سمسار)",
            f"{agg_cm['broker_as_broker_sec']:.2f}s ({(agg_cm['broker_as_broker_sec']/max(0.1, total_broker_sec))*100:.1f}%)",
            f"{agg_cm['broker_as_client_sec']:.2f}s ({(agg_cm['broker_as_client_sec']/max(0.1, total_broker_sec))*100:.1f}%)"
        ],
        [
            "True Client (عميل)",
            f"{agg_cm['client_as_broker_sec']:.2f}s ({(agg_cm['client_as_broker_sec']/max(0.1, total_client_sec))*100:.1f}%)",
            f"{agg_cm['client_as_client_sec']:.2f}s ({(agg_cm['client_as_client_sec']/max(0.1, total_client_sec))*100:.1f}%)"
        ]
    ]
    cm_headers = ["Actual \\ Predicted", "Classified as Broker", "Classified as Client"]
    print("CONFUSION MATRIX (Speech Duration in Seconds & Percentage):")
    print(tabulate(cm_table, headers=cm_headers, tablefmt="fancy_grid"))
    print("=" * 80)
    
    # Save Report
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = {
        "benchmark_task": "Report how accurately the system separates the broker from the client",
        "dataset": "Smart Semsar Real Estate Calls & MagicHub Corpus",
        "summary": summary,
        "confusion_matrix": cm_table,
        "per_call_details": benchmark_results["per_call_details"]
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
    csv_path = output_report_path.with_suffix(".csv")
    import pandas as pd
    df = pd.DataFrame(per_call_table, columns=headers)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    print(f"\n[✓] JSON report saved: {output_report_path}")
    print(f"[✓] CSV report saved : {csv_path}")
    return report_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Broker vs Client Separation Accuracy")
    parser.add_argument("--annotations_dir", type=Path, default=ANNOTATIONS_DIR, help="Path to annotations folder")
    parser.add_argument("--output_file", type=Path, default=REPORTS_DIR / "diarization_role_separation_report.json", help="Report output path")
    args = parser.parse_args()
    
    run_diarization_separation_benchmark(args.annotations_dir, args.output_file)

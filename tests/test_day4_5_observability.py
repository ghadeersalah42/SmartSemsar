"""
Unit Tests for Day 4-5 Observability, Latency, Cost & Fallback Alerting
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langsmith_ops.tracer import trace_full_session, langsmith_trace, build_langchain_config, compute_cost_egp
from langsmith_ops.fallback_monitor import log_fallback_alert, classify_model_failure
from langsmith_ops.merged_router import run_merged_router_streaming
from langsmith_ops.latency_benchmark import run_50_session_benchmark


class TestDay45Observability(unittest.TestCase):

    def test_stage_and_actual_model_tagging(self):
        cfg = build_langchain_config(
            agent_name="intent_detector",
            prompt_version="v1.1.0",
            model_name="openai/gpt-oss-20b",
            stage_name="detect_micro_intent",
            session_id="sess_test_1",
        )
        self.assertEqual(cfg["metadata"]["stage_name"], "detect_micro_intent")
        self.assertIn("detect_micro_intent", cfg["tags"])
        self.assertIn("v1.1.0", cfg["tags"])
        self.assertIn("openai/gpt-oss-20b", cfg["tags"])

    def test_single_session_trace_wraps_stages(self):
        with trace_full_session(session_id="sess_unit", user_query="عايز شقة في التجمع") as sess:
            res = run_merged_router_streaming("عايز شقة في التجمع", stream=True, actual_model="openai/gpt-oss-20b")
            self.assertEqual(res["macro_intent"], "search_by_reqs")
        self.assertEqual(len(sess.stages), 1)
        self.assertEqual(sess.stages[0]["stage_name"], "merged_supervisor_intent")
        self.assertGreater(sess.total_cost_egp, 0.0)

    def test_fallback_alert_classification(self):
        self.assertEqual(classify_model_failure("404 model retired"), "RETIRED_OR_MISSING")
        self.assertEqual(classify_model_failure("503 service overloaded"), "OVERLOADED_OR_TIMEOUT")
        rec = log_fallback_alert("llm_furnisher", "gemini-2.5-flash", "404 retired", "openai/gpt-oss-120b")
        self.assertEqual(rec["status"], "RETIRED_OR_MISSING")

    def test_50_run_benchmark_and_cost_egp(self):
        report = run_50_session_benchmark()
        self.assertEqual(report["total_runs"], 50)
        self.assertIn("llm_furnisher", report["per_stage_latency"])
        self.assertLess(report["cost_analysis_egp"]["avg_cost_egp"], 5.20)
        self.assertGreaterEqual(report["session_latency_before_ms"]["p99"], 10000.0)


if __name__ == "__main__":
    unittest.main()

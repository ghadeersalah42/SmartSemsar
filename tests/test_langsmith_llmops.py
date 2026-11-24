"""
Unit Tests for Smart Semsar LangSmith LLMOps Integration
Verifies:
1. Version every prompt in LangSmith Prompt Hub.
2. Tag each trace with prompt_version and the model that answered.
3. Run an eval set automatically on prompt changes using evaluate().
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langsmith_ops.prompt_hub import get_prompt_hub
from langsmith_ops.tracer import langsmith_trace, build_langchain_config
from langsmith_ops.eval_pipeline import LangSmithEvalRunner
from langsmith_ops.compat import Client


class TestLangSmithLLMOps(unittest.TestCase):

    def setUp(self):
        self.hub = get_prompt_hub()
        self.runner = LangSmithEvalRunner()

    def test_prompt_hub_versioning(self):
        # Push version
        identifier = self.hub.push_prompt(
            agent_name="test_agent",
            system_prompt="Test Prompt v1.0",
            version="v1.0.0",
            description="Unit test prompt"
        )
        self.assertIn("v1.0.0", identifier)

        # Pull version
        pulled = self.hub.pull_prompt("test_agent", version="v1.0.0")
        self.assertEqual(pulled["version"], "v1.0.0")
        self.assertIn("Test Prompt", pulled["system_prompt"])

    def test_trace_tagging_with_version_and_model(self):
        @langsmith_trace(
            agent_name="intent_agent",
            prompt_version="v1.1.0",
            default_model="openai/gpt-oss-20b"
        )
        def sample_agent(query: str):
            return {"intent": "new_search", "query": query}

        res = sample_agent("شقة للبيع")
        self.assertEqual(res["intent"], "new_search")

        # Test LangChain config generator
        config = build_langchain_config("intent_agent", "v1.1.0", "openai/gpt-oss-20b")
        self.assertEqual(config["metadata"]["prompt_version"], "v1.1.0")
        self.assertEqual(config["metadata"]["model_name"], "openai/gpt-oss-20b")
        self.assertIn("v1.1.0", config["tags"])

    def test_automated_eval_run_on_dataset(self):
        eval_result = self.runner.run_agent_eval("intent_detector", prompt_version="v1.1.0")
        self.assertEqual(eval_result["agent_name"], "intent_detector")
        self.assertEqual(eval_result["prompt_version"], "v1.1.0")
        self.assertEqual(eval_result["status"], "PASS")
        self.assertGreaterEqual(eval_result["accuracy_score"], 0.90)


if __name__ == "__main__":
    unittest.main()

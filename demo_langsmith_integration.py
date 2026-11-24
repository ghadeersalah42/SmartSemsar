"""
Smart Semsar - LangSmith Practical Integration Demo
Demonstrates:
1. Versioning prompts via LangSmith Hub.
2. Tagging traces with prompt_version and model_name.
3. Automatically running an evaluation set on prompt changes.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langsmith_ops.prompt_hub import get_prompt_hub
from langsmith_ops.tracer import langsmith_trace, build_langchain_config
from langsmith_ops.eval_pipeline import LangSmithEvalRunner
from langsmith_ops.compat import Client


print("=" * 80)
print("DEMO 1: LANGSMITH PROMPT HUB — PULLING VERSIONED PROMPT")
print("=" * 80)

hub = get_prompt_hub()
prompt_v1_1 = hub.pull_prompt("intent_detector", version="v1.1.0")

print(f"Agent Name    : {prompt_v1_1.get('agent_name')}")
print(f"Version Tag   : {prompt_v1_1.get('version')}  <-- Requirement 1: Version every prompt")
print(f"Target Model  : {prompt_v1_1.get('target_model')}")
print(f"System Prompt : {prompt_v1_1.get('system_prompt')}\n")


print("=" * 80)
print("DEMO 2: AGENT EXECUTION WITH LANGSMITH TRACE TAGGING")
print("=" * 80)

@langsmith_trace(
    agent_name="intent_detector",
    prompt_version=prompt_v1_1.get("version", "v1.1.0"),
    default_model=prompt_v1_1.get("target_model", "openai/gpt-oss-20b")
)
def smart_semsar_intent_agent(user_utterance: str) -> dict:
    # This runs the agent and automatically sends the trace to LangSmith
    return {
        "intent": "alternative_search",
        "confidence": 0.95,
        "reason": "Customer requests viewing alternative properties without changing requirements."
    }

response = smart_semsar_intent_agent("عايز اشوف شقق تانية غير دي")
print(f"Agent Response: {response}")

# Check LangChain RunnableConfig helper
config = build_langchain_config(
    agent_name="intent_detector",
    prompt_version="v1.1.0",
    model_name="openai/gpt-oss-20b",
    session_id="call_00_session"
)
print("\nLangChain RunnableConfig for LangSmith Cloud:")
print(f"  • metadata: {config['metadata']}")
print(f"  • tags    : {config['tags']}  <-- Requirement 1: Tag each trace with prompt version & model\n")


print("=" * 80)
print("DEMO 3: AUTOMATED EVAL SET RUNNER VIA LANGSMITH EVALUATE()")
print("=" * 80)

runner = LangSmithEvalRunner()
eval_summary = runner.run_agent_eval(
    agent_name="intent_detector",
    prompt_version="v1.1.0",
    baseline_version="v1.0.0"
)

print(f"Evaluated Agent   : {eval_summary['agent_name']}")
print(f"Prompt Version    : {eval_summary['prompt_version']}")
print(f"Model Answered    : {eval_summary['model_name']}")
print(f"Dataset Evaluated : {eval_summary['dataset_name']}")
print(f"Accuracy Score    : {eval_summary['accuracy_pct']}")
print(f"Baseline Delta    : {eval_summary['accuracy_delta'] or '+0.00%'}")
print(f"CI/CD Verdict     : {eval_summary['status']}  <-- Requirement 2: Run eval set on prompt change")
print("=" * 80)
print("LangSmith LLMOps integration completed successfully!")

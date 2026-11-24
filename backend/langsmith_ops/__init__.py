"""
Smart Semsar - LangSmith LLMOps & Prompt Versioning Module
Provides:
- LangSmith Prompt Hub integration (push/pull versioned prompts)
- Trace tagging decorator (@langsmith_trace) and RunnableConfig builder (tagging prompt_version & model_name)
- Automated evaluation pipeline (langsmith.evaluate() with golden datasets & regression detection)
- Dual-Mode: Live LangSmith Cloud or Offline local fallback engine
"""

from .prompt_hub import LangSmithPromptHub, get_prompt_hub
from .tracer import langsmith_trace, build_langchain_config
from .eval_pipeline import LangSmithEvalRunner
from .compat import Client, evaluate, traceable, IS_LIVE_LANGSMITH

__all__ = [
    "LangSmithPromptHub",
    "get_prompt_hub",
    "langsmith_trace",
    "build_langchain_config",
    "LangSmithEvalRunner",
    "Client",
    "evaluate",
    "traceable",
    "IS_LIVE_LANGSMITH"
]

"""
LangSmith Tracing & Metadata Tagging Decorator
Fulfills the core requirement:
"Tag each trace with the prompt version and the model that answered."
"""

import functools
from typing import Dict, Any, Optional, Callable
from .compat import traceable


def langsmith_trace(
    agent_name: str,
    prompt_version: str = "v1.0.0",
    default_model: str = "openai/gpt-oss-20b"
):
    """
    Decorator that instruments any agent call for LangSmith,
    tagging it with prompt_version and the model that answered.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            model_used = kwargs.get("model_name", default_model)
            version_used = kwargs.get("prompt_version", prompt_version)

            meta = {
                "agent_name": agent_name,
                "prompt_version": version_used,
                "model_name": model_used
            }
            tags_list = [agent_name, version_used, model_used]

            # Wrap execution with traceable
            traced_func = traceable(
                name=f"{agent_name}_execution",
                metadata=meta,
                tags=tags_list
            )(func)

            return traced_func(*args, **kwargs)
        return wrapper
    return decorator


def build_langchain_config(
    agent_name: str,
    prompt_version: str = "v1.0.0",
    model_name: str = "openai/gpt-oss-20b",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs a LangChain / LangGraph RunnableConfig dict with
    prompt_version and model_name tags for LangSmith.
    """
    return {
        "metadata": {
            "agent_name": agent_name,
            "prompt_version": prompt_version,
            "model_name": model_name,
            "session_id": session_id or "default_session"
        },
        "tags": [agent_name, prompt_version, model_name]
    }

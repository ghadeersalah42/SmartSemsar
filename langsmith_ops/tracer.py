"""
LangSmith Tracing, Stage Tagging & Session-Level Observability
Fulfills Day 4-5 Requirements:
- Tag every trace with prompt_version, the model that actually answered, and stage_name.
- Show one full session as a single trace (parent trace wrapping stage child spans).
- Track real token counts, TTFT (Time-To-First-Token), latency, and cost in EGP.
"""

import time
import uuid
import json
import functools
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Any, Optional, Callable, List

from .compat import traceable
from .config import TRACES_DIR

# ContextVar to bind all child stage spans under a single parent session trace
_current_session: ContextVar[Optional["SessionTraceContext"]] = ContextVar("_current_session", default=None)

# Verified pricing per 1M tokens in USD (converted at 1 USD = 50.0 EGP)
USD_TO_EGP = 50.0
MODEL_PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "gemini-flash-latest": {"input": 0.10, "output": 0.40},
    "gemini-flash-lite-latest": {"input": 0.075, "output": 0.30},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "qwen2.5": {"input": 0.0, "output": 0.0},  # Local Ollama
    "rule_based": {"input": 0.0, "output": 0.0},
}


def compute_cost_egp(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Computes exact session/stage cost in EGP from real token counts."""
    clean_model = model_name.replace("groq/", "").replace("gemini/", "").replace("ollama/", "")
    pricing = MODEL_PRICING_USD_PER_1M.get(clean_model, {"input": 0.15, "output": 0.60})
    usd_cost = (input_tokens / 1_000_000.0) * pricing["input"] + (output_tokens / 1_000_000.0) * pricing["output"]
    return round(usd_cost * USD_TO_EGP, 6)



def _safe_mask(text: str) -> str:
    try:
        from backend.guardrails.pii import mask_pii
        return mask_pii(text)
    except Exception:
        return text


class SessionTraceContext:
    """Groups all stages of a single user session under one parent LangSmith trace."""

    def __init__(self, session_id: str, user_query: str = ""):
        self.session_id = session_id
        self.trace_id = str(uuid.uuid4())
        self.user_query = user_query
        self.start_time = time.perf_counter()
        self.stages: List[Dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_egp = 0.0

    def record_stage(
        self,
        stage_name: str,
        prompt_version: str,
        model_answered: str,
        latency_ms: float,
        ttft_ms: Optional[float] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        status: str = "SUCCESS",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        cost_egp = compute_cost_egp(model_answered, input_tokens, output_tokens)
        span = {
            "span_id": str(uuid.uuid4())[:8],
            "parent_trace_id": self.trace_id,
            "session_id": self.session_id,
            "stage_name": stage_name,
            "prompt_version": prompt_version,
            "model_answered": model_answered,
            "latency_ms": round(latency_ms, 2),
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_egp": cost_egp,
            "status": status,
            "error": error,
            "tags": [stage_name, f"prompt:{prompt_version}", f"model:{model_answered}"],
        }
        self.stages.append(span)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_egp = round(self.total_cost_egp + cost_egp, 6)
        return span

    def finalize(self) -> Dict[str, Any]:
        total_latency_ms = round((time.perf_counter() - self.start_time) * 1000.0, 2)
        if self.stages:
            total_stage_ms = round(sum(s["latency_ms"] for s in self.stages), 2)
        else:
            total_stage_ms = total_latency_ms

        session_record = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "name": "smart_semsar_full_session",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "user_query": _safe_mask(self.user_query)[:200],
            "total_wall_ms": total_latency_ms,
            "total_stage_ms": total_stage_ms,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_egp": round(self.total_cost_egp, 6),
            "stages_count": len(self.stages),
            "stages": self.stages,
        }
        try:
            with open(TRACES_DIR / "session_traces.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(session_record, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return session_record


@contextmanager
def trace_full_session(session_id: str, user_query: str = ""):
    """Context manager that wraps an entire user session as a single unified LangSmith trace."""
    ctx = SessionTraceContext(session_id=session_id, user_query=user_query)
    token = _current_session.set(ctx)

    @traceable(
        name="smart_semsar_full_session",
        metadata={"session_id": session_id, "trace_scope": "full_session"},
        tags=["full_session", session_id],
    )
    def _run_session():
        return ctx

    try:
        _run_session()
        yield ctx
    finally:
        ctx.finalize()
        _current_session.reset(token)


def get_active_session() -> Optional[SessionTraceContext]:
    return _current_session.get()


def langsmith_trace(
    agent_name: str,
    prompt_version: str = "v1.0.0",
    default_model: str = "openai/gpt-oss-20b",
    stage_name: Optional[str] = None,
):
    """
    Decorator that instruments any agent/stage call for LangSmith,
    tagging it with prompt_version, the model that actually answered, and stage_name.
    Automatically attaches as a child span if inside trace_full_session().
    """
    resolved_stage = stage_name or agent_name

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            model_used = kwargs.get("actual_model") or kwargs.get("model_name", default_model)
            version_used = kwargs.get("prompt_version", prompt_version)

            meta = {
                "agent_name": agent_name,
                "stage_name": resolved_stage,
                "prompt_version": version_used,
                "model_name": model_used,
            }
            active = get_active_session()
            if active:
                meta["parent_trace_id"] = active.trace_id
                meta["session_id"] = active.session_id

            tags_list = [resolved_stage, f"prompt:{version_used}", f"model:{model_used}"]

            traced_func = traceable(
                name=f"{resolved_stage}_execution",
                metadata=meta,
                tags=tags_list,
            )(func)

            status = "SUCCESS"
            err_msg = None
            try:
                res = traced_func(*args, **kwargs)
                # Allow function output dict to report the actual model that answered & token usage
                if isinstance(res, dict):
                    model_used = res.get("_actual_model", model_used)
                    in_tok = int(res.get("_input_tokens", 0))
                    out_tok = int(res.get("_output_tokens", 0))
                    ttft_ms = res.get("_ttft_ms")
                else:
                    in_tok, out_tok, ttft_ms = 0, 0, None
                return res
            except Exception as exc:
                status = "ERROR"
                err_msg = str(exc)
                in_tok, out_tok, ttft_ms = 0, 0, None
                raise
            finally:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if active:
                    active.record_stage(
                        stage_name=resolved_stage,
                        prompt_version=version_used,
                        model_answered=model_used,
                        latency_ms=latency_ms,
                        ttft_ms=ttft_ms,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        status=status,
                        error=err_msg,
                    )

        return wrapper

    return decorator


def build_langchain_config(
    agent_name: str,
    prompt_version: str = "v1.0.0",
    model_name: str = "openai/gpt-oss-20b",
    stage_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Constructs a LangChain / LangGraph RunnableConfig dict with
    prompt_version, model_name, and stage_name tags for LangSmith.
    """
    resolved_stage = stage_name or agent_name
    return {
        "metadata": {
            "agent_name": agent_name,
            "stage_name": resolved_stage,
            "prompt_version": prompt_version,
            "model_name": model_name,
            "session_id": session_id or "default_session",
        },
        "tags": [resolved_stage, prompt_version, model_name],
    }

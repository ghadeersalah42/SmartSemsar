"""
LangSmith Compatibility & Bridge Layer
Provides seamless transition between real LangSmith Cloud and an offline local engine.
"""

import os
import json
import time
import uuid
import functools
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from .config import TRACES_DIR, LANGCHAIN_API_KEY


class Example:
    """Emulates LangSmith Example object."""
    def __init__(self, inputs: Dict[str, Any], outputs: Dict[str, Any], id: Optional[str] = None):
        self.id = id or str(uuid.uuid4())
        self.inputs = inputs
        self.outputs = outputs


class Run:
    """Emulates LangSmith Run execution object."""
    def __init__(self, inputs: Dict[str, Any], outputs: Dict[str, Any], id: Optional[str] = None):
        self.id = id or str(uuid.uuid4())
        self.inputs = inputs
        self.outputs = outputs


class OfflineLangSmithClient:
    """
    Offline/Local Client emulating LangSmith Client interface:
    - push_prompt / pull_prompt (stores/retrieves versioned prompts locally in JSON/YAML)
    - create_dataset / has_dataset / create_examples
    - list_runs / query traces
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or LANGCHAIN_API_KEY
        self.base_dir = Path(__file__).resolve().parent
        self.prompts_dir = self.base_dir / "prompts"
        self.datasets_dir = self.base_dir / "golden_datasets"
        self.traces_file = TRACES_DIR / "traces.jsonl"
        self._datasets: Dict[str, List[Example]] = {}
        self._load_local_datasets()

    def _load_local_datasets(self):
        """Loads golden datasets from disk."""
        for f in self.datasets_dir.glob("*.json"):
            ds_name = f.stem
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    examples = []
                    for item in data:
                        inp = {"conversation": item.get("input", "")} if "input" in item else item.get("inputs", {})
                        out = item.get("expected", {}) if "expected" in item else item.get("outputs", {})
                        examples.append(Example(inputs=inp, outputs=out, id=item.get("id")))
                    self._datasets[ds_name] = examples
            except Exception:
                pass

    def push_prompt(self, prompt_identifier: str, object: Any, description: str = "") -> str:
        """Emulates pushing a prompt to LangSmith Prompt Hub with a version commit."""
        clean_name = prompt_identifier.split("/")[-1].replace(":", "_")
        p_dir = self.prompts_dir / clean_name
        p_dir.mkdir(parents=True, exist_ok=True)
        
        # Save prompt with commit/timestamp
        commit_id = time.strftime("%Y%m%d_%H%M%S")
        file_path = p_dir / f"{commit_id}.json"
        
        content = str(object) if not hasattr(object, "to_json") else object.to_json()
        payload = {
            "identifier": prompt_identifier,
            "commit_id": commit_id,
            "description": description,
            "content": content
        }
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
            
        # Update latest pointer
        with open(p_dir / "latest.json", "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
            
        return f"{prompt_identifier}:{commit_id}"

    def pull_prompt(self, prompt_identifier: str) -> str:
        """Emulates pulling a prompt from LangSmith Prompt Hub."""
        parts = prompt_identifier.split(":")
        repo_name = parts[0].split("/")[-1]
        version = parts[1] if len(parts) > 1 else "latest"
        
        p_dir = self.prompts_dir / repo_name
        file_path = p_dir / f"{version}.json"
        if not file_path.exists():
            file_path = p_dir / "latest.json"
            
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt {prompt_identifier} not found in local hub.")
            
        with open(file_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            return data.get("content", "")

    def has_dataset(self, dataset_name: str) -> bool:
        return dataset_name in self._datasets or (self.datasets_dir / f"{dataset_name}.json").exists()

    def create_dataset(self, dataset_name: str, description: str = "") -> Any:
        class DatasetObj:
            def __init__(self, name, id): self.name = name; self.id = id
        ds = DatasetObj(dataset_name, str(uuid.uuid4()))
        self._datasets[dataset_name] = []
        return ds

    def create_examples(self, inputs: List[Dict], outputs: List[Dict], dataset_name: Optional[str] = None, dataset_id: Optional[str] = None):
        name = dataset_name or "default_dataset"
        if name not in self._datasets:
            self._datasets[name] = []
        for inp, out in zip(inputs, outputs):
            self._datasets[name].append(Example(inputs=inp, outputs=out))


def offline_traceable(
    name: str = "agent_call",
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
):
    """
    Offline decorator emulating LangSmith @traceable.
    Tags each trace with prompt_version, model_name, inputs, outputs, and latency.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            status = "SUCCESS"
            error_msg = None
            out = None
            
            # Extract tags and metadata
            meta = dict(metadata or {})
            if "prompt_version" in kwargs:
                meta["prompt_version"] = kwargs["prompt_version"]
            if "model_name" in kwargs:
                meta["model_name"] = kwargs["model_name"]
                
            try:
                out = func(*args, **kwargs)
                return out
            except Exception as e:
                status = "ERROR"
                error_msg = str(e)
                raise
            finally:
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0
                
                # Build trace record
                trace_entry = {
                    "trace_id": str(uuid.uuid4()),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "name": name,
                    "prompt_version": meta.get("prompt_version", "unknown"),
                    "model_name": meta.get("model_name", meta.get("model", "unknown")),
                    "latency_ms": round(latency_ms, 2),
                    "tags": tags or [],
                    "metadata": meta,
                    "status": status,
                    "error": error_msg,
                    "input": str(args[0])[:300] if args else str(kwargs)[:300],
                    "output": out if isinstance(out, (dict, list, str, int, float)) else str(out)[:300]
                }
                
                # Append to traces.jsonl
                log_file = TRACES_DIR / "traces.jsonl"
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass
        return wrapper
    return decorator


def offline_evaluate(
    target: Callable,
    data: Any,
    evaluators: List[Callable],
    experiment_prefix: str = "eval",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Offline evaluator emulating langsmith.evaluate().
    Executes target against dataset examples and computes evaluator scores.
    """
    meta = metadata or {}
    client = OfflineLangSmithClient()
    
    # Resolve dataset
    if isinstance(data, str):
        examples = client._datasets.get(data, [])
        if not examples:
            client._load_local_datasets()
            examples = client._datasets.get(data, [])
    elif isinstance(data, list):
        examples = [e if isinstance(e, Example) else Example(inputs=e.get("inputs", {}), outputs=e.get("outputs", {})) for e in data]
    else:
        examples = []
        
    results = []
    total_score = 0.0
    
    for ex in examples:
        t0 = time.perf_counter()
        pred = target(ex.inputs)
        t1 = time.perf_counter()
        
        run_obj = Run(inputs=ex.inputs, outputs=pred)
        scores = {}
        for ev in evaluators:
            res = ev(run_obj, ex)
            if isinstance(res, dict):
                scores[res.get("key", "score")] = res.get("score", 0.0)
                total_score += res.get("score", 0.0)
            else:
                total_score += float(res)
                
        results.append({
            "example_id": ex.id,
            "inputs": ex.inputs,
            "expected": ex.outputs,
            "predicted": pred,
            "latency_ms": round((t1 - t0) * 1000.0, 2),
            "scores": scores
        })
        
    num_ex = len(examples) if examples else 1
    accuracy_score = total_score / num_ex
    
    summary = {
        "experiment_name": f"{experiment_prefix}_{time.strftime('%Y%m%d_%H%M%S')}",
        "prompt_version": meta.get("prompt_version", "latest"),
        "model_name": meta.get("model", meta.get("model_name", "unknown")),
        "total_examples": len(examples),
        "accuracy_score": round(accuracy_score, 4),
        "accuracy_pct": f"{accuracy_score * 100:.2f}%",
        "results": results
    }
    return summary


# Export factory functions: use real LangSmith if available, else offline client
try:
    import langsmith
    from langsmith import Client, evaluate, traceable
    IS_LIVE_LANGSMITH = True
except Exception:
    Client = OfflineLangSmithClient
    evaluate = offline_evaluate
    traceable = offline_traceable
    IS_LIVE_LANGSMITH = False

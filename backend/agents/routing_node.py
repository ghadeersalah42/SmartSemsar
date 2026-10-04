"""routing_node.py: LangGraph node that decides what happens after the 3D
model step. There is no "classic" / 2D fallback path here - once the GLB
model is ready, the only next stop is furniture placement.

Reads from state:
    model_state : ModelState

Writes to state:
    next_step : "3d_staging" | "retry" | "error" | "furniture"
    error     : str | None
"""
from backend.schema.state import GraphState


def routing_node(state: GraphState) -> dict:
    model_state = state.get("model_state")

    if model_state is None or model_state.status == "pending":
        return {"next_step": "3d_staging"}

    if model_state.status == "building":
        return {"next_step": "retry"}

    if model_state.status == "failed":
        return {"next_step": "error", "error": model_state.error or "3D model build failed."}

    # status == "ready" -> hand off to furniture placement
    return {"next_step": "furniture"}

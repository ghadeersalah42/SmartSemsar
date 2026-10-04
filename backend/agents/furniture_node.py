"""furniture_node.py: placeholder for the furniture-placement step.

The real implementation (catalog + placer, see origin/Furniture_Feature)
has not been merged into this pipeline yet. This stub only gives the
graph somewhere to route once routing_node sees the 3D model is ready.

Reads from state:
    model_state : ModelState

Writes to state:
    next_step : "done"
"""
from backend.schema.state import GraphState


def furniture_node(state: GraphState) -> dict:
    # TODO: replace with the real furniture placer once Furniture_Feature is merged.
    return {"next_step": "done"}

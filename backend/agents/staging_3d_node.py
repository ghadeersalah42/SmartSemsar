"""staging_3d_node.py: LangGraph node that turns the selected property's
plan.json into a .glb + walkthrough page, and records the result in
model_state so routing_node knows when it can send the graph on to
furniture placement.

Reads from state:
    selected_property : PropertyItem

Writes to state:
    model_state : ModelState
"""
from pathlib import Path

from backend.schema.plan import load_plan
from backend.schema.state import GraphState, ModelState
from backend.services.plan_to_glb import export_glb
from backend.services.walkthrough import build_walkthrough

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "data" / "models"


def staging_3d_node(state: GraphState) -> dict:
    prop = state.get("selected_property")
    if prop is None or not prop.plan_path:
        return {"model_state": ModelState(status="failed", error="No property/plan selected.")}

    try:
        plan = load_plan(prop.plan_path)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        glb_path = MODELS_DIR / f"{prop.property_id}.glb"
        html_path = MODELS_DIR / f"{prop.property_id}_walkthrough.html"

        export_glb(plan, glb_path)
        build_walkthrough(prop.plan_path, html_path, glb_path=glb_path, listing=prop.model_dump())

        return {
            "model_state": ModelState(
                status="ready",
                plan_path=str(prop.plan_path),
                glb_path=str(glb_path),
                walkthrough_path=str(html_path),
            )
        }
    except Exception as exc:
        return {"model_state": ModelState(status="failed", error=str(exc))}

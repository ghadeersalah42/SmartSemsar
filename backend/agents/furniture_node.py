"""furniture_node.py: LangGraph node that furnishes the 3D model staging_3d_node built.

The user's own pieces (custom_pieces: raw GLBs from photos or uploads) take the place of catalog
items; then the rules or the LLM place everything, the validator checks it, and a furnished
walkthrough is built next to the empty one (services/multi_furnish.furnish_with_pieces).

Reads from state:
    selected_property : PropertyItem
    model_state       : ModelState      (status "ready": plan_path, glb_path)
    furniture_state   : FurnitureState  (preferences, planner, custom_pieces)

Writes to state:
    furniture_state : FurnitureState (status, staging_path, walkthrough_path, report ...)
    next_step       : "done" | "error"
    error           : str | None
"""
from pathlib import Path

from backend.schema.state import FurnitureState, GraphState
from backend.services.custom_furniture import FRONT_TURN_DEG
from backend.services.multi_furnish import Piece, furnish_with_pieces

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "data" / "models"


def furniture_node(state: GraphState) -> dict:
    prop = state.get("selected_property")
    model_state = state.get("model_state")
    fs = state.get("furniture_state") or FurnitureState()
    if prop is None or model_state is None or model_state.status != "ready" or not model_state.plan_path:
        return _failed(fs, "The 3D model is not ready, so there is nothing to furnish.")

    report: list[str] = []
    # each piece is turned for the generator that made it ("colab" models face backwards)
    pieces = [Piece(p.glb_path, p.slot, p.width_m, (p.yaw_deg + FRONT_TURN_DEG.get(p.generator or "", 0.0)) % 360,
                    p.label or Path(p.glb_path).name)
              for p in fs.custom_pieces]
    try:
        result = furnish_with_pieces(model_state.plan_path, pieces, MODELS_DIR / f"{prop.property_id}_furnished",
                                     fs.preferences, fs.planner, listing=prop.model_dump(),
                                     glb_path=model_state.glb_path, say=report.append)
    except Exception as exc:
        return _failed(fs.model_copy(update={"report": report}), f"{type(exc).__name__}: {exc}")

    furniture_state = fs.model_copy(update={
        "status": "ready" if result.approved else "rejected",
        "preferences": result.prefs,
        "staging_path": str(result.staging_path),
        "walkthrough_path": str(result.walkthrough) if result.walkthrough else None,
        "placed_count": len(result.staging.items),
        "unplaced": list(result.staging.unplaced),
        "failed_checks": result.failed_checks,
        "report": report,
        "error": None,
    })
    if not result.approved:
        return {"furniture_state": furniture_state, "next_step": "error",
                "error": "The furnished staging did not pass the checks: " + "; ".join(result.failed_checks)}
    return {"furniture_state": furniture_state, "next_step": "done"}


def _failed(fs: FurnitureState, message: str) -> dict:
    return {"furniture_state": fs.model_copy(update={"status": "failed", "error": message}),
            "next_step": "error", "error": message}

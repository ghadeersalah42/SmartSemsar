"""The LangGraph leg: plan -> 3D model -> furniture, with the user's wishes and own pieces in furniture_state."""
import pytest
import trimesh

from backend.agents import furniture_node as furniture_module
from backend.agents import staging_3d_node as staging_module
from backend.agents.furniture_node import furniture_node
from backend.agents.graph import compiled_graph
from backend.schema.design import DesignPreferences
from backend.schema.staging import load_staging
from backend.schema.state import CustomPiece, FurnitureState, ModelState, PropertyItem

PROP = PropertyItem(property_id="PROP_1002", title="Apartment for Rent in Al Marasem Compound", price=0,
                    location="New Cairo", bedrooms=3, area_sqm=170, plan_path="data/plans/PROP_1002.json")


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(staging_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(furniture_module, "MODELS_DIR", tmp_path)
    return tmp_path


def raw_glb(path):
    box = trimesh.creation.box(extents=(0.5, 0.2, 1.4))      # unit-less and facing sideways, like a generated one
    box.export(path)
    return str(path)


def run(furniture_state):
    return compiled_graph.invoke({"session_id": "test", "selected_property": PROP,
                                  "model_state": ModelState(), "furniture_state": furniture_state})


def test_graph_builds_the_3d_model_then_furnishes_it(models_dir):
    out = run(FurnitureState())
    fs = out["furniture_state"]
    assert out["model_state"].status == "ready" and out["next_step"] == "done"
    assert fs.status == "ready" and fs.placed_count > 10 and not fs.failed_checks
    assert fs.walkthrough_path and (models_dir / "PROP_1002_furnished" / "walkthrough.html").exists()
    # the empty walkthrough of the 3D step stays as it was, next to the furnished one
    assert out["model_state"].walkthrough_path != fs.walkthrough_path
    assert len(load_staging(fs.staging_path).items) == fs.placed_count


def test_wishes_and_own_piece_reach_the_furnished_apartment(models_dir):
    piece = CustomPiece(glb_path=raw_glb(models_dir / "chair.glb"), slot="armchair", width_m=0.8, label="chair.jpg")
    prefs = DesignPreferences.from_loose({"must_have": ["desk"], "exclude": ["tv", "armchair"], "dining_seats": 4})
    out = run(FurnitureState(preferences=prefs, custom_pieces=[piece]))
    fs = out["furniture_state"]
    placed = [i.catalog_id for i in load_staging(fs.staging_path).items]
    assert fs.status == "ready"
    assert "armchair" in placed            # the user's own piece: "leave out armchair" does not remove it
    assert "desk" in placed and not any(c.startswith("tv") for c in placed)
    assert "dining_table_4" in placed
    assert any("chair.jpg -> armchair: 0.8 x" in line for line in fs.report)


def test_nothing_to_furnish_without_a_ready_3d_model():
    out = furniture_node({"selected_property": PROP, "model_state": ModelState(status="failed"),
                          "furniture_state": FurnitureState()})
    assert out["next_step"] == "error" and out["furniture_state"].status == "failed"

"""
Smart Semsar - Shared LangGraph state (GraphState)
العقد الموحد لحالة الـ Graph: كل Node بيقرأ ويكتب في نفس الـ State.

Rules:
  - One 3D pipeline only: a Plan is extruded deterministically into a GLB
    (backend/services/plan_to_glb.py). There is no "classic" 2D-staging
    mode here - staging always works on the 3D model.
  - model_state tracks that GLB build so routing_node knows when the graph
    can move on to furniture placement.
  - furniture_state carries the user's wishes and own pieces into furniture_node,
    and what it placed in that same 3D model (staging + furnished walkthrough).
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from backend.schema.design import DesignPreferences


class CustomerRequirements(BaseModel):
    location: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    area_sqm: Optional[float] = None
    bedrooms: Optional[int] = None
    property_type: Optional[Literal[
        "apartment", "villa", "duplex", "studio", "townhouse", "unknown"
    ]] = "unknown"
    purpose: Optional[Literal["living", "investment", "unknown"]] = "unknown"
    notes: Optional[str] = None
    missing_fields: list[str] = Field(default_factory=list)


class PropertyItem(BaseModel):
    """One match from the matching_node, enough to drive the 3D step."""
    property_id: str
    title: str
    price: float
    location: str
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    area_sqm: Optional[float] = None
    plan_path: Optional[str] = None  # path to this property's plan.json


# ---------- 3D GLB model state ----------
# No "classic" / 2D-render mode: plan.json -> .glb is the only staging path.
ModelStatus = Literal["pending", "building", "ready", "failed"]


class ModelState(BaseModel):
    status: ModelStatus = "pending"
    plan_path: Optional[str] = None
    glb_path: Optional[str] = None
    walkthrough_path: Optional[str] = None
    error: Optional[str] = None


# ---------- furniture placed in that 3D model ----------
FurnitureStatus = Literal["pending", "ready", "rejected", "failed"]


class CustomPiece(BaseModel):
    """The user's own piece: a raw GLB (from a photo or uploaded) that replaces one catalog item."""
    glb_path: str
    slot: str                               # catalog id or kind it replaces, e.g. "sofa" / "sofa_3_seat"
    width_m: Optional[float] = None         # real width; None = the stock item's width
    yaw_deg: float = 0.0                    # turn it if it faces sideways
    generator: Optional[str] = None         # what made the GLB ("colab" models face backwards and are turned)
    label: str = ""                         # e.g. the photo's file name, for the report


class FurnitureState(BaseModel):
    status: FurnitureStatus = "pending"
    # what the user asked for (filled before the graph runs, or by an intake node)
    preferences: DesignPreferences = Field(default_factory=DesignPreferences)
    planner: Literal["rules", "llm"] = "rules"
    custom_pieces: list[CustomPiece] = Field(default_factory=list)
    # what furniture_node produced
    staging_path: Optional[str] = None      # staging.json: every placed item
    walkthrough_path: Optional[str] = None  # the furnished walkthrough (model_state's one stays empty)
    placed_count: int = 0
    unplaced: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    report: list[str] = Field(default_factory=list)
    error: Optional[str] = None


NextStep = Literal["matching", "3d_staging", "furniture", "retry", "error", "done"]


class GraphState(TypedDict):
    session_id: str

    # intake (input_node)
    audio_path: Optional[str]
    user_text: Optional[str]
    requirements: Optional[CustomerRequirements]

    # matching (matching_node)
    top_properties: Optional[list[PropertyItem]]
    selected_property: Optional[PropertyItem]

    # 3D model (staging_3d_node)
    model_state: ModelState

    # furniture in the 3D model (furniture_node)
    furniture_state: FurnitureState

    # routing / control (routing_node)
    next_step: Optional[NextStep]
    error: Optional[str]

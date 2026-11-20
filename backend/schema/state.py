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
from typing import Literal, TypedDict, List, Optional, Dict, Any, Annotated
import operator

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field, model_validator

from backend.schema.design import DesignPreferences


# ---------- حدود منطقية للطلبات (Schema sanity ranges) ----------
# أي قيمة برّه الحدود دي غالباً الاستخراج فهمها غلط ("ميزانيتي 5" مش 5 جنيه)، فبتتشال
# وبتتسجل في unclear_fields، والبوت بيسأل العميل عنها بدل ما يدور بقيمة غلط.
BEDROOMS_RANGE = (0, 10)                 # 0 = ستوديو
AREA_RANGE = (20, 2_000)                 # متر مربع
BUDGET_RANGES = {
    "rent": (1_000, 1_000_000),          # إيجار شهري بالجنيه
    "buy": (100_000, 1_000_000_000),     # سعر الشقة كله بالجنيه
    "unknown": (1_000, 1_000_000_000),   # لسه مش عارفين إيجار ولا تمليك: الحدود الأوسع
}


def budget_range(deal_type: Optional[str]) -> tuple[float, float]:
    return BUDGET_RANGES.get(deal_type or "unknown", BUDGET_RANGES["unknown"])


class CustomerRequirements(BaseModel):
    location: Optional[str] = Field(default=None, description="Preferred Location")
    budget_min: Optional[float] = Field(default=None, description="Min budget in EGP")
    budget_max: Optional[float] = Field(default=None, description="Max budget in EGP")
    area_sqm: Optional[float] = Field(default=None, description="Area of property")
    bedrooms: Optional[int] = Field(default=None, description="Bedroom count")
    property_type: Optional[Literal["apartment", "villa", "duplex", "studio", "townhouse", "unknown"]] = Field(default="unknown", description="Type of property")
    purpose: Optional[Literal["living", "investment", "unknown"]] = Field(default="unknown", description="Purpose of Searching for Apartment")
    deal_type: Optional[Literal["rent", "buy", "unknown"]] = Field(default="unknown", description="Rent (إيجار) or buy (تمليك/شراء)")
    # style_preference: Optional[str] = Field(default=None, description="Design style e.g., Modern, Classic")
    notes: Optional[str] = Field(default=None, description="Notes from customer inputs regards the property")
    missing_fields: list[str] = Field(default_factory=list)
    # الحقول اللي اتقالت بس مش واضحة أو برّه الحدود: {"budget_max": "out_of_range:5"}
    unclear_fields: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_sanity_ranges(self):
        """القيم اللي مش منطقية بتتشال وبتتسجل في unclear_fields (مش بنخمن قيمة بدالها)"""
        def out_of_range(field, value, low, high):
            if value is not None and not (low <= value <= high):
                setattr(self, field, None)
                self.unclear_fields[field] = f"out_of_range:{value:g}"

        out_of_range("bedrooms", self.bedrooms, *BEDROOMS_RANGE)
        out_of_range("area_sqm", self.area_sqm, *AREA_RANGE)
        low, high = budget_range(self.deal_type)
        out_of_range("budget_min", self.budget_min, low, high)
        out_of_range("budget_max", self.budget_max, low, high)
        if self.budget_min is not None and self.budget_max is not None and self.budget_min > self.budget_max:
            self.budget_min, self.budget_max = self.budget_max, self.budget_min
        return self

class PropertyItem(BaseModel):
    """Property details from the matching agent"""
    property_id: str = Field(description="Unique ID for the property")
    title: str = Field(description="Property title")
    price: float = Field(description="Price in EGP")
    location: str = Field(description="Location/Area")
    pf_bedrooms: Optional[int] = Field(default=None, description="No. of Bedrooms")
    pf_bathrooms: Optional[int] = Field(default=None, description="Bathrooms number")
    pf_area_sqm: Optional[float] = Field(default=None, description="Area")
    plan_path: Optional[str] = None
    cubicasa_id: Optional[int] = Field(default=None, description="CubiCasa dataset ID")
    image_path: Optional[str] = Field(default=None, description="Path of 2D Sketch")
    svg_path: Optional[str] = Field(default=None, description="Path of 2D Sketch")

    


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
NextStep = Literal["matching", "3d_staging", "furniture", "retry", "error", "done"]

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




def update_profile_reducer(old_profile: Optional[CustomerRequirements], new_profile: Optional[CustomerRequirements]) -> Optional[CustomerRequirements]:
    """Profile partial Update"""
    
    if not old_profile:
        return new_profile
    if not new_profile:
        return old_profile
      
    updated_data = old_profile.model_dump()
    new_data = new_profile.model_dump(exclude_unset=True)
    updated_data.update(new_data)
    
    return CustomerRequirements(**updated_data)



class AgentState(TypedDict):
    session_id: str

    # Inputs
    user_input_type: Optional[str]=Field(default=None)   # 'voice' | 'pdf' | 'text' | 'sketch'
    # raw_input_path_or_text: Optional[str]
    user_req_audio_path:Optional[str]=Field(default=None)
    user_req_text: Optional[str]=Field(default=None)

    user_query: Optional[str]
    uploaded_sketch_path: Optional[str] 
    
    macro_intent: Optional[str]   # 'search_by_reqs', 'direct_sketch_staging', 'general_browse', 'sketch_based_search'
    micro_intent: Optional[str]   # 'new_search', 'modify_search', 'alternative_search', etc.
    current_stage: Optional[str]
    
    intent_confidence: Optional[float]

    # Profile
    customer_reqs: Annotated[Optional[CustomerRequirements], update_profile_reducer]
    svg_data: Optional[str]

    #matching service vars
    top_properties: Optional[List[PropertyItem]]
    used_fallback: Optional[bool]
    total_found: Optional[int]

    selected_property_id: Optional[str]
    selected_property: Optional[PropertyItem]

    # 3D model (staging_3d_node)
    model_state: ModelState

    # furniture in the 3D model (furniture_node)
    furniture_state: FurnitureState
    
    custom_sketch_path: Optional[str]
    matching_status: Optional[str]

    telegram_chat_id: Optional[str]
    booking_data: Optional[Dict[str, Any]]
    telegram_report_sent: bool        # True بس لو send_property_report رجعت success
    # user_decision: Optional[str]  # 'proceed_3d', 'furnish', 'send_telegram', 'stop'
    agent_response: Optional[str]

    # Workflow Control & Errors
    error_message: Optional[str]
    next_step: Optional[str]


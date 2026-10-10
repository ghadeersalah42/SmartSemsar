from typing import Literal, TypedDict, List, Optional, Dict, Any, Annotated
import operator

from pydantic import BaseModel, Field



class CustomerRequirements(BaseModel):
    location: Optional[str] = Field(default=None, description="Preferred Location")
    budget_min: Optional[float] = Field(default=None, description="Min budget in EGP")
    budget_max: Optional[float] = Field(default=None, description="Max budget in EGP")
    area_sqm: Optional[float] = Field(default=None, description="Area of property")
    bedrooms: Optional[int] = Field(default=None, description="Bedroom count")
    property_type: Optional[Literal["apartment", "villa", "duplex", "studio", "townhouse", "unknown"]] = Field(default="unknown", description="Type of property")
    purpose: Optional[Literal["living", "investment", "unknown"]] = Field(default="unknown", description="Purpose of Searching for Apartment")
    # style_preference: Optional[str] = Field(default=None, description="Design style e.g., Modern, Classic")
    notes: Optional[str] = Field(default=None, description="Notes from customer inputs regards the property")
    missing_fields: list[str] = Field(default_factory=list)
    
class PropertyItem(BaseModel):
    """Property details from the matching agent"""
    property_id: str = Field(description="Unique ID for the property")
    title: str = Field(description="Property title")
    price: float = Field(description="Price in EGP")
    location: str = Field(description="Location/Area")
    pf_bedrooms: Optional[int] = Field(default=None, description="No. of Bedrooms")
    pf_bathrooms: Optional[int] = Field(default=None, description="Bathrooms number")
    pf_area_sqm: Optional[float] = Field(default=None, description="Area")
    cubicasa_id: Optional[int] = Field(default=None, description="CubiCasa dataset ID")
    image_path: Optional[str] = Field(default=None, description="Path of 2D Sketch")

    
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
    user_input_type: Optional[str]   # 'voice' | 'pdf' | 'text' | 'sketch'
    # raw_input_path_or_text: Optional[str]
    audio_path:Optional[str]
    user_text: Optional[str]

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
    
    custom_sketch_path: Optional[str]
    matching_status: Optional[str]

    # CV & 3D Staging
    render_3d_url: Optional[str]
    staging_3d_result: Optional[Dict[str, Any]]
    furnished_result: Optional[Dict[str, Any]]
    whatsapp_report_sent: bool
    # user_decision: Optional[str]  # 'proceed_3d', 'furnish', 'send_whatsapp', 'stop'
    agent_response: Optional[str]

    # Workflow Control & Errors
    error_message: Optional[str]
    next_step: Optional[str]
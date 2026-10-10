from backend.services.MatchingFeature import PropertyMatchingService, CustomerRequirements
from backend.schema.state import AgentState, PropertyItem
from typing import Optional, Literal, List, Dict, Any

Property_matching_service= PropertyMatchingService(csv_path="data/final_merged_dataset.csv", chroma_path="data/chroma_db")

def match_properties_node(state: AgentState)-> Dict[str, Any]:
    print("Execute Matching Property")

    requirements= state.get("customer_reqs") or {}
    macro_intent= state.get('macro_intent')
    micro_intent= state.get('micro_intent')
    all_prop=""
    property_objects = []

    if macro_intent=="general_browse":
        all_prop="general_browse"

    # if not requirements:
    #     return{
    #         "top_properties": [],
    #         "used_fallback": False,
    #         "total_found": -1,
    #         "selected_property": None,
    #         "custom_sketch_path": "",
    #         "matching_status": ""
    #     }
    

    results= Property_matching_service.search_properties(requirements, all_prop)
    
    for prop in results["properties"]:
        item = PropertyItem(
            property_id=str(prop.get("property_id")),
            title=str(prop.get("title", "")),
            price=float(prop.get("price", 0.0)),
            location=str(prop.get("location", "")),
            pf_bedrooms=prop.get("pf_bedrooms"),
            pf_bathrooms=prop.get("pf_bathrooms"),
            pf_area_sqm=prop.get("pf_area_sqm"),
            cubicasa_id=prop.get("cubicasa_id"),
            image_path=prop.get("image_path")
        )
        property_objects.append(item)
    return {
        "top_properties": property_objects,
        "total_found": results["total_found"],
        "used_fallback": results["used_fallback"],
        "current_stage": "property_results_displayed"
    }

def select_property_node(state: AgentState) -> dict:
    print("\n🎯 [Node] Executing Property Selection Node...")
    selected_id = state.get("selected_property_id")
    top_props = state.get("top_properties", []) or []

    if not selected_id:
        return {"selected_property": None}

    chosen_property = next(
        (p for p in top_props if str(p.property_id) == str(selected_id)), 
        None
    )

    if not chosen_property and hasattr(Property_matching_service, 'get_property_by_id'):
        raw_p = Property_matching_service.get_property_by_id(selected_id)
        if raw_p:
            chosen_property = PropertyItem(
                property_id=str(raw_p.get("property_id")),
                title=str(raw_p.get("title", "")),
                price=float(raw_p.get("price", 0.0)),
                location=str(raw_p.get("location", "")),
                pf_bedrooms=raw_p.get("pf_bedrooms"),
                pf_bathrooms=raw_p.get("pf_bathrooms"),
                pf_area_sqm=raw_p.get("pf_area_sqm"),
                cubicasa_id=raw_p.get("cubicasa_id"),
                image_path=raw_p.get("image_path")
            )

    if chosen_property:
        print(f"Chosen Property is: {chosen_property.title} (ID: {chosen_property.property_id})")

    return {
        "selected_property": chosen_property,
        "custom_sketch_path": chosen_property.image_path,
        "current_stage": "property_selected"
    }


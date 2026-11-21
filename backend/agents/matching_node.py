from backend.services.MatchingFeature import PropertyMatchingService, CustomerRequirements
from backend.schema.state import AgentState, PropertyItem
from typing import Optional, Literal, List, Dict, Any
from langgraph.types import interrupt


Property_matching_service= PropertyMatchingService(csv_path="data/final_merged_dataset.csv", chroma_path="data/chroma_db")

def match_properties_node(state: AgentState)-> Dict[str, Any]:
    print("Execute Matching Property")

    raw_reqs= state.get("customer_reqs") or {}
    macro_intent= state.get('macro_intent')
    micro_intent= state.get('micro_intent')
    # لو الـ customer_reqs راجعة كـ Dictionary، نحولها لـ CustomerRequirements Object
    if isinstance(raw_reqs, dict):
        requirements = CustomerRequirements(
            location=raw_reqs.get("location"),
            budget_max=raw_reqs.get("budget_max"),
            bedrooms=raw_reqs.get("bedrooms"),
            area_sqm=raw_reqs.get("area_sqm"),
            property_type=raw_reqs.get("property_type"),
            deal_type=raw_reqs.get("deal_type") or "unknown"
        )
    else:
        requirements = raw_reqs

        
    all_prop=""
    property_objects = []

    if macro_intent=="general_browse":
        all_prop="general_browse"
    elif getattr(requirements, "deal_type", None) == "buy":
        # كل الشقق اللي في الداتا إيجار (مفيش عمود بيع ولا إعلانات تمليك)، فطلب التمليك ملوش نتايج
        print("No sale listings in the dataset (all rentals)")
        return {"top_properties": [], "total_found": 0, "used_fallback": False,
                "current_stage": "no_sale_listings"}

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
            plan_path= prop.get("plan_path"),
            image_path=prop.get("image_path"),
            svg_path=prop.get("svg_path")
        )
        property_objects.append(item)
    return {
        "top_properties": property_objects,
        "total_found": results["total_found"],
        "used_fallback": results["used_fallback"],
        "current_stage": "property_results_displayed"
    }

def property_item_by_id(property_id) -> Optional[PropertyItem]:
    """شقة من الداتا كـ PropertyItem (None لو الـ id مش موجود)"""
    raw_p = Property_matching_service.get_property_by_id(property_id)
    if not raw_p:
        return None
    return PropertyItem(
        property_id=str(raw_p.get("property_id")),
        title=str(raw_p.get("title", "")),
        price=float(raw_p.get("price", 0.0)),
        location=str(raw_p.get("location", "")),
        pf_bedrooms=raw_p.get("pf_bedrooms"),
        pf_bathrooms=raw_p.get("pf_bathrooms"),
        pf_area_sqm=raw_p.get("pf_area_sqm"),
        cubicasa_id=raw_p.get("cubicasa_id"),
        plan_path= raw_p.get("plan_path"),
        image_path=raw_p.get("image_path"),
        svg_path=raw_p.get("svg_path")
    )


def select_property_node(state: AgentState) -> dict:
    print("\n🎯 [Node] Executing Property Selection Node...")
    selected_id = state.get("selected_property_id")
    top_props = state.get("top_properties", []) or []

    # لو اليوزر لسه مَسَكّش ولا اختيار، نعمل Interrupt ونبعت الـ 5 شقق للواجهة
    
    print("interuptp")
    user_choice = interrupt({
        "type": "property_selection",
        "message": "إليك أفضل العقارات المطابقة لطلباتك، يرجى اختيار العقار المناسب:",
        "properties": top_props  
    })
    # لو اليوزر اختار من الواجهة، الـ user_choice هيرجع فيه الـ property_id (أو الـ ID المختار)
    selected_id = user_choice.get("property_id") if isinstance(user_choice, dict) else user_choice

    # وهنا باقي الكود بتاعك زي ما هو...
    chosen_property = next(
        (p for p in top_props if str(p.property_id) == str(selected_id)), 
        None
    )

    if not chosen_property:
        chosen_property = property_item_by_id(selected_id)

    if chosen_property:
        print(f"Chosen Property is: {chosen_property.title} (ID: {chosen_property.property_id})")
    
    print("finish selection node")

    return {
        "selected_property_id": chosen_property.property_id,
        "selected_property": chosen_property,
        "current_stage": "property_selected"
    }


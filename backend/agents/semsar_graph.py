from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.guardrails.pii import install_langsmith_masking
install_langsmith_masking()      # أرقام التليفونات والأسامي بتتخفي قبل ما أي trace يتبعت لـ LangSmith

from backend.schema.state import AgentState, PropertyItem
from backend.agents.routing_node import supervisor_router_node
from backend.agents.intent_detector import detect_intent_node
from backend.agents.input_node import input_node
from backend.agents.sketch_2D_to_svg_node import sketch_to_svg_node
from backend.agents.matching_node import match_properties_node, select_property_node
from backend.agents.final_report_node import telegram_report_node
from backend.agents.furniture_node import furniture_node
from backend.agents.routing_node import routing_node ##
from backend.agents.staging_3d_node import staging_3d_node


builder = StateGraph(AgentState)
builder.add_node("supervisor_router", supervisor_router_node)

builder.add_node("detect_micro_intent", detect_intent_node)
builder.add_node("input_extraction", input_node)
builder.add_node("match_properties", match_properties_node)
builder.add_node("select_property", select_property_node)
builder.add_node("sketch_to_svg", sketch_to_svg_node)
builder.add_node("staging_3d", staging_3d_node)
builder.add_node("routing", routing_node)
builder.add_node("furniture", furniture_node)

builder.add_edge(START, "supervisor_router")

def route_macro_intent(state: AgentState) -> str:
    intent = state.get("macro_intent")
    extra=""
   
    if intent== "search_by_reqs":
        return "user_enter_reqs_in_intent_detector" #intent detector + micro intent= new search
    
    elif intent == "general_browse":
        return "Match_all_properties"
    
    elif intent == "sketch_based_search":
        return "sketch_to_svg"

    elif intent == "direct_sketch_staging":
        return "3D design stage"

    elif intent in ["request_sketch_upload"]: #route to S2 or S4
        # يتوقف الجراف وينتظر رفع الصورة من العميل في الـ UI
        return END
        
 

 
builder.add_conditional_edges(
    "supervisor_router",
    route_macro_intent,
    {
        "user_enter_reqs_in_intent_detector": "detect_micro_intent",
        "sketch_to_svg": "sketch_to_svg",
        "Match_all_properties": "match_properties",
        "3D design stage": "staging_3d",
        END: END
    }
)


def route_micro_intent(state: AgentState) -> str:
    micro_intent = state.get("micro_intent")
    
    if micro_intent in ["new_search", "modify_search"]:
        print("new search here")
        return "input_reqs"
        
    elif micro_intent == "alternative_search":
        return "Alternate_match"
        
    elif micro_intent in ["property_inquiry", "booking_request", "positive_feedback", "negative_feedback"]:
        print("go to select property node")
        return "select_property"
        
    else:
        return "Match_all_properties"


builder.add_conditional_edges(
    "detect_micro_intent",
    route_micro_intent, 
    {
        "input_reqs": "input_extraction",
        "Alternate_match": "match_properties",
        "Match_all_properties": "match_properties",
        "select_property": "select_property"
    }
)

builder.add_edge("input_extraction", "match_properties")
builder.add_edge("match_properties", END)

builder.add_edge("select_property", "staging_3d")
builder.add_edge("staging_3d", "routing")

builder.add_conditional_edges(
    "routing",
    lambda state: state.get("next_step"),
    {
        "furniture": "furniture",
        "retry": "staging_3d",
        "error": END,
    },
)
builder.add_edge("furniture", END)



memory = MemorySaver()
the_Orchestrator = builder.compile(checkpointer=memory)
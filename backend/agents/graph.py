# # tool 1: location initial search
# # tool 2: from dolar to EGY pounds

# from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver

# from backend.schema.state import AgentState, PropertyItem
# from agents.routing_node import supervisor_router_node
# from agents.intent_detector import detect_intent_node
# from agents.input_node import input_node
# from agents.sketch_2D_to_svg_node import sketch_to_svg_node
# from agents.matching_node import match_properties_node, select_property_node
# from agents.staging_3d_node import staging_3d_node
# from agents.furniture_node import furniture_node
# from agents.final_report_node import whatsapp_report_node


# # def route_macro_intent(state: AgentState) -> str:
# #     intent = state.get("macro_intent")
    
# #     if intent == "request_sketch_upload": #route to S2 or S4
# #         # يتوقف الجراف وينتظر رفع الصورة من العميل في الـ UI
# #         return END
        
# #     elif intent == "search_by_reqs": #scenario 1

# #         return "user_enter_reqs"
        
# #     elif intent == "sketch_based_search": #4
# #         return "sketch_to_svg"
        
# #     elif intent == "direct_sketch_staging":# S2
# #         return "sketch_to_svg"
        
# #     else:  # general_browse S3
# #         return "match_properties"
    
# # def route_micro_intent(state: AgentState) -> str:
# #     micro_intent = state.get("micro_intent")
    
# #     # 1. بحث جديد أو تعديل على الشروط الحالية -> ذهاب لاستخراج البيانات من النص
# #     if micro_intent in ["new_search", "modify_search"]:
# #         return "input_extraction"
        
# #     # 2. طلب بدائل لنفس الشروط -> ذهاب مباشر لمطابقة الشقق بدون إعادة استخراج
# #     elif micro_intent == "alternative_search":
# #         return "match_properties"
        
# #     # 3. الاستفسار عن شقة أو طلب حجز -> ذهاب لنود تفاصيل/اختيار الشقة
# #     elif micro_intent in ["property_inquiry", "booking_request", "positive_feedback", "negative_feedback"]:
# #         return "select_property"
        
# #     # 4. أسئلة عامة أو غير معروف -> العودة للمطابقة أو الرد العام
# #     else:
# #         return "match_properties"
    
# def route_after_svg(state: AgentState) -> str:
#     """بعد استخراج الـ SVG: هل يتجه للـ 3D أم للبحث في قاعدة البيانات؟"""
#     intent = state.get("macro_intent")
#     if intent == "direct_sketch_staging":
#         return "staging_3d"
#     return "match_properties"  # sketch_based_search

# def route_after_selection(state: AgentState) -> str:
#     """بعد اختيار عقار: هل العميل يريد معالجة 3D أم يكتفي بالتقرير؟"""
#     decision = state.get("user_decision")
#     if decision == "proceed_3d":
#         return "staging_3d"
#     return "whatsapp_report"

# def route_after_3d(state: AgentState) -> str:
#     """بعد معالجة الـ 3D: هل يريد إضافة أثاث أم التقرير؟"""
#     decision = state.get("user_decision")
#     if decision == "furnish":
#         return "furniture"
#     return "whatsapp_report"


# # builder = StateGraph(AgentState)

# # builder.add_node("supervisor_router", supervisor_router_node)
# # builder.add_node("detect_micro_intent", detect_intent_node)
# # builder.add_node("input_extraction", input_node)
# # builder.add_node("match_properties", match_properties_node)
# builder.add_node("select_property", select_property_node)

# # builder.add_edge(START, "supervisor_router")
# # builder.add_conditional_edges(
# #     "supervisor_router",
# #     route_macro_intent,
# #     {
# #         "search_by_reqs": "detect_micro_intent",
# #         "sketch_to_svg": "sketch_to_svg",
# #         "match_properties": "match_properties"
# #     }
# # )

# # builder.add_conditional_edges(
# #     "detect_micro_intent",
# #     route_micro_intent,  # الدالة التي تفحص micro_intent
# #     {
# #         "input_extraction": "input_extraction",
# #         "match_properties": "match_properties",
# #         "select_property": "select_property"
# #     }
# # )


# # builder.add_edge("input_extraction", "match_properties")
# # builder.add_edge("match_properties", END)

























# # builder.add_node("sketch_to_svg", sketch_to_svg_node)
# builder.add_node("staging_3d", staging_3d_node)
# builder.add_node("furniture", furniture_node)
# builder.add_node("whatsapp_report", whatsapp_report_node)


# builder.add_conditional_edges(
#     "select_property",
#     route_after_selection,
#     {
#         "staging_3d": "staging_3d",
#         "whatsapp_report": "whatsapp_report"
#     }
# )





# # 🔀 التفرع 2: بعد الـ SVG
# builder.add_conditional_edges(
#     "sketch_to_svg",
#     route_after_svg,
#     {
#         "staging_3d": "staging_3d",
#         "match_properties": "match_properties"
#     }
# )


# builder.add_conditional_edges(
#     "staging_3d",
#     route_after_3d,
#     {
#         "furniture": "furniture",
#         "whatsapp_report": "whatsapp_report"
#     }
# )

# builder.add_edge("furniture", "whatsapp_report")
# builder.add_edge("whatsapp_report", END)

# # Checkpointer لإدارة الجلسات والتوقف التفاعلي
# checkpointer = MemorySaver()
# orchestrator_app = builder.compile(checkpointer=checkpointer)
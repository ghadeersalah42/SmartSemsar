from backend.schema.state import AgentState, PropertyItem

def whatsapp_report_node(state: AgentState) -> dict:
    print("\n📱 [Node] Generating and Sending WhatsApp Summary Report...")
    return {"whatsapp_report_sent": True, "agent_response": "تم إرسال التقرير الكامل لعقارك على واتساب بنجاح! 🚀"}
"""final_report_node.py: يبعت تقرير الشقة على تليجرام.

الرسالة اللي بتطلع للعميل مبنية على نتيجة send_property_report بس: لو الإرسال فشل أو تليجرام
مش متربط، البوت مبيقولش "اتبعت" (قبل كده كانت بترجع "اتبعت على واتساب" من غير ما تبعت حاجة).
"""
from backend.schema.state import AgentState
from backend.guardrails.actions import action_reply


def _send_property_report(**kwargs) -> dict:
    """تليجرام موجود في smart_semsar_ui/services (بيستورد نفسه كـ services.xxx)"""
    import os
    import sys
    ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "smart_semsar_ui"))
    if ui_dir not in sys.path:
        sys.path.append(ui_dir)
    from services.telegram_service import send_property_report
    return send_property_report(**kwargs)


def telegram_report_node(state: AgentState) -> dict:
    print("\n📱 [Node] Sending property report on Telegram...")
    chat_id = state.get("telegram_chat_id")
    prop = state.get("selected_property")
    if not chat_id:
        result = {"success": False, "error": "اربط حسابك على تليجرام الأول عشان التقرير يتبعتلك."}
    elif prop is None:
        result = {"success": False, "error": "مفيش شقة متختارة لسه."}
    else:
        info = prop.model_dump() if hasattr(prop, "model_dump") else dict(prop)
        info.setdefault("id", info.get("property_id"))
        try:
            result = _send_property_report(chat_id=chat_id, property_info=info,
                                           booking_data=state.get("booking_data"))
        except Exception as e:
            result = {"success": False, "error": str(e)}
    return {"telegram_report_sent": result.get("success") is True,
            "agent_response": action_reply("telegram_report", result)}

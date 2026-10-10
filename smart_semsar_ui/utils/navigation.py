import streamlit as st

PAGES = {
    "Dashboard": "🏠 Dashboard",
    "Voice Analysis": "🎙️ Voice Analysis",
    "Customer Profile": "👤 Customer Profile",
    "Properties": "🏢 Properties",
    "Recommendations": "⭐ Recommendations",
    "Alternatives": "🔄 Alternatives",
    "Virtual Staging": "🎨 Virtual Staging",
    "Booking": "📅 Booking",
}


def navigate_to(page_name: str) -> None:
    if page_name in PAGES:
        st.session_state["current_page"] = page_name


def get_current_page() -> str:
    return st.session_state.get("current_page", "Dashboard")

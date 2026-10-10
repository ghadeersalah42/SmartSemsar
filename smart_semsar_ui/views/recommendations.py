import streamlit as st
from components.cards import page_header
from services.property_service import get_properties, get_property_by_id
from services.agent_service import get_demo_recommendations


def render_recommendations() -> None:
    page_header("Smart matching", "Recommendations", "Demo ranking based on illustrative match scores. No AI model is running.")
    properties = get_demo_recommendations(get_properties())
    customer = st.session_state.get("customer_profile", {})
    st.info(f"Customer: **{customer.get('name', 'Demo customer')}** · Budget: **EGP {customer.get('budget', 0):,.0f}**")
    for item in properties:
        with st.container(border=True):
            c1, c2, c3 = st.columns([5, 2, 2])
            c1.subheader(item["title"])
            c1.caption(f"{item['location']} · {item['bedrooms']} bedrooms · {item['area_sqm']} m²")
            c2.metric("Demo match score", f"{item.get('match_score', 0)}%")
            c3.metric("Price", f"EGP {item['price']:,.0f}")
            a, b = st.columns(2)
            if a.button("Select property", key=f"select_{item['id']}", use_container_width=True):
                st.session_state["selected_property_id"] = item["id"]
                st.session_state["current_page"] = "Alternatives"
                st.rerun()
            if b.button("View in catalogue", key=f"catalogue_{item['id']}", use_container_width=True):
                st.session_state["selected_property_id"] = item["id"]
                st.session_state["current_page"] = "Properties"
                st.rerun()

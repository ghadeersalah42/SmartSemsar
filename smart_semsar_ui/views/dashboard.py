import streamlit as st
from components.cards import page_header, metric_row
from services.property_service import get_properties


def render_dashboard() -> None:
    page_header(
        "Smart workspace",
        "Good afternoon 👋",
        "Your AI-powered real-estate workspace. This is a UI demo; no AI services are connected.",
    )

    st.info("🏡 **Smart Real Estate AI**  \nFind the right property, intelligently.")
    properties = get_properties()

    metric_row([
        ("Customers", "124", "Demo metric"),
        ("Properties", "39K+", "Illustrative catalogue size"),
        ("Match Rate", "87%", "Demo metric, not a live model result"),
        ("Viewings", "18", "Demo metric"),
    ])

    st.divider()
    st.subheader("Quick Actions")
    cols = st.columns(4)
    actions = [
        ("🎙️ Voice Analysis", "Voice Analysis"),
        ("👤 Customer Profile", "Customer Profile"),
        ("🏢 Browse Properties", "Properties"),
        ("⭐ Recommendations", "Recommendations"),
    ]
    for col, (label, page) in zip(cols, actions):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["current_page"] = page
                st.rerun()

    st.divider()
    st.subheader("Featured Properties")
    cols = st.columns(3)
    for col, item in zip(cols, properties[:3]):
        with col:
            with st.container(border=True):
                st.caption(item["id"])
                st.markdown(f"### {item['title']}")
                st.write(item["location"])
                st.metric("Price", f"EGP {item['price']:,.0f}")
                st.write(f"{item['bedrooms']} bedrooms · {item['area_sqm']} m²")
                if st.button("View property", key=f"dash_{item['id']}", use_container_width=True):
                    st.session_state["selected_property_id"] = item["id"]
                    st.session_state["current_page"] = "Properties"
                    st.rerun()

import streamlit as st
from utils.navigation import PAGES, navigate_to, get_current_page


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🏡 Smart Semsar")
        st.caption("AI Real Estate Assistant")
        st.divider()
        st.caption("WORKSPACE")

        current_page = get_current_page()
        for page_name, label in PAGES.items():
            if st.button(
                label,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if page_name == current_page else "secondary",
            ):
                navigate_to(page_name)
                st.rerun()

        st.divider()
        st.caption("SYSTEM STATUS")
        st.success("UI Prototype • Demo Mode")
        st.caption("AI services are not connected.")

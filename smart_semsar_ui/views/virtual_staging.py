import streamlit as st
from components.cards import page_header
from services.property_service import get_property_by_id, get_properties


def render_virtual_staging() -> None:
    page_header("Design preview", "Virtual Staging", "Preview a future room-design workflow. Image generation is not connected in this UI prototype.")
    selected_id = st.session_state.get("selected_property_id", "PROP-001")
    selected = get_property_by_id(selected_id) or get_properties()[0]
    st.info(f"Selected property: **{selected['title']}**")
    with st.form("staging_preferences"):
        c1, c2 = st.columns(2)
        style = c1.selectbox("Interior style", ["Modern", "Minimalist", "Scandinavian", "Classic", "Industrial"])
        room = c2.selectbox("Room", ["Living Room", "Bedroom", "Kitchen", "Dining Room"])
        palette = st.selectbox("Color palette", ["Warm neutrals", "White and wood", "Earth tones", "Monochrome", "Soft pastels"])
        submitted = st.form_submit_button("Save design preferences", type="primary", use_container_width=True)
    if submitted:
        st.session_state["staging_preferences"] = {"style": style, "room": room, "palette": palette}
        st.success("Design preferences saved for this session. No image was generated.")
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Before")
        with st.container(border=True):
            st.markdown("### 🏠 Empty / Original Space")
            st.caption("Original property image placeholder")
            st.info("An original room image will appear here after an image is supplied.")
    with right:
        st.subheader("After")
        with st.container(border=True):
            st.markdown("### ✨ Staged Design Preview")
            st.caption("Generated image placeholder")
            st.info("The staged room image will appear here after image-generation integration.")

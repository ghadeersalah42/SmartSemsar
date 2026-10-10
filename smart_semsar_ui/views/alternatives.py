import streamlit as st
from components.cards import page_header
from services.property_service import get_property_by_id, get_properties


def render_alternatives() -> None:
    page_header("Flexible search", "Property Alternatives", "Explore alternatives and adjust constraints. Matching is demo logic only.")
    selected_id = st.session_state.get("selected_property_id", "PROP-001")
    selected = get_property_by_id(selected_id) or get_properties()[0]
    st.subheader("Currently selected")
    with st.container(border=True):
        st.markdown(f"### {selected['title']}")
        st.write(f"{selected['location']} · EGP {selected['price']:,.0f} · {selected['bedrooms']} bedrooms")
    st.subheader("Adjust your search")
    c1, c2 = st.columns(2)
    budget = c1.slider("Maximum budget (EGP)", 1_000_000, 15_000_000, min(max(int(selected["price"] * 1.15), 1_000_000), 15_000_000), step=250_000)
    bedrooms = c2.slider("Minimum bedrooms", 1, 6, max(1, int(selected["bedrooms"]) - 1))
    alternatives = [
        p for p in get_properties()
        if p["id"] != selected["id"] and p["price"] <= budget and p["bedrooms"] >= bedrooms
    ]
    st.divider()
    st.subheader(f"Alternatives ({len(alternatives)})")
    if not alternatives:
        st.warning("No demo alternatives match these settings. Increase the budget or reduce bedrooms.")
    for item in alternatives:
        with st.container(border=True):
            st.subheader(item["title"])
            st.write(f"{item['location']} · EGP {item['price']:,.0f}")
            st.write(f"{item['bedrooms']} bedrooms · {item['area_sqm']} m²")
            if st.button("Choose this alternative", key=f"alt_{item['id']}", use_container_width=True):
                st.session_state["selected_property_id"] = item["id"]
                st.session_state["current_page"] = "Virtual Staging"
                st.rerun()

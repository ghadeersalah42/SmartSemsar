import streamlit as st
from components.cards import page_header
from services.property_service import get_property_by_id, get_properties


def render_booking() -> None:
    page_header("Next step", "Schedule a Viewing", "Create a demo viewing request. This form does not contact a customer or booking service.")
    selected_id = st.session_state.get("selected_property_id", "PROP-001")
    selected = get_property_by_id(selected_id) or get_properties()[0]
    with st.container(border=True):
        st.subheader(selected["title"])
        st.write(f"{selected['location']} · EGP {selected['price']:,.0f}")
    with st.form("booking_form"):
        name = st.text_input("Customer name", st.session_state.get("customer_profile", {}).get("name", ""))
        phone = st.text_input("Phone number", st.session_state.get("customer_profile", {}).get("phone", ""))
        date = st.date_input("Preferred date")
        time = st.selectbox("Preferred time", ["10:00", "12:00", "14:00", "16:00", "18:00"])
        notes = st.text_area("Additional notes")
        submitted = st.form_submit_button("Confirm demo request", type="primary", use_container_width=True)
    if submitted:
        st.session_state["booking_data"] = {
            "property_id": selected["id"], "customer_name": name,
            "phone": phone, "date": str(date), "time": time, "notes": notes,
            "status": "Demo request saved in this session",
        }
        st.success("Demo viewing request saved. No message was sent and no external booking was created.")
        st.json(st.session_state["booking_data"])

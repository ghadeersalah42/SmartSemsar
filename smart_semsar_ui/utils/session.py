import streamlit as st


def initialize_session_state() -> None:
    defaults = {
        "current_page": "Dashboard",
        "uploaded_audio_name": None,
        "audio_demo_result": None,
        "customer_profile": {
            "name": "Ahmed Mohamed",
            "phone": "+20 100 123 4567",
            "budget": 5_000_000,
            "location": "New Cairo",
            "property_type": "Apartment",
            "bedrooms": 3,
            "area_sqm": 150,
            "purpose": "Family Living",
            "finishing": "Fully Finished",
            "payment_plan": "Installments",
        },
        "properties": None,
        "selected_property_id": "PROP-001",
        "favorite_property_ids": [],
        "booking_data": {},
        "staging_preferences": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value

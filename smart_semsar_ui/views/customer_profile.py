import streamlit as st
from components.cards import page_header


def render_customer_profile() -> None:
    page_header(
        "Customer intelligence",
        "Customer Profile",
        "Review and edit the customer's real-estate requirements. Data is stored only for this demo session.",
    )
    defaults = {
        "name": "Ahmed Mohamed", "phone": "+20 100 123 4567",
        "budget": 5_000_000, "location": "New Cairo",
        "property_type": "Apartment", "bedrooms": 3, "area_sqm": 150,
        "purpose": "Family Living", "finishing": "Fully Finished",
        "payment_plan": "Installments",
    }
    current = st.session_state.get("customer_profile") or defaults
    locations = ["New Cairo", "New Administrative Capital", "Sheikh Zayed", "6th of October", "Maadi", "Nasr City"]
    types = ["Apartment", "Villa", "Duplex", "Penthouse", "Townhouse", "Twin House"]
    purposes = ["Family Living", "Investment", "Vacation Home", "Rental Income"]
    finishings = ["Fully Finished", "Semi Finished", "Core & Shell", "Any"]
    plans = ["Cash", "Installments", "Any"]

    with st.form("customer_profile_form"):
        left, right = st.columns(2)
        with left:
            name = st.text_input("Customer name", current.get("name", ""))
            phone = st.text_input("Phone number", current.get("phone", ""))
            budget = st.number_input("Maximum budget (EGP)", min_value=0, value=int(current.get("budget") or 0), step=100_000)
            location = st.selectbox("Preferred location", locations, index=locations.index(current["location"]) if current.get("location") in locations else 0)
            property_type = st.selectbox("Property type", types, index=types.index(current["property_type"]) if current.get("property_type") in types else 0)
        with right:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=int(current.get("bedrooms") or 3))
            area = st.number_input("Preferred area (m²)", min_value=20, max_value=5000, value=int(current.get("area_sqm") or 150), step=10)
            purpose = st.selectbox("Purpose", purposes, index=purposes.index(current["purpose"]) if current.get("purpose") in purposes else 0)
            finishing = st.selectbox("Finishing", finishings, index=finishings.index(current["finishing"]) if current.get("finishing") in finishings else 0)
            payment_plan = st.selectbox("Payment plan", plans, index=plans.index(current["payment_plan"]) if current.get("payment_plan") in plans else 0)
        submitted = st.form_submit_button("Save Profile", type="primary", use_container_width=True)

    if submitted:
        st.session_state["customer_profile"] = {
            "name": name, "phone": phone, "budget": budget, "location": location,
            "property_type": property_type, "bedrooms": bedrooms, "area_sqm": area,
            "purpose": purpose, "finishing": finishing, "payment_plan": payment_plan,
        }
        st.success("Profile saved for this session.")
        st.rerun()

    st.divider()
    if st.button("Browse matching properties", type="primary", use_container_width=True):
        st.session_state["current_page"] = "Properties"
        st.rerun()

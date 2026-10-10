import streamlit as st

from components.sidebar import render_sidebar
from utils.navigation import get_current_page
from utils.session import initialize_session_state
from views.dashboard import render_dashboard
from views.voice_analysis import render_voice_analysis
from views.customer_profile import render_customer_profile
from views.properties import render_properties
from views.recommendations import render_recommendations
from views.alternatives import render_alternatives
from views.virtual_staging import render_virtual_staging
from views.booking import render_booking

st.set_page_config(
    page_title="Smart Semsar",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()
render_sidebar()

page = get_current_page()
routes = {
    "Dashboard": render_dashboard,
    "Voice Analysis": render_voice_analysis,
    "Customer Profile": render_customer_profile,
    "Properties": render_properties,
    "Recommendations": render_recommendations,
    "Alternatives": render_alternatives,
    "Virtual Staging": render_virtual_staging,
    "Booking": render_booking,
}
routes.get(page, render_dashboard)()

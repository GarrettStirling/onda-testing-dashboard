import os

# Local laptops: newer google-auth probes GCE metadata for universe domain unless set.
os.environ.setdefault("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "googleapis.com")

import streamlit as st

from streamlit_dashboard.forecast_tab import render_forecast_tab
from streamlit_dashboard.k_coefficients import render_k_coefficients_tab
from streamlit_dashboard.map_tab import render_map_tab


st.set_page_config(
    page_title="ONDA Testing Dashboard",
    layout="wide",
)

tabs = st.tabs(["K Coefficients", "Map", "Forecasts"])

with tabs[0]:
    render_k_coefficients_tab()

with tabs[1]:
    render_map_tab()

with tabs[2]:
    render_forecast_tab()


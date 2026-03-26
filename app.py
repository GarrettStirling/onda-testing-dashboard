import os

# Local laptops: newer google-auth probes GCE metadata for universe domain unless set.
os.environ.setdefault("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "googleapis.com")

import streamlit as st

from streamlit_dashboard.cdip_buoy_spectra_tab import render_cdip_buoy_spectra_tab
from streamlit_dashboard.forecast_tab import render_forecast_tab
from streamlit_dashboard.gfs_wave_components_tab import render_gfs_wave_components_tab
from streamlit_dashboard.gfs_wave_radar_tab import render_gfs_wave_radar_tab
from streamlit_dashboard.k_coefficients import render_k_coefficients_tab


st.set_page_config(
    page_title="ONDA Testing Dashboard",
    layout="wide",
)

tabs = st.tabs(
    [
        "K Coefficients",
        "Forecasts",
        "CDIP buoy spectra",
        "GFS wave components",
        "GFS wave radar",
    ]
)

with tabs[0]:
    render_k_coefficients_tab()

with tabs[1]:
    render_forecast_tab()

with tabs[2]:
    render_cdip_buoy_spectra_tab()

with tabs[3]:
    render_gfs_wave_components_tab()

with tabs[4]:
    render_gfs_wave_radar_tab()


import os

# Local laptops: newer google-auth probes GCE metadata for universe domain unless set.
os.environ.setdefault("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "googleapis.com")

import streamlit as st

from streamlit_dashboard.cdip_buoy_spectra_tab import render_cdip_buoy_spectra_tab
from streamlit_dashboard.forecast_tab import render_forecast_tab
from streamlit_dashboard.gfs_wave_components_tab import render_gfs_wave_components_tab
from streamlit_dashboard.k_coefficients import render_k_coefficients_tab
from streamlit_dashboard.qc_calibration_tab import render_qc_calibration_tab
from streamlit_dashboard.scaling_comparison_tab import render_scaling_comparison_tab


st.set_page_config(
    page_title="ONDA Testing Dashboard",
    layout="wide",
)

tabs = st.tabs(
    [
        "K Coefficients",
        "Forecasts",
        "QC Calibration",
        "CDIP buoy spectra",
        "GFS wave components",
        "Scaling compare (temp)",
    ]
)

with tabs[0]:
    render_k_coefficients_tab()

with tabs[1]:
    render_forecast_tab()

with tabs[2]:
    render_qc_calibration_tab()

with tabs[3]:
    render_cdip_buoy_spectra_tab()

with tabs[4]:
    render_gfs_wave_components_tab()

with tabs[5]:
    render_scaling_comparison_tab()


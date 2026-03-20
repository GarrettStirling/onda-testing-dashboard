import streamlit as st

from streamlit_dashboard.k_coefficients import render_k_coefficients_tab
from streamlit_dashboard.map_tab import render_map_tab


st.set_page_config(
    page_title="ONDA Testing Dashboard",
    layout="wide",
)

tabs = st.tabs(["K Coefficients", "Map"])

with tabs[0]:
    render_k_coefficients_tab()

with tabs[1]:
    render_map_tab()


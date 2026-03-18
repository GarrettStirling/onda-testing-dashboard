import streamlit as st

from streamlit_dashboard.k_coefficients import render_k_coefficients_tab


st.set_page_config(
    page_title="ONDA Testing Dashboard",
    layout="wide",
)

tabs = st.tabs(["K Coefficients"])

with tabs[0]:
    render_k_coefficients_tab()


from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
GFS_COMPONENTS_CSV = REPO_ROOT / "data" / "NOAA GFS Wave" / "gfs_wave_components.csv"
M_TO_FT = 3.28084


@st.cache_data(show_spinner=False)
def _load_components(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.sort_values(["point_id", "time_utc", "component_rank"])
    return df


def _component_multiline(df: pd.DataFrame, y_col: str, title: str, y_label: str):
    fig = px.line(
        df,
        x="time_utc",
        y=y_col,
        color="component_label",
        markers=True,
        category_orders={"component_label": ["primary", "secondary", "tertiary"]},
        labels={
            "time_utc": "Time (UTC)",
            y_col: y_label,
            "component_label": "Component",
        },
        title=title,
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render_gfs_wave_components_tab() -> None:
    st.header("GFS wave forecast components")
    st.caption("Source: `data/NOAA GFS Wave/gfs_wave_components.csv`")

    if not GFS_COMPONENTS_CSV.exists():
        st.error(f"Missing file: `{GFS_COMPONENTS_CSV}`")
        return

    df = _load_components(str(GFS_COMPONENTS_CSV))
    if df.empty:
        st.warning("No rows found in GFS components CSV.")
        return

    points = (
        df[["point_id", "point_name"]]
        .drop_duplicates()
        .sort_values(["point_id"])
        .assign(label=lambda x: x["point_id"] + " - " + x["point_name"].astype(str))
    )
    point_labels = points["label"].tolist()
    selected_label = st.selectbox("Offshore point", point_labels, index=0)
    selected_id = selected_label.split(" - ", 1)[0]

    sub = df[df["point_id"] == selected_id].copy()
    if sub.empty:
        st.warning("No rows for selected point.")
        return

    sub["component_wave_height_ft"] = pd.to_numeric(sub["component_wave_height_m"], errors="coerce") * M_TO_FT

    st.plotly_chart(
        _component_multiline(
            sub,
            "component_wave_height_ft",
            "Wave height by component",
            "Wave height (ft)",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        _component_multiline(
            sub,
            "component_period_sec",
            "Period by component",
            "Period (s)",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        _component_multiline(
            sub,
            "component_direction_deg",
            "Direction by component",
            "Direction (deg)",
        ),
        use_container_width=True,
    )

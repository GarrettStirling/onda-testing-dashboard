from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
GFS_COMPONENTS_PARQUET = REPO_ROOT / "data" / "NOAA GFS Wave" / "gfs_wave_components.parquet"
GFS_COMPONENTS_CSV = REPO_ROOT / "data" / "NOAA GFS Wave" / "gfs_wave_components.csv"
M_TO_FT = 3.28084


@st.cache_data(show_spinner=False)
def _load_components(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # Support both parquet (`forecast_time_pst`) and older CSV (`time_utc`) schemas.
    if "forecast_time_pst" in df.columns:
        df["forecast_time_pst"] = pd.to_datetime(df["forecast_time_pst"], errors="coerce")
    elif "time_utc" in df.columns:
        df["forecast_time_pst"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    else:
        df["forecast_time_pst"] = pd.NaT

    # Be robust to mixed/object types from different export environments.
    for col in ["component_wave_height_m", "component_period_sec", "component_direction_deg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["point_id", "forecast_time_pst", "component_rank"])
    return df


def _resolve_components_source() -> Path | None:
    if GFS_COMPONENTS_PARQUET.exists():
        return GFS_COMPONENTS_PARQUET
    if GFS_COMPONENTS_CSV.exists():
        return GFS_COMPONENTS_CSV
    return None


def _component_multiline(df: pd.DataFrame, y_col: str, title: str, y_label: str):
    fig = px.line(
        df,
        x="forecast_time_pst",
        y=y_col,
        color="component_label",
        markers=True,
        category_orders={"component_label": ["primary", "secondary", "tertiary"]},
        labels={
            "forecast_time_pst": "Time (PST/PDT)",
            y_col: y_label,
            "component_label": "Component",
        },
        title=title,
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render_gfs_wave_components_tab() -> None:
    st.header("GFS wave forecast components")
    source = _resolve_components_source()
    if source is None:
        st.error(
            "Missing file: expected either "
            f"`{GFS_COMPONENTS_PARQUET}` or `{GFS_COMPONENTS_CSV}`"
        )
        return
    st.caption(f"Source: `{source}`")

    df = _load_components(str(source))
    if df.empty:
        st.warning("No rows found in GFS components data.")
        return

    points = (
        df[["point_id", "point_name"]]
        .drop_duplicates()
        .sort_values(["point_id"])
        .assign(label=lambda x: x["point_id"] + " - " + x["point_name"].astype(str))
    )

    for _, row in points.iterrows():
        point_id = str(row["point_id"])
        point_name = str(row["point_name"])
        st.subheader(f"{point_id} - {point_name}")
        sub = df[df["point_id"] == point_id].copy()
        if sub.empty:
            st.caption(f"No data for `{point_id}`.")
            continue

        sub["component_wave_height_ft"] = pd.to_numeric(sub["component_wave_height_m"], errors="coerce") * M_TO_FT
        n_h = sub["component_wave_height_ft"].notna().sum()
        n_p = pd.to_numeric(sub["component_period_sec"], errors="coerce").notna().sum()
        n_d = pd.to_numeric(sub["component_direction_deg"], errors="coerce").notna().sum()
        if (n_h + n_p + n_d) == 0:
            st.caption(f"No data for `{point_id}`.")
            st.divider()
            continue

        st.plotly_chart(
            _component_multiline(
                sub,
                "component_wave_height_ft",
                "Wave height by component",
                "Wave height (ft)",
            ),
            width="stretch",
            key=f"gfs_components_height_{point_id}",
        )
        st.plotly_chart(
            _component_multiline(
                sub,
                "component_period_sec",
                "Period by component",
                "Period (s)",
            ),
            width="stretch",
            key=f"gfs_components_period_{point_id}",
        )
        st.plotly_chart(
            _component_multiline(
                sub,
                "component_direction_deg",
                "Direction by component",
                "Direction (deg)",
            ),
            width="stretch",
            key=f"gfs_components_direction_{point_id}",
        )
        st.divider()

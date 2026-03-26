from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
GFS_ENERGY_PARQUET = REPO_ROOT / "data" / "NOAA GFS Wave" / "gfs_wave_energy_2d_long.parquet"
GFS_ENERGY_CSV = REPO_ROOT / "data" / "NOAA GFS Wave" / "gfs_wave_energy_2d_long.csv"


@st.cache_data(show_spinner=False)
def _load_energy(path: str) -> pd.DataFrame:
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

    df["energy_density"] = pd.to_numeric(df["energy_density"], errors="coerce").fillna(0.0)
    df["direction_deg"] = pd.to_numeric(df["direction_deg"], errors="coerce")
    df["period_sec"] = pd.to_numeric(df["period_sec"], errors="coerce")
    return df.sort_values(["point_id", "forecast_time_pst", "direction_deg", "period_sec"])


def _resolve_energy_source() -> Path | None:
    if GFS_ENERGY_PARQUET.exists():
        return GFS_ENERGY_PARQUET
    if GFS_ENERGY_CSV.exists():
        return GFS_ENERGY_CSV
    return None


def render_gfs_wave_radar_tab() -> None:
    st.header("GFS wave radar evolution")
    source = _resolve_energy_source()
    if source is None:
        st.error(
            "Missing file: expected either "
            f"`{GFS_ENERGY_PARQUET}` or `{GFS_ENERGY_CSV}`"
        )
        return
    st.caption(f"Animated **full 2D spectra** (direction x period) from `{source}`.")

    df = _load_energy(str(source))
    if df.empty:
        st.warning("No rows found in GFS energy data.")
        return

    points = (
        df[["point_id", "point_name"]]
        .drop_duplicates()
        .sort_values(["point_id"])
        .assign(label=lambda x: x["point_id"] + " - " + x["point_name"].astype(str))
    )
    selected_label = st.selectbox("Offshore point", points["label"].tolist(), index=0, key="gfs_radar_point")
    point_id = selected_label.split(" - ", 1)[0]
    point_df = df[df["point_id"] == point_id].copy()
    if point_df.empty:
        st.warning("No rows for selected point.")
        return

    point_df = point_df.dropna(subset=["direction_deg", "period_sec"]).copy()
    if point_df.empty:
        st.warning("No valid direction/period rows for selected point.")
        return

    # Full 2D grid can span many orders of magnitude; log10 color scale improves contrast.
    point_df["energy_for_color"] = point_df["energy_density"].clip(lower=1e-12)
    point_df["log10_energy"] = np.log10(point_df["energy_for_color"])
    point_df["time_frame"] = point_df["forecast_time_pst"].dt.strftime("%Y-%m-%d %H:%M PST/PDT")

    c_min = float(point_df["log10_energy"].quantile(0.02))
    c_max = float(point_df["log10_energy"].quantile(0.995))
    if not np.isfinite(c_min) or not np.isfinite(c_max) or c_max <= c_min:
        c_min, c_max = -12.0, -2.0

    fig = px.scatter_polar(
        point_df,
        r="period_sec",
        theta="direction_deg",
        color="log10_energy",
        animation_frame="time_frame",
        color_continuous_scale="Turbo",
        range_color=[c_min, c_max],
        hover_data={
            "period_sec": ":.2f",
            "direction_deg": ":.1f",
            "energy_density": ":.6f",
            "log10_energy": ":.2f",
        },
    )
    fig.update_traces(
        mode="markers",
        marker=dict(size=7, opacity=0.95, symbol="square"),
        selector=dict(type="scatterpolar"),
    )
    fig.update_layout(
        height=760,
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90),
            radialaxis=dict(title="Period (s)", autorange="reversed"),
        ),
        coloraxis_colorbar=dict(title="log10(E)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

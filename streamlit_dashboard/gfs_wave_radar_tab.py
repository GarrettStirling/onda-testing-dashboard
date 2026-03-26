from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
GFS_ENERGY_CSV = REPO_ROOT / "data" / "NOAA GFS Wave" / "gfs_wave_energy_2d_long.csv"


@st.cache_data(show_spinner=False)
def _load_energy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["energy_density"] = pd.to_numeric(df["energy_density"], errors="coerce").fillna(0.0)
    df["period_sec"] = pd.to_numeric(df["period_sec"], errors="coerce")
    return df.sort_values(["point_id", "time_utc", "direction_deg", "frequency_hz"])


def render_gfs_wave_radar_tab() -> None:
    st.header("GFS wave radar evolution")
    st.caption("Animated directional-frequency energy view from `gfs_wave_energy_2d_long.csv`.")

    if not GFS_ENERGY_CSV.exists():
        st.error(f"Missing file: `{GFS_ENERGY_CSV}`")
        return

    df = _load_energy(str(GFS_ENERGY_CSV))
    if df.empty:
        st.warning("No rows found in GFS energy CSV.")
        return

    points = (
        df[["point_id", "point_name"]]
        .drop_duplicates()
        .sort_values(["point_id"])
        .assign(label=lambda x: x["point_id"] + " - " + x["point_name"].astype(str))
    )
    selected_label = st.selectbox("Offshore point", points["label"].tolist(), index=0)
    point_id = selected_label.split(" - ", 1)[0]

    max_bins_per_frame = st.slider("Max bins per timestep", min_value=50, max_value=400, value=180, step=10)
    point_df = df[df["point_id"] == point_id].copy()
    if point_df.empty:
        st.warning("No rows for selected point.")
        return

    # Keep the strongest bins per timestep so animation is responsive.
    point_df = (
        point_df.sort_values(["time_utc", "energy_density"], ascending=[True, False])
        .groupby("time_utc", as_index=False)
        .head(max_bins_per_frame)
    )
    point_df["time_frame"] = point_df["time_utc"].dt.strftime("%Y-%m-%d %H:%M UTC")

    e_max = float(point_df["energy_density"].quantile(0.995)) if len(point_df) > 10 else float(point_df["energy_density"].max())
    e_max = max(e_max, 1e-8)

    fig = px.scatter_polar(
        point_df,
        r="period_sec",
        theta="direction_deg",
        color="energy_density",
        size="energy_density",
        animation_frame="time_frame",
        color_continuous_scale="Turbo",
        range_color=[0.0, e_max],
        size_max=9,
        hover_data={
            "frequency_hz": ":.3f",
            "period_sec": ":.2f",
            "direction_deg": ":.1f",
            "energy_density": ":.6f",
        },
    )
    fig.update_layout(
        height=760,
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90),
            radialaxis=dict(title="Period (s)"),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

"""Temporary tab: compare four GFS scaling variants (buoy components) per break.

Stacks forecasts vertically in order: ALLSCALED, ONLYANCHORING, ONLYKCOEFF, NOSCALE.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_dashboard.forecast_tab import (
    BREAKS_CSV,
    REPO_ROOT,
    _empty_cdip_join_columns,
    _load_break_labels,
    _load_buoy_forecast,
    _plot_break_forecast,
)

FORECAST_DIR = REPO_ROOT / "data" / "forecasts"

# Order matches user request (top → bottom).
VARIANTS: list[tuple[str, str]] = [
    ("ALLSCALED", "20260327_GFS_ALLSCALED_buoy_scaled_components.csv"),
    ("ONLYANCHORING", "20260327_GFS_ONLYANCHORING_buoy_scaled_components.csv"),
    ("ONLYKCOEFF", "20260327_GFS_ONLYKCOEFF_buoy_scaled_components.csv"),
    ("NOSCALE", "20260327_GFS_NOSCALE_buoy_scaled_components.csv"),
]


def _collapse_duplicate_times(df: pd.DataFrame) -> pd.DataFrame:
    """Mean numeric columns per (break_id, wave_time_pst); first non-numeric (e.g. bool)."""
    if df.empty:
        return df
    key = ["break_id", "wave_time_pst"]
    agg: dict[str, str] = {}
    for c in df.columns:
        if c in key:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            agg[c] = "mean"
        else:
            agg[c] = "first"
    out = df.groupby(key, as_index=False).agg(agg)
    return out.sort_values(key)


@st.cache_data(show_spinner=False)
def _load_scaling_variant_csv(path_str: str) -> pd.DataFrame:
    df = _load_buoy_forecast(path_str)
    return _collapse_duplicate_times(df)


def render_scaling_comparison_tab() -> None:
    st.header("Scaling comparison (temporary)")
    st.caption(
        "Four GFS scaling runs stacked per spot (top → bottom): ALLSCALED, ONLYANCHORING, "
        "ONLYKCOEFF, NOSCALE. Buoy components only (no CDIP overlay)."
    )

    missing: list[str] = []
    paths_ok: list[tuple[str, Path]] = []
    for short, fname in VARIANTS:
        p = FORECAST_DIR / fname
        if not p.exists():
            missing.append(fname)
        else:
            paths_ok.append((short, p))

    if missing:
        st.warning("Missing CSV(s) under `data/forecasts/`: " + ", ".join(missing))

    if not paths_ok:
        st.error("No variant files found — add the dated `*_buoy_scaled_components.csv` files.")
        return

    labels = _load_break_labels(str(BREAKS_CSV))

    break_ids: set[int] = set()
    for short, p in paths_ok:
        df = _load_scaling_variant_csv(str(p))
        if not df.empty:
            break_ids.update(int(x) for x in df["break_id"].dropna().unique())
    break_ids_sorted = sorted(break_ids)

    if not break_ids_sorted:
        st.warning("No break_id values found in the loaded CSVs.")
        return

    id_to_label = {bid: labels.get(bid, f"Break {bid}") for bid in break_ids_sorted}

    # Four full-height charts per spot — default to a small subset so first load stays responsive.
    default_n = min(12, len(break_ids_sorted))
    default_sel = break_ids_sorted[:default_n]

    selected = st.multiselect(
        "Surf spots (breaks)",
        options=break_ids_sorted,
        format_func=lambda bid: id_to_label.get(bid, f"Break {bid}"),
        default=default_sel,
        help="Add more breaks from the list if needed; each spot renders four stacked figures.",
    )

    if not selected:
        st.info("Select at least one break.")
        return

    for bid in selected:
        spot_label = id_to_label.get(bid, f"Break {bid}")
        st.subheader(spot_label)

        for short, p in paths_ok:
            df = _load_scaling_variant_csv(str(p))
            df_b = df[df["break_id"] == bid].copy()
            if df_b.empty:
                st.caption(f"**{short}** — no rows for this break.")
                continue

            joined = _empty_cdip_join_columns(df_b)
            title = f"{short} — {spot_label}"
            try:
                fig = _plot_break_forecast(
                    joined,
                    show_cdip_sig=False,
                    overlay_cdip_mop=False,
                    label=title,
                )
                st.plotly_chart(fig, width="stretch", key=f"scale_cmp_{bid}_{short}")
            except Exception as exc:
                st.error(f"{short}: plot failed — {exc}")

        st.divider()

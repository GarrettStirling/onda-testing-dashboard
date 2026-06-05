"""Observation QC tab: distribution of captured observations per break.

Data source: ``onda-maverick.surf_calibration_data.observations_with_cdip`` (BigQuery).
Breaks ranked most → least by observation count; 4 histograms per break.
Display labels follow the SURF_SPOT_MAPPING_GUIDE format: ``Spot — Break``.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ID = "onda-maverick"
OBS_TABLE = f"`{PROJECT_ID}.surf_calibration_data.observations_with_cdip`"

BG_DARK = "#0e1117"
BG_PANEL = "#161b22"
GRID_COLOR = "#2a2d35"
TEXT_COLOR = "#c9d1d9"

# (column, panel title, bar color)
HIST_PANELS: list[tuple[str, str, str]] = [
    ("observed_hs_ft",      "Observed Wave Height (ft)", "#38bdf8"),
    ("per_weighted_s_mop",  "Per Weighted (s MOP)",      "#818cf8"),
    ("dir_mean_deg_buoy",   "Mean Direction — Buoy (°)", "#34d399"),
    ("hs_mop_ft",           "MOP Hs (ft)",               "#fb923c"),
]


# ---------------------------------------------------------------------------
# BigQuery helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Loading observations from BigQuery…")
def _load_obs_bigquery(_cache_buster: int = 0) -> pd.DataFrame:
    """Fetch all rows from observations_with_cdip.  Cache busted via _cache_buster."""
    from streamlit_dashboard.bq_forecast_loader import forecast_bigquery_client

    client = forecast_bigquery_client()
    sql = f"SELECT * FROM {OBS_TABLE}"
    job = client.query(sql)
    return job.to_dataframe(create_bqstorage_client=False)


def _force_refresh() -> None:
    """Increment the cache-buster counter so the next load bypasses the TTL."""
    st.session_state["obs_qc_cache_buster"] = (
        st.session_state.get("obs_qc_cache_buster", 0) + 1
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _spot_hist_figure(df_spot: pd.DataFrame, *, spot_label: str, n_obs: int) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[lbl for _, lbl, _ in HIST_PANELS],
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    for i, (col, label, color) in enumerate(HIST_PANELS):
        r = i // 2 + 1
        c = i % 2 + 1
        if col not in df_spot.columns:
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0],
                    mode="text",
                    text=[f"Column '{col}' not found in table"],
                    textfont=dict(color="#64748b", size=11),
                    showlegend=False,
                ),
                row=r, col=c,
            )
            continue

        vals = pd.to_numeric(df_spot[col], errors="coerce").dropna()
        if vals.empty:
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0],
                    mode="text",
                    text=["No non-null values"],
                    textfont=dict(color="#64748b", size=11),
                    showlegend=False,
                ),
                row=r, col=c,
            )
            continue

        fig.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=40,
                marker_color=color,
                marker_line=dict(width=0.5, color=BG_DARK),
                opacity=0.85,
                name=label,
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Range: %{x}<br>"
                    "Count: %{y}"
                    "<extra></extra>"
                ),
            ),
            row=r,
            col=c,
        )

    title_html = (
        f"{spot_label}"
        f"<span style='font-size:12px; color:#64748b'>  {n_obs:,} observations</span>"
    )
    fig.update_layout(
        title=dict(
            text=title_html,
            font=dict(color=TEXT_COLOR, size=15),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_COLOR, size=11),
        margin=dict(l=56, r=28, t=100, b=56),
        height=580,
        bargap=0.03,
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        showgrid=True,
        zeroline=False,
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        showgrid=True,
        zeroline=False,
        title_text="Count",
        tickfont=dict(size=10),
    )
    fig.update_annotations(font=dict(color=TEXT_COLOR, size=11))

    return fig


# ---------------------------------------------------------------------------
# Break label helpers  (same convention as qc_calibration_tab)
# ---------------------------------------------------------------------------

def _break_display_label(row: pd.Series) -> str:
    """Return 'Spot — Break' (or single name when equal/missing)."""
    spot = str(row.get("spot_name") or row.get("spot") or "").strip()
    brk  = str(row.get("break_name") or row.get("break") or "").strip()
    bid_raw = row.get("break_id")
    try:
        bid = int(bid_raw) if pd.notna(bid_raw) else 0
    except (TypeError, ValueError):
        bid = 0

    if spot and brk and spot.lower() != brk.lower():
        return f"{spot} — {brk}"
    if brk:
        return brk
    if spot:
        return spot
    return f"Break {bid}"


def _build_break_index(df: pd.DataFrame) -> tuple[list[int], dict[int, str]]:
    """
    Return (break_ids_sorted_by_count_desc, label_by_break_id).

    Falls back to a synthetic break_id derived from the spot column when
    break_id is absent from the table.
    """
    if "break_id" not in df.columns:
        # No break_id — synthesise one from spot name so the rest of the code
        # stays uniform.
        spot_col = next(
            (c for c in ("spot_name", "spot", "location_name", "break_name")
             if c in df.columns and df[c].notna().any()),
            None,
        )
        if spot_col is None:
            raise ValueError(
                "Table has no `break_id` and no spot/location name column. "
                f"Columns present: {list(df.columns)}"
            )
        cats = df[spot_col].astype("category")
        df["break_id"] = cats.cat.codes
        # Inject a matching break_name so label logic picks it up
        df["break_name"] = df[spot_col]

    df = df.copy()
    df["break_id"] = pd.to_numeric(df["break_id"], errors="coerce")
    df = df.loc[df["break_id"].notna()]
    df["break_id"] = df["break_id"].astype(int)

    counts = df.groupby("break_id").size().sort_values(ascending=False)
    sorted_ids = counts.index.tolist()

    label_by_bid: dict[int, str] = {}
    for bid in sorted_ids:
        first_row = df.loc[df["break_id"] == bid].iloc[0]
        label_by_bid[bid] = _break_display_label(first_row)

    return sorted_ids, label_by_bid, df


# ---------------------------------------------------------------------------
# Tab entry point
# ---------------------------------------------------------------------------

def render_obs_qc_tab() -> None:
    st.header("Observation QC")
    st.caption(
        "Distribution of captured observations per break from "
        f"`{OBS_TABLE}`. "
        "Breaks ranked **most → least** observations. "
        "Labels follow the `Spot — Break` convention."
    )

    # ── Controls row ──────────────────────────────────────────────────────
    ctrl_left, ctrl_right = st.columns([3, 1])
    with ctrl_left:
        use_bq = st.toggle(
            "Load observations from BigQuery",
            value=False,
            key="obs_qc_use_bq",
            help=(
                f"Queries `{OBS_TABLE}`. "
                "Results are cached for 10 minutes. "
                "Use **Refresh** to force a new fetch."
            ),
        )
    with ctrl_right:
        refresh_clicked = st.button(
            "↺ Refresh",
            disabled=not use_bq,
            help="Clear the cache and re-query BigQuery.",
            use_container_width=True,
        )

    if refresh_clicked:
        _force_refresh()
        st.toast("Cache cleared — fetching fresh data…", icon="🔄")

    if not use_bq:
        st.info(
            "Enable **Load observations from BigQuery** above to fetch data. "
            "Results are cached for 10 minutes; use **↺ Refresh** to force a reload."
        )
        return

    # ── Load data ─────────────────────────────────────────────────────────
    cache_buster = st.session_state.get("obs_qc_cache_buster", 0)
    try:
        df = _load_obs_bigquery(cache_buster)
    except Exception as exc:
        st.error(f"BigQuery load failed: {exc}")
        st.caption(
            "Make sure you have credentials configured "
            "(`gcloud auth application-default login` or Streamlit secrets)."
        )
        return

    if df.empty:
        st.warning("No observations returned from BigQuery.")
        return

    # ── Build break index (break_id → label, sorted by count desc) ────────
    try:
        sorted_ids, label_by_bid, df = _build_break_index(df)
    except ValueError as exc:
        st.error(str(exc))
        return

    break_counts: dict[int, int] = {
        bid: int((df["break_id"] == bid).sum()) for bid in sorted_ids
    }

    # ── Summary expander ──────────────────────────────────────────────────
    with st.expander("Data summary", expanded=False):
        m1, m2 = st.columns(2)
        m1.metric("Total observations", f"{len(df):,}")
        m2.metric("Unique breaks", f"{len(sorted_ids):,}")

        st.markdown("**Histogram column status**")
        col_status_rows = []
        for col, label, _ in HIST_PANELS:
            if col not in df.columns:
                status = "❌ not found in table"
                null_pct = "—"
            else:
                null_pct = f"{df[col].isna().mean() * 100:.1f}% null"
                status = "✅ found"
            col_status_rows.append({"Column": col, "Panel": label, "Status": status, "Null %": null_pct})
        st.dataframe(col_status_rows, hide_index=True, use_container_width=True)

        st.markdown("**All columns in table**")
        st.code(", ".join(sorted(df.columns.tolist())), language="text")

        st.markdown("**Observations per break (most → least)**")
        st.dataframe(
            pd.DataFrame([
                {"Break": label_by_bid[bid], "break_id": bid, "Observations": break_counts[bid]}
                for bid in sorted_ids
            ]),
            hide_index=True,
            use_container_width=True,
            height=min(400, 35 * len(sorted_ids) + 38),
        )

    # ── Break filter ──────────────────────────────────────────────────────
    selected_ids: list[int] = st.multiselect(
        "Filter breaks (ranked most → least observations)",
        options=sorted_ids,
        default=sorted_ids,
        format_func=lambda bid: f"{label_by_bid[bid]}  ({break_counts.get(bid, 0):,} obs)",
        help="Deselect a break to hide its histograms.",
    )

    if not selected_ids:
        st.info("Select at least one break to see histograms.")
        return

    # ── Histogram grid per break ──────────────────────────────────────────
    for bid in selected_ids:
        sub = df[df["break_id"] == bid]
        label = label_by_bid[bid]
        n = break_counts.get(bid, len(sub))
        fig = _spot_hist_figure(sub, spot_label=label, n_obs=n)
        st.plotly_chart(fig, use_container_width=True)

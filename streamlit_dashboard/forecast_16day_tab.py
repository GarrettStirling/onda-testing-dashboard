"""Forecast (16 day): served swell from Firestore ``surfingConditions``.

Reads ``surfing_breaks/{geohash}`` → ``surfingConditions``, the coalesced
CDIP + GFS series ``ingest_swell`` writes (``wavesHeight`` = calibrated_hs).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from streamlit_dashboard.field_observations import (
    DEFAULT_MIN_SELECTED_OBS,
    default_break_ids_with_min_obs,
    sort_break_ids_by_obs_count,
)
from streamlit_dashboard.bq_forecast_loader import (
    load_calibration_observation_counts_bigquery,
    load_reference_heights_bigquery,
)
from streamlit_dashboard.firestore_forecast_loader import PROJECT_ID, load_served_swell_forecast
from streamlit_dashboard.forecast_tab import (
    ALPHA_OVERLAY,
    ALPHA_SIG_RAW,
    BG_DARK,
    BG_PANEL,
    BREAKS_CSV,
    C_PRIMARY,
    C_SECONDARY,
    C_SIG,
    C_TERTIARY,
    DEFAULT_NEAREST_TOLERANCE_HOURS,
    GRID_COLOR,
    LW_MAIN,
    LW_OVERLAY,
    LW_SEC,
    LW_SIG,
    LW_SIG_RAW,
    LW_TER,
    PST,
    SPINE_COLOR,
    TEXT_COLOR,
    XAXIS_DAY_GRID_COLOR,
    _add_ts_line,
    _forecast_hover_xaxis_tickformat,
    _forecast_panel_layout,
    _heights_to_ft,
    _legend_show_once,
    _load_break_labels,
    _pacific_midnight_ticks_for_plot,
    pacific_wall_clock_for_plot,
    stored_utc_instant_to_pacific,
    _y_range_padded,
)

C_CDIP_MOP_RAW = "#94a3b8"
C_OFFSHORE_BUOY = "#2dd4bf"
C_GFS_HTSGW = "#f472b6"

REF_HEIGHT_COLS = ("cdip_mop_hs_raw_m", "offshore_buoy_hs_m", "gfs_htsgw_m")
CALIBRATED_HEIGHT_LABEL = "Calibrated Height"
REF_HEIGHT_LABELS = {
    "cdip_mop_hs_raw_m": "Uncalibrated CDIP MOP Height",
    "offshore_buoy_hs_m": "Uncalibrated CDIP Buoy Height",
    "gfs_htsgw_m": "Uncalibrated GFS Height",
}
REF_HEIGHT_COLORS = {
    "cdip_mop_hs_raw_m": C_CDIP_MOP_RAW,
    "offshore_buoy_hs_m": C_OFFSHORE_BUOY,
    "gfs_htsgw_m": C_GFS_HTSGW,
}


def _records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "forecast_datetime" not in df.columns:
        return pd.DataFrame()

    # Firestore stores UTC instants (see ingest_swell); display as US/Pacific.
    df["wave_time_pst"] = stored_utc_instant_to_pacific(df["forecast_datetime"])
    df = df.sort_values("wave_time_pst").reset_index(drop=True)
    return df


def _partition_columns(df: pd.DataFrame, prefix: str) -> tuple[str, str, str] | None:
    h, p, d = f"{prefix}1Height", f"{prefix}1Period", f"{prefix}1Direction"
    if h in df.columns and df[h].notna().any():
        return h, p, d
    return None


def _merge_reference_heights(
    fs_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    *,
    tolerance_hours: float = DEFAULT_NEAREST_TOLERANCE_HOURS,
) -> pd.DataFrame:
    """Attach BQ reference bulk-Hs columns to served rows (nearest time match)."""
    if fs_df.empty or ref_df.empty:
        return fs_df
    out = fs_df.sort_values("wave_time_pst").copy()
    ref = ref_df.copy()
    ref["wave_time_pst"] = stored_utc_instant_to_pacific(ref["wave_time_utc"])
    ref = ref.sort_values("wave_time_pst")
    tol = pd.Timedelta(hours=tolerance_hours)
    for col in REF_HEIGHT_COLS:
        if col not in ref.columns:
            continue
        merged = pd.merge_asof(
            out[["wave_time_pst"]],
            ref[["wave_time_pst", col]],
            on="wave_time_pst",
            direction="nearest",
            tolerance=tol,
        )
        out[col] = merged[col].values
    return out


def _plot_served_forecast(
    df: pd.DataFrame,
    *,
    label: str,
    partition_prefix: str,
    show_raw_height: bool,
    show_cdip_mop_raw: bool,
    show_offshore_buoy: bool,
    show_gfs_htsgw: bool,
    show_direction: bool,
    show_period: bool,
) -> go.Figure:
    df = df.copy().sort_values("wave_time_pst")
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No surfingConditions rows",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=TEXT_COLOR, size=14),
        )
        fig.update_layout(paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL, height=220)
        return fig

    t_x = pacific_wall_clock_for_plot(df["wave_time_pst"])
    span_days = max((t_x.max() - t_x.min()).total_seconds() / 86400.0, 0.25)
    part = _partition_columns(df, partition_prefix)
    part_label = "WW3 deep water" if partition_prefix == "ww3Swell" else "Nearshore"

    panel_titles, row_heights, dir_row, period_row, plot_height = _forecast_panel_layout(
        show_direction, show_period,
    )
    fig = make_subplots(
        rows=len(panel_titles),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=row_heights,
        subplot_titles=panel_titles,
    )
    leg: set[str] = set()

    if "wavesHeight" in df.columns and df["wavesHeight"].notna().any():
        _add_ts_line(
            fig,
            1,
            t_x,
            _heights_to_ft(df["wavesHeight"]),
            name=CALIBRATED_HEIGHT_LABEL,
            color=C_SIG,
            width=LW_SIG,
            unit=" ft",
            showlegend=_legend_show_once(leg, CALIBRATED_HEIGHT_LABEL),
        )

    if show_raw_height and "wavesHeightRaw" in df.columns and df["wavesHeightRaw"].notna().any():
        _add_ts_line(
            fig,
            1,
            t_x,
            _heights_to_ft(df["wavesHeightRaw"]),
            name="wavesHeightRaw",
            color=C_SIG,
            dash="dash",
            width=LW_SEC,
            unit=" ft",
            opacity=0.55,
            showlegend=_legend_show_once(leg, "wavesHeightRaw"),
        )

    if show_cdip_mop_raw and "cdip_mop_hs_raw_m" in df.columns and df["cdip_mop_hs_raw_m"].notna().any():
        _add_ts_line(
            fig, 1, t_x, _heights_to_ft(df["cdip_mop_hs_raw_m"]),
            name=REF_HEIGHT_LABELS["cdip_mop_hs_raw_m"],
            color=C_CDIP_MOP_RAW, dash="dot", width=LW_SIG_RAW, unit=" ft",
            opacity=ALPHA_SIG_RAW,
            showlegend=_legend_show_once(leg, REF_HEIGHT_LABELS["cdip_mop_hs_raw_m"]),
        )
    if show_offshore_buoy and "offshore_buoy_hs_m" in df.columns and df["offshore_buoy_hs_m"].notna().any():
        _add_ts_line(
            fig, 1, t_x, _heights_to_ft(df["offshore_buoy_hs_m"]),
            name=REF_HEIGHT_LABELS["offshore_buoy_hs_m"],
            color=C_OFFSHORE_BUOY, dash="dashdot", width=LW_OVERLAY, unit=" ft",
            opacity=ALPHA_OVERLAY,
            showlegend=_legend_show_once(leg, REF_HEIGHT_LABELS["offshore_buoy_hs_m"]),
        )
    if show_gfs_htsgw and "gfs_htsgw_m" in df.columns and df["gfs_htsgw_m"].notna().any():
        _add_ts_line(
            fig, 1, t_x, _heights_to_ft(df["gfs_htsgw_m"]),
            name=REF_HEIGHT_LABELS["gfs_htsgw_m"],
            color=C_GFS_HTSGW, dash="longdash", width=LW_OVERLAY, unit=" ft",
            opacity=ALPHA_OVERLAY,
            showlegend=_legend_show_once(leg, REF_HEIGHT_LABELS["gfs_htsgw_m"]),
        )

    if part:
        h1, p1, d1 = part[0], part[1], part[2]
        h2, p2, d2 = h1.replace("1", "2"), p1.replace("1", "2"), d1.replace("1", "2")
        h3, p3, d3 = h1.replace("1", "3"), p1.replace("1", "3"), d1.replace("1", "3")

        _add_ts_line(
            fig, 1, t_x, _heights_to_ft(df[h1]),
            name=f"Primary ({part_label})",
            color=C_PRIMARY, width=LW_MAIN, unit=" ft",
            showlegend=_legend_show_once(leg, f"Primary ({part_label})"),
        )
        _add_ts_line(
            fig, 1, t_x, _heights_to_ft(df[h2]),
            name=f"Secondary ({part_label})",
            color=C_SECONDARY, dash="dash", width=LW_SEC, unit=" ft",
            showlegend=_legend_show_once(leg, f"Secondary ({part_label})"),
        )
        if h3 in df.columns and df[h3].notna().any():
            _add_ts_line(
                fig, 1, t_x, _heights_to_ft(df[h3]),
                name=f"Tertiary ({part_label})",
                color=C_TERTIARY, dash="dot", width=LW_TER, unit=" ft",
                showlegend=_legend_show_once(leg, f"Tertiary ({part_label})"),
            )

        if show_direction and dir_row is not None:
            _add_ts_line(
                fig, dir_row, t_x, pd.to_numeric(df[d1], errors="coerce"),
                name=f"Primary ({part_label})",
                color=C_PRIMARY, width=LW_MAIN, unit="°",
                showlegend=False,
            )
            _add_ts_line(
                fig, dir_row, t_x, pd.to_numeric(df[d2], errors="coerce"),
                name=f"Secondary ({part_label})",
                color=C_SECONDARY, dash="dash", width=LW_SEC, unit="°",
                showlegend=False,
            )
            if d3 in df.columns and df[d3].notna().any():
                _add_ts_line(
                    fig, dir_row, t_x, pd.to_numeric(df[d3], errors="coerce"),
                    name=f"Tertiary ({part_label})",
                    color=C_TERTIARY, dash="dot", width=LW_TER, unit="°",
                    showlegend=False,
                )

        if show_period and period_row is not None:
            _add_ts_line(
                fig, period_row, t_x, pd.to_numeric(df[p1], errors="coerce"),
                name=f"Primary ({part_label})",
                color=C_PRIMARY, width=LW_MAIN, unit=" s",
                showlegend=False,
            )
            _add_ts_line(
                fig, period_row, t_x, pd.to_numeric(df[p2], errors="coerce"),
                name=f"Secondary ({part_label})",
                color=C_SECONDARY, dash="dash", width=LW_SEC, unit=" s",
                showlegend=False,
            )
            if p3 in df.columns and df[p3].notna().any():
                _add_ts_line(
                    fig, period_row, t_x, pd.to_numeric(df[p3], errors="coerce"),
                    name=f"Tertiary ({part_label})",
                    color=C_TERTIARY, dash="dot", width=LW_TER, unit=" s",
                    showlegend=False,
                )

    h_for_range: list[pd.Series] = []
    if "wavesHeight" in df.columns:
        h_for_range.append(_heights_to_ft(df["wavesHeight"]))
    for col in REF_HEIGHT_COLS:
        if col in df.columns and df[col].notna().any():
            h_for_range.append(_heights_to_ft(df[col]))
    if part:
        h_for_range.append(_heights_to_ft(df[part[0]]))
        h2 = part[0].replace("1", "2")
        h3 = part[0].replace("1", "3")
        if h2 in df.columns:
            h_for_range.append(_heights_to_ft(df[h2]))
        if h3 in df.columns:
            h_for_range.append(_heights_to_ft(df[h3]))
    y_r_h = _y_range_padded(*h_for_range, frac=0.08, floor_zero=True, min_span=0.35) if h_for_range else None

    y_r_d = None
    if show_direction and part:
        d_for_range: list[pd.Series] = []
        for slot in ("1", "2", "3"):
            d_col = part[2].replace("1", slot)
            if d_col in df.columns:
                d_for_range.append(pd.to_numeric(df[d_col], errors="coerce"))
        if d_for_range:
            y_r_d = _y_range_padded(*d_for_range, frac=0.08, floor_zero=False, min_span=12.0)

    y_r_p = None
    if show_period and part:
        p_for_range: list[pd.Series] = []
        for slot in ("1", "2", "3"):
            p_col = part[1].replace("1", slot)
            if p_col in df.columns:
                p_for_range.append(pd.to_numeric(df[p_col], errors="coerce"))
        if p_for_range:
            y_r_p = _y_range_padded(*p_for_range, frac=0.08, floor_zero=False, min_span=0.75)

    tick_fmt = _forecast_hover_xaxis_tickformat(span_days)
    x_tickvals = _pacific_midnight_ticks_for_plot(df["wave_time_pst"].min(), df["wave_time_pst"].max())

    fig.update_layout(
        title=dict(
            text=f"Forecast — {label}",
            font=dict(color=TEXT_COLOR, size=15),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_COLOR, size=11),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG_PANEL, bordercolor=SPINE_COLOR, font_size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(22,27,34,0.92)",
            bordercolor=SPINE_COLOR,
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=56, r=20, t=96, b=48),
        height=plot_height,
    )
    fig.update_xaxes(
        showspikes=True,
        spikecolor="#8b949e",
        spikesnap="cursor",
        spikemode="across",
        spikethickness=1,
        tickmode="array",
        tickvals=x_tickvals,
        gridcolor=XAXIS_DAY_GRID_COLOR,
        showgrid=True,
        zeroline=False,
        tickformat=tick_fmt,
    )
    bottom_row = period_row or dir_row or 1
    for r in range(1, bottom_row):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    fig.update_xaxes(title_text="Date (US/Pacific)", row=bottom_row, col=1)
    fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True, zeroline=False)
    if y_r_h:
        fig.update_yaxes(range=list(y_r_h), row=1, col=1)
    if show_direction and dir_row is not None and y_r_d is not None:
        fig.update_yaxes(range=list(y_r_d), row=dir_row, col=1)
    if show_period and period_row is not None and y_r_p is not None:
        fig.update_yaxes(range=list(y_r_p), row=period_row, col=1)
    fig.update_annotations(font=dict(color=TEXT_COLOR, size=12))
    return fig


def _fmt_ts(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(PST).strftime("%Y-%m-%d %H:%M %Z")


def render_forecast_16day_tab() -> None:
    st.header("Forecast (16 day)")
    st.caption(
        "Production swell in the app: Firestore `surfing_breaks/{geohash}` → "
        "`surfingConditions`. Calibrated Height is the coalesced CDIP/GFS bulk Hs "
        "(`COALESCE(cdip_calibrated_hs, gfs_calibrated_hs)`); the window extends to the "
        "GFS f384 horizon (~16 days). Chart times are **US/Pacific**; Firestore "
        "`forecast_datetime` values are stored as UTC instants from `ingest_swell`."
    )

    labels = _load_break_labels(str(BREAKS_CSV))
    if not BREAKS_CSV.exists():
        st.error(f"Missing break labels: `{BREAKS_CSV}`")
        return

    break_ids = sorted(pd.read_csv(BREAKS_CSV, usecols=["break_id"])["break_id"].astype(int).unique())

    obs_counts = pd.Series(dtype=int)
    try:
        obs_counts = load_calibration_observation_counts_bigquery()
    except Exception as exc:
        st.warning(f"Could not load observation counts from BigQuery: {exc}")

    sorted_break_ids, obs_count_by_bid = sort_break_ids_by_obs_count(
        [int(b) for b in break_ids],
        obs_counts,
    )
    default_selected = default_break_ids_with_min_obs(
        sorted_break_ids,
        obs_count_by_bid,
        min_obs=DEFAULT_MIN_SELECTED_OBS,
    )
    id_to_label = {bid: labels.get(bid, f"Break {bid}") for bid in sorted_break_ids}

    partition_choice = st.radio(
        "Swell partitions",
        options=["Nearshore (swell1/2/3)", "WW3 deep water (ww3Swell1/2/3)"],
        horizontal=True,
        help="Nearshore partitions come from CDIP MOP in the CDIP window; WW3 partitions are deep-water overlays.",
    )
    partition_prefix = "ww3Swell" if "WW3" in partition_choice else "swell"
    show_raw_height = st.checkbox(
        "Show wavesHeightRaw (pre-override model value)",
        value=False,
    )
    show_cdip_mop_raw = st.checkbox(
        f"Overlay {REF_HEIGHT_LABELS['cdip_mop_hs_raw_m']} (`cdip_data_p.significant_wave_height_raw`)",
        value=True,
    )
    show_offshore_buoy = st.checkbox(
        f"Overlay {REF_HEIGHT_LABELS['offshore_buoy_hs_m']} (`offshore_buoy_data_p.significant_wave_height`)",
        value=True,
    )
    show_gfs_htsgw = st.checkbox(
        f"Overlay {REF_HEIGHT_LABELS['gfs_htsgw_m']} (`gfs_offshore_wave_data_p.hs_total_m_gfs`)",
        value=True,
    )

    selected: list[int] = st.multiselect(
        f"Surf spots (breaks, ranked most → least observations; ≥{DEFAULT_MIN_SELECTED_OBS} obs selected by default)",
        options=sorted_break_ids,
        format_func=lambda bid: (
            f"{id_to_label[bid]}  ({obs_count_by_bid[bid]:,} obs)"
            if obs_count_by_bid[bid]
            else f"{id_to_label[bid]}  (0 obs)"
        ),
        default=default_selected,
        help=(
            "Break order and default selection use observation counts from "
            f"BigQuery `calibration_observations`. "
            f"Spots with at least {DEFAULT_MIN_SELECTED_OBS} observations are selected on load."
        ),
    )
    if not selected:
        st.info("Select at least one break.")
        return

    with st.expander("Data source", expanded=False):
        st.write(f"- GCP project: `{PROJECT_ID}`")
        st.write("- Calibrated Height: Firestore `surfing_breaks` → `surfingConditions` (`wavesHeight`)")
        st.write(
            "  - `forecast_datetime` / start/end metadata: **UTC instants** "
            "(CDIP `wave_time_utc`; buoy/GFS `TIMESTAMP(wave_time_pst, 'America/Los_Angeles')`)"
        )
        st.write("  - Charts: converted to **US/Pacific** for display")
        st.write("- Reference overlays (BigQuery):")
        st.write(f"  - {REF_HEIGHT_LABELS['cdip_mop_hs_raw_m']} — `cdip_data_p.significant_wave_height_raw`")
        st.write(f"  - {REF_HEIGHT_LABELS['offshore_buoy_hs_m']} — `offshore_buoy_data_p.significant_wave_height`")
        st.write(f"  - {REF_HEIGHT_LABELS['gfs_htsgw_m']} — `gfs_offshore_wave_data_p.hs_total_m_gfs`")
        st.write("  - Buoy mapping: `surf_intermediates.break_to_buoy_map`")
        st.write("  - GFS point mapping: `surf_intermediates.break_to_gfs_map`")
        st.write("- Cache TTL: 5 minutes (Firestore) / 5 minutes (BigQuery overlays)")
        st.write("- Observation counts / ranking: BigQuery `calibration_observations`")
        if not obs_counts.empty:
            st.write(f"- {int(obs_counts.sum()):,} total observations across breaks")

    selected_set = set(int(b) for b in selected)
    selected_tuple = tuple(sorted(selected_set))

    # Load Firestore payloads first so we know the time window for BQ overlays.
    payloads: dict[int, dict[str, Any]] = {}
    time_mins: list[pd.Timestamp] = []
    time_maxs: list[pd.Timestamp] = []
    for bid in sorted_break_ids:
        if bid not in selected_set:
            continue
        try:
            payload = load_served_swell_forecast(int(bid))
        except Exception as exc:
            st.error(f"Firestore load failed for break {bid}: {exc}")
            st.caption("Try `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS`.")
            continue
        if not payload or not payload.get("surfingConditions"):
            st.warning(f"No `surfingConditions` in Firestore for break {bid}.")
            continue
        df_fs = _records_to_dataframe(payload["surfingConditions"])
        if df_fs.empty:
            continue
        payloads[int(bid)] = payload
        time_mins.append(df_fs["wave_time_pst"].min())
        time_maxs.append(df_fs["wave_time_pst"].max())

    ref_by_break: dict[int, pd.DataFrame] = {}
    if payloads and (show_cdip_mop_raw or show_offshore_buoy or show_gfs_htsgw):
        min_utc = min(time_mins).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
        max_utc = max(time_maxs).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
        try:
            ref_all = load_reference_heights_bigquery(selected_tuple, min_utc, max_utc)
            if not ref_all.empty:
                ref_all["break_id"] = pd.to_numeric(ref_all["break_id"], errors="coerce").astype("Int64")
                for bid in selected_tuple:
                    sub = ref_all[ref_all["break_id"].astype(int) == int(bid)].copy()
                    if not sub.empty:
                        ref_by_break[bid] = sub
        except Exception as exc:
            st.warning(f"BigQuery reference-height load failed: {exc}")

    for bid in sorted_break_ids:
        if bid not in selected_set or bid not in payloads:
            continue
        label = id_to_label.get(bid, f"Break {bid}")
        n_obs = obs_count_by_bid.get(bid, 0)
        st.subheader(f"{label} ({n_obs:,} obs)")

        payload = payloads[bid]
        df = _records_to_dataframe(payload["surfingConditions"])
        if bid in ref_by_break:
            df = _merge_reference_heights(df, ref_by_break[bid])

        n = len(df)
        span_days = 0.0
        if n >= 2:
            span_days = (df["wave_time_pst"].max() - df["wave_time_pst"].min()).total_seconds() / 86400.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{n:,}")
        c2.metric("Span (days)", f"{span_days:.1f}")
        c3.metric("Step (h)", str(payload.get("period_hours") or "—"))
        c4.metric("Geohash", str(payload.get("geohash") or "—"))

        st.caption(
            f"Window: {_fmt_ts(payload.get('start_time'))} → {_fmt_ts(payload.get('end_time'))} · "
            f"Updated: {_fmt_ts(payload.get('updated_at'))}"
        )

        pc1, pc2 = st.columns(2)
        with pc1:
            show_direction = st.checkbox(
                "Show direction subplot",
                value=False,
                key=f"16d_dir_{bid}",
            )
        with pc2:
            show_period = st.checkbox(
                "Show period subplot",
                value=False,
                key=f"16d_per_{bid}",
            )

        fig = _plot_served_forecast(
            df,
            label=label,
            partition_prefix=partition_prefix,
            show_raw_height=show_raw_height,
            show_cdip_mop_raw=show_cdip_mop_raw,
            show_offshore_buoy=show_offshore_buoy,
            show_gfs_htsgw=show_gfs_htsgw,
            show_direction=show_direction,
            show_period=show_period,
        )
        st.plotly_chart(fig, use_container_width=True)

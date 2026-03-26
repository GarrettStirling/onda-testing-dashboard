"""Forecast tab: buoy components + CDIP via nearest-time join.

Builds ``buoy_cdip_nearest_join.csv``: each buoy row gets the nearest CDIP row (same break)
within a time tolerance (default 2.5 h), so 1/4/7 vs 2/5/8 style grids still match.
Plots use buoy ``wave_time_pst`` as x; CDIP values are drawn at that time from the join.
Forecasts render with **Plotly** (hover, vertical spike across panels, unified tooltips).

Sources:
  - data/forecasts/buoy_scaled_components.csv
  - data/forecasts/cdip_data_p.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
BUOY_CSV = REPO_ROOT / "data" / "forecasts" / "buoy_scaled_components.csv"
CDIP_CSV = REPO_ROOT / "data" / "forecasts" / "cdip_data_p.csv"
JOINED_CSV = REPO_ROOT / "data" / "forecasts" / "buoy_cdip_nearest_join.csv"
BREAKS_CSV = REPO_ROOT / "data" / "reference" / "breaks_with_names.csv"

# Max |buoy_time − cdip_time| for a match (nearest). 2.5 h covers ~1 h skew on 3 h grids.
DEFAULT_NEAREST_TOLERANCE_HOURS = 2.5

CDIP_MERGE_COLUMNS = [
    "significant_wave_height",
    "primary_wave_height",
    "primary_period",
    "primary_direction",
    "secondary_wave_height",
    "secondary_period",
    "secondary_direction",
    "tertiary_wave_height",
    "tertiary_period",
    "tertiary_direction",
    "source",
    "ingested_at",
]

PST = pytz.timezone("US/Pacific")
M_TO_FT = 3.28084

# Dark theme (aligned with onda-backend/scripts/plot_cdip_data.py)
BG_DARK = "#0e1117"
BG_PANEL = "#161b22"
GRID_COLOR = "#2a2d35"
TEXT_COLOR = "#c9d1d9"
SPINE_COLOR = "#30363d"

C_PRIMARY = "#38bdf8"
C_SECONDARY = "#fb923c"
C_TERTIARY = "#a78bfa"
C_SIG = "#f1f5f9"

LW_MAIN = 1.8
LW_SEC = 1.4
LW_TER = 1.2
LW_SIG = 2.8
LW_OVERLAY = 1.2
ALPHA_OVERLAY = 0.55


def _format_break_label(spot_name: str, break_name: str, break_id: int) -> str:
    spot = (spot_name or "").strip()
    brk = (break_name or "").strip()
    if spot and brk and spot.lower() != brk.lower():
        return f"{spot} — {brk}"
    if brk:
        return brk
    if spot:
        return spot
    return f"Break {break_id}"


@st.cache_data(show_spinner=False)
def _load_break_labels(path: str) -> dict[int, str]:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(
        p,
        usecols=["break_id", "spot_name", "break_name"],
        dtype={"break_id": "int32"},
    )
    out: dict[int, str] = {}
    for _, row in df.iterrows():
        bid = int(row["break_id"])
        out[bid] = _format_break_label(
            str(row.get("spot_name", "") or ""),
            str(row.get("break_name", "") or ""),
            bid,
        )
    return out


def _heights_to_ft(series: pd.Series) -> pd.Series:
    """Meters → feet; coerce non-numeric to NaN (avoids overlay plot crashes)."""
    return pd.to_numeric(series, errors="coerce").astype(float) * M_TO_FT


def _normalize_buoy_component_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Support current CSV schema (``primary_wave_height``, …) and legacy ``*_buoy_scaled`` names."""
    out = df.copy()
    pairs = [
        ("primary_wave_height_buoy_scaled", "primary_wave_height"),
        ("secondary_wave_height_buoy_scaled", "secondary_wave_height"),
        ("tertiary_wave_height_buoy_scaled", "tertiary_wave_height"),
        ("primary_period_buoy_scaled", "primary_period"),
        ("secondary_period_buoy_scaled", "secondary_period"),
        ("tertiary_period_buoy_scaled", "tertiary_period"),
        ("primary_direction_buoy_scaled", "primary_direction"),
        ("secondary_direction_buoy_scaled", "secondary_direction"),
        ("tertiary_direction_buoy_scaled", "tertiary_direction"),
    ]
    for old, new in pairs:
        if old not in out.columns:
            continue
        if new in out.columns:
            out = out.drop(columns=[old])
        else:
            out = out.rename(columns={old: new})
    return out


@st.cache_data(show_spinner=False)
def _load_buoy_forecast(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wave_time_pst"] = pd.to_datetime(df["wave_time_pst"])
    if df["wave_time_pst"].dt.tz is None:
        # infer handles DST fall-back ambiguous local times better than NaT
        df["wave_time_pst"] = df["wave_time_pst"].dt.tz_localize(
            PST, ambiguous="infer", nonexistent="shift_forward"
        )
    else:
        df["wave_time_pst"] = df["wave_time_pst"].dt.tz_convert(PST)
    df = _normalize_buoy_component_columns(df)
    return df.sort_values(["break_id", "wave_time_pst"])


@st.cache_data(show_spinner=False)
def _load_cdip_forecast(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wave_time_pst"] = pd.to_datetime(df["wave_time_utc"], utc=True).dt.tz_convert(PST)
    return df.sort_values(["break_id", "wave_time_pst"])


def _empty_cdip_join_columns(df_buoy: pd.DataFrame) -> pd.DataFrame:
    out = df_buoy.copy()
    for col in CDIP_MERGE_COLUMNS:
        out[f"cdip_{col}"] = np.nan
    out["cdip_obs_time_pst"] = pd.NaT
    out["cdip_match_delta_seconds"] = np.nan
    return out


def build_buoy_cdip_nearest_join(
    df_buoy: pd.DataFrame,
    df_cdip: pd.DataFrame | None,
    tolerance_hours: float,
) -> pd.DataFrame:
    """Left = every buoy row; attach nearest CDIP row per ``break_id`` (``merge_asof``, ``direction=nearest``)."""
    if df_cdip is None or df_cdip.empty:
        return _empty_cdip_join_columns(df_buoy).sort_values(["break_id", "wave_time_pst"])

    tol = pd.Timedelta(hours=float(tolerance_hours))
    parts: list[pd.DataFrame] = []

    for bid, b_raw in df_buoy.groupby("break_id", sort=True):
        b = b_raw.sort_values("wave_time_pst").copy()
        c = df_cdip[df_cdip["break_id"] == bid].sort_values("wave_time_pst").copy()
        if c.empty:
            parts.append(_empty_cdip_join_columns(b))
            continue

        c2 = c.rename(columns={"wave_time_pst": "cdip_obs_time_pst"})
        rename_map = {col: f"cdip_{col}" for col in CDIP_MERGE_COLUMNS if col in c2.columns}
        c2 = c2.rename(columns=rename_map)
        right_cols = ["cdip_obs_time_pst"] + [rename_map[c] for c in CDIP_MERGE_COLUMNS if c in rename_map]
        right_cols = [x for x in right_cols if x in c2.columns]
        c2 = c2[right_cols].sort_values("cdip_obs_time_pst")

        m = pd.merge_asof(
            b,
            c2,
            left_on="wave_time_pst",
            right_on="cdip_obs_time_pst",
            direction="nearest",
            tolerance=tol,
        )
        m["cdip_match_delta_seconds"] = (m["wave_time_pst"] - m["cdip_obs_time_pst"]).dt.total_seconds()
        parts.append(m)

    return pd.concat(parts, ignore_index=True).sort_values(["break_id", "wave_time_pst"])


def _buoy_cdip_mtime_pair() -> tuple[float, float]:
    b = BUOY_CSV.stat().st_mtime if BUOY_CSV.exists() else 0.0
    c = CDIP_CSV.stat().st_mtime if CDIP_CSV.exists() else 0.0
    return (b, c)


@st.cache_data(show_spinner="Building buoy↔CDIP nearest join…")
def _joined_forecast_dataframe(
    buoy_mtime: float,
    cdip_mtime: float,
    tolerance_hours: float,
) -> pd.DataFrame:
    df_b = _load_buoy_forecast(str(BUOY_CSV))
    df_c = _load_cdip_forecast(str(CDIP_CSV)) if CDIP_CSV.exists() else None
    joined = build_buoy_cdip_nearest_join(df_b, df_c, tolerance_hours)
    JOINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(JOINED_CSV, index=False)
    return joined


def _legend_show_once(seen: set[str], name: str) -> bool:
    """Plotly one legend entry per logical series (same name repeated on 3 subplots)."""
    if name in seen:
        return False
    seen.add(name)
    return True


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _forecast_hover_xaxis_tickformat(span_days: float) -> str:
    return "%b %d %H:%M" if span_days <= 3.5 else "%b %d"


def _y_range_padded(
    *series: pd.Series,
    frac: float = 0.08,
    floor_zero: bool = False,
    min_span: float = 0.0,
) -> tuple[float, float]:
    """Finite min/max across series, expand span by ``frac`` on each side; optional floor at 0 and minimum span."""
    parts: list[pd.Series] = []
    for s in series:
        x = pd.to_numeric(s, errors="coerce").dropna()
        if len(x) > 0:
            parts.append(x)
    if not parts:
        return (0.0, 1.0)
    c = pd.concat(parts, ignore_index=True)
    lo, hi = float(c.min()), float(c.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0)
    if lo > hi:
        lo, hi = hi, lo
    span = hi - lo
    if span < min_span:
        mid = (lo + hi) / 2.0
        lo = mid - min_span / 2.0
        hi = mid + min_span / 2.0
        span = min_span
    pad = max(span * frac, 1e-9)
    lo2, hi2 = lo - pad, hi + pad
    if floor_zero:
        lo2 = max(0.0, lo2)
    return (lo2, hi2)


def _add_ts_line(
    fig: go.Figure,
    row: int,
    x: pd.Series,
    y: pd.Series,
    *,
    name: str,
    color: str,
    dash: str = "solid",
    width: float = 2,
    unit: str,
    opacity: float = 1.0,
    showlegend: bool = True,
) -> None:
    """One time-series trace with hover text ``name`` + value + ``unit``."""
    line_color = _hex_to_rgba(color, opacity) if opacity < 1.0 else color
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            showlegend=showlegend,
            line=dict(color=line_color, width=width, dash=dash),
            hovertemplate=f"<b>{name}</b><br>%{{y:.3f}}{unit}<extra></extra>",
        ),
        row=row,
        col=1,
    )


def _plot_break_forecast(
    joined_g: pd.DataFrame,
    *,
    show_cdip_sig: bool,
    overlay_cdip_mop: bool,
    label: str,
) -> go.Figure:
    """Interactive Plotly figure: shared x, vertical spike across panels, unified hover per row."""
    joined_g = joined_g.copy().sort_values("wave_time_pst")

    if joined_g.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No buoy forecast rows in window",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=TEXT_COLOR, size=14),
        )
        fig.update_layout(
            paper_bgcolor=BG_DARK,
            plot_bgcolor=BG_PANEL,
            height=220,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

    t_b = joined_g["wave_time_pst"]
    span_days = max(
        (joined_g["wave_time_pst"].max() - joined_g["wave_time_pst"].min()).total_seconds() / 86400.0,
        0.25,
    )

    show_any_cdip = show_cdip_sig or overlay_cdip_mop
    has_sig = "cdip_significant_wave_height" in joined_g.columns and joined_g["cdip_significant_wave_height"].notna().any()
    has_mop = "cdip_primary_wave_height" in joined_g.columns and joined_g["cdip_primary_wave_height"].notna().any()

    title = label if label else f"Break {int(joined_g['break_id'].iloc[0])}"
    has_any_match = has_sig or has_mop
    if not show_any_cdip:
        cdip_note = " — buoy only"
    elif not has_any_match:
        cdip_note = " — buoy (no CDIP match within tolerance)"
    elif show_cdip_sig and overlay_cdip_mop and has_sig and has_mop:
        cdip_note = " — buoy + CDIP sig + MOP"
    elif show_cdip_sig and has_sig:
        cdip_note = " — buoy + CDIP sig"
    elif overlay_cdip_mop and has_mop:
        cdip_note = " — buoy + CDIP MOP"
    else:
        cdip_note = " — buoy (CDIP on, no data for selected layers)"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.38, 0.31, 0.31],
        subplot_titles=("Height (ft)", "Direction (° from north)", "Period (s)"),
    )

    leg: set[str] = set()

    # --- Row 1: height ---
    _add_ts_line(
        fig,
        1,
        t_b,
        _heights_to_ft(joined_g["primary_wave_height"]),
        name="Primary (buoy components)",
        color=C_PRIMARY,
        dash="solid",
        width=LW_MAIN,
        unit=" ft",
        opacity=1.0,
        showlegend=_legend_show_once(leg, "Primary (buoy components)"),
    )
    _add_ts_line(
        fig,
        1,
        t_b,
        _heights_to_ft(joined_g["secondary_wave_height"]),
        name="Secondary (buoy components)",
        color=C_SECONDARY,
        dash="dash",
        width=LW_SEC,
        unit=" ft",
        opacity=1.0,
        showlegend=_legend_show_once(leg, "Secondary (buoy components)"),
    )
    _add_ts_line(
        fig,
        1,
        t_b,
        _heights_to_ft(joined_g["tertiary_wave_height"]),
        name="Tertiary (buoy components)",
        color=C_TERTIARY,
        dash="dot",
        width=LW_TER,
        unit=" ft",
        opacity=1.0,
        showlegend=_legend_show_once(leg, "Tertiary (buoy components)"),
    )

    if overlay_cdip_mop and has_mop:
        _add_ts_line(
            fig,
            1,
            t_b,
            _heights_to_ft(joined_g["cdip_primary_wave_height"]),
            name="Primary (CDIP MOP)",
            color=C_PRIMARY,
            dash="solid",
            width=LW_OVERLAY,
            unit=" ft",
            opacity=ALPHA_OVERLAY,
            showlegend=_legend_show_once(leg, "Primary (CDIP MOP)"),
        )
        _add_ts_line(
            fig,
            1,
            t_b,
            _heights_to_ft(joined_g["cdip_secondary_wave_height"]),
            name="Secondary (CDIP MOP)",
            color=C_SECONDARY,
            dash="dash",
            width=LW_OVERLAY,
            unit=" ft",
            opacity=ALPHA_OVERLAY,
            showlegend=_legend_show_once(leg, "Secondary (CDIP MOP)"),
        )
        ter_h = joined_g["cdip_tertiary_wave_height"] if "cdip_tertiary_wave_height" in joined_g.columns else None
        if ter_h is not None and ter_h.notna().any():
            _add_ts_line(
                fig,
                1,
                t_b,
                _heights_to_ft(ter_h),
                name="Tertiary (CDIP MOP)",
                color=C_TERTIARY,
                dash="dot",
                width=LW_OVERLAY,
                unit=" ft",
                opacity=ALPHA_OVERLAY,
                showlegend=_legend_show_once(leg, "Tertiary (CDIP MOP)"),
            )

    if show_cdip_sig and has_sig:
        _add_ts_line(
            fig,
            1,
            t_b,
            _heights_to_ft(joined_g["cdip_significant_wave_height"]),
            name="Sig. height (CDIP)",
            color=C_SIG,
            dash="solid",
            width=LW_SIG,
            unit=" ft",
            opacity=0.95,
            showlegend=_legend_show_once(leg, "Sig. height (CDIP)"),
        )

    # --- Row 2: direction (raw 0–360°; no artificial NaNs — steep segments = wrap past north) ---
    def _add_dir(name: str, series: pd.Series, color: str, dash: str, width: float, opacity: float = 1.0) -> None:
        y = pd.to_numeric(series, errors="coerce")
        line_color = _hex_to_rgba(color, opacity) if opacity < 1.0 else color
        fig.add_trace(
            go.Scatter(
                x=t_b,
                y=y,
                mode="lines",
                name=name,
                showlegend=_legend_show_once(leg, name),
                line=dict(color=line_color, width=width, dash=dash),
                hovertemplate=f"<b>{name}</b><br>%{{y:.1f}}°<extra></extra>",
            ),
            row=2,
            col=1,
        )

    _add_dir("Primary (buoy components)", joined_g["primary_direction"], C_PRIMARY, "solid", LW_MAIN)
    _add_dir("Secondary (buoy components)", joined_g["secondary_direction"], C_SECONDARY, "dash", LW_SEC)
    _add_dir("Tertiary (buoy components)", joined_g["tertiary_direction"], C_TERTIARY, "dot", LW_TER)

    if overlay_cdip_mop and has_mop:
        _add_dir(
            "Primary (CDIP MOP)",
            pd.to_numeric(joined_g["cdip_primary_direction"], errors="coerce"),
            C_PRIMARY,
            "solid",
            LW_OVERLAY,
            ALPHA_OVERLAY,
        )
        _add_dir(
            "Secondary (CDIP MOP)",
            pd.to_numeric(joined_g["cdip_secondary_direction"], errors="coerce"),
            C_SECONDARY,
            "dash",
            LW_OVERLAY,
            ALPHA_OVERLAY,
        )
        if "cdip_tertiary_direction" in joined_g.columns and joined_g["cdip_tertiary_direction"].notna().any():
            _add_dir(
                "Tertiary (CDIP MOP)",
                pd.to_numeric(joined_g["cdip_tertiary_direction"], errors="coerce"),
                C_TERTIARY,
                "dot",
                LW_OVERLAY,
                ALPHA_OVERLAY,
            )

    # --- Row 3: period ---
    _add_ts_line(
        fig,
        3,
        t_b,
        pd.to_numeric(joined_g["primary_period"], errors="coerce"),
        name="Primary (buoy components)",
        color=C_PRIMARY,
        width=LW_MAIN,
        unit=" s",
        showlegend=_legend_show_once(leg, "Primary (buoy components)"),
    )
    _add_ts_line(
        fig,
        3,
        t_b,
        pd.to_numeric(joined_g["secondary_period"], errors="coerce"),
        name="Secondary (buoy components)",
        color=C_SECONDARY,
        dash="dash",
        width=LW_SEC,
        unit=" s",
        showlegend=_legend_show_once(leg, "Secondary (buoy components)"),
    )
    _add_ts_line(
        fig,
        3,
        t_b,
        pd.to_numeric(joined_g["tertiary_period"], errors="coerce"),
        name="Tertiary (buoy components)",
        color=C_TERTIARY,
        dash="dot",
        width=LW_TER,
        unit=" s",
        showlegend=_legend_show_once(leg, "Tertiary (buoy components)"),
    )

    if overlay_cdip_mop and has_mop:
        _add_ts_line(
            fig,
            3,
            t_b,
            pd.to_numeric(joined_g["cdip_primary_period"], errors="coerce"),
            name="Primary (CDIP MOP)",
            color=C_PRIMARY,
            width=LW_OVERLAY,
            unit=" s",
            opacity=ALPHA_OVERLAY,
            showlegend=_legend_show_once(leg, "Primary (CDIP MOP)"),
        )
        _add_ts_line(
            fig,
            3,
            t_b,
            pd.to_numeric(joined_g["cdip_secondary_period"], errors="coerce"),
            name="Secondary (CDIP MOP)",
            color=C_SECONDARY,
            dash="dash",
            width=LW_OVERLAY,
            unit=" s",
            opacity=ALPHA_OVERLAY,
            showlegend=_legend_show_once(leg, "Secondary (CDIP MOP)"),
        )
        if "cdip_tertiary_period" in joined_g.columns and joined_g["cdip_tertiary_period"].notna().any():
            _add_ts_line(
                fig,
                3,
                t_b,
                pd.to_numeric(joined_g["cdip_tertiary_period"], errors="coerce"),
                name="Tertiary (CDIP MOP)",
                color=C_TERTIARY,
                dash="dot",
                width=LW_OVERLAY,
                unit=" s",
                opacity=ALPHA_OVERLAY,
                showlegend=_legend_show_once(leg, "Tertiary (CDIP MOP)"),
            )

    # Y ranges from everything drawn on each row (small padding; not fixed 0–360 / tozero)
    h_for_range: list[pd.Series] = [
        _heights_to_ft(joined_g["primary_wave_height"]),
        _heights_to_ft(joined_g["secondary_wave_height"]),
        _heights_to_ft(joined_g["tertiary_wave_height"]),
    ]
    if overlay_cdip_mop and has_mop:
        h_for_range.append(_heights_to_ft(joined_g["cdip_primary_wave_height"]))
        h_for_range.append(_heights_to_ft(joined_g["cdip_secondary_wave_height"]))
        if "cdip_tertiary_wave_height" in joined_g.columns:
            h_for_range.append(_heights_to_ft(joined_g["cdip_tertiary_wave_height"]))
    if show_cdip_sig and has_sig:
        h_for_range.append(_heights_to_ft(joined_g["cdip_significant_wave_height"]))
    y_r_h = _y_range_padded(*h_for_range, frac=0.08, floor_zero=True, min_span=0.35)

    d_for_range: list[pd.Series] = [
        joined_g["primary_direction"],
        joined_g["secondary_direction"],
        joined_g["tertiary_direction"],
    ]
    if overlay_cdip_mop and has_mop:
        d_for_range.append(joined_g["cdip_primary_direction"])
        d_for_range.append(joined_g["cdip_secondary_direction"])
        if "cdip_tertiary_direction" in joined_g.columns:
            d_for_range.append(joined_g["cdip_tertiary_direction"])
    y_r_d = _y_range_padded(*d_for_range, frac=0.08, floor_zero=False, min_span=12.0)

    p_for_range: list[pd.Series] = [
        joined_g["primary_period"],
        joined_g["secondary_period"],
        joined_g["tertiary_period"],
    ]
    if overlay_cdip_mop and has_mop:
        p_for_range.append(joined_g["cdip_primary_period"])
        p_for_range.append(joined_g["cdip_secondary_period"])
        if "cdip_tertiary_period" in joined_g.columns:
            p_for_range.append(joined_g["cdip_tertiary_period"])
    y_r_p = _y_range_padded(*p_for_range, frac=0.08, floor_zero=False, min_span=0.75)

    tick_fmt = _forecast_hover_xaxis_tickformat(span_days)
    fig.update_layout(
        title=dict(
            text=f"Forecast — {title}{cdip_note}",
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
        height=820,
    )
    fig.update_xaxes(
        showspikes=True,
        spikecolor="#8b949e",
        spikesnap="cursor",
        spikemode="across",
        spikethickness=1,
        gridcolor=GRID_COLOR,
        showgrid=True,
        zeroline=False,
        tickformat=tick_fmt,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(title_text="Date (US/Pacific)", row=3, col=1)

    fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True, zeroline=False)
    fig.update_yaxes(range=list(y_r_h), row=1, col=1)
    fig.update_yaxes(range=list(y_r_d), row=2, col=1)
    fig.update_yaxes(range=list(y_r_p), row=3, col=1)

    fig.update_annotations(font=dict(color=TEXT_COLOR, size=12))

    return fig


def render_forecast_tab() -> None:
    st.header("Forecasts")
    st.caption(
        "Buoy rows drive the time axis. CDIP values are attached with **nearest** timestamps per break "
        f"(≤ **match window** hours apart) and written to `{JOINED_CSV.name}`. "
        "Staggered grids (e.g. buoy 1/4/7 vs CDIP 2/5/8) still match; if buoy and CDIP date ranges "
        "never overlap, CDIP columns stay empty. "
        "**Tertiary** period/direction gaps usually mean no third swell in the CSV (NaN there). "
        "Direction may show a **steep diagonal** when bearing wraps past north (0°/360°) — that is not missing data."
    )

    if not BUOY_CSV.exists():
        st.error(f"Missing buoy forecast CSV: `{BUOY_CSV}`")
        return
    if not CDIP_CSV.exists():
        st.warning(f"CDIP file not found: `{CDIP_CSV}` — joined table will have empty `cdip_*` columns.")

    labels = _load_break_labels(str(BREAKS_CSV))

    def _resolve_default_break_ids() -> list[int]:
        # Resolve user-friendly spot names to `break_id`s as defined in
        # `data/reference/breaks_with_names.csv`.
        breaks_df = pd.read_csv(BREAKS_CSV)
        spot_col = "spot_name" if "spot_name" in breaks_df.columns else "spot"
        break_col = "break_name" if "break_name" in breaks_df.columns else "break"

        # Each entry is (spot_name_substring, break_name_substring_or_None).
        # (Matches are case-insensitive via `str.contains`.)
        targets: list[tuple[str, str | None]] = [
            ("Ocean Beach", "North"),
            ("Ocean Beach", "Central"),
            ("Waddell Creek", "Reef"),
            ("Davenport", "Landing"),
            ("Four Mile", None),
            ("Swift Street", None),
            ("Steamer Lane", "Point"),
            ("Steamer Lane", "Middle Peak"),
            ("Cowells", None),
            ("Pleasure Point", "Sewers"),
            ("Pleasure Point", "First Peak"),
            ("Capitola", None),
        ]

        ids: list[int] = []
        for spot_sub, break_sub in targets:
            m = breaks_df[spot_col].astype(str).str.contains(spot_sub, case=False, na=False)
            if break_sub:
                m = m & breaks_df[break_col].astype(str).str.contains(break_sub, case=False, na=False)
            found = sorted(breaks_df.loc[m, "break_id"].dropna().astype(int).unique().tolist())
            ids.extend(found)

        # Deduplicate, keep sorted stable.
        return sorted(set(ids))

    tol_h = st.number_input(
        "Nearest CDIP match window (hours)",
        min_value=1.0,
        max_value=6.0,
        value=float(DEFAULT_NEAREST_TOLERANCE_HOURS),
        step=0.5,
        help="Each buoy timestep gets the closest CDIP row within this many hours (same break_id). "
        "Increase slightly if your pipelines use a larger clock/grid offset.",
    )

    bm, cm = _buoy_cdip_mtime_pair()
    df_joined = _joined_forecast_dataframe(bm, cm, float(tol_h))

    break_ids = sorted(df_joined["break_id"].unique().astype(int).tolist())
    if not break_ids:
        st.warning("No rows in buoy forecast file.")
        return

    with st.expander("Data sources", expanded=False):
        st.write(f"Buoy scaled components: `{BUOY_CSV}`")
        st.write(f"CDIP MOP processed: `{CDIP_CSV}`")
        st.write(f"Nearest join (saved on load): `{JOINED_CSV}`")
        st.write(f"Break labels: `{BREAKS_CSV}`")

    show_cdip_sig = st.checkbox(
        "Show CDIP significant wave height (`cdip_data_p.csv`)",
        value=True,
        help="Bold line on the height panel (values from nearest-join columns).",
    )
    overlay_cdip_mop = st.checkbox(
        "Overlay CDIP MOP (pri/sec/ter on all three panels)",
        value=False,
        help="Semi-transparent CDIP component lines. Off by default.",
    )

    options = break_ids
    labels_opt = [labels.get(bid, f"Break {bid}") for bid in options]
    id_to_label = dict(zip(options, labels_opt))
    selected = st.multiselect(
        "Surf spots (breaks)",
        options=options,
        format_func=lambda bid: id_to_label.get(bid, f"Break {bid}"),
        default=[bid for bid in _resolve_default_break_ids() if bid in set(options)],
    )

    if not selected:
        st.info("Select at least one break.")
        return

    for bid in selected:
        st.subheader(id_to_label.get(bid, f"Break {bid}"))
        jsub = df_joined[df_joined["break_id"] == bid]

        try:
            fig = _plot_break_forecast(
                jsub,
                show_cdip_sig=show_cdip_sig,
                overlay_cdip_mop=overlay_cdip_mop,
                label=id_to_label.get(bid, f"Break {bid}"),
            )
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:
            st.error(f"Plot failed for break {bid}: {exc}")

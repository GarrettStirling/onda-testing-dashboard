"""Forecast tab: buoy components + CDIP via nearest-time join.

Builds ``buoy_cdip_nearest_join.csv``: each buoy row gets the nearest CDIP row (same break)
within a time tolerance (default 2.5 h), so 1/4/7 vs 2/5/8 style grids still match.
Plots use buoy ``wave_time_pst`` as x; CDIP values are drawn at that time from the join.
Forecasts render with **Plotly** (hover, vertical spike across panels, unified tooltips).

Sources:
  - data/forecasts/buoy_scaled_components.csv
  - data/forecasts/cdip_data_p.csv
  - optional: data/forecasts/offshore_buoy_data_p.csv

Or BigQuery tables in ``onda-maverick.surf_forecast_data`` (today onward, US/Pacific).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from plotly.subplots import make_subplots

from streamlit_dashboard.field_observations import (
    DEFAULT_MIN_SELECTED_OBS,
    FIELD_OBS_CSV,
    default_break_ids_with_min_obs,
    load_field_observations,
    obs_counts_by_break_id,
    sort_break_ids_by_obs_count,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUOY_CSV = REPO_ROOT / "data" / "forecasts" / "buoy_scaled_components.csv"
CDIP_CSV = REPO_ROOT / "data" / "forecasts" / "cdip_data_p.csv"
OFFSHORE_CSV = REPO_ROOT / "data" / "forecasts" / "offshore_buoy_data_p.csv"
JOINED_CSV = REPO_ROOT / "data" / "forecasts" / "buoy_cdip_nearest_join.csv"
BREAKS_CSV = REPO_ROOT / "data" / "reference" / "breaks_with_names.csv"

# Max |buoy_time − cdip_time| for a match (nearest). 2.5 h covers ~1 h skew on 3 h grids.
DEFAULT_NEAREST_TOLERANCE_HOURS = 2.5

CDIP_MERGE_COLUMNS = [
    "significant_wave_height",
    "significant_wave_height_raw",
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
XAXIS_DAY_GRID_COLOR = "#4f5866"  # lighter gray for daily vertical gridlines
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
LW_SIG_RAW = 1.15
ALPHA_SIG_RAW = 0.5
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


def _coerce_datetime_to_ns(s: pd.Series) -> pd.Series:
    """Normalize resolution to nanoseconds. BigQuery/pyarrow often yields ``datetime64[us]``; CSV is
    typically ``datetime64[ns]`` — ``merge_asof`` requires matching dtypes.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(s, errors="coerce"))
    if hasattr(idx, "as_unit"):
        idx = idx.as_unit("ns")
    return pd.Series(idx, index=s.index)


def _prepare_buoy_forecast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize buoy forecast columns (CSV or BigQuery)."""
    if df.empty:
        return df
    df = df.copy()
    if "wave_time_pst" in df.columns:
        df["wave_time_pst"] = pd.to_datetime(df["wave_time_pst"])
        if df["wave_time_pst"].dt.tz is None:
            df["wave_time_pst"] = df["wave_time_pst"].dt.tz_localize(
                PST, ambiguous="infer", nonexistent="shift_forward"
            )
        else:
            df["wave_time_pst"] = df["wave_time_pst"].dt.tz_convert(PST)
    elif "wave_time_utc" in df.columns:
        df["wave_time_pst"] = pd.to_datetime(df["wave_time_utc"], utc=True).dt.tz_convert(PST)
    else:
        raise ValueError("Buoy forecast data needs `wave_time_pst` or `wave_time_utc`.")

    df["wave_time_pst"] = _coerce_datetime_to_ns(df["wave_time_pst"])

    df = _normalize_buoy_component_columns(df)
    if "break_id" in df.columns:
        df["break_id"] = pd.to_numeric(df["break_id"], errors="coerce")
        df = df.loc[df["break_id"].notna()].copy()
        df["break_id"] = df["break_id"].astype(int)
    return df.sort_values(["break_id", "wave_time_pst"])


def _prepare_cdip_forecast_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "wave_time_pst" not in df.columns:
        if "wave_time_utc" not in df.columns:
            raise ValueError("CDIP data needs `wave_time_utc` or `wave_time_pst`.")
        df["wave_time_pst"] = pd.to_datetime(df["wave_time_utc"], utc=True).dt.tz_convert(PST)
    else:
        df["wave_time_pst"] = pd.to_datetime(df["wave_time_pst"])
        if df["wave_time_pst"].dt.tz is None:
            df["wave_time_pst"] = df["wave_time_pst"].dt.tz_localize(
                PST, ambiguous="infer", nonexistent="shift_forward"
            )
        else:
            df["wave_time_pst"] = df["wave_time_pst"].dt.tz_convert(PST)

    df["wave_time_pst"] = _coerce_datetime_to_ns(df["wave_time_pst"])

    # Merge prefixes `significant_wave_height_raw` → `cdip_significant_wave_height_raw`; align BQ name if needed.
    if "significant_wave_height_raw" not in df.columns and "cdip_significant_wave_height_raw" in df.columns:
        df = df.rename(columns={"cdip_significant_wave_height_raw": "significant_wave_height_raw"})

    if "break_id" in df.columns:
        df["break_id"] = pd.to_numeric(df["break_id"], errors="coerce")
        df = df.loc[df["break_id"].notna()].copy()
        df["break_id"] = df["break_id"].astype(int)
    return df.sort_values(["break_id", "wave_time_pst"])


def _prepare_offshore_buoy_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "wave_time_pst" not in df.columns:
        if "wave_time_utc" not in df.columns:
            raise ValueError("Offshore buoy data needs `wave_time_utc` or `wave_time_pst`.")
        df["wave_time_pst"] = pd.to_datetime(df["wave_time_utc"], utc=True).dt.tz_convert(PST)
    else:
        df["wave_time_pst"] = pd.to_datetime(df["wave_time_pst"])
        if df["wave_time_pst"].dt.tz is None:
            df["wave_time_pst"] = df["wave_time_pst"].dt.tz_localize(
                PST, ambiguous="infer", nonexistent="shift_forward"
            )
        else:
            df["wave_time_pst"] = df["wave_time_pst"].dt.tz_convert(PST)

    df["wave_time_pst"] = _coerce_datetime_to_ns(df["wave_time_pst"])

    if "buoy_id" in df.columns:
        df["buoy_id"] = pd.to_numeric(df["buoy_id"], errors="coerce")
        df = df.loc[df["buoy_id"].notna()].copy()
        df["buoy_id"] = df["buoy_id"].astype(int)
    return df.sort_values(["buoy_id", "wave_time_pst"])


@st.cache_data(show_spinner=False)
def _load_buoy_forecast(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _prepare_buoy_forecast_df(df)


@st.cache_data(show_spinner=False)
def _load_cdip_forecast(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _prepare_cdip_forecast_df(df)


@st.cache_data(show_spinner=False)
def _load_offshore_forecast_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _prepare_offshore_buoy_df(df)


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
    if df_buoy.empty:
        return df_buoy.copy()
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


def _offshore_csv_mtime() -> float:
    return OFFSHORE_CSV.stat().st_mtime if OFFSHORE_CSV.exists() else 0.0


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


@st.cache_data(ttl=120, show_spinner="Building buoy↔CDIP nearest join…")
def _bq_forecast_bundle(tolerance_hours: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BigQuery: joined break forecasts + offshore buoy rows (same ingest; cached together)."""
    from streamlit_dashboard.bq_forecast_loader import load_forecast_tables_bigquery

    raw_b, raw_c, raw_o = load_forecast_tables_bigquery()
    df_b = _prepare_buoy_forecast_df(raw_b)
    df_c: pd.DataFrame | None
    if raw_c.empty:
        df_c = None
    else:
        df_c = _prepare_cdip_forecast_df(raw_c)
        if df_c.empty:
            df_c = None
    joined = build_buoy_cdip_nearest_join(df_b, df_c, tolerance_hours)
    offshore = _prepare_offshore_buoy_df(raw_o) if not raw_o.empty else pd.DataFrame()
    return joined, offshore


@st.cache_data(show_spinner=False)
def _local_offshore_dataframe(offshore_mtime: float) -> pd.DataFrame:
    if offshore_mtime <= 0:
        return pd.DataFrame()
    return _load_offshore_forecast_csv(str(OFFSHORE_CSV))


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


def _pacific_daily_midnight_ticks(t_min: pd.Timestamp, t_max: pd.Timestamp) -> list[pd.Timestamp]:
    """US/Pacific local midnights for each calendar day overlapping ``[t_min, t_max]``."""
    t_min = pd.Timestamp(t_min)
    t_max = pd.Timestamp(t_max)
    if pd.isna(t_min) or pd.isna(t_max):
        return []
    if t_min.tzinfo is None:
        t_min = t_min.tz_localize(PST, ambiguous="infer", nonexistent="shift_forward")
    else:
        t_min = t_min.tz_convert(PST)
    if t_max.tzinfo is None:
        t_max = t_max.tz_localize(PST, ambiguous="infer", nonexistent="shift_forward")
    else:
        t_max = t_max.tz_convert(PST)
    if t_min > t_max:
        t_min, t_max = t_max, t_min
    dr = pd.date_range(start=t_min.normalize(), end=t_max.normalize(), freq="D", tz=PST)
    return list(dr)


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
    has_sig_raw = (
        "cdip_significant_wave_height_raw" in joined_g.columns
        and joined_g["cdip_significant_wave_height_raw"].notna().any()
    )
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

    if show_cdip_sig and has_sig_raw:
        _add_ts_line(
            fig,
            1,
            t_b,
            _heights_to_ft(joined_g["cdip_significant_wave_height_raw"]),
            name="Sig. height raw (CDIP)",
            color=C_SIG,
            dash="solid",
            width=LW_SIG_RAW,
            unit=" ft",
            opacity=ALPHA_SIG_RAW,
            showlegend=_legend_show_once(leg, "Sig. height raw (CDIP)"),
        )
    if show_cdip_sig and has_sig:
        _add_ts_line(
            fig,
            1,
            t_b,
            _heights_to_ft(joined_g["cdip_significant_wave_height"]),
            name="Sig. height (Calibrated CDIP)",
            color=C_SIG,
            dash="solid",
            width=LW_SIG,
            unit=" ft",
            opacity=0.95,
            showlegend=_legend_show_once(leg, "Sig. height (Calibrated CDIP)"),
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
    if show_cdip_sig and has_sig_raw:
        h_for_range.append(_heights_to_ft(joined_g["cdip_significant_wave_height_raw"]))
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
    x_tickvals = _pacific_daily_midnight_ticks(
        joined_g["wave_time_pst"].min(),
        joined_g["wave_time_pst"].max(),
    )
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
        tickmode="array",
        tickvals=x_tickvals,
        gridcolor=XAXIS_DAY_GRID_COLOR,
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


def _plot_offshore_buoy_forecast(df_sub: pd.DataFrame, *, title: str) -> go.Figure:
    """Offshore buoy time series (``buoy_id`` keyed; heights in meters → ft)."""
    df_sub = df_sub.sort_values("wave_time_pst")
    t_x = df_sub["wave_time_pst"]
    span_days = max(
        (df_sub["wave_time_pst"].max() - df_sub["wave_time_pst"].min()).total_seconds() / 86400.0,
        0.25,
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.38, 0.31, 0.31],
        subplot_titles=("Height (ft)", "Direction (° from north)", "Period (s)"),
    )

    leg: set[str] = set()

    def _ln(row: int, series: pd.Series | None, name: str, color: str, dash: str, lw: float, *, unit: str) -> None:
        if series is None:
            return
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() == 0:
            return
        show = _legend_show_once(leg, name)
        line_color = color
        fig.add_trace(
            go.Scatter(
                x=t_x,
                y=s if unit != " ft" else _heights_to_ft(s),
                mode="lines",
                name=name,
                showlegend=show,
                line=dict(color=line_color, width=lw, dash=dash),
                hovertemplate=f"<b>{name}</b><br>%{{y:.3f}}{unit}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    sig = df_sub["significant_wave_height"] if "significant_wave_height" in df_sub.columns else None
    _ln(1, sig, "Sig. height (offshore buoy)", C_SIG, "solid", LW_SIG, unit=" ft")

    _ln(
        1,
        df_sub["primary_wave_height"] if "primary_wave_height" in df_sub.columns else None,
        "Primary (offshore buoy)",
        C_PRIMARY,
        "solid",
        LW_MAIN,
        unit=" ft",
    )
    _ln(
        1,
        df_sub["secondary_wave_height"] if "secondary_wave_height" in df_sub.columns else None,
        "Secondary (offshore buoy)",
        C_SECONDARY,
        "dash",
        LW_SEC,
        unit=" ft",
    )
    _ln(
        1,
        df_sub["tertiary_wave_height"] if "tertiary_wave_height" in df_sub.columns else None,
        "Tertiary (offshore buoy)",
        C_TERTIARY,
        "dot",
        LW_TER,
        unit=" ft",
    )

    _ln(
        2,
        df_sub["primary_direction"] if "primary_direction" in df_sub.columns else None,
        "Primary dir (offshore buoy)",
        C_PRIMARY,
        "solid",
        LW_MAIN,
        unit="°",
    )
    _ln(
        2,
        df_sub["secondary_direction"] if "secondary_direction" in df_sub.columns else None,
        "Secondary dir (offshore buoy)",
        C_SECONDARY,
        "dash",
        LW_SEC,
        unit="°",
    )
    _ln(
        2,
        df_sub["tertiary_direction"] if "tertiary_direction" in df_sub.columns else None,
        "Tertiary dir (offshore buoy)",
        C_TERTIARY,
        "dot",
        LW_TER,
        unit="°",
    )

    _ln(
        3,
        df_sub["primary_period"] if "primary_period" in df_sub.columns else None,
        "Primary period (offshore buoy)",
        C_PRIMARY,
        "solid",
        LW_MAIN,
        unit=" s",
    )
    _ln(
        3,
        df_sub["secondary_period"] if "secondary_period" in df_sub.columns else None,
        "Secondary period (offshore buoy)",
        C_SECONDARY,
        "dash",
        LW_SEC,
        unit=" s",
    )
    _ln(
        3,
        df_sub["tertiary_period"] if "tertiary_period" in df_sub.columns else None,
        "Tertiary period (offshore buoy)",
        C_TERTIARY,
        "dot",
        LW_TER,
        unit=" s",
    )

    tick_fmt = _forecast_hover_xaxis_tickformat(span_days)
    x_tickvals = _pacific_daily_midnight_ticks(df_sub["wave_time_pst"].min(), df_sub["wave_time_pst"].max())
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=15), x=0.5, xanchor="center"),
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
        tickmode="array",
        tickvals=x_tickvals,
        gridcolor=XAXIS_DAY_GRID_COLOR,
        showgrid=True,
        zeroline=False,
        tickformat=tick_fmt,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(title_text="Date (US/Pacific)", row=3, col=1)
    fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True, zeroline=False)
    fig.update_annotations(font=dict(color=TEXT_COLOR, size=12))
    return fig


def render_forecast_tab() -> None:
    st.header("Forecasts")

    use_bigquery = st.toggle(
        "Load forecasts from BigQuery (vs local CSV under `data/forecasts/`)",
        value=False,
        help=(
            "BigQuery uses `onda-maverick.surf_forecast_data` "
            "(buoy_scaled_components_p, cdip_data_p, offshore_buoy_data_p). "
            "Only rows from the current calendar day (US/Pacific) onward are loaded. "
            "Requires Application Default Credentials or `GOOGLE_APPLICATION_CREDENTIALS`."
        ),
    )

    labels = _load_break_labels(str(BREAKS_CSV))

    obs_mtime = FIELD_OBS_CSV.stat().st_mtime if FIELD_OBS_CSV.exists() else 0.0
    field_obs, unmatched_obs_spots = load_field_observations(
        str(FIELD_OBS_CSV),
        obs_mtime,
        str(BREAKS_CSV),
    )
    obs_counts = obs_counts_by_break_id(field_obs)

    tol_h = float(DEFAULT_NEAREST_TOLERANCE_HOURS)

    if use_bigquery:
        try:
            df_joined, df_offshore = _bq_forecast_bundle(tol_h)
        except Exception as exc:
            st.error(f"BigQuery load failed: {exc}")
            st.caption(
                "Try `gcloud auth application-default login` or set "
                "`GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON path."
            )
            return
    else:
        if not BUOY_CSV.exists():
            st.error(f"Missing buoy forecast CSV: `{BUOY_CSV}`")
            return
        if not CDIP_CSV.exists():
            st.warning(
                f"CDIP file not found: `{CDIP_CSV}` — joined table will have empty `cdip_*` columns."
            )
        bm, cm = _buoy_cdip_mtime_pair()
        df_joined = _joined_forecast_dataframe(bm, cm, tol_h)
        df_offshore = _local_offshore_dataframe(_offshore_csv_mtime())

    break_ids = sorted(df_joined["break_id"].unique().astype(int).tolist())
    if not break_ids:
        st.warning("No rows in buoy forecast file.")
        return

    sorted_break_ids, obs_count_by_bid = sort_break_ids_by_obs_count(break_ids, obs_counts)
    default_selected = default_break_ids_with_min_obs(
        sorted_break_ids,
        obs_count_by_bid,
        min_obs=DEFAULT_MIN_SELECTED_OBS,
    )

    with st.expander("Data sources", expanded=False):
        if use_bigquery:
            st.write(
                "**BigQuery** (`onda-maverick.surf_forecast_data`): rows from the current "
                "calendar day onward (**US/Pacific**)."
            )
            st.write("- `buoy_scaled_components_p` — break-scaled buoy components")
            st.write("- `cdip_data_p` — CDIP / WW3 mop + significant height")
            st.write("- `offshore_buoy_data_p` — offshore buoy hourly series (`buoy_id`)")
        else:
            st.write(f"Buoy scaled components: `{BUOY_CSV}`")
            st.write(f"CDIP MOP processed: `{CDIP_CSV}`")
            st.write(f"Offshore buoy (optional): `{OFFSHORE_CSV}`")
            st.write(f"Nearest join (saved on load): `{JOINED_CSV}`")
        st.write(f"Break labels: `{BREAKS_CSV}`")
        st.write(f"Field observations (ranking / defaults): `{FIELD_OBS_CSV}`")
        if not field_obs.empty:
            st.write(f"- {len(field_obs):,} rows matched to breaks")
        if unmatched_obs_spots:
            st.warning(
                "Could not match these observation spot names to a break:\n"
                + "\n".join(f"- `{s}`" for s in unmatched_obs_spots)
            )

    show_cdip_sig = st.checkbox(
        "Show CDIP significant wave height",
        value=True,
        help=(
            "Height panel: bold CDIP sig height + thinner/semi-transparent raw sig height when "
            "`significant_wave_height_raw` / `cdip_significant_wave_height_raw` is present in CDIP data."
        ),
    )
    overlay_cdip_mop = st.checkbox(
        "Overlay CDIP MOP (pri/sec/ter on all three panels)",
        value=False,
        help="Semi-transparent CDIP component lines. Off by default.",
    )

    id_to_label = {bid: labels.get(bid, f"Break {bid}") for bid in sorted_break_ids}
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
            f"Break order and default selection come from `{FIELD_OBS_CSV.name}`. "
            f"Spots with at least {DEFAULT_MIN_SELECTED_OBS} matched observations are selected on load."
        ),
    )

    if not selected:
        st.info("Select at least one break.")
        return

    selected_set = set(selected)
    for bid in sorted_break_ids:
        if bid not in selected_set:
            continue
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

    if not df_offshore.empty and "buoy_id" in df_offshore.columns:
        st.divider()
        st.subheader("Offshore buoys")
        if use_bigquery:
            st.caption("Source: BigQuery `offshore_buoy_data_p` (same date window as above).")
        else:
            st.caption(f"Source: `{OFFSHORE_CSV}`")
        buoy_ids_off = sorted(df_offshore["buoy_id"].dropna().unique().astype(int).tolist())
        if buoy_ids_off:
            default_n = min(3, len(buoy_ids_off))
            pick_off = st.multiselect(
                "Offshore buoy ID",
                options=buoy_ids_off,
                default=buoy_ids_off[:default_n],
            )
            for oid in pick_off:
                osub = df_offshore[df_offshore["buoy_id"] == oid]
                try:
                    ofig = _plot_offshore_buoy_forecast(
                        osub,
                        title=f"Offshore buoy {oid}",
                    )
                    st.plotly_chart(ofig, width="stretch")
                except Exception as exc:
                    st.error(f"Offshore plot failed for buoy {oid}: {exc}")

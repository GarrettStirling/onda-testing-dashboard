"""20260610 South Swell validation: wave-height time series per break.

Compares uncalibrated vs buoy/CDIP scalar (complex + minimal) forecasts from
``data/20261006 Wave Height Validation/``. Breaks are ranked most → least by
observation count from the local calibration CSV.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_dashboard.forecast_tab import (
    BG_DARK,
    BG_PANEL,
    BREAKS_CSV,
    GRID_COLOR,
    LW_MAIN,
    LW_SEC,
    LW_SIG,
    PST,
    SPINE_COLOR,
    TEXT_COLOR,
    XAXIS_DAY_GRID_COLOR,
    C_PRIMARY,
    C_SECONDARY,
    C_SIG,
    _forecast_hover_xaxis_tickformat,
    _load_break_labels,
    _pacific_daily_midnight_ticks,
    _y_range_padded,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "20261006 Wave Height Validation"
BUOY_DIR = DATA_ROOT / "buoy_based_scalar"
CDIP_DIR = DATA_ROOT / "cdip_based_scalar"
FIELD_OBS_CSV = REPO_ROOT / "data" / "observations" / "20260610_Onda_Calibration_Wave_Height.csv"
FIELD_OBS_DATE = pd.Timestamp("2026-06-10", tz=PST).date()

C_OBS = "#34d399"  # field observations (distinct from forecast lines)

# Bump when series keys or source CSV paths change (invalidates Streamlit cache).
BUNDLE_CACHE_VERSION = 4

# Manual fixes when observation ``Spot`` text does not match reference labels.
SPOT_ALIASES: dict[str, int] = {
    "4 mile": 15,
    "four mile": 15,
    "mitchell s": 19,
    "mitchells": 19,
    "waddell reef": 12,
    "waddell beach": 11,
}

# (series key, legend label, color, dash, line width)
SERIES: list[tuple[str, str, str, str, float]] = [
    ("uncalibrated_buoy", "Uncalibrated (buoy)", C_SIG, "dash", LW_SIG),
    ("uncalibrated_cdip", "Uncalibrated (CDIP)", C_SIG, "solid", LW_SIG),
    ("complex_buoy_scalar", "Complex (buoy scalar)", C_PRIMARY, "solid", LW_MAIN),
    ("minimal_buoy_scalar", "Minimal (buoy scalar)", C_PRIMARY, "dash", LW_SEC),
    ("complex_cdip_scalar", "Complex (CDIP scalar)", C_SECONDARY, "solid", LW_MAIN),
    ("minimal_cdip_scalar", "Minimal (CDIP scalar)", C_SECONDARY, "dash", LW_SEC),
]


def _parse_wave_time_pst(s: pd.Series) -> pd.Series:
    """Parse ``wave_time_pst`` strings (may include ``PDT``/``PST``) to tz-aware US/Pacific."""
    raw = s.astype(str).str.replace(r"\s+(PDT|PST)\s*$", "", regex=True)
    ts = pd.to_datetime(raw, errors="coerce")
    if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
        return ts.dt.tz_convert(PST)
    return ts.dt.tz_localize(PST, ambiguous="infer", nonexistent="shift_forward")


def _prepare_validation_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["wave_time_pst"] = _parse_wave_time_pst(out["wave_time_pst"])
    out["break_id"] = pd.to_numeric(out["break_id"], errors="coerce")
    out = out.loc[out["break_id"].notna()].copy()
    out["break_id"] = out["break_id"].astype(int)
    out["significant_wave_height"] = pd.to_numeric(
        out["significant_wave_height"], errors="coerce"
    )
    return out.sort_values(["break_id", "wave_time_pst"])


def _validation_csv_paths() -> dict[str, Path]:
    return {
        "uncalibrated_buoy": BUOY_DIR / "20260610_forecast_uncalibrated_buoy.csv",
        "uncalibrated_cdip": CDIP_DIR / "20260610_forecast_uncalibrated_cdip.csv",
        "complex_buoy_scalar": BUOY_DIR / "20260610_forecast_complex_model.csv",
        "minimal_buoy_scalar": BUOY_DIR / "20260610_forecast_minimal_model.csv",
        "complex_cdip_scalar": CDIP_DIR / "20260610_forecast_complex_model.csv",
        "minimal_cdip_scalar": CDIP_DIR / "20260610_forecast_minimal_model.csv",
    }


def _validation_mtime_key() -> tuple[int | float, ...]:
    mtimes = tuple(p.stat().st_mtime if p.exists() else 0.0 for p in _validation_csv_paths().values())
    return (BUNDLE_CACHE_VERSION, *mtimes)


@st.cache_data(show_spinner="Loading South Swell validation CSVs…")
def _load_validation_bundle(_mtime_key: tuple[int | float, ...]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, path in _validation_csv_paths().items():
        if not path.exists():
            raise FileNotFoundError(f"Missing validation CSV: `{path}`")
        out[key] = _prepare_validation_df(pd.read_csv(path))
    return out


def _norm_spot_text(s: str) -> str:
    t = (s or "").lower().strip()
    t = t.replace("'", "").replace("’", "")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@st.cache_data(show_spinner=False)
def _build_spot_to_break_id(path: str) -> dict[str, int]:
    """Map normalized observation ``Spot`` strings → ``break_id``."""
    df = pd.read_csv(path, usecols=["break_id", "spot_name", "break_name"])
    lookup: dict[str, int] = {}

    def _add(key: str, bid: int) -> None:
        k = _norm_spot_text(key)
        if k:
            lookup.setdefault(k, int(bid))

    for _, row in df.iterrows():
        bid = int(row["break_id"])
        spot = str(row.get("spot_name") or "").strip()
        brk = str(row.get("break_name") or "").strip()
        _add(spot, bid)
        _add(brk, bid)
        if spot and brk:
            _add(f"{spot} {brk}", bid)
            _add(f"{spot} — {brk}", bid)
            if spot.lower() != brk.lower():
                _add(f"{spot}{brk}", bid)

    for alias, bid in SPOT_ALIASES.items():
        lookup[_norm_spot_text(alias)] = bid
    return lookup


def _parse_field_wave_height_ft(raw: object) -> float | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    if "-" in s:
        parts = [p.strip() for p in s.split("-", 1)]
        try:
            lo, hi = float(parts[0]), float(parts[1])
            return (lo + hi) / 2.0
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_field_obs_datetime(date_s: object, time_s: object) -> pd.Timestamp | pd.NaT:
    d = pd.to_datetime(date_s, errors="coerce")
    if pd.isna(d):
        return pd.NaT
    t = pd.to_datetime(str(time_s).strip(), errors="coerce")
    if pd.isna(t):
        return pd.NaT
    combined = pd.Timestamp(
        year=d.year,
        month=d.month,
        day=d.day,
        hour=t.hour,
        minute=t.minute,
        second=getattr(t, "second", 0),
    )
    return combined.tz_localize(PST, ambiguous=True, nonexistent="shift_forward")


@st.cache_data(show_spinner=False)
def _load_calibration_observations(
    csv_path: str,
    csv_mtime: float,
    breaks_path: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    All calibration CSV rows with a resolved ``break_id``.

    Returns (dataframe, unmatched_spot_names from Jun 10 rows only).
    """
    p = Path(csv_path)
    if not p.exists() or csv_mtime <= 0:
        return pd.DataFrame(), []

    raw = pd.read_csv(p)
    if raw.empty or "Spot" not in raw.columns:
        return pd.DataFrame(), []

    spot_lookup = _build_spot_to_break_id(breaks_path)
    rows: list[dict] = []
    unmatched_june10: set[str] = set()

    for _, r in raw.iterrows():
        obs_dt = _parse_field_obs_datetime(r.get("Date"), r.get("Time"))
        if pd.isna(obs_dt):
            continue
        hs = _parse_field_wave_height_ft(r.get("Wave_Height_ft"))
        if hs is None:
            continue
        spot_raw = str(r.get("Spot") or "").strip()
        bid = spot_lookup.get(_norm_spot_text(spot_raw))
        if bid is None:
            if obs_dt.date() == FIELD_OBS_DATE:
                unmatched_june10.add(spot_raw)
            continue
        rows.append(
            {
                "break_id": bid,
                "spot_raw": spot_raw,
                "obs_time_pst": obs_dt,
                "wave_height_ft": hs,
                "observation_type": str(r.get("Observation_Type") or "").strip(),
            }
        )

    if not rows:
        return pd.DataFrame(), sorted(unmatched_june10)

    out = pd.DataFrame(rows)
    return out.sort_values(["break_id", "obs_time_pst"]), sorted(unmatched_june10)


def _obs_counts_from_calibration(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby("break_id").size().astype(int)


def _snap_obs_to_forecast_times(
    field_obs: pd.DataFrame,
    forecast_times: pd.Series,
) -> pd.DataFrame:
    """Attach nearest forecast ``wave_time_pst`` per observation (``merge_asof``)."""
    if field_obs.empty or forecast_times.empty:
        return field_obs.copy()

    fc = (
        pd.DataFrame({"wave_time_pst": pd.to_datetime(forecast_times).sort_values()})
        .drop_duplicates(subset=["wave_time_pst"])
        .sort_values("wave_time_pst")
    )
    parts: list[pd.DataFrame] = []
    for bid, grp in field_obs.groupby("break_id", sort=True):
        left = grp.sort_values("obs_time_pst").copy()
        merged = pd.merge_asof(
            left,
            fc,
            left_on="obs_time_pst",
            right_on="wave_time_pst",
            direction="nearest",
        )
        merged["match_delta_min"] = (
            (merged["obs_time_pst"] - merged["wave_time_pst"]).dt.total_seconds().abs() / 60.0
        )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def _field_obs_for_break(
    field_obs: pd.DataFrame,
    *,
    break_id: int,
    forecast_times: pd.Series,
) -> pd.DataFrame:
    sub = field_obs.loc[field_obs["break_id"] == break_id]
    if sub.empty:
        return sub
    return _snap_obs_to_forecast_times(sub, forecast_times)


def _get_validation_bundle() -> dict[str, pd.DataFrame]:
    """Load bundle; clear stale Streamlit cache if series keys changed."""
    expected = frozenset(_validation_csv_paths().keys())
    mtime_key = _validation_mtime_key()
    bundle = _load_validation_bundle(mtime_key)
    if expected.issubset(bundle.keys()):
        return bundle
    _load_validation_bundle.clear()
    bundle = _load_validation_bundle(mtime_key)
    if not expected.issubset(bundle.keys()):
        missing = sorted(expected - set(bundle.keys()))
        raise KeyError(f"Validation bundle missing series: {missing}")
    return bundle


def _break_label_from_row(row: pd.Series, labels: dict[int, str]) -> str:
    bid = int(row["break_id"])
    if bid in labels:
        return labels[bid]
    spot = str(row.get("spot_name") or "").strip()
    brk = str(row.get("break_name") or "").strip()
    if spot and brk and spot.lower() != brk.lower():
        return f"{spot} — {brk}"
    if brk:
        return brk
    if spot:
        return spot
    return f"Break {bid}"


def _add_height_line(
    fig: go.Figure,
    x: pd.Series,
    y: pd.Series,
    *,
    name: str,
    color: str,
    dash: str,
    width: float,
    showlegend: bool,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            showlegend=showlegend,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"<b>{name}</b><br>%{{y:.3f}} ft<extra></extra>",
        )
    )


def _bundle_ref_df(bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reference frame for break list / labels (any non-empty series)."""
    for key in ("uncalibrated_buoy", "uncalibrated_cdip", "complex_buoy_scalar"):
        df = bundle.get(key)
        if df is not None and not df.empty:
            return df
    return next(iter(bundle.values()))


def _sorted_break_ids(
    bundle: dict[str, pd.DataFrame],
    obs_counts: pd.Series | None,
) -> tuple[list[int], dict[int, str], dict[int, int]]:
    """Return break_ids (most → least obs), labels, and obs count per break."""
    ref = _bundle_ref_df(bundle)
    break_ids = sorted(ref["break_id"].unique().astype(int).tolist())
    labels_map = _load_break_labels(str(BREAKS_CSV))

    label_by_bid: dict[int, str] = {}
    for bid in break_ids:
        rows = ref.loc[ref["break_id"] == bid]
        label_by_bid[bid] = _break_label_from_row(rows.iloc[0], labels_map)

    count_by_bid: dict[int, int] = {}
    for bid in break_ids:
        if obs_counts is not None and bid in obs_counts.index:
            count_by_bid[bid] = int(obs_counts[bid])
        else:
            count_by_bid[bid] = 0

    sorted_ids = sorted(break_ids, key=lambda b: (-count_by_bid[b], b))
    return sorted_ids, label_by_bid, count_by_bid


def _plot_break_validation(
    bundle: dict[str, pd.DataFrame],
    *,
    break_id: int,
    label: str,
    n_obs: int,
    field_obs: pd.DataFrame | None = None,
) -> go.Figure:
    fig = go.Figure()
    leg: set[str] = set()
    height_series: list[pd.Series] = []
    t_min, t_max = None, None

    for key, name, color, dash, width in SERIES:
        df = bundle.get(key)
        if df is None:
            continue
        sub = df.loc[df["break_id"] == break_id].sort_values("wave_time_pst")
        if sub.empty:
            continue
        t = sub["wave_time_pst"]
        # Validation CSVs store significant_wave_height in feet (not meters).
        h = pd.to_numeric(sub["significant_wave_height"], errors="coerce")
        height_series.append(h)
        if t_min is None:
            t_min, t_max = t.min(), t.max()
        else:
            t_min = min(t_min, t.min())
            t_max = max(t_max, t.max())

        show = name not in leg
        leg.add(name)
        _add_height_line(
            fig,
            t,
            h,
            name=name,
            color=color,
            dash=dash,
            width=width,
            showlegend=show,
        )

    obs_label = "Field obs (Jun 10)"
    if field_obs is not None and not field_obs.empty:
        ref_fc = bundle.get("uncalibrated_buoy")
        fc_times = (
            ref_fc.loc[ref_fc["break_id"] == break_id, "wave_time_pst"]
            if ref_fc is not None
            else pd.Series(dtype="datetime64[ns, US/Pacific]")
        )
        obs_sub = _field_obs_for_break(field_obs, break_id=break_id, forecast_times=fc_times)
        if not obs_sub.empty:
            height_series.append(obs_sub["wave_height_ft"])
            fig.add_trace(
                go.Scatter(
                    x=obs_sub["wave_time_pst"],
                    y=obs_sub["wave_height_ft"],
                    mode="markers",
                    name=obs_label,
                    marker=dict(
                        color=C_OBS,
                        size=10,
                        symbol="circle",
                        line=dict(width=1.2, color=BG_DARK),
                    ),
                    customdata=np.stack(
                        [
                            obs_sub["obs_time_pst"].dt.strftime("%b %d %I:%M %p"),
                            obs_sub["match_delta_min"].round(0).astype(int),
                            obs_sub["spot_raw"].astype(str),
                            obs_sub.get("observation_type", pd.Series([""] * len(obs_sub))).astype(str),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        f"<b>{obs_label}</b><br>"
                        "%{y:.2f} ft<br>"
                        "Obs: %{customdata[0]}<br>"
                        "Spot: %{customdata[2]}<br>"
                        "Δ forecast: %{customdata[1]} min"
                        "<extra></extra>"
                    ),
                )
            )

    if not height_series:
        fig.add_annotation(
            text="No forecast rows for this break",
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
            height=360,
            margin=dict(l=56, r=20, t=72, b=48),
        )
        return fig

    span_days = max((t_max - t_min).total_seconds() / 86400.0, 0.25)
    tick_fmt = _forecast_hover_xaxis_tickformat(span_days)
    x_tickvals = _pacific_daily_midnight_ticks(t_min, t_max)
    y_r = _y_range_padded(*height_series, frac=0.08, floor_zero=True, min_span=0.35)

    obs_note = f"  ·  {n_obs:,} observations" if n_obs else "  ·  no calibration observations"
    fig.update_layout(
        title=dict(
            text=f"{label}{obs_note}",
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
        margin=dict(l=56, r=20, t=88, b=48),
        height=460,
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
        title_text="Date (US/Pacific)",
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        showgrid=True,
        zeroline=False,
        title_text="Wave height (ft)",
        range=list(y_r),
    )
    return fig


def render_south_swell_validation_tab() -> None:
    st.header("20260610 South Swell validation")
    st.caption(
        "Significant wave height (ft) vs time (US/Pacific) per break. "
        "Six forecast variants plus **Jun 10 field observations** (green dots; x snapped to "
        "nearest forecast time). Breaks ranked **most → least** by observation count "
        f"from `{FIELD_OBS_CSV.name}`."
    )

    paths = _validation_csv_paths()
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        st.error("Missing validation CSV(s):\n" + "\n".join(f"- `{p}`" for p in missing))
        return

    try:
        bundle = _get_validation_bundle()
    except Exception as exc:
        st.error(f"Failed to load validation CSVs: {exc}")
        return

    obs_mtime = FIELD_OBS_CSV.stat().st_mtime if FIELD_OBS_CSV.exists() else 0.0
    all_cal_obs, unmatched_spots = _load_calibration_observations(
        str(FIELD_OBS_CSV),
        obs_mtime,
        str(BREAKS_CSV),
    )
    obs_counts = _obs_counts_from_calibration(all_cal_obs)
    field_obs = all_cal_obs.loc[
        all_cal_obs["obs_time_pst"].dt.date == FIELD_OBS_DATE
    ].copy()

    sorted_ids, label_by_bid, count_by_bid = _sorted_break_ids(bundle, obs_counts)

    with st.expander("Data sources", expanded=False):
        st.write(f"Data root: `{DATA_ROOT}`")
        for key, path in paths.items():
            st.write(f"- **{key}**: `{path.name}`")
        st.write(
            "Uncalibrated baselines: `20260610_forecast_uncalibrated_buoy.csv` (buoy folder) "
            "and `20260610_forecast_uncalibrated_cdip.csv` (CDIP folder)."
        )
        st.write(f"Calibration observations: `{FIELD_OBS_CSV.name}`")
        if not all_cal_obs.empty:
            st.write(f"- {len(all_cal_obs):,} total rows matched to breaks (all dates)")
        if not field_obs.empty:
            st.write(f"- {len(field_obs):,} on Jun 10 (plotted as green dots)")
        if unmatched_spots:
            st.warning(
                "Could not match these Jun 10 spot names to a break — "
                "let Garrett know if any should be mapped:\n"
                + "\n".join(f"- `{s}`" for s in unmatched_spots)
            )

    selected: list[int] = st.multiselect(
        "Surf breaks (ranked most → least observations)",
        options=sorted_ids,
        default=sorted_ids,
        format_func=lambda bid: (
            f"{label_by_bid[bid]}  ({count_by_bid[bid]:,} obs)"
            if count_by_bid[bid]
            else f"{label_by_bid[bid]}  (0 obs)"
        ),
        help="Deselect breaks to hide their charts.",
    )

    if not selected:
        st.info("Select at least one break.")
        return

    for bid in selected:
        try:
            fig = _plot_break_validation(
                bundle,
                break_id=bid,
                label=label_by_bid[bid],
                n_obs=count_by_bid[bid],
                field_obs=field_obs,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Plot failed for {label_by_bid[bid]}: {exc}")

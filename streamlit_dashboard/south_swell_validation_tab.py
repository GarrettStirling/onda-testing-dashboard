"""20260610 South Swell validation: wave-height time series per break.

Compares uncalibrated vs buoy/CDIP scalar (complex + minimal) forecasts from
``data/20261006 Wave Height Validation/``. Breaks are ranked most → least by
observation count from ``surf_calibration_data.observations_with_cdip`` (BigQuery).
"""

from __future__ import annotations

from pathlib import Path

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

PROJECT_ID = "onda-maverick"
OBS_TABLE = f"`{PROJECT_ID}.surf_calibration_data.observations_with_cdip`"

# (series key, legend label, color, dash, line width)
SERIES: list[tuple[str, str, str, str, float]] = [
    ("uncalibrated", "Uncalibrated", C_SIG, "solid", LW_SIG),
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
        "uncalibrated": BUOY_DIR / "20260610_forecast_uncalibrated.csv",
        "complex_buoy_scalar": BUOY_DIR / "20260610_forecast_complex_model.csv",
        "minimal_buoy_scalar": BUOY_DIR / "20260610_forecast_minimal_model.csv",
        "complex_cdip_scalar": CDIP_DIR / "20260610_forecast_complex_model.csv",
        "minimal_cdip_scalar": CDIP_DIR / "20260610_forecast_minimal_model.csv",
    }


def _validation_mtime_key() -> tuple[float, ...]:
    return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in _validation_csv_paths().values())


@st.cache_data(show_spinner="Loading South Swell validation CSVs…")
def _load_validation_bundle(_mtime_key: tuple[float, ...]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, path in _validation_csv_paths().items():
        if not path.exists():
            raise FileNotFoundError(f"Missing validation CSV: `{path}`")
        out[key] = _prepare_validation_df(pd.read_csv(path))
    return out


@st.cache_data(ttl=600, show_spinner="Loading observation counts from BigQuery…")
def _load_obs_counts_bigquery(_cache_buster: int = 0) -> pd.Series:
    from streamlit_dashboard.bq_forecast_loader import forecast_bigquery_client

    client = forecast_bigquery_client()
    sql = f"""
    SELECT CAST(break_id AS INT64) AS break_id, COUNT(*) AS n_obs
    FROM {OBS_TABLE}
    WHERE break_id IS NOT NULL
    GROUP BY break_id
    """
    df = client.query(sql).to_dataframe(create_bqstorage_client=False)
    if df.empty:
        return pd.Series(dtype=int)
    return df.set_index("break_id")["n_obs"].astype(int)


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


def _sorted_break_ids(
    bundle: dict[str, pd.DataFrame],
    obs_counts: pd.Series | None,
) -> tuple[list[int], dict[int, str], dict[int, int]]:
    """Return break_ids (most → least obs), labels, and obs count per break."""
    ref = bundle["uncalibrated"]
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
) -> go.Figure:
    fig = go.Figure()
    leg: set[str] = set()
    height_series: list[pd.Series] = []
    t_min, t_max = None, None

    for key, name, color, dash, width in SERIES:
        df = bundle[key]
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

    obs_note = f"  ·  {n_obs:,} observations" if n_obs else "  ·  no BQ observations"
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
        "Five forecast variants: uncalibrated baseline plus buoy- and CDIP-based "
        "complex/minimal scalar models. Breaks ranked **most → least** observations "
        f"from `{OBS_TABLE}`."
    )

    paths = _validation_csv_paths()
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        st.error("Missing validation CSV(s):\n" + "\n".join(f"- `{p}`" for p in missing))
        return

    try:
        bundle = _load_validation_bundle(_validation_mtime_key())
    except Exception as exc:
        st.error(f"Failed to load validation CSVs: {exc}")
        return

    use_bq = st.toggle(
        "Rank breaks by BigQuery observation count",
        value=True,
        key="south_swell_use_bq",
        help=(
            f"Uses `{OBS_TABLE}` for per-break observation counts. "
            "Disable to sort by break_id only (no credentials needed)."
        ),
    )

    obs_counts: pd.Series | None = None
    if use_bq:
        refresh = st.button("↺ Refresh observation counts", key="south_swell_bq_refresh")
        if refresh:
            st.session_state["south_swell_bq_buster"] = (
                st.session_state.get("south_swell_bq_buster", 0) + 1
            )
        try:
            obs_counts = _load_obs_counts_bigquery(
                st.session_state.get("south_swell_bq_buster", 0)
            )
        except Exception as exc:
            st.warning(f"BigQuery observation counts unavailable: {exc}")
            st.caption("Falling back to break_id order.")

    sorted_ids, label_by_bid, count_by_bid = _sorted_break_ids(bundle, obs_counts)

    with st.expander("Data sources", expanded=False):
        st.write(f"Data root: `{DATA_ROOT}`")
        for key, path in paths.items():
            st.write(f"- **{key}**: `{path.name}`")
        st.write(
            "Uncalibrated file is shared between buoy and CDIP folders "
            "(identical baseline forecast)."
        )
        if obs_counts is not None:
            st.write(f"Observation ranking: `{OBS_TABLE}`")

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
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Plot failed for {label_by_bid[bid]}: {exc}")

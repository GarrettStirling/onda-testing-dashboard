"""Forecast tab: buoy components + CDIP via nearest-time join.

Builds ``buoy_cdip_nearest_join.csv``: each buoy row gets the nearest CDIP row (same break)
within a time tolerance (default 2.5 h), so 1/4/7 vs 2/5/8 style grids still match.
Plots use buoy ``wave_time_pst`` as x; CDIP values are drawn at that time from the join.

Sources:
  - data/forecasts/buoy_scaled_components.csv
  - data/forecasts/cdip_data_p.csv
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pytz
import streamlit as st

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


def _break_circular(series: pd.Series, threshold: float = 180.0) -> pd.Series:
    s = series.copy().astype(float)
    jumps = s.diff().abs() > threshold
    s[jumps] = np.nan
    return s


def _apply_dark_style(fig: plt.Figure, axes: list) -> None:
    fig.patch.set_facecolor(BG_DARK)
    for ax in axes:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.8)
        ax.set_axisbelow(True)


def _format_x_axis(ax, span_days: float) -> None:
    """Tick density from plotted time span (no user slider)."""
    if span_days <= 3.5:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", which="minor", length=3, color=SPINE_COLOR)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _plot_break_forecast(
    joined_g: pd.DataFrame,
    *,
    show_cdip_sig: bool,
    overlay_cdip_mop: bool,
    label: str,
) -> plt.Figure:
    """x-axis = buoy ``wave_time_pst``; CDIP series from nearest-join columns (same timestamps)."""
    joined_g = joined_g.copy().sort_values("wave_time_pst")

    if joined_g.empty:
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_PANEL)
        ax.text(0.5, 0.5, "No buoy forecast rows in window", ha="center", va="center", color=TEXT_COLOR)
        ax.axis("off")
        return fig

    t_b = joined_g["wave_time_pst"]
    span_days = max(
        (joined_g["wave_time_pst"].max() - joined_g["wave_time_pst"].min()).total_seconds() / 86400.0,
        0.25,
    )

    show_any_cdip = show_cdip_sig or overlay_cdip_mop
    has_sig = "cdip_significant_wave_height" in joined_g.columns and joined_g["cdip_significant_wave_height"].notna().any()
    has_mop = "cdip_primary_wave_height" in joined_g.columns and joined_g["cdip_primary_wave_height"].notna().any()

    # Subplots: height, direction, period
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"hspace": 0.08, "height_ratios": [1.2, 1, 1]},
    )
    ax_h, ax_d, ax_p = axes
    _apply_dark_style(fig, list(axes))

    # --- Height: buoy pri/sec/ter, then optional CDIP MOP overlay, then bold CDIP sig ---
    ax_h.plot(
        t_b,
        _heights_to_ft(joined_g["primary_wave_height_buoy_scaled"]),
        color=C_PRIMARY,
        lw=LW_MAIN,
        label="Primary (buoy components)",
        zorder=4,
    )
    ax_h.plot(
        t_b,
        _heights_to_ft(joined_g["secondary_wave_height_buoy_scaled"]),
        color=C_SECONDARY,
        lw=LW_SEC,
        linestyle="--",
        label="Secondary (buoy components)",
        zorder=3,
    )
    ax_h.plot(
        t_b,
        _heights_to_ft(joined_g["tertiary_wave_height_buoy_scaled"]),
        color=C_TERTIARY,
        lw=LW_TER,
        linestyle=":",
        label="Tertiary (buoy components)",
        zorder=2,
    )

    if overlay_cdip_mop and has_mop:
        ax_h.plot(
            t_b,
            _heights_to_ft(joined_g["cdip_primary_wave_height"]),
            color=C_PRIMARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            linestyle="-",
            label="Primary (CDIP MOP)",
            zorder=3,
        )
        ax_h.plot(
            t_b,
            _heights_to_ft(joined_g["cdip_secondary_wave_height"]),
            color=C_SECONDARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            linestyle="--",
            label="Secondary (CDIP MOP)",
            zorder=2,
        )
        ter_h = joined_g["cdip_tertiary_wave_height"] if "cdip_tertiary_wave_height" in joined_g.columns else None
        if ter_h is not None and ter_h.notna().any():
            ax_h.plot(
                t_b,
                _heights_to_ft(ter_h),
                color=C_TERTIARY,
                lw=LW_OVERLAY,
                alpha=ALPHA_OVERLAY,
                linestyle=":",
                label="Tertiary (CDIP MOP)",
                zorder=1,
            )

    if show_cdip_sig and has_sig:
        ax_h.plot(
            t_b,
            _heights_to_ft(joined_g["cdip_significant_wave_height"]),
            color=C_SIG,
            lw=LW_SIG,
            label="Sig. height (CDIP)",
            zorder=6,
            alpha=0.95,
        )

    ax_h.set_ylabel("Height (ft)", fontsize=10)
    cols_h = [
        joined_g["primary_wave_height_buoy_scaled"],
        joined_g["secondary_wave_height_buoy_scaled"],
        joined_g["tertiary_wave_height_buoy_scaled"],
    ]
    ymax = float(pd.concat(cols_h).max()) * M_TO_FT
    if show_cdip_sig and has_sig:
        sig_max = pd.to_numeric(joined_g["cdip_significant_wave_height"], errors="coerce").max()
        if pd.notna(sig_max):
            ymax = max(ymax, float(sig_max) * M_TO_FT)
    ax_h.set_ylim(bottom=0, top=max(ymax * 1.12, 1.0))
    ax_h.legend(
        loc="upper right",
        fontsize=7,
        framealpha=0.25,
        facecolor=BG_PANEL,
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
        ncol=2,
    )

    # --- Direction ---
    ax_d.plot(
        t_b,
        _break_circular(joined_g["primary_direction_buoy_scaled"]),
        color=C_PRIMARY,
        lw=LW_MAIN,
        label="Primary (buoy components)",
        zorder=4,
    )
    ax_d.plot(
        t_b,
        _break_circular(joined_g["secondary_direction_buoy_scaled"]),
        color=C_SECONDARY,
        lw=LW_SEC,
        linestyle="--",
        label="Secondary (buoy components)",
        zorder=3,
    )
    ax_d.plot(
        t_b,
        _break_circular(joined_g["tertiary_direction_buoy_scaled"]),
        color=C_TERTIARY,
        lw=LW_TER,
        linestyle=":",
        label="Tertiary (buoy components)",
        zorder=2,
    )

    if overlay_cdip_mop and has_mop:
        ax_d.plot(
            t_b,
            _break_circular(pd.to_numeric(joined_g["cdip_primary_direction"], errors="coerce")),
            color=C_PRIMARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            label="Primary (CDIP MOP)",
            zorder=3,
        )
        ax_d.plot(
            t_b,
            _break_circular(pd.to_numeric(joined_g["cdip_secondary_direction"], errors="coerce")),
            color=C_SECONDARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            linestyle="--",
            label="Secondary (CDIP MOP)",
            zorder=2,
        )
        if "cdip_tertiary_direction" in joined_g.columns and joined_g["cdip_tertiary_direction"].notna().any():
            ax_d.plot(
                t_b,
                _break_circular(pd.to_numeric(joined_g["cdip_tertiary_direction"], errors="coerce")),
                color=C_TERTIARY,
                lw=LW_OVERLAY,
                alpha=ALPHA_OVERLAY,
                linestyle=":",
                label="Tertiary (CDIP MOP)",
                zorder=1,
            )

    ax_d.set_ylabel("Direction (° from north)", fontsize=10)
    ax_d.set_ylim(0, 360)
    ax_d.set_yticks(range(0, 361, 45))
    ax_d.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d°"))
    for deg in [0, 90, 180, 270, 360]:
        ax_d.axhline(deg, color=SPINE_COLOR, linewidth=0.5, linestyle="-", alpha=0.6)
    ax_d.legend(
        loc="upper right",
        fontsize=7,
        framealpha=0.25,
        facecolor=BG_PANEL,
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
        ncol=2,
    )

    # --- Period ---
    ax_p.plot(
        t_b,
        joined_g["primary_period_buoy_scaled"],
        color=C_PRIMARY,
        lw=LW_MAIN,
        label="Primary (buoy components)",
    )
    ax_p.plot(
        t_b,
        joined_g["secondary_period_buoy_scaled"],
        color=C_SECONDARY,
        lw=LW_SEC,
        linestyle="--",
        label="Secondary (buoy components)",
    )
    ax_p.plot(
        t_b,
        joined_g["tertiary_period_buoy_scaled"],
        color=C_TERTIARY,
        lw=LW_TER,
        linestyle=":",
        label="Tertiary (buoy components)",
    )

    if overlay_cdip_mop and has_mop:
        ax_p.plot(
            t_b,
            pd.to_numeric(joined_g["cdip_primary_period"], errors="coerce"),
            color=C_PRIMARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            label="Primary (CDIP MOP)",
        )
        ax_p.plot(
            t_b,
            pd.to_numeric(joined_g["cdip_secondary_period"], errors="coerce"),
            color=C_SECONDARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            linestyle="--",
            label="Secondary (CDIP MOP)",
        )
        if "cdip_tertiary_period" in joined_g.columns and joined_g["cdip_tertiary_period"].notna().any():
            ax_p.plot(
                t_b,
                pd.to_numeric(joined_g["cdip_tertiary_period"], errors="coerce"),
                color=C_TERTIARY,
                lw=LW_OVERLAY,
                alpha=ALPHA_OVERLAY,
                linestyle=":",
                label="Tertiary (CDIP MOP)",
            )

    ax_p.set_ylabel("Period (s)", fontsize=10)
    ax_p.set_ylim(bottom=0)
    ax_p.legend(
        loc="upper right",
        fontsize=7,
        framealpha=0.25,
        facecolor=BG_PANEL,
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
        ncol=2,
    )

    _format_x_axis(ax_p, span_days)
    ax_p.set_xlabel("Date (US/Pacific)", fontsize=10)

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
    fig.suptitle(
        f"Forecast — {title}{cdip_note}",
        color=TEXT_COLOR,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1, hspace=0.12)
    return fig


def render_forecast_tab() -> None:
    st.header("Forecasts")
    st.caption(
        "Buoy rows drive the time axis. CDIP values are attached with **nearest** timestamps per break "
        f"(≤ **match window** hours apart) and written to `{JOINED_CSV.name}`. "
        "Staggered grids (e.g. buoy 1/4/7 vs CDIP 2/5/8) still match; if buoy and CDIP date ranges "
        "never overlap, CDIP columns stay empty."
    )

    if not BUOY_CSV.exists():
        st.error(f"Missing buoy forecast CSV: `{BUOY_CSV}`")
        return
    if not CDIP_CSV.exists():
        st.warning(f"CDIP file not found: `{CDIP_CSV}` — joined table will have empty `cdip_*` columns.")

    labels = _load_break_labels(str(BREAKS_CSV))

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
        default=options[: min(6, len(options))],
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
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=BG_DARK)
            plt.close(fig)
            buf.seek(0)
            st.image(buf, width="stretch")
        except Exception as exc:
            st.error(f"Plot failed for break {bid}: {exc}")

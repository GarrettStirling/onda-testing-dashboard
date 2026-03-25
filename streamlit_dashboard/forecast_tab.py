"""Forecast tab: buoy-scaled swell components + CDIP sig height + optional CDIP MOP overlay.

Data:
  - data/forecasts/buoy_scaled_components.csv  (pri/sec/ter scaled from buoys)
  - data/forecasts/cdip_data_p.csv             (CDIP MOP processed; sig + components)
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
BREAKS_CSV = REPO_ROOT / "data" / "reference" / "breaks_with_names.csv"

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


def _best_cdip_time_shift_hours(buoy_g: pd.DataFrame, cdip_g: pd.DataFrame) -> float:
    """Pick shift in {-2..2} h that minimizes median clock skew (buoy vs shifted CDIP times).

    Uses ``merge_asof`` (nearest, 4 h tolerance — enough for typical 3 h buoy/CDIP cadence).
    Tie-break: smallest ``|h|`` so a 1 h pipeline skew prefers ±1 over ±2 when medians tie.
    If nothing matches well (all medians > 4 h or <5 pairs), returns 0.
    """
    b = buoy_g.sort_values("wave_time_pst")[
        ["wave_time_pst", "primary_wave_height_buoy_scaled"]
    ].dropna(subset=["primary_wave_height_buoy_scaled"])
    if len(b) < 5 or cdip_g.empty or "primary_wave_height" not in cdip_g.columns:
        return 0.0
    c = cdip_g.sort_values("wave_time_pst")[["wave_time_pst", "primary_wave_height"]].dropna(
        subset=["primary_wave_height"]
    )
    if len(c) < 3:
        return 0.0

    b = b.rename(columns={"wave_time_pst": "buoy_t"})
    tol = pd.Timedelta(hours=4)
    best_h = 0.0
    best_key: tuple[float, int] | None = None

    for h in (-2, -1, 0, 1, 2):
        c2 = c.copy()
        c2["cdip_t"] = c2["wave_time_pst"] + pd.Timedelta(hours=h)
        c2 = c2.sort_values("cdip_t")[["cdip_t", "primary_wave_height"]]

        m = pd.merge_asof(
            b.sort_values("buoy_t"),
            c2,
            left_on="buoy_t",
            right_on="cdip_t",
            direction="nearest",
            tolerance=tol,
        )
        m = m.dropna(subset=["primary_wave_height", "primary_wave_height_buoy_scaled"])
        if len(m) < 5:
            continue
        med_dt = float((m["buoy_t"] - m["cdip_t"]).abs().dt.total_seconds().median())
        key = (med_dt, abs(int(h)))
        if best_key is None or key < best_key:
            best_key = key
            best_h = float(h)

    if best_key is None:
        return 0.0
    med_best, _ = best_key
    if med_best > 4 * 3600:
        return 0.0
    return best_h


def _resolve_cdip_shift_hours(
    buoy_g: pd.DataFrame,
    cdip_g: pd.DataFrame | None,
    mode: str,
) -> float:
    """``mode``: ``auto`` | numeric hour string like ``0``, ``-1``, ``+1``."""
    if cdip_g is None or cdip_g.empty:
        return 0.0
    if mode == "auto":
        return _best_cdip_time_shift_hours(buoy_g, cdip_g)
    try:
        return float(mode)
    except ValueError:
        return 0.0


def _cdip_window_and_times(
    buoy_g: pd.DataFrame,
    cdip_g: pd.DataFrame,
    shift_hours: float,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Rows whose *plotted* time (CDIP PST + shift) overlaps the buoy span; returns aligned plot times."""
    t_min_b = buoy_g["wave_time_pst"].min()
    t_max_b = buoy_g["wave_time_pst"].max()
    margin = pd.Timedelta(hours=8)
    shift_td = pd.Timedelta(hours=float(shift_hours))
    c = cdip_g.sort_values("wave_time_pst").copy()
    t_plot = c["wave_time_pst"] + shift_td
    mask = (t_plot >= t_min_b - margin) & (t_plot <= t_max_b + margin)
    win = c.loc[mask]
    if win.empty:
        return pd.DataFrame(), None
    t_c = win["wave_time_pst"] + shift_td
    return win, t_c


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
    break_id: int,
    buoy_g: pd.DataFrame,
    cdip_g: pd.DataFrame | None,
    *,
    show_cdip_sig: bool,
    overlay_cdip_mop: bool,
    cdip_align_mode: str,
    label: str,
) -> plt.Figure:
    """Three subplots: buoy components always; optional CDIP sig and/or MOP with shared time shift."""
    buoy_g = buoy_g.copy()
    buoy_g = buoy_g.sort_values("wave_time_pst")

    if buoy_g.empty:
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_PANEL)
        ax.text(0.5, 0.5, "No buoy forecast rows in window", ha="center", va="center", color=TEXT_COLOR)
        ax.axis("off")
        return fig

    t_b = buoy_g["wave_time_pst"]
    span_days = max(
        (buoy_g["wave_time_pst"].max() - buoy_g["wave_time_pst"].min()).total_seconds() / 86400.0,
        0.25,
    )

    show_any_cdip = show_cdip_sig or overlay_cdip_mop
    cdip_win = pd.DataFrame()
    t_c = None
    if show_any_cdip and cdip_g is not None and not cdip_g.empty:
        shift_h = _resolve_cdip_shift_hours(buoy_g, cdip_g, cdip_align_mode)
        cdip_win, t_c = _cdip_window_and_times(buoy_g, cdip_g, shift_h)

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
        _heights_to_ft(buoy_g["primary_wave_height_buoy_scaled"]),
        color=C_PRIMARY,
        lw=LW_MAIN,
        label="Primary (buoy components)",
        zorder=4,
    )
    ax_h.plot(
        t_b,
        _heights_to_ft(buoy_g["secondary_wave_height_buoy_scaled"]),
        color=C_SECONDARY,
        lw=LW_SEC,
        linestyle="--",
        label="Secondary (buoy components)",
        zorder=3,
    )
    ax_h.plot(
        t_b,
        _heights_to_ft(buoy_g["tertiary_wave_height_buoy_scaled"]),
        color=C_TERTIARY,
        lw=LW_TER,
        linestyle=":",
        label="Tertiary (buoy components)",
        zorder=2,
    )

    if overlay_cdip_mop and not cdip_win.empty and t_c is not None:
        ax_h.plot(
            t_c,
            _heights_to_ft(cdip_win["primary_wave_height"]),
            color=C_PRIMARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            linestyle="-",
            label="Primary (CDIP MOP)",
            zorder=3,
        )
        ax_h.plot(
            t_c,
            _heights_to_ft(cdip_win["secondary_wave_height"]),
            color=C_SECONDARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            linestyle="--",
            label="Secondary (CDIP MOP)",
            zorder=2,
        )
        ter_h = cdip_win["tertiary_wave_height"]
        if ter_h.notna().any():
            ax_h.plot(
                t_c,
                _heights_to_ft(ter_h),
                color=C_TERTIARY,
                lw=LW_OVERLAY,
                alpha=ALPHA_OVERLAY,
                linestyle=":",
                label="Tertiary (CDIP MOP)",
                zorder=1,
            )

    if show_cdip_sig and not cdip_win.empty and t_c is not None and "significant_wave_height" in cdip_win.columns:
        ax_h.plot(
            t_c,
            _heights_to_ft(cdip_win["significant_wave_height"]),
            color=C_SIG,
            lw=LW_SIG,
            label="Sig. height (CDIP)",
            zorder=6,
            alpha=0.95,
        )

    ax_h.set_ylabel("Height (ft)", fontsize=10)
    cols_h = [
        buoy_g["primary_wave_height_buoy_scaled"],
        buoy_g["secondary_wave_height_buoy_scaled"],
        buoy_g["tertiary_wave_height_buoy_scaled"],
    ]
    ymax = float(pd.concat(cols_h).max()) * M_TO_FT
    if show_cdip_sig and not cdip_win.empty and "significant_wave_height" in cdip_win.columns:
        sig_max = pd.to_numeric(cdip_win["significant_wave_height"], errors="coerce").max()
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
        _break_circular(buoy_g["primary_direction_buoy_scaled"]),
        color=C_PRIMARY,
        lw=LW_MAIN,
        label="Primary (buoy components)",
        zorder=4,
    )
    ax_d.plot(
        t_b,
        _break_circular(buoy_g["secondary_direction_buoy_scaled"]),
        color=C_SECONDARY,
        lw=LW_SEC,
        linestyle="--",
        label="Secondary (buoy components)",
        zorder=3,
    )
    ax_d.plot(
        t_b,
        _break_circular(buoy_g["tertiary_direction_buoy_scaled"]),
        color=C_TERTIARY,
        lw=LW_TER,
        linestyle=":",
        label="Tertiary (buoy components)",
        zorder=2,
    )

    if overlay_cdip_mop and not cdip_win.empty and t_c is not None:
        ax_d.plot(
            t_c,
            _break_circular(pd.to_numeric(cdip_win["primary_direction"], errors="coerce")),
            color=C_PRIMARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            label="Primary (CDIP MOP)",
            zorder=3,
        )
        ax_d.plot(
            t_c,
            _break_circular(pd.to_numeric(cdip_win["secondary_direction"], errors="coerce")),
            color=C_SECONDARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            linestyle="--",
            label="Secondary (CDIP MOP)",
            zorder=2,
        )
        if "tertiary_direction" in cdip_win.columns and cdip_win["tertiary_direction"].notna().any():
            ax_d.plot(
                t_c,
                _break_circular(pd.to_numeric(cdip_win["tertiary_direction"], errors="coerce")),
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
        buoy_g["primary_period_buoy_scaled"],
        color=C_PRIMARY,
        lw=LW_MAIN,
        label="Primary (buoy components)",
    )
    ax_p.plot(
        t_b,
        buoy_g["secondary_period_buoy_scaled"],
        color=C_SECONDARY,
        lw=LW_SEC,
        linestyle="--",
        label="Secondary (buoy components)",
    )
    ax_p.plot(
        t_b,
        buoy_g["tertiary_period_buoy_scaled"],
        color=C_TERTIARY,
        lw=LW_TER,
        linestyle=":",
        label="Tertiary (buoy components)",
    )

    if overlay_cdip_mop and not cdip_win.empty and t_c is not None:
        ax_p.plot(
            t_c,
            pd.to_numeric(cdip_win["primary_period"], errors="coerce"),
            color=C_PRIMARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            label="Primary (CDIP MOP)",
        )
        ax_p.plot(
            t_c,
            pd.to_numeric(cdip_win["secondary_period"], errors="coerce"),
            color=C_SECONDARY,
            lw=LW_OVERLAY,
            alpha=ALPHA_OVERLAY,
            linestyle="--",
            label="Secondary (CDIP MOP)",
        )
        if "tertiary_period" in cdip_win.columns and cdip_win["tertiary_period"].notna().any():
            ax_p.plot(
                t_c,
                pd.to_numeric(cdip_win["tertiary_period"], errors="coerce"),
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

    title = label if label else f"Break {break_id}"
    has_cdip = not cdip_win.empty
    if not show_any_cdip:
        cdip_note = " — buoy only"
    elif not has_cdip:
        cdip_note = " — buoy (no CDIP in buoy window)"
    elif show_cdip_sig and overlay_cdip_mop:
        cdip_note = " — buoy + CDIP sig + MOP"
    elif show_cdip_sig:
        cdip_note = " — buoy + CDIP sig"
    else:
        cdip_note = " — buoy + CDIP MOP"
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
        "Buoy components from `buoy_scaled_components.csv`. Optional CDIP sig height and/or MOP overlay from "
        "`cdip_data_p.csv`; **Auto** aligns CDIP time to buoy by correlating primary heights (±2 h). "
        "If date ranges do not overlap, no CDIP lines appear until you refresh the CSVs."
    )

    if not BUOY_CSV.exists():
        st.error(f"Missing buoy forecast CSV: `{BUOY_CSV}`")
        return
    if not CDIP_CSV.exists():
        st.warning(f"CDIP file not found: `{CDIP_CSV}` — plots will show buoy data only (no sig height / overlay).")

    labels = _load_break_labels(str(BREAKS_CSV))

    with st.spinner("Loading forecast CSVs..."):
        df_buoy = _load_buoy_forecast(str(BUOY_CSV))
        df_cdip = _load_cdip_forecast(str(CDIP_CSV)) if CDIP_CSV.exists() else None

    break_ids = sorted(df_buoy["break_id"].unique().astype(int).tolist())
    if not break_ids:
        st.warning("No rows in buoy forecast file.")
        return

    with st.expander("Data sources", expanded=False):
        st.write(f"Buoy scaled components: `{BUOY_CSV}`")
        st.write(f"CDIP MOP processed: `{CDIP_CSV}`")
        st.write(f"Break labels: `{BREAKS_CSV}`")

    show_cdip_sig = st.checkbox(
        "Show CDIP significant wave height (`cdip_data_p.csv`)",
        value=True,
        help="Bold line on the height panel. Uses the same time alignment as MOP (below).",
    )
    overlay_cdip_mop = st.checkbox(
        "Overlay CDIP MOP (pri/sec/ter on all three panels)",
        value=False,
        help="Semi-transparent CDIP component lines for comparison with buoy components. Off by default.",
    )
    align_labels = {
        "Auto (match primary height)": "auto",
        "0 h (no shift)": "0",
        "-1 h": "-1",
        "+1 h": "1",
        "-2 h": "-2",
        "+2 h": "2",
    }
    choice = st.selectbox(
        "CDIP time vs buoy clock",
        options=list(align_labels.keys()),
        index=0,
        help="Auto picks the shift in ±2 h that best matches buoy vs CDIP primary height. "
        "Choose a fixed offset if Auto is wrong or your files share no overlap (Auto → 0 h).",
    )
    cdip_align_mode = align_labels[choice]

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
        bsub = df_buoy[df_buoy["break_id"] == bid]
        csub = df_cdip[df_cdip["break_id"] == bid] if df_cdip is not None else None

        try:
            fig = _plot_break_forecast(
                int(bid),
                bsub,
                csub,
                show_cdip_sig=show_cdip_sig,
                overlay_cdip_mop=overlay_cdip_mop,
                cdip_align_mode=cdip_align_mode,
                label=id_to_label.get(bid, f"Break {bid}"),
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=BG_DARK)
            plt.close(fig)
            buf.seek(0)
            st.image(buf, width="stretch")
        except Exception as exc:
            st.error(f"Plot failed for break {bid}: {exc}")

"""Obs vs CDIP MOP: observation–model difference and scalar vs MOP/tide drivers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
OBS_VS_CDIP_CSV = REPO_ROOT / "data" / "obs_enriched" / "observations_vs_cdip_diff_and_scale.csv"

BG_DARK = "#0e1117"
BG_PANEL = "#161b22"
GRID_COLOR = "#2a2d35"
TEXT_COLOR = "#c9d1d9"
SPINE_COLOR = "#30363d"
ACCENT_DIFF = "#38bdf8"
ACCENT_SCALAR = "#fb923c"

# Taller rows: total figure height scales with this (two columns side by side).
ROW_HEIGHT_PX = 450


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _spot_break_columns(df: pd.DataFrame) -> tuple[str, str]:
    spot_c = _first_existing_column(df, ["spot", "Spot"])
    break_c = _first_existing_column(df, ["break", "break_id"])
    if spot_c is None:
        raise ValueError("CSV needs a `spot`/`Spot` column.")
    if break_c is None:
        return spot_c, "__no_break__"
    return spot_c, break_c


def _combo_label(spot: str, brk: str) -> str:
    s = (spot or "").strip()
    b = (brk or "").strip()
    if b and b.lower() != s.lower():
        return f"{s} — {b}"
    if s:
        return s
    return b or "(unnamed)"


def _histogram_nbins(x: np.ndarray) -> int:
    """Freedman–Diaconis–style bin count, clamped for small samples."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return max(5, n)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = float(q75 - q25)
    rng = float(np.ptp(x))
    if rng <= 0 or not np.isfinite(rng):
        return min(20, max(5, int(np.sqrt(n)) + 1))
    if not np.isfinite(iqr) or iqr <= 0:
        width = rng / min(25, max(8, int(np.sqrt(n))))
    else:
        width = 2.0 * iqr * (n ** (-1.0 / 3.0))
        width = max(width, rng / 45.0)
    nb = int(np.ceil(rng / width)) if width > 0 else 15
    return int(np.clip(nb, 8, 45))


def _histogram_nbins_narrower(x: np.ndarray) -> int:
    """Like `_histogram_nbins` but ~60% bin width → ~1/0.6 more bins (capped)."""
    nb = _histogram_nbins(x)
    return int(max(8, min(100, round(nb / 0.6))))


def _apply_dark_layout(fig: go.Figure, *, title: str, height: int) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=15), x=0.5, xanchor="center"),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_COLOR, size=11),
        margin=dict(l=56, r=24, t=64, b=52),
        height=height,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, showgrid=True)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, showgrid=True)
    fig.update_annotations(font=dict(color=TEXT_COLOR, size=11))


def _obs_csv_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _plot_combined_diff_scalar(
    df_sub: pd.DataFrame,
    *,
    combo_title: str,
    has_tide: bool,
) -> go.Figure:
    """Two columns: left = mop_surfline_scalar, right = mop_obs_scalar. Histograms unchanged; scatters
    use diff/scalar on x-axis and MOP, CDIP buoy fields, or tide on y-axis."""
    # Rows: dist, Hs, MOP period, buoy primary period, buoy primary direction,
    # weighted period (MOP), weighted direction (buoy), [tide].
    n_rows = 8 if has_tide else 7
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        vertical_spacing=0.055,
        horizontal_spacing=0.09,
    )

    left = pd.to_numeric(df_sub["mop_surfline_scalar"], errors="coerce")
    scal = pd.to_numeric(df_sub["mop_obs_scalar"], errors="coerce")

    def _empty_both(message: str) -> go.Figure:
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=TEXT_COLOR, size=14),
        )
        _apply_dark_layout(fig, title=combo_title, height=400)
        return fig

    # Row 1: distributions (value on x, count on y — not flipped)
    for col, (series, color, label) in enumerate(
        (
            (left, ACCENT_DIFF, "mop_surfline_scalar"),
            (scal, ACCENT_SCALAR, "mop_obs_scalar"),
        ),
        start=1,
    ):
        sv = series.dropna()
        if len(sv) == 0:
            fig.add_trace(go.Histogram(x=[], nbinsx=10, showlegend=False), row=1, col=col)
        else:
            nb = _histogram_nbins_narrower(sv.to_numpy())
            fig.add_trace(
                go.Histogram(
                    x=sv,
                    nbinsx=nb,
                    marker_color=color,
                    marker_line_width=0.5,
                    marker_line_color=SPINE_COLOR,
                ),
                row=1,
                col=col,
            )
        fig.update_yaxes(title_text="Row count", row=1, col=col)

    fig.update_xaxes(
        title_text="Distribution<br>scalar (surfline Hs / MOP Hs)",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Distribution<br>scalar (obs Hs / MOP Hs)",
        row=1,
        col=2,
    )

    if left.notna().sum() == 0 and scal.notna().sum() == 0:
        return _empty_both("No finite `mop_surfline_scalar` or `mop_obs_scalar` values")

    def _scatter_flipped(
        v: pd.Series,
        mop: pd.Series,
        *,
        row: int,
        col: int,
        x_title: str,
        y_title: str,
        color: str,
        x_key: str,
        y_key: str,
    ) -> None:
        m = v.notna() & mop.notna()
        if m.sum() == 0:
            fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False), row=row, col=col)
        else:
            fig.add_trace(
                go.Scatter(
                    x=v[m],
                    y=mop[m],
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=0.55, line=dict(width=0)),
                    hovertemplate=f"{x_key}=%{{x:.4f}}<br>{y_key}=%{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text=x_title, row=row, col=col)
        fig.update_yaxes(title_text=y_title, row=row, col=col)

    x_left = "mop_surfline_scalar (surfline / MOP)"
    x_scal = "mop_obs_scalar (obs / MOP)"

    mop_hs = pd.to_numeric(df_sub["mop_hs_ft"], errors="coerce")
    mop_per = pd.to_numeric(df_sub["mop_period_s"], errors="coerce")
    dir_buoy = pd.to_numeric(df_sub["dir_pri_deg_buoy"], errors="coerce")
    per_pri_buoy = pd.to_numeric(df_sub["per_pri_s_buoy"], errors="coerce")
    period_weighted_mop = pd.to_numeric(df_sub["period_weighted_s_mop"], errors="coerce")
    direction_weighted_buoy = pd.to_numeric(df_sub["direction_weighted_deg_buoy"], errors="coerce")

    r_hs = 2
    _scatter_flipped(left, mop_hs, row=r_hs, col=1, x_title=x_left, y_title="Sig. Wave Height (CDIP MOP, ft)", color=ACCENT_DIFF, x_key="mop_surfline_scalar", y_key="mop_hs_ft")
    _scatter_flipped(scal, mop_hs, row=r_hs, col=2, x_title=x_scal, y_title="Sig. Wave Height (CDIP MOP, ft)", color=ACCENT_SCALAR, x_key="mop_obs_scalar", y_key="mop_hs_ft")

    r_per = 3
    _scatter_flipped(left, mop_per, row=r_per, col=1, x_title=x_left, y_title="Primary Period (CDIP MOP, s)", color=ACCENT_DIFF, x_key="mop_surfline_scalar", y_key="mop_period_s")
    _scatter_flipped(scal, mop_per, row=r_per, col=2, x_title=x_scal, y_title="Primary Period (CDIP MOP, s)", color=ACCENT_SCALAR, x_key="mop_obs_scalar", y_key="mop_period_s")

    r_per_pri = 4
    _scatter_flipped(
        left,
        per_pri_buoy,
        row=r_per_pri,
        col=1,
        x_title=x_left,
        y_title="Primary Period (CDIP Buoy, s)",
        color=ACCENT_DIFF,
        x_key="mop_surfline_scalar",
        y_key="per_pri_s_buoy",
    )
    _scatter_flipped(
        scal,
        per_pri_buoy,
        row=r_per_pri,
        col=2,
        x_title=x_scal,
        y_title="Primary Period (CDIP Buoy, s)",
        color=ACCENT_SCALAR,
        x_key="mop_obs_scalar",
        y_key="per_pri_s_buoy",
    )

    r_dir_buoy = 5
    _scatter_flipped(
        left,
        dir_buoy,
        row=r_dir_buoy,
        col=1,
        x_title=x_left,
        y_title="Primary Direction (CDIP Buoy, deg)",
        color=ACCENT_DIFF,
        x_key="mop_surfline_scalar",
        y_key="dir_pri_deg_buoy",
    )
    _scatter_flipped(
        scal,
        dir_buoy,
        row=r_dir_buoy,
        col=2,
        x_title=x_scal,
        y_title="Primary Direction (CDIP Buoy, deg)",
        color=ACCENT_SCALAR,
        x_key="mop_obs_scalar",
        y_key="dir_pri_deg_buoy",
    )

    r_period_weighted = 6
    _scatter_flipped(
        left,
        period_weighted_mop,
        row=r_period_weighted,
        col=1,
        x_title=x_left,
        y_title="Weighted Period (CDIP MOP, s)",
        color=ACCENT_DIFF,
        x_key="mop_surfline_scalar",
        y_key="period_weighted_s_mop",
    )
    _scatter_flipped(
        scal,
        period_weighted_mop,
        row=r_period_weighted,
        col=2,
        x_title=x_scal,
        y_title="Weighted Period (CDIP MOP, s)",
        color=ACCENT_SCALAR,
        x_key="mop_obs_scalar",
        y_key="period_weighted_s_mop",
    )

    r_direction_weighted = 7
    _scatter_flipped(
        left,
        direction_weighted_buoy,
        row=r_direction_weighted,
        col=1,
        x_title=x_left,
        y_title="Weighted Direction (CDIP Buoy, deg)",
        color=ACCENT_DIFF,
        x_key="mop_surfline_scalar",
        y_key="direction_weighted_deg_buoy",
    )
    _scatter_flipped(
        scal,
        direction_weighted_buoy,
        row=r_direction_weighted,
        col=2,
        x_title=x_scal,
        y_title="Weighted Direction (CDIP Buoy, deg)",
        color=ACCENT_SCALAR,
        x_key="mop_obs_scalar",
        y_key="direction_weighted_deg_buoy",
    )

    if has_tide:
        tide = pd.to_numeric(df_sub["tide_ft"], errors="coerce")
        r_tide = 8
        _scatter_flipped(left, tide, row=r_tide, col=1, x_title=x_left, y_title="Tide (ft)", color=ACCENT_DIFF, x_key="mop_surfline_scalar", y_key="tide_ft")
        _scatter_flipped(scal, tide, row=r_tide, col=2, x_title=x_scal, y_title="Tide (ft)", color=ACCENT_SCALAR, x_key="mop_obs_scalar", y_key="tide_ft")

    height = ROW_HEIGHT_PX * n_rows
    _apply_dark_layout(fig, title=combo_title, height=height)
    return fig


@st.cache_data(show_spinner=False)
def _load_obs_vs_cdip(path_str: str, csv_mtime: float) -> pd.DataFrame:
    # `csv_mtime` is part of the cache key so edits to the CSV invalidate cached data.
    _ = csv_mtime
    df = pd.read_csv(path_str)
    # Column aliases for cleaned header variants.
    alias_map = {
        "mop_hs_ft": ["mop_hs_ft", "hs_mop_ft"],
        "mop_period_s": ["mop_period_s", "per_mop_s"],
        "tide_ft": ["tide_ft", "tide_linear_ft"],
    }
    for canonical, candidates in alias_map.items():
        if canonical in df.columns:
            continue
        found = _first_existing_column(df, candidates)
        if found is not None:
            df[canonical] = df[found]
    for c in (
        "mop_obs_diff",
        "mop_obs_scalar",
        "mop_surfline_scalar",
        "mop_hs_ft",
        "mop_period_s",
        "dir_pri_deg_buoy",
        "per_pri_s_buoy",
        "period_weighted_s_mop",
        "direction_weighted_deg_buoy",
        "tide_ft",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def render_obs_vs_cdip_mop_tab() -> None:
    st.header("CDIP MOP Comparison")
    st.caption(
        "Per spot+break: **mop_surfline_scalar** (left) and **mop_obs_scalar** (right). Histograms show "
        "each metric on the x-axis; scatter panels put diff/scalar on **x** and MOP, **CDIP buoy** "
        "direction / primary period, or tide on **y**. Source: "
        "`data/obs_enriched/observations_vs_cdip_diff_and_scale.csv`."
    )

    if not OBS_VS_CDIP_CSV.exists():
        st.error(f"Missing CSV: `{OBS_VS_CDIP_CSV}`")
        return

    csv_mtime = _obs_csv_mtime(OBS_VS_CDIP_CSV)
    df = _load_obs_vs_cdip(str(OBS_VS_CDIP_CSV), csv_mtime)
    need = {
        "mop_surfline_scalar",
        "mop_obs_scalar",
        "mop_hs_ft",
        "mop_period_s",
        "dir_pri_deg_buoy",
        "per_pri_s_buoy",
        "period_weighted_s_mop",
        "direction_weighted_deg_buoy",
    }
    miss = need - set(df.columns)
    if miss:
        st.error(f"CSV missing required columns: {sorted(miss)}")
        return

    spot_c, break_c = _spot_break_columns(df)
    df["_spot_k"] = df[spot_c].astype(str).str.strip()
    if break_c == "__no_break__":
        df["_break_k"] = ""
    else:
        df["_break_k"] = df[break_c].fillna("").astype(str).str.strip()
    df["_combo"] = [_combo_label(s, b) for s, b in zip(df["_spot_k"], df["_break_k"])]

    # Keep dropdown options to combos that actually have observations.
    # Primary signal is `observed_hs_ft`; fallback to scalar columns if needed.
    if "observed_hs_ft" in df.columns:
        has_obs_mask = pd.to_numeric(df["observed_hs_ft"], errors="coerce").notna()
    else:
        has_obs_mask = (
            pd.to_numeric(df["mop_obs_scalar"], errors="coerce").notna()
            | pd.to_numeric(df["mop_surfline_scalar"], errors="coerce").notna()
        )
    df_opts = df.loc[has_obs_mask].copy()

    combo_counts = df_opts.groupby("_combo", sort=False).size().sort_values(ascending=False)
    options = combo_counts.index.tolist()
    if not options:
        st.warning("No spot+break rows with observations found in the CSV.")
        return

    default_n = min(8, len(options))
    selected = st.multiselect(
        "Spot + break",
        options=options,
        default=options[:default_n],
        help="Each selection renders one wide figure: diff (left column), scalar (right).",
    )

    if not selected:
        st.info("Select at least one spot+break.")
        return

    for combo in selected:
        sub = df[df["_combo"] == combo].copy()
        n = len(sub)
        has_tide = "tide_ft" in sub.columns and sub["tide_ft"].notna().any()
        if not has_tide and "tide_ft" in sub.columns:
            st.caption(f"{combo}: no non-null `tide_ft` — tide row omitted.")

        try:
            fig = _plot_combined_diff_scalar(
                sub,
                combo_title=f"{combo} ({n} rows)",
                has_tide=has_tide,
            )
            st.plotly_chart(fig, width="stretch", key=f"obs_cdip_combo_{combo}")
        except Exception as exc:
            st.error(f"Plot failed: {exc}")

        st.divider()

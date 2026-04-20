"""QC Calibration tab: residual diagnostics vs MOP and buoy predictors per break.

Indexed by ``break_id`` (pipeline join key) with display labels from spot + break names per
``SURF_SPOT_MAPPING_GUIDE`` (Spot — Break when names differ).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QC_CSV = REPO_ROOT / "data" / "regression_qc" / "observation_residuals_test_averaged.csv"

BG_DARK = "#0e1117"
BG_PANEL = "#161b22"
GRID_COLOR = "#2a2d35"
TEXT_COLOR = "#c9d1d9"
SPINE_COLOR = "#30363d"
MARKER_COLOR = "#38bdf8"
HLINE_COLOR = "#64748b"

Y_COL = "residual_hs_ft_mean"

REQUIRED_FOR_HOVER = ("observed_hs_ft", "pred_hs_ft_mean")

# Pipeline key; one chart per distinct break_id (spot + break in the title)
REQUIRED_INDEX = ("break_id",)

# (column_name, x-axis label) — row-major: top row MOP (a–c), bottom row buoy (d–f)
X_PANELS: list[tuple[str, str]] = [
    ("mop_hs_ft", "MOP Hs (ft)"),
    ("mop_period_s", "MOP period (s)"),
    ("mop_direction", "MOP direction (°)"),
    ("buoy_primary_period", "Buoy primary period (s)"),
    ("buoy_primary_direction", "Buoy primary direction (°)"),
    ("buoy_mean_direction", "Buoy mean direction (°)"),
]


def _resolve_spot_column(df: pd.DataFrame) -> str:
    """Which column carries spot labels (`spot_name` preferred). Used for CSV validation."""
    if "spot_name" in df.columns and df["spot_name"].notna().any():
        return "spot_name"
    if "spot" in df.columns:
        return "spot"
    raise ValueError("QC CSV needs a `spot_name` or `spot` column (surf mapping / display).")


def _format_break_display_label(spot_name: str, break_name: str, break_id: int) -> str:
    """Align with SURF_SPOT_MAPPING_GUIDE: ``Spot — Break`` or single name when equal."""
    spot = (spot_name or "").strip()
    brk = (break_name or "").strip()
    if spot and brk and spot.lower() != brk.lower():
        return f"{spot} — {brk}"
    if brk:
        return brk
    if spot:
        return spot
    return f"Break {break_id}"


def _display_label_for_row(row: pd.Series) -> str:
    spot = str(row.get("spot_name") or row.get("spot") or "").strip()
    brk = str(row.get("break_name") or row.get("break") or "").strip()
    bid_raw = row.get("break_id")
    try:
        bid = int(bid_raw) if pd.notna(bid_raw) else 0
    except (TypeError, ValueError):
        bid = 0
    return _format_break_display_label(spot, brk, bid)


def _fmt_hover_num(v: object, *, nd: int = 2, suffix: str = "") -> str:
    if v is None:
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except TypeError:
        return "—"
    try:
        return f"{float(v):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _hover_text_for_row(row: pd.Series) -> str:
    """Rich hover: observed/predicted Hs, CDIP MOP swell, nearshore buoy CDIP."""
    parts: list[str] = [f"<b>{_display_label_for_row(row)}</b>"]
    if "break_id" in row.index and pd.notna(row.get("break_id")):
        try:
            parts.append(f"<b>break_id:</b> {int(row['break_id'])}")
        except (TypeError, ValueError):
            pass

    if "observed_hs_ft" in row.index:
        parts.append(f"<b>Observed Hs:</b> {_fmt_hover_num(row.get('observed_hs_ft'), nd=2)} ft")
    if "pred_hs_ft_mean" in row.index:
        parts.append(f"<b>Predicted Hs:</b> {_fmt_hover_num(row.get('pred_hs_ft_mean'), nd=2)} ft")

    if "observed_at" in row.index and pd.notna(row.get("observed_at")):
        parts.append(f"<b>Observed at:</b> {row['observed_at']}")

    parts.append("<b>CDIP MOP (offshore swell)</b>")
    parts.append(
        f"Hs {_fmt_hover_num(row.get('mop_hs_ft'), nd=3)} ft · "
        f"Period {_fmt_hover_num(row.get('mop_period_s'), nd=2)} s · "
        f"Dir {_fmt_hover_num(row.get('mop_direction'), nd=1)}°"
    )

    parts.append("<b>CDIP buoy (nearshore)</b>")
    parts.append(
        f"Pri. period {_fmt_hover_num(row.get('buoy_primary_period'), nd=2)} s · "
        f"Pri. dir {_fmt_hover_num(row.get('buoy_primary_direction'), nd=1)}° · "
        f"Mean dir {_fmt_hover_num(row.get('buoy_mean_direction'), nd=1)}°"
    )

    if "cdip_transect" in row.index and pd.notna(row.get("cdip_transect")):
        parts.append(f"<b>CDIP transect:</b> {row['cdip_transect']}")

    return "<br>".join(parts)


@st.cache_data(show_spinner=False)
def _load_qc_csv(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    for col in list(REQUIRED_FOR_HOVER) + list(REQUIRED_INDEX) + [c for c, _ in X_PANELS] + [Y_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in QC CSV.")
    _resolve_spot_column(df)
    return df


def _qc_break_figure(df_break: pd.DataFrame, *, title_label: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[lbl for _, lbl in X_PANELS],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, (x_col, x_label) in enumerate(X_PANELS):
        row = i // 3 + 1
        col = i % 3 + 1
        x = pd.to_numeric(df_break[x_col], errors="coerce")
        y = pd.to_numeric(df_break[Y_COL], errors="coerce")
        mask = x.notna() & y.notna()
        sub = df_break.loc[mask]
        hover_texts = [_hover_text_for_row(r) for _, r in sub.iterrows()]
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                marker=dict(size=9, color=MARKER_COLOR, opacity=0.72, line=dict(width=0.3, color="#0c4a6e")),
                hovertext=hover_texts,
                hovertemplate=(
                    "%{hovertext}<br><br>"
                    f"<b>{x_label}</b> (this panel x): %{{x:.4f}}<br>"
                    "<b>Mean residual</b>: %{y:.4f} ft"
                    "<extra></extra>"
                ),
                name=x_label,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_hline(
            y=0.0,
            line_dash="dash",
            line_width=1,
            line_color=HLINE_COLOR,
            opacity=0.75,
            row=row,
            col=col,
        )

    fig.update_layout(
        title=dict(
            text=f"Residual QC — {title_label}",
            font=dict(color=TEXT_COLOR, size=16),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_COLOR, size=11),
        margin=dict(l=52, r=28, t=88, b=52),
        height=640,
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        showgrid=True,
        zeroline=False,
        title_font=dict(size=11),
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        showgrid=True,
        zeroline=False,
        title_font=dict(size=11),
        tickfont=dict(size=10),
    )

    # Shared y-axis label on left column only
    fig.update_yaxes(title_text="Residual Hs (ft)", row=1, col=1)
    fig.update_yaxes(title_text="Residual Hs (ft)", row=2, col=1)

    fig.update_annotations(font=dict(color=TEXT_COLOR, size=11))

    return fig


def render_qc_calibration_tab() -> None:
    st.header("QC Calibration")
    st.caption(
        "Mean test-split residual (`residual_hs_ft_mean`) vs offshore/MOP and buoy inputs. "
        "Charts are **per break** (`break_id`); titles use **spot + break** like the surf mapping guide "
        "(e.g. `Steamer Lane — Middle Peak`). Dashed line: zero residual."
    )

    csv_path = DEFAULT_QC_CSV
    if not csv_path.exists():
        st.error(f"QC CSV not found: `{csv_path}`")
        return

    try:
        df = _load_qc_csv(str(csv_path))
    except Exception as exc:
        st.error(f"Failed to load QC CSV: {exc}")
        return

    df = df.loc[pd.notna(df["break_id"])].copy()
    df["break_id"] = pd.to_numeric(df["break_id"], errors="coerce")
    df = df.loc[df["break_id"].notna()].copy()
    df["break_id"] = df["break_id"].astype(int)

    break_ids = sorted(df["break_id"].unique().tolist())
    if not break_ids:
        st.warning("No `break_id` values found in the QC file.")
        return

    label_by_bid: dict[int, str] = {}
    for bid in break_ids:
        row0 = df.loc[df["break_id"] == bid].iloc[0]
        label_by_bid[int(bid)] = _display_label_for_row(row0)

    sorted_bids = sorted(break_ids, key=lambda b: label_by_bid[int(b)].lower())

    selected = st.multiselect(
        "Breaks",
        options=sorted_bids,
        format_func=lambda bid: label_by_bid[int(bid)],
        default=sorted_bids,
        help="One 3×2 residual grid per break (pipeline key: break_id). Label = spot + break display name.",
    )

    if not selected:
        st.info("Select at least one break.")
        return

    with st.expander("Data source", expanded=False):
        st.code(str(csv_path.resolve()), language="text")

    for bid in selected:
        sub = df[df["break_id"] == int(bid)]
        if sub.empty:
            continue
        n = len(sub)
        title = label_by_bid[int(bid)]
        st.subheader(title)
        st.caption(f"`break_id={bid}` · {n} observations")
        fig = _qc_break_figure(sub, title_label=title)
        st.plotly_chart(fig, width="stretch")

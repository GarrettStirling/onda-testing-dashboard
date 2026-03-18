from __future__ import annotations

from pathlib import Path

import matplotlib

# Matplotlib backend selection helps avoid issues in Streamlit environments.
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT.parent / "onda-backend"

ANALYTIC_CSV = REPO_ROOT / "data" / "k_coef" / "k_coefficient_matrix_analytic.csv"
SWAN_CSV = REPO_ROOT / "data" / "k_coef" / "k_coefficient_matrix_swan.csv"

# Matches defaults from `onda-backend/scripts/plot_refraction_coefficients.py`
DATA_REFERENCE_BREAKS_CSV = REPO_ROOT / "data" / "reference" / "breaks_with_names.csv"
BACKEND_BREAKS_WITH_NAMES_CSV = (
    BACKEND_ROOT / "temp_component_testing" / "intermediates" / "breaks_with_names.csv"
)


# ── K-factor color scale (kept aligned to backend plot script) ──────────────
K_VMIN = 0.0
K_VMAX = 2.5
CMAP = "RdYlGn"


def _format_break_label(spot_name: str, break_name: str, break_id: int) -> str:
    """
    Display convention from `onda-backend/DATA_MODEL.md`:
    - spot != break: "Spot — Break"
    - spot == break: show once as "Mavericks" (break name)
    """
    spot = (spot_name or "").strip()
    brk = (break_name or "").strip()

    if spot and brk and spot.lower() != brk.lower():
        return f"{spot} — {brk}"
    if brk:
        return brk
    if spot:
        return spot
    return f"Break {break_id}"


def _resolve_breaks_with_names_csv_path() -> Path:
    """
    Prefer the local reference CSV in this repo.
    If missing, fall back to the backend intermediates (useful during local runs).
    """
    if DATA_REFERENCE_BREAKS_CSV.exists():
        return DATA_REFERENCE_BREAKS_CSV
    return BACKEND_BREAKS_WITH_NAMES_CSV


@st.cache_data(show_spinner=False)
def load_break_labels(breaks_csv_path: str) -> dict[int, str]:
    if not Path(breaks_csv_path).exists():
        return {}

    df = pd.read_csv(
        breaks_csv_path,
        usecols=["break_id", "spot_name", "break_name"],
        dtype={"break_id": "int32", "spot_name": "string", "break_name": "string"},
    )

    labels: dict[int, str] = {}
    for _, row in df.iterrows():
        bid = int(row["break_id"])
        labels[bid] = _format_break_label(
            spot_name=str(row.get("spot_name", "") or ""),
            break_name=str(row.get("break_name", "") or ""),
            break_id=bid,
        )
    return labels


@st.cache_data(show_spinner=False)
def load_k_matrix(csv_path: str) -> pd.DataFrame:
    """
    Load full k-matrix CSV once; files are small (~tens of thousands of rows).
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(csv_path)

    return pd.read_csv(
        csv_path,
        dtype={
            "break_id": "int32",
            "swell_dir": "float32",
            "swell_period": "float32",
            "k_factor": "float32",
        },
    )


def _make_polar_kfactor_plot(group: pd.DataFrame, *, title: str | None = None) -> plt.Figure:
    """
    Dark-themed polar heatmap (radar-style) for one break.
    Mirrors the structure of `plot_refraction_coefficients.py::plot_break`.
    """
    if group.empty:
        raise ValueError("Empty group")

    # Use ints for indexing consistency with the backend plotting approach.
    local = group.copy()
    local["swell_dir"] = np.rint(local["swell_dir"]).astype(int)
    local["swell_period"] = np.rint(local["swell_period"]).astype(int)

    dirs_deg = sorted(local["swell_dir"].unique())
    periods = sorted(local["swell_period"].unique())

    pivot = local.pivot(index="swell_dir", columns="swell_period", values="k_factor")
    pivot = pivot.reindex(index=dirs_deg, columns=periods)

    p_max = int(max(periods))
    r_vals = np.array([p_max + 1 - p for p in periods], dtype=float)  # reversed periods

    # Continuity grid for 0..360 step=5 (fill missing directions with NaN).
    all_dirs_deg = np.arange(0, 365, 5)
    k_full = np.full((len(all_dirs_deg), len(r_vals)), np.nan, dtype=float)
    for i, d in enumerate(all_dirs_deg):
        if d in pivot.index:
            k_full[i, :] = pivot.loc[d].values.astype(float)

    theta_edges = np.radians(np.append(all_dirs_deg, all_dirs_deg[-1] + 5))
    r_edges = np.append(r_vals, r_vals[-1] + 1)
    T, R = np.meshgrid(theta_edges[:-1], r_edges[:-1], indexing="ij")

    fig, ax = plt.subplots(figsize=(4.9, 4.9), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0f111a")
    ax.set_facecolor("#0f111a")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    norm = mcolors.Normalize(vmin=K_VMIN, vmax=K_VMAX)
    mesh = ax.pcolormesh(
        T,
        R,
        k_full,
        cmap=CMAP,
        norm=norm,
        shading="auto",
    )

    # Angular ticks
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=7, color="#cbd5e1")

    # Radial ticks
    tick_periods = [p for p in [6, 8, 10, 13, 16, 20, 22] if p in periods]
    tick_r = [p_max + 1 - p for p in tick_periods]
    ax.set_yticks(tick_r)
    ax.set_yticklabels([f"{p}s" for p in tick_periods], fontsize=6, color="#cbd5e1")
    ax.set_ylim(0, float(max(r_edges)))

    # Optional title (we usually show break name above the two columns in Streamlit)
    if title:
        ax.set_title(title, pad=16, fontsize=10, fontweight="bold", color="#e5e7eb")

    # Subtle helper label (kept small to reduce clutter)
    ax.text(
        np.radians(135),
        float(max(r_edges)) * 1.08,
        "← longer period   shorter period →",
        fontsize=6,
        color="#94a3b8",
        ha="center",
        transform=ax.transData,
    )

    # Ensure spines/ticks are legible on dark background
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # `mesh` is intentionally unused beyond creation (no per-plot colorbar to keep layout compact).
    _ = mesh
    return fig


def _csv_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _plot_cache_dir(*, breaks_with_names_csv_path: Path) -> Path:
    key = (
        f"{int(_csv_mtime(ANALYTIC_CSV))}_"
        f"{int(_csv_mtime(SWAN_CSV))}_"
        f"{int(_csv_mtime(breaks_with_names_csv_path))}"
    )
    d = REPO_ROOT / ".streamlit_cache" / "k_coefficients" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_plot_png(
    *,
    cache_dir: Path,
    df: pd.DataFrame,
    break_id: int,
    dataset_name: str,
) -> Path:
    out_path = cache_dir / f"{dataset_name}_break_{break_id}.png"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    group = df[df["break_id"] == break_id]
    if group.empty:
        raise ValueError(f"No data for break_id={break_id} in dataset={dataset_name}")

    fig = _make_polar_kfactor_plot(group)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_k_coefficients_tab() -> None:
    st.header("K Coefficients")

    if not ANALYTIC_CSV.exists():
        st.error(f"Missing analytic CSV: `{ANALYTIC_CSV}`")
        return
    if not SWAN_CSV.exists():
        st.error(f"Missing swan CSV: `{SWAN_CSV}`")
        return

    breaks_csv_path = _resolve_breaks_with_names_csv_path()

    if not DATA_REFERENCE_BREAKS_CSV.exists():
        st.warning(
            "Using fallback `breaks_with_names.csv` path. "
            "Prefer placing it at `data/reference/breaks_with_names.csv`."
        )

    if not breaks_csv_path.exists():
        st.error(
            "Missing `breaks_with_names.csv` for break label formatting. "
            "Expected `data/reference/breaks_with_names.csv`."
        )
        return

    cache_dir = _plot_cache_dir(breaks_with_names_csv_path=breaks_csv_path)

    with st.expander("Data sources", expanded=False):
        st.write(f"Analytical CSV: `{ANALYTIC_CSV}`")
        st.write(f"Swan CSV: `{SWAN_CSV}`")
        st.write(f"Break labels: `{breaks_csv_path}`")

    # Load data (cached by Streamlit)
    with st.spinner("Loading k-coefficient CSVs..."):
        df_analytic = load_k_matrix(str(ANALYTIC_CSV))
        df_swan = load_k_matrix(str(SWAN_CSV))

    labels = load_break_labels(str(breaks_csv_path))

    break_ids = sorted(
        {
            int(x)
            for x in set(df_analytic["break_id"].unique()).intersection(df_swan["break_id"].unique())
        }
    )
    if not break_ids:
        st.error("No overlapping break_id values found between the two CSVs.")
        return

    st.caption(
        "Dark polar heatmaps. Each row shows one break: analytical (left) and swan (right)."
    )

    selected_break_ids = st.multiselect(
        "Select breaks to display",
        options=break_ids,
        default=break_ids,
    )

    if not selected_break_ids:
        st.info("Select at least one break.")
        return

    for break_id in selected_break_ids:
        label = labels.get(int(break_id)) or f"Break {break_id}"

        st.markdown(f"### {label}")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            analytic_png = _ensure_plot_png(
                cache_dir=cache_dir,
                df=df_analytic,
                break_id=int(break_id),
                dataset_name="analytical",
            )
            st.image(analytic_png, use_container_width=True)

        with c2:
            swan_png = _ensure_plot_png(
                cache_dir=cache_dir,
                df=df_swan,
                break_id=int(break_id),
                dataset_name="swan",
            )
            st.image(swan_png, use_container_width=True)


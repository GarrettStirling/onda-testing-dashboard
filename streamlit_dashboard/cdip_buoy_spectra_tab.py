from __future__ import annotations

from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECTRA_DIR = REPO_ROOT / "data" / "buoy_spectrum_animation"


def _humanize_name(p: Path) -> str:
    # Example: buoy_029_1d_spectrum_qc.gif -> buoy 029 (1d spectrum qc)
    stem = p.stem.replace("_", " ").strip()
    stem = stem.replace("buoy ", "buoy ")
    return stem


def render_cdip_buoy_spectra_tab() -> None:
    st.header("CDIP buoy spectra")
    st.caption("Animated buoy spectra GIFs from `data/buoy_spectrum_animation/`.")

    if not SPECTRA_DIR.exists():
        st.error(f"Missing folder: `{SPECTRA_DIR}`")
        return

    gifs = sorted(SPECTRA_DIR.glob("*.gif"))
    if not gifs:
        st.warning(f"No `.gif` files found in `{SPECTRA_DIR}`")
        return

    st.divider()
    for p in gifs:
        st.subheader(_humanize_name(p))
        st.image(str(p))


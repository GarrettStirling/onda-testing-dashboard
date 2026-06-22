"""Field observation CSV loading and spot → break_id mapping."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
FIELD_OBS_CSV = REPO_ROOT / "data" / "observations" / "20260621_Onda_Observed_Wave_Height.csv"
DEFAULT_MIN_SELECTED_OBS = 10

PST = pytz.timezone("US/Pacific")

# Manual fixes when observation ``Spot`` text does not match reference labels.
SPOT_ALIASES: dict[str, int] = {
    "4 mile": 15,
    "four mile": 15,
    "mitchell s": 19,
    "mitchells": 19,
    "waddell reef": 12,
    "waddell beach": 11,
    "swift": 18,
    "steamer point": 20,
    "pleasure point rockview": 26,
    "pleaseure point rockview": 26,
}


def norm_spot_text(s: str) -> str:
    t = (s or "").lower().strip()
    t = t.replace("'", "").replace("\u2019", "")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@st.cache_data(show_spinner=False)
def build_spot_to_break_id(path: str) -> dict[str, int]:
    """Map normalized observation ``Spot`` strings → ``break_id``."""
    df = pd.read_csv(path, usecols=["break_id", "spot_name", "break_name"])
    lookup: dict[str, int] = {}

    def _add(key: str, bid: int) -> None:
        k = norm_spot_text(key)
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
        lookup[norm_spot_text(alias)] = bid
    return lookup


def parse_field_wave_height_ft(raw: object) -> float | None:
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


def parse_field_obs_datetime(date_s: object, time_s: object) -> pd.Timestamp | pd.NaT:
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
def load_field_observations(
    csv_path: str,
    csv_mtime: float,
    breaks_path: str,
) -> tuple[pd.DataFrame, list[str]]:
    """All observation CSV rows with a resolved ``break_id``.

    Returns ``(dataframe, unmatched_spot_names)``.
    """
    p = Path(csv_path)
    if not p.exists() or csv_mtime <= 0:
        return pd.DataFrame(), []

    raw = pd.read_csv(p)
    if raw.empty or "Spot" not in raw.columns:
        return pd.DataFrame(), []

    spot_lookup = build_spot_to_break_id(breaks_path)
    rows: list[dict] = []
    unmatched: set[str] = set()

    for _, r in raw.iterrows():
        obs_dt = parse_field_obs_datetime(r.get("Date"), r.get("Time"))
        if pd.isna(obs_dt):
            continue
        hs = parse_field_wave_height_ft(r.get("Wave_Height_ft"))
        if hs is None:
            continue
        spot_raw = str(r.get("Spot") or "").strip()
        bid = spot_lookup.get(norm_spot_text(spot_raw))
        if bid is None:
            unmatched.add(spot_raw)
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
        return pd.DataFrame(), sorted(unmatched)

    out = pd.DataFrame(rows)
    return out.sort_values(["break_id", "obs_time_pst"]), sorted(unmatched)


def obs_counts_by_break_id(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby("break_id").size().astype(int)


def sort_break_ids_by_obs_count(
    break_ids: list[int],
    obs_counts: pd.Series | None,
) -> tuple[list[int], dict[int, int]]:
    """Return break_ids sorted most → least observations, plus count per break."""
    count_by_bid: dict[int, int] = {}
    for bid in break_ids:
        if obs_counts is not None and bid in obs_counts.index:
            count_by_bid[bid] = int(obs_counts[bid])
        else:
            count_by_bid[bid] = 0
    sorted_ids = sorted(break_ids, key=lambda b: (-count_by_bid[b], b))
    return sorted_ids, count_by_bid


def default_break_ids_with_min_obs(
    break_ids: list[int],
    count_by_bid: dict[int, int],
    *,
    min_obs: int = DEFAULT_MIN_SELECTED_OBS,
) -> list[int]:
    """Break IDs with at least ``min_obs`` observations (preserves ``break_ids`` order)."""
    return [bid for bid in break_ids if count_by_bid.get(bid, 0) >= min_obs]

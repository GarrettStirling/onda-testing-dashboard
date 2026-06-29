"""Load surf forecast slices from BigQuery (`surf_forecast_data` dataset).

Uses Application Default Credentials (``gcloud auth application-default login``),
``GOOGLE_APPLICATION_CREDENTIALS``, or Streamlit secrets ``[gcp_service_account]``.
"""

from __future__ import annotations

import os
from pathlib import Path

import google.auth
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID = "onda-maverick"
BUOY_SCALED_TABLE = f"`{PROJECT_ID}.surf_forecast_data.buoy_scaled_components_p`"
CDIP_TABLE = f"`{PROJECT_ID}.surf_forecast_data.cdip_data_p`"
OFFSHORE_BUOY_TABLE = f"`{PROJECT_ID}.surf_forecast_data.offshore_buoy_data_p`"
GFS_OFFSHORE_TABLE = f"`{PROJECT_ID}.surf_forecast_data.gfs_offshore_wave_data_p`"
BREAK_TO_BUOY_TABLE = f"`{PROJECT_ID}.surf_intermediates.break_to_buoy_map`"
BREAK_TO_GFS_TABLE = f"`{PROJECT_ID}.surf_intermediates.break_to_gfs_map`"
CALIBRATION_OBS_TABLE = f"`{PROJECT_ID}.surf_calibration_data.calibration_observations`"


def _query_to_df(client: bigquery.Client, sql: str) -> pd.DataFrame:
    job = client.query(sql)
    return job.to_dataframe(create_bqstorage_client=False)


@st.cache_resource(show_spinner=False)
def forecast_bigquery_client() -> bigquery.Client:
    os.environ.setdefault("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "googleapis.com")
    try:
        secrets = st.secrets
        if "gcp_service_account" in secrets:
            info = dict(secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(info)
            project = str(info.get("project_id") or PROJECT_ID)
            return bigquery.Client(project=project, credentials=creds)
    except Exception:
        pass

    explicit_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit_path and Path(explicit_path).exists():
        creds, project = google.auth.load_credentials_from_file(explicit_path)
        return bigquery.Client(project=PROJECT_ID or project, credentials=creds)

    win_adc = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
    unix_adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    adc_path = win_adc if win_adc.exists() else unix_adc
    if adc_path.exists():
        creds, project = google.auth.load_credentials_from_file(str(adc_path))
        return bigquery.Client(project=PROJECT_ID or project, credentials=creds)

    creds, project = google.auth.default()
    return bigquery.Client(project=PROJECT_ID or project, credentials=creds)


@st.cache_data(ttl=3600, show_spinner=False)
def _table_columns(dataset: str, table_id: str) -> frozenset[str]:
    client = forecast_bigquery_client()
    q = f"""
    SELECT column_name
    FROM `{PROJECT_ID}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table_id}'
    """
    df = _query_to_df(client, q)
    return frozenset(str(c) for c in df["column_name"].tolist())


def _start_of_today_pacific_sql() -> str:
    """Midnight today in America/Los_Angeles as a UTC TIMESTAMP."""
    return "TIMESTAMP(CURRENT_DATE('America/Los_Angeles'), 'America/Los_Angeles')"


def _time_predicate_sql(columns: frozenset[str], table_alias: str | None = None) -> str:
    """SQL fragment ``col >= start_of_today_pacific`` using the best available time column."""
    start = _start_of_today_pacific_sql()
    prefix = f"{table_alias}." if table_alias else ""
    quoted = lambda c: f"{prefix}`{c}`"

    if "wave_time_utc" in columns:
        return f"{quoted('wave_time_utc')} >= {start}"
    if "wave_time_pst" in columns:
        return f"TIMESTAMP({quoted('wave_time_pst')}, 'America/Los_Angeles') >= {start}"
    raise ValueError(
        "Forecast table needs `wave_time_utc` or `wave_time_pst` for date filtering."
    )


@st.cache_data(ttl=120, show_spinner="Loading buoy / CDIP / offshore from BigQuery…")
def load_forecast_tables_bigquery() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch rows from today (US/Pacific calendar date) onward for all three forecast tables."""
    client = forecast_bigquery_client()

    buoy_cols = _table_columns("surf_forecast_data", "buoy_scaled_components_p")
    cdip_cols = _table_columns("surf_forecast_data", "cdip_data_p")
    offshore_cols = _table_columns("surf_forecast_data", "offshore_buoy_data_p")

    buoy_where = _time_predicate_sql(buoy_cols)
    cdip_where = _time_predicate_sql(cdip_cols)
    offshore_where = _time_predicate_sql(offshore_cols)

    q_buoy = f"SELECT * FROM {BUOY_SCALED_TABLE} WHERE {buoy_where}"
    q_cdip = f"SELECT * FROM {CDIP_TABLE} WHERE {cdip_where}"
    q_off = f"SELECT * FROM {OFFSHORE_BUOY_TABLE} WHERE {offshore_where}"

    df_buoy = _query_to_df(client, q_buoy)
    df_cdip = _query_to_df(client, q_cdip)
    df_off = _query_to_df(client, q_off)

    return df_buoy, df_cdip, df_off


def _break_ids_sql(break_ids: tuple[int, ...]) -> str:
    return ", ".join(str(int(b)) for b in break_ids)


@st.cache_data(ttl=300, show_spinner="Loading CDIP / buoy / GFS reference heights from BigQuery…")
def load_reference_heights_bigquery(
    break_ids: tuple[int, ...],
    min_wave_time_utc: str,
    max_wave_time_utc: str,
) -> pd.DataFrame:
    """Per-break reference bulk Hs series aligned to the served forecast window.

    Returns columns:
      break_id, wave_time_utc,
      cdip_mop_hs_raw_m  — ``cdip_data_p.significant_wave_height_raw``
      offshore_buoy_hs_m — ``offshore_buoy_data_p.significant_wave_height`` at mapped buoy
      gfs_htsgw_m        — ``gfs_offshore_wave_data_p.hs_total_m_gfs`` at mapped GFS point
    """
    if not break_ids:
        return pd.DataFrame()

    client = forecast_bigquery_client()
    ids_sql = _break_ids_sql(break_ids)
    sql = f"""
    WITH b2b AS (
      SELECT break_id, buoy_id
      FROM {BREAK_TO_BUOY_TABLE}
      WHERE break_id IN ({ids_sql})
    ),
    b2g AS (
      SELECT break_id, buoy_id AS gfs_point_id
      FROM {BREAK_TO_GFS_TABLE}
      WHERE break_id IN ({ids_sql})
    ),
    cdip AS (
      SELECT
        break_id,
        wave_time_utc,
        significant_wave_height_raw AS cdip_mop_hs_raw_m
      FROM {CDIP_TABLE}
      WHERE break_id IN ({ids_sql})
        AND wave_time_utc >= TIMESTAMP('{min_wave_time_utc}')
        AND wave_time_utc <= TIMESTAMP('{max_wave_time_utc}')
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY break_id, wave_time_utc
        ORDER BY ingested_at DESC
      ) = 1
    ),
    buoy AS (
      SELECT
        b.break_id,
        o.wave_time_utc,
        o.significant_wave_height AS offshore_buoy_hs_m
      FROM {OFFSHORE_BUOY_TABLE} o
      INNER JOIN b2b b ON o.buoy_id = b.buoy_id
      WHERE o.wave_time_utc >= TIMESTAMP('{min_wave_time_utc}')
        AND o.wave_time_utc <= TIMESTAMP('{max_wave_time_utc}')
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY b.break_id, o.wave_time_utc
        ORDER BY o.ingested_at DESC
      ) = 1
    ),
    gfs AS (
      SELECT
        b.break_id,
        g.wave_time_utc,
        g.hs_total_m_gfs AS gfs_htsgw_m
      FROM {GFS_OFFSHORE_TABLE} g
      INNER JOIN b2g b ON g.gfs_point_id = b.gfs_point_id
      WHERE g.wave_time_utc >= TIMESTAMP('{min_wave_time_utc}')
        AND g.wave_time_utc <= TIMESTAMP('{max_wave_time_utc}')
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY b.break_id, g.wave_time_utc
        ORDER BY g.ingested_at DESC
      ) = 1
    )
    SELECT
      COALESCE(c.break_id, b.break_id, g.break_id) AS break_id,
      COALESCE(c.wave_time_utc, b.wave_time_utc, g.wave_time_utc) AS wave_time_utc,
      c.cdip_mop_hs_raw_m,
      b.offshore_buoy_hs_m,
      g.gfs_htsgw_m
    FROM cdip c
    FULL OUTER JOIN buoy b
      ON c.break_id = b.break_id AND c.wave_time_utc = b.wave_time_utc
    FULL OUTER JOIN gfs g
      ON COALESCE(c.break_id, b.break_id) = g.break_id
     AND COALESCE(c.wave_time_utc, b.wave_time_utc) = g.wave_time_utc
    ORDER BY break_id, wave_time_utc
    """
    return _query_to_df(client, sql)


@st.cache_data(ttl=3600, show_spinner=False)
def load_calibration_observation_counts_bigquery() -> pd.Series:
    """Observation counts per ``break_id`` from ``calibration_observations``."""
    client = forecast_bigquery_client()
    sql = f"""
    SELECT break_id, COUNT(*) AS obs_count
    FROM {CALIBRATION_OBS_TABLE}
    WHERE break_id IS NOT NULL
    GROUP BY break_id
    """
    df = _query_to_df(client, sql)
    if df.empty:
        return pd.Series(dtype=int)
    out = df.set_index("break_id")["obs_count"]
    return pd.to_numeric(out, errors="coerce").dropna().astype(int)

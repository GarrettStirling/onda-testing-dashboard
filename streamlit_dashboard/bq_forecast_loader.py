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

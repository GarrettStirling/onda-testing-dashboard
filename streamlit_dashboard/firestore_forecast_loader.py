"""Load served swell forecasts from Firestore ``surfing_breaks`` documents.

``ingest_swell`` writes coalesced CDIP + GFS values into
``surfingConditions`` (``wavesHeight`` = calibrated_hs). This is what the
mobile app reads for the 16-day chart (GFS extends to f384 ≈ 16 days).

Auth mirrors ``bq_forecast_loader``: ADC, ``GOOGLE_APPLICATION_CREDENTIALS``,
or Streamlit ``[gcp_service_account]`` secrets.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import google.auth
import pandas as pd
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account

PROJECT_ID = "onda-maverick"


def _firestore_client() -> firestore.Client:
    os.environ.setdefault("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "googleapis.com")
    try:
        secrets = st.secrets
        if "gcp_service_account" in secrets:
            info = dict(secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(info)
            project = str(info.get("project_id") or PROJECT_ID)
            return firestore.Client(project=project, credentials=creds)
    except Exception:
        pass

    explicit_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit_path and Path(explicit_path).exists():
        creds, project = google.auth.load_credentials_from_file(explicit_path)
        return firestore.Client(project=PROJECT_ID or project, credentials=creds)

    win_adc = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
    unix_adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    adc_path = win_adc if win_adc.exists() else unix_adc
    if adc_path.exists():
        creds, project = google.auth.load_credentials_from_file(str(adc_path))
        return firestore.Client(project=PROJECT_ID or project, credentials=creds)

    creds, project = google.auth.default()
    return firestore.Client(project=PROJECT_ID or project, credentials=creds)


@st.cache_resource(show_spinner=False)
def forecast_firestore_client() -> firestore.Client:
    return _firestore_client()


def _to_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        ts = pd.Timestamp(value)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_served_swell_forecast(break_id: int) -> dict[str, Any] | None:
    """Fetch ``surfingConditions`` for one break (same query as backend ``forecast_repo``)."""
    # Firestore rejects numpy scalar types (e.g. np.int64 from pandas multiselect).
    break_id = int(break_id)
    db = forecast_firestore_client()
    docs = (
        db.collection("surfing_breaks")
        .where("break_id", "==", break_id)
        .limit(1)
        .stream()
    )
    for doc in docs:
        data = doc.to_dict() or {}
        return {
            "break_id": data.get("break_id", break_id),
            "geohash": doc.id,
            "surfingConditions": data.get("surfingConditions") or [],
            "start_time": _to_utc_datetime(data.get("surfingConditionsStartTime")),
            "end_time": _to_utc_datetime(data.get("surfingConditionsEndTime")),
            "period_hours": data.get("surfingConditionsPeriod"),
            "updated_at": _to_utc_datetime(data.get("surfingConditionsUpdateTime")),
        }
    return None

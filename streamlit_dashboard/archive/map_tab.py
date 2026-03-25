from __future__ import annotations

"""
ARCHIVED (not currently used by app.py).

This was the Map tab implementation (Folium + BigQuery). It is kept here so it can
be brought back later without losing the code.
"""

import os
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

# Before any google.* import: newer google-auth probes GCE metadata for
# "universe domain" unless this is set — on a laptop that times out and breaks BQ.
os.environ.setdefault("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "googleapis.com")

import folium
import google.auth
import pandas as pd
import streamlit as st
from folium.plugins import Draw, MousePosition
from google.cloud import bigquery
from google.oauth2 import service_account
from streamlit_folium import st_folium


PROJECT_ID = "onda-maverick"
OFFSHORE_BUOYS_TABLE = "onda-maverick.surf_system_data.offshore_buoys"
VIRTUAL_OFFSHORE_POINTS_TABLE = "onda-maverick.surf_system_data.virtual_offshore_points"
QUERY_LIMIT = 500

# Canonical schemas (BigQuery) — tables differ; queries are built per-table.
# offshore_buoys:     buoy_id INT, buoy_name, lat, lon, depth_m, ...
# virtual_offshore_points: vop_id STRING, lat, long, depth_m, ...


def _query_to_df(client: bigquery.Client, sql: str) -> pd.DataFrame:
    """REST path only — avoids BigQuery Storage client + extra auth/metadata probes."""
    job = client.query(sql)
    return job.to_dataframe(create_bqstorage_client=False)


@st.cache_resource(show_spinner=False)
def _bq_client() -> bigquery.Client:
    # Hosted Streamlit: service account in secrets. Locally there is often no
    # secrets.toml — touching st.secrets then raises "No secrets found", so
    # we must try/except and fall through to ADC.
    try:
        if "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(info)
            project = str(info.get("project_id") or PROJECT_ID)
            return bigquery.Client(project=project, credentials=creds)
    except Exception:
        pass

    # Prefer explicit local ADC file loading to avoid metadata-server fallbacks.
    explicit_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit_path and Path(explicit_path).exists():
        creds, project = google.auth.load_credentials_from_file(explicit_path)
        return bigquery.Client(project=PROJECT_ID or project, credentials=creds)

    # Local-dev fallback to standard gcloud ADC file locations.
    win_adc = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
    unix_adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    adc_path = win_adc if win_adc.exists() else unix_adc
    if adc_path.exists():
        creds, project = google.auth.load_credentials_from_file(str(adc_path))
        return bigquery.Client(project=PROJECT_ID or project, credentials=creds)

    # Last resort: standard ADC resolution.
    creds, project = google.auth.default()
    return bigquery.Client(project=PROJECT_ID or project, credentials=creds)


@st.cache_data(ttl=600, show_spinner=False)
def _table_columns(table_fqn: str) -> set[str]:
    client = _bq_client()
    project_id, dataset_id, table_id = table_fqn.split(".")
    query = f"""
    SELECT column_name
    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table_id}'
    """
    cols_df = _query_to_df(client, query)
    return {str(c) for c in cols_df["column_name"].tolist()}


def _pick_first_column(columns: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _bq_ident(col: str) -> str:
    """Quote column names for BigQuery (e.g. reserved word `long`)."""
    safe = col.replace("`", "")
    return f"`{safe}`"


@st.cache_data(ttl=600, show_spinner=False)
def _load_map_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    client = _bq_client()
    buoy_cols = _table_columns(OFFSHORE_BUOYS_TABLE)
    vop_cols = _table_columns(VIRTUAL_OFFSHORE_POINTS_TABLE)

    buoy_id_col = _pick_first_column(buoy_cols, ["buoy_id", "id"])
    buoy_name_col = _pick_first_column(buoy_cols, ["buoy_name", "name", "station_name"])
    buoy_lat_col = _pick_first_column(buoy_cols, ["lat", "latitude"])
    buoy_lon_col = _pick_first_column(buoy_cols, ["lon", "long", "lng", "longitude"])
    buoy_depth_col = _pick_first_column(buoy_cols, ["depth", "depth_m", "water_depth"])

    vop_id_col = _pick_first_column(vop_cols, ["vop_id", "id", "point_id"])
    vop_lat_col = _pick_first_column(vop_cols, ["lat", "latitude"])
    vop_lon_col = _pick_first_column(vop_cols, ["lon", "long", "lng", "longitude"])
    vop_depth_col = _pick_first_column(vop_cols, ["depth", "depth_m", "water_depth"])

    if not buoy_id_col or not buoy_lat_col or not buoy_lon_col:
        raise RuntimeError(f"Missing required buoy columns in `{OFFSHORE_BUOYS_TABLE}`: {sorted(buoy_cols)}")
    if not vop_id_col or not vop_lat_col or not vop_lon_col:
        raise RuntimeError(
            f"Missing required VOP columns in `{VIRTUAL_OFFSHORE_POINTS_TABLE}`: {sorted(vop_cols)}"
        )

    bid_b = _bq_ident(buoy_id_col)
    blat_b = _bq_ident(buoy_lat_col)
    blon_b = _bq_ident(buoy_lon_col)
    vid = _bq_ident(vop_id_col)
    vlat = _bq_ident(vop_lat_col)
    vlon = _bq_ident(vop_lon_col)

    buoy_name_expr = f"CAST({_bq_ident(buoy_name_col)} AS STRING)" if buoy_name_col else "''"
    buoy_depth_expr = f"CAST({_bq_ident(buoy_depth_col)} AS FLOAT64)" if buoy_depth_col else "CAST(NULL AS FLOAT64)"
    vop_depth_expr = f"CAST({_bq_ident(vop_depth_col)} AS FLOAT64)" if vop_depth_col else "CAST(NULL AS FLOAT64)"

    buoys_query = f"""
    SELECT
      CAST({bid_b} AS INT64) AS id,
      {buoy_name_expr} AS name,
      CAST({blat_b} AS FLOAT64) AS lat,
      CAST({blon_b} AS FLOAT64) AS lon,
      {buoy_depth_expr} AS depth
    FROM `{OFFSHORE_BUOYS_TABLE}`
    WHERE {blat_b} IS NOT NULL AND {blon_b} IS NOT NULL
    LIMIT {QUERY_LIMIT}
    """

    # vop_id is STRING in surf_system_data.virtual_offshore_points — do not cast to INT64.
    vop_query = f"""
    SELECT
      CAST({vid} AS STRING) AS id,
      CAST({vlat} AS FLOAT64) AS lat,
      CAST({vlon} AS FLOAT64) AS lon,
      {vop_depth_expr} AS depth
    FROM `{VIRTUAL_OFFSHORE_POINTS_TABLE}`
    WHERE {vlat} IS NOT NULL AND {vlon} IS NOT NULL
    LIMIT {QUERY_LIMIT}
    """

    buoys = _query_to_df(client, buoys_query)
    vops = _query_to_df(client, vop_query)
    if "name" not in buoys.columns:
        buoys["name"] = ""

    return buoys, vops


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(min(1.0, sqrt(a)))
    return r * c


def _line_distance_km(coords_lon_lat: list[list[float]]) -> float:
    if len(coords_lon_lat) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(coords_lon_lat)):
        lon1, lat1 = coords_lon_lat[i - 1]
        lon2, lat2 = coords_lon_lat[i]
        total += _haversine_km(float(lat1), float(lon1), float(lat2), float(lon2))
    return total


def render_map_tab() -> None:
    st.header("Buoys and Virtual Offshore Points")
    st.caption("Interactive map with BigQuery-backed points, click-to-copy coordinates, and line measurement.")

    try:
        buoys_df, vops_df = _load_map_points()
    except Exception as exc:
        st.error(f"BigQuery query failed: {exc}")
        st.info(
            "Make sure Application Default Credentials are available for this Streamlit runtime. "
            "For local PowerShell, set "
            "`$env:GOOGLE_APPLICATION_CREDENTIALS='C:\\Users\\garre\\AppData\\Roaming\\gcloud\\application_default_credentials.json'` "
            "before running `streamlit run app.py`. For hosted Streamlit, add a full service-account JSON "
            "under `[gcp_service_account]` in app secrets."
        )
        return

    if buoys_df.empty and vops_df.empty:
        st.warning("No map points returned from BigQuery.")
        return

    depth_values = []
    if not buoys_df.empty:
        depth_values.extend([d for d in buoys_df["depth"].dropna().tolist()])
    if not vops_df.empty:
        depth_values.extend([d for d in vops_df["depth"].dropna().tolist()])

    if depth_values:
        min_depth = float(min(depth_values))
        max_depth = float(max(depth_values))
    else:
        min_depth, max_depth = 0.0, 1000.0

    selected_depth = st.slider(
        "Depth filter (m)",
        min_value=float(min_depth),
        max_value=float(max_depth),
        value=(float(min_depth), float(max_depth)),
        step=1.0,
    )

    lo, hi = selected_depth
    buoys_plot = buoys_df[(buoys_df["depth"].isna()) | ((buoys_df["depth"] >= lo) & (buoys_df["depth"] <= hi))]
    vops_plot = vops_df[(vops_df["depth"].isna()) | ((vops_df["depth"] >= lo) & (vops_df["depth"] <= hi))]

    st.caption(f"Showing {len(buoys_plot)} offshore buoys and {len(vops_plot)} virtual offshore points.")

    all_lats = pd.concat([buoys_plot["lat"], vops_plot["lat"]], ignore_index=True).dropna()
    all_lons = pd.concat([buoys_plot["lon"], vops_plot["lon"]], ignore_index=True).dropna()
    center_lat = float(all_lats.mean()) if not all_lats.empty else 36.8
    center_lon = float(all_lons.mean()) if not all_lons.empty else -122.0

    # Esri World Ocean Base gives an ocean-focused look.
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Ocean Base",
        control_scale=True,
    )

    # Add a second ocean reference layer users can switch to.
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Ocean Reference",
        name="Ocean Labels",
        overlay=True,
        control=True,
    ).add_to(m)

    buoy_fg = folium.FeatureGroup(name="Offshore Buoys", show=True)
    for _, row in buoys_plot.iterrows():
        buoy_id = row.get("id")
        buoy_name = row.get("name", "")
        depth = row.get("depth")
        tooltip = f"Buoy {buoy_id} - {buoy_name} - depth: {depth} m"
        popup = folium.Popup(
            f"<b>Buoy ID:</b> {buoy_id}<br><b>Name:</b> {buoy_name}<br><b>Depth:</b> {depth} m",
            max_width=280,
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=6,
            color="#38bdf8",
            fill=True,
            fill_color="#38bdf8",
            fill_opacity=0.85,
            tooltip=tooltip,
            popup=popup,
        ).add_to(buoy_fg)
    buoy_fg.add_to(m)

    vop_fg = folium.FeatureGroup(name="Virtual Offshore Points", show=True)
    for _, row in vops_plot.iterrows():
        vop_id = row.get("id")
        depth = row.get("depth")
        tooltip = f"VOP {vop_id} - depth: {depth} m"
        popup = folium.Popup(
            f"<b>VOP ID:</b> {vop_id}<br><b>Depth:</b> {depth} m",
            max_width=260,
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=5,
            color="#f97316",
            fill=True,
            fill_color="#f97316",
            fill_opacity=0.85,
            tooltip=tooltip,
            popup=popup,
        ).add_to(vop_fg)
    vop_fg.add_to(m)

    Draw(
        export=False,
        draw_options={
            "polyline": True,
            "polygon": False,
            "rectangle": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)
    MousePosition(position="bottomright", separator=" | ", prefix="Lat/Lon", num_digits=6).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    map_data = st_folium(m, width=None, height=700, returned_objects=["last_clicked", "all_drawings"])

    st.subheader("Map Tools")
    if map_data and map_data.get("last_clicked"):
        lat = float(map_data["last_clicked"]["lat"])
        lon = float(map_data["last_clicked"]["lng"])
        st.write("Last clicked coordinate")
        st.code(f"{lat:.6f}, {lon:.6f}")
        st.caption("Use the copy icon in the code block to copy lat/lon.")
    else:
        st.caption("Click anywhere on the map to capture coordinates.")

    drawings = map_data.get("all_drawings") if map_data else None
    latest_line_km = None
    if drawings:
        for feature in drawings:
            geom = feature.get("geometry", {})
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                if len(coords) >= 2:
                    latest_line_km = _line_distance_km(coords)

    if latest_line_km is not None:
        latest_line_mi = latest_line_km * 0.621371
        st.write(f"Measured line distance: **{latest_line_km:.2f} km** (**{latest_line_mi:.2f} miles**)")
    else:
        st.caption("Use the polyline tool on the map to measure distance.")


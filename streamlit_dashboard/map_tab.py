from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw, MousePosition
from google.cloud import bigquery
from streamlit_folium import st_folium


PROJECT_ID = "onda-maverick"
OFFSHORE_BUOYS_TABLE = "onda-maverick.surf_system_data.offshore_buoys"
VIRTUAL_OFFSHORE_POINTS_TABLE = "onda-maverick.surf_system_data.virtual_offshore_points"
QUERY_LIMIT = 500


@st.cache_resource(show_spinner=False)
def _bq_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


@st.cache_data(ttl=600, show_spinner=False)
def _load_map_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    client = _bq_client()

    buoys_query = f"""
    SELECT
      CAST(buoy_id AS INT64) AS id,
      CAST(name AS STRING) AS name,
      CAST(lat AS FLOAT64) AS lat,
      CAST(lon AS FLOAT64) AS lon,
      CAST(depth AS FLOAT64) AS depth
    FROM `{OFFSHORE_BUOYS_TABLE}`
    WHERE lat IS NOT NULL AND lon IS NOT NULL
    LIMIT {QUERY_LIMIT}
    """

    vop_query = f"""
    SELECT
      CAST(id AS INT64) AS id,
      CAST(lat AS FLOAT64) AS lat,
      CAST(lon AS FLOAT64) AS lon,
      CAST(depth AS FLOAT64) AS depth
    FROM `{VIRTUAL_OFFSHORE_POINTS_TABLE}`
    WHERE lat IS NOT NULL AND lon IS NOT NULL
    LIMIT {QUERY_LIMIT}
    """

    buoys = client.query(buoys_query).to_dataframe()
    vops = client.query(vop_query).to_dataframe()
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
        st.info("Make sure Application Default Credentials are available for this Streamlit runtime.")
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


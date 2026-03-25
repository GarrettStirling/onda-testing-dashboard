# ONDA Testing Dashboard (Streamlit)

Streamlit app to visualize outputs from the **`onda-backend`** pipeline: K-coefficient radar plots from local CSVs, **forecast** time series (buoy-scaled components + CDIP), and an interactive map of offshore buoys / virtual offshore points from **BigQuery**.

## Repo layout

| Path | Purpose |
|------|---------|
| `app.py` | Entrypoint: tabs **K Coefficients**, **Map**, **Forecasts** |
| `streamlit_dashboard/k_coefficients.py` | K-matrix CSV loading, polar plots, caching |
| `streamlit_dashboard/map_tab.py` | Folium map + BigQuery queries |
| `streamlit_dashboard/forecast_tab.py` | Buoy + CDIP forecast plots |
| `.streamlit/config.toml` | Dark UI theme |
| `data/k_coef/` | K-coefficient CSVs (written by backend or copied here) |
| `data/forecasts/` | `buoy_scaled_components.csv`, `cdip_data_p.csv` |
| `data/reference/breaks_with_names.csv` | Break labels (`break_id`, spot/break names) |

## Requirements

- Python 3.10+ recommended  
- Dependencies: `requirements.txt` (`streamlit`, `pandas`, `numpy`, `matplotlib`, `pytz`, `google-cloud-bigquery`, `folium`, `streamlit-folium`, `db-dtypes`)

## Run locally (recommended: venv)

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If `streamlit` is not on your PATH, always use `python -m streamlit` (or `.\.venv\Scripts\python.exe -m streamlit`).

---

## K Coefficients tab

Loads polar (“radar-style”) heatmaps per **break** (dark theme, colorbar). Direction bins follow the actual `swell_dir` sampling in each CSV; **Analytic (VOP)** and **SWAN** panels only render when that dataset has rows for the break.

### Data files (`data/k_coef/`)

| Column | File | Role in UI |
|--------|------|------------|
| 1 | `k_coefficient_matrix_analytic.csv` | **Analytic (Buoy)** |
| 2 | `k_coefficient_matrix_analytic_vop.csv` | **Analytic (VOP)** — optional until generated |
| 3 | `k_coefficient_matrix_analytic_fan.csv` | **Analytical Fan** — optional until generated |
| 4 | `k_coefficient_matrix_swan.csv` | **SWAN** |

Expected columns (same schema for all four): `break_id`, `swell_dir`, `swell_period`, `k_factor`.

### Break labels

1. **Preferred:** `data/reference/breaks_with_names.csv`  
2. **Fallback:** `../onda-backend/temp_component_testing/intermediates/breaks_with_names.csv`

Display names follow `onda-backend/DATA_MODEL.md` (spot vs break naming).

### Caching

Rendered plot PNGs are cached under `.streamlit_cache/k_coefficients/` (keyed by CSV mtimes). Use **Streamlit menu → Clear cache** if plots look stale after regenerating CSVs.

### Plot logic

Aligned with `onda-backend/scripts/plot_refraction_coefficients.py` (polar layout, K color scale). Implementation lives in `streamlit_dashboard/k_coefficients.py`.

---

## Map tab (BigQuery)

Queries:

- `onda-maverick.surf_system_data.offshore_buoys` (blue markers)  
- `onda-maverick.surf_system_data.virtual_offshore_points` (orange markers)

**Features:** depth slider, hover/popup (id / buoy name / depth where columns exist), ocean basemap, click-to-copy lat/lon, polyline measure (km + miles).

**Schemas differ** between tables (e.g. buoys use `buoy_name`, `lon`, `depth_m`; VOPs use `vop_id` STRING, `long`, `depth_m`). The app resolves columns via `INFORMATION_SCHEMA` and quotes reserved names like `` `long` ``.

### Dependencies

`db-dtypes` is required for `job.to_dataframe()` on BigQuery results.

---

## Forecasts tab

Reads local CSVs under `data/forecasts/`:

| File | Role |
|------|------|
| `buoy_scaled_components.csv` | Primary time series: pri/sec/ter **height**, **direction**, **period** (buoy-scaled columns). |
| `cdip_data_p.csv` | CDIP: **`significant_wave_height`** plus MOP **pri/sec/ter** (heights, directions, periods). |

For each selected **break** (surf spot), the tab shows **three stacked subplots** (top → bottom): **wave height (ft)**, **direction (°)**, **period (s)**. Each subplot draws three lines for primary / secondary / tertiary from the buoy-scaled file over the **full buoy time range** in the CSV.

- **Show CDIP significant wave height** (checkbox): adds the bold CDIP sig-height line on the height panel when `cdip_data_p.csv` has rows whose *aligned* times fall in the buoy window.
- **Overlay CDIP MOP** (checkbox, **off** by default): adds semi-transparent CDIP pri/sec/ter on all three panels (see `onda-backend/scripts/plot_cdip_data.py` palette).
- **CDIP time vs buoy clock**: **Auto** tries shifts in ±2 h and picks the one that best correlates buoy vs CDIP **primary wave height** (`merge_asof`, 2 h tolerance). Fixed **−2 … +2 h** overrides Auto when you know the clock offset. The same shift applies to sig height and MOP so overlays line up.

If the two CSVs use **non-overlapping date ranges**, no CDIP lines appear until both files cover the same period.

Break titles use `data/reference/breaks_with_names.csv` when present.

---

## Authentication

### Local development

The app sets `GOOGLE_CLOUD_UNIVERSE_DOMAIN=googleapis.com` in `app.py` so `google-auth` does not probe `metadata.google.internal` on your laptop.

Use **Application Default Credentials**, e.g. after:

```powershell
gcloud auth application-default login
```

Then either rely on the default ADC path, or set explicitly before starting Streamlit:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\garre\AppData\Roaming\gcloud\application_default_credentials.json"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Hosted Streamlit (Community Cloud) — for later

Your laptop ADC file **is not available** on Streamlit Cloud. The Map tab expects a **service account** supplied via **app secrets** (see below). Without secrets, BigQuery calls will fail with metadata / auth errors.

#### 1) Where to add secrets in Streamlit

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and select your deployed app.  
2. Open **App settings** (gear / **⋮** menu).  
3. Open **Secrets**.  
4. Paste the TOML block below (fill in values from your downloaded JSON).  
5. **Save**, then **Reboot** the app so `st.secrets` reloads.

The code reads `st.secrets["gcp_service_account"]` in `streamlit_dashboard/map_tab.py` and builds a BigQuery client from that.

#### 2) Where the secret values come from (Google Cloud)

When you have project access:

1. **Google Cloud Console** → **IAM & Admin** → **Service accounts**.  
2. Create (or pick) a service account for this dashboard.  
3. Grant at least:  
   - **BigQuery Job User** (run queries)  
   - **BigQuery Data Viewer** on dataset `surf_system_data` (or narrower table-level if you prefer)  
4. **Keys** → **Add key** → **JSON** → download.  
5. Copy each field from that JSON into the Secrets TOML (same keys as in the file).

#### 3) Secrets TOML template (paste into Streamlit Secrets UI)

Use the section header **`[gcp_service_account]`** and the keys from your downloaded JSON:

```toml
[gcp_service_account]
type = "service_account"
project_id = "onda-maverick"
private_key_id = "YOUR_PRIVATE_KEY_ID"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_KEY_LINES_HERE\n-----END PRIVATE KEY-----\n"
client_email = "your-sa@onda-maverick.iam.gserviceaccount.com"
client_id = "YOUR_CLIENT_ID"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
universe_domain = "googleapis.com"
```

**Security:** never commit service account JSON or Secrets content to git. For **local** testing of secrets, you can create `.streamlit/secrets.toml` with the same structure; that path is listed in `.gitignore` in this repo.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Stale K plots after CSV change | Streamlit **Clear cache**; or delete `.streamlit_cache/k_coefficients/` |
| Map auth errors locally | Confirm `gcloud auth application-default login` and optional `GOOGLE_APPLICATION_CREDENTIALS` |
| Map auth errors on Streamlit Cloud | Add `[gcp_service_account]` secrets and reboot app |
| `db-dtypes` error | `pip install db-dtypes` (already in `requirements.txt`) |
| Forecast tab empty / error | Ensure `data/forecasts/buoy_scaled_components.csv` exists; `cdip_data_p.csv` optional for sig height + MOP overlay |

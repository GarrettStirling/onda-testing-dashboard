# ONDA Testing Dashboard (Streamlit)

This repo provides a Streamlit dashboard to visualize outputs produced by the `onda-backend` repo.

## Run

From the repo root:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## K Coefficients Tab

The `K Coefficients` tab loads:

1. `data/k_coef/k_coefficient_matrix_analytic.csv`
2. `data/k_coef/k_coefficient_matrix_swan.csv`
3. Break labels from `../onda-backend/temp_component_testing/intermediates/breaks_with_names.csv`

Each selected break is displayed as:
- left: analytical
- right: swan

All plots use a dark theme and are generated from the same logic as
`onda-backend/scripts/plot_refraction_coefficients.py`.

Break labels are read from `data/reference/breaks_with_names.csv` (local reference).
If that file is missing, the dashboard falls back to
`../onda-backend/temp_component_testing/intermediates/breaks_with_names.csv`.

## Map Tab (BigQuery)

The `Map` tab queries BigQuery and renders:

- `onda-maverick.surf_system_data.offshore_buoys` (blue)
- `onda-maverick.surf_system_data.virtual_offshore_points` (orange)

Features:
- depth slider filter
- hover/click point metadata (id/name/depth)
- click-anywhere coordinate capture with easy copy
- draw polyline to measure distance (km and miles)

Authentication:
- uses Google Application Default Credentials for the Streamlit runtime.

### Local Windows (venv + ADC)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\garre\AppData\Roaming\gcloud\application_default_credentials.json"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app sets `GOOGLE_CLOUD_UNIVERSE_DOMAIN=googleapis.com` so `google-auth` does not try to reach
`metadata.google.internal` on your laptop (that causes timeouts).

If the Map tab still shows an old error after a code change: **Streamlit menu → Clear cache**, then refresh.


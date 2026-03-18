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


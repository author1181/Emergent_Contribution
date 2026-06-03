# Data dictionary and provenance

This directory holds the five real-data panels placed in the phase diagram
(Table 2 of the paper). Each is a multivariate panel time series; the pipeline
fits a neural vector-autoregression per panel and measures the departure of the
emergent-contribution node ordering from average controllability.

## Panels

### `my_data_75_years.csv` — Democracy (V-Dem)
89 countries × 75 years (1950–2024), 16 component indicators of the Varieties of
Democracy project. Columns: `country_id`, `year`, and the 16 component columns.
This panel is taken in depth as the deep-domain worked example.

**Provenance.** Derived from the Varieties of Democracy (V-Dem) dataset v15
(V-Dem Institute, CC BY 4.0).
> Coppedge, M., et al. 2025. V-Dem Dataset v15. Varieties of Democracy Project.

### `gmd_macro_core.csv` — Macro-finance
9 core macroeconomic series from the Global Macro Database.

### `rv_dataset.csv` — Realized volatility
Daily realized volatilities of 8 major global equity indices.

### `wdi_reversal_panel.csv` — Economic development
8 World Bank World Development Indicators at annual resolution.

### `beijing/` — Air quality
11 hourly pollutant and meteorological channels from multi-site monitoring in
Beijing, 2013–2017. Preprocessed by `src/beijing_prep.py`.

## Notes
- Per-panel preprocessing, standardization, and splits are applied in the
  notebook; the V-Dem split is leakage-safe (frozen pre-2000 standardization),
  implemented in `src/leakage_safe_split.py`.
- Component names are anonymization-safe (no author/institution identifiers).

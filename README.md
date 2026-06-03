# Predicting Crop Yield in Kyrgyzstan Using Climate and Remote Sensing Data

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **MSc Thesis — Data Science**  
> Dastan Karygulov · Ala-Too International University, Bishkek · June 2026  
> Supervisor: Dr. Mohammad Tauheed Khan

---

## Overview

This repository contains the complete source code for the MSc thesis:
**"Predicting Crop Yield in Kyrgyzstan Using Climate and Remote Sensing Data"**

The pipeline integrates three open-access data sources (ERA5 reanalysis, MODIS NDVI,
FAO FAOSTAT) into a national-scale machine learning framework for predicting annual
yields of winter wheat, barley, and potato across a 34-year period (1992–2025).
Eight ML models are trained and evaluated across three feature sets. A SHAP
explainability analysis quantifies individual climate feature contributions, and a
five-stage early-warning protocol translates forecasts into operational food security outputs.

### Key Results

| Feature Set | Best Model | R² | RMSE (hg/ha) | MAPE |
|-------------|------------|-----|--------------|------|
| A — Lags Only | Random Forest | **0.9978** | 335 | 7.3% |
| B — ERA5+NDVI+Lags | CatBoost | **0.9969** | 393 | 8.4% |
| C — Climate Only (no yield lags) | CatBoost | **0.9947** | 519 | 10.4% |

**Headline finding:** A climate-only model (no historical yield data) achieves R²=0.9947
— only ΔR²=−0.003 below the full model — demonstrating that ERA5 climate variables and
MODIS NDVI genuinely explain yield variation, not merely autocorrelation.

---

## Project Structure

```
kyrgyzstan-crop-yield-prediction/
│
├── config.py                    # All paths, hyperparameters, thresholds
├── run_all.py                   # Master pipeline runner
│
├── step1_data_cleaning.py       # ERA5 + MODIS + FAOSTAT cleaning & merging
├── step2_eda.py                 # Exploratory data analysis & EDA figures
├── step3_ml_pipeline.py         # 8-model training, CV, SHAP analysis
├── step4_early_warning.py       # Early-warning protocol & 2026–2028 forecasts
│
├── src/
│   └── data/
│       └── download_era5.py     # ERA5 CDS API download helper
│
├── notebooks/
│   └── 01_full_analysis.ipynb   # Interactive Jupyter walkthrough
│
├── data/
│   ├── raw/                     # Input data (see data/raw/README.md)
│   │   └── README.md
│   └── processed/               # Outputs from step1 (auto-created)
│
├── figures/                     # All thesis figures (auto-created)
├── models_saved/                # Serialised models (auto-created)
│
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/kyrgyzstan-crop-yield-prediction.git
cd kyrgyzstan-crop-yield-prediction
```

### 2. Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the raw data files

Copy the four raw data files into `data/raw/`.
See `data/raw/README.md` for download instructions.

```
data/raw/
├── era5_temperature_kyrgyzstan_1992_2024.csv
├── era5_precipitation_kyrgyzstan_1992_2024.csv
├── modis_ndvi_monthly.csv
└── FAOSTAT_combined_2026.csv
```

### 5. Run the full pipeline

```bash
python run_all.py
```

Or run individual steps:

```bash
python step1_data_cleaning.py    # ~30 seconds
python step2_eda.py              # ~20 seconds
python step3_ml_pipeline.py      # ~5–10 minutes
python step4_early_warning.py    # ~30 seconds
```

Or run a subset of steps:

```bash
python run_all.py --steps 3,4    # only ML training and forecasting
python run_all.py --steps 2-4    # steps 2, 3, and 4
```

---

## Step-by-Step Pipeline Description

### Step 1 — Data Cleaning (`step1_data_cleaning.py`)

**Input:** Four raw CSV files in `data/raw/`  
**Output:** `data/processed/kyrgyzstan_climate_national.csv`, `data/processed/kyrgyzstan_clean.csv`

What it does:
- Loads ERA5 monthly temperature (Kelvin → Celsius) and precipitation (m/day → mm/month)
- Computes national spatial means across 893 ERA5 grid points
- Aggregates to seasonal features: growing season (Apr–Sep), spring (Mar–May), winter (Dec–Feb)
- Computes growing degree days (GDD, base 5°C): `GDD = Σmax(T_month − 5, 0) × 30`
- Computes climate anomalies (deviation from 1992–2025 long-term mean)
- Aggregates MODIS NDVI to growing-season mean, maximum, and spring mean
- Merges FAOSTAT crop yields with climate features
- Engineers lagged yield features: `yield_lag1`, `yield_lag2`, `yield_roll3`
- Encodes crop dummy variables

**Feature engineering summary:**

| Feature Group | Features | Count |
|---------------|----------|-------|
| ERA5 seasonal temperature | temp_grow_c, temp_spring_c, temp_summer_c, temp_winter_c, anomalies | 8 |
| ERA5 seasonal precipitation | precip_grow_mm, precip_spring_mm, precip_winter_mm, anomalies | 6 |
| GDD + drought index | gdd, gdd_anom, drought_idx | 3 |
| MODIS NDVI | ndvi_grow, ndvi_max, ndvi_spring, anomalies | 5 |
| Lagged yield | yield_lag1, yield_lag2, yield_roll3, yield_change | 4 |
| Crop dummies + area | crop_Wheat, crop_Barley, crop_Potato, area_ha, area_change | 5 |
| Time trend | year_norm | 1 |
| **Total** | | **32** |

---

### Step 2 — EDA (`step2_eda.py`)

**Output:** Figures saved to `figures/`

Generates:
- `fig1_yield_trends.png` — crop yield trends 1992–2025
- `fig3_era5_climate_trends.png` — ERA5 temperature, precipitation, and GDD
- `fig5_correlation_heatmap.png` — per-crop Pearson correlations with p-values
- `fig8_climate_yield_anomaly.png` — climate anomaly vs yield anomaly scatter

Key findings reported:
- Growing-season precipitation dominates for barley (r=+0.650, p<0.001)
- Spring NDVI is the second-strongest predictor for wheat (r=+0.586, p<0.005)
- Temperature correlations are negative but not statistically significant at n=25

---

### Step 3 — ML Pipeline (`step3_ml_pipeline.py`)

**Output:** `data/processed/results_summary.csv`, saved models, SHAP figures

**Three feature sets:**

| Set | Description | Features |
|-----|-------------|----------|
| A | Lags Only | Lagged yields + crop dummies |
| B | Full | ERA5 + NDVI + Lags |
| C | Climate Only | ERA5 + NDVI only (no yield lags) |

**Eight models:**

| Model | Library | Notes |
|-------|---------|-------|
| Ridge | scikit-learn | Linear baseline |
| Random Forest | scikit-learn | 400 trees, max_depth=8 |
| Gradient Boosting | scikit-learn | 400 iterations, lr=0.05 |
| Extra Trees | scikit-learn | Randomised thresholds |
| XGBoost | xgboost | Regularised boosting |
| LightGBM | lightgbm | Leaf-wise growth |
| CatBoost | catboost | Ordered boosting (best on small data) |
| Voting Ensemble | scikit-learn | RF + GB + ET averaged |

**Validation strategy:**
- Temporal train/test split: 1992–2016 (train), 2017–2025 (test)
- 5-fold TimeSeriesSplit cross-validation within training set
- All preprocessing (imputation, scaling) fitted on training data only

**SHAP analysis:**
- TreeSHAP computed for the best Feature Set B model
- Mean absolute SHAP values quantify each feature's average contribution
- Dependence plots reveal non-linear precipitation threshold (~290mm)

---

### Step 4 — Early Warning & Forecasting (`step4_early_warning.py`)

**Output:** Forecast CSV, early warning status JSON, figures

**Five-stage early-warning protocol:**

| Stage | Timing | Trigger Condition |
|-------|--------|-------------------|
| 1. Monitoring | April–June | Download ERA5 + MODIS at 5-day latency |
| 2. Level 1 Alert | By June 30 | Precip anomaly < −50mm |
| 3. Level 2 Confirm | By July 15 | +temp > 0.8°C OR NDVI < 10th pct OR drought_idx < 1.0 |
| 4. Yield Forecast | August 1 | Run CatBoost Set C; issue range ± MAPE |
| 5. Policy Response | By August 5 | Mild/Moderate/Severe protocol |

**2026–2028 forecast scenarios:**

| Scenario | Temperature | Precipitation |
|----------|-------------|---------------|
| Baseline | 2023–2025 avg | 2023–2025 avg |
| Warm-Dry | +1.5°C | −15% |
| Cool-Wet | −0.5°C | +10% |

---

## Reproducing the Thesis Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place raw data in data/raw/ (see data/raw/README.md)

# 3. Run the full pipeline
python run_all.py

# 4. Check results
cat data/processed/results_summary.csv

# 5. View figures
ls figures/
```

Expected runtime on a modern laptop:

| Step | Time |
|------|------|
| Step 1 (data cleaning) | ~30 sec |
| Step 2 (EDA) | ~20 sec |
| Step 3 (ML pipeline, all models) | 5–15 min |
| Step 4 (forecasting) | ~30 sec |

---

## Data Sources

| Source | URL | Access |
|--------|-----|--------|
| ERA5 Reanalysis | https://cds.climate.copernicus.eu | Free registration |
| MODIS MOD13A3 | https://appeears.earthdatacloud.nasa.gov | Free NASA Earthdata account |
| FAO FAOSTAT | https://www.fao.org/faostat | Open access, no registration |

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@mastersthesis{karygulov2026,
  author = {Karygulov, Dastan},
  title  = {Predicting Crop Yield in Kyrgyzstan Using Climate and Remote Sensing Data},
  school = {Ala-Too International University},
  year   = {2026},
  month  = {June},
  address= {Bishkek, Kyrgyzstan},
  type   = {MSc Thesis in Data Science}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

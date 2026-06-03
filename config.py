"""
config.py
=========
Central configuration for the Kyrgyzstan Crop Yield Prediction project.
Edit DATA_DIR to point to wherever you placed the raw data files.
"""

import os

# ── Root directories ─────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(ROOT_DIR, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FIGURES_DIR   = os.path.join(ROOT_DIR, "figures")
MODELS_DIR    = os.path.join(ROOT_DIR, "models_saved")

# ── Raw data file names (place these in data/raw/) ───────────────────────────
ERA5_TEMP_FILE   = os.path.join(RAW_DIR, "era5_temperature_kyrgyzstan_1992_2024.csv")
ERA5_PRECIP_FILE = os.path.join(RAW_DIR, "era5_precipitation_kyrgyzstan_1992_2024.csv")
MODIS_FILE       = os.path.join(RAW_DIR, "modis_ndvi_monthly.csv")
FAOSTAT_FILE     = os.path.join(RAW_DIR, "FAOSTAT_combined_2026.csv")

# ── Processed output files ───────────────────────────────────────────────────
CLIMATE_NATIONAL_FILE = os.path.join(PROCESSED_DIR, "kyrgyzstan_climate_national.csv")
CLEAN_DATASET_FILE    = os.path.join(PROCESSED_DIR, "kyrgyzstan_clean.csv")

# ── Study period ─────────────────────────────────────────────────────────────
STUDY_START = 1992
STUDY_END   = 2025
TRAIN_END   = 2016   # 1992–2016 → training
TEST_START  = 2017   # 2017–2025 → held-out test

# ── Crops ────────────────────────────────────────────────────────────────────
CROPS = ["Wheat", "Barley", "Potato"]

# ── Climate windows (month numbers) ─────────────────────────────────────────
GROW_MONTHS   = list(range(4, 10))   # April–September
SPRING_MONTHS = list(range(3, 6))    # March–May
SUMMER_MONTHS = list(range(6, 9))    # June–August
WINTER_MONTHS = [12, 1, 2]           # Dec–Feb

GDD_BASE_TEMP = 5.0    # °C base for growing degree days

# ── Feature sets ─────────────────────────────────────────────────────────────
FEATURES_A = [
    "yield_lag1", "yield_lag2", "yield_roll3",
    "year_norm", "yield_change", "area_ha", "area_change",
    "crop_Barley", "crop_Potato", "crop_Wheat",
]

FEATURES_B = FEATURES_A + [
    "temp_grow_c", "temp_spring_c", "temp_summer_c", "temp_winter_c",
    "temp_grow_c_anom", "temp_spring_c_anom",
    "precip_grow_mm", "precip_spring_mm", "precip_winter_mm",
    "precip_grow_mm_anom", "precip_spring_mm_anom",
    "gdd", "gdd_anom", "drought_idx",
    "ndvi_grow", "ndvi_max", "ndvi_spring",
    "ndvi_grow_anom", "ndvi_spring_anom",
]

FEATURES_C = [
    "temp_grow_c", "temp_spring_c", "temp_summer_c", "temp_winter_c",
    "temp_grow_c_anom", "temp_spring_c_anom",
    "precip_grow_mm", "precip_spring_mm", "precip_winter_mm",
    "precip_grow_mm_anom", "precip_spring_mm_anom",
    "gdd", "gdd_anom", "drought_idx",
    "ndvi_grow", "ndvi_max", "ndvi_spring",
    "ndvi_grow_anom", "ndvi_spring_anom",
    "area_ha", "year_norm",
    "crop_Barley", "crop_Potato", "crop_Wheat",
]

TARGET = "yield_hg_ha"

# ── Model hyperparameters ────────────────────────────────────────────────────
RANDOM_SEED = 42
CV_FOLDS    = 5

MODEL_PARAMS = {
    "RandomForest": dict(
        n_estimators=400, max_depth=8, min_samples_leaf=2,
        max_features=0.6, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "GradientBoosting": dict(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=RANDOM_SEED
    ),
    "ExtraTrees": dict(
        n_estimators=400, max_depth=8, min_samples_leaf=2,
        max_features=0.6, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "XGBoost": dict(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
        random_state=RANDOM_SEED, n_jobs=-1
    ),
    "LightGBM": dict(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
        min_child_samples=5, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "CatBoost": dict(
        iterations=400, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0, subsample=0.8, random_seed=RANDOM_SEED,
        verbose=0
    ),
}

# ── Early-warning thresholds ─────────────────────────────────────────────────
EW_PRECIP_ANOMALY_THRESHOLD = -50.0   # mm below long-term mean
EW_TEMP_ANOMALY_THRESHOLD   =  0.8    # °C above long-term mean
EW_NDVI_PERCENTILE          = 10      # NDVI below 10th pct → alert
EW_DROUGHT_IDX_THRESHOLD    =  1.0    # drought_idx < 1 → confirm

# ── Forecast scenarios ───────────────────────────────────────────────────────
SCENARIOS = {
    "Baseline":  {"temp_delta": 0.0,  "precip_scale": 1.00},
    "Warm-Dry":  {"temp_delta": 1.5,  "precip_scale": 0.85},
    "Cool-Wet":  {"temp_delta": -0.5, "precip_scale": 1.10},
}
FORECAST_YEARS = [2026, 2027, 2028]

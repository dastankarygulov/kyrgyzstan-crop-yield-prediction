"""
step1_data_cleaning.py
======================
Reads raw ERA5, MODIS NDVI, and FAOSTAT files.
Cleans, converts units, and aggregates ERA5 to
national monthly means. Outputs:
  - data/processed/kyrgyzstan_climate_national.csv
  - data/processed/kyrgyzstan_clean.csv  (merged crop + climate)

Run:
    python step1_data_cleaning.py
"""

import os
import calendar
import warnings
import pandas as pd
import numpy as np
from config import (
    ERA5_TEMP_FILE, ERA5_PRECIP_FILE, MODIS_FILE, FAOSTAT_FILE,
    CLIMATE_NATIONAL_FILE, CLEAN_DATASET_FILE,
    PROCESSED_DIR, CROPS,
    GROW_MONTHS, SPRING_MONTHS, SUMMER_MONTHS, WINTER_MONTHS,
    GDD_BASE_TEMP, STUDY_START, STUDY_END,
)

warnings.filterwarnings("ignore")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── 1. ERA5 Temperature ───────────────────────────────────────────────────────
def load_era5_temperature(path):
    print(f"  Loading ERA5 temperature: {path}")
    df = pd.read_csv(path, parse_dates=["valid_time"])
    df["year"]  = df["valid_time"].dt.year
    df["month"] = df["valid_time"].dt.month
    # National spatial mean per month
    monthly = df.groupby(["year", "month"])["t2m"].mean().reset_index()
    monthly["temp_c"] = monthly["t2m"] - 273.15
    return monthly[["year", "month", "temp_c"]]


# ── 2. ERA5 Precipitation ────────────────────────────────────────────────────
def load_era5_precipitation(path):
    print(f"  Loading ERA5 precipitation: {path}")
    df = pd.read_csv(path, parse_dates=["valid_time"])
    df["year"]  = df["valid_time"].dt.year
    df["month"] = df["valid_time"].dt.month
    monthly = df.groupby(["year", "month"])["tp"].mean().reset_index()
    # tp is in m/day → convert to mm/month
    monthly["days"] = monthly.apply(
        lambda r: calendar.monthrange(int(r.year), int(r.month))[1], axis=1
    )
    monthly["precip_mm"] = monthly["tp"] * 1000 * monthly["days"]
    return monthly[["year", "month", "precip_mm"]]


# ── 3. Aggregate monthly climate to annual seasonal features ─────────────────
def build_annual_climate(temp_df, precip_df):
    print("  Building annual climate features …")
    merged = temp_df.merge(precip_df, on=["year", "month"])
    rows = []

    for year, grp in merged.groupby("year"):
        def seasonal_temp(months):
            sub = grp[grp["month"].isin(months)]["temp_c"]
            return sub.mean() if len(sub) else np.nan

        def seasonal_precip(months):
            sub = grp[grp["month"].isin(months)]["precip_mm"]
            return sub.sum() if len(sub) else np.nan

        grow_temps = grp[grp["month"].isin(GROW_MONTHS)]["temp_c"].values
        gdd = float(np.sum(np.maximum(grow_temps - GDD_BASE_TEMP, 0.0) * 30))

        rows.append({
            "year":              year,
            "temp_annual_c":     grp["temp_c"].mean(),
            "temp_grow_c":       seasonal_temp(GROW_MONTHS),
            "temp_spring_c":     seasonal_temp(SPRING_MONTHS),
            "temp_summer_c":     seasonal_temp(SUMMER_MONTHS),
            "temp_winter_c":     seasonal_temp(WINTER_MONTHS),
            "precip_annual_mm":  grp["precip_mm"].sum(),
            "precip_grow_mm":    seasonal_precip(GROW_MONTHS),
            "precip_spring_mm":  seasonal_precip(SPRING_MONTHS),
            "precip_winter_mm":  seasonal_precip(WINTER_MONTHS),
            "gdd":               gdd,
        })

    climate = pd.DataFrame(rows).query(f"{STUDY_START} <= year <= {STUDY_END}").reset_index(drop=True)

    # Drought index = precip / temp ratio (normalised)
    climate["drought_idx"] = (
        climate["precip_grow_mm"] / climate["precip_grow_mm"].mean()
    ) / (
        climate["temp_grow_c"] / climate["temp_grow_c"].mean()
    )

    # Anomaly features
    for col in ["precip_grow_mm", "precip_spring_mm", "precip_annual_mm",
                "temp_grow_c", "temp_spring_c", "gdd"]:
        climate[f"{col}_anom"] = climate[col] - climate[col].mean()

    # Additional derived metrics
    climate["temp_tminmax_grow"] = climate["temp_summer_c"] - climate["temp_winter_c"]
    climate["heat_stress_months"] = (climate["temp_summer_c"] > 28).astype(int)

    return climate


# ── 4. MODIS NDVI ────────────────────────────────────────────────────────────
def load_ndvi(path):
    print(f"  Loading MODIS NDVI: {path}")
    df = pd.read_csv(path)
    annual = []
    for year, grp in df.groupby("year"):
        grow   = grp[grp["month"].isin(GROW_MONTHS)]["ndvi_mean"]
        spring = grp[grp["month"].isin(SPRING_MONTHS)]["ndvi_mean"]
        annual.append({
            "year":       year,
            "ndvi_grow":  grow.mean()   if len(grow)   else np.nan,
            "ndvi_max":   grow.max()    if len(grow)   else np.nan,
            "ndvi_spring":spring.mean() if len(spring) else np.nan,
        })
    ndvi = pd.DataFrame(annual)
    for col in ["ndvi_grow", "ndvi_spring"]:
        ndvi[f"{col}_anom"] = ndvi[col] - ndvi[col].mean()
    return ndvi


# ── 5. FAOSTAT crop yields ────────────────────────────────────────────────────
def load_faostat(path):
    print(f"  Loading FAOSTAT: {path}")
    df = pd.read_csv(path)
    yields = (
        df[df["Element"] == "Yield"][["Item", "Year", "Value"]]
        .rename(columns={"Item": "crop", "Year": "year", "Value": "yield_hg_ha"})
    )
    area = (
        df[df["Element"] == "Area harvested"][["Item", "Year", "Value"]]
        .rename(columns={"Item": "crop", "Year": "year", "Value": "area_ha"})
    )
    prod = (
        df[df["Element"] == "Production"][["Item", "Year", "Value"]]
        .rename(columns={"Item": "crop", "Year": "year", "Value": "production_t"})
    )

    crops_df = yields.merge(area, on=["crop", "year"], how="left")
    crops_df = crops_df.merge(prod, on=["crop", "year"], how="left")

    # Standardise crop names
    crops_df["crop"] = crops_df["crop"].str.replace("Potatoes", "Potato")
    crops_df = crops_df[crops_df["crop"].isin(CROPS)]
    crops_df = crops_df.query(f"{STUDY_START} <= year <= {STUDY_END}").reset_index(drop=True)
    return crops_df


# ── 6. Merge into final clean dataset ─────────────────────────────────────────
def build_clean_dataset(crops_df, climate_df, ndvi_df):
    print("  Merging all datasets …")
    df = crops_df.merge(climate_df, on="year", how="left")

    # NDVI available 2000+; impute with median for 1992-1999
    df = df.merge(ndvi_df, on="year", how="left")
    ndvi_cols = [c for c in df.columns if c.startswith("ndvi")]
    for col in ndvi_cols:
        df[col] = df[col].fillna(df[col].median())

    # Encode crop dummies
    for crop in CROPS:
        df[f"crop_{crop}"] = (df["crop"] == crop).astype(int)

    # Lagged yield features (within each crop)
    df = df.sort_values(["crop", "year"]).reset_index(drop=True)
    df["yield_lag1"]  = df.groupby("crop")["yield_hg_ha"].shift(1)
    df["yield_lag2"]  = df.groupby("crop")["yield_hg_ha"].shift(2)
    df["yield_roll3"] = df.groupby("crop")["yield_hg_ha"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    df["yield_change"] = df.groupby("crop")["yield_hg_ha"].pct_change() * 100
    df["area_change"]  = df.groupby("crop")["area_ha"].pct_change() * 100

    # Normalised time trend
    df["year_norm"] = (df["year"] - STUDY_START) / (STUDY_END - STUDY_START)

    # Additional features
    df["log_yield"]    = np.log(df["yield_hg_ha"])
    df["prod_per_ha"]  = df["production_t"] / df["area_ha"]

    df = df.dropna(subset=["yield_lag1"]).reset_index(drop=True)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Step 1: Data Cleaning & Preprocessing ===\n")

    print("[1/4] Processing ERA5 climate data …")
    temp_df   = load_era5_temperature(ERA5_TEMP_FILE)
    precip_df = load_era5_precipitation(ERA5_PRECIP_FILE)
    climate   = build_annual_climate(temp_df, precip_df)
    climate.to_csv(CLIMATE_NATIONAL_FILE, index=False)
    print(f"  Saved: {CLIMATE_NATIONAL_FILE} ({len(climate)} rows)")

    print("\n[2/4] Processing MODIS NDVI …")
    ndvi_df = load_ndvi(MODIS_FILE)

    print("\n[3/4] Processing FAOSTAT crop statistics …")
    crops_df = load_faostat(FAOSTAT_FILE)

    print("\n[4/4] Merging into clean dataset …")
    clean = build_clean_dataset(crops_df, climate, ndvi_df)
    clean.to_csv(CLEAN_DATASET_FILE, index=False)
    print(f"  Saved: {CLEAN_DATASET_FILE}")
    print(f"  Shape: {clean.shape[0]} rows × {clean.shape[1]} columns")
    print(f"  Crops: {clean['crop'].unique().tolist()}")
    print(f"  Years: {clean['year'].min()} – {clean['year'].max()}")
    print(f"  Missing values: {clean.isnull().sum().sum()}")
    print("\nStep 1 complete.\n")


if __name__ == "__main__":
    main()

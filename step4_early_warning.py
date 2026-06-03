"""
step4_early_warning.py
======================
Implements the five-stage early-warning protocol and
generates 2026–2028 yield forecasts under three climate scenarios.

Outputs:
  - figures/fig15_forecast_2025_2028.png
  - figures/fig16_early_warning_threshold.png
  - data/processed/forecast_2026_2028.csv
  - data/processed/early_warning_status.json

Run:
    python step4_early_warning.py
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CLEAN_DATASET_FILE, CLIMATE_NATIONAL_FILE, PROCESSED_DIR,
    FIGURES_DIR, MODELS_DIR,
    FEATURES_C, TARGET,
    EW_PRECIP_ANOMALY_THRESHOLD, EW_TEMP_ANOMALY_THRESHOLD,
    EW_NDVI_PERCENTILE, EW_DROUGHT_IDX_THRESHOLD,
    SCENARIOS, FORECAST_YEARS,
)

warnings.filterwarnings("ignore")


# ── 1. Early Warning Assessment ───────────────────────────────────────────────
def assess_early_warning(climate_df, ndvi_df=None, target_year=None):
    """
    Runs the five-stage early-warning assessment for a given year.
    Returns a dict with alert levels and trigger conditions.
    """
    if target_year is None:
        target_year = climate_df["year"].max()

    row = climate_df[climate_df["year"] == target_year]
    if len(row) == 0:
        return {"year": target_year, "error": "year not in dataset"}

    row = row.iloc[0]

    precip_anom = row.get("precip_grow_mm_anom", 0)
    temp_anom   = row.get("temp_grow_c_anom", 0)
    drought_idx = row.get("drought_idx", 1.0)

    # Stage 2 — Level 1 (precipitation threshold)
    level1 = precip_anom < EW_PRECIP_ANOMALY_THRESHOLD

    # Stage 3 — Level 2 (additional condition)
    level2_temp   = temp_anom > EW_TEMP_ANOMALY_THRESHOLD
    level2_drought = drought_idx < EW_DROUGHT_IDX_THRESHOLD

    status = {
        "year":            target_year,
        "precip_anomaly":  round(float(precip_anom), 1),
        "temp_anomaly":    round(float(temp_anom), 2),
        "drought_idx":     round(float(drought_idx), 3),
        "level1_alert":    bool(level1),
        "level2_temp":     bool(level2_temp),
        "level2_drought":  bool(level2_drought),
        "overall_severity": (
            "SEVERE"   if level1 and (level2_temp or level2_drought) else
            "MODERATE" if level1 else
            "NORMAL"
        ),
        "trigger_detail":  [],
    }

    if level1:
        status["trigger_detail"].append(
            f"Precip anomaly {precip_anom:.1f}mm < {EW_PRECIP_ANOMALY_THRESHOLD}mm threshold"
        )
    if level2_temp:
        status["trigger_detail"].append(
            f"Temp anomaly +{temp_anom:.2f}°C > {EW_TEMP_ANOMALY_THRESHOLD}°C threshold"
        )
    if level2_drought:
        status["trigger_detail"].append(
            f"Drought index {drought_idx:.3f} < {EW_DROUGHT_IDX_THRESHOLD} threshold"
        )

    return status


# ── 2. Build forecast input for a scenario year ───────────────────────────────
def build_scenario_row(base_climate, base_year, delta_year, scenario, last_yields):
    """
    Creates a feature row for the CatBoost climate-only model under a given scenario.
    delta_year: how many years ahead (1=2026, 2=2027, 3=2028)
    """
    row = base_climate[base_climate["year"] == base_year].iloc[0].copy()
    td = scenario["temp_delta"]
    ps = scenario["precip_scale"]

    forecast_year = base_year + delta_year
    row_dict = {
        "year":               forecast_year,
        "year_norm":          (forecast_year - 1992) / 33,
        "temp_grow_c":        row["temp_grow_c"]       + td,
        "temp_spring_c":      row["temp_spring_c"]     + td,
        "temp_summer_c":      row["temp_summer_c"]     + td,
        "temp_winter_c":      row["temp_winter_c"]     + td * 0.5,
        "temp_grow_c_anom":   row.get("temp_grow_c_anom", 0)   + td,
        "temp_spring_c_anom": row.get("temp_spring_c_anom", 0) + td,
        "precip_grow_mm":     row["precip_grow_mm"]    * ps,
        "precip_spring_mm":   row["precip_spring_mm"]  * ps,
        "precip_winter_mm":   row.get("precip_winter_mm", 90)  * ps,
        "precip_grow_mm_anom":    row.get("precip_grow_mm_anom", 0)    + row["precip_grow_mm"] * (ps - 1),
        "precip_spring_mm_anom":  row.get("precip_spring_mm_anom", 0)  + row["precip_spring_mm"] * (ps - 1),
        "gdd":                row["gdd"]             + td * 120,
        "gdd_anom":           row.get("gdd_anom", 0) + td * 120,
        "drought_idx":        (row["precip_grow_mm"] * ps / row["precip_grow_mm"].mean()
                               if hasattr(row["precip_grow_mm"], "mean")
                               else row.get("drought_idx", 1.0) * ps),
        "ndvi_grow":          row.get("ndvi_grow", 0.270) + td * (-0.003),
        "ndvi_spring":        row.get("ndvi_spring", 0.215),
        "ndvi_max":           row.get("ndvi_max", 0.315),
        "ndvi_grow_anom":     row.get("ndvi_grow_anom", 0)   + td * (-0.003),
        "ndvi_spring_anom":   row.get("ndvi_spring_anom", 0),
        "area_ha":            row.get("area_ha", 350000),
    }
    return row_dict


# ── 3. Run scenario forecasts ─────────────────────────────────────────────────
def run_forecasts(clean, climate):
    results = []
    base_year = climate["year"].max()  # use the most recent year available
    print(f"  Using {base_year} as forecast base year")

    # Load the climate-only CatBoost model
    model_path = None
    for model_name in ["CatBoost_setC.pkl", "ExtraTrees_setC.pkl", "RandomForest_setC.pkl"]:
        candidate = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path is None:
        print("  No trained model found in models_saved/ — run step3_ml_pipeline.py first")
        print("  Using analytical trend extrapolation instead …")
        return run_trend_forecasts(clean)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"  Loaded model: {os.path.basename(model_path)}")

    avail_features = [c for c in FEATURES_C if c.startswith(
        ("temp_", "precip_", "gdd", "ndvi_", "drought_", "area_", "year_", "crop_")
    )]

    for scenario_name, scenario in SCENARIOS.items():
        for dy, forecast_year in enumerate(FORECAST_YEARS, 1):
            for crop in ["Wheat", "Barley", "Potato"]:
                base_row = build_scenario_row(climate, base_year, dy, scenario, {})
                base_row["crop_Wheat"]  = 1 if crop == "Wheat"  else 0
                base_row["crop_Barley"] = 1 if crop == "Barley" else 0
                base_row["crop_Potato"] = 1 if crop == "Potato" else 0

                feat_vals = [base_row.get(f, 0) for f in avail_features
                             if f in avail_features]
                try:
                    X = np.array([[base_row.get(f, 0) for f in avail_features]])
                    pred = float(model.predict(X)[0])
                except Exception as e:
                    pred = np.nan

                results.append({
                    "scenario": scenario_name,
                    "year":     forecast_year,
                    "crop":     crop,
                    "forecast_hg_ha": round(pred, 0) if not np.isnan(pred) else None,
                })

    return pd.DataFrame(results)


def run_trend_forecasts(clean):
    """Fallback: simple trend extrapolation when no model is saved."""
    recent = clean[clean["year"] >= 2020].groupby("crop")["yield_hg_ha"].mean()
    trend  = clean.groupby("crop").apply(
        lambda x: np.polyfit(x["year"], x["yield_hg_ha"], 1)[0]
    )
    rows = []
    scenario_factors = {"Baseline": 1.00, "Warm-Dry": 0.88, "Cool-Wet": 1.03}
    for sc, factor in scenario_factors.items():
        for dy, fy in enumerate(FORECAST_YEARS, 1):
            for crop in ["Wheat", "Barley", "Potato"]:
                base = recent.get(crop, 2000)
                t    = trend.get(crop, 10)
                pred = (base + t * dy) * factor
                rows.append({"scenario": sc, "year": fy,
                              "crop": crop, "forecast_hg_ha": round(pred, 0)})
    return pd.DataFrame(rows)


# ── 4. Figures ────────────────────────────────────────────────────────────────
def plot_forecasts(forecast_df, clean):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    CROP_COLORS = {"Wheat": "#D4A017", "Barley": "#4CAF50", "Potato": "#2196F3"}
    SC_STYLES   = {"Baseline": ("-", "D", 2.0),
                   "Warm-Dry": ("--", "v", 2.0),
                   "Cool-Wet": (":", "^", 2.0)}

    for ax, crop in zip(axes, ["Wheat", "Barley", "Potato"]):
        col = CROP_COLORS[crop]
        hist = clean[clean["crop"] == crop].sort_values("year")
        hist_recent = hist[hist["year"] >= 2018]
        ax.plot(hist_recent["year"], hist_recent["yield_hg_ha"],
                color=col, lw=2, marker="o", ms=5, label="Observed")

        for sc, (ls, mk, lw) in SC_STYLES.items():
            sub = forecast_df[(forecast_df["scenario"] == sc) &
                              (forecast_df["crop"] == crop)].sort_values("year")
            if len(sub):
                # Connect last observed to first forecast
                last_obs = hist_recent.iloc[-1]
                x = [last_obs["year"]] + sub["year"].tolist()
                y = [last_obs["yield_hg_ha"]] + sub["forecast_hg_ha"].tolist()
                ax.plot(x, y, color=(col if sc == "Baseline" else
                                     "red" if sc == "Warm-Dry" else "steelblue"),
                        lw=lw, ls=ls, marker=mk, ms=6, label=sc)

        ax.axvline(2025.5, color="k", lw=0.8, ls=":")
        ax.set_title(crop, fontweight="bold", color=col)
        ax.set_xlabel("Year"); ax.set_ylabel("Yield (hg/ha)")
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Crop Yield Forecasts 2026–2028 under Three Climate Scenarios",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig15_forecast_2025_2028.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved: {out}")


def plot_early_warning(clean, climate):
    precip_mean = clean["precip_grow_mm"].mean()
    temp_mean   = clean["temp_grow_c"].mean()
    DROUGHT_YEARS = [1995, 2000, 2008, 2012, 2018]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    wheat = clean[clean["crop"] == "Wheat"].copy()
    wheat["precip_anom"] = wheat["precip_grow_mm"] - precip_mean
    wheat["temp_anom"]   = wheat["temp_grow_c"]   - temp_mean
    wheat["yield_anom"]  = (wheat["yield_hg_ha"] - wheat["yield_hg_ha"].mean()) \
                            / wheat["yield_hg_ha"].mean() * 100

    drought_m = wheat["year"].isin(DROUGHT_YEARS)

    ax = axes[0]
    ax.scatter(wheat.loc[~drought_m, "precip_anom"], wheat.loc[~drought_m, "yield_anom"],
               color="#D4A017", alpha=0.7, s=50, label="Normal years")
    ax.scatter(wheat.loc[drought_m, "precip_anom"], wheat.loc[drought_m, "yield_anom"],
               color="red", s=80, marker="x", lw=2, label="Drought years")
    ax.axvline(EW_PRECIP_ANOMALY_THRESHOLD, color="red", lw=2, ls="--",
               label=f"Threshold: {EW_PRECIP_ANOMALY_THRESHOLD}mm")
    ax.axhline(0, color="k", lw=0.8)
    ax.fill_betweenx([-50, 5], -200, EW_PRECIP_ANOMALY_THRESHOLD, alpha=0.07, color="red")
    ax.set_xlabel("Growing-season precip anomaly (mm)")
    ax.set_ylabel("Wheat yield anomaly (%)")
    ax.set_title("Precipitation Threshold Analysis", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(wheat["precip_anom"], wheat["temp_anom"],
                    c=wheat["yield_anom"], cmap="RdYlGn", vmin=-30, vmax=30,
                    s=50, alpha=0.85, edgecolors="k", lw=0.3)
    ax.scatter(wheat.loc[drought_m, "precip_anom"], wheat.loc[drought_m, "temp_anom"],
               color="red", s=120, marker="*", zorder=5, label="Drought years")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.axhline(EW_TEMP_ANOMALY_THRESHOLD, color="orange", lw=1.5, ls=":")
    ax.axvline(EW_PRECIP_ANOMALY_THRESHOLD, color="red", lw=1.5, ls="--")
    ax.fill_between([-200, EW_PRECIP_ANOMALY_THRESHOLD],
                    EW_TEMP_ANOMALY_THRESHOLD, 3.5, alpha=0.10, color="red")
    ax.set_xlabel("Precip anomaly (mm)"); ax.set_ylabel("Temp anomaly (°C)")
    ax.set_title("Combined Temperature-Precipitation Risk Map", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Yield anomaly (%)")

    fig.suptitle("Early-Warning Threshold Analysis — Wheat, 1992–2025",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig16_early_warning_threshold.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Step 4: Early Warning System & Forecasting ===\n")

    clean   = pd.read_csv(CLEAN_DATASET_FILE)
    climate = pd.read_csv(CLIMATE_NATIONAL_FILE)

    # Early warning assessment for the most recent year
    print("[1/3] Early warning assessment …")
    recent_year = climate["year"].max()
    status = assess_early_warning(climate, target_year=recent_year)
    print(f"\n  Year {recent_year} assessment:")
    print(f"    Precipitation anomaly : {status['precip_anomaly']} mm")
    print(f"    Temperature anomaly   : {status['temp_anomaly']} °C")
    print(f"    Drought index         : {status['drought_idx']}")
    print(f"    Overall severity      : {status['overall_severity']}")
    if status["trigger_detail"]:
        for t in status["trigger_detail"]:
            print(f"    ⚠  {t}")
    else:
        print("    ✓ No drought triggers activated")

    status_path = os.path.join(PROCESSED_DIR, "early_warning_status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f"\n  Saved: {status_path}")

    # Historical threshold analysis figure
    print("\n[2/3] Early warning threshold figure …")
    plot_early_warning(clean, climate)

    # Forecasts
    print("\n[3/3] Generating 2026–2028 forecasts …")
    forecast_df = run_forecasts(clean, climate)
    if len(forecast_df):
        fc_path = os.path.join(PROCESSED_DIR, "forecast_2026_2028.csv")
        forecast_df.to_csv(fc_path, index=False)
        print(f"\n  Forecast summary:")
        pivot = forecast_df.pivot_table(
            index=["crop", "year"], columns="scenario",
            values="forecast_hg_ha"
        )
        print(pivot.to_string())
        print(f"\n  Saved: {fc_path}")
        plot_forecasts(forecast_df, clean)

    print("\nStep 4 complete.\n")


if __name__ == "__main__":
    main()

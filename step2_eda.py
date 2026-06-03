"""
step2_eda.py
============
Exploratory Data Analysis.
Reads the clean dataset and generates all EDA figures:
  - Yield trends 1992–2025 (with 2025 highlighted)
  - Decade comparison
  - ERA5 climate trends
  - MODIS NDVI trends and anomalies
  - Per-crop Pearson correlations with p-values
  - NDVI vs yield scatter
  - Climate anomaly maps

Run:
    python step2_eda.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

from config import (
    CLEAN_DATASET_FILE, CLIMATE_NATIONAL_FILE, FIGURES_DIR,
    CROPS, GROW_MONTHS, SPRING_MONTHS, STUDY_START, STUDY_END,
)

warnings.filterwarnings("ignore")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "wheat":  "#D4A017",
    "barley": "#4CAF50",
    "potato": "#2196F3",
    "era5":   "#E53935",
    "ndvi":   "#43A047",
    "blue":   "#1565C0",
    "grey":   "#757575",
    "orange": "#EF6C00",
}
DROUGHT_YEARS = [1995, 2000, 2008, 2012, 2018]


def load_data():
    clean   = pd.read_csv(CLEAN_DATASET_FILE)
    climate = pd.read_csv(CLIMATE_NATIONAL_FILE)
    return clean, climate


# ── Figure 1: Yield trends ─────────────────────────────────────────────────
def plot_yield_trends(clean):
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    for crop, col, ax, marker in [
        ("Wheat",  C["wheat"],  ax1, "o"),
        ("Barley", C["barley"], ax1, "s"),
        ("Potato", C["potato"], ax2, "^"),
    ]:
        sub = clean[clean["crop"] == crop].sort_values("year")
        (ax1 if crop != "Potato" else ax2).plot(
            sub["year"], sub["yield_hg_ha"],
            color=col, lw=2, marker=marker, ms=4, label=crop
        )
        # Highlight 2025
        row2025 = sub[sub["year"] == 2025]
        if len(row2025):
            (ax1 if crop != "Potato" else ax2).scatter(
                row2025["year"], row2025["yield_hg_ha"],
                color=col, marker="D", s=80, zorder=6
            )

    for dy in DROUGHT_YEARS:
        ax1.axvline(dy, color="red", lw=0.8, ls="--", alpha=0.55)

    ax1.set_xlabel("Year", fontsize=11)
    ax1.set_ylabel("Yield (hg/ha) — Cereals", fontsize=11)
    ax2.set_ylabel("Yield (hg/ha) — Potato", fontsize=11, color=C["potato"])
    ax1.set_xlim(STUDY_START - 1, STUDY_END + 1)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=9)
    ax1.set_title("Crop Yield Trends, Kyrgyzstan 1992–2025", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig1_yield_trends.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 3: ERA5 climate trends ─────────────────────────────────────────────
def plot_era5_trends(climate):
    yrs = climate["year"].values
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(yrs, climate["temp_annual_c"], color=C["era5"], lw=1.8)
    m, b = np.polyfit(yrs, climate["temp_annual_c"], 1)
    ax.plot(yrs, m * yrs + b, "--", color=C["grey"], lw=1.2,
            label=f"Trend +{m*10:.2f}°C/decade")
    ax.set_title("Annual Mean Temperature", fontweight="bold")
    ax.set_ylabel("°C"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(yrs, climate["temp_grow_c"], color=C["era5"], lw=1.8)
    for dy in DROUGHT_YEARS:
        ax.axvline(dy, color="orange", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(climate["temp_grow_c"].mean(), color=C["grey"], lw=1, ls=":")
    ax.set_title("Growing-Season Temperature (Apr–Sep)", fontweight="bold")
    ax.set_ylabel("°C"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.bar(yrs, climate["precip_annual_mm"], color=C["blue"], alpha=0.7, width=0.8)
    ax.axhline(climate["precip_annual_mm"].mean(), color=C["era5"], lw=1.5, ls="--",
               label=f'Mean {climate["precip_annual_mm"].mean():.0f} mm')
    ax.set_title("Annual Precipitation", fontweight="bold")
    ax.set_ylabel("mm"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.plot(yrs, climate["gdd"], color="#388E3C", lw=1.8)
    m2, b2 = np.polyfit(yrs, climate["gdd"], 1)
    ax.plot(yrs, m2 * yrs + b2, "--", color=C["grey"], lw=1.2,
            label=f"Trend +{m2*10:.1f} GDD/decade")
    ax.set_title("Growing Degree Days (base 5°C, Apr–Sep)", fontweight="bold")
    ax.set_ylabel("°C·days"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Year")
        ax.set_xlim(STUDY_START - 1, STUDY_END + 1)

    fig.suptitle("ERA5 National Climate Trends, Kyrgyzstan 1992–2025",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig3_era5_climate_trends.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved: {out}")


# ── Figure 5: Correlation heatmap with p-values ────────────────────────────
def plot_correlations(clean):
    features = {
        "precip_grow_mm":  "Grow-season precip",
        "ndvi_grow":       "NDVI grow-season",
        "precip_spring_mm":"Spring precip",
        "drought_idx":     "Drought index",
        "temp_grow_c":     "Grow-season temp",
        "ndvi_spring":     "NDVI spring",
    }
    crop_colors = {"Wheat": C["wheat"], "Barley": C["barley"], "Potato": C["potato"]}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Store results for printing
    results = {}
    for ax, crop in zip(axes, CROPS):
        sub = clean[clean["crop"] == crop].dropna(subset=list(features.keys()))
        rs, ps, labs = [], [], []
        for col, lab in features.items():
            if col in sub.columns and len(sub) > 5:
                r, p = scipy_stats.pearsonr(sub[col], sub["yield_hg_ha"])
                rs.append(r); ps.append(p); labs.append(lab)

        rs = np.array(rs)
        bar_colors = [crop_colors[crop] if r >= 0 else "#E53935" for r in rs]
        y_pos = range(len(labs))
        ax.barh(y_pos, rs, color=bar_colors, alpha=0.85, edgecolor="white")

        for i, (r, p, lab) in enumerate(zip(rs, ps, labs)):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            x_txt = r + 0.02 if r >= 0 else r - 0.02
            ha_txt = "left" if r >= 0 else "right"
            ax.text(x_txt, i, sig, va="center", ha=ha_txt, fontsize=8, fontweight="bold")

        ax.set_yticks(y_pos); ax.set_yticklabels(labs, fontsize=9)
        ax.set_xlim(-0.85, 0.85)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(crop, fontsize=12, fontweight="bold", color=crop_colors[crop])
        ax.set_xlabel("Pearson r"); ax.grid(axis="x", alpha=0.3)
        results[crop] = list(zip(labs, rs.tolist(), ps))

    fig.suptitle("Per-Crop Climate–Yield Correlations (2000–2024, n=25)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig5_correlation_heatmap.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved: {out}")

    # Print significance table
    print("\n  Pearson correlation significance (p-values):")
    for crop, vals in results.items():
        print(f"\n  {crop}:")
        for lab, r, p in sorted(vals, key=lambda x: abs(x[1]), reverse=True):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    {lab:<25} r={r:+.3f}  p={p:.4f}  {sig}")


# ── Figure 8: Climate anomaly vs yield anomaly ─────────────────────────────
def plot_anomaly_map(clean, climate):
    precip_mean = climate["precip_grow_mm"].mean()
    temp_mean   = climate["temp_grow_c"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    crop_colors = {"Wheat": C["wheat"], "Barley": C["barley"], "Potato": C["potato"]}

    for ax, crop in zip(axes, CROPS):
        sub = clean[clean["crop"] == crop].merge(
            climate[["year", "precip_grow_mm", "temp_grow_c"]], on="year", how="left"
        ).dropna()

        sub["precip_anom"] = sub["precip_grow_mm"] - precip_mean
        sub["temp_anom"]   = sub["temp_grow_c"] - temp_mean
        sub["yield_anom"]  = (sub["yield_hg_ha"] - sub["yield_hg_ha"].mean()) \
                              / sub["yield_hg_ha"].mean() * 100

        sc = ax.scatter(sub["precip_anom"], sub["yield_anom"],
                        c=sub["temp_anom"], cmap="RdBu_r", vmin=-1.5, vmax=1.5,
                        s=45, alpha=0.85, edgecolors="k", lw=0.3)

        for _, row in sub[sub["year"].isin(DROUGHT_YEARS)].iterrows():
            ax.annotate(str(int(row["year"])),
                        (row["precip_anom"], row["yield_anom"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points", color="red")

        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Precip anomaly (mm)"); ax.set_ylabel("Yield anomaly (%)")
        ax.set_title(crop, fontweight="bold", color=crop_colors[crop])
        ax.grid(alpha=0.25)

    plt.colorbar(sc, ax=axes[-1], label="Temp anomaly (°C)")
    fig.suptitle("Climate Anomalies vs. Yield Anomalies, 1992–2025",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig8_climate_yield_anomaly.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Step 2: Exploratory Data Analysis ===\n")
    clean, climate = load_data()
    print(f"  Dataset loaded: {clean.shape[0]} rows × {clean.shape[1]} cols")

    print("\n[1/4] Yield trends …")
    plot_yield_trends(clean)

    print("\n[2/4] ERA5 climate trends …")
    plot_era5_trends(climate)

    print("\n[3/4] Per-crop correlations with p-values …")
    plot_correlations(clean)

    print("\n[4/4] Climate anomaly map …")
    plot_anomaly_map(clean, climate)

    print("\nStep 2 complete — figures saved to:", FIGURES_DIR, "\n")


if __name__ == "__main__":
    main()

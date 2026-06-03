"""
src/data/download_era5.py
=========================
Downloads ERA5 monthly 2m temperature and precipitation
for Kyrgyzstan (1992–2025) using the ECMWF Climate Data Store API.

Requirements:
    pip install cdsapi
    Create ~/.cdsapirc with your CDS API key:
        url: https://cds.climate.copernicus.eu/api/v2
        key: <your-uid>:<your-api-key>

Usage:
    python src/data/download_era5.py
"""

import os
import sys

try:
    import cdsapi
except ImportError:
    print("ERROR: cdsapi not installed.")
    print("Run: pip install cdsapi")
    print("Then register at https://cds.climate.copernicus.eu and create ~/.cdsapirc")
    sys.exit(1)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_DIR

# Kyrgyzstan bounding box: lat 39–43N, lon 69–80E
AREA   = [43, 69, 39, 80]   # [N, W, S, E]
YEARS  = [str(y) for y in range(1992, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def download_temperature():
    out = os.path.join(RAW_DIR, "era5_temperature_kyrgyzstan_1992_2024.nc")
    if os.path.exists(out):
        print(f"  Already exists: {out}")
        return

    print("  Downloading ERA5 temperature (this may take several minutes) …")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable":     "2m_temperature",
            "year":         YEARS,
            "month":        MONTHS,
            "time":         "00:00",
            "area":         AREA,
            "format":       "netcdf",
        },
        out,
    )
    print(f"  Saved: {out}")


def download_precipitation():
    out = os.path.join(RAW_DIR, "era5_precipitation_kyrgyzstan_1992_2024.nc")
    if os.path.exists(out):
        print(f"  Already exists: {out}")
        return

    print("  Downloading ERA5 precipitation …")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable":     "total_precipitation",
            "year":         YEARS,
            "month":        MONTHS,
            "time":         "00:00",
            "area":         AREA,
            "format":       "netcdf",
        },
        out,
    )
    print(f"  Saved: {out}")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    print("=== Downloading ERA5 data ===")
    download_temperature()
    download_precipitation()
    print("\nNote: the project already includes pre-processed CSV versions")
    print("of ERA5 data. Only run this script if you want to re-download")
    print("from scratch from the ECMWF CDS API.")

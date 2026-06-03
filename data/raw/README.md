# Raw Data Files

This folder holds the four raw data files used in the pipeline.
They are excluded from Git (`.gitignore`) because of their size.
Download or copy them here before running the pipeline.

## Files Required

| File | Source | Size (approx.) |
|------|--------|---------------|
| `era5_temperature_kyrgyzstan_1992_2024.csv` | ECMWF ERA5 CDS | ~40 MB |
| `era5_precipitation_kyrgyzstan_1992_2024.csv` | ECMWF ERA5 CDS | ~40 MB |
| `modis_ndvi_monthly.csv` | NASA MODIS MOD13A3 | <1 MB |
| `FAOSTAT_combined_2026.csv` | FAO FAOSTAT | <1 MB |

## How to Obtain Each File

### ERA5 Reanalysis (Temperature + Precipitation)

1. Register at https://cds.climate.copernicus.eu (free)
2. Install the CDS API client:
   ```bash
   pip install cdsapi
   ```
3. Create `~/.cdsapirc` with your API key (shown on your CDS profile page):
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: <your-uid>:<your-api-key>
   ```
4. Run the download helper:
   ```bash
   python src/data/download_era5.py
   ```
   This downloads NetCDF files. The pipeline expects CSV format.
   Pre-converted CSVs are provided on the project's Zenodo record.

### MODIS NDVI

1. Visit https://appeears.earthdatacloud.nasa.gov/
2. Register for a free NASA Earthdata account
3. Submit a point/region request for:
   - Product: MOD13A3 v061 (Terra Vegetation Indices, Monthly, 1km)
   - Layer: _1_km_monthly_NDVI
   - Region: Kyrgyzstan boundary shapefile
   - Date: 2000-01-01 to 2025-12-31
4. Download the CSV output and rename to `modis_ndvi_monthly.csv`

### FAO FAOSTAT

1. Visit https://www.fao.org/faostat/en/#data/QCL
2. Select:
   - Country: Kyrgyzstan
   - Items: Wheat, Barley, Potatoes
   - Elements: Area harvested, Yield, Production
   - Years: 1992–2025
3. Download as CSV and rename to `FAOSTAT_combined_2026.csv`

## Pre-processed Data (Recommended)

The project Zenodo record provides pre-processed versions of all four files
ready to copy into this folder:

**DOI:** https://doi.org/[insert-zenodo-doi]

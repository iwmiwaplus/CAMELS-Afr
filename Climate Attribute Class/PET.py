"""
extract_pet_daily.py
---------------------
Area-weighted daily PET extraction from GeoTIFF files into a CSV.
Uses rasterstats.zonal_stats with coverage fractions (all_touched=False)
— equivalent to exactextractr's weighted mean in R.

Dependencies:
    pip install geopandas rasterio rasterstats numpy pandas tqdm

Usage:
    python extract_pet_daily.py
"""

import os
import re
import gc
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS  –  edit these
# ─────────────────────────────────────────────
BASIN_SHP = r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"
PET_DIR   = r"...\PET_daily"  # floder path of the PET tif files
OUT_CSV   = r"...\CAMELS_Afr_Climate_Daily_L1_pet_mean.csv"


# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
ID_FIELD   = "Basin_ID"
NA_VALUE   = -999.0    # raster NoData flag
CHUNK_SIZE = 100       # files to process before writing to CSV

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def list_pet_files(pet_dir):
    """Return sorted list of PET .tif/.tiff files matching Ep_*.tif(f)."""
    files = sorted(
        glob.glob(os.path.join(pet_dir, "Ep_*.tif")) +
        glob.glob(os.path.join(pet_dir, "Ep_*.tiff"))
    )
    if not files:
        raise FileNotFoundError(f"No matching PET files found in {pet_dir}")
    return files


def parse_date(filename):
    """Extract YYYY.MM.DD from filename and return as M/D/YYYY."""
    m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", os.path.basename(filename))
    if not m:
        raise ValueError(f"Cannot parse date from filename: {filename}")

    date = pd.to_datetime(
        f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    )

    return f"{date.month}/{date.day}/{date.year}"

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # 1. Load basins and reproject to match raster CRS
    print("Loading basin shapefile …")
    pet_files = list_pet_files(PET_DIR)
    print(f"Found {len(pet_files)} PET files.")

    basins = gpd.read_file(BASIN_SHP)
    with rasterio.open(pet_files[0]) as src:
        raster_crs = src.crs
    basins = basins.to_crs(raster_crs)

    basin_ids = basins[ID_FIELD].astype(str).tolist()
    geoms     = basins.geometry.values   # numpy array of shapely geometries

    # 2. Write CSV header
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    header_df = pd.DataFrame(columns=["Date"] + basin_ids)
    header_df.to_csv(OUT_CSV, index=False)

    # 3. Process files in chunks, appending to CSV
    total   = len(pet_files)
    results = []

    for i, fpath in enumerate(tqdm(pet_files, desc="Extracting PET"), start=1):

        with rasterio.open(fpath) as src:
            arr    = src.read(1).astype(np.float64)
            affine = src.transform
            nodata = src.nodata

        # ── NoData / invalid masking (mirrors R script logic) ──────────
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr == NA_VALUE] = np.nan   # explicit -999 flag
        arr[arr < 0]         = np.nan   # PET cannot be negative

        # ── Area-weighted mean via coverage fractions ───────────────────
        # all_touched=False  → only pixels whose centre falls inside, or
        # that are partially covered, contribute proportionally to their
        # coverage fraction — equivalent to exactextractr's weighted mean.
        stats = zonal_stats(
            geoms,
            arr,
            affine=affine,
            stats=["mean"],
            nodata=np.nan,
            all_touched=False,
        )

        means = [
            s["mean"] if (s["mean"] is not None and np.isfinite(s["mean"]))
            else np.nan
            for s in stats
        ]
        results.append([parse_date(fpath)] + means)

        # Write chunk to CSV and free memory
        if i % CHUNK_SIZE == 0 or i == total:
            chunk_df = pd.DataFrame(results, columns=["date"] + basin_ids)
            chunk_df.to_csv(OUT_CSV, mode="a", index=False, header=False)
            results.clear()
            del chunk_df
            gc.collect()

    print(f"\n✅ PET extraction complete → {OUT_CSV}")


if __name__ == "__main__":
    main()
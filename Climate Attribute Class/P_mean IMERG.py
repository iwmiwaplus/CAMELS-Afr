"""
IMERG Daily Precipitation Extraction to CSV
ith Date + HYBAS_ID columns

Requirements:
    pip install netCDF4 numpy pandas geopandas rasterio shapely exactextract tqdm

Usage:
    python extract_imerg_precipitation.py
"""

import os
import re
import glob
import warnings
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from exactextract import exact_extract
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---- edit these ----

BASIN_SHP   = r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"
IMERG_DIR   = r"...\IMERG"   # floder path of the imerg_percipitation tif files 
OUT_CSV     = r"...\CAMELS_Afr_Climate_Daily_L1_p_mean_imerg.csv"

VARIABLE  = "precipitation"
ID_FIELD  = "Basin_ID"
CHUNK_SIZE  = 50      # Files processed per batch before writing to disk
MAX_WORKERS = 4       # Parallel workers for raster loading (tune to your CPU count)

# --------------------


def load_imerg_array(nc4_path: str, sub_dataset: str = "precipitation"):
    """
    Load an IMERG .nc4 file, apply column-major orientation fix,
    clamp negatives to NaN, return (date_str, array, rasterio_meta).
    """
    basename   = os.path.basename(nc4_path)
    date_match = re.search(r"(\d{8})", basename)
    if not date_match:
        raise ValueError(f"Cannot parse date from filename: {basename}")
    date_str = date_match.group(1)

    with nc.Dataset(nc4_path, "r") as ds:
        raw = ds.variables[sub_dataset][:]

    # Remove size-1 dimensions (e.g. time=1), force float32
    data = np.array(raw, dtype=np.float32)
    data = np.squeeze(data)          # → (lon, lat) column-major

    # IMERG orientation fix: mirrors R's t() then flip(r, "vertical")
    data = data.T                    # → (lat, lon)
    data = np.flipud(data)           # → north-up

    # Clamp negatives to NaN; zero is valid
    data[data < 0] = np.nan

    n_lat, n_lon = data.shape        # e.g. 1800 × 3600 for 0.1°
    transform = from_bounds(
        west=-180, south=-90, east=180, north=90,
        width=n_lon, height=n_lat
    )
    meta = {
        "driver":    "GTiff",
        "height":    n_lat,
        "width":     n_lon,
        "count":     1,
        "dtype":     np.float32,
        "crs":       CRS.from_epsg(4326),
        "transform": transform,
        "nodata":    np.nan,
    }
    return date_str, data, meta

def process_file(args):
    """Worker: load one file, return serialisable tuple for the main process."""
    nc4_path, sub_dataset = args
    try:
        date_str, arr, meta = load_imerg_array(nc4_path, sub_dataset)
        return date_str, arr, meta, None
    except Exception as e:
        return None, None, None, f"{os.path.basename(nc4_path)}: {e}"
    
def extract_means(arr, meta, basins):
    """
    Write array to a temp GeoTIFF on /tmp (RAM-backed on Linux),
    pass file path to exact_extract, clean up, return list of means.
    """
    with tempfile.NamedTemporaryFile(suffix=".tif", dir="/tmp", delete=False) as tf:
        tmp_path = tf.name

    try:
        with rasterio.open(tmp_path, "w", **meta) as ds:
            ds.write(arr, 1)

        stats = exact_extract(
            rast   = tmp_path,
            vec    = basins,
            ops    = ["mean"],
            output = "pandas",
        )
        return stats["mean"].tolist()

    finally:
        os.remove(tmp_path)
        
        
    
def main():
    # ── Load shapefile ────────────────────────────────────────────────────────
    print("Reading shapefile …")
    basins = gpd.read_file(BASIN_SHP)

    if basins.crs is None:
        basins = basins.set_crs(epsg=4326)
    else:
        basins = basins.to_crs(epsg=4326)

    basin_ids = basins[ID_FIELD].astype(str).tolist()
    print(f"  {len(basin_ids)} basin(s) loaded.")

    # ── List IMERG files ──────────────────────────────────────────────────────
    pattern = os.path.join(IMERG_DIR, "3B-DAY*.nc4")
    imerg_files = sorted(glob.glob(pattern))

    if not imerg_files:
        raise FileNotFoundError(
            f"No .nc4 files found in: {IMERG_DIR}"
        )

    print(f"Found {len(imerg_files)} IMERG files.\n")

    # ── Initialise output CSV ─────────────────────────────────────────────────
    pd.DataFrame(columns=["Date"] + basin_ids).to_csv(
        OUT_CSV, index=False
    )

    # ── Chunked + parallel processing ────────────────────────────────────────
    total_chunks = (len(imerg_files) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(imerg_files))
        chunk = imerg_files[start:end]

        print(
            f"Chunk {chunk_idx + 1}/{total_chunks} "
            f"(files {start + 1}–{end})"
        )

        # ── Parallel raster loading ───────────────────────────────────────────
        args = [(f, SUB_DATASET) for f in chunk]
        results = []

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_file, a): a
                for a in args
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="  Loading",
                unit="file",
                leave=False
            ):
                date_str, arr, meta, err = future.result()

                if err:
                    print(f"  ⚠  {err}")
                else:
                    results.append((date_str, arr, meta))

        # ── Sort chronologically within chunk ─────────────────────────────────
        results.sort(key=lambda x: x[0])

        if not results:
            print("  ⚠  No valid files in this chunk, skipping.\n")
            continue

        # ── Zonal statistics (area-weighted mean) ─────────────────────────────
        rows = []

        for date_str, arr, meta in tqdm(
            results,
            desc="  Extracting",
            unit="date",
            leave=False
        ):
            means = extract_means(arr, meta, basins)

            # Convert date to MM/DD/YYYY
            dt = datetime.strptime(date_str, "%Y%m%d")
            fmt = dt.strftime("%m/%d/%Y")

            rows.append([fmt] + means)

        # ── Append chunk to CSV ──────────────────────────────────────────────
        pd.DataFrame(
            rows,
            columns=["Date"] + basin_ids
        ).to_csv(
            OUT_CSV,
            mode="a",
            header=False,
            index=False
        )

        print(f"  ✓ {len(rows)} dates written\n")

    print(
        f"🚀 Extraction complete. Results saved to:\n   {OUT_CSV}"
    )


if __name__ == "__main__":
    main()

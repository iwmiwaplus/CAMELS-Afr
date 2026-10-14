import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
 
# -------------------------------------------------
# Paths
# -------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
 
basin_shp = os.path.join(
    base_dir,
    "Basin shp files 2",
    "Basin shp files",
    "Level 04.shp"
)
 
mswep_dir = base_dir
 
out_csv = os.path.join(
    base_dir,
    "CAMELS_Afr_Climate_Daily_L1_p_mean_mswep.csv"
)
 
id_field = "Basin_ID"
 
# -------------------------------------------------
# Check input files
# -------------------------------------------------
if not os.path.exists(basin_shp):
    raise FileNotFoundError(
        f"Basin shapefile not found: {basin_shp}"
    )
 
mswep_files = sorted(
    glob.glob(os.path.join(mswep_dir, "mswep_*.tif")) +
    glob.glob(os.path.join(mswep_dir, "mswep_*.tiff"))
)
 
if not mswep_files:
    raise FileNotFoundError(
        f"No MSWEP raster files found in: {mswep_dir}"
    )
 
print(f"Found {len(mswep_files)} MSWEP raster files.")
 
# -------------------------------------------------
# Read basin shapefile
# -------------------------------------------------
basins = gpd.read_file(basin_shp)
 
if basins.empty:
    raise ValueError("The basin shapefile contains no basins.")
 
if id_field not in basins.columns:
    raise ValueError(
        f"'{id_field}' not found. "
        f"Available columns: {basins.columns.tolist()}"
    )
 
if basins.crs is None:
    raise ValueError("The basin shapefile has no CRS.")
 
if basins[id_field].isna().any():
    raise ValueError(f"Missing values found in {id_field}.")
 
if basins[id_field].duplicated().any():
    raise ValueError(
        f"Duplicate basin IDs found in {id_field}."
    )
 
# -------------------------------------------------
# Match basin CRS to raster CRS
# -------------------------------------------------
with rasterio.open(mswep_files[0]) as src:
    mswep_crs = src.crs
 
if mswep_crs is None:
    raise ValueError("The MSWEP raster has no CRS.")
 
print("Original basin CRS:", basins.crs)
print("MSWEP raster CRS:", mswep_crs)
 
if basins.crs != mswep_crs:
    print("Reprojecting basins to raster CRS...")
    basins = basins.to_crs(mswep_crs)
 
print("Basin CRS after reprojection:", basins.crs)
print("Number of basins:", len(basins))
 
basin_ids = basins[id_field].astype(str).tolist()
geoms = list(basins.geometry)
 
# -------------------------------------------------
# Extract date from filename
# Expected: mswep_YYYY.MM.DD.tif
# Output: YYYY-MM-DD
# -------------------------------------------------
def extract_date(filename):
    match = re.search(
        r"(\d{4})\.(\d{2})\.(\d{2})",
        os.path.basename(filename)
    )
 
    if match is None:
        raise ValueError(
            f"Cannot extract date from: {filename}"
        )
 
    year, month, day = map(int, match.groups())
 
    return pd.Timestamp(
        year=year,
        month=month,
        day=day
    )
 
# -------------------------------------------------
# Process each MSWEP raster
# -------------------------------------------------
results = []
 
for i, raster_file in enumerate(mswep_files, start=1):
 
    date = extract_date(raster_file)
 
    with rasterio.open(raster_file) as src:
        arr = src.read(1).astype(np.float64)
        affine = src.transform
        nodata = src.nodata
        raster_crs = src.crs
 
    if raster_crs != mswep_crs:
        raise ValueError(
            f"Raster CRS mismatch: {raster_file}"
        )
 
    # Mask NoData values
    if nodata is not None:
        if np.isnan(nodata):
            arr[np.isnan(arr)] = np.nan
        else:
            arr[arr == nodata] = np.nan
 
    # Remove invalid values and negative precipitation
    arr[~np.isfinite(arr)] = np.nan
    arr[arr < 0] = np.nan
 
    # Calculate mean precipitation for each basin
    stats = zonal_stats(
        geoms,
        arr,
        affine=affine,
        stats=["mean"],
        nodata=np.nan,
        all_touched=False
    )
 
    means = [
        stat["mean"]
        if stat["mean"] is not None
        else np.nan
        for stat in stats
    ]
 
    results.append([date] + means)
 
    if i % 100 == 0 or i == len(mswep_files):
        print(f"Processed {i}/{len(mswep_files)} rasters.")
 
# -------------------------------------------------
# Create output DataFrame
# -------------------------------------------------
columns = ["Date"] + basin_ids
 
df = pd.DataFrame(
    results,
    columns=columns
)
 
# Sort chronologically and format dates
df = df.sort_values("Date").reset_index(drop=True)
 
df["Date"] = df["Date"].dt.strftime("%m/%d/%Y")
 
# -------------------------------------------------
# Save CSV
# -------------------------------------------------
df.to_csv(out_csv, index=False)
 
# -------------------------------------------------
# Validation summary
# -------------------------------------------------
print("\nMSWEP precipitation extraction completed!")
print("Output file:", out_csv)
print("Number of dates:", len(df))
print("Number of basins:", len(basin_ids))
print("Date range:", df["Date"].iloc[0], "to", df["Date"].iloc[-1])
print(
    "Missing precipitation values:",
    int(df.iloc[:, 1:].isna().sum().sum())
)
 
print("\nFirst five rows:")
print(df.head().to_string(index=False))
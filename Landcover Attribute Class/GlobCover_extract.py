"""
GlobCover 2009 land-cover extraction — CAMELS-Africa Level 1 only.

Aligned with the attached notebook:
- Uses the existing Basin_ID field from the basin shapefile.
- Does not use HYBAS_ID and does not rename Basin_ID.
- Processes Level 1 basins only (no Level 3 or Level 4 processing).
- Reprojects basin geometry to the GlobCover raster CRS when required.
- Produces only the land-cover-percentage CSV output.

Update the input/output paths before running.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from tqdm import tqdm


# ======================================================
# 1. INPUT AND OUTPUT PATHS — UPDATE IF NEEDED
# ======================================================

basin_path = Path(
    r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"
)

globcover_tif = Path(
    r"...\2009 Version\GLOBCOVER_L4_200901_200912_V2.3.tif"
)

output_dir = Path(r"D:\Camel_Checking_Scripts\LandCover")
output_dir.mkdir(parents=True, exist_ok=True)

percentages_csv = output_dir / "CAMELS_Afr_Landcover_Static_L1.csv"

BASIN_ID_COLUMN = "Basin_ID"


# ======================================================
# 2. LOAD LEVEL 1 BASINS AND CHECK INPUTS
# ======================================================

if not basin_path.exists():
    raise FileNotFoundError(f"Basin shapefile not found: {basin_path}")
if not globcover_tif.exists():
    raise FileNotFoundError(f"GlobCover raster not found: {globcover_tif}")

basins = gpd.read_file(basin_path)

if basins.empty:
    raise ValueError("The basin shapefile contains no features.")
if BASIN_ID_COLUMN not in basins.columns:
    raise KeyError(
        f"'{BASIN_ID_COLUMN}' was not found in the shapefile. "
        f"Available columns: {list(basins.columns)}"
    )
if basins.crs is None:
    raise ValueError(
        "The basin shapefile CRS is undefined. Check the .prj file before proceeding."
    )
if basins[BASIN_ID_COLUMN].isna().any():
    raise ValueError("Missing Basin_ID values were found in the shapefile.")
if basins[BASIN_ID_COLUMN].duplicated().any():
    raise ValueError("Duplicate Basin_ID values were found in the shapefile.")

print(f"Loaded Level 1 basins: {len(basins):,}")
print("Original basin CRS:", basins.crs)


# ======================================================
# 3. READ RASTER METADATA AND MATCH CRS
# ======================================================

with rasterio.open(globcover_tif) as src:
    raster_crs = src.crs
    transform = src.transform
    raster_shape = (src.height, src.width)
    raster_nodata = src.nodata
    block_windows = list(src.block_windows(1))

if raster_crs is None:
    raise ValueError("The GlobCover raster CRS is undefined.")

print("GlobCover raster CRS:", raster_crs)
if basins.crs != raster_crs:
    print("Reprojecting Level 1 basin geometries to the raster CRS...")
    basins = basins.to_crs(raster_crs)
if basins.crs != raster_crs:
    raise ValueError("Basin CRS and raster CRS do not match.")
print("CRS used for extraction:", basins.crs)


# ======================================================
# 4. CREATE INTERNAL RASTER IDS; KEEP Basin_ID UNCHANGED
# ======================================================

basins = basins.reset_index(drop=True).copy()
basins["raster_id"] = np.arange(1, len(basins) + 1, dtype=np.int32)
id_map = basins[["raster_id", BASIN_ID_COLUMN]].copy()

shapes = [
    (geom, int(raster_id))
    for geom, raster_id in zip(basins.geometry, basins["raster_id"])
    if geom is not None and not geom.is_empty
]
if not shapes:
    raise ValueError("No valid basin geometries were found.")

print("Rasterizing Level 1 basin boundaries...")
basin_raster = rasterize(
    shapes=shapes,
    out_shape=raster_shape,
    transform=transform,
    fill=0,
    dtype="int32"
)


# ======================================================
# 5. COUNT RASTER PIXELS BY BASIN AND LAND-COVER CLASS
# ======================================================

results = []
with rasterio.open(globcover_tif) as src:
    for _, window in tqdm(block_windows, desc="Counting GlobCover pixels"):
        lc = src.read(1, window=window)
        row_start = int(window.row_off)
        row_end = row_start + int(window.height)
        col_start = int(window.col_off)
        col_end = col_start + int(window.width)
        basin_chunk = basin_raster[row_start:row_end, col_start:col_end]

        # Keep the notebook's pixel-selection rule.
        mask = (lc > 0) & (basin_chunk > 0)
        if raster_nodata is not None:
            if np.isnan(raster_nodata):
                mask &= ~np.isnan(lc)
            else:
                mask &= lc != raster_nodata

        if np.any(mask):
            results.append(pd.DataFrame({
                "raster_id": basin_chunk[mask],
                "lc": lc[mask]
            }))

if not results:
    raise ValueError(
        "No valid raster pixels were found inside the basin boundaries. "
        "Check CRS, raster extent, geometries, and raster values."
    )

pixel_data = pd.concat(results, ignore_index=True)
pixel_count = (
    pixel_data.groupby(["raster_id", "lc"])
    .size()
    .reset_index(name="pixel_count")
)
pixel_count = pixel_count.merge(
    id_map, on="raster_id", how="left", validate="many_to_one"
)

# Keep Basin_ID as extracted from the shapefile.
pixel_count = pixel_count[[BASIN_ID_COLUMN, "raster_id", "lc", "pixel_count"]]


# ======================================================
# 6. PIVOT PIXEL COUNTS — ONE ROW PER Basin_ID
# ======================================================

pivot = pixel_count.pivot_table(
    index=BASIN_ID_COLUMN,
    columns="lc",
    values="pixel_count",
    aggfunc="sum",
    fill_value=0
).reset_index()
pivot.columns.name = None
class_columns = [c for c in pivot.columns if c != BASIN_ID_COLUMN]
pivot["total_pixels"] = pivot[class_columns].sum(axis=1)


# ======================================================
# 7. AGGREGATE GLOBCOVER CLASSES 
# ======================================================

def sum_cols(dataframe, classes):
    available = [c for c in classes if c in dataframe.columns]
    if not available:
        return np.zeros(len(dataframe), dtype=np.int64)
    return dataframe[available].sum(axis=1)

out = pd.DataFrame({
    BASIN_ID_COLUMN: pivot[BASIN_ID_COLUMN],
    "bare": sum_cols(pivot, [200]),
    "forest": sum_cols(pivot, [40, 50, 60, 70, 90, 100, 160, 170]),
    "urban": sum_cols(pivot, [190]),
    "cover_grass": sum_cols(pivot, [140]),
    "cover_shrub": sum_cols(pivot, [130]),
    "cover_crops": sum_cols(pivot, [11, 14]),
    "mosaic": sum_cols(pivot, [20, 30, 110, 120]),
    "veg": sum_cols(pivot, [150, 180]),
    "water": sum_cols(pivot, [210]),
    "total_pixels": pivot["total_pixels"]
})


# ======================================================
# 8. CALCULATE PERCENTAGES
# ======================================================

perc = out.copy()
count_columns = [
    "bare", "forest", "urban", "cover_grass", "cover_shrub",
    "cover_crops", "mosaic", "veg", "water"
]
for col in count_columns:
    perc[col + "_perc"] = np.where(
        perc["total_pixels"] > 0,
        (perc[col] / perc["total_pixels"]) * 100.0,
        0.0
    )

perc = perc.drop(columns=count_columns)
lc_cols = [c for c in perc.columns if c.endswith("_perc")]
perc["lc_dom"] = perc[lc_cols].idxmax(axis=1).str.replace(
    "_perc", "", regex=False
)
perc["dom_land_cover_perc"] = perc[lc_cols].max(axis=1)

# Keep all Level 1 Basin_ID values, including basins with no counted pixels.
all_basins = basins[[BASIN_ID_COLUMN]].copy()
final_output = all_basins.merge(
    perc, on=BASIN_ID_COLUMN, how="left", validate="one_to_one"
)
percentage_columns = [c for c in perc.columns if c.endswith("_perc")]
final_output[percentage_columns] = final_output[percentage_columns].fillna(0)
final_output = final_output.drop(columns=["total_pixels"])
final_output.to_csv(percentages_csv, index=False)


# ======================================================
# 9. SUMMARY
# ======================================================

print("\n========== LEVEL 1 EXTRACTION SUMMARY ==========")
print(f"Level 1 basins in shapefile: {len(basins):,}")
print(f"Basins with counted pixels: {pixel_count[BASIN_ID_COLUMN].nunique():,}")
print(f"Basins in final percentage CSV: {len(final_output):,}")
print(f"Basins with all-zero land-cover percentages: {(final_output[percentage_columns].sum(axis=1) == 0).sum():,}")
print("Percentage CSV:", percentages_csv)
print("\nLevel 1 GlobCover extraction completed.")

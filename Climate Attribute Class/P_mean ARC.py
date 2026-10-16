'''
Install the required libraries if they are not already installed.
Run the following command:
pip install numpy pandas geopandas rasterio rasterstats

'''

# Import below libraries first
import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from rasterio.warp import calculate_default_transform

#-------------------------------------------------
# Paths
# -------------------------------------------------

basin_shp = r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"
arc_dir   = r"...\ARC_daily"      # floder path of the ARC percipitation tif files
out_csv   = r"...\CAMELS_Afr_Climate_Daily_L1_p_mean_arc.csv"

id_field  = "Basin_ID"

# -------------------------------------------------
# List ARC files
# -------------------------------------------------
arc_files = sorted(glob.glob(os.path.join(arc_dir, "africa_arc_*.tif")) +
                    glob.glob(os.path.join(arc_dir, "africa_arc_*.tiff")))
assert len(arc_files) > 0, "No ARC files found."
print(f"Found {len(arc_files)} ARC files.")

# -------------------------------------------------
# Read basins and match CRS to first raster
# -------------------------------------------------
basins = gpd.read_file(basin_shp)

with rasterio.open(arc_files[0]) as src:
    arc_crs = src.crs

print("Transforming shapefile to match raster CRS...")
basins = basins.to_crs(arc_crs)

basin_ids = basins[id_field].astype(str).tolist()

# -------------------------------------------------
# Date extraction helper
# -------------------------------------------------
def format_date(fname):
    m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", os.path.basename(fname))
    y, mo, d = m.groups()
    return f"{int(mo)}/{int(d)}/{int(y)}"

# -------------------------------------------------
# Process each raster
# -------------------------------------------------
geoms = basins.geometry.values

results = []

for i, f in enumerate(arc_files, 1):
    with rasterio.open(f) as src:
        arr = src.read(1).astype(np.float64)
        affine = src.transform
        nodata = src.nodata

    # Mask invalid values: set <0 (and original nodata) to NaN so they are excluded
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < 0] = np.nan

    stats = zonal_stats(
        geoms,
        arr,
        affine=affine,
        stats=["mean"],
        nodata=np.nan,
        all_touched=False
    )

    means = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
    results.append([format_date(f)] + means)

    if i % 100 == 0 or i == len(arc_files):
        print(f"Processed {i}/{len(arc_files)} files")

# -------------------------------------------------
# Write to CSV
# -------------------------------------------------
columns = ["Date"] + basin_ids
df = pd.DataFrame(results, columns=columns)
df.to_csv(out_csv, index=False)

print("ARC Precipitation extraction completed!")
print("Output saved to:", out_csv)
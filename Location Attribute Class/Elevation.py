import geopandas as gpd
import rasterio
import pandas as pd
from exactextract import exact_extract

# 1. File Paths
dem_path = r"...\FABDEM_merged.tif"
basins_path = r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"
output_path = r"...\CAMELS_Afr_Location_Static_L1_elevation.csv"

# 2. Load Data
print("Loading Basins...")
basins = gpd.read_file(basins_path)

# 3. Align CRS (Vector to Raster is faster)
with rasterio.open(dem_path) as src:
    dem_crs = src.crs
    if basins.crs != dem_crs:
        print(f"Reprojecting basins from {basins.crs} to {dem_crs}...")
        basins = basins.to_crs(dem_crs)

# 4. Fix Geometries (Equivalent to st_make_valid)
print("Validating geometries...")
basins['geometry'] = basins.geometry.buffer(0)

# 5. Extract Stats
# exactextract weights all stats by pixel coverage fraction (area) by default
print("Starting extraction... (This may take time for large DEMs)")

ops = [
    'mean',
    'min',
    'max',
    'quantile(q=0.10)',
    'quantile(q=0.25)',
    'quantile(q=0.50)',
    'quantile(q=0.75)',
    'quantile(q=0.90)',
]

results = exact_extract(
    dem_path,
    basins,
    ops,
    include_cols=['Basin_ID'],
    output='pandas'
)

# 6. Rename and Format
print("Formatting and renaming columns...")

# Calculate Range
results['elev_range'] = results['max'] - results['min']

results = results.rename(columns={
    'Basin_ID': 'Basin_ID',
    'mean':        'elev_mean',
    'min':         'elev_min',
    'max':         'elev_max',
    'quantile_10': 'elev_p10',
    'quantile_25': 'elev_p25',
    'quantile_50': 'elev_median',
    'quantile_75': 'elev_p75',
    'quantile_90': 'elev_p90',
})

# Select and Order Columns
final_cols = [
    'Basin_ID', 'elev_mean', 'elev_min', 'elev_max',
    'elev_p10', 'elev_p25', 'elev_median',
    'elev_p75', 'elev_p90', 'elev_range'
]

# Save to CSV
results[final_cols].to_csv(output_path, index=False)

print(f"Done! Results saved to: {output_path}")
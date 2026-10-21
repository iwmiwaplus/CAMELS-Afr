import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from shapely.geometry import mapping

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
flow_path = r".....\FABDEM_flowdir_d8_90m.tif"
shp_path  = r"....\CAMELS_Afr_L1_Basin_Boundaries.shp"
out_csv   = r".......\CAMELS_Afr_Location_Static_L1_flow_direction.csv"

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
print("Loading vector data...")
basins = gpd.read_file(shp_path)

# Open Raster connection
src = rasterio.open(flow_path)

# ------------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------------

# 1. Ensure CRS match (Reproject Basins to Raster CRS)
if basins.crs != src.crs:
    print(f"Reprojecting basins from {basins.crs} to {src.crs}...")
    basins = basins.to_crs(src.crs)

# 2. Fix Geometries
print("Fixing invalid geometries...")
basins['geometry'] = basins.make_valid()

# Prepare columns (Only ID and Mode)
out_columns = ["Basin_ID", "flowdir_mode"]

# Initialize output CSV
if not os.path.exists(out_csv):
    pd.DataFrame(columns=out_columns).to_csv(out_csv, index=False)

# ------------------------------------------------------------------
# Extract Mode (Iterative)
# ------------------------------------------------------------------
total_basins = len(basins)
print(f"Starting extraction for {total_basins} basins...")

for idx, row in basins.iterrows():
    Basin_ID = row['Basin_ID'] 
    
    # 1. Geometry Prep
    geom_fixed = row['geometry'].buffer(0) # Fix micro-topologies
    geoms = [mapping(geom_fixed)] # Convert to GeoJSON for rasterio
    
    if (idx + 1) % 50 == 0:
        print(f"Processing basin {idx + 1} of {total_basins}...")

    try:
        # 2. Extract Data
        # all_touched=True ensures we catch pixels for small/thin polygons
        out_image, out_transform = mask(src, geoms, crop=True, nodata=src.nodata, all_touched=True)
        
        # Flatten array
        vals = out_image[0].flatten()
        
        # 3. Clean Data
        # Remove NoData
        if src.nodata is not None:
            vals = vals[vals != src.nodata]
            
        # Remove NaNs
        vals = vals[~np.isnan(vals)]

        if len(vals) == 0:
            mode_val = np.nan
        else:
            # 4. Calculate Mode
            # np.unique returns sorted unique elements and their counts
            values, counts = np.unique(vals, return_counts=True)
            # Find the index of the maximum count
            mode_index = np.argmax(counts)
            # Retrieve the value corresponding to that index
            mode_val = values[mode_index]

        # Prepare Result
        data = {
            "Basin_ID": Basin_ID,
            "flowdir_mode": mode_val
        }

    except Exception as e:
        print(f"Error on Basin {Basin_ID}: {e}")
        data = {
            "Basin_ID": Basin_ID,
            "flowdir_mode": np.nan
        }

    # 5. Append to CSV
    pd.DataFrame([data]).to_csv(out_csv, mode='a', header=False, index=False)

# Close raster connection
src.close()
print(f"Processing complete. Saved to {out_csv}")
"""
=============================================================================
iSDAsoil Basin Extraction Tool: Categorical Variables (Percentages)
=============================================================================
This script extracts zonal statistics from high-resolution iSDAsoil raster 
datasets (30m) across hydrological basins for categorical variables.

METHODOLOGY & MEMORY MANAGEMENT:
Extracting zonal statistics over massive, continent-scale basins (e.g., the 
Nile or Congo) frequently leads to MemoryError when attempting to load the 
entire masked array into RAM. To solve this, the script utilizes a Dynamic 
Execution switch:
1. Standard Processing: For basins smaller than 5000 x 5000 pixels, the 
   script uses fast, direct masking via `rasterio.mask`.
2. Chunked Processing: For massive basins, the script dynamically switches 
   to processing the bounding box in 2048 x 2048 tiled windows. It calculates 
   statistics incrementally, entirely bypassing memory limits while strictly 
   respecting the exact polygon boundaries.

CATEGORICAL VARIABLE HANDLING:
Specifically built for variables like Texture Class, which cannot be averaged 
mathematically. This script iterates through the basin, tallies the absolute 
pixel count for every categorical ID, maps the ID to an internal dictionary, 
and outputs:
* The Majority Class (Mode) string (e.g., "Sandy Clay Loam").
* The Percentage (0.00 to 100.00) of every possible class present within that 
  specific basin.
=============================================================================
"""

import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.transform import rowcol
import geopandas as gpd
import numpy as np
import pandas as pd
from collections import Counter
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=rasterio.errors.NodataShadowWarning)

# ==========================================
# 1. CONFIGURATION
# ==========================================

SPATIAL_LEVEL_NAME = {SPATIAL_LEVEL_NAME} # Examples: "Level_01", "Level_02", "Level_04"
TIF_FILE = REPO_ROOT / "Data" / "isDA" / "texture_class.tif"
SHP_FILE = REPO_ROOT / "CAMELS_Afr" / "Basin_Boundaries" / f"CAMELS_Afr_{SPATIAL_LEVEL_NAME}_Basin_Boundaries.shp"
OUTPUT_CSV = REPO_ROOT / "CAMELS-Africa" / {SPATIAL_LEVEL_NAME} / "Soil Attribute Class" / "isDa_variables" / "texture_class.csv"
BASIN_ID_FIELD = "Basin_ID"
LARGE_BASIN_LIMIT = 5000

# Texture Class Definitions
TEXTURE_CLASSES = {
    1: "Clay",
    2: "Silty Clay",
    3: "Sandy Clay",
    4: "Clay Loam",
    5: "Silty Clay Loam",
    6: "Sandy Clay Loam",
    7: "Loam",
    8: "Silt Loam",
    9: "Sandy Loam",
    10: "Silt",
    11: "Loamy Sand",
    12: "Sand"
}

# ==========================================
# 2. CORE PROCESSING FUNCTIONS
# ==========================================
def get_pixel_bounds(geom, transform):
    minx, miny, maxx, maxy = geom.bounds
    row_min, col_min = rowcol(transform, minx, maxy)
    row_max, col_max = rowcol(transform, maxx, miny)
    return max(0, row_min), row_max, max(0, col_min), col_max

def get_counts_standard(src, geometry, bands):
    out_image, _ = mask(src, [geometry], crop=True, all_touched=True)
    basin_counts = {b: Counter() for b in bands}
    
    for band_idx in bands:
        data = out_image[band_idx - 1]
        valid_mask = np.ones_like(data, dtype=bool)
        if src.nodata is not None:
            valid_mask &= (data != src.nodata)
            
        if np.any(valid_mask):
            basin_counts[band_idx].update(data[valid_mask].astype(int))
    return basin_counts

def get_counts_chunked(src, geometry, bands, chunk_size=2048):
    row_min, row_max, col_min, col_max = get_pixel_bounds(geometry, src.transform)
    row_max, col_max = min(src.height, row_max), min(src.width, col_max)
    basin_counts = {b: Counter() for b in bands}
    
    for r in range(row_min, row_max, chunk_size):
        for c in range(col_min, col_max, chunk_size):
            h, w = min(chunk_size, row_max - r), min(chunk_size, col_max - c)
            if h <= 0 or w <= 0: continue
            
            window = Window(c, r, w, h)
            try:
                data = src.read(window=window)
                win_transform = src.window_transform(window)
                mask_arr = geometry_mask([geometry], out_shape=(h, w), 
                                       transform=win_transform, invert=True, all_touched=True)
                
                if not np.any(mask_arr): continue
                for band_idx in bands:
                    band_data = data[band_idx - 1]
                    valid = mask_arr
                    if src.nodata is not None:
                        valid &= (band_data != src.nodata)
                        
                    if np.any(valid):
                        basin_counts[band_idx].update(band_data[valid].astype(int))
            except Exception:
                continue
    return basin_counts

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading Basins...")
    class_ids = list(TEXTURE_CLASSES.keys())
    
    gdf = gpd.read_file(SHP_FILE)
    gdf[BASIN_ID_FIELD] = gdf[BASIN_ID_FIELD].astype(str)
    results = []

    with rasterio.open(TIF_FILE) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
            
        target_bands = list(range(1, src.count + 1))
        depth_labels = ["0_20", "20_50"][:src.count]
        
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Extracting"):
            basin_id, geometry = row[BASIN_ID_FIELD], row.geometry
            if geometry is None or geometry.is_empty: continue
                
            try:
                row_min, row_max, col_min, col_max = get_pixel_bounds(geometry, src.transform)
                if (col_max - col_min) > LARGE_BASIN_LIMIT or (row_max - row_min) > LARGE_BASIN_LIMIT:
                    counts_dict = get_counts_chunked(src, geometry, target_bands)
                else:
                    counts_dict = get_counts_standard(src, geometry, target_bands)
                
                res_row = {"Basin_ID": basin_id}
                
                for band_idx, depth_label in zip(target_bands, depth_labels):
                    counter = counts_dict[band_idx]
                    total_pixels = sum(counter.values())
                    
                    if total_pixels > 0:
                        mode_val = counter.most_common(1)[0][0]
                        res_row[f"soil_textclass_{depth_label}"] = TEXTURE_CLASSES.get(mode_val, f"Unknown_{mode_val}")
                        
                        for cid in class_ids:
                            desc = TEXTURE_CLASSES[cid].replace(' ', '_').replace('/', '_')
                            # Changed to calculate percentage and round to 2 decimals
                            percentage = (counter[cid] / total_pixels) * 100.0
                            res_row[f"{desc}_perc_{depth_label}"] = percentage
                    else:
                        res_row[f"texture_{depth_label}"] = np.nan
                        for cid in class_ids:
                            desc = TEXTURE_CLASSES[cid].replace(' ', '_').replace('/', '_')
                            res_row[f"{desc}_{depth_label}_perc"] = np.nan
                            
                results.append(res_row)
            except Exception as e:
                print(f"Error on basin {basin_id}: {e}")
                results.append({"Basin_ID": basin_id})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone in {(time.time() - start_time)/60:.2f} minutes. Saved to {OUTPUT_CSV}")
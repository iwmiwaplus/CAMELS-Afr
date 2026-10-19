"""
SoilGrids Coarse Fragment Volume Extraction
-------------------------------------------
Extracts basin-wise mean coarse fragment volume (CRFVOL)
from SoilGrids raster datasets for six soil depth intervals
using CAMELS African Level 1 basin boundaries.
 
Methodology:
1. Processes basins in chunks to manage memory usage.
2. Reprojects basin geometries to match each raster's CRS.
3. Calculates mean raster values within each basin using
   zonal statistics, excluding NoData values.
 
Output:
A CSV file containing one row per basin and a separate
column for each soil depth interval.
"""
 
 
import geopandas as gpd
import pandas as pd
import glob
import os
import rasterio
import numpy as np
import warnings
from rasterstats import zonal_stats
from tqdm import tqdm
 
# Suppress benign warnings during spatial operations
warnings.filterwarnings('ignore', category=UserWarning)
 
 

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
RASTER_DIRECTORY = os.path.join(
    BASE_DIR, "Soil Grids"
)
 
SHAPEFILE_PATH = os.path.join(
    BASE_DIR,
    "CAMELS_Afr_L1_Basin_Boundaries",
    "CAMELS_Afr_L1_Basin_Boundaries.shp"
)
 
OUTPUT_FILE = os.path.join(
    BASE_DIR, "CAMELS_Afr_Soil_Static_L1_pct_gravel.csv"
)
 
BASIN_ID_COL = "Basin_ID"
CHUNK_SIZE = 1000

 
# Depth Mapping for standardized column nomenclature
DEPTH_NAME_MAP = {
    "1": "pct_gravel_0_5cm",
    "2": "pct_gravel_5_15cm",
    "3": "pct_gravel_15_30cm",
    "4": "pct_gravel_30_60cm",
    "5": "pct_gravel_60_100cm",
    "6": "pct_gravel_100_200cm"
}
 
def extract_soil_metrics():
    # Load raster file list
    tif_files = sorted(glob.glob(os.path.join(RASTER_DIRECTORY, "*_sd*_250m.tif")))
   
    # Get total basins without fiona by ignoring geometry for speed
    print("Calculating total basins...")
    total_basins = len(gpd.read_file(SHAPEFILE_PATH, ignore_geometry=True))
 
    print(f"Initiating extraction for {total_basins} basins across {len(tif_files)} depth layers.")
 
    final_results = []
 
    # ------------------------------------------------------------
    # Processing Loop
    # ------------------------------------------------------------
    for i in tqdm(range(0, total_basins, CHUNK_SIZE), desc="Processing Basin Chunks"):
       
        # Load vector slice
        gdf_chunk = gpd.read_file(SHAPEFILE_PATH, rows=slice(i, i + CHUNK_SIZE))
       
        # Break out of loop if chunk is empty
        if gdf_chunk.empty:
            break
           
        chunk_result = gdf_chunk[[BASIN_ID_COL]].copy()
        chunk_result.rename(columns={BASIN_ID_COL: 'Basin_ID'}, inplace=True)
 
        # Iterate through depth-specific rasters
        for tif_path in tif_files:
            try:
                depth_code = os.path.basename(tif_path).split("_sd")[1].split("_")[0]
                col_name = DEPTH_NAME_MAP.get(depth_code, f"pct_gravel_sd{depth_code}")
            except IndexError:
                continue
 
            with rasterio.open(tif_path) as src:
                raster_crs = src.crs
                raster_nodata = src.nodata if src.nodata is not None else 255
 
            # Ensure CRS alignment
            gdf_proc = gdf_chunk.to_crs(raster_crs) if gdf_chunk.crs != raster_crs else gdf_chunk
 
            # Calculate mean zonal statistics
            stats = zonal_stats(
                vectors=gdf_proc,
                raster=tif_path,
                stats="mean",
                nodata=raster_nodata
            )
           
            chunk_result[col_name] = [s['mean'] if s.get('mean') is not None else np.nan for s in stats]
 
        final_results.append(chunk_result)
 
    # ------------------------------------------------------------
    # Final Compilation
    # ------------------------------------------------------------
    if final_results:
        final_df = pd.concat(final_results, ignore_index=True)
        # Ensure Basin_ID is leading
        cols = ['Basin_ID'] + [c for c in final_df.columns if c != 'Basin_ID']
       
        # Create output directory if it doesn't exist to prevent FileNotFoundError
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
       
        final_df[cols].to_csv(OUTPUT_FILE, index=False)
        print(f"Extraction complete. Results saved to: {OUTPUT_FILE}")
    else:
        print("Error: No data processed.")
 
if __name__ == "__main__":
    extract_soil_metrics()
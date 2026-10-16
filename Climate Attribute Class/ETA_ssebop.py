import os
import glob
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats   #install rasterstats via pip if not already installed
from datetime import datetime

# 1. Define Paths
raster_folder = r".....\ETA_SSEBOP"
shapefile_path = r".....\CAMELS_Afr_L1_Basin_Boundaries.shp"
output_csv = r".....\CAMELS_Afr_Climate_Monthly_L1_eta_ssebop.csv"

# 2. Load the Basin Shapefile
print("Loading shapefile...")
basins = gpd.read_file(shapefile_path)



# --- CRS HANDLING ---
# Your shapefile is in ESRI:102022 (Africa Albers) 
# Your rasters are in EPSG:4326 (WGS84)
print(f"Original Shapefile CRS: {basins.crs}")
print("Reprojecting shapefile to EPSG:4326 to match TIFF files...")
basins = basins.to_crs("EPSG:4326")
# --------------------

# 3. Get list of TIF files and sort them chronologically
tif_files = sorted(glob.glob(os.path.join(raster_folder, "*.tif")))

results = []

print(f"Starting extraction for {len(tif_files)} files...")

for tif in tif_files:
    filename = os.path.basename(tif)
    
    # Extract date from filename: 'm201202_...' -> '201202'
    try:
        date_part = filename[1:7] 
        date_obj = datetime.strptime(date_part, "%Y%m")
        formatted_date = date_obj.strftime("%m/01/%Y")
    except ValueError:
        print(f"Skipping file with unexpected name format: {filename}")
        continue
    
    # 4. Perform Zonal Statistics
    # We use 'mean' to get the average ET value for each polygon
    stats = zonal_stats(basins, tif, stats="mean")
    
    # 5. Create a row dictionary for this month
    row_data = {"Date": formatted_date}
    
    # Map the extracted mean values to their respective CAMELS_ID
    for i, stat in enumerate(stats):
        basin_id = basins.iloc[i]["Basin_ID"]
        row_data[basin_id] = stat['mean']
        
    results.append(row_data)
    print(f"Processed: {formatted_date}")

# 6. Create DataFrame and Save to CSV
df = pd.DataFrame(results)

# Reorder columns: "Date" first, then CAMELS_IDs
cols = ['Date'] + [c for c in df.columns if c != 'Date']
df = df[cols]

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

df.to_csv(output_csv, index=False)

print(f"\nSuccess! File saved to: {output_csv}")
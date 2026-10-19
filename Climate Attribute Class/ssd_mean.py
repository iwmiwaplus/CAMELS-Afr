import os
import re
import xarray as xr
import rioxarray
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
 
warnings.filterwarnings('ignore')
 
 
# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
 
base_path = os.getcwd()
 
basin_shp = os.path.join(
    base_path,
    "Basin shp files",
    "CAMELS_Afr_L1_Basin_Boundaries.shp"
)
 
output_filename = "CAMELS_Afr_Climate_Daily_L1_ssd_mean.csv"
 
output_path = os.path.join(
    base_path,
    output_filename
)
 
 
# --------------------------------------------------
# 2. Load basins
# --------------------------------------------------
 
print("Loading basins...")
 
basins = gpd.read_file(basin_shp)
basins = basins.to_crs("EPSG:4326")
 
print(f"Loaded {len(basins)} basins")
 
 
# --------------------------------------------------
# 3. SDU extraction function
# --------------------------------------------------
 
def extract_sdu_rioxarray(nc_file, basin_geometry):
 
    try:
        with xr.open_dataset(nc_file) as ds:
 
            if 'SDU' not in ds:
                return np.nan
 
            sdu = ds['SDU']
 
            if 'time' in sdu.dims:
                sdu = sdu.isel(time=0, drop=True)
 
            # Assume the data uses EPSG:4326
            sdu = sdu.rio.write_crs("EPSG:4326")
 
            clipped = sdu.rio.clip(
                [basin_geometry],
                crs=sdu.rio.crs,
                drop=True
            )
 
            if clipped.size == 0 or np.all(np.isnan(clipped.values)):
                return np.nan
 
            return float(clipped.mean(skipna=True).values)
 
    except Exception:
        return np.nan
 
 
# --------------------------------------------------
# 4. Find NetCDF files
# --------------------------------------------------
 
print("\nFinding NetCDF files...")
 
nc_files = []
 
for root, dirs, files in os.walk(base_path):
 
    # Avoid searching inside any .git or intermediate output folders
    dirs[:] = [
        directory for directory in dirs
        if directory != ".git"
    ]
 
    for file in files:
 
        if file.lower().endswith('.nc'):
            nc_files.append(
                os.path.join(root, file)
            )
 
nc_files = sorted(nc_files)
 
print(f"Found {len(nc_files)} NetCDF files")
 
if not nc_files:
    raise FileNotFoundError(
        "No NetCDF files found in the task folder or its subfolders."
    )
 
 
# --------------------------------------------------
# 5. Extract SDU and combine basin data in memory
# --------------------------------------------------
 
print("\nExtracting SDU for all basins...")
 
all_data = []
 
for basin_idx, (_, basin) in enumerate(basins.iterrows()):
 
    basin_id = str(basin['Basin_ID'])
    geometry = basin['geometry']
 
    print(
        f"\nProcessing Basin {basin_id} "
        f"({basin_idx + 1}/{len(basins)})"
    )
 
    results = []
 
    for nc_file in tqdm(
        nc_files,
        desc=f"Basin {basin_id}"
    ):
 
        # Extract date from filename (YYYYMMDD)
        match = re.search(
            r"(\d{8})",
            os.path.basename(nc_file)
        )
 
        if not match:
            continue
 
        date = pd.to_datetime(
            match.group(1),
            format='%Y%m%d'
        )
 
        sdu_value = extract_sdu_rioxarray(
            nc_file,
            geometry
        )
 
        results.append({
            'Date': date,
            basin_id: sdu_value
        })
 
    basin_df = pd.DataFrame(results)
 
    if basin_df.empty:
        print(f"No dated NetCDF files processed for Basin {basin_id}")
        continue
 
    basin_df = basin_df.sort_values('Date')
 
    # Ensure one record per date before merging
    if basin_df['Date'].duplicated().any():
        raise ValueError(
            f"Duplicate dates found for Basin {basin_id}. "
            "Check the NetCDF filenames."
        )
 
    all_data.append(basin_df)
 
 
# --------------------------------------------------
# 6. Merge all basins by Date
# --------------------------------------------------
 
if not all_data:
    raise ValueError(
        "No basin data was extracted successfully."
    )
 
print("\nCombining basin data...")
 
combined_df = all_data[0]
 
for basin_df in all_data[1:]:
 
    combined_df = pd.merge(
        combined_df,
        basin_df,
        on='Date',
        how='outer',
        validate='one_to_one'
    )
 
 
# --------------------------------------------------
# 7. Sort dates chronologically
# --------------------------------------------------
 
combined_df = (
    combined_df
    .sort_values('Date')
    .reset_index(drop=True)
)
 
 
# --------------------------------------------------
# 8. Format dates as M/D/YYYY
# --------------------------------------------------
 
combined_df['Date'] = combined_df['Date'].apply(
    lambda d: f"{d.month}/{d.day}/{d.year}"
)
 
 
# --------------------------------------------------
# 9. Save only the final CSV
# --------------------------------------------------
 
combined_df.to_csv(
    output_path,
    index=False
)
 
 
# --------------------------------------------------
# 10. Summary
# --------------------------------------------------
 
print("\n====================================")
print("SDU workflow completed successfully!")
print("====================================")
 
print(f"Basins combined: {len(all_data)}")
print(f"Basin columns: {len(combined_df.columns) - 1}")
print(f"Date records: {len(combined_df)}")
print(f"First date: {combined_df['Date'].iloc[0]}")
print(f"Last date: {combined_df['Date'].iloc[-1]}")
print(f"Output file: {output_path}")
 
print("\nFirst five rows:")
print(combined_df.head())
 
print("\nMissing values by basin:")
print(combined_df.iloc[:, 1:].isna().sum())
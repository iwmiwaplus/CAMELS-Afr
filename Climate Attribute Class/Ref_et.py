import xarray as xr
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
import os

# -----------------------------
# Input paths
# -----------------------------
base_folder = r"...\CAMELS"
shapefile = r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"
output_folder = r"...\Ref_ET_L1"
output_filename = "CAMELS_Afr_Climate_Daily_L1_ref_et_mean.csv"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Load and Prep Basins
# -----------------------------
print("Loading shapefile...")
basins = gpd.read_file(shapefile)
if basins.crs != "EPSG:4326":
    basins = basins.to_crs("EPSG:4326")

basins = basins.sort_values('Basin_ID').reset_index(drop=True)
basin_ids = basins['Basin_ID'].astype(str).tolist()

# -----------------------------
# Get Spatial Info from NetCDF
# -----------------------------
year_folders = sorted([f for f in os.listdir(base_folder) if f.startswith('reference_evapotranspiration_')])
first_year_path = os.path.join(base_folder, year_folders[0])
sample_nc = os.path.join(first_year_path, [f for f in os.listdir(first_year_path) if f.endswith('.nc')][0])

with xr.open_dataset(sample_nc) as ds:
    # Fix Longitude if it's 0-360
    if ds.lon.max() > 180:
        print("Detected 0-360 longitude. Adjusting to -180-180...")
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    
    lon = ds.lon.values
    lat = ds.lat.values
    
    # Spatial Debug Info
    print("-" * 30)
    print("SPATIAL DEBUG INFO:")
    print(f"NetCDF Lon Range: {lon.min():.2f} to {lon.max():.2f}")
    print(f"NetCDF Lat Range: {lat.min():.2f} to {lat.max():.2f}")
    bbox = basins.total_bounds # [minx, miny, maxx, maxy]
    print(f"Basins Lon Range: {bbox[0]:.2f} to {bbox[2]:.2f}")
    print(f"Basins Lat Range: {bbox[1]:.2f} to {bbox[3]:.2f}")
    print("-" * 30)

    # Calculate Resolution
    res_lon = (lon.max() - lon.min()) / (len(lon) - 1)
    res_lat = (lat.max() - lat.min()) / (len(lat) - 1)
    
    # We use from_bounds for safety. It needs (west, south, east, north)
    # This assumes the data array will be oriented Top-Down (North at index 0)
    transform = rasterio.transform.from_bounds(
        lon.min() - res_lon/2, 
        lat.min() - res_lat/2, 
        lon.max() + res_lon/2, 
        lat.max() + res_lat/2, 
        len(lon), len(lat)
    )
    out_shape = (len(lat), len(lon))
    is_lat_descending = lat[0] > lat[-1]

print(f"Pre-calculating masks for {len(basin_ids)} basins...")
basin_masks = []
for idx, basin in basins.iterrows():
    mask = geometry_mask([basin['geometry']],
                         transform=transform,
                         invert=True,
                         out_shape=out_shape,
                         all_touched=True) 
    basin_masks.append(mask)

# -----------------------------
# Processing loop
# -----------------------------
all_rows = []

print("Starting daily processing...")
for year_folder in year_folders:
    year_path = os.path.join(base_folder, year_folder)
    print(f"  Processing folder: {year_folder}")
    
    nc_files = sorted([f for f in os.listdir(year_path) if f.endswith('.nc')])
    
    for nc_file in nc_files:
        file_path = os.path.join(year_path, nc_file)
        
        try:
            with xr.open_dataset(file_path) as ds:
                # Apply the same longitude fix to every file
                if ds.lon.max() > 180:
                    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
                
                date_val = pd.to_datetime(ds.time.values[0]).strftime('%m/%d/%Y')
                et_raw = ds['ReferenceET_PenmanMonteith_FAO56'].values[0]
                et_raw[et_raw == -9999] = np.nan
                
                # IMPORTANT: If the NetCDF is South-to-North, but our transform is North-to-South
                # we must flip the data array vertically to match the mask.
                if not is_lat_descending:
                    et_raw = np.flipud(et_raw)

                current_day_data = {'Date': date_val}
                found_count = 0
                
                for i, b_id in enumerate(basin_ids):
                    pixel_values = et_raw[basin_masks[i]]
                    if pixel_values.size > 0 and not np.all(np.isnan(pixel_values)):
                        mean_val = np.nanmean(pixel_values)
                        found_count += 1
                    else:
                        mean_val = np.nan
                    current_day_data[b_id] = mean_val
                
                all_rows.append(current_day_data)
                
                if len(all_rows) == 1:
                    print(f"    Check: First file processed. Valid data found for {found_count} out of {len(basin_ids)} basins.")

        except Exception as e:
            print(f"    Error processing {nc_file}: {e}")

# -----------------------------
# Save Results
# -----------------------------
print("Saving to CSV...")
final_df = pd.DataFrame(all_rows)
final_df = final_df[['Date'] + basin_ids]
final_df.to_csv(os.path.join(output_folder, output_filename), index=False)
print("Process complete.")
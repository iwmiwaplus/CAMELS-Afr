"""
HiHydro v2.0 Soil Attribute Extraction Module
---------------------------------------------
This module extracts spatial statistics from HiHydro v2.0 raster datasets.
Automatically handles single files or directories containing multiple depth levels.

These variables has 6 columns for each depth
(e.g: ksat_0-5cm    ksat_5-15cm ksat_15-30cm    ksat_30-60cm    ksat_60-100cm   ksat_100-200cm)
ksat -  Saturated hydraulic conductivity (Ksat)
FC -    water content at field capacity (WCpF2) 
THS -   Saturated water content (WCsat) 
WP -    water content at wilting point (WCpF4.2)
pct_gravel -    Volumetric coarse fragments content


Remaining variables
hsg_major   Major hydrological soil group
hsg_prop    Proportion of hydrological soil group
(hsg_A_prop hsg_B_prop  hsg_C_prop  hsg_D_prop  hsg_A/D_prop    hsg_B/D_prop    hsg_C/D_prop) 


Author: Kavishka Palihena
"""

import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.ops import transform
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
SHAPEFILE_PATH = r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"
BASIN_ID_FIELD = "Basin_ID"  # Change this to match the ID column in your shapefile
OUTPUT_CSV     = r"...\CAMELS_Afr_Soil_Static_L1.csv"

# You can now provide EITHER a direct path to a .tif file OR a folder containing .tif files.
# The script will automatically detect which it is.
INPUT_PATHS = {
    "HSG":  r"...\HSG",  # Script will find the .tif inside
    "ksat": r"...\Ksat", # Script will loop through depth .tifs
    "FC":   r"...\WCpF2",
    "THS":  r"...\WCsat",
    "WP":   r"...\WCpF4.2"
}

def get_tif_files(base_path):
    """Helper function to safely get all .tif files whether a folder or file is provided."""
    if os.path.isfile(base_path) and base_path.endswith('.tif'):
        return [base_path]
    elif os.path.isdir(base_path):
        return sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.tif')])
    return []

def process_soil_data(shapefile_path, basin_id_field, input_paths, output_csv):
    
    # Configuration
    equal_area_crs = "ESRI:102022"
    hsg_val_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'A/D': 14, 'B/D': 24, 'C/D': 34}
    scale_factor = 0.0001
    
    # Initialize GeoDataFrame
    basins = gpd.read_file(shapefile_path).to_crs(equal_area_crs)
    
    if basin_id_field not in basins.columns:
        raise ValueError(f"Column '{basin_id_field}' not found. Available: {list(basins.columns)}")
        
    basins['Total_Area_m2'] = basins.geometry.area
    
    # Map the original shapefile ID to the standardized 'Basin_ID' column
    final_df = pd.DataFrame({'Basin_ID': basins[basin_id_field]})
    
    # ---------------------------------------------------------
    # 1. HSG Extraction
    # ---------------------------------------------------------
    hsg_files = get_tif_files(input_paths.get("HSG", ""))
    if hsg_files:
        hsg_path = hsg_files[0] # Assume only one HSG raster is needed
        print(f"Processing Hydrologic Soil Groups from: {os.path.basename(hsg_path)}")
        
        with rasterio.open(hsg_path) as src:
            project = pyproj.Transformer.from_crs(equal_area_crs, src.crs, always_xy=True).transform
            hsg_results = []
            
            for _, row in tqdm(basins.iterrows(), total=len(basins), desc="Extracting HSG"):
                geom_raster = transform(project, row['geometry'])
                try:
                    win = rasterio.windows.from_bounds(*geom_raster.bounds, transform=src.transform)
                    data = src.read(1, window=win)
                    mask_arr = geometry_mask([geom_raster], transform=src.window_transform(win), invert=True, out_shape=data.shape)
                    counts = {cls: np.sum((data[mask_arr] == val)) for cls, val in hsg_val_map.items()}
                except ValueError:
                    counts = {cls: 0 for cls in hsg_val_map.keys()}
                    
                row_res = {f'hsg_{cls}_perc': (count * 250 * 250 / row['Total_Area_m2']) * 100 
                           for cls, count in counts.items()}
                hsg_results.append(row_res)
                
        hsg_df = pd.DataFrame(hsg_results)
        perc_cols = [c for c in hsg_df.columns if c.endswith('_perc')]
        if not hsg_df.empty:
            hsg_df['hsg_major'] = hsg_df[perc_cols].idxmax(axis=1).str.replace('hsg_', '', regex=False).str.replace('_perc', '', regex=False)
        final_df = pd.concat([final_df, hsg_df], axis=1)
    else:
        print("HSG raster not found. Skipping HSG extraction.")

    # ---------------------------------------------------------
    # 2. Continuous Attribute Extraction (ksat, FC, THS, WP with Depth Levels)
    # ---------------------------------------------------------
    for attr, path in input_paths.items():
        if attr.upper() == "HSG": 
            continue
            
        tif_files = get_tif_files(path)
        if not tif_files:
            print(f"Warning: No .tif files found for {attr} at {path}. Skipping.")
            continue
            
        for r_path in tif_files:
            # Extract depth name from filename (e.g., "Ksat_0-5cm_M_250m.tif" -> "0-5cm")
            base = os.path.splitext(os.path.basename(r_path))[0]
            parts = base.split("_")
            depth_name = parts[1] if len(parts) > 1 else "level"
            col_name = f"{attr}_{depth_name}"
            
            with rasterio.open(r_path) as src:
                project_hihydro = pyproj.Transformer.from_crs(equal_area_crs, src.crs, always_xy=True).transform
                vals = []
                
                for _, row in tqdm(basins.iterrows(), total=len(basins), desc=f"Extracting {col_name}"):
                    try:
                        geom_hihydro = transform(project_hihydro, row['geometry'])
                        out_img, _ = mask(src, [geom_hihydro], crop=True)
                        data = out_img[0].astype(float)
                        
                        if src.nodata is not None:
                            data = np.where(data == src.nodata, np.nan, data)
                            
                        with np.errstate(invalid='ignore'):
                            if np.all(np.isnan(data)):
                                vals.append(np.nan)
                            else:
                                vals.append(np.nanmean(data) * scale_factor)
                                
                    except ValueError:
                        vals.append(np.nan)
                        
                final_df[col_name] = vals

    # ---------------------------------------------------------
    # 3. Save Results
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"\n Extraction complete. Data saved to {output_csv}")

if __name__ == "__main__":
    process_soil_data(
        shapefile_path=SHAPEFILE_PATH,
        basin_id_field=BASIN_ID_FIELD,
        input_paths=INPUT_PATHS,
        output_csv=OUTPUT_CSV
    )
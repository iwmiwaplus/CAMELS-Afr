"""
CAMELS-Africa Land Cover MODIS MCD12Q1 Combined Pipeline

Runs zonal land-cover extraction, then processes those basin CSV files
and restructures them into annual basin-wise matrices.

Update the directory paths below to match your local folders.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
import geopandas as gpd
from tqdm import tqdm

# =====================================================================
# SPATIAL LEVEL GENERALIZATION PARAMETERS
# =====================================================================
SPATIAL_LEVEL_NAME = "Level_01"  # Examples: "Level_01", "Level_02", "Level_04"

# =====================================================================
# REPOSITORY DIRECTORIES
# Wrapped in Path() so that .exists(), .glob(), and / operators work
# =====================================================================
REPO_ROOT            = Path(__file__).resolve().parent

# Input from Part 1
ALBERS_FOLDER        = Path(r"...\mosaic_albers")
BASIN_SHP            = Path(r"...\CAMELS_Afr_L1_Basin_Boundaries.shp")

# Outputs
EXTRACTED_CSV_FOLDER = Path(r"...\MCD12Q1")


# Spatial Processing Constants
TARGET_CRS = "ESRI:102022"  # Africa Albers Equal Area Coordinate System

# IGBP Land Cover Class Lookup Map
CLASS_DICT = {
    1: "EvergreenNeedleleafForests", 
    2: "EvergreenBroadleafForests", 
    3: "DeciduousNeedleleafForests", 
    4: "DeciduousBroadleafForests", 
    5: "MixedForests", 
    6: "ClosedShrublands", 
    7: "OpenShrublands",
    8: "WoodySavannas", 
    9: "Savannas", 
    10: "Grasslands", 
    11: "PermanentWetlands", 
    12: "Croplands", 
    13: "UrbanAndBuiltUpLands", 
    14: "CroplandNaturalVegetationMosaics", 
    15: "PermanentSnowAndIce",
    16: "Barren", 
    17: "WaterBodies", 
    255: "Unclassified"
}

def setup_directories():
    """Initializes the output directory framework safely."""
    EXTRACTED_CSV_FOLDER.mkdir(parents=True, exist_ok=True)

def extract_basin_landcover():
    """Calculates land cover fractional distributions across basin polygon layers."""
    print(f"\nParsing vector boundaries for {SPATIAL_LEVEL_NAME}...")
    if not BASIN_SHP.exists():
        print(f"Error: Boundary shapefile not found at {BASIN_SHP}")
        return

    # Align vectors to projection
    basins = gpd.read_file(BASIN_SHP).to_crs(TARGET_CRS)
    
    # Extract geometries into micro-dictionaries to minimize object lookup overhead
    basin_data = [
        {"id": row["Basin_ID"], "geometry": row["geometry"], "bounds": row["geometry"].bounds}
        for _, row in basins.iterrows()
    ]
    
    all_results = {b["id"]: [] for b in basin_data}
    tif_files = sorted(ALBERS_FOLDER.glob("*.tif"))
    classes = list(CLASS_DICT.values())
    
    if not tif_files:
        print(f"Error: No Albers tif files found in {ALBERS_FOLDER}. Run Part 1 first.")
        return
        
    for tif_file in tqdm(tif_files, desc="Processing Spatial Years"):
        match = re.search(r"(\d{4})", tif_file.name)
        if not match:
            continue
        year = int(match.group(1))
        
        with rasterio.open(tif_file) as src:
            for basin in basin_data:
                minx, miny, maxx, maxy = basin["bounds"]
                
                # Windowed read: isolates RAM footprint to the basin boundary envelope
                window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                data = src.read(1, window=window, boundless=True, fill_value=255)
                
                # Spatial geometry array masking
                mask_arr = geometry_mask(
                    [basin["geometry"]], 
                    transform=src.window_transform(window), 
                    invert=True, 
                    out_shape=data.shape
                )
                masked_data = data[mask_arr]
                
                if len(masked_data) == 0:
                    continue
                
                # Vectorized calculations
                unique, counts = np.unique(masked_data, return_counts=True)
                total_pixels = counts.sum()
                
                row_dict = {"Year": year}
                row_dict.update({c: 0.0 for c in classes})
                
                for u, c in zip(unique, counts):
                    if u in CLASS_DICT:
                        row_dict[CLASS_DICT[u]] = c / total_pixels
                        
                all_results[basin["id"]].append(row_dict)

    print("\nExporting localized basin files to CSV...")
    for basin_id, records in tqdm(all_results.items(), desc="Writing Files"):
        if records:
            df = pd.DataFrame(records).sort_values("Year")
            cols = ["Year"] + [c for c in classes if c in df.columns]
            df[cols].to_csv(EXTRACTED_CSV_FOLDER / f"{basin_id}.csv", index=False)

# =====================================================================
# RESTRUCTURING STAGE
# =====================================================================
"""
CAMELS-Africa Land Cover MODIS MCD12Q1 Extraction Pipeline
Part 2: Attribute Post-Processing and Level Restructuring

This script imports localized basin profiles, resolves ecological equations,
normalizes feature classifications, runs contributor checks, and aggregates 
cross-basin matrix tables.

Output:
-------
For each LandCover variable:
    - Annual time series (rows = Year)
    - Basin-wise values (columns = Basin_IDs)

Author: Kavishka Palihena
Date: 5/15/2026
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from tqdm import tqdm

# =====================================================================
# SPATIAL LEVEL GENERALIZATION PARAMETERS
# Change these to match the configurations set up in Part 1
# =====================================================================
OUTPUT_FILE_LEVEL_TAG = "L1"       # Keep consistent with SPATIAL_LEVEL_NAME = "Level_01"

# =====================================================================
# RELATIVE REPOSITORY DIRECTORIES
# =====================================================================
REPO_ROOT     = Path(__file__).resolve().parent

INPUT_FOLDER  = EXTRACTED_CSV_FOLDER  # Output of the extraction stage in this same script
OUTPUT_FOLDER = Path(r"........\Annual_L1")

# ECOSYSTEM DEEP-ROOT COEFFICIENTS (Zeng, 2001)
ZENG_PARAMS = {
    "EvergreenNeedleleafForests": (6.706, 2.175),
    "EvergreenBroadleafForests": (7.344, 1.303),
    "DeciduousNeedleleafForests": (7.066, 1.953),
    "DeciduousBroadleafForests": (5.990, 1.955),
    "MixedForests": (4.453, 1.631),
    "ClosedShrublands": (6.326, 1.567),
    "OpenShrublands": (7.718, 1.262),
    "WoodySavannas": (7.604, 2.300),
    "Savannas": (8.235, 1.627),
    "Grasslands": (10.74, 2.608),
    "Croplands": (5.558, 2.614),
    "UrbanAndBuiltUpLands": (5.558, 2.614),
    "CroplandNaturalVegetationMosaics": (4.372, 0.978),
    "Barren": (2.614, 1.567),
    "WaterBodies": (0.978, 1.631)
}

# DATABASE FIELD STANDARDIZATION MAP
RENAME_MAP = {
    "EvergreenNeedleleafForests": "evergreen_needleleaf_perc",
    "EvergreenBroadleafForests": "evergreen_broadleaf_perc",
    "DeciduousNeedleleafForests": "deciduous_needleleaf_perc",
    "DeciduousBroadleafForests": "deciduous_broadleaf_perc",
    "MixedForests": "mixed_forest_perc",
    "ClosedShrublands": "closed_shrubland_perc",
    "OpenShrublands": "open_shrubland_perc",
    "WoodySavannas": "woody_savanna_perc",
    "Savannas": "savanna_perc",
    "Grasslands": "cover_grass_perc",
    "PermanentWetlands": "wetland_perc",
    "Croplands": "cover_crops_perc",
    "UrbanAndBuiltUpLands": "urban_perc",
    "CroplandNaturalVegetationMosaics": "mosaic_perc",
    "PermanentSnowAndIce": "snow_ice_perc",
    "Barren": "bare_perc",
    "WaterBodies": "lake_perc"
}


def get_depth(a, b, p):
    """Numerically determines target vertical soil boundaries at a given percentile profile."""
    func = lambda d: 1 - 0.5 * (np.exp(-a * d) + np.exp(-b * d)) - p
    return fsolve(func, 1.0)[0]


# Precalculate ecosystem limits to prevent nested runtime calculations
depth_50 = {lc: get_depth(a, b, 0.5) for lc, (a, b) in ZENG_PARAMS.items()}
depth_99 = {lc: get_depth(a, b, 0.99) for lc, (a, b) in ZENG_PARAMS.items()}


def compute_root_depths(df):
    """Applies a weighted average model to determine annual rooting limits."""
    rd50_list, rd99_list = [], []

    for _, row in df.iterrows():
        rd50 = sum(row[lc] * depth_50[lc] for lc in ZENG_PARAMS if lc in df.columns)
        rd99 = sum(row[lc] * depth_99[lc] for lc in ZENG_PARAMS if lc in df.columns)
        rd50_list.append(rd50)
        rd99_list.append(rd99)

    df["root_depth_50"] = rd50_list
    df["root_depth_99"] = rd99_list
    return df


def convert_and_rename(df):
    """Transforms tracking fields from fractions to percentage values, then maps database headers."""
    cols_to_convert = [c for c in RENAME_MAP.keys() if c in df.columns]
    df[cols_to_convert] = df[cols_to_convert] * 100
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    return df


def enhance_landcover(df):
    """Compiles thematic aggregates, calculates spatial dominance, and executes rollback validation rules."""
    forest_cols = [
        "evergreen_needleleaf_perc", "evergreen_broadleaf_perc",
        "deciduous_needleleaf_perc", "deciduous_broadleaf_perc", "mixed_forest_perc"
    ]
    shrub_cols = ["closed_shrubland_perc", "open_shrubland_perc"]
    
    exist_forest = [c for c in forest_cols if c in df.columns]
    exist_shrubs = [c for c in shrub_cols if c in df.columns]

    if exist_forest:
        df["forest_perc"] = df[exist_forest].sum(axis=1, skipna=True)
    if exist_shrubs:
        df["cover_shrub_perc"] = df[exist_shrubs].sum(axis=1, skipna=True)

    # Filter out helper and specialized metadata layers from dominance evaluations
    exclude_cols = ["Year", "system:index", ".geo", "Unclassified", "snow_ice_perc"] + exist_forest + exist_shrubs
    candidate_cols = [c for c in df.columns if c not in exclude_cols and c.endswith("_perc")]

    if candidate_cols:
        df["lc_dom"] = df[candidate_cols].idxmax(axis=1)
        df["dom_land_cover_perc"] = df[candidate_cols].max(axis=1)

    def resolve_sole_contributor(row):
        """Rollback logic: evaluates if an aggregate category was single-handedly driven by one sub-class."""
        dom_class = row.get("lc_dom")
        
        if dom_class == "forest_perc" and exist_forest:
            max_sub_col = row[exist_forest].astype(float).idxmax()
            if row[max_sub_col] == row["forest_perc"] and row["forest_perc"] > 0:
                return max_sub_col
                
        elif dom_class == "cover_shrub_perc" and exist_shrubs:
            max_sub_col = row[exist_shrubs].astype(float).idxmax()
            if row[max_sub_col] == row["cover_shrub_perc"] and row["cover_shrub_perc"] > 0:
                return max_sub_col
                
        return dom_class

    if "lc_dom" in df.columns:
        df["lc_dom"] = df.apply(resolve_sole_contributor, axis=1)
        # Strip suffix to standardize feature name fields
        df["lc_dom"] = df["lc_dom"].astype(str).str.replace("_perc", "", regex=False)

    df = df.drop(columns=["Unclassified", "snow_ice_perc"], errors="ignore")

    # Column ordering template
    main_cols = ["Year", "lc_dom", "dom_land_cover_perc", "forest_perc", "cover_shrub_perc"]
    ordered_cols = [c for c in main_cols if c in df.columns] + [c for c in df.columns if c not in main_cols]
    
    return df[ordered_cols]


def restructure_outputs():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_FOLDER.exists():
        print(f"Error: Input data folder not found at {INPUT_FOLDER}. Run Part 1 first.")
        return
        
    all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(f"\nDiscovered {len(all_files)} basin profiles for {SPATIAL_LEVEL_NAME}. Processing math transformations...")
    variable_store = {}

    for filename in tqdm(all_files, desc="Processing Attribute Analytics"):
        basin_id = os.path.splitext(os.path.basename(filename))[0]
        
        try:
            df = pd.read_csv(filename)
            
            # Sequentially process and transform data in-memory
            df = compute_root_depths(df)
            df = convert_and_rename(df)
            df = enhance_landcover(df)
            
            df.set_index('Year', inplace=True)
            
            # Restructure from long to side-by-side matrices
            for variable_name in df.columns:
                if variable_name not in variable_store:
                    variable_store[variable_name] = []
                series = df[variable_name].rename(basin_id)
                variable_store[variable_name].append(series)
                
        except Exception as e:
            print(f"Error handling file {filename}: {str(e)}")
            continue

    print(f"\nStructural mapping complete. Writing master sheets out to: {OUTPUT_FOLDER}")
    for variable_name, series_list in tqdm(variable_store.items(), desc="Saving Restructured Outputs"):
        new_df = pd.concat(series_list, axis=1).sort_index(axis=1)
        
        safe_var_name = str(variable_name).replace(" ", "_").replace("/", "-")
        out_filename = f"CAMELS_Afr_LandCover_Annual_{OUTPUT_FILE_LEVEL_TAG}_{safe_var_name}.csv"
        
        new_df.to_csv(OUTPUT_FOLDER / out_filename)

if __name__ == "__main__":
    print("======================================================")
    print("Starting combined CAMELS-Africa land-cover pipeline")
    print("======================================================")

    setup_directories()
    extract_basin_landcover()

    # The restructuring stage reads the CSV files created above.
    if INPUT_FOLDER.exists() and list(INPUT_FOLDER.glob("*.csv")):
        restructure_outputs()
        print("\nCombined pipeline finalized successfully!")
    else:
        print(f"\nNo extracted CSV files found in {INPUT_FOLDER}. Restructuring skipped.")

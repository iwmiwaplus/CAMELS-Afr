import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
 
# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
BASIN_SHP = os.path.join(
    BASE_DIR,
    "CAMELS_Afr_L1_Basin_Boundaries.shp"
)
 
OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "CAMELS_Afr_Climate_Daily_L1_win_mean.csv"
)
 
# --------------------------------------------------
# 2. FIND AND MATCH U/V FILES BY YEAR
# --------------------------------------------------
u_files = glob.glob(
    os.path.join(BASE_DIR, "10m_u_component_of_wind_*_daily.nc")
)
 
v_files = glob.glob(
    os.path.join(BASE_DIR, "10m_v_component_of_wind_*_daily.nc")
)
 
 
def extract_year(filepath):
    match = re.search(
        r"_(\d{4})_daily\.nc$",
        os.path.basename(filepath)
    )
    return match.group(1) if match else None
 
 
u_by_year = {
    extract_year(path): path
    for path in u_files
    if extract_year(path) is not None
}
 
v_by_year = {
    extract_year(path): path
    for path in v_files
    if extract_year(path) is not None
}
 
if not u_by_year or not v_by_year:
    raise FileNotFoundError(
        "Could not find the required U and V daily NetCDF files "
        "in the script folder."
    )
 
if set(u_by_year) != set(v_by_year):
    raise ValueError(
        "U and V files do not have matching years.\n"
        f"U years: {sorted(u_by_year)}\n"
        f"V years: {sorted(v_by_year)}"
    )
 
years = sorted(u_by_year)
 
print("Years found:", ", ".join(years))
 
for year in years:
    print(f"U ({year}): {os.path.basename(u_by_year[year])}")
    print(f"V ({year}): {os.path.basename(v_by_year[year])}")
 
 
# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------
def find_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
 
    for name in ds.coords:
        standard = str(
            ds[name].attrs.get("standard_name", "")
        ).lower()
 
        if any(c.lower() in name.lower() for c in candidates):
            return name
 
        if standard in [c.lower() for c in candidates]:
            return name
 
    raise ValueError(
        f"Could not find coordinate {candidates}. "
        f"Available coordinates: {list(ds.coords)}"
    )
 
 
def find_data_variable(ds):
    if not ds.data_vars:
        raise ValueError("No data variables found in NetCDF.")
 
    return list(ds.data_vars)[0]
 
 
def extract_weighted_mean(data, geometry):
    """
    Calculate the daily basin mean using
    cosine-latitude weighting method.
    """
 
    minx, miny, maxx, maxy = geometry.bounds
 
    # Select the basin bounding box first
    if data["latitude"].values[0] > data["latitude"].values[-1]:
        lat_slice = slice(maxy, miny)
    else:
        lat_slice = slice(miny, maxy)
 
    subset = data.sel(
        latitude=lat_slice,
        longitude=slice(minx, maxx)
    )
 
    if (
        subset.sizes.get("latitude", 0) == 0
        or subset.sizes.get("longitude", 0) == 0
    ):
        raise ValueError("No grid cells within the basin bounding box.")
 
    # Clip the raster to the basin boundary
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
 
        clipped = subset.rio.clip(
            [geometry],
            crs="EPSG:4326",
            drop=True,
            all_touched=True
        )
 
    if (
        clipped.sizes.get("latitude", 0) == 0
        or clipped.sizes.get("longitude", 0) == 0
    ):
        raise ValueError("No raster cells intersect the basin.")
 
    # Original cosine-latitude weighting method
    weights = np.cos(
        np.deg2rad(clipped["latitude"])
    )
 
    weights = weights / weights.sum()
    weights.name = "weights"
 
    # Calculate weighted mean
    mean_values = clipped.weighted(weights).mean(
        dim=("latitude", "longitude"),
        skipna=True
    )
 
    return np.asarray(mean_values.values, dtype=float)
 
 
# --------------------------------------------------
# 4. LOAD AND PREPARE BASIN SHAPEFILE
# --------------------------------------------------
if not os.path.isfile(BASIN_SHP):
    raise FileNotFoundError(
        f"Shapefile not found:\n{BASIN_SHP}"
    )
 
basins = gpd.read_file(BASIN_SHP)
 
if basins.empty:
    raise ValueError("The basin shapefile contains no features.")
 
if basins.crs is None:
    raise ValueError(
        "The shapefile CRS is missing. "
        "Check the original CRS before assigning one."
    )
 
basins = basins[
    basins.geometry.notnull()
    & ~basins.geometry.is_empty
].copy()
 
basins["geometry"] = basins.geometry.buffer(0)
basins = basins.to_crs("EPSG:4326")
 
# Use readable basin names
id_field = "Basin_ID"
 
if id_field not in basins.columns:
    raise ValueError(
        f"{id_field} not found. Available columns: "
        f"{basins.columns.tolist()}"
    )
 
if basins[id_field].isna().any():
    raise ValueError("Missing basin names found in Basin_ID.")
 
basins[id_field] = basins[id_field].astype(str)
 
if basins[id_field].duplicated().any():
    raise ValueError("Duplicate basin names found in Basin_ID.")
 
 
print(f"\nNumber of basins: {len(basins)}")
print("Basin names sorted alphabetically.")
 
 
# --------------------------------------------------
# 5. PROCESS ALL YEARS
# --------------------------------------------------
yearly_results = []
 
for year in years:
    print(f"\nProcessing year: {year}")
 
    with xr.open_dataset(u_by_year[year]) as u_ds, \
         xr.open_dataset(v_by_year[year]) as v_ds:
 
        # Identify variables and coordinates
        u_var = find_data_variable(u_ds)
        v_var = find_data_variable(v_ds)
 
        u_time = find_coord(u_ds, ["time", "valid_time", "date"])
        v_time = find_coord(v_ds, ["time", "valid_time", "date"])
 
        u_lat = find_coord(u_ds, ["latitude", "lat"])
        v_lat = find_coord(v_ds, ["latitude", "lat"])
 
        u_lon = find_coord(u_ds, ["longitude", "lon"])
        v_lon = find_coord(v_ds, ["longitude", "lon"])
 
        # Standardise coordinate names
        u = u_ds[u_var].rename({
            u_time: "time",
            u_lat: "latitude",
            u_lon: "longitude"
        })
 
        v = v_ds[v_var].rename({
            v_time: "time",
            v_lat: "latitude",
            v_lon: "longitude"
        })
 
        # Sort latitude consistently
        u = u.sortby("latitude", ascending=False)
        v = v.sortby("latitude", ascending=False)
 
        # Check that U and V grids match
        if not np.allclose(u.latitude.values, v.latitude.values):
            raise ValueError(
                f"U and V latitude grids do not match for {year}."
            )
 
        if not np.allclose(u.longitude.values, v.longitude.values):
            raise ValueError(
                f"U and V longitude grids do not match for {year}."
            )
 
        # Match dates between U and V
        u_dates = pd.DatetimeIndex(pd.to_datetime(u.time.values))
        v_dates = pd.DatetimeIndex(pd.to_datetime(v.time.values))
 
        common_dates = u_dates.intersection(v_dates).sort_values()
 
        if len(common_dates) == 0:
            raise ValueError(f"No matching dates found for {year}.")
 
        if not all(date.year == int(year) for date in common_dates):
            raise ValueError(
                f"Unexpected year found in files labelled {year}."
            )
 
        u = u.sel(time=common_dates)
        v = v.sel(time=common_dates)
 
        # Set spatial dimensions and CRS
        u = u.rio.set_spatial_dims(
            x_dim="longitude",
            y_dim="latitude"
        ).rio.write_crs("EPSG:4326")
 
        v = v.rio.set_spatial_dims(
            x_dim="longitude",
            y_dim="latitude"
        ).rio.write_crs("EPSG:4326")
 
        # Initialise output for this year
        output = {
            "Date": [
                f"{date.month}/{date.day}/{date.year}"
                for date in common_dates
            ]
        }
 
        # Process basins in alphabetical order
        for _, basin in basins.iterrows():
            basin_name = basin[id_field]
 
            print(f"  Processing basin: {basin_name}")
 
            u_mean = extract_weighted_mean(u, basin.geometry)
            v_mean = extract_weighted_mean(v, basin.geometry)
 
            # Vector mean: calculate wind speed from mean U and V
            wind_speed = np.sqrt(
                u_mean ** 2 + v_mean ** 2
            )
 
            if len(wind_speed) != len(common_dates):
                raise ValueError(
                    f"Date count mismatch for {basin_name}, {year}."
                )
 
            # Round to 3 decimal places, as in the original script
            output[basin_name] = wind_speed
 
        yearly_results.append(pd.DataFrame(output))
 
        print(
            f"Completed {year}: {len(common_dates)} dates processed."
        )
 
 
# --------------------------------------------------
# 6. COMBINE YEARS AND SAVE ONE CSV
# --------------------------------------------------
if not yearly_results:
    raise ValueError("No yearly results were generated.")
 
result = pd.concat(yearly_results, ignore_index=True)
 
# Sort rows chronologically
result["_sort_date"] = pd.to_datetime(
    result["Date"],
    format="%m/%d/%Y"
)
 
result = result.sort_values(
    "_sort_date"
).reset_index(drop=True)
 
# Check for duplicate dates
if result["_sort_date"].duplicated().any():
    duplicate_dates = result.loc[
        result["_sort_date"].duplicated(keep=False),
        "Date"
    ].unique()
 
    raise ValueError(
        "Duplicate dates found across input files: "
        f"{duplicate_dates.tolist()}"
    )
 
result = result.drop(columns="_sort_date")
 
# Save a single combined CSV
result.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8"
)
 
print("\nExtraction completed successfully.")
print(f"Output file: {OUTPUT_CSV}")
print(f"Years processed: {', '.join(years)}")
print(f"Number of dates: {len(result)}")
print(f"Number of basins: {len(result.columns) - 1}")
print("Basin columns: alphabetical order")
print("Date format: M/D/YYYY")
print(result.head())
 
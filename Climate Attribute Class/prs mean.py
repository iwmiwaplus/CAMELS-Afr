# ------------------------------------------------------------------
# Surface Pressure Extraction from NetCDF files to Basin CSV
#
# This script extracts basin-averaged surface pressure values from
# NetCDF files using area-weighted extraction.
#
# Each NetCDF file contains multiple time steps (valid_time).
# The script calculates the mean surface pressure for each basin
# polygon and saves the results in a CSV file.
#
# Author: Kasthuri Subramaniam
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Import required libraries
# ------------------------------------------------------------------
import os
import glob
import tempfile
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from exactextract import exact_extract
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin


# ------------------------------------------------------------------
# File paths (EDIT THESE PATHS IF NEEDED)
# ------------------------------------------------------------------
nc_path = "......\surface pressure"
shp_path = "........\CAMELS_Afr_L1_Basin_Boundaries 3\CAMELS_Afr_L1_Basin_Boundaries.shp"
out_csv = "..........\surface_pressure_sinai_lvl1.csv"


# ------------------------------------------------------------------
# Load basin shapefile
# ------------------------------------------------------------------
basins = gpd.read_file(shp_path)


print("Original basin CRS:", basins.crs)

if basins.crs is None:
    raise ValueError("The basin shapefile has no CRS defined.")

basins = basins.to_crs("EPSG:4326")

print("Reprojected basin CRS:", basins.crs)

# Fix possible geometry issues
basins["geometry"] = basins.buffer(0)

# Basin ID field
basin_ids = basins["Basin_ID"].astype(str).tolist()


# ------------------------------------------------------------------
# Function to process one NetCDF file
# ------------------------------------------------------------------
def process_year(nc_file):

    # -------------------------------
    # Read time dimension
    # -------------------------------
    nc = Dataset(nc_file)

    time_raw = nc.variables["valid_time"][:]
    time_units = nc.variables["valid_time"].units

    origin = time_units.replace("days since ", "")
    dates = pd.to_datetime(time_raw, unit="D", origin=pd.Timestamp(origin))

    nc.close()

    # -------------------------------
    # Load surface pressure variable
    # -------------------------------
    ds = xr.open_dataset(nc_file, chunks={"valid_time": 50})

    sp = ds["sp"]

    # Ensure correct dimension order
    sp = sp.transpose("valid_time", "latitude", "longitude")

    
    # -------------------------------
    # Extract basin means
    # -------------------------------
    results = []

    # Read coordinate values
    lons = sp["longitude"].values
    lats = sp["latitude"].values

    # Calculate pixel resolution
    xres = abs(float(np.median(np.diff(lons))))
    yres = abs(float(np.median(np.diff(lats))))

    # GeoTIFF coordinates use the upper-left pixel edge
    west = float(lons.min()) - xres / 2
    north = float(lats.max()) + yres / 2

    transform = from_origin(west, north, xres, yres)

    for i in range(sp.sizes["valid_time"]):

        print(f"  Processing time step {i + 1}/{sp.sizes['valid_time']}")

        layer = sp.isel(valid_time=i).compute()
        data = np.asarray(layer.values, dtype=np.float32)

        # Ensure latitude runs from north to south
        if lats[0] < lats[-1]:
            data = np.flipud(data)

        # Ensure longitude runs from west to east
        if lons[0] > lons[-1]:
            data = np.fliplr(data)

        # Convert missing values to raster NoData
        nodata = -9999.0
        data = np.where(np.isfinite(data), data, nodata)

        profile = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": nodata,
        }

        # Create an in-memory GeoTIFF
        
        # Save the current raster layer temporarily
        with tempfile.NamedTemporaryFile(
            suffix=".tif", delete=False
        ) as tmp:
            temp_path = tmp.name

        try:
            with rasterio.open(temp_path, "w", **profile) as raster:
                raster.write(data, 1)

            # Pass the raster file path to exactextract
            vals = exact_extract(
                temp_path,
                basins,
                ["mean"]
            )

            means = [
                v["properties"]["mean"]
                for v in vals
            ]

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        results.append(means)

    # -------------------------------
    # Convert results to dataframe
    # -------------------------------
    df = pd.DataFrame(results, columns=basin_ids)

    # Format Date column
    df.insert(0, "Date", dates.day.astype(str) + "/" + dates.month.astype(str) + "/" + dates.year.astype(str))

    return df


# ------------------------------------------------------------------
# Process all NetCDF files
# ------------------------------------------------------------------
nc_files = sorted(glob.glob(os.path.join(nc_path, "*.nc")))

all_results = []

for f in nc_files:
    print(f"Processing: {os.path.basename(f)}")
    all_results.append(process_year(f))

final_df = pd.concat(all_results, ignore_index=True)


# ------------------------------------------------------------------
# Save final dataset
# ------------------------------------------------------------------
final_df.to_csv(out_csv, index=False, float_format="%.2f")

print("Extraction completed successfully.")
print(f"Output saved to: {out_csv}")
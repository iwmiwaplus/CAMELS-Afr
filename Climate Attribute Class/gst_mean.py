"""
Daily Skin Temperature (SKT) Extraction from ERA5-Land NetCDF Files
===================================================================
- Reads daily .nc files.
- Extracts mean skin temperature for each basin.
- Converts Kelvin to Celsius.
- Formats dates as M/D/YYYY without leading zeros.
- Saves daily results to a CSV file.

Required libraries:
    pip install netCDF4 geopandas rasterio numpy pandas
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc4

from rasterio.transform import from_origin
from rasterio.features import geometry_mask

warnings.filterwarnings("ignore")


# ============================================================
# 1. FILE PATHS - CHANGE THESE TO YOUR OWN PATHS
# ============================================================


NC_DIR = r"...\Skin_temp_era5Land"  # folder with skin temperature files

SHP_PATH =  r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"
    
OUT_CSV = r"...\CAMELS_Afr_Sinai_Climate_Daily_L4_gst_mean.csv"


ID_FIELD = "Basin_ID"
KELVIN_OFFSET = 273.15


# ============================================================
# 2. READ BASIN SHAPEFILE
# ============================================================

print("Reading basin shapefile...")

basins = gpd.read_file(SHP_PATH)

if basins.crs is None:
    raise ValueError(
        "The basin shapefile has no CRS. "
        "Please define its correct CRS in QGIS first."
    )

# Convert basin coordinates to latitude/longitude
basins = basins.to_crs("EPSG:4326")

if ID_FIELD not in basins.columns:
    raise ValueError(
        f"'{ID_FIELD}' not found in the shapefile. "
        f"Available columns: {list(basins.columns)}"
    )

if basins.empty:
    raise ValueError("The basin shapefile contains no features.")

basin_ids = basins[ID_FIELD].tolist()


# ============================================================
# 3. FIND NETCDF FILES
# ============================================================

nc_files = sorted(
    glob.glob(os.path.join(NC_DIR, "*.nc"))
)

if not nc_files:
    raise FileNotFoundError(
        f"No .nc files found in: {NC_DIR}"
    )

print(f"Found {len(nc_files)} NetCDF file(s).")


# ============================================================
# 4. PROCESS ONE NETCDF FILE
# ============================================================

def process_nc(nc_path):

    print("Processing:", os.path.basename(nc_path))

    with nc4.Dataset(nc_path) as ds:

        # Read dates
        time_var = ds.variables["valid_time"]

        dates = nc4.num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(
                time_var, "calendar", "standard"
            ),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True
        )

        dates = pd.to_datetime(
            [str(d) for d in np.asarray(dates).ravel()]
        ).normalize()

        # Read latitude and longitude
        lats = np.asarray(
            ds.variables["latitude"][:]
        )

        lons = np.asarray(
            ds.variables["longitude"][:]
        )

        # Read skin temperature
        skt_var = ds.variables["skt"]
        original_dims = list(skt_var.dimensions)

        skt_raw = np.ma.filled(
            np.ma.asarray(
                skt_var[:],
                dtype=np.float64
            ),
            np.nan
        )

        # Mask fill and missing values
        for attr in ("_FillValue", "missing_value"):
            value = getattr(skt_var, attr, None)

            if value is not None:
                skt_raw[skt_raw == value] = np.nan

    # Remove dimensions whose size is 1
    keep = [
        i for i, size in enumerate(skt_raw.shape)
        if size > 1
    ]

    skt_raw = np.squeeze(skt_raw)

    dims = [
        original_dims[i] for i in keep
    ]

    # Identify time, latitude and longitude axes
    time_dim = next(
        (d for d in dims if d in ("valid_time", "time")),
        None
    )

    lat_dim = next(
        (d for d in dims if d in ("latitude", "lat")),
        None
    )

    lon_dim = next(
        (d for d in dims if d in ("longitude", "lon")),
        None
    )

    if time_dim and lat_dim and lon_dim:

        order = [
            dims.index(time_dim),
            dims.index(lat_dim),
            dims.index(lon_dim)
        ]

    else:
        expected = (
            len(dates),
            len(lats),
            len(lons)
        )

        if skt_raw.shape != expected:
            raise ValueError(
                f"Cannot identify SKT dimensions in {nc_path}. "
                f"Shape: {skt_raw.shape}; expected: {expected}"
            )

        order = [0, 1, 2]

    # Reorder to (time, latitude, longitude)
    skt = np.transpose(skt_raw, order)

    if skt.shape != (
        len(dates), len(lats), len(lons)
    ):
        raise ValueError(
            f"Unexpected SKT shape: {skt.shape}"
        )

    # Remove invalid and extremely large values
    skt[~np.isfinite(skt)] = np.nan
    skt[skt > 1e30] = np.nan

    # ========================================================
    # 5. PREPARE RASTER GRID
    # ========================================================

    if len(lats) < 2 or len(lons) < 2:
        raise ValueError(
            "At least two latitude and longitude values "
            "are required to calculate pixel size."
        )

    dlat = abs(float(np.median(np.diff(lats))))
    dlon = abs(float(np.median(np.diff(lons))))

    # Ensure the first raster row is the northernmost row
    if lats[0] < lats[-1]:
        lats = lats[::-1]
        skt = skt[:, ::-1, :]

    # Ensure longitude runs from west to east
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        skt = skt[:, :, ::-1]

    xmin = float(lons[0]) - dlon / 2
    ymax = float(lats[0]) + dlat / 2

    transform = from_origin(
        xmin,
        ymax,
        dlon,
        dlat
    )

    # ========================================================
    # 6. CREATE A MASK FOR EACH BASIN
    # ========================================================

    basin_masks = []

    for geom in basins.geometry:

        mask = geometry_mask(
            [geom.__geo_interface__],
            out_shape=(len(lats), len(lons)),
            transform=transform,
            invert=True,
            all_touched=False
        )

        basin_masks.append(mask)

    # ========================================================
    # 7. EXTRACT DAILY MEAN TEMPERATURE
    # ========================================================

    results = np.full(
        (len(dates), len(basin_ids)),
        np.nan,
        dtype=np.float64
    )

    for t in range(len(dates)):

        daily_grid = skt[t]

        for i, mask in enumerate(basin_masks):

            # Select valid pixel centres inside the basin
            valid = mask & np.isfinite(daily_grid)

            if np.any(valid):
                results[t, i] = np.mean(
                    daily_grid[valid]
                )

    # ========================================================
    # 8. CONVERT KELVIN TO CELSIUS
    # ========================================================

    results -= KELVIN_OFFSET

    # ========================================================
    # 9. CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        results,
        columns=basin_ids
    )

    df.insert(
        0,
        "Date",
        dates.strftime("%m/%d/%Y")
    )

    return df


# ============================================================
# 10. PROCESS ALL FILES AND SAVE CSV
# ============================================================

def main():

    frames = []

    for nc_path in nc_files:
        df = process_nc(nc_path)
        frames.append(df)

    if not frames:
        raise ValueError("No NetCDF files were processed.")

    print("\nCombining daily results...")

    final = pd.concat(
        frames,
        ignore_index=True
    )

    # Convert Date strings to datetime values
    # The input date format is MM/DD/YYYY
    final["Date"] = pd.to_datetime(
        final["Date"],
        format="%m/%d/%Y",
        errors="coerce"
    )

    # Check for dates that could not be parsed
    if final["Date"].isna().any():
        invalid_count = int(final["Date"].isna().sum())
        raise ValueError(
            f"{invalid_count} date(s) could not be parsed. "
            "Please check the date values in the input files."
        )

    # Sort dates and remove duplicate dates
    final = (
        final.sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    # Format dates as M/D/YYYY without leading zeros
    # Example: 01/03/2025 becomes 1/3/2025
    final["Date"] = (
        final["Date"].dt.month.astype(str) + "/" +
        final["Date"].dt.day.astype(str) + "/" +
        final["Date"].dt.year.astype(str)
    )

    # Create output folder if it does not exist
    output_folder = os.path.dirname(OUT_CSV)

    if output_folder:
        os.makedirs(
            output_folder,
            exist_ok=True
        )

    # Save CSV
    final.to_csv(
        OUT_CSV,
        index=False
    )

    print("\nExtraction completed successfully!")
    print("Output file:", OUT_CSV)
    print("Output shape:", final.shape)
    print("\nFirst five rows:")
    print(final.head())


if __name__ == "__main__":
    main()
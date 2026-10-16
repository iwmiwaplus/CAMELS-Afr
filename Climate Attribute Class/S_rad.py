"""
Extract daily solar radiation (SSR) from NetCDF files
for CAMELS Level 01 basins using area-weighted mean.
 
Output CSV columns: Date, <Basin_ID_1>, <Basin_ID_2>, ...
Units: J m-2 (raw accumulated SSR)
"""
 
import os
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc4
 
from datetime import date, timedelta
from exactextract.raster import NumPyRasterSource
from exactextract import exact_extract
 
warnings.filterwarnings("ignore")
 
# --------------------------------------------------
# PATHS
# --------------------------------------------------
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NC_DIR = BASE_DIR
 
SHP_PATH = os.path.join(
    BASE_DIR,
    "CAMELS_Afr_L1_Basin_Boundaries.shp"
)
 
OUT_CSV = os.path.join(
    BASE_DIR,
    "CAMELS_Afr_Climate_Daily_L1_srad.csv"
)
 
FILL_THRESHOLD = 1e37
 
WKT_4326 = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)
 
# --------------------------------------------------
# LOAD BASINS
# --------------------------------------------------
 
print("Loading shapefile...")
 
if not os.path.exists(SHP_PATH):
    raise FileNotFoundError(f"Shapefile not found: {SHP_PATH}")
 
basins = gpd.read_file(SHP_PATH)
 
print(f"Original CRS: {basins.crs}")
 
basins = basins[["Basin_ID", "geometry"]].copy()
basins["Basin_ID"] = basins["Basin_ID"].astype(str)
 
# Convert basin geometries to longitude/latitude
basins = basins.to_crs("EPSG:4326")
 
basin_ids = basins["Basin_ID"].tolist()
 
print(f"{len(basins)} basins loaded.")
print(f"Basin names: {basin_ids}")
print(f"Reprojected CRS: {basins.crs}")
 
 
# --------------------------------------------------
# DECODE DATES
# --------------------------------------------------
 
def decode_dates(ds: nc4.Dataset) -> list:
 
    time_var = ds.variables["valid_time"]
    raw_dates = np.asarray(time_var[:]).ravel()
 
    origin = (
        time_var.units
        .split("days since ")[-1]
        .strip()
        .split()[0]
    )
 
    start_date = date.fromisoformat(origin)
 
    return [
        start_date + timedelta(days=float(day))
        for day in raw_dates
    ]
 
 
# --------------------------------------------------
# PROCESS ONE NETCDF FILE
# --------------------------------------------------
 
def process_nc(nc_path: str) -> pd.DataFrame:
 
    print(f"\nProcessing {os.path.basename(nc_path)}...", flush=True)
 
    with nc4.Dataset(nc_path) as ds:
 
        dates = decode_dates(ds)
 
        lats = np.asarray(ds.variables["latitude"][:])
        lons = np.asarray(ds.variables["longitude"][:])
 
        ssr_var = ds.variables["ssr"]
        ssr_dims = ssr_var.dimensions
        ssr_shape = tuple(ssr_var.shape)
 
        ssr_raw = np.asarray(
            ssr_var[:],
            dtype=np.float64
        )
 
    print(f"Raw shape: {ssr_raw.shape}")
    print(f"Dimensions: {ssr_dims}")
 
    # Remove dimensions of size 1, if present
    keep = [
        i for i, size in enumerate(ssr_shape)
        if size > 1
    ]
 
    if len(keep) < ssr_raw.ndim:
        ssr_raw = np.squeeze(ssr_raw)
        ssr_dims = tuple(ssr_dims[i] for i in keep)
 
    # Arrange data as (time, latitude, longitude)
    dim_to_ax = {
        dim: i for i, dim in enumerate(ssr_dims)
    }
 
    t_ax = dim_to_ax.get("valid_time")
    lat_ax = dim_to_ax.get("latitude")
    lon_ax = dim_to_ax.get("longitude")
 
    if None in (t_ax, lat_ax, lon_ax):
        raise ValueError(
            f"Cannot identify time/latitude/longitude dimensions: {ssr_dims}"
        )
 
    ssr = np.transpose(
        ssr_raw,
        (t_ax, lat_ax, lon_ax)
    )
 
    print(f"Rearranged shape: {ssr.shape}")
 
    # Mask invalid fill values
    ssr[ssr >= FILL_THRESHOLD] = np.nan
 
    # Ensure latitude increases from south to north
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        ssr = ssr[:, ::-1, :]
 
    # Calculate raster pixel-edge bounds
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])
 
    xmin = float(lons[0]) - dlon / 2
    xmax = float(lons[-1]) + dlon / 2
    ymin = float(lats[0]) - dlat / 2
    ymax = float(lats[-1]) + dlat / 2
 
    n_time = ssr.shape[0]
    n_basins = len(basins)
 
    results = np.full(
        (n_time, n_basins),
        np.nan,
        dtype=np.float64
    )
 
    # Extract mean solar radiation for each basin and day
    for t in range(n_time):
 
        src = NumPyRasterSource(
            ssr[t],
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            nodata=np.nan,
            srs_wkt=WKT_4326
        )
 
        extracted = exact_extract(
            src,
            basins,
            ["mean"],
            include_cols=["Basin_ID"],
            output="pandas",
            progress=False
        )
 
        extracted["Basin_ID"] = (
            extracted["Basin_ID"].astype(str)
        )
 
        extracted = extracted.set_index("Basin_ID")["mean"]
 
        for i, basin_id in enumerate(basin_ids):
            results[t, i] = extracted.get(basin_id, np.nan)
 
        if (t + 1) % 50 == 0 or t + 1 == n_time:
            print(f"  Processed {t + 1}/{n_time} days", flush=True)
 
    df = pd.DataFrame(
        results,
        columns=basin_ids
    )
 
    # Date format: month/day/year
    df.insert(
        0,
        "Date",
        [d.strftime("%m/%d/%Y") for d in dates]
    )
 
    return df
 
 
# --------------------------------------------------
# MAIN
# --------------------------------------------------
 
def main():
 
    nc_files = sorted(
        glob.glob(
            os.path.join(NC_DIR, "solar_radiation_*.nc")
        )
    )
 
    if not nc_files:
        raise FileNotFoundError(
            f"No solar_radiation_*.nc files found in: {NC_DIR}"
        )
 
    print(f"\nFound {len(nc_files)} NetCDF file(s).")
 
    frames = [
        process_nc(nc_path)
        for nc_path in nc_files
    ]
 
    print("\nCombining results...")
 
    final = (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            "Date",
            key=lambda col: pd.to_datetime(col, format="%m/%d/%Y")
        )
        .drop_duplicates(subset=["Date"], keep="first")
        .reset_index(drop=True)
    )
 
    final.to_csv(OUT_CSV, index=False)
 
    print("\nProcessing completed.")
    print(f"Output shape: {final.shape}")
    print(f"Saved CSV: {OUT_CSV}")
 
 
if __name__ == "__main__":
    main()
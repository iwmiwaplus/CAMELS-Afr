import os
import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from exactextract import exact_extract
 
# 1. File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
slope_path = os.path.join(
    BASE_DIR,
    "FABDEM_slope_degrees.tif"
)
 
basin_path = os.path.join(
    BASE_DIR,
    "CAMELS_Afr_L1_Basin_Boundaries.shp"
)
 
out_csv = os.path.join(
    BASE_DIR,
    "CAMELS_Afr_Location_Static_L1.csv"
)
 
# 2. Load and Prepare Data
print("Loading Basins...")
basins_all = gpd.read_file(basin_path)
 
with rasterio.open(slope_path) as src:
    raster_crs = src.crs
 
# Align CRS
if basins_all.crs != raster_crs:
    print("Reprojecting basins to match raster CRS...")
    basins_all = basins_all.to_crs(raster_crs)
 
# Fix geometries
basins_all["geometry"] = basins_all.geometry.buffer(0)
 
# 3. Resume Logic
if os.path.exists(out_csv):
    existing_df = pd.read_csv(out_csv)
    processed_ids = existing_df["Basin_ID"].unique()
    basins_to_process = basins_all[
        ~basins_all["Basin_ID"].isin(processed_ids)
    ].copy()
    print(
        f"Resuming. {len(processed_ids)} already processed. "
        f"{len(basins_to_process)} remaining."
    )
else:
    basins_to_process = basins_all.copy()
    print(f"Starting fresh. {len(basins_to_process)} basins to process.")
 
# 4. Ops list — quantiles passed inline, no separate 'quantiles' argument
ops_stats = [
    "min",
    "mean",
    "max",
    "quantile(q=0.10)",
    "quantile(q=0.25)",
    "quantile(q=0.50)",
    "quantile(q=0.75)",
    "quantile(q=0.90)",
]
 
# 5. Processing Loop
for idx, row in basins_to_process.iterrows():
    Basin_ID = row["Basin_ID"]
    single_basin = basins_all.loc[[idx]]
    pos = list(basins_all.index).index(idx) + 1
    print(f"Processing basin {Basin_ID} ({pos}/{len(basins_all)})...")
 
    try:
        # --- Standard extraction ---
        res = exact_extract(
            slope_path,
            single_basin,
            ops_stats,
            include_cols=["Basin_ID"],
            output="pandas"
        )
 
        raw = exact_extract(
            slope_path,
            single_basin,
            ["values", "coverage"],
            output="pandas"
        )
 
        vals = np.array(raw["values"].iloc[0])
        weights = np.array(raw["coverage"].iloc[0])
        mask = ~np.isnan(vals)
        vals = vals[mask]
        weights = weights[mask]
 
        # --- Fallback: buffer ~1 pixel if no overlap ---
        if len(vals) == 0:
            print(f"  -> Warning: No overlap for {Basin_ID}. Buffering...")
            buf_dist = 0.00083 if single_basin.crs.is_geographic else 100
            single_basin_buf = single_basin.copy()
            single_basin_buf["geometry"] = (
                single_basin_buf.geometry.buffer(buf_dist)
            )
 
            res = exact_extract(
                slope_path,
                single_basin_buf,
                ops_stats,
                include_cols=["Basin_ID"],
                output="pandas"
            )
 
            raw = exact_extract(
                slope_path,
                single_basin_buf,
                ["values", "coverage"],
                output="pandas"
            )
 
            vals = np.array(raw["values"].iloc[0])
            weights = np.array(raw["coverage"].iloc[0])
            mask = ~np.isnan(vals)
            vals = vals[mask]
            weights = weights[mask]
 
        # --- Calculate results ---
        if len(vals) > 0:
            total_w = weights.sum()
            flat_perc = (weights[vals < 3].sum() / total_w) * 100
            steep_perc = (weights[vals > 15].sum() / total_w) * 100
 
            data = {
                "Basin_ID": Basin_ID,
                "slope_min": res["min"].iloc[0],
                "slope_mean": res["mean"].iloc[0],
                "slope_max": res["max"].iloc[0],
                "slope_median": res["quantile_50"].iloc[0],
                "slope_10": res["quantile_10"].iloc[0],
                "slope_25": res["quantile_25"].iloc[0],
                "slope_75": res["quantile_75"].iloc[0],
                "slope_90": res["quantile_90"].iloc[0],
                "flat_area_perc": round(flat_perc, 4),
                "steep_area_perc": round(steep_perc, 4),
            }
        else:
            # Truly outside raster coverage — fill with NaN
            data = {col: np.nan for col in [
                "slope_min", "slope_mean", "slope_max", "slope_median",
                "slope_10", "slope_25", "slope_75", "slope_90",
                "flat_area_perc", "steep_area_perc"
            ]}
            data["Basin_ID"] = Basin_ID
 
    except Exception as e:
        print(f"  -> Error processing {Basin_ID}: {e}")
        continue
 
    # 6. Append row to CSV immediately (safe resume)
    result_df = pd.DataFrame([data])
    write_header = not os.path.exists(out_csv)
    result_df.to_csv(
        out_csv,
        mode="a",
        index=False,
        header=write_header
    )
 
print(f"\nDone! Results saved to: {out_csv}")
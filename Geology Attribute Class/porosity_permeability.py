import geopandas as gpd
import pandas as pd
import os
import numpy as np
import shapely

# 1. File Paths
basin_path = r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"
glhymps_path = r"...\GLHYMPS_africa_extent.shp"
output_csv = r"...\CAMELS_Afr_Geology_Static_L1_porosity_permeability.csv"

# Tune these if memory is limited.
BASINS_PER_BATCH = 10
PAIRS_PER_BATCH = 1000

print("Loading basin shapefile...", flush=True)

basins = gpd.read_file(
    basin_path, engine="pyogrio",
    columns=["HYBAS_ID", "geometry"]
)

print("Loading GLHYMPS shapefile...", flush=True)
glhymps = gpd.read_file(
    glhymps_path, engine="pyogrio",
    columns=["Porosity_x", "logK_Ferr_", "geometry"]
)

# Remove empty/null geometries and rows without useful geometry.
basins = basins.loc[
    basins.geometry.notna() & ~basins.geometry.is_empty
].reset_index(drop=True)
glhymps = glhymps.loc[
    glhymps.geometry.notna() & ~glhymps.geometry.is_empty
].reset_index(drop=True)

# Align CRS.
if basins.crs != glhymps.crs:
    print(f"Reprojecting basins to {glhymps.crs}...", flush=True)
    basins = basins.to_crs(glhymps.crs)

# NOTE: area-weighted calculations require a suitable projected CRS
# whose units are linear (usually metres). Check the source CRS before
# relying on the resulting areas. This preserves the original CRS logic.

glhymps["Porosity_x"] = pd.to_numeric(
    glhymps["Porosity_x"], errors="coerce"
)
glhymps["logK_Ferr_"] = pd.to_numeric(
    glhymps["logK_Ferr_"], errors="coerce"
)

# Keep an explicit row ID so we can recover geology geometries reliably
# regardless of GeoPandas index_right naming behavior.
glhymps["geol_id"] = np.arange(len(glhymps), dtype=np.int64)

print("Building spatial index...", flush=True)
geol_sindex = glhymps.sindex

# Accumulate sums per basin ID without retaining intersection geometries.
records = []
n_basins = len(basins)

for batch_start in range(0, n_basins, BASINS_PER_BATCH):
    batch_end = min(batch_start + BASINS_PER_BATCH, n_basins)
    basin_batch = basins.iloc[batch_start:batch_end]

    print(
        f"Processing basins {batch_start + 1:,}-{batch_end:,} "
        f"of {n_basins:,}...",
        flush=True
    )

    # Spatial join limits candidate pairs to polygons that intersect.
    joined = gpd.sjoin(
        basin_batch,
        glhymps[["geol_id", "Porosity_x", "logK_Ferr_", "geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="basin",
        rsuffix="geol"
    )

    if joined.empty:
        continue

    # Fetch geology geometry using the explicit geology row ID.
    right_idx = joined["geol_id"].to_numpy(dtype=np.int64)
    basin_geoms = joined.geometry.to_numpy()
    geol_geoms = glhymps.geometry.iloc[right_idx].to_numpy()
    por_values = joined["Porosity_x"].to_numpy(dtype=float)
    per_values = joined["logK_Ferr_"].to_numpy(dtype=float)
    basin_ids = joined["HYBAS_ID"].to_numpy()

    # Process matched pairs in smaller chunks.
    for pair_start in range(0, len(joined), PAIRS_PER_BATCH):
        pair_end = min(pair_start + PAIRS_PER_BATCH, len(joined))

        intersections = shapely.intersection(
            basin_geoms[pair_start:pair_end],
            geol_geoms[pair_start:pair_end]
        )
        areas = shapely.area(intersections)

        # Only positive-area polygon fragments contribute to area-weighted means.
        valid = np.isfinite(areas) & (areas > 0)
        if not valid.any():
            continue

        ids = basin_ids[pair_start:pair_end][valid]
        a = areas[valid]
        p = por_values[pair_start:pair_end][valid]
        k = per_values[pair_start:pair_end][valid]

        # Missing attribute values do not contribute to that attribute's
        # numerator OR denominator. This avoids treating missing values as zero.
        temp = pd.DataFrame({
            "HYBAS_ID": ids,
            "area": a,
            "por_area": np.where(np.isfinite(p), p * a, 0.0),
            "por_valid_area": np.where(np.isfinite(p), a, 0.0),
            "per_area": np.where(np.isfinite(k), k * a, 0.0),
            "per_valid_area": np.where(np.isfinite(k), a, 0.0),
        })

        partial = temp.groupby("HYBAS_ID", as_index=False).sum()
        records.append(partial)

# Combine partial sums from all batches.
print("Combining weighted results...", flush=True)
if not records:
    raise RuntimeError(
        "No positive-area intersections were found. Check the CRS, "
        "input geometries, and spatial overlap."
    )

all_sums = pd.concat(records, ignore_index=True)
results = all_sums.groupby("HYBAS_ID", as_index=False).sum()

results["por_mean"] = (
    results["por_area"] / results["por_valid_area"].replace(0, np.nan)
)
results["per_mean"] = (
    results["per_area"] / results["per_valid_area"].replace(0, np.nan)
)

# Output only the three requested columns, with clear names.
# Values remain area-weighted means; no extra scaling is applied.
final_df = results[["HYBAS_ID", "por_mean", "per_mean"]].copy()
final_df = final_df.rename(columns={
    "HYBAS_ID": "Basin_ID",
    "por_mean": "Porosity",
    "per_mean": "Permeability"
})

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
final_df.to_csv(output_csv, index=False)

print(f"Process complete! Output saved to: {output_csv}", flush=True)

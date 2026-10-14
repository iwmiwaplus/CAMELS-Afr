from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------------------
SHAPEFILE_PATH = (".../CAMELS_Afr_L1_Basin_Boundaries.shp")

OUTPUT_CSV = (".../CAMELS_Afr_Location_Static_L1.csv") 

ROUND_DIGITS = 8

# Only set this if you have independently confirmed the shapefile's CRS.
# Leave as None to stop safely if the CRS is missing.
MISSING_CRS_OVERRIDE = None


def find_basin_id_column(columns):
    """Find the basin identifier field, allowing common capitalization variants."""
    normalized = {str(column).strip().lower(): column for column in columns}

    for candidate in ("basin_id", "basinid", "basin_id_1", "id"):
        if candidate in normalized:
            return normalized[candidate]

    raise ValueError(
        "Could not identify the basin ID column. "
        f"Available fields: {list(columns)}"
    )


def main():
    if not SHAPEFILE_PATH.exists():
        raise FileNotFoundError(
            f"Shapefile not found:\n{SHAPEFILE_PATH}\n\n"
            "Check the filename and folder, and ensure all shapefile "
            "components are together."
        )

    print(f"Reading shapefile: {SHAPEFILE_PATH}")
    basins = gpd.read_file(SHAPEFILE_PATH, engine="pyogrio")

    if basins.empty:
        raise ValueError("The shapefile contains no features.")

    id_column = find_basin_id_column(basins.columns)

    # Remove features without usable geometries.
    basins = basins[
        basins.geometry.notna() & ~basins.geometry.is_empty
    ].copy()

    if basins.crs is None:
        if MISSING_CRS_OVERRIDE is None:
            raise ValueError(
                "The shapefile CRS is missing. Check the .prj file and "
                "set MISSING_CRS_OVERRIDE only after confirming the CRS."
            )
        basins = basins.set_crs(MISSING_CRS_OVERRIDE)

    print(f"Input CRS: {basins.crs}")
    print("Projecting geometries to EPSG:6933 for area/perimeter calculations...")
    metric_basins = basins.to_crs("EPSG:6933")

    # Area in square kilometres and perimeter in kilometres.
    area_km2 = metric_basins.geometry.area / 1_000_000
    perimeter_km = metric_basins.geometry.boundary.length / 1_000

    valid = (
        np.isfinite(area_km2)
        & np.isfinite(perimeter_km)
        & (area_km2 > 0)
        & (perimeter_km > 0)
    )

    circul_ratio = pd.Series(np.nan, index=metric_basins.index, dtype=float)
    compact_co = pd.Series(np.nan, index=metric_basins.index, dtype=float)

    circul_ratio.loc[valid] = (
        4 * math.pi * area_km2.loc[valid] / perimeter_km.loc[valid] ** 2
    )
    compact_co.loc[valid] = (
        perimeter_km.loc[valid]
        / (2 * np.sqrt(math.pi * area_km2.loc[valid]))
    )

    result = pd.DataFrame({
        "Basin_ID": metric_basins[id_column].astype("string").str.strip().to_numpy(),
        "catchment_area": area_km2.round(ROUND_DIGITS).to_numpy(),
        "catchment_perimeter": perimeter_km.round(ROUND_DIGITS).to_numpy(),
        "circul_ratio": circul_ratio.round(ROUND_DIGITS).to_numpy(),
        "compact_co": compact_co.round(ROUND_DIGITS).to_numpy(),
    })

    result.to_csv(OUTPUT_CSV, index=False)

    print(f"\nFeatures processed: {len(result)}")
    print(f"Calculated values saved to:\n{OUTPUT_CSV}")
    print("\nFirst rows:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()

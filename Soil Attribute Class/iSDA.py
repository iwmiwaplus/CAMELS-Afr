"""
iSDAsoil Basin-wise Mean Extraction
 
Method:
1. Standard processing for smaller basins.
2. Chunked processing for large basins to reduce memory usage.
3. Apply the configured transformation to valid raster pixels.
4. Calculate basin-wise means and save the results to a CSV file.
"""
 
import os
import time
import warnings
 
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NodataShadowWarning
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.windows import Window
 
from tqdm import tqdm
 
 
# Suppress the repeated NoData/alpha-band warning.
warnings.filterwarnings(
    "ignore",
    category=NodataShadowWarning
)
 
 
# ==================================================
# 1. CONFIGURATION
# ==================================================
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
RASTER_FOLDER = r"...\iSDA_soil"
 
BASIN_PATH = os.path.join(
    BASE_DIR,
    "CAMELS_Afr_L1_Basin_Boundaries.shp"
)
 
# Current raster
TIF_NAME = "bulk_density.tif"
 
TIF_FILE = os.path.join(  
    RASTER_FOLDER,
    TIF_NAME
)
 
# Output in the same folder as this script
OUTPUT_FOLDER = BASE_DIR
 
BASIN_ID_FIELD = "Basin_ID"
 
# Switch to chunked processing when either
# basin dimension exceeds this threshold.
LARGE_BASIN_LIMIT = 5000
 
# Window size for large-basin processing
CHUNK_SIZE = 2048
 
# ==================================================
# 2. TRANSFORMATION FUNCTIONS
# ==================================================
 
def transform_expm1(x):
    return np.expm1(x / 10.0)
 
 
def transform_div10(x):
    return x / 10.0
 
 
def transform_div100(x):
    return x / 100.0
 
 
def transform_none(x):
    return x
 
 
TRANSFORMATIONS = {
    "expm1": transform_expm1,
    "div10": transform_div10,
    "div100": transform_div100,
    "none": transform_none,
}
 
 
# ==================================================
# 3. RASTER CONFIGURATION
# ==================================================
 
RASTER_CONFIG = {
    "aluminium_extractable.tif": {
        "columns": [
            "aluminium_ext_0_20",
            "aluminium_ext_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "bedrock_depth.tif": {
        "columns": ["bedrk_dep_0_200"],
        "bands": [1],
        "transform": "none",
    },
 
    "bulk_density.tif": {
        "columns": [
            "soil_bulkdensity_0_20",
            "soil_bulkdensity_20_50"
        ],
        "bands": [1, 2],
        "transform": "div100",
    },
 
    "calcium_extractable.tif": {
        "columns": [
            "calcium_ext_0_20",
            "calcium_ext_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "carbon_organic.tif": {
        "columns": [
            "oc_perc_0_20",
            "oc_perc_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "carbon_total.tif": {
        "columns": [
            "carbon_tot_0_20",
            "carbon_tot_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "cation_exchange_capacity.tif": {
        "columns": [
            "cecsol_0_20",
            "cecsol_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "clay_content.tif": {
        "columns": [
            "clay_perc_0_20",
            "clay_perc_20_50"
        ],
        "bands": [1, 2],
        "transform": "none",
    },
 
    "iron_extractable.tif": {
        "columns": [
            "iron_ext_0_20",
            "iron_ext_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "magnesium_extractable.tif": {
        "columns": [
            "magnesium_ext_0_20",
            "magnesium_ext_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "nitrogen_total.tif": {
        "columns": [
            "nitrogen_tot_0_20",
            "nitrogen_tot_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "ph.tif": {
        "columns": [
            "phihox_0_20",
            "phihox_20_50"
        ],
        "bands": [1, 2],
        "transform": "div10",
    },
 
    "phosphorous_extractable.tif": {
        "columns": [
            "phosphorous_ext_0_20",
            "phosphorous_ext_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "potassium_extractable.tif": {
        "columns": [
            "potassium_ext_0_20",
            "potassium_ext_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "sand_content.tif": {
        "columns": [
            "sand_perc_0_20",
            "sand_perc_20_50"
        ],
        "bands": [1, 2],
        "transform": "none",
    },
 
    "silt_content.tif": {
        "columns": [
            "silt_perc_0_20",
            "silt_perc_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "stone_content.tif": {
        "columns": [
            "stone_perc_0_20",
            "stone_perc_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "sulphur_extractable.tif": {
        "columns": [
            "sulphur_ext_0_20",
            "sulphur_ext_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
 
    "zinc_extractable.tif": {
        "columns": [
            "zinc_ext_0_20",
            "zinc_ext_20_50"
        ],
        "bands": [1, 2],
        "transform": "expm1",
    },
}
 
 
# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================
 
def get_pixel_bounds(geometry, transform, width, height):
    """Calculate raster pixel bounds intersecting a basin."""
 
    minx, miny, maxx, maxy = geometry.bounds
 
    # Convert all four corners to pixel coordinates.
    corners = [
        rasterio.transform.rowcol(transform, minx, miny),
        rasterio.transform.rowcol(transform, minx, maxy),
        rasterio.transform.rowcol(transform, maxx, miny),
        rasterio.transform.rowcol(transform, maxx, maxy),
    ]
 
    rows = [point[0] for point in corners]
    cols = [point[1] for point in corners]
 
    row_min = max(0, min(rows))
    row_max = min(height, max(rows) + 1)
 
    col_min = max(0, min(cols))
    col_max = min(width, max(cols) + 1)
 
    return row_min, row_max, col_min, col_max
 
 
def update_statistics(stats, band_number, values, transform_func):
    """Update running sum and count for valid transformed pixels."""
 
    if values.size == 0:
        return
 
    valid_values = values[np.isfinite(values)]
 
    if valid_values.size == 0:
        return
 
    transformed_values = transform_func(
        valid_values.astype(np.float64)
    )
 
    transformed_values = transformed_values[
        np.isfinite(transformed_values)
    ]
 
    if transformed_values.size == 0:
        return
 
    stats[band_number]["sum"] += np.sum(
        transformed_values,
        dtype=np.float64
    )
 
    stats[band_number]["count"] += transformed_values.size
 
 
# ==================================================
# 5. STANDARD PROCESSING
# ==================================================
 
def process_basin_standard(
    src,
    geometry,
    band_numbers,
    transform_func
):
    """Process a smaller basin using a cropped raster mask."""
 
    stats = {
        band: {"sum": 0.0, "count": 0}
        for band in band_numbers
    }
 
    for band_number in band_numbers:
 
        masked_data, _ = mask(
            src,
            [geometry],
            indexes=band_number,
            crop=True,
            all_touched=True,
            filled=False
        )
 
        band_data = masked_data[0]
 
        # Preserve the raster's masked NoData pixels.
        values = band_data.compressed()
 
        update_statistics(
            stats,
            band_number,
            values,
            transform_func
        )
 
    return stats
 
 
# ==================================================
# 6. CHUNKED PROCESSING
# ==================================================
 
def process_basin_chunked(
    src,
    geometry,
    band_numbers,
    transform_func,
    chunk_size=CHUNK_SIZE
):
    """Process a large basin in small raster windows."""
 
    row_min, row_max, col_min, col_max = get_pixel_bounds(
        geometry,
        src.transform,
        src.width,
        src.height
    )
 
    stats = {
        band: {"sum": 0.0, "count": 0}
        for band in band_numbers
    }
 
    for row_start in range(row_min, row_max, chunk_size):
 
        for col_start in range(col_min, col_max, chunk_size):
 
            window_height = min(
                chunk_size,
                row_max - row_start
            )
 
            window_width = min(
                chunk_size,
                col_max - col_start
            )
 
            if window_height <= 0 or window_width <= 0:
                continue
 
            window = Window(
                col_start,
                row_start,
                window_width,
                window_height
            )
 
            window_transform = src.window_transform(window)
 
            # Create a mask for the current chunk only.
            inside_basin = geometry_mask(
                [geometry],
                out_shape=(window_height, window_width),
                transform=window_transform,
                invert=True,
                all_touched=True
            )
 
            if not inside_basin.any():
                continue
 
            # Read only the required bands and current window.
            for band_number in band_numbers:
 
                chunk = src.read(
                    band_number,
                    window=window,
                    masked=True
                )
 
                # Exclude pixels outside the basin and NoData.
                valid_mask = inside_basin & ~np.ma.getmaskarray(chunk)
 
                if not valid_mask.any():
                    continue
 
                values = np.asarray(chunk.data)[valid_mask]
 
                update_statistics(
                    stats,
                    band_number,
                    values,
                    transform_func
                )
 
    return stats
 
 
# ==================================================
# 7. MAIN EXECUTION
# ==================================================
 
def main():
 
    start_time = time.time()
 
    # Validate input files.
    if TIF_NAME == "texture_class.tif":
        raise ValueError(
            "texture_class.tif is excluded from extraction."
        )
 
    if TIF_NAME not in RASTER_CONFIG:
        raise ValueError(
            f"No raster configuration found for {TIF_NAME}"
        )
 
    if not os.path.isfile(TIF_FILE):
        raise FileNotFoundError(
            f"Raster not found: {TIF_FILE}"
        )
 
    if not os.path.isfile(BASIN_PATH):
        raise FileNotFoundError(
            f"Basin shapefile not found: {BASIN_PATH}"
        )
 
    config = RASTER_CONFIG[TIF_NAME]
 
    band_numbers = config["bands"]
    output_columns = config["columns"]
 
    transform_func = TRANSFORMATIONS[
        config["transform"]
    ]
 
    if len(band_numbers) != len(output_columns):
        raise ValueError(
            "The configured band numbers and output columns do not match."
        )
 
    print("Reading basin shapefile...")
 
    basins = gpd.read_file(BASIN_PATH)
 
    if basins.crs is None:
        raise ValueError(
            "The basin shapefile has no CRS defined."
        )
 
    if BASIN_ID_FIELD not in basins.columns:
        raise ValueError(
            f"Missing basin ID field: {BASIN_ID_FIELD}"
        )
 
    basins = basins[
        [BASIN_ID_FIELD, "geometry"]
    ].copy()
 
    basins = basins[
        basins.geometry.notna()
        & ~basins.geometry.is_empty
    ]
 
    results = []
 
    print(f"Processing raster: {TIF_NAME}")
 
    with rasterio.open(TIF_FILE) as src:
 
        if src.crs is None:
            raise ValueError(
                "The raster has no CRS defined."
            )
 
        if max(band_numbers) > src.count:
            raise ValueError(
                f"{TIF_NAME} contains {src.count} band(s), "
                f"but the configuration requests band "
                f"{max(band_numbers)}."
            )
 
        if basins.crs != src.crs:
            print("Reprojecting basins to raster CRS...")
            basins = basins.to_crs(src.crs)
 
        print(
            f"Number of basins: {len(basins)}"
        )
 
        for _, basin in tqdm(
            basins.iterrows(),
            total=len(basins),
            desc="Extracting"
        ):
 
            basin_id = basin[BASIN_ID_FIELD]
            geometry = basin.geometry
 
            result_row = {
                "Basin_ID": basin_id
            }
 
            for column in output_columns:
                result_row[column] = np.nan
 
            try:
 
                row_min, row_max, col_min, col_max = (
                    get_pixel_bounds(
                        geometry,
                        src.transform,
                        src.width,
                        src.height
                    )
                )
 
                row_size = row_max - row_min
                col_size = col_max - col_min
 
                if row_size <= 0 or col_size <= 0:
                    results.append(result_row)
                    continue
 
                is_large = (
                    row_size > LARGE_BASIN_LIMIT
                    or col_size > LARGE_BASIN_LIMIT
                )
 
                if is_large:
                    stats = process_basin_chunked(
                        src,
                        geometry,
                        band_numbers,
                        transform_func
                    )
                else:
                    stats = process_basin_standard(
                        src,
                        geometry,
                        band_numbers,
                        transform_func
                    )
 
                for band_number, column_name in zip(
                    band_numbers,
                    output_columns
                ):
 
                    band_stats = stats[band_number]
                    count = band_stats["count"]
 
                    if count > 0:
                        result_row[column_name] = (
                            band_stats["sum"] / count
                        )
 
            except Exception as error:
 
                print(
                    f"\nError processing basin {basin_id}: {error}"
                )
 
            results.append(result_row)
 
    # --------------------------------------------------
    # 8. SAVE OUTPUT CSV
    # --------------------------------------------------
 
    output_df = pd.DataFrame(
        results,
        columns=["Basin_ID"] + output_columns
    )
 
    variable_name = os.path.splitext(TIF_NAME)[0]
 
    output_filename = (
        f"CAMELS_Afr_Soil_Static_L1_{variable_name}.csv"
    )
 
    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_filename
    )
 
    output_df.to_csv(
        output_path,
        index=False
    )
 
    elapsed_minutes = (
        time.time() - start_time
    ) / 60
 
    print("\nExtraction completed.")
    print(f"Raster: {TIF_NAME}")
    print(f"Number of basins: {len(output_df)}")
    print(f"Output saved to: {output_path}")
    print(f"Execution time: {elapsed_minutes:.2f} minutes")
    print("\nPreview:")
    print(output_df.head())
 
 
if __name__ == "__main__":
    main()
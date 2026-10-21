# -*- coding: utf-8 -*-
"""
Monthly MODIS MCD15A3H Basin Statistics Extraction (Google Earth Engine)

This script computes monthly basin-scale vegetation statistics using MODIS
MCD15A3H Leaf Area Index (LAI) and Fraction of Photosynthetically Active
Radiation (FPAR) products.

Outputs are exported as CSV files to Google Drive.

"""

import ee
import time
from tqdm import tqdm


# =========================================================
# Configuration
# =========================================================

PROJECT_ID = "your-google-earth-engine-project-id"

BASIN_ASSET_ID = "projects/your-project/assets/Basin_Boundaries.shp"
BASIN_ID_FIELD = "Basin_ID"

OUTPUT_FOLDER = "EarthEngine_Exports"
OUTPUT_PREFIX = "LandCover_MODIS_MCD15A3H_Monthly"

START_YEAR = 2002
END_YEAR = 2024
CHUNK_SIZE = 1

SPATIAL_SCALE = 500
TILE_SCALE = 4
OUTPUT_CRS = "EPSG:4326"

FPAR_MAX = 0.95

MODIS_COLLECTION = "MODIS/061/MCD15A3H"


# =========================================================
# Initialize Earth Engine
# =========================================================

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)


# =========================================================
# Load datasets
# =========================================================

basins = ee.FeatureCollection(BASIN_ASSET_ID)
dataset = ee.ImageCollection(MODIS_COLLECTION)

print("Basin dataset loaded.")


# =========================================================
# Monthly processing
# =========================================================

def process_year_range(start_year, end_year):

    start_date = ee.Date.fromYMD(start_year, 1, 1)
    end_date = ee.Date.fromYMD(end_year, 12, 31)

    n_months = end_date.difference(start_date, "month").round()
    month_offsets = ee.List.sequence(0, n_months.subtract(1))

    def process_month(offset, acc):

        offset = ee.Number(offset)
        acc = ee.FeatureCollection(acc)

        date = start_date.advance(offset, "month")
        year = date.get("year")
        month = date.get("month")

        monthly_images = dataset.filterDate(
            date,
            date.advance(1, "month")
        )

        # -------------------------
        # No data case
        # -------------------------
        def empty_month():

            def empty_feature(f):
                return ee.Feature(None, {
                    "Year": year,
                    "Month": month,
                    "Basin_ID": f.get(BASIN_ID_FIELD),
                    "gvf_mean": "NA",
                    "gvf_min": "NA",
                    "gvf_max": "NA",
                    "gvf_diff": "NA",
                    "lai_mean": "NA",
                    "lai_min": "NA",
                    "lai_max": "NA",
                    "lai_diff": "NA",
                    "valid_pixels": 0,
                    "total_pixels": 0,
                    "valid_pct": 0
                })

            return basins.map(empty_feature)

        # -------------------------
        # Processing function
        # -------------------------
        def compute_monthly():

            def extract(img):

                fpar = img.select("Fpar")
                lai = img.select("Lai")

                valid = (
                    fpar.lte(100)
                    .Or(fpar.eq(253))
                    .Or(fpar.eq(252))
                )

                fpar = fpar.updateMask(valid)
                lai = lai.updateMask(valid)

                fpar = fpar.where(fpar.gte(252), 0)
                lai = lai.where(lai.gte(252), 0)

                gvf = (
                    fpar.multiply(0.01)
                    .divide(FPAR_MAX)
                    .clamp(0, 1)
                    .rename("gvf")
                )

                lai = lai.multiply(0.1).rename("lai")

                return img.addBands([gvf, lai]).select(["gvf", "lai"])

            monthly = monthly_images.map(extract)

            mean_img = monthly.mean().rename(["gvf_mean", "lai_mean"])
            min_img = monthly.min().rename(["gvf_min", "lai_min"])
            max_img = monthly.max().rename(["gvf_max", "lai_max"])

            diff_img = max_img.subtract(min_img).rename(
                ["gvf_diff", "lai_diff"]
            )

            total = ee.Image.constant(1).rename("total")

            combined = mean_img.addBands([
                min_img,
                max_img,
                diff_img,
                total
            ])

            reducer = (
                ee.Reducer.mean()
                .combine(
                    ee.Reducer.count(),
                    sharedInputs=True
                )
            )

            stats = combined.reduceRegions(
                collection=basins,
                reducer=reducer,
                scale=SPATIAL_SCALE,
                crs=OUTPUT_CRS,
                tileScale=TILE_SCALE
            )

            def format_feature(f):

                props = f.toDictionary()

                total_px = ee.Number(
                    ee.Algorithms.If(
                        props.contains("total_count"),
                        props.get("total_count"),
                        0
                    )
                )

                valid_px = ee.Number(
                    ee.Algorithms.If(
                        props.contains("gvf_mean_count"),
                        props.get("gvf_mean_count"),
                        0
                    )
                )

                valid_pct = ee.Algorithms.If(
                    total_px.gt(0),
                    valid_px.divide(total_px).multiply(100),
                    0
                )

                def safe_get(k):
                    return ee.Algorithms.If(
                        props.contains(k),
                        props.get(k),
                        "NA"
                    )

                return ee.Feature(None, {
                    "Year": year,
                    "Month": month,
                    "Basin_ID": f.get(BASIN_ID_FIELD),

                    "gvf_mean": safe_get("gvf_mean_mean"),
                    "gvf_min": safe_get("gvf_min_mean"),
                    "gvf_max": safe_get("gvf_max_mean"),
                    "gvf_diff": safe_get("gvf_diff_mean"),

                    "lai_mean": safe_get("lai_mean_mean"),
                    "lai_min": safe_get("lai_min_mean"),
                    "lai_max": safe_get("lai_max_mean"),
                    "lai_diff": safe_get("lai_diff_mean"),

                    "valid_pixels": valid_px,
                    "total_pixels": total_px,
                    "valid_pct": valid_pct
                })

            return stats.map(format_feature)

        monthly_result = ee.Algorithms.If(
            monthly_images.size().gt(0),
            compute_monthly(),
            empty_month()
        )

        return acc.merge(ee.FeatureCollection(monthly_result))

    return ee.FeatureCollection(
        month_offsets.iterate(process_month, ee.FeatureCollection([]))
    )


# =========================================================
# Export execution
# =========================================================

years = range(START_YEAR, END_YEAR + 1)

chunks = [
    years[i:i + CHUNK_SIZE]
    for i in range(0, len(years), CHUNK_SIZE)
]

print(f"Submitting {len(chunks)} export tasks.")

for c in tqdm(chunks, desc="Submitting"):

    start = c[0]
    end = c[-1]

    task_name = f"{OUTPUT_PREFIX}_{start}_{end}"

    fc = process_year_range(start, end)

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=task_name,
        folder=OUTPUT_FOLDER,
        fileNamePrefix=task_name,
        fileFormat="CSV",
        selectors=[
            "Basin_ID", "Year", "Month",
            "gvf_mean", "gvf_min", "gvf_max", "gvf_diff",
            "lai_mean", "lai_min", "lai_max", "lai_diff",
            "valid_pixels", "total_pixels", "valid_pct"
        ]
    )

    try:
        task.start()
    except Exception as e:
        print(f"Task submission failed for {task_name}: {e}")
        time.sleep(5)


print("All export tasks submitted. Monitor progress in Earth Engine Tasks.")
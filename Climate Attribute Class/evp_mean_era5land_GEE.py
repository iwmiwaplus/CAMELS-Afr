# -*- coding: utf-8 -*-
"""
ERA5-Land Monthly Total Evaporation
Area-weighted basin mean (mm/month)
Period: 1980-01-01 to 2024-12-31
"""

import ee

# ----------------------------------------------------------------------
# Authenticate / initialise Earth Engine
# ----------------------------------------------------------------------
# Run once if you haven't authenticated on this machine:
#   ee.Authenticate()
ee.Initialize(project="my-project-475604")

START_YEAR = 1980
END_YEAR = 2024

# Basin shapefile (uploaded as an Earth Engine asset)
basins = ee.FeatureCollection("projects/my-project-475604/assets/CAMELS")


# ----------------------------------------------------------------------
# Function to calculate area-weighted mean for one monthly image
# ----------------------------------------------------------------------
def extract_month(img):
    # Convert: m --> mm, negative --> positive
    evap = img.multiply(-1000).rename("evap")

    # Pixel area (m^2)
    area = ee.Image.pixelArea().rename("area")

    # Weighted evaporation
    weighted = evap.multiply(area).rename("weighted")

    # Sum(weighted) and Sum(area) per basin
    sums = weighted.addBands(area).reduceRegions(
        collection=basins,
        reducer=ee.Reducer.sum(),
        scale=11132,
        crs="EPSG:4326",
    )

    date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")

    def to_feature(f):
        weighted_sum = ee.Number(f.get("weighted"))
        area_sum = ee.Number(f.get("area"))

        mean = ee.Algorithms.If(
            area_sum.gt(0),
            weighted_sum.divide(area_sum),
            None,
        )

        return ee.Feature(
            None,
            {
                "Date": date,
                "Basin_ID": f.get("Basin_ID"),
                "Evap_mm": mean,
            },
        )

    return sums.map(to_feature)


# ----------------------------------------------------------------------
# Create one export task for each year
# ----------------------------------------------------------------------
tasks = []

for year in range(START_YEAR, END_YEAR + 1):
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, "year")

    yearly_collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(start_date, end_date)
        .select("total_evaporation_sum")
    )

    yearly_results = yearly_collection.map(extract_month).flatten()

    name = f"ERA5Land_TotalEvaporation_{year}"

    task = ee.batch.Export.table.toDrive(
        collection=yearly_results,
        description=name,
        folder="EarthEngine_L3",
        fileNamePrefix=name,
        fileFormat="CSV",
    )
    task.start()
    tasks.append(task)
    print(f"Started export task: {name}")

print(f"\n{len(tasks)} tasks submitted. Monitor them at https://code.earthengine.google.com/tasks")
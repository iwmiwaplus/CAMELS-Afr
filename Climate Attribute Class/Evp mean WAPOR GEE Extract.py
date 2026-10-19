"""
FAO WaPOR Evaporation Extraction Pipeline
 
This script extracts, aggregates, and exports dekadal (10-day) actual evaporation 
data from the FAO WaPOR dataset (Level 1) for a specified set of hydrological basins.
 
Data Sources:
- Evaporation Dataset: FAO WaPOR Level 1 Actual Evaporation (FAO/WAPOR/2/L1_E_D).
- Vector Geometries: Custom basin boundaries defined by BASIN_ASSET_ID.
 
Outputs:
The script submits batch tasks to the Earth Engine servers. Outputs are exported 
directly to Google Drive as CSV files, containing the following columns:
- Basin_ID: Unique identifier for the catchment geometry.
- Year: Calendar year.
- Month: Calendar month (1-12).
- mean: Spatial mean of cumulative actual evaporation (mm/month).
 
Usage:
Update the Configuration parameters (PROJECT_ID, BASIN_ASSET_ID, START_YEAR, END_YEAR) 
before execution. Ensure the Earth Engine API is authenticated locally.
"""
 
import time
import ee
from tqdm import tqdm
 
# ==========================================
# Configuration and Constants
# ==========================================
PROJECT_ID = 'camels-afr'
BASIN_ASSET_ID = 'projects/camels-afr/assets/CAMELS_Afr_L1_Basin_Boundaries'
BASIN_ID_FIELD = 'Basin_ID'
 
OUTPUT_FOLDER = 'Evaporation_WAPOR_Lev_01'
OUTPUT_PREFIX = 'CAMELS_Lev01_Evap'
 
START_YEAR = 2009
END_YEAR = 2023  # Inclusive end year implies range(2009, 2024)
 
def initialize_earth_engine(project_id: str):
    """Initializes the Earth Engine API, prompting authentication if necessary."""
    try:
        ee.Initialize(project=project_id)
        print("Earth Engine initialized successfully.")
    except Exception:
        print("Authentication required. Initiating Earth Engine authentication...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
 
def process_year_chunk(year: int, dataset: ee.ImageCollection, basins: ee.FeatureCollection) -> ee.FeatureCollection:
    """
    Processes evaporation data for a given year.
    Converts dekadal mm/day rates to total dekadal mm, computes monthly sums,
    and calculates spatial means for the provided basin geometries.
    Args:
        year (int): The processing year.
        dataset (ee.ImageCollection): The WaPOR ImageCollection.
        basins (ee.FeatureCollection): The basin geometries.
    Returns:
        ee.FeatureCollection: A flattened collection of spatial statistics per basin per month.
    """
    start_date = ee.Date.fromYMD(year, 1, 1)
    months = ee.List.sequence(1, 12)
 
    def process_month(m):
        m = ee.Number(m)
        current_date = start_date.advance(m.subtract(1), 'month')
 
        # Filter collection for the specific month
        month_col = dataset.filterDate(current_date, current_date.advance(1, 'month'))
 
        def compute_stats():
            def prep_image(img):
                # Apply Scale Factor (0.1) to convert to mm/day
                evap_rate = img.select('L1_E_D').multiply(0.1)
 
                # Calculate multiplier for the dekad
                img_date = ee.Date(img.get('system:time_start'))
                day_of_month = img_date.get('day')
                days_in_month = img_date.advance(1, 'month').difference(img_date.update(day=1), 'day')
 
                multiplier_val = ee.Number(ee.Algorithms.If(
                    day_of_month.lte(20),
                    10.0,
                    days_in_month.subtract(20)
                ))
 
                multiplier_img = ee.Image.constant(multiplier_val).toFloat()
 
                # Convert (mm/day) to (Total mm per dekad)
                evap_total_mm = evap_rate.multiply(multiplier_img).rename('evap_dekad_total')
 
                return evap_total_mm.copyProperties(img, ['system:time_start'])
 
            processed_col = month_col.map(prep_image)
 
            # Aggregate monthly total
            monthly_total_img = processed_col.sum().rename("evap_monthly_sum")
 
            # Compute spatial mean across basins
            stats = monthly_total_img.reduceRegions(
                collection=basins,
                reducer=ee.Reducer.mean(),
                scale=250,
                crs='EPSG:4326',
                tileScale=4
            )
 
            return stats.map(lambda f: f.set({
                'Year': year,
                'Month': m,
                'Basin_ID': f.get(BASIN_ID_FIELD)
            }))
 
        def return_empty():
            return ee.FeatureCollection([])
 
        return ee.FeatureCollection(
            ee.Algorithms.If(month_col.size().gt(0), compute_stats(), return_empty())
        )
 
    monthly_features = months.map(process_month)
    return ee.FeatureCollection(monthly_features).flatten()
 
def main():
    """Main execution routine for exporting spatial statistics."""
    initialize_earth_engine(PROJECT_ID)
 
    # Load spatial features
    basins = ee.FeatureCollection(BASIN_ASSET_ID)
    print(f"Basins loaded. Total count: {basins.size().getInfo()}")
 
    # Load WaPOR Level 1 Evaporation (Dekadal)
    wapor_dataset = ee.ImageCollection("FAO/WAPOR/2/L1_E_D")
 
    years = range(START_YEAR, END_YEAR + 1)
    print("Initiating WaPOR Evaporation workflow (Spatial Mean Aggregation)...")
 
    for year in tqdm(years, desc="Submitting Earth Engine Tasks"):
        task_desc = f"{OUTPUT_PREFIX}_{year}"
        fc_out = process_year_chunk(year, wapor_dataset, basins)
 
        # Filter out features with null mean values
        fc_out = fc_out.filter(ee.Filter.notNull(['mean']))
 
        columns = ['Basin_ID', 'Year', 'Month', 'mean']
 
        task = ee.batch.Export.table.toDrive(
            collection=fc_out,
            description=task_desc,
            folder=OUTPUT_FOLDER,
            fileNamePrefix=task_desc,
            fileFormat="CSV",
            selectors=columns
        )
 
        try:
            task.start()
        except Exception as e:
            print(f"Error submitting task for {task_desc}: {e}")
            time.sleep(5)
 
    print("\nAll tasks successfully submitted to the Earth Engine batch queue.")
 
if __name__ == '__main__':
    main()
"""
Earth Engine Workflow: Actual Evapotranspiration and Interception (AETI)
 
This script extracts and scales dekadal (10-day) Actual Evapotranspiration
and Interception (AETI) data from the FAO WAPOR v2 dataset. The data is spatially
averaged over specified hydrological basins and exported year-by-year.
 
Methodology:
- Data Source: FAO WAPOR V2 (L1_AETI_D)
- Temporal Resolution: Dekadal temporally integrated to Monthly
- Spatial Aggregation: Mean AETI per basin
- Gap Filling: Monthly sum is scaled based on the ratio of available dekads
  to the expected dekads (3) per month.
"""
 
import ee
import time
import logging
 
# Configure logging for academic/professional output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 
def initialize_earth_engine(project_id: str):
    """Initializes the Earth Engine API securely."""
    try:
        ee.Initialize(project=project_id)
        logging.info("Earth Engine initialized successfully.")
    except Exception:
        logging.warning("Standard initialization failed. Prompting for authentication.")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        logging.info("Earth Engine authenticated and initialized.")
 
def run_wapor_aeti_extraction():
    """Main workflow to process and export WAPOR AETI data annually."""
   
    # --------------------------
    # 1. Configuration & Assets
    # --------------------------
    PROJECT_ID = 'camels-afr'
    ASSET_ID = 'projects/camels-afr/assets/CAMELS_Afr_L1_Basin_Boundaries'
    BASIN_ID_FIELD = 'Basin_ID'
    OUTPUT_FOLDER = "Actual_Evotranspiration_WAPOR_Lev_02"
    OUTPUT_PREFIX = "CAMELS_Lev02_eta"
   
    START_YEAR = 2009
    END_YEAR = 2023
 
    initialize_earth_engine(PROJECT_ID)
 
    basins = ee.FeatureCollection(ASSET_ID)
    dataset = ee.ImageCollection("FAO/WAPOR/2/L1_AETI_D")
 
    # --------------------------
    # 2. Server-Side Processing Function
    # --------------------------
    def process_month_gee(ym_dict):
        """Processes a single Year-Month combination on EE servers."""
        ym = ee.Dictionary(ym_dict)
        year = ym.getNumber('year')
        month = ym.getNumber('month')
       
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')
 
        # Filter for the specific month
        month_col = dataset.filterDate(start_date, end_date).select('L1_AETI_D')
 
        # DUMMY IMAGE INJECTION:
        # Prevents server-side graph failures if a month has 0 images
        dummy_img = ee.Image.constant(0) \
            .rename('L1_AETI_D') \
            .set('system:time_start', start_date.millis()) \
            .updateMask(0)
       
        safe_col = month_col.merge(ee.ImageCollection([dummy_img]))
 
        def calculate_dekad_aeti(img):
            img_date = img.date()
            day = img_date.get('day')
           
            # Calculate days in the month to appropriately scale the 3rd dekad
            days_in_month = img_date.advance(1, 'month').difference(img_date.update(day=1), 'day')
            duration = ee.Number(ee.Algorithms.If(day.gt(20), days_in_month.subtract(20), 10.0))
 
            # Multiply by scale factor (0.1) and temporal duration
            aeti = img.multiply(0.1).multiply(duration).rename('aeti_total').float()
           
            # Presence mask (1 for data, 0 for missing)
            count = img.mask().rename('dekad_count').float()
 
            return aeti.addBands(count)
 
        # Sum dekads for the month
        monthly_img = safe_col.map(calculate_dekad_aeti).sum()
       
        aeti_sum = monthly_img.select('aeti_total')
        d_count = monthly_img.select('dekad_count')
       
        # GAP SCALING MATH
        # Scaled Result = (Sum of available data / number of available dekads) * 3
        # updateMask(d_count.gt(0)) ensures areas with absolutely zero data return Null rather than 0.
        scaled_aeti = aeti_sum.divide(d_count.max(0.001)).multiply(3) \
            .rename('aeti_monthly_sum') \
            .updateMask(d_count.gt(0))
 
        # Reducer: Spatially average the scaled monthly totals per basin
        reducer = ee.Reducer.mean().setOutputs(['aeti_monthly_sum'])
 
        stats = scaled_aeti.reduceRegions(
            collection=basins,
            reducer=reducer,
            scale=250,
            crs='EPSG:4326',
            tileScale=16
        )
 
        # Standardize feature properties and rename the ID field to 'Basin_ID'
        def format_feature(f):
            return ee.Feature(None, {
                'Year': year,
                'Month': month,
                'Basin_ID': f.get(BASIN_ID_FIELD),
                'aeti_monthly_sum': f.get('aeti_monthly_sum')
            })
 
        return stats.map(format_feature)
 
    # --------------------------
    # 3. Client-Side Loop for Task Submission
    # --------------------------
    logging.info("Starting batch submission of annual tasks...")
 
    # We iterate through the years locally in Python to create one task per year
    for year in range(START_YEAR, END_YEAR + 1):
        task_desc = f"{OUTPUT_PREFIX}_{year}"
       
        try:
            # Create a server-side list of 12 months for this specific year
            months_seq = ee.List.sequence(1, 12)
            ym_list = months_seq.map(lambda m: ee.Dictionary({'year': year, 'month': m}))
           
            # Process all 12 months on Google's servers and flatten into one FeatureCollection
            annual_fc = ee.FeatureCollection(ym_list.map(process_month_gee)).flatten()
 
            # Ensure only the required columns are exported, enforcing 'Basin_ID'
            selectors = ['Basin_ID', 'Year', 'Month', 'aeti_monthly_sum']
 
            task = ee.batch.Export.table.toDrive(
                collection=annual_fc,
                description=task_desc,
                folder=OUTPUT_FOLDER,
                fileNamePrefix=task_desc,
                fileFormat="CSV",
                selectors=selectors
            )
 
            task.start()
            logging.info(f"Submitted Task: {task_desc}")
           
            # Brief pause to prevent hitting Earth Engine's batch API rate limits
            time.sleep(0.5)
 
        except Exception as e:
            logging.error(f"Failed to submit task for {year}: {e}")
 
    logging.info("All tasks have been submitted to the Earth Engine Task Manager.")
 
if __name__ == "__main__":
    run_wapor_aeti_extraction()
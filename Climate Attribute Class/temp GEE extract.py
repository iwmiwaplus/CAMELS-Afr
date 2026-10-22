import ee
 
# Initialize Earth Engine. Please follow the authentication prompts in the output.
ee.Authenticate()
ee.Initialize(project='my-project-512206')
 
# GEE Script: Africa Daily Basin Temp (1980-2024)
# Asset: Custom Shapefile
# Resolution: Native (11132m)
 
# 1. Load your Asset
basins = ee.FeatureCollection(".../assets/CAMELS_Afr_L1_Basin_Boundaries")
 
# 2. Loop through each year from 1980 to 2024
for year in range(1980, 2026):
    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = startDate.advance(1, 'year')
   
    # Load ERA5-Land Hourly
    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            .filterDate(startDate, endDate)
            .select('temperature_2m'))
 
    # Create list of days for this specific year
    nDays = endDate.difference(startDate, 'day')
    dayList = ee.List.sequence(0, nDays.subtract(1))
 
    # 3. Process each day
    def process_day(dayOffset):
        date = startDate.advance(dayOffset, 'day')
        dayColl = era5.filterDate(date, date.advance(1, 'day'))
       
        # Create Daily Mean, Min, Max (K to C)
        dailyImg = ee.Image.cat([
            dayColl.mean().subtract(273.15).rename('Tmean'),
            dayColl.min().subtract(273.15).rename('Tmin'),
            dayColl.max().subtract(273.15).rename('Tmax')
        ])
 
        # 4. Spatial Reduction (Native Scale)
        reduced = dailyImg.reduceRegions(
            collection=basins,
            reducer=ee.Reducer.mean(),
            scale=11132,
            tileScale=4
        )
 
        def format_feature(f):
            # Rounding to 1 decimal
            def round_val(val):
                num = ee.Number(val)
                return ee.Algorithms.If(num, num.multiply(100).round().divide(100), None)
           
            # Strip geometry for CSV stability
            return ee.Feature(None, {
                'Basin_ID': f.get('Basin_ID'),
                'Date': date.format('YYYY-MM-DD'),
                'temp_mean': round_val(f.get('Tmean')),
                'temp_min': round_val(f.get('Tmin')),
                'temp_max': round_val(f.get('Tmax'))
            })
           
        return reduced.map(format_feature)
 
    dailyStats = dayList.map(process_day)
   
    finalCollection = (ee.FeatureCollection(dailyStats)
                       .flatten()
                       .filter(ee.Filter.notNull(['temp_mean'])))
 
    # 5. Create an Export Task for the current year
    task = ee.batch.Export.table.toDrive(
        collection=finalCollection,
        description=f'Africa_L1_Temp_Daily_{year}',
        folder='Africa_L1_Temp_Data_1980_2025',
        fileNamePrefix=f'Africa_L1_Temp_{year}',
        fileFormat='CSV',
        selectors=['Basin_ID', 'Date', 'temp_mean', 'temp_min', 'temp_max']
    )
    task.start()
 
print('Tasks for 1980-2025 have been started. Check your GEE Tasks tab/dashboard.')

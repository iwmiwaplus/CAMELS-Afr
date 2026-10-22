'''

Requirements:  pip install pandas geopandas

'''

import pandas as pd
import geopandas as gpd
import os
from datetime import datetime, timedelta

# 1. DEFINE PATHS
input_dir = r"...\Africa_L1_Temp_Data_1980_2025"  # Folder containing all the input files
shp_path = r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"
output_dir = r"...\L1"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def fix_broken_date(date_str):
    """
    Fixes dates like '1980-02-32' by treating the last part
    as the Day of the Year.
    """
    date_str = str(date_str).strip()

    # If it contains '-', it's likely the broken format from GEE (YYYY-MM-DOY)
    if "-" in date_str:
        try:
            parts = date_str.split("-")
            year = int(parts[0])
            day_of_year = int(parts[2])
            # Calculate actual date: Jan 1st + (DayCount - 1)
            actual_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
            return actual_date
        except:
            return pd.NaT

    # If it's the standard format (1/31/1980), parse normally
    return pd.to_datetime(date_str, errors='coerce')

# 2. CREATE ID-TO-NAME MAPPING FROM SHAPEFILE
print("Reading shapefile attribute table...")
gdf = gpd.read_file(shp_path)

# Detect appropriate ID column in shapefile (handle potential case variations)
columns_list = list(gdf.columns)
print("Columns found in shapefile:", columns_list)

if 'HYBAS_ID' in gdf.columns:
    id_col = 'HYBAS_ID'
elif 'hybas_id' in gdf.columns:
    id_col = 'hybas_id'
else:
    # Fallback/Error handling: identify which column contains 'id'
    potential_cols = [c for c in gdf.columns if 'id' in c.lower()]
    if potential_cols:
        id_col = potential_cols[0]
        print(f"  Using '{id_col}' as the ID column mapping.")
    else:
        raise KeyError("Could not find 'HYBAS_ID', 'hybas_id', or any other ID column in the shapefile. Please verify columns list printed above.")

# Dictionary: {ID: ID}
id_map = pd.Series(gdf[id_col].values, index=gdf[id_col]).to_dict()

# 3. LOAD AND COMBINE YEARLY CSVS
all_dfs = []

print("Loading and fixing yearly CSVs...")
for year in range(1980, 2026):
    # We dynamically check for both L4 and L3 file patterns
    file_name = f"Africa_L4_Temp_{year}.csv"
    file_path = os.path.join(input_dir, file_name)

    # If L4 doesn't exist, check for L3 format
    if not os.path.exists(file_path):
        file_name = f"Africa_L3_Temp_{year}.csv"
        file_path = os.path.join(input_dir, file_name)

    if os.path.exists(file_path):
        print(f"  Processing: {file_name}")
        df = pd.read_csv(file_path)

        # APPLY THE FIX TO THE DATE COLUMN FIRST
        df['Date'] = df['Date'].apply(fix_broken_date)

        # Drop rows where date couldn't be fixed (if any)
        df = df.dropna(subset=['Date'])

        # Dynamically detect ID column in CSV
        csv_id_col = None
        for col in ['HYBAS_ID', 'hybas_id', 'Basin_ID', 'basin_id']:
            if col in df.columns:
                csv_id_col = col
                break
        if not csv_id_col:
            potential_csv_id_cols = [c for c in df.columns if 'id' in c.lower()]
            if potential_csv_id_cols:
                csv_id_col = potential_csv_id_cols[0]

        if not csv_id_col:
            print(f"  Warning: Could not find an ID column in {file_name}. Skipping.")
            continue

        # Replace numeric ID with shapefile mappings
        df['HYBAS_ID'] = df[csv_id_col].map(id_map).fillna(df[csv_id_col])

        all_dfs.append(df[['Date', 'HYBAS_ID', 'temp_mean', 'temp_min', 'temp_max']])
    else:
        # Silence the warnings to keep the log clean
        pass

# Merge all years into one master dataframe
if len(all_dfs) == 0:
    raise FileNotFoundError(f"No files found in '{input_dir}' matching 'Africa_L4_Temp_*.csv' or 'Africa_L3_Temp_*.csv'. Please check your directory contents.")

print(f"Merging {len(all_dfs)} years...")
master_df = pd.concat(all_dfs, ignore_index=True)

# 4. CREATE THE 3 FINAL Wide-Format CSVS
metrics = {
    'temp_mean': 'CAMELS_Afr_Climate_Daily_L4_t_mean.csv',
    'temp_min': 'CAMELS_Afr_Climate_Daily_L4_t_min.csv',
    'temp_max': 'CAMELS_Afr_Climate_Daily_L4_t_max.csv'
}

for metric, filename in metrics.items():
    print(f"Pivoting data for {metric}...")

    # Pivot: Dates as rows, Basin Names as columns
    final_df = master_df.pivot_table(index='Date', columns='HYBAS_ID', values=metric, aggfunc='mean')

    # Sort dates chronologically
    final_df.sort_index(inplace=True)

    # Format Date index to M/D/YYYY (No time, No leading zeros)
    final_df.index = final_df.index.map(lambda x: f"{x.month}/{x.day}/{x.year}")
    final_df.index.name = 'Date'

    # Save the file
    save_path = os.path.join(output_dir, filename)
    final_df.to_csv(save_path)
    print(f"  Successfully saved: {filename}")

print("\nAll tasks complete! Your files are in:", output_dir)
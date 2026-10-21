import pandas as pd
import os
import glob

# 1. Define the input and output paths
# Using 'r' before the string to handle Windows backslashes correctly

input_folder = ".....\Africa_Sinai_ERA5Land_TotalEvaporation_Data"
output_file = "......\CAMELS_Afr_Sinai_Climate_Monthly_L4_evp_era5_land.csv"

# 2. Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

print(f"Found {len(csv_files)} files. Starting processing...")

all_data_frames = []

# 3. Loop through files and read relevant columns
for file in csv_files:
    # Read only the necessary columns to save memory
    # We use dtype={'Basin_ID': str} to prevent IDs like '1070072290_1' from causing errors
    df = pd.read_csv(file, usecols=['Date', 'Basin_ID', 'Evap_mm'], dtype={'Basin_ID': str})
    all_data_frames.append(df)

# 4. Combine all years into one large "Long Format" dataframe
combined_df = pd.concat(all_data_frames, ignore_index=True)

# 5. Convert Date column to datetime objects to ensure they sort chronologically
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# 6. Pivot the data: Rows = Date, Columns = Basin_ID, Values = Evap_mm
# We use pivot_table in case there are any duplicate entries for a date/basin combo
final_df = combined_df.pivot_table(index='Date', columns='Basin_ID', values='Evap_mm')

# 7. Sort by date just in case
final_df = final_df.sort_index()

# 8. Ensure the output directory exists
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 9. Save to the final CSV
# We format the date back to a string style (e.g., 1/1/1980) as seen in your example
final_df.to_csv(output_file, date_format='%m/%d/%Y')

print(f"Success! Final file saved at: {output_file}")
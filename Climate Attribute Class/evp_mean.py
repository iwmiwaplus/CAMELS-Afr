import pandas as pd
import glob
import os

# 1. Define Paths
input_folder = r"...\Evaporation_WAPOR_Lev_01"
output_folder = r"...\ETA WAPOR"
output_filename = "CAMELS_Afr_Climate_L1_evp_mean.csv"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Get all CSV files
file_pattern = os.path.join(input_folder, "*.csv")
all_files = glob.glob(file_pattern)

all_data = []
print(f"Found {len(all_files)} files. Starting processing...")

# 3. Load and prepare data
for file in all_files:
    try:
        df = pd.read_csv(file)
        # Required columns for Evaporation files
        cols = ['Basin_ID', 'Year', 'Month', 'mean']
        
        if all(col in df.columns for col in cols):
            # Ensure Month and Year are integers
            df['Month'] = df['Month'].astype(int)
            df['Year'] = df['Year'].astype(int)
            all_data.append(df[cols])
        else:
            missing = [c for c in cols if c not in df.columns]
            print(f"Skipping {os.path.basename(file)}: Missing columns {missing}")
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not all_data:
    print("No valid data found to process.")
else:
    # 4. Combine and Restructure
    combined_df = pd.concat(all_data, ignore_index=True)

    # Create Date object for sorting
    combined_df['Date_obj'] = pd.to_datetime({
        'year': combined_df['Year'], 
        'month': combined_df['Month'], 
        'day': 1
    })

    # Pivot: Basin_ID becomes columns, 'mean' is the value
    master_df = combined_df.pivot_table(
        index='Date_obj', 
        columns='Basin_ID', 
        values='mean'
    )

    # Sort rows chronologically
    master_df = master_df.sort_index()

    # Format Date index to mm/dd/yy (e.g., 01/01/09)
    master_df.index = master_df.index.strftime('%m/%d/%y')

    # Move 'Date' to the first column
    master_df.reset_index(inplace=True)
    master_df.rename(columns={'Date_obj': 'Date'}, inplace=True)

    # 5. Save output
    final_path = os.path.join(output_folder, output_filename)
    master_df.to_csv(final_path, index=False)

    print(f"Successfully saved master CSV to: {final_path}")
    print(f"Final shape: {master_df.shape}")
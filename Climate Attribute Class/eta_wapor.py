import pandas as pd
import glob
import os

# 1. Define Paths
input_folder = r"...\Actual_Evotranspiration_WAPOR_Lev_02"
output_folder = r"...\ETA WAPOR"
output_filename = "CAMELS_Afr_Sinai_Climate_L3_eta_wapor.csv"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Find all CSV files
file_pattern = os.path.join(input_folder, "*.csv")
all_files = glob.glob(file_pattern)

all_data = []
print(f"Found {len(all_files)} files in Level 1 folder. Starting processing...")

# 3. Load and prepare data
for file in all_files:
    try:
        df = pd.read_csv(file)
        # Required columns
        cols = ['Basin_ID', 'Year', 'Month', 'aeti_monthly_sum']
        
        if all(col in df.columns for col in cols):
            # Clean Month and Year types
            df['Month'] = df['Month'].astype(int)
            df['Year'] = df['Year'].astype(int)
            all_data.append(df[cols])
        else:
            missing = [c for c in cols if c not in df.columns]
            print(f"Skipping {os.path.basename(file)}: Missing {missing}")
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not all_data:
    print("No valid data found to process.")
else:
    # 4. Combine and Pivot
    combined_df = pd.concat(all_data, ignore_index=True)

    # Convert Year/Month to a sortable Date object
    combined_df['Date_obj'] = pd.to_datetime({
        'year': combined_df['Year'], 
        'month': combined_df['Month'], 
        'day': 1
    })

    # Restructure: Rows = Date, Columns = Basin_ID
    master_df = combined_df.pivot_table(
        index='Date_obj', 
        columns='Basin_ID', 
        values='aeti_monthly_sum'
    )

    # Sort rows chronologically
    master_df = master_df.sort_index()

    # Format Date index to mm/dd/yy (e.g., 01/01/09)
    master_df.index = master_df.index.strftime('%m/%d/%y')

    # Reset index to make 'Date' the first column
    master_df.reset_index(inplace=True)
    master_df.rename(columns={'Date_obj': 'Date'}, inplace=True)

    # 5. Save final result
    final_path = os.path.join(output_folder, output_filename)
    master_df.to_csv(final_path, index=False)

    print(f"Successfully saved Level 1 master CSV to: {final_path}")
    print(f"Total Rows (Months): {master_df.shape[0]}")
    print(f"Total Columns (Basins): {master_df.shape[1] - 1}")
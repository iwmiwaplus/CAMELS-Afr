import pandas as pd
import os
 
# 1. File paths
daily_file = ".../CAMELS_Afr_Climate_Daily_L1_t_mean.csv"
output_file = ".../CAMELS_Afr_Climate_Annual_L1_t_mean.csv"
 
# 2. LOAD DATA
if os.path.exists(daily_file):
    print(f"Reading daily data from: {daily_file}")
 
    df = pd.read_csv(daily_file)
 
    # 3. PROCESS DATES
    print("Processing dates...")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
 
    # 4. AGGREGATE BY YEAR
    print("Calculating annual mean temperatures...")
 
    df["Year"] = df["Date"].dt.year
 
    # Identify basin columns
    basin_columns = [
        col for col in df.columns
        if col not in ["Date", "Year"]
    ]
 
    # Convert basin values to numeric
    df[basin_columns] = df[basin_columns].apply(
        pd.to_numeric, errors="coerce"
    )
 
    # Calculate annual mean temperature for each basin
    annual_df = df.groupby("Year")[basin_columns].mean()
 
    # 5. SAVE RESULTS (without rounding)
    annual_df.to_csv(output_file)
 
    print("-" * 30)
    print("SUCCESS!")
    print(
        f"Processed years: {annual_df.index.min()} "
        f"to {annual_df.index.max()}"
    )
    print(f"Number of basins: {len(basin_columns)}")
    print(f"File saved to: {output_file}")
    print("-" * 30)
 
else:
    print(f"Error: Could not find the file: {daily_file}")
    print("Please check the input filename and folder.")
"""
CAMELS-Africa Land Cover Processing Pipeline
============================================
 
This script processes multi-basin land cover datasets (GVF and LAI),
performs in-memory quality filtering and temporal structuring,
and reshapes the data into basin-wise monthly matrices.
 
Output:
-------
For each GVF/LAI variable:
    - Monthly time series (rows = Month)
    - Basin-wise values (columns = Basin_IDs)
 
Author: Kavishka Palihena
Date: 5/15/2026
"""
 
# =========================================================
# 1. IMPORTS
# =========================================================
 
import os
import glob
import pandas as pd
 
# =========================================================
# 2. USER CONFIGURATION
# =========================================================
 
INPUT_DIR = "/path/to/your/gogle_earth_engine outputs/spatial_level"
OUTPUT_DIR =  REPO_ROOT / "CAMELS-Africa" / {SPATIAL_LEVEL_NAME} / "Land Cover Attribute Class" / "CAMELS_Afr_LandCover_Monthly_{SPATIAL_LEVEL_NAME}"
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# =========================================================
# 3. MASTER DATASET CREATION (Read & Clean In-Memory)
# =========================================================
 
def build_master_dataset(input_dir: str) -> pd.DataFrame:
    """
    Loads all yearly CSV files, applies quality filtering on the fly,
    and concatenates them into a single master dataset.
 
    Returns
    -------
    pd.DataFrame
    """
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    df_list = []
   
    print(f"Found {len(files)} CSV files in input directory. Processing...")
 
    for file in files:
        df = pd.read_csv(file)
 
        # -------------------------------------------------
        # Identify GVF and LAI variables
        # -------------------------------------------------
        value_cols = [
            c for c in df.columns
            if ("gvf" in c.lower()) or ("lai" in c.lower())
        ]
 
        # Safety Check: Ensure necessary columns exist before processing
        if not value_cols or not {"Year", "Month", "Basin_ID"}.issubset(df.columns):
            continue
 
        # -------------------------------------------------
        # Quality filtering using valid percentage
        # -------------------------------------------------
        #if "valid_pct" in df.columns:
        #    df.loc[df["valid_pct"] < 50, value_cols] = 0
 
        # -------------------------------------------------
        # Create proper monthly datetime index
        # -------------------------------------------------
        df["Month"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
 
        # -------------------------------------------------
        # Subset, clean NaNs, and append to list
        # -------------------------------------------------
        keep_cols = ["Basin_ID", "Month"] + value_cols
        df_clean = df[keep_cols].copy()
       
        # fillna(0) inherently handles 'NA' strings parsed as missing values
        df_clean = df_clean.fillna(0)
        df_list.append(df_clean)
 
    if not df_list:
        raise ValueError("No valid data found to process. Check input files and column names.")
 
    # -----------------------------------------------------
    # Combine all cleaned yearly data into one master frame
    # -----------------------------------------------------
    master_df = pd.concat(df_list, ignore_index=True)
 
    # Temporal filtering: remove 2002 Jan–Jun
    master_df = master_df[master_df["Month"] >= "2002-07-01"]
 
    return master_df
 
# =========================================================
# 4. RESHAPING FUNCTION (FINAL OUTPUT)
# =========================================================
 
def export_gvf_lai_timeseries(master_df: pd.DataFrame, output_dir: str):
    """
    Converts long-format dataset into basin-wise monthly matrices.
 
    Output:
    -------
    One CSV per GVF/LAI variable:
        Rows    → Monthly timestamps
        Columns → Basin_IDs
    """
    # Identify GVF/LAI variables for export
    variables = [
        c for c in master_df.columns
        if ("gvf" in c.lower()) or ("lai" in c.lower())
    ]
 
    print(f"\nIdentified {len(variables)} GVF/LAI variables to export.")
 
    for var in variables:
        # Pivot to Month × Basin_ID structure
        df_pivot = master_df.pivot_table(
            index="Month",
            columns="Basin_ID",
            values=var,
            aggfunc="mean"
        ).reset_index()
 
        # Save output
        output_file = os.path.join(
            output_dir,
            f"CAMELS_Afr_LandCover_Monthly_{SPATIAL_LEVEL}_{var}.csv"
        )
 
        df_pivot.to_csv(output_file, index=False)
        print(f"Saved: {os.path.basename(output_file)}")
 
# =========================================================
# 5. PIPELINE EXECUTION
# =========================================================
 
if __name__ == "__main__":
 
    print("\n==================================")
    print(" CAMELS-AFRICA PROCESSING PIPELINE")
    print("==================================\n")
 
    print("Step 1: Reading, cleaning, and building master dataset...")
    master_data = build_master_dataset(INPUT_DIR)
 
    print("\nStep 2: Exporting final variable-wise matrices...")
    export_gvf_lai_timeseries(master_data, OUTPUT_DIR)
 
    print("\nProcessing completed successfully.")
import pandas as pd
from pathlib import Path
 
# File paths
precip_file = ".../Daily/CAMELS_Afr_Climate_Daily_L1_p_mean_arc.csv"
pet_file = ".../Daily/CAMELS_Afr_Climate_Daily_L1_pet_mean.csv"
output_file = ".../CAMELS_Afr_Climate_Annual_L1_aridity_arc.csv"
 
# 1. Load the datasets
print("Loading files...")
df_precip = pd.read_csv(precip_file)
df_pet = pd.read_csv(pet_file)
 
# 2. Process dates
print("Processing dates...")
df_precip["Date"] = pd.to_datetime(
    df_precip["Date"], errors="coerce"
)
df_pet["Date"] = pd.to_datetime(
    df_pet["Date"], errors="coerce"
)
 
df_precip.dropna(subset=["Date"], inplace=True)
df_pet.dropna(subset=["Date"], inplace=True)
 
df_precip.set_index("Date", inplace=True)
df_pet.set_index("Date", inplace=True)
 
# 3. Match basin IDs
df_precip.columns = df_precip.columns.astype(str)
df_pet.columns = df_pet.columns.astype(str)
 
common_basins = df_precip.columns.intersection(
    df_pet.columns
)
 
if len(common_basins) == 0:
    raise ValueError(
        "No common basin IDs found between the two files."
    )
 
df_precip = df_precip[common_basins].apply(
    pd.to_numeric, errors="coerce"
)
df_pet = df_pet[common_basins].apply(
    pd.to_numeric, errors="coerce"
)
 
# 4. Calculate annual totals
print(f"Resampling data for {len(common_basins)} basins...")
 
annual_precip_sum = df_precip.resample("YE").sum(min_count=1)
annual_pet_sum = df_pet.resample("YE").sum(min_count=1)
 
# 5. Calculate annual aridity index
annual_aridity = annual_precip_sum / annual_pet_sum.replace(
    0, float("nan")
)
 
# 6. Keep years with at least one valid result
annual_aridity.dropna(how="all", inplace=True)
 
# 7. Format the year index
annual_aridity.index = annual_aridity.index.year
annual_aridity.index.name = "Year"
 
# 8. Save the result
annual_aridity.to_csv(output_file)
 
print("-" * 30)
print("SUCCESS!")
print(
    f"Processed years: {annual_aridity.index.min()} "
    f"to {annual_aridity.index.max()}"
)
print(f"Number of basins: {len(common_basins)}")
print(f"File saved to: {output_file}")
print("-" * 30)
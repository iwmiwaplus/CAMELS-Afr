#----------------------------------------------------------------------------------------------------
#-------------------Identifying and extracting for multiple outlet basins----------------------------
#----------------------------------------------------------------------------------------------------
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon
import pandas as pd
import numpy as np
from scipy.stats import skew
import re

basins = gpd.read_file(".../CAMELS_Afr_Level1_Basins.shp")

merged_geom = basins.geometry.unary_union

if merged_geom.geom_type == 'Polygon':
    # Create a new polygon using only the exterior ring
    outer_poly = Polygon(merged_geom.exterior)
elif merged_geom.geom_type == 'MultiPolygon':
    # If there are islands (like Madagascar), do this for each part
    outer_polys = [Polygon(p.exterior) for p in merged_geom.geoms]
    outer_poly = unary_union(outer_polys)

africa_outline = outer_poly.boundary

boundary_gdf = gpd.GeoDataFrame(geometry=[africa_outline], crs=basins.crs)
boundary_gdf.to_file(".../africa_continental_boundary.shp")
print("Saved: africa_continental_boundary.shp (No inland lines)")

coastal_basins = basins[basins.intersects(africa_outline.buffer(0.001))]

coastal_basins.to_file(".../multiple_outlet_basins.shp")
print(f"Saved: coastal_basins_code.shp ({len(coastal_basins)} basins found)")

rivers = gpd.read_file(".../nextdownid_zero_seg.shp")
multi_basins = gpd.read_file(".../multiple_outlet_basins.shp")
boundary = gpd.read_file(".../africa_continental_boundary.shp")

print("Projecting layers...")
rivers_m = rivers.to_crs(epsg=3857)
multi_basins_m = multi_basins.to_crs(epsg=3857)
boundary_m = boundary.to_crs(epsg=3857)

boundary_buffer = boundary_m.geometry.buffer(500)

# Find river segments that intersect the boundary (the Outlets)
print("Finding outlet segments...")
outlet_segments = gpd.sjoin(
    rivers_m, 
    gpd.GeoDataFrame(geometry=boundary_buffer, crs=3857), 
    how="inner", 
    predicate="intersects"
)

# Clean up join artifacts to avoid column conflicts later
if 'index_right' in outlet_segments.columns:
    outlet_segments = outlet_segments.drop(columns=['index_right'])

# Spatial Join: Link outlet segments to their specific Level 1 Basin
# This identifies which 'multiple outlet' basin each segment belongs to
outlets_in_basins = gpd.sjoin(
    outlet_segments, 
    multi_basins_m[['Basin_ID', 'geometry']], 
    how="inner", 
    predicate="intersects"
)

# Identify the Monthly Discharge Columns
# From your schema, these are the columns like '1993_01', '2024_12', etc.
discharge_cols = [col for col in outlets_in_basins.columns if "_" in col and col[0].isdigit()]

# Aggregate (Sum) the discharge by Basin ID
print(f"Aggregating discharge for {len(discharge_cols)} months...")

# We group by the basin ID and sum all the monthly columns
# If your Level 1 file uses 'Basin_ID', replace 'basin_id' below
final_discharge_df = outlets_in_basins.groupby('Basin_ID')[discharge_cols].sum().reset_index()

# Save the final CSV
final_discharge_df.to_csv(".../multiple_outlet_basin_discharge_lvl1.csv", index=False)

print("-" * 30)
print(f"Success! Saved discharge for {len(final_discharge_df)} multiple-outlet basins.")
print(f"Data spans from {discharge_cols[0]} to {discharge_cols[-1]}.")

#----------------------------------------------------------------------------------------------------------------
#--------------------------Identifying and extracting for endorheic basins---------------------------------------
#----------------------------------------------------------------------------------------------------------------

import geopandas as gpd
import pandas as pd

basins_path = ".../CAMELS_Afr_Level1_Basins.shp"
coastal_path = ".../multiple_outlet_basins.shp"

basins = gpd.read_file(basins_path)
coastal = gpd.read_file(coastal_path)

if basins.crs != coastal.crs:
    coastal = coastal.to_crs(basins.crs)

print("Removing entire coastal basins...")

coastal_ids = coastal["Basin_ID"].unique()

# Keep only basins NOT in coastal list
basins_no_coastal = basins[~basins["Basin_ID"].isin(coastal_ids)].copy()

print(f"Original basins: {len(basins)}")
print(f"Coastal basins removed: {len(coastal_ids)}")
print(f"Remaining inland basins: {len(basins_no_coastal)}")

# Save result
output_path = ".../inland_basins.shp"

basins_no_coastal.to_file(output_path)

print("✔ Done! Saved to:", output_path)

import geopandas as gpd
import pandas as pd

rivers = gpd.read_file(".../river_network.shp")

endorheic_basins = gpd.read_file(".../inland_basins.shp")

if rivers.crs != endorheic_basins.crs:
    rivers = rivers.to_crs(endorheic_basins.crs)

# Ensure uparea numeric
rivers["uparea"] = pd.to_numeric(rivers["uparea"], errors="coerce")

#SPATIAL JOIN — ALL RIVERS INSIDE BASINS
print("Joining ALL river segments to basins...")

all_joined = gpd.sjoin(
    rivers,
    endorheic_basins[["Basin_ID", "geometry"]],
    how="inner",
    predicate="within"
)

all_joined = all_joined.drop(columns=["index_right"])

print(f"Total river segments inside basins: {len(all_joined)}")

#FIND MAX UPAREA PER BASIN (considering ALL segments)
basin_max_uparea = (
    all_joined.groupby("Basin_ID")["uparea"]
    .max()
    .reset_index()
    .rename(columns={"uparea": "max_uparea"})
)

# Merge max values back
all_joined = all_joined.merge(
    basin_max_uparea,
    on="Basin_ID",
    how="left"
)

# CHECK TERMINAL SEGMENTS (NextDownID == 0)
#    Keep only if they have basin max uparea
terminal_segments = all_joined[all_joined["NextDownID"] == 0].copy()

# Keep only those where uparea equals basin max
valid_terminal_segments = terminal_segments[
    terminal_segments["uparea"] == terminal_segments["max_uparea"]
].copy()

print(f"Valid endorheic terminal segments: {len(valid_terminal_segments)}")

output_path = ".../endorheic_segments.shp"

valid_terminal_segments.to_file(output_path)

print(f"Saved filtered terminal segments to:\n{output_path}")

# SELECT BASINS CONTAINING VALID ENDORHEIC TERMINALS
print("Selecting basins containing valid endorheic terminals...")

valid_basin_ids = valid_terminal_segments["Basin_ID"].unique()

endorheic_candidate_basins = endorheic_basins[
    endorheic_basins["Basin_ID"].isin(valid_basin_ids)
].copy()

print(f"Candidate endorheic basins: {len(endorheic_candidate_basins)}")

candidate_output = ".../endorheic_candidate_basins.shp"
endorheic_candidate_basins.to_file(candidate_output)
print("Saved candidate basins.")

# REMOVE BASINS WITHIN 10 km OF AFRICA BOUNDARY
print("Removing basins within 10 km of African boundary...")

boundary = gpd.read_file(".../africa_continental_boundary.shp")

# Ensure CRS match
if boundary.crs != endorheic_candidate_basins.crs:
    boundary = boundary.to_crs(endorheic_candidate_basins.crs)

# Convert to metric CRS for buffering
basins_proj = endorheic_candidate_basins.to_crs(3857)
boundary_proj = boundary.to_crs(3857)

# Create 10 km buffer
boundary_buffer = boundary_proj.buffer(10000)

boundary_union = boundary_buffer.unary_union

# Identify basins intersecting buffer
near_boundary = basins_proj.intersects(boundary_union)

removed_basins = basins_proj[near_boundary]
final_endorheic_basins = basins_proj[~near_boundary].copy()

print(f"Basins removed near boundary: {len(removed_basins)}")
print(f"Final endorheic basins: {len(final_endorheic_basins)}")

# Convert back to original CRS
final_endorheic_basins = final_endorheic_basins.to_crs(endorheic_candidate_basins.crs)
removed_basins = removed_basins.to_crs(endorheic_candidate_basins.crs)

final_output = ".../endorheic_basins_final.shp"
final_endorheic_basins.to_file(final_output)
print("Boundary filtering completed.")

# EXTRACT DISCHARGE FOR FINAL ENDORHEIC BASINS
print("Extracting discharge for final endorheic basins...")

endorheic = ".../endorheic_basins_final.shp"

# Load final basins
endorheic_basins_final = gpd.read_file(endorheic)

# Spatial join rivers with basins
rivers_in_basins = gpd.sjoin(
    rivers,
    endorheic_basins_final[["Basin_ID","geometry"]],
    how="inner",
    predicate="within"
)

rivers_in_basins = rivers_in_basins.drop(columns=["index_right"])

# Select terminal segments
terminal_segments = rivers_in_basins[
    rivers_in_basins["NextDownID"] == 0
].copy()

print(f"Terminal segments inside final basins: {len(terminal_segments)}")

# IDENTIFY MONTHLY DISCHARGE COLUMNS
discharge_cols = [
    col for col in terminal_segments.columns
    if col[:4].isdigit() and "_" in col
]

print(f"Monthly discharge columns found: {len(discharge_cols)}")

terminal_segments[discharge_cols] = (
    terminal_segments[discharge_cols]
    .apply(pd.to_numeric, errors="coerce")
)

# SUM DISCHARGE PER BASIN
basin_discharge = (
    terminal_segments
    .groupby("Basin_ID")[discharge_cols]
    .sum()
    .reset_index()
)

print(f"Basins in discharge table: {len(basin_discharge)}")

csv_output = ".../endorheic_basin_discharge.csv"
basin_discharge.to_csv(csv_output, index=False)
print("Discharge CSV saved.")

# --------------------------------------------------------------------------------------------------------------------
# -------------------------------Identifying and extracting for single outlet basins----------------------------------
# --------------------------------------------------------------------------------------------------------------------
basins_path = ".../inland_basins.shp"
coastal_path = ".../endorheic_basins_final.shp"

basins = gpd.read_file(basins_path)
coastal = gpd.read_file(coastal_path)

if basins.crs != coastal.crs:
    coastal = coastal.to_crs(basins.crs)

#Basin-level difference (REMOVE ENTIRE BASINS)
print("Removing entire coastal basins...")

# Get list of coastal HYBAS_IDs
coastal_ids = coastal["HYBAS_ID"].unique()

# Keep only basins NOT in coastal list
basins_no_coastal = basins[~basins["HYBAS_ID"].isin(coastal_ids)].copy()

print(f"Original basins: {len(basins)}")
print(f"Coastal basins removed: {len(coastal_ids)}")
print(f"Remaining inland basins: {len(basins_no_coastal)}")

output_path = ".../single_outlet_basins.shp"
basins_no_coastal.to_file(output_path)
print("✔ Done! Saved to:", output_path)

basins_path = ".../single_outlet_basins.shp"

rivers_path = ".../river_network.shp"

basins = gpd.read_file(basins_path)
rivers = gpd.read_file(rivers_path)

if rivers.crs != basins.crs:
    rivers = rivers.to_crs(basins.crs)

# Ensure uparea numeric
rivers["uparea"] = pd.to_numeric(rivers["uparea"], errors="coerce")

# Spatial join rivers to basins
print("Joining rivers to basins...")

joined = gpd.sjoin(
    rivers,
    basins[["Basin_ID", "geometry"]],
    how="inner",
    predicate="intersects"
)

joined = joined.drop(columns=["index_right"])

print(f"Total river segments inside basins: {len(joined)}")

# Find segment with MAX UPAREA in each basin
print("Selecting max uparea segment per basin...")

idx = joined.groupby("Basin_ID")["uparea"].idxmax()

max_segments = joined.loc[idx].copy()

print(f"Segments selected: {len(max_segments)}")

segments_output = ".../single_segments.shp"
max_segments.to_file(segments_output)
print("Saved max uparea segments shapefile.")

# Identify discharge columns
discharge_cols = [
    col for col in max_segments.columns
    if col[:4].isdigit() and "_" in col
]

print(f"Discharge columns found: {len(discharge_cols)}")

# Ensure numeric
max_segments[discharge_cols] = (
    max_segments[discharge_cols]
    .apply(pd.to_numeric, errors="coerce")
)

# Create discharge table
basin_discharge = max_segments[["Basin_ID"] + discharge_cols].copy()
print(f"Basins in discharge table: {len(basin_discharge)}")

csv_output = ".../single_outlet_basin_discharge.csv"
basin_discharge.to_csv(csv_output, index=False)
print("✔ Discharge CSV saved.")

# ----------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------Formatting---------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
single_fp     = ".../single_outlet_basins.csv"
multiple_fp   = ".../multiple_outlet_basins.csv"
endorheic_fp  = ".../endorheic_basins.csv"

output_file   = ".../CAMELS_Afr_Hydrology_L1_q.csv"

df_single = pd.read_csv(single_fp)
df_multiple = pd.read_csv(multiple_fp)
df_endo = pd.read_csv(endorheic_fp)

df_all = pd.concat([df_single, df_multiple, df_endo], ignore_index=True)

# Optional: remove duplicates if any basin appears twice
df_all = df_all.drop_duplicates(subset="Basin_ID")

# TRANSPOSE (FLIP)
df_t = df_all.set_index("Basin_ID").T.reset_index()

# Rename first column
df_t = df_t.rename(columns={"index": "Month"})

# CONVERT DATE FORMAT
df_t["Month"] = pd.to_datetime(df_t["Month"], format="%Y_%m").dt.strftime("%Y-%m-%d")

# Remove column index name
df_t.columns.name = None

df_t.to_csv(output_file, index=False)
print(f"✅ Final combined + reshaped file saved at:\n{output_file}")

# ---------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------Extracting for static variables------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

q_fp    = ".../CAMELS_Afr_Hydrology_L1_q.csv"
p_fp    = ".../CAMELS_Afr_Climate_Monthly_L1_p_chirps.csv"
area_fp = ".../Area_L1.csv"

out_fp  = ".../CAMELS_Afr_Hydrology_Static_L1.csv"

q_long = pd.read_csv(q_fp, parse_dates=["Month"])
p_long = pd.read_csv(p_fp, parse_dates=["Month"])
area_df = pd.read_csv(area_fp)

q_long = q_long.set_index("Month")
p_long = p_long.set_index("Month")

common_time = q_long.index.intersection(p_long.index)
q_long = q_long.loc[common_time]
p_long = p_long.loc[common_time]

basins = q_long.columns.tolist()

# CONVERT TO MATRIX
q_matrix = q_long[basins].apply(pd.to_numeric, errors="coerce").values
p_matrix = p_long[basins].apply(pd.to_numeric, errors="coerce").values

n_time = q_matrix.shape[0]

# CORE METRICS
q_mean = np.nanmean(q_matrix, axis=0)
annual_q = np.nansum(q_matrix, axis=0) / (n_time / 12)

# Percentiles
q_5  = np.nanpercentile(q_matrix, 5, axis=0)
q_10 = np.nanpercentile(q_matrix, 10, axis=0)
q_25 = np.nanpercentile(q_matrix, 25, axis=0)
q_50 = np.nanpercentile(q_matrix, 50, axis=0)
q_75 = np.nanpercentile(q_matrix, 75, axis=0)
q_90 = np.nanpercentile(q_matrix, 90, axis=0)
q_95 = np.nanpercentile(q_matrix, 95, axis=0)

# FDC slope
q_33 = np.nanpercentile(q_matrix, 33, axis=0)
q_66 = np.nanpercentile(q_matrix, 66, axis=0)
slope_fdc = (np.log(q_66 + 1e-6) - np.log(q_33 + 1e-6)) / 0.33

# Variability
q_cv   = np.nanstd(q_matrix, axis=0) / (q_mean + 1e-6)
q_skew = skew(q_matrix, axis=0, nan_policy='omit')

# HIGH / LOW FLOW
median_q = np.nanmedian(q_matrix, axis=0)
mean_q_long = np.nanmean(q_matrix, axis=0)

def get_freq_dur(matrix, threshold, condition='above'):
    freqs, durs = [], []
    for i in range(matrix.shape[1]):
        series = matrix[:, i]

        if condition == 'above':
            mask = series > threshold[i]
        else:
            mask = series <= threshold[i]

        mask = np.nan_to_num(mask, nan=False)

        events = np.diff(mask.astype(int)) == 1
        freqs.append(np.sum(events) / (len(series)/12))

        if np.any(mask):
            runs = np.diff(np.where(np.concatenate(([mask[0]], mask[:-1] != mask[1:], [True])))[0])[::2]
            durs.append(np.mean(runs))
        else:
            durs.append(np.nan)

    return freqs, durs

high_q_freq, high_q_dur = get_freq_dur(q_matrix, 9 * median_q, 'above')
low_q_freq,  low_q_dur  = get_freq_dur(q_matrix, 0.2 * mean_q_long, 'below')

zero_q_freq = np.sum(q_matrix == 0, axis=0) / n_time

# RLD
diffs = np.diff(q_matrix, axis=0)
rld = np.sum(diffs > 0, axis=0) / n_time

# MONTHLY CLIMATOLOGY
monthly_means = {}

for m in range(1, 13):
    mask = q_long.index.month == m
    monthly_means[m] = np.nanmean(q_matrix[mask, :], axis=0)

# CLIMATE METRICS
mean_p = np.nanmean(p_matrix, axis=0)

# RUNOFF RATIO (AREA IN km²)

# Convert area to m²
area_series = area_df.set_index("Basin_ID").loc[basins]["catchment_area"].values * 1e6

# Seconds in a month
seconds_per_month = 60 * 60 * 24 * 30

# Convert discharge (m³/s) → runoff depth (mm/month)
runoff_mm = (q_mean * seconds_per_month) / area_series * 1000

# Avoid division issues
runoff_mm = np.where(area_series > 0, runoff_mm, np.nan)

# Runoff ratio (dimensionless)
runoff_ratio = runoff_mm / mean_p

# Elasticity
stream_elas = []

for i in range(len(basins)):
    q = q_matrix[:, i]
    p = p_matrix[:, i]

    qy = pd.Series(q, index=q_long.index).resample("YE").sum()
    py = pd.Series(p, index=p_long.index).resample("YE").sum()

    df = pd.concat([qy, py], axis=1).dropna()
    df.columns = ["q", "p"]

    if len(df) < 5:
        stream_elas.append(np.nan)
        continue

    q_bar, p_bar = df["q"].mean(), df["p"].mean()
    dq = df["q"] - q_bar
    dp = df["p"] - p_bar

    elas = (dq/q_bar)/(dp/p_bar)
    stream_elas.append(np.nanmedian(elas))

# HFD
def calc_hfd(series):
    hfd = []

    for year, qy in series.groupby(series.index.year):
        total = qy.sum()
        if total <= 0:
            continue

        cum = qy.cumsum()
        hfd.append(cum[cum >= total/2].index[0].month)

    return pd.Series(hfd).mode().iloc[0] if len(hfd) else np.nan

hfd_mean = [calc_hfd(q_long[b]) for b in basins]

# BASEFLOW INDEX
def lyne_hollick(q, alpha=0.925):
    bf = np.zeros_like(q)
    for i in range(1, len(q)):
        bf[i] = alpha * bf[i-1] + (1+alpha)/2 * (q[i] - q[i-1])
        bf[i] = min(bf[i], q[i])
    bf[bf < 0] = 0
    return bf

baseflow_index = []
for b in basins:
    q = q_long[b].dropna().values
    if len(q) < 12:
        baseflow_index.append(np.nan)
        continue
    bf = lyne_hollick(q)
    baseflow_index.append(np.sum(bf)/np.sum(q))

# FINAL DATAFRAME
results = pd.DataFrame({
    "Basin_ID": basins,
    "q_mean": q_mean,
    "annual_q": annual_q,
    "runoff_ratio": runoff_ratio,
    "stream_elas": stream_elas,
    "hfd_mean": hfd_mean,
    "baseflow_index": baseflow_index,
    "slope_fdc": slope_fdc,
    "q_cv": q_cv,
    "Q5": q_5,
    "Q95": q_95,
    "Q_skew": q_skew,
    "high_q_freq": high_q_freq,
    "high_q_dur": high_q_dur,
    "low_q_freq": low_q_freq,
    "low_q_dur": low_q_dur,
    "zero_q_freq": zero_q_freq,
    "RLD": rld
})

# Add percentiles
results["Q10"] = q_10
results["Q25"] = q_25
results["Q50"] = q_50
results["Q75"] = q_75
results["Q90"] = q_90

# Monthly means
month_names = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
for i, m in enumerate(month_names):
    results[f"mean_{m}_flow"] = monthly_means[i+1]

results.to_csv(out_fp, index=False)

print(f"✅ DONE: {len(results.columns)} variables for {len(results)} basins")

import geopandas as gpd
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ======================================================
# 1. FILE PATHS
# ======================================================

glim_fp = r"...\LiMW_GIS 2015.gdb.zip"
glim_layer = "GLiM_export"

basin_fp = r"...\CAMELS_Afr_L1_Basin_Boundaries.shp"

out_dir = Path(
    r"...\Output"
)
out_dir.mkdir(parents=True, exist_ok=True)

# ======================================================
# 2. LOAD DATA
# ======================================================

glim = gpd.read_file(glim_fp, layer=glim_layer)
basins = gpd.read_file(basin_fp)

print("GLiM CRS:", glim.crs)
print("Basins CRS:", basins.crs)

# Validate required columns
if "xx" not in glim.columns:
    raise KeyError("The GLiM dataset does not contain the 'xx' field.")

if "Basin_ID" not in basins.columns:
    raise KeyError("The basin dataset does not contain 'Basin_ID'.")

# Use xx as the lithology class
glim = glim[["xx", "geometry"]].copy()
glim = glim.rename(columns={"xx": "dom_class"})
glim = glim.dropna(subset=["dom_class"])

basins["Basin_ID"] = basins["Basin_ID"].astype(str)

# ======================================================
# 3. REPROJECT TO EQUAL-AREA CRS
# ======================================================

target_crs = "ESRI:54012"

glim = glim.to_crs(target_crs)
basins = basins.to_crs(target_crs)

# ======================================================
# 4. PROCESS EACH BASIN
# ======================================================

records = []

for _, basin in tqdm(
    basins.iterrows(),
    total=len(basins),
    desc="Processing basins"
):
    basin_id = basin["Basin_ID"]

    basin_gdf = gpd.GeoDataFrame(
        {"Basin_ID": [basin_id]},
        geometry=[basin.geometry],
        crs=basins.crs
    )

    try:
        clipped = gpd.overlay(
            glim,
            basin_gdf,
            how="intersection",
            keep_geom_type=False
        )
    except Exception as e:
        print(f"Overlay error for basin {basin_id}: {e}")
        continue

    if clipped.empty:
        continue

    # Keep polygon intersections only
    clipped = clipped[
        clipped.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if clipped.empty:
        continue

    # Calculate area in km²
    clipped["area_km2"] = clipped.geometry.area / 1_000_000

    clipped = clipped[clipped["area_km2"] > 0]

    if clipped.empty:
        continue

    # Sum area by lithology class
    summary = (
        clipped.groupby("dom_class", dropna=True)["area_km2"]
        .sum()
        .reset_index()
    )

    total_area = summary["area_km2"].sum()

    if total_area <= 0:
        continue

    # Percentage of mapped lithology within this basin
    summary["percentage"] = (
        summary["area_km2"] / total_area * 100
    )

    summary["Basin_ID"] = basin_id
    records.append(summary)

# ======================================================
# 5. COMBINE RESULTS
# ======================================================

if not records:
    raise ValueError(
        "No lithology intersections were produced. "
        "Check the input files, CRS, and geometries."
    )

df = pd.concat(records, ignore_index=True)

# ======================================================
# 6. PRIMARY AND SECONDARY LITHOLOGY
# ======================================================

df_sorted = df.sort_values(
    ["Basin_ID", "percentage"],
    ascending=[True, False]
)

dominant = (
    df_sorted.groupby("Basin_ID")
    .agg(
        prim_class=("dom_class", "first"),
        prim_class_perc=("percentage", "first"),
        sec_class=(
            "dom_class",
            lambda x: x.iloc[1] if len(x) > 1 else None
        ),
        sec_class_perc=(
            "percentage",
            lambda x: x.iloc[1] if len(x) > 1 else None
        )
    )
    .reset_index()
)

# ======================================================
# 7. LITHOLOGY PERCENTAGES IN WIDE FORMAT
# ======================================================

fractions = df.pivot_table(
    index="Basin_ID",
    columns="dom_class",
    values="percentage",
    aggfunc="sum",
    fill_value=0
)

fractions.columns = [
    f"{col}_perc" for col in fractions.columns
]

fractions = fractions.reset_index()

# ======================================================
# 8. FINAL SUMMARY — INCLUDE ALL BASINS
# ======================================================

basin_summary = dominant.merge(
    fractions,
    on="Basin_ID",
    how="left"
)

all_basins = basins[["Basin_ID"]].drop_duplicates()

basin_summary = all_basins.merge(
    basin_summary,
    on="Basin_ID",
    how="left"
)


# Ensure all expected lithology percentage columns exist

expected_perc_cols = [
    "ev_perc",
    "mt_perc",
    "nd_perc",
    "pa_perc",
    "pb_perc",
    "pi_perc",
    "sc_perc",
    "sm_perc",
    "ss_perc",
    "su_perc",
    "va_perc",
    "vb_perc",
    "vi_perc",
    "wb_perc",
    "py_perc"
]

for col in expected_perc_cols:
    if col not in basin_summary.columns:
        basin_summary[col] = 0
    else:
        basin_summary[col] = basin_summary[col].fillna(0)

# ======================================================
# 9. SAVE OUTPUTS
# ======================================================

summary_path = out_dir / "basin_lith_summary.csv"
areas_path = out_dir / "CAMELS_Afr_Geology_Static_L1_GLiM.csv"

basin_summary.to_csv(summary_path, index=False)
df.to_csv(areas_path, index=False)

print("\nProcessing completed successfully.")
print("Total input basins:", len(all_basins))
print("Basins with lithology results:", df["Basin_ID"].nunique())
print("Summary CSV:", summary_path)
print("Lithology area CSV:", areas_path)
 
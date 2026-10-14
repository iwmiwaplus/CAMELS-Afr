import os
import geopandas as gpd
import pandas as pd

# 1. File Paths
basin_shp_path = r".../CAMELS_Afr_L1_Basin_Boundaries.shp"
hydrogeology_root_folder = ".../Hydrogeology/"
output_csv = ".../CAMELS_Afr_Geology_Static_L1_Hydrogeology.csv"

# 2. Load Basins
print("Loading basins shapefile...")
basins_gdf = gpd.read_file(basin_shp_path)

if 'Basin_ID' not in basins_gdf.columns:
    raise ValueError("ERROR: Column 'Basin_ID' not found in basins shapefile.")

# Ensure geometries are valid
basins_gdf['geometry'] = basins_gdf.geometry.buffer(0)
basin_crs = basins_gdf.crs

# 3. Find and Process Country Hydrogeology Shapefiles
print("Searching for hydrogeology shapefiles...")
possible_hg_endings = ("HGComb", "HGCo", "HCComb", "HGCom")
country_layers = []

for root, dirs, files in os.walk(hydrogeology_root_folder):
    for file in files:
        if file.endswith("_HG.shp"):
            shp_path = os.path.join(root, file)
            country_name = os.path.basename(root)
            
            try:
                # Load country shapefile
                temp_gdf = gpd.read_file(shp_path)
                
                # Align CRS
                if temp_gdf.crs != basin_crs:
                    temp_gdf = temp_gdf.to_crs(basin_crs)
                
                temp_gdf['geometry'] = temp_gdf.geometry.buffer(0)
                
                # Identify columns
                cols = temp_gdf.columns
                glg_col = [c for c in cols if c.endswith("GLG")]
                hg_col = [c for c in cols if c.endswith(possible_hg_endings)]
                
                if glg_col and hg_col:
                    # Rename to standard names and keep only necessary columns
                    temp_gdf = temp_gdf[[glg_col[0], hg_col[0], 'geometry']]
                    temp_gdf.columns = ['GLG', 'HGComb', 'geometry']
                    country_layers.append(temp_gdf)
                    print(f"  Successfully loaded: {country_name}")
                else:
                    print(f"  WARNING: {country_name} missing GLG or HGComb columns.")
                    
            except Exception as e:
                print(f"  ERROR processing {country_name}: {e}")

# Combine all hydrogeology layers into one GeoDataFrame
print("Combining hydrogeology layers...")
all_hydro_gdf = pd.concat(country_layers, ignore_index=True)

# 4. Spatial Intersection
print("Intersecting basins with hydrogeology data (this may take a few minutes)...")
# overlay 'intersection' is the equivalent of st_intersection
intersections = gpd.overlay(basins_gdf[['Basin_ID', 'geometry']], 
                            all_hydro_gdf, 
                            how='intersection')

# Calculate area for each fragment
intersections['area'] = intersections.geometry.area

# 5. Determine Dominant Value Per Basin
print("Determining dominant attributes per basin...")

def get_dominant(df, id_col, attr_col):
    # Sum area by Basin and Attribute
    grouped = df.groupby([id_col, attr_col])['area'].sum().reset_index()
    # Sort and pick the one with the maximum area for each Basin
    dominant = grouped.sort_values([id_col, 'area'], ascending=[True, False])
    dominant = dominant.drop_duplicates(subset=[id_col])
    return dominant[[id_col, attr_col]]

# Calculate dominance for GLG
dom_glg = get_dominant(intersections, 'Basin_ID', 'GLG')
dom_glg.columns = ['Basin_ID', 'AlgGLG']

# Calculate dominance for HGComb
dom_hg = get_dominant(intersections, 'Basin_ID', 'HGComb')
dom_hg.columns = ['Basin_ID', 'AlgHGComb']

# 6. Merge results back to original basin list
final_results = pd.DataFrame({'Basin_ID': basins_gdf['Basin_ID']})
final_results = final_results.merge(dom_glg, on='Basin_ID', how='left')
final_results = final_results.merge(dom_hg, on='Basin_ID', how='left')

# Rename Basin_ID to Basin_ID for output consistency if desired
final_results = final_results.rename(columns={'Basin_ID': 'Basin_ID'})

# 7. Save to CSV
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
final_results.to_csv(output_csv, index=False)

print(f"Processing complete! File saved to: {output_csv}")
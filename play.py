import numpy as np
import geopandas as gpd

gdf = gpd.read_file('data/北京市全部_Point.shp')
print(gdf)
# gdf=gdf.to_crs(3857)
# # print(gdf.crs)
# print(gdf.geometry.area)

# gdf = gpd.read_file('data_use/geometry/六环内城中村.shp')
# print(gdf)
# gdf=gdf.to_crs(3857)
# # gdf = gpd.read_file('data_use/geometry/urban_villages_filtered.shp')
# print(gdf.loc[244].geometry.area)
# print(gdf.loc[243].geometry.area)
# print(gdf.loc[245].geometry.area)
# print(gdf.loc[242].geometry.area)
# print(gdf.loc[241].geometry.area)
# print(gdf.crs)

# import pandas as pd
# import re

# # Raw data as a string


# # Extract Iteration and Reward values

# def extract_iteration_reward(file_path):
#     with open(file_path, 'r') as file:
#         raw_data = file.read()
    
#     # Extract Iteration and Reward values
#     data = re.findall(r"Iteration (\d+)/\d+ - Average Reward: ([\d\.]+)", raw_data)
    
#     # Create DataFrame
#     df = pd.DataFrame(data, columns=["Iteration", "Reward"]).astype({"Iteration": int, "Reward": float})
#     df.set_index("Iteration", inplace=True)
    
#     return df

# df = extract_iteration_reward("logging/normal.out")
# df.to_csv('dynamic.csv')
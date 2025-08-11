import geopandas as gpd

gdf = gpd.read_file('urban_villages_filtered.shp')
print(gdf)
print(gdf.geometry.area)
# print(gdf['area'])
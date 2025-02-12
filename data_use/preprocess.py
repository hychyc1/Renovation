import numpy as np
import geopandas as gpd

geo = gpd.read_file('geometry/urban_villages.shp')
print(geo['assign_row'][10])
geo = geo[geo['assign_row'].notnull()]
geo['area'] = geo.geometry.area
geo=geo.drop(columns=['geometry'])
# print(geo)
geo.to_csv('village_with_col.csv', index=False)
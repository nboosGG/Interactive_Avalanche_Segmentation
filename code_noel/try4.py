

import geopandas as gpd

gdf = gpd.read_file("/home/boosnoel/temp1.shp")

print(gdf.geometry)
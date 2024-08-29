import numpy as np
import rasterio
import geopandas as gpd

from rasterio import Affine


mapp = rasterio.open("/home/boosnoel/Documents/eagle_project/GD_map_atm_20190810_20190822.tif")

profile = mapp.profile
profile.update(count=3)

print(profile)

data = mapp.read()
with rasterio.open("/home/boosnoel/Documents/eagle_project/GD_map_atm_20190810_20190822_RGBd.tif", 'w+', **profile) as dst:
    data_rgbd = np.vstack((data, data, data))
    print("data shape: ", np.shape(data_rgbd))
    #dst.write(data_rgbd)
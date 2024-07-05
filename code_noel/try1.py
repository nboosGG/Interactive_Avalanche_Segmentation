import os
import gc
import numpy as np

from scipy.ndimage import gaussian_filter


import rasterio
import rasterio.mask
from rasterio.enums import Resampling
from rasterio.profiles import DefaultGTiffProfile


import matplotlib.pyplot as pl
import shapefile

import cv2

def show_matrix(matrix, verbose: bool, title: str):
    """visualizes a 2d matrix/array if verbose evaluates to True"""
    if (verbose):
        pl.imshow(matrix, cmap='hot')
        pl.title(title)
        pl.show()


def calc_resolution(bounds, pixel_per_meter):
    """calculate the number of pixels per dimensions"""
    dist_x = bounds[2] - bounds[0]
    dist_y = bounds[3] - bounds[1]
    assert(dist_x > 0 and dist_y > 0 and "invalid bounds, critical error")

    pixel_x_direc = dist_x * pixel_per_meter
    pixel_y_direc = dist_y * pixel_per_meter
    return tuple(np.array([pixel_y_direc, pixel_x_direc]).astype(int))

def get_bounds_from_shp(shp):
    """get maximal bounds from an shp file 
    by iterating over all polygons"""
    #iterate over all points of the polygon and get the left, right, top and bottom bounds
    top = None
    bottom = None
    right = None
    left = None

    if len(shp.points) == 0:
        return False, None


    for point in shp.points:
        if top is None:
            top = point[1]
        if bottom is None:
            bottom = point[1]
        if right is None:
            right = point[0]
        if left is None:
            left = point[0]

        top = np.maximum(top, point[1])
        bottom = np.minimum(bottom, point[1])
        right = np.maximum(right, point[0])
        left = np.minimum(left, point[0])
    
    bounds = np.array([np.floor(left), np.floor(bottom), np.ceil(right), np.ceil(top)])
    return True, bounds.astype(int)


def rescale(data):
    return data *(255/43000)


path = "/media/boosnoel/LaCie/noel/DS_v3_Sammlung/Ultracam_Davos_20219/"

ortho_map = rasterio.open(path + 'Orthofoto_ultra19_high_noprec_withIMU_withangles_optimize2GCPadapted_165.tif')
src_polys = shapefile.Reader(path + '20190316_UltracamAvalanches_extended.shp')
max_val = 0
min_val = 60000
sample_counter = 0
for iPoly in range(len(src_polys)):
    print("sampe nr: ", sample_counter)

    cur_poly = src_polys.shape(iPoly)
    suceeded, shp_bounds = get_bounds_from_shp(cur_poly)
    print("shp bounds: ", shp_bounds)

    if not suceeded:
        continue

    ortho_data_read_window = rasterio.windows.from_bounds(*shp_bounds, ortho_map.transform)
    shape = calc_resolution(shp_bounds, 2)

    ortho_data = ortho_map.read(out_shape=(ortho_map.count, shape[0], shape[1]), window=ortho_data_read_window)
    ortho_data = ortho_data[:3,:,:]
    ortho_data = rescale(ortho_data)
    #show_matrix(ortho_data[0,:,:], 1, "ortho")
    print("stats: ", np.amin(ortho_data), np.amax(ortho_data), np.mean(ortho_data), np.std(ortho_data))
    max_val = np.maximum(max_val, np.amax(ortho_data))
    min_val = np.minimum(min_val, np.amin(ortho_data))


print("max val: ", max_val)
print("min val: ", min_val)




a = 5
import os
import gc
import numpy as np


import rasterio
import rasterio.mask
from rasterio.enums import Resampling
from rasterio.profiles import DefaultGTiffProfile


import matplotlib.pyplot as pl
import shapefile

def show_matrix(matrix, verbose: bool, title: str):
    """visualizes a 2d matrix/array if verbose evaluates to True"""
    if (verbose):
        pl.imshow(matrix, cmap='hot')
        pl.title(title)
        pl.show()

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

def extend_bounds(bounds, extend, points_per_meter):
    """extend the bounds in all direction by [extend] with a minimum shape of (600,600)"""

    dist_x = bounds[2] - bounds[0]
    dist_y = bounds[3] - bounds[1]

    bound_x = extend * dist_x / 100
    bound_y = extend * dist_y / 100

    extended_bounds = np.array([bounds[0] - bound_x,
                     bounds[1] - bound_y, 
                     bounds[2] + bound_x, 
                     bounds[3] + bound_y])
    
    #increase image to at least (600,600)
    x_to_extend = 600 - (extended_bounds[2]-extended_bounds[0]) * points_per_meter
    y_to_extend = 600 - (extended_bounds[3]-extended_bounds[1]) * points_per_meter

    x_to_extend /= points_per_meter
    y_to_extend /= points_per_meter
    
    #if to_extend uneven, some more code to get exactly 600 in each dimension
    if x_to_extend > 0:
        left_to_extend = x_to_extend // 2
        right_to_extend = x_to_extend - left_to_extend
        extended_bounds[0] -= left_to_extend
        extended_bounds[2] += right_to_extend
    
    if y_to_extend > 0:
        top_to_extend = y_to_extend // 2
        bottom_to_extend = y_to_extend - top_to_extend
        extended_bounds[1] -= bottom_to_extend
        extended_bounds[3] += top_to_extend 

    return extended_bounds

def check_map_bounds(cur_bounds, map_bounds):
    """should be used, but somehow not needed yet"""
    a=5

def calc_resolution(bounds, pixel_per_meter):
    """calculate the number of pixels per dimensions"""
    dist_x = bounds[2] - bounds[0]
    dist_y = bounds[3] - bounds[1]
    assert(dist_x > 0 and dist_y > 0 and "invalid bounds, critical error")

    pixel_x_direc = dist_x * pixel_per_meter
    pixel_y_direc = dist_y * pixel_per_meter
    return tuple(np.array([pixel_y_direc, pixel_x_direc]).astype(int))

def create_directiories(folder_list):
    """check if directories exist, if not create them
    input: a list of folder paths"""
    for path in folder_list:
        if not os.path.exists(path):
            # Create the directory
            os.makedirs(path)
            print("created: ", path)



def main():
    data_path = "/home/boosnoel/Documents/data/DS_v2_sammlung/"
    target_path = "/home/boosnoel/Documents/data/temp/"

    path_storage_dsm = target_path + "dsm/"
    path_storage_ortho = target_path + "images/"
    path_storage_mask = target_path + "masks/"

    create_directiories([path_storage_dsm, path_storage_ortho, path_storage_mask])

    sample_counter = 0

    verbose_show_data = False

    datapoints_per_meter = 1

    extend = 5 #in % of the image size that gets padded in each direction

    for folder in os.listdir(data_path):
        #print("name: ", folder)
        ortho_map = None
        dsm_map = None
        src_polys = None
        for filename in os.listdir(data_path + folder + "/"):
            #print("filename: ", filename)

            if filename.startswith("Ortho"):
                ortho_map = rasterio.open(data_path + folder + "/" + filename)

            if filename.startswith("DSM"):
                dsm_map = rasterio.open(data_path + folder + "/" + filename)

            if filename.endswith(".shp"):
                src_polys = shapefile.Reader(data_path + folder + "/" + filename)


        if ortho_map is None or dsm_map is None or src_polys is None:
            print("could not find all 3 needed files (orho, dsm, shp). skip this one: ", folder)
            ortho_map = None
            dsm_map = None
            src_polys = None
            gc.collect()
            continue

        for iPoly in range(len(src_polys)):
            print("sampe nr: ", sample_counter)

            cur_poly = src_polys.shape(iPoly)
            suceeded, shp_bounds = get_bounds_from_shp(cur_poly)

            if not suceeded:
                assert(False)
                continue

            continue

            
            bounds_extended = extend_bounds(shp_bounds, extend, datapoints_per_meter)

            top_left_coordinate = np.array([bounds_extended[0], bounds_extended[3]])

            shape = calc_resolution(bounds_extended, datapoints_per_meter)
            height, width = shape

            ortho_data_read_window = rasterio.windows.from_bounds(*bounds_extended, ortho_map.transform)
            ortho_data = ortho_map.read(out_shape=(ortho_map.count, shape[0], shape[1]), window=ortho_data_read_window)
            
            
            dsm_data_read_window = rasterio.windows.from_bounds(*bounds_extended, dsm_map.transform)
            dsm_data = dsm_map.read(1, out_shape=shape, window=dsm_data_read_window)

            show_matrix(ortho_data[2,:,:], verbose_show_data, "ortho data, blue channel")
            show_matrix(dsm_data, verbose_show_data, "dsm data")

            print("shape: ", np.shape(ortho_data), np.shape(dsm_data))

            #now write the 2 maps. 
            #then use the shapefile to make a polygonmask with the same dimesnion as the maps

            resolution = 1. / datapoints_per_meter

            transform = ortho_map.transform
            transform_adjusted = rasterio.Affine(resolution, transform[1], top_left_coordinate[0],
                                        transform[3], -1 * resolution, top_left_coordinate[1])
            

            profile_ortho = ortho_map.profile.copy()
            profile_ortho.update(transform=transform_adjusted)
            
            profile_ortho.update(width=width, height=height)

            map_sample_ortho = rasterio.open(path_storage_ortho + str(sample_counter) + ".tif", 'w+', **profile_ortho)
            map_sample_ortho.write(ortho_data.astype(np.uint8))
            


            profile_dsm = dsm_map.profile.copy()
            profile_dsm.update(transform = transform_adjusted)
            profile_dsm.update(width=width, height=height)

            map_sample_dsm = rasterio.open(path_storage_dsm + str(sample_counter) + ".tif", 'w+', **profile_dsm)

            map_sample_dsm.write_band(1, dsm_data.astype(np.float32))


            polygon_mask, _, _ = rasterio.mask.raster_geometry_mask(map_sample_ortho, [cur_poly], invert=True)
            polygon_mask = np.where(polygon_mask == 1, 1, 0)


            show_matrix(polygon_mask, verbose_show_data, "avalanche mask")

            #store mask
            profile_mask = dsm_map.profile.copy()
            profile_mask.update(transform=transform_adjusted)
            profile_mask.update(dtype="uint8", nodata=0, width=width, height=height)
            #print("profile: ", profile_mask)
            map_sample_mask = rasterio.open(path_storage_mask + str(sample_counter) + ".tif", 'w+', **profile_mask)

            map_sample_mask.write_band(1, polygon_mask.astype(np.uint8))

            sample_counter += 1
            gc.collect()


if __name__ == '__main__':
    main()
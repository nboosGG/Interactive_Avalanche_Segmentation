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
    if (verbose):
        pl.imshow(matrix, cmap='hot')
        pl.title(title)
        pl.show()

def get_bounds_from_shp(shp):
    #iterate over all points of the polygon and get the left, right, top and bottom bounds
    top = None
    bottom = None
    right = None
    left = None

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
    return bounds.astype(int)

def extend_bounds(bounds, extend):
    """extend the bounds in all direction by [extend]"""
    return np.array([bounds[0] - extend,
                     bounds[1] - extend, 
                     bounds[2] + extend, 
                     bounds[3] + extend])

def calc_resolution(bounds, pixel_per_meter):
    """calculate the number of pixels per dimensions"""
    dist_x = bounds[2] - bounds[0]
    dist_y = bounds[3] - bounds[1]
    assert(dist_x > 0 and dist_y > 0 and "invalid bounds, critical error")

    pixel_x_direc = dist_x * pixel_per_meter
    pixel_y_direc = dist_y * pixel_per_meter
    return tuple(np.array([pixel_y_direc, pixel_x_direc]).astype(int))




def main():
    path = "/home/boosnoel/Documents/data/small_dataset/"

    path_storage_dsm = path + "sample_dsm/"
    path_storage_ortho = path + "sample_ortho/"
    path_storage_mask = path + "groundtruth/"

    sample_counter = 0

    verbose_show_data = False

    #print(DefaultGTiffProfile(count=3))
    #assert(False)

    for folder in os.listdir(path):
        print("name: ", folder)
        ortho_map = None
        dsm_map = None
        src_polys = None
        for filename in os.listdir(path + folder + "/"):
            print("filename: ", filename)

            if filename.startswith("Ortho"):
                ortho_map = rasterio.open(path + folder + "/" + filename)
                print("ortho map: ", ortho_map.name)

            if filename.startswith("DSM"):
                dsm_map = rasterio.open(path + folder + "/" + filename)
                print("dsm map: ", dsm_map.name)

            if filename.endswith(".shp"):
                src_polys = shapefile.Reader(path + folder + "/" + filename)

                print("subname: ", filename, "#polys: ", len(src_polys))

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
            shp_bounds = get_bounds_from_shp(cur_poly)

            extend = 100 #in meters
            bounds_extended = extend_bounds(shp_bounds, extend)

            top_left_coordinate = np.array([bounds_extended[0], bounds_extended[3]])

            shape = calc_resolution(bounds_extended, 1)
            height, width = shape

            ortho_data_read_window = rasterio.windows.from_bounds(*bounds_extended, ortho_map.transform)
            ortho_data = ortho_map.read(out_shape=(ortho_map.count, shape[0], shape[1]), window=ortho_data_read_window)
            
            
            dsm_data_read_window = rasterio.windows.from_bounds(*bounds_extended, dsm_map.transform)
            dsm_data = dsm_map.read(1, out_shape=shape, window=dsm_data_read_window)

            show_matrix(ortho_data[2,:,:], verbose_show_data, "ortho data, blue channel")
            show_matrix(dsm_data, verbose_show_data, "dsm data")

            #now write the 2 maps. 
            #then use the shapefile to make a polygonmask with the same dimesnion as the maps

            transform = ortho_map.transform
            transform_adjusted = rasterio.Affine(1, transform[1], top_left_coordinate[0],
                                        transform[3], -1, top_left_coordinate[1])
            

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


            show_matrix(polygon_mask, verbose_show_data, "avalange mask")

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
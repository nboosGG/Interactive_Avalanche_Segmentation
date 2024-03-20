import os
import gc
import numpy as np


import rasterio
import rasterio.mask
from rasterio.enums import Resampling


import matplotlib.pyplot as pl
import shapefile

def show_matrix(matrix, verbose: bool, title: str):
    if (verbose):
        pl.imshow(matrix, cmap='hot')
        pl.title(title)
        pl.show()



def reshape_arr(arr):
    return arr[0,:,:]

def window_data_from_arr(arr, window):
    window = np.array(window.flatten())
    return arr[window[1]:window[1]+window[3], window[0]:window[0]+window[2]]


def write_arr(map, arr, band, dtype):
    windows = [window for rij, window in map.block_windows()]
    windows_arr = np.array(windows)
    #print("windows: ", windows_arr)

    for window in windows:
        window_data = window_data_from_arr(arr, window)
        map.write_band(band, window_data.astype(dtype), window=window)

def window_data_from_arr2(arr, window):
    window = np.array(window.flatten())
    return arr[:, window[1]:window[1]+window[3], window[0]:window[0]+window[2]]

def write_arr2(map, arr, dtype):
    windows = [window for rij, window in map.block_windows()]
    windows_arr = np.array(windows)
    #print("windows: ", windows_arr)

    for window in windows:
        window_data = window_data_from_arr2(arr, window)
        map.write(window_data.astype(dtype), window=window)






def get_profiles(example_map):
    profile = example_map.profile
    profile_dsm = profile.copy()
    profile_dsm.update(dtype="float32", count=1, nodatavals=np.array([0]))

    profile_ortho = profile.copy()
    profile_ortho.update(dtype="uint8", count = 4)

    profile_mask = profile.copy()
    profile_mask.update(count = 1, nodatavals=np.array([0]))

    return profile_ortho, profile_dsm, profile_mask

def main():
    path = "/home/boosnoel/Documents/data/small_dataset/"

    path_storage_dsm = path + "sample_dsm/"
    path_storage_ortho = path + "sample_ortho/"
    path_storage_mask = path + "groundtruth/"

    sample_counter = 0

    for folder in os.listdir(path):
        print("name: ", folder)
        ortho_map = None
        dsm_map = None
        src_polys = None
        for filename in os.listdir(path + folder + "/"):
            

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

            if sample_counter < 3:
                sample_counter += 1
                continue


            cur_poly = src_polys.shape(iPoly)
            
            
            polygon_mask, polygon_mask_transfrom = rasterio.mask.mask(ortho_map, [cur_poly], crop=False, all_touched = False, pad=False)
            polygon_mask = reshape_arr(polygon_mask)
            show_matrix(polygon_mask, 0, "polygon mask")
            polygon_mask = np.where(polygon_mask > 0.0001, 1, 0) #build the true boolean mask
            ortho_data = ortho_map.read(out_shape=(ortho_map.count, np.shape(polygon_mask)[0], np.shape(polygon_mask)[1]), resampling=Resampling.bilinear) 
            dsm_data = dsm_map.read(1,out_shape=(np.shape(polygon_mask))) #resampling=Resampling.bilinear

            #assert("all shapes equal: ", np.shape(polygon_mask) == np.shape(ortho_data) and np.shape(ortho_data) == np.shape(dsm_data))
            print("maps shape: ", np.shape(ortho_data), np.shape(dsm_data), np.shape(polygon_mask))
            
            #print masks
            #print("bla1")
            profile_ortho, profile_dsm, profile_mask = get_profiles(ortho_map)

            print("bla2")
            map_sample_ortho = rasterio.open(path_storage_ortho + str(sample_counter) + ".tif", 'w+', **profile_ortho)
            map_sample_dsm = rasterio.open(path_storage_dsm + str(sample_counter) + ".tif", "w+", **profile_dsm)
            map_sample_mask = rasterio.open(path_storage_mask + str(sample_counter) + ".tif", "w+", **profile_mask)
            #print("bla3")
            #print("shapes: ", np.shape(map_sample_dsm), np.shape(map_sample_mask), np.shape(map_sample_ortho))

            write_arr(map_sample_dsm, dsm_data, 1, np.float32)
            del dsm_data
            print("bla4")
            write_arr(map_sample_mask, polygon_mask, 1, np.uint8)
            del polygon_mask

            gc.collect()
            print("bla5")


            #write_arr(map_sample_ortho, ortho_data[0,:,:], 1, np.uint8)
            #write_arr(map_sample_ortho, ortho_data[1,:,:], 2, np.uint8)
            #write_arr(map_sample_ortho, ortho_data[2,:,:], 3, np.uint8)
            #write_arr(map_sample_ortho, ortho_data[3,:,:], 4, np.uint8)
            write_arr2(map_sample_ortho, ortho_data, np.uint8)

            del ortho_data
            gc.collect()


            
            sample_counter += 1
            


            






if __name__ == '__main__':
    main()





print("finished successfully")
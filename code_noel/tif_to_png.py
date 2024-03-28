import os
import gc
import numpy as np

import cv2
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



def main():

    path = "/home/boosnoel/Documents/InteractiveAvalancheSegmentation/datasets/small_dataset/"

    path_storage_dsm = path + "dsm/"
    path_storage_ortho = path + "ortho/"
    path_storage_mask = path + "mask/"

    for folder in os.listdir(path):
        print("name: ", folder)
        ortho_map = None
        dsm_map = None
        src_polys = None
        for filename in os.listdir(path + folder + "/"):
            if filename.endswith(".tif"):
                mapp = rasterio.open(path + folder + "/" + filename)
                data = mapp.read()

                #show_matrix(data, 1, "data")
                #print("image shape: ", np.shape(data))
                if np.shape(data)[0] == 4:
                    #show_matrix(data[0,:,:], 1, "channel 0")
                    #show_matrix(data[1,:,:], 1, "channel 1")
                    #show_matrix(data[2,:,:], 1, "channel 2")
                    #show_matrix(data[3,:,:], 1, "channel 3")
                    data[0,:,:] = np.where(data[3,:,:] > 0, data[0,:,:], 0)
                    data[1,:,:] = np.where(data[3,:,:] > 0, data[1,:,:], 0)
                    data[2,:,:] = np.where(data[3,:,:] > 0, data[2,:,:], 0)

                    data = data[:3,:,:]
                
                assert(np.shape(data)[0] < 4)
                print("image shape: ", np.shape(data))

                cv2.imwrite(path + folder + "/" + filename[:-4] + ".png", np.moveaxis(data, 0, -1))






if __name__ == '__main__':
    main()

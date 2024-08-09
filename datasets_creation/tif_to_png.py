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

    path = "/home/boosnoel/Documents/InteractiveAvalancheSegmentation/datasets/ds_v3_0p5m_NoBlur_3/"

    for folder in os.listdir(path):
        print("name: ", folder)

        for filename in os.listdir(path + folder + "/"):
            if filename.endswith(".tif"):
                mapp = rasterio.open(path + folder + "/" + filename)
                data = mapp.read()
                if np.shape(data)[0] == 4:
                    data[0,:,:] = np.where(data[3,:,:] > 0, data[0,:,:], 0)
                    data[1,:,:] = np.where(data[3,:,:] > 0, data[1,:,:], 0)
                    data[2,:,:] = np.where(data[3,:,:] > 0, data[2,:,:], 0)

                    data = data[:3,:,:]
                
                assert(np.shape(data)[0] < 4)
                print("image stats: ", np.shape(data), np.amin(data), np.amax(data))
                cv2.imwrite(path + folder + "/" + filename[:-4] + ".png", np.moveaxis(data, 0, -1))

                






if __name__ == '__main__':
    main()

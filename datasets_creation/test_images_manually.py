import os
import gc
import numpy as np

from scipy.ndimage import gaussian_filter, median_filter

from scipy import fftpack


import rasterio
import rasterio.mask
#from rasterio.enums import Resampling
#from rasterio.profiles import DefaultGTiffProfile


import matplotlib.pyplot as pl
import shapefile

import cv2

from PIL import Image, ImageDraw

def show_matrix(matrix, verbose: bool, title: str):
    """visualizes a 2d matrix/array if verbose evaluates to True"""
    if (verbose):
        pl.imshow(matrix, cmap='hot')
        pl.title(title)
        pl.show()



def main():
    path = "/home/boosnoel/Documents/InteractiveAvalancheSegmentation/datasets/"
    folder_name = "ds_v3_0p5m_RGB_0to1/images/"

    image_name = "218.png"

    image = Image.open(path+folder_name+image_name)
    image = np.asarray(image)
    print("image stats: ", np.shape(image), np.amax(image), np.amin(image))

    show_matrix(image[:,:,0], 1, "image")


if __name__ == '__main__':
    main()
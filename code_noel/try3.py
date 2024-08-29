
import numpy as np

from PIL import Image
import cv2
import rasterio

import matplotlib.pyplot as pl

def show_matrix(matrix, verbose: bool, title: str):
    if (verbose):
        pl.imshow(matrix, cmap='hot')
        pl.title(title)
        pl.show()

ortho_map = rasterio.open("/home/boosnoel/Documents/clay/examples/201113_RGB_new.tif")
profile = ortho_map.profile
print("ohrto profile: ", ortho_map.profile)

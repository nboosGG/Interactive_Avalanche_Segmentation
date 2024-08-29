import os
import gc
import numpy as np

from scipy.ndimage import gaussian_filter


import rasterio
import rasterio.mask
from rasterio.enums import Resampling
from rasterio.profiles import DefaultGTiffProfile


import matplotlib.pyplot as plt
import shapefile

import cv2


def show_matrix(image, title: str):

    plt.imshow(image, cmap='hot')
    plt.title(title)
    plt.show()



def main():
    file_path = "/home/boosnoel/Documents/data/temp/dsm/14.tif"

    path_storage_ortho = file_path


    sample_counter = 145

    verbose_show_data = False
    is_16bit_uint = False

    datapoints_per_meter = 5

    extend = 5 #in % of the image size that gets padded in each direction

    ortho_map = rasterio.open(path_storage_ortho)
    """data = ortho_map.read()
    print(ortho_map.dtypes)
    if ortho_map.dtypes[0] == 'uint16':
        data = (data.astype(np.float64) / 65535 * 255).astype(np.uint8)
    print("ortho shape: ", np.shape(data))
    print("stats: ", np.amin(data), np.amax(data))"""

    profile = ortho_map.profile
    print("profile: ", profile)
    profile.update(count=3, dtype = 'uint8', nodata=0)
    print("profile2: ", profile)

    data = ortho_map.read()
    print("data shape: ", np.shape(data))
    



if __name__ == '__main__':
    main()








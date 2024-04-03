import numpy as np

import cv2


path = "/home/boosnoel/Documents/data/small_dataset/dsm/2.tif"

img = cv2.imread(path)
print(type(img))
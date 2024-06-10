import torch
import torch.nn as nn
import numpy as np

from PIL import Image
import cv2


import matplotlib.pyplot as pl

def show_matrix(matrix, verbose: bool, title: str):
    if (verbose):
        pl.imshow(matrix, cmap='hot')
        pl.title(title)
        pl.show()

storage_path = "/home/boosnoel/Downloads/temp2/"
image_easy = "45.png"
image_hard = "46.png"

mask = np.array(Image.open(storage_path + image_easy))

print("mask stuff: ", type(mask), np.amin(mask), np.amax(mask))

verbose = 1


contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("hierarchy: ", hierarchy)

print("contours: ", contours)




show_matrix(mask, verbose, "first mask")


import os
import shutil
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from matplotlib import image
from pycocotools.coco import COCO

# Dataset
dataset_in = 'Lawine_3_instance'
dataset_out = 'Lawine_3_instance_resized'

# Input data
dataset_in_path = f"/home/oberson/slfhome/RITM/ritm_interactive_segmentation/datasets/{dataset_in}"
images_in_path = Path(f"{dataset_in_path}/images")

images_in = [x.name for x in images_in_path.glob('*.*')]


# Output folder
dataset_out_path = f"/home/oberson/slfhome/RITM/ritm_interactive_segmentation/datasets/{dataset_out}"
images_out_path = f"{dataset_out_path}/images"
masks_out_path = f"{dataset_out_path}/masks"
Path(images_out_path).mkdir(parents=True, exist_ok=True)
Path(masks_out_path).mkdir(parents=True, exist_ok=True)

for idx, image_in_path in enumerate(images_in):
    image_path = str(images_in_path / image_in_path)
    mask_path = image_path.replace("images", "masks").replace("jpg", "png")

    image_in = plt.imread(image_path)
    mask_in = plt.imread(mask_path)

    image_out = cv2.resize(image_in, dsize=(600,300))
    mask_out = cv2.resize(mask_in, (600,300), interpolation =cv2.INTER_NEAREST)

    image.imsave(image_path.replace(dataset_in, dataset_out), image_out)
    image.imsave(mask_path.replace(dataset_in, dataset_out), mask_out, cmap="gray")

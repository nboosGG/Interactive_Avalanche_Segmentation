import os
import shutil
from pathlib import Path
from matplotlib import image
from pycocotools.coco import COCO

# Dataset
dataset = 'Avalanche_1'

# Input data
imgage_original_dir = f"/data/annotated_avalanches/{dataset}/images_original" #Rohdaten aus dem Scalabel
ann_original_file = f"/data/annotated_avalanches/{dataset}/annotations_original/instances.json"

# Output folder
image_dir = f"/data/annotated_avalanches/{dataset}/{dataset}_all/images"
mask_dir = f"/data/annotated_avalanches/{dataset}/{dataset}_all/masks"
Path(image_dir).mkdir(parents=True, exist_ok=True)
Path(mask_dir).mkdir(parents=True, exist_ok=True)

# Load annotations
coco = COCO(ann_original_file)

# Get categories and image IDs
catIds = coco.getCatIds()
imgIds = coco.getImgIds()

# Create mask for each image 
for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]
    img_path_original = os.path.join(imgage_original_dir, img['file_name'])

    img_path = os.path.join(image_dir, img['file_name'])
    mask_path = os.path.join(mask_dir, img['file_name'].replace('jpg', 'png').replace('JPG', 'png'))

    anns_ids = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    
    try:
        mask = coco.annToMask(anns[0])>0
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])>0
        shutil.copy(img_path_original, img_path)
        image.imsave(mask_path, mask, cmap="gray")
    except:
        print("No annotations for given image")
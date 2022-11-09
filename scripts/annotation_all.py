import os
import shutil
from pathlib import Path
from matplotlib import image
from pycocotools.coco import COCO

# Dataset
dataset = 'Lawine_3'

# Input data
img_org_dir = f"/home/oberson/slfhome/annotated_avalanches/{dataset}/images_original"
ann_file = f"/home/oberson/slfhome/annotated_avalanches/{dataset}/annotations_original/instances.json"

# Output folder
img_dir = f"/home/oberson/slfhome/annotated_avalanches/{dataset}/{dataset}_all/images"
mask_dir = f"/home/oberson/slfhome/annotated_avalanches/{dataset}/{dataset}_all/masks"
Path(img_dir).mkdir(parents=True, exist_ok=True)
Path(mask_dir).mkdir(parents=True, exist_ok=True)

# Load annotations
coco = COCO(ann_file)

# Get categories and image IDs
catIds = coco.getCatIds()
imgIds = coco.getImgIds()

# Create mask for each image 
for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]

    img_path_original = os.path.join(img_org_dir, img['file_name'])

    img_path = os.path.join(img_dir, img['file_name'])
    mask_path = os.path.join(mask_dir, img['file_name'].replace('jpg', 'png'))
    mask_path = mask_path.replace('JPG', 'png')

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
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
image_instance_dir = f"/home/oberson/slfhome/annotated_avalanches/{dataset}/{dataset}_instance/images"
mask_instance_dir = f"/home/oberson/slfhome/annotated_avalanches/{dataset}/{dataset}_instance/masks"
Path(image_instance_dir).mkdir(parents=True, exist_ok=True)
Path(mask_instance_dir).mkdir(parents=True, exist_ok=True)

# Load annotations
coco = COCO(ann_original_file)

# Get categories and image IDs
catIds = coco.getCatIds()
imgIds = coco.getImgIds()

# Create mask for each image 
for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]
    img_path_original = os.path.join(imgage_original_dir, img['file_name'])
   
    anns_ids = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
    for anns_id in anns_ids:
        anns = coco.loadAnns(anns_id)

        filename = img['file_name'].split('.')
        filename = f"{filename[0]}_{anns_id}.{filename[1]}"
        
        img_path = os.path.join(image_instance_dir, filename)
        mask_path = os.path.join(mask_instance_dir, filename.replace('jpg', 'png').replace('JPG', 'png'))

        try:
            mask = coco.annToMask(anns[0])>0
            for i in range(len(anns)):
                mask += coco.annToMask(anns[i])>0

            shutil.copy(img_path_original, img_path)    
            image.imsave(mask_path, mask, cmap="gray")
        except:
            print("No annotations for given image")
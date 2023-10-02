import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, '.')
from isegm.utils.misc import get_bbox_from_mask, expand_bbox, clamp_bbox

def main():
    # Target Size, e.g. 1000x1000 pixel
    target_size = 1000
    
    # Dataset
    dataset_in = 'Avalanche_4' #DEFINE!!
    dataset_out = 'Avalanche_train'
    
    # Input data
    images_in_path = Path(f"/data/ritm_interactive_segmentation/datasets/{dataset_in}/images")
    images_in = [x.name for x in images_in_path.glob('*.*')]

    # Output folder
    dataset_out_path = f"/data/ritm_interactive_segmentation/datasets/{dataset_out}"
    images_out_path = f"{dataset_out_path}/images"
    masks_out_path = f"{dataset_out_path}/masks"
    
    Path(images_out_path).mkdir(parents=True, exist_ok=True)
    Path(masks_out_path).mkdir(parents=True, exist_ok=True)

    for image_in_path in images_in:
        print(image_in_path)
        image_path = str(images_in_path / image_in_path)
        mask_path = image_path.replace("images", "masks").replace("jpg", "png").replace("JPG", "png")

        image_in = Image.open(image_path)
        mask_in = Image.open(mask_path)

        #print("shape of mask" ,np.shape(mask_in))

        mask = plt.imread(mask_path)[:, :, 0].astype(np.int32)
        mask[mask > 0] = 1
        
        h, w = mask.shape[0], mask.shape[1]

        bbox = get_bbox_from_mask(mask)
        bbox = expand_bbox(bbox, 1.1, target_size)
        bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

        rmin, rmax, cmin, cmax = bbox
        
        height = rmax - rmin + 1
        width = cmax - cmin + 1
        
        scale = target_size / max(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))
        
        image_out = image_in.resize((new_height, new_width), Image.BILINEAR, (cmin, rmin, cmax + 1, rmax + 1))   
        mask_out = mask_in.resize((new_height, new_width), Image.BILINEAR, (cmin, rmin, cmax + 1, rmax + 1))           

        image_out.save(image_path.replace(dataset_in, dataset_out), quality=100, subsampling=0)
        mask_out.save(mask_path.replace(dataset_in, dataset_out))

if __name__ == '__main__':
    main()
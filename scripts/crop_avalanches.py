import sys
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import image
from PIL import Image
import numpy as np


sys.path.insert(0, '.')
from isegm.utils.misc import get_bbox_from_mask, expand_bbox, clamp_bbox


def main():
    # Dataset
    dataset_in = 'Lawine_3_instance'
    dataset_out = 'Lawine_3_instance_crop_PIL_test'
    
    library = 'PIL' # cv2, plt, PIL

    # Input data
    dataset_in_path = f"/home/oberson/slfhome/ritm_interactive_segmentation/datasets/{dataset_in}"
    images_in_path = Path(f"{dataset_in_path}/images")

    images_in = [x.name for x in images_in_path.glob('*.*')]

    # Output folder
    dataset_out_path = f"/home/oberson/slfhome/ritm_interactive_segmentation/datasets/{dataset_out}"
    images_out_path = f"{dataset_out_path}/images"
    masks_out_path = f"{dataset_out_path}/masks"
    
    Path(images_out_path).mkdir(parents=True, exist_ok=True)
    Path(masks_out_path).mkdir(parents=True, exist_ok=True)

    for image_in_path in images_in:
        print(image_in_path)
        image_path = str(images_in_path / image_in_path)
        mask_path = image_path.replace("images", "masks").replace("jpg", "png").replace("JPG", "png")

        if library == 'cv2':
            image_in = cv2.imread(image_path)
            mask_in = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
            mask_in[mask_in > 0] = 1
            
            h, w = mask_in.shape[0], mask_in.shape[1]

            bbox = get_bbox_from_mask(mask_in)
            bbox = expand_bbox(bbox, 1.4, 200)
            bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

            rmin, rmax, cmin, cmax = bbox
            image_out = image_in[rmin:rmax + 1, cmin:cmax + 1]
            mask_out = mask_in[rmin:rmax + 1, cmin:cmax + 1]

            cv2.imwrite(image_path.replace(dataset_in, dataset_out), image_out)
            image.imsave(mask_path.replace(dataset_in, dataset_out), mask_out, cmap="gray")
        elif library == 'plt':
            image_in = plt.imread(image_path)
            mask_in = plt.imread(mask_path)[:, :, 0].astype(np.int32)
            mask_in[mask_in > 0] = 1
            
            h, w = mask_in.shape[0], mask_in.shape[1]

            bbox = get_bbox_from_mask(mask_in)
            bbox = expand_bbox(bbox, 1.4, None)
            bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

            rmin, rmax, cmin, cmax = bbox
            image_out = image_in[rmin:rmax + 1, cmin:cmax + 1]
            mask_out = mask_in[rmin:rmax + 1, cmin:cmax + 1]

            image.imsave(image_path.replace(dataset_in, dataset_out), image_out)
            image.imsave(mask_path.replace(dataset_in, dataset_out), mask_out, cmap="gray")
        elif library == 'PIL':
            image_in = Image.open(image_path)
            mask_in = Image.open(mask_path)
                        
            mask = plt.imread(mask_path)[:, :, 0].astype(np.int32)
            mask[mask > 0] = 1
            
            h, w = mask.shape[0], mask.shape[1]

            bbox = get_bbox_from_mask(mask)
            bbox = expand_bbox(bbox, 1.1, 1000)
            bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

            rmin, rmax, cmin, cmax = bbox
            #image_out = image_in.crop((cmin, rmin, cmax + 1, rmax + 1))
            #mask_out = mask_in.crop((cmin, rmin, cmax + 1, rmax + 1))
            
            target_size = 1000
            height = rmax - rmin + 1
            width = cmax - cmin + 1
            
            scale = target_size / max(height, width)
            new_height = int(round(height * scale))
            new_width = int(round(width * scale))
            
            image_out = image_in.resize((new_height, new_width), Image.BILINEAR, (cmin, rmin, cmax + 1, rmax + 1))   
            mask_out = mask_in.resize((new_height, new_width), Image.BILINEAR, (cmin, rmin, cmax + 1, rmax + 1))           

            image_out.save(image_path.replace(dataset_in, dataset_out), quality=100, subsampling=0)
            mask_out.save(mask_path.replace(dataset_in, dataset_out))
        else: 
            return()

if __name__ == '__main__':
    main()
import cv2
import shutil
import numpy as np
from pathlib import Path
from matplotlib import image

def main():
    # Dataset
    dataset_in = 'test' #DEFINE!!
    dataset_out = 'test_out'

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
        image_path = str(images_in_path / image_in_path)
        mask_path = image_path.replace("images", "masks").replace("jpg", "png").replace("JPG", "png")

        mask_in = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        
        for idx, instance in enumerate(np.unique(mask_in)):
            if instance != 0:           
                mask_out = np.zeros_like(mask_in)
                mask_out[mask_in == instance] = 1
                
                temp = image_path.split('.')
                image_path_out = f"{temp[0]}_{idx}.{temp[1]}".replace(dataset_in, dataset_out)
                
                temp = mask_path.split('.')
                mask_path_out = f"{temp[0]}_{idx}.{temp[1]}".replace(dataset_in, dataset_out)
                        
                shutil.copy(image_path, image_path_out)
                image.imsave(mask_path_out, mask_out, cmap="gray")
            
if __name__ == '__main__':
    main()
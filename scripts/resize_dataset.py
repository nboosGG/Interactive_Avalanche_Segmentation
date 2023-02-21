import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import image

def main():
    # Dataset
    dataset_in = 'Avalanche_1'
    dataset_out = 'Avalanche_1_resized'

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

        image_in = plt.imread(image_path)
        mask_in = plt.imread(mask_path)

        image_out = cv2.resize(image_in, dsize=(600,400))
        mask_out = cv2.resize(mask_in, (600,400), interpolation=cv2.INTER_NEAREST)

        image.imsave(image_path.replace(dataset_in, dataset_out), image_out)
        image.imsave(mask_path.replace(dataset_in, dataset_out), mask_out, cmap="gray")

if __name__ == '__main__':
    main()
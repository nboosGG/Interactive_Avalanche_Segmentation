import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class AvalancheDataset(ISDataset):
    def __init__(self, dataset_path, split=None,
                 images_dir_name='images', masks_dir_name='masks', **kwargs):
        super(AvalancheDataset, self).__init__(**kwargs)
        assert split in {None, 'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._masks_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*'))]
        self.masks_samples = [x.name for x in sorted(self._masks_path.glob('*'))]
        
        if split is not None:
            X_train, X_val, y_train, y_val = train_test_split(self.dataset_samples, self.masks_samples, test_size=0.2, random_state=42)

            if split == 'train':
                self.dataset_samples = X_train
                self.masks_samples = y_train
            elif split == 'val':
                self.dataset_samples = X_val
                self.masks_samples = y_val

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        mask_name = self.masks_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_path / mask_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask > 0] = 1
        
        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)

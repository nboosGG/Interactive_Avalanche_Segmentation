import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from isegm.data.base import ISDataset
from isegm.data.sample import DSample



class DroneAvalancheDataset(ISDataset):
    def __init__(self, dataset_path, split=None,
                 ortho_dir_name='images', dsm_dir_name='dsm', masks_dir_name='masks', **kwargs):
        super(DroneAvalancheDataset, self).__init__(**kwargs)
        assert split in {None, 'train', 'val'}

        dataset_path = "datasets/small_dataset/"
        

        self.dataset_path = Path(dataset_path)
        self._ortho_path = self.dataset_path / ortho_dir_name
        self._dsm_path = self.dataset_path / dsm_dir_name
        self._masks_path = self.dataset_path / masks_dir_name

        print("paths: ", self.dataset_path, self._ortho_path, self._dsm_path, self._masks_path)

        self.dataset_ortho= [x.name for x in sorted(self._ortho_path.glob('*'))]
        self.dataset_dsm = [x.name for x in sorted(self._dsm_path.glob('*'))]
        self.masks = [x.name for x in sorted(self._masks_path.glob('*'))]



        if split is not None:
            X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(self.dataset_ortho, self.dataset_dsm, self.masks, test_size=0.2, random_state=42)
            if split == 'train':
                self.dataset_ortho = X1_train
                self.dataset_dsm = X2_train
                self.masks = y_train
            elif split == 'val':
                self.dataset_ortho = X1_val
                self.dataset_dsm = X2_val
                self.masks = y_val

        self.dataset_samples = self.dataset_ortho

    def get_sample(self, index) -> DSample:
        ortho_name = self.dataset_ortho[index]
        dsm_name = self.dataset_dsm[index]
        mask_name = self.masks[index]
        ortho_path = str(self._ortho_path / ortho_name)
        dsm_path = str(self._dsm_path / dsm_name)
        mask_path = str(self._masks_path / mask_name)
        
        #print("paths: ", ortho_path, dsm_path, mask_path)

        ortho = cv2.imread(ortho_path)
        #print("pure ortho shaep: ", np.shape(ortho))
        #ortho = cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB)
        #dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
        dsm = cv2.imread(dsm_path, cv2.IMREAD_GRAYSCALE)
        #print("asdfsadf", type(dsm))

        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)

        assert(np.shape(ortho)[0] == np.shape(dsm)[0] 
               and np.shape(ortho)[1] == np.shape(dsm)[1] 
               and np.shape(dsm) == np.shape(instances_mask) 
               and "input dataset shape missmatch (of otho, dsm, mask)")
        
        dsm = dsm[:,:,np.newaxis]

        sample = np.concatenate((ortho, dsm), axis=2)
        #sample = ortho
        #print("sample shape: ", np.shape(sample))
        #print("sample shapes: ", np.shape(ortho), np.shape(dsm), np.shape(sample))

        
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask > 0] = 1

        #print("shapes: ", np.shape(sample), np.shape(instances_mask))
        #print("ortho, dsm min max: ", np.amin(ortho), np.amax(ortho), np.amin(dsm), np.amax(dsm))
        #print("min max: ", np.amin(sample), np.amax(sample), np.amin(instances_mask), np.amax(instances_mask))
        
        return DSample(sample, instances_mask, objects_ids=[1], sample_id=index)


import torch

from typing import List
from isegm.inference.clicker import Click
from .base import BaseTransform


class CropBBox(BaseTransform):
    def __init__(self,
                 bbox):
        super().__init__()
        self.bbox = bbox
        
    def transform(self, image_nd, clicks_lists: List[List[Click]]):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        self.image_changed = True

        clicks_list = clicks_lists[0]
        
        self.original_image = image_nd
        
        rmin, rmax, cmin, cmax = self.bbox
        self._roi_image = image_nd[:, :, rmin:rmax + 1, cmin:cmax + 1]
        
        tclicks_lists = [self._transform_clicks(clicks_list)]
        return self._roi_image.to(image_nd.device), tclicks_lists

    def inv_transform(self, prob_map):
        assert prob_map.shape[0] == 1
        rmin, rmax, cmin, cmax = self.bbox

        new_prob_map = torch.zeros_like(self.original_image[:, :1, :, :], device=prob_map.device, dtype=prob_map.dtype)
        new_prob_map[:, :, rmin:rmax + 1, cmin:cmax + 1] = prob_map
        
        return new_prob_map
    
    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def reset(self):
        pass

    def _transform_clicks(self, clicks_list):

        rmin, rmax, cmin, cmax = self.bbox
        crop_height, crop_width = self._roi_image.shape[2:]

        transformed_clicks = []
        for click in clicks_list:
            new_r = crop_height * (click.coords[0] - rmin) / (rmax - rmin + 1)
            new_c = crop_width * (click.coords[1] - cmin) / (cmax - cmin + 1)
            transformed_clicks.append(click.copy(coords=(new_r, new_c)))
        return transformed_clicks

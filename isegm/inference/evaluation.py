from time import time

import numpy as np
import torch
import cv2

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, resize=None, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        image = sample.image
        gt_mask = sample.gt_mask

        if not resize==None:
            img_height, img_width, _ = image.shape
            target_width, target_height = resize

            # Check if the image dimensions are larger than the resize dimensions, if yes, resize
            if img_width > target_width or img_height > target_height:
                image = cv2.resize(image, dsize=resize)
                gt_mask = cv2.resize(gt_mask, resize, interpolation=cv2.INTER_NEAREST)

        _, sample_ious, _ = evaluate_sample(image, gt_mask, predictor,
                                            sample_id=index, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time

def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.5, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            
            if clicker.not_improving:
                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx-1, clicker.clicks_list, clicker.not_improving)
                break
            
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr
            
            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)
                break

            if click_indx == max_clicks-1:
                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)
            
        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs

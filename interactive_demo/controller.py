import torch
import numpy as np
import os
import cv2
import csv
from pathlib import Path
from tkinter import messagebox

from isegm.inference import clicker
from isegm.inference import utils
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks
from datetime import datetime


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.49): #set to 0.49 to equalize with evaluation instead of 0.75
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.image_name = None #
        self.dsm = None
        self.dsm_name = None
        self.vis = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()
        print("predictior params: ", predictor_params)

    def set_image(self, image, filename):
        self.image = image
        print("controller.py: set image called, shape: ", np.shape(image))
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)
        self.image_name = filename
        """
        # added aval-e for saving data for User study
        my_list = [self.image_name,] #write name of image to file
        list2 = ["time_of_click", "click_count", "type_click", "x", "y", "IoU"]
        with open('/data/ritm_interactive_segmentation/datasets/User_Study/Results/tmp.csv', 'a', newline='') as file: # Opening a CSV file in append mode
            writer = csv.writer(file) # Using csv.writer to write the list to the CSV file
            writer.writerow(my_list)  # Use writerow for single list
            writer.writerow(list2)"""

    def set_dsm(self, dsm, filename):
        print("controller.py: set_dsm called, shape:", np.shape(dsm))
        self.dsm = dsm
        self.dsm_name = filename
        self.reset_last_object(update_image=False) #needed to update dsm somewhere

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, is_positive):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S") #add to file
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })


        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        print("click type: ", type(click))

        self.clicker.add_click(click)
        print("clicker: ", type(self.clicker), type(self._init_mask))
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
            self._save_mask_callback

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()
        """
        ##added aval-e
        ## exclude from here if it is not supposed to write the masks after each click and save coordinates and IoU
        i = self.clicker.num_neg_clicks + self.clicker.num_pos_clicks  ##+1 to account for python counting from 0
        i_i = str(i)
        script_dir = "/data/ritm_interactive_segmentation/datasets/User_Study/Results"
        filename = os.path.join(script_dir,self.image_name + '_' + i_i + '.png')
        temp_mask = self.result_mask
        gt_dir = "/data/ritm_interactive_segmentation/datasets/User_Study/GT-masks"
        gt_file = os.path.join(gt_dir, self.image_name +'.png')
        #print("gt", gt_file)
        gt_mask = cv2.imread(gt_file)[:, :, 0] > 127
        iou = utils.get_iou(gt_mask, temp_mask)
        #print("IoU:", iou)
        temp_mask = temp_mask.astype(np.uint8)
        temp_mask *= 255 // temp_mask.max()
        if not cv2.imwrite(filename, temp_mask):
            raise Exception("Could not write prediction")
        #print("click:", i, is_positive, "x", x, "y", y, "IoU", iou) #save this to file for user study
        my_list = [current_time, i, is_positive, x, y, iou]

        with open('/data/ritm_interactive_segmentation/datasets/User_Study/Results/tmp.csv', 'a',
                  newline='') as file:  # Opening a CSV file in append mode
            writer = csv.writer(file)  # Using csv.writer to write the list to the CSV file
            writer.writerow(my_list)  # Use writerow for single list"""

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()
        my_list = ["last click undone"]

        with open('/data/ritm_interactive_segmentation/datasets/User_Study/Results/tmp.csv', 'a',
                  newline='') as file:  # Opening a CSV file in append mode
            writer = csv.writer(file)  # Using csv.writer to write the list to the CSV file
            writer.writerow(my_list)  # Use writerow for single list

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S") #add to file
        print(current_time)
        my_list = [current_time, "avalanche finished"] #write name of image to file
        with open('/data/ritm_interactive_segmentation/datasets/User_Study/Results/tmp.csv', 'a', newline='') as file: # Opening a CSV file in append mode
            writer = csv.writer(file) # Using csv.writer to write the list to the CSV file
            writer.writerow(my_list)  # Use writerow for single list

    def reset_last_object(self, update_image=True):
        print("reset_last_object callt")
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        print("reset_predictior called")
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)
        if self.dsm is not None:
            self.predictor.set_input_dsm(self.dsm)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask

    def get_visualization(self, alpha_blend, click_radius, bbox=None):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        if bbox is not None:           
            self.vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                            clicks_list=self.clicker.clicks_list, radius=click_radius, bbox=bbox, vis=self.vis)
        else:
            self.vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                            clicks_list=self.clicker.clicks_list, radius=click_radius, bbox=bbox)        

        return self.vis

    def dismiss_bbox(self):
        pass

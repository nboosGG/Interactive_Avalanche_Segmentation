import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from tkcalendar import Calendar, DateEntry
from tktimepicker import AnalogPicker, AnalogThemes, constants

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame
from datetime import datetime

import rasterio
import rasterio.mask
import rasterio.windows
import rasterio.transform

import pandas as pd
import geopandas as gpd

import shapefile

from shapely import geometry

import matplotlib.pyplot as plt


def show_matrix(arr, title: str):
    plt.imshow(arr, cmap='hot')
    plt.title(title)
    plt.show()

class InteractiveDemoApp(ttk.Frame):

    image_name = None
    dsm_name = None
    loaded_tif = False
    loaded_dsm = False
    current_bounds = None
    current_resolution = None
    polygonlist = None
    image_transform = None
    use_DSM = None
    load_InSAR = None
    zoomed_out = None

    def __init__(self, master, args, model):
        super().__init__(master)
        self.master = master
        master.title("Reviving Iterative Training with Mask Guidance for Interactive Segmentation")
        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.zoomed_out = False
        self.loaded_tif = False
        self.loaded_dsm = False
        self.map_ortho = None
        self.map_dsm = None
        self.cur_shape = None
        self.map_temp_results = None
        self.use_DSM = tk.BooleanVar(value=False)
        self.load_InSAR = tk.BooleanVar(value=False)
        self.image_resolution = tk.DoubleVar(value=1)
        self.fringe = tk.DoubleVar(value=0.5)

        self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = args.limit_longest_size

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image,
                                                get_polylist_callback=self._get_polylist_transform_callback)

        self._init_state()
        self._add_menu()
        self._add_canvas()
        self._add_buttons()

        master.bind('<space>', lambda event: self._finish_object())
        master.bind('a', lambda event: self.controller.partially_finish_object())

        self.state['zoomin_params']['skip_clicks'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['target_size'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w', callback=self._reset_predictor)
        self.state['predictor_params']['net_clicks_limit'].trace(mode='w', callback=self._change_brs_mode)
        self.state['lbfgs_max_iters'].trace(mode='w', callback=self._change_brs_mode)
        self._change_brs_mode()

    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=False),
                'fixed_crop': tk.BooleanVar(value=False),
                'skip_clicks': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=min(400, self.limit_longest_size)),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },

            'predictor_params': {
                'net_clicks_limit': tk.IntVar(value=8)
            },
            'brs_mode': tk.StringVar(value='NoBRS'),
            'prob_thresh': tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20), #20

            'alpha_blend': tk.DoubleVar(value=0.1),
            'click_radius': tk.IntVar(value=5),

        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill='x')

        button = FocusButton(self.menubar, text='Load image', command=self._load_image_callback)
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text='Load DSM', command=self._load_dsm_callback)
        button.pack(side=tk.LEFT)
        self.save_mask_btn = FocusButton(self.menubar, text='Save SHP', command=self._save_polygon_callback)
        self.save_mask_btn.pack(side=tk.LEFT)
        self.save_mask_btn.configure(state=tk.DISABLED)

        self.load_mask_btn = FocusButton(self.menubar, text='Load mask', command=self._load_mask_callback)
        self.load_mask_btn.pack(side=tk.LEFT)
        self.load_mask_btn.configure(state=tk.DISABLED)

        # Add label to display the image name
        self.image_name_label = tk.Label(self.menubar, text='', anchor='e', padx=10)
        self.image_name_label.pack(side=tk.LEFT, fill='x', expand=True)

        #Add label to display the dsm name
        self.dsm_name_label = tk.Label(self.menubar, text='', anchor='e', padx=10)
        self.dsm_name_label.pack(side=tk.LEFT, fill='x', expand=True)


        button = FocusButton(self.menubar, text='About', command=self._about_callback)
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text='Exit', command=self.master.quit)
        button.pack(side=tk.LEFT)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)
        
        self.bbox = None

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.control_frame

        self.clicks_options_frame = FocusLabelFrame(master, text="Clicks management")
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.finish_object_button = \
            FocusButton(self.clicks_options_frame, text='Finish\nobject', bg='#b6d7a8', fg='black', width=15, height=2,
                        state=tk.DISABLED, command=self._finish_object)
        self.finish_object_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.undo_click_button = \
            FocusButton(self.clicks_options_frame, text='Undo click', bg='#ffe599', fg='black', width=15, height=2,
                        state=tk.DISABLED, command=self.controller.undo_click)
        self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_clicks_button = \
            FocusButton(self.clicks_options_frame, text='Reset clicks', bg='#ea9999', fg='black', width=15, height=2,
                        state=tk.DISABLED, command=self._reset_last_object)
        self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        
        self.bbx_options_frame = FocusLabelFrame(master, text="Bounding Box management")
        self.bbx_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.confirm_bbx_button = \
            FocusButton(self.bbx_options_frame, text='Confirm Bounding Box', bg='#b6d7a8', fg='black', width=15, height=2,
                        state=tk.DISABLED, command=self._confirm_bbox)
        self.confirm_bbx_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_bbx_button = \
            FocusButton(self.bbx_options_frame, text='Reset Bounding Box', bg='#ea9999', fg='black', width=15, height=2,
                        state=tk.NORMAL, command=self._reset_bbox)
        self.reset_bbx_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        #dismiss bbox button
        self.dismiss_bbx_button = \
            FocusButton(self.bbx_options_frame, text='Dismiss Box', bg='#ffc966', fg='black', width=15, height=2,
                        state=tk.NORMAL, command=self._dismiss_bbox)
        self.dismiss_bbx_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        #shapefile buttons
        self.shapefile_options_frame = FocusLabelFrame(master, text="Shapefile management")
        self.shapefile_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.safe_mask_button = \
            FocusButton(self.shapefile_options_frame, text='Save prediction', bg='#b6d7a8', fg='black', width=15, height=2,
                        state=tk.NORMAL, command=self._store_polygon)
        self.safe_mask_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.display_all_masks_button = \
            FocusButton(self.shapefile_options_frame, text='Display prev prediction', bg='#b6d7a8', fg='black', width=15, height=2,
                        state=tk.DISABLED, command=self._display_all_masks)
        self.display_all_masks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_polygonlist = \
            FocusButton(self.shapefile_options_frame, text='Temp', bg='#b6d7a8', fg='black', width=15, height=2,
                        state=tk.NORMAL, command=self._temp)
        self.reset_polygonlist.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.image_prediction_frame = FocusLabelFrame(master, text="Flags")
        self.image_prediction_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.use_DSM_button = \
            FocusCheckButton(self.image_prediction_frame, text='use DSM', variable=self.use_DSM, onvalue = 1, offvalue = 0,
                              height = 2, width = 15)
        self.use_DSM_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.loadInSAR_button = \
            FocusCheckButton(self.image_prediction_frame, text='load InSAR', variable=self.load_InSAR, onvalue = 1, offvalue = 0,
                              height = 2, width = 15)
        self.loadInSAR_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)


        #zooming not used
        self.zoomin_options_frame = FocusLabelFrame(master, text="ZoomIn options")
        self.zoomin_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        """
        FocusCheckButton(self.zoomin_options_frame, text='Use ZoomIn', command=self._reset_predictor,
                         variable=self.state['zoomin_params']['use_zoom_in']).grid(row=0, column=0, padx=10)
        FocusCheckButton(self.zoomin_options_frame, text='Fixed crop', command=self._reset_predictor,
                         variable=self.state['zoomin_params']['fixed_crop']).grid(row=1, column=0, padx=10)
        tk.Label(self.zoomin_options_frame, text="Skip clicks").grid(row=0, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Target size").grid(row=1, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Expand ratio").grid(row=2, column=1, pady=1, sticky='e')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['skip_clicks'],
                              min_value=-1, max_value=None, vartype=int,
                              name='zoom_in_skip_clicks').grid(row=0, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['target_size'],
                              min_value=100, max_value=self.limit_longest_size, vartype=int,
                              name='zoom_in_target_size').grid(row=1, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['expansion_ratio'],
                              min_value=1.0, max_value=2.0, vartype=float,
                              name='zoom_in_expansion_ratio').grid(row=2, column=2, padx=10, pady=1, sticky='w')
        self.zoomin_options_frame.columnconfigure((0, 1, 2), weight=1)
        """

        #always use noBRS mode
        self.brs_options_frame = FocusLabelFrame(master, text="BRS options")
        """
        #self.brs_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        #menu = tk.OptionMenu(self.brs_options_frame, self.state['brs_mode'],
        #                     *self.brs_modes, command=self._change_brs_mode)
        #menu.config(width=11)
        #menu.grid(rowspan=2, column=0, padx=10)
        """

        self.net_clicks_label = tk.Label(self.brs_options_frame, text="Network clicks")
        self.net_clicks_label.grid(row=0, column=1, pady=2, sticky='e')
        self.net_clicks_entry = BoundedNumericalEntry(self.brs_options_frame,
                                                      variable=self.state['predictor_params']['net_clicks_limit'],
                                                      min_value=0, max_value=None, vartype=int, allow_inf=True,
                                                      name='net_clicks_limit')
        self.net_clicks_entry.grid(row=0, column=2, padx=10, pady=2, sticky='w')
        self.lbfgs_iters_label = tk.Label(self.brs_options_frame, text="L-BFGS\nmax iterations")
        self.lbfgs_iters_label.grid(row=1, column=1, pady=2, sticky='e')
        self.lbfgs_iters_entry = BoundedNumericalEntry(self.brs_options_frame, variable=self.state['lbfgs_max_iters'],
                                                       min_value=1, max_value=1000, vartype=int,
                                                       name='lbfgs_max_iters')
        self.lbfgs_iters_entry.grid(row=1, column=2, padx=10, pady=2, sticky='w')
        self.brs_options_frame.columnconfigure((0, 1), weight=1)

        self.image_resolution_frame = FocusLabelFrame(master, text="Image Resolution (pixel size in meters [m])")
        self.image_resolution_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.image_resolution_frame, from_=0.1, to=5.0, resolution=0.1, command=self._update_image_resolution,
                             variable=self.image_resolution, sliderlength=30).pack(padx=10, side=tk.LEFT)
        self.reload_image_button = \
            FocusButton(self.image_resolution_frame, text='Reload Image', bg='#b6d7a8', fg='black', width=15, height=2,
                        state=tk.NORMAL, command=self._reload_image)
        self.reload_image_button.pack(side=tk.RIGHT, fill=tk.X, padx=10, pady=3)

        self.prob_thresh_frame = FocusLabelFrame(master, text="Predictions threshold")
        self.prob_thresh_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.prob_thresh_frame, from_=0.0, to=1.0, resolution=0.1, command=self._update_prob_thresh,
                             variable=self.state['prob_thresh']).pack(padx=10, side=tk.LEFT)

        self.alpha_blend_frame = FocusLabelFrame(master, text="Alpha blending coefficient")
        self.alpha_blend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.alpha_blend_frame, from_=0.0, to=1.0, resolution=0.1, command=self._update_blend_alpha,
                             variable=self.state['alpha_blend']).pack(padx=10, anchor=tk.CENTER, side=tk.LEFT)

        self.click_radius_frame = FocusLabelFrame(master, text="Visualisation click radius")
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.click_radius_frame, from_=0, to=7, resolution=1, command=self._update_click_radius,
                             variable=self.state['click_radius']).pack(padx=10, anchor=tk.CENTER, side=tk.LEFT)
        
        self.insar_fringes_frame = FocusLabelFrame(master, text="Fringe count")
        self.insar_fringes_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.insar_fringes_frame, from_=0.25, to=5, resolution=0.25, command=self._nothing,
                             variable=self.fringe).pack(padx=10, anchor=tk.CENTER, side=tk.LEFT)
        


                                  

        ################################################
        #avalanche properties
        self.avalanche_properties_frame = FocusLabelFrame(self, text="Avalanche Properties")
        self.avalanche_properties_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)

        ## avalanche information
        self.avalanche_information_frame = FocusLabelFrame(self.avalanche_properties_frame, text="Avalanche Information")
        self.avalanche_information_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        self.avalanche_area_lbl = tk.Label(self.avalanche_information_frame, text='Avalanche Area:   not calculated yet')
        self.avalanche_area_lbl.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)


        ## avalanche date entry
        self.release_date_frame = FocusLabelFrame(self.avalanche_properties_frame, text="Release Date + Time")
        self.release_date_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        self.cal_date = DateEntry(self.release_date_frame, width=12, background='darkblue',
                                  foreground='white', borderwidth=2)
        self.cal_date.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.time_lbl = tk.Label(self.release_date_frame, text='Time:')
        self.time_lbl.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.release_time_btn = FocusButton(self.release_date_frame, text='Set Time', bg='#b6d7a8', fg='black', width=15, height=2,
                        state=tk.NORMAL, command=self._get_time)
        self.release_time_btn.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=10)


        ## avalanche type entry
        ### part 1
        self.avalanche_type_frame = FocusLabelFrame(self.avalanche_properties_frame, text="     Avalanche Type     |     Avalanche Size    |")
        self.avalanche_type_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        self.avalanche_type_selection = ttk.Combobox(self.avalanche_type_frame, state="readonly", values=["slab", "glide snow", "loose snow", "unknown"])
        self.avalanche_type_selection.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=10)
        
        self.avalanche_size_selection = ttk.Combobox(self.avalanche_type_frame, state="readonly", values=["1", "2", "3", "4", "5", "unknown"])
        self.avalanche_size_selection.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=10)


        ### part 2
        self.avalanche_type_frame2 = FocusLabelFrame(self.avalanche_properties_frame, text="     Snow Moisture      |     Release Type      |")
        self.avalanche_type_frame2.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        self.snow_moisture_selection = ttk.Combobox(self.avalanche_type_frame2, state="readonly", values=["wet", "dry", "unknown"])
        self.snow_moisture_selection.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=10)

        self.release_type_selection = ttk.Combobox(self.avalanche_type_frame2, state="readonly", values=["natural", "person", "explosive", "snow groomer", "unknown"])
        self.release_type_selection.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=10)

        
    def _nothing(self):
        """empty function to use as playceholder"""
        return
    
    def _list_changer(self, ll):
        ll_ret = []

        for l in ll:
            ll_ret.append((l,255))
        return ll_ret
    
    def _translate_polygon(self, polylist, aff: rasterio.Affine):
        #toDo: iterate over polygon, and subtract topleft coordinate, then also multiply with resolution
        x_offset = aff[2]
        y_offset = aff[5]

        ret_poly_list = []


        for poly in polylist:
            xx, yy = poly.exterior.coords.xy
            #print(xx, yy)

            xx = np.subtract(xx, int(x_offset))
            yy = np.subtract(int(y_offset), yy)

            ret_poly_list.append(geometry.Polygon(zip(xx,yy)))
        
        return ret_poly_list

    def _get_current_stored_polygons(self):
        return self.polygonlist

    def _temp(self):
        print("#polygons: ", len(self.polygonlist))
        print("stuff: ", type(self.polygonlist))
        """print("cur affine transform: ", self.image_transform)
        print("bounding box: ", self.current_bounds)
        aff = rasterio.transform.Affine(int(self.image_transform[0]),int(self.image_transform[1]),int(self.image_transform[2]),
                                        int(self.image_transform[3]),int(self.image_transform[4]),int(self.image_transform[5]))
        print("poly translation:")
        polylist = self._translate_polygon(self.polygonlist, aff)
        #for poly in self.polygonlist:
        #    print("it is: ", poly)
        
        shappe = np.array(self.cur_shape)
        #shape = shape[:,:,0]

        print("shape: ", shappe, type(shappe), shappe[0], shappe[1], type(shappe[0]))
        shapppe = np.zeros((2,), dtype=np.int)
        
        shapppe[0] = shappe[0].astype(int)
        shapppe[1] = shappe[1].astype(int)
        print("new: ", np.shape(shapppe), shapppe)
        #geom = [shapes for shapes in self.polygonlist.geometry]
        #templist = []
        #img = rasterio.features.rasterize(self._list_changer(self.polygonlist), shapppe, self.image_transform)
        #out = np.empty(shapppe)
        #print("out shape: ", np.shape(out))

        #for poly in polylist:
        #    print(poly)

        #img = rasterio.features.rasterize(self.polygonlist, shapppe, transform=self.image_transform, default_value=1, fill=5)

        #print("image shape: ", np.shape(img), np.amax(img), np.amin(img))
        #show_matrix(img, "daaaamn")

        #self.controller._result_mask = img

        self.controller.update_result_mask(self.polygonlist, self.image_transform)

        print("shapes: ", np.shape(self.controller._result_mask), np.shape(self.controller.image))"""


    def _get_polylist_transform_callback(self):
        return self.polygonlist, self.image_transform

    
    def _reset_avalanche_properties(self):
        #reset all avalanche properties
        #reset date
        today = datetime.now().date()
        self.cal_date.set_date(today)
        
        #reset time:
        self._reset_time()

        #reset dropdown menus
        self.avalanche_type_selection.select_clear()
        self.avalanche_size_selection.select_clear()
        self.snow_moisture_selection.select_clear()
        self.release_type_selection.select_clear()
        

    def _updateTime(self, time):
        print("time datatype: ", type(time), time, type(time[0]), type(time[2]))

        self.time_lbl.configure(text="{}:{} {}".format(*time)) # remove 3rd flower bracket in case of 24 hrs time

    def _reset_time(self):
        self.time_lbl.configure(text="time unknown")
        self._updateTime(tuple([0,0,'AM']))
    
    
    def _get_time(self):
        top = tk.Toplevel(self)

        time_picker = AnalogPicker(top, type=constants.HOURS12)
        time_picker.pack(expand=True, fill="both")

        theme = AnalogThemes(time_picker)
        theme.setDracula()
        #theme.setNavyBlue()
        #theme.setPurple()
        ok_btn = tk.Button(top, text="set time", command=lambda: self._updateTime(time_picker.time()))
        ok_btn.pack()
        unkown_time_btn = tk.Button(top, text="time unknown", command=lambda: self._resetTime())
        unkown_time_btn.pack()
        

    def _calculate_resolution(self, bounds):
        x_dist = bounds[2]-bounds[0]
        y_dist = bounds[3]-bounds[1]

        x_npixel = x_dist / self.image_resolution.get()
        y_npixel = y_dist / self.image_resolution.get()

        max_npixel = np.maximum(x_npixel, y_npixel)

        divider = 1
        if max_npixel > 2000:
            divider = 2000. / max_npixel

        x_npixel = x_npixel / divider
        y_npixel = y_npixel / divider

        return np.array([y_npixel, x_npixel]).astype(int)
    
    def inSAR_image_preprocessing(self, image):
        print("inSAR image preprocessing: ", np.shape(image))
        image = image[0,:,:]
        

        max_val = np.amax(image)
        min_val = np.amin(image)
        image = (image - min_val) / (max_val - min_val)
        image *= 255



        image = cv2.GaussianBlur(image, (3,3), 0)

        image = np.array([image,image,image])


        print("inSAR preprocessing, return image shape: ", np.shape(image))
        return image.astype(np.uint8)

    def _load_image(self, filename, bounds=None, use_current_bounds=False):
        load_entire_new_image_flag = False
        print("load image2 called, use_DSM state: ")
        map = rasterio.open(filename)
        if bounds is None:
            if use_current_bounds:
                bounds = self.current_bounds
            else:
                load_entire_new_image_flag = True
                bounds = map.bounds
        assert(bounds is not None)

        out_shape = self._calculate_resolution(bounds)

        window = rasterio.windows.from_bounds(*bounds, map.transform)

        #print("widnow dir: ", dir(window))

        image = map.read(window=window, out_shape=(map.count, out_shape[0], out_shape[1]))
        print("raw image shape: ", np.shape(image))
        print("inSAR flag: ", self.load_InSAR.get())
        image = image[:3,:,:]
        
        if self.load_InSAR.get():
            image = self.inSAR_image_preprocessing(image)

        image = np.rollaxis(image, 0,3) #from [3,ydim,xdim] to [ydim,xdim,3]

        print("loaded image shape: ", np.shape(image))

        self.image_transform = rasterio.Affine(self.image_resolution.get(), 0, bounds[0], 0, -self.image_resolution.get(), bounds[3])
        self.controller.set_image(image, self.image_name, reset_result_mask=load_entire_new_image_flag)
        self.map_ortho = map
        self.current_bounds = np.array(bounds)
        self.cur_shape = np.shape(np.array(image))
        return image
    
    def _load_dsm(self, filename, bounds=None, use_current_bounds=False):
        print("load dsm2 called")
        map = rasterio.open(filename)
        if bounds is None:
            if use_current_bounds:
                bounds = self.current_bounds
            else:
                bounds = map.bounds
        assert(bounds is not None)

        out_shape = self._calculate_resolution(bounds)

        window = rasterio.windows.from_bounds(*bounds, map.transform)

        #print("widnow dir: ", dir(window))

        image = map.read(window=window, out_shape=(map.count, out_shape[0], out_shape[1]))
        image = np.rollaxis(image, 0,3) #from [1,ydim,xdim] to [ydim,xdim,1]

        self.controller.set_dsm(image, self.image_name)

        self.map_dsm = map
        return image

    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif"),
                ("All files", "*.*"),
            ], title="Chose a dsm file")
            self.image_name = Path(filename).stem

            if len(filename) > 0:
                self.polygonlist = []
                if filename[-4:] == ".tif":
                    image = self._load_image(filename)
                    self.loaded_tif = True
                    self.image_name = filename
                else:

                    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                    print("loaded image, shape: ", np.shape(image))
                    self.controller.set_image(image, self.image_name)

                if self.image_on_canvas is not None:
                    if self.image_on_canvas.bbox is not None:
                        self._reset_bbox()
                
                self.save_mask_btn.configure(state=tk.NORMAL)
                self.load_mask_btn.configure(state=tk.NORMAL)
    
                # Update the image name label
                self.image_name_label.config(text=f'Image: {self.image_name}')

                #update some UI stuff
                #reset avalanche properties
                self._reset_avalanche_properties()


    def _load_dsm_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("DSM", "*.tif"),
                ("All files", "*.*"),
            ], title="Chose a DSM file")
            self.dsm_name = Path(filename).stem
            print("dsm name: ", self.dsm_name)

            if len(filename) > 0:
                #image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                #dsm = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) #maybe for real dsm this needs to be used
                print("entered _load_dsm call")
                self._load_dsm(filename)
                self.loaded_dsm = True
                self.dsm_name = filename
                self.dsm_name_label.config(text=f'DSM: {self.dsm_name}')


    def translate_polygon(self, contour, translate_x, translate_y):
        ll = []
        for i, cnt in enumerate(contour):
            a=5


    def _store_contours(self, contour, hierarchy):
        affine_trans = self.image_transform
        x_resolution = affine_trans[0]
        y_resolution = affine_trans[4]

        top_left_x = self.current_bounds[0]
        top_left_y = self.current_bounds[3]


        cur_polygon_id = 0

        has_next_outer_polygon = hierarchy[0,0,0] != -1

        while(True):
            
            print("polygon generater state: id, continue flag:", cur_polygon_id, has_next_outer_polygon)

            main_polygon = []

            if len(contour[cur_polygon_id]) <= 3:
                print("found small polygon (<3points). skip it")
                if not has_next_outer_polygon:
                    break
                cur_polygon_id = hierarchy[0,cur_polygon_id,0]
                has_next_outer_polygon = cur_polygon_id != -1
                continue

            for p in contour[cur_polygon_id]:
                pp = p[0]
                #print("pp: ", pp)
                pp[0] = pp[0] * x_resolution + top_left_x
                pp[1] = pp[1] * y_resolution + top_left_y
                main_polygon.append(pp)


            #iterate over holes
            cur_hole_polygon_id = hierarchy[0,cur_polygon_id,2]
            has_next_hole_polygon = cur_hole_polygon_id != -1
            #print("next hole polygon id, + flag: ", cur_hole_polygon_id, has_next_hole_polygon)
            
            hole_polygons = []
            hole_polygon_counter = 0
            

            while(has_next_hole_polygon):
                #make_hole_polygon
                hole_polygon = []

                
                for p in contour[cur_hole_polygon_id]:
                    pp = p[0]
                    pp[0] = pp[0] * x_resolution + top_left_x
                    pp[1] = pp[1]* y_resolution + top_left_y
                    hole_polygon.append(pp)

                hole_polygons.append(hole_polygon)
                hole_polygon_counter += 1

                cur_hole_polygon_id = hierarchy[0,cur_hole_polygon_id,0]
                has_next_hole_polygon = cur_hole_polygon_id != -1
            
            polygonomy = geometry.Polygon(main_polygon, holes=hole_polygons)

            self.polygonlist.append(polygonomy)

            if not has_next_outer_polygon:
                break

            cur_polygon_id = hierarchy[0,cur_polygon_id,0]
            has_next_outer_polygon = hierarchy[0,cur_polygon_id,0] != -1

        print("polygon list: ", len(self.polygonlist))
        return

    def _store_polygon(self):

        mask = (self.controller.result_mask * 255).astype(np.uint8)

        if mask is None:
            return
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self._store_contours(contours, hierarchy)
        

        #update result map in the UI
        self.controller.update_result_mask()

        #delet clicks since they are stored
        self._reset_last_object()


    def _save_polygon_callback(self):
        self.menubar.focus_set()

        if len(self.polygonlist) == 0:
            return


        cur_polygonlist = []
        counter = 0
        for poly in self.polygonlist:

            cur_polygonlist.append([counter,poly])
            counter += 1

        print("#polygons found: ", counter)
        
        gdf = gpd.GeoDataFrame(cur_polygonlist, columns=['ID','geometry'], crs='EPSG:2056', geometry='geometry')

        filename = filedialog.asksaveasfilename(parent=self.master, initialfile=f'{self.image_name}.png', filetypes=[
            ("SHP image", "*.shp"),
        ], title="Save the current mask as...")
        print("output filename: ", filename)

        gdf.to_file(filename)



    def _load_mask_callback(self):
        if not self.controller.net.with_prev_mask:
            messagebox.showwarning("Warning", "The current model doesn't support loading external masks. "
                                              "Please use ITER-M models for that purpose.")
            return

        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Binary mask (png, bmp)", "*.png *.bmp"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                mask = cv2.imread(filename)[:, :, 0] > 127
                self.controller.set_mask(mask)
                self._update_image()

    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            "Developed by:",
            "K.Sofiiuk and I. Petrov",
            "The MIT License, 2021"
        ]

        messagebox.showinfo("About Demo", '\n'.join(text))
        
    def _finish_object(self):
        self._reset_bbox(reset_last_object=False)
        self.controller.finish_object()
        
    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()
    
    def _reload_image(self):
        self._load_image(self.image_name, use_current_bounds=True)
        if self.loaded_dsm:
            self._load_dsm(self.dsm_name, use_current_bounds=True)

    def _update_image_resolution(self, value):
        self.image_resolution.set(value)

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    def _change_brs_mode(self, *args):
        if self.state['brs_mode'].get() == 'NoBRS':
            self.net_clicks_entry.set('INF')
            self.net_clicks_entry.configure(state=tk.DISABLED)
            self.net_clicks_label.configure(state=tk.DISABLED)
            self.lbfgs_iters_entry.configure(state=tk.DISABLED)
            self.lbfgs_iters_label.configure(state=tk.DISABLED)
        else:
            if self.net_clicks_entry.get() == 'INF':
                self.net_clicks_entry.set(8)
            self.net_clicks_entry.configure(state=tk.NORMAL)
            self.net_clicks_label.configure(state=tk.NORMAL)
            self.lbfgs_iters_entry.configure(state=tk.NORMAL)
            self.lbfgs_iters_label.configure(state=tk.NORMAL)

        self._reset_predictor()

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = self.state['brs_mode'].get()
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None if brs_mode == 'NoBRS' else self.state['predictor_params']['net_clicks_limit'].get()

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None

        predictor_params = {
            'brs_mode': brs_mode,
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': net_clicks_limit,
                'max_size': self.limit_longest_size
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': self.state['lbfgs_max_iters'].get()}
        }

        if self.bbox != None:
            predictor_params['bbox'] = self.bbox

        self.controller.reset_predictor(predictor_params)

    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()

        if self.image_on_canvas.bbox is not None:
            messagebox.showwarning("Warning", "Confirm or reset the drawn bounding box before selecting any points.")
            return

        if self.bbox is not None:
            if not (self.bbox[0] <= y <= self.bbox[1]) or not (self.bbox[2] <= x <= self.bbox[3]):
                messagebox.showwarning("Warning", "Selected point is not within the drawn bounding box.")
                return

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        if self._check_entry(self):
            self.controller.add_click(x, y, is_positive)

    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get(),
                                                  bbox=self.bbox)
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas, self.bbox)
            self.image_on_canvas.register_click_callback(self._click_callback)
            self.image_on_canvas.register_widget_state_callback(self._set_click_dependent_widgets_state)

        self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)
    
    def _confirm_bbox(self):
        if self.image_on_canvas.bbox_x1 > self.image_on_canvas.bbox_x2:
            self.image_on_canvas.bbox_x1, self.image_on_canvas.bbox_x2 = self.image_on_canvas.bbox_x2, self.image_on_canvas.bbox_x1
        if self.image_on_canvas.bbox_y1 > self.image_on_canvas.bbox_y2:
            self.image_on_canvas.bbox_y1, self.image_on_canvas.bbox_y2 = self.image_on_canvas.bbox_y2, self.image_on_canvas.bbox_y1

        #bbox = [y1,y2,x1,x2], different than a real bounding box!!!
        bbox = [self.image_on_canvas.bbox_y1, self.image_on_canvas.bbox_y2, self.image_on_canvas.bbox_x1, self.image_on_canvas.bbox_x2]

        print("bbox: ", bbox)

        initial_image_pixel_height = bbox[1]-bbox[0] + 1
        initial_image_pixel_width = bbox[3]-bbox[2] + 1

        #calculate windows and its bound coordinates
        window_ortho = rasterio.windows.Window(bbox[2],bbox[0], initial_image_pixel_width, initial_image_pixel_height)


        bounds_ch_coordinates = rasterio.windows.bounds(window_ortho,self.image_transform)

        self._load_image(self.image_name, bounds=bounds_ch_coordinates)

        if self.loaded_dsm:
            self._load_dsm(self.dsm_name, bounds=bounds_ch_coordinates)
            #assert(False and "dsm boundinng box not implemented yet")

        self.save_mask_btn.configure(state=tk.NORMAL)
        self.load_mask_btn.configure(state=tk.NORMAL)
        self.reset_bbx_button.configure(state=tk.NORMAL)

        #remove the drawn bbox
        self._dismiss_bbox()
        
        #dont know if this stuff is needed
        self._reset_last_object() #to delet previous clicks and stuff
        self._reset_predictor()

    def _dismiss_bbox(self):
        self.bbox = None
        self.canvas.delete("bbox")
        self.image_on_canvas.bbox = None
        self.image_on_canvas._bbox = None
        

    def _reset_bbox(self, reset_last_object = True):               
        self.bbox = None
        
        self.canvas.delete("bbox")
        self.image_on_canvas.bbox = None
        self.image_on_canvas._bbox = None

        if self.loaded_tif:
            self.image_resolution.set(1) #set to fixed resolution
            self._load_image(self.image_name)
        if self.loaded_dsm:
            self.image_resolution.set(1) #set to fixed resolution
            self._load_dsm(self.dsm_name)
        
        if reset_last_object:
            self._reset_last_object()
        self._reset_predictor()


    #everything to create the shapefile    
    def _display_all_masks(self):
        a=5

    def _reset_polygonlist(self):
        self.polygonlist = []
        
    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED
        before_1st_click_state = tk.DISABLED if self.controller.is_incomplete_mask else tk.NORMAL

        #self.finish_object_button.configure(state=after_1st_click_state) #this button is currently not used
        self.undo_click_button.configure(state=after_1st_click_state)
        self.reset_clicks_button.configure(state=after_1st_click_state)
        self.zoomin_options_frame.set_frame_state(before_1st_click_state)
        self.brs_options_frame.set_frame_state(before_1st_click_state)
                
        if self.image_on_canvas.bbox == None:
            self.confirm_bbx_button.configure(state=tk.DISABLED)
        else:
            self.state['zoomin_params']['use_zoom_in'].set(False)
            self.state['zoomin_params']['fixed_crop'].set(False)
            self.state['brs_mode'].set('NoBRS')
            
            self.reset_bbx_button.configure(state=tk.NORMAL)
            self.zoomin_options_frame.set_frame_state(state=tk.DISABLED)
            self.brs_options_frame.set_frame_state(state=tk.DISABLED)
            if self.bbox is None:
                self.confirm_bbx_button.configure(state=tk.NORMAL)
            else:
                self.confirm_bbx_button.configure(state=tk.DISABLED)

        
        if self.polygonlist is not None and len(self.polygonlist) > 0:
            self.reset_polygonlist.configure(state=tk.NORMAL)
        else:
            #self.reset_polygonlist.configure(state=tk.DISABLED)
            a=5

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked

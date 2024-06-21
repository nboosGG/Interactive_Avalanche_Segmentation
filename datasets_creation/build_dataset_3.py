import os
import gc
import numpy as np

from scipy.ndimage import gaussian_filter, median_filter

from scipy import fftpack


import rasterio
import rasterio.mask
#from rasterio.enums import Resampling
#from rasterio.profiles import DefaultGTiffProfile


import matplotlib.pyplot as pl
import shapefile

import cv2

from PIL import Image, ImageDraw


def show_matrix(matrix, verbose: bool, title: str):
    """visualizes a 2d matrix/array if verbose evaluates to True"""
    if (verbose):
        pl.imshow(matrix, cmap='hot')
        pl.title(title)
        pl.show()

def get_bounds_from_shp(shp):
    """get maximal bounds from an shp file 
    by iterating over all polygons"""
    #iterate over all points of the polygon and get the left, right, top and bottom bounds
    top = None
    bottom = None
    right = None
    left = None

    if len(shp.points) == 0:
        return False, None


    for point in shp.points:
        if top is None:
            top = point[1]
        if bottom is None:
            bottom = point[1]
        if right is None:
            right = point[0]
        if left is None:
            left = point[0]

        top = np.maximum(top, point[1])
        bottom = np.minimum(bottom, point[1])
        right = np.maximum(right, point[0])
        left = np.minimum(left, point[0])
    
    bounds = np.array([np.floor(left), np.floor(bottom), np.ceil(right), np.ceil(top)])
    return True, bounds.astype(int)

def extend_bounds(bounds, extend, points_per_meter):
    """extend the bounds in all direction by [extend] with a minimum shape of (600,600)"""

    dist_x = bounds[2] - bounds[0]
    dist_y = bounds[3] - bounds[1]

    bound_x = extend * dist_x / 100
    bound_y = extend * dist_y / 100

    extended_bounds = np.array([bounds[0] - bound_x,
                     bounds[1] - bound_y, 
                     bounds[2] + bound_x, 
                     bounds[3] + bound_y])
    
    #increase image to at least (600,600)
    x_to_extend = 600 - (extended_bounds[2]-extended_bounds[0]) * points_per_meter
    y_to_extend = 600 - (extended_bounds[3]-extended_bounds[1]) * points_per_meter

    x_to_extend /= points_per_meter
    y_to_extend /= points_per_meter
    
    #if to_extend uneven, some more code to get exactly 600 in each dimension
    if x_to_extend > 0:
        left_to_extend = x_to_extend // 2
        right_to_extend = x_to_extend - left_to_extend
        extended_bounds[0] -= left_to_extend
        extended_bounds[2] += right_to_extend
    
    if y_to_extend > 0:
        top_to_extend = y_to_extend // 2
        bottom_to_extend = y_to_extend - top_to_extend
        extended_bounds[1] -= bottom_to_extend
        extended_bounds[3] += top_to_extend 

    return extended_bounds



def create_directiories(folder_list):
    """check if directories exist, if not create them
    input: a list of folder paths"""
    for path in folder_list:
        if not os.path.exists(path):
            # Create the directory
            os.makedirs(path)
            print("created: ", path)


def window_createre(bounds, counter, some_map):
    main_window = rasterio.windows.Window()
    if counter == 0:
        return 
    
def frag_window_bounds(window_bounds):
    return window_bounds


def calc_resolution(bounds, pixel_per_meter):
    """calculate the number of pixels per dimensions"""
    dist_x = bounds[2] - bounds[0]
    dist_y = bounds[3] - bounds[1]
    assert(dist_x > 0 and dist_y > 0 and "invalid bounds, critical error")

    pixel_x_direc = dist_x * pixel_per_meter
    pixel_y_direc = dist_y * pixel_per_meter
    return tuple(np.array([pixel_y_direc, pixel_x_direc]).astype(int))

def gaussian_blur(data, sigma):
    return gaussian_filter(data, sigma)

def median_blur(data, size):
    return median_filter(data, size)





def lpf(image, radius):
    show_matrix(image, 0, "initial image")
    image1_np = np.array(image)
    #fft of image
    fft1 = fftpack.fftshift(fftpack.fft2(image1_np))

    #Create a low pass filter image
    x,y = image1_np.shape[1],image1_np.shape[0]
    #size of circle
    #e_x,e_y=50,50
    e_x,e_y=radius,radius
    #create a box 
    bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))

    low_pass=Image.new("L",(image1_np.shape[1],image1_np.shape[0]),color=0)
    

    draw1=ImageDraw.Draw(low_pass)
    draw1.ellipse(bbox, fill=1)

    low_pass_np=np.array(low_pass)

    

    #print("low pass shape: ", np.shape(low_pass_np))
    show_matrix(low_pass_np.astype(np.float32), 0, "low pass filter")

    #multiply both the images
    #fft1 = np.transpose(fft1)
    #low_pass_np = np.transpose(low_pass_np)
    filtered=np.multiply(fft1,low_pass_np)

    #inverse fft
    ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
    ifft2 = np.maximum(0, np.minimum(ifft2, 255))
    ifft2 = ifft2.astype(np.uint8)

    #print("ifft2 shape: ", np.shape(ifft2), type(ifft2[0,0]))

    #cv2.imshow("lpf image", ifft2)
    show_matrix(ifft2, 0, "blurred image")
    return ifft2

def blur(data, method):
     #print("datashape: ", np.shape(data))

    nchannels = np.shape(data)[0]
    for i in range(nchannels):
        if method == 1:
            sigma = 0.5
            data[i,:,:] = gaussian_blur(data[i,:,:], sigma)
        elif method == 2:
            size = 3
            data[i,:,:] = median_blur(data[i,:,:], size)
        elif method == 3:
            radius = 400 #in pixels
            #print("initial shape: ", np.shape(data))
            #data[i,:,:] = lpf(data[i,:,:], radius)
            #data[i,:,:] = butter_lowpass_filter()
            data[i,:,:] = lpf(data[i,:,:], radius)

    return data

def rgb_normalization1(data):
    return data / 255

def rgb_normalization2(data):
    min_val = np.amin(data)
    max_val = np.amax(data)
    data = (data - min_val) / (max_val - min_val)
    return data


def main():
    data_path = "/media/boosnoel/LaCie/noel/DS_v3_Sammlung/"
    target_path = "/media/boosnoel/LaCie/noel/ds_v3_0p5m_RGBnormalization3/"

    path_storage_dsm = target_path + "dsm/"
    path_storage_ortho = target_path + "images/"
    path_storage_mask = target_path + "masks/"

    create_directiories([path_storage_dsm, path_storage_ortho, path_storage_mask])

    sample_counter = 0

    verbose_show_data = False

    ultracam_flagrgb_normalization = False

    datapoints_per_meter_read = 10
    datapoints_per_meter_write = 2
    

    extend = 5 #in % of the image size that gets padded in each direction

    for folder in os.listdir(data_path):
        print("----------------------------------")
        print("folder name: ", folder)
        ortho_map = None
        dsm_map = None
        src_polys = None
        profile = None
        ultracam_flag = False
        for filename in os.listdir(data_path + folder + "/"):
            #print("filename: ", filename)

            if filename.startswith("Ortho"):
                ortho_map = rasterio.open(data_path + folder + "/" + filename)
                profile = ortho_map.profile

                if 'ultra19' in filename:
                    ultracam_flag = True
                else:
                    ultracam_flag = False

            if filename.startswith("DSM"):
                dsm_map = rasterio.open(data_path + folder + "/" + filename)

            if filename.endswith(".shp"):
                src_polys = shapefile.Reader(data_path + folder + "/" + filename)


        if ortho_map is None or dsm_map is None or src_polys is None:
            print("could not find all 3 needed files (orho, dsm, shp). skip this one: ", folder)
            ortho_map = None
            dsm_map = None
            src_polys = None
            gc.collect()
            continue
            
        #if not ultracam_flag: #for testing reasons
        #    continue

        for iPoly in range(len(src_polys)):
            print("sampe nr: ", sample_counter)


            cur_poly = src_polys.shape(iPoly)
            suceeded, shp_bounds = get_bounds_from_shp(cur_poly)

            if not suceeded:
                continue
            
            bounds_extended = extend_bounds(shp_bounds, extend, datapoints_per_meter_read)

            top_left_coordinate = np.array([bounds_extended[0], bounds_extended[3]])

            

            shape = calc_resolution(bounds_extended, datapoints_per_meter_read)
            height, width = shape

            ortho_data_read_window = rasterio.windows.from_bounds(*bounds_extended, ortho_map.transform)
            ortho_data = ortho_map.read(out_shape=(ortho_map.count, shape[0], shape[1]), window=ortho_data_read_window)
            ortho_data = ortho_data[:3,:,:]

            if ultracam_flagrgb_normalization:
                ortho_data = (ortho_data * (255/43000)).astype(np.uint8)

            dsm_data_read_window = rasterio.windows.from_bounds(*bounds_extended, dsm_map.transform)
            dsm_data = dsm_map.read(1, out_shape=shape, window=dsm_data_read_window)
            dsm_data = np.expand_dims(dsm_data, axis=0)

            show_matrix(ortho_data[0,:,:], verbose_show_data, "ortho data, blue channel")
            show_matrix(dsm_data, verbose_show_data, "dsm data")

            print("initial shapes: ", np.shape(ortho_data), np.shape(dsm_data))


            #blur and downscale data
            downscale_factor = datapoints_per_meter_read / datapoints_per_meter_write

            blur_method = 0
            ortho_data = blur(ortho_data, blur_method)
            #dsm_data = blur(dsm_data, blur_method)


            ortho_data = rgb_normalization2(ortho_data)

            #show_matrix(ortho_data[0,:,:], 1, "ortho norma")

            print("data stats: ", np.amin(ortho_data), np.amax(ortho_data))

            show_matrix(ortho_data[0,:,:], verbose_show_data, "ortho blurred")

            #now downscale
            write_shape = calc_resolution(bounds_extended, datapoints_per_meter_write)
            write_shape = np.array(write_shape)
            temp_shape = write_shape[0]
            write_shape[0] = write_shape[1]
            write_shape[1] = temp_shape

            ortho_data = np.moveaxis(ortho_data, 0, 2)
            dsm_data = np.moveaxis(dsm_data, 0,2)
            

            ortho_data = cv2.resize(ortho_data, 
                                    dsize=write_shape,
                                    interpolation=cv2.INTER_AREA)
            dsm_data = cv2.resize(dsm_data,
                                  dsize=write_shape,
                                  interpolation=cv2.INTER_AREA)

            ortho_data = np.moveaxis(ortho_data, 2, 0)

            show_matrix(ortho_data[0,:,:], verbose_show_data, "ortho resampled")


            print("write shapes: ", np.shape(ortho_data), np.shape(dsm_data))
            print("write stats: ", np.amin(ortho_data), np.amax(ortho_data))

            resolution_write = 1. / datapoints_per_meter_write

            transform = ortho_map.transform
            transform_adjusted = rasterio.Affine(resolution_write, transform[1], top_left_coordinate[0],
                                        transform[3], -1 * resolution_write, top_left_coordinate[1])

            

            profile_ortho = ortho_map.profile.copy()
            profile_ortho.update(transform=transform_adjusted, count=3, dtype='float32')


            _, height, width = np.shape(ortho_data)
            profile_ortho.update(width=width, height=height)


            map_sample_ortho = rasterio.open(path_storage_ortho + str(sample_counter) + ".tif", 'w+', **profile_ortho)
            window_ortho_write = rasterio.windows.from_bounds(*bounds_extended, map_sample_ortho.transform)
            map_sample_ortho.write(ortho_data.astype(np.uint8))

            profile_dsm = dsm_map.profile.copy()
            profile_dsm.update(transform = transform_adjusted)
            profile_dsm.update(width=width, height=height, nodata=0)

            map_sample_dsm = rasterio.open(path_storage_dsm + str(sample_counter) + ".tif", 'w+', **profile_dsm)

            dsm_data = np.where(dsm_data < 0, 0, dsm_data)

            map_sample_dsm.write_band(1, dsm_data.astype(np.float32))


            polygon_mask, _, _ = rasterio.mask.raster_geometry_mask(map_sample_ortho, [cur_poly], invert=True)
            polygon_mask = np.where(polygon_mask == 1, 255, 0)

            print("mask shape: ", np.shape(polygon_mask))


            show_matrix(polygon_mask, verbose_show_data, "avalange mask")

            #store mask
            profile_mask = dsm_map.profile.copy()
            profile_mask.update(transform=transform_adjusted)
            profile_mask.update(dtype="uint8", nodata=0, width=width, height=height)
            #print("profile: ", profile_mask)
            map_sample_mask = rasterio.open(path_storage_mask + str(sample_counter) + ".tif", 'w+', **profile_mask)

            map_sample_mask.write_band(1, polygon_mask.astype(np.uint8))

            sample_counter += 1
            gc.collect()


            """

            #try to read entire bounds
            read_successful = False
            counter = 0

            window_bounds = [bounds_extended]

            while True: #do while loop

                try:
                    a=5
                    #read alll maps
                    for bounds in window_bounds:



                    read_successful = True
                
                except OSError:
                    print("os error, uuups")
                

                if read_successful:
                    break

                window_bounds = frag_window_bounds(window_bounds) """




if __name__ == '__main__':
    main()
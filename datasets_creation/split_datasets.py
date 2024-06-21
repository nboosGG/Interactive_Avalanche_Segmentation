import os
import numpy as np

import cv2

import rasterio

import shutil


import matplotlib.pyplot as plt

def get_test_indices(n_samples, test_size, fixed_indices):
    test_indices = np.random.choice(n_samples, test_size, replace=False)
    #print("test indices: ", test_indices)

    #print("test: ",)

    for ind in fixed_indices:
        if not np.any(ind == test_indices):
            flag = True
            while(flag):
                add_ind = np.random.randint(test_size, size=1)
                if not np.any(test_indices[add_ind] == fixed_indices):
                    #print("arrived, jipeee")
                    test_indices[add_ind] = ind
                    flag = False
    test_indices = np.sort(test_indices)
    print("test indices: ", test_indices)
    print("length: ", len(test_indices))

    return test_indices

def create_directiories(folder_list):
    """check if directories exist, if not create them
    input: a list of folder paths"""
    for path in folder_list:
        if not os.path.exists(path):
            # Create the directory
            os.makedirs(path)
            print("created: ", path)



def main():

    n_samples = 346
    test_set_size = 50
    #fixed_test_indices = np.array([151, 152, 177, 227, 231, 341])
    fixed_test_indices = np.array([])

    #test_indices = get_test_indices(n_samples, test_set_size, fixed_test_indices)
    #v3 main indices
    #test_indices = np.array([3, 14,  20,  23,  24,  38,  40,  42,  48,  65,  80,  90,  95,  99, 109, 111, 113, 115,116, 117, 120, 127, 135, 143, 151, 152, 159, 162, 172, 177, 178, 179, 181, 186, 225, 227, 231, 239, 240, 241, 254, 261, 267, 271, 296, 319, 325, 333, 336, 341]).astype(int)
    #test indices for gaussian blur sets
    test_indices  = np.array([ 5, 12, 16, 29, 32, 36, 37, 42, 51, 62, 71, 91, 98, 125, 129, 130, 154, 156, 158, 164, 166, 171, 185, 187, 195, 205, 206, 217, 222, 227, 242, 249, 251, 255, 256, 261, 262, 264, 279, 287, 290, 291, 309, 313, 315, 332, 336, 338, 339, 340])
    
    #print("test_ind len: ", len(test_indices))


    #build datasets

    folder_path = "/home/boosnoel/Documents/InteractiveAvalancheSegmentation/datasets/"
    main = "ds_v3_0p5m_NoBlur"
    train_extension = "train"
    test_extension = "test"

    initial_path = folder_path + main + "/"
    assert(os.path.exists(initial_path))
    train_path = folder_path + main + "_" + train_extension + "/"
    test_path = folder_path + main + "_" + test_extension + "/"

    #create folders
    subfolders = np.array(['dsm', 'images', 'masks'])

    for subfolder in subfolders:
        create_directiories([train_path + subfolder + "/", test_path + subfolder + "/"])

    for ind in range(n_samples):
        for subfolder in subfolders:
            src_path = initial_path + subfolder + "/" + str(ind) + ".png"
            #print("path: ", src_path, os.path.exists(src_path))
            assert(os.path.exists(src_path))
            dst_path = None
            if np.any(ind == test_indices):
                #file belongs into test folder
                dst_path = test_path + subfolder + "/" + str(ind) + ".png"
            else:
                dst_path = train_path + subfolder + "/" + str(ind) + ".png"
            
            #print("src: ", src_path)
            #print("dst: ", dst_path)

            shutil.copyfile(src_path, dst_path)

            
            

    



if __name__ == '__main__':
    main()


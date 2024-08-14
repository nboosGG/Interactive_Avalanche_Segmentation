import os
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def mean_IoU(path_file):

    df = pd.read_csv(path_file)
    df = df.fillna(0)

    data = df.to_numpy()

    #go over all values, if 0, copy last value
    nrows, ncols = np.shape(data)
    for i in range(nrows):
        for j in range(ncols):
            if data[i,j] == 0:
                assert(i>=0 and data[i-1,j] >= 0)
                data[i,j] = data[i-1, j]

    assert(not np.any(data < 0))

    return np.mean(data, axis=1)


def create_IoU_per_Clicks(path_storage, path_file):

    f = pd.read_csv(path_file)
    f = f.fillna(0)
    print(f)
    data = f.to_numpy()

    max_clicks, n_images = np.shape(data)

    plt.figure(figsize=(16, 8), dpi=150) 

    for image_indx in range(n_images):
        f[:,image_indx].plot(label="image nr " + str(image_indx))
    
    plt.title('IoU per clicks')
    plt.xlabel('#clicks')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(visible=True, which='both', axis='both')


        


    

    print(path_file, type(f), np.shape(data))

def create_IoU_per_Clicks2(path_storage, path_file):

    df = pd.read_csv(path_file)
    #df = df.fillna(0)

    print(df)


    for series_name, series in df.items():
        #print(series_name)
        #print(series)

        series.plot(label=series_name)
    

    plt.title('IoU per clicks')
    plt.xlabel('#clicks')
    plt.ylabel('IoU')
    plt.legend()

    plt.show()

def all_IoU_models_performance(path):

    x = np.arange(20)+1
    y = (np.arange(7)+4)/10

    #set fixed layout positiion for every plot
    position_lib = {
        "Baseline": (0,0),
        "20cm Model": (0,1),
        "50cm Model": (1,0),
        "1m Model": (1,1),
        "2m Model": (2,0),
        "5m Model": (2,1)

    }

    #set fixed color per model
    color_lib = {
        "Baseline": 'y',
        "20cm_Model": 'b',
        "50cm_Model": 'r',
        "1m_Model": 'g',
        "2m_Model": 'c',
        "5m_Model": 'm'

    }

    fig, axis = plt.subplots(3,2, figsize=(12,12), constrained_layout=True)
    fig.suptitle("Performance of all models on all datasets", fontsize=16)

    #figure.tight_layout()
    figure_counter = 0
    
    for folder in os.listdir(path):

        foldername = os.fsdecode(folder)
        print("foldername: ", foldername)
        title_name = foldername

        if "_0p2m" in title_name:
            title_name = "20cm Model"
        elif "_0p5m" in title_name:
            title_name = "50cm Model"
        elif "_1m" in title_name:
            title_name = "1m Model"
        elif "_2m" in title_name:
            title_name = "2m Model"
        elif "_5m" in title_name:
            title_name = "5m Model"
        else:
            title_name = "Baseline"
        
        subplot_yposition, subplot_xposition = position_lib[title_name]

        for file in os.listdir(path + "/" + foldername):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"):
                print("found file: ", filename, "file: ", file)

                #create_IoU_per_Clicks2(storage_path, file_path+filename)
                series = mean_IoU(path+"/"+foldername+"/"+filename)
                print("title name: ", title_name)
                filename = filename[10:-4]
                series_name = filename
                print("faile name: ", filename)
                if "Baseline" in filename:
                    series_name = "Baseline"
                elif "_0p2m" in filename:
                    series_name = "20cm_Model"
                elif "_0p5m" in filename:
                    series_name = "50cm_Model"
                elif "_1m" in filename:
                    series_name = "1m_Model"
                elif "_2m" in filename:
                    series_name = "2m_Model"
                elif "_5m" in filename:
                    series_name = "5m_Model"
                else:
                    series_name = "unknown model"

                color = color_lib[series_name]
                
                
                print("did it work: ", subplot_yposition, subplot_xposition)
                
                print("series name: ", series_name)
                #axis[figure_counter // 2, figure_counter % 2].plot(x, series, label=series_name)
                axis[subplot_yposition, subplot_xposition].plot(x, series, label=series_name, color=color)
                #plt.plot(x, series, label=series_name)

                #handles, labels = axis[subplot_yposition, subplot_xposition].get_legend_handles_labels()
                #order = [3,0,2,1]
                #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
                


            
        

        axis[subplot_yposition, subplot_xposition].set_title("Performance of " + title_name)
        axis[subplot_yposition, subplot_xposition].set_xlabel('Nr. of Clicks []')
        axis[subplot_yposition, subplot_xposition].set_ylabel('mean IoU []')
        axis[subplot_yposition, subplot_xposition].set_xticks(x,x)
        axis[subplot_yposition, subplot_xposition].set_yticks(y,y)
        axis[subplot_yposition, subplot_xposition].grid()
        #print(dir(axis[figure_counter // 2, figure_counter % 2]))

        lines, labels = axis[subplot_yposition, subplot_xposition].get_legend_handles_labels()
        print("lines: ", lines)
        print("labels: ", labels)
        #order = [4,3,0,2,1]
        #plt.legend([lines[idx] for idx in order],[labels[idx] for idx in order])
        axis[subplot_yposition, subplot_xposition].legend(lines, labels)        
        
        
        #plt.title("Performance of " + title_name)
        #plt.xlabel('Nr. of Clicks')
        #plt.ylabel('mean IoU')
        #handles, labels = plt.gca().get_legend_handles_labels()
        #order = [4,3,0,2,1]
        #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

        #plt.xticks(x, x)
        #plt.yticks(y, y)
        

        #plt.grid()

        figure_counter += 1

    #figure.set_title('Performances of all trained model on all 5 datasets')

    plt.show()

    # //ToDO:
    # build a nice graph per sub folder (e.g. per trained model, with model name as title and all datasets with correct name)
    # build matplotlib graph with multiple graphs that contain all these 6 graphs. (3,2) shape of graphs


def create_IoU_graph(path):

    x = np.arange(20)+1
    y = (np.arange(6)+5)/10

    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 

            #create_IoU_per_Clicks2(storage_path, file_path+filename)
            series = mean_IoU(path+filename)
            filename = filename[:-4]
            series_name = filename[10:]
            print("series name: ", series_name)
            if "Baseline" in filename:
                series_name = "Baseline"
            elif "_0p2m" in filename:
                series_name = "20cm_Model"
            elif "_0p5m" in filename:
                series_name = "50cm_Model"
            elif "_1m" in filename:
                series_name = "1m_Model"
            elif "_2m" in filename:
                series_name = "2m_Model"
            elif "_5m" in filename:
                series_name = "5m_Model"
            else:
                series_name = "unknown model"
            
            print("series name: ", series_name)

            plt.plot(x, series, label=series_name)


    plt.title('Model Performance on 0.5m Test Dataset')
    plt.xlabel('Nr. of Clicks')
    plt.ylabel('mean IoU')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [4,3,0,2,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    plt.xticks(x, x)
    plt.yticks(y, y)
    

    plt.grid()

    plt.show()

def create_IoU_graph_blurs(path):
    plt.style.use('_classic_test_patch')
    print(plt.style.available)
    #assert(False)

    x = np.arange(20)+1
    y = (np.arange(7)+4)/10

    #skip_list = np.array(["DS_v3_0p5m_NoDSM_lpf_r50_test_0p2m_NoBRS_20.csv", "DS_v3_0p5m_NoDSM_gB_sig5_test_0p2m_NoBRS_20.csv",
    #                      "DS_v3_0p5m_NoDSM_mB_s9_test_0p2m_NoBRS_20.csv"])
    
    #skip_list = np.array(["DS_v3_0p5m_NoDSM_lpf_r50_test_1m_NoBRS_20.csv", "DS_v3_0p5m_NoDSM_gB_sig5_test_1m_NoBRS_20.csv",
    #                      "DS_v3_0p5m_NoDSM_mB_s9_test_1m_NoBRS_20.csv"])

    skip_list = np.array(["DS_v3_0p5m_NoDSM_lpf_r50_test_0p5m_NoBRS_20.csv", "DS_v3_0p5m_NoDSM_gB_sig5_test_NoBRS_20.csv",
                          "DS_v3_0p5m_NoDSM_mB_s9_test_NoBRS_20.csv"])

    #set fixed color per model
    color_lib = {
        "No Blur": 'y',
        "Gaussian Blur, sig=0.5": 'forestgreen',
        "Gaussian Blur, sig=1": 'lime',
        "Median Blur, size=3": 'navy',
        "Median Blur, size=5": 'blue',
        "LowPassFilter, radius=100p": 'lightcoral',
        "LowPassFilter, radius=200p": 'red',
        "LowPassFilter, radius=400p": 'crimson',
        "wtf": 'black'

    }

    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            print("-------------------------------------------")
            print("filename: ", filename)

            if filename in skip_list:
                print("found name: ", filename)
                continue
            else:
                print("else name: ", filename)

            #create_IoU_per_Clicks2(storage_path, file_path+filename)
            series = mean_IoU(path+filename)
            filename = filename[:-4]
            series_name = filename[10:]
            print("series name: ", series_name)
            if "_gB_sig0p5" in filename:
                series_name = "Gaussian Blur, sig=0.5"
            elif "_gB_sig1" in filename:
                series_name = "Gaussian Blur, sig=1"
            elif "_mB_s3" in filename:
                series_name = "Median Blur, size=3"
            elif "_mB_s5" in filename:
                series_name = "Median Blur, size=5"
            elif "_lpf_r100" in filename:
                series_name = "LowPassFilter, radius=100p"
            elif "_lpf_r200" in filename:
                series_name = "LowPassFilter, radius=200p"
            elif "_lpf_r400" in filename:
                series_name = "LowPassFilter, radius=400p"
            elif "_NoBlur" in filename:
                series_name = "No Blur"
            else:
                print("unkown model name: ", series_name)
                series_name = "wtf"
            
            print("series name: ", series_name)
            color = color_lib[series_name]

            plt.plot(x, series, label=series_name, color=color)


    plt.title('Models Performance on 50cm Test Dataset')
    plt.xlabel('Nr. of Clicks')
    plt.ylabel('mean IoU')
    handles, labels = plt.gca().get_legend_handles_labels()
    print("# lines ", len(handles))
    order = [5,4,0,6,1,7,3,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')

    plt.xticks(x, x)
    plt.yticks(y, y)
    

    plt.grid()

    plt.show()


def create_mean_IoU_graph(folder_path):

    #set fixed color per model
    color_lib = {
        "Baseline": 'y',
        "20cm_Model": 'b',
        "50cm_Model": 'r',
        "1m_Model": 'g',
        "2m_Model": 'c',
        "5m_Model": 'm'

    }

    x = np.arange(20)+1
    #y = (np.arange(7)+4)/10
    y = np.round(np.arange(10)/10 + 0.1,decimals=1)

    for directory in os.listdir(folder_path):
        print("directory path: ", directory)
        data = []
        for file in os.listdir(folder_path+"/"+directory):
            filename = os.fsdecode(file)
            print("file name: ", filename)
            
            if filename.endswith(".csv"):
                series = mean_IoU(folder_path+"/"+directory+"/"+filename)
                data.append(series)
        data = np.array(data)
        print("data shape: ", np.shape(data))
        print(data)
        mean_data = np.mean(data, axis=0)
        series_name = directory
        print("series name: ", series_name)
        
        if "baseline" in directory:
            series_name = "Baseline"
        elif "_0p2m" in directory:
            series_name = "20cm_Model"
        elif "_0p5m" in directory:
            series_name = "50cm_Model"
        elif "_1m" in directory:
            series_name = "1m_Model"
        elif "_2m" in directory:
            series_name = "2m_Model"
        elif "_5m" in directory:
            series_name = "5m_Model"
        else:
            series_name = "unknown model"
        
        print("series name: ", series_name)

        color = color_lib[series_name]

        plt.plot(x, mean_data, label=series_name, color=color)
            
    plt.title('Mean Model Performance over all Image Resolutions')
    plt.xlabel('Nr. of Clicks []')
    plt.ylabel('mean IoU []')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [4,1,2,0,5,3]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    
    plt.xticks(x, x)
    plt.yticks(y, y)


    plt.grid()

    plt.show()

def create_aug_flip_graph(path):
    plt.style.use('_classic_test_patch')
    print(plt.style.available)
    #assert(False)

    x = np.arange(20)+1
    y = (np.arange(7)+4)/10


    #set fixed color per model
    color_lib = {
        "Randomly Flip Images": 'forestgreen',
        "No Flipping": 'red',
        "wtf": 'black'
    }

    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            print("-------------------------------------------")
            print("filename: ", filename)

            #create_IoU_per_Clicks2(storage_path, file_path+filename)
            series = mean_IoU(path+filename)
            filename = filename[:-4]
            series_name = filename[10:]
            print("series name: ", series_name)
            if "_NoBlur_test_NoBRS_20_Aug_Flip" in filename:
                series_name = "Randomly Flip Images"
            elif "_NoBlur_test_NoBRS_20_Aug_NoFlip" in filename:
                series_name = "No Flipping"
            else:
                print("unkown model name: ", series_name)
                series_name = "wtf"
            
            print("series name: ", series_name)
            color = color_lib[series_name]

            plt.plot(x, series, label=series_name, color=color)


    plt.title('Augmentation Performance on Test Dataset')
    plt.xlabel('Nr. of Clicks []')
    plt.ylabel('mean IoU []')
    handles, labels = plt.gca().get_legend_handles_labels()
    print("# lines ", len(handles))
    order = [0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')

    plt.xticks(x, x)
    plt.yticks(y, y)
    

    plt.grid()

    plt.show()

def create_aug_rotation_graph(path):

    plt.style.use('_classic_test_patch')
    print(plt.style.available)
    #assert(False)

    x = np.arange(20)+1
    y = (np.arange(7)+4)/10


    #set fixed color per model
    color_lib = {
        "No rotation": 'y',
        "15° rotation": 'forestgreen',
        "25° rotation": 'lime',
        "50° rotation": 'navy',
        "75° rotation": 'blue',
        "90° rotation": 'lightcoral',
        "180° rotation": 'red',
        "LowPassFilter, radius=400p": 'crimson',
        "wtf": 'black'

    }

    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            print("-------------------------------------------")
            print("filename: ", filename)

            #create_IoU_per_Clicks2(storage_path, file_path+filename)
            series = mean_IoU(path+filename)
            filename = filename[:-4]
            series_name = filename[10:]
            print("series name: ", series_name)
            #continue
            if "Rot75" in filename:
                series_name = "75° rotation"
            elif "Rot50" in filename:
                series_name = "50° rotation"
            elif "Rot25" in filename:
                series_name = "25° rotation"
            elif "Rot15" in filename:
                series_name = "15° rotation"
            elif "Rot90" in filename:
                series_name = "90° rotation"
            elif "Rot180" in filename:
                series_name = "180° rotation"
            elif "NoRot" in filename:
                series_name = "No rotation"
            else:
                print("unkown model name: ", series_name)
                series_name = "wtf"
            
            print("series name: ", series_name)
            color = color_lib[series_name]

            plt.plot(x, series, label=series_name, color=color)


    plt.title('Augmentation Performance on Test Dataset')
    plt.xlabel('Nr. of Clicks []')
    plt.ylabel('mean IoU []')
    handles, labels = plt.gca().get_legend_handles_labels()
    print("# lines ", len(handles))
    order = [1,4,3,2,0,5,6]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')

    plt.xticks(x, x)
    plt.yticks(y, y)
    

    plt.grid()

    plt.show()

def create_DSM_test1_graph(path):
    plt.style.use('_classic_test_patch')
    print(plt.style.available)
    #assert(False)

    x = np.arange(20)+1
    y = (np.arange(7)+4)/10


    #set fixed color per model
    color_lib = {
        "DSM per Sample normlaization": 'lightcoral',
        "DSM overall normalization": 'red',
        "DSM Hillshade": 'navy',
        "Baseline (No DSM)": 'lime',
        "wtf": 'black'
    }

    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            print("-------------------------------------------")
            print("filename: ", filename)

            #create_IoU_per_Clicks2(storage_path, file_path+filename)
            series = mean_IoU(path+filename)
            filename = filename[:-4]
            series_name = filename[10:]
            print("series name: ", series_name)
            #continue
            if "DSM_only_test" in filename:
                series_name = "DSM per Sample normlaization"
            elif "DSM_only_max4k" in filename:
                series_name = "DSM overall normalization"
            elif "hillshade" in filename:
                series_name = "DSM Hillshade"
            elif "NoBlur" in filename:
                series_name = "Baseline (No DSM)"
            else:
                print("unkown model name: ", series_name)
                series_name = "wtf"
            
            print("series name: ", series_name)
            color = color_lib[series_name]

            plt.plot(x, series, label=series_name, color=color)

    plt.title('DSM Test 1')
    plt.xlabel('Nr. of Clicks []')
    plt.ylabel('Mean IoU []')
    handles, labels = plt.gca().get_legend_handles_labels()
    print("# lines ", len(handles))
    order = [1,0,2,3]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')

    plt.xticks(x, x)
    plt.yticks(y, y)
    

    plt.grid()

    plt.show()

def create_DSM_test2_graph(path):
    plt.style.use('_classic_test_patch')
    print(plt.style.available)
    #assert(False)

    x = np.arange(20)+1
    y = (np.arange(7)+4)/10


    #set fixed color per model
    color_lib = {
        "RGH": 'y',
        "RHB": 'red',
        "HGB": 'navy',
        "Baseline (RGB)": 'lime',
        "wtf": 'black'
    }

    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            print("-------------------------------------------")
            print("filename: ", filename)

            #create_IoU_per_Clicks2(storage_path, file_path+filename)
            series = mean_IoU(path+filename)
            filename = filename[:-4]
            series_name = filename[10:]
            print("series name: ", series_name)
            #continue
            if "RGH" in filename:
                series_name = "RGH"
            elif "HGB" in filename:
                series_name = "HGB"
            elif "RHB" in filename:
                series_name = "RHB"
            elif "RGB" in filename:
                series_name = "Baseline (RGB)"
            else:
                print("unkown model name: ", series_name)
                series_name = "wtf"
            
            print("series name: ", series_name)
            color = color_lib[series_name]

            plt.plot(x, series, label=series_name, color=color)

    plt.title('DSM Test 2')
    plt.xlabel('Nr. of Clicks []')
    plt.ylabel('Mean IoU []')
    handles, labels = plt.gca().get_legend_handles_labels()
    print("# lines ", len(handles))
    order = [1,0,2,3]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')

    plt.xticks(x, x)
    plt.yticks(y, y)
    

    plt.grid()

    plt.show()

def main():

    
    file_path = "/home/boosnoel/Documents/data/graphs/ds_v3/iou_data/ds_v3_0p2m_test_blurs/"

    #create_IoU_graph_blurs(file_path)

    #file_path = "/home/boosnoel/Documents/InteractiveAvalancheSegmentation/experiments/evaluation_logs/others/Mv3_0p5m_wDSMt1_0522_095/ious/"
    #create_IoU_graph(file_path)

    file_path2 = "/home/boosnoel/Documents/data/graphs/ds_v3/iou_data/mean/"

    #create_mean_IoU_graph(file_path2)

    file_path3 = file_path2
    #all_IoU_models_performance(file_path3)


    file_path4 = "/home/boosnoel/Documents/data/graphs/ds_v3/iou_data/Aug_Flip/"
    #create_aug_flip_graph(file_path4)

    file_path5 = "/home/boosnoel/Documents/data/graphs/ds_v3/iou_data/Aug_Rotation/"
    #create_aug_rotation_graph(file_path5)


    file_path6 = "/home/boosnoel/Documents/data/graphs/ds_v3/iou_data/dsm_tests/test1/"
    #create_DSM_test1_graph(file_path6)

    file_path7 = "/home/boosnoel/Documents/data/graphs/ds_v3/iou_data/dsm_tests/test2/"
    create_DSM_test2_graph(file_path7)



if __name__ == '__main__':
    main()
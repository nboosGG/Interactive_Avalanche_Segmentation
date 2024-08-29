import os
import pandas as pd
import numpy as np

def compute_mean_IoU(data, target):
    x1 = 0
    x2 = 1
    if data[x1] >= target:
        
        return 1 - (target - data[0]) / (-data[0])
    
    while(data[x2] < target):
        x1 += 1
        x2 += 1
        if x2 == 20:
            return -1
        
    return 1 + x2 - (target - data[x2]) / (data[x1] -  data[x2])


def compute_nr_of_failed_pred(data, target_score):
    return np.sum(data[19,:] < target_score)

path = "/home/boosnoel/Documents/data/graphs/ds_v3/iou_data/ds_v3_0p2m_test_blurs/"

for file in os.listdir(path):
    filename = os.fsdecode(file)
    print("filename: ", filename)
    if filename.endswith(".csv"): 
        df = pd.read_csv(path + filename)
        df = df.fillna(0)

        data = df.to_numpy()

        print("data: ", np.shape(data))

        avg_data = np.mean(data, axis=1)
        print("avg data: ", np.shape(avg_data))
        print(avg_data)
        print("mean IoU@80: ", compute_mean_IoU(avg_data, 0.8))
        print("mean IoU@90: ", compute_mean_IoU(avg_data, 0.9))
        print(">=20@80: ,", compute_nr_of_failed_pred(data, 0.8))
        print(">=20@90: ,", compute_nr_of_failed_pred(data, 0.9))
        print("mIoU@1: ", avg_data[0])
        print("mIoU@2: ", avg_data[1])

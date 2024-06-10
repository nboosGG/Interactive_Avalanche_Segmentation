import os
import csv
import numpy as np

import pandas as pd









def main():
    a=5

    path = "/home/boosnoel/Documents/InteractiveAvalancheSegmentation/experiments/evaluation_logs/others/082_epo090/ious/Small_dataset_NoBRS_20.csv"


    with open(path,'r') as dest_f:
        data_iter = csv.reader(dest_f,
                           delimiter = ",",
                           quotechar = '"',
                           header=False)
        data = [data for data in data_iter]
    
    data_array = np.asarray(data, dtype = np.float32)
    #data_array = data_array[1:,:].astype(np.float32)
    print("data shape: ", np.shape(data_array))
    print(data_array)













if __name__ == '__main__':
    main()
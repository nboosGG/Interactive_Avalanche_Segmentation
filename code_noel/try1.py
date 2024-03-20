import numpy as np



arr = np.arange(20).reshape((4,5))
print("arr: ", arr)

window = np.array([1,2,2,2]) #col_off, row_off, width, height

sub_arr = arr[window[1]:window[1]+window[3], window[0]:window[0]+window[2]]
print("subarr: ", sub_arr)
print(arr[2])
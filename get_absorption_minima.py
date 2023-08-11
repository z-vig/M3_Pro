import numpy as np

def find_min(wvl_array:np.ndarray,rfl_array:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    diff_list = []
    for count,i in enumerate(rfl_array):
        if count>0:
            diff_list.append(i-rfl_array[count-1])
    
    min_list = []
    for count,i in enumerate(diff_list):
        if count>0 and diff_list[count-1]*i<0 and diff_list[count-1]<0:
            min_list.append(diff_list.index(i))
    
    return [list(wvl_array)[i]/1000 for i in min_list]
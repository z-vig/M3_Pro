# -*- coding: utf-8 -*-
"""
Script for averaging spectral signal
"""

import numpy as np
import matplotlib.pyplot as plt
import spectral as sp

def spec_avg(spec_list,wvl_list,box_size):
    avg_array = np.zeros((box_size,2))
    avg_rfl = []
    std_rfl = []
    avg_wvl = []
    n = 1
    for spec,wvl in zip(spec_list,wvl_list):
        if n%box_size != 0:
            avg_array[n-1] = (spec,wvl)
            n+=1
        elif n%box_size == 0:
            avg_array[n-1] = (spec,wvl)
            avg_rfl.append(np.average(avg_array[:,0]))
            std_rfl.append(np.std(avg_array[:,0]))
            avg_wvl.append(np.average(avg_array[:,1]))
            avg_array = np.zeros((box_size,2))
            n=1
    
    rfl_last = []
    wvl_last = []
    if np.all(np.where(avg_array==0,True,False)):
        return avg_rfl,std_rfl,avg_wvl
    
    for rfl,wvl in zip(avg_array[:,0],avg_array[:,1]):
        if rfl != 0:
            rfl_last.append(rfl)
            wvl_last.append(wvl)
            
    avg_rfl.append(np.average(rfl_last))
    std_rfl.append(np.std(rfl_last))
    avg_wvl.append(np.average(wvl_last))
    
    return avg_rfl,std_rfl,avg_wvl

if __name__ == "__main__":
    hdr = sp.envi.open(r"D:/Data/20230209T095534013597/extracted_files/hdr_files/m3g20090417t193320_v01_rfl/m3g20090417t193320_v01_rfl.hdr")
    wvl,rfl = hdr.bands.centers[21:72],hdr.read_pixel(91,100)[21:72]
    y_avg,y_std,x_avg = spec_avg(rfl,wvl,3)
    plt.plot(wvl,rfl)
    plt.plot(x_avg,y_avg)
    plt.show()
# -*- coding: utf-8 -*-
"""
Cubic Spline Interpolation for an Entire Image
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from spectrum_averaging import moving_avg
from spectrum_averaging import spec_avg
from spectrum_averaging import nd_avg
import spectral as sp
import math
import time
import tifffile as tf
import os

def spline_fit(inputImage:np.ndarray,avg_num:int,wvl:np.ndarray):
    averageBands = np.zeros((*inputImage.shape[:2],0))
    averageWvl = []
    for band_num in range(inputImage.shape[2]):
        wavelength = wvl[band_num]
        if band_num%avg_num==0:
            wvl_list = []
            avgBand_Array = np.zeros((*inputImage.shape[:2],0))
            avgBand_Array = np.concatenate((avgBand_Array,np.expand_dims(inputImage[:,:,band_num],2)),axis=2)
            wvl_list.append(wavelength)
        elif band_num%avg_num<avg_num-1:
            avgBand_Array = np.concatenate((avgBand_Array,np.expand_dims(inputImage[:,:,band_num],2)),axis=2)
            wvl_list.append(wavelength)
        elif band_num%avg_num==avg_num-1:
            avgBand_Array = np.concatenate((avgBand_Array,np.expand_dims(inputImage[:,:,band_num],2)),axis=2)
            averageBands = np.concatenate((averageBands,np.expand_dims(np.average(avgBand_Array,axis=2),2)),axis=2)
            wvl_list.append(wavelength)
            averageWvl.append(np.average(wvl_list))
    
    averageBands = np.concatenate((averageBands,np.expand_dims(np.average(avgBand_Array,axis=2),2)),axis=2)
    averageWvl.append(np.average(wvl_list))
    averageWvl_index = []
    for av in averageWvl:
        diff_list = [np.abs(av-wv) for wv in wvl]
        averageWvl_index.append(diff_list.index(min(diff_list)))

    #print (averageBands.shape)
    #print (wvl,averageWvl)
    def spline_func(averaged_spectrum_pixel):
        f = CubicSpline(averageWvl_index,averaged_spectrum_pixel)
        resample_x = np.linspace(0,len(wvl),inputImage.shape[2])
        return f(resample_x)
    
    print ('Applying spline along axis 2...')
    smoothArray = np.apply_along_axis(spline_func,2,averageBands)

    return averageWvl,averageBands,smoothArray
if __name__ == "__main__":
    start = time.time()
    
    hdr = sp.envi.open(f"D:/Data/L2_Data/extracted_files/hdr_files/m3g20090417t193320_v01_rfl/m3g20090417t193320_v01_rfl.hdr")
    
    bandCenters = np.array(hdr.bands.centers)
    allowedIndices = np.where((bandCenters>900)&(bandCenters<2600))[0]
    minIndex,maxIndex = allowedIndices.min(),allowedIndices.max()
    
    allowedWvl = bandCenters[allowedIndices]
    signalArray = hdr.read_bands(allowedIndices)
    avgWvl,avgImg,splineImg, = splineFit(signalArray,5,allowedWvl)

    fig = plt.figure()
    plt.plot(allowedWvl,signalArray[91,100,:],label='Original')
    plt.plot(avgWvl,avgImg[91,100,:],label='Average')
    plt.plot(allowedWvl,splineImg[91,100,:],label='Spline')
    plt.legend()
    plt.show()

    end = time.time()
    runtime = end-start
    if runtime < 1:
        print(f'Program Executed in {runtime*10**3:.3f} milliseconds')
    elif runtime < 60 and runtime > 1:
        print(f'Program Executed in {runtime:.3f} seconds')
    else:
        print(f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')
    





    
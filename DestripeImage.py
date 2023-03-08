# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:59:01 2023

@author: zacha
"""

import numpy as np
import os
import pandas as pd
from HySpec_Image_Processing import HDR_Image
from fancy_spec_plot import fancy_spec_plot
import matplotlib.pyplot as plt
from spec_average import spec_avg
from scipy import interpolate as interp
import numpy.random as r
from scipy import signal
import spectral as sp
import tifffile as tf

hdr = sp.envi.open(r'D:\Data\20230209T095534013597\extracted_files\hdr_files\m3g20090417t193320_v01_rfl\m3g20090417t193320_v01_rfl.hdr')
bands = np.array(hdr.bands.centers)
validBandIndexes = np.where((bands>1000)&(bands<2500))
bands = bands[validBandIndexes]

img = hdr.read_bands(range(np.min(validBandIndexes),np.max(validBandIndexes)))
dstripImage = np.zeros(img.shape)

for band in range(img.shape[2]):
    imArray = img[:,:,band]
    newArray = np.zeros(imArray.shape)
    columnIndex = np.arange(1,imArray.shape[1]+1,1)
    
    for i in range(imArray.shape[0]):
        print (f'\rStatus: {i+1} out of 1017 ... {band+1} out of 51',end='')
        avg,std,wvl = spec_avg(imArray[i,:],columnIndex,box_size=5)
        f = interp.CubicSpline(wvl,avg)
        ferr = interp.CubicSpline(wvl,std)
        x = np.linspace(0,imArray.shape[1],304)
        newArray[i,:] = f(x)
    
    dstripImage[:,:,band] = newArray
    

tf.imsave(r"D:\Data\Destriped Images\0417.tif",dstripImage,photometric='rgb')
    
fig = plt.figure()
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

ax1.imshow(imArray,cmap='gray')
ax2.imshow(newArray,cmap='gray')
    



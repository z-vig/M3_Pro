# -*- coding: utf-8 -*-
"""
Script for Destriping M3 images using convolution
"""

import numpy as np
import os
import pandas as pd
from spec_plotting import fancy_spec_plot
import matplotlib.pyplot as plt
from spectrum_averaging import spec_avg
from scipy import interpolate as interp
import numpy.random as rand
from scipy import signal
import spectral as sp
import tifffile as tf
from spec_plotting import plot_numpy_images
from copy import copy

def destripe(image,box_size,**kwargs):
    defaultKwargs = {"plotImg":False}
    kwargs = {**defaultKwargs,**kwargs}

    destripedImage = np.zeros(image.shape)
    n=0
    for band in range(image.shape[2]):
        print (f'\rBand {band+1} of {image.shape[2]} processed. ({(band+1)/image.shape[2]:.0%})',end='\r')
        for row in range(image.shape[0]):
            xCoords = np.arange(image.shape[1])
            yAvg,std,xAvg = spec_avg(image[row,:,band],xCoords,box_size)
            f = interp.CubicSpline(xAvg,yAvg)
            xtest = np.linspace(0,image.shape[1],304)
            destripedImage[row,:,band] = f(xtest)
            if kwargs.get('plotImg') == True:
                if n<3:
                    fig,ax = plt.subplots(1,1)
                    ax.plot(xCoords,image[row,:,0])
                    ax.plot(xAvg,yAvg)
                    ax.plot(xtest,f(xtest))
                else:
                    pass
                fig,ax = plt.subplots(1,1)
                ax.imshow(destripedImage[:,:,0])

        n+=1
    
    print ('\n')
    
    # sharp = np.array(([0,-1,0],[-1,10,-1],[0,-1,0]))
    # destripedImage_sharp = signal.convolve2d(destripedImage[:,:,0],sharp)

    return destripedImage

if __name__ == "__main__":
    hdr = sp.envi.open(r'D:\Data/20230209T095534013597/extracted_files/hdr_files/m3g20090417t193320_v01_rfl/m3g20090417t193320_v01_rfl.hdr')
    bandCenters = hdr.bands.centers
    bandCenters = np.array(bandCenters)

    allowedIndices = np.where((bandCenters>900)&(bandCenters<2600))[0]
    allowedWvl = bandCenters[allowedIndices]

    image = hdr.read_bands(allowedIndices)

    # im1,im2,im3,im4 = destripe(image,3),destripe(image,5),destripe(image,7),destripe(image,10)
    img = destripe(image,7,plotImg=True)

    plot_numpy_images(img[:,:,10],titles=['7'],figtitle='Destriping Images')
    plt.show()




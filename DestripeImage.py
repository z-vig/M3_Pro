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



def destripe(image):
    for row in range(image.shape[0]):
        print (len(image[row,:,:]))

hdr = sp.envi.open(r'D:\Data/20230209T095534013597/extracted_files/hdr_files/m3g20090417t193320_v01_rfl/m3g20090417t193320_v01_rfl.hdr')
bandCenters = hdr.bands.centers
bandCenters = np.array(bandCenters)

allowedIndices = np.where((bandCenters>900)&(bandCenters<2600))[0]
allowedWvl = bandCenters[allowedIndices]

image = hdr.read_bands(allowedIndices)

destripe(image)




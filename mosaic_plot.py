# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:55:01 2023

@author: zacha
"""

from sympy import *
from spec_average import spec_avg
from scipy import interpolate as interp
import tifffile as tf
from tkinter.filedialog import askopenfile as askfile
from tkinter.filedialog import askdirectory as askdir
from tkinter import Tk
import zipfile
import os
from celluloid import Camera
import spectral as sp
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
import random
import pandas as pd
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import time
from spec_plotting import fancy_spec_plot
#from HySpec_Image_Processing import all_stats


# Timing the program
start = time.time()


#print (all_stats)
image = tf.imread('D:/Data/Image_Mosaic_Normalized.tif')
plt.imshow(image[:,:,0],cmap='gray',vmin=np.min(image[:,:,0])*0.1,vmax=np.max(image[:,:,0])*0.1,aspect='auto')


end = time.time()
runtime = end-start
if runtime < 1:
    print(f'Program Executed in {runtime*10**3:.3f} milliseconds')
elif runtime < 60 and runtime > 1:
    print(f'Program Executed in {runtime:.3f} seconds')
elif runtime > 60:
    print(
        f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')
# -*- coding: utf-8 -*-
"""
test.py
"""


# def multiply(x,y):
#     x = float(x)
#     y = float(y)
#     return x*y


# if __name__ == "__main__":
#     x_,y_ = input('X: '),input('Y: ')
#     product = multiply(x_,y_)
#     print(f'The Answer is {product}')

# import spectral as sp
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import scipy.stats as ss
# import scipy.signal as sig
#from tkinter.filedialog import askdirectory as askdir

# =============================================================================
# hdr = sp.envi.open(r"D:\Data\20230209T095534013597\extracted_files\hdr_files\m3g20090417t193320_v01_rfl\m3g20090417t193320_v01_rfl.hdr")
# 
# band = hdr.read_band(35)
# norm_band = (band-np.average(band))/np.std(band)
# band_01 = (band-np.min(band))
# band_01 = 256*band_01/np.max(band_01)
# 
# 
# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)
# 
# ax1.imshow(band,cmap='gray')
# ax2.imshow(band_01,cmap='gray')
# 
# =============================================================================

# =============================================================================
# class MyClass:
#     def __init__(self,value):
#         self._value = value
#         
#     @property 
#     def value(self):
#         print ('Getting Value')
#         return self._value
#     
#     @value.setter 
#     def value(self,value):
#         print ('Setting value to ' + value)
#         self._value = value
#     
#     @value.deleter 
#     def value(self):
#         print ('Deleting Value')
#         del self._value
#         
#     def printvalue(self):
#         print (self.value)
# =============================================================================

# =============================================================================
# def func(**kwargs):
#     defaultKwargs = {'arg1':3,'arg2':4,'arg3':5}
#     kwargs = {**defaultKwargs,**kwargs}
#     print (kwargs["arg1"])
#     print (kwargs["arg2"])
#     print (kwargs["arg3"])
#     
# func(arg2=27)
# =============================================================================

# =============================================================================
# x = np.linspace(0,11,20)
# y = sig.square(x,0.5)
# 
# def moving_avg(xdata,ydata,length):
#     def norm(x,center):
#         return math.e**((-(x-center)**2)/2)
#     if length%2 == 0:
#         length += 1
#     moving_avg_len = length
#     
#     moving_avg = np.arange(0,moving_avg_len)
#     moving_avg = norm(moving_avg,np.median(moving_avg))
#     moving_avg = moving_avg/moving_avg.sum()
#     print (f'Convolution Array: {moving_avg} of length {length}')
#     
#     #moving_avg = [1/moving_avg_len]*moving_avg_len
#     
#     
#     #moving_avg = [0.1,0.2,0.4,0.2,0.1]
#     
#     
#     fig,ax = plt.subplots(1,1)
#     ax.plot(xdata,ydata,label='original')
#     ex_val = int(len(moving_avg)//2)
#     
#     conv = np.convolve(moving_avg,ydata)
#     conv = conv[ex_val:len(conv)-ex_val]
#     
#     ax.plot(xdata,conv,label='moving avg')
#     
#     ax.legend()
# 
# for i in np.arange(1,20,2):
#     moving_avg(x,y,i)
# ============================================================================
import numpy as np
from get_pixel_mosaic import create_arrays

if 'shadow' not in locals():
    shadow,imgStats,mosaic,mosaicStats = create_arrays(r'/run/media/zvig/My Passport/Data')
elif 'shadow' in locals():
    print ('Already Done')




# =============================================================================
# path = askdir()
# print (path)
# =============================================================================

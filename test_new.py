# -*- coding: utf-8 -*-
"""
test.py
"""

import spectral as sp
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]
# =============================================================================
# def create_scale(x):
#     def scaled_num(y):
#         return y*x
#     
#     return scaled_num
# 
# scale_10 = create_scale(10)
# 
# for i in range(10):
#     print (scale_10(i))
# =============================================================================

# =============================================================================
# 
# if __name__ == "__main__":
#     x_,y_ = input('X: '),input('Y: ')
#     product = multiply(x_,y_)
#     print(f'The Answer is {product}')
# =============================================================================

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

class MyClass:
    def __init__(self,path):
        self.hdr = sp.envi.open(path)
        data_str = self.hdr.fid.name[find(self.hdr.fid.name, '\\')[-1]+1:len(self.hdr.fid.name)-4]
        date = f"{data_str[3:7]}/{data_str[7:9]}/{data_str[9:11]}"
        date = date.replace('/','-')
        time = f"{data_str[12:14]}:{data_str[14:16]}:{data_str[16:18]}"
        date_time = date+'_'+time
        self._datetime = date_time
    
    @property
    def datetime(self):
        print ('Getting Date/Time...')
        return self._datetime
        
        
        
img1 = MyClass(r"D:\Data\20230209T095534013597\extracted_files\hdr_files\m3g20090417t193320_v01_rfl\m3g20090417t193320_v01_rfl.hdr")

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



# =============================================================================
# path = askdir()
# print (path)
# =============================================================================

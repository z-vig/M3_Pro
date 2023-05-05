# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:51:44 2023

@author: zacha
"""
import matplotlib.pyplot as plt
import numpy as np
import math

def moving_avg(xdata,ydata,length,**kwargs):
    defaultKwargs = {'showPlot':False,'cType':'Gauss'}
    kwargs = {**defaultKwargs,**kwargs}
    
    def norm(x,center):
        return math.e**((-(x-center)**2)/(2*3**2))
    
    if length%2 == 0:
        length += 1
        
    moving_avg_len = length
    
    if kwargs.get('cType') == 'Gauss':
        moving_avg = np.arange(0,moving_avg_len)
        moving_avg = norm(moving_avg,np.median(moving_avg))
        moving_avg = moving_avg/moving_avg.sum()
        #print (f'Convolution Array: {moving_avg} of length {length}')
    elif kwargs.get('cType') == 'Equal':
        moving_avg = [1/moving_avg_len]*moving_avg_len
    else:
        raise ValueError(f'cType cannot be {kwargs.get("cType")}')
    

    conv = np.convolve(moving_avg,ydata,mode='valid')
    buffer = int((len(xdata)-len(conv))/2)
    xdata_valid = xdata[buffer:len(xdata)-buffer]

    if kwargs.get('showPlot') == True:
        fig,ax = plt.subplots(1,1)
        ax.plot(xdata,ydata,label='original')
        ax.plot(xdata_valid,conv,label='moving avg')
        ax.legend()
        
        return xdata_valid,conv
        
    elif kwargs.get('showPlot') == False:
        return xdata_valid,conv
    
    else:
        raise ValueError(f'cType cannot be {kwargs.get("cType")}')
    
    
    
    
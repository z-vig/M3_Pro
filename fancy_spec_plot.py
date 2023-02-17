# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:22:34 2023

@author: zacha
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np


def fancy_spec_plot(x,y,**kwargs):
    defaultKwargs = {'std':np.zeros(len(y)),'ylabel':False,'xlabel':False,'title':'','minorticks':True,'wvl_lim':(1000,2500)}
    kwargs = {**defaultKwargs,**kwargs}
    error = "Keyword Error in"
    
    ##Setting up figure
    fig,ax = plt.subplots(1,1)
    
    ##Plotting Line and/or Standard Deviation                    
    ax.fill_between(x,y-kwargs.get('std'),y+kwargs.get('std'),color='k',alpha=0.3)
    ax.plot(x,y,color='red',linewidth=0.8)
    if type(kwargs.get('std')) != np.ndarray:
        raise Exception(f'{error} std')
    
    ##Setting wavelength range
    ax.set_xlim([*kwargs.get('wvl_lim')])
    if type(kwargs.get('wvl_lim')) != tuple:
        raise Exception(f'{error} wvl_lim')
    
    if kwargs.get('minorticks') == True:
        ax.set_xticks(np.linspace(1000,2500,4),fontname="Source Code Pro")
        ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
        ax.set_yticks(np.linspace(0.05,0.35,7),fontname="Source Code Pro")
        ax.yaxis.set_minor_locator(tck.MultipleLocator(0.01))
    elif kwargs.get('minorticks') == False:
        pass
    else:
        raise Exception(f'{error} minorticks')
    
    if type(kwargs.get('ylabel')) == str:
        ax.set_ylabel(kwargs.get('ylabel'),fontname="Times New Roman",fontsize=12)
    elif kwargs.get('ylabel') == False:
        pass
    else:
        raise Exception(f'{error} ylabel')
        
    if type(kwargs.get('xlabel')) == str:
        ax.set_xlabel(kwargs.get('xlabel'),fontname="Times New Roman",fontsize=12)
    elif kwargs.get('xlabel') == False:
        pass
    else:
        raise Exception(f'{error} xlabel')
        
    ax.set_title(kwargs.get('title'),fontname="Times New Roman",fontsize=14)
    if type(kwargs.get('title')) != str:
        raise Exception(f'{error} title')
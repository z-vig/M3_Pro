# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:22:34 2023

@author: zacha
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np


def fancy_spec_plot(fig,ax,x,y,**kwargs):
    defaultKwargs = {'std':np.zeros(len(y)),'ylabel':False,'xlabel':False,'title':'','minorticks':True,'wvl_lim':(1000,2500),
                     'line_style':'solid','line_color':'red','std_color':'black','label':'plot'}
    kwargs = {**defaultKwargs,**kwargs}
    error = "Keyword Error in"
    
    ##Setting up figure
    fig,ax = fig,ax
    
    ##Plotting Line and/or Standard Deviation                    
    ax.fill_between(x,y-kwargs.get('std'),y+kwargs.get('std'),color=kwargs.get('std_color'),alpha=0.3)
    ax.plot(x,y,color=kwargs.get('line_color'),ls=kwargs.get('line_style'),linewidth=0.8,label=kwargs.get('label'))
    if type(kwargs.get('std')) != np.ndarray:
        raise Exception(f'{error} std')
    
    ##Setting wavelength range
    ax.set_xlim([*kwargs.get('wvl_lim')])
    if type(kwargs.get('wvl_lim')) != tuple:
        raise Exception(f'{error} wvl_lim')
    
    if kwargs.get('minorticks') == True:
        ax.set_xticks(np.linspace(kwargs.get('wvl_lim')[0],kwargs.get('wvl_lim')[1],4),fontname="Source Code Pro")
        ax.xaxis.set_minor_locator(tck.MultipleLocator(100))
        ax.set_yticks(np.arange(round(min(y)-0.1*min(y),2),round(max(y)+0.1*max(y),2),
                                round(((max(y)+0.1*max(y)-(min(y)-0.1*min(y)))/2),3)),fontname="Source Code Pro")
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
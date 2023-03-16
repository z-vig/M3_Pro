# -*- coding: utf-8 -*-
"""
Cubic Spline Interpolation for an Entire Image
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import LinearNDInterpolator as LNI
from scipy import interpolate as interp
from spectrum_averaging import moving_avg
from spectrum_averaging import spec_avg
from spectrum_averaging import nd_avg
import spectral as sp
import math
import time

def removeNAN(array):
    array[np.isnan(array)] = 0
    xs,ys,zs = np.where(array!=0)
    array = array[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1]
    nan_locations = [(min(xs),max(xs)+1),(min(ys),max(ys)+1),(min(zs),max(zs)+1)]
    
    return array,np.array(nan_locations)

def splineFit(signalArray,wavelengthValues,box_size,**kwargs):
    defaultKwargs = {"saveAvg":False,"saveCubic":False,"findMin":False}
    kwargs = {**defaultKwargs,**kwargs}


    print (f'An array of size {signalArray.shape} and band \
centers from {wavelengthValues.min()}\u03BCm to {wavelengthValues.max()}\u03BCm had been loaded')
    

    ##nrows = y, ncols = x
    yLen,xLen,wvlLen = signalArray.shape[0],signalArray.shape[1],signalArray.shape[2]
    yCoords,xCoords,wvlCoords = range(yLen),range(xLen),range(wvlLen)
    yMesh,wvlMesh = np.meshgrid(wvlCoords,yCoords)
    print (yMesh.shape,signalArray[:,0,:].shape)
    print ('Averaging across x coordinate...')
# =============================================================================
#     imgAverage = np.zeros(signalArray.shape)
#     imgAverage_dense = np.zeros((yLen,xLen,int(math.ceil(wvlLen/box_size))))
# =============================================================================
    imgLinInterp = np.zeros(signalArray.shape)
    imgCubicSpline = np.zeros(signalArray.shape)
    for count,x in enumerate(xCoords):
        print (f'\r{x+1}/{len(xCoords)} lines complete. ({x/len(xCoords):.0%})',end='\r')
        signalAvg,xAvg,yAvg,signalAvg_dense = nd_avg(yMesh,wvlMesh,signalArray[:,x,:],box_size,weighted=True)
        
        ptNum = xAvg.flatten().shape[0]
        points = np.zeros((ptNum,2))
        for n in range(ptNum):
            points[n] = (xAvg.flatten()[n],yAvg.flatten()[n])
            
        linear_interp = LNI(points,signalAvg_dense.flatten())
        signalAvg_filled = linear_interp(yMesh,wvlMesh)
        
        imgLinInterp[:,x,:] = signalAvg_filled
        
        signal_CubicSpline = np.zeros((signalAvg.shape[0],59))
        for row in range(signalAvg.shape[0]):
            #print (row)
            allY = signalAvg[row,:]
            xtest = np.where(allY!=0)[0]
            yNonZero = allY[xtest]
            f = interp.CubicSpline(xtest,yNonZero)
            xinterp = np.linspace(0,signalAvg.shape[1],59)
            signal_CubicSpline[row,:] = f(xinterp)
        
        imgCubicSpline[:,x,:] = signal_CubicSpline

        if kwargs.get('saveAvg') == True:
            np.save(r"/run/media/zvig/My Passport/Data/avg_image.npy",imgLinInterp)
        if kwargs.get('saveCubic') == True:
            np.save(r"/run/media/zvig/My Passport/Data/cubic_spline_image.npy",imgCubicSpline)

        if kwargs.get("findMin") == True:
            print ('Find Minimum is true')
        
    return imgLinInterp,imgCubicSpline

if __name__ == "__main__":
    start = time.time()
    
    hdr = sp.envi.open(r"D:\Data/20230209T095534013597/extracted_files/hdr_files/m3g20090417t193320_v01_rfl/m3g20090417t193320_v01_rfl.hdr")
    
    bandCenters = np.array(hdr.bands.centers)
    allowedIndices = np.where((bandCenters>900)&(bandCenters<2600))[0]
    minIndex,maxIndex = allowedIndices.min(),allowedIndices.max()
    
    allowedWvl = bandCenters[allowedIndices]
    signalArray = hdr.read_bands(allowedIndices)
    imgAverage,imgCubic = splineFit(signalArray,allowedWvl,5)

    def plt_stuff(xpt,ypt):
        fig,ax = plt.subplots(1,1)
        ax.plot(allowedWvl,signalArray[xpt,ypt,:],label='Original')
        ax.plot(allowedWvl,imgAverage[xpt,ypt,:],label='Calculated Average')
        y,std,x = spec_avg(signalArray[xpt,ypt,:],allowedWvl,5)
        ax.plot(x,y,label='Real Average')
        ax.set_title(f'{xpt},{ypt}')
        ax.legend()

    plt_stuff(302,2)
    plt_stuff(302,0)
    plt.show()
    # print ('\nRemoving NANs...')
    # imgAverage_NoNAN,nan_loc = removeNAN(imgAverage)
    
    # maxdel = len(wavelengthValues)-nan_loc[2,1]
    # wvl = np.delete(wavelengthValues,slice(0,nan_loc[2,0]))
    # wvl = np.delete(wvl,slice(len(wvl)-maxdel,len(wvl)))

    #np.save(r"/run/media/zvig/My Passport/Data/cubic_spline_image.npy",imgCubic)
    
    end = time.time()
    runtime = end-start
    if runtime < 1:
        print(f'Program Executed in {runtime*10**3:.3f} milliseconds')
    elif runtime < 60 and runtime > 1:
        print(f'Program Executed in {runtime:.3f} seconds')
    else:
        print(f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')
    





    
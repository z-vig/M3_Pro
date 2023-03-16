# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:51:44 2023

@author: zacha
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import spectral as sp
from scipy.interpolate import LinearNDInterpolator as LNI

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

def spec_avg(spec_list,wvl_list,box_size):
    avg_array = np.zeros((box_size,2))
    avg_rfl = []
    std_rfl = []
    avg_wvl = []
    n = 1
    for spec,wvl in zip(spec_list,wvl_list):
        if n%box_size != 0:
            avg_array[n-1] = (spec,wvl)
            n+=1
        elif n%box_size == 0:
            avg_array[n-1] = (spec,wvl)
            avg_rfl.append(np.average(avg_array[:,0]))
            std_rfl.append(np.std(avg_array[:,0]))
            avg_wvl.append(np.average(avg_array[:,1]))
            avg_array = np.zeros((box_size,2))
            n=1
    
    rfl_last = []
    wvl_last = []
    #print (avg_array)
    if np.all(np.where(avg_array==0,True,False)):
        return avg_rfl,std_rfl,avg_wvl
    
    for rfl,wvl in zip(avg_array[:,0],avg_array[:,1]):
        if rfl != 0:
            rfl_last.append(rfl)
            wvl_last.append(wvl)
            
    #print (wvl_last)
    avg_rfl.append(np.average(rfl_last))
    std_rfl.append(np.std(rfl_last))
    avg_wvl.append(np.average(wvl_last))
    
    return avg_rfl,std_rfl,avg_wvl

def nd_avg(X,Y,z,box_size,**kwargs):
    defaultKwargs = {"Weighted":False}
    kwargs = {**defaultKwargs,**kwargs}
    
    if X.shape != z.shape or Y.shape != z.shape:
        raise Exception('x and y must be mesh grids!')
    if box_size%2 == 0:
        raise Exception('Box Size must be odd!')
        
    def weighted_average(array,weights):
        return np.sum(array*weights)/np.sum(weights)
        
    M,N = 1,box_size
    new_zArray = np.zeros(z.shape)
    xCoords = np.zeros((z.shape[0],int(math.ceil(z.shape[1]/box_size))))
    yCoords = np.zeros(xCoords.shape)
    zArray_dense = np.zeros(xCoords.shape)
    tiles = [z[x:x+M,y:y+N] for x in range(0,z.shape[0],M) for y in range(0,z.shape[1],N)]
    x_pts = [X[x:x+M,y:y+N] for x in range(0,X.shape[0],M) for y in range(0,X.shape[1],N)]
    y_pts = [Y[x:x+M,y:y+N] for x in range(0,Y.shape[0],M) for y in range(0,Y.shape[1],N)]
    col = 0
    row = 0
    if kwargs.get('Weighted') == False:
        for tile,x_arr,y_arr in zip(tiles,x_pts,y_pts):
            #fig,ax = plt.subplots(1,1)
            #ax.imshow(tile)
            zAvg = np.average(tile)
            xAvg = int(np.average(x_arr))
            yAvg = int(np.average(y_arr))
            new_zArray[yAvg,xAvg] = zAvg
            zArray_dense[row,col] = zAvg
            xCoords[row,col] = xAvg
            yCoords[row,col] = yAvg
            if col<yCoords.shape[1]-1:
                col+=1
                continue
            else:
                col = 0
                row+=1
                continue
        
        return new_zArray,xCoords,yCoords,zArray_dense
    
    elif kwargs.get('Weighted') == True:
        weights = np.array(([0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]))
        
        for tile,x_arr,y_arr in zip(tiles,x_pts,y_pts):
            if tile.shape == (5,5):
                zAvg = weighted_average(tile,weights)
            elif tile.shape != (5,5):
                zAvg = np.average(tile)
            xAvg = int(np.average(x_arr))
            yAvg = int(np.average(y_arr))
            new_zArray[yAvg,xAvg] = zAvg
            zArray_dense[row,col] = zAvg
            xCoords[row,col] = xAvg
            yCoords[row,col] = yAvg
            if col<yCoords.shape[1]-1:
                col+=1
                continue
            else:
                col = 0
                row+=1
                continue
                
            
        return new_zArray,xCoords,yCoords,zArray_dense


if __name__ == "__main__":
    hdr = sp.envi.open(r"D:/Data/20230209T095534013597/extracted_files/hdr_files/m3g20090417t193320_v01_rfl/m3g20090417t193320_v01_rfl.hdr")
    bandCenters = np.array(hdr.bands.centers)
    allowedIndices = np.where((bandCenters>900)&(bandCenters<2600))[0]
    print (allowedIndices)
    allowedWvl = bandCenters[allowedIndices]
    image = hdr.read_bands(allowedIndices)

    yCoords,xCoords,wvlCoords = range(image.shape[0]),range(image.shape[1]),range(image.shape[2])
    yMesh,wvlMesh = np.meshgrid(wvlCoords,yCoords)
    print (image[:,0,:].shape)
    print (yMesh.shape)

    imageAvg,xAvg,yAvg,zDense = nd_avg(yMesh,wvlMesh,image[:,0,:],5,weighted=True)
    print (xAvg.shape)

    ptNum = xAvg.flatten().shape[0]
    points = np.zeros((ptNum,2))
    for n in range(ptNum):
        points[n] = (xAvg.flatten()[n],yAvg.flatten()[n])
            
    linear_interp = LNI(points,zDense.flatten())
    signalAvg_filled = linear_interp(yMesh,wvlMesh)

    def plt_stuff(xpt,ypt):
        fig,ax = plt.subplots(1,1)
        ax.plot(allowedWvl,image[xpt,ypt,:],label='Original')
        ax.plot(allowedWvl,signalAvg_filled[xpt,:],label='Calculated Average')
        y,std,x = spec_avg(image[xpt,ypt,:],allowedWvl,5)
        ax.plot(x,y,label='Real Average')
        ax.set_title(f'{xpt},{ypt}')
        ax.legend()

    plt_stuff(302,2)
    plt_stuff(302,0)
    plt.show()

    
    


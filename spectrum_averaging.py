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
        weights = np.array(([1,1,1],[100,200,100],[1,1,1]))
        weights = weights/sum(weights)
        
        for tile,x_arr,y_arr in zip(tiles,x_pts,y_pts):
            if tile.shape == (3,3):
                zAvg = weighted_average(tile,weights)
            elif tile.shape != (3,3):
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
            
    
    #new_zArray = new_zArray[np.where(new_zArray!=0)].reshape(xCoords.shape)

    

# =============================================================================
# x = np.arange(0,10,1)
# y = np.arange(0,10,1)
# xMesh,yMesh = np.meshgrid(x,y)
# z = np.random.choice(np.array((1,2)),(10,10))
# z = locals()['z']
# M,N=3,1
# new_zArray,x,y,z_dense = nd_avg(xMesh,yMesh,z,3)
# =============================================================================
#new_zArray1,x1,y1,z_dense1 = nd_avg(xMesh,yMesh,z,3,Weighted=True)




# =============================================================================
# if __name__ == "__main__":
#     x = np.arange(0,300,1)
#     y = np.arange(0,300,1)
#     xMesh,yMesh = np.meshgrid(x,y)
#     z = np.random.choice(np.array((0,1)),(300,300))
#     z = locals()['z']
#     M,N=3,3
#     new_zArray,x,y = nd_avg(xMesh,yMesh,z,3)
# =============================================================================
    
    


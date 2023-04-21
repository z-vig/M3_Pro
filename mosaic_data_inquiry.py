'''
Python script for obtaining necessary information about uploaded mosaic dataset
'''

##Import necessary modules
#%%
import numpy as np
import os
import pandas as pd
import M3_UnZip
import time
import spectral as sp
import Locate_Ice
from tkinter.filedialog import askdirectory as askdir
import copy

def mosaic_data_inquiry():
    print ('Select L2 folder')
    L2_fileList,L2_filePath = M3_UnZip.M3_unzip(select=True)
    print ('Select L1 folder')
    L1_fileList,L1_filePath = M3_UnZip.M3_unzip(select=True)
    print ('Select output/save folder')
    saveFolder = askdir()

    try:
        os.mkdir(f'{saveFolder}/mosaicStatistics')
    except:
        pass

    rfl_fileList = [i for i in L2_fileList if i.find('rfl')>-1]
    loc_fileList = [i for i in L1_fileList if i.find('loc')>-1]
    obs_fileList = [i for i in L1_fileList if i.find('obs')>-1]

    ##Mosaic data inquiry
    M3stamp_sample = Locate_Ice.HDR_Image(rfl_fileList[0],loc_fileList[0],obs_fileList[0])
    nBands = M3stamp_sample.analyzedWavelengths.shape[0]
    nStamps = len(rfl_fileList)

    imageStatsArray = np.zeros((nBands,5,nStamps))
    mosaicStatsArray = np.zeros((nBands,5))
    mosaicArray = np.zeros((nBands,0))
    illuminatedMosaic = np.zeros((nBands,0))
    illuminatedMosaicStats = np.zeros((nBands,5))

    stampNum = 0
    progress = 1
    for rflPath,locPath,obsPath in zip(rfl_fileList,loc_fileList,obs_fileList):
        M3stamp = Locate_Ice.HDR_Image(rflPath,locPath,obsPath)

        image = M3stamp.unprocessedImage
        xLight,yLight = M3stamp.get_illuminated_coords()
        shape = image.shape
        pixels = image.reshape(shape[0]*shape[1],shape[2])
        illuminatedPixels = image[xLight,yLight]


        imageAvg = np.average(pixels,axis=0)
        imageMedian = np.median(pixels,axis=0)
        imageStd = np.std(pixels,axis=0)
        imageMax = np.max(pixels,axis=0)
        imageMin = np.min(pixels,axis=0)

        imageStatsArray[:,0,stampNum] = imageAvg
        imageStatsArray[:,1,stampNum] = imageMedian
        imageStatsArray[:,2,stampNum] = imageStd
        imageStatsArray[:,3,stampNum] = imageMax
        imageStatsArray[:,4,stampNum] = imageMin
        stampNum +=1

        mosaicArray = np.concatenate((mosaicArray,pixels.T),axis=1)
        illuminatedMosaic = np.concatenate((illuminatedMosaic,illuminatedPixels.T),axis=1)

        print (f'\rBuilding Mosaic: ({progress/nStamps:.0%})',end='\r')
        progress+=1
    
    mosaicAvg = np.average(mosaicArray,axis=1)
    mosaicMedian = np.median(mosaicArray,axis=1)
    mosaicStd = np.std(mosaicArray,axis=1)
    mosaicMax = np.max(mosaicArray,axis=1)
    mosaicMin = np.min(mosaicArray,axis=1)

    mosaicStatsArray[:,0] = mosaicAvg
    mosaicStatsArray[:,1] = mosaicMedian
    mosaicStatsArray[:,2] = mosaicStd
    mosaicStatsArray[:,3] = mosaicMax
    mosaicStatsArray[:,4] = mosaicMin

    illuminatedAvg = np.average(illuminatedMosaic,axis=1)
    illuminatedMedian = np.median(illuminatedMosaic,axis=1)
    illuminatedStd = np.std(illuminatedMosaic,axis=1)
    illuminatedMax = np.max(illuminatedMosaic,axis=1)
    illuminatedMin = np.min(illuminatedMosaic,axis=1)

    illuminatedMosaicStats[:,0] = illuminatedAvg
    illuminatedMosaicStats[:,1] = illuminatedMedian
    illuminatedMosaicStats[:,2] = illuminatedStd
    illuminatedMosaicStats[:,3] = illuminatedMax
    illuminatedMosaicStats[:,4] = illuminatedMin

    
    return imageStatsArray,mosaicArray,mosaicStatsArray,illuminatedMosaic,illuminatedMosaicStats

if __name__ == "__main__":
    print('Creating image and mosaic statistics...')
    imageStatsArray,mosaicArray,mosaicStatsArray,illuminatedMosaic,illuminatedMosaicStats = mosaic_data_inquiry()
    print ('Statistics calculated! Success!')
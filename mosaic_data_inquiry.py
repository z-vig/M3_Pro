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
import spectral as sp

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

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

    imageStatsArray = np.zeros((nBands,5,nStamps)) #Row are {Average,Median,Standard Deviation, Max, Min}
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

    fileNameList = ['imageStatsArray','mosaicArray','mosaicStatsArray','illuminatedMosaic','illuminatedMosaicStats']
    for file,name in zip([imageStatsArray,mosaicArray,mosaicStatsArray,illuminatedMosaic,illuminatedMosaicStats],fileNameList):
            start = time.time()
            print (f'Saving {name}...')
            np.save(f'{saveFolder}/mosaicStatistics/{name}.npy',file)
            print (f'{name} took {time.time()-start:.2f} seconds\n')

    return imageStatsArray,mosaicArray,mosaicStatsArray,illuminatedMosaic,illuminatedMosaicStats

def mosaic_data_inquiry_large():
    print ('Select L2 folder')
    L2_fileList,L2_filePath = M3_UnZip.M3_unzip(select=True)
    print ('Select L1 folder')
    L1_fileList,L1_filePath = M3_UnZip.M3_unzip(select=True)
    print ('Select output/save folder')
    saveFolder = 'D:/Data/Ice_Pipeline_Out_5-9-23' #askdir()

    try:
        os.mkdir(f'{saveFolder}/mosaicStatistics')
    except:
        pass

    rfl_fileList = [i for i in L2_fileList if i.find('rfl')>-1]
    loc_fileList = [i for i in L1_fileList if i.find('loc')>-1]
    obs_fileList = [i for i in L1_fileList if i.find('obs')>-1]

    M3stamp_sample = Locate_Ice.HDR_Image(rfl_fileList[0],loc_fileList[0],obs_fileList[0])
    nBands = M3stamp_sample.analyzedWavelengths.shape[0]
    nStamps = len(rfl_fileList)

    illuminated_average_array = np.zeros((0,nBands))
    prog = 1
    tot = len(rfl_fileList)
    for rflPath,locPath,obsPath in zip(rfl_fileList,loc_fileList,obs_fileList):
        name_ind1 = find_all(rflPath,'\\')[-1]
        name_ind2 = find_all(rflPath,'_')[0]
        name = f'{rflPath[name_ind1+1:name_ind2]}'

        hdr_rfl = sp.envi.open(f'{rflPath}')
        hdr_obs = sp.envi.open(obsPath)
        #hdr_loc = sp.envi.open(locPath)
        
        wvl = np.array(hdr_rfl.bands.centers)
        allowedInd = np.where((wvl>900)&(wvl<2600))[0]
        allowedWvl = wvl[allowedInd]
        
        rfl_image = hdr_rfl.read_bands(allowedInd)
        #loc_image = hdr_loc.read_bands([0,1])
        obs_image = hdr_obs.read_band(-1)
        
        solar_incidence_image = 180*np.arccos(obs_image)/np.pi
        xLight,yLight = (np.where(solar_incidence_image<90))
        shape = rfl_image.shape
        pixels = rfl_image.reshape(shape[0]*shape[1],shape[2])
        
        illuminatedPixels = rfl_image[xLight,yLight]
        illuminatedAvg = np.mean(illuminatedPixels,axis=0)
        
        illuminated_average_array = np.concatenate([illuminated_average_array,np.expand_dims(illuminatedAvg,0)])
        print (f'\r{prog} of {tot} complete ({prog/tot:.0%}), ({illuminated_average_array.shape})',end='\r')
        prog+=1

    np.save(f'{saveFolder}/illuminated_average_array.npy',illuminated_average_array)
    return illuminated_average_array
     

if __name__ == "__main__":
    print('Creating image and mosaic statistics...')
    illuminated_average_array = mosaic_data_inquiry_large()
    print ('Statistics calculated! Success!')
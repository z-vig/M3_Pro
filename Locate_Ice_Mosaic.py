'''
HDR Image Class and Script for locating Ice Pixels using both L1 and L2 data from the moon mineralogy mapper
'''
#%%
import time
import spectral as sp
import numpy as np
import spec_plotting
import matplotlib.pyplot as plt
import destripe_image
from copy import copy
import cubic_spline_image as csi
import pandas as pd
import tifffile as tf
import os
import M3_UnZip
from tkinter.filedialog import askdirectory as askdir
import datetime
import shutil
from get_USGS_H2OFrost import get_USGS_H2OFrost
import rasterio

def find_all(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def get_metaDataList(imgList:list)->list:
    dictList = []
    for img in imgList:
        if len(img.shape)<3:
            img=np.expand_dims(img,2)
        bandIndex = img.shape.index(min(img.shape))
        imgSave = np.moveaxis(img,bandIndex,0)
        dictList.append({'driver':'GTiff',\
                            'height':imgSave.shape[1],\
                            'width':imgSave.shape[2],\
                            'count':imgSave.shape[0],\
                            'dtype':imgSave.dtype,\
                            'interleave':'pixel',\
                            'nodata':-9999})
    return dictList

class M3_Mosaic():
    def __init__(self,saveFolderPath) -> None:
        self.saveFolderPath = saveFolderPath

        print ('Fetching rfl Images...')
        rflPathList = os.listdir(f'{self.saveFolderPath}/rfl_cropped')
        self.rflImgList = []
        prog,tot = 1,len(rflPathList)
        for img in rflPathList:
            print (f'\r{prog} of {tot} ({prog/tot:.0%})',end='\r')
            self.rflImgList.append(tf.imread(f'{self.saveFolderPath}/rfl_cropped/{img}'))
            prog+=1
    
    @property
    def stampNames(self)->list:
        with open(f'{self.saveFolderPath}/stampNames.txt','r') as f:
            names = f.readlines()
        return [name[:-2] for name in names]
    
    @property
    def rflImages(self)->list:
        return self.rflImgList
    
    @property
    def locImages(self)->list:
        pathList = os.listdir(f'{self.saveFolderPath}/loc_cropped')
        imgList = []
        for img in pathList:
            imgList.append(tf.imread(f'{self.saveFolderPath}/locs_cropped/{img}'))
        return imgList
    
    @property
    def obsImages(self)->list:
        pathList = os.listdir(f'{self.saveFolderPath}/obs_cropped')
        imgList = []
        for img in pathList:
            imgList.append(tf.imread(f'{self.saveFolderPath}/obs_cropped/{img}'))
        return imgList
    
    @property
    def analyzedWavelengths(self)->np.ndarray:
        df = pd.read_csv(f'{self.saveFolderPath}/bandInfo.csv')
        bandArr = df.to_numpy()
        return bandArr[:,2]
    
    @property
    def statistics(self)->np.ndarray:
        try:
            stats_arr = np.load(f'{self.saveFolderPath}/mosaic_stats_array.npy')
        except:
            raise FileNotFoundError('Run the mosaic data inquiry script first!')
        return stats_arr
    
    def destripe_images(self)->dict:
        startTime = time.time()
        self.destripeDict = {}
        prog,tot = 1,len(self.stampNames)
        for name,image in zip(self.stampNames,self.rflImages):
            destripeImage = destripe_image.fourier_filter(image)
            self.destripeDict.update({name:destripeImage})
            print(f'\r{prog} of {tot} ({prog/tot:.0%})',end='\r')
            prog+=1
        print (f'>>>Destriping complete in {time.time()-startTime:.1f} seconds')
        return self.destripeDict
    
    def shadow_correction(self,**kwargs)->dict:
        defaultKwargs = {'inputImageDictionary':self.destripeDict}
        kwargs = {**defaultKwargs,**kwargs}
        startTime = time.time()
        nameList,imageList = kwargs.get('inputImageDictionary').keys(),kwargs.get('inputImageDictionary').values()
        
        self.correctedImageDict = {}
        R_BIDIRECTIONAL = np.mean(self.statistics[:,:,0],axis=0)
        prog,tot = 1,len(nameList)
        for name,image in zip(nameList,imageList):
            self.correctedImageDict.update({name:image/R_BIDIRECTIONAL})
            print(f'\r{prog} of {tot} ({prog/tot:.0%})',end='\r')
            prog+=1
        print (f'>>>Shadow correction complete in {time.time()-startTime:.1f} seconds')
        return self.correctedImageDict
    
    def spectrum_smoothing(self,**kwargs)->dict:
        defaultKwargs = {'inputImageDictionary':self.correctedImageDict}
        kwargs = {**defaultKwargs,**kwargs}
        startTime = time.time()
        nameList,imageList = kwargs.get('inputImageDictionary').keys(),kwargs.get('inputImageDictionary').values()

        self.smoothDict = {}
        prog,tot = 1,len(nameList)
        for name,image in zip(nameList,imageList):
            avgWvl,avgSpectrumImage,smoothSpectrumImage = csi.splineFit(image,5,self.analyzedWavelengths)
            self.smoothDict.update({name:smoothSpectrumImage})
            print(f'{prog} of {tot} ({prog/tot:.0%})')
            prog+=1
        
        print (f'>>>Spectrum Smoothing complete in {time.time()-startTime:.1f} seconds')
        return self.smoothDict
    
    def locate_ice(self,**kwargs)->tuple[np.ndarray,np.ndarray,pd.DataFrame]:
        defaultKwargs = {'inputImageDictionary':self.smoothDict}
        kwargs = {**defaultKwargs,**kwargs}
        startTime = time.time()
        nameList,imageList = kwargs.get('inputImageDictionary').keys(),kwargs.get('inputImageDictionary').values()

        band1_indices = np.where((self.analyzedWavelengths>1242)&(self.analyzedWavelengths<1323))[0]
        band2_indices = np.where((self.analyzedWavelengths>1503)&(self.analyzedWavelengths<1659))[0]
        band3_indices = np.where((self.analyzedWavelengths>1945)&(self.analyzedWavelengths<2056))[0]
        allBand_indices = np.concatenate((band1_indices,band2_indices,band3_indices))

        waterLocateDict = {}
        for name,image,mapCoords in zip(nameList,imageList,self.locImages):
            diff_array = np.zeros(image.shape)
            for band in range(image.shape[2]-1): #The last band will be all zeros
                diff_array[:,:,band] = image[:,:,band]>image[:,:,band+1]
            
            def get_bandArray(band_indices:np.ndarray,bandName:str)->np.ndarray:
                band_arr = np.zeros((image.shape[0:2]))
                for i in range(band_indices.min()-1,band_indices.max()):
                    band_arr[np.where((diff_array[:,:,i]!=diff_array[:,:,i+1])&(diff_array[:,:,i]==True))] = 1
                return band_arr

            band1_Array = get_bandArray(band1_indices,'Band 1')
            band2_Array = get_bandArray(band2_indices,'Band 2')
            band3_Array = get_bandArray(band3_indices,'Band 3')

            self.waterLocations = np.zeros(band1_Array.shape)
            self.waterCoords_numpy = np.where((band1_Array==1)&(band2_Array==1)&(band3_Array==1)&(np.average(kwargs.get('inputImage'),axis=2)>0))
            self.waterLocations[self.waterCoords_numpy] = 1

            self.waterCoords_map = mapCoords[self.waterCoords_numpy[0],self.waterCoords_numpy[1],:]
            waterDf = pd.DataFrame(self.waterCoords_map)
            waterDf.columns = ['Latitude','Longitude','Elevation']

            waterLocateDict.update({name:waterDf})
        
        print(f'>>>Ice located in {time.time()-startTime:.1f} seconds')
        return waterLocateDict
    

if __name__ == '__main__':
    print ('Select Analysis Folder:')
    folderPath = askdir()
    largeMosaic = M3_Mosaic(folderPath)
    print (f'-----Beginning Mosaic analysis of {len(largeMosaic.stampNames)} images-----')
    print ('Destriping Images...')
    dedict = largeMosaic.destripe_images()
    print ('Running Li et al., 2018 Shadow Correction...')
    cordict = largeMosaic.shadow_correction()
    print ('Smoothing spectrum...')
    smoothdict = largeMosaic.spectrum_smoothing()
    print ('locating ice...')
    waterlocatedict = largeMosaic.locate_ice()





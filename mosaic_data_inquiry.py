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
import Locate_Ice_TIF
from tkinter.filedialog import askdirectory as askdir
import copy
import spectral as sp
import tifffile as tf
import rasterio as rio
import copy
import time

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

def mosaic_data_inquiry():
    print ('Select RFL Folder')
    rfl_fileList = 'D:/Data/OP2C_Downloads/L2_sorted/sorted_tif_files/rfl_files' #askdir()
    print (f'{rfl_fileList} selected\nSelect LOC Folder')
    loc_fileList = 'D:/Data/OP2C_Downloads/L1_sorted/sorted_tif_files/loc_files' #askdir()
    print (f'{loc_fileList} selected\nSelect OBS Folder')
    obs_fileList = 'D:/Data/OP2C_Downloads/L1_sorted/sorted_tif_files/obs_files' #askdir()
    print (f'{obs_fileList} selected')
    print ('Select output/save folder')
    saveFolder = askdir()

    try:
        os.mkdir(f'{saveFolder}/mosaicStatistics')
    except:
        pass

    ##Mosaic data inquiry
    M3stamp_sample = Locate_Ice_TIF.TIF_Image(rfl_fileList[0],loc_fileList[0],obs_fileList[0])
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
        M3stamp = Locate_Ice_TIF.TIF_Image(rflPath,locPath,obsPath)
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

def mosaic_data_inquiry_large_withi(imsave=False):
    print ('Select Output Folder')
    saveFolder = askdir()

    try:
        os.mkdir(f'{saveFolder}/shaded_removed')
    except:
        pass

    rfl_fileList = os.listdir(f'{saveFolder}/rfl_cropped')
    obs_fileList = os.listdir(f'{saveFolder}/obs_cropped')

    stats_array = np.zeros((0,59,2))

    rflImgList,obsImgList,pixList = [],[],[]
    bright_bool_list = []
    prog = 0
    tot = len(rfl_fileList)
    for rfl,obs in zip(rfl_fileList,obs_fileList):
        rfl_img = tf.imread(os.path.join(saveFolder,'rfl_cropped',rfl))
        obs_img = tf.imread(os.path.join(saveFolder,'obs_cropped',obs))
        deg_incidence = 180*np.arccos(obs_img[:,:,-1])/np.pi
        obs_img[:,:,-1] = deg_incidence

        rflImgList.append(rfl_img)
        obsImgList.append(obs_img)
       
        bright_pixels = np.where(deg_incidence<90)
        pixList.append(bright_pixels)

        bright_bool = -9999*np.ones((rfl_img.shape[:2]))
        bright_bool[bright_pixels] = 1
        bright_bool_list.append(bright_bool)

        bright_array = copy.copy(rfl_img)
        bright_array = bright_array[bright_pixels[0],bright_pixels[1],:]
        image_avg = np.mean(bright_array,axis=0)
        image_std = np.std(bright_array,axis=0)
        stats_array = np.concatenate([stats_array,np.expand_dims(np.array([image_avg,image_std]).T,0)],axis=0)
        
        print (f'\r{prog+1} of {tot} ({prog/tot:.0%})',end='\r')
        prog+=1
    
    if imsave == True:
        print ('Saving Images...')
        try:
            os.mkdir(os.path.join(saveFolder,'incidence_bright_bool_arrays'))
        except:
            pass
        prog,tot=1,len(rfl_fileList)
        for rfl,bool_array in zip(rfl_fileList,bright_bool_list):
            bool_array_save = copy.copy(bool_array)
            bool_array_save = bool_array_save[np.newaxis,:]
            with rio.open(os.path.join(saveFolder,'incidence_bright_bool_arrays',f'{rfl[3:-16]}_inc_bright.tif'),'w',\
                            driver='GTiff',height=bool_array_save.shape[1],\
                            width=bool_array_save.shape[2],\
                            count=1,\
                            dtype='float32',nodata=-9999) as f:
                            f.write(bool_array_save)

            print (f'\r{prog} of {tot} ({prog/tot:.0%})',end='\r')
            prog+=1
        

def mosaic_data_inquiry_large_thresholdRFL(imsave=False):
    print ('Select Output Folder:')
    saveFolder = askdir()
    print (f'{saveFolder} selected')

    rfl_fileList = os.listdir(f'{saveFolder}/rfl_cropped')
    loc_fileList = os.listdir(f'{saveFolder}/loc_cropped')
    obs_fileList = os.listdir(f'{saveFolder}/obs_cropped')

    THRESH_RFL = 0.05

    stats_array = np.zeros((0,59,2))
    brightImageList = []
    prog,tot = 1,len(rfl_fileList)
    for file in rfl_fileList:
        img_rfl = tf.imread(f'{saveFolder}/rfl_cropped/{file}')
        brightPixels = np.where(np.mean(img_rfl,axis=2)>THRESH_RFL)
        brightImg = -9999*np.ones((img_rfl.shape[:2]))
        brightImg[brightPixels] = 1
        brightImageList.append(brightImg)

        bright_array = copy.copy(img_rfl)
        bright_array = bright_array[brightPixels[0],brightPixels[1],:]
        image_avg = np.mean(bright_array,axis=0)
        image_std = np.std(bright_array,axis=0)
        stats_array = np.concatenate([stats_array,np.expand_dims(np.array([image_avg,image_std]).T,0)],axis=0)
        
        print (f'\r{prog} of {tot} complete ({prog/tot:.0%})',end='\r')
        prog+=1
    
    np.save(f'{saveFolder}/mosaic_stats_array.npy',stats_array)
    
    if imsave==True:
        print('Saving shaded images...')
        try:
            os.mkdir(f'{saveFolder}/bright_bool_arrays')
        except:
            pass
        
        prog,tot = 1,len(rfl_fileList)
        for file,brightImg in zip(rfl_fileList,brightImageList):
            brightImg_save = copy.copy(brightImg)
            brightImg_save = np.expand_dims(brightImg_save,2)
            brightImg_save = np.moveaxis(brightImg_save,2,0)
            with rio.open(os.path.join(saveFolder,'bright_bool_arrays',f'{file[3:-16]}_bright.tif'),'w',\
                            height=brightImg_save.shape[1],width=brightImg_save.shape[2],\
                            count=brightImg_save.shape[0],dtype='float32',\
                            nodata=-9999) as f:
                            f.write(brightImg_save)
            print (f'\r{prog} of {tot} complete ({prog/tot:.0%})',end='\r')
            prog+=1
            pass
        
    return stats_array,brightImageList
     

if __name__ == "__main__":
    start = time.time()
    print('Creating image and mosaic statistics...')
    stats_array = mosaic_data_inquiry_large_thresholdRFL(imsave=True)
    print ('Statistics calculated! Success!')
    end = time.time()

    print (f'Program completed in {(end-start)/60:.3f} minutes')
# %%
# def plot_avg(stats_array):    
#     for i in range(1,stats_array.shape[0]):
#         plt.plot(stats_array[i,:,0],ls='--',color='k',alpha=0.2)
#     av,std = np.mean(stats_array[:,:,0],axis=0),np.mean(stats_array[:,:,1],axis=0)
#     plt.plot(av,color='red')
#     plt.fill_between(range(len(av)),av-std,av+std,color='k',alpha=0.3)
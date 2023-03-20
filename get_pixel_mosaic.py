##Import necessary modules
import numpy as np
import os
import pandas as pd
import Locate_Ice
from HySpec_Image_Processing import HDR_Image
from M3_UnZip import M3_unzip
import time

def create_arrays(dataFolder,saveFolder):
    ##Timing script
    start = time.time()

    ##Get hdr_file_list
    hdrFileList,hdrFilePath = M3_unzip(select=False,folder=dataFolder)

    ##Load in all images for mosaic from source directory
    print ('Loading Images...')
    obj_list = []
    for file in hdrFileList:
        if file.find('rfl') > -1:
            obj_list.append(Locate_Ice.HDR_Image(file))
    print (f'Images Loaded at {time.time()-start} seconds')

    ##Getting list of shadow arrays
    print ('Getting list of shadow arrays...')

    try:
        os.mkdir(os.path.join(saveFolder,'Shadow_Arrays'))
    except:
        pass

    shadowImage_arrayDict = {}
    processedFiles = os.listdir(os.path.join(saveFolder,'Shadow_Arrays'))
    for count,obj in enumerate(obj_list):
        if obj.datetime+'_shadow.npy' in processedFiles:
            print (obj.datetime+'_shadow.npy is already saved.')
            array = np.load(os.path.join(saveFolder,'Shadow_Arrays',obj.datetime+'_shadow.npy'))
            shadowImage_arrayDict[obj.datetime] = array
            continue
        array = obj.find_shadows(saveImage=False,showPlot=False)
        np.save(os.path.join(saveFolder,'Shadow_Arrays',obj.datetime+'_shadow.npy'),array)
        shadowImage_arrayDict[obj.datetime] = array
        im_id = obj.datetime
        print (f'Image {im_id} complete ({count}/{len(obj_list)})')
    print (f'List of shadows arrays filled at {time.time()-start} seconds')

    ##Getting wavelength, average and standard deviation of each image

    print ('Getting individual image statistics...')

    try:
        os.mkdir(os.path.join(saveFolder,'Image_Statistics'))
    except:
        pass

    data_arr = np.zeros((obj_list[0].hdr.nbands-2, 3))
    imageStatsArray = np.zeros((*data_arr.shape,14))
    processedFiles = os.listdir(os.path.join(saveFolder,'Image_Statistics'))
    n = 0
    for obj, arr in zip(obj_list, shadowImage_arrayDict.values()):
        if obj.datetime+'_statistics.csv' in processedFiles:
            imageStatistics = pd.read_csv(os.path.join(saveFolder,'Image_Statistics',obj.datetime+'_statistics.csv'),index_col=0)
            imageStatistics = imageStatistics.to_numpy()
            imageStatsArray[:,:,n] = imageStatistics
            print (obj.datetime+'_statistics.csv is already saved.')
            continue
        wvl, avg, std = obj.get_average_rfl(arr,avg_by_img=True)
        data_arr[:,:] = np.array([wvl, avg, std]).T
        imageStatistics = pd.DataFrame(data_arr)
        imageStatsArray[:,:,n] = imageStatistics.to_numpy()
        imageStatistics.to_csv(os.path.join(saveFolder,'Image_Statistics',obj.datetime+'_statistics.csv'))
        im_id = obj.datetime
        print (f'Image {im_id} complete ({n+1}/{len(obj_list)})')
        n += 1

    print (f'Image statistics obtained at {time.time()-start:.1f} seconds')

    ##Getting large array with size (num of bands, num of all pixels in mosaic)
    print ('Getting mosaic pixel statistics...')
    mosaicArray = np.zeros((data_arr.shape[0],1))
    n = 0
    try:
        mosaicArray = np.load(os.path.join(saveFolder,'mosaic_pixels.npy'))
        if np.count_nonzero(mosaicArray==0) == mosaicArray.shape[0]*mosaicArray.shape[1]:
            raise Exception('Array not filled')
        else:
            print ('Mosaic Pixel Array already saved')
    except:
        for obj,array in zip(obj_list,shadowImage_arrayDict.values()):
            wvl,newImage_array = obj.get_average_rfl(array,avg_by_img=False)
            mosaicArray = np.concatenate((mosaicArray,newImage_array),axis=1)
            im_id = obj.datetime
            print (f'Image {im_id} complete ({n+1}/{len(obj_list)})')
            n+=1

        mosaicArray = np.delete(mosaicArray, 0, 1)
        np.save(os.path.join(saveFolder,'mosaic_pixels.npy'),mosaicArray)


    mosaicStatsArray = np.zeros((mosaicArray.shape[0],3))
    n = 0
    for row in range(mosaicArray.shape[0]):
        avg,median,std = np.average(mosaicArray[n,:]),np.std(mosaicArray[n,:]),np.median(mosaicArray[n,:])
        mosaicStatsArray[n,:] = [avg,median,std]
        n+=1
    df = pd.DataFrame(mosaicStatsArray)
    df.to_csv(os.path.join(saveFolder,'Mosaic_Statistics.csv'))
    print (f'Mosaic Statistics obtained at {time.time()-start:.1f} seconds')

    return shadowImage_arrayDict,imageStatsArray,mosaicArray,mosaicStatsArray


if __name__ == "__main__":
    print('Creating image and mosaic statistics...')
    shadow,imstats,mosaic,mosaicStats = create_arrays(r"D:/Data/20230209T095534013597/",r'D:/Data/')
    print (len(shadow))
    print ('Statistics calculated! Success!')
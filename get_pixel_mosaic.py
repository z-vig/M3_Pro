##Import necessary modules
import numpy as np
import os
import pandas as pd
from HySpec_Image_Processing import HDR_Image
from M3_UnZip import M3_unzip
import time

def create_arrays(saveFolder):
    ##Timing script
    start = time.time()

    ##Get hdr_file_list
    hdrFileList,hdrFilePath = M3_unzip(select=False,folder=r'/run/media/zvig/My Passport/Data/20230209T095534013597')

    ##Load in all images for mosaic from source directory
    print ('Loading Images...')
    obj_list = []
    for file in hdrFileList:
        if file.find('rfl') > -1:
            obj_list.append(HDR_Image(file))
    print (f'Images Loaded at {time.time()-start} seconds')

    ##Getting list of shadow arrays
    print ('Getting list of shadow arrays...')
    shadowImage_arrayList = []
    for count,obj in enumerate(obj_list):
        array = obj.find_shadows(saveImage=False,showPlot=False)
        np.save(os.path.join(saveFolder,'Shadow_Arrays',obj.datetime+'_shadow.npy'),array)
        shadowImage_arrayList.append(array)
        im_id = obj.datetime
        print (f'Image {im_id} complete ({count}/{len(obj_list)})')
    print (f'List of shadows arrays filled at {time.time()-start} seconds')

    ##Getting wavelength, average and standard deviation of each image
    print ('Getting individual image statistics...')
    data_arr = np.zeros((obj_list[0].hdr.nbands-2, 3))
    n = 0
    for obj, arr in zip(obj_list, shadowImage_arrayList):
        wvl, avg, std = obj.get_average_rfl(arr,avg_by_img=True)
        data_arr[n,:] = [wvl, avg, std]
        df = pd.DataFrame(data_arr)
        df.to_csv(os.join.path(saveFolder,'Image_Statistics',obj.datetime+'_statistics'))
        print (f'Image {im_id} complete ({n}/{len(obj_list)}')
        n += 1
    print (f'Image statistics obtained at {time.time()-start} seconds')

    ##Getting large array with size (num of bands, num of all pixels in mosaic)
    print ('Getting mosaic pixel statistics...')
    large_arr = np.zeros(data_arr.shape[0],1)
    n = 0
    for obj,array in zip(obj_list,shadowImage_arrayList):
        wvl,newImage_array = obj.get_average_rfl(array,avg_by_img=False)
        large_arr = np.concatenate((large_arr,newImage_array),axis=1)
        print (f'Image {im_id} complete ({n}/{len(obj_list)}')
        n+=1

    large_arr = np.delete(large_arr, 0, 1)
    np.save(os.path.join(saveFolder,'mosaic_pixels.npy'),large_arr)

    mosaicStatsArray = np.zeros(large_arr.shape[0],3)
    n = 0
    for row in range(large_arr.shape[0]):
        avg,median,std = np.average(large_arr[n,:]),np.std(large_arr[n,:]),np.median(large_arr[n,:])
        mosaicStatsArray[n,:] = [avg,median,std]
    df = pd.DataFrame(mosaicStatsArray)
    df.to_csv(os.path.join(saveFolder),'Mosaic_Statistics')
    print (f'Mosaic Statistics obtained at {time.time()-start} seconds')

if __name__ == "__main__":
    print('Creating image and mosaic statistics...')
    create_arrays(r'/run/media/zvig/My Passport/Data')
    print ('Statistics calculated! Success!')
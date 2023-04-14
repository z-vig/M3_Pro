##Import necessary modules
#%%
import numpy as np
import os
import pandas as pd
from M3_UnZip import M3_unzip
import time
import spectral as sp

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def create_arrays(dataFolder,saveFolder,**kwargs):
    defaultKwargs = {'allowedWvl':False}
    kwargs = {**defaultKwargs,**kwargs}

    ##Timing script
    start = time.time()

    ##Get hdr_file_list
    hdrFileList,hdrFilePath = M3_unzip(select=False,folder=dataFolder)

    ##Load in all images for mosaic from source directory and get ID list
    print ('Loading Images...')
    hdr_list = []
    fileID_list = []
    for file in hdrFileList:
        if file.find('rfl') > -1:
            hdr_list.append(sp.envi.open(file))
            slash_index = find(file,"\\")[-1]
            fileID = list(file[slash_index+4:slash_index+19].replace('t','_'))
            for i in [4,7,13,16]:
                fileID.insert(i,'-')
            fileID = ''.join(fileID)
            fileID_list.append(fileID)
    
    ##Get the bands we are analyzing to trim hdr image objects
    def get_allowedWvl(lower,upper):
        allBands = np.array(hdr_list[0].bands.centers)
        allowed_indices = np.where((allBands>lower)&(allBands<upper))[0]
        allowedWvl = allBands[allowed_indices]

        return allowed_indices,allowedWvl

    if kwargs.get('allowedWvl') == False:
        allowed_indices,allowedWvl = get_allowedWvl(900,2600)
    else:
        lower,upper = kwargs.get('allowedWvl')[0],kwargs.get('allowedWvl')[1]
        allowed_indices,allowedWvl = get_allowedWvl(lower,upper)

    ##Trim the image objects
    hdr_trimmed = []
    for obj in hdr_list:
        hdr_trimmed.append(obj.read_bands(allowed_indices))
    
    #Getting statistics arrays with 3rd dimension labels: {average,median,standard deviation,min,max}
    image_array_dict = {}
    for img,imgName in zip(hdr_trimmed,fileID_list):
        stats_array = np.zeros((img.shape[0],img.shape[1],5))
        stats_array[:,:,0] = np.average(img,axis=2)
        stats_array[:,:,1] = np.median(img,axis=2)
        stats_array[:,:,2] = np.std(img,axis=2)
        stats_array[:,:,3] = np.min(img,axis=2)
        stats_array[:,:,4] = np.max(img,axis=2)
        image_array_dict.update({imgName:stats_array})

    
    print (f'Images Loaded at {time.time()-start} seconds')

    return stats_array

    #return shadowImage_arrayDict,imageStatsArray,mosaicArray,mosaicStatsArray

create_arrays(r"E:/Data/20230209T095534013597/",r'D:/Data/Ice_Pipeline_Out_4-13-23/')
#%%


if __name__ == "__main__":
    print('Creating image and mosaic statistics...')
    hdr_list = create_arrays(r"E:/Data/20230209T095534013597/",r'D:/Data/Ice_Pipeline_Out_4-13-23/')
    print ('Statistics calculated! Success!')
#%%
import numpy as np
import os
import tifffile as tf
import rasterio as rio
import copy
from tkinter.filedialog import askdirectory as askdir

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

def save_tif(tif_array:np.ndarray,meta_data:dict,tifPath:str):
    if len(tif_array.shape)<3:
        tif_array = np.expand_dims(tif_array,2)
    bandAxis = tif_array.shape.index(min(tif_array.shape))
    tif_array = np.moveaxis(tif_array,bandAxis,0)
    with rio.open(tifPath,'w',**meta_data) as f:
        f.write(tif_array)
        

def crop(srcFolder,locFolder,dstFolder):
    srcFileList = [i for i in os.listdir(srcFolder) if i.find('.ovr')==-1]

    nameIndex = find_all(srcFolder,'/')[-1]
    newFolderName = f'{srcFolder[nameIndex:]}_cropped'
    try:
        os.mkdir(f'{dstFolder}/{newFolderName}')
    except:
        pass

    prog = 1
    tot = len(srcFileList)
    croppedNames = []
    for file,loc in zip(srcFileList,os.listdir(locFolder)):
        ##Check if files exist
        if f'{file[:-4]}_cropped.tif' in os.listdir(f'{dstFolder}/{newFolderName}'):
            print (f'Image {prog} of {tot} already cropped\tImageID: {file[:-8]}')
            prog+=1
            continue

        ##Getting source metadata
        with rio.open(f'{srcFolder}/{file}','r') as f:
            meta_data = f.profile
        
        #print ('Reading images...')
        img = tf.imread(f'{srcFolder}/{file}')
        loc_img = tf.imread(f'{locFolder}/{loc}')
        crop_img = copy.copy(img)

        #print ('Cropping images...')
        rows_to_del = np.where(loc_img[:,:,1]>-70)[0]
        if len(rows_to_del) == 0:
            rows_to_del = []
        else:
            min_row,max_row = rows_to_del.min(),rows_to_del.max()
            rows_to_del = np.arange(min_row,max_row)

        if len(rows_to_del) == img.shape[0]-1:
            print (f'Image {prog} of {tot} is outside of target latitude range\tImageID: {file[:-8]}\n')
            prog+=1
            continue

        elif len(rows_to_del)==0:
            #print ('Writing images...')
            save_tif(crop_img,meta_data,f'{dstFolder}/{newFolderName}/{file}_cropped.tif')
            print(f'Image {prog} of {tot} is entirely within target latitude range\tImageID: {file[:-8]}\n')
            prog+=1
            croppedNames.append(file[:-8])
            continue
        else:
            crop_img = np.delete(crop_img,rows_to_del,axis=0)
            #print ('Writing images...')
            save_tif(crop_img,meta_data,f'{dstFolder}/{newFolderName}/{file}_cropped.tif')
            print(f'\r{prog} of {tot} saved ({prog/tot:.0%})\tImageID: {file[:-8]}',end='\r')
            prog+=1
            croppedNames.append(file[:-8])
            continue


    # with open(f'{dstFolder}/cropped_image_names.txt','w') as f:
    #     for name in croppedNames:
    #         f.write(f'{name}*\n')

if __name__ == "__main__":
    print ('Select folder of images to crop')
    srcFolder = askdir() #'D:/Data/OP2C_Downloads/L2_sorted/sorted_tif_files/rfl_files'
    print (f'{srcFolder} selected\nSelect LOC Folder')
    locFolder = askdir() #'D:/Data/OP2C_Downloads/L1_sorted/sorted_tif_files/loc_files'
    print (f'{locFolder} selected\nSelect Save Folder')
    saveFolder = askdir() #'D:/Data/Ice_Pipeline_Out_5-16-23'
    print (f'{saveFolder} selected')
    crop(srcFolder,locFolder,saveFolder)
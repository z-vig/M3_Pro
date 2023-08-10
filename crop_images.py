#%%
import numpy as np
import os
import tifffile as tf
import rasterio as rio
import copy
from tkinter.filedialog import askdirectory as askdir

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

def stereo_project_x(lat:np.array,long:np.array)->np.array:
    return 2*1737400*np.tan(np.pi/4-np.pi*abs(lat/360))*np.sin(np.pi*long/180)

def stereo_project_y(lat:np.array,long:np.array)->np.array:
    return 2*1737400*np.tan(np.pi/4-np.pi*abs(lat/360))*np.cos(np.pi*long/180)

def save_tif(tif_array:np.ndarray,meta_data:dict,tifPath:str):
    if len(tif_array.shape)<3:
        tif_array = np.expand_dims(tif_array,2)
    bandAxis = 2
    tif_array = np.moveaxis(tif_array,bandAxis,0)
    with rio.open(tifPath,'w',**meta_data) as f:
        f.write(tif_array)
        

def crop(srcFolder,locFolder,dstFolder):
    srcFileList = [i for i in os.listdir(srcFolder) if i.find('.ovr')==-1]

    nameIndex = find_all(srcFolder,'/')[-1]
    newFolderName = f'{srcFolder[nameIndex:]}_shoemaker'
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

        loc_img_xmeters = stereo_project_x(loc_img[:,:,1],loc_img[:,:,0])
        loc_img_ymeters = stereo_project_y(loc_img[:,:,1],loc_img[:,:,0])
        loc_img_meters = np.concatenate([loc_img_xmeters[...,np.newaxis],loc_img_ymeters[...,np.newaxis]],axis=2)

        #print ('Cropping images...')
        MAX_X = 60000
        MIN_X = 15000
        MAX_Y = 65000
        MIN_Y = 14000

        bool_array = np.full(img.shape[:2],False)
        bool_array[np.where((loc_img_meters[:,:,0]>MIN_X)&(loc_img_meters[:,:,0]<MAX_X)\
                            &(loc_img_meters[:,:,1]>MIN_Y)&(loc_img_meters[:,:,1]<MAX_Y))] = True

        
        rows_to_keep = np.unique(np.where(bool_array==True)[0])
        cols_to_keep = np.unique(np.where(bool_array==True)[1])

        rows_to_del,cols_to_del = np.arange(0,img.shape[0]),np.arange(0,img.shape[1])
        rows_to_del,cols_to_del = np.delete(rows_to_del,rows_to_keep),np.delete(cols_to_del,cols_to_keep)

        if len(rows_to_del) == 0:
            rows_to_del = []
        else:
            pass
        
        if len(cols_to_del) == 0:
            cols_to_del = []
        else:
            pass


        if len(rows_to_del) == img.shape[0]:
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
            crop_img = np.delete(crop_img,cols_to_del,axis=1)
            meta_data.update({'height':crop_img.shape[0],'width':crop_img.shape[1]})
            #print (crop_img.shape)
            #print ('Writing images...')
            save_tif(crop_img,meta_data,f'{dstFolder}/{newFolderName}/{file[:-4]}_cropped.tif')
            print(f'\r{prog} of {tot} saved ({prog/tot:.0%})\tImageID: {file[:-4]}',end='\r')
            prog+=1
            croppedNames.append(file[:-8])
            continue


    # with open(f'{dstFolder}/cropped_image_names.txt','w') as f:
    #     for name in croppedNames:
    #         f.write(f'{name}*\n')

if __name__ == "__main__":
    print ('Select folder of images to crop')
    srcFolder = askdir()
    #srcFolder = 'D:/Data/Ice_Pipeline_Out_7-19-23/rfl_cropped' 
    print (f'{srcFolder} selected\nSelect LOC Folder')
    locFolder = askdir()
    #locFolder = 'D:/Data/Ice_Pipeline_Out_7-19-23/loc_cropped' 
    print (f'{locFolder} selected\nSelect Save Folder')
    saveFolder = askdir()
    #saveFolder = 'D:/Data/Ice_Pipeline_Out_7-19-23'  
    crop(srcFolder,locFolder,saveFolder)
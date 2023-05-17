#%%
import numpy as np
import os
import tifffile as tf
import copy
from tkinter.filedialog import askdirectory as askdir

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

def crop(rflFolderPath,locFolderPath,obsFolderPath):
    rflBasePath = rflFolderPath[:find_all(rflFolderPath,'/')[-1]]
    locBasePath = locFolderPath[:find_all(locFolderPath,'/')[-1]]
    obsBasePath = obsFolderPath[:find_all(obsFolderPath,'/')[-1]]

    cropFolderNames = ['rfl_cropped','loc_cropped','obs_cropped']
    basePathList = [rflBasePath,locBasePath,obsBasePath]
    for basePath,cropFolder in zip(basePathList,cropFolderNames):
        try:
            os.mkdir(f'{basePath}/{cropFolder}')
        except:
            pass

    prog = 1
    tot = len(os.listdir(rflFolderPath))
    for rfl,loc,obs in zip(os.listdir(rflFolderPath),os.listdir(locFolderPath),os.listdir(obsFolderPath)):
        ##Check if files exist
        if f'{rfl[:-4]}_cropped.tif' in os.listdir(f'{rflBasePath}/{cropFolderNames[0]}') and\
            f'{loc[:-4]}_cropped.tif' in os.listdir(f'{locBasePath}/{cropFolderNames[1]}') and\
            f'{obs[:-4]}_cropped.tif' in os.listdir(f'{obsBasePath}/{cropFolderNames[2]}'):
            
            print (f'Image {prog} of {tot} already cropped\tImageID: {rfl[:-8]}')
            prog+=1
            continue
        
        print ('Reading images...')
        im_rfl = tf.imread(f'{rflFolderPath}/{rfl}')
        im_loc = tf.imread(f'{locFolderPath}/{loc}')
        im_obs = tf.imread(f'{obsFolderPath}/{obs}')
        crop_rfl = copy.copy(im_rfl)
        crop_loc = copy.copy(im_loc)
        crop_obs = copy.copy(im_obs)

        print ('Cropping images...')
        rows_to_del = np.where(crop_loc[:,:,1]>-70)[0]
        if len(rows_to_del) == 0:
            rows_to_del = []
        else:
            min_row,max_row = rows_to_del.min(),rows_to_del.max()
            rows_to_del = np.arange(min_row,max_row)

        if len(rows_to_del) == im_rfl.shape[0]-1:
            print (f'Image {prog} of {tot} is outside of target latitude range\tImageID: {rfl[:-8]}')
            prog+=1
            continue

        elif len(rows_to_del)==0:
            print ('Writing images...')
            tf.imwrite(f'{rflBasePath}/{cropFolderNames[0]}/{rfl[:-4]}_cropped.tif',crop_rfl,photometric='rgb')
            tf.imwrite(f'{locBasePath}/{cropFolderNames[1]}/{loc[:-4]}_cropped.tif',crop_loc,photometric='rgb')
            tf.imwrite(f'{obsBasePath}/{cropFolderNames[2]}/{obs[:-4]}_cropped.tif',crop_obs,photometric='rgb')
            print(f'Image {prog} of {tot} is entirely within target latitude range\tImageID: {rfl[:-8]}')
            prog+=1
            continue
        else:
            crop_rfl = np.delete(crop_rfl,rows_to_del,axis=0)
            crop_loc = np.delete(crop_loc,rows_to_del,axis=0)
            crop_obs = np.delete(crop_obs,rows_to_del,axis=0)

            print ('Writing images...')
            tf.imwrite(f'{rflBasePath}/{cropFolderNames[0]}/{rfl[:-4]}_cropped.tif',crop_rfl,photometric='rgb')
            tf.imwrite(f'{locBasePath}/{cropFolderNames[1]}/{loc[:-4]}_cropped.tif',crop_loc,photometric='rgb')
            tf.imwrite(f'{obsBasePath}/{cropFolderNames[2]}/{obs[:-4]}_cropped.tif',crop_obs,photometric='rgb')

            print(f'{prog} of {tot} saved ({prog/tot:.0%})\tImageID: {rfl[:-8]}')
            prog+=1
            continue

if __name__ == "__main__":
    print ('Select RFL Folder')
    rflFolder = 'D:/Data/OP2C_Downloads/L2_sorted/sorted_tif_files/rfl_files' #askdir()
    print (f'{rflFolder} selected\nSelect LOC Folder')
    locFolder = 'D:/Data/OP2C_Downloads/L1_sorted/sorted_tif_files/loc_files' #askdir()
    print (f'{locFolder} selected\nSelect OBS Folder')
    obsFolder = 'D:/Data/OP2C_Downloads/L1_sorted/sorted_tif_files/obs_files' #askdir()
    print (f'{obsFolder} selected')
    crop(rflFolder,locFolder,obsFolder)
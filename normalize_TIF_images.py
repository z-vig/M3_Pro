#%%
import numpy as np
import tifffile as tf
from tkinter.filedialog import askdirectory as askdir
from georeference_TIF import georef
import os

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

def norm_folder(tifFolder:str)->list:
    fileListIn = os.listdir(tifFolder)
    fileListOut = []
    nameListOut = []
    prog,tot = 1,len(fileListIn)
    for file in fileListIn:
        img = tf.imread(f'{tifFolder}/{file}')
        nameListOut.append(file)
        for band in range(img.shape[img.shape.index(min(img.shape))]):
            img[:,:,band] = img[:,:,band]-img[:,:,band].min()
            img[:,:,band] = img[:,:,band]/img[:,:,band].max()
            img[:,:,band] = 255*img[:,:,band]
        fileListOut.append(img)
        print (f'\r{prog} of {tot} normalized ({prog/tot:.0%})',end='\r')
        prog+=1
    
    return fileListOut,nameListOut

if __name__ == '__main__':
    print ('Select tifFolder to Normalize:')
    tifFolder = askdir()
    print ('Normalizing images...')
    normImageList,nameList = norm_folder(tifFolder)
    print ('Saving normalized images...')
    index = find_all(tifFolder,'/')[-1]
    print (f'{tifFolder[:index]}/{tifFolder}_normalized')
    try:
        os.mkdir(f'{tifFolder[:index]}/{os.path.basename(tifFolder)}_normalized')
    except:
        pass
    prog,tot = 0,len(normImageList)
    for file,name in zip(normImageList,nameList):
        tf.imwrite(f'{tifFolder[:index]}/{os.path.basename(tifFolder)}_normalized/{name[:-4]}_normalized.tif',file,photometric='rgb')
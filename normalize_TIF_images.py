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
    prog,tot = 0,len(fileListIn)
    for file in fileListIn:
        img = tf.imread(f'{tifFolder}/{file}')
        for band in range(img.shape[img.shape.index(min(img.shape))]):
            img[:,:,band] = img[:,:,band]-img[:,:,band].min()
            img[:,:,band] = img[:,:,band]/img[:,:,band].max()
            img[:,:,band] = 255*img[:,:,band]
        fileListOut.append(img)
        print (f'\r{prog} of {tot} normalized ({prog/tot:.0%})',end='\r')
    
    return fileListOut

if __name__ == '__main__':
    print ('Select tifFolder to Normalize:')
    tifFolder = askdir()
    print ('Normalizing images...')
    normImageList = norm_folder(tifFolder)
    print ('Saving normalized images...')
    index = find_all(tifFolder,'/')[-1]
    try:
        os.mkdir(f'{tifFolder[:index]}/{tifFolder}_normalized')
    except:
        pass
    prog,tot = 0,len(normImageList)
    for file in normImageList:
        tf.imwrite(f'{tifFolder[:index]}/{tifFolder}_normalized.tif',file,photometric='rgb')
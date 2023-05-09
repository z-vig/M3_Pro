#%%
import os
import shutil
from typing import List
from tkinter.filedialog import askdirectory as askdir

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

def sort(dataFolder,sortedFolder):
    try: ##Making necessary sorting folders
        os.mkdir(f'{sortedFolder}/extracted_files')
    except:
        pass

    for folder in ['hdr_files','lbl_files']:
        try:
            os.mkdir(f'{sortedFolder}/extracted_files/{folder}')
        except:
            pass

    n=0 ##Making image folders within HDR images folder and getting .img and .hdr file paths
    imgIDList = []
    nameList = []
    hdrPathList = []
    imgPathList = []
    for file in os.listdir(f'{dataFolder}'):
        if file[-4:].find('.img')>-1 or file[-4:].find('.hdr')>-1:
            name = file[:-4]
            nameList.append(name)
            imgID = file[3:find_all(file,'_')[0]]
            if imgID not in imgIDList:
                imgIDList.append(imgID)
            if file[-4:].find('.img')>-1:
                imgPathList.append(f'{dataFolder}/{file}')
            elif file[-4:].find('.hdr')>-1:
                hdrPathList.append(f'{dataFolder}/{file}')
            try:
                os.mkdir(f'{sortedFolder}/extracted_files/hdr_files/{name}')
            except:
                pass
    
    print ('Sorting HDR...')
    for hdr in hdrPathList:
        ind = find_all(hdr,'/')[-1]
        fileName = f'{hdr[ind+1:]}'
        folder = f'{hdr[ind+1:][:-4]}'
        shutil.move(hdr,f'{sortedFolder}/extracted_files/hdr_files/{folder}')

    print ('Sorting IMG...')
    for img in imgPathList:
        ind = find_all(img,'/')[-1]
        fileName = f'{img[ind+1:]}'
        folder = f'{img[ind+1:][:-4]}'
        shutil.move(img,f'{sortedFolder}/extracted_files/hdr_files/{folder}')
    
    print ('Sorting Complete!')


if __name__ == '__main__':
    # dataFolder = 'D:/Data/L2_Data_allSP_test2/'
    # sortedFolder = 'D:/Data/L2_sorted/'
    print ('Select data folder')
    dataFolder = askdir()
    print (f'{dataFolder} selected.')
    print ('Select save folder')
    sortedFolder = askdir()
    print (f'{sortedFolder} selected.')
    sort(dataFolder,sortedFolder)
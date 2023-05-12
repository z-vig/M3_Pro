#%%
import spectral as sp
import tifffile as tf
import os
import numpy as np

def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

def extract(folderPath):
    try:
        os.mkdir(f'{folderPath}/sorted_tif_files')
    except:
        pass
    walk = os.walk(folderPath)
    fileTypeList = []
    filePathList = []
    for root,dir,files in walk:
        if root.find('hdr_files')>-1:
            for file in files:
                if file[-4:].find('.hdr')>-1:
                    fileType = file[-7:-4]
                    filePath = f'{root}/{file}'
                    fileTypeList.append(fileType)
                    filePathList.append(filePath)

    for fileType in set(fileTypeList):
        try:
            os.mkdir(f'{folderPath}/sorted_tif_files/{fileType}_files')
        except:
            pass
    
    print ('Folders created!')
    prog = 1
    tot = len(filePathList)
    for filePath in filePathList:
        name_index_start = find_all(filePath,'/')[-1]
        name_index_end = find_all(filePath,'_')[-2]
        stampName = filePath[name_index_start+1:name_index_end]
        stampType = filePath[-7:-4]
        if stampType == 'rfl':
            if f'{stampName}_rfl.tif' not in os.listdir(f'{folderPath}/sorted_tif_files/{stampType}_files'):
                
                hdr = sp.envi.open(filePath)
    
                wvl = np.array(hdr.bands.centers)
                allowedInd = np.where((wvl>900)&(wvl<2600))[0]
                
                img = hdr.read_bands(allowedInd)
                
                tf.imwrite(f'{folderPath}/sorted_tif_files/{stampType}_files/{stampName}_rfl.tif',img.astype('float32'))
            else:
                pass
        elif stampType == 'loc':
            if f'{stampName}_loc.tif' not in os.listdir(f'{folderPath}/sorted_tif_files/{stampType}_files'):
                hdr = sp.envi.open(filePath)
                img = hdr.read_bands([0,1,2])
                tf.imwrite(f'{folderPath}/sorted_tif_files/{stampType}_files/{stampName}_loc.tif',img.astype('float32'))
            else:
                pass
        elif stampType == 'obs':
            if f'{stampName}_obs.tif' not in os.listdir(f'{folderPath}/sorted_tif_files/{stampType}_files'):
                hdr = sp.envi.open(filePath)
                img = hdr.read_bands([0,1,2,3,4,5,6,7,8,9])
                tf.imwrite(f'{folderPath}/sorted_tif_files/{stampType}_files/{stampName}_obs.tif',img.astype('float32'))
            else:
                pass
        else:
            pass
        #print (f'{prog} of {tot}')
        print (f'\r{prog} of {tot} files extracted ({prog/tot:.0%})',end='\r')
        prog+=1

if __name__ == "__main__":
    l1_folder = 'D:/Data/OP2C_Downloads/L1_sorted'
    l2_folder = 'D:/Data/OP2C_Downloads/L2_sorted'
    print (f'Extracting files from {l1_folder}...')
    extract(l1_folder)
    print (f'Extracting files from {l2_folder}...')
    extract(l2_folder)
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import Locate_Ice
import M3_UnZip

#%%
def find_multi(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

locateIceSavesPath = r'E:/Data/Locate_Ice_Saves'

class Water_Image():
    def __init__(self,smoothPath,waterPath) -> None:
        self.smoothPath = smoothPath
        self.waterPath = waterPath
        self.smoothCube = np.load(smoothPath)
        self.waterCube = np.load(waterPath)

    @property
    def fileID(self):
        fileID_slice = slice(*find_multi(self.smoothPath,'\\')[-2:],1) #Getting fileID location in path
        self._fileID = path[fileID_slice][1:] #Removing \ at the beginning
        return self._fileID

    def find_water_pixels(self):
        self.waterCoords = np.where(self.waterCube[:,:,0]==1)[0],np.where(self.waterCube[:,:,0]==1)[1]
        return self.waterCoords
    
    def get_water_spectra(self):
        x,y = self.waterCoords[0],self.waterCoords[1]
        self.water_spectra = self.smoothCube[x,y,:]
        return self.water_spectra
    
    def calculate_band_depth(self):
        fig = plt.figure()
        

        
#%%
waterImagePathList = []
smoothImagePathList = []
for root,dirs,files in os.walk(locateIceSavesPath):
    for file in files:
        if root.find('2009')>-1 and file == 'Water_Locations.npy':
            path = os.path.join(root,file)
            fileID_slice = slice(*find_multi(path,'\\')[-2:],1) #Getting fileID location in path
            fileID = path[fileID_slice][1:] #Removing \ at the beginning
            waterImagePathList.append(path)
            #print (f'Image {fileID}: {os.path.join(root,file)}')
        elif root.find('2009')>-1 and file == "Smooth_Spectrum_Image.npy":
            path = os.path.join(root,file)
            smoothImagePathList.append(path)



#%%
import spectral as sp
import random

hdr = sp.envi.open(r'E:\Data\20230209T095534013597\extracted_files\hdr_files\m3g20090417t193320_v01_rfl\m3g20090417t193320_v01_rfl.hdr')
wavelengths = np.array(hdr.bands.centers)
allowedIndices = np.where((wavelengths>900)&(wavelengths<2600))
allowedWavelengths = wavelengths[allowedIndices]

for waterPath,smoothPath in zip(waterImagePathList,smoothImagePathList):
    waterImageObject = Water_Image(smoothPath,waterPath)
    fileID = waterImageObject.fileID
    waterCoords = waterImageObject.find_water_pixels()
    waterCoordArray = np.array(waterCoords)
    waterSpectra = waterImageObject.get_water_spectra()
    
    plotPerImage = 2
    get_random_index = random.sample(range(waterSpectra.shape[0]),plotPerImage)
    fig = plt.figure()
    for index in get_random_index:
        plt.plot(allowedWavelengths,waterSpectra[index,:],label=f'{waterCoordArray[0,index]},{waterCoordArray[1,index]}')
    plt.title(f'Image: {fileID}')
    plt.legend()
plt.show()

    
#%%


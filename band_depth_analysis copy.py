#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import spec_plotting

#%%
def find_multi(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

locateIceSavesPath = r'E:/Data/Locate_Ice_Saves'
## Defining image class
class Water_Mosaic():
    def __init__(self,iceDataPath) -> None:
        self.iceDataPath = iceDataPath

    @property
    def fileIDList(self):
        self._fileIDList = []
        for root,dirs,files in os.walk(self.iceDataPath):
            for file in files:
                if root.find('2009')>-1 and file.find('npy')>-1:
                    fileID_Index = find_multi(root,'\\')
                    file_ID = root[fileID_Index[-1]+1:]
                    #print (file_ID)
                    if file_ID not in self._fileIDList:
                        self._fileIDList.append(file_ID) #Removing \ at the beginning
        
        return self._fileIDList
    
    @property
    def imagePathDictionary(self):
        imagePath_dict = {}
        for root,dirs,files in os.walk(self.iceDataPath):
            for file in files:
                if root.find('2009')>-1 and file.find('.npy')>-1:
                    if file[:-4] not in imagePath_dict.keys():
                        imagePath_dict.update({file[:-4]:[os.path.join(root,file)]})
                    else:
                        imagePath_dict.get(file[:-4]).append(os.path.join(root,file))

        return imagePath_dict
    
    @property
    def waterArrayList(self):
        return [np.load(i) for i in self.imagePathDictionary.get('Water_Locations')]

    def get_water_coordinates(self):
        coordinateList = []
        for array in self.waterArrayList:
            coordinateList.append(np.array(np.where(array[:,:,0]==1)))
        return coordinateList
    
    def get_water_spectra(self):
        x,y = self.waterCoords[0],self.waterCoords[1]
        self.water_spectra = self.smoothCube[x,y,:]
        return self.water_spectra
    
    def get_water_spectra_original(self):
        x,y = self.waterCoords[0],self.waterCoords[1]
        self.water_spectra_original = self.originalCube[x,y,:]
        return self.water_spectra_original
    
    def get_destripe_spectra(self):
        x,y = self.waterCoords[0],self.waterCoords[1]
        self.water_spectra_destripe = self.destripeCube[x,y,:]
        return self.water_spectra_destripe
    
    def calculate_spectral_angle(self):
        minLocate = np.array(([1.242,1.323],[1.503,1.659],[1.945,2.056]))
        shoulderLocate = np.array(([1.13,1.35],[1.42,1.74],[1.82,2.2]))
        

        
#%%
##Getting Paths

obj = Water_Mosaic(r'E:\Data\Locate_Ice_Saves')
fileIDList = obj.fileIDList
imagePathDictionary = obj.imagePathDictionary
for i in obj.get_water_coordinates():
    print (type(i),i.shape)


water_coordinates = obj.get_water_coordinates()[0]

testim = obj.waterArrayList[0][:,:,0]
fig = plt.figure()
plt.imshow(testim)

cross_check = np.zeros(testim.shape)
cross_check[water_coordinates[0,:],water_coordinates[1,:]] = 1
fig = plt.figure()
plt.imshow(cross_check)
plt.show()

print (imagePathDictionary.keys())

                


    
#%%


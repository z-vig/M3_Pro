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
        _fileIDList = []
        for root,dirs,files in os.walk(self.iceDataPath):
            for file in files:
                if root.find('2009')>-1 and file.find('npy')>-1:
                    fileID_Index = find_multi(root,'\\')
                    file_ID = root[fileID_Index[-1]+1:]
                    #print (file_ID)
                    if file_ID not in _fileIDList:
                        _fileIDList.append(file_ID) #Removing \ at the beginning
        
        return _fileIDList
    
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
    def waterArrayDict(self) -> dict:
        waterArrayList = [np.load(i) for i in self.imagePathDictionary.get('Water_Locations')]
        _waterArrayDict = {}
        for array,arrayName in zip(waterArrayList,self.fileIDList):
            _waterArrayDict.update({arrayName:array})
        
        return _waterArrayDict

    @property
    def waterCoordinates(self) -> dict:
        coordinateDict = {}
        for array,arrayName in zip(self.waterArrayDict.values(),self.fileIDList):
            coordinateDict.update({arrayName:np.array(np.where(array[:,:,0]==1))})
        return coordinateDict
    
    def get_spectra(self,**kwargs):
        defaultKwargs = {'spectraType':'Original_Image','imageID':['2009-04-17_19-33-20']}
        kwargs = {**defaultKwargs,**kwargs}

        if kwargs.get('imageID') == 'all':
            imageID_list = self.fileIDList
        else:
            imageID_list = kwargs.get('imageID')

        spectrumImageDict = {}
        imageNum = 1
        for imageID in imageID_list:
            print (f'\rCreating water spectrum image for {imageID}... ({imageNum/len(imageID_list):.0%})',end='\r')
            xWater,yWater = self.waterCoordinates.get(imageID)[0,:],self.waterCoordinates.get(imageID)[1,:]

            spectraType = kwargs.get('spectraType')
            spectrumImage_list = imagePathDictionary.get(spectraType) #Gets list of all images with spectrum type "spectraType"
            spectrumImage = np.load(spectrumImage_list[[index for index,ID in enumerate(imageID_list) if imageID in ID][0]]) #Gets image defined by "imageID"
            spectrumImageDict.update({imageID:spectrumImage})

            imageNum+=1
        
        return spectrumImageDict

    def calculate_spectral_angle(self):
        minLocate = np.array(([1.242,1.323],[1.503,1.659],[1.945,2.056]))
        shoulderLocate = np.array(([1.13,1.35],[1.42,1.74],[1.82,2.2]))
        

        
#%%
##Getting Paths

obj = Water_Mosaic(r'E:\Data\Locate_Ice_Saves')
fileIDList = obj.fileIDList
imagePathDictionary = obj.imagePathDictionary
# for i in obj.get_water_coordinates():
#     print (type(i),i.shape)

# print (obj.waterArrayDict.keys())
# for value in obj.waterArrayDict.values():
#     print (value.shape)

# print (obj.waterCoordinates.keys())
# for value in obj.waterCoordinates.values():
#     print (value.shape)

waterSpectra_original_0417 = obj.get_spectra(spectraType='Original_Image',imageID='all')
print (waterSpectra_original_0417.keys())
for value in waterSpectra_original_0417.values():
    print (value.shape)



# testim = obj.waterArrayDict.get('2009-04-17_19-33-20')[:,:,0]
# fig = plt.figure()
# plt.imshow(testim)

# water_coordinates = obj.waterCoordinates.get('2009-04-17_19-33-20')
# cross_check = np.zeros(testim.shape)
# cross_check[water_coordinates[0,:],water_coordinates[1,:]] = 1
# fig = plt.figure()
# plt.imshow(cross_check)
# plt.show()

                


    
#%%


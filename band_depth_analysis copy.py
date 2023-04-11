#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import spec_plotting
import spectral as sp
import tifffile as tf

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
    
    @property
    def waterBandDict(self) -> pd.DataFrame:
        _waterBandDict = {}
        
        n=0
        for root,dirs,files in os.walk(r'E:/Data/Locate_Ice_Saves'):
            if root.find('2009')>-1:
                fileID = self.fileIDList[n]
                n+=1
                for file in files:
                    if file.find('water_locations.csv')>-1:
                        waterDf = pd.read_csv(os.path.join(root,file))
                        _waterBandDict.update({fileID:waterDf})
            
        return _waterBandDict

    
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
            spectrumImageDict.update({imageID:spectrumImage[xWater,yWater]})

            imageNum+=1
        
        return spectrumImageDict
    
    def get_band_images(self,smoothed_waterSpectra_dict,allowedWvl):
        shoulderValues = np.array(([1130,1350],[1420,1740],[1820,2200]))
        shoulderValues_exact = np.zeros((3,2))
        n=0
        for Ra,Rc in zip(shoulderValues[:,0],shoulderValues[:,1]):
            Ra_wvl_list = [abs(Ra-index) for index in allowedWvl]
            Rc_wvl_list = [abs(Rc-index) for index in allowedWvl]
            shoulderValues_exact[n,:]=allowedWvl[np.where((Ra_wvl_list==min(Ra_wvl_list))|(Rc_wvl_list==min(Rc_wvl_list)))]
            n+=1

        self.bandImages_dict = {}
        for stamp,stampName in zip(smoothed_waterSpectra_dict.values(),smoothed_waterSpectra_dict.keys()):
            waterDf = obj.waterBandDict.get(stampName)
            waterDf = waterDf.sort_values(['x','y'])
            band1,band2,band3 = waterDf.iloc[:,2],waterDf.iloc[:,3],waterDf.iloc[:,4]

            Rc_wvlIndices = np.zeros((band1.shape[0],3))
            for row in range(Rc_wvlIndices.shape[0]):
                Rc_wvlIndices[row,:] = np.where((allowedWvl==band1[row])|(allowedWvl==band2[row])|(allowedWvl==band3[row]))[0].astype(int)
            
            Ra_wvlIndices = np.where((allowedWvl==shoulderValues_exact[0,0])|(allowedWvl==shoulderValues_exact[1,0])|(allowedWvl==shoulderValues_exact[2,0]))[0]
            Rb_wvlIndices = np.where((allowedWvl==shoulderValues_exact[0,1])|(allowedWvl==shoulderValues_exact[1,1])|(allowedWvl==shoulderValues_exact[2,1]))[0]
            
            Ra_wvlIndices = np.tile(Ra_wvlIndices,(Rc_wvlIndices.shape[0],1))
            Rb_wvlIndices = np.tile(Rb_wvlIndices,(Rc_wvlIndices.shape[0],1))

            '''
            Values of dictionary are a tuple : 
            (array with columns {Rc1,Rc2,Rc3,Ra1,Ra2,Ra3,Rb1,Rb2,Rb3}, 
                            array with columns {[Rc_wvlIndices],[Ra_wvlIndices],[Rb_wvlIndices]})
            '''

            rValueArray = np.zeros((Rc_wvlIndices.shape[0],9))
            rIndexArray = np.concatenate([Rc_wvlIndices,Ra_wvlIndices,Rb_wvlIndices],axis=1)
            for row in range(Rc_wvlIndices.shape[0]):
                addRow_RValues = stamp[row,np.concatenate([Rc_wvlIndices[row,:],Ra_wvlIndices[row,:],Rb_wvlIndices[row,:]]).astype(int)]
                rValueArray[row,:] = addRow_RValues

            self.bandImages_dict.update({stampName:(rValueArray,rIndexArray)})

        return self.bandImages_dict       
    
    def calculate_band_depth(self,**kwargs):
        defaultKwargs = {'plot':False}
        kwargs = {**defaultKwargs,**kwargs}

        #print (shoulderValues_exact[0,0])
        bandDepthImage_dict = {}
        for imageTuple,stampName in zip(self.bandImages_dict.values(),self.bandImages_dict.keys()):
            
            bandImage,wvlIndices = imageTuple


            Rc_bands,Ra_bands,Rb_bands = [0,1,2],[3,4,5],[6,7,8]
            Rc_wvlIndices,Ra_wvlIndices,Rb_wvlIndices = wvlIndices[:,Rc_bands].astype(int),wvlIndices[:,Ra_bands].astype(int),wvlIndices[:,Rb_bands].astype(int)

            Rc = bandImage[:,Rc_bands]
            Ra = bandImage[:,Ra_bands]
            Rb = bandImage[:,Rb_bands]
            #print (f'wvlIndices:{wvlIndices.shape},Rc:{Rc_wvlIndices.shape}')
            b = (allowedWvl[Rc_wvlIndices]-allowedWvl[Ra_wvlIndices])/(allowedWvl[Rb_wvlIndices]-allowedWvl[Ra_wvlIndices])
            a = 1-b
            
            Rc_star = a*Ra+b*Rb

            bandDepths = 1-(Rc/Rc_star)

            bandDepthImage_dict.update({stampName:bandDepths})
        
            waterDf = obj.waterBandDict.get(stampName)
            waterDf = waterDf.sort_values(['x','y'])
            #print (f'a:{a},b:{b}')
            if kwargs.get('plot')==True:
                test_spectra = waterSpectra_smooth.get('2009-04-17_19-33-20')[0,:]
                plt.plot(allowedWvl,waterSpectra_corrected.get('2009-04-17_19-33-20')[0,:])
                plt.plot(allowedWvl,test_spectra)
                plt.vlines(waterDf.iloc[0,2:],test_spectra.min(),test_spectra.max(),ls='-.',color='red')
                plt.vlines([1242,1323,1503,1659,1945,2056],test_spectra.min(),test_spectra.max(),ls='-.',color='k')
                #plt.xlim(1000,1450)
                plt.scatter(allowedWvl[self.Rc_wvlIndices],test_spectra[self.Rc_wvlIndices],marker='x',color='k')
                plt.scatter(allowedWvl[self.Ra_wvlIndices],test_spectra[self.Ra_wvlIndices],marker='x',color='green')
                plt.scatter(allowedWvl[self.Rb_indices],test_spectra[self.Rb_indices],marker='x',color='purple')
                plt.scatter(allowedWvl[self.Rc_wvlIndices],Rc_star,marker='x')
                for i in range(3):
                    plt.plot([allowedWvl[self.Ra_wvlIndices][i],allowedWvl[self.Rb_indices][i]],[test_spectra[self.Ra_wvlIndices][i],test_spectra[self.Rb_indices][i]])

        return bandDepthImage_dict

    def calculate_spectral_angle(self):
        minLocate = np.array(([1.242,1.323],[1.503,1.659],[1.945,2.056]))
        shoulderLocate = np.array(([1.13,1.35],[1.42,1.74],[1.82,2.2]))
        

##Getting Paths

obj = Water_Mosaic(r'E:\Data\Locate_Ice_Saves')
fileIDList = obj.fileIDList
imagePathDictionary = obj.imagePathDictionary


#%%
waterSpectra_original = obj.get_spectra(spectraType='Original_Image',imageID='all')
print (waterSpectra_original.keys())
for value in waterSpectra_original.values():
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
waterSpectra_smooth = obj.get_spectra(spectraType='Smooth_Spectrum_Image',imageID='all')
print (waterSpectra_smooth.keys())
for value in waterSpectra_smooth.values():
    print (value.shape)
#%%
waterSpectra_corrected = obj.get_spectra(spectraType='Correced_Image',imageID='all')
print (waterSpectra_corrected.keys())
for value in waterSpectra_corrected.values():
    print (value.shape)

    
#%%
hdr = sp.envi.open(r'E:\Data\20230209T095534013597\extracted_files\hdr_files\m3g20090417t193320_v01_rfl\m3g20090417t193320_v01_rfl.hdr')
wvl = np.array(hdr.bands.centers)
allowedIndices = np.where((wvl>900)&(wvl<2600))
allowedWvl = wvl[allowedIndices]
image = hdr.read_bands(allowedIndices[0])
print (image.shape)

def plot_spec(coordNum):
    fig = plt.figure()
    coordinate = obj.waterCoordinates.get('2009-04-17_19-33-20')[:,coordNum]
    plt.plot(allowedWvl,waterSpectra_corrected.get('2009-04-17_19-33-20')[coordNum,:])
    #plt.plot(allowedWvl,image[155,265,:])
    plt.plot(allowedWvl,waterSpectra_smooth.get('2009-04-17_19-33-20')[coordNum,:])
    ymin = waterSpectra_corrected.get('2009-04-17_19-33-20')[coordNum,:].min()
    ymax = waterSpectra_corrected.get('2009-04-17_19-33-20')[coordNum,:].max()
    plt.vlines([1242,1323,1503,1659,1945,2056],ymin,ymax,ls='-.',color='k')
    plt.title(f'({coordinate[0]},{coordinate[1]})')

plot_spec(57)
plot_spec(0)
plot_spec(3)

#%%
for row in range(waterSpectra_smooth.get('2009-04-17_19-33-20').shape[0]):
    spectrum = waterSpectra_smooth.get('2009-04-17_19-33-20')[row,:]



#%%
bandImages_dict = obj.get_band_images(waterSpectra_smooth,allowedWvl)
print (bandImages_dict.keys())
for i in bandImages_dict.values():
    print(i[0].shape,i[1].shape)

bandDepthImage_dict = obj.calculate_band_depth()
print (bandDepthImage_dict.keys())
for i in bandDepthImage_dict.values():
    print (i.shape)

#%%
bandDepthTestImg = bandDepthImage_dict.get('2009-04-17_19-33-20')
originalImg = waterSpectra_original.get('2009-04-17_19-33-20')
correctedImg = waterSpectra_corrected.get('2009-04-17_19-33-20')
smoothImg = waterSpectra_smooth.get('2009-04-17_19-33-20')
waterCoords = obj.waterCoordinates.get('2009-04-17_19-33-20')

test_img = np.zeros(hdr.read_band(0).shape).astype('float32')
test_img[waterCoords[0],waterCoords[1]] = bandDepthTestImg[:,0]

fig,(ax1,ax2,ax3) = plt.subplots(3,1)
ax1.plot(bandDepthTestImg[:,0])
ax2.plot(bandDepthTestImg[:,1])
ax3.plot(bandDepthTestImg[:,2])

fig= plt.figure()

x,y = np.where(test_img==test_img[np.argpartition(test_img,-5)][-5:])
x,y = x[0],y[0]
coordNumber = np.where((waterCoords[0,:]==x)&(waterCoords[1,:]==y))[0][0]
print (f'Max Band Depth Location: {x,y} \n CoordNumber: {coordNumber}')

plt.plot(allowedWvl,originalImg[coordNumber,:],label='Original')
plt.plot(allowedWvl,correctedImg[coordNumber,:],label='Corrected')
plt.plot(allowedWvl,smoothImg[coordNumber,:],label='Cubic Spline')
plt.legend()
tf.imwrite(f'E:/Data/Figures/test_1.25BandDepth.tif',test_img)


#%%
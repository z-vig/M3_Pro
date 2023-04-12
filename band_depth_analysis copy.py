#%%
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import spec_plotting
import spectral as sp
import tifffile as tf
import get_n_extrema
importlib.reload(get_n_extrema)
from get_n_extrema import get_nmax
from get_n_extrema import get_nmin
import get_USGS_H2OFrost
importlib.reload(get_USGS_H2OFrost)
from get_USGS_H2OFrost import get_USGS_H2OFrost

#%%
def find_multi(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

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

            #bandDepths = 1-(Rc/Rc_star)
            bandDepths = Rc_star-Rc

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

    def calculate_spectral_angle(self,smoothed_waterSpectra_dict,allowedWvl):
        usgs_spec = get_USGS_H2OFrost()
        specAngle_dict = {}
        for stamp,stampName in zip(smoothed_waterSpectra_dict.values(),smoothed_waterSpectra_dict.keys()):
            specAngle_array = np.zeros(stamp.shape[0])
            rowNum = 0
            #print (f'Stamp:{stamp.shape},usgs: {usgs_spec.shape}')
            for stampRow in stamp:
                stampNorm,usgsNorm = np.sqrt(np.sum(stampRow**2)),np.sqrt(np.sum(usgs_spec**2))
                dotProd = np.dot(stampRow,usgs_spec)
                specAngle = np.arccos(dotProd/(stampNorm*usgsNorm))
                specAngle_array[rowNum] = specAngle*180/np.pi
                rowNum += 1
            specAngle_dict.update({stampName:specAngle_array})

        return specAngle_dict

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
def norm_arrayDict(arrDict:dict)->dict:
    newDict = {}
    for stamp,stampName in zip(arrDict.values(),arrDict.keys()):
        normStamp = (stamp-stamp.min(axis=1)[:,np.newaxis])/ \
            (stamp-stamp.min(axis=1)[:,np.newaxis]).max(axis=1)[:,np.newaxis]
        newDict.update({stampName:normStamp})
    return newDict

originalDict_norm = norm_arrayDict(waterSpectra_original)
correctedDict_norm = norm_arrayDict(waterSpectra_corrected)
smoothDict_norm = norm_arrayDict(waterSpectra_smooth)

#%%
bandImages_dict = obj.get_band_images(smoothDict_norm,allowedWvl)
print (bandImages_dict.keys())
for i in bandImages_dict.values():
    print(i[0].shape,i[1].shape)

bandDepthImage_dict = obj.calculate_band_depth()
print (bandDepthImage_dict.keys())
for i in bandDepthImage_dict.values():
    print (i.shape)

#%%
def get_bandDepth_image(imageID,displayMax=3,**kwargs):
    defaultKwargs = {'saveImage':False,'analyzedBand':1.25}
    kwargs = {**defaultKwargs,**kwargs}

    print (f'Getting Data for {imageID}...')
    bandDepthTestImg = bandDepthImage_dict.get(imageID)
    originalImg = waterSpectra_original.get(imageID)
    correctedImg = waterSpectra_corrected.get(imageID)
    smoothImg = waterSpectra_smooth.get(imageID)
    waterCoords = obj.waterCoordinates.get(imageID)

    print (f'Plotting {imageID}...')
    fig = plt.figure()
    plt.hist(bandDepthTestImg[:,0],20)
    plt.title(f'Image <{imageID}> Band Depth Distribution')
    
    imageShape = np.load(os.path.join(r'E:/Data/Locate_Ice_Saves/'+imageID+'/Original_Image.npy')).shape
    BD_img = np.zeros((imageShape[0],imageShape[1],3)).astype('float32')
    BD_img[waterCoords[0],waterCoords[1]] = bandDepthTestImg
    if kwargs.get('saveImage') == True:
        tf.imwrite(f'E:/Data/Figures/BandDepthImage_{imageID}.tif',photometric='rgb')

    if kwargs.get('analyzedBand')==1.25:   
        x,y = get_nmax(BD_img[:,:,0],displayMax)
        print (f'Max Values are {BD_img[x,y,0]}')
        max_loc = [(x,y) for x,y in zip(x,y)]
        print (f'Max Values Located at {max_loc}')

    #coordNumber = np.where((waterCoords[0,:]==x)&(waterCoords[1,:]==y))[0][0]
    coordNumber = [np.where((waterCoords[0,:]==max[0])&(waterCoords[1,:]==max[1]))[0][0] \
                for max in [(x,y) for x,y in zip(x,y)]]
    print (f'Max Band Depth Location: {x,y}\nCoordNumber: {coordNumber}')

    fig,axList = plt.subplots(displayMax,1,figsize=(8,displayMax*4.5))
    n = 1
    for ax,num,coord in zip(axList,coordNumber,max_loc):
        #ax.plot(allowedWvl,originalImg[num,:],label='Original')
        ax.plot(allowedWvl,correctedImg[num,:],label='Corrected')
        ax.plot(allowedWvl,smoothImg[num,:],label='Cubic Spline')
        ax.legend()
        ax.set_title(f'Image <{imageID}>, Point [{coord}] ({n}/{displayMax})')
        n+=1

    return BD_img

for imageID in obj.fileIDList:
    BD_img = get_bandDepth_image(imageID,displayMax=10)
    #tf.imwrite(f'E:/Data/Figures/1.25um_BandDepth_NonRatio.tif',BD_img,photometric='rgb')
    break
#%%
specAngle_dict = obj.calculate_spectral_angle(waterSpectra_smooth,allowedWvl)
def get_specAngle_image(imageID,displayMin=3,**kwargs):
    defaultKwargs = {}
    kwargs = {**defaultKwargs,**kwargs}

    specAngle_water = specAngle_dict.get(imageID)
    smooth_water = waterSpectra_smooth.get(imageID)
    corrected_water = waterSpectra_corrected.get(imageID)

    fig,ax = plt.subplots(1,1)
    ax.hist(specAngle_water,15)
    ax.set_title(f'{imageID} Spectral Angle Distribution')

    waterCoords = obj.waterCoordinates.get(imageID)
    print ('Loading image Shape...')
    imageShape = np.load(os.path.join(r'E:/Data/Locate_Ice_Saves/'+imageID+'/Original_Image.npy')).shape
    
    SA_img = np.zeros((imageShape[0],imageShape[1])).astype('float32')
    SA_img[waterCoords[0],waterCoords[1]] = specAngle_water
    print ('Getting min...')
    print (SA_img)
    x,y = get_nmin(SA_img,displayMin)
    print ('Min obtained!')
    max_loc = np.array([(x,y) for x,y in zip(x,y)])
    #print (np.flip(np.sort(maxArray,axis=0),axis=0))
    
    coordNumber = [np.where((waterCoords[0,:]==max[0])&(waterCoords[1,:]==max[1]))[0][0] \
                   for max in [(x,y) for x,y in zip(x,y)]]

    fig,axList = plt.subplots(displayMin,1,figsize=(8,displayMin*4.5))
    n = 1
    for ax,num,coord in zip(axList,coordNumber,max_loc):
        print (coord)
        SA_val = SA_img[coord[0],coord[1]]
        #ax.plot(allowedWvl,originalImg[num,:],label='Original')
        ax.plot(allowedWvl,corrected_water[num,:],label='Corrected')
        ax.plot(allowedWvl,smooth_water[num,:],label='Cubic Spline')
        ax.legend()
        ax.set_title(f'Image: {imageID[0:10]}, Point: {coord}, SA: {SA_val:.0f} ({n}/{displayMin})')
        n+=1

    return SA_img


for imageID in obj.fileIDList:
    print (imageID)
    SA_img = get_specAngle_image(imageID,displayMin=10)
    tf.imwrite(r'E:/Data/Figures/SpectralAngle_Image.tif',SA_img,photometric='minisblack')
    break
    


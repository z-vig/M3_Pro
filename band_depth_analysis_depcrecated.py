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
class Water_Image():
    def __init__(self,smoothPath,waterPath,originalPath,destripePath) -> None:
        self.smoothPath = smoothPath
        self.waterPath = waterPath
        self.originalPath = originalPath
        self.destripePath = destripePath
        self.smoothCube = np.load(smoothPath)
        self.waterCube = np.load(waterPath)
        self.originalCube = np.load(originalPath)
        self.destripeCube = np.load(destripePath)

    @property
    def fileID(self):
        fileID_slice = slice(*find_multi(self.smoothPath,'\\')[-2:],1) #Getting fileID location in path
        self._fileID = self.waterPath[fileID_slice][1:] #Removing \ at the beginning
        return self._fileID

    def find_water_pixels(self):
        self.waterCoords = np.where(self.waterCube[:,:,0]==1)[0],np.where(self.waterCube[:,:,0]==1)[1]
        return self.waterCoords
    
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
waterImagePathList = []
smoothImagePathList = []
originalImagePathList = []
destripeImagePathList = []
for root,dirs,files in os.walk(locateIceSavesPath):
    fileType_list = []
    for file in files:
        if root.find('2009')>-1 and file.find('.npy')>-1:
            if file not in fileType_list:
                print (root,file)
                fileType_list.append(file)

 

#print (smoothImagePathList)

#%%
import spectral as sp
import random
import matplotlib.cm as cm
import matplotlib.ticker as tck
import scipy.interpolate as interp
import spectrum_averaging

hdr = sp.envi.open(r'E:\Data\20230209T095534013597\extracted_files\hdr_files\m3g20090417t193320_v01_rfl\m3g20090417t193320_v01_rfl.hdr')
wavelengths = np.array(hdr.bands.centers)
allowedIndices = np.where((wavelengths>900)&(wavelengths<2600))
allowedWavelengths = wavelengths[allowedIndices]

averageWater = np.zeros((14,59))
unfilteredAverageWater = np.zeros((14,59))
correctedAverageWater = np.zeros((14,59))
fileIDList = []
imageNumber = 0

for waterPath,smoothPath,originalPath,destripePath in zip(waterImagePathList,smoothImagePathList,originalImagePathList,destripeImagePathList):
    waterImageObject = Water_Image(smoothPath,waterPath,originalPath,destripePath)
    file_ID = waterImageObject.fileID
    fileIDList.append(file_ID)
    print(f'\rPlotting {waterImageObject.fileID}... {(imageNumber+1)/14:.0%}',end='\r')
    waterCoords = waterImageObject.find_water_pixels()
    waterCoordArray = np.array(waterCoords)
    waterSpectra = waterImageObject.get_water_spectra()
    waterSpectraUnfiltered = waterImageObject.get_water_spectra_original()
    waterSpectraCorrected = waterImageObject.get_destripe_spectra()
    ##Normaling Reflectance Values
    # for spectrum in range(waterSpectra.shape[0]):
    #     waterSpectra[spectrum,:] = waterSpectra[spectrum,:]-waterSpectra[spectrum,:].min()
    #     waterSpectra[spectrum,:] = waterSpectra[spectrum,:]/waterSpectra[spectrum,:].max()
        #print (f'Max: {waterSpectra[spectrum,:].max()},Min: {waterSpectra[spectrum,:].min()}')

    
    plotPerImage = 5
    cmap = cm.viridis
    colors = cmap(np.arange(0,plotPerImage,1)/plotPerImage)
    if waterSpectra.shape[0]<plotPerImage:
        get_random_index = random.sample(range(waterSpectra.shape[0]),waterSpectra.shape[0])
    else:
        get_random_index = random.sample(range(waterSpectra.shape[0]),plotPerImage)
    fig,ax = plt.subplots(1,1,figsize=(15,10))
    maxReflectance = 0
    for index,color in zip(get_random_index,colors):
        if waterSpectra[index,:].max() > maxReflectance:
            maxReflectance = waterSpectra[index,:].max()
        spec_plotting.fancy_spec_plot(fig,ax,allowedWavelengths,waterSpectra[index,:],
                                      title=f'Randomly Selected Water-Like Spectra from Image: {file_ID}',
                                      line_color=color,legend=False)
    
    averageWater[imageNumber,:] = np.average(waterSpectra,axis=0)
    unfilteredAverageWater[imageNumber,:] = np.average(waterSpectraUnfiltered,axis=0)
    correctedAverageWater[imageNumber,:] = np.average(waterSpectraCorrected,axis=0)
    
    ytick_locations = np.arange(0,round(maxReflectance+0.1*maxReflectance,2),
                                round(((maxReflectance+0.1*maxReflectance)/3),3)).round(2)
    ax.set_yticks(ytick_locations,labels=ytick_locations,fontname="Times New Roman")
    ax.yaxis.set_minor_locator(tck.MultipleLocator(0.01))
    ax.vlines([1242,1323],0,maxReflectance,ls='-.',color='blue')
    ax.vlines([1503,1659],0,maxReflectance,ls='-.',color='orange')
    ax.vlines([1945,2056],0,maxReflectance,ls='-.',color='red')
    ##[1.242,1.323],[1.503,1.659],[1.945,2.056]
    ax.text(1250,maxReflectance,'1250\u03BCm')
    ax.text(1500,maxReflectance,'1500\u03BCm')
    ax.text(1900,maxReflectance,'1900\u03BCm')
    ax.set_title(f'Randomly Selected Water-Like Spectra from Image: {file_ID}')
    imageNumber+=1
    plt.close()
    #plt.savefig(f'E:/Data/Figures/Rnadom Water Spectra/{file_ID}')plt.close()



water = pd.read_csv(r"D:/Data/USGS_Water_Ice/splib07a_H2O-Ice_GDS136_77K_BECKa_AREF.txt")
wavelengths = pd.read_csv(r"D:/Data/USGS_Water_Ice/splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt")
water.columns = ['']
wavelengths.columns = ['']

goodIndices = np.where(water>0)[0]
#print (goodIndices)

wvl,rfl = wavelengths.iloc[goodIndices,0]*1000,water.iloc[goodIndices,0]
print (f'Water Length: {water.shape}, Wavelength Length: {wavelengths.shape}')
f = interp.CubicSpline(wvl,rfl)
xtest = np.linspace(wvl.min(),wvl.max(),90)
usgsMin = f(xtest).min()
usgsMax = f(xtest).max()

for i,fileID in zip(range(averageWater.shape[0]),fileIDList):

    normalizedDestripe = correctedAverageWater[i,:] - correctedAverageWater[i,:].min()
    normalizedDestripe = normalizedDestripe/normalizedDestripe.max()
    avg_Destripe,std_Destripe,wvl_Destripe = spectrum_averaging.spec_avg(correctedAverageWater[i,:],allowedWavelengths,5)
    avg_NormDestripe,std_NormDestripe,wvl_NormDestripe = spectrum_averaging.spec_avg(normalizedDestripe,allowedWavelengths,5)
    f_destripe = interp.CubicSpline(np.array(wvl_Destripe),np.array(avg_Destripe))
    f_destripe_norm = interp.CubicSpline(np.array(wvl_NormDestripe),np.array(avg_NormDestripe))
    x_destripe = np.linspace(np.array(wvl_Destripe).min(),np.array(wvl_Destripe).max(),100)

    normalizedUnfilteredAverage = unfilteredAverageWater[i,:] - unfilteredAverageWater[i,:].min()
    normalizedUnfilteredAverage = normalizedUnfilteredAverage/normalizedUnfilteredAverage.max()
    avg_UA,std_UA,wvl_UA = spectrum_averaging.spec_avg(unfilteredAverageWater[i,:],allowedWavelengths,5)
    avg_NUA,std_NUA,wvl_NUA = spectrum_averaging.spec_avg(normalizedUnfilteredAverage,allowedWavelengths,5)
    f1 = interp.CubicSpline(np.array(wvl_NUA),np.array(avg_NUA))
    xtest1 = np.linspace(np.array(wvl_NUA).min(),np.array(wvl_NUA).max(),100)
    f1_unfiltered = interp.CubicSpline(np.array(wvl_UA),np.array(avg_UA))

    normalizedAverage = averageWater[i,:] - averageWater[i,:].min()
    normalizedAverage = normalizedAverage/normalizedAverage.max()

    normalizedUSGS = f(xtest)-usgsMin
    normalizedUSGS = normalizedUSGS/usgsMax

    fig = plt.figure()
    plt.plot(allowedWavelengths,normalizedAverage,label = 'Filtered Average Water')
    plt.plot(allowedWavelengths,normalizedUnfilteredAverage,label='Unfiltered Average Water',color='k',ls='--',alpha=0.4)
    plt.plot(xtest1,f1(xtest1),label='Unfiltered Average Fit')
    plt.plot(xtest,normalizedUSGS,label='USGS Water')
    plt.title(f'Average Water Spectrum for {fileID}')
    plt.legend()
    #plt.savefig(f'E:/Data/Figures/Random Water Spectra/Average_{fileID}.png')

    fig = plt.figure()
    plt.plot(allowedWavelengths,unfilteredAverageWater[i,:],label='Original Average')
    plt.plot(allowedWavelengths,correctedAverageWater[i,:],label='Corrected Average')
    plt.plot(x_destripe,f_destripe(x_destripe),label='Corrected Average Fit')
    #plt.plot(xtest,f(xtest),label='USGS Water')
    plt.title(f'Non-Normalized for {fileID}')
    plt.legend()

plt.show()

    
#%%


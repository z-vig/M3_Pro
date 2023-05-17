'''
HDR Image Class and Script for locating Ice Pixels using both L1 and L2 data from the moon mineralogy mapper
'''
#%%
import time
import spectral as sp
import numpy as np
import spec_plotting
import matplotlib.pyplot as plt
import destripe_image
from copy import copy
import cubic_spline_image as csi
import pandas as pd
import tifffile as tf
import os
import M3_UnZip
from tkinter.filedialog import askdirectory as askdir
import datetime
import shutil
from get_USGS_H2OFrost import get_USGS_H2OFrost

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

class TIF_Image():
    # Constructor method
    def __init__(self,rfl_path,loc_path,obs_path,hdr_path):
        self.rfl_image = tf.imread(rfl_path)
        self.loc_data = tf.imread(loc_path)
        self.obs_data = tf.imread(obs_path)

        self.hdr = sp.envi.open(hdr_path)

        stampFileName = self.hdr.filename
        dateTimeIndex = find(stampFileName,'t')[-1]

        date = f"{stampFileName[dateTimeIndex-8:dateTimeIndex-4]}-{stampFileName[dateTimeIndex-4:dateTimeIndex-2]}"\
                f"-{stampFileName[dateTimeIndex-2:dateTimeIndex]}"

        time = f"{stampFileName[dateTimeIndex+1:dateTimeIndex+3]}-{stampFileName[dateTimeIndex+3:dateTimeIndex+5]}"\
                f"-{stampFileName[dateTimeIndex+5:dateTimeIndex+7]}"
        
        self._dateTime = f'{date}_{time}'

        self.bandCenters = np.array(self.hdr.bands.centers)
        self.allowedIndices = np.where((self.bandCenters>900)&(self.bandCenters<2600))[0]

    @property
    def datetime(self)->str:
        return self._dateTime
    
    @property
    def obsBandNames(self)->list:
        return self.obs_data.__dict__.get('metadata').get('band names')
    
    @property
    def allWavelengths(self)->np.ndarray:
        return self.bandCenters
    
    @property
    def analyzedWavelengths(self)->np.ndarray:
        self._analyzedWavelengths = self.bandCenters[self.allowedIndices]
        return self._analyzedWavelengths
    
    @property
    def unprocessedImage(self)->np.ndarray:
        return self.rfl_image.read_bands(self.allowedIndices)
    
    @property
    def coordinateGrid(self)->np.ndarray: #Array with dim 3 labels: {lat,long,elevation}
        coordArray = np.zeros((*self.unprocessedImage.shape[0:2],3))
        coordArray[:,:,0] = self.loc_data.read_band(1)
        coordArray[:,:,1] = self.loc_data.read_band(0)
        coordArray[:,:,2] = self.loc_data.read_band(2)
        return coordArray

    @property
    def solarIncidenceImage(self)->np.ndarray:
        solar_incidence_index = self.obsBandNames.index('Facet Cos(i) (unitless)')
        solar_incidence_image = 180*np.arccos(self.obs_data.read_band(solar_incidence_index))/np.pi
        return solar_incidence_image

    def get_illuminated_coords(self)->tuple:
        solar_incidence_index = self.obsBandNames.index('Facet Cos(i) (unitless)')
        solar_incidence_image = 180*np.arccos(self.obs_data.read_band(solar_incidence_index))/np.pi
        xLight,yLight = (np.where(solar_incidence_image<90))
        return (xLight,yLight)
    
    def destripe_image(self)->np.ndarray:
        startTime = time.time()
        self.filteredImage = destripe_image.fourier_filter(self.unprocessedImage)
        print (f'Destriping complete in {time.time()-startTime:.1f} seconds')
        return self.filteredImage
    
    def shadow_correction(self,saveFolder,**kwargs)->np.ndarray:
        defaultKwargs = {'inputImage':self.filteredImage}
        kwargs = {**defaultKwargs,**kwargs}

        startTime = time.time()
        self.correctedImage = copy(kwargs.get('inputImage'))
        try:
            stats_arr = np.load(f'{saveFolder}/mosaic_stats_array.npy')
        except:
            raise FileNotFoundError('Run the mosaic data inquiry script first!')
        
        avg_arr = stats_arr[:,:,0]
        R_bi = np.mean(avg_arr,axis=0)
        self.correctedImage = self.correctedImage/R_bi
        print (f'>>>Shadow correction complete in {time.time()-startTime:.1f} seconds')
        return self.correctedImage
    
    def spectrum_smoothing(self,**kwargs)->tuple[np.ndarray,np.ndarray]:
        defaultKwargs = {'inputImage':self.correctedImage}
        kwargs = {**defaultKwargs,**kwargs}

        startTime = time.time()
        self.avgWvl,self.avgSpectrumImage,self.smoothSpectrumImage = \
            csi.splineFit(kwargs.get('inputImage'),5,self.analyzedWavelengths)
        print (f'>>>Spectrum Smoothing complete in {time.time()-startTime:.1f} seconds')
        return self.avgSpectrumImage,self.smoothSpectrumImage
        
    def locate_ice(self,**kwargs)->tuple[np.ndarray,np.ndarray,pd.DataFrame]:
        defaultKwargs = {'inputImage':self.smoothSpectrumImage}
        kwargs = {**defaultKwargs,**kwargs}

        startTime = time.time()
        band1_indices = np.where((self.analyzedWavelengths>1242)&(self.analyzedWavelengths<1323))[0]
        band2_indices = np.where((self.analyzedWavelengths>1503)&(self.analyzedWavelengths<1659))[0]
        band3_indices = np.where((self.analyzedWavelengths>1945)&(self.analyzedWavelengths<2056))[0]

        diff_array = np.zeros(kwargs.get('inputImage').shape)
        for band in range(kwargs.get('inputImage').shape[2]-1): #The last band will be all zeros
            diff_array[:,:,band] = kwargs.get('inputImage')[:,:,band]>kwargs.get('inputImage')[:,:,band+1]

        allBand_indices = np.concatenate((band1_indices,band2_indices,band3_indices))

        def get_bandArray(band_indices:np.ndarray,bandName:str)->np.ndarray:
            band_arr = np.zeros((kwargs.get('inputImage').shape[0:2]))
            for i in range(band_indices.min()-1,band_indices.max()):
                band_arr[np.where((diff_array[:,:,i]!=diff_array[:,:,i+1])&(diff_array[:,:,i]==True))] = 1
            return band_arr

        band1_Array = get_bandArray(band1_indices,'Band 1')
        band2_Array = get_bandArray(band2_indices,'Band 2')
        band3_Array = get_bandArray(band3_indices,'Band 3')

        self.waterLocations = np.zeros(band1_Array.shape)
        self.waterCoords_numpy = np.where((band1_Array==1)&(band2_Array==1)&(band3_Array==1)&(np.average(kwargs.get('inputImage'),axis=2)>0.05))
        self.waterLocations[self.waterCoords_numpy] = 1

        self.waterCoords_map = self.coordinateGrid[self.waterCoords_numpy[0],self.waterCoords_numpy[1],:]
        waterDf = pd.DataFrame(self.waterCoords_map)
        waterDf.columns = ['Latitude','Longitude','Elevation']
        print(f'>>>Ice located in {time.time()-startTime:.1f} seconds')
        
        return self.waterLocations,self.waterCoords_map,waterDf

    def spectral_angle_mapping(self,threshold:float,**kwargs)->tuple[np.ndarray,np.ndarray]:
        defaultKwargs = {'inputImage':self.smoothSpectrumImage}
        kwargs = {**defaultKwargs,**kwargs}

        total_pixels = self.unprocessedImage.shape[0]*self.unprocessedImage.shape[1]

        wvl,USGS_Frost = get_USGS_H2OFrost(USGS_folder='D:/Data/USGS_Water_Ice')
        USGS_Frost = np.expand_dims(USGS_Frost,1)
        USGS_Frost_Array = np.repeat(USGS_Frost,total_pixels,1).T
        USGS_Frost_Array = USGS_Frost_Array.reshape((self.unprocessedImage.shape[0],self.unprocessedImage.shape[1],59))

        M,I = kwargs.get('inputImage'),USGS_Frost_Array

        SAM = 180*np.arccos(np.einsum('ijk,ijk->ij',M,I)/(np.linalg.norm(M,axis=2)*np.linalg.norm(I,axis=2)))/np.pi
        no_water_indices = np.where(self.waterLocations==0)
        high_spec_angle_indices = np.where(SAM>threshold)

        threshIceMap = copy(kwargs.get('inputImage'))
        threshIceMap[no_water_indices]=-9999
        threshIceMap[high_spec_angle_indices] = -9999

        return SAM,threshIceMap

#For testing:    

# #%%
# L2_fileList,L2_filePath = M3_UnZip.M3_unzip(select=True)
# L1_fileList,L1_filePath = M3_UnZip.M3_unzip(select=True)
# saveFolder = askdir()
# rfl_fileList = [i for i in L2_fileList if i.find('rfl')>-1]
# loc_fileList = [i for i in L1_fileList if i.find('loc')>-1]
# obs_fileList = [i for i in L1_fileList if i.find('obs')>-1]
# #%%
# for rflPath,locPath,obsPath in zip(rfl_fileList,loc_fileList,obs_fileList):
#     M3stamp = HDR_Image(rflPath,locPath,obsPath)
#     print (f'{M3stamp.datetime} Started')
#     filterImg = M3stamp.destripe_image()
#     correctedImg = M3stamp.shadow_correction(saveFolder)
#     avgImg,smoothImg = M3stamp.spectrum_smoothing()
#     waterImage,waterCoords,waterDf = M3stamp.locate_ice(inputImage=smoothImg)
#     noWater_indices = np.where(waterImage==0)
#     waterImage_complete = copy(smoothImg)
#     waterImage_complete[noWater_indices] = 0
#     SAM,IceMap = M3stamp.spectral_angle_mapping(30,inputImage=smoothImg)
#     xWater,yWater = (np.where(IceMap[:,:,0]!=-9999))
#     fig = plt.figure()
#     for x,y in zip(xWater,yWater):
#         rfl = IceMap[x,y,:]
#         plt.plot(M3stamp.analyzedWavelengths,IceMap[x,y,:],label=f'({x},{y})')
#         plt.fill_betweenx(np.arange(rfl.min(),rfl.max(),0.01),1242,1323,color='gray',alpha=0.5)
#         plt.fill_betweenx(np.arange(rfl.min(),rfl.max(),0.01),1503,1659,color='gray',alpha=0.5)
#         plt.fill_betweenx(np.arange(rfl.min(),rfl.max(),0.01),1945,2056,color='gray',alpha=0.5)
#     #tf.imwrite(f'D:/Data/Ice_Pipeline_Out_4-26-23/water_convolutionFilter.tif',waterImage.astype('float32'))
#     break
#%%



#%%
if __name__ == "__main__":
    ##Setup and folder selection
    start = time.time()
    print ('Select rfl folder')
    rfl_folder = askdir() 
    rfl_fileList = os.listdir(rfl_folder)
    print (f'{rfl_folder} selected')
    print ('Select loc folder')
    loc_folder = askdir()
    loc_fileList = os.listdir(loc_folder)
    print (f'{loc_folder} selected')
    print ('Select obs folder')
    obs_folder = askdir()
    obs_fileList = os.listdir(obs_folder)
    print (f'{obs_folder} selected')
    print ('Select output/save folder')
    saveFolder = askdir()

    imageProductList = ['originalImages','locationInfo','solarIncidenceImages',\
                        'water_locations','destripedImages','spectralAngleMaps','sampleWaterSpectra']
    for dir in imageProductList:
        try:
            os.mkdir(f'{saveFolder}/{dir}')
        except:
            continue

    ##Image Processing
    allWater_array = np.zeros((0,59))
    totalStampsLoaded = len(rfl_fileList)
    stamp_progress = 1
    print (f'{totalStampsLoaded} image stamps will be processed.')
    for rflPath,locPath,obsPath in zip(rfl_fileList,loc_fileList,obs_fileList):
        print (f'-------Beginning Stamp {stamp_progress} of {totalStampsLoaded}-------')
        stampStartTime = time.time()
        M3stamp = TIF_Image(rflPath,locPath,obsPath)

        originalImage = M3stamp.unprocessedImage
        print (f'Stamp ID: {M3stamp.datetime}\n'\
            f'Number of Analyzed Wavelengths: {M3stamp.analyzedWavelengths.shape[0]}\n'\
            f'Stamp Size: {originalImage.shape}\n')
        print (f'Stamp {stamp_progress} of {totalStampsLoaded} has been loaded in {time.time()-stampStartTime:.1f} seconds')

        print ('Destriping reflectance image...')
        filteredImage = M3stamp.destripe_image()

        print ('Making Li et al., 2018 Shadow Correction...')
        correctedImage = M3stamp.shadow_correction(saveFolder,inputImage=filteredImage)

        print ('Smoothing spectra...')
        avgSpecImg,smoothSpecImg = M3stamp.spectrum_smoothing()

        print ('Locating water-like spectra..')
        waterImage,waterPixels,waterDf = M3stamp.locate_ice()

        print ('Building Spectral Angle Map...')
        SpecAngleMap,iceMap = M3stamp.spectral_angle_mapping(30)
        iceMap_df = pd.DataFrame(M3stamp.coordinateGrid[np.where(iceMap[:,:,0]>-9999)])
        iceMap_df.columns = ['Latitude','Longitude','Elevation']

        print ('Sampling water spectra...')
        try:
            os.mkdir(f'{saveFolder}/sampleWaterSpectra/{M3stamp.datetime}')
        except:
            pass

        XY_array = np.array(np.where(iceMap>-9999)).T
        if 0 in XY_array.shape:
            print (f'{M3stamp.datetime} has no water detections\n\n')
            stamp_progress+=1
            continue
        
        print (f'There were {XY_array.shape} water detections in {M3stamp.datetime}')
        randomXY = XY_array[np.random.choice(np.arange(0,XY_array.shape[0],1),10,replace=False),:]
        random10spectra = iceMap[randomXY[:,0],randomXY[:,1],:]
        fig,axList = plt.subplots(10,1,figsize=(8,30),dpi=500,layout='constrained')
        for ax,spec,loc in zip(axList,random10spectra,range(randomXY.shape[0])):
            ax.plot(M3stamp.analyzedWavelengths,spec)
            ax.set_title(f'({randomXY[loc,0]},{randomXY[loc,1]})')
            ax.set_ylabel('Reflectance')
        fig.suptitle('10 Random Water-Like Spectra')
        fig.supxlabel('Wavelength (\u03BCm)')
        plt.savefig(f'{saveFolder}/sampleWaterSpectra/{M3stamp.datetime}/tenRandomSpectra.jpg')
        
        for spec,loc in zip(random10spectra,range(randomXY.shape[0])):
            fig = plt.figure(dpi=300)
            plt.plot(M3stamp.analyzedWavelengths,spec)
            plt.title(f'{randomXY[loc,:]}')
            plt.xlabel('Wavelength (\u03BCm)')
            plt.ylabel('Reflectance')
            plt.savefig(f'{saveFolder}/sampleWaterSpectra/{M3stamp.datetime}/{randomXY[loc,0]}_{randomXY[loc,1]}.jpg')
        
        allWater_array = np.concatenate((allWater_array,iceMap[XY_array[:,0],XY_array[:,1],:]))

        def save_everything_to(folder): #Function for saving images
            saveStartTime = time.time()
            try:
                os.mkdir(f'{folder}/{M3stamp.datetime}')
            except:
                pass

            try:
                os.mkdir(f'{folder}/aux_data')
            except:
                pass

            try:
                os.mkdir(f'{folder}/aux_data/{M3stamp.datetime}')
            except:
                pass

            print ('Saving Data...')
            np.save(f'{folder}/{M3stamp.datetime}/filter_{M3stamp.datetime}.npy',filteredImage)
            np.save(f'{folder}/{M3stamp.datetime}/corrected_{M3stamp.datetime}.npy',correctedImage)
            np.save(f'{folder}/{M3stamp.datetime}/smoothSpec_{M3stamp.datetime}.npy',smoothSpecImg)
            np.save(f'{folder}/{M3stamp.datetime}/waterImage_{M3stamp.datetime}.npy',waterImage)
            np.save(f'{folder}/{M3stamp.datetime}/specAngleMap_{M3stamp.datetime}.npy',SpecAngleMap)

            print ('Saving Images...')
            tf.imwrite(f'{folder}/{M3stamp.datetime}/original_{M3stamp.datetime}_readable.tif',originalImage[:,:,0].astype('float32'))
            tf.imwrite(f'{folder}/{M3stamp.datetime}/filter_{M3stamp.datetime}.tif',filteredImage.astype('float32'),photometric='rgb')
            tf.imwrite(f'{folder}/{M3stamp.datetime}/corrected_{M3stamp.datetime}.tif',correctedImage.astype('float32'),photometric='rgb')
            tf.imwrite(f'{folder}/{M3stamp.datetime}/smoothSpec_{M3stamp.datetime}.tif',smoothSpecImg.astype('float32'),photometric='rgb')
            tf.imwrite(f'{folder}/{M3stamp.datetime}/waterImage_{M3stamp.datetime}.tif',waterImage.astype('float32'))
            tf.imwrite(f'{folder}/{M3stamp.datetime}/specAngleImage_{M3stamp.datetime}.tif',iceMap.astype('float32'),photometric='rgb')
            tf.imwrite(f'{folder}/{M3stamp.datetime}/specAngleImage_{M3stamp.datetime}_readable.tif',iceMap[:,:,0].astype('float32'))

            print ('Saving Auxilliary Data...')
            waterDf.to_csv(f'{folder}/aux_data/{M3stamp.datetime}/all_water_locations.csv')
            iceMap_df.to_csv(f'{folder}/aux_data/{M3stamp.datetime}/specAngle_water_locations.csv')
            np.save(f'{folder}/aux_data/{M3stamp.datetime}/avgSpec.npy',avgSpecImg)
            tf.imwrite(f'{folder}/aux_data/{M3stamp.datetime}/correctedImg.tif',correctedImage.astype('float32'),photometric='rgb')
            tf.imwrite(f'{folder}/aux_data/{M3stamp.datetime}/smoothSpecImg.tif',smoothSpecImg.astype('float32'),photometric='rgb')
            tf.imwrite(f'{folder}/solarIncidenceImages/incidence_{M3stamp.datetime}.tif',M3stamp.solarIncidenceImage.astype('float32'))
            coordDf = pd.DataFrame({'Latitude':M3stamp.coordinateGrid[:,:,0].flatten(),'Longitude':M3stamp.coordinateGrid[:,:,1].flatten(),'Elevation':M3stamp.coordinateGrid[:,:,2].flatten()})
            tf.imwrite(f'{folder}/locationInfo/coordGrid_{M3stamp.datetime}.tif',M3stamp.coordinateGrid.astype('float32'),photometric='rgb')
            fig = plt.figure()
            plt.plot()

            print ('Making copies...')
            shutil.copy(f'{folder}/{M3stamp.datetime}/original_{M3stamp.datetime}.tif',\
                        f'{folder}/originalImages/original_{M3stamp.datetime}.tif')
            shutil.copy(f'{folder}/{M3stamp.datetime}/filter_{M3stamp.datetime}.tif',\
                        f'{folder}/destripedImages/filter_{M3stamp.datetime}.tif')
            shutil.copy(f'{folder}/aux_data/{M3stamp.datetime}/specAngle_water_locations.csv',\
                        f'{folder}/water_locations/iceCoords_{M3stamp.datetime}.csv')
            shutil.copy(f'{folder}/{M3stamp.datetime}/specAngleImage_{M3stamp.datetime}.tif',\
                        f'{folder}/spectralAngleMaps/specAngleImage_{M3stamp.datetime}.tif')
            
            print (f'Save complete in {time.time()-saveStartTime:.2f} seconds')
        save_everything_to(saveFolder)

        print (f'Stamp ID: {M3stamp.datetime} ({stamp_progress} of {totalStampsLoaded}) complete ({stamp_progress/totalStampsLoaded:.0%})\n\n')
        plt.close()
        stamp_progress+=1

    np.save(f'{saveFolder}/allWaterSpectra.npy',allWater_array)
    averageWaterSpectrum = np.mean(allWater_array,axis=0)
    fig=plt.figure(dpi=300)
    plt.plot(M3stamp.analyzedWavelengths,averageWaterSpectrum)
    plt.title('Average Water Spectrum')
    plt.xlabel('Wavelength (\u03BCm)')
    plt.ylabel('Reflectance')
    plt.savefig(f'{saveFolder}/Mean Water Spectrum.jpg')

    end = time.time()
    runtime = end-start
    print (f'Image Processing finished at {datetime.datetime.now()}')
    if runtime < 1:
        print(f'Program Executed in {runtime*10**3:.3f} milliseconds')
    elif runtime < 60 and runtime > 1:
        print(f'Program Executed in {runtime:.3f} seconds')
    else:
        print(f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')
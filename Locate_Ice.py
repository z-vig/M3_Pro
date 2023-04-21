'''
HDR Image Class and Script for locating Ice Pixels
'''

#%%
import time
import spectral as sp
import numpy as np
import spec_plotting
import matplotlib.pyplot as plt
import DestripeImage
from copy import copy
import cubic_spline_image as csi
import pandas as pd
import tifffile as tf
import os
import M3_UnZip
from tkinter.filedialog import askdirectory as askdir
import datetime
import shutil

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

class HDR_Image():
    # Constructor method
    def __init__(self,rfl_path,loc_path,obs_path):
        self.rfl_image = sp.envi.open(rfl_path)
        self.loc_data = sp.envi.open(loc_path)
        self.obs_data = sp.envi.open(obs_path)

        stampFileName = self.rfl_image.filename
        dateTimeIndex = find(stampFileName,'t')[-1]

        date = f"{stampFileName[dateTimeIndex-8:dateTimeIndex-4]}-{stampFileName[dateTimeIndex-4:dateTimeIndex-2]}"\
                f"-{stampFileName[dateTimeIndex-2:dateTimeIndex]}"

        time = f"{stampFileName[dateTimeIndex+1:dateTimeIndex+3]}-{stampFileName[dateTimeIndex+3:dateTimeIndex+5]}"\
                f"-{stampFileName[dateTimeIndex+5:dateTimeIndex+7]}"
        
        self._dateTime = f'{date}_{time}'

        self.bandCenters = np.array(self.rfl_image.bands.centers)
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
    
    def get_illuminated_coords(self)->tuple:
        solar_incidence_index = self.obsBandNames.index('Facet Cos(i) (unitless)')
        solar_incidence_image = 180*np.arccos(self.obs_data.read_band(solar_incidence_index))/np.pi
        xLight,yLight = (np.where(solar_incidence_image<90))
        return (xLight,yLight)
    
    def destripe_image(self)->np.ndarray:
        startTime = time.time()
        self.filteredImage = DestripeImage.fourier_filter(self.unprocessedImage)
        print (f'Destriping complete in {time.time()-startTime:.1f} seconds')
        return self.filteredImage
    
    def shadow_correction(self):
        self.correctedImage = copy(self.filteredImage)




#For testing:        

#%%
L2_fileList,L2_filePath = M3_UnZip.M3_unzip(select=True)
L1_fileList,L1_filePath = M3_UnZip.M3_unzip(select=True)
rfl_fileList = [i for i in L2_fileList if i.find('rfl')>-1]
loc_fileList = [i for i in L1_fileList if i.find('loc')>-1]
obs_fileList = [i for i in L1_fileList if i.find('obs')>-1]
#%%
for rflPath,locPath,obsPath in zip(rfl_fileList,loc_fileList,obs_fileList):
    M3stamp = HDR_Image(rflPath,locPath,obsPath)
    xLight,yLight = M3stamp.get_illuminated_coords()

#%%
if __name__ == "__main__":
    ##Setup and folder selection
    start = time.time()
    print ('Select L2 folder')
    L2_fileList,L2_filePath = M3_UnZip.M3_unzip(select=True)
    print ('Select L1 folder')
    L1_fileList,L1_filePath = M3_UnZip.M3_unzip(select=True)
    print ('Select output/save folder')
    saveFolder = askdir()
    imageProductList = ['originalImages','locationInfo','solarIncidenceImages',\
                        'water_locations','destripedImages']
    for dir in imageProductList:
        try:
            os.mkdir(f'{saveFolder}/{dir}')
        except:
            continue

    rfl_fileList = [i for i in L2_fileList if i.find('rfl')>-1]
    loc_fileList = [i for i in L1_fileList if i.find('loc')>-1]
    obs_fileList = [i for i in L1_fileList if i.find('obs')>-1]

    ##Image Processing
    totalStampsLoaded = len(rfl_fileList)
    stamp_progress = 1
    print (f'{totalStampsLoaded} image stamps will be processed.')
    for rflPath,locPath,obsPath in zip(rfl_fileList,loc_fileList,obs_fileList):
        print (f'-------Beginning Stamp {stamp_progress} of {totalStampsLoaded}-------')
        stampStartTime = time.time()
        M3stamp = HDR_Image(rflPath,locPath,obsPath)

        originalImage = M3stamp.unprocessedImage
        print (f'Stamp ID: {M3stamp.datetime}\n'\
            f'Number of Analyzed Wavelengths: {M3stamp.analyzedWavelengths.shape[0]}\n'\
            f'Stamp Size: {originalImage.shape}\n')
        print (f'Stamp {stamp_progress} of {totalStampsLoaded} has been loaded in {time.time()-stampStartTime:.1f} seconds')

        print ('Destriping reflectance image...')
        filteredImage = M3stamp.destripe_image()

        print ()

        def save_everything_to(folder):
            saveStartTime = time.time()
            try:
                os.mkdir(f'{folder}/{M3stamp.datetime}')
            except:
                pass

            print ('Saving Data...')
            np.save(f'{folder}/{M3stamp.datetime}/original_{M3stamp.datetime}.npy',originalImage)

            print ('Saving Images...')
            tf.imwrite(f'{folder}/{M3stamp.datetime}/original_{M3stamp.datetime}.tif',originalImage.astype('float32'),photometric='rgb')
            tf.imwrite(f'{folder}/{M3stamp.datetime}/original_{M3stamp.datetime}_readable.tif',originalImage[:,:,0].astype('float32'))

            print ('Making copies...')
            shutil.copy(f'{folder}/{M3stamp.datetime}/original_{M3stamp.datetime}.tif',\
                        f'{folder}/originalImages/original_{M3stamp.datetime}.tif')
            
            print (f'Save complete in {time.time()-saveStartTime:.2f} seconds')
        save_everything_to(saveFolder)
        print (f'Stamp {stamp_progress} of {totalStampsLoaded} complete ({stamp_progress/totalStampsLoaded:.0%})\n\n')
        stamp_progress+=1

    end = time.time()
    runtime = end-start
    print (f'Image Processing finished at {datetime.datetime.now()}')
    if runtime < 1:
        print(f'Program Executed in {runtime*10**3:.3f} milliseconds')
    elif runtime < 60 and runtime > 1:
        print(f'Program Executed in {runtime:.3f} seconds')
    else:
        print(f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')
        
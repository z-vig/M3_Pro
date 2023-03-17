'''
HDR Image Class and Script for locating Ice Pixels
'''
import time
import spectral as sp
import numpy as np
import spec_plotting
import matplotlib.pyplot as plt
import DestripeImage
from copy import copy

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

class HDR_Image():
    # Constructor method
    def __init__(self, path):
        self.hdr = sp.envi.open(path)
        data_str = self.hdr.filename
        dateTimeIndex = find(data_str,'t')[-1]

        date = f"{data_str[dateTimeIndex-8:dateTimeIndex-4]}-{data_str[dateTimeIndex-4:dateTimeIndex-2]}-{data_str[dateTimeIndex-2:dateTimeIndex]}"

        time = f"{data_str[dateTimeIndex+1:dateTimeIndex+3]}-{data_str[dateTimeIndex+3:dateTimeIndex+5]}-{data_str[dateTimeIndex+5:dateTimeIndex+7]}"

        date_time = date+'_'+time

        if data_str[dateTimeIndex-9] == 'g':
            obs_type = 'Global'
        elif data_str[dateTimeIndex-9] == 't':
            obs_type = 'Target'
        else:
            print (f'HDR File Name: {self.hdr.filename}')
            print (find(self.hdr.filename, '/'))
            print (data_str)
            print (f'Observation Type loaded as: {data_str[2]}')
            raise Exception("Data String Error!")
            
        if data_str[len(data_str)-7:len(data_str)-4] == 'sup':
            data_type = "Supplemental"
        elif data_str[len(data_str)-7:len(data_str)-4] == 'rfl':
            data_type = "Reflectance"
            
        self._dateTime = date_time
        self._obsType = obs_type
        self._dataType = data_type
        self.meta_data_str = f"Observation Type: {obs_type}\nData Type: {data_type}\nDate (Y/M/D): {date}\nTime: {time}"
        self.meta_data_dict = {"ObservationType": obs_type,
                               "DataType": data_type, "Date(Y/M/D)": date, "Time": time}
        
    # String print method
    def __str__(self):
        return f"HDR Image Class: {self.hdr.fid.name[find(self.hdr.fid.name,'/')[-1]+1:len(self.hdr.fid.name)-4]}"

    # Returns HDR Object for testing purposes
    def hdr(self):
        return self.hdr

    # Method that returns date, time, and data types of HDR Image
    @property
    def datetime(self):
        return self._dateTime
    
    @property
    def allowedWavelengths(self):
        bandCenters = np.array(self.hdr.bands.centers)
        self.allowedIndices = np.where((bandCenters>900)&(bandCenters<2600))[0]
        self._allowedWvl = bandCenters[self.allowedIndices]
        return self._allowedWvl

    def original_image(self,**kwargs):
        defaultKwargs = {'imshow':False}
        kwargs = {**defaultKwargs,**kwargs}

        self.originalImage = self.hdr.read_bands(self.allowedIndices)
        if kwargs.get('imshow') == True:
            spec_plotting.plot_numpy_images(self.originalImage[:,:,0],self.originalImage[:,:,1],
                                            titles=[f'{self._allowedWvl[0]}\u03BCm',f'{self._allowedWvl[1]}\u03BCm'])
            plt.show()
        
        return self.originalImage
    
    def destripe_image(self,**kwargs):
        defaultKwargs = {'imshow':False}
        kwargs = {**defaultKwargs,**kwargs}

        self.destripeImage = DestripeImage.destripe(self.originalImage,7)
        if kwargs.get('imshow') == True:
            spec_plotting.plot_numpy_images(self.originalImage[:,:,0],self.destripeImage[:,:,1],
                                            titles=['Original','Destriped Image'])
            plt.show()

        return self.destripeImage
    
    def shadow_correction(self,R_bi:np.ndarray,shadowLocations:np.ndarray,**kwargs)->np.ndarray:
        defaultKwargs = {'imshow':False}
        kwargs = {**defaultKwargs,**kwargs}

        self.correctedImage = copy(self.originalImage)

        R_bi = R_bi[self.allowedIndices]
        xShade,yShade = np.where(shadowLocations[0]==0)
        xLight,yLight = np.where(shadowLocations[0]!=0)

        self.correctedImage[xShade,yShade,:] = self.correctedImage[xShade,yShade,:]/R_bi

        if kwargs.get('imshow') == True:
            spec_plotting.plot_numpy_images(self.originalImage[:,:,0],self.correctedImage[:,:,0],
                                            titles=['Original','Li et al., 2018\nCorrection'])
            plt.show()
        
        return self.correctedImage
        


        
        
if __name__ == "__main__":
    start = time.time()

    img1 = HDR_Image(r"D:\Data/20230209T095534013597/extracted_files/hdr_files/m3g20090417t193320_v01_rfl/m3g20090417t193320_v01_rfl.hdr")
        
    print (f'Image {img1.datetime} Loaded:\n\
    Analyzed Wavelengths: {img1.allowedWavelengths}')
    originalImage = img1.original_image()
    print ('Destriping Image...')
    destripeImage = img1.destripe_image(imshow=True)
    print (f'Image destriped at {time.time()-start}')

    end = time.time()
    runtime = end-start
    if runtime < 1:
        print(f'Program Executed in {runtime*10**3:.3f} milliseconds')
    elif runtime < 60 and runtime > 1:
        print(f'Program Executed in {runtime:.3f} seconds')
    else:
        print(f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')

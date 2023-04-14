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
from get_pixel_mosaic import create_arrays
import cubic_spline_image as csi
import pandas as pd
import tifffile as tf
import os
import M3_UnZip
from tkinter.filedialog import askdirectory as askdir
import datetime

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
        
        self.bandCenters = np.array(self.hdr.bands.centers)
        self.allowedIndices = np.where((self.bandCenters>900)&(self.bandCenters<2600))[0]
        
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
        self._allowedWvl = self.bandCenters[self.allowedIndices]
        return self._allowedWvl
    
    @property
    def allWavelengths(self):
        self._allWavelengths = self.bandCenters
        return self._allWavelengths

    def original_image(self,**kwargs):
        defaultKwargs = {'imshow':False}
        kwargs = {**defaultKwargs,**kwargs}

        self.originalImage = self.hdr.read_bands(self.allowedIndices)
        if kwargs.get('imshow') == True:
            spec_plotting.plot_numpy_images(self.originalImage[:,:,0],self.originalImage[:,:,1],
                                            titles=[f'{self._allowedWvl[0]}\u03BCm',f'{self._allowedWvl[1]}\u03BCm'])
        
        return self.originalImage
    
    def destripe_image(self,**kwargs):
        defaultKwargs = {'imshow':False}
        kwargs = {**defaultKwargs,**kwargs}

        print (self.allowedIndices)
        self.destripeImage = DestripeImage.destripe(self.originalImage,7)
        if kwargs.get('imshow') == True:
            spec_plotting.plot_numpy_images(self.originalImage[:,:,0],self.destripeImage[:,:,1],
                                            titles=['Original','Destriped Image'])

        return self.destripeImage
    
    def get_average_rfl(self, mosaicPixels, **kwargs):
        defaultKwargs = {'plotAverage':False}
        kwargs = {**defaultKwargs,**kwargs}

        mosaicPixels_allowed = mosaicPixels[self.allowedIndices,:]
        means = mosaicPixels_allowed.mean(axis=1)
        std = mosaicPixels_allowed.std(axis=1)

        if kwargs.get('plotAverage') == True:
            plt.plot(self.allowedWavelengths,means,color='red')
            plt.fill_between(self.allowedWavelengths,means-std,means+std,color='gray',alpha=0.5)

        return means,std
    
    def shadow_correction(self,R_bi:np.ndarray,shadowLocations:np.ndarray,**kwargs)->np.ndarray:
        defaultKwargs = {'imshow':False}
        kwargs = {**defaultKwargs,**kwargs}

        self.correctedImage = copy(self.destripeImage)

        xShade,yShade = np.where(shadowLocations==0)[0],np.where(shadowLocations==0)[1]
        xLight,yLight = np.where(shadowLocations!=0)[0],np.where(shadowLocations!=0)[1]

        self.correctedImage[xShade,yShade,:] = self.correctedImage[xShade,yShade,:]/R_bi

        if kwargs.get('imshow') == True:
            spec_plotting.plot_numpy_images(self.originalImage[:,:,0],self.correctedImage[:,:,0],
                                            titles=['Original','Li et al., 2018\nCorrection'])
        
        return self.correctedImage

    def spectrum_smoothing(self,**kwargs)->np.ndarray:
        defaultKwargs = {'imshow':False,'specshow':False,'plottedPoints':(91,100)}
        kwargs = {**defaultKwargs,**kwargs}

        self.avgSpectrumImage,self.smoothSpectrumImage = csi.splineFit(self.correctedImage,self.allowedWavelengths,5)
        if kwargs.get('imshow') == True:
            spec_plotting.plot_numpy_images(self.originalImage[:,:,0],self.correctedImage[:,:,0],self.smoothSpectrumImage[:,:,0],
                                            titles=['Original','Li et al., 2018\nCorrection','Smoothed Spectra'])
        if kwargs.get('imshow') == True:
            fig,ax = plt.subplots(1,1)
            x,y = kwargs.get('plottedPoints')[0],kwargs.get('plottedPoints')[1]
            spec_plotting.fancy_spec_plot(fig,ax,self.allowedWavelengths,self.smoothSpectrumImage[x,y,:],
                                          ylabel='Reflectance',xlabel='Wavelength \u03BCm',title=f'Smoothed Spectrum of point ({x},{y})',label = 'Cubic Spline Fit')
            spec_plotting.fancy_spec_plot(fig,ax,self.allowedWavelengths,self.originalImage[x,y,:],label='Original',line_color='black')
            spec_plotting.fancy_spec_plot(fig,ax,self.allowedWavelengths,self.correctedImage[x,y,:],label='Corrected Spectrum',line_color='green')

        return self.avgSpectrumImage,self.smoothSpectrumImage
        
    def locate_ice(self,**kwargs)->np.ndarray:
        defaultKwargs = {'imshow':False,'specshow':False,'plottedPoints':(91,100)}
        kwargs = {**defaultKwargs,**kwargs}

        def get_minima(wvlValues:'np.ndarray',img:'np.ndarray',x:'int', y:'int',**kwargs) -> 'tuple':
            defaultKwargs = {"plotMins":False}
            kwargs = {**defaultKwargs,**kwargs}

            rflValues = img[x,y,:]

            diff_list = []
            wvlMinima = ()
            for n in range(0, len(rflValues)):
                if n < len(rflValues)-1:
                    diff = rflValues[n]-rflValues[n+1]
                    diff_list.append(diff)
                    if n > 2 and diff < 0 and diff_list[-2] > 0:
                        wvlMinima += (wvlValues[n],)
            
            ##Check if minima are indicative of water spectrum
            validWvl = np.array(([1.242,1.323],[1.503,1.659],[1.945,2.056]))
            waterBands = ()
            for val in wvlMinima:
                if any(val/1000 > i and val/1000 < j for i,j in zip(validWvl[:,0],validWvl[:,1])):
                    waterBands += (val,)
                else:
                    pass
            
            if kwargs.get('plotMins') == True:
                fig,ax = plt.subplots(1,1)
                ax.plot(wvlValues,rflValues,label='Cubic Spline Fit')
                for val in wvlMinima:
                    ax.vlines(val,rflValues.min(),rflValues.max())

            if len(wvlMinima) == 3 and len(waterBands) == 3 and np.average(rflValues)>0.01:
                return waterBands,True,False
            elif len(wvlMinima) == 3 and len(waterBands) == 3 and np.average(rflValues)<0.01:
                return waterBands,True,True
            else:
                return wvlMinima,False,True
            
        self.waterPixels = np.zeros([1,5]).astype(int)
        self.waterPixels_noise = np.zeros([1,5]).astype(int)
        xCoords,yCoords = range(int(self.smoothSpectrumImage.shape[0])),range(int(self.smoothSpectrumImage.shape[1]))
        xMesh,yMesh = np.meshgrid(xCoords,yCoords)

        num = 0
        for x,y in zip(xMesh.flatten(),yMesh.flatten()):
            if num < 1000000:
                print (f'\r{num+1}/{len(xMesh.flatten())} Points Processed ({num/len(xMesh.flatten()):.0%})',end='\r')
                waterBands,add,noise = get_minima(self.allowedWavelengths,self.smoothSpectrumImage,x,y)
                if add == True and noise == False:
                    self.waterPixels = np.concatenate([self.waterPixels,np.array([(x,y,*waterBands)])])
                elif add == True and noise == True:
                    self.waterPixels_noise = np.concatenate([self.waterPixels_noise,np.array([(x,y,*waterBands)])])
                elif add == False:
                    pass
                num+=1
            else:
                print (f'\nLoop broken at {num}')
                break
            
        waterDf = pd.DataFrame(self.waterPixels)
        waterDf.columns = ['x','y','Band 1','Band 2','Band 3']
        waterDf.drop(waterDf.index[0],inplace=True)
        waterDf.set_index(['x','y'],inplace=True)


        def plot_correction_minima(x,y,**kwargs):
            defaultKwargs = {'showMinText':True}
            kwargs = {**defaultKwargs,**kwargs}

            fig,ax = plt.subplots(1,1)
            spec_plotting.fancy_spec_plot(fig,ax,self.allowedWavelengths,self.originalImage[x,y,:],label='Original',color='k',
                                          ylabel='Reflectance',xlabel='Wavelength (\u03BCm)')
            spec_plotting.fancy_spec_plot(fig,ax,self.allowedWavelengths,self.correctedImage[x,y,:],label='Corrected',line_color='red',alpha=0.75)
            spec_plotting.fancy_spec_plot(fig,ax,self.allowedWavelengths,self.avgSpectrumImage[x,y,:],label='Average',line_color='red',line_style='--')
            spec_plotting.fancy_spec_plot(fig,ax,self.allowedWavelengths,self.smoothSpectrumImage[x,y,:],label='Cubic Spline',line_color='Green',
                                          title=f'Possible Water Spectra at ({x},{y})')

            if (x,y) in waterDf.index:
                wvlMin = waterDf.loc[(x,y)]
                if kwargs.get('showMinText') == True:
                    for val in wvlMin:
                        ax.vlines(val,self.smoothSpectrumImage[x,y,:].min(),self.smoothSpectrumImage[x,y,:].max(),ls='-.')
                        ax.text(val-100,self.smoothSpectrumImage[x,y,:].max()+0.05*self.smoothSpectrumImage[x,y,:].max(),f'{val:.0f} \u03BCm')

            ax.legend()

        if kwargs.get('specshow') == True:
            plot_correction_minima(kwargs.get('plottedPoints')[0],kwargs.get('plottedPoints')[1])

        water_where = np.zeros(self.originalImage.shape)
        water_where_overlaid = copy(self.originalImage)
        water_bandDepth = copy(self.originalImage)
        for x,y in waterDf.index:
            water_where[int(x),int(y),:] = np.ones((len(self.allowedWavelengths)))
            water_where_overlaid[int(x),int(y),:] = np.ones((len(self.allowedWavelengths)))

        if kwargs.get('imshow') == True:
            spec_plotting.plot_numpy_images(self.originalImage[:,:,32],self.smoothSpectrumImage[:,:,32],
                                            self.smoothSpectrumImage[:,:,32],water_where[:,:,32], water_where_overlaid[:,:,32],
                                            titles=['Original','Average\n(Shade Corrected)','Cubic Spline\n(Shade Corrected)',
                                                    'Possible H\u2082O\nPixels','Possible H\u2082O\nPixels'])
        
        return water_where,self.waterPixels_noise,self.waterPixels,waterDf
#%%
#hdrFileList,hdrFilesPath = M3_UnZip.M3_unzip(select=True)
with open("E:/Data/Locate_Ice_Saves/Product_IDs.txt",'w') as writeFile:
    for file in hdrFileList:
        img = HDR_Image(file)
        string = 'M3G'+img.datetime.replace('-','').replace('_','T')+'*'
        writeFile.write(string+'\n')
writeFile.close()

#%%
if __name__ == "__main__":
    start = time.time()
    print ('Begin by selecting a folder to save your data!')
    savefolder = askdir()
    print (f'Image Processing started at {datetime.datetime.now()}')
    hdrFileList,hdrFilesPath = M3_UnZip.M3_unzip(select=True)
    totalImages = len(hdrFileList)
    img_num = 1
    print (f'{totalImages} images are about to be processed. Estimated time: {totalImages*15} minutes')
    for file in hdrFileList:
        imgStartTime = time.time()
        img = HDR_Image(file)
        
        print (f'Image {img.datetime} Loaded:\n\
        Analyzed Wavelengths: {img.allowedWavelengths}')
        originalImage = img.original_image()

        print ('Destriping Image...')
        destripeImage = img.destripe_image(imshow=True)
        print (f'Image destriped at {time.time()-start:.1f} seconds')

        print ('Obtaining image and mosaic statistics...')

        shadowDict,imageStats,mosaicPixels,mosaicStats = create_arrays(r"E:/Data/20230209T095534013597/",savefolder)
        print (f'Image and mosaic statistics obtained at {time.time()-start:.1f} seconds')

        print ('Calculating Average Mosaic Reflectance...')
        averageRfl,stdRfl = img.get_average_rfl(mosaicPixels)
        print (f'Average reflectance obtained at {time.time()-start:.1f} seconds')

        print ('Making Li et al., 2018 Shadow Correction...')
        correctedImage = img.shadow_correction(averageRfl,shadowDict[img.datetime])
        print (f'Correction completed at {time.time()-start:.1f} seconds')

        print('Smoothing spectra...')
        avgSpecImg,smoothSpecImg = img.spectrum_smoothing(imshow=True,specshow=True,plottedPoints = (91,100))
        print (f'Smooth spectra obtained at {time.time()-start:.1f} seconds')

        print('Locating water-like spectra...')
        waterLocations,waterPixels_noise,waterPixels,waterDf=img.locate_ice(imshow=True)
        print (f'Water-like spectra located at {time.time()-start:.1f} seconds')

        #plt.show()

        #%%
        def save_everything_to(folder):
            print ('Saving Data...')

            try:
                os.mkdir(f'{folder}/{img.datetime}')
            except:
                pass

            waterDf.to_csv(f'{folder}/{img.datetime}/water_locations.csv')
            np.save(f'{folder}/{img.datetime}/Original_Image.npy',originalImage)
            np.save(f'{folder}/{img.datetime}/Destriped_Image.npy',destripeImage)
            np.save(f'{folder}/{img.datetime}/Correced_Image.npy',correctedImage)
            np.save(f'{folder}/{img.datetime}/Smooth_Spectrum_Image.npy',smoothSpecImg)
            np.save(f'{folder}/{img.datetime}/Water_Locations.npy',waterLocations)
            print (f'Data saved at {time.time()-start} seconds')

            print ('Saving Images...')
            tf.imwrite(f'{folder}/{img.datetime}/original.tif',originalImage,photometric='rgb')
            tf.imwrite(f'{folder}/{img.datetime}/destriped.tif',destripeImage,photometric='rgb')
            tf.imwrite(f'{folder}/{img.datetime}/corrected.tif',correctedImage,photometric='rgb')
            tf.imwrite(f'{folder}/{img.datetime}/smoothed.tif',smoothSpecImg,photometric='rgb')
            tf.imwrite(f'{folder}/{img.datetime}/water_locations.tif',waterLocations,photometric='rgb')
        
        #save_everything_to(savefolder)
        #%%
        print (f'Images Saved at {time.time()-start:.0f} seconds')

        imgtime = time.time()-imgStartTime
        if imgtime < 1:
            print(f'Image {img.datetime} completed in {imgtime*10**3:.3f} milliseconds')
        elif imgtime < 60 and imgtime > 1:
            print(f'Image {img.datetime} completed in {imgtime:.3f} seconds')
        else:
            print(f'Image {img.datetime} completed in {imgtime/60:.0f} minutes and {imgtime%60:.3f} seconds')

        print (f'{img_num} out of {totalImages} images complete ({img_num/totalImages:.0%})')
        img_num+=1


    end = time.time()
    runtime = end-start
    print (f'Image Processing finished at {datetime.datetime.now()}')
    if runtime < 1:
        print(f'Program Executed in {runtime*10**3:.3f} milliseconds')
    elif runtime < 60 and runtime > 1:
        print(f'Program Executed in {runtime:.3f} seconds')
    else:
        print(f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')
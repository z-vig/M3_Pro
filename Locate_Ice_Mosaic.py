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
import get_USGS_H2OFrost
from get_USGS_H2OFrost import get_USGS_H2OFrost
import rasterio
import threading

def find_all(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def load_tifs(pathList:list)->list:
    tif_list = []
    prog,tot = 1,len(pathList)
    for path in pathList:
        tif_list.append(tf.imread(path))
        print (f'\r{prog} of {tot} retrived. ({prog/tot:.0%})',end='\r')
        prog+=1
    return tif_list

def get_metaDataList(imgList:list)->list:
    dictList = []
    for img in imgList:
        if len(img.shape)<3:
            img=np.expand_dims(img,2)
        bandIndex = img.shape.index(min(img.shape))
        imgSave = np.moveaxis(img,bandIndex,0)
        dictList.append({'driver':'GTiff',\
                            'height':imgSave.shape[1],\
                            'width':imgSave.shape[2],\
                            'count':imgSave.shape[0],\
                            'dtype':imgSave.dtype,\
                            'interleave':'pixel',\
                            'nodata':-9999})
    return dictList

class M3_Mosaic():
    def __init__(self,rflImages:list,locImages:list,obsImages:list,stampNames:list,folderPath:str) -> None:
        self.rflImages = rflImages
        self.locImages = locImages
        self.obsImages = obsImages
        self.stampNames = stampNames
        self.folderPath = folderPath
    
    @property
    def analyzedWavelengths(self)->np.ndarray:
        df = pd.read_csv(os.path.join(self.folderPath,'bandInfo.csv'))
        bandArr = df.to_numpy()
        return bandArr[:,2]
    
    @property
    def statistics(self)->np.ndarray:
        try:
            stats_arr = np.load(f'{self.folderPath}/mosaic_stats_array.npy')
        except:
            raise FileNotFoundError('Run the mosaic data inquiry script first!')
        return stats_arr
    
    def destripe_images(self)->dict:
        startTime = time.time()
        self.destripeDict = {}
        prog,tot = 1,len(self.stampNames)
        for name,image in zip(self.stampNames,self.rflImages):
            destripeImage = destripe_image.fourier_filter(image)
            self.destripeDict.update({name:destripeImage})
            print(f'\r{prog} of {tot} ({prog/tot:.0%})',end='\r')
            prog+=1
        print (f'>>>Destriping complete in {time.time()-startTime:.1f} seconds')
        return self.destripeDict
    
    def shadow_correction(self,**kwargs)->dict:
        defaultKwargs = {'inputImageDictionary':{i:j for i,j in zip(self.stampNames,self.rflImages)},'shadowOnly':False}
        kwargs = {**defaultKwargs,**kwargs}
        startTime = time.time()
        nameList,imageList = kwargs.get('inputImageDictionary').keys(),kwargs.get('inputImageDictionary').values()

        try:
            os.mkdir(os.path.join(self.folderPath,'rfl_correction'))
        except:
            pass
        
        self.correctedImageDict = {}
        R_BIDIRECTIONAL = np.mean(self.statistics[:,:,0],axis=0)
        prog,tot = 1,len(nameList)
        for name,image in zip(nameList,imageList):
            bool_array = tf.imread(os.path.join(self.folderPath,'bright_bool_arrays',f'{name}_bright.tif'))
            shaded_regions = image[np.where(bool_array==-9999)]
            shaded_regions_corrected = shaded_regions/R_BIDIRECTIONAL
            image[np.where(bool_array==-9999)] = shaded_regions_corrected
            self.correctedImageDict.update({name:image})
            tf.imwrite(os.path.join(self.folderPath,'rfl_correction',f'{name}_corrected.tif'),image)
            print(f'\r{prog} of {tot} ({prog/tot:.0%})',end='\r')
            prog+=1
        print (f'>>>Shadow correction complete in {time.time()-startTime:.1f} seconds')
        return self.correctedImageDict
    
    def spectrum_smoothing(self,**kwargs)->dict:
        defaultKwargs = {'inputImageDictionary':self.correctedImageDict,'shadowOnly':False}
        kwargs = {**defaultKwargs,**kwargs}
        startTime = time.time()
        nameList,imageList = kwargs.get('inputImageDictionary').keys(),kwargs.get('inputImageDictionary').values()
        
        try:
            os.mkdir(os.path.join(self.folderPath,'rfl_smooth'))
        except:
            pass

        prog,tot = 1,len(nameList)
        self.smoothDict = {}
        if kwargs.get('shadowOnly')==False:
            for name,image in zip(nameList,imageList):
                avgWvl,avgSpectrumImage,smoothSpectrumImage = csi.splineFit(image,5,self.analyzedWavelengths)
                self.smoothDict.update({name:smoothSpectrumImage})
                tf.imwrite(os.path.join(self.folderPath,'rfl_smooth',f'{name}_smooth.tif'),smoothSpectrumImage.astype('float32'),photometric='rgb')
                print(f'{prog} of {tot} ({prog/tot:.0%})')
                prog+=1
        elif kwargs.get('shadowOnly')==True:
            for name,image in zip(nameList,imageList):
                bool_array = tf.imread(os.path.join(self.folderPath),'bright_bool_arrays',f'{name}_bright.tif')
                
        
        print (f'>>>Spectrum Smoothing complete in {time.time()-startTime:.1f} seconds')
    
    def locate_ice(self,**kwargs)->tuple[np.ndarray,np.ndarray,pd.DataFrame]:
        ##Loading all necessary data
        try:
            defaultKwargs = {'inputImageDictionary':self.smoothDict}
        except:
            defaultKwargs = {'inputImageDictionary':None}

        kwargs = {**defaultKwargs,**kwargs}
        startTime = time.time()
        nameList,imageList = kwargs.get('inputImageDictionary').keys(),kwargs.get('inputImageDictionary').values()

        ##Making save directory
        try:
            os.mkdir(os.path.join(self.folderPath,'water_locations'))
        except:
            pass
        
        ##Getting relevant band indices
        band1_indices = np.where((self.analyzedWavelengths>1242)&(self.analyzedWavelengths<1323))[0]
        band2_indices = np.where((self.analyzedWavelengths>1503)&(self.analyzedWavelengths<1659))[0]
        band3_indices = np.where((self.analyzedWavelengths>1945)&(self.analyzedWavelengths<2056))[0]
        allBand_indices = np.concatenate((band1_indices,band2_indices,band3_indices))

        ##For each image in batch, find all the minima
        self.waterLocateDict = {}
        prog,tot=1,len(imageList)
        for name,image,mapCoords in zip(nameList,imageList,self.locImages):
            diff_array = np.zeros(image.shape)
            for band in range(image.shape[2]-1): #The last band will be all zeros
                diff_array[:,:,band] = image[:,:,band]>image[:,:,band+1]
            
            def get_bandArray(band_indices:np.ndarray,bandName:str)->np.ndarray:
                band_arr = np.zeros((image.shape[0:2]))
                band_min_loc_arr = np.zeros((image.shape[0:2]))
                for i in range(band_indices.min()-1,band_indices.max()):
                    band_arr[np.where((diff_array[:,:,i]!=diff_array[:,:,i+1])&(diff_array[:,:,i]==True))] = 1
                    band_min_loc_arr[np.where((diff_array[:,:,i]!=diff_array[:,:,i+1])&(diff_array[:,:,i]==True))] = i+1
                return band_arr,band_min_loc_arr

            band1_array,band1_minloc = get_bandArray(band1_indices,'Band 1')
            band2_array,band2_minloc = get_bandArray(band2_indices,'Band 2')
            band3_array,band3_minloc = get_bandArray(band3_indices,'Band 3')
            allband_array = np.zeros(image.shape)
            for i in range(1,image.shape[2]-1):
                x,y  = (np.where((diff_array[:,:,i]!=diff_array[:,:,i+1])&(diff_array[:,:,i]==True)))
                allband_array[x,y,i+1] = 1

            self.waterLocations = np.zeros(band1_array.shape)
            self.waterCoords_numpy = np.where((band1_array==1)&(band2_array==1)&(band3_array==1)&(np.average(image,axis=2)>0))
            band1_minima = band1_minloc[self.waterCoords_numpy]
            band2_minima = band2_minloc[self.waterCoords_numpy]
            band3_minima = band3_minloc[self.waterCoords_numpy]
            self.waterLocations[self.waterCoords_numpy] = 1
            
            self.waterCoords_map = mapCoords[self.waterCoords_numpy[0],self.waterCoords_numpy[1],:]
            all_band_minima = np.vstack([band1_minima,band2_minima,band3_minima]).T
            
            df_data = np.concatenate([self.waterCoords_map,np.array([*self.waterCoords_numpy]).T,all_band_minima],axis=1)
            waterDf = pd.DataFrame(df_data)
            waterDf.columns = ['Longitude','Latitude','Elevation','x','y','band1_index','band2_index','band3_index']
            self.waterLocateDict.update({name:allband_array})

            waterDf.to_csv(os.path.join(self.folderPath,'water_locations',f'{name}.csv'))
            print (f'\r{prog} of {tot} ({prog/tot:.1%})',end='\r')
            prog+=1
        
        print(f'>>>Ice located in {time.time()-startTime:.1f} seconds')
        return self.waterLocateDict

    def spectral_angle_mapping(self,threshold:float,**kwargs)->tuple[np.ndarray,np.ndarray]:
        try:
            defaultKwargs = {'inputImageDict':self.smoothDict}
        except:
            defaultKwargs = {'inputImageDict':None}

        ##Making save directory
        try:
            os.mkdir(os.path.join(self.folderPath,'spectral_angle_maps'))
        except:
            pass

        kwargs = {**defaultKwargs,**kwargs}
        nameList,imageList = kwargs.get('inputImageDictionary').keys(),kwargs.get('inputImageDictionary').values()
        
        ##Getting waterlocations in numpy coordinates
        try:
            water_loc_path_list = [os.path.join(self.folderPath,'water_locations',f'{i}.csv') for i in kwargs.get('inputImageDictionary').keys()]
        except:
            raise FileNotFoundError('Run the locate_ice method first!')
        
        spec_ang_map_dict,thresh_map_dict = {},{}
        prog,tot=1,len(imageList)
        for name,image,water_loc_path in zip(nameList,imageList,water_loc_path_list):
            total_pixels = image.shape[0]*image.shape[1]
            water_loc = pd.read_csv(water_loc_path)
            x,y = np.array(water_loc.iloc[:,4]).astype(int),np.array(water_loc.iloc[:,5]).astype(int)
            water_locations_array = np.zeros((image.shape[:2]))
            water_locations_array[x,y] = 1
            
            wvl,USGS_Frost = get_USGS_H2OFrost('D:/Data/USGS_Water_Ice',self.analyzedWavelengths)
            USGS_Frost = np.expand_dims(USGS_Frost,1)
            USGS_Frost_Array = np.repeat(USGS_Frost,total_pixels,1).T
            USGS_Frost_Array = USGS_Frost_Array.reshape((image.shape[0],image.shape[1],59))

            M,I = image,USGS_Frost_Array
            specAngleMap = 180*np.arccos(np.einsum('ijk,ijk->ij',M,I)/(np.linalg.norm(M,axis=2)*np.linalg.norm(I,axis=2)))/np.pi
            spec_ang_map_dict.update({name:specAngleMap})
            no_water_indices = np.where(water_locations_array==0)
            high_spec_angle_indices = np.where(specAngleMap>threshold)

            threshIceMap = copy(image)
            threshIceMap[no_water_indices]=-9999
            threshIceMap[high_spec_angle_indices] = -9999
            thresh_map_dict.update({name:threshIceMap})

            tf.imwrite(os.path.join(self.folderPath,'spectral_angle_maps',f'{name}_SAM.tif'),specAngleMap.astype('float32'))

            print (f'\r{prog} of {tot} ({prog/tot:.1%})',end='\r')
            prog+=1

        return spec_ang_map_dict,thresh_map_dict,USGS_Frost

    def calculate_band_depth(self,**kwargs):
        try:
            defaultKwargs = {'inputImageDictionary':self.smoothDict}
        except:
            defaultKwargs = {'inputImageDictionary':None}

        kwargs = {**defaultKwargs,**kwargs}

        ##Making save directory
        try:
            os.mkdir(os.path.join(self.folderPath,'band_depth_maps'))
        except:
            pass

        try:
            os.mkdir(os.path.join(self.folderPath,'min_position_maps'))
        except:
            pass
        
        ##Finding saved water locations
        try:
            water_loc_path_list = [os.path.join(self.folderPath,'water_locations',f'{i}.csv') for i in kwargs.get('inputImageDictionary').keys()]
        except:
            raise FileNotFoundError('Run the locate_ice method first!')

        ##Finding exact indices that correlate to given shoulder values
        allowedWvl = self.analyzedWavelengths
        nameList,imageList = kwargs.get('inputImageDictionary').keys(),kwargs.get('inputImageDictionary').values()
        shoulderValues = np.array(([1130,1350],[1420,1740],[1820,2200]))
        shoulderValues_exact = np.zeros((3,2))
        n=0
        for Rs,Rl in zip(shoulderValues[:,0],shoulderValues[:,1]):
            Rs_wvl_list = [abs(Rs-index) for index in allowedWvl]
            Rl_wvl_list = [abs(Rl-index) for index in allowedWvl]
            shoulderValues_exact[n,:]=allowedWvl[np.where((Rs_wvl_list==min(Rs_wvl_list))|(Rl_wvl_list==min(Rl_wvl_list)))]
            n+=1

        for name,image,water_loc_path in zip(nameList,imageList,water_loc_path_list):
            waterDf = pd.read_csv(water_loc_path)
            band1,band2,band3 = waterDf.iloc[:,6],waterDf.iloc[:,7],waterDf.iloc[:,8]
            waterX,waterY = (np.array(waterDf)[:,4:6].astype(int).T)
            water_loc_mask = np.zeros((image.shape[:2]),dtype=bool)
            water_loc_mask[waterX,waterY] = 1

            Rc_band_loc = np.zeros((band1.shape[0],3)).astype(int)
            for row in range(Rc_band_loc.shape[0]):
                Rc_band_loc[row,:] = np.array((band1[row],band2[row],band3[row])).astype(int)

            Rc,Rs,Rl = np.full((*image.shape[:2],3),np.nan),np.full((*image.shape[:2],3),np.nan),np.full((*image.shape[:2],3),np.nan)
            lamb_c,lamb_s,lamb_l = np.full((*image.shape[:2],3),np.nan),np.full((*image.shape[:2],3),np.nan),np.full((*image.shape[:2],3),np.nan)
            
            for num,col in enumerate(Rc_band_loc.T):
                Rc[waterX,waterY,num] = image[waterX,waterY,col]

            Rs_wvlIndices = np.where((allowedWvl==shoulderValues_exact[0,0])|(allowedWvl==shoulderValues_exact[1,0])|(allowedWvl==shoulderValues_exact[2,0]))[0]
            Rl_wvlIndices = np.where((allowedWvl==shoulderValues_exact[0,1])|(allowedWvl==shoulderValues_exact[1,1])|(allowedWvl==shoulderValues_exact[2,1]))[0]

            for i in range(3):
                Rs[waterX,waterY,i] = image[waterX,waterY,Rs_wvlIndices[i]]
                Rl[waterX,waterY,i] = image[waterX,waterY,Rl_wvlIndices[i]]

                lamb_c[waterX,waterY,i] = np.array(self.analyzedWavelengths[Rc_band_loc.flatten()]).reshape(Rc_band_loc.shape)[:,i]
                lamb_s[waterX,waterY,i] = np.repeat(np.array(self.analyzedWavelengths[Rs_wvlIndices[i]]),len(waterX))
                lamb_l[waterX,waterY,i] = np.repeat(np.array(self.analyzedWavelengths[Rl_wvlIndices[i]]),len(waterX))

            #print (f'wvlIndices:{wvlIndices.shape},Rc:{Rc_wvlIndices.shape}')
            b = (lamb_c-lamb_s)/(lamb_l-lamb_s)
            a = 1-b
            
            Rc_star = a*Rs+b*Rl
            
            test_ar = b[np.where(np.isnan(b)==0)]
            test_ar2 = a[np.where(np.isnan(b)==0)]
            test_ar3 = Rc[np.where(np.isnan(b)==0)]
            test_ar4 = Rs[np.where(np.isnan(b)==0)]
            test_ar5 = Rl[np.where(np.isnan(b)==0)]
            # print (Rc_band_loc[0])
            # print (waterX[0],waterY[0], image[15,147,Rc_band_loc[0]])#,image[15,147,:lamb_s[0]],image[15,147,:lamb_l[0]])
            # print (f'b: {test_ar.reshape(int(test_ar.shape[0]/3),3)[0,:]}')
            # print (f'a: {test_ar2.reshape(int(test_ar2.shape[0]/3),3)[0,:]}')
            # print (f'Rc: {test_ar3.reshape(int(test_ar.shape[0]/3),3)[0,:]}')
            # print (f'Rs: {test_ar4.reshape(int(test_ar.shape[0]/3),3)[0,:]}')
            # print (f'Rl: {test_ar5.reshape(int(test_ar.shape[0]/3),3)[0,:]}')

            band_depth_map = 1-(Rc/Rc_star)
            min_position_map = np.full_like(band_depth_map,np.nan)
            # print (min_position_map.shape)
            # print ((np.where(np.isnan(band_depth_map[:,:,0])==0)))
            # print (min_position_map[(np.where(np.isnan(band_depth_map[:,:,0])==0))].shape)
            for i in range(3):
                min_position_map[(np.where(np.isnan(band_depth_map[:,:,0])==0))]=Rc_band_loc

           #=Rc_band_loc[:,0]


            # tf.imwrite(os.path.join(self.folderPath,'band_depth_maps',f'{name}_BD_map.tif'),band_depth_map)
            # tf.imwrite(os.path.join(self.folderPath,'min_position_maps',f'{name}_min_positions.tif'),min_position_map)

        return band_depth_map,min_position_map
    

if __name__ == '__main__':
    print ('Select Analysis Folder:')
    folderPath = askdir()
    all_rfl_paths = [os.path.join(folderPath,'rfl_cropped',i) for i in os.listdir(os.path.join(folderPath,'rfl_cropped'))]
    all_loc_paths = [os.path.join(folderPath,'loc_cropped',i) for i in os.listdir(os.path.join(folderPath,'loc_cropped'))]
    all_obs_paths = [os.path.join(folderPath,'obs_cropped',i) for i in os.listdir(os.path.join(folderPath,'obs_cropped'))]

    with open(os.path.join(folderPath,'stampNames.txt')) as f:
        all_names = f.readlines()
    all_names = [i[:-2] for i in all_names]
    
    def batch_list(input,n):
        return [input[i:i+n] for i in range(0,len(input),n)]

    N = 5
    batch_rfl_paths = batch_list(all_rfl_paths,N)
    batch_loc_paths = batch_list(all_loc_paths,N)
    batch_obs_paths = batch_list(all_obs_paths,N)
    all_names_split = batch_list(all_names,N)

    prog,tot=0,len(all_rfl_paths)
    for n in range(len(batch_rfl_paths)):
        print ('Retrieving RFL Tifs...')
        batch_rfl = load_tifs(batch_rfl_paths[n])
        print ('\nRetrieving LOC Tifs...')
        batch_loc = load_tifs(batch_loc_paths[n])
        print ('\nRetrieving OBS Tifs...')
        batch_obs = load_tifs(batch_obs_paths[n])
        batch_names = all_names_split[n]
    
        batchMosaic = M3_Mosaic(batch_rfl,batch_loc,batch_obs,batch_names,folderPath)
        prog = prog+len(batchMosaic.stampNames)
        print (f'\n-----Beginning Mosaic analysis of {len(batchMosaic.stampNames)} ({prog} of {tot})images-----')
        print ('Running Li et al., 2018 Shadow Correction...')
        cordict = batchMosaic.shadow_correction()
        print ('Smoothing spectrum...')
        smoothdict = batchMosaic.spectrum_smoothing()

        print ('\nRemoval from memory...')
        del batch_rfl,batch_loc,batch_obs
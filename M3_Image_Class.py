'''
Class for .tif images downloaded from the PDS. See download_PDS_files.py to process download files up to this point
'''
#Importing necessary modules
import numpy as np
import os
import pandas as pd
import time
import tifffile as tf
import copy

#Importing project defined modules
import destripe_image
import cubic_spline_image as csi

#Defining helper functions
def get_summary_dataframe(boolean_array:np.ndarray,map_coord_array:np.ndarray,value_name_list:list,*args)->pd.DataFrame: #Function to format a summary dataframe of analyses
    true_coords = np.where(boolean_array==True) #Gets positive values from boolean_array
    value_array = np.array([i[true_coords] for i in args]).T #N by len(args) array of values at each True boolean pixel
    map_coords = map_coord_array[true_coords[0],true_coords[1],:] #Gets map coordinates from map_coord_array for each positive value

    summary_array = np.concatenate([map_coords,np.array([true_coords[0],true_coords[1]]).T,value_array],axis=1) #Defines summary array by concatenteing map coordinates,
                                                                                                                #numpy coordinates and pixel values
    summary_df = pd.DataFrame(summary_array)
    summary_df.columns = ['Longitude','Latitude','Elevation','x','y',*value_name_list] #Puts everything into pandas dataframe

    return summary_df

class M3_Stamp():
    '''
    Class containing the properties of and methods to be performed on individual M3 stamps. The methods of this class define the M3 image processing pipeline as follows:

    1. destripe_image // attempts to remove strping affects commonly found in M3 stamps.
    2. shadow_correction // Uses the method from Li et al., 2018 to remove re-reflectance affects from shadowed M3 pixels.
    3. spectrum_smoothing // Smoothes each pixel of an M3 stamp down the spectral dimension.
    4. ice_band_pos_map // Creates a boolean map of pixel with absorption bands in the range of ice and a .csv summary file with columns: [Long,Lat,Elev,x,y,band1,band2,band3]
    5. spectral_angle_map // Given a reference spectrum, creates a boolean map based off of a threshold spectral angle and a map of all spectral angle values.
    6. euclidian_distance_map // Given a reference spectrum, creates a boolean map based off a threshold distance and a map of all euclidian distance values.
    7. band_depth_map // Creates a boolean map based on a threshold applied to every water ice band and a 3-dimensional image with each of the 3 water ice band values

    **Note that the input_im variable will allow users to begin analyses at whatever stage is necessary for their script. For example, if a script is written to destripe every 
    image using the M3_stamp.destripe_image method, the user can now take the output images from that script and use them as inputs into a script that, say, performs the 
    M3_stamp.shadow_correction method on each image.
    '''
    def __init__(self,input_im:np.ndarray,loc_im:np.ndarray,obs_im:np.ndarray,stamp_name:str,folder_path:str) -> None:
        self.input_im = input_im #Numpy array containing the input M3 stamp.
        self.loc_im = loc_im #Numpy array containing the geospatial information for each M3 stamp (3 dimensions: lat,long,elev.)
        self.obs_im = obs_im #Numpy array containing the observation orientation information for each M3 stamp
        self.stamp_name = stamp_name #The name of the specific M3 stamp as a string
        self.folder_path = folder_path #The path to the output folder as a string
    
    @property
    def analyzed_wavelengths(self)->np.ndarray: #Property that gets the analyzed wavelengths for the M3 stamp (all stamps are the same)
        df = pd.read_csv(os.path.join(self.folder_path,'bandInfo.csv')) #Accesses the bandInfo.csv file that is made during the mosaic_data_inquiry.py script
        bandArr = df.to_numpy()
        return bandArr[:,2] #Returns numpy array of analyzed wavelengths
    
    @property
    def statistics(self)->np.ndarray: #Property that gets the statistics for each M3 stamp from a numpy save file
        try:
            stats_arr = np.load(f'{self.folder_path}/mosaic_stats_array.npy')
        except:
            raise FileNotFoundError('Run the mosaic data inquiry script first!')
        return stats_arr #Returns an array with the average spectrum of each M3 stamp and the standard deviation of each M3 stamp
    
    
    def destripe_image(self,**kwargs)->np.ndarray:
        defaultKwargs = {'save_step':False}
        kwargs = {**defaultKwargs,**kwargs}
        startTime = time.time()
        self.destripe_im = destripe_image.fourier_filter(self.input_im)
        if kwargs.get('save_step')==True:
            try:
                os.mkdir(os.path.join(self.folder_path,'rfl_destriped'))
            except:
                pass
            tf.imwrite(os.path.join(self.folder_path,'rfl_destriped',f'{self.stamp_name}.tif'),self.destripe_im.astype('float32'))
        else:
            pass
        return self.destripe_im
    
    
    def shadow_correction(self,**kwargs)->np.ndarray:
        #Defining keyword arguments
        defaultKwargs = {'save_step':False}
        kwargs = {**defaultKwargs,**kwargs}
        #Beginning Li et al., 2018 Correction
        startTime = time.time()
        self.corrected_im = copy.copy(self.input_im) #copy for mutability

        R_BIDIRECTIONAL = np.mean(self.statistics[:,:,0],axis=0) #Bidirectional reflectance defined as the average of all bright pixels over the entire image mosaic (polar region)

        bright_bool_array = tf.imread(os.path.join(self.folder_path,'bright_bool_arrays',f'{self.stamp_name}_bright.tif')) #True is a bright pixel, false is a dark pixel
        shaded_regions = self.input_im[np.where(bright_bool_array==-9999)]
        shaded_regions_corrected = shaded_regions/R_BIDIRECTIONAL #Li et al., 2018 Correction term
        self.corrected_im[np.where(bright_bool_array==-9999)] = shaded_regions_corrected

        if kwargs.get('save_step')==True:
            try:
                os.mkdir(os.path.join(self.folder_path,'rfl_correction'))
            except:
                pass
            tf.imwrite(os.path.join(self.folder_path,'rfl_correction',f'{self.stamp_name}.tif'),self.corrected_im.astype('float32'))
        else:
            pass

        return self.corrected_im
    
    def spectrum_smoothing(self,**kwargs)->np.ndarray:
        #Defining keyword arguments
        defaultKwargs = {'save_step':False}
        kwargs = {**defaultKwargs,**kwargs}
        #Beginning cubic spline fitting for each pixel
        startTime = time.time()
        avgWvl,avg_spec_im,self.smooth_im = csi.spline_fit(self.input_im,5,self.analyzed_wavelengths)
 
        if kwargs.get('save_step')==True:
            try:
                os.mkdir(os.path.join(self.folder_path,'rfl_smooth'))
            except:
                pass
            tf.imwrite(os.path.join(self.folder_path,'rfl_smooth',f'{self.stamp_name}.tif'),self.smooth_im.astype('float32'),photometric='rgb')
        else:
            pass

        return self.smooth_im
    
    def ice_band_pos_map(self,**kwargs)->tuple[np.ndarray,pd.DataFrame]:
        #Defining keyword arguments
        defaultKwargs = {'save_step':False}
        kwargs = {**defaultKwargs,**kwargs}
        #Beginning location of each pixel with water-like absorption bands ({1.242-1.323},{1.503-1.1659},{1.945-2.056})
        startTime = time.time()
        self.ice_band_pos_map = copy.copy(self.input_im) #copy for mutability
        
        ##Getting relevant band indices
        band1_indices = np.where((self.analyzed_wavelengths>1242)&(self.analyzed_wavelengths<1323))[0]
        band2_indices = np.where((self.analyzed_wavelengths>1503)&(self.analyzed_wavelengths<1659))[0]
        band3_indices = np.where((self.analyzed_wavelengths>1945)&(self.analyzed_wavelengths<2056))[0]
        allBand_indices = np.concatenate((band1_indices,band2_indices,band3_indices))

        ##For a given input image, find all the minima. diff_array is a boolean array where True indicates an increase spectrum at that band and False indicates a decreasing spectrum
        diff_array = np.zeros(self.input_im.shape)
        for band in range(self.input_im.shape[2]-1): #The last band will be all zeros
            diff_array[:,:,band] = self.input_im[:,:,band]>self.input_im[:,:,band+1]

        def get_band_array(band_indices:np.ndarray)->None:
            '''
            Function that uses a differential array to find where the spectra experiences local minima for each pixel using a vectorized method.
            The input band_indices instructs the function to only look in that subset of bands, which will correspond to each of the three water ice absorption bands
            band_arr is a boolean image where each pixel shows where the image has a minima for a given band
            band_min_loc_arr is an image where the pixels are the indices of the minimum that was found
            band_arr says "these are the pixels that experience this absorption minima" and band_min_loc_arr says "here is the band index where this minima occurs"
            '''
            band_arr = np.zeros((self.input_im.shape[0:2]))
            band_min_loc_arr = np.zeros((self.input_im.shape[0:2]))
            for i in range(band_indices.min()-1,band_indices.max()):
                band_arr[np.where((diff_array[:,:,i]!=diff_array[:,:,i+1])&(diff_array[:,:,i]==True))] = 1
                band_min_loc_arr[np.where((diff_array[:,:,i]!=diff_array[:,:,i+1])&(diff_array[:,:,i]==True))] = i+1
            return band_arr,band_min_loc_arr

        #Creating absorption band location arrays for each of the three water ice absorption bands (1,2,3)
        band1_array,band1_minloc = get_band_array(band1_indices)
        band2_array,band2_minloc = get_band_array(band2_indices)
        band3_array,band3_minloc = get_band_array(band3_indices)

        ice_bool = band1_array.astype(bool)&band2_array.astype(bool)&band3_array.astype(bool)&(np.mean(self.input_im,axis=2)>0).astype(bool)
        band_loc_df = get_summary_dataframe(ice_bool,self.loc_im,['band1_index','band2_index','band3_index'],band1_minloc,band2_minloc,band3_minloc)
        
        if kwargs.get('save_step')==True:
            try:
                os.mkdir(os.path.join(self.folder_path,'ice_band_bool'))
            except:
                pass
            try:
                os.mkdir(os.path.join(self.folder_path,'ice_band_locations'))
            except:
                pass

            #band_loc_df.to_csv(os.path.join(self.folder_path,'ice_band_location_summary.csv'))
            tf.imwrite(os.path.join(self.folder_path,'ice_band_bool',f'{self.stamp_name}.tif'),ice_bool.astype('float32'))
            band_loc_df.to_csv(os.path.join(self.folder_path,'ice_band_locations',f'{self.stamp_name}.csv'))
        
        return ice_bool,band_loc_df

    def spectral_angle_map(self,reference_spectrum:np.ndarray,threshold:float,**kwargs)->np.ndarray:
        #Defining keyword arguments
        defaultKwargs = {'save_step':False}
        kwargs = {**defaultKwargs,**kwargs}
        #Beginning location of each pixel with water-like absorption bands ({1.242-1.323},{1.503-1.1659},{1.945-2.056})
        start_time = time.time()
        total_pixels = self.input_im.shape[0]*self.input_im.shape[1]
        
        reference_spectrum = np.expand_dims(reference_spectrum,1)
        ref_spec_array = np.repeat(reference_spectrum,total_pixels,1).T
        ref_spec_array = ref_spec_array.reshape((self.input_im.shape[0],self.input_im.shape[1],59))

        M,I = self.input_im,ref_spec_array
        self.spec_ang_map = 180*np.arccos(np.einsum('ijk,ijk->ij',M,I)/(np.linalg.norm(M,axis=2)*np.linalg.norm(I,axis=2)))/np.pi
        spec_ang_bool = (self.spec_ang_map<threshold).astype(bool)
        spec_ang_threshold_df = get_summary_dataframe(spec_ang_bool,self.loc_im,['Spectral Angle (\u00B0)'],self.spec_ang_map)

        if kwargs.get('save_step')==True:
            try:
                os.mkdir(os.path.join(self.folder_path,f'spectral_angle_bool_{threshold}'))
            except:
                pass
            try:
                os.mkdir(os.path.join(self.folder_path,f'spectral_angle_values'))
            except:
                pass

            #band_loc_df.to_csv(os.path.join(self.folder_path,'ice_band_location_summary.csv'))
            tf.imwrite(os.path.join(self.folder_path,f'spectral_angle_values',f'{self.stamp_name}.tif'),self.spec_ang_map.astype('float32'))
            tf.imwrite(os.path.join(self.folder_path,f'spectral_angle_bool_{threshold}',f'{self.stamp_name}.tif'),spec_ang_bool.astype('float32'))

        return spec_ang_threshold_df,spec_ang_bool
    
    def euclidian_distance_map(self,reference_spectrum:np.ndarray,threshold:float,**kwargs)->np.ndarray:
        #Defining keyword arguments
        defaultKwargs = {'save_step':False}
        kwargs = {**defaultKwargs,**kwargs}
        #Beginning location of each pixel with water-like absorption bands ({1.242-1.323},{1.503-1.1659},{1.945-2.056})
        start_time = time.time()
        total_pixels = self.input_im.shape[0]*self.input_im.shape[1]
        
        reference_spectrum = np.expand_dims(reference_spectrum,1)
        ref_spec_array = np.repeat(reference_spectrum,total_pixels,1).T
        ref_spec_array = ref_spec_array.reshape((self.input_im.shape[0],self.input_im.shape[1],59))

        M,I = self.input_im,ref_spec_array
        self.euc_dist_map = np.linalg.norm(M-I,axis=2)
        euc_dist_bool = (self.euc_dist_map<threshold).astype(bool)
        euc_dist_threshold_df = get_summary_dataframe(euc_dist_bool,self.loc_im,['Euclidian Distance'],self.euc_dist_map)

        if kwargs.get('save_step')==True:
            try:
                os.mkdir(os.path.join(self.folder_path,f'euclidian_distance_bool_{threshold}'))
            except:
                pass
            try:
                os.mkdir(os.path.join(self.folder_path,f'euclidian_distance_values'))
            except:
                pass

            #band_loc_df.to_csv(os.path.join(self.folder_path,'ice_band_location_summary.csv'))
            tf.imwrite(os.path.join(self.folder_path,f'euclidian_distance_values',f'{self.stamp_name}.tif'),self.euc_dist_map.astype('float32'))
            tf.imwrite(os.path.join(self.folder_path,f'euclidian_distance_bool_{threshold}',f'{self.stamp_name}.tif'),euc_dist_bool.astype('float32'))

        return euc_dist_threshold_df,euc_dist_bool

    
    def band_depth_map(self,ice_bool_array:np.ndarray,ice_band_df:np.ndarray,threshold:float,**kwargs):
        #Defining keyword arguments
        defaultKwargs = {'save_step':False}
        kwargs = {**defaultKwargs,**kwargs}
        #Beginning band depth mapping
        start_time = time.time()

        ##Finding exact indices that correlate to given shoulder values
        allowed_wvl = self.analyzed_wavelengths
        shoulderValues = np.array(([1130,1350],[1420,1740],[1820,2200]))
        shoulderValues_exact = np.zeros((3,2))
        n=0
        for Rs,Rl in zip(shoulderValues[:,0],shoulderValues[:,1]):
            Rs_wvl_list = [abs(Rs-index) for index in allowed_wvl]
            Rl_wvl_list = [abs(Rl-index) for index in allowed_wvl]
            shoulderValues_exact[n,:]=allowed_wvl[np.where((Rs_wvl_list==min(Rs_wvl_list))|(Rl_wvl_list==min(Rl_wvl_list)))]
            n+=1

        band1,band2,band3 = ice_band_df.iloc[:,6],ice_band_df.iloc[:,7],ice_band_df.iloc[:,8]
        iceX,iceY = (np.array(ice_band_df)[:,4:6].astype(int).T)

        Rc_band_index = np.zeros((band1.shape[0],3)).astype(int)
        for row in range(Rc_band_index.shape[0]):
            Rc_band_index[row,:] = np.array((band1[row],band2[row],band3[row])).astype(int)

        Rc,Rs,Rl = np.full((*self.input_im.shape[:2],3),np.nan),np.full((*self.input_im.shape[:2],3),np.nan),np.full((*self.input_im.shape[:2],3),np.nan)
        lamb_c,lamb_s,lamb_l = np.full((*self.input_im.shape[:2],3),np.nan),np.full((*self.input_im.shape[:2],3),np.nan),np.full((*self.input_im.shape[:2],3),np.nan)
        
        for num,col in enumerate(Rc_band_index.T):
            Rc[iceX,iceY,num] = self.input_im[iceX,iceY,col]

        Rs_wvlIndices = np.where((allowed_wvl==shoulderValues_exact[0,0])|(allowed_wvl==shoulderValues_exact[1,0])|(allowed_wvl==shoulderValues_exact[2,0]))[0]
        Rl_wvlIndices = np.where((allowed_wvl==shoulderValues_exact[0,1])|(allowed_wvl==shoulderValues_exact[1,1])|(allowed_wvl==shoulderValues_exact[2,1]))[0]

        for i in range(3):
            Rs[iceX,iceY,i] = self.input_im[iceX,iceY,Rs_wvlIndices[i]]
            Rl[iceX,iceY,i] = self.input_im[iceX,iceY,Rl_wvlIndices[i]]
            lamb_c[iceX,iceY,i] = np.array(self.analyzed_wavelengths[Rc_band_index.flatten()]).reshape(Rc_band_index.shape)[:,i]
            lamb_s[iceX,iceY,i] = np.repeat(np.array(self.analyzed_wavelengths[Rs_wvlIndices[i]]),len(iceX))
            lamb_l[iceX,iceY,i] = np.repeat(np.array(self.analyzed_wavelengths[Rl_wvlIndices[i]]),len(iceX))

        #print (f'wvlIndices:{wvlIndices.shape},Rc:{Rc_wvlIndices.shape}')
        b = (lamb_c-lamb_s)/(lamb_l-lamb_s)
        a = 1-b
        
        Rc_star = a*Rs+b*Rl

        band_depth_map = 1-(Rc/Rc_star)
        band_depth_bool = np.zeros(band_depth_map.shape[:2]).astype(bool)

        bd_values = band_depth_map[np.where(np.isnan(band_depth_map)==False)].reshape(-1,3)
        above_thresh_index = np.where((bd_values[:,0]>0.1)&(bd_values[:,1]>0.1)&(bd_values[:,2]>threshold))[0]
        pos_det_loc = np.array(np.where(np.isnan(band_depth_map[:,:,0])==False))[:,above_thresh_index]
        band_depth_bool[(pos_det_loc[0,:],pos_det_loc[1,:])] = True

        band_depth_thresh_df = get_summary_dataframe(band_depth_bool,self.loc_im,\
                                                     ['Band Depth 1','Band Depth 2','Band Depth 3'],\
                                                        band_depth_map[:,:,0],band_depth_map[:,:,1],band_depth_map[:,:,2])
        
        if kwargs.get('save_step')==True:
            try:
                os.mkdir(os.path.join(self.folder_path,f'band_depth_bool_{threshold:.2f}'))
            except:
                pass

            try:
                os.mkdir(os.path.join(self.folder_path,f'band_depth_values'))
            except:
                pass
            
            tf.imwrite(os.path.join(self.folder_path,f'band_depth_values',f'{self.stamp_name}.tif'),band_depth_map.astype('float32'))
            tf.imwrite(os.path.join(self.folder_path,f'band_depth_bool_{threshold:.2f}',f'{self.stamp_name}.tif'),band_depth_bool.astype('float32'))

        return band_depth_bool,band_depth_thresh_df
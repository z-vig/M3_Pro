'''
Script for utilizing the M3 Image Class to run a spectral angle mapping algorithm with reference to a given reference spectrum
'''
#%%
import M3_Image_Class
from importlib import reload
reload(M3_Image_Class)
from M3_Image_Class import M3_Stamp
from get_USGS_H2OFrost import get_USGS_H2OFrost

import numpy as np
import pandas as pd
import os
import time
from tkinter.filedialog import askdirectory as askdir
import tifffile as tf

def map_sa(save_step:bool,reference_spectrum:np.ndarray,ref_spec_name:str)->None:
    print ('Select Analysis Folder:')
    #folder_path = 'D:/Data/Ice_Pipeline_Out_8-7-23'
    folder_path = askdir()
    print (f'{folder_path} selected for analysis')
    print ('Select input folder for spectral angle mapping step:')
    input_path = askdir()
    #input_path = 'D:/Data/Ice_Pipeline_Out_8-7-23/rfl_smooth_complete'
    THRESH = float(input('Enter the spectral angle threshold value for ice-positive detection\n>>>'))
    print (f'The {os.path.basename(input_path)} folder has been selected as the processing input') #os.path.join(folder_path,'rfl_cropped')
    all_input_paths = [os.path.join(input_path,i) for i in os.listdir(input_path)]
    all_loc_paths = [os.path.join(folder_path,'loc_cropped',i) for i in os.listdir(os.path.join(folder_path,'loc_cropped'))]
    all_obs_paths = [os.path.join(folder_path,'obs_cropped',i) for i in os.listdir(os.path.join(folder_path,'obs_cropped'))]

    with open(os.path.join(folder_path,'stampNames.txt')) as f:
        stamp_names = f.readlines()
    stamp_names = [i[:-2] for i in stamp_names]

    prog,tot = 0,len(stamp_names)
    start_time = time.time()
    df_list = []
    for input_path,loc_path,obs_path,stamp_name in zip(all_input_paths,all_loc_paths,all_obs_paths,stamp_names):
        input_im,loc_im,obs_im = tf.imread(input_path),tf.imread(loc_path),tf.imread(obs_path)
        stamp_object = M3_Stamp(input_im,loc_im,obs_im,stamp_name,folder_path)
        summary_df,spec_ang_bool = stamp_object.spectral_angle_map(reference_spectrum=reference_spectrum,ref_spec_name=ref_spec_name,threshold=THRESH,save_step=save_step)
        df_list.append(summary_df)
        prog+=1
        print (f'\rAnalysis for {stamp_name} complete ({prog/tot:.2%})',end='\r')
    
    if save_step==True:
        all_stamps_df = pd.concat(df_list,ignore_index=True)
        all_stamps_df.to_csv(os.path.join(folder_path,f'spectral_angle_{THRESH}_summary.csv'))
    else:
        pass

    print (f'\n>>>Spectral Angle Mapping complete for {THRESH} degrees!')

if __name__ == "__main__":
    ##Running spectral angle mapping step with reference to USGS frost spectrum
    folder_path = 'D:/Data/Ice_Pipeline_Out_8-7-23'
    sample_in,sample_loc,sample_obs = (os.path.join(folder_path,i,os.listdir(os.path.join(folder_path,i))[0]) for i in ['rfl_smooth_complete','loc_cropped','obs_cropped'])
    sample_in,sample_loc,sample_obs = [tf.imread(i) for i in [sample_in,sample_loc,sample_obs]]

    with open(os.path.join(folder_path,'stampNames.txt')) as f:
        stamp_names = f.readlines()
    stamp_names = [i[:-2] for i in stamp_names]

    analyzed_wavelengths = M3_Stamp(sample_in,sample_loc,sample_obs,stamp_names[0],folder_path).analyzed_wavelengths

    wvl,USGS_frost = get_USGS_H2OFrost('D:/Data/USGS_Water_Ice',analyzed_wavelengths)
    map_sa(False,reference_spectrum=USGS_frost)
    


'''
Script for utilizing the M3 Image Class to run a band depth mapping algorithm for each water ice absorption band
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

def bd_mapping(save_step:bool)->None:
    print ('Select Analysis Folder:')
    #folder_path = 'D:/Data/Ice_Pipeline_Out_8-7-23'
    folder_path = askdir()
    print (f'{folder_path} selected for analysis')
    print ('Select input folder for band depth mapping step:')
    input_path = askdir()
    #input_path = 'D:/Data/Ice_Pipeline_Out_8-7-23/rfl_smooth_complete'
    print (f'The {os.path.basename(input_path)} folder has been selected as the processing input') #os.path.join(folder_path,'rfl_cropped')

    THRESH = float(input('Select threshold value for band depth\n>>>'))

    if ('ice_band_bool' in os.listdir(folder_path)) and ('ice_band_locations' in os.listdir(folder_path)):
        ice_band_bool_path = os.path.join(folder_path,'ice_band_bool')
        ice_band_loc_path = os.path.join(folder_path,'ice_band_locations')
    else:
        raise OSError('ice_band_bool and ice_band_locations cannot be found. Please run the ice band position mapping first.')

    all_input_paths = [os.path.join(input_path,i) for i in os.listdir(input_path)]
    all_loc_paths = [os.path.join(folder_path,'loc_cropped',i) for i in os.listdir(os.path.join(folder_path,'loc_cropped'))]
    all_obs_paths = [os.path.join(folder_path,'obs_cropped',i) for i in os.listdir(os.path.join(folder_path,'obs_cropped'))]
    all_ice_bool_paths = [os.path.join(ice_band_bool_path,i) for i in os.listdir(os.path.join(ice_band_bool_path))]
    all_ice_loc_paths = [os.path.join(ice_band_loc_path,i) for i in os.listdir(os.path.join(ice_band_loc_path))]

    with open(os.path.join(folder_path,'stampNames.txt')) as f:
        stamp_names = f.readlines()
    stamp_names = [i[:-2] for i in stamp_names]

    prog,tot = 0,len(stamp_names)
    start_time = time.time()
    df_list = []
    for input_path,loc_path,obs_path,stamp_name,ice_bool_path,ice_loc_path in\
            zip(all_input_paths,all_loc_paths,all_obs_paths,stamp_names,all_ice_bool_paths,all_ice_loc_paths):
        
        input_im,loc_im,obs_im,ice_bool,ice_loc = tf.imread(input_path),tf.imread(loc_path),tf.imread(obs_path),tf.imread(ice_bool_path),pd.read_csv(ice_loc_path)
        stamp_object = M3_Stamp(input_im,loc_im,obs_im,stamp_name,folder_path)
        bd_bool,summary_df = stamp_object.band_depth_mapping(ice_bool_array=ice_bool,ice_band_df=ice_loc,threshold=THRESH,save_step=save_step)
        df_list.append(summary_df)
        prog+=1
        print (f'\rAnalysis for {stamp_name} complete ({prog/tot:.2%})',end='\r')
        del input_im,loc_im,obs_im
    
    all_stamps_df = pd.concat(df_list,ignore_index=True)
    all_stamps_df.to_csv(os.path.join(folder_path,f'band_depth_{THRESH}_summary.csv'))

    print (f'\n>>>Band Depth Mapping complete!\n>>>Script completed in {(time.time()-start_time)/60:.2f} minutes.')

if __name__ == "__main__":
    ##Running band depth mapping step
    bd_mapping(True)
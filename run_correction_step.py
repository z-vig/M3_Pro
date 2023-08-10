'''
Script for utilizing the M3 Image Class to run the Li et al., 2018 Shadow Correction
'''

##Importing Modules
import M3_Image_Class
from importlib import reload
reload(M3_Image_Class)
from M3_Image_Class import M3_Stamp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf
import rasterio as rio
import os
from os import path as path
import time
from tkinter.filedialog import askdirectory as askdir

def correction(save_step:bool)->None:
    print ('Select Analysis Folder:')
    folder_path = askdir()
    print (f'{folder_path} selected for analysis')
    input_path = askdir()
    print ('Select input folder for reflected light correction step:')
    print (f'The {os.path.basename(input_path)} folder has been selected as the processing input') #os.path.join(folder_path,'rfl_cropped')
    all_input_paths = [os.path.join(input_path,i) for i in os.listdir(input_path)]
    all_loc_paths = [os.path.join(folder_path,'loc_cropped',i) for i in os.listdir(os.path.join(folder_path,'loc_cropped'))]
    all_obs_paths = [os.path.join(folder_path,'obs_cropped',i) for i in os.listdir(os.path.join(folder_path,'obs_cropped'))]

    with open(os.path.join(folder_path,'stampNames.txt')) as f:
        stamp_names = f.readlines()
    stamp_names = [i[:-2] for i in stamp_names]

    prog,tot = 0,len(stamp_names)
    start_time = time.time()
    for input_path,loc_path,obs_path,stamp_name in zip(all_input_paths,all_loc_paths,all_obs_paths,stamp_names):
        input_im,loc_im,obs_im = tf.imread(input_path),tf.imread(loc_path),tf.imread(obs_path)
        stamp_object = M3_Stamp(input_im,loc_im,obs_im,stamp_name,folder_path)
        stamp_object.shadow_correction(save_step=save_step)
        prog+=1
        print (f'\rAnalysis for {stamp_name} complete ({prog/tot:.2%})',end='\r')

if __name__ == "__main__":
    ##Running shadow correction step
    correction(True)

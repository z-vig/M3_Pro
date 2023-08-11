import M3_Image_Class
from importlib import reload
reload(M3_Image_Class)
from M3_Image_Class import M3_Stamp

import os
from tkinter.filedialog import askdirectory as askdir
import tifffile as tf

def destripe(save_step:bool)->None:
    print ('Select analysis folder:')
    folder_path = askdir()
    print (f'{folder_path} selected for analysis')
    print ('Select input folder for spectral angle mapping step:')
    input_path = askdir()
    print (f'The {os.path.basename(input_path)} folder has been selected as the processing input')
    all_input_paths = [os.path.join(input_path,i) for i in os.listdir(os.path.join(input_path))]
    all_loc_paths = [os.path.join(folder_path,'loc_cropped',i) for i in os.listdir(os.path.join(folder_path,'loc_cropped'))]
    all_obs_paths = [os.path.join(folder_path,'obs_cropped',i) for i in os.listdir(os.path.join(folder_path,'obs_cropped'))]

    with open(os.path.join(folder_path,'stampNames.txt')) as f:
        stamp_names = f.readlines()
    stamp_names = [i[:-2] for i in stamp_names]

    prog,tot = 0,len(stamp_names)
    for rfl_path,loc_path,obs_path,stamp_name in zip(all_input_paths,all_loc_paths,all_obs_paths,stamp_names):
        rfl_im,loc_im,obs_im = tf.imread(rfl_path),tf.imread(loc_path),tf.imread(obs_path)
        stamp_object = M3_Stamp(rfl_im,loc_im,obs_im,stamp_name,folder_path)
        stamp_object.destripe_images(save_step=save_step)
        prog+=1
        print (f'\rAnalysis for {stamp_name} complete ({prog/tot:.2%})',end='\r')
        del rfl_im,loc_im,obs_im

if __name__ == "__main__":
    destripe(True)
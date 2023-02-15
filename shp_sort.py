# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:04:57 2023

@author: zacha
"""
from M3_UnZip import M3_unzip
import os
import os.path as path
import shutil

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

##Sorting shape files

if 'hdr_file_list' in locals():
    print ('Necessary Variables are Defined')
else:
    hdr_file_list,hdr_files_path = M3_unzip(False,folder=r"D:/Data/20230209T095534013597")

ex_files_path = hdr_files_path[:find(hdr_files_path,'/')[-1]]

shp_files = os.path.join(ex_files_path,"shape_files")

try:
    os.mkdir(os.path.join(ex_files_path,"shp"))
    os.mkdir(os.path.join(ex_files_path,"prj"))
    os.mkdir(os.path.join(ex_files_path,"dbf"))
    print ('Creating Directories...')
except:
    pass

if os.listdir(os.path.join(ex_files_path,"prj")) == []:
    print ('Copying Files...')
    for i in os.listdir(shp_files):
        for file in os.listdir(os.path.join(shp_files,i)):
            try:
                existing_files = os.listdir(os.path.join(ex_files_path,file[-3:]))
            except:
                pass
            
            try:
                shutil.copyfile(os.path.join(shp_files,i,file),os.path.join(ex_files_path,file[-3:],file))
            except:
                pass

for walk_list in os.walk(shp_files):
    for file in walk_list[2]:
        print (file[0:find(file,'_')[-1]])
        shutil.copyfile(path.join(shp_files,file[0:find(file,'_')[-1]],file),path.join(ex_files_path,"all",file))
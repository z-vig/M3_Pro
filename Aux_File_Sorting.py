# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:58:50 2023

@author: zacha
"""

import os
import shutil
import numpy

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def str_insert(s,ch,index):
    new = list(s)
    new.insert(index,ch)
    new = ''.join(new)
    return new

home_dir = r"D:/Data/Lunar_Ice_Images"

allBands_tiff = []
allBands_aux = []
allBands_tfw = []
name_list = []
allBands_paths = []
for i in os.walk(home_dir):
    if i[0].find('all') > -1:
        allBands_tiff.append(i[2][0])
        allBands_paths.append(i[0])
        
for i in os.walk(home_dir):
    if i[0].find('1289.41') > -1:
        name = (os.path.split(os.path.split(i[0])[0])[1])
        name_list.append(name)
        for file in i[2]:
            if file.find('aux.xml') > -1:
                allBands_aux.append(file)
            elif file.find('.tfwx') > -1:
                allBands_tfw.append(file)

# =============================================================================
# print (allBands_aux)
# print (allBands_tfw)        
# =============================================================================

# =============================================================================
# print (allBands)
# print (allBands_paths)
# =============================================================================

try:
    for path in allBands_paths:
        os.mkdir(os.path.join(os.path.split(path)[0],"Raster_World_Files"))
except:
    pass

for path,aux,tfw,name in zip(allBands_paths,allBands_aux,allBands_tfw,name_list):
    shutil.copyfile(os.path.join(os.path.split(path)[0],'1289.41',aux),os.path.join(os.path.split(path)[0],"Raster_World_Files",name+aux[11:]))
    shutil.copyfile(os.path.join(os.path.split(path)[0],'1289.41',tfw),os.path.join(os.path.split(path)[0],"Raster_World_Files",name+tfw[11:]))


for folder in os.listdir(home_dir):
    for file in os.walk(home_dir+'/'+folder):
        if 'Raster_World_Files' in file[1]:
            for world_file in os.listdir(os.path.join(file[0],file[1][-1])):
                #print (os.path.join(file[0],file[1][-1],world_file))
                if len(find(world_file,'_'))>1:
                    #print (os.path.join(file[0],file[1][-1],world_file))
                    os.remove(os.path.join(file[0],file[1][-1],world_file))
                    

for folder in os.walk(home_dir):
    if len(folder[1]) > 0 and folder[1][-1] == 'Raster_World_Files':
        world_files = os.listdir(os.path.join(folder[0],"Raster_World_Files"))
        aux = world_files[0]
        tfw = world_files[1]
        str_aux = str_insert(aux,'_allBands',find(aux,'.')[0])
        str_tfw = str_insert(tfw,'_allBands',find(tfw,'.')[0])
        
        src_aux = os.path.join(folder[0],"Raster_World_Files",aux)
        dst_aux = os.path.join(folder[0],"all",str_aux)
        
        src_tfw = os.path.join(folder[0],"Raster_World_Files",tfw)
        dst_tfw = os.path.join(folder[0],"all",str_tfw)
        
        shutil.copyfile(src_aux,dst_aux)
        shutil.copyfile(src_tfw,dst_tfw)


        
        
        
        
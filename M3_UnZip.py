# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
import matplotlib.cm as cm
import spectral as sp
from celluloid import Camera
import os
import zipfile
from tkinter import Tk
from tkinter.filedialog import askdirectory as askdir
from tkinter.filedialog import askopenfile as askfile
import tifffile as tf
from scipy import interpolate as interp
import shutil

def M3_unzip(select,**kwargs):
    ##Asks user what the source directory is for M3 Data
    Tk().withdraw()
    if select == True:    
        hdr_folder_path = askdir()
    elif select == False:
        hdr_folder_path = kwargs['folder']
    else:
        raise Exception("Select is either True or False")
    
    
    ##Unzips files and places them in "extracted_files" folder in source directory
    hdr_folder = os.listdir(hdr_folder_path)
    if 'extracted_files' not in hdr_folder:
        for file in hdr_folder:
            if file.find(".zip") > -1:
                myfile = zipfile.ZipFile(hdr_folder_path+"/"+file)
                myfile.extractall(path=hdr_folder_path+"/extracted_files/"+file[0:-4])
                
        try:
            os.mkdir(hdr_folder_path+'/extracted_files')
        except:
            pass
    
    try:
        os.mkdir(hdr_folder_path+'/extracted_files/shape_files')
    except:
        pass
    
    try:
        os.mkdir(hdr_folder_path+'/extracted_files/hdr_files')
    except:
        pass
    
        
    for i in os.walk(hdr_folder_path+'/extracted_files'):
        if 'dershp' in i[1]:
            for i in os.walk(i[0]+'/dershp'):
                if len(i[2]) != 0 and ''.join(i[2]).find('.zip')>-1:
                    for n in range(0,len(i[2])):
                        myfile = zipfile.ZipFile(i[0]+'/'+i[2][n])
                        myfile.extractall(path=hdr_folder_path+'/extracted_files/shape_files/'+i[2][n][0:-4])
                        
    ##Gets list of all .hdr files in the source directory
    ex_files = os.listdir(hdr_folder_path+'/'+'extracted_files')
    hdr_file_list = []
    hdr_folder_list = []
    for i in os.walk(hdr_folder_path):
        if len(i[2]) != 0 and ''.join(i[2]).find('.hdr') > -1 and i[0].find('hdr_files')==-1:
            for file in i[2]:
                if file.find('hdr') > -1:
                    #print (i[0],file)
                    hdr_file_list.append(file)
                    hdr_folder_list.append(i[0])
                    
    for file,folder in zip(hdr_file_list,hdr_folder_list):
        try:
            os.mkdir(hdr_folder_path+'/extracted_files/hdr_files/'+file[0:-4])
        except:
            pass
        
        try:
            shutil.copyfile(folder+'/'+file,hdr_folder_path+'/extracted_files/hdr_files/'+file[0:-4]+'/'+file)
            shutil.copyfile(folder+'/'+file[0:-4]+'.img',hdr_folder_path+'/extracted_files/hdr_files/'+file[0:-4]+'/'+file[0:-4]+'.img')
        except:
            print('failed')
            pass
        
    hdr_files_path = hdr_folder_path+'/extracted_files/hdr_files'
        
                    
    ##Counts the number of images used
    n = 0
    for i in hdr_file_list:
        if i.find('rfl') > -1:
            n+=1
    print (f"Number of Files Parsed & Sorted: {n}")
    
    return hdr_file_list,hdr_files_path

hdr_file_list,hdr_files_path = M3_unzip(False,folder=r"D:/Data/20230209T095534013597")

##Getting list of files to copy into PDS
for i in os.walk(hdr_folder_path+'/extracted_files/hdr_files'):
    for file in i[2]:
        if file.find('_rfl')>-1:
            print (file[0:-12]+'*')

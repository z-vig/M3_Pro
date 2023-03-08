"""
Python Script for parsing M3 files from PDS
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
import matplotlib.cm as cm
import spectral as sp
import os
import zipfile
from tkinter import Tk
from tkinter.filedialog import askdirectory as askdir
from tkinter.filedialog import askopenfile as askfile
import tifffile as tf
from scipy import interpolate as interp
import shutil

def M3_unzip(select,**kwargs):
    defaultKwargs = {'folder':None}
    kwargs = {**defaultKwargs,**kwargs}

    ##Asks user what the source directory is for M3 Data
    Tk().withdraw()
    if select == True:    
        hdrFolderPath = askdir()
    elif select == False and kwargs.get('folder') != None:
        hdrFolderPath = kwargs.get('folder')
    elif kwargs.get('folder') == None:
        raise ValueError(f"Folder cannot be {type(kwargs.get('folder'))}")
    else:
        raise Exception("Select is either True or False")
    
    
    ##Exits script if files are already unzipped 
    sourcedir_FileList = os.listdir(hdrFolderPath)  
    if 'extracted_files' in sourcedir_FileList:
        hdrFilesPath = os.path.join(hdrFolderPath,'extracted_files','hdr_files')
        hdrFileList=[]
        for root,dirs,files in os.walk(hdrFilesPath):
            for file in files:
                if file[len(file)-4:].find('.hdr')>-1 and root.find('sup') == -1:
                    hdrFileList.append(os.path.join(root,file))

        print ('.zip Files have already been extracted')

        return hdrFileList,hdrFilesPath
        

    ##Unzips files and places them in "extracted_files" folder in source directory
    elif 'extracted_files' not in sourcedir_FileList:
        os.mkdir(hdrFolderPath+'/extracted_files')
        for file in sourcedir_FileList:
            if file.find(".zip") > -1:
                myfile = zipfile.ZipFile(hdrFolderPath+"/"+file)
                myfile.extractall(path=hdrFolderPath+"/extracted_files/"+file[0:-4])

    extractedFilesPath = os.path.join(hdrFolderPath+'/extracted_files')
    
    ##Makes directories for .lbl, .HDR and shape files
    lblFilesPath = os.path.join(extractedFilesPath,'lbl_files') #lbl folders containing shape info
    os.mkdir(lblFilesPath)
    hdrFilesPath = os.path.join(extractedFilesPath,'hdr_files') #hdr and img files in the same folder
    os.mkdir(hdrFilesPath)
    shapeFilesPath = os.path.join(extractedFilesPath,'shape_files') #all shape file info (.dbf,.prj,.shp,.lock,.xml,.shx) into a single folder
    os.mkdir(shapeFilesPath)

    ##Filling .lbl directory
    for root,dirs,files in os.walk(extractedFilesPath):
        for file in files:
            if file.find('.zip')>-1:
                myfile = zipfile.ZipFile(os.path.join(root,file))
                myfile.extractall(path=os.path.join(lblFilesPath,file[0:-4]))
                        
    ##Filling .HDR directory
    for root,dirs,files in os.walk(extractedFilesPath):
        if root.find('hdr_files') == -1:
            for file in files:
                file_ext = file[len(file)-4:]
                if file_ext.find('.hdr')>-1 and file.find('sup') == -1: ##Copy _rfl.hdr files
                    copyto_hdr = os.path.join(hdrFilesPath,file[0:-4])
                    os.mkdir(copyto_hdr)
                    shutil.copyfile(os.path.join(root,file),os.path.join(copyto_hdr,file))
                elif file_ext.find('.img')>-1 and file.find('sup') == -1: ##Copy _rfl.img files
                    copyto_hdr = os.path.join(hdrFilesPath,file[0:-4])
                    shutil.copyfile(os.path.join(root,file),os.path.join(copyto_hdr,file))
                elif file_ext.find('.hdr')>-1 and file.find('sup') > -1: ##Copy _sup.hdr files
                    copyto_sup = os.path.join(hdrFilesPath,file[0:-4])
                    os.mkdir(copyto_sup)
                    shutil.copyfile(os.path.join(root,file),os.path.join(copyto_sup,file))
                elif file_ext.find('.img')>-1 and file.find('sup') > -1: ##Copy _sup.img files
                    copyto_sup = os.path.join(hdrFilesPath,file[0:-4])
                    shutil.copyfile(os.path.join(root,file),os.path.join(copyto_sup,file))

    ##Filling shape file directory
    for root,dirs,files in os.walk(lblFilesPath):
        for file in files:
            shutil.copyfile(os.path.join(root,file),os.path.join(shapeFilesPath,file))
        
    ##Getting HDR file list
    hdrFileList=[]
    for root,dirs,files in os.walk(hdrFilesPath):
        for file in files:
            if file[len(file)-4:].find('.hdr')>-1 and root.find('sup') == -1:
                hdrFileList.append(os.path.join(root,file))
                    
    ##Counts the number of images used
    print (f"Number of Files Parsed & Sorted: {len(hdrFileList)}")
    
    return hdrFileList,hdrFilesPath

##hdrFileList,hdrFilesPath = M3_unzip(select=False,folder=r'/run/media/zvig/My Passport/Data/20230209T095534013597 - Copy')

if __name__ == "__main__":
    hdrFileList,hdrFilesPath = M3_unzip(select=True)
    print (f'HDR files exist in: {hdrFilesPath} \n\
             HDR files are:')
    for file in hdrFileList:
        print (file)

##Getting list of files to copy into PDS
# =============================================================================
# for i in os.walk(hdrFolderPath+'/extracted_files/hdr_files'):
#     for file in i[2]:
#         if file.find('_rfl')>-1:
#             print (file[0:-12]+'*')
# 
# =============================================================================

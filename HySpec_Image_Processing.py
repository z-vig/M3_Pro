# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:34:52 2023

@author: zacha
"""
##Timing the program
import time
start = time.time()

##Importing Necessary Libraries
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
#from Linear_Spline import linear_spline

def get_hdr_files(select):
    ##Asks user what the source directory is for M3 Data
    Tk().withdraw()
    if select == True:    
        hdr_folder_path = askdir()
    elif select == False:
        hdr_folder_path = r"D:\Data\20230131T17554836251"
    else:
        raise Exception("Select is either True or False")
    
    
    ##Unzips files and places them in "extracted_files" folder in source directory
    hdr_folder = os.listdir(hdr_folder_path)
    if 'extracted_files' not in hdr_folder:
        for file in hdr_folder:
            if file.find(".zip") > -1:
                myfile = zipfile.ZipFile(hdr_folder_path+"/"+file)
                myfile.extractall(path=hdr_folder_path+"/extracted_files/"+file[0:-4])
    
    
    ##Gets list of all .hdr files in the source directory
    ex_files = os.listdir(hdr_folder_path+'/'+'extracted_files')
    hdr_file_list = []
    hdr_folder_list = []
    for i in os.walk(hdr_folder_path):
        if len(i[2]) != 0 and ''.join(i[2]).find('.hdr') > -1:
            for file in i[2]:
                if file.find('hdr') > -1:
                    #print (i[0],file)
                    hdr_file_list.append(file)
                    hdr_folder_list.append(i[0])
    
    
    ##Counts the number of images used
    n = 0
    for i in hdr_file_list:
        if i.find('rfl') > -1:
            n+=1
    print (f"Number of Files: {n}")
    
    return hdr_file_list,hdr_folder_list


##Unused plotting functions
# =============================================================================
# step = 100
# img_array = np.zeros((rows,cols,bands))
# for x in range(0,rows,step):
#     rows = hdr.read_subregion((x,x+step),(0,cols))
# =============================================================================
    
# =============================================================================
# fig = plt.figure()
# camera = Camera(fig)
#     
# for band,lamb in zip(range(bands),wvl):
#     bands_plot = hdr.read_band(band)
#     plt.imshow(bands_plot,cmap = 'gray',interpolation='nearest')
#     plt.text(100,1000,f'{float(lamb)/1000:.3f} \u03BCm',color="white")
#     camera.snap()
# 
# animation = camera.animate(interval = 100, repeat = True, repeat_delay = 500)
# animation.save(r"C:\Users\zacha\OneDrive\Desktop\m3.gif",writer='Pillow')
# =============================================================================


##Function to multiple locations of a character in a string
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


##HDR Image class, input hdr image file path
class HDR_Image():
    ##Constructor method
    def __init__ (self,hdr):
        self.hdr = sp.envi.open(hdr)
    
    ##String print method
    def __str__(self):
        return f"HDR Image Class: {self.hdr.fid.name[find(self.hdr.fid.name,'/')[-1]+1:len(self.hdr.fid.name)-4]}"
    
    ##Returns HDR Object for testing purposes
    def hdr(self):
        return self.hdr
    
    ##Method that returns date, time, and data types of HDR Image
    def datetime(self):
        data_str = self.hdr.fid.name[find(self.hdr.fid.name,'/')[-1]+1:len(self.hdr.fid.name)-4]
        if data_str[2] == 'g':
            obs_type = 'Global'
        elif data_str[2] == 't':
            obs_type = 'Target'
        else:
            raise 'String Error'
            
        if data_str[len(data_str)-3:] == 'sup':
            data_type = "Supplemental"
        elif data_str[len(data_str)-3:] == 'rfl':
            data_type = "Reflectance"
        else:
            data_type = "String Error"
        
        date = f"{data_str[3:7]}/{data_str[7:9]}/{data_str[9:11]}"
        time = f"{data_str[12:14]}:{data_str[14:16]}:{data_str[16:18]}"
        
        self.date = data_str[3:11]
        self.time = data_str[12:18]
        
        print (f"Observation Type: {obs_type}\nData Type: {data_type}\nDate (Y/M/D): {date}\nTime: {time}\n")
        
        self.meta_data_str = f"Observation Type: {obs_type}\nData Type: {data_type}\nDate (Y/M/D): {date}\nTime: {time}"
        self.meta_data_dict = {"ObservationType": obs_type,"DataType": data_type,"Date(Y/M/D)":date,"Time":time}
# =============================================================================
#         print (f"Data Type: {data_type}")
#         print (f"Date (Y/M/D): {date}")
#         print (f"Time: {time}")
# =============================================================================
        
        return [obs_type,data_type,date,time]
    
    
    ##Method that returns list of wavelengths of HDR Image
    def wavelengths(self):
        print (f"Wavelengths: {self.hdr.bands.centers}")
        #return self.hdr.bands.centers
    
    
    ##Plots image of a given wavelength or all three supplemental image, if using a .sup file as input
    def plt_img(self,wavelen):
        if self.hdr.bands.centers == None:
            fig_big = plt.figure(figsize=(20,20))
            
            plot_topo = self.hdr.read_band(0)
            plot_temp = self.hdr.read_band(1)
            plot_longwav = self.hdr.read_band(2)
            plot_all = self.hdr.read_bands([0,1,2])
            
            ax1 = fig_big.add_subplot(1,3,1)
            ax2 = fig_big.add_subplot(1,3,2)
            ax3 = fig_big.add_subplot(1,3,3)
            ax1.imshow(plot_topo,cmap="gray")
            ax2.imshow(plot_temp,cmap="jet")
            ax3.imshow(plot_longwav,cmap="gray")

            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3.set_xticks([])
            ax3.set_yticks([])
            
            try:
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time)
            except:
                pass
            
            #plt.savefig(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+"supplemental.tif")
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+"topo.tif",
                       plot_topo,
                       photometric='minisblack',
                       imagej = True,
                       metadata = self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+"temp.tif",
                       plot_temp,
                       photometric='minisblack',
                       imagej=True,
                       metadata = self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+"longwave.tif",
                       plot_longwav,
                       photometric='minisblack',
                       imagej=True,
                       metadata = self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+"supp.tif",
                       plot_all,
                       photometric='rgb',
                       metadata = self.meta_data_dict)
            
        
        elif self.hdr.bands.centers != None:
            fig_big = plt.figure(figsize=(20,20))
            
            wvl = self.hdr.bands.centers
            plot_band = self.hdr.read_band(wvl.index(wavelen))
            
            print (type(wvl))
            wavelength = wvl[wvl.index(wavelen)]
            
            ax = fig_big.add_subplot()
            ax.imshow(plot_band,cmap="gray")
            ax.text(305,110,f"Wavelength: {wavelength} nm",fontsize="xx-large")
            ax.text(305,100,self.meta_data_str,color="black",fontsize="xx-large")
            ax.set_xticks([])
            ax.set_yticks([])
            
            try:
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time)
            except:
                pass
                
            #plt.savefig(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+str(wavelength)+".tif")
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+str(wavelength)+"_raw.tif",
                       plot_band,
                       photometric='minisblack',
                       imagej = True,
                       metadata = self.meta_data_dict)
    
    
    ## Plots spectrum of a given x,y point of HDR Image
    def plot_spec(self,x,y,**kwargs):
        if self.hdr.bands.centers != None:
            fig,ax = plt.subplots(1,1)
            
            wvl = self.hdr.bands.centers
            spec_list = []
            wvl_list = []
            for data_val,wavelength in zip(self.hdr.read_pixel(x,y),wvl):
                if abs(data_val)<500 and wavelength<2500 and wavelength>1000:
                    spec_list.append(data_val)
                    wvl_list.append(wavelength)
                    
            if kwargs["plot_og"] == True:
                ax.plot(wvl_list,spec_list,alpha=0.6,ls='--',label='Original Spectrum')
            else:
                pass
            
            f = interp.CubicSpline(np.array(wvl_list),np.array(spec_list))
            x = np.linspace(min(wvl_list),max(wvl_list),30)
            
            if kwargs["plot_cspline"] == True:
                ax.plot(x,f(x),'-',color="orange",label='Cubic Spline')
            else:
                pass

            
            try:
                if kwargs["filter_img"] == True:
                    if np.all(np.array(spec_list)>0.1) == True:
                        print (np.array(spec_list))
                    plt.close()
                elif kwargs["filter_img"] == False:
                    plt.show()
            except:
                pass
                
            box_size = int(input("Box Size: "))
            avg_array = np.zeros((box_size,2))
            avg_rfl = []
            std_rfl = []
            avg_wvl = []
            n = 1
            for spec,wvl in zip(spec_list,wvl_list):
                if n%box_size != 0:
                    avg_array[n-1] = (spec,wvl)
                    n+=1
                elif n%box_size == 0:
                    avg_array[n-1] = (spec,wvl)
                    avg_rfl.append(np.average(avg_array[:,0]))
                    std_rfl.append(np.std(avg_array[:,0]))
                    avg_wvl.append(np.average(avg_array[:,1]))
                    avg_array = np.zeros((box_size,2))
                    n=1
            
            rfl_last = []
            wvl_last = []
            for rfl,wvl in zip(avg_array[:,0],avg_array[:,1]):
                if rfl > 0:
                    rfl_last.append(rfl)
                    wvl_last.append(wvl)
                    
            
            avg_rfl.append(np.average(rfl_last))
            std_rfl.append(np.std(rfl_last))
            avg_wvl.append(np.average(wvl_last))

            if kwargs["plot_boxcar"] == True:
                ax.plot(avg_wvl,avg_rfl,c='red',label=f'Boxcar({box_size} pts.)')
                ax.fill_between(avg_wvl,np.array(avg_rfl)+np.array(std_rfl),
                                np.array(avg_rfl)-np.array(std_rfl),alpha=0.1,color='red')
            else: 
                pass
            
            f1 = interp.CubicSpline(np.array(avg_wvl),np.array(avg_rfl))
            f1_err = interp.CubicSpline(np.array(avg_wvl),np.array(std_rfl))
            x1 = np.linspace(min(wvl_list),max(wvl_list),30)

            if kwargs["plot_cspline_boxcar"] == True:
                ax.plot(x1,f1(x1),c='black',ls='dashdot',label=f'Boxcar({box_size} pts.)  & Cubic Spline')
                ax.fill_between(x1,f1(x1)+f1_err(x1),f1(x1)-f1_err(x1),alpha=0.1,color='black')
            else:
                pass
                    
            #ax.plot(wvl,avg_rfl,c="red")    
            ax.set_xlabel('Wavelength (\u03BCm)')
            ax.set_ylabel('Reflectance')
            
            ax.legend()
            
            #return spec_list,wvl_list
            
        elif self.hdr.bands.centers == None:
            spec_list,wvl_list = [],[]
            return spec_list,wvl_list
        
    def good_spectra(self):
        bands = 1
        for i in range(0,self.hdr.nbands-1):
            band = self.hdr.read_band(i)
            bands+=1
        print (f'There are {bands} bands and {band.shape[0]*band.shape[1]} pixels')
    def H2O_p1(self):
        wv1 = self.hdr.read_band(10)
        wv2 = self.hdr.read_band(12)
        wv3 = self.hdr.read_band(14)
        
        wv_plot = (wv1-wv2)/wv3
        
        plt.imshow(wv_plot,cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
hdr_file_list,hdr_folder_list = get_hdr_files(False)

obj_list = []
for path,file in zip(hdr_folder_list,hdr_file_list):
    if file.find('rfl') > -1:
        obj_list.append(HDR_Image(path+'/'+file))

# =============================================================================
# for obj in obj_list:
#     obj.datetime()
# =============================================================================

# =============================================================================
# for obj in obj_list:
#     spec1,wv1 = obj.plot_spec(90,103)
# =============================================================================
#obj_list[0].wavelengths()

# =============================================================================
# pixels = 1
# for x in range(1015,obj_list[0].hdr.nrows):
#     for y in range(302,obj_list[0].hdr.ncols):
#         spec,wvl = obj_list[0].plot_spec(x,y,filter_img=True)
#         print (f'{pixels} out of {obj_list[0].hdr.nrows*obj_list[0].hdr.ncols} pixels processed')
#         pixels += 1
# =============================================================================
        
#obj_list[0].plot_spec(100,100,plot_og=True,plot_boxcar=False,plot_cspline=False,plot_cspline_boxcar=True)

# =============================================================================
# for obj in obj_list:
#     obj.good_spectra()
# =============================================================================

# =============================================================================
# for obj in obj_list:
#     obj.plt_img(1289.41)
# =============================================================================



end = time.time()
runtime = end-start
if runtime < 1:
    print (f'Program Executed in {runtime*10**3:.3f} milliseconds')
elif runtime < 60 and runtime > 1:
    print (f'Program Executed in {runtime:.3f} seconds')
elif runtime > 60:
    print (f'Program Executed in {runtime/60:.1f} minutes and {runtime%60:.3f} seconds')
    

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
from spec_average import spec_avg
#from Linear_Spline import linear_spline
    
    



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
    def plt_img(self,wavelen,**kwargs):
        if self.hdr.bands.centers == None and kwargs["All_Bands"] == False:
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
            
            plt.close()
            
            try:
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time)
            except:
                pass
            
            #plt.savefig(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+"supplemental.tif")
            try:
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/topo")
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/temp")
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/longwave")
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/supp")
            except:
                pass
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/topo/"+"topo.tif",
                       plot_topo,
                       photometric='minisblack',
                       imagej = True,
                       metadata = self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/temp/"+"temp.tif",
                       plot_temp,
                       photometric='minisblack',
                       imagej=True,
                       metadata = self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/longwave/"+"longwave.tif",
                       plot_longwav,
                       photometric='minisblack',
                       imagej=True,
                       metadata = self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/all_img/"+"supp.tif",
                       plot_all,
                       photometric='rgb',
                       metadata = self.meta_data_dict)
            
        
        elif self.hdr.bands.centers != None and kwargs["All_Bands"] == False:
            fig_big = plt.figure(figsize=(20,20))
            
            wvl = self.hdr.bands.centers
            
            plot_band = self.hdr.read_band(wvl.index(wavelen))
            
            ##Normalizing
            norm_band = plot_band-np.min(plot_band)
            norm_band = norm_band/np.max(norm_band)
            
            print (type(wvl))
            wavelength = wvl[wvl.index(wavelen)]
            
            ax = fig_big.add_subplot()
            ax.imshow(plot_band,cmap="gray")
            ax.text(305,110,f"Wavelength: {wavelength} nm",fontsize="xx-large")
            ax.text(305,100,self.meta_data_str,color="black",fontsize="xx-large")
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.close()
            
            try:
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time)
            except:
                pass
                
            #plt.savefig(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+str(wavelength)+".tif")
            try:
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+str(wavelength))
            except:
                pass
            
            #tf.imwrite(r"C:\Users\zacha/OneDrive/Desktop/test1.tif",norm_band,photometric='minisblack')
            #print (norm_band.dtype)
            
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+str(wavelength)+'/'+str(wavelength)+"_raw.tif",
                       norm_band,
                       photometric='minisblack',
                       metadata = self.meta_data_dict)
        
        elif kwargs["All_Bands"] == True:
            wvl = self.hdr.bands.centers
            band_index = []
            for index,w in enumerate(wvl):
                if w>1000 and w<2500:
                    band_index.append(index)
            rfl = self.hdr.read_bands(band_index)
            
            norm_array = np.zeros(rfl.shape).astype("float32")
            for i in range(0,rfl.shape[2]):
                band = rfl[:,:,i]
                band_norm = band-np.min(band)
                band_norm = band_norm.astype("float32")
                band_norm = 255*band_norm/np.max(band_norm).astype("float32")
                print (np.max(band_norm),np.min(band_norm))
                norm_array[:,:,i] = band_norm
                
            print (norm_array.dtype)
            try:
                os.mkdir(r"D:/Data/Lunar_Ice_Images/"+self.date+"_"+self.time+'/all')
            except:
                pass
            
            tf.imwrite(r"D:/Data/Lunar_Ice_Images/"+self.date+"_"+self.time+'/all/'+self.date+"_"+self.time+"_allBands.tif",
                       norm_array,
                       photometric='rgb',
                       metadata=self.meta_data_dict)
            
            print ("TIFF File Written")
    
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
            
            f1 = interp.CubicSpline(np.array(wvl_list),np.array(spec_list))
            x1 = np.linspace(min(wvl_list),max(wvl_list),80)
            
            if kwargs["plot_cspline"] == True:
                ax.plot(x1,f1(x1),'-',color="orange",label='Cubic Spline')
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
            
            try:
                if kwargs["box_size"] == "select":
                    box_size = int(input("Box Size: "))
                else:
                    box_size = kwargs["box_size"]
            except:
                pass
            
            avg_rfl,std_rfl,avg_wvl = spec_avg(spec_list,wvl_list,5)

            if kwargs["plot_boxcar"] == True:
                ax.plot(avg_wvl,avg_rfl,c='red',label=f'Boxcar({box_size} pts.)')
                ax.fill_between(avg_wvl,np.array(avg_rfl)+np.array(std_rfl),
                                np.array(avg_rfl)-np.array(std_rfl),alpha=0.1,color='red')
            else: 
                pass
            
            f2 = interp.CubicSpline(np.array(avg_wvl),np.array(avg_rfl))
            f2_err = interp.CubicSpline(np.array(avg_wvl),np.array(std_rfl))
            x2 = np.linspace(min(wvl_list),max(wvl_list),80)

            if kwargs["plot_cspline_boxcar"] == True:
                ax.plot(x2,f2(x2),c='black',ls='dashdot',label=f'Boxcar({box_size} pts.)  & Cubic Spline')
                ax.fill_between(x2,f2(x2)+f2_err(x2),f2(x2)-f2_err(x2),alpha=0.1,color='black')
            else:
                pass
                    
            #ax.plot(wvl,avg_rfl,c="red")    
            ax.set_xlabel('Wavelength (\u03BCm)')
            ax.set_ylabel('Reflectance')
            
            ax.legend()
            
            name_str = "_"
            for key,value in zip(kwargs.keys(),kwargs.values()):
                if value == True:
                    name_str = name_str+key[key.find('_')+1:]+'_'
            
            plt.savefig(r"D:Data/Figures/"+self.hdr.filename[59:67]+"_"+str(x)+"_"+str(y)+name_str+".png")
            
            #return spec_list,wvl_list
            
        elif self.hdr.bands.centers == None:
            spec_list,wvl_list = [],[]
            return spec_list,wvl_list
        
    def good_spectra(self):
        pixel_list=[]
        test_pixel = self.hdr.read_pixel(np.random.randint(0,self.hdr.nrows),np.random.randint(0,self.hdr.ncols))
        nbands = len(test_pixel[test_pixel>-900])
        good_pixel_array = np.zeros((self.hdr.nrows,self.hdr.ncols,nbands))
        for x in range(0,self.hdr.nrows):
            for y in range(0,self.hdr.ncols):
                pixel = self.hdr.read_pixel(x,y)
                pixel = pixel[pixel>-999]
                if np.average(pixel)>0.05:
                    good_pixel_array[x,y] = pixel
                    pixel_list.append(np.average(pixel))
        
        bin_array = good_pixel_array
        bin_array[bin_array>0]=1
        bin_array = bin_array.astype('float32')

        #plt.imshow(bin_array[:,:,10],cmap='Spectral')
        
        try:
            os.mkdir(r"D:Data\Lunar_Ice_Images\Shaded_Regions")
        except:
            pass
        
        tf.imwrite(r"D:Data\Lunar_Ice_Images\Shaded_Regions/"+self.date+"_"+self.time+"_raw.tif",
                   bin_array[:,:,10],
                   photometric='minisblack',
                   imagej=True,
                   metadata = self.meta_data_dict)
        
        good_pix = np.count_nonzero(bin_array[:,:,10])
        total_pix = self.hdr.nrows*self.hdr.ncols
        pct = good_pix/total_pix
            
        print (bin_array.shape)
        print (f'There are {good_pix} good pixels out of {total_pix} total pixels ({pct:.2%})')
        
    def H2O_p1(self,x,y):
        rfl = []
        wvl = []
        for r,w in zip(self.hdr.read_pixel(x,y),self.hdr.bands.centers):
            if r>-900 and w<2500 and w>1000:
                rfl.append(r)
                wvl.append(w)
        
        rfl,wvl = np.array(rfl),np.array(wvl)
        avg_rfl,std_rfl,avg_wvl = spec_avg(rfl,wvl,5)
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        f = interp.CubicSpline(avg_wvl,avg_rfl)
        f_err = interp.CubicSpline(avg_wvl,std_rfl)
        x = np.linspace(min(wvl),max(wvl),80)
        
# =============================================================================
#         ax.plot(wvl,rfl,ls='--')
#         ax.plot(avg_wvl,avg_rfl)
# =============================================================================
        ax.plot(x,f(x),ls='dashdot',c='k')
        
        return x,f(x)
        
        
        

if 'hdr_file_list' in locals():
    print ('Necessary Variables are Defined')
else:
    from M3_UnZip import *
    hdr_file_list,hdr_files_path = M3_unzip(False,folder=r"D:/Data/20230209T095534013597")

obj_list = []
for file in hdr_file_list:
    if file.find('rfl') > -1:
        obj_list.append(HDR_Image(hdr_files_path+'/'+file[0:-4]+'/'+file))

for obj in obj_list:
    obj.datetime()

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
        
#obj_list[0].plot_spec(91,100,plot_og=True,plot_boxcar=True,plot_cspline=False,plot_cspline_boxcar=False,box_size=5)

#x,y = obj_list[0].H2O_p1(91,100)

# =============================================================================
# for obj in obj_list:
#     obj.good_spectra()
# =============================================================================
#obj_list[2].plt_img(1209.57)

# =============================================================================
# for obj in obj_list:
#     obj.plt_img(1289.41)
# =============================================================================
arr = obj_list[0].plt_img(1289.41,All_Bands=True)

end = time.time()
runtime = end-start
if runtime < 1:
    print (f'Program Executed in {runtime*10**3:.3f} milliseconds')
elif runtime < 60 and runtime > 1:
    print (f'Program Executed in {runtime:.3f} seconds')
elif runtime > 60:
    print (f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')
    

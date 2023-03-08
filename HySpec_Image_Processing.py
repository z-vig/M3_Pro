# -*- coding: utf-8 -*-
"""
Welcome to HySpecPy! 
"""
# Importing Necessary Modules
from spec_average import spec_avg
from scipy import interpolate as interp
import tifffile as tf
from tkinter.filedialog import askopenfile as askfile
from tkinter.filedialog import askdirectory as askdir
from tkinter import Tk
import zipfile
import os
import spectral as sp
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
import random
import pandas as pd
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import time


from fancy_spec_plot import fancy_spec_plot
from moving_avg import moving_avg

# Timing the program
start = time.time()


# Function to multiple locations of a character in a string
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


# HDR Image class, input hdr image file path
class HDR_Image():
    # Constructor method
    def __init__(self, path):
        self.hdr = sp.envi.open(path)
        data_str = self.hdr.fid.name[find(self.hdr.fid.name, '\\')[-1]+1:len(self.hdr.fid.name)-4]
        date = f"{data_str[3:7]}/{data_str[7:9]}/{data_str[9:11]}"
        date = date.replace('/','-')
        time = f"{data_str[12:14]}:{data_str[14:16]}:{data_str[16:18]}"
        time = time.replace(':','-')
        date_time = date+'_'+time
        
        
        if data_str[2] == 'g':
            obs_type = 'Global'
        elif data_str[2] == 't':
            obs_type = 'Target'
            
        if data_str[len(data_str)-3:] == 'sup':
            data_type = "Supplemental"
        elif data_str[len(data_str)-3:] == 'rfl':
            data_type = "Reflectance"
            
        self._dateTime = date_time
        self._obsType = obs_type
        self._dataType = data_type
        self.meta_data_str = f"Observation Type: {obs_type}\nData Type: {data_type}\nDate (Y/M/D): {date}\nTime: {time}"
        self.meta_data_dict = {"ObservationType": obs_type,
                               "DataType": data_type, "Date(Y/M/D)": date, "Time": time}

    # String print method
    def __str__(self):
        return f"HDR Image Class: {self.hdr.fid.name[find(self.hdr.fid.name,'/')[-1]+1:len(self.hdr.fid.name)-4]}"

    # Returns HDR Object for testing purposes
    def hdr(self):
        return self.hdr

    # Method that returns date, time, and data types of HDR Image
    @property
    def datetime(self):
        print (f'Getting Date/Time... \n\
               Data Type: {self._dataType} \n\
                Observation Type: {self._obsType} \n\
                Date/Time: {self._dateTime}')
        return self._dateTime

    # Plots image of a given wavelength or all three supplemental image, if using a .sup file as input
    def plot_image(self, wavelen, **kwargs):
        defaultKwargs = {"All_Bands": True,
                         "Norm": "All", 'allMax': np.ones(52), 'allMin': np.zeros(52),
                         "saveImage":False}
        kwargs = {**defaultKwargs, **kwargs}
        # For Supplemental HDR Files
        if self.hdr.bands.centers == None and kwargs["All_Bands"] == False:
            fig_big = plt.figure(figsize=(20, 20))

            plot_topo = self.hdr.read_band(0)
            plot_temp = self.hdr.read_band(1)
            plot_longwav = self.hdr.read_band(2)
            plot_all = self.hdr.read_bands([0, 1, 2])

            ax1 = fig_big.add_subplot(1, 3, 1)
            ax2 = fig_big.add_subplot(1, 3, 2)
            ax3 = fig_big.add_subplot(1, 3, 3)
            ax1.imshow(plot_topo, cmap="gray")
            ax2.imshow(plot_temp, cmap="jet")
            ax3.imshow(plot_longwav, cmap="gray")

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

            # plt.savefig(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+"supplemental.tif")
            try:
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date +
                         "_"+self.time+"/supplemental/topo")
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date +
                         "_"+self.time+"/supplemental/temp")
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date +
                         "_"+self.time+"/supplemental/longwave")
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date +
                         "_"+self.time+"/supplemental/supp")
            except:
                pass
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self._dateTime+"/supplemental/topo/"+"topo.tif",
                       plot_topo,
                       photometric='minisblack',
                       imagej=True,
                       metadata=self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self._dateTime+"/supplemental/temp/"+"temp.tif",
                       plot_temp,
                       photometric='minisblack',
                       imagej=True,
                       metadata=self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self._dateTime+"/supplemental/longwave/"+"longwave.tif",
                       plot_longwav,
                       photometric='minisblack',
                       imagej=True,
                       metadata=self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self._dateTime+"/supplemental/all_img/"+"supp.tif",
                       plot_all,
                       photometric='rgb',
                       metadata=self.meta_data_dict)

        # For regular hdr files
        elif self.hdr.bands.centers != None and kwargs["All_Bands"] == False:
            fig_big = plt.figure(figsize=(20, 20))

            wvl = self.hdr.bands.centers

            plot_band = self.hdr.read_band(wvl.index(wavelen))

            # Normalizing
            norm_band = plot_band-np.min(plot_band)
            norm_band = norm_band/np.max(norm_band)

            print(type(wvl))
            wavelength = wvl[wvl.index(wavelen)]

            ax = fig_big.add_subplot()
            ax.imshow(plot_band, cmap="gray")
            ax.text(
                305, 110, f"Wavelength: {wavelength} nm", fontsize="xx-large")
            ax.text(305, 100, self.meta_data_str,
                    color="black", fontsize="xx-large")
            ax.set_xticks([])
            ax.set_yticks([])

            plt.close()

            try:
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self._dateTime)
            except:
                pass

            # plt.savefig(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+str(wavelength)+".tif")
            try:
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self._dateTime+"/"+str(wavelength))
            except:
                pass

            # tf.imwrite(r"C:\Users\zacha/OneDrive/Desktop/test1.tif",norm_band,photometric='minisblack')
            #print (norm_band.dtype)

            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self._dateTime+"/"+str(wavelength)+'/'+str(wavelength)+"_raw.tif",
                       norm_band,
                       photometric='minisblack',
                       metadata=self.meta_data_dict)

        # To save a tiff with all bands saved
        elif kwargs["All_Bands"] == True:
            wvl = self.hdr.bands.centers
            band_index = []
            for index, w in enumerate(wvl):
                if w > 1000 and w < 2500:
                    band_index.append(index)
            rfl = self.hdr.read_bands(band_index)

            # Normalizing
            
            try:
                os.mkdir(r"D:/Data/Lunar_Ice_Images/" +self._dateTime+'/all')
            except:
                pass
  
            if kwargs.get('Norm') == 'Image':
                norm_array = np.zeros(rfl.shape)
                for i in range(rfl.shape[2]):
                    band = rfl[:, :, i]
                    band_norm = band-np.min(band)
                    band_norm = 255*band_norm/np.max(band_norm)
                    norm_array[:, :, i] = band_norm
                    
                        
                if kwargs.get('saveImage') == True:
                    tf.imwrite(r"D:/Data/Lunar_Ice_Images/"+self._dateTime+'/all/'+self.date+"_"+self.time+"_allBands.tif",
                               norm_array,
                               photometric='rgb',
                               metadata=self.meta_data_dict)
        
                    print("TIFF File Written")
                
                print ('Normalized w.r.t each Image')
                return norm_array
                        
            elif kwargs.get('Norm') == 'All':
                norm_array = np.zeros(rfl.shape)
                for i in range(0, rfl.shape[2]):
                    band = rfl[:, :, i]
                    band_norm = 255*(band-kwargs.get('allMin')[i])/kwargs.get('allMax')[i]
                    band_norm = band_norm.astype('int32')
                    #print (np.max(band_norm),np.min(band_norm))
                    norm_array[:, :, i] = band_norm

                norm_array = norm_array.astype("int")
                
                if kwargs.get('saveImage') == True:
                    tf.imwrite(r"D:/Data/Img_Supp_Files/"+self._dateTime+"_allBands.tif",
                               norm_array,
                               photometric='rgb',
                               metadata=self.meta_data_dict)
                    print("TIFF File Written")
                
                print ('Normalized w.r.t all Images')
                return norm_array
            
            elif kwargs.get('Norm') == 'None':
                return rfl


    # Plots spectrum of a given x,y point of HDR Image
    def plot_spec(self, x, y, **kwargs):
        defaultKwargs = {'plot_og': True, 'plot_boxcar': False, 'plot_cspline': False, 'plot_cspline_boxcar': False,
                         'box_size': "select", 'plot_movingAvg':False, 'movingAvg_size':'select','saveFig': False, "showPlot": True,
                         'plot_movingAvg_cspline':False}
        kwargs = {**defaultKwargs, **kwargs}

        if self.hdr.bands.centers != None:
            fig, ax = plt.subplots(1, 1)

            wvl = self.hdr.bands.centers
            rfl_vals = []
            wvl_vals = []
            for data_val, wavelength in zip(self.hdr.read_pixel(x, y), wvl):
                if abs(data_val) < 500 and wavelength < 2500 and wavelength > 1000:
                    rfl_vals.append(data_val)
                    wvl_vals.append(wavelength)

            ####Plotting Original####
            if kwargs["plot_og"] == True:
                fancy_spec_plot(fig, ax, np.array(wvl_vals), np.array(rfl_vals), title='Original Spectrum',
                                ylabel='Reflectance', xlabel='Wavelength (\u03BCm)',
                                line_color='red', line_style='--', label='Original Spectrum', ylim=(0.05, 0.2))
            elif kwargs["plot_og"] == False:
                pass
            else:
                raise Exception("Invalid kwarg for plot_og")
                
                
            ####Plotting Cubic Spline without Boxcar Average####
            if kwargs["plot_cspline"] == True:
                f1 = interp.CubicSpline(np.array(wvl_vals), np.array(rfl_vals))
                x1 = np.linspace(min(wvl_vals), max(wvl_vals), 80)
                fancy_spec_plot(fig, ax, x1, f1(x1), line_style='solid', line_color='orange',
                                title='Original Spectrum', label='Cubic Spline')
            elif kwargs["plot_cspline"] == False:
                pass
            else:
                raise Exception('Invalid kwarg for plot_cspline')

            ####Plotting Boxcar Average####
            if kwargs["plot_boxcar"] == True:
                if kwargs["box_size"] == "select":
                    box_size = int(input("Box Size: "))
                elif type(kwargs['box_size']) == int:
                    box_size = kwargs["box_size"]
                else:
                    raise Exception("Invalid kwarg for box_size")
                avg_rfl, std_rfl, avg_wvl = spec_avg(rfl_vals, wvl_vals, box_size)
                fancy_spec_plot(fig, ax, np.array(avg_wvl), np.array(avg_rfl), std=np.array(std_rfl),
                                line_color='red', std_color='red', title='Original Spectrum', label='Boxcar Average')
            elif kwargs["plot_boxcar"] == False:
                pass
            else:
                raise Exception("Invalid kwarg for plot_boxcar")
                
                
            ####Plotting Cubic Spline Fit onto Boxcar Average####
            if kwargs["plot_cspline_boxcar"] == True:
                if kwargs["box_size"] == "select":
                    box_size = int(input("Box Size: "))
                elif type(kwargs['box_size']) == int:
                    box_size = kwargs["box_size"]
                else:
                    raise Exception("Invalid kwarg for box_size")
                avg_rfl, std_rfl, avg_wvl = spec_avg(rfl_vals, wvl_vals, box_size)
                f2 = interp.CubicSpline(np.array(avg_wvl), np.array(avg_rfl))
                f2_err = interp.CubicSpline(np.array(avg_wvl), np.array(std_rfl))
                x2 = np.linspace(min(wvl_vals), max(wvl_vals), 80)
                fancy_spec_plot(fig, ax, x2, f2(x2), std=f2_err(x2), title='Original Spectrum',
                                line_style='solid', line_color='k', std_color='gray',
                                label='Cubic Spline', ylim=(0.05, 0.2))
            elif kwargs["plot_cspline_boxcar"] == False:
                pass
            else:
                raise Exception("Invalid kwarg for plot_cspline_boxcar")
                
                
                
            ####Plotting Moving Average####
            if kwargs.get('plot_movingAvg') == True:
                if kwargs.get('movingAvg_size') == 'select':
                    length = int(input('movingAvg_size: '))
                elif type(kwargs.get('movingAvg_size')) == int:
                    length = kwargs.get('movingAvg_size')
                conv_x,conv_y = moving_avg(np.array(wvl_vals),np.array(rfl_vals),length)
                fancy_spec_plot(fig,ax,conv_x,conv_y,title='Original Spectrum',line_color='purple',label='Moving Average')
                
                
            ####Plotting Cubic Spline on Moving Average####
            if kwargs.get('plot_movingAvg_cspline') == True:
                if kwargs.get('movingAvg_size') == 'select':
                    length = int(input('movingAvg_size: '))
                elif type(kwargs.get('movingAvg_size')) == int:
                    length = kwargs.get('movingAvg_size')
                conv_x,conv_y = moving_avg(np.array(wvl_vals),np.array(rfl_vals),length)
                f3 = interp.CubicSpline(conv_x,conv_y)
                x3 = np.linspace(min(conv_x),max(conv_x),80)
                fancy_spec_plot(fig,ax,x3,f3(x3),title='Original Spectrum',line_color='orange',
                                label = 'Cubic Spline (Moving Avg)')
            
            ax.legend()

            name_str = "_"
            for key, value in zip(kwargs.keys(), kwargs.values()):
                if value == True:
                    name_str = name_str+key[key.find('_')+1:]+'_'

            if kwargs.get('saveFig') == True:
                plt.savefig(
                    r"D:Data/Figures/"+self.hdr.filename[59:67]+"_"+str(x)+"_"+str(y)+name_str+".png")
            elif kwargs.get('saveFig') == False:
                pass
            else:
                raise Exception('Keyword Error in savefig')

            if kwargs.get('showPlot') == True:
                pass
            elif kwargs.get('showPlot') == False:
                plt.close()

            return np.array(rfl_vals), np.array(wvl_vals)

        elif self.hdr.bands.centers == None:
            rfl_vals, wvl_vals = [], []
            return rfl_vals, wvl_vals

    def find_shadows(self, **kwargs):
        defaultKwargs = {"saveImage": False, 'showPlot': True,'threshold':0.5}
        kwargs = {**defaultKwargs, **kwargs}

        #pixel_list = []
        test_pixel = self.hdr.read_pixel(np.random.randint(
            0, self.hdr.nrows), np.random.randint(0, self.hdr.ncols))
        nbands = len(test_pixel[test_pixel > -900])
        good_pixel_array = np.zeros((self.hdr.nrows, self.hdr.ncols, nbands))
        for x in range(0, self.hdr.nrows):
            for y in range(0, self.hdr.ncols):
                pixel = self.hdr.read_pixel(x, y)
                pixel = pixel[pixel > -999]
                if np.average(pixel) > 0.05:
                    good_pixel_array[x, y] = pixel
                    # pixel_list.append(np.average(pixel))

        bin_array = good_pixel_array
        bin_array[np.abs(bin_array) > 0] = 1
        bin_array = bin_array.astype('float32')

        # plt.imshow(bin_array[:,:,10],cmap='Spectral')

        try:
            os.mkdir(r"D:Data\Lunar_Ice_Images\Shaded_Regions")
        except:
            pass

        if kwargs.get("saveImage") == True:
            tf.imwrite(r"D:Data\Lunar_Ice_Images\Shaded_Regions/"+self._dateTime+"_raw.tif",
                       bin_array[:, :, 0],
                       photometric='minisblack',
                       imagej=True,
                       metadata=self.meta_data_dict)
            print('Image Saved to Shaded_Regions')
        elif kwargs.get('saveImage') == False:
            pass
        else:
            raise Exception(
                "Keyword Argument Error in good_spectra, saveImage")

        if kwargs.get('showPlot') == True:
            plt.imshow(bin_array[:, :, 0], cmap='Spectral')
        elif kwargs.get('showPlot') == False:
            plt.close()

        good_pix = np.count_nonzero(bin_array[:, :, 10])
        total_pix = self.hdr.nrows*self.hdr.ncols
        pct = good_pix/total_pix

        print(bin_array.shape)
        print(f'There are {good_pix} good pixels out of {total_pix} total pixels ({pct:.2%})')

        return bin_array[:, :, 0]

    def wvl_smoothing(self, x, y):
        rfl = []
        wvl = []
        for r, w in zip(self.hdr.read_pixel(x, y), self.hdr.bands.centers):
            if r > -900 and w < 2500 and w > 1000:
                rfl.append(r)
                wvl.append(w)

        rfl, wvl = np.array(rfl), np.array(wvl)
        avg_rfl, std_rfl, avg_wvl = spec_avg(rfl, wvl, 5)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(wvl, rfl)

        #print (avg_wvl)
        f = interp.CubicSpline(avg_wvl, avg_rfl)
        f_err = interp.CubicSpline(avg_wvl, std_rfl)
        x = np.linspace(min(wvl), max(wvl), 100)
        x = np.array([x])

# =============================================================================
#         ax.plot(wvl,rfl,ls='--')
#         ax.plot(avg_wvl,avg_rfl)
# =============================================================================
        # ax.plot(x,f(x),ls='dashdot',c='k')

        x_test = np.linspace(-2, 2, 20)

        def func(x):
            return 1+-1.6*x+0.3*x**2+x**3+0.45*x**4
        y_test = []
        
        for i in x_test:
            y_test.append(func(i))

        x_test = np.array([x_test])

        z = np.polyfit(x[0], f(x)[0], 12)

        z = np.array([np.flip(z)])

        powers = np.array([np.arange(0, len(z[0]))])

        def poly(x, z):
            return np.dot(z, np.repeat(x, len(z[0]), axis=0)**powers.T)

        f1 = poly(x, z)

        #print (f1.shape)

        ax.scatter(x, f(x), c='orange')
        ax.plot(x[0], f1[0], c='k')

        plt.close()

        self.rfl_og = rfl
        self.wvl_og = wvl
        self.wvl_cubfit = x[0]
        self.rfl_cubfit = f(x)[0]
        self.wvl_polyfit = x[0]
        self.rfl_polyfit = f1[0]
        self.z = z[0]

        return x, f1

    def get_minima(self, x, y):
        if self.rfl_cubfit == None:
            raise Exception("Please run wvl_smoothing for this point first")
        else:
            pass
            
        fig, ax = plt.subplots(1, 1)
        # ax.plot(self.wvl_cubfit,self.rfl_cubfit)
        # ax.plot(self.wvl_polyfit,self.rfl_polyfit)

        diff_list = []
        wvl_min_list = []
        for n in range(0, len(self.rfl_cubfit)):
            if n < len(self.rfl_cubfit)-1:
                diff = self.rfl_cubfit[n]-self.rfl_cubfit[n+1]
                diff_list.append(diff)
                if n > 2 and diff < 0 and diff_list[-2] > 0:
                    (diff, diff_list[-2])
                    print(
                        f"Differential: {diff} \nWavelength: {self.wvl_cubfit[n]} \nReflectance: {self.rfl_cubfit[n]} \n")
                    ax.vlines(self.wvl_cubfit[n], min(self.rfl_og), max(self.rfl_og), color='k', ls='dashdot',
                              label=f'{self.wvl_cubfit[n]/1000:.3f} \u03BCm')
                    ax.plot(self.wvl_cubfit, self.rfl_cubfit)
                    wvl_min_list.append(self.wvl_cubfit[n])

        ax.plot(self.wvl_og, self.rfl_og, color='blue', ls='--', alpha=0.3)
                
        return wvl_min_list

    def get_average_rfl(self, true_arr, **kwargs):
        defaultKwargs = {'avg_by_img': False}
        kwargs = {**defaultKwargs, **kwargs}

        rfl_avg = []
        rfl_std = []
        wvl_avg = []
        bands = self.hdr.read_bands(range(2, self.hdr.nbands))
        wvl = self.hdr.bands.centers
        wlength = []
        for i in wvl:
            if i > 1000 and i < 2500:
                wlength.append(i)

        arr = true_arr == 1

        avg_rfl_arr = np.zeros(
            (len(wvl[0:-2]), np.count_nonzero(true_arr == 1)))
        for i in range(0, len(wvl[0:-2])):
            n = 0
            for x, y in zip(np.where(arr == True)[0], np.where(arr == True)[1]):
                avg_rfl_arr[i, n] = bands[x, y, i]
                n += 1

        for i in range(0, avg_rfl_arr.shape[0]):
            rfl_avg.append(np.average(avg_rfl_arr[i, :]))
            rfl_std.append(np.std(avg_rfl_arr[i, :]))

        rfl_avg = np.array(rfl_avg)
        rfl_std = np.array(rfl_std)

        if kwargs["avg_by_img"] == True:
            return wvl[0:-2], rfl_avg, rfl_std
        elif kwargs["avg_by_img"] == False:
            return wvl[0:-2], avg_rfl_arr


end = time.time()
runtime = end-start
if runtime < 1:
    print(f'Program Executed in {runtime*10**3:.3f} milliseconds')
elif runtime < 60 and runtime > 1:
    print(f'Program Executed in {runtime:.3f} seconds')
else:
    print(f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')

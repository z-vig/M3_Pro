# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:34:52 2023

@author: zacha
"""
# Timing the program
from sympy import *
from spec_average import spec_avg
from scipy import interpolate as interp
import tifffile as tf
from tkinter.filedialog import askopenfile as askfile
from tkinter.filedialog import askdirectory as askdir
from tkinter import Tk
import zipfile
import os
from celluloid import Camera
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
start = time.time()

# Importing Necessary Libraries
#from Linear_Spline import linear_spline


# Unused plotting functions
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


# Function to multiple locations of a character in a string
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


# HDR Image class, input hdr image file path
class HDR_Image():
    # Constructor method
    def __init__(self, hdr):
        self.hdr = sp.envi.open(hdr)

    # String print method
    def __str__(self):
        return f"HDR Image Class: {self.hdr.fid.name[find(self.hdr.fid.name,'/')[-1]+1:len(self.hdr.fid.name)-4]}"

    # Returns HDR Object for testing purposes
    def hdr(self):
        return self.hdr

    # Method that returns date, time, and data types of HDR Image
    def datetime(self):
        data_str = self.hdr.fid.name[find(
            self.hdr.fid.name, '/')[-1]+1:len(self.hdr.fid.name)-4]
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

        print(
            f"Observation Type: {obs_type}\nData Type: {data_type}\nDate (Y/M/D): {date}\nTime: {time}\n")

        self.meta_data_str = f"Observation Type: {obs_type}\nData Type: {data_type}\nDate (Y/M/D): {date}\nTime: {time}"
        self.meta_data_dict = {"ObservationType": obs_type,
                               "DataType": data_type, "Date(Y/M/D)": date, "Time": time}
# =============================================================================
#         print (f"Data Type: {data_type}")
#         print (f"Date (Y/M/D): {date}")
#         print (f"Time: {time}")
# =============================================================================

        return [obs_type, data_type, date, time]

    # Method that returns list of wavelengths of HDR Image

    def wavelengths(self):
        print(f"Wavelengths: {self.hdr.bands.centers}")
        # return self.hdr.bands.centers

    # Plots image of a given wavelength or all three supplemental image, if using a .sup file as input
    def plt_img(self, wavelen, **kwargs):
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
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/topo/"+"topo.tif",
                       plot_topo,
                       photometric='minisblack',
                       imagej=True,
                       metadata=self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/temp/"+"temp.tif",
                       plot_temp,
                       photometric='minisblack',
                       imagej=True,
                       metadata=self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/longwave/"+"longwave.tif",
                       plot_longwav,
                       photometric='minisblack',
                       imagej=True,
                       metadata=self.meta_data_dict)
            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/supplemental/all_img/"+"supp.tif",
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
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time)
            except:
                pass

            # plt.savefig(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+str(wavelength)+".tif")
            try:
                os.mkdir(r"D:Data\Lunar_Ice_Images/"+self.date +
                         "_"+self.time+"/"+str(wavelength))
            except:
                pass

            # tf.imwrite(r"C:\Users\zacha/OneDrive/Desktop/test1.tif",norm_band,photometric='minisblack')
            #print (norm_band.dtype)

            tf.imwrite(r"D:Data\Lunar_Ice_Images/"+self.date+"_"+self.time+"/"+str(wavelength)+'/'+str(wavelength)+"_raw.tif",
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
                os.mkdir(r"D:/Data/Lunar_Ice_Images/" +
                         self.date+"_"+self.time+'/all')
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
                    tf.imwrite(r"D:/Data/Lunar_Ice_Images/"+self.date+"_"+self.time+'/all/'+self.date+"_"+self.time+"_allBands.tif",
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
                    tf.imwrite(r"D:/Data/Img_Supp_Files/"+self.date+"_"+self.time+"_allBands.tif",
                               norm_array,
                               photometric='rgb',
                               metadata=self.meta_data_dict)
                    print("TIFF File Written")
                
                print ('Normalized w.r.t all Images')
                return norm_array


    # Plots spectrum of a given x,y point of HDR Image
    def plot_spec(self, x, y, **kwargs):
        defaultKwargs = {'plot_og': True, 'plot_boxcar': False, 'plot_cspline': False, 'plot_cspline_boxcar': False,
                         'box_size': "select", 'saveFig': False, "showPlot": True}
        kwargs = {**defaultKwargs, **kwargs}

        if self.hdr.bands.centers != None:
            fig, ax = plt.subplots(1, 1)

            wvl = self.hdr.bands.centers
            spec_list = []
            wvl_list = []
            for data_val, wavelength in zip(self.hdr.read_pixel(x, y), wvl):
                if abs(data_val) < 500 and wavelength < 2500 and wavelength > 1000:
                    spec_list.append(data_val)
                    wvl_list.append(wavelength)

            if kwargs["plot_og"] == True:
                fancy_spec_plot(fig, ax, np.array(wvl_list), np.array(spec_list), title='Original Spectrum',
                                ylabel='Reflectance', xlabel='Wavelength (\u03BCm)',
                                line_color='red', line_style='--', label='Original Spectrum', ylim=(0.05, 0.2))
            elif kwargs["plot_og"] == False:
                pass
            else:
                raise Exception("Invalid kwarg for plot_og")

            f1 = interp.CubicSpline(np.array(wvl_list), np.array(spec_list))
            x1 = np.linspace(min(wvl_list), max(wvl_list), 80)

            if kwargs["plot_cspline"] == True:
                fancy_spec_plot(fig, ax, x1, f1(x1), line_style='solid', line_color='orange',
                                title='Original Spectrum', label='Cubic Spline')
            elif kwargs["plot_cspline"] == False:
                pass
            else:
                raise Exception('Invalid kwarg for plot_cspline')

            if kwargs["box_size"] == "select":
                box_size = int(input("Box Size: "))
            elif type(kwargs['box_size']) == int:
                box_size = kwargs["box_size"]
            else:
                raise Exception("Invalid kwarg for box_size")

            avg_rfl, std_rfl, avg_wvl = spec_avg(spec_list, wvl_list, 5)

            if kwargs["plot_boxcar"] == True:
                fancy_spec_plot(fig, ax, np.array(avg_wvl), np.array(avg_rfl), std=np.array(std_rfl),
                                line_color='red', std_color='red', title='Original Spectrum', label='Boxcar Average')

            elif kwargs["plot_boxcar"] == False:
                pass
            else:
                raise Exception("Invalid kwarg for plot_boxcar")

            f2 = interp.CubicSpline(np.array(avg_wvl), np.array(avg_rfl))
            f2_err = interp.CubicSpline(np.array(avg_wvl), np.array(std_rfl))
            x2 = np.linspace(min(wvl_list), max(wvl_list), 80)

            if kwargs["plot_cspline_boxcar"] == True:
                fancy_spec_plot(fig, ax, x2, f2(x2), std=f2_err(x2), title='Original Spectrum',
                                line_style='solid', line_color='k', std_color='gray',
                                label='Cubic Spline', ylim=(0.05, 0.2))
            elif kwargs["plot_cspline_boxcar"] == False:
                pass
            else:
                raise Exception("Invalid kwarg for plot_cspline_boxcar")

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

            return np.array(spec_list), np.array(wvl_list)

        elif self.hdr.bands.centers == None:
            spec_list, wvl_list = [], []
            return spec_list, wvl_list

    def good_spectra(self, **kwargs):
        defaultKwargs = {"saveImage": False, 'showPlot': True}
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
            tf.imwrite(r"D:Data\Lunar_Ice_Images\Shaded_Regions/"+self.date+"_"+self.time+"_raw.tif",
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
        print(
            f'There are {good_pix} good pixels out of {total_pix} total pixels ({pct:.2%})')

        return bin_array[:, :, 0]

    def H2O_p1(self, x, y):
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

        ax.plot(self.wvl_og, self.rfl_og, color='blue', ls='--', alpha=0.2)

        num = 0
        w_list = []
        min_list = []
        max_list = []
        target_arr = np.array([[1.242, 1.503, 1.945], [1.323, 1.659, 2.056]])
        for wvl, tar_min, tar_max in zip(wvl_min_list, target_arr[0], target_arr[1]):
            if wvl/1000 > tar_min and wvl/1000 < tar_max:
                w_list.append(wvl/1000)
                min_list.append(tar_min)
                max_list.append(tar_max)
                num += 1
            if num == 3:
                for wvl1, min_, max_ in zip(w_list, min_list, max_list):
                    print(
                        f"Wavelength {wvl1:.3f} is between {min_} and {max_}")
                    ax.vlines(1000*wvl1, min(self.rfl_og), max(self.rfl_og),
                              color='k', ls='dashdot', label=f'{wvl1:.3f} \u03BCm')
                print(f'{num} H\u2082O Peaks Located at {x},{y}\n')
                ax.plot(self.wvl_cubfit, self.rfl_cubfit)
                ax.plot(self.wvl_polyfit, self.rfl_polyfit)
                ax.plot(self.wvl_og, self.rfl_og,
                        color='blue', ls='--', alpha=0.2)
                ax.legend(title="Absorption Minima")

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


if 'hdr_file_list' in locals():
    print('HDR File List is Defined')
else:
    from M3_UnZip import *
    hdr_file_list, hdr_files_path = M3_unzip(
        False, folder=r"D:/Data/20230209T095534013597")

obj_list = []
for file in hdr_file_list:
    if file.find('rfl') > -1:
        obj_list.append(HDR_Image(hdr_files_path+'/'+file[0:-4]+'/'+file))
        
def get_data_fromsave():
    # Get all pixel array from .npy file
    big_arr = np.load("D:Data/big_array.npy")
    
    # Get arr_list with arrays saved
    arr_list = []
    for file in os.listdir(r"D:/Data/Shadow_Arrays"):
        df = pd.read_csv(os.path.join('D:/Data/Shadow_Arrays', file))
        arr = df.to_numpy()
        arr = np.delete(arr, 0, axis=1)
        arr_list.append(arr)
        
    return big_arr,arr_list
        
def get_data_nosave():
    arr_list = []
    for obj in obj_list:
        arr = obj.good_spectra(saveImage=True)
        arr_list.append(arr)

    data_arr = np.zeros((14, 3, len(obj_list[0].hdr.bands.centers)-2))
    n = 0
    for obj, arr in zip(obj_list, arr_list):
        x, avg, std = obj.get_average_rfl(arr)
        data_arr[n, :] = [x, avg, std]
        print('Data Added')
        n += 1

    def avg_by_img(data_arr):
        wav_num = []
        for i in obj_list[0].hdr.bands.centers:
            if i > 1000 and i < 2500:
                wav_num.append(i)

        rfl_total_avglist = np.zeros((14, 83))
        rfl_total_stdlist = np.zeros((14, 83))

        for i in range(0, len(data_arr[:, 0, 0])):

            x, y, std = data_arr[i, 0, :], data_arr[i, 1, :], data_arr[i, 2, :]
            rfl_total_avglist[i, :] = y
            rfl_total_stdlist[i, :] = std

        num_bands = len(rfl_total_avglist[0, :])
        y = np.zeros((num_bands))
        std = np.zeros((num_bands))
        for i in range(0, num_bands):
            y[i] = np.average(rfl_total_avglist[:, i])
            std[i] = np.sqrt(np.sum(rfl_total_stdlist[:, i]**2)
                             )/len(rfl_total_stdlist[:, i])

        fancy_spec_plot(x, y, std=std,
                        title="Average Reflectance of Non-Shaded Lunar South Pole",
                        ylabel='Reflectance', xlabel='Wavelength (\u03BCm)')
    # avg_by_img(data_arr)

    def avg_by_pixel(arr_list):
        rfl_by_pix_arr = np.zeros((83, 1))
        n = 0
        for obj, arr in zip(obj_list, arr_list):
            x, all_arr = obj.get_average_rfl(arr, avg_by_img=False)
            rfl_by_pix_arr = np.concatenate([rfl_by_pix_arr, all_arr], axis=1)
            print(f'Array Added. It is now {rfl_by_pix_arr.shape} big')
            n += 1

        return rfl_by_pix_arr
    big_arr = avg_by_pixel(arr_list)
    big_arr = np.delete(big_arr, 0, 1)
    print("Big Array Filled")
        
if 'big_arr' in locals() and 'arr_list' in locals():
    print ('Pixel Arrays are defined from files')
else:
    try:
        print ('Defining Pixel Arrays')
        big_arr,arr_list = get_data_fromsave()
    except:
        raise Exception('Error: Pixel Arrays may not be saved! Run get_data_nosave()')


##Gets arr_list and big_arr without save

# get_data_nosave()

##Get Date and Time of each stamp
# =============================================================================
# for obj in obj_list:
#     obj.datetime()
# =============================================================================


##Get Minima for a set amount of points
# =============================================================================
# x_pix = range(0,obj_list[0].hdr.nrows)
# y_pix = range(0,obj_list[0].hdr.ncols)
#
# for x in x_pix[91:92]:
#     for y in y_pix[100:101]:
#         print (x,y)
#         obj_list[0].H2O_p1(x,y)
#         obj_list[0].get_minima(x,y)
# =============================================================================


##Get average reflectance data
def get_avg_rfl_data(plot_data=False):
    wvl, rfl_all_pixels = obj_list[0].get_average_rfl(arr_list[0])
    rfl_avgSP = np.zeros((83))
    rfl_stdSP = np.zeros((83))
    for count, band in enumerate(rfl_all_pixels):
        rfl_avgSP[count] = np.average(band)
        rfl_stdSP[count] = np.std(band)
    
    # Plot Average Reflectance Data
    if plot_data == True:
        fig,ax = plt.subplots(1,1)
        fancy_spec_plot(fig,ax,wvl[21:73],rfl_avgSP[21:73],std=rfl_stdSP[21:73],
                        title="Average Reflectance of Non-Shaded Lunar South Pole",
                        ylabel= 'Reflectance', xlabel = 'Wavelength (\u03BCm)')
        
get_avg_rfl_data(plot_data=True)




# plt.savefig(r"D:/Data/Figures/Average_SPole_Reflectance.png")


def get_single_minima(wvl, rfl):
    diff_list = []
    wvl_min_list = []
    for n in range(0, len(rfl)):
        if n < len(rfl)-1:
            diff = rfl[n]-rfl[n+1]
            diff_list.append(diff)
            if n > 2 and diff < 0 and diff_list[-2] > 0:
                (diff, diff_list[-2])
                print(
                    f"Differential: {diff} \nWavelength: {wvl[n]} \nReflectance: {rfl[n]} \n")
                wvl_min_list.append(wvl[n])
    return wvl_min_list


def shade_correction(x_pt, y_pt, **kwargs):
    defaultKwargs = {'plot_og': True, 'plot_avg': False,
                     'plot_cspline': False, 'plot_minima': True, 'saveImage': False}
    kwargs = {**defaultKwargs, **kwargs}

    R_bi = rfl_avgSP[21:73]

    #R_meas,w = obj_list[0].plot_spec(0,10,plot_cspline_boxcar=True,box_size=5)
    R_meas, w = obj_list[0].plot_spec(
        x_pt, y_pt, plot_cspline_boxcar=True, box_size=5)

    R_T = R_meas/R_bi

    avg_rfl, std_rfl, avg_wvl = spec_avg(R_T, w, 5)
    f = interp.CubicSpline(np.array(avg_wvl), np.array(avg_rfl))
    ferr = interp.CubicSpline(np.array(avg_wvl), np.array(std_rfl))
    x = np.linspace(min(w), max(w), 100)

    wvl_min_list = get_single_minima(x, f(x))
    print(wvl_min_list)

    fig, ax = plt.subplots(1, 1)
    if kwargs['plot_og'] == True:
        fancy_spec_plot(fig, ax, w, R_T, line_style='dashdot', title="Shadow-Corrected Spectrum",
                        ylabel='Reflectance', xlabel='Wavelength (\u03BCm)', label='Original')
    if kwargs['plot_avg'] == True:
        fancy_spec_plot(fig, ax, np.array(avg_wvl), np.array(avg_rfl), std=np.array(std_rfl),
                        line_style='--', line_color='red', std_color='red', title='Shadow-Corrected Spectrum', label='Boxcar Average')
    if kwargs['plot_cspline'] == True:
        fancy_spec_plot(fig, ax, x, f(x), std=ferr(x), title="Shadow-Corrected Spectrum",
                        line_style='solid', line_color='k', std_color='gray', label='Cubic Spline')
    if kwargs['plot_minima'] == True:
        for _min in wvl_min_list:
            ax.vlines(_min, min(f(x)), max(f(x)), ls='--',
                      color='k', label=str(round(_min, 1)))

    ax.legend()

    if kwargs.get('saveImage') == True:
        plt.savefig(r"D:Data/Figures/"+str(x_pt)+'_' +
                    str(y_pt)+"_shadow_correction.png")

# shade_correction(90,91,plot_avg=False,plot_cspline=True,saveImage=True)

def normalize_all_images():
    all_stats = np.zeros((52,2))
    for num,row in enumerate(big_arr[21:73]):
        all_stats[num,0] = min(row)
        all_stats[num,1] = max(row)
    
    norm_img_list = []
    for count,obj in enumerate(obj_list):
        norm_img = obj.plt_img(1449.11,All_Bands=True,Norm='All',allMax=all_stats[:,1],allMin=all_stats[:,0],saveImage=False)
        norm_img_list.append(norm_img) 
        print (f'Image {count+1} out of {len(obj_list)} complete in {time.time()-start} seconds')
#normalize_all_images()

end = time.time()
runtime = end-start
if runtime < 1:
    print(f'Program Executed in {runtime*10**3:.3f} milliseconds')
elif runtime < 60 and runtime > 1:
    print(f'Program Executed in {runtime:.3f} seconds')
elif runtime > 60:
    print(
        f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')

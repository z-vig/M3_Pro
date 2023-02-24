# -*- coding: utf-8 -*-
"""
Supplementary file for HySpec_Image_Processing.py
"""

##Importing Necessary Modules
import numpy as np
import os
import pandas as pd
from HySpec_Image_Processing import HDR_Image
from fancy_spec_plot import fancy_spec_plot
import matplotlib.pyplot as plt
from spec_average import spec_avg
from scipy import interpolate as interp

##Timing the program
import time
start = time.time()

if 'hdr_file_list' in locals():
    print('HDR File List is Defined')
else:
    from M3_UnZip import M3_unzip
    hdr_file_list, hdr_files_path = M3_unzip(select = True)
    #hdr_file_list, hdr_files_path = M3_unzip(select = False, folder=r'/run/media/zvig/My Passport/Data/20230209T095534013597')

obj_list = []
for file in hdr_file_list:
    if file.find('rfl') > -1:
        obj_list.append(HDR_Image(file))
        
for obj in obj_list:
    obj.datetime
        
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
        arr = obj.find_shadows(saveImage=True)
        arr_list.append(arr)

    data_arr = np.zeros((14, 3, len(obj_list[0].hdr.bands.centers)-2))
    n = 0
    for obj, arr in zip(obj_list, arr_list):
        x, avg, std = obj.get_average_rfl(arr)
        data_arr[n, :] = [x, avg, std]
        print(f'Data Added. Size is now {data_arr.shape}')
        n += 1

    return arr_list,data_arr

arr_list,data_arr = get_data_nosave()

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
        print ('Pixel Arrays Defined')
    except:
        raise Exception('Error: Pixel Arrays may not be saved! Run get_data_nosave()')


##Gets arr_list and big_arr without save

# get_data_nosave()

##Get Date and Time of each stamp
# =============================================================================
# for obj in obj_list:
#     obj.datetime()
# =============================================================================


##Get Minima for a set amount of point
# =============================================================================
# print ('Noise Reduction')
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
    return wvl,rfl_avgSP,rfl_stdSP
        
wvl, rfl_avgSP, rfl_stdSP = get_avg_rfl_data(plot_data=False)




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
                #print(f"Differential: {diff} \nWavelength: {wvl[n]} \nReflectance: {rfl[n]} \n")
                wvl_min_list.append(wvl[n])
    return wvl_min_list


def shade_correction(x_pt, y_pt, **kwargs):
    defaultKwargs = {'plotImage':True,'plot_og': True, 'plot_avg': False,
                     'plot_cspline': False, 'plot_minima': True, 'saveImage': False,'returnShadeCorrection':False}
    kwargs = {**defaultKwargs, **kwargs}

    R_bi = rfl_avgSP[21:73]

    #R_meas,w = obj_list[0].plot_spec(0,10,plot_cspline_boxcar=True,box_size=5)
    R_meas, w = obj_list[0].plot_spec(x_pt, y_pt, plot_cspline_boxcar=True, box_size=5,showPlot=False)

    R_T = R_meas/R_bi

    avg_rfl, std_rfl, avg_wvl = spec_avg(R_T, w, 5)
    f = interp.CubicSpline(np.array(avg_wvl), np.array(avg_rfl))
    ferr = interp.CubicSpline(np.array(avg_wvl), np.array(std_rfl))
    x = np.linspace(min(w), max(w), 100)

    wvl_min_list = get_single_minima(x, f(x))
    
    if kwargs.get('returnShadeCorrection') == True:
        return f(x)
    #print(wvl_min_list)
    
    if kwargs.get('plotImage') == True:
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
    else:
        pass
        

    if kwargs.get('saveImage') == True:
        plt.savefig(r"D:Data/Figures/"+str(x_pt)+'_' +
                    str(y_pt)+"_shadow_correction.png")
        
    return f(x)

print (f'Shadow Correction started at {time.time()-start}')
orig_img = obj_list[0].plt_img(1449.11, All_Bands=True,Norm='None',saveImage=False)
dark_img = arr_list[0]
total_dark = np.count_nonzero(dark_img==0)
n=0
for x,y in zip(np.where(dark_img==0)[0],np.where(dark_img==0)[1]):
    shade_correction(x,y,returnShadeCorrection=True)
    print (f'{n} out of {total_dark} ({n/total_dark:.0%})')
    n+=1


#pt = shade_correction(91,100,plot_avg=False,plot_cspline=True,saveImage=False,plotImage=False)


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
else:
    print(f'Program Executed in {runtime/60:.0f} minutes and {runtime%60:.3f} seconds')

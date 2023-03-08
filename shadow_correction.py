#%%
from HySpec_Image_Processing import HDR_Image
from get_pixel_mosaic import create_arrays
from M3_UnZip import M3_unzip
import os
import matplotlib.pyplot as plt
import numpy as np
from fancy_spec_plot import fancy_spec_plot
from spec_average import spec_avg
from scipy import interpolate as interp
from matplotlib.animation import FuncAnimation
#%%
## Getting necessary data arrays and M3 stamp list
if 'shadow' not in locals():
    shadow,imgStats,mosaicArray,mosaicStats = create_arrays(r'/run/media/zvig/My Passport/Data')
    print ('Arrays Loaded')
elif 'shadow' in locals():
    print ('Arrays Exist')

hdrFileList,hdrFilesPath = M3_unzip(select=False,folder=r'/run/media/zvig/My Passport/Data/20230209T095534013597')
stampList = []
for file in hdrFileList:
    stampList.append(HDR_Image(os.path.join(hdrFilesPath,file)))

## Getting average reflectance of South Pole (R_bi)
def get_avg_rfl_data(plot_data=False):
    wvl = stampList[0].hdr.bands.centers
    rfl_avgSouthPole=np.zeros(83)
    rfl_stdSouthPole=np.zeros(83)
    for n in range(mosaicArray.shape[0]):
        rfl_avgSouthPole[n] = np.average(mosaicArray[n,:])
        rfl_stdSouthPole[n] = np.std(mosaicArray[n,:])

    
    # Plot Average Reflectance Data
    if plot_data == True:
        fig,ax = plt.subplots(1,1)
        fancy_spec_plot(fig,ax,wvl[21:73],rfl_avgSouthPole[21:73],std=rfl_stdSouthPole[21:73],
                        title="Average Reflectance of Non-Shaded Lunar South Pole",
                        ylabel= 'Reflectance', xlabel = 'Wavelength (\u03BCm)')
        
    return wvl,rfl_avgSouthPole,rfl_stdSouthPole
        
wvl, rfl_avgSP, rfl_stdSP = get_avg_rfl_data(plot_data=False)

## Shadow Correction
def shadow_correction(x_pt, y_pt, **kwargs):
    defaultKwargs = {'plotImage':True,'plot_og': True, 'plot_avg': False,
                     'plot_cspline': False, 'plot_minima': True, 'saveImage': False,'returnShadeCorrection':False}
    kwargs = {**defaultKwargs, **kwargs}

    R_bi = rfl_avgSP[21:73]

    R_meas, w = stampList[0].plot_spec(x_pt, y_pt, plot_cspline_boxcar=True, box_size=5,showPlot=False)

    R_T = R_meas/R_bi

    avg_rfl, std_rfl, avg_wvl = spec_avg(R_T, w, 5)
    f = interp.CubicSpline(np.array(avg_wvl), np.array(avg_rfl))
    ferr = interp.CubicSpline(np.array(avg_wvl), np.array(std_rfl))
    x = np.linspace(min(w), max(w), 100)
    
    if kwargs.get('returnShadeCorrection') == True:
        return f(x)
    
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

#%%
plt.imshow(shadow[0])
original = stampList[0].hdr.read_band(2)
darkX,darkY = np.where(shadow[0]==0)
fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot(1,1,1)
#ax2 = plt.subplot(1,2,2)
og_img = ax1.imshow(original)

data = np.array(([0,0,0],[0,1,0],[0,0,0]))
xlist,ylist=np.where(data==0)

#img = ax1.imshow(data)

print (darkX,darkY)
def animation_frame(i):
    x = darkX[i]
    y = darkY[i]
    R_T = original[x,y]
    R_bi = shadow_correction(x,y,returnShadeCorrection=True)[0]
    original[x,y] = R_T/R_bi
    og_img.set_data(original)

    return og_img

animation = FuncAnimation(fig,func=animation_frame,frames=np.arange(0,len(darkX),1),interval=0.01)
plt.show()

# n = 0
# for x,y in zip(darkPixels[0],darkPixels[1]):
#     print (f'\rPlotting {x},{y}',end='')
#     #original[x,y] = shadow_correction(x,y,returnShadeCorrection=True)[0]

#ax2.imshow(original)


# %%

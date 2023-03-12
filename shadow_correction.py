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
from cubic_spline_image import cubic_spline_image
from cubic_spline_image import removeNAN
import spectral as sp

## Getting necessary data arrays and M3 stamp list
if 'shadow' not in locals():
    shadow,imgStats,mosaicArray,mosaicStats = create_arrays(r'D:/Data')
    print ('Arrays Loaded')
elif 'shadow' in locals():
    print ('Arrays Exist')

hdrFileList,hdrFilesPath = M3_unzip(select=False,folder=r'D:/Data/20230209T095534013597')
stampList = []
for file in hdrFileList:
    stampList.append(HDR_Image(os.path.join(hdrFilesPath,file)))

#cubic_img = np.load(r"/run/media/zvig/My Passport/Data/cubic_spline_image.npy")

## Getting average reflectance of South Pole (R_bi)
def get_avg_rfl_data(plot_data=False):
    wvl = stampList[0].hdr.bands.centers[2:]
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
        


## Shadow Correction
def shadow_correction(x_pt, y_pt, **kwargs):
    defaultKwargs = {'plotImage':True,'plot_og': True, 'plot_avg': False,
                     'plot_cspline': False, 'plot_minima': True, 'saveImage': False,'returnShadeCorrection':False}
    kwargs = {**defaultKwargs, **kwargs}

    R_bi = rfl_avgSP[21:73]

    R_meas, w = stampList[0].plot_spec(x_pt, y_pt, plot_cspline_boxcar=True, box_size=5,showPlot=False)
    imgAverage,imgCubic = cubic_spline_image()

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
# =============================================================================
#         if kwargs['plot_minima'] == True:
#             for _min in wvl_min_list:
#                 ax.vlines(_min, min(f(x)), max(f(x)), ls='--',
#                           color='k', label=str(round(_min, 1)))
# =============================================================================
                
        ax.legend()

if __name__ == "__main__":
    print('Plotting...')
    wvl, rfl_avgSP, rfl_stdSP = get_avg_rfl_data(plot_data=False)
    wvl = np.array(wvl)
    allowedIndices = np.where((wvl>900)&(wvl<2600))[0]
    allowedWvl = wvl[allowedIndices]
    hdr = sp.envi.open(r"D:/Data/20230209T095534013597/extracted_files/hdr_files/m3g20090417t193320_v01_rfl/m3g20090417t193320_v01_rfl.hdr")
    R_c = hdr.read_bands(allowedIndices)
    R_meas = hdr.read_bands(allowedIndices)

    R_bi = rfl_avgSP[allowedIndices]

    xShade,yShade = np.where(shadow[0]==0)
    xLight,yLight = np.where(shadow[0]!=0)

    R_c[xShade,yShade,:] = R_c[xShade,yShade,:]/R_bi

    #imgAverageCorrected,imgCubicCorrected = cubic_spline_image(R_c,wvl,5)

    cubic_img = np.load(r"D:/Data/cubic_spline_image.npy")
    plt.imshow(cubic_img[:,:,0])

    def plot_correction(x,y):
        fig,ax = plt.subplots(1,1)
        ax.plot(R_c[x,y,:],label='Corrected')
        ax.plot(R_meas[x,y,:],label='Original')
        ax.set_title(f'{x},{y} Spectrum')
        ax.legend()
    
    plot_correction(1,2)
    plot_correction(91,100)
    plot_correction(1000,300)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.imshow(R_meas[:,:,0])
    ax2.imshow(R_c[:,:,0])
    plt.show()

    

    # fig,ax = plt.subplots(1,1)
    # ax.plot(allowedWvl,original_img[91,100,:])
    # ax.plot(allowedWvl,cubic_img[91,100,:])
    # plt.show()



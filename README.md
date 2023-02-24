# M3 Image Processing
This package aims to process M3 Hyperspectral data with goal of better constraining the location of H~2~O Ice on the Lunar Surface

## HySpec Image Processing File
Here you will find the HDR_Image() class. This class is intended to perform basic image processing techniques such as noise reduction and continuum removal. To begin, load your image into the HDR_Image class as follows:
```
from HySpec_Image_Processing import HDR_Image
myimage = HDR_Image(path_to_image)
```
From here, several methods are available for various kinds of exploratory analysis. 

### HDR_Image().hdr
Creates a specPy BilFile object. From this you can get band names, pixel info, metadata and much more. See specPy documentation. \n
**return:** BilFile object

### HDR_Image().plot_image(wvl,\*\*kwargs)
Plots reflectance hdr files and supplemental hdr files using a single band or all bands in a 3-dimensional image, and saves these images to a specified folder \n
**kwargs:** All_Bands = Boolen, Norm = (All,Image,None), allMax = ndarray, allMin = ndarray, saveImage = Boolean \n
**return:** ndarray of image nomalized to itself OR ndarray of image normalized to set max and min OR ndarray of original reflectance image

### HDR_Image().plot_spec(x,y,\*\*kwargs)
Plots the spectrum for a given pixel on the image and saves this to a specified folder \n
**kwargs:** saveFig = Boolean, showPlot = Boolean, plot_og, plot_boxcar, plot_cspline, plot_cspline_boxcar, plot_movingAvg, plot_movingAvg_cspline (all plot_ kwargs = Boolean), box_size=(select or type(float)), movingAvg_size=(select of type(float)) \n
**return:** ndarray of reflectance values, ndarray of wavelength values

### HDR_Image().find_shadows(\*\*kwargs)
Tests each pixel in the image to see if the average reflectance over all the bands is below a certain threshold. \n
**kwargs:** saveImage = Boolean, showPlot = Boolean, threshold = type(float) \n
**return:** ndarray with size equal to the original image with pixel values = 1 for non-shadowed regions and pixel values = 0 for shadowed regions

### HDR_Image().wvl_smoothing(x,y)
Smooths the reflectance spectrum and reduces noise using a cubic spline fit. \n
**return:** ndarray of modeled wavelengths, ndarray of modeled reflectance

### HDR_Image().get_minima(x,y)
Plots the smoothed spectrum from wvl_smoothing and draws vertical lines at identified absorption bands. \n
**return:** list containing the wavelength values of all identified absorption bands

### HDR_Image().get_average_rfl(true_arr,\*\*kwargs)
Assists in finding the average reflectance values for a given mosaic dataset. \n
**kwargs:** avg_by_img = Boolean \n
**return:** wavelnegths,average reflectances and standard deviation reflectances OR wavelengths, ndarray of all pixels to be averaged in mosaic




# M3 Image Processing
This package aims to process M3 Hyperspectral data with goal of better constraining the location of H~2~O Ice on the Lunar Surface

## HySpec Image Processing File
Here you will find the HDR_Image() class. This class is intended to perform basic image processing techniques such as noise reduction and continuum removal. 

```
import HDR_Image()

image1 = HDR_Image(path+'/'+file)
image1.datetime()
image1.plot_img()
image1.plot_spec(x,y,**kwargs)
```

'''
Script for georeferencing .tiff image for use in ArcGIS Pro using ground control points.
The script requires a 2-band backplane image with first plane as the latitude and the second band as the longitude.
'''
#%%
import rasterio as rio
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
from rasterio.control import GroundControlPoint as GCP
import spectral as sp
from rasterio.transform import from_gcps
import tifffile as tf
from rasterio.crs import CRS
import numpy as np
import matplotlib.pyplot as plt
import spectral as sp
from tkinter.filedialog import askopenfilename as askfile
from tkinter.filedialog import askdirectory as askdir
import os

##Utility fucntion
def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

##Function for projecting a lat-long point into a stereographic projection
def stereo_project(lat:float,long:float)->tuple:
    return (2*1737400*np.tan(np.pi/4-np.pi*abs(lat/360))*np.sin(np.pi*long/180),\
            2*1737400*np.tan(np.pi/4-np.pi*abs(lat/360))*np.cos(np.pi*long/180))

#Function for adding georeference tags to a tiff image, given a location backplane
def georef(originalStampPath:str,locBackplanePath:str,saveFolder:str,crs_wkt:str)->np.ndarray:
    ##Loading images from paths and defining lat/long coordinates
    originalStamp = tf.imread(originalStampPath)
    originalStamp = np.moveaxis(originalStamp,2,0)

    rio_obj = rio.open(originalStampPath)
    im = rio_obj.read(rio_obj.indexes)

    #backplane_hdr = sp.envi.open(locBackplanePath)
    #backplaneStamp = backplane_hdr.read_bands[0,1,2]
    backplaneStamp = tf.imread(locBackplanePath)
    long,lat = backplaneStamp[:,:,0],backplaneStamp[:,:,1]

    ##Setting lat-long of corner points
    lon1,lat1 = (long[0,0],lat[0,0])
    lon2,lat2 = (long[0,backplaneStamp.shape[1]-1],lat[0,backplaneStamp.shape[1]-1])
    lon3,lat3 = (long[backplaneStamp.shape[0]-1,backplaneStamp.shape[1]-1], lat[backplaneStamp.shape[0]-1,backplaneStamp.shape[1]-1])
    lon4,lat4 = (long[backplaneStamp.shape[0]-1,0],lat[backplaneStamp.shape[0]-1,0])

    corner_coords = [(lon1,lat1),(lon2,lat2),(lon3,lat3),(lon4,lat4)]
    corner_coords_meters = [stereo_project(corner_coords[i][1],corner_coords[i][0]) for i in range(len(corner_coords))]
    ul,ur,lr,ll = [i for i in corner_coords_meters]

    gcps_corners = [
        GCP(0.5, 0.5, *ul),
        GCP(0.5,im.shape[0]-0.5,*ur),
        GCP(im.shape[1]-0.5,0.5,*ll),
        GCP(im.shape[1]-0.5, im.shape[0]-0.5, *lr)
    ]

    all_coords = [(i,j) for i,j in zip(backplaneStamp[:,:,0].flatten(),backplaneStamp[:,:,1].flatten())]

    random_ind = np.random.choice(range(len(all_coords)),10000,replace=False)
    Y,X = np.meshgrid(np.arange(backplaneStamp.shape[1]),np.arange(backplaneStamp.shape[0]))
    random_np_coords = np.array([(X.flatten()[i],Y.flatten()[i]) for i in list(random_ind)])
    random_coords_latlong = np.array([all_coords[i] for i in list(random_ind)])
    random_coords_meters = stereo_project(random_coords_latlong[:,1],random_coords_latlong[:,0])
    random_coords_meters = np.array((random_coords_meters[0],random_coords_meters[1])).T

    gcps_random = [GCP(*random_np_coords[i,:],*random_coords_meters[i,:]) for i in range(len(random_ind))]

    gcps = gcps_corners+gcps_random

    transform = from_gcps(gcps)
    
    name_index = find_all(originalStampPath,'/')[-1]
    stampName = originalStampPath[name_index:-4]
    with rio.open(f'{saveFolder}/{stampName}_georef.tif','w',
            driver='GTiff',
            height=originalStamp.shape[1],
            width = originalStamp.shape[2],
            count=originalStamp.shape[0],
            dtype=originalStamp.dtype,
            crs=crs_wkt,
            transform=transform,
            interleave = 'pixel'
                ) as img:
        img.write(originalStamp)
    
    return originalStamp

if __name__ == "__main__":
    # print ('Select image to be georeferenced:')
    # originalImage = askfile()
    # print ('Select location backplane image:')
    # locBackplane = askfile()
    # print ('Select folder to save georeferenced raster:')
    # saveFolder = askdir()
    # crs = CRS.from_wkt('PROJCS["Moon 2000 South Pole Stereographic",\
    #                     GEOGCS["Moon 2000",\
    #                     DATUM["D_Moon_2000",\
    #                     SPHEROID["Moon_2000_IAU_IAG",1737400,0]],\
    #                     PRIMEM["Reference_Meridian",0],\
    #                     UNIT["degree",0.0174532925199433]],\
    #                     PROJECTION["Stereographic"],\
    #                     PARAMETER["latitude_of_origin",-90],\
    #                     PARAMETER["central_meridian",0],\
    #                     PARAMETER["scale_factor",1],\
    #                     PARAMETER["false_easting",0],\
    #                     PARAMETER["false_northing",0],\
    #                     UNIT["meter",1]]')
    # print (f'Georeferencing {originalImage} and saving to {saveFolder}...')
    # img = georef(originalImage,locBackplane,saveFolder,crs)
    # plt.imshow(img[0,:,:])
    # plt.show()

    og_img_path = 'D:/Data/Ice_Pipeline_Out_5-4-23/originalImages'
    loc_img_path = 'D:/Data/Ice_Pipeline_Out_5-4-23/locationInfo'
    saveFolder = 'D:/Data/Ice_Pipeline_Out_5-4-23/georeference_originals/'
    crs = CRS.from_wkt('PROJCS["Moon 2000 South Pole Stereographic",\
                        GEOGCS["Moon 2000",\
                        DATUM["D_Moon_2000",\
                        SPHEROID["Moon_2000_IAU_IAG",1737400,0]],\
                        PRIMEM["Reference_Meridian",0],\
                        UNIT["degree",0.0174532925199433]],\
                        PROJECTION["Stereographic"],\
                        PARAMETER["latitude_of_origin",-90],\
                        PARAMETER["central_meridian",0],\
                        PARAMETER["scale_factor",1],\
                        PARAMETER["false_easting",0],\
                        PARAMETER["false_northing",0],\
                        UNIT["meter",1]]')
    n = 1
    tot = len(os.listdir(og_img_path))
    for og,bp in zip(os.listdir(og_img_path),os.listdir(loc_img_path)):
        ogPath = f'{og_img_path}/{og}'
        locPath = f'{loc_img_path}/{bp}'
        print (f'\rProcessing {n} of {tot} ({n/tot:.0%})',end='\r')
        georef(ogPath,locPath,saveFolder,crs)
        n+=1

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
import pandas as pd
import time

##Utility fucntion
def find_all(s,c):
    return [n for n,i in enumerate(s) if i==c]

##Function for projecting a lat-long point into a stereographic projection
def stereo_project(lat:float,long:float)->tuple:
    return (2*1737400*np.tan(np.pi/4-np.pi*abs(lat/360))*np.sin(np.pi*long/180),\
            2*1737400*np.tan(np.pi/4-np.pi*abs(lat/360))*np.cos(np.pi*long/180))

#Function for adding georeference tags to a tiff image, given a location backplane
def georef(originalStampPath:str,locBackplanePath:str,saveFolder:str,crs_wkt:str,no_data_value:int)->np.ndarray:
    ##Loading images from paths and defining lat/long coordinates
    originalStamp = tf.imread(originalStampPath)
    if len(originalStamp.shape)==2:
        originalStamp = np.expand_dims(originalStamp,2)
    else:
        pass
    band_index = originalStamp.shape.index(min(originalStamp.shape))
    originalStamp = np.moveaxis(originalStamp,band_index,0)

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

    ##Setting lat-long of border points
    right_ind = backplaneStamp.shape[1]-1
    left_ind = 0
    top_ind = 0
    bot_ind = backplaneStamp.shape[0]-1
    rBorder,rBorder_np = [(i,j) for i,j in zip(backplaneStamp[:,right_ind,0],backplaneStamp[:,right_ind,1])],[(i,j) for i,j in zip(np.arange(bot_ind+1),right_ind*np.ones(bot_ind+1))]
    lBorder,lBorder_np = [(i,j) for i,j in zip(backplaneStamp[:,left_ind,0],backplaneStamp[:,left_ind,1])],[(i,j) for i,j in zip(np.arange(bot_ind+1),np.zeros(bot_ind+1))]
    tBorder,tBorder_np = [(i,j) for i,j in zip(backplaneStamp[top_ind,:,0],backplaneStamp[top_ind,:,1])],[(i,j) for i,j in zip(np.zeros(right_ind+1),np.arange(right_ind+1))]
    bBorder,bBorder_np = [(i,j) for i,j in zip(backplaneStamp[bot_ind,:,0],backplaneStamp[bot_ind,:,1])],[(i,j) for i,j in zip(bot_ind*np.ones(right_ind+1),np.arange(right_ind+1))]
    borderCoords = np.array(rBorder+lBorder+tBorder+bBorder)
    borderCoords_np = np.array(rBorder_np+lBorder_np+tBorder_np+bBorder_np)
    borderCoords_meters = stereo_project(borderCoords[:,1],borderCoords[:,0])
    borderCoords_meters = np.array((borderCoords_meters[0],borderCoords_meters[1])).T

    gcps_border = [GCP(*borderCoords_np[i,:],*borderCoords_meters[i,:]) for i in range(borderCoords.shape[0])]
    

    ##Setting lat-long of random center points
    all_coords = [(i,j) for i,j in zip(backplaneStamp[:,:,0].flatten(),backplaneStamp[:,:,1].flatten())]

    GCPS_NUM = 50000
    if backplaneStamp.shape[0]*backplaneStamp.shape[1] < GCPS_NUM:
        print (f'Could not georeference a small image of size: {backplaneStamp.shape}')
        return None,None
    else:
        random_ind = np.random.choice(range(len(all_coords)),GCPS_NUM,replace=False)
        Y,X = np.meshgrid(np.arange(backplaneStamp.shape[1]),np.arange(backplaneStamp.shape[0]))
        random_np_coords = np.array([(X.flatten()[i],Y.flatten()[i]) for i in list(random_ind)])
        random_coords_latlong = np.array([all_coords[i] for i in list(random_ind)])
        random_coords_meters = stereo_project(random_coords_latlong[:,1],random_coords_latlong[:,0])
        random_coords_meters = np.array((random_coords_meters[0],random_coords_meters[1])).T

        gcps_random = [GCP(*random_np_coords[i,:],*random_coords_meters[i,:]) for i in range(len(random_ind))]

        gcps = gcps_corners+gcps_random+gcps_border

        transform = from_gcps(gcps)
        
        #name_index = find_all(originalStampPath,'/')[-1]
        stampName = os.path.basename(originalStampPath)
        with rio.open(f'{saveFolder}/{stampName[:-4]}_georef.tif','w',
                driver='GTiff',
                height=originalStamp.shape[1],
                width = originalStamp.shape[2],
                count=originalStamp.shape[0],
                dtype=originalStamp.dtype,
                crs=crs_wkt,
                transform=transform,
                interleave = 'pixel',
                nodata = no_data_value
                    ) as img:
            img.write(originalStamp)

        allgcps_array = np.concatenate([borderCoords_meters,random_coords_meters],axis=0)
        allgcps_np = np.concatenate([borderCoords_np,random_np_coords],axis=0)
        #print (allgcps_np.shape)
        allgcps_array = np.concatenate([allgcps_np,allgcps_array],axis=1)

        return originalStamp,allgcps_array

if __name__ == "__main__":
    start = time.time()
    processing = input('Batch or Single Image?')
    if processing=='Single Image':
        start = time.time()
        print ('Select image to be georeferenced:')
        originalImage = askfile()
        if originalImage == '':
            raise InterruptedError('Georeference canceled!')
        else:
            print (f'{originalImage} selected.')
        print ('Select location backplane image:')
        locBackplane = askfile()
        if originalImage == '':
            raise InterruptedError('Georeference canceled!')
        else:
            print (f'{locBackplane} selected.')
        print ('Select folder to save georeferenced raster:')
        saveFolder = askdir()
        if originalImage == '':
            raise InterruptedError('Georeference canceled!')
        else:
            print (f'{saveFolder} selected.')
        no_data_value = input('No Data Value: ')

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
        print (f'Georeferencing {originalImage} and saving to {saveFolder}...')
        img,allgcps = georef(originalImage,locBackplane,saveFolder,crs,no_data_value)
        with open('D:/Data/gcps.txt','w') as f:
            for i in allgcps:
                num_list = list(i)
                str_list = [str(i) for i in num_list]
                write_line = '\t'.join(str_list)
                f.write(f'{write_line}\n')
        f.close()
        print (f'Georeferencing complete in {(time.time()-start)/60:.2} minutes!')
        plt.imshow(img[0,:,:])
        plt.show()

    elif processing=='Batch':
        print ('Select Original Images Folder:')
        og_img_path = askdir() #'D:/Data/OP2C_Downloads/L2_sorted/sorted_tif_files/rfl_cropped'
        print (f'{og_img_path} selected\nSelect Location Backplane Folder:')
        loc_img_path = askdir() #'D:/Data/OP2C_Downloads/L1_sorted/sorted_tif_files/loc_cropped'
        print (f'{loc_img_path} selected\nSelect Save Folder:')
        try:
            os.mkdir(f'{og_img_path}_georeferenced')
        except:
            pass
        saveFolder = f'{og_img_path}_georeferenced' #'D:/Data/OP2C_Downloads/georeferenced_images'
        print(f'{saveFolder} selected\n')
        no_data_value = input('No Data Value: ')
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
        srcFolder = [i for i in os.listdir(og_img_path) if i.find('.ovr')==-1]
        dstFolder = os.listdir(loc_img_path)
        srcFolder.sort(),dstFolder.sort()
        tot = len(srcFolder)
        for og,bp in zip(srcFolder,dstFolder):
            ogPath = os.path.join(og_img_path,og)
            locPath = os.path.join(loc_img_path,bp)
            index = find_all(os.path.basename(ogPath),'_')[0]
            #print (os.path.basename(ogPath)[:index],os.path.basename(locPath)[3:index+3])
            if os.path.basename(ogPath)[:index]!=os.path.basename(locPath)[3:index+3]:
                print (f'{os.path.basename(ogPath)}!={os.path.basename(locPath)}')
                raise ValueError('Backplane does not match image!')
            print (f'\rProcessing {n} of {tot} ({n/tot:.0%})',end='\r')
            georef(ogPath,locPath,saveFolder,crs,no_data_value)
            n+=1
        print (f'Georeferencing complete!-----Elapsed time: {(time.time()-start)/60:.2f} minutes.')
    
    else:
        raise SyntaxError('Please select a valid processing option.')
# %%

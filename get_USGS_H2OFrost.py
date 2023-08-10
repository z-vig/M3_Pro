##Water Frost Spectrum
#%%
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory as askdir

def get_USGS_H2OFrost(USGS_folder:str,resample_wvl:list):
    '''
    Function for obtaining the USGS Water Frost Spectrum used in Li et al., 2018
    '''
    water = pd.read_csv(f'{USGS_folder}/splib07a_H2O-Ice_GDS136_77K_BECKa_AREF.txt')
    wavelengths = pd.read_csv(f'{USGS_folder}/splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt')
    water.columns = ['']
    wavelengths.columns = ['']

    goodIndices = np.where((wavelengths*1000>900)&(wavelengths*1000<2600))[0]

    wvl,rfl = wavelengths.iloc[goodIndices,0]*1000,water.iloc[goodIndices,0]
    #print (f'Water Length: {water.shape}, Wavelength Length: {wavelengths.shape}')
    f = interp.CubicSpline(wvl,rfl)
    xtest = resample_wvl

    #fig,ax = plt.subplots(1,1)
    #ax.plot(xtest,f(xtest))

    #plt.show()

    return xtest,f(xtest)

if __name__ == "__main__":
    USGS_folder = askdir()
    
    df = pd.read_csv('D:/Data/Ice_Pipeline_Out_5-23-23/bandInfo.csv')
    wvl = df.iloc[:,2]/1000
    _wvl,usgs_spec = get_USGS_H2OFrost(USGS_folder,wvl)
    plt.plot(_wvl,usgs_spec)




##Water Frost Spectrum
#%%
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory as askdir

def get_USGS_H2OFrost(USGS_folder):
    water = pd.read_csv(f'{USGS_folder}/splib07a_H2O-Ice_GDS136_77K_BECKa_AREF.txt')
    wavelengths = pd.read_csv(f'{USGS_folder}/splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt')
    water.columns = ['']
    wavelengths.columns = ['']

    goodIndices = np.where((wavelengths*1000>900)&(wavelengths*1000<2600))[0]

    wvl,rfl = wavelengths.iloc[goodIndices,0],water.iloc[goodIndices,0]
    #print (f'Water Length: {water.shape}, Wavelength Length: {wavelengths.shape}')
    f = interp.CubicSpline(wvl,rfl)
    xtest = np.linspace(wvl.min(),wvl.max(),59)

    #fig,ax = plt.subplots(1,1)
    #ax.plot(xtest,f(xtest))

    #plt.show()

    return wvl,f(xtest)

if __name__ == "__main__":
    USGS_folder = askdir()
    wvl,usgs_spec = get_USGS_H2OFrost(USGS_folder)
    plt.plot(wvl,usgs_spec)



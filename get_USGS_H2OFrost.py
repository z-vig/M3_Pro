##Water Frost Spectrum
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_USGS_H2OFrost():
    water = pd.read_csv(r"D:/Data/USGS_Water_Ice/splib07a_H2O-Ice_GDS136_77K_BECKa_AREF.txt")
    wavelengths = pd.read_csv(r"D:/Data/USGS_Water_Ice/splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt")
    water.columns = ['']
    wavelengths.columns = ['']

    goodIndices = np.where(water>0)[0]

    wvl,rfl = wavelengths.iloc[goodIndices,0],water.iloc[goodIndices,0]
    #print (f'Water Length: {water.shape}, Wavelength Length: {wavelengths.shape}')
    #plt.plot(wvl,rfl)
    f = interp.CubicSpline(wvl,rfl)
    xtest = np.linspace(wvl.min(),wvl.max(),59)

    #fig,ax = plt.subplots(1,1)
    #ax.plot(xtest,f(xtest))

    #plt.show()

    return f(xtest)

if __name__ == "__main__":
    usgs_spec = get_USGS_H2OFrost()
    print (np.cos(usgs_spec))



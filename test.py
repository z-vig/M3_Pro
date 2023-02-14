import spectral as sp
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# hdr = sp.envi.open(r"D:\Data\20230209T095534013597\extracted_files\hdr_files\m3g20090417t193320_v01_rfl\m3g20090417t193320_v01_rfl.hdr")
# 
# band = hdr.read_band(35)
# norm_band = (band-np.average(band))/np.std(band)
# band_01 = (band-np.min(band))
# band_01 = 256*band_01/np.max(band_01)
# 
# 
# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)
# 
# ax1.imshow(band,cmap='gray')
# ax2.imshow(band_01,cmap='gray')
# 
# =============================================================================

class MyClass:
    def __init__(self,value):
        self._value = value
        
    @property 
    def value(self):
        print ('Getting Value')
        return self._value
    
    @value.setter 
    def value(self,value):
        print ('Setting value to ' + value)
        self._value = value
    
    @value.deleter 
    def value(self):
        print ('Deleting Value')
        del self._value
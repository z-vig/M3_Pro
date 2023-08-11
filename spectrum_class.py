import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##helper function
def normalize_numpy(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return np.array(norm_arr)

class Spectrum():
    '''
    Class for each analyzed spectrum in the lab
    '''
    def __init__(self,wvl_values:np.ndarray,rfl_values:np.ndarray,meta_data:pd.DataFrame) -> None:
        self.rfl_values = rfl_values
        self.wvl_values = wvl_values
        self.meta_data = meta_data
        self.rfl_values_normalized =  normalize_numpy(self.rfl_values,0,1)
    
    @property
    def description(self):
        return self.meta_data['Description']
    
    @property
    def ice2regolith(self)->tuple:
        if pd.isnull(self.meta_data['Ice wt.%']) == False and type(self.meta_data['Ice wt.%'])!=str:
            return (round(100*self.meta_data["Ice wt.%"],1),round(100*self.meta_data["Regolith wt.%"],1))
        else:
            return (np.nan,np.nan)
    
    @property
    def notes(self):
        if pd.isnull(self.meta_data['Notes'])==False:
            return self.meta_data['Notes']
        else:
            return 'None'
        
    @property
    def test_day(self):
        return self.meta_data['Test Day Name'][1:]
    

    def add_to_plot(self,ax,color:str=None):
        '''
        Adds spectrum to current plot
        '''
        if color == None:
            ax.plot(self.wvl_values,self.rfl_values,label=f'{self.description},{self.ice2regolith}')
        else:
            ax.plot(self.wvl_values,self.rfl_values,label=f'{self.description},{self.ice2regolith}',color=color)
        ax.legend()
    
    def add_to_plot_normalized(self,ax,color:str=None,mean_modifier:float=None,label:str=None):
        '''
        Adds the normalized[0-1] spectrum to current plot
        '''
        if color == None and mean_modifier == None and label == None:
            ax.plot(self.wvl_values,self.rfl_values_normalized,label=f'{self.description},{self.ice2regolith}')
        elif color != None and mean_modifier != None and label == None:
            ax.plot(self.wvl_values,self.rfl_values_normalized+mean_modifier,\
                    label=f'{self.description},{self.ice2regolith}',color=color)
        elif color != None and mean_modifier != None and label != None:
            ax.plot(self.wvl_values,self.rfl_values_normalized+mean_modifier,\
                label=label,color=color)
        else:
            ax.plot(self.wvl_values,self.rfl_values_normalized,label=f'{self.description},{self.ice2regolith}',color=color)
        

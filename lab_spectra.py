'''
Script for pulling spectrum from laboratory data to be used in further image analysis
'''

#Importing all the needed modules and defining helper functions
import spectrum_class as spc
from importlib import reload
reload(spc)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_absorption_minima import find_min
from spectrum_averaging import spec_avg

def find_chars(string,character):
    return [n for n,i in enumerate(string) if i == character]

def normalize_numpy(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return np.array(norm_arr)

def get_spectrum(spectral_data_folder:str,data_acquisition_file:str,**kwargs)->np.ndarray:
    ##Loading all the .txt files into dataframes, and then into a dictionary with keys being test days and values being dataframes
    all_txt_file_paths:list = os.listdir(spectral_data_folder)
    all_txt_file_paths = [all_txt_file_paths[i] for i in [1,2,3,0]]
    df_list:list = [pd.read_csv(os.path.join(spectral_data_folder,i)) for i in all_txt_file_paths]


    test_name_list = [[j[find_chars(j,'_')[1]+1:-8] for j in list(i.columns[1:])] for i in df_list]
    test_number_list = [[j[-5:] for j in i] for i in test_name_list]
    test_day_name_list = [i[:find_chars(i,'.')[0]] for i in all_txt_file_paths]

    for df in df_list:
        df.set_index('Wavelength',inplace=True)
        
    for n,i in enumerate(df_list):
        i.columns = test_number_list[n]

    df_dict = {i:j for i,j in zip(test_day_name_list,df_list)}

    ##Obtaining metadata from the Spectral Data Acquisition spreadsheet
    meta_df = pd.read_excel(data_acquisition_file,header=None,sheet_name='Sheet2')
    row_bool = [True if type(i)==str and i.find('Testing Session:')>-1 else False for i in meta_df.iloc[:,0]]
    col_bool = [True if type(i)==str else False for i in meta_df.iloc[0,:]]

    daily_info = meta_df.iloc[row_bool,col_bool]
    daily_info.columns = ['Testing Session','User(s)','Date']

    def replace_colons(x):
        return x[find_chars(x,':')[0]+1:]

    daily_info = daily_info.applymap(replace_colons)

    all_test_meta = meta_df.iloc[1:,:]
    all_test_meta.columns = all_test_meta.iloc[0,:] 

    all_test_meta = all_test_meta[[False if type(i)==str else True for i in all_test_meta.iloc[:,0]]]

    all_test_meta = all_test_meta[~pd.isnull(all_test_meta.iloc[:,1])]

    n=0
    test_per_day = []
    for i in all_test_meta['Measurement Number']:
        if i!=0:
            n+=1
        else:
            test_per_day.append(n)
            n=1
    test_per_day.append(n)
    del(test_per_day[0])

    test_day_name_list = daily_info.iloc[:,0].tolist()

    test_day_name_col = ([k for i,j in zip(test_day_name_list,test_per_day) for k in [i]*j])
    all_test_meta['Test Day Name'] = test_day_name_col
    cols = all_test_meta.columns.tolist()
    cols = [cols[-1]]+cols[:-1]
    all_test_meta = all_test_meta[cols]


    all_test_reduced_df = all_test_meta[['Test Day Name','Measurement Number','Description','Regolith Mass (g)','Ice Mass (g)',\
                                    'Regolith wt.%','Ice wt.%','Notes']]
    
    ##Eliminating bad analyses and creating spectrum objects
    wvl_array = list(df_dict.values())[0].index
    good_spectra_array = np.zeros((len(wvl_array),0))
    n=0
    for df in df_dict.values():
        for column_count,col in enumerate(df):
            if int(col) == all_test_reduced_df['Measurement Number'].iloc[n]:
                good_spectra_array = np.concatenate([good_spectra_array,np.array(df.iloc[:,column_count])[:,np.newaxis]],axis=1)
            else:
                continue
            n+=1

    spec_obj_list = []
    for spec_index in range(0,good_spectra_array.shape[1]):
        meta = all_test_reduced_df.iloc[spec_index,:]
        spec_obj = spc.Spectrum(wvl_array,good_spectra_array[:,spec_index],meta)
        spec_obj_list.append(spec_obj)
        
    true_dict = {i:j for i,j in zip(kwargs.keys(),kwargs.values()) if j==True}
    for i in true_dict.keys():
        if i == 'Description':
            print (np.unique([obj.description for obj in spec_obj_list]))
        if i == 'Ice Percentage':
            print (np.unique([obj.ice2regolith[0] for obj in spec_obj_list]))
        if i == 'Test Day':
            print (np.unique([obj.test_day for obj in spec_obj_list]))
        if i == 'Notes':
            print (np.unique([obj.notes for obj in spec_obj_list]))
                
    fig,ax=plt.subplots(1,1)
    description_match = set()
    ice_pct_match = set()
    test_day_match = set()
    notes_match = set()
    
    for i in spec_obj_list:
        if i.description == kwargs.get('Description'):
            description_match.add(i)
        if i.ice2regolith[0] == kwargs.get('Ice Percentage'):
            ice_pct_match.add(i)
        if i.test_day == kwargs.get('Test Day'):
            test_day_match.add(i)
        if i.notes == kwargs.get('Notes'):
            notes_match.add(i)
    
    #print (len(description_match),len(ice_pct_match),len(test_day_match),len(notes_match))
    for i in ice_pct_match:
        print (i.description,i.ice2regolith,i.test_day)
    
    intersection_list = [i for i in [description_match,test_day_match,notes_match] if len(i)>0]
    print (intersection_list)
    search_results = ice_pct_match.intersection(*intersection_list)
    spectrum_list = [i.rfl_values for i in list(search_results)]
    wvl_list = [i.wvl_values for i in list(search_results)]

    for i in search_results:
        i.add_to_plot(ax)

    return spectrum_list,wvl_list

if __name__ == "__main__":
    spectral_analysis_folder = 'C:/Users/zvig/OneDrive - University of Iowa/Desktop/Python Code/spectral_analysis'

    search_criteria = {
        'Description':'LMS-1D'
    }

    spectrum_list = get_spectrum(os.path.join(spectral_analysis_folder,'Spectral_Data'),\
                 os.path.join(spectral_analysis_folder,'Spectral Data Acquisition.xlsx'),**search_criteria)
    
    print (spectrum_list)
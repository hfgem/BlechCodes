#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:15:54 2025

@author: Hannah Germaine

Cross-Animal Analysis Test Support: Use to test updates to functions and debug
"""

#%% Compare Multiday Support

import os
import numpy as np
from tkinter.filedialog import askdirectory
from functions.compare_multiday_analysis import run_compare_multiday_analysis
from functions.compare_conditions_funcs import int_input, bool_input

# Grab current directory and data directory / metadata
script_path = os.path.realpath(__file__)
blechcodes_dir = os.path.dirname(script_path)

multiday_data_dict = dict()
save_dir = ''

# _____Prompt user if they'd like to use previously stored correlation data_____
print("If you previously started an analysis, you may have a multiday_data_dict.npy file in the analysis folder.")
bool_val = bool_input(
    "Do you have a file stored you'd like to continue analyzing [y/n]? ")
if bool_val == 'y':
    print('Please select the storage folder.')
    save_dir = askdirectory()
    try:
        multiday_data_dict = np.load(os.path.join(save_dir,'multiday_data_dict.npy'),allow_pickle=True).item()
    except:
        print("Multiday data dict not found in given save folder. Aborting.")
        quit()
else:
    # _____Prompt user for the number of datasets needed in the analysis_____
    print("Conditions include across days and across animals (the number of separate spike sorted datasets).")
    num_cond = int_input(
        "How many conditions-worth of correlation data do you wish to import for this comparative analysis (integer value)? ")
    if num_cond >= 1:
        print("Multiple file import selected.")
    else:
        print("Single file import selected.")

    # _____Pull all data into a dictionary_____
    multiday_data_dict = dict()
    for nc in range(num_cond):
        # _____Get the directory of the hdf5 file_____
        print("Please select the folder where the data # " +
              str(nc+1) + " is stored.")
        data_dir = askdirectory()
        subfolders = os.listdir(data_dir)
        corr_exists = 0
        for sf in subfolders:
            if sf == 'Correlations':
                corr_exists = 1
             
        for corr_attempts in range(3):
            while corr_exists == 0:
                print("Try again - directory selected does not contain expected subfolders.")
                data_dir = askdirectory()
                subfolders = os.listdir(data_dir)
                corr_exists = 0
                for sf in subfolders:
                    if sf == 'Correlations':
                        corr_exists = 1
        
        if corr_exists == 0:
            print("Number of folder selection attempts exceeded. Program quitting.")
            quit()
            
        # Grab colloquial name
        print("Give a more colloquial name to the dataset.")
        data_name = data_dir.split('/')[-2]

        multiday_data_dict[data_name] = dict()
        multiday_data_dict[data_name]['data_dir'] = data_dir
        
        del data_dir, data_name
    del nc
    
    print('Please select the storage folder for comparative multiday analyses.')
    save_dir = askdirectory()
    np.save(os.path.join(save_dir,'multiday_data_dict.npy'),multiday_data_dict,allow_pickle=True)

#%% imports

import os
import warnings
import pickle
import numpy as np
from tkinter.filedialog import askdirectory

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

warnings.filterwarnings("ignore")

#%% gather_corr_data()

corr_dict_path = os.path.join(save_dir,'corr_data_dict.npy')
try:
    corr_dict = np.load(corr_dict_path,allow_pickle=True).item()
except:
    corr_dict = dict()
    data_names = list(multiday_data_dict.keys())
    for nc_i, dn in enumerate(data_names):
        corr_dict[dn] = dict()
        data_dir = multiday_data_dict[dn]['data_dir']
        corr_dir = os.path.join(data_dir,'Correlations')
        corr_types = os.listdir(corr_dir)
        for ct in corr_types:
            corr_dict[dn][ct] = dict()
            corr_type_files = os.listdir(os.path.join(corr_dir,ct))
            for f in corr_type_files:
                if f.split('.')[-1] == 'npy':
                    f_name = f.split('.')[0]
                    if f_name == 'all_taste_names':
                        taste_names = np.load(os.path.join(corr_dir,ct,f),allow_pickle=True)
                        taste_name_list = [str(tn) for tn in taste_names]
                        corr_dict[dn][ct]['tastes'] = taste_name_list
                    else:
                        seg_name = f_name.split('_')[0]
                        corr_dict_keys = list(corr_dict[dn][ct].keys())
                        if len(np.where(np.array(corr_dict_keys) == seg_name)[0]) == 0: #Segment not stored yet
                            corr_dict[dn][ct][seg_name] = dict()
                            
                        if f_name.split('_')[-1] == 'dict': #Dictionary of all correlation values
                            f_data = np.load(os.path.join(corr_dir,ct,f),allow_pickle=True).item()
                            corr_dict[dn][ct][seg_name]['all'] = dict()
                            num_tastes = len(f_data)
                            for nt_i in range(num_tastes):
                                taste_name = f_data[nt_i]['name']
                                num_cp = len(f_data[nt_i]['data'])
                                num_points = len(f_data[nt_i]['data'][0])
                                data_concat = np.zeros((num_cp,num_points))
                                for cp_i in range(num_cp):
                                    data_concat[cp_i,:] = np.array(f_data[nt_i]['data'][cp_i])
                                corr_dict[dn][ct][seg_name]['all'][taste_name] = dict()
                                corr_dict[dn][ct][seg_name]['all'][taste_name]['data'] = data_concat
                                try:
                                    corr_dict[dn][ct][seg_name]['all'][taste_name]['num_dev'] = f_data[nt_i]['num_dev']
                                    corr_dict[dn][ct][seg_name]['all'][taste_name]['taste_num_deliv'] = f_data[nt_i]['taste_num_deliv']
                                except:
                                    skip_val = 1 #Place holder skip
                        else: #best correlations file
                            f_data = np.load(os.path.join(corr_dir,ct,f),allow_pickle=True)
                            corr_dict[dn][ct][seg_name]['best'] = f_data
    np.save(corr_dict_path,corr_dict,allow_pickle=True)   
    
#%% find_corr_groupings()

num_datasets = len(corr_dict)
unique_given_names = list(corr_dict.keys())
#Pull unique correlation analysis names
unique_corr_names = []
for name in unique_given_names:
    unique_corr_names.extend(list(corr_dict[name].keys()))
unique_corr_indices = np.sort(
    np.unique(unique_corr_names, return_index=True)[1])
unique_corr_names = [unique_corr_names[i] for i in unique_corr_indices]
#Pull unique segment and taste names and max cp
unique_segment_names = []
unique_taste_names = []
max_cp = 0
for name in unique_given_names:
    for corr_name in unique_corr_names:
        seg_names = list(corr_dict[name][corr_name].keys())
        taste_names = corr_dict[name][corr_name]['tastes']
        nacl_ind = [i for i in range(len(taste_names)) if taste_names[i] == 'NaCl_1']
        if len(nacl_ind) > 0: #Stupid on my end - rename so they're all salt_1
            taste_names[nacl_ind[0]] = 'salt_1'
            corr_dict[name][corr_name]['tastes'] = list(taste_names)
            np.save(corr_dict_path,corr_dict,allow_pickle=True)
        unique_taste_names.extend(list(taste_names))
        for s_n in seg_names:
            if type(corr_dict[name][corr_name][s_n]) == dict:
                unique_segment_names.extend([s_n])
            try:
                num_cp, _ = np.shape(corr_dict[name][corr_name][s_n]['all'][taste_names[0]]['data'])
                if num_cp > max_cp:
                    max_cp = num_cp
            except:
                error = "Unable to grab changepoint count."
unique_seg_indices = np.sort(
    np.unique(unique_segment_names, return_index=True)[1])
unique_segment_names = [unique_segment_names[i] for i in unique_seg_indices]
unique_taste_indices = np.sort(
    np.unique(unique_taste_names, return_index=True)[1])
unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]

    
#%% run_corr_analysis()

import functions.compare_multiday_funcs as cmf

cmf.compare_corr_data(corr_dict, multiday_data_dict, unique_given_names,
                      unique_corr_names, unique_segment_names, unique_taste_names, 
                      max_cp, save_dir)

#%% gather_decode_data()

verbose = False

decode_dict_path = os.path.join(save_dir,'decode_data_dict.npy')
try:
    decode_dict = np.load(decode_dict_path,allow_pickle=True).item()
except:
    decode_dict = dict()
    data_names = list(multiday_data_dict.keys())
    for nc_i, dn in enumerate(data_names):
        decode_dict[dn] = dict()
        data_dir = multiday_data_dict[dn]['data_dir']
        decode_dir = os.path.join(data_dir,'Decodes')
        decode_types = os.listdir(decode_dir)
        for dt in decode_types:
            decode_dict[dn][dt] = dict()
            decode_type_files = os.listdir(os.path.join(decode_dir,dt))
            for f in decode_type_files:
                if f.split('.')[-1] == 'npy':
                    f_name = f.split('.')[0]
                    name_components = f_name.split('_')
                    #Check if dict has a segment storage started yet and make if not
                    seg_name = name_components[0]
                    decode_dict_keys = list(decode_dict[dn][dt].keys())
                    if len(np.where(np.array(decode_dict_keys) == seg_name)[0]) == 0: #Segment not stored yet
                        decode_dict[dn][dt][seg_name] = dict()
                    #Create storage for the type of decode
                    decode_type = ('_').join(name_components[-2:])
                    decode_dict[dn][dt][seg_name][decode_type] = \
                        np.load(os.path.join(decode_dir,dt,f),allow_pickle=True)
            #Add storage of taste order info
            corr_type_keys = corr_dict[dn].keys()
            for ctk in corr_type_keys:
                try:
                    decode_dict[dn][dt]['tastes'] = corr_dict[dn][ctk]['tastes']
                except:
                    if verbose == True:
                        error = 'No taste info for animal ' + dn + ' corr key ' + str(ctk)
            
    np.save(decode_dict_path,decode_dict,allow_pickle=True)

#%% find_decode_groupings():
    
num_datasets = len(decode_dict)
unique_given_names = list(decode_dict.keys())
#Pull unique decode analysis names
unique_decode_names = []
for name in unique_given_names:
    unique_decode_names.extend(list(decode_dict[name].keys()))
unique_decode_indices = np.sort(
    np.unique(unique_decode_names, return_index=True)[1])
unique_decode_names = [unique_decode_names[i] for i in unique_decode_indices]
    
#%% run_decode_analysis()

import functions.compare_multiday_funcs as cmf

cmf.compare_decode_data(decode_dict, multiday_data_dict, unique_given_names,
                       unique_decode_names, unique_segment_names, 
                       unique_taste_names, max_cp, save_dir)

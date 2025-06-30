#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:03:51 2024

@author: Hannah Germaine

Cross-Animal Analysis Test Support: Use to test updates to functions and debug
"""

#%% Compare Conditions Support

import os
import easygui
import numpy as np
from utils.replay_utils import import_metadata
from utils.data_utils import import_data
from functions.compare_conditions_analysis import run_compare_conditions_analysis
from functions.compare_conditions_funcs import int_input, bool_input
import functions.hdf5_handling as hf5

# Grab current directory and data directory / metadata
script_path = os.path.realpath(__file__)
blechcodes_dir = os.path.dirname(script_path)

all_data_dict = dict()
save_dir = ''

# _____Prompt user if they'd like to use previously stored correlation data_____
print("If you previously started an analysis, you may have a all_data_dict.npy file in the analysis folder.")
bool_val = bool_input(
    "Do you have a file stored you'd like to continue analyzing [y/n]? ")
if bool_val == 'y':
    save_dir = easygui.diropenbox(
        title='Please select the storage folder.')
    try:
        all_data_dict = np.load(os.path.join(save_dir,'all_data_dict.npy'),allow_pickle=True).item()
    except:
        print("All data dict not found in given save folder. Aborting.")
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
    all_data_dict = dict()
    for nc in range(num_cond):
        # _____Get the directory of the hdf5 file_____
        print("Please select the folder where the data # " +
              str(nc+1) + " is stored.")

        # _____Import relevant data_____
        metadata_handler = import_metadata([blechcodes_dir])
        try:
            dig_in_names = metadata_handler.info_dict['taste_params']['tastes']
        except:
            dig_in_names = []

        # import data from hdf5
        data_handler = import_data(
            [metadata_handler.dir_name, metadata_handler.hdf5_dir, dig_in_names])

        # repackage data from all handlers
        metadata = dict()
        for var in vars(metadata_handler):
            metadata[var] = getattr(metadata_handler, var)
        del metadata_handler

        data_dict = dict()
        for var in vars(data_handler):
            data_dict[var] = getattr(data_handler, var)
        del data_handler

        # Grab colloquial name
        print("Give a more colloquial name to the dataset.")
        data_name = data_dict['data_path'].split('/')[-2]
        given_name = input("How would you rename " + data_name + "? ")

        all_data_dict[given_name] = dict()
        all_data_dict[given_name]['data'] = data_dict
        all_data_dict[given_name]['metadata'] = metadata

        del data_dict, data_name, given_name, metadata, dig_in_names
    del nc
    
min_best_cutoff = 0.5


import os
import warnings
import easygui
import pickle
import numpy as np
import functions.hdf5_handling as hf5
from tkinter.filedialog import askdirectory

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.compare_datasets_funcs as cdf
import functions.compare_conditions_funcs as ccf

if len(save_dir) == 0:
    print("Please select a storage folder for results.")
    print('Please select the storage folder.')
    save_dir = askdirectory()
    np.save(os.path.join(save_dir,'all_data_dict.npy'),\
            all_data_dict,allow_pickle=True)

#%% import dev true x null data

try:
    dict_save_dir = os.path.join(save_dir, 'dev_null_data.npy')
    dev_null_data = np.load(dict_save_dir,allow_pickle=True).item()
    dev_null_data = dev_null_data
    if not os.path.isdir(os.path.join(save_dir,'Dev_Null')):
        os.mkdir(os.path.join(save_dir,'Dev_Null'))
    dev_null_results_dir = os.path.join(save_dir,'Dev_Null')
except:
    num_datasets = len(all_data_dict)
    dataset_names = list(all_data_dict.keys())
    dev_null_data = dict()
    for n_i in range(num_datasets):
        data_name = dataset_names[n_i]
        data_dict = all_data_dict[data_name]['data']
        metadata = all_data_dict[data_name]['metadata']
        data_save_dir = data_dict['data_path']
        
        dev_null_save_dir = os.path.join(
            data_save_dir, 'Deviations','null_x_true_deviations')
        dev_dir_files = os.listdir(dev_null_save_dir)
        dev_dict_dirs = []
        for dev_f in dev_dir_files:
            if dev_f[-4:] == '.npy':
                dev_dict_dirs.append(dev_f)
        dev_null_data[data_name] = dict()
        dev_null_data[data_name]['num_neur'] = data_dict['num_neur']
        segments_to_analyze = metadata['params_dict']['segments_to_analyze']
        dev_null_data[data_name]['segments_to_analyze'] = segments_to_analyze
        dev_null_data[data_name]['segment_names'] = data_dict['segment_names']
        segment_names_to_analyze = np.array(data_dict['segment_names'])[segments_to_analyze]
        segment_times = data_dict['segment_times']
        num_segments = len(dev_null_data[data_name]['segment_names'])
        dev_null_data[data_name]['segment_times_reshaped'] = [
            [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
        dig_in_names = data_dict['dig_in_names']
        dev_null_data[data_name]['dig_in_names'] = dig_in_names
        dev_null_data[data_name]['dev_null'] = dict()
        for stat_i in range(len(dev_dict_dirs)):
            stat_dir_name = dev_dict_dirs[stat_i]
            null_name = stat_dir_name.split('.')[0]
            result_dir = os.path.join(dev_null_save_dir, stat_dir_name)
            result_dict = np.load(result_dir,allow_pickle=True).item()
            result_keys = list(result_dict.keys())
            dev_null_data[data_name]['dev_null'][null_name] = dict()
            for s_i, s_name in enumerate(segment_names_to_analyze):
                dev_null_data[data_name]['dev_null'][null_name][s_name] = dict()
                for rk_i, rk in enumerate(result_keys):
                    if rk[:len(s_name)] == s_name:
                        rk_type = rk.split('_')[1]
                        dev_null_data[data_name]['dev_null'][null_name][s_name][rk_type] = \
                            result_dict[rk]
                
    dev_null_data = dev_null_data
    dict_save_dir = os.path.join(save_dir, 'dev_null_data.npy')
    np.save(dict_save_dir,dev_null_data,allow_pickle=True)
    # _____Analysis Storage Directory_____
    if not os.path.isdir(os.path.join(save_dir,'Dev_Null')):
        os.mkdir(os.path.join(save_dir,'Dev_Null'))
    dev_null_results_dir = os.path.join(save_dir,'Dev_Null')
    
#%% find_dev_null_groupings()

unique_given_names = list(dev_null_data.keys())
unique_given_indices = np.sort(
    np.unique(unique_given_names, return_index=True)[1])
unique_given_names = [unique_given_names[i]
                      for i in unique_given_indices]
unique_dev_null_names = []
for name in unique_given_names:
    unique_dev_null_names.extend(list(dev_null_data[name]['dev_null'].keys()))
unique_dev_null_names = np.array(unique_dev_null_names)
unique_dev_null_indices = np.sort(
    np.unique(unique_dev_null_names, return_index=True)[1])
unique_dev_null_names = [unique_dev_null_names[i] for i in unique_dev_null_indices]
unique_segment_names = []
for name in unique_given_names:
    for dev_null_name in unique_dev_null_names:
        try:
            seg_names = list(
                dev_null_data[name]['dev_null'][dev_null_name].keys())
            unique_segment_names.extend(seg_names)
        except:
            print(name + " does not have correlation data for " + dev_null_name)
unique_segment_indices = np.sort(
    np.unique(unique_segment_names, return_index=True)[1])
unique_segment_names = [unique_segment_names[i]
                        for i in unique_segment_indices]

#%% plot_dev_null_results()

import functions.cross_animal_dev_null_plots as cadnp

num_cond = len(dev_null_data)
results_dir = dev_null_results_dir

print("Beginning Plots.")
if num_cond > 1:
    cadnp.cross_dataset_dev_null_plots(dev_null_data, unique_given_names, 
                                     unique_dev_null_names, unique_segment_names, 
                                     results_dir)
else:
    print("Not enough animals for cross-animal dev stat plots.")

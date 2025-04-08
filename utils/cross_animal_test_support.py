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
        
#%% import_rate_corr_data()

num_datasets = len(all_data_dict)
dataset_names = list(all_data_dict.keys())
rate_corr_data = dict()
for n_i in range(num_datasets):
    data_name = dataset_names[n_i]
    data_dict = all_data_dict[data_name]['data']
    metadata = all_data_dict[data_name]['metadata']
    data_save_dir = data_dict['data_path']
    rate_corr_save_dir = os.path.join(data_save_dir,'Sliding_Correlations')
    num_corr_types = os.listdir(rate_corr_save_dir)
    rate_corr_data[data_name] = dict()
    rate_corr_data[data_name]['num_neur'] = data_dict['num_neur']
    segments_to_analyze = metadata['params_dict']['segments_to_analyze']
    rate_corr_data[data_name]['segments_to_analyze'] = segments_to_analyze
    rate_corr_data[data_name]['segment_names'] = data_dict['segment_names']
    segment_times = data_dict['segment_times']
    num_segments = len(rate_corr_data[data_name]['segment_names'])
    rate_corr_data[data_name]['segment_times_reshaped'] = [
        [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
    dig_in_names = data_dict['dig_in_names']
    rate_corr_data[data_name]['dig_in_names'] = dig_in_names
    seg_names_to_analyze = np.array(rate_corr_data[data_name]['segment_names'])[segments_to_analyze]
    rate_corr_data[data_name]['rate_corr_data'] = dict()
    for nct in range(len(num_corr_types)):
        corr_type = num_corr_types[nct]
        if corr_type[0] != '.': #Ignore '.DS_Store'
            rate_corr_data[data_name]['rate_corr_data'][corr_type] = dict()
            corr_dir = os.path.join(rate_corr_save_dir,corr_type)
            try:
                rate_corr_data[data_name]['rate_corr_data'][corr_type] = np.load(os.path.join(corr_dir,'popfr_corr_storage.npy'), allow_pickle=True).item()
            except:
                print("No population fr x taste correlation dictionary found for " + data_name + " corr " + corr_type)
            #This data is organized by [seg_name][bin_size] gives the result array
rate_corr_data = rate_corr_data
np.save(os.path.join(save_dir, 'rate_corr_data.npy'),rate_corr_data,allow_pickle=True)
# Save the combined dataset somewhere...
# _____Analysis Storage Directory_____
if not os.path.isdir(os.path.join(save_dir,'Sliding_Correlation_Comparison')):
    os.mkdir(os.path.join(save_dir,'Sliding_Correlation_Comparison'))
rate_corr_results_dir = os.path.join(save_dir,'Sliding_Correlation_Comparison')

#%% find_rate_corr_groupings()

rate_corr_data = rate_corr_data
unique_given_names = list(rate_corr_data.keys())
unique_given_indices = np.sort(
    np.unique(unique_given_names, return_index=True)[1])
unique_given_names = [unique_given_names[i]
                      for i in unique_given_indices]
unique_corr_types = []
for name in unique_given_names:
    unique_corr_types.extend(list(rate_corr_data[name]['rate_corr_data'].keys()))
unique_corr_types = np.array(unique_corr_types)
unique_corr_indices = np.sort(
    np.unique(unique_corr_types, return_index=True)[1])
unique_corr_types = [unique_corr_types[i] for i in unique_corr_indices]
unique_segment_names = []
unique_taste_names = []
for name in unique_given_names:
    for corr_name in unique_corr_types:
        try:
            segment_names = list(rate_corr_data[name]['rate_corr_data'][corr_name].keys())
            unique_segment_names.extend(segment_names)
            for seg_name in segment_names:
                taste_names = list(rate_corr_data[name]['rate_corr_data'][corr_name][seg_name].keys())
                unique_taste_names.extend(taste_names)
        except:
            print(name + " does not have data for " + corr_name)
unique_segment_indices = np.sort(
    np.unique(unique_segment_names, return_index=True)[1])
unique_segment_names = [unique_segment_names[i]
                        for i in unique_segment_indices]
unique_taste_indices = np.sort(
    np.unique(unique_taste_names, return_index=True)[1])
unique_taste_names = [unique_taste_names[i]
                      for i in unique_taste_indices]

#%% plot_rate_corr_results()
num_cond = len(rate_corr_data)
results_dir = rate_corr_results_dir
cdf.cross_dataset_pop_rate_taste_corr_plots(rate_corr_data, unique_given_names, 
                                            unique_corr_types, unique_segment_names, 
                                            unique_taste_names, results_dir)
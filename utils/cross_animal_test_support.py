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

#%% import taste response data
from functions.data_description_analysis import run_data_description_analysis
import functions.dependent_decoding_funcs as ddf
    
try:
    dict_save_dir = os.path.join(save_dir, 'taste_resp_data.npy')
    taste_data = np.load(dict_save_dir,allow_pickle=True).item()
    taste_data = taste_data
    if not os.path.isdir(os.path.join(save_dir,'Taste_Responses')):
        os.mkdir(os.path.join(save_dir,'Taste_Responses'))
    taste_results_dir = os.path.join(save_dir,'Taste_Responses')
except:
    num_datasets = len(all_data_dict)
    dataset_names = list(all_data_dict.keys())
    taste_data = dict()
    for n_i in range(num_datasets):
        data_name = dataset_names[n_i]
        data_dict = all_data_dict[data_name]['data']
        metadata = all_data_dict[data_name]['metadata']
        data_description_results = run_data_description_analysis([metadata, data_dict])
        data_dict = data_description_results.data_dict
        hdf5_dir = metadata['hdf5_dir']
        data_save_dir = data_dict['data_path']
        taste_data[data_name] = dict()
        num_neur = data_dict['num_neur']
        taste_data[data_name]['num_neur'] = num_neur
        segments_to_analyze = metadata['params_dict']['segments_to_analyze']
        taste_data[data_name]['segments_to_analyze'] = segments_to_analyze
        segment_names = data_dict['segment_names']
        taste_data[data_name]['segment_names'] = segment_names
        segment_times = data_dict['segment_times']
        num_segments = len(segment_names)
        taste_data[data_name]['segment_times_reshaped'] = [
            [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
        dig_in_names = data_dict['dig_in_names']
        taste_data[data_name]['dig_in_names'] = dig_in_names
        tastant_spike_times = data_dict['tastant_spike_times']
        taste_data[data_name]['tastant_spike_times'] = tastant_spike_times
        segment_spike_times = data_dict['segment_spike_times']
        taste_data[data_name]['segment_spike_times'] = segment_spike_times
        start_dig_in_times = data_dict['start_dig_in_times']
        taste_data[data_name]['start_dig_in_times'] = start_dig_in_times
        taste_data[data_name]['end_dig_in_times'] = data_dict['end_dig_in_times']
        pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(hdf5_dir, \
                            'changepoint_data', 'pop_taste_cp_raster_inds')
        bayes_fr_bins = metadata['params_dict']['bayes_params']['fr_bins']
        pre_taste = metadata['params_dict']['pre_taste']
        post_taste = metadata['params_dict']['post_taste']
        pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
        post_taste_dt = np.ceil(post_taste*1000).astype('int')
        bin_time = metadata['params_dict']['bayes_params']['z_score_bin_time']
        bin_dt = np.ceil(bin_time*1000).astype('int')
        trial_start_frac = metadata['params_dict']['bayes_params']['trial_start_frac']
        tastant_fr_dist_z_pop, taste_num_deliv, max_hz_z_pop, \
            min_hz_z_pop = ddf.taste_fr_dist_zscore(num_neur, tastant_spike_times,
                                                    segment_spike_times, segment_names,
                                                    segment_times, pop_taste_cp_raster_inds,
                                                    bayes_fr_bins, start_dig_in_times, 
                                                    pre_taste_dt, post_taste_dt, 
                                                    bin_dt, trial_start_frac)
        taste_data[data_name]['tastant_fr_dist_z_pop'] = tastant_fr_dist_z_pop
        taste_data[data_name]['taste_num_deliv'] = taste_num_deliv
        taste_data[data_name]['max_hz_z_pop'] = max_hz_z_pop
        taste_data[data_name]['min_hz_z_pop'] = min_hz_z_pop
        
    taste_data = taste_data
    dict_save_dir = os.path.join(save_dir, 'taste_resp_data.npy')
    np.save(dict_save_dir,taste_data,allow_pickle=True)
    # Save the combined dataset somewhere...
    # _____Analysis Storage Directory_____
    if not os.path.isdir(os.path.join(save_dir,'Taste_Responses')):
        os.mkdir(os.path.join(save_dir,'Taste_Responses'))
    taste_results_dir = os.path.join(save_dir,'Taste_Responses')
    
#%% 

import functions.cross_animal_taste_stats as cats

taste_stats = cats.taste_stat_collection(taste_data, taste_results_dir)
np.save(os.path.join(taste_results_dir,'taste_stats.npy'),taste_stats,allow_pickle=True)

cats.plot_corr_outputs(taste_stats,taste_results_dir)
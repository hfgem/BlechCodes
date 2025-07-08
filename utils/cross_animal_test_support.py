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

import os

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import warnings
import numpy as np
import functions.hdf5_handling as hf5
from tkinter.filedialog import askdirectory
from itertools import combinations
from functions.data_description_analysis import run_data_description_analysis
import functions.compare_datasets_funcs as cdf
import functions.compare_conditions_funcs as ccf
import functions.cross_animal_seg_stats as cass
import functions.cross_animal_taste_stats as cats
import functions.cross_animal_dev_stats as cads
import functions.cross_animal_dev_null_plots as cadnp
import functions.dependent_decoding_funcs as ddf

verbose = False

try:
    dict_save_dir = os.path.join(save_dir, 'dev_split_decode_data.npy')
    dev_split_decode_data = np.load(dict_save_dir,allow_pickle=True).item()
    dev_split_decode_data = dev_split_decode_data
    if not os.path.isdir(os.path.join(save_dir,'Dev_Split_Decode')):
        os.mkdir(os.path.join(save_dir,'Dev_Split_Decode'))
    dev_split_decode_results_dir = os.path.join(save_dir,'Dev_Split_Decode')
except:
    num_datasets = len(all_data_dict)
    dataset_names = list(all_data_dict.keys())
    dev_split_decode_data = dict()
    for n_i in range(num_datasets):
        data_name = dataset_names[n_i]
        data_dict = all_data_dict[data_name]['data']
        metadata = all_data_dict[data_name]['metadata']
        dev_split_decode_data[data_name] = dict()
        dev_split_decode_data[data_name]['num_neur'] = data_dict['num_neur']
        segments_to_analyze = metadata['params_dict']['segments_to_analyze']
        dev_split_decode_data[data_name]['segments_to_analyze'] = segments_to_analyze
        dev_split_decode_data[data_name]['segment_names'] = data_dict['segment_names']
        segment_names_to_analyze = np.array(data_dict['segment_names'])[segments_to_analyze]
        segment_times = data_dict['segment_times']
        num_segments = len(dev_split_decode_data[data_name]['segment_names'])
        dev_split_decode_data[data_name]['segment_times_reshaped'] = [
            [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
        dig_in_names = data_dict['dig_in_names']
        dev_split_decode_data[data_name]['dig_in_names'] = dig_in_names
        data_save_dir = data_dict['data_path']
        dev_split_save_dir = os.path.join(
            data_save_dir, 'Deviation_Sequence_Analysis')
        #Subfolders we care about: corr_tests and decode_splits
        #First load correlation data
        try:
            group_dict = np.load(os.path.join(dev_split_save_dir,'group_dict.npy'),\
                                 allow_pickle=True).item()
            allgroups = list(group_dict.keys())
            dev_split_decode_data[data_name]['groups'] = allgroups
        except:
            dev_split_decode_data[data_name]['groups'] = []
            if verbose == True:
                print('Missing group dict for ' + data_name)
        dev_split_dir = os.path.join(dev_split_save_dir,'decode_splits')
        dev_split_decode_dir = os.path.join(dev_split_dir,'zscore_firing_rates')
        dev_split_decode_files = os.listdir(dev_split_decode_dir)
        dev_split_decode_dict_files = []
        for dev_dec_f in dev_split_decode_files:
            if dev_dec_f[-4:] == '.npy':
                dev_split_decode_dict_files.append(dev_dec_f)
        dev_split_decode_data[data_name]['decode_data'] = dict()
        for sna in segment_names_to_analyze:
            dev_split_decode_data[data_name]['decode_data'][sna] = dict()
        for stat_i, stat_filename in enumerate(dev_split_decode_dict_files):
            stat_filename_split = (stat_filename.split('.')[0]).split('_')
            stat_seg = stat_filename_split[0]
            if stat_filename_split[-1] == 'array': #Probability or argmax array
                if stat_filename_split[-2] == 'prob': #Probabilities
                    dev_split_decode_data[data_name]['decode_data'][stat_seg]['probabilities'] = \
                        np.load(os.path.join(dev_split_decode_dir,stat_filename),allow_pickle=True)
                elif stat_filename_split[-2] == 'decode': #Argmax
                    dev_split_decode_data[data_name]['decode_data'][stat_seg]['argmax'] = \
                        np.load(os.path.join(dev_split_decode_dir,stat_filename),allow_pickle=True)
            elif stat_filename_split[-1] == 'dict': #Dictionary of counts
                dev_split_decode_data[data_name]['decode_data'][stat_seg]['group_dict'] = \
                    np.load(os.path.join(dev_split_decode_dir,stat_filename),allow_pickle=True).item()
        
    dev_split_decode_data = dev_split_decode_data
    dict_save_dir = os.path.join(save_dir, 'dev_split_decode_data.npy')
    np.save(dict_save_dir,dev_split_decode_data,allow_pickle=True)
    # _____Analysis Storage Directory_____
    if not os.path.isdir(os.path.join(save_dir,'Dev_Split_Decode')):
        os.mkdir(os.path.join(save_dir,'Dev_Split_Decode'))
    dev_split_decode_results_dir = os.path.join(save_dir,'Dev_Split_Decode')

#%% find_dev_split_decode_groupings()

unique_given_names = list(dev_split_decode_data.keys())
unique_given_indices = np.sort(
    np.unique(unique_given_names, return_index=True)[1])
unique_given_names = [unique_given_names[i]
                      for i in unique_given_indices]
unique_segment_names = []
unique_taste_names = []
unique_group_names = []
unique_group_pair_names = []
for name in unique_given_names:
    group_names = list(dev_split_decode_data[name]['groups'])
    unique_group_names.extend(group_names)
    segment_names = list(dev_split_decode_data[name]['decode_data'].keys())
    unique_segment_names.extend(segment_names)
    for seg_name in segment_names:
        taste_names = list(dev_split_decode_data[name]['dig_in_names'][:-1])
        unique_taste_names.extend(taste_names)
        unique_group_pair_names.extend(list(dev_split_decode_data[name]['decode_data'][seg_name]['group_dict'].keys()))
        
unique_segment_indices = np.sort(
    np.unique(unique_segment_names, return_index=True)[1])
unique_segment_names = [unique_segment_names[i] for i in unique_segment_indices]
unique_taste_indices = np.sort(
    np.unique(unique_taste_names, return_index=True)[1])
unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
unique_group_indices = np.sort(
    np.unique(unique_group_names, return_index=True)[1])
unique_group_names = [unique_group_names[i] for i in unique_group_indices]
unique_group_pair_inds = np.sort(
    np.unique(unique_group_pair_names, return_index=True)[1])
unique_group_pair_names = [unique_group_pair_names[i] for i in unique_group_pair_inds]
non_null_groups = [ugn for ugn in unique_group_names if ugn != 'No Taste Control']
non_null_pairs = list(combinations(np.arange(len(non_null_groups)),2))    
same_pairs = [(gn_i,gn_i) for gn_i in range(len(non_null_groups))]
group_pair_dict = dict()
for nnp_1, nnp_2 in non_null_pairs:
    name_1 = unique_group_names[nnp_1]
    name_2 = unique_group_names[nnp_2]
    name_sort = np.sort([name_1,name_2])
    group_pair_dict[name_sort[0] + ',' + name_sort[1]] = []
for sp_1, sp_2 in same_pairs:
    group_pair_dict[unique_group_names[sp_1] + ',' + unique_group_names[sp_2]] = []
group_pair_dict['1+ Null'] = []
group_pair_dict_keys = list(group_pair_dict.keys())
for ugpn in unique_group_pair_names:
    ugpn_split = np.sort(ugpn.split(', '))
    ind = [i for i in range(len(group_pair_dict_keys)) if (',').join(ugpn_split) == group_pair_dict_keys[i]]
    if len(ind) == 0:
        group_pair_dict['1+ Null'].append(ugpn)
    else:
        group_pair_dict[group_pair_dict_keys[ind[0]]].append(ugpn)
    
    
    
#%% plot results

import functions.cross_animal_dev_split_stats as cadss

num_cond = len(dev_split_decode_data)
results_dir = dev_split_decode_results_dir

cadss.run_cross_animal_dev_split_decode_analyses(dev_split_decode_data, group_pair_dict,
                                               unique_given_names,unique_segment_names, 
                                               unique_taste_names,unique_group_names,
                                               unique_group_pair_names,results_dir)

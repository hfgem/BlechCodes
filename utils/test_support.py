#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:13:46 2024

@author: Hannah Germaine

Script to grab data for testing new code out of the pipeline.
"""

#Import necessary packages and functions
import os
import json
import gzip
import tqdm
import numpy as np
current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-2]) + '/'
os.chdir(blech_codes_path)
import functions.analysis_funcs as af
import functions.hdf5_handling as hf5
import functions.dependent_decoding_funcs as ddf
import functions.decoding_funcs as df
from utils.replay_utils import import_metadata, state_tracker
from utils.data_utils import import_data

# Grab current directory and data directory / metadata
script_path = os.path.realpath(__file__)
blechcodes_dir = os.path.dirname(script_path)

metadata_handler = import_metadata([blechcodes_dir])
try:
	dig_in_names = metadata_handler.info_dict['taste_params']['tastes']
except:
	dig_in_names = []

# import data from hdf5
data_handler = import_data([metadata_handler.dir_name, metadata_handler.hdf5_dir, dig_in_names])

state_handler = state_tracker([metadata_handler.dir_name])

# repackage data from all handlers
metadata = dict()
for var in vars(metadata_handler):
	metadata[var] = getattr(metadata_handler,var)
del metadata_handler

data_dict = dict()
for var in vars(data_handler):
	data_dict[var] = getattr(data_handler,var)
del data_handler

state_dict = dict()
for var in vars(state_handler):
	state_dict[var] = getattr(state_handler,var)
del state_handler


segment_spike_times = af.calc_segment_spike_times(data_dict['segment_times'],data_dict['spike_times'],data_dict['num_neur'])
tastant_spike_times = af.calc_tastant_spike_times(data_dict['segment_times'],data_dict['spike_times'],
						  data_dict['start_dig_in_times'],data_dict['end_dig_in_times'],
						  metadata['params_dict']['pre_taste'],metadata['params_dict']['post_taste'],data_dict['num_tastes'],data_dict['num_neur'])
data_dict['segment_spike_times'] = segment_spike_times
data_dict['tastant_spike_times'] = tastant_spike_times


#%% CP Dist Plots

import matplotlib.pyplot as plt
import numpy as np

colors = ['g','b','o','p','r']

hdf5_dir = metadata['hdf5_dir']

data_group_name = 'changepoint_data'
pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'pop_taste_cp_raster_inds')

for t_i in range(len(dig_in_names)):
    taste_cp_data = pop_taste_cp_raster_inds[t_i] #num deliv x num_cp + 2
    taste_cp_diff = np.diff(taste_cp_data)
    taste_cp_realigned = np.cumsum(taste_cp_diff,1)
    f = plt.figure(figsize=(5,5))
    for cp_i in range(2):
        plt.hist(taste_cp_realigned[:,cp_i],density=True,alpha=0.3,label='Changepoint ' + str(cp_i+1),color=colors[cp_i])
        dist_mean = np.nanmean(taste_cp_realigned[:,cp_i])
        dist_median = np.median(taste_cp_realigned[:,cp_i])
        plt.axvline(dist_mean,label='Mean CP ' + str(cp_i+1) + ' = ' + str(np.round(dist_mean,2)), color=colors[cp_i])
        plt.axvline(dist_median,label='Median CP ' + str(cp_i+1) + ' = ' + str(np.round(dist_median,2)), linestyle='dashed', color=colors[cp_i])
    plt.legend()
    plt.title(dig_in_names[t_i])
    plt.xlabel('Time Post Taste Delivery (ms)')
    plt.ylabel('Density')

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

import os
import warnings
import easygui
import pickle
import numpy as np

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.compare_datasets_funcs as cdf
import functions.compare_conditions_funcs as ccf

warnings.filterwarnings("ignore")

dict_save_dir = os.path.join(save_dir, 'corr_data.npy')
corr_data = np.load(dict_save_dir,allow_pickle=True).item()
if not os.path.isdir(os.path.join(save_dir,'Correlations')):
    os.mkdir(os.path.join(save_dir,'Correlations'))
corr_results_dir = os.path.join(save_dir,'Correlations')

unique_given_names = list(corr_data.keys())
unique_given_indices = np.sort(
    np.unique(unique_given_names, return_index=True)[1])
unique_given_names = [unique_given_names[i]
                      for i in unique_given_indices]
unique_corr_names = []
for name in unique_given_names:
    unique_corr_names.extend(list(corr_data[name]['corr_data'].keys()))
unique_corr_names = np.array(unique_corr_names)
unique_corr_indices = np.sort(
    np.unique(unique_corr_names, return_index=True)[1])
unique_corr_names = [unique_corr_names[i] for i in unique_corr_indices]
unique_segment_names = []
unique_taste_names = []
for name in unique_given_names:
    for corr_name in unique_corr_names:
        try:
            seg_names = list(
                corr_data[name]['corr_data'][corr_name].keys())
            unique_segment_names.extend(seg_names)
            for seg_name in seg_names:
                taste_names = list(
                    corr_data[name]['corr_data'][corr_name][seg_name].keys())
                unique_taste_names.extend(taste_names)
        except:
            print(name + " does not have correlation data for " + corr_name)
unique_segment_indices = np.sort(
    np.unique(unique_segment_names, return_index=True)[1])
unique_segment_names = [unique_segment_names[i]
                        for i in unique_segment_indices]
unique_taste_indices = np.sort(
    np.unique(unique_taste_names, return_index=True)[1])
unique_taste_names = [unique_taste_names[i]
                      for i in unique_taste_indices]

num_cond = len(corr_data)
results_dir = corr_results_dir

cross_segment_dir = os.path.join(
    results_dir, 'cross_segment_plots')
if os.path.isdir(cross_segment_dir) == False:
    os.mkdir(cross_segment_dir)
    


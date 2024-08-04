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

#%% Dev Corr Support

import os
import json
import gzip
import tqdm
import numpy as np
import functions.dev_plot_funcs as dpf
import functions.dev_funcs as df
import functions.hdf5_handling as hf5

dev_dir = metadata['dir_name'] + 'Deviations/'
hdf5_dir = metadata['hdf5_dir']
comp_dir = metadata['dir_name'] + 'dev_x_taste/'
if os.path.isdir(comp_dir) == False:
    os.mkdir(comp_dir)
corr_dir = comp_dir + 'corr/'
if os.path.isdir(corr_dir) == False:
    os.mkdir(corr_dir)
# Params/Variables
num_neur = data_dict['num_neur']
pre_taste = metadata['params_dict']['pre_taste']
post_taste = metadata['params_dict']['post_taste']
segments_to_analyze = metadata['params_dict']['segments_to_analyze']
epochs_to_analyze = metadata['params_dict']['epochs_to_analyze']
segment_names = data_dict['segment_names']
num_segments = len(segment_names)
segment_spike_times = data_dict['segment_spike_times']
segment_times = data_dict['segment_times']
segment_times_reshaped = [
    [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
# Remember this is 1 less than the number of epochs
num_cp = metadata['params_dict']['num_cp']
tastant_spike_times = data_dict['tastant_spike_times']
start_dig_in_times = data_dict['start_dig_in_times']
end_dig_in_times = data_dict['end_dig_in_times']
dig_in_names = data_dict['dig_in_names']
z_bin = metadata['params_dict']['z_bin']

print("\tNow importing calculated deviations")
segment_deviations = []
for s_i in tqdm.tqdm(segments_to_analyze):
    filepath = dev_dir + \
        segment_names[s_i] + '/deviations.json'
    with gzip.GzipFile(filepath, mode="r") as f:
        json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)
        segment_deviations.append(data)
print("\tNow pulling true deviation rasters")
num_segments = len(segments_to_analyze)
segment_spike_times_reshaped = [segment_spike_times[i]
                       for i in segments_to_analyze]
segment_times_reshaped = np.array(
    [segment_times_reshaped[i] for i in segments_to_analyze])
segment_dev_rasters, segment_dev_times, segment_dev_vec, segment_dev_vec_zscore = df.create_dev_rasters(num_segments,
                                                                                                        segment_spike_times_reshaped,
                                                                                                        segment_times_reshaped,
                                                                                                        segment_deviations, z_bin)

data_group_name = 'changepoint_data'
pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'pop_taste_cp_raster_inds')
pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
num_pt_cp = num_cp + 2
# Import discriminability data
data_group_name = 'taste_discriminability'
peak_epochs = np.squeeze(hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'peak_epochs'))
discrim_neur = np.squeeze(hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'discrim_neur'))
# Convert discriminatory neuron data into pop_taste_cp_raster_inds shape
# TODO: Test this first, then if going with this rework functions to fit instead!
num_discrim_cp = np.shape(discrim_neur)[0]
discrim_cp_raster_inds = []
for t_i in range(len(dig_in_names)):
    t_cp_vec = np.ones(
        (np.shape(pop_taste_cp_raster_inds[t_i])[0], num_discrim_cp))
    t_cp_vec = (peak_epochs[:num_pt_cp] +
                int(pre_taste*1000))*t_cp_vec
    discrim_cp_raster_inds.append(t_cp_vec)
num_discrim_cp = len(peak_epochs)

current_corr_dir = corr_dir + 'all_neur/'
if os.path.isdir(current_corr_dir) == False:
    os.mkdir(current_corr_dir)
#neuron_keep_indices = np.ones((num_neur,num_cp+1))
neuron_keep_indices = np.ones(np.shape(discrim_neur))
# Calculate correlations
df.calculate_vec_correlations(num_neur, segment_dev_vec, tastant_spike_times,
                              start_dig_in_times, end_dig_in_times, segment_names,
                              dig_in_names, pre_taste, post_taste, pop_taste_cp_raster_inds,
                              current_corr_dir, neuron_keep_indices, segments_to_analyze)  # For all neurons in dataset
# Calculate significant events
sig_dev, sig_dev_counts = df.calculate_significant_dev(segment_dev_times, 
                                                       segment_times, dig_in_names,
                                                       segment_names, current_corr_dir,
                                                       segments_to_analyze)

best_dir = current_corr_dir + 'best/'
if os.path.isdir(best_dir) == False:
    os.mkdir(best_dir)

corr_data_dir = current_corr_dir
save_dir = best_dir

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

num_datasets = len(all_data_dict)
dataset_names = list(all_data_dict.keys())

cp_data = dict()
for n_i in range(num_datasets):
    data_name = dataset_names[n_i]
    data_dict = all_data_dict[data_name]['data']
    metadata = all_data_dict[data_name]['metadata']
    hdf5_dir = metadata['hdf5_dir']
    cp_data[data_name] = dict()
    cp_data[data_name]['num_neur'] = data_dict['num_neur']
    segments_to_analyze = metadata['params_dict']['segments_to_analyze']
    cp_data[data_name]['segments_to_analyze'] = segments_to_analyze
    cp_data[data_name]['segment_names'] = data_dict['segment_names']
    segment_times = data_dict['segment_times']
    num_segments = len(cp_data[data_name]['segment_names'])
    cp_data[data_name]['segment_times_reshaped'] = [
        [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
    dig_in_names = data_dict['dig_in_names']
    cp_data[data_name]['dig_in_names'] = dig_in_names
    cp_data[data_name]['cp_data'] = dict()
    data_group_name = 'changepoint_data'
    pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
        hdf5_dir, data_group_name, 'pop_taste_cp_raster_inds')
    for t_i in range(len(dig_in_names)):
        taste_cp_data = pop_taste_cp_raster_inds[t_i] #num deliv x num_cp + 2
        cp_data[data_name]['cp_data'][dig_in_names[t_i]] = taste_cp_data
           
np.save(os.path.join(save_dir, 'cp_data.npy'),cp_data,allow_pickle=True)
# Save the combined dataset somewhere...
# _____Analysis Storage Directory_____
if not os.path.isdir(os.path.join(save_dir,'Changepoint_Statistics')):
    os.mkdir(os.path.join(save_dir,'Changepoint_Statistics'))
cp_results_dir = os.path.join(save_dir,'Changepoint_Statistics')

unique_given_names = list(cp_data.keys())
unique_given_indices = np.sort(
    np.unique(unique_given_names, return_index=True)[1])
unique_given_names = [unique_given_names[i]
                      for i in unique_given_indices]
unique_taste_names = np.array([list(cp_data[name]['cp_data'].keys(
)) for name in unique_given_names]).flatten()  # How many types of segment analyses
unique_taste_indices = np.sort(
    np.unique(unique_taste_names, return_index=True)[1])
unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
max_cp_counts = 0
for name in unique_given_names:
    for taste_name in unique_taste_names:
        try:
            taste_cp_data = cp_data[name]['cp_data'][taste_name]
            num_cp = np.shape(taste_cp_data)[1] - 2
            if num_cp > max_cp_counts:
                max_cp_counts = num_cp
        except:
            print(name + " does not have data for " + taste_name)

results_dir = cp_results_dir

cdf.cross_dataset_cp_plots(cp_data, unique_given_names, 
                           unique_taste_names, max_cp_counts,
                           results_dir)


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

#%% 

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

#%% gather_corr_data()

num_datasets = len(all_data_dict)
dataset_names = list(all_data_dict.keys())
corr_data = dict()
for n_i in range(num_datasets):
    data_name = dataset_names[n_i]
    data_dict = all_data_dict[data_name]['data']
    metadata = all_data_dict[data_name]['metadata']
    data_save_dir = data_dict['data_path']
    dev_corr_save_dir = os.path.join(
        data_save_dir, 'dev_x_taste', 'corr')
    num_corr_types = os.listdir(dev_corr_save_dir)
    corr_data[data_name] = dict()
    corr_data[data_name]['num_neur'] = data_dict['num_neur']
    segments_to_analyze = metadata['params_dict']['segments_to_analyze']
    corr_data[data_name]['segments_to_analyze'] = segments_to_analyze
    corr_data[data_name]['segment_names'] = data_dict['segment_names']
    segment_times = data_dict['segment_times']
    num_segments = len(corr_data[data_name]['segment_names'])
    corr_data[data_name]['segment_times_reshaped'] = [
        [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
    dig_in_names = data_dict['dig_in_names']
    corr_data[data_name]['dig_in_names'] = dig_in_names
    corr_data[data_name]['corr_data'] = dict()
    for nct_i in range(len(num_corr_types)):
        nct = num_corr_types[nct_i]
        if nct[0] != '.':
            result_dir = os.path.join(dev_corr_save_dir, nct)
            corr_data[data_name]['corr_data'][nct] = dict()
            for s_i in segments_to_analyze:
                seg_name = corr_data[data_name]['segment_names'][s_i]
                corr_data[data_name]['corr_data'][nct][seg_name] = dict()
                filename_best_corr = os.path.join(result_dir,seg_name + '_best_taste_epoch_array.npy')
                best_data = np.load(filename_best_corr)
                corr_data[data_name]['corr_data'][nct][seg_name]['best'] = best_data
                for t_i in range(len(dig_in_names)):
                    taste_name = dig_in_names[t_i]
                    corr_data[data_name]['corr_data'][nct][seg_name][taste_name] = dict(
                    )
                    try:
                        filename_corr_pop_vec = os.path.join(
                            result_dir, seg_name + '_' + taste_name + '_pop_vec.npy')
                        data = np.load(filename_corr_pop_vec)
                        corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['data'] = data
                        num_dev, num_deliv, num_cp = np.shape(data)
                        corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['num_dev'] = num_dev
                        corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['num_deliv'] = num_deliv
                        corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['num_cp'] = num_cp
                    except:
                        print("No data in directory " + result_dir)
dict_save_dir = os.path.join(save_dir, 'corr_data.npy')
np.save(dict_save_dir,corr_data,allow_pickle=True)
# Save the combined dataset somewhere...
# _____Analysis Storage Directory_____
if not os.path.isdir(os.path.join(save_dir,'Correlations')):
    os.mkdir(os.path.join(save_dir,'Correlations'))
corr_results_dir = os.path.join(save_dir,'Correlations')

#%% find_corr_groupings()

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
                taste_names = list(np.setdiff1d(list(
                    corr_data[name]['corr_data'][corr_name][seg_name].keys()),['best']))
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

#%% plot_corr_results()

num_cond = len(corr_data)
results_dir = corr_results_dir

# ____Deviation Event Frequencies____
dev_freq_dir = os.path.join(results_dir, 'dev_frequency_plots')
if os.path.isdir(dev_freq_dir) == False:
    os.mkdir(dev_freq_dir)
print("\tCalculating Cross-Segment Deviation Frequencies")
cdf.cross_dataset_dev_freq(corr_data, unique_given_names,
                                 unique_corr_names, unique_segment_names,
                                 unique_taste_names, dev_freq_dir)
# ____Correlation Distributions____
cross_segment_dir = os.path.join(
    results_dir, 'cross_segment_plots')
if os.path.isdir(cross_segment_dir) == False:
    os.mkdir(cross_segment_dir)
print("\tComparing Segments")
cdf.cross_segment_diffs(corr_data, cross_segment_dir, unique_given_names,
                        unique_corr_names, unique_segment_names, unique_taste_names)
cdf.combined_corr_by_segment_dist(corr_data, cross_segment_dir, unique_given_names, 
                                  unique_corr_names,unique_segment_names, unique_taste_names)
cross_taste_dir = os.path.join(results_dir, 'cross_taste_plots')
if os.path.isdir(cross_taste_dir) == False:
    os.mkdir(cross_taste_dir)
print("\tComparing Tastes")
cdf.cross_taste_diffs(corr_data, cross_taste_dir, unique_given_names,
                      unique_corr_names, unique_segment_names, unique_taste_names)
cdf.combined_corr_by_taste_dist(corr_data, cross_taste_dir, unique_given_names, 
                                  unique_corr_names,unique_segment_names, unique_taste_names)
cross_epoch_dir = os.path.join(results_dir, 'cross_epoch_plots')
if os.path.isdir(cross_epoch_dir) == False:
    os.mkdir(cross_epoch_dir)
print("\tComparing Epochs")
cdf.cross_epoch_diffs(corr_data, cross_epoch_dir, unique_given_names,
                      unique_corr_names, unique_segment_names, unique_taste_names)
cdf.combined_corr_by_epoch_dist(corr_data, cross_epoch_dir, unique_given_names, unique_corr_names,
                        unique_segment_names, unique_taste_names)

#%% gather_seg_data()

num_datasets = len(all_data_dict)
dataset_names = list(all_data_dict.keys())
seg_data = dict()
for n_i in range(num_datasets):
    data_name = dataset_names[n_i]
    data_dict = all_data_dict[data_name]['data']
    metadata = all_data_dict[data_name]['metadata']
    data_save_dir = data_dict['data_path']
    seg_save_dir = os.path.join(data_save_dir,'Segment_Comparison','indiv_distributions')
    num_dicts = os.listdir(seg_save_dir)
    seg_data[data_name] = dict()
    seg_data[data_name]['num_neur'] = data_dict['num_neur']
    segments_to_analyze = metadata['params_dict']['segments_to_analyze']
    seg_data[data_name]['segments_to_analyze'] = segments_to_analyze
    seg_data[data_name]['segment_names'] = data_dict['segment_names']
    segment_times = data_dict['segment_times']
    num_segments = len(seg_data[data_name]['segment_names'])
    seg_data[data_name]['segment_times_reshaped'] = [
        [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
    dig_in_names = data_dict['dig_in_names']
    seg_data[data_name]['dig_in_names'] = dig_in_names
    seg_data[data_name]['seg_data'] = dict()
    for nd_i in range(len(num_dicts)):
        dict_name = num_dicts[nd_i]
        dict_dir = os.path.join(seg_save_dir,dict_name)
        seg_data[data_name]['seg_data'][dict_name.split('.npy')[0]] = \
            np.load(dict_dir,allow_pickle=True).item()
        #This data is organized by [seg_name][bin_size] gives the result array
np.save(os.path.join(save_dir, 'seg_data.npy'),seg_data,allow_pickle=True)
# Save the combined dataset somewhere...
# _____Analysis Storage Directory_____
if not os.path.isdir(os.path.join(save_dir,'Segment_Comparison')):
    os.mkdir(os.path.join(save_dir,'Segment_Comparison'))
seg_results_dir = os.path.join(save_dir,'Segment_Comparison')

#%% find_seg_groupings()

unique_given_names = list(seg_data.keys())
unique_given_indices = np.sort(
    np.unique(unique_given_names, return_index=True)[1])
unique_given_names = [unique_given_names[i]
                      for i in unique_given_indices]
unique_analysis_names = np.array([list(seg_data[name]['seg_data'].keys(
)) for name in unique_given_names]).flatten()  # How many types of segment analyses
unique_analysis_indices = np.sort(
    np.unique(unique_analysis_names, return_index=True)[1])
unique_analysis_names = [unique_analysis_names[i] for i in unique_analysis_indices]
unique_segment_names = []
unique_bin_sizes = []
for name in unique_given_names:
    for analysis_name in unique_analysis_names:
        try:
            seg_to_analyze = seg_data[name]['segments_to_analyze']
            segment_names = np.array(seg_data[name]['segment_names'])
            seg_names = list(
                segment_names[seg_to_analyze])
            unique_segment_names.extend(seg_names)
            for seg_name in seg_names:
                bin_sizes = list(
                    seg_data[name]['seg_data'][analysis_name][seg_name].keys())
                bin_sizes_float = [float(bs) for bs in bin_sizes]
                unique_bin_sizes.extend(bin_sizes_float)
        except:
            print(name + " does not have binned data for " + analysis_name)
unique_segment_indices = np.sort(
    np.unique(unique_segment_names, return_index=True)[1])
unique_segment_names = [unique_segment_names[i]
                        for i in unique_segment_indices]
unique_bin_indices = np.sort(
    np.unique(unique_bin_sizes, return_index=True)[1])
unique_bin_sizes = [unique_bin_sizes[i]
                      for i in unique_bin_indices]

#%% plot_seg_results()

num_cond = len(seg_data)
results_dir = seg_results_dir

cdf.cross_dataset_seg_compare_combined_dist(seg_data,unique_given_names,
                              unique_analysis_names,
                              unique_segment_names,
                              unique_bin_sizes,
                              results_dir)
cdf.cross_dataset_seg_compare_means(seg_data,unique_given_names,
                              unique_analysis_names,
                              unique_segment_names,
                              unique_bin_sizes,
                              results_dir)
cdf.cross_dataset_seg_compare_mean_diffs(seg_data,unique_given_names,
                              unique_analysis_names,
                              unique_segment_names,
                              unique_bin_sizes,
                              results_dir)

#%% gather_cp_data()

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
cp_data = cp_data
np.save(os.path.join(save_dir, 'cp_data.npy'),cp_data,allow_pickle=True)
# Save the combined dataset somewhere...
# _____Analysis Storage Directory_____
if not os.path.isdir(os.path.join(save_dir,'Changepoint_Statistics')):
    os.mkdir(os.path.join(save_dir,'Changepoint_Statistics'))
cp_results_dir = os.path.join(save_dir,'Changepoint_Statistics')

#%% find_cp_groupings()

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

#%% plot_cp_results()

num_cond = len(cp_data)
results_dir = cp_results_dir
cdf.cross_dataset_cp_plots(cp_data, unique_given_names, 
                           unique_taste_names, max_cp_counts,
                           results_dir)

#%% gather_rate_corr_data()

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

#%% gather_dev_stats_data()

num_datasets = len(all_data_dict)
dataset_names = list(all_data_dict.keys())
dev_stats_data = dict()
for n_i in range(num_datasets):
    data_name = dataset_names[n_i]
    data_dict = all_data_dict[data_name]['data']
    metadata = all_data_dict[data_name]['metadata']
    data_save_dir = data_dict['data_path']
    
    dev_stats_save_dir = os.path.join(
        data_save_dir, 'Deviations')
    dev_dir_files = os.listdir(dev_stats_save_dir)
    dev_dict_dirs = []
    for dev_f in dev_dir_files:
        if dev_f[-4:] == '.npy':
            dev_dict_dirs.append(dev_f)
    dev_stats_data[data_name] = dict()
    dev_stats_data[data_name]['num_neur'] = data_dict['num_neur']
    segments_to_analyze = metadata['params_dict']['segments_to_analyze']
    dev_stats_data[data_name]['segments_to_analyze'] = segments_to_analyze
    dev_stats_data[data_name]['segment_names'] = data_dict['segment_names']
    segment_names_to_analyze = np.array(data_dict['segment_names'])[segments_to_analyze]
    segment_times = data_dict['segment_times']
    num_segments = len(dev_stats_data[data_name]['segment_names'])
    dev_stats_data[data_name]['segment_times_reshaped'] = [
        [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
    dig_in_names = data_dict['dig_in_names']
    dev_stats_data[data_name]['dig_in_names'] = dig_in_names
    dev_stats_data[data_name]['dev_stats'] = dict()
    for stat_i in range(len(dev_dict_dirs)):
        stat_dir_name = dev_dict_dirs[stat_i]
        stat_name = stat_dir_name.split('.')[0]
        result_dir = os.path.join(dev_stats_save_dir, stat_dir_name)
        result_dict = np.load(result_dir,allow_pickle=True).item()
        dev_stats_data[data_name]['dev_stats'][stat_name] = dict()
        for s_i, s_name in enumerate(segment_names_to_analyze):
            dev_stats_data[data_name]['dev_stats'][stat_name][s_name] = result_dict[s_i]
            
dev_stats_data = dev_stats_data
dict_save_dir = os.path.join(save_dir, 'dev_stats_data.npy')
np.save(dict_save_dir,dev_stats_data,allow_pickle=True)
# _____Analysis Storage Directory_____
if not os.path.isdir(os.path.join(save_dir,'Dev_Stats')):
    os.mkdir(os.path.join(save_dir,'Dev_Stats'))
dev_stats_results_dir = os.path.join(save_dir,'Dev_Stats')

#%% find_dev_stats_groupings()

unique_given_names = list(dev_stats_data.keys())
unique_given_indices = np.sort(
    np.unique(unique_given_names, return_index=True)[1])
unique_given_names = [unique_given_names[i]
                      for i in unique_given_indices]
unique_dev_stats_names = []
for name in unique_given_names:
    unique_dev_stats_names.extend(list(dev_stats_data[name]['dev_stats'].keys()))
unique_dev_stats_names = np.array(unique_dev_stats_names)
unique_dev_stats_indices = np.sort(
    np.unique(unique_dev_stats_names, return_index=True)[1])
unique_dev_stats_names = [unique_dev_stats_names[i] for i in unique_dev_stats_indices]
unique_segment_names = []
for name in unique_given_names:
    for dev_stat_name in unique_dev_stats_names:
        try:
            seg_names = list(
                dev_stats_data[name]['dev_stats'][dev_stat_name].keys())
            unique_segment_names.extend(seg_names)
        except:
            print(name + " does not have correlation data for " + dev_stat_name)
unique_segment_indices = np.sort(
    np.unique(unique_segment_names, return_index=True)[1])
unique_segment_names = [unique_segment_names[i]
                        for i in unique_segment_indices]

#%% plot_dev_stat_results()

num_cond = len(dev_stats_data)
results_dir = dev_stats_results_dir

cdf.cross_dataset_dev_stats_plots(dev_stats_data, unique_given_names, 
                                  unique_dev_stats_names, 
                                  unique_segment_names, 
                                  results_dir)

#%% gather_dev_null_data()

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

num_cond = len(dev_null_data)
results_dir = dev_null_results_dir

cdf.cross_dataset_dev_null_plots(dev_null_data, unique_given_names, 
                                 unique_dev_null_names, unique_segment_names, 
                                 results_dir)

#%% gather_dev_split_data()

num_datasets = len(all_data_dict)
dataset_names = list(all_data_dict.keys())
dev_split_corr_data = dict()
for n_i in range(num_datasets):
    data_name = dataset_names[n_i]
    data_dict = all_data_dict[data_name]['data']
    metadata = all_data_dict[data_name]['metadata']
    dev_split_corr_data[data_name] = dict()
    dev_split_corr_data[data_name]['num_neur'] = data_dict['num_neur']
    segments_to_analyze = metadata['params_dict']['segments_to_analyze']
    dev_split_corr_data[data_name]['segments_to_analyze'] = segments_to_analyze
    dev_split_corr_data[data_name]['segment_names'] = data_dict['segment_names']
    segment_names_to_analyze = np.array(data_dict['segment_names'])[segments_to_analyze]
    segment_times = data_dict['segment_times']
    num_segments = len(dev_split_corr_data[data_name]['segment_names'])
    dev_split_corr_data[data_name]['segment_times_reshaped'] = [
        [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
    dig_in_names = data_dict['dig_in_names']
    dev_split_corr_data[data_name]['dig_in_names'] = dig_in_names
    data_save_dir = data_dict['data_path']
    dev_split_save_dir = os.path.join(
        data_save_dir, 'Deviation_Sequence_Analysis')
    #Subfolders we care about: corr_tests and decode_splits
    #First load correlation data
    dev_split_corr_dir = os.path.join(dev_split_save_dir,'corr_tests','zscore_firing_rates')
    dev_split_corr_files = os.listdir(dev_split_corr_dir)
    dev_split_corr_dict_files = []
    for dev_corr_f in dev_split_corr_files:
        if dev_corr_f[-4:] == '.npy':
            dev_split_corr_dict_files.append(dev_corr_f)
    dev_split_corr_data[data_name]['corr_data'] = dict()
    for stat_i in range(len(dev_split_corr_dict_files)):
        stat_filename = dev_split_corr_dict_files[stat_i]
        stat_segment = (stat_filename.split('.')[0]).split('_')[0]
        dev_split_corr_data[data_name]['corr_data'][stat_segment] = dict()
        dict_data = np.load(os.path.join(dev_split_corr_dir,stat_filename),allow_pickle=True).item()
        epoch_pairs = list(dict_data.keys())
        dev_split_corr_data[data_name]['corr_data'][stat_segment]['epoch_pairs'] = epoch_pairs
        num_tastes = len(dig_in_names)
        for t_i, t_name in enumerate(dig_in_names):
            taste_corr_data = []
            for ep_i, ep in enumerate(epoch_pairs):
                taste_corr_data.append(dict_data[ep]['taste_corrs'][t_i])
            taste_corr_data = np.array(taste_corr_data)
            _, num_dev = np.shape(taste_corr_data)
            dev_split_corr_data[data_name]['corr_data'][stat_segment][t_name] = taste_corr_data
            
dict_save_dir = os.path.join(save_dir, 'dev_split_corr_data.npy')
np.save(dict_save_dir,dev_split_corr_data,allow_pickle=True)
# _____Analysis Storage Directory_____
if not os.path.isdir(os.path.join(save_dir,'Dev_Split_Corr')):
    os.mkdir(os.path.join(save_dir,'Dev_Split_Corr'))
dev_split_corr_results_dir = os.path.join(save_dir,'Dev_Split_Corr')

#%% find_dev_split_corr_groupings()

unique_given_names = list(dev_split_corr_data.keys())
unique_given_indices = np.sort(
    np.unique(unique_given_names, return_index=True)[1])
unique_given_names = [unique_given_names[i]
                      for i in unique_given_indices]
unique_epoch_pairs = []
unique_segment_names = []
unique_taste_names = []
for name in unique_given_names:
    segment_names = list(dev_split_corr_data[name]['corr_data'].keys())
    unique_segment_names.extend(segment_names)
    for seg_name in segment_names:
        epoch_pairs = dev_split_corr_data[name]['corr_data'][seg_name]['epoch_pairs']
        epoch_pair_strings = [str(ep) for ep in epoch_pairs]
        unique_epoch_pairs.extend(epoch_pair_strings)
        taste_names = list(dev_split_corr_data[name]['corr_data'][seg_name].keys())
        epoch_pairs_ind = np.where(np.array(taste_names) == 'epoch_pairs')
        if len(np.shape(np.where(np.array(taste_names) == 'epoch_pairs'))) == 2:
            taste_names.pop(epoch_pairs_ind[0][0])
        else:
            taste_names.pop(epoch_pairs_ind[0])
        unique_taste_names.extend(taste_names)

unique_epoch_indices = np.sort(
    np.unique(unique_epoch_pairs, return_index=True)[1])
unique_epoch_pairs = [unique_epoch_pairs[i] for i in unique_epoch_indices]
unique_segment_indices = np.sort(
    np.unique(unique_segment_names, return_index=True)[1])
unique_segment_names = [unique_segment_names[i] for i in unique_segment_indices]
unique_taste_indices = np.sort(
    np.unique(unique_taste_names, return_index=True)[1])
unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]

#%% plot_dev_split_results()
num_cond = len(dev_split_corr_data)
results_dir = dev_split_corr_results_dir

print("Beginning Plots.")
if num_cond > 1:
    cdf.cross_dataset_dev_split_corr_plots(dev_split_corr_data, unique_given_names, 
                                     unique_epoch_pairs, unique_segment_names, 
                                     unique_taste_names, results_dir)
    cdf.cross_dataset_dev_split_best_corr_plots(dev_split_corr_data, unique_given_names, 
                                     unique_epoch_pairs, unique_segment_names, 
                                     unique_taste_names, results_dir)
else:
    print("Not enough animals for cross-animal dev stat plots.")

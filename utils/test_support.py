#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:13:46 2024

@author: Hannah Germaine

Script to grab data for testing new code out of the pipeline.
"""

#Import necessary packages and functions
import os
current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-2]) + '/'
#functions_path = os.path.join(blech_codes_path,'functions')
os.chdir(blech_codes_path)
from utils.replay_utils import import_metadata, state_tracker
from utils.data_utils import import_data
import multiprocess
from functions.run_analysis_handler import run_analysis_steps

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

# run_analysis_handler

import functions.analysis_funcs as af

segment_spike_times = af.calc_segment_spike_times(data_dict['segment_times'],data_dict['spike_times'],data_dict['num_neur'])
tastant_spike_times = af.calc_tastant_spike_times(data_dict['segment_times'],data_dict['spike_times'],
						  data_dict['start_dig_in_times'],data_dict['end_dig_in_times'],
						  metadata['params_dict']['pre_taste'],metadata['params_dict']['post_taste'],data_dict['num_tastes'],data_dict['num_neur'])
data_dict['segment_spike_times'] = segment_spike_times
data_dict['tastant_spike_times'] = tastant_spike_times


# function imports
import os
import sys
import warnings
import tqdm
import gzip
import itertools
import json
from multiprocessing import Pool
import numpy as np
import scipy.stats as stats

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.null_distributions as nd
import functions.dev_plot_funcs as dpf
import functions.dev_funcs as df

# gather_variables()

num_neur = data_dict['num_neur']
pre_taste = metadata['params_dict']['pre_taste']
post_taste = metadata['params_dict']['post_taste']
segments_to_analyze = metadata['params_dict']['segments_to_analyze']
segment_spike_times = data_dict['segment_spike_times']
segment_names = data_dict['segment_names']
num_segments = len(segment_names)
segment_times = data_dict['segment_times']
segment_times_reshaped = [
    [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
local_size = metadata['params_dict']['local_size']
min_dev_size = metadata['params_dict']['min_dev_size']
num_null = metadata['params_dict']['num_null']
max_plot = metadata['params_dict']['max_plot']
count_cutoff = np.arange(1, num_neur)
bin_size = metadata['params_dict']['compare_null_params']['bin_size']
dev_dir = metadata['dir_name'] + 'Deviations/'
null_dir = metadata['dir_name'] + 'null_data/'
if os.path.isdir(null_dir) == False:
    os.mkdir(null_dir)
bin_dir = dev_dir + 'null_x_true_deviations/'
if os.path.isdir(bin_dir) == False:
    os.mkdir(bin_dir)
    
# import_deviations()

try:  # test if the data exists by trying to import the last
    filepath = dev_dir + \
        segment_names[segments_to_analyze[-1]
                           ] + '/deviations.json'
    with gzip.GzipFile(filepath, mode="r") as f:
        json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)

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
    segment_deviations = segment_deviations
except:
    print("ERROR! ERROR! ERROR!")
    print("Deviations were not calculated previously as expected.")
    print("Something went wrong in the analysis pipeline.")
    print("Please try reverting your analysis_state_tracker.csv to 1 to rerun.")
    print("If issues persist, contact Hannah.")
    sys.exit()
    
# gen_null_distributions()

# _____Generate null datasets_____
for s_i in segments_to_analyze:
    seg_null_dir = null_dir + segment_names[s_i] + '/'
    if os.path.isdir(seg_null_dir) == False:
        os.mkdir(seg_null_dir)
    try:
        filepath = seg_null_dir + 'null_' + \
            str(num_null-1) + '.json'
        with gzip.GzipFile(filepath, mode="r") as f:
            json_bytes = f.read()
            json_str = json_bytes.decode('utf-8')
            null_segment_spike_times = json.loads(json_str)
        print('\t' + segment_names[s_i] +
              ' null distributions previously created')
    except:
        # First create a null distribution set
        print('\tNow creating ' +
              segment_names[s_i] + ' null distributions')
        with Pool(processes=4) as pool:  # start 4 worker processes
            pool.map(nd.run_null_create_parallelized, zip(np.arange(num_null),
                                                          itertools.repeat(
                                                              segment_spike_times[s_i]),
                                                          itertools.repeat(
                                                              segment_times[0]),
                                                          itertools.repeat(
                                                              segment_times[-1]),
                                                          itertools.repeat(seg_null_dir)))
        pool.close()
print('\tCalculating null distribution spike times')
# _____Grab null dataset spike times_____
all_null_segment_spike_times = []
for null_i in range(num_null):
    null_segment_spike_times = []
    
    for s_i in segments_to_analyze:
        seg_null_dir = null_dir + segment_names[s_i] + '/'
        # Import the null distribution into memory
        filepath = seg_null_dir + 'null_' + str(null_i) + '.json'
        with gzip.GzipFile(filepath, mode="r") as f:
            json_bytes = f.read()
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)

        seg_null_dir = null_dir + segment_names[s_i] + '/'

        seg_start = segment_times_reshaped[s_i][0]
        seg_end = segment_times_reshaped[s_i][1]
        null_seg_st = []
        for n_i in range(num_neur):
            seg_spike_inds = np.where(
                (data[n_i] >= seg_start)*(data[n_i] <= seg_end))[0]
            null_seg_st.append(
                list(np.array(data[n_i])[seg_spike_inds]))
        null_segment_spike_times.append(null_seg_st)
    all_null_segment_spike_times.append(null_segment_spike_times)
all_null_segment_spike_times = all_null_segment_spike_times

# _____Import or calculate null deviations for all segments_____
try:  # test if the data exists by trying to import the last
    filepath = dev_dir + 'null_data/' + \
        segment_names[segments_to_analyze[-1]] + \
        '/null_'+str(num_null - 1)+'_deviations.json'
    with gzip.GzipFile(filepath, mode="r") as f:
        json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)
except:
    print("\nNow calculating deviations")
    for null_i in range(num_null):
        try:  # Not to have to restart if deviation calculation was interrupted partway
            last_seg = segment_names[segments_to_analyze[-1]]
            filepath = dev_dir + 'null_data/' + last_seg + \
                '/null_'+str(null_i)+'_deviations.json'
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)
            print("\t\tNull " + str(null_i) +
                  " Deviations Previously Calculated.")
            # Puts the onus on the user to delete the null deviations if they want them completely recalculated
        except:
            print("\tCreating Null " + str(null_i))
            seg_dirs = []
            # Import the null distribution into memory
            filepath = seg_null_dir + 'null_' + str(null_i) + '.json'
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)

            for s_i in segments_to_analyze:
                # create storage directory for null deviation data
                if os.path.isdir(dev_dir + 'null_data/') == False:
                    os.mkdir(dev_dir + 'null_data/')
                seg_dir = dev_dir + 'null_data/' + \
                    segment_names[s_i] + '/'
                if os.path.isdir(seg_dir) == False:
                    os.mkdir(seg_dir)
                seg_dir = dev_dir + 'null_data/' + \
                    segment_names[s_i] + \
                    '/null_' + str(null_i) + '_'
                seg_dirs.append(seg_dir)

            null_segment_spike_times = all_null_segment_spike_times[null_i]
            segment_times_reshaped = [
                segment_times_reshaped[i] for i in segments_to_analyze]
            with Pool(processes=4) as pool:  # start 4 worker processes
                pool.map(df.run_dev_pull_parallelized, zip(null_segment_spike_times,
                                                           itertools.repeat(
                                                               local_size),
                                                           itertools.repeat(
                                                               min_dev_size),
                                                           segment_times_reshaped,
                                                           seg_dirs))
            pool.close()

print("\tNow importing calculated null deviations")
all_null_deviations = []
for null_i in tqdm.tqdm(range(num_null)):
    null_segment_deviations = []
    for s_i in segments_to_analyze:
        filepath = dev_dir + 'null_data/' + \
            segment_names[s_i] + '/null_' + \
            str(null_i) + '_deviations.json'
        with gzip.GzipFile(filepath, mode="r") as f:
            json_bytes = f.read()
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)
            null_segment_deviations.append(data)
    all_null_deviations.append(null_segment_deviations)
del null_i, null_segment_deviations, s_i, filepath, json_bytes, json_str, data

# convert_to_rasters()

# Calculate segment deviation spikes
print("\tNow pulling true deviation rasters")
num_seg = len(segments_to_analyze)
seg_spike_times = [segment_spike_times[i]
                   for i in segments_to_analyze]
seg_times_reshaped = np.array(segment_times_reshaped)[
    segments_to_analyze, :]
z_bin = metadata['params_dict']['z_bin']
segment_dev_rasters, segment_dev_times, _, _, _, _ = df.create_dev_rasters(num_seg,
                                                                     seg_spike_times,
                                                                     seg_times_reshaped,
                                                                     segment_deviations, z_bin)
segment_dev_rasters = segment_dev_rasters
segment_dev_times = segment_dev_times

# Calculate segment deviation spikes
print("\tNow pulling null deviation rasters")
null_dev_rasters = []
null_dev_times = []
for null_i in tqdm.tqdm(range(num_null)):
    null_segment_deviations = all_null_deviations[null_i]
    null_segment_spike_times = all_null_segment_spike_times[null_i]
    null_segment_dev_rasters_i, null_segment_dev_times_i, _, _, _, _ = df.create_dev_rasters(num_seg,
                                                                                       null_segment_spike_times,
                                                                                       seg_times_reshaped,
                                                                                       null_segment_deviations, z_bin)
    null_dev_rasters.append(null_segment_dev_rasters_i)
    null_dev_times.append(null_segment_dev_times_i)

del all_null_deviations

# calc_statistics()

print('\tCalculating Deviation Statistics')
neur_fr_dict = dict()
for s_ind, s_i in tqdm.tqdm(enumerate(segments_to_analyze)):
    # Gather data / parameters
    seg_name = segment_names[s_i]
    # Create segment save dir for figures
    seg_fig_save_dir = os.path.join(bin_dir, seg_name)
    if not os.path.isdir(seg_fig_save_dir):
        os.mkdir(seg_fig_save_dir)
    # _____Gather null data deviation event stats_____
    null_dev_lengths = []
    null_dev_neuron_counts = []
    null_dev_spike_counts = []
    null_dev_frs = []
    for null_i in range(num_null):
        all_rast = null_dev_rasters[null_i][s_ind]
        null_i_num_neur, null_i_num_spikes, all_len = df.calculate_dev_null_stats(
            all_rast, null_dev_times[null_i][s_ind])
        null_dev_neuron_counts.append(null_i_num_neur)
        null_dev_spike_counts.append(null_i_num_spikes)
        null_dev_lengths.append(all_len)
        null_dev_frs.append(np.array(null_i_num_spikes)/(all_len/1000)/num_neur)
    # _____Gather true data deviation event stats_____
    all_rast = segment_dev_rasters[s_ind]
    true_dev_neuron_counts, true_dev_spike_counts, true_dev_lengths = df.calculate_dev_null_stats(
        all_rast, segment_dev_times[s_ind])
    true_dev_frs = np.array(true_dev_spike_counts)/(true_dev_lengths/1000)/num_neur
    
    # _____Gather data as dictionary of number of events as a function of cutoff
    #FR data
    null_max_fr = np.max([np.max(null_dev_frs[null_i]) for null_i in range(num_null)])
    max_fr = np.max([null_max_fr,np.max(true_dev_frs)])
    fr_x_vals = np.arange(np.ceil(max_fr).astype('int'))
    true_fr_x_val_counts = np.nan*np.ones(len(fr_x_vals))
    null_fr_x_val_counts_all = []
    null_fr_x_val_counts_mean = np.nan*np.ones(len(fr_x_vals))
    null_fr_x_val_counts_std = np.nan*np.ones(len(fr_x_vals))
    for n_cut_i, n_cut in enumerate(fr_x_vals):
        true_fr_x_val_counts[n_cut_i] = np.sum(
            (np.array(true_dev_frs) > n_cut).astype('int'))
        null_fr_x_val_counts = []
        for null_i in range(num_null):
            null_fr_x_val_counts.append(
                np.sum((np.array(null_dev_frs[null_i]) > n_cut).astype('int')))
        null_fr_x_val_counts_all.append(null_fr_x_val_counts)
        null_fr_x_val_counts_mean[n_cut_i] = np.nanmean(
            null_fr_x_val_counts)
        null_fr_x_val_counts_std[n_cut_i] = np.nanstd(
            null_fr_x_val_counts)
        # Plot the individual distribution
        dpf.plot_dev_x_null_single_dist(
            null_fr_x_val_counts, true_fr_x_val_counts[n_cut_i], 'dev_fr_cutoff_' + str(n_cut), seg_fig_save_dir)
    neur_fr_dict[seg_name + '_true'] = [list(fr_x_vals),
                                           list(true_fr_x_val_counts)]
    neur_fr_dict[seg_name + '_null'] = [list(fr_x_vals),
                                           list(
                                               null_fr_x_val_counts_mean),
                                           list(null_fr_x_val_counts_std)]
    # Calculate percentiles
    percentiles = []  # Calculate percentile of true data point in null data distribution
    for n_cut in fr_x_vals:
        try:
            percentiles.extend([round(stats.percentileofscore(
                null_fr_x_val_counts_all[n_cut-1], true_fr_x_val_counts[n_cut-1]), 2)])
        except:
            percentiles.extend([100])
    neur_fr_dict[seg_name +
                    '_percentile'] = [list(fr_x_vals), percentiles]

# Save the dictionaries
np.save(os.path.join(bin_dir,'neur_fr_dict.npy'), neur_fr_dict)

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


#%%

import matplotlib.pyplot as plt

#Calculate spike times for selected segment
pre_seg_spikes = segment_spike_times[0]
num_neur = len(pre_seg_spikes)
time_to_plot = 5*60*1000
spike_raster = np.zeros((num_neur,time_to_plot))
spike_list = []
for n_i in range(num_neur):
    neur_spikes = np.array(pre_seg_spikes[n_i]).astype('int')
    keep_spikes = (np.where(np.array(neur_spikes) < time_to_plot)[0]).astype('int')
    spike_raster[n_i,np.array(neur_spikes)[keep_spikes]] = 1
    spike_list.append(list(np.array(neur_spikes)[keep_spikes]))

#Calculate Population Rate (Hz)
bin_size = 40
bin_starts = np.arange(0,time_to_plot,bin_size)
pop_rate = np.zeros(np.shape(bin_starts))
for b_ind, b_i in enumerate(bin_starts):
    pop_rate[b_ind] = np.sum(spike_raster[:,b_i:b_i+bin_size])/num_neur/(bin_size/1000)

#Firing Rate Vectors
fr_vecs = np.zeros((num_neur,len(bin_starts)))
for b_ind, b_i in enumerate(bin_starts):
    fr_vecs[:,b_ind] = np.sum(spike_raster[:,b_i:b_i+bin_size],1)/(bin_size/1000)

#Full Raster
plt.figure(figsize=(20,2))
plt.eventplot(spike_list, colors='k')
plt.xticks(np.arange(0,time_to_plot,60000),np.arange(0,5,1))
plt.xlabel('Time (min)')
plt.ylabel('Neuron Index')
plt.title('5 Minutes of Pre-Taste Rest')

#Population Rate
plt.figure(figsize=(20,2))
plt.plot(pop_rate,c='k')
len_bin_starts = len(bin_starts)
min_step = len_bin_starts/5
x_tick_vals = np.arange(0,len_bin_starts,min_step)
plt.xticks(x_tick_vals,np.arange(0,5,1))
plt.xlabel('Time (min)')
plt.ylabel('Population Rate (Hz)')
plt.title('5 Minutes of Pre-Taste Rest Population Rate')

#Plot the firing rate vectors as an image
plt.figure(figsize=(20,2))
plt.imshow(np.flipud(fr_vecs),aspect='auto',cmap='jet')
plt.xticks(x_tick_vals,np.arange(0,5,1))
plt.xlabel('Time (min)')
plt.ylabel('Population Rate (Hz)')
plt.title('5 Minutes of Pre-Taste Rest Population Rate')


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

#%% Generate null deviations support

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

import functions.null_distributions as nd
import functions.dev_plot_funcs as dpf
import functions.dev_funcs as df

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
            segment_times_reshaped_lim = [
                segment_times_reshaped[i] for i in segments_to_analyze]
            with Pool(processes=4) as pool:  # start 4 worker processes
                pool.map(df.run_dev_pull_parallelized, zip(null_segment_spike_times,
                                                           itertools.repeat(
                                                               local_size),
                                                           itertools.repeat(
                                                               min_dev_size),
                                                           segment_times_reshaped_lim,
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

# Calculate segment deviation spikes
print("\tNow pulling true deviation rasters")
num_seg = len(segments_to_analyze)
seg_spike_times = [segment_spike_times[i]
                   for i in segments_to_analyze]
seg_times_reshaped = np.array(segment_times_reshaped)[
    segments_to_analyze, :]
z_bin = metadata['params_dict']['z_bin']
segment_dev_rasters, segment_dev_times, _, _ = df.create_dev_rasters(num_seg,
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
    null_segment_dev_rasters_i, null_segment_dev_times_i, _, _ = df.create_dev_rasters(num_seg,
                                                                                       null_segment_spike_times,
                                                                                       seg_times_reshaped,
                                                                                       null_segment_deviations, z_bin)
    null_dev_rasters.append(null_segment_dev_rasters_i)
    null_dev_times.append(null_segment_dev_times_i)

try:  # Import calculated dictionaries if they exist
    filepath = bin_dir + 'neur_count_dict.npy'
    neur_count_dict = np.load(filepath, allow_pickle=True).item()
    filepath = bin_dir + 'neur_spike_dict.npy'
    neur_spike_dict = np.load(filepath, allow_pickle=True).item()
    filepath = bin_dir + 'neur_len_dict.npy'
    neur_spike_dict = np.load(filepath, allow_pickle=True).item()
    print('\tTruexNull deviation datasets previously calculated.')
except:  # Calculate dictionaries
    print('\tCalculating Deviation Statistics')
    neur_count_dict = dict()
    neur_spike_dict = dict()
    neur_len_dict = dict()
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
        for null_i in range(num_null):
            all_rast = null_dev_rasters[null_i][s_ind]
            null_i_num_neur, null_i_num_spikes, all_len = df.calculate_dev_null_stats(
                all_rast, null_dev_times[null_i][s_ind])
            null_dev_neuron_counts.append(null_i_num_neur)
            null_dev_spike_counts.append(null_i_num_spikes)
            null_dev_lengths.append(all_len)
        # _____Gather true data deviation event stats_____
        true_dev_neuron_counts = []
        true_dev_spike_counts = []
        all_rast = segment_dev_rasters[s_ind]
        true_dev_neuron_counts, true_dev_spike_counts, true_dev_lengths = df.calculate_dev_null_stats(
            all_rast, segment_dev_times[s_ind])
        # _____Gather data as dictionary of number of events as a function of cutoff
        # Neuron count data
        null_max_neur_count = np.max(
            [np.max(null_dev_neuron_counts[null_i]) for null_i in range(num_null)])
        max_neur_count = int(
            np.max([np.max(null_max_neur_count), np.max(true_dev_neuron_counts)]))
        neur_x_vals = np.arange(10, max_neur_count)
        true_neur_x_val_counts = np.nan*np.ones(np.shape(neur_x_vals))
        null_neur_x_val_counts_all = []
        null_neur_x_val_counts_mean = np.nan * \
            np.ones(np.shape(neur_x_vals))
        null_neur_x_val_counts_std = np.nan * \
            np.ones(np.shape(neur_x_vals))
        for n_cut_i, n_cut in enumerate(neur_x_vals):
            true_neur_x_val_counts[n_cut_i] = np.sum(
                (np.array(true_dev_neuron_counts) > n_cut).astype('int'))
            null_neur_x_val_counts = []
            for null_i in range(num_null):
                null_neur_x_val_counts.append(
                    np.sum((np.array(null_dev_neuron_counts[null_i]) > n_cut).astype('int')))
            null_neur_x_val_counts_all.append(null_neur_x_val_counts)
            null_neur_x_val_counts_mean[n_cut_i] = np.nanmean(
                null_neur_x_val_counts)
            null_neur_x_val_counts_std[n_cut_i] = np.nanstd(
                null_neur_x_val_counts)
            # Plot the individual distribution
            dpf.plot_dev_x_null_single_dist(
                null_neur_x_val_counts, true_neur_x_val_counts[n_cut_i], 'neur_count_cutoff_' + str(n_cut), seg_fig_save_dir)
        neur_count_dict[seg_name + '_true'] = [list(neur_x_vals),
                                               list(true_neur_x_val_counts)]
        neur_count_dict[seg_name + '_null'] = [list(neur_x_vals),
                                               list(
                                                   null_neur_x_val_counts_mean),
                                               list(null_neur_x_val_counts_std)]
        # Calculate percentiles
        percentiles = []  # Calculate percentile of true data point in null data distribution
        for n_cut in neur_x_vals:
            try:
                percentiles.extend([round(stats.percentileofscore(
                    null_neur_x_val_counts_all[n_cut-1], true_neur_x_val_counts[n_cut-1]), 2)])
            except:
                percentiles.extend([100])
        neur_count_dict[seg_name +
                        '_percentile'] = [list(neur_x_vals), percentiles]

        # Spike count data
        null_max_neur_spikes = np.max(
            [np.max(null_dev_spike_counts[null_i]) for null_i in range(num_null)])
        max_spike_count = int(
            np.max([np.max(null_max_neur_spikes), np.max(true_dev_spike_counts)]))
        spike_x_vals = np.arange(1, max_spike_count)
        true_neur_x_val_spikes = np.nan*np.ones(np.shape(spike_x_vals))
        null_neur_x_val_spikes_all = []
        null_neur_x_val_spikes_mean = np.nan * \
            np.ones(np.shape(spike_x_vals))
        null_neur_x_val_spikes_std = np.nan * \
            np.ones(np.shape(spike_x_vals))
        for s_cut_i, s_cut in enumerate(spike_x_vals):
            true_neur_x_val_spikes[s_cut_i] = np.sum(
                (np.array(true_dev_spike_counts) > s_cut).astype('int'))
            null_neur_x_val_spikes = []
            for null_i in range(num_null):
                null_neur_x_val_spikes.append(
                    np.sum((np.array(null_dev_spike_counts[null_i]) > s_cut).astype('int')))
            null_neur_x_val_spikes_all.append(null_neur_x_val_spikes)
            null_neur_x_val_spikes_mean[s_cut_i] = np.nanmean(
                null_neur_x_val_spikes)
            null_neur_x_val_spikes_std[s_cut_i] = np.nanstd(
                null_neur_x_val_spikes)
            # Plot the individual distribution
            dpf.plot_dev_x_null_single_dist(
                null_neur_x_val_spikes, true_neur_x_val_spikes[s_cut_i], 'spike_count_cutoff_' + str(s_cut), seg_fig_save_dir)
        neur_spike_dict[seg_name + '_true'] = [list(spike_x_vals),
                                               list(true_neur_x_val_spikes)]
        neur_spike_dict[seg_name + '_null'] = [list(spike_x_vals),
                                               list(
                                                   null_neur_x_val_spikes_mean),
                                               list(null_neur_x_val_spikes_std)]
        percentiles = []  # Calculate percentile of true data point in null data distribution
        for s_cut in spike_x_vals:
            try:
                percentiles.extend([round(stats.percentileofscore(
                    null_neur_x_val_spikes_all[s_cut-1], true_neur_x_val_spikes[s_cut-1]), 2)])
            except:
                percentiles.extend([100])
        neur_spike_dict[seg_name +
                        '_percentile'] = [list(spike_x_vals), percentiles]

        # Burst length data
        null_max_neur_len = np.max(
            [np.max(null_dev_lengths[null_i]) for null_i in range(num_null)])
        max_len = int(
            np.max([np.max(null_max_neur_len), np.max(true_dev_lengths)]))
        len_x_vals = np.arange(min_dev_size, max_len)
        true_neur_x_val_lengths = np.nan*np.ones(np.shape(len_x_vals))
        null_neur_x_val_lengths_all = []
        null_neur_x_val_lengths_mean = np.nan * \
            np.ones(np.shape(len_x_vals))
        null_neur_x_val_lengths_std = np.nan * \
            np.ones(np.shape(len_x_vals))
        for l_cut_i, l_cut in enumerate(len_x_vals):
            true_neur_x_val_lengths[l_cut_i] = np.sum(
                (np.array(true_dev_lengths) > l_cut).astype('int'))
            null_neur_x_val_lengths = []
            for null_i in range(num_null):
                null_neur_x_val_lengths.append(
                    np.sum((np.array(null_dev_lengths[null_i]) > l_cut).astype('int')))
            null_neur_x_val_lengths_all.append(null_neur_x_val_lengths)
            null_neur_x_val_lengths_mean[l_cut_i] = np.nanmean(
                null_neur_x_val_lengths)
            null_neur_x_val_lengths_std[l_cut_i] = np.nanstd(
                null_neur_x_val_lengths)
            # Plot the individual distribution
            dpf.plot_dev_x_null_single_dist(
                null_neur_x_val_lengths, true_neur_x_val_lengths[l_cut_i], 'length_cutoff_' + str(l_cut), seg_fig_save_dir)
        neur_len_dict[seg_name + '_true'] = [list(len_x_vals),
                                             list(true_neur_x_val_lengths)]
        neur_len_dict[seg_name + '_null'] = [list(len_x_vals),
                                             list(
                                                 null_neur_x_val_lengths_mean),
                                             list(null_neur_x_val_lengths_std)]
        percentiles = []  # Calculate percentile of true data point in null data distribution
        for l_cut in len_x_vals:
            try:
                percentiles.extend([round(stats.percentileofscore(
                    null_neur_x_val_lengths_all[l_cut-1], true_neur_x_val_lengths[l_cut-1]), 2)])
            except:
                percentiles.extend([100])
        neur_len_dict[seg_name +
                      '_percentile'] = [list(len_x_vals), percentiles]

    # Save the dictionaries
    filepath = bin_dir + 'neur_count_dict.npy'
    np.save(filepath, neur_count_dict)
    filepath = bin_dir + 'neur_spike_dict.npy'
    np.save(filepath, neur_spike_dict)
    filepath = bin_dir + 'neur_len_dict.npy'
    np.save(filepath, neur_len_dict)

print('\tPlotting deviation statistics datasets')
filepath = bin_dir + 'neur_count_dict.npy'
neur_count_dict = np.load(filepath, allow_pickle=True).item()
filepath = bin_dir + 'neur_spike_dict.npy'
neur_spike_dict = np.load(filepath, allow_pickle=True).item()
filepath = bin_dir + 'neur_len_dict.npy'
neur_len_dict = np.load(filepath, allow_pickle=True).item()

# _____Plotting_____
neur_true_count_x = []
neur_true_count_vals = []
neur_null_count_x = []
neur_null_count_mean = []
neur_null_count_std = []
neur_true_spike_x = []
neur_true_spike_vals = []
neur_null_spike_x = []
neur_null_spike_mean = []
neur_null_spike_std = []
neur_true_len_x = []
neur_true_len_vals = []
neur_null_len_x = []
neur_null_len_mean = []
neur_null_len_std = []
for s_ind, s_i in tqdm.tqdm(enumerate(segments_to_analyze)):
    seg_name = segment_names[s_i]
    segment_start_time = segment_times[s_i]
    segment_end_time = segment_times[s_i+1]
    segment_length = segment_end_time - segment_start_time
    neur_true_count_data = neur_count_dict[seg_name + '_true']
    neur_null_count_data = neur_count_dict[seg_name + '_null']
    percentile_count_data = neur_count_dict[seg_name + '_percentile']
    neur_true_spike_data = neur_spike_dict[seg_name + '_true']
    neur_null_spike_data = neur_spike_dict[seg_name + '_null']
    percentile_spike_data = neur_spike_dict[seg_name + '_percentile']
    neur_true_len_data = neur_len_dict[seg_name + '_true']
    neur_null_len_data = neur_len_dict[seg_name + '_null']
    percentile_len_data = neur_len_dict[seg_name + '_percentile']
    # Plot the neuron count data
    # Normalizing the number of bins to number of bins / second
    norm_val = segment_length/1000
    nd.plot_indiv_truexnull(np.array(neur_true_count_data[0]), np.array(neur_null_count_data[0]), np.array(neur_true_count_data[1]), np.array(neur_null_count_data[1]),
                            np.array(neur_null_count_data[2]), segment_length, norm_val, bin_dir, 'Neuron Counts', seg_name, np.array(percentile_count_data[1]))
    neur_true_count_x.append(np.array(neur_true_count_data[0]))
    neur_null_count_x.append(np.array(neur_null_count_data[0]))
    neur_true_count_vals.append(np.array(neur_true_count_data[1]))
    neur_null_count_mean.append(np.array(neur_null_count_data[1]))
    neur_null_count_std.append(np.array(neur_null_count_data[2]))
    # Plot the spike count data
    nd.plot_indiv_truexnull(np.array(neur_true_spike_data[0]), np.array(neur_null_spike_data[0]), np.array(neur_true_spike_data[1]), np.array(neur_null_spike_data[1]),
                            np.array(neur_null_spike_data[2]), np.array(segment_length), norm_val, bin_dir, 'Spike Counts', seg_name, np.array(percentile_spike_data[1]))
    neur_true_spike_x.append(np.array(neur_true_spike_data[0]))
    neur_null_spike_x.append(np.array(neur_null_spike_data[0]))
    neur_true_spike_vals.append(np.array(neur_true_spike_data[1]))
    neur_null_spike_mean.append(np.array(neur_null_spike_data[1]))
    neur_null_spike_std.append(np.array(neur_null_spike_data[2]))
    # Plot the length data
    nd.plot_indiv_truexnull(np.array(neur_true_len_data[0]), np.array(neur_null_len_data[0]), np.array(neur_true_len_data[1]), np.array(neur_null_len_data[1]),
                            np.array(neur_null_len_data[2]), np.array(segment_length), norm_val, bin_dir, 'Lengths', seg_name, np.array(percentile_len_data[1]))
    neur_true_len_x.append(np.array(neur_true_len_data[0]))
    neur_null_len_x.append(np.array(neur_null_len_data[0]))
    neur_true_len_vals.append(np.array(neur_true_len_data[1]))
    neur_null_len_mean.append(np.array(neur_null_len_data[1]))
    neur_null_len_std.append(np.array(neur_null_len_data[2]))
# Plot all neuron count data
nd.plot_all_truexnull(neur_true_count_x, neur_null_count_x, neur_true_count_vals, neur_null_count_mean,
                      neur_null_count_std, norm_val, bin_dir, 'Neuron Counts', list(np.array(segment_names)[segments_to_analyze]))
# Plot all spike count data
nd.plot_all_truexnull(neur_true_spike_x, neur_null_spike_x, neur_true_spike_vals, neur_null_spike_mean,
                      neur_null_spike_std, norm_val, bin_dir, 'Spike Counts', list(np.array(segment_names)[segments_to_analyze]))

# Plot all length data
nd.plot_all_truexnull(neur_true_len_x, neur_null_len_x, neur_true_len_vals, neur_null_len_mean,
                      neur_null_len_std, norm_val, bin_dir, 'Lengths', list(np.array(segment_names)[segments_to_analyze]))

#%% Calculate null dev corr support
import os
import sys
import warnings

import tqdm
import gzip
import json
import numpy as np

import functions.dev_funcs as df
import functions.dev_plot_funcs as dpf
import functions.hdf5_handling as hf5

#These directories should already exist
hdf5_dir = metadata['hdf5_dir']
null_dir = metadata['dir_name'] + 'null_data/'
dev_dir = metadata['dir_name'] + 'Deviations/'
comp_dir = metadata['dir_name'] + 'dev_x_taste/'
if os.path.isdir(comp_dir) == False:
    os.mkdir(comp_dir)
corr_dir = comp_dir + 'corr/'
if os.path.isdir(corr_dir) == False:
    os.mkdir(corr_dir)

num_neur = data_dict['num_neur']
tastant_spike_times = data_dict['tastant_spike_times']
start_dig_in_times = data_dict['start_dig_in_times']
end_dig_in_times = data_dict['end_dig_in_times']
dig_in_names = data_dict['dig_in_names']
segment_names = data_dict['segment_names']
num_segments = len(segment_names)
pre_taste = metadata['params_dict']['pre_taste']
post_taste = metadata['params_dict']['post_taste']
# Import changepoint data
num_cp = metadata['params_dict']['num_cp']+ 1
data_group_name = 'changepoint_data'
pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'pop_taste_cp_raster_inds')
pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
data_group_name = 'taste_discriminability'
discrim_neur = np.squeeze(hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'discrim_neur'))
discrim_neur = discrim_neur
num_null = metadata['params_dict']['num_null']
segments_to_analyze = metadata['params_dict']['segments_to_analyze']
segment_names = data_dict['segment_names']
segment_spike_times = data_dict['segment_spike_times']
segment_times = data_dict['segment_times']
segment_times_reshaped = [
    [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
z_bin = metadata['params_dict']['z_bin']

try:  # test if the data exists by trying to import the last from each segment
    null_i = num_null - 1
    for s_i in tqdm.tqdm(segments_to_analyze):
        filepath = dev_dir + 'null_data/' + \
             segment_names[s_i] + '/null_' + \
                 str(null_i) + '_deviations.json'
        with gzip.GzipFile(filepath, mode="r") as f:
            json_bytes = f.read()
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)
except:
    print("ERROR! ERROR! ERROR!")
    print("Null deviations were not calculated previously as expected.")
    print("Something went wrong in the analysis pipeline.")
    print("Please try reverting your analysis_state_tracker.csv to 2 to rerun.")
    print("If issues persist, contact Hannah.")
    sys.exit()
    
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

print("\tNow pulling null deviation rasters")
num_seg = len(segments_to_analyze)
seg_times_reshaped = np.array(segment_times_reshaped)[
    segments_to_analyze, :]

null_dev_vecs = []
null_dev_vecs_zscore = []
for s_i in range(num_seg):
    null_dev_vecs.append([])
    null_dev_vecs_zscore.append([])
for null_i in tqdm.tqdm(range(num_null)):
    null_segment_deviations = all_null_deviations[null_i]
    null_segment_spike_times = all_null_segment_spike_times[null_i]
    _, _, null_segment_dev_vecs_i, null_segment_dev_vecs_zscore_i = df.create_dev_rasters(num_seg,
                                                             null_segment_spike_times,
                                                             seg_times_reshaped,
                                                             null_segment_deviations,
                                                             z_bin, no_z = False)
    #Compiled all into a single segment group, rather than keeping separated by null dist
    for s_i in range(num_seg):
        null_dev_vecs[s_i].extend(null_segment_dev_vecs_i[s_i])
        null_dev_vecs_zscore[s_i].extend(null_segment_dev_vecs_zscore_i[s_i])

print('\tCalculating null correlation distributions')
if os.path.isdir(corr_dir + 'all_neur/') == False:
    os.mkdir(corr_dir + 'all_neur/')
current_corr_dir = corr_dir + 'all_neur/' + 'null/'
if os.path.isdir(current_corr_dir) == False:
    os.mkdir(current_corr_dir)
neuron_keep_indices = np.ones(np.shape(discrim_neur))
# Calculate correlations
df.calculate_vec_correlations(num_neur, null_dev_vecs, tastant_spike_times,
                              start_dig_in_times, end_dig_in_times, segment_names,
                              dig_in_names, pre_taste, post_taste, pop_taste_cp_raster_inds,
                              current_corr_dir, neuron_keep_indices, segments_to_analyze)  # For all neurons in dataset
# Now plot and calculate significance!
# Plot dir setup
print('\tPlotting null correlation distributions')
plot_dir = current_corr_dir + 'plots/'
if os.path.isdir(plot_dir) == False:
    os.mkdir(plot_dir)
plot_dir = plot_dir
corr_dev_stats = df.pull_corr_dev_stats(
    segment_names, dig_in_names, current_corr_dir, segments_to_analyze)
dpf.plot_stats(corr_dev_stats, segment_names, dig_in_names, plot_dir,
               'Correlation', neuron_keep_indices, segments_to_analyze)
segment_pop_vec_data = dpf.plot_combined_stats(corr_dev_stats, segment_names, dig_in_names,
                                               plot_dir, 'Correlation', neuron_keep_indices, segments_to_analyze)
null_corr_percentiles = df.null_dev_corr_90_percentiles(corr_dev_stats, segment_names, dig_in_names, 
                                 current_corr_dir, segments_to_analyze)

print('\tCalculating null correlation distributions')
if os.path.isdir(corr_dir + 'all_neur_zscore/') == False:
    os.mkdir(corr_dir + 'all_neur_zscore/')
current_corr_dir = corr_dir + 'all_neur_zscore/' + 'null/'
if os.path.isdir(current_corr_dir) == False:
    os.mkdir(current_corr_dir)
neuron_keep_indices = np.ones(np.shape(discrim_neur))
# Calculate correlations
df.calculate_vec_correlations_zscore(num_neur, z_bin, null_dev_vecs_zscore, tastant_spike_times,
                              segment_times, segment_spike_times, start_dig_in_times, end_dig_in_times,
                              segment_names, dig_in_names, pre_taste, post_taste, pop_taste_cp_raster_inds,
                              current_corr_dir, neuron_keep_indices, segments_to_analyze)  # For all neurons in dataset
# Now plot and calculate significance!
# Plot dir setup
print('\tPlotting null correlation distributions')
plot_dir = current_corr_dir + 'plots/'
if os.path.isdir(plot_dir) == False:
    os.mkdir(plot_dir)
plot_dir = plot_dir
corr_dev_stats = df.pull_corr_dev_stats(
    segment_names, dig_in_names, current_corr_dir, segments_to_analyze)
dpf.plot_stats(corr_dev_stats, segment_names, dig_in_names, plot_dir,
               'Correlation', neuron_keep_indices, segments_to_analyze)
segment_pop_vec_data = dpf.plot_combined_stats(corr_dev_stats, segment_names, dig_in_names,
                                               plot_dir, 'Correlation', neuron_keep_indices, segments_to_analyze)
null_corr_percentiles = df.null_dev_corr_90_percentiles(corr_dev_stats, segment_names, dig_in_names, 
                                 current_corr_dir, segments_to_analyze)


#%% Indiv animal correlation support

import os
import json
import gzip
import tqdm
import numpy as np
import functions.dev_plot_funcs as dpf
import functions.dev_funcs as df
import functions.hdf5_handling as hf5

# Directories
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
min_dev_size = metadata['params_dict']['min_dev_size']

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

print("\tNow pulling changepoints")
# Import changepoint data
data_group_name = 'changepoint_data'
pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'pop_taste_cp_raster_inds')
pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
num_pt_cp = num_cp + 2

data_group_name = 'taste_discriminability'
peak_epochs = np.squeeze(hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'peak_epochs'))
discrim_neur = np.squeeze(hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'discrim_neur'))

current_corr_dir = corr_dir + 'all_neur_zscore/'
if os.path.isdir(current_corr_dir) == False:
    os.mkdir(current_corr_dir)
neuron_keep_indices = np.ones(np.shape(discrim_neur))
# Calculate correlations
df.calculate_vec_correlations_zscore(num_neur, z_bin, segment_dev_vec_zscore, tastant_spike_times,
                                     segment_times, segment_spike_times, start_dig_in_times, end_dig_in_times,
                                     segment_names, dig_in_names, pre_taste, post_taste, pop_taste_cp_raster_inds,
                                     current_corr_dir, neuron_keep_indices, segments_to_analyze)
# Calculate significant events
sig_dev, sig_dev_counts = df.calculate_significant_dev(segment_dev_times, 
                                                       segment_times, dig_in_names,
                                                       segment_names, current_corr_dir,
                                                       segments_to_analyze)

plot_dir = current_corr_dir + 'plots/'
if os.path.isdir(plot_dir) == False:
    os.mkdir(plot_dir)

# Calculate stats
print("\tCalculating Correlation Statistics")
corr_dev_stats = df.pull_corr_dev_stats(
    segment_names, dig_in_names, current_corr_dir, segments_to_analyze)
print("\tPlotting Correlation Statistics")
dpf.plot_stats(corr_dev_stats, segment_names, dig_in_names, plot_dir,
               'Correlation', neuron_keep_indices, segments_to_analyze)
print("\tPlotting Combined Correlation Statistics")
segment_pop_vec_data = dpf.plot_combined_stats(corr_dev_stats, segment_names, dig_in_names,
                                               plot_dir, 'Correlation', neuron_keep_indices, segments_to_analyze)
segment_pop_vec_data = segment_pop_vec_data
df.top_dev_corr_bins(corr_dev_stats, segment_names, dig_in_names,
                     plot_dir, neuron_keep_indices, segments_to_analyze)

print("\tCalculate statistical significance between correlation distributions.")
current_stats_dir = current_corr_dir + 'stats/'
if os.path.isdir(current_stats_dir) == False:
    os.mkdir(current_stats_dir)

# KS-test
df.stat_significance(segment_pop_vec_data, segment_names, dig_in_names,
                     current_stats_dir, 'population_vec_correlation', segments_to_analyze)

# T-test less
df.stat_significance_ttest_less(segment_pop_vec_data, segment_names,
                                dig_in_names, current_stats_dir,
                                'population_vec_correlation_ttest_less', segments_to_analyze)

# T-test more
df.stat_significance_ttest_more(segment_pop_vec_data, segment_names,
                                dig_in_names, current_stats_dir,
                                'population_vec_correlation_ttest_more', segments_to_analyze)

# Mean compare
df.mean_compare(segment_pop_vec_data, segment_names, dig_in_names,
                current_stats_dir, 'population_vec_mean_difference', segments_to_analyze)

print("\tDetermine best correlation per deviation and plot stats.")
best_dir = current_corr_dir + 'best/'
if os.path.isdir(best_dir) == False:
    os.mkdir(best_dir)

dpf.best_corr_calc_plot(dig_in_names, epochs_to_analyze,
                        segments_to_analyze, segment_names,
                        segment_times_reshaped, segment_dev_times, 
                        dev_dir, min_dev_size, segment_spike_times,
                        current_corr_dir, pop_taste_cp_raster_inds, 
                        tastant_spike_times, start_dig_in_times, 
                        end_dig_in_times, pre_taste, 
                        post_taste, num_neur,
                        best_dir, no_indiv_plot = False)

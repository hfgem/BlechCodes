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

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
#functions_path = os.path.join(blech_codes_path,'functions')
os.chdir(blech_codes_path)
import functions.analysis_funcs as af
import functions.hdf5_handling as hf5
import functions.dependent_decoding_funcs as ddf
#utils_path = os.path.join(blech_codes_path,'utils')
#os.chdir(utils_path)
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



import functions.decoder_tuning as dt
import functions.decoding_funcs as df
import functions.plot_decoding_funcs as pdf
import functions.dependent_decoding_funcs as ddf
import functions.dev_funcs as dev_f

#Directories
hdf5_dir = metadata['hdf5_dir']
seq_dir = metadata['dir_name'] + 'Deviation_Sequence_Analysis/'
if os.path.isdir(seq_dir) == False:
	os.mkdir(seq_dir)
dev_dir = metadata['dir_name'] + 'Deviations/'
#General Params/Variables
num_neur = data_dict['num_neur']
pre_taste = metadata['params_dict']['pre_taste']
post_taste = metadata['params_dict']['post_taste']
pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
post_taste_dt = np.ceil(post_taste*1000).astype('int')
segments_to_analyze = metadata['params_dict']['segments_to_analyze']
epochs_to_analyze = metadata['params_dict']['epochs_to_analyze']
segment_names = data_dict['segment_names']
num_segments = len(segment_names)
segment_spike_times = data_dict['segment_spike_times']
segment_times = data_dict['segment_times']
segment_times_reshaped = [[segment_times[i],segment_times[i+1]] for i in range(num_segments)]
num_cp = metadata['params_dict']['num_cp'] + 1 #Remember this imported value is 1 less than the number of epochs
tastant_spike_times = data_dict['tastant_spike_times']
start_dig_in_times = data_dict['start_dig_in_times']
end_dig_in_times = data_dict['end_dig_in_times']
dig_in_names = data_dict['dig_in_names']
num_tastes = len(dig_in_names)
fr_bins = metadata['params_dict']['fr_bins']
#Bayes Params/Variables
skip_time = metadata['params_dict']['bayes_params']['skip_time']
skip_dt = np.ceil(skip_time*1000).astype('int')
e_skip_time = metadata['params_dict']['bayes_params']['e_skip_time']
e_skip_dt = np.ceil(e_skip_time*1000).astype('int')
taste_e_len_time = metadata['params_dict']['bayes_params']['taste_e_len_time']
taste_e_len_dt = np.ceil(taste_e_len_time*1000).astype('int') 
seg_e_len_time = metadata['params_dict']['bayes_params']['seg_e_len_time']
seg_e_len_dt = np.ceil(seg_e_len_time*1000).astype('int') 
bayes_fr_bins = metadata['params_dict']['bayes_params']['fr_bins']
neuron_count_thresh = metadata['params_dict']['bayes_params']['neuron_count_thresh']
max_decode = metadata['params_dict']['bayes_params']['max_decode']
seg_stat_bin = metadata['params_dict']['bayes_params']['seg_stat_bin']
trial_start_frac = metadata['params_dict']['bayes_params']['trial_start_frac']
decode_prob_cutoff = metadata['params_dict']['bayes_params']['decode_prob_cutoff']
bin_time = metadata['params_dict']['bayes_params']['z_score_bin_time']
bin_dt = np.ceil(bin_time*1000).astype('int')
#Import changepoint data
pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(hdf5_dir,'changepoint_data','pop_taste_cp_raster_inds')
num_pt_cp = num_cp + 2

print("\tNow importing calculated deviations")

num_seg_to_analyze = len(segments_to_analyze)
segment_names_to_analyze = [segment_names[i] for i in segments_to_analyze]
segment_times_to_analyze_reshaped = [
    [segment_times[i], segment_times[i+1]] for i in segments_to_analyze]
segment_spike_times_to_analyze = [segment_spike_times[i] for i in segments_to_analyze]

segment_deviations = []
for s_i in tqdm.tqdm(range(num_seg_to_analyze)):
    filepath = dev_dir + \
        segment_names_to_analyze[s_i] + '/deviations.json'
    with gzip.GzipFile(filepath, mode="r") as f:
        json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)
        segment_deviations.append(data)

print("\tNow pulling true deviation rasters")
segment_dev_rasters, segment_dev_times, segment_dev_fr_vecs, \
    segment_dev_fr_vecs_zscore, segment_zscore_means, segment_zscore_stds = \
        dev_f.create_dev_rasters(num_seg_to_analyze, 
                segment_spike_times_to_analyze,
                np.array(segment_times_to_analyze_reshaped),
                segment_deviations, pre_taste)

print("\tPulling taste rasters")
tastant_raster_dict = af.taste_response_rasters(num_tastes, num_neur, 
                           tastant_spike_times, start_dig_in_times, 
                           pop_taste_cp_raster_inds, pre_taste_dt)

print("\tPulling FR Distributions")
tastant_fr_dist_pop, taste_num_deliv, max_hz_pop = ddf.taste_fr_dist(num_neur, tastant_spike_times,
                                                                  pop_taste_cp_raster_inds, bayes_fr_bins,
                                                                  start_dig_in_times, pre_taste_dt,
                                                                  post_taste_dt, trial_start_frac)

tastant_fr_dist_z_pop, taste_num_deliv, max_hz_z_pop, min_hz_z_pop = ddf.taste_fr_dist_zscore(num_neur, tastant_spike_times,
                                                                                           segment_spike_times, segment_names,
                                                                                           segment_times, pop_taste_cp_raster_inds,
                                                                                           bayes_fr_bins, start_dig_in_times, pre_taste_dt,
                                                                                           post_taste_dt, bin_dt, trial_start_frac)


import functions.dev_sequence_funcs as dsf
import time

num_null = 100

#%%
tic = time.time()
dsf.split_match_calc(num_neur,segment_dev_rasters,segment_zscore_means,segment_zscore_stds,
                   tastant_raster_dict,tastant_fr_dist_pop,tastant_fr_dist_z_pop,
                   dig_in_names,segment_names,num_null, seq_dir, 
                   segments_to_analyze, epochs_to_analyze)
toc = time.time()
print('Total Pair Correlation Analysis Time = ' + str(np.round((toc-tic)/60, 2)) + ' (min).')
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

#%% Decoder Support

import functions.decoder_tuning as dt
import functions.decoding_funcs as df
import functions.plot_decoding_funcs as pdf
import functions.dependent_decoding_funcs as ddf

#Directories
hdf5_dir = metadata['hdf5_dir']
bayes_dir = metadata['dir_name'] + 'Bayes_Dependent_Decoding/'
if os.path.isdir(bayes_dir) == False:
	os.mkdir(bayes_dir)
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
#Import taste selectivity data
try:
	select_neur = hf5.pull_data_from_hdf5(hdf5_dir, 'taste_selectivity', 'taste_select_neur_epoch_bin')[0]
except:
	print("\tNo taste selectivity data found. Skipping.")
#Import discriminability data
peak_epochs = np.squeeze(hf5.pull_data_from_hdf5(hdf5_dir,'taste_discriminability','peak_epochs'))
discrim_neur = np.squeeze(hf5.pull_data_from_hdf5(hdf5_dir,'taste_discriminability','discrim_neur'))
#Convert discriminatory neuron changepoint data into pop_taste_cp_raster_inds shape
#TODO: Add a flag for a user to select whether to use discriminatory neurons or selective neurons
num_discrim_cp = np.shape(peak_epochs)[0]
min_cp = np.min((num_pt_cp,num_discrim_cp))

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

main_decode_dir = bayes_dir + 'All_Neurons/'
if os.path.isdir(main_decode_dir) == False:
	os.mkdir(main_decode_dir)
cur_dist = tastant_fr_dist_pop

print("\tDecoding all neurons")
all_neur_dir = bayes_dir + 'All_Neurons/'
if os.path.isdir(all_neur_dir) == False:
    os.mkdir(all_neur_dir)

#%% Decoder tuning support

# dt.test_decoder_params(dig_in_names, start_dig_in_times, num_neur, tastant_spike_times,
#                         tastant_fr_dist_pop, pop_taste_cp_raster_inds, pre_taste_dt, post_taste_dt,
#                         epochs_to_analyze, select_neur, e_skip_dt, taste_e_len_dt, 
#                         max_hz_pop, main_decode_dir)

tastant_fr_dist = tastant_fr_dist_pop
cp_raster_inds = pop_taste_cp_raster_inds
taste_select_neur = np.ones(np.shape(discrim_neur))
max_hz = max_hz_pop
save_dir = main_decode_dir
e_len_dt = taste_e_len_dt

# Get trial indices for train/test sets
num_tastes = len(tastant_spike_times)
all_trial_inds = []
for t_i in range(num_tastes):
    taste_trials = len(tastant_spike_times[t_i])
    all_trial_inds.append(list(np.arange(taste_trials)))

del t_i, taste_trials

dt.multistep_epoch_decoder(num_neur, start_dig_in_times, tastant_fr_dist, 
                all_trial_inds, tastant_spike_times, cp_raster_inds,
                pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names,
                max_hz, save_dir, epochs_to_analyze)

dt.multistep_taste_decoder(num_neur, start_dig_in_times, tastant_fr_dist, 
                all_trial_inds, tastant_spike_times, cp_raster_inds,
                pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names,
                max_hz, save_dir, epochs_to_analyze)

#%% Decoder pipeline support

decode_dir = all_neur_dir + 'GMM_Decoding/'
if os.path.isdir(decode_dir) == False:
    os.mkdir(decode_dir)
    
taste_select_neur = np.ones(np.shape(discrim_neur))

seg_e_len_dt = 20

ddf.decode_epochs(tastant_fr_dist_pop, 	segment_spike_times,
                  	post_taste_dt, 	e_skip_dt, 	seg_e_len_dt,
                  	dig_in_names, 	segment_times, 	segment_names,
                  	start_dig_in_times, 	taste_num_deliv, select_neur,
                  	max_hz_pop, decode_dir, 	neuron_count_thresh, decode_prob_cutoff,
                  	False, epochs_to_analyze, segments_to_analyze)

tastant_fr_dist = tastant_fr_dist_pop
e_len_dt = seg_e_len_dt
taste_select_epoch = select_neur
max_hz = max_hz_pop
save_dir = decode_dir

decode_prob_cutoff = 0.9 #1 - 1/num_tastes

print("\t\tPlotting Decoded Results")
pdf.plot_decoded(	tastant_fr_dist_pop, num_tastes, num_neur,
                segment_spike_times, 	tastant_spike_times,
                	start_dig_in_times, 	end_dig_in_times, 	post_taste_dt,
                	pre_taste_dt, 	pop_taste_cp_raster_inds, 	bin_dt, dig_in_names,
                	segment_times, 	segment_names, 	taste_num_deliv,
                	taste_select_epoch, 	decode_dir, 	max_decode, 	max_hz_pop,
                	seg_stat_bin, 	neuron_count_thresh, 	seg_e_len_dt, trial_start_frac,
                	epochs_to_analyze, 	segments_to_analyze, 	decode_prob_cutoff)

#%% Deviation Decoding Pipeline support

import functions.dev_funcs as dev_f
import functions.dependent_decoding_funcs as ddf

dev_dir = metadata['dir_name'] + 'Deviations/'
bayes_dir = metadata['dir_name'] + 'Deviation_Dependent_Decoding/'
if os.path.isdir(bayes_dir) == False:
    os.mkdir(bayes_dir)

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
segment_dev_rasters, segment_dev_times, segment_dev_fr_vecs, segment_dev_fr_vecs_zscore = dev_f.create_dev_rasters(num_seg_to_analyze, 
                                                        segment_spike_times_to_analyze,
                                                        np.array(segment_times_to_analyze_reshaped),
                                                        segment_deviations, pre_taste)

tastant_fr_dist = tastant_fr_dist_pop

print("\tDecoding all neurons")
all_neur_dir = bayes_dir + 'All_Neurons/'
if os.path.isdir(all_neur_dir) == False:
    os.mkdir(all_neur_dir)
    
taste_select_neur = np.ones(np.shape(discrim_neur))

decode_dir = all_neur_dir + 'GMM_Decoding/'
if os.path.isdir(decode_dir) == False:
    os.mkdir(decode_dir)

#Normal Decode
ddf.decode_deviations_epochs(tastant_fr_dist, segment_spike_times, dig_in_names, 
                  segment_times, segment_names, start_dig_in_times, taste_num_deliv,
                  segment_dev_times, segment_dev_fr_vecs, taste_select_neur, bin_dt,
                  decode_dir, False, epochs_to_analyze, segments_to_analyze)

import functions.plot_dev_decoding_funcs as pddf
pddf.plot_decoded(num_tastes, num_neur, segment_spike_times, tastant_spike_times,
                 start_dig_in_times, post_taste_dt, pre_taste_dt,
                 pop_taste_cp_raster_inds, bin_dt, dig_in_names, segment_times,
                 segment_names, taste_select_neur, decode_dir, max_hz_pop,
                 segment_dev_times, segment_dev_fr_vecs, segment_dev_fr_vecs_zscore,
                 neuron_count_thresh, seg_e_len_dt, trial_start_frac,
                 epochs_to_analyze, segments_to_analyze, decode_prob_cutoff)


print("\tDecoding all neurons z-scored")
all_neur_dir = bayes_dir + 'All_Neurons_Z_Scored/'
if os.path.isdir(all_neur_dir) == False:
    os.mkdir(all_neur_dir)
    
taste_select_neur = np.ones(np.shape(discrim_neur))

decode_dir = all_neur_dir + 'GMM_Decoding/'
if os.path.isdir(decode_dir) == False:
    os.mkdir(decode_dir)

#Z-Scored Decode
ddf.decode_deviations_epochs(tastant_fr_dist_z_pop, segment_spike_times, dig_in_names, 
                  segment_times, segment_names, start_dig_in_times, taste_num_deliv,
                  segment_dev_times, segment_dev_fr_vecs_zscore, taste_select_neur, 
                  bin_dt, decode_dir, True, epochs_to_analyze, segments_to_analyze)

import functions.plot_dev_decoding_funcs as pddf
pddf.plot_decoded(num_tastes, num_neur, segment_spike_times, tastant_spike_times,
                 start_dig_in_times, post_taste_dt, pre_taste_dt,
                 pop_taste_cp_raster_inds, bin_dt, dig_in_names, segment_times,
                 segment_names, taste_select_neur, decode_dir, max_hz_pop,
                 segment_dev_times, segment_dev_fr_vecs, segment_dev_fr_vecs_zscore,
                 neuron_count_thresh, seg_e_len_dt, trial_start_frac,
                 epochs_to_analyze, segments_to_analyze, decode_prob_cutoff)

# cp_raster_inds = pop_taste_cp_raster_inds
# z_bin_dt = bin_dt
# taste_select_epoch = taste_select_neur
# save_dir = decode_dir
# max_hz = max_hz_pop
# e_len_dt = seg_e_len_dt

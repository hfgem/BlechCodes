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
os.chdir(blech_codes_path)
from utils.replay_utils import import_metadata, state_tracker
from utils.data_utils import import_data
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

import functions.analysis_funcs as af
segment_spike_times = af.calc_segment_spike_times(data_dict['segment_times'],data_dict['spike_times'],data_dict['num_neur'])
tastant_spike_times = af.calc_tastant_spike_times(data_dict['segment_times'],data_dict['spike_times'],
						  data_dict['start_dig_in_times'],data_dict['end_dig_in_times'],
						  metadata['params_dict']['pre_taste'],metadata['params_dict']['post_taste'],data_dict['num_tastes'],data_dict['num_neur'])
data_dict['segment_spike_times'] = segment_spike_times
data_dict['tastant_spike_times'] = tastant_spike_times

#%%

import os

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import numpy as np
import functions.analysis_funcs as af
import functions.dependent_decoding_funcs as ddf
import functions.decoding_funcs as df

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
#Bayes Params/Variables
skip_time = metadata['params_dict']['bayes_params']['skip_time']
skip_dt = np.ceil(skip_time*1000).astype('int')
e_skip_time = metadata['params_dict']['bayes_params']['e_skip_time']
e_skip_dt = np.ceil(e_skip_time*1000).astype('int')
e_len_time = metadata['params_dict']['bayes_params']['e_len_time']
e_len_dt = np.ceil(e_len_time*1000).astype('int')
neuron_count_thresh = metadata['params_dict']['bayes_params']['neuron_count_thresh']
max_decode = metadata['params_dict']['bayes_params']['max_decode']
seg_stat_bin = metadata['params_dict']['bayes_params']['seg_stat_bin']
trial_start_frac = metadata['params_dict']['bayes_params']['trial_start_frac']
decode_prob_cutoff = metadata['params_dict']['bayes_params']['decode_prob_cutoff']
bin_time = metadata['params_dict']['bayes_params']['bin_time']
bin_dt = np.ceil(bin_time*1000).astype('int')
#Import changepoint data
pop_taste_cp_raster_inds = af.pull_data_from_hdf5(hdf5_dir,'changepoint_data','pop_taste_cp_raster_inds')
pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
num_pt_cp = num_cp + 2
#Import taste selectivity data
try:
	select_neur = af.pull_data_from_hdf5(hdf5_dir, 'taste_selectivity', 'taste_select_neur_epoch_bin')[0]
	select_neur = select_neur
except:
	print("\tNo taste selectivity data found. Skipping.")
#Import discriminability data
peak_epochs = np.squeeze(af.pull_data_from_hdf5(hdf5_dir,'taste_discriminability','peak_epochs'))
discrim_neur = np.squeeze(af.pull_data_from_hdf5(hdf5_dir,'taste_discriminability','discrim_neur'))
#Convert discriminatory neuron changepoint data into pop_taste_cp_raster_inds shape
#TODO: Add a flag for a user to select whether to use discriminatory neurons or selective neurons
num_discrim_cp = np.shape(peak_epochs)[0]
min_cp = np.min((num_pt_cp,num_discrim_cp))
discrim_cp_raster_inds = []
for t_i in range(len(dig_in_names)):
	t_cp_vec = np.ones((np.shape(pop_taste_cp_raster_inds[t_i])[0],num_discrim_cp+1))
	t_cp_vec = (peak_epochs[:min_cp] + int(pre_taste*1000))*t_cp_vec[:,:min_cp]
	discrim_cp_raster_inds.append(t_cp_vec)
discrim_cp_raster_inds = discrim_cp_raster_inds
discrim_neur = discrim_neur

tastant_fr_dist_pop, taste_num_deliv, max_hz_pop = ddf.taste_fr_dist(num_neur, tastant_spike_times,
                                                                        discrim_cp_raster_inds,
                                                                        start_dig_in_times, pre_taste_dt,
                                                                        post_taste_dt, trial_start_frac)
tastant_fr_dist_z_pop, taste_num_deliv, max_hz_z_pop, min_hz_z_pop = ddf.taste_fr_dist_zscore(num_neur, tastant_spike_times,
                                                                                                 segment_spike_times, segment_names,
                                                                                                 segment_times, discrim_cp_raster_inds,
                                                                                                 start_dig_in_times, pre_taste_dt,
                                                                                                 post_taste_dt, bin_dt, trial_start_frac)

print("\tDecoding all neurons")
decode_dir = bayes_dir + 'All_Neurons/'
if os.path.isdir(decode_dir) == False:
	os.mkdir(decode_dir)
cur_dist = tastant_fr_dist_pop
select_neur = np.ones(np.shape(discrim_neur))

taste_select_neur = select_neur
save_dir = decode_dir
tastant_fr_dist = cur_dist
max_hz = max_hz_pop
cp_raster_inds = discrim_cp_raster_inds

	
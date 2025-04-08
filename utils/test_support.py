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


import os
import numpy as np
import functions.analysis_funcs as af
import functions.dev_funcs as df
import functions.hdf5_handling as hf5
import functions.dev_plot_funcs as dpf
import functions.slide_plot_funcs as spf

slide_dir = metadata['dir_name'] + 'Sliding_Correlations/'
if os.path.isdir(slide_dir) == False:
    os.mkdir(slide_dir)
hdf5_dir = metadata['hdf5_dir']
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
# Remember this is 1 less than the number of epochs so add 1
num_cp = metadata['params_dict']['num_cp'] + 1
tastant_spike_times = data_dict['tastant_spike_times']
start_dig_in_times = data_dict['start_dig_in_times']
end_dig_in_times = data_dict['end_dig_in_times']
dig_in_names = data_dict['dig_in_names']
bin_size = metadata['params_dict']['min_dev_size'] #Use the same minimal size as deviation events
z_bin = metadata['params_dict']['z_bin']
data_group_name = 'changepoint_data'
pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'pop_taste_cp_raster_inds')
data_group_name = 'taste_discriminability'
discrim_neur = np.squeeze(hf5.pull_data_from_hdf5(
    hdf5_dir, data_group_name, 'discrim_neur'))


bin_times, bin_pop_fr, bin_fr_vecs, bin_fr_vecs_zscore = af.get_bin_activity(segment_times_reshaped,
                                                            segment_spike_times, bin_size, 
                                                            segments_to_analyze, False)

corr_dir = os.path.join(slide_dir,'all_neur_zscore')
if os.path.isdir(corr_dir) == False:
    os.mkdir(corr_dir)
    
neuron_keep_indices = np.ones((num_cp,num_neur))

df.calculate_vec_correlations_zscore(num_neur, z_bin, bin_fr_vecs_zscore, bin_pop_fr, tastant_spike_times,
                                     segment_times, segment_spike_times, start_dig_in_times, end_dig_in_times,
                                     segment_names, dig_in_names, pre_taste, post_taste, pop_taste_cp_raster_inds,
                                     corr_dir, neuron_keep_indices, segments_to_analyze)

# Plot dir setup
plot_dir = os.path.join(corr_dir,'plots/')
if os.path.isdir(plot_dir) == False:
    os.mkdir(plot_dir)
# Pull statistics into dictionary and plot
corr_slide_stats = df.pull_corr_dev_stats(
    segment_names, dig_in_names, corr_dir, 
    segments_to_analyze, False)
print("\tPlotting Correlation Statistics")
# dpf.plot_stats(corr_slide_stats, segment_names, dig_in_names, plot_dir,
#                'Correlation', neuron_keep_indices, segments_to_analyze)
# segment_pop_vec_data = dpf.plot_combined_stats(corr_slide_stats, segment_names, dig_in_names,
#                                                plot_dir, 'Correlation', neuron_keep_indices, segments_to_analyze)


#Correlation calculations and plots
popfr_corr_storage = spf.slide_corr_vs_rate(corr_slide_stats,bin_times,bin_pop_fr,
                       num_cp,plot_dir,corr_dir,
                       segment_names,dig_in_names,
                       segments_to_analyze)
# #90th-Percentile Correlations and the related pop rates
# spf.top_corr_rate_dist(corr_slide_stats,bin_times,bin_pop_fr,
#                        num_cp,plot_dir,corr_dir,
#                        segment_names,dig_in_names,
#                        segments_to_analyze)
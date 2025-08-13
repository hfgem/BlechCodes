#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:11:00 2025

@author: Hannah Germaine

Multiday test support: file to help test out development of multiday 
analysis pipeline
"""

#%% Import metadata and data files

import os
import csv
import numpy as np
from tkinter.filedialog import askdirectory
from functions.blech_held_units_funcs import int_input
from utils.multiday_utils import import_metadata
from utils.data_utils import import_multiday_data
from utils.input_funcs import *

script_path = os.path.realpath(__file__)
blechcodes_dir = os.path.dirname(script_path)

print('Where did you save the held units pickle file?')
held_save_dir = askdirectory()
held_data_dict = np.load(os.path.join(held_save_dir,'data_dict.npy'),allow_pickle=True).item()
held_unit_csv = os.path.join(held_save_dir,'held_units.csv')
held_units = []
with open(held_unit_csv, 'r') as heldunitcsv:
    heldreader = csv.reader(heldunitcsv, delimiter=' ', quotechar='|')
    for row in heldreader:
        row_vals = row[0].split(',')
        try:
            is_int = int(row_vals[0])
            held_units.append([int(row_vals[i]) for i in range(len(row_vals))])
        except:
            is_header = row_vals

num_days = len(held_units[0])

metadata_handler = import_metadata([held_data_dict])

metadata = dict()
metadata['held_units'] = np.array(held_units)
data_dict = dict()

#Now go day by day and import data
for n_i in range(num_days):
    print("Day " + str(n_i+1))
    day_metadata = metadata_handler.metadata_dict[n_i]
    metadata[n_i] = day_metadata
    try:
        dig_in_names = day_metadata.info_dict['taste_params']['tastes']
    except:
        dig_in_names = []
    
    data_handler = import_multiday_data([day_metadata, dig_in_names])
    day_data = dict()
    for var in vars(data_handler):
        day_data[var] = getattr(data_handler,var)
    del data_handler
    data_dict[n_i] = day_data

import os
import tqdm
import gzip
import json
import numpy as np
from functions.blech_held_units_funcs import *
import functions.analysis_funcs as af
import functions.dev_funcs as dev_f
import functions.hdf5_handling as hf5
import functions.dependent_decoding_funcs as ddf
from tkinter.filedialog import askdirectory

# Using the directories of the different days find a common root folder and create save dir there
joint_name = metadata[0]['info_dict']['name']
day_dirs = []
for n_i in range(num_days):
    day_name = metadata[n_i]['info_dict']['exp_type']
    joint_name = joint_name + '_' + day_name
    day_dir = data_dict[n_i]['data_path']
    day_dirs.append(os.path.split(day_dir))
day_dir_array = np.array(day_dirs)
stop_ind = -1
while stop_ind == -1:
    for i in range(np.shape(day_dir_array)[1]):
        if len(np.unique(day_dir_array[:,i])) > 1:
            stop_ind = i
    if stop_ind == -1:
        day_dirs = []
        for n_i in range(num_days):
            day_dir_list = day_dirs[n_i]
            new_day_dir_list = os.path.split(day_dir_list[0])
            new_day_dir_list.extend(day_dir_list[1:])
            day_dirs.append(new_day_dir_list)
#Now create the new folder in the shared root path
root_path = os.path.join(*list(day_dir_array[0,:stop_ind]))
save_dir = os.path.join(root_path,joint_name)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
bayes_dir = os.path.join(save_dir,\
                              'Deviation_Dependent_Decoding/')
if os.path.isdir(bayes_dir) == False:
    os.mkdir(bayes_dir)

def get_spike_time_datasets(args):
    segment_times, spike_times, num_neur, keep_neur, start_dig_in_times, \
        end_dig_in_times, pre_taste, post_taste, num_tastes = args
    
    # _____Pull out spike times for all tastes (and no taste)_____
    segment_spike_times = af.calc_segment_spike_times(
        segment_times, spike_times, num_neur)
    tastant_spike_times = af.calc_tastant_spike_times(segment_times, spike_times,
                                                      start_dig_in_times, end_dig_in_times,
                                                      pre_taste, post_taste, 
                                                      num_tastes, num_neur)
    
    # _____Update spike times for only held units_____
    keep_segment_spike_times = []
    for s_i in range(len(segment_spike_times)):
        seg_neur_keep = []
        for n_i in keep_neur:
            seg_neur_keep.append(segment_spike_times[s_i][n_i])
        keep_segment_spike_times.append(seg_neur_keep)
        
    keep_tastant_spike_times = []
    for t_i in range(len(tastant_spike_times)):
        taste_deliv_list = []
        for d_i in range(len(tastant_spike_times[t_i])):
            deliv_neur_keep = []
            for n_i in keep_neur:
                deliv_neur_keep.append(tastant_spike_times[t_i][d_i][n_i])
            taste_deliv_list.append(deliv_neur_keep)
        keep_tastant_spike_times.append(taste_deliv_list)
        
    return keep_segment_spike_times, keep_tastant_spike_times

#For each day store the relevant variables/parameters
day_vars = dict()
for n_i in range(num_days):
    day_vars[n_i] = dict()
    # Directories
    day_vars[n_i]['hdf5_dir'] = os.path.join(metadata[n_i]['dir_name'], metadata[n_i]['hdf5_dir'])
    day_vars[n_i]['dev_dir'] = os.path.join(metadata[n_i]['dir_name'],'Deviations')
    day_vars[n_i]['null_dir'] = os.path.join(metadata[n_i]['dir_name'],'null_data')
    # General Params/Variables
    num_neur = data_dict[n_i]['num_neur']
    keep_neur = metadata['held_units'][:,n_i]
    day_vars[n_i]['keep_neur'] = keep_neur
    day_vars[n_i]['pre_taste'] = metadata[n_i]['params_dict']['pre_taste']
    day_vars[n_i]['post_taste'] = metadata[n_i]['params_dict']['post_taste']
    day_vars[n_i]['pre_taste_dt'] = np.ceil(day_vars[n_i]['pre_taste']*1000).astype('int')
    day_vars[n_i]['post_taste_dt'] = np.ceil(day_vars[n_i]['pre_taste']*1000).astype('int')
    day_vars[n_i]['segments_to_analyze'] = metadata[n_i]['params_dict']['segments_to_analyze']
    day_vars[n_i]['epochs_to_analyze'] = metadata[n_i]['params_dict']['epochs_to_analyze']
    day_vars[n_i]['segment_names'] = data_dict[n_i]['segment_names']
    day_vars[n_i]['num_segments'] = len(day_vars[n_i]['segment_names'])
    day_vars[n_i]['segment_times'] = data_dict[n_i]['segment_times']
    day_vars[n_i]['segment_times_reshaped'] = [
        [day_vars[n_i]['segment_times'][i], day_vars[n_i]['segment_times'][i+1]] for i in range(day_vars[n_i]['num_segments'])]
    # Remember this imported value is 1 less than the number of epochs
    day_vars[n_i]['num_cp'] = metadata[n_i]['params_dict']['num_cp'] + 1
    day_vars[n_i]['start_dig_in_times'] = data_dict[n_i]['start_dig_in_times']
    day_vars[n_i]['end_dig_in_times'] = data_dict[n_i]['end_dig_in_times']
    day_vars[n_i]['dig_in_names'] = data_dict[n_i]['dig_in_names']
    day_vars[n_i]['num_tastes'] = len(day_vars[n_i]['dig_in_names'])
    day_vars[n_i]['fr_bins'] = metadata[n_i]['params_dict']['fr_bins']
    day_vars[n_i]['z_bin'] = metadata[n_i]['params_dict']['z_bin']
    segment_spike_times, tastant_spike_times = get_spike_time_datasets(
        [day_vars[n_i]['segment_times'],data_dict[n_i]['spike_times'],
         num_neur, keep_neur, day_vars[n_i]['start_dig_in_times'],
         day_vars[n_i]['end_dig_in_times'], day_vars[n_i]['pre_taste'],
         day_vars[n_i]['post_taste'], day_vars[n_i]['num_tastes']])
    day_vars[n_i]['segment_spike_times'] = segment_spike_times
    day_vars[n_i]['tastant_spike_times'] = tastant_spike_times
    #Bayes Params/Variables
    day_vars[n_i]['skip_time'] = metadata[n_i]['params_dict']['bayes_params']['skip_time']
    day_vars[n_i]['skip_dt'] = np.ceil(day_vars[n_i]['skip_time']*1000).astype('int')
    day_vars[n_i]['e_skip_time'] = metadata[n_i]['params_dict']['bayes_params']['e_skip_time']
    day_vars[n_i]['e_skip_dt'] = np.ceil(day_vars[n_i]['e_skip_time']*1000).astype('int')
    day_vars[n_i]['taste_e_len_time'] = metadata[n_i]['params_dict']['bayes_params']['taste_e_len_time']
    day_vars[n_i]['taste_e_len_dt'] = np.ceil(day_vars[n_i]['taste_e_len_time']*1000).astype('int') 
    day_vars[n_i]['seg_e_len_time'] = metadata[n_i]['params_dict']['bayes_params']['seg_e_len_time']
    day_vars[n_i]['seg_e_len_dt'] = np.ceil(day_vars[n_i]['seg_e_len_time']*1000).astype('int') 
    day_vars[n_i]['bayes_fr_bins'] = metadata[n_i]['params_dict']['bayes_params']['fr_bins']
    day_vars[n_i]['neuron_count_thresh'] = metadata[n_i]['params_dict']['bayes_params']['neuron_count_thresh']
    day_vars[n_i]['max_decode'] = metadata[n_i]['params_dict']['bayes_params']['max_decode']
    day_vars[n_i]['seg_stat_bin'] = metadata[n_i]['params_dict']['bayes_params']['seg_stat_bin']
    day_vars[n_i]['trial_start_frac'] = metadata[n_i]['params_dict']['bayes_params']['trial_start_frac']
    day_vars[n_i]['decode_prob_cutoff'] = metadata[n_i]['params_dict']['bayes_params']['decode_prob_cutoff']
    day_vars[n_i]['bin_time'] = metadata[n_i]['params_dict']['bayes_params']['z_score_bin_time']
    day_vars[n_i]['bin_dt'] = np.ceil(day_vars[n_i]['bin_time']*1000).astype('int')
    day_vars[n_i]['num_null'] = 100 #metadata['params_dict']['num_null']
    # Import changepoint data
    day_vars[n_i]['pop_taste_cp_raster_inds'] = hf5.pull_data_from_hdf5(
        day_vars[n_i]['hdf5_dir'], 'changepoint_data', 'pop_taste_cp_raster_inds')
    day_vars[n_i]['num_pt_cp'] = day_vars[n_i]['num_cp'] + 2
    

#%% get taste response matrices

import functions.lstm_decoding_funcs as lstm

num_bins = 4

taste_unique_categories, training_matrices, training_labels = lstm.create_taste_matrices(num_neur, tastant_spike_times, segment_spike_times,
                         segment_names, segment_times, cp_raster_inds, fr_bins,
                         start_dig_in_times, pre_taste_dt, post_taste_dt, 
                         all_dig_in_names, num_bins, z_bin_dt, start_bins=0)


# import_deviations()

print("\tNow importing calculated deviations for first day")

num_seg_to_analyze = len(day_vars[0]['segments_to_analyze'])
segment_names_to_analyze = [day_vars[0]['segment_names'][i] for i in day_vars[0]['segments_to_analyze']]
segment_times_to_analyze_reshaped = [
    [day_vars[0]['segment_times'][i], day_vars[0]['segment_times'][i+1]] for i in day_vars[0]['segments_to_analyze']]
segment_spike_times_to_analyze = [day_vars[0]['segment_spike_times'][i] for i in day_vars[0]['segments_to_analyze']]
segment_names_to_analyze = segment_names_to_analyze

segment_deviations = []
for s_i in tqdm.tqdm(range(num_seg_to_analyze)):
    filepath = os.path.join(day_vars[0]['dev_dir'],segment_names_to_analyze[s_i],'deviations.json')
    with gzip.GzipFile(filepath, mode="r") as f:
        json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)
        segment_deviations.append(data)

print("\tNow pulling true deviation rasters")
#Note, these will already reflect the held units
segment_dev_rasters, segment_dev_times, segment_dev_fr_vecs, \
    segment_dev_fr_vecs_zscore, _, _ = dev_f.create_dev_rasters(num_seg_to_analyze, 
                                                        segment_spike_times_to_analyze,
                                                        np.array(segment_times_to_analyze_reshaped),
                                                        segment_deviations, day_vars[0]['pre_taste'])

# decode_groups()

print("Determine decoding groups")
#Create fr vector grouping instructions: list of epoch,taste pairs
non_none_tastes = [taste for taste in all_dig_in_names if taste[:4] != 'none']
non_none_tastes = non_none_tastes
# group_list, group_names = ddf.multiday_decode_groupings(day_vars[0]['epochs_to_analyze'],
#                                                 all_dig_in_names,
#                                                 non_none_tastes)
group_list, group_names = ddf.multiday_decode_groupings_split_identity(day_vars[0]['epochs_to_analyze'],
                                                all_dig_in_names,
                                                
                                                non_none_tastes)

# decode_zscored()

decode_dir = os.path.join(bayes_dir,'split_identity_test') #All_Neurons_Z_Scored #split_identity_test
if os.path.isdir(decode_dir) == False:
    os.mkdir(decode_dir)

#Save the group information for cross-animal use 
group_dict = dict()
for gn_i, gn in enumerate(group_names):
    group_dict[gn] = group_list[gn_i]
np.save(os.path.join(decode_dir,'group_dict.npy'),group_dict,allow_pickle=True)

z_score = True
tastant_fr_dist = tastant_fr_dist_z_pop
dev_vecs = segment_dev_fr_vecs_zscore
segment_spike_times = day_vars[0]['segment_spike_times']
segment_times = day_vars[0]['segment_times']
segment_names = day_vars[0]['segment_names']
start_dig_in_times = day_vars[0]['start_dig_in_times']
bin_dt = day_vars[0]['bin_dt']
epochs_to_analyze = day_vars[0]['epochs_to_analyze']
segments_to_analyze = day_vars[0]['segments_to_analyze']

# # decoder_accuracy()
# ddf.decoder_accuracy_tests(tastant_fr_dist, segment_spike_times, 
#                 all_dig_in_names, segment_times, segment_names, 
#                 start_dig_in_times, taste_num_deliv,
#                 group_list, group_names, non_none_tastes, 
#                 decode_dir, bin_dt, z_score, 
#                 epochs_to_analyze, segments_to_analyze)

# # decode_sliding_bins()
# ddf.decode_sliding_bins(tastant_fr_dist, segment_spike_times, all_dig_in_names, 
#                   segment_times, segment_names, start_dig_in_times, taste_num_deliv,
#                   bin_dt, group_list, group_names, non_none_tastes, decode_dir, 
#                   z_score, segments_to_analyze)

# # decode_deviations()
ddf.decode_deviations(tastant_fr_dist, tastant_spike_times,
                      segment_spike_times, all_dig_in_names, 
                      segment_times, segment_names, 
                      start_dig_in_times, taste_num_deliv, 
                      segment_dev_times, dev_vecs, 
                      bin_dt, group_list, group_names, 
                      non_none_tastes, decode_dir, z_score, 
                      segments_to_analyze)
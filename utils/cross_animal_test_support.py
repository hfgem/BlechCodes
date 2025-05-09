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
    
min_best_cutoff = 0.5


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
        
#%% correlation plots updates to rates

#Gather data
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
                        filename_corr_pop_vec_null_data = os.path.join(
                            result_dir, 'null', seg_name + '_' + taste_name + '_pop_vec.npy')
                        null_data = np.load(filename_corr_pop_vec_null_data)
                        corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['data'] = data
                        corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['null_data'] = null_data
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

#Pull unique info
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

#%% cross_dataset_dev_by_corr_cutoff - Create rate plots

# Set parameters
warnings.filterwarnings('ignore')
colors = ['green','magenta','royalblue','blueviolet','teal','deeppink', \
          'springgreen','turquoise', 'midnightblue', 'lightskyblue', \
          'palevioletred', 'darkslateblue']
corr_cutoffs = np.round(np.arange(0,1.01,0.01),2)

corr_cutoff_save = os.path.join(save_dir, 'Corr_Cutoff_Fracs')
if not os.path.isdir(corr_cutoff_save):
    os.mkdir(corr_cutoff_save)

# _____Reorganize data by unique correlation type_____
unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, min_best_cutoff,
                                                unique_corr_names, unique_given_names,
                                                unique_segment_names,unique_taste_names)

# Collect fractions by cutoff
for corr_name in unique_corr_names:
    dev_corr_frac_dict = dict() #Collect fraction of events by corr cutoff by animal
    dev_corr_rate_dict = dict() #Collect the rate of events by corr cutoff by animal
    dev_corr_total_frac_dict = dict() #Collect total fraction of events by corr cutoff (sum counts across animals and divide by total events across animals)
    dev_corr_ind_dict = dict() #Collect indices of events above corr cutoff for comparison
    for seg_name in unique_segment_names:
        dev_corr_frac_dict[seg_name] = dict()
        dev_corr_rate_dict[seg_name] = dict()
        dev_corr_total_frac_dict[seg_name] = dict()
        dev_corr_ind_dict[seg_name] = dict()
        for taste in unique_taste_names:
            dev_corr_frac_dict[seg_name][taste] = dict()
            dev_corr_rate_dict[seg_name][taste] = dict()
            dev_corr_total_frac_dict[seg_name][taste] = dict()
            dev_corr_ind_dict[seg_name][taste] = dict()
            for cp_i in range(max_epochs):
                dev_corr_frac_dict[seg_name][taste][cp_i] = []
                dev_corr_rate_dict[seg_name][taste][cp_i] = []
                dev_corr_total_frac_dict[seg_name][taste][cp_i] = []
                dev_corr_ind_dict[seg_name][taste][cp_i] = []
    
    for s_i, seg_name in enumerate(unique_segment_names):
        for t_i, taste in enumerate(unique_taste_names):
            try:
                for cp_i in range(max_epochs):
                    animal_inds = []
                    animal_counts = []
                    animal_all_dev_counts = []
                    animal_lens = []
                    for g_n in unique_given_names:
                        data = corr_data[g_n]['corr_data'][corr_name][seg_name][taste]['data']
                        null_data = corr_data[g_n]['corr_data'][corr_name][seg_name][taste]['null_data']
                        segment_times_reshaped = corr_data[g_n]['segment_times_reshaped']
                        unique_segments_ind = [i for i in range(len(corr_data[g_n]['segment_names'])) if corr_data[g_n]['segment_names'][i] == seg_name][0]
                        segment_len_sec = (segment_times_reshaped[unique_segments_ind][1]-segment_times_reshaped[unique_segments_ind][0])/1000
                        animal_lens.append(segment_len_sec)
                        num_dev, _, _ = np.shape(data)
                        animal_all_dev_counts.append(num_dev)
                        if taste == 'none':
                            taste_corr_vals = np.array([np.nanmean(data[d_i,:,:]) for d_i in range(num_dev)])
                            null_taste_corr_vals = np.array([np.nanmean(null_data[d_i,:,:]) for d_i in range(np.shape(null_data)[0])])
                            corr_cut_inds = [np.where(taste_corr_vals >= cc)[0] for cc in corr_cutoffs]
                            corr_cut_count = [len(cc_i) for cc_i in corr_cut_inds]
                            animal_counts.append(corr_cut_count)
                            animal_inds.append(corr_cut_inds)
                        else:
                            taste_corr_vals = np.nanmean(data,1) #num dev x num cp
                            corr_cut_inds = [np.where(taste_corr_vals[:,cp_i] >= cc)[0] for cc in corr_cutoffs]
                            corr_cut_count = [len(cc_i) for cc_i in corr_cut_inds]
                            animal_counts.append(corr_cut_count)
                            animal_inds.append(corr_cut_inds)
                    animal_counts = np.array(animal_counts)
                    animal_all_dev_counts = np.array(animal_all_dev_counts)
                    animal_lens = np.array(animal_lens)
                    dev_corr_ind_dict[seg_name][taste][cp_i] = animal_inds
                    dev_corr_frac_dict[seg_name][taste][cp_i] = animal_counts/np.expand_dims(animal_all_dev_counts,1)
                    dev_corr_rate_dict[seg_name][taste][cp_i] = animal_counts/np.expand_dims(animal_lens,1)
                    dev_corr_total_frac_dict[seg_name][taste][cp_i] = np.sum(animal_counts,0)/np.sum(animal_all_dev_counts)
            except:
                print("No data.")
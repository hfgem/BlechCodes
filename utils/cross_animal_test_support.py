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

#%% segment stats

from functions.cross_animal_seg_stats import seg_stat_collection

segments_to_analyze = [0,2,4]

unique_given_names, unique_segment_names, neur_rates, pop_rates, isis, \
    cvs = seg_stat_collection(all_data_dict)
        
seg_stat_save_dir = os.path.join(save_dir,'seg_stats')
if not os.path.isdir(seg_stat_save_dir):
    os.mkdir(seg_stat_save_dir)


#%% dev stats import

try:
    dict_save_dir = os.path.join(save_dir, 'dev_stats_data.npy')
    dev_stats_data = np.load(dict_save_dir,allow_pickle=True).item()
    dev_stats_data = dev_stats_data
    if not os.path.isdir(os.path.join(save_dir,'Dev_Stats')):
        os.mkdir(os.path.join(save_dir,'Dev_Stats'))
    dev_stats_results_dir = os.path.join(save_dir,'Dev_Stats')
except:
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
        dev_stats_data[data_name]['dev_stats']['dev_freq_dict'] = dict()
        for s_i, s_ind in enumerate(segments_to_analyze):
            seg_len = dev_stats_data[data_name]['segment_times_reshaped'][s_ind]
            seg_len_s = (seg_len[1] - seg_len[0])/1000
            seg_name = segment_names_to_analyze[s_i]
            dev_freq = len(dev_stats_data[data_name]['dev_stats'][stat_name][seg_name])/seg_len_s
            dev_stats_data[data_name]['dev_stats']['dev_freq_dict'][seg_name] = dev_freq
        
    dev_stats_data = dev_stats_data
    dict_save_dir = os.path.join(save_dir, 'dev_stats_data.npy')
    np.save(dict_save_dir,dev_stats_data,allow_pickle=True)
    # _____Analysis Storage Directory_____
    if not os.path.isdir(os.path.join(save_dir,'Dev_Stats')):
        os.mkdir(os.path.join(save_dir,'Dev_Stats'))
    dev_stats_results_dir = os.path.join(save_dir,'Dev_Stats')

#%%
verbose = False
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
            if verbose == True:
                print(name + " does not have segment name data for " + dev_stat_name)
unique_segment_indices = np.sort(
    np.unique(unique_segment_names, return_index=True)[1])
unique_segment_names = [unique_segment_names[i]
                        for i in unique_segment_indices]

#%%

cdf.cross_dataset_dev_stats_plots(dev_stats_data, unique_given_names, 
                                  unique_dev_stats_names, 
                                  unique_segment_names, 
                                  dev_stats_results_dir)

#%% basic stats plot
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, ks_2samp
from itertools import combinations

plot_side = np.ceil(np.sqrt(len(unique_dev_stats_names))).astype('int')
plot_inds = np.reshape(np.arange(plot_side**2),(plot_side,plot_side))
seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
f_stats, ax_stats = plt.subplots(nrows=plot_side,ncols=plot_side,
                                 figsize=(8,8),sharex=True)
for ds_i, ds in enumerate(unique_dev_stats_names):
    ds_name = (' ').join(ds.split('_')[:-1])
    ax_r, ax_c = np.where(plot_inds == ds_i)
    all_animal_stats = np.zeros((len(unique_given_names),len(unique_segment_names)))
    for gn_i, gn in enumerate(unique_given_names):
        seg_means = np.zeros(len(unique_segment_names))
        for s_i, sn in enumerate(unique_segment_names):
            all_animal_stats[gn_i,s_i] = np.nanmean(dev_stats_data[gn]['dev_stats'][ds][sn])
    ax_stats[ax_r[0],ax_c[0]].boxplot(all_animal_stats)
    for s_i in range(len(unique_segment_names)):
        scat_x = s_i+1+np.random.randn(len(unique_given_names))/10
        ax_stats[ax_r[0],ax_c[0]].scatter(scat_x,all_animal_stats[:,s_i],\
                                          color='g',alpha=0.3)
    ax_stats[ax_r[0],ax_c[0]].set_xticks(np.arange(len(unique_segment_names))+1,unique_segment_names,\
                             rotation=45)
    ax_stats[ax_r[0],ax_c[0]].set_ylabel(ds_name)
    max_y = np.nanmax(all_animal_stats)
    #ANOVA test
    result = f_oneway(*list(all_animal_stats.T))
    if result.pvalue <= 0.05:
        ax_stats[ax_r[0],ax_c[0]].set_title(ds_name + ' *ANOVA')
    else:
        ax_stats[ax_r[0],ax_c[0]].set_title(ds_name)
    #Pairwise TTest
    for sp_0, sp_1 in seg_pairs:
        result = ttest_ind(all_animal_stats[:,sp_0],all_animal_stats[:,sp_1])
        if result.pvalue <= 0.05:
            max_y += max_y*0.1
            plt.plot([sp_0+1,sp_1+1],[max_y,max_y])
            max_y += max_y*0.1
            plt.text(1+sp_0+(sp_1-sp_0)/2,max_y,'*TT')
        result = ks_2samp(all_animal_stats[:,sp_0],all_animal_stats[:,sp_1])
        if result.pvalue <= 0.05:
            max_y += max_y*0.1
            plt.plot([sp_0+1,sp_1+1],[max_y,max_y])
            max_y += max_y*0.1
            plt.text(1+sp_0+(sp_1-sp_0)/2,max_y,'*KS')
plt.tight_layout()
f_stats.savefig(os.path.join(dev_stats_results_dir,'overview_boxplots.png'))
f_stats.savefig(os.path.join(dev_stats_results_dir,'overview_boxplots.svg'))

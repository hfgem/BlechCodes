#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:37:35 2025

@author: hannahgermaine
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def cross_dataset_dev_null_plots(dev_null_data, unique_given_names, 
                                 unique_dev_null_names, unique_segment_names, 
                                 results_dir):   
    """This function is dedicated to plotting deviation statistics compared
    to null distributions across animals.
    INPUTS:
        - dev_null_data: dictionary containing data regarding deviation statistics
            organized as follows:
                - dev_null_data.keys() are the unique_given_names
                - dev_null_data[name]['dev_null'] = dict containing deviation stats
                - dev_null_data[name]['dev_null'].keys() are the unique_dev_stats_names
                - dev_null_data[name]['dev_null'][null_name] = dict containing specific statistic results
                - dev_null_data[name]['dev_null'][null_name].keys() are the unique_segment_names
                - dev_null_data[name]['dev_null'][null_name][seg_name].keys() = ['null','true','percentile']
                - dev_null_data[name]['dev_null'][null_name][seg_name]['null'] = list of [[cutoff value],[mean count],[std]]
                - dev_null_data[name]['dev_null'][null_name][seg_name]['true'] = list of [[cutoff value],[count]]
        - unique_given_names: names of imported datasets in dict
        - unique_dev_null_names: names of types of statistics
        - unique_segment_names: names of segments analyzed
        - results_dir: storage directory
    OUTPUTS:
        - Plots with statistical results
    """
    colors = ['green','royalblue','blueviolet','teal','deeppink', \
              'springgreen','turquoise', 'midnightblue', 'lightskyblue', \
              'palevioletred', 'darkslateblue']
    for dev_null_stat in unique_dev_null_names:
        max_val = 0
        f, ax = plt.subplots(ncols=len(unique_segment_names),figsize=(4*len(unique_segment_names),4))
        f_log, ax_log = plt.subplots(ncols=len(unique_segment_names),figsize=(4*len(unique_segment_names),4))
        for s_i,seg_name in enumerate(unique_segment_names):
            #Collect across animals data
            max_cutoff_val = 0
            min_cutoff_val = 100000
            combined_cutoff_values = []
            seg_animal_true_vals = []
            seg_animal_null_means = []
            seg_animal_null_stds = []
            for name in unique_given_names:
                null_dataset = dev_null_data[name]['dev_null'][dev_null_stat][seg_name]['null']
                true_dataset = dev_null_data[name]['dev_null'][dev_null_stat][seg_name]['true']
                combined_cutoff_values.append(null_dataset[0])
                seg_animal_null_means.append(null_dataset[1])
                seg_animal_null_stds.append(null_dataset[2])
                seg_animal_true_vals.append(true_dataset[1])
                if max(np.array(true_dataset[1])) > max_val:
                    max_val = max(np.array(true_dataset[1]))
                if max(np.array(null_dataset[1])) + max(np.array(null_dataset[2])) > max_val:
                    max_val = max(np.array(null_dataset[1])) + max(np.array(null_dataset[2]))
                if max(np.array(null_dataset[0])) > max_cutoff_val:
                    max_cutoff_val = max(np.array(null_dataset[0]))
                if min(np.array(null_dataset[0])) < min_cutoff_val:
                    min_cutoff_val = min(np.array(null_dataset[0]))
            #Reorganize data
            cutoff_array = np.arange(min_cutoff_val,max_cutoff_val+1)
            num_anim = len(seg_animal_true_vals)
            num_x_vals = len(cutoff_array)
            true_array = np.nan*np.ones((num_anim,num_x_vals))
            null_mean_array = np.nan*np.ones((num_anim,num_x_vals))
            null_std_array = np.nan*np.ones((num_anim,num_x_vals))
            for a_i in range(num_anim):
                anim_cutoffs = combined_cutoff_values[a_i]
                anim_true_means = seg_animal_true_vals[a_i]
                anim_null_means = seg_animal_null_means[a_i]
                anim_null_stds = seg_animal_null_stds[a_i]
                for c_i, c_val in enumerate(anim_cutoffs):
                    c_array_ind = np.where(cutoff_array == c_val)[0]
                    true_array[a_i,c_array_ind] = anim_true_means[c_i]
                    null_mean_array[a_i,c_array_ind] = anim_null_means[c_i]
                    null_std_array[a_i,c_array_ind] = anim_null_stds[c_i]
            #Generate null data from mean and std data
            new_null_mean_array = np.nan*np.ones(len(cutoff_array))
            new_null_std_array = np.nan*np.ones(len(cutoff_array))
            for c_i,c_val in enumerate(cutoff_array):
                collection = []
                for a_i in range(num_anim):
                    if ~np.isnan(null_mean_array[a_i][c_i]) * ~np.isnan(null_std_array[a_i][c_i]):
                        collection.extend(list(np.random.normal(null_mean_array[a_i][c_i],\
                                                                null_std_array[a_i][c_i],\
                                                                    50)))
                new_null_mean_array[c_i] = np.nanmean(collection)
                new_null_std_array[c_i] = np.nanstd(collection)
            true_mean_array = np.nanmean(true_array,0)
            true_std_array = np.nanstd(true_array,0)
            #Plot normal
            ax[s_i].plot(cutoff_array,true_mean_array,color=colors[0],label='True')
            true_min = true_mean_array-true_std_array
            true_min[true_min < 0] = 0
            true_max = true_mean_array+true_std_array
            ax[s_i].fill_between(cutoff_array,true_max,\
                                 true_min,color=colors[0],\
                                     alpha=0.3,label='True Std')
            ax[s_i].plot(cutoff_array,new_null_mean_array,color=colors[1],label='Null')
            null_min = new_null_mean_array-new_null_std_array
            null_min[null_min < 0] = 0
            null_max = new_null_mean_array+new_null_std_array
            ax[s_i].fill_between(cutoff_array,null_min,\
                                 null_max,color=colors[1],\
                                     alpha=0.3,label='Null Std')
            ax[s_i].legend(loc='upper right')
            ax[s_i].set_xlabel(dev_null_stat + ' cutoffs')
            ax[s_i].set_ylabel('Count')
            ax[s_i].set_title(seg_name)
            #Plot log scale
            ax_log[s_i].plot(cutoff_array,np.log(true_mean_array),color=colors[0],label='True')
            log_min = np.log(true_min)
            log_max = np.log(true_max)
            ax_log[s_i].fill_between(cutoff_array,np.log(true_min),\
                                 np.log(true_max),color=colors[0],\
                                     alpha=0.3,label='True Std')
            ax_log[s_i].plot(cutoff_array,np.log(new_null_mean_array),color=colors[1],label='Null')
            ax_log[s_i].fill_between(cutoff_array,np.log(null_min),\
                                 np.log(null_max),color=colors[1],\
                                     alpha=0.3,label='Null Std')
            ax_log[s_i].legend(loc='upper right')
            ax_log[s_i].set_xlabel(dev_null_stat + ' cutoffs')
            ax_log[s_i].set_ylabel('Log(Count)')
            ax_log[s_i].set_title(seg_name)
        for s_i in range(len(unique_segment_names)):
            ax[s_i].set_ylim([-1/10*max_val,max_val+1/10*max_val])
            ax_log[s_i].set_ylim([0,np.log(max_val+1/10*max_val)])
        f.suptitle(dev_null_stat)
        plt.tight_layout()
        f.savefig(os.path.join(results_dir,dev_null_stat+'_combined.png'))
        f.savefig(os.path.join(results_dir,dev_null_stat+'_combined.svg'))
        plt.close(f)
        f_log.suptitle(dev_null_stat)
        plt.tight_layout()
        f_log.savefig(os.path.join(results_dir,dev_null_stat+'_combined_log.png'))
        f_log.savefig(os.path.join(results_dir,dev_null_stat+'_combined_log.svg'))
        plt.close(f_log)
        
def null_cutoff_stats_combined(dev_null_data, unique_given_names, 
                                 unique_dev_null_names, unique_segment_names, 
                                 results_dir):
    
    
    
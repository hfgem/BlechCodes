#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:33:13 2024

@author: Hannah Germaine

A collection of functions for comparing different datasets against each other 
in their correlation trends.
"""

import os
import warnings
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu, percentileofscore
from scipy.signal import savgol_filter
from matplotlib import colormaps


def cross_dataset_dev_freq(corr_data, min_best_cutoff, unique_given_names, unique_corr_names,
                                 unique_segment_names, unique_taste_names, save_dir):
    """This function collects deviation correlation statistics across different
    datasets and plots the frequencies of deviation correlation events for 
    different conditions
    INPUTS:
            - corr_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    corr_data[name] = dictionary of dataset data
                    corr_data[name]['corr_data'] = dict of length #correlation types
                    corr_data[name]['corr_data'][corr_name] = dict of length #segments
                    corr_data[name]['corr_data'][corr_name][seg_name] = dict of length #tastes
                    corr_data[name]['corr_data'][corr_name][seg_name][taste_name]['data'] = numpy 
                            array of population average vector correlations [num_dev x num_trials x num_epochs]
            - unique_given_names: unique names of datasets
            - unique_corr_names: unique names of correlation analysis types
            - unique_segment_names: unique names of experimental segments
            - unique_taste_names: unique names of tastants
            - save_dir: where to save resulting plots
    OUTPUTS: plots and statistical significance tests comparing taste rates to each other
    """
    # Set parameters
    warnings.filterwarnings('ignore')

    max_epochs = 0
    for corr_name in unique_corr_names:
        for seg_name in unique_segment_names:
            for g_n in unique_given_names:
                for taste_name in unique_taste_names:
                    try:
                        num_cp = corr_data[g_n]['corr_data'][corr_name][seg_name][taste_name]['num_cp']
                        if num_cp > max_epochs:
                            max_epochs = num_cp
                    except:
                        max_epochs = max_epochs

    # Plot all combinations
    for corr_name in unique_corr_names:
        dev_freq_dict = dict() #Overall frequency of deviation events
        best_dev_freq_dict = dict() #Frequency of specialized deviation events
        significance_storage = dict()
        max_freq = 0
        f_all = plt.figure(figsize=(5,5))
        for seg_name in unique_segment_names:
            dev_freq_dict[seg_name] = []
            best_dev_freq_dict[seg_name] = dict()
            for taste in unique_taste_names:
                best_dev_freq_dict[seg_name][taste] = dict()
                for cp_i in range(max_epochs):
                    best_dev_freq_dict[seg_name][taste][cp_i] = []
            
            for g_n in unique_given_names:
                seg_ind = np.where([corr_data[g_n]['segment_names'][s_i] == seg_name for s_i in range(
                    len(corr_data[g_n]['segment_names']))])[0][0]
                seg_len = corr_data[g_n]['segment_times_reshaped'][seg_ind][1] - \
                    corr_data[g_n]['segment_times_reshaped'][seg_ind][0]
                seg_len_s = seg_len/1000
                try:
                    taste_names = list(np.intersect1d(list(corr_data[g_n]['corr_data'][corr_name][seg_name].keys()),unique_taste_names))
                    data = corr_data[g_n]['corr_data'][corr_name][seg_name]
                    one_taste = taste_names[0]
                    num_cp = data[one_taste]['num_cp']
                    num_dev = data[one_taste]['num_dev']
                    #Pull out frequency
                    dev_freq_dict[seg_name].extend([num_dev/seg_len_s])
                    #Pull out best designations
                    best_data = corr_data[g_n]['corr_data'][corr_name][seg_name]['best'] #num_dev x 2 (col 1 = taste, col2 = epoch)
                    
                    #Now store the correlation values for each dev event by taste and epoch
                    for t_i, taste in enumerate(taste_names):
                        for cp_i in range(num_cp):
                            dev_inds = np.where((best_data[:,0] == t_i)*(best_data[:,1] == cp_i))[0]
                            best_dev_freq_dict[seg_name][taste][cp_i].extend([len(dev_inds)/seg_len_s])
                except:
                    print("No data.")
        #____DEVIATION RATES____
        # For each segment create boxplots of the rate distributions
        plt.axhline(0, label='_', alpha=0.2, color='k', linestyle='dashed')
        all_segment_data = []
        all_segment_data_indices = []
        for s_i,seg_name in enumerate(unique_segment_names):
            seg_frequencies = dev_freq_dict[seg_name]
            plt.scatter(np.random.normal(s_i+1, 0.04,size=len(seg_frequencies)),
                          seg_frequencies, color='g',alpha=0.2)
            plt.boxplot([seg_frequencies], positions=[
                                s_i+1], sym='', meanline=True, medianprops=dict(linestyle='-', color='blue'), showcaps=True, showbox=True)
            if len(seg_frequencies) > 0:
                if np.nanmax(seg_frequencies) > max_freq:
                    max_freq = np.nanmax(seg_frequencies)
            all_segment_data.append(seg_frequencies)
            all_segment_data_indices.extend([s_i])
        plt.xticks(np.arange(1, len(unique_segment_names)+1), unique_segment_names)
        plt.xlabel('Segment')
        plt.ylabel('Frequency (Hz)')
        # Test pairwise significance #TTEST
        pair_nums = list(combinations(all_segment_data_indices, 2))
        for pair_i, pair in enumerate(pair_nums):
            data_1 = all_segment_data[pair[0]]
            data_2 = all_segment_data[pair[1]]
            if (len(data_1) > 0)*(len(data_2) > 0):
                result = ttest_ind(data_1, data_2)
                if result[1] <= 0.05:
                    ind_1 = pair[0] + 1
                    ind_2 = pair[1] + 1
                    significance_storage[pair_i] = dict()
                    significance_storage[pair_i]['ind_1'] = ind_1
                    significance_storage[pair_i]['ind_2'] = ind_2
        #Plot pairwise significance
        step = max_freq/10
        sig_height = max_freq + step
        for sp_i in list(significance_storage.keys()):
            ind_1 = significance_storage[sp_i]['ind_1']
            ind_2 = significance_storage[sp_i]['ind_2']
            plt.plot([ind_1, ind_2], [
                             sig_height, sig_height], color='k', linestyle='solid')
            plt.plot([ind_1, ind_1], [
                             sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
            plt.plot([ind_2, ind_2], [
                             sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
            plt.text(ind_1 + (ind_2-ind_1)/2, sig_height + step/2,
                             '*', horizontalalignment='center', verticalalignment='center')
            sig_height += step
        #Finish up and save
        plt.suptitle(corr_name)
        plt.tight_layout()
        f_all.savefig(os.path.join(save_dir, corr_name) + '.png')
        f_all.savefig(os.path.join(save_dir, corr_name) + '.svg')
        plt.close(f_all)
        #____BEST DEVIATION RATES____
        #best_dev_freq_dict[seg_name][taste][cp_i] = list of length num animals in Hz
        f_best_taste, ax_best_taste = plt.subplots(ncols = len(unique_segment_names), \
                                                    nrows = max_epochs, figsize=(len(unique_segment_names)*4,max_epochs*4), \
                                                        sharex = True, sharey = True) #x-axis is tastes
        for s_i,seg_name in enumerate(unique_segment_names):
            for cp_i in range(max_epochs):
                all_taste_freq = []
                all_taste_freq_ind = []
                max_freq = 0
                # f_best_taste = plt.figure(figsize=(5,5))
                #Scatter/boxplot the frequency data
                for t_i,t_name in enumerate(unique_taste_names):
                    try:
                        taste_freq = best_dev_freq_dict[seg_name][t_name][cp_i]
                        all_taste_freq.append(taste_freq)
                        all_taste_freq_ind.extend([t_i])
                        ax_best_taste[cp_i,s_i].scatter(np.random.normal(t_i+1, 0.04,size=len(taste_freq)),
                                  taste_freq, color='g',alpha=0.2)
                        ax_best_taste[cp_i,s_i].boxplot([taste_freq], positions=[
                                        t_i+1], sym='', meanline=True, medianprops=dict(linestyle='-', color='blue'), \
                                showcaps=True, showbox=True)
                        # plt.scatter(np.random.normal(t_i+1, 0.04,size=len(taste_freq)),
                        #           taste_freq, color='g',alpha=0.2)
                        # plt.boxplot([taste_freq], positions=[
                        #                 t_i+1], sym='', meanline=True, medianprops=dict(linestyle='-', color='blue'), \
                        #         showcaps=True, showbox=True)
                        if np.nanmax(taste_freq) > max_freq:
                            max_freq = np.nanmax(taste_freq)
                    except:
                        print("No data.")
                ax_best_taste[cp_i,s_i].set_xticks(np.arange(1,len(unique_taste_names)+1),unique_taste_names)
                if s_i == 0:
                    ax_best_taste[cp_i,s_i].set_ylabel('Epoch ' + str(cp_i) + '\nFrequency (Hz)')
                if cp_i == 0:
                    ax_best_taste[cp_i,s_i].set_title(seg_name)
                # plt.xticks(np.arange(1,len(unique_taste_names)+1),unique_taste_names)
                # plt.ylabel('Epoch ' + str(cp_i) + '\nFrequency (Hz)')
                # plt.title(seg_name)
                #Test pairwise significance
                significance_storage = dict()
                pair_nums = list(combinations(all_taste_freq_ind, 2))
                for pair_i, pair in enumerate(pair_nums):
                    data_1 = all_taste_freq[pair[0]]
                    data_2 = all_taste_freq[pair[1]]
                    if (len(data_1) > 0)*(len(data_2) > 0):
                        result = ttest_ind(data_1, data_2)
                        if result[1] <= 0.05:
                            ind_1 = pair[0] + 1
                            ind_2 = pair[1] + 1
                            significance_storage[pair_i] = dict()
                            significance_storage[pair_i]['ind_1'] = ind_1
                            significance_storage[pair_i]['ind_2'] = ind_2
                #Plot pairwise significance
                step = max_freq/10
                sig_height = max_freq + step
                for sp_i in list(significance_storage.keys()):
                    ind_1 = significance_storage[sp_i]['ind_1']
                    ind_2 = significance_storage[sp_i]['ind_2']
                    ax_best_taste[cp_i,s_i].plot([ind_1, ind_2], [
                                      sig_height, sig_height], color='k', linestyle='solid')
                    ax_best_taste[cp_i,s_i].plot([ind_1, ind_1], [
                                      sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                    ax_best_taste[cp_i,s_i].plot([ind_2, ind_2], [
                                      sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                    ax_best_taste[cp_i,s_i].text(ind_1 + (ind_2-ind_1)/2, sig_height + step/2,
                                      '*', horizontalalignment='center', verticalalignment='center')
                    # plt.plot([ind_1, ind_2], [
                    #                  sig_height, sig_height], color='k', linestyle='solid')
                    # plt.plot([ind_1, ind_1], [
                    #                  sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                    # plt.plot([ind_2, ind_2], [
                    #                  sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                    # plt.text(ind_1 + (ind_2-ind_1)/2, sig_height + step/2,
                    #                  '*', horizontalalignment='center', verticalalignment='center')
                    sig_height += step
                # plt.tight_layout()
                # f_best_taste.savefig(os.path.join(save_dir, corr_name) + '_best_taste_'+str(cp_i)+'_'+seg_name+'.png')
                # f_best_taste.savefig(os.path.join(save_dir, corr_name) + '_best_taste_'+str(cp_i)+'_'+seg_name+'.svg')
        plt.suptitle(corr_name + '\nBest Deviation Rates')
        plt.tight_layout()
        f_best_taste.savefig(os.path.join(save_dir, corr_name) + '_best_taste.png')
        f_best_taste.savefig(os.path.join(save_dir, corr_name) + '_best_taste.svg')
        plt.close(f_best_taste)
        
        f_best_seg, ax_best_seg = plt.subplots(ncols = len(unique_taste_names), \
                                               nrows = max_epochs, figsize=(4*len(unique_taste_names),4*max_epochs), \
                                                   sharex = True, sharey = True) #x-axis is segments
        for t_i, taste in enumerate(unique_taste_names):
            for cp_i in range(max_epochs):
                all_seg_freq = []
                all_seg_freq_ind = []
                max_freq = 0
                #Scatter/boxplot the frequency data
                for s_i,seg_name in enumerate(unique_segment_names):
                    try:
                        seg_freq = best_dev_freq_dict[seg_name][taste][cp_i]
                        all_seg_freq.append(seg_freq)
                        all_seg_freq_ind.extend([s_i])
                        ax_best_seg[cp_i,t_i].scatter(np.random.normal(s_i+1, 0.04,size=len(seg_freq)),
                                  seg_freq, color='g',alpha=0.2)
                        ax_best_seg[cp_i,t_i].boxplot([seg_freq], positions=[
                                        s_i+1], sym='', meanline=True, medianprops=dict(linestyle='-', color='blue'), \
                                showcaps=True, showbox=True)
                        if np.nanmax(seg_freq) > max_freq:
                            max_freq = np.nanmax(seg_freq)
                    except:
                        print("No data.")
                ax_best_seg[cp_i,t_i].set_xticks(np.arange(1,len(unique_segment_names)+1),unique_segment_names)
                if t_i == 0:
                    ax_best_seg[cp_i,t_i].set_ylabel('Epoch ' + str(cp_i) + '\nFrequency (Hz)')
                if cp_i == 0:
                    ax_best_seg[cp_i,t_i].set_title(taste)
                #Test pairwise significance
                significance_storage = dict()
                pair_nums = list(combinations(all_seg_freq_ind, 2))
                for pair_i, pair in enumerate(pair_nums):
                    data_1 = all_seg_freq[pair[0]]
                    data_2 = all_seg_freq[pair[1]]
                    if (len(data_1) > 0)*(len(data_2) > 0):
                        result = ttest_ind(data_1, data_2)
                        if result[1] <= 0.05:
                            ind_1 = pair[0] + 1
                            ind_2 = pair[1] + 1
                            significance_storage[pair_i] = dict()
                            significance_storage[pair_i]['ind_1'] = ind_1
                            significance_storage[pair_i]['ind_2'] = ind_2
                #Plot pairwise significance
                step = max_freq/10
                sig_height = max_freq + step
                for sp_i in list(significance_storage.keys()):
                    ind_1 = significance_storage[sp_i]['ind_1']
                    ind_2 = significance_storage[sp_i]['ind_2']
                    ax_best_seg[cp_i,t_i].plot([ind_1, ind_2], [
                                     sig_height, sig_height], color='k', linestyle='solid')
                    ax_best_seg[cp_i,t_i].plot([ind_1, ind_1], [
                                     sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                    ax_best_seg[cp_i,t_i].plot([ind_2, ind_2], [
                                     sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                    ax_best_seg[cp_i,t_i].text(ind_1 + (ind_2-ind_1)/2, sig_height + step/2,
                                     '*', horizontalalignment='center', verticalalignment='center')
                    sig_height += step
        plt.suptitle(corr_name + '\nBest Deviation Rates')
        plt.tight_layout()
        f_best_seg.savefig(os.path.join(save_dir, corr_name) + '_best_segment.png')
        f_best_seg.savefig(os.path.join(save_dir, corr_name) + '_best_segment.svg')
        plt.close(f_best_seg)
        
def cross_dataset_dev_by_corr_cutoff(corr_data, min_best_cutoff, unique_given_names, unique_corr_names,
                                 unique_segment_names, unique_taste_names, save_dir):
    """This function collects deviation correlation statistics across different
    datasets and plots the frequencies of deviation correlation events above
    a correlation cutoff for different conditions
    INPUTS:
            - corr_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    corr_data[name] = dictionary of dataset data
                    corr_data[name]['corr_data'] = dict of length #correlation types
                    corr_data[name]['corr_data'][corr_name] = dict of length #segments
                    corr_data[name]['corr_data'][corr_name][seg_name] = dict of length #tastes
                    corr_data[name]['corr_data'][corr_name][seg_name][taste_name]['data'] = numpy 
                            array of population average vector correlations [num_dev x num_trials x num_epochs]
            - unique_given_names: unique names of datasets
            - unique_corr_names: unique names of correlation analysis types
            - unique_segment_names: unique names of experimental segments
            - unique_taste_names: unique names of tastants
            - save_dir: where to save resulting plots
    OUTPUTS: plots and statistical significance tests comparing taste rates to each other
    """
    # Set parameters
    warnings.filterwarnings('ignore')
    colors = ['green','magenta','royalblue','blueviolet','teal','deeppink', \
              'springgreen','turquoise', 'midnightblue', 'lightskyblue', \
              'palevioletred', 'darkslateblue']
    corr_cutoffs = np.round(np.arange(0,1.01,0.01),2)
    num_null_samples = 100
    
    corr_cutoff_save = os.path.join(save_dir, 'Corr_Cutoff_Fracs')
    if not os.path.isdir(corr_cutoff_save):
        os.mkdir(corr_cutoff_save)
        
    corr_cutoff_indiv_save = os.path.join(corr_cutoff_save, 'Indiv_Combinations')
    if not os.path.isdir(corr_cutoff_indiv_save):
        os.mkdir(corr_cutoff_indiv_save)
    
    # _____Reorganize data by unique correlation type_____
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, min_best_cutoff,
                                                    unique_corr_names, unique_given_names,
                                                    unique_segment_names,unique_taste_names)
    
    # Collect fractions by cutoff
    for corr_name in unique_corr_names:
        try:
            dev_corr_ind_dict = np.load(os.path.join(corr_cutoff_save,corr_name+'_dev_corr_ind_dict.npy'),allow_pickle=True).item()
            dev_corr_frac_dict = np.load(os.path.join(corr_cutoff_save,corr_name+'_dev_corr_frac_dict.npy'),allow_pickle=True).item()
            dev_corr_rate_dict = np.load(os.path.join(corr_cutoff_save,corr_name+'_dev_corr_rate_dict.npy'),allow_pickle=True).item()
            dev_corr_total_frac_dict = np.load(os.path.join(corr_cutoff_save,corr_name+'_dev_corr_total_frac_dict.npy'),allow_pickle=True).item()
        except:
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
                        dev_corr_rate_dict[seg_name][taste][cp_i] = dict()
                        dev_corr_total_frac_dict[seg_name][taste][cp_i] = []
                        dev_corr_ind_dict[seg_name][taste][cp_i] = []
                
            for s_i, seg_name in enumerate(unique_segment_names):
                for t_i, taste in enumerate(unique_taste_names):
                    try:
                        for cp_i in range(max_epochs):
                            animal_inds = []
                            animal_counts = []
                            animal_null_counts = []
                            animal_all_dev_counts = []
                            animal_lens = []
                            null_animal_lens = []
                            for g_n in unique_given_names:
                                data = corr_data[g_n]['corr_data'][corr_name][seg_name][taste]['data']
                                null_data = corr_data[g_n]['corr_data'][corr_name][seg_name][taste]['null_data']
                                num_null = corr_data[g_n]['num_null']
                                segment_times_reshaped = corr_data[g_n]['segment_times_reshaped']
                                unique_segments_ind = [i for i in range(len(corr_data[g_n]['segment_names'])) if corr_data[g_n]['segment_names'][i] == seg_name][0]
                                segment_len_sec = (segment_times_reshaped[unique_segments_ind][1]-segment_times_reshaped[unique_segments_ind][0])/1000
                                animal_lens.append(segment_len_sec)
                                num_dev, _, _ = np.shape(data)
                                animal_all_dev_counts.append(num_dev)
                                null_sample_counts = []
                                if taste == 'none':
                                    #True values
                                    taste_corr_vals = np.array([np.nanmean(data[d_i,:,:]) for d_i in range(num_dev)])
                                    corr_cut_inds = [np.where(taste_corr_vals >= cc)[0] for cc in corr_cutoffs]
                                    corr_cut_count = [len(cc_i) for cc_i in corr_cut_inds]
                                    #Null values
                                    null_taste_corr_vals = np.array([np.nanmean(null_data[d_i,:,:]) for d_i in range(np.shape(null_data)[0])])
                                    for ns_i in range(num_null_samples): #Bootstrap to find counts
                                        null_sample_vals = random.sample(list(null_taste_corr_vals),num_dev)
                                        avg_null_sample_corr_cut_count = [len(np.where(null_sample_vals >= cc)[0])/num_null for cc in corr_cutoffs]
                                        null_sample_counts.append(avg_null_sample_corr_cut_count)
                                    animal_counts.append(corr_cut_count)
                                    animal_inds.append(corr_cut_inds)
                                else:
                                    #True values
                                    taste_corr_vals = np.nanmean(data,1) #num dev x num cp
                                    corr_cut_inds = [np.where(taste_corr_vals[:,cp_i] >= cc)[0] for cc in corr_cutoffs]
                                    corr_cut_count = [len(cc_i) for cc_i in corr_cut_inds]
                                    #Null values
                                    null_taste_corr_vals = np.nanmean(null_data,1)[:,cp_i] #num null dev
                                    for ns_i in range(num_null_samples): #Bootstrap to find counts
                                        null_sample_vals = random.sample(list(null_taste_corr_vals),num_dev)
                                        avg_null_corr_cut_count = [len(np.where(null_sample_vals >= cc)[0])/num_null for cc in corr_cutoffs]
                                        null_sample_counts.append(avg_null_corr_cut_count)
                                    animal_counts.append(corr_cut_count)
                                    animal_inds.append(corr_cut_inds)
                                animal_null_counts.extend(null_sample_counts)
                                null_animal_lens.extend(list(segment_len_sec*np.ones(len(null_sample_counts))))
                            animal_counts = np.array(animal_counts)
                            animal_null_counts = np.array(animal_null_counts)
                            animal_all_dev_counts = np.array(animal_all_dev_counts)
                            animal_lens = np.array(animal_lens)
                            null_animal_lens = np.array(null_animal_lens)
                            dev_corr_ind_dict[seg_name][taste][cp_i] = animal_inds
                            dev_corr_frac_dict[seg_name][taste][cp_i] = animal_counts/np.expand_dims(animal_all_dev_counts,1)
                            dev_corr_rate_dict[seg_name][taste][cp_i]['true'] = animal_counts/np.expand_dims(animal_lens,1)
                            dev_corr_rate_dict[seg_name][taste][cp_i]['null'] = animal_null_counts/np.expand_dims(null_animal_lens,1)
                            dev_corr_total_frac_dict[seg_name][taste][cp_i] = np.sum(animal_counts,0)/np.sum(animal_all_dev_counts)
                    except:
                        print("No data.")
            
            #Save dicts
            np.save(os.path.join(corr_cutoff_save,corr_name+'_dev_corr_ind_dict.npy'),dev_corr_ind_dict,allow_pickle=True)
            np.save(os.path.join(corr_cutoff_save,corr_name+'_dev_corr_frac_dict.npy'),dev_corr_frac_dict,allow_pickle=True)
            np.save(os.path.join(corr_cutoff_save,corr_name+'_dev_corr_rate_dict.npy'),dev_corr_rate_dict,allow_pickle=True)
            np.save(os.path.join(corr_cutoff_save,corr_name+'_dev_corr_total_frac_dict.npy'),dev_corr_total_frac_dict,allow_pickle=True)
            
        #Create rate plots
        rate_plots(dev_corr_rate_dict, corr_cutoffs, unique_segment_names, 
                       max_epochs, unique_taste_names, corr_name, corr_cutoff_save, 
                       corr_cutoff_indiv_save)
        
        #Plot tastes against each other
        f_cc_taste, ax_cc_taste = plt.subplots(nrows = len(unique_segment_names),\
                                               ncols = max_epochs, sharex = True,\
                                               sharey = True, figsize = (8,8))
        f_cc_taste_frac, ax_cc_taste_frac = plt.subplots(nrows = len(unique_segment_names),\
                                               ncols = max_epochs, sharex = True,\
                                               sharey = True, figsize = (8,8))
        for s_i, seg_name in enumerate(unique_segment_names):
            for cp_i in range(max_epochs):
                taste_inds = []
                for t_i, taste in enumerate(unique_taste_names):
                    taste_mean = np.nanmean(dev_corr_frac_dict[seg_name][taste][cp_i],0)
                    taste_inds.append(dev_corr_ind_dict[seg_name][taste][cp_i])
                    # taste_std = np.nanstd(dev_corr_frac_dict[seg_name][taste][cp_i],0)
                    # taste_mean_minus_std = taste_mean - taste_std
                    # taste_mean_minus_std[np.where(taste_mean_minus_std < 0)[0]] = 0
                    # ax_cc_epoch[s_i,cp_i].fill_between(corr_cutoffs,taste_mean+taste_std,\
                    #                                    taste_mean_minus_std,color = colors[t_i],
                    #                                    alpha=0.3,label='_')
                    ax_cc_taste[s_i,cp_i].plot(corr_cutoffs,taste_mean,\
                                               c=colors[t_i],label=taste)
                    try:
                        zero_val = corr_cutoffs[np.where(taste_mean <= 0)[0][0]]
                    except:
                        zero_val = 1
                    ax_cc_taste[s_i,cp_i].axvline(zero_val,c = colors[t_i],\
                                                      linestyle='dashed',alpha=0.3,\
                                                          label='_')
                    ax_cc_taste[s_i,cp_i].text(zero_val,0.1,str(zero_val),rotation=90)
                if s_i == 0:
                    ax_cc_taste[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                    ax_cc_taste_frac[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                if cp_i == 0:
                    ax_cc_taste[s_i,cp_i].set_ylabel(seg_name + '\nMean Fraction of Events')
                    ax_cc_taste_frac[s_i,cp_i].set_ylabel(seg_name + '\nMean Fraction of Events')
                water_fracs = []
                saccharin_fracs = []
                none_fracs = []
                overlap_fracs = []
                for cc_i, cc in enumerate(corr_cutoffs):
                    water_count = 0
                    saccharin_count = 0
                    none_count = 0
                    overlap_count = 0
                    for gn_i, gn in enumerate(unique_given_names):
                        animal_inds = []
                        for t_i, taste in enumerate(['water','saccharin','none']):
                            animal_inds.append(taste_inds[t_i][gn_i][cc_i])
                        water_count += len(np.setdiff1d(animal_inds[0],\
                                                        np.concatenate((np.array(animal_inds[1]),\
                                                                        np.array(animal_inds[2])))))
                        saccharin_count += len(np.setdiff1d(animal_inds[1],\
                                                        np.concatenate((np.array(animal_inds[0]),\
                                                                        np.array(animal_inds[2])))))
                        none_count += len(np.setdiff1d(animal_inds[2],\
                                                        np.concatenate((np.array(animal_inds[0]),\
                                                                        np.array(animal_inds[1])))))
                        overlap_count += len(np.intersect1d(np.intersect1d(animal_inds[0],animal_inds[1]),animal_inds[2]))
                    total_count = np.sum([water_count,saccharin_count,none_count,overlap_count])
                    water_fracs.append(water_count/total_count)
                    saccharin_fracs.append(saccharin_count/total_count)
                    none_fracs.append(none_count/total_count)
                    overlap_fracs.append(overlap_count/total_count)
                ax_cc_taste_frac[s_i,cp_i].plot(corr_cutoffs,water_fracs,\
                                           color='royalblue',label='Water Fraction')
                ax_cc_taste_frac[s_i,cp_i].plot(corr_cutoffs,saccharin_fracs,\
                                           color='magenta',label='Saccharin Fraction')
                ax_cc_taste_frac[s_i,cp_i].plot(corr_cutoffs,none_fracs,\
                                           color='green',label='None Fraction')
                ax_cc_taste_frac[s_i,cp_i].plot(corr_cutoffs,overlap_fracs,\
                                           color='orange',label='Overlap Fraction')
        ax_cc_taste[0,0].set_xlim([-0.01,1.01])
        ax_cc_taste[0,0].legend(loc='upper left')
        f_cc_taste.suptitle(corr_name + ' Mean Fraction of Events by Cutoff')
        plt.tight_layout()
        save_name = corr_name + '_mean_frac_by_cutoff_comp_tastes'
        f_cc_taste.savefig(os.path.join(corr_cutoff_save,save_name + '.png'))
        f_cc_taste.savefig(os.path.join(corr_cutoff_save,save_name + '.svg'))
        plt.close(f_cc_taste)
        ax_cc_taste_frac[0,0].set_xlim([-0.01,1.01])
        ax_cc_taste_frac[0,0].legend(loc='upper left')
        f_cc_taste_frac.suptitle(corr_name + ' Fraction of Unique Taste Events by Cutoff')
        plt.tight_layout()
        save_name = corr_name + '_unique_frac_by_cutoff_comp_tastes'
        f_cc_taste_frac.savefig(os.path.join(corr_cutoff_save,save_name + '.png'))
        f_cc_taste_frac.savefig(os.path.join(corr_cutoff_save,save_name + '.svg'))
        plt.close(f_cc_taste_frac)
        
        #Plot epochs against each other
        f_cc_epoch, ax_cc_epoch = plt.subplots(nrows = len(unique_segment_names),\
                                               ncols = len(unique_taste_names), sharex = True,\
                                               sharey = True, figsize = (8,8))
        f_cc_epoch_frac, ax_cc_epoch_frac = plt.subplots(nrows = len(unique_segment_names),\
                                               ncols = len(unique_taste_names), sharex = True,\
                                               sharey = True, figsize = (8,8))
        for s_i, seg_name in enumerate(unique_segment_names):
            for t_i, taste in enumerate(unique_taste_names):
                epoch_inds = []
                for cp_i in range(max_epochs):
                    epoch_mean = np.nanmean(dev_corr_frac_dict[seg_name][taste][cp_i],0)
                    epoch_inds.append(dev_corr_ind_dict[seg_name][taste][cp_i])
                    # epoch_std = np.nanstd(dev_corr_frac_dict[seg_name][taste][cp_i],0)
                    # epoch_mean_minus_std = epoch_mean - epoch_std
                    # epoch_mean_minus_std[np.where(epoch_mean_minus_std < 0)[0]] = 0
                    # ax_cc_epoch[s_i,t_i].fill_between(corr_cutoffs,epoch_mean+epoch_std,\
                    #                                    epoch_mean_minus_std,color = colors[cp_i],
                    #                                    alpha=0.3,label='_')
                    ax_cc_epoch[s_i,t_i].plot(corr_cutoffs,epoch_mean,\
                                               c=colors[cp_i],label='Epoch ' + str(cp_i))
                    try:
                        zero_val = corr_cutoffs[np.where(epoch_mean == 0)[0][0]]
                        
                    except:
                        zero_val = 1
                    ax_cc_epoch[s_i,t_i].axvline(zero_val,c = colors[cp_i],\
                                              linestyle='dashed',alpha=0.3,\
                                                  label='_')
                    ax_cc_epoch[s_i,t_i].text(zero_val,0.1,str(zero_val),rotation=90)
                if s_i == 0:
                    ax_cc_epoch[s_i,t_i].set_title(taste)
                    ax_cc_epoch_frac[s_i,t_i].set_title(taste)
                if t_i == 0:
                    ax_cc_epoch[s_i,t_i].set_ylabel(seg_name + '\nMean Fraction of Events')
                    ax_cc_epoch_frac[s_i,t_i].set_ylabel(seg_name + '\nMean Fraction of Events')
                epoch_0_fracs = []
                epoch_1_fracs = []
                epoch_2_fracs = []
                overlap_fracs = []
                for cc_i, cc in enumerate(corr_cutoffs):
                    e_0_count = 0
                    e_1_count = 0
                    e_2_count = 0
                    overlap_count = 0
                    for gn_i, gn in enumerate(unique_given_names):
                        animal_inds = []
                        for cp_i in range(max_epochs):
                            animal_inds.append(epoch_inds[cp_i][gn_i][cc_i])
                        e_0_count += len(np.setdiff1d(animal_inds[0],\
                                                        np.concatenate((np.array(animal_inds[1]),\
                                                                        np.array(animal_inds[2])))))
                        e_1_count += len(np.setdiff1d(animal_inds[1],\
                                                        np.concatenate((np.array(animal_inds[0]),\
                                                                        np.array(animal_inds[2])))))
                        e_2_count += len(np.setdiff1d(animal_inds[2],\
                                                        np.concatenate((np.array(animal_inds[0]),\
                                                                        np.array(animal_inds[1])))))
                        overlap_count += len(np.intersect1d(np.intersect1d(animal_inds[0],animal_inds[1]),animal_inds[2]))
                    total_count = np.sum([e_0_count,e_1_count,e_2_count,overlap_count])
                    epoch_0_fracs.append(e_0_count/total_count)
                    epoch_1_fracs.append(e_1_count/total_count)
                    epoch_2_fracs.append(e_2_count/total_count)
                    overlap_fracs.append(overlap_count/total_count)
                ax_cc_epoch_frac[s_i,t_i].plot(corr_cutoffs,epoch_0_fracs,\
                                           color='royalblue',label='Epoch 0 Fraction')
                ax_cc_epoch_frac[s_i,t_i].plot(corr_cutoffs,epoch_1_fracs,\
                                           color='magenta',label='Epoch 1 Fraction')
                ax_cc_epoch_frac[s_i,t_i].plot(corr_cutoffs,epoch_2_fracs,\
                                           color='green',label='Epoch 2 Fraction')
                ax_cc_epoch_frac[s_i,t_i].plot(corr_cutoffs,overlap_fracs,\
                                           color='orange',label='Overlap Fraction')
        ax_cc_epoch[0,0].legend(loc='upper left')
        f_cc_epoch.suptitle(corr_name + ' Mean Fraction of Events by Cutoff')
        plt.tight_layout()
        save_name = corr_name + '_mean_frac_by_cutoff_comp_epochs'
        f_cc_epoch.savefig(os.path.join(corr_cutoff_save,save_name + '.png'))
        f_cc_epoch.savefig(os.path.join(corr_cutoff_save,save_name + '.svg'))
        plt.close(f_cc_epoch)
        ax_cc_epoch_frac[0,0].legend(loc='upper left')
        f_cc_epoch_frac.suptitle(corr_name + ' Fraction of Unique Epoch Events by Cutoff')
        plt.tight_layout()
        save_name = corr_name + '_unique_frac_by_cutoff_comp_epochs'
        f_cc_epoch_frac.savefig(os.path.join(corr_cutoff_save,save_name + '.png'))
        f_cc_epoch_frac.savefig(os.path.join(corr_cutoff_save,save_name + '.svg'))
        plt.close(f_cc_epoch_frac)
        
        #Plot segments against each other
        f_cc_seg, ax_cc_seg = plt.subplots(nrows = max_epochs, ncols = len(unique_taste_names),\
                                           sharex = True, sharey = True, figsize = (8,8))
        for cp_i in range(max_epochs):
            for t_i, taste in enumerate(unique_taste_names):
                for s_i, seg_name in enumerate(unique_segment_names):
                    seg_mean = np.nanmean(dev_corr_frac_dict[seg_name][taste][cp_i],0)
                    # seg_std = np.nanstd(dev_corr_frac_dict[seg_name][taste][cp_i],0)
                    # seg_mean_minus_std = seg_mean - seg_std
                    # seg_mean_minus_std[np.where(seg_mean_minus_std < 0)[0]] = 0
                    # ax_cc_epoch[cp_i,t_i].fill_between(corr_cutoffs,seg_mean+seg_std,\
                    #                                    seg_mean_minus_std,color = colors[s_i],
                    #                                    alpha=0.3,label='_')
                    ax_cc_seg[cp_i,t_i].plot(corr_cutoffs,seg_mean,\
                                               c=colors[s_i],label=seg_name)
                    try:
                        zero_val = corr_cutoffs[np.where(seg_mean == 0)[0][0]]
                    except:
                        zero_val = 1
                    ax_cc_seg[cp_i,t_i].axvline(zero_val,c = colors[s_i],\
                                                  linestyle='dashed',alpha=0.3,\
                                                      label='_')
                    ax_cc_seg[cp_i,t_i].text(zero_val,0.1,str(zero_val),rotation=90)
                if cp_i == 0:
                    ax_cc_seg[cp_i,t_i].set_title(taste)
                if t_i == 0:
                    ax_cc_seg[cp_i,t_i].set_ylabel('Epoch ' + str(cp_i) + '\nMean Fraction of Events')
        ax_cc_seg[0,0].legend(loc='upper left')
        f_cc_seg.suptitle(corr_name + ' Mean Fraction of Events by Cutoff')
        plt.tight_layout()
        save_name = corr_name + '_mean_frac_by_cutoff_comp_segments'
        f_cc_seg.savefig(os.path.join(corr_cutoff_save,save_name + '.png'))
        f_cc_seg.savefig(os.path.join(corr_cutoff_save,save_name + '.svg'))
        plt.close(f_cc_seg)
        
        #Plot tastes against each other
        f_cc_taste, ax_cc_taste = plt.subplots(nrows = len(unique_segment_names),\
                                               ncols = max_epochs, sharex = True,\
                                               sharey = True, figsize = (8,8))
        for s_i, seg_name in enumerate(unique_segment_names):
            for cp_i in range(max_epochs):
                for t_i, taste in enumerate(unique_taste_names):
                    taste_mean = dev_corr_total_frac_dict[seg_name][taste][cp_i]
                    ax_cc_taste[s_i,cp_i].plot(corr_cutoffs,taste_mean,\
                                               c=colors[t_i],label=taste)
                    try:
                        zero_val = corr_cutoffs[np.where(taste_mean <= 0)[0][0]]
                    except:
                        zero_val = 1
                    ax_cc_taste[s_i,cp_i].axvline(zero_val,c = colors[t_i],\
                                                      linestyle='dashed',alpha=0.3,\
                                                          label='_')
                    ax_cc_taste[s_i,cp_i].text(zero_val,0.1,str(zero_val),rotation=90)
                if s_i == 0:
                    ax_cc_taste[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                if cp_i == 0:
                    ax_cc_taste[s_i,cp_i].set_ylabel(seg_name + '\nFraction of Events')
        ax_cc_taste[0,0].legend(loc='upper left')
        f_cc_taste.suptitle(corr_name + ' Fraction of All Events by Cutoff')
        plt.tight_layout()
        save_name = corr_name + '_frac_by_cutoff_comp_tastes'
        f_cc_taste.savefig(os.path.join(corr_cutoff_save,save_name + '.png'))
        f_cc_taste.savefig(os.path.join(corr_cutoff_save,save_name + '.svg'))
        plt.close(f_cc_taste)
        
        #Plot epochs against each other
        f_cc_epoch, ax_cc_epoch = plt.subplots(nrows = len(unique_segment_names),\
                                               ncols = len(unique_taste_names), sharex = True,\
                                               sharey = True, figsize = (8,8))
        for s_i, seg_name in enumerate(unique_segment_names):
            for t_i, taste in enumerate(unique_taste_names):
                for cp_i in range(max_epochs):
                    epoch_mean = dev_corr_total_frac_dict[seg_name][taste][cp_i]
                    ax_cc_epoch[s_i,t_i].plot(corr_cutoffs,epoch_mean,\
                                               c=colors[cp_i],label='Epoch ' + str(cp_i))
                    try:
                        zero_val = corr_cutoffs[np.where(epoch_mean == 0)[0][0]]
                        
                    except:
                        zero_val = 1
                    ax_cc_epoch[s_i,t_i].axvline(zero_val,c = colors[cp_i],\
                                              linestyle='dashed',alpha=0.3,\
                                                  label='_')
                    ax_cc_epoch[s_i,t_i].text(zero_val,0.1,str(zero_val),rotation=90)
                if s_i == 0:
                    ax_cc_epoch[s_i,t_i].set_title(taste)
                if t_i == 0:
                    ax_cc_epoch[s_i,t_i].set_ylabel(seg_name + '\nFraction of Events')
        ax_cc_epoch[0,0].legend(loc='upper left')
        f_cc_epoch.suptitle(corr_name + ' Fraction of All Events by Cutoff')
        plt.tight_layout()
        save_name = corr_name + '_frac_by_cutoff_comp_epochs'
        f_cc_epoch.savefig(os.path.join(corr_cutoff_save,save_name + '.png'))
        f_cc_epoch.savefig(os.path.join(corr_cutoff_save,save_name + '.svg'))
        plt.close(f_cc_epoch)
        
        #Plot segments against each other
        f_cc_seg, ax_cc_seg = plt.subplots(nrows = max_epochs, ncols = len(unique_taste_names),\
                                           sharex = True, sharey = True, figsize = (8,8))
        for cp_i in range(max_epochs):
            for t_i, taste in enumerate(unique_taste_names):
                for s_i, seg_name in enumerate(unique_segment_names):
                    seg_mean = dev_corr_total_frac_dict[seg_name][taste][cp_i]
                    ax_cc_seg[cp_i,t_i].plot(corr_cutoffs,seg_mean,\
                                               c=colors[s_i],label=seg_name)
                    try:
                        zero_val = corr_cutoffs[np.where(seg_mean == 0)[0][0]]
                    except:
                        zero_val = 1
                    ax_cc_seg[cp_i,t_i].axvline(zero_val,c = colors[s_i],\
                                                  linestyle='dashed',alpha=0.3,\
                                                      label='_')
                    ax_cc_seg[cp_i,t_i].text(zero_val,0.1,str(zero_val),rotation=90)
                if cp_i == 0:
                    ax_cc_seg[cp_i,t_i].set_title(taste)
                if t_i == 0:
                    ax_cc_seg[cp_i,t_i].set_ylabel('Epoch ' + str(cp_i) + '\nFraction of Events')
        ax_cc_seg[0,0].legend(loc='upper left')
        f_cc_seg.suptitle(corr_name + ' Fraction of All Events by Cutoff')
        plt.tight_layout()
        save_name = corr_name + '_frac_by_cutoff_comp_segments'
        f_cc_seg.savefig(os.path.join(corr_cutoff_save,save_name + '.png'))
        f_cc_seg.savefig(os.path.join(corr_cutoff_save,save_name + '.svg'))
        plt.close(f_cc_seg)


def cross_segment_diffs(corr_data, min_best_cutoff, save_dir, unique_given_names, unique_corr_names,
                        unique_segment_names, unique_taste_names):
    """This function collects statistics across different correlation types and
    plots them together
    INPUTS:
            - corr_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    corr_data[name] = dictionary of dataset data
                    corr_data[name]['corr_data'] = dict of length #correlation types
                    corr_data[name]['corr_data'][corr_name] = dict of length #segments
                    corr_data[name]['corr_data'][corr_name][seg_name] = dict of length #tastes
                    corr_data[name]['corr_data'][corr_name][seg_name][taste_name]['data'] = numpy 
                            array of population average vector correlations [num_dev x num_trials x num_epochs]
            - save_dir: directory to save the resulting plots
            - unique_given_names: unique names of datasets
            - unique_corr_names: unique names of correlation analysis types
            - unique_segment_names: unique names of experimental segments
            - unique_taste_names: unique names of tastants
    OUTPUTS: plots and statistical significance tests
    """
    # Set parameters
    warnings.filterwarnings('ignore')
    bin_edges = np.linspace(0, 1, 1001)
    bin_x_vals = np.arange(0, 1, 1/1000)
    
    # Create further save dirs
    mean_diff_save = os.path.join(save_dir, 'Corr_Mean_Diffs')
    if not os.path.isdir(mean_diff_save):
        os.mkdir(mean_diff_save)

    class cross_segment_attributes:
        def __init__(self, combo, names, i_1, i_2, i_3, unique_corr_names,
                     unique_taste_names, unique_epochs):
            setattr(self, names[0], eval(combo[0])[i_1])
            setattr(self, names[1], eval(combo[1])[i_2])
            setattr(self, names[2], eval(combo[2])[i_3])

    # _____Reorganize data by unique correlation type_____
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, min_best_cutoff,
                                                    unique_corr_names, unique_given_names,
                                                    unique_segment_names,unique_taste_names)

    # Plot all combinations
    unique_epochs = np.arange(max_epochs)
    characteristic_list = ['unique_corr_names',
                           'unique_taste_names', 'unique_epochs']
    characteristic_dict = dict()
    for cl in characteristic_list:
        characteristic_dict[cl] = eval(cl)
    name_list = ['corr_name', 'taste_name', 'e_i']
    # Get attribute pairs for plotting views
    all_combinations = list(combinations(characteristic_list, 2))
    all_combinations_full = []
    all_names_full = []
    for ac in all_combinations:
        ac_list = list(ac)
        missing = np.setdiff1d(characteristic_list, ac_list)
        full_combo = ac_list
        full_combo.extend(missing)
        all_combinations_full.append(full_combo)
        names_combo = [
            name_list[characteristic_list.index(c)] for c in full_combo]
        all_names_full.append(names_combo)
    # Get segment pairs for comparison
    segment_combinations = list(combinations(unique_segment_names, 2))

    for c_i in range(len(all_combinations_full)):
        combo = all_combinations_full[c_i]
        combo_lengths = [len(characteristic_dict[combo[i]])
                         for i in range(len(combo))]
        names = all_names_full[c_i]
        for i_1 in range(combo_lengths[0]):
            combo_1 = eval(combo[0])[i_1]
            if type(combo_1) == np.int64:
                combo_1 = "epoch_" + str(combo_1)
            for i_2 in range(combo_lengths[1]):
                combo_2 = eval(combo[1])[i_2]
                if type(combo_2) == np.int64:
                    combo_2 = "epoch_" + str(combo_2)
                f_mean_diff, ax_mean_diff = plt.subplots(
                    ncols=combo_lengths[2], nrows=1, \
                        figsize=(combo_lengths[2]*5, 5), \
                            sharex = True, sharey = True)
                f_best_mean_diff, ax_best_mean_diff = plt.subplots(
                    ncols=combo_lengths[2], nrows=1, \
                        figsize=(combo_lengths[2]*5, 5), \
                        sharex = True, sharey = True)
                max_mean_diff = -1
                min_mean_diff = 1
                max_best_mean_diff = -1
                min_best_mean_diff = 1
                significance_storage = dict()
                best_significance_storage = dict()
                
                for i_3 in range(combo_lengths[2]):
                    significance_storage[i_3] = dict()
                    best_significance_storage[i_3] = dict()
                    max_mean_diff_i = -1
                    min_mean_diff_i = 1
                    max_best_mean_diff_i = -1
                    min_best_mean_diff_i = 1
                    xlabel = eval(combo[2])[i_3]
                    if type(xlabel) == np.int64:
                        xlabel = "epoch_" + str(xlabel)
                    # Begin pulling data
                    att = cross_segment_attributes(combo, names, i_1, i_2, i_3, unique_corr_names,
                                                   unique_taste_names, unique_epochs)
                    corr_name = att.corr_name
                    taste_name = att.taste_name
                    e_i = att.e_i
                    # Pit segment pairs against each other
                    mean_diff_collection = dict()
                    best_mean_diff_collection = dict()
                    mean_diff_labels = []
                    for sp_i, sp in enumerate(segment_combinations):
                        seg_1 = sp[0]
                        seg_2 = sp[1]
                        mean_diffs = []
                        best_mean_diffs = []
                        cum_dist_labels = []
                        best_cum_dist_labels = []
                        counter = 0
                        best_counter = 0
                        for g_n in unique_given_names:
                            try:
                                #Reg Means
                                data_1 = unique_data_dict[corr_name][g_n][seg_1][taste_name]['data'][:, e_i]
                                data_2 = unique_data_dict[corr_name][g_n][seg_2][taste_name]['data'][:, e_i]
                                mean_diffs.extend(
                                    [np.nanmean(data_2) - np.nanmean(data_1)])
                                cum_dist_labels.append(
                                    [g_n + '(' + str(counter) + ')'])
                                counter += 1
                            except:
                                print("\tSkipping invalid dataset.")
                            try:    
                                #Best Means
                                data_1 = unique_best_data_dict[corr_name][g_n][seg_1][taste_name][e_i]
                                data_2 = unique_best_data_dict[corr_name][g_n][seg_2][taste_name][e_i]
                                best_mean_diffs.extend(
                                    [np.nanmean(data_2) - np.nanmean(data_1)])
                                best_cum_dist_labels.append(
                                    [g_n + '(' + str(best_counter) + ')'])
                                best_counter += 1
                            except:
                                print("\tSkipping invalid dataset.")
                        # Collect mean distribution differences
                        mean_diff_collection[sp_i] = dict()
                        mean_diff_collection[sp_i]['data'] = mean_diffs
                        mean_diff_collection[sp_i]['labels'] = cum_dist_labels
                        best_mean_diff_collection[sp_i] = dict()
                        best_mean_diff_collection[sp_i]['data'] = best_mean_diffs
                        best_mean_diff_collection[sp_i]['labels'] = best_cum_dist_labels
                        mean_diff_labels.append(seg_2 + ' - ' + seg_1)
                    # Plot box plots and trends of mean correlation differences with significance
                    ax_mean_diff[i_3].axhline(
                        0, label='_', alpha=0.2, color='k', linestyle='dashed')
                    ax_best_mean_diff[i_3].axhline(
                        0, label='_', alpha=0.2, color='k', linestyle='dashed')
                    points_boxplot = []
                    points_boxplot_best = []
                    for m_i in range(len(mean_diff_collection)):
                        points = np.array(mean_diff_collection[m_i]['data'])
                        if len(points) > 0:
                            if np.nanmax(points) > max_mean_diff_i:
                                max_mean_diff_i = np.nanmax(points)
                            if np.nanmin(points) < min_mean_diff_i:
                                min_mean_diff_i = np.nanmin(points)
                            ax_mean_diff[i_3].scatter(np.random.normal(
                                m_i+1, 0.04, size=len(points)), points, color='g', alpha=0.2)
                            ax_mean_diff[i_3].boxplot([points[~np.isnan(points)]], positions=[
                                                         m_i+1], sym='', medianprops=dict(linestyle='-', color='blue'), showcaps=True, showbox=True)
                        points_boxplot.append(list(points))
                    for m_i in range(len(best_mean_diff_collection)):
                        points_best = np.array(best_mean_diff_collection[m_i]['data'])
                        points_best = points_best[~np.isnan(points_best)]
                        if len(points_best) > 0:
                            if np.nanmax(points_best) > max_best_mean_diff_i:
                                max_best_mean_diff_i = np.nanmax(points_best)
                            if np.nanmin(points_best) < min_best_mean_diff_i:
                                min_best_mean_diff_i = np.nanmin(points_best)
                            ax_best_mean_diff[i_3].scatter(np.random.normal(
                                m_i+1, 0.04, size=len(points_best)), points_best, color='g', alpha=0.2)
                            ax_best_mean_diff[i_3].boxplot([points_best], positions=[
                                                         m_i+1], sym='', medianprops=dict(linestyle='-', color='blue'), showcaps=True, showbox=True)
                        points_boxplot_best.append(list(points_best))
                    if len(points_boxplot) > 0:
                        if max_mean_diff < max_mean_diff_i:
                            max_mean_diff = max_mean_diff_i
                        if min_mean_diff > min_mean_diff_i:
                            min_mean_diff = min_mean_diff_i
                    if len(points_boxplot_best) > 0:
                        if max_best_mean_diff < max_best_mean_diff_i:
                            max_best_mean_diff = max_best_mean_diff_i
                        if min_best_mean_diff > min_best_mean_diff_i:
                            min_best_mean_diff = min_best_mean_diff_i
                    # Now plot the points by animal as lines
                    ax_mean_diff[i_3].axhline(
                        0, label='_', alpha=0.2, color='k', linestyle='dashed')
                    ax_best_mean_diff[i_3].axhline(
                        0, label='_', alpha=0.2, color='k', linestyle='dashed')
                    all_data_labels = []
                    for m_i in range(len(mean_diff_collection)):
                        all_data_labels.extend(
                            mean_diff_collection[m_i]['labels'])
                    all_best_data_labels = []
                    for m_i in range(len(best_mean_diff_collection)):
                        all_best_data_labels.extend(
                            best_mean_diff_collection[m_i]['labels'])
                    unique_data_labels = np.array(
                        list(np.unique(all_data_labels)))
                    unique_best_data_labels = np.array(
                        list(np.unique(all_best_data_labels)))
                    # Calculate significances
                    # __Significance from 0
                    sig_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot):
                        # First check if positive or negative distribution on average
                        # Then test significance accordingly
                        if np.mean(points_dist) > 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile <= 5:
                                sig_inds.extend([points_ind+1])
                        if np.mean(points_dist) < 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile >= 95:
                                sig_inds.extend([points_ind+1])
                    significance_storage[i_3]['sig_dists'] = sig_inds
                    # __All points above or below 0
                    greater_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot):
                        if len(np.where(np.array(points_dist) > 0)[0]) == len(points_dist):
                            greater_inds.extend([points_ind+1])
                    significance_storage[i_3]['greater_dists'] = greater_inds
                    less_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot):
                        if len(np.where(np.array(points_dist) < 0)[0]) == len(points_dist):
                            less_inds.extend([points_ind+1])
                    significance_storage[i_3]['less_dists'] = less_inds
                    sig_inds_best = []
                    for points_ind, points_dist in enumerate(points_boxplot_best):
                        # First check if positive or negative distribution on average
                        # Then test significance accordingly
                        if np.mean(points_dist) > 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile <= 5:
                                sig_inds_best.extend([points_ind+1])
                        if np.mean(points_dist) < 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile >= 95:
                                sig_inds_best.extend([points_ind+1])
                    best_significance_storage[i_3]['sig_dists'] = sig_inds_best
                    # __All points above or below 0
                    greater_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot_best):
                        if len(np.where(np.array(points_dist) > 0)[0]) == len(points_dist):
                            greater_inds.extend([points_ind+1])
                    best_significance_storage[i_3]['greater_dists'] = greater_inds
                    less_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot_best):
                        if len(np.where(np.array(points_dist) < 0)[0]) == len(points_dist):
                            less_inds.extend([points_ind+1])
                    best_significance_storage[i_3]['less_dists'] = less_inds
                    # __Pairwise significance
                    pair_diffs = list(combinations(np.arange(len(mean_diff_collection)),2))
                    step = max_mean_diff_i/10
                    for pair_i, pair in enumerate(pair_diffs):
                       data_1 = mean_diff_collection[pair[0]]['data']
                       data_2 = mean_diff_collection[pair[1]]['data']
                       if (len(data_1) > 0)*(len(data_2) > 0):
                            result = ks_2samp(data_1,data_2)
                            if result[1] <= 0.05:
                               marker='*'
                               ind_1 = pair[0] + 1
                               ind_2 = pair[1] + 1
                               significance_storage[i_3][pair_i] = dict()
                               significance_storage[i_3][pair_i]['ind_1'] = ind_1
                               significance_storage[i_3][pair_i]['ind_2'] = ind_2
                               significance_storage[i_3][pair_i]['marker'] = marker
                    pair_diffs = list(combinations(np.arange(len(mean_diff_collection)),2))
                    step = max_mean_diff_i/10
                    for pair_i, pair in enumerate(pair_diffs):
                       data_1 = best_mean_diff_collection[pair[0]]['data']
                       data_2 = best_mean_diff_collection[pair[1]]['data']
                       if (len(data_1) > 0)*(len(data_2) > 0):
                            result = ks_2samp(data_1,data_2)
                            if result[1] <= 0.05:
                               marker='*'
                               ind_1 = pair[0] + 1
                               ind_2 = pair[1] + 1
                               best_significance_storage[i_3][pair_i] = dict()
                               best_significance_storage[i_3][pair_i]['ind_1'] = ind_1
                               best_significance_storage[i_3][pair_i]['ind_2'] = ind_2
                               best_significance_storage[i_3][pair_i]['marker'] = marker
                    ax_mean_diff[i_3].set_xticks(
                        np.arange(1, len(mean_diff_collection)+1), mean_diff_labels, rotation=45)
                    ax_mean_diff[i_3].set_title(xlabel)
                    ax_mean_diff[i_3].set_ylabel(
                        'Mean Correlation Difference')
                    ax_best_mean_diff[i_3].set_xticks(
                        np.arange(1, len(best_mean_diff_collection)+1), mean_diff_labels, rotation=45)
                    ax_best_mean_diff[i_3].set_title(xlabel)
                    ax_best_mean_diff[i_3].set_ylabel(
                        'Mean Correlation Difference')
                # Update all plots with remaining values
                for i_3 in range(combo_lengths[2]):
                    # Add individual significance data
                    step = max_mean_diff/10
                    sig_height = max_mean_diff + step
                    sig_taste_data = significance_storage[i_3]
                    for key in list(sig_taste_data.keys()):
                        if key == 'sig_dists':  # Individual dist significance
                            dist_sig_data = sig_taste_data[key]
                            for sig_plot_ind in dist_sig_data:
                                ax_mean_diff[i_3].text(
                                    sig_plot_ind, sig_height, '*', horizontalalignment='center', \
                                        verticalalignment='center',color='g')
                        elif key == 'greater_dists':  
                            side_data = sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_mean_diff[i_3].text(
                                    sig_plot_ind-0.25, sig_height, '>0', horizontalalignment='center', \
                                        verticalalignment='center',color='b')
                        elif key == 'less_dists':  
                            side_data = sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_mean_diff[i_3].text(
                                    sig_plot_ind+0.25, sig_height, '<0', horizontalalignment='center', \
                                        verticalalignment='center',color='r')
                        else: # Pair data
                            ind_1 = sig_taste_data[key]['ind_1']
                            ind_2 = sig_taste_data[key]['ind_2']
                            marker = sig_taste_data[key]['marker']
                            ax_mean_diff[i_3].plot(
                                [ind_1, ind_2], [sig_height, sig_height], color='k', linestyle='solid')
                            ax_mean_diff[i_3].plot([ind_1, ind_1], [
                                                      sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                            ax_mean_diff[i_3].plot([ind_2, ind_2], [
                                                      sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                            ax_mean_diff[i_3].text(
                                ind_1 + (ind_2-ind_1)/2, sig_height + step/2, marker,\
                                    horizontalalignment='center', verticalalignment='center',\
                                        color='k')
                            sig_height += step
                   
                    # Add best significance data
                    best_step = max_best_mean_diff/10
                    best_sig_height = max_best_mean_diff + best_step
                    best_sig_taste_data = best_significance_storage[i_3]
                    for key in list(best_sig_taste_data.keys()):
                        if key == 'sig_dists':  # Individual dist significance
                            dist_sig_data = best_sig_taste_data[key]
                            for sig_plot_ind in dist_sig_data:
                                ax_best_mean_diff[i_3].text(
                                    sig_plot_ind, best_sig_height, '*', horizontalalignment='center', \
                                        verticalalignment='center',color='g')
                        elif key == 'greater_dists':  
                            side_data = best_sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_best_mean_diff[i_3].text(
                                    sig_plot_ind+0.25, best_sig_height, '>0', horizontalalignment='center', \
                                        verticalalignment='center',color='b')
                        elif key == 'less_dists':  
                            side_data = best_sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_best_mean_diff[i_3].text(
                                    sig_plot_ind-0.25, best_sig_height, '<0', horizontalalignment='center', \
                                        verticalalignment='center',color='r')
                        else: # Pair data
                            ind_1 = best_sig_taste_data[key]['ind_1']
                            ind_2 = best_sig_taste_data[key]['ind_2']
                            marker = best_sig_taste_data[key]['marker']
                            ax_best_mean_diff[i_3].plot(
                                [ind_1, ind_2], [best_sig_height, best_sig_height], color='k', linestyle='solid')
                            ax_best_mean_diff[i_3].plot([ind_1, ind_1], [
                                                      best_sig_height-best_step/2, best_sig_height+best_step/2], color='k', linestyle='solid')
                            ax_best_mean_diff[i_3].plot([ind_2, ind_2], [
                                                      best_sig_height-best_step/2, best_sig_height+best_step/2], color='k', linestyle='solid')
                            ax_best_mean_diff[i_3].text(
                                ind_1 + (ind_2-ind_1)/2, best_sig_height + best_step/2, marker,\
                                    horizontalalignment='center', verticalalignment='center',\
                                        color='k')
                            best_sig_height += best_step
                    
                f_pop_vec_plot_name = combo_1.replace(
                    ' ', '_') + '_' + combo_2.replace(' ', '_')
                f_mean_diff.suptitle('Mean Correlation Difference: ' +
                                     combo_1.replace(' ', '_') + ' x ' + combo_2.replace(' ', '_'))
                f_mean_diff.tight_layout()
                f_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff.png')
                f_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff.svg')
                plt.close(f_mean_diff)
                f_best_mean_diff.suptitle('Mean Correlation Difference: ' +
                                     combo_1.replace(' ', '_') + ' x ' + combo_2.replace(' ', '_'))
                f_best_mean_diff.tight_layout()
                f_best_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff_best.png')
                f_best_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff_best.svg')
                plt.close(f_best_mean_diff)

def cross_taste_diffs(corr_data, min_best_cutoff, save_dir, unique_given_names, unique_corr_names,
                      unique_segment_names, unique_taste_names):
    """This function collects statistics across different correlation types and
    plots them together
    INPUTS:
            - corr_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    corr_data[name] = dictionary of dataset data
                    corr_data[name]['corr_data'] = dict of length #correlation types
                    corr_data[name]['corr_data'][corr_name] = dict of length #segments
                    corr_data[name]['corr_data'][corr_name][seg_name] = dict of length #tastes
                    corr_data[name]['corr_data'][corr_name][seg_name][taste_name]['data'] = numpy 
                            array of population average vector correlations [num_dev x num_trials x num_epochs]
            - results_dir: directory to save the resulting plots
            - unique_corr_names: unique names of correlation analyses to compare
    OUTPUTS: plots and statistical significance tests
    """
    # Set parameters
    warnings.filterwarnings('ignore')
    bin_edges = np.linspace(0, 1, 1001)
    bin_x_vals = np.arange(0, 1, 1/1000)
    
    # Create further save dirs
    mean_diff_save = os.path.join(save_dir, 'Corr_Mean_Diffs')
    if not os.path.isdir(mean_diff_save):
        os.mkdir(mean_diff_save)

    class cross_taste_attributes:
        def __init__(self, combo, names, i_1, i_2, i_3, unique_corr_names,
                     unique_segment_names, unique_epochs):
            setattr(self, names[0], eval(combo[0])[i_1])
            setattr(self, names[1], eval(combo[1])[i_2])
            setattr(self, names[2], eval(combo[2])[i_3])

    # _____Reorganize data by unique correlation type_____
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, min_best_cutoff,
                                                            unique_corr_names, unique_given_names,
                                                            unique_segment_names,unique_taste_names)

    # Plot all combinations
    unique_epochs = np.arange(max_epochs)
    characteristic_list = ['unique_corr_names',
                           'unique_segment_names', 'unique_epochs']
    characteristic_dict = dict()
    for cl in characteristic_list:
        characteristic_dict[cl] = eval(cl)
    name_list = ['corr_name', 'seg_name', 'e_i']
    # Get attribute pairs for plotting views
    all_combinations = list(combinations(characteristic_list, 2))
    all_combinations_full = []
    all_names_full = []
    for ac in all_combinations:
        ac_list = list(ac)
        missing = np.setdiff1d(characteristic_list, ac_list)
        full_combo = ac_list
        full_combo.extend(missing)
        all_combinations_full.append(full_combo)
        names_combo = [
            name_list[characteristic_list.index(c)] for c in full_combo]
        all_names_full.append(names_combo)
    # Get segment pairs for comparison
    taste_combinations = list(combinations(unique_taste_names, 2))

    for c_i in range(len(all_combinations_full)):
        combo = all_combinations_full[c_i]
        combo_lengths = [len(characteristic_dict[combo[i]])
                         for i in range(len(combo))]
        names = all_names_full[c_i]
        for i_1 in range(combo_lengths[0]):
            combo_1 = eval(combo[0])[i_1]
            if type(combo_1) == np.int64:
                combo_1 = "epoch_" + str(combo_1)
            for i_2 in range(combo_lengths[1]):
                combo_2 = eval(combo[1])[i_2]
                if type(combo_2) == np.int64:
                    combo_2 = "epoch_" + str(combo_2)
                f_mean_diff, ax_mean_diff = plt.subplots(
                    ncols=combo_lengths[2], figsize=(combo_lengths[2]*5, 5),\
                        sharex = True, sharey = True)
                f_best_mean_diff, ax_best_mean_diff = plt.subplots(
                    ncols=combo_lengths[2], figsize=(combo_lengths[2]*5, 5),\
                        sharex = True, sharey = True)
                max_mean_diff = 1
                min_mean_diff = -1
                max_best_mean_diff = -1
                min_best_mean_diff = 1
                significance_storage = dict()
                best_significance_storage = dict()
                
                for i_3 in range(combo_lengths[2]):
                    significance_storage[i_3] = dict()
                    best_significance_storage[i_3] = dict()
                    max_mean_diff_i = -1
                    min_mean_diff_i = 1
                    max_best_mean_diff_i = -1
                    min_best_mean_diff_i = 1
                    xlabel = eval(combo[2])[i_3]
                    if type(xlabel) == np.int64:
                        xlabel = "epoch_" + str(xlabel)
                    # Begin pulling data
                    att = cross_taste_attributes(combo, names, i_1, i_2, i_3, unique_corr_names,
                                                 unique_segment_names, unique_epochs)
                    corr_name = att.corr_name
                    seg_name = att.seg_name
                    e_i = att.e_i
                    # Pit segment pairs against each other
                    mean_diff_collection = dict()
                    best_mean_diff_collection = dict()
                    mean_diff_labels = []
                    for tp_i, tp in enumerate(taste_combinations):
                        taste_1 = tp[0]
                        taste_2 = tp[1]
                        title = taste_2 + ' - ' + taste_1
                        mean_diffs = []
                        best_mean_diffs = []
                        cum_dist_labels = []
                        best_cum_dist_labels = []
                        counter = 0
                        best_counter = 0
                        for g_n in unique_given_names:
                            try:
                                data_1 = unique_data_dict[corr_name][g_n][seg_name][taste_1]['data'][:, e_i]
                                data_2 = unique_data_dict[corr_name][g_n][seg_name][taste_2]['data'][:, e_i]
                                mean_diffs.extend(
                                    [np.nanmean(data_2) - np.nanmean(data_1)])
                                cum_dist_labels.append(
                                    [g_n + '(' + str(counter) + ')'])
                                counter += 1
                            except:
                                print("\tSkipping invalid dataset.")
                            try:    
                                #Best Means
                                data_1 = unique_best_data_dict[corr_name][g_n][seg_name][taste_1][e_i]
                                data_2 = unique_best_data_dict[corr_name][g_n][seg_name][taste_2][e_i]
                                best_mean_diffs.extend(
                                    [np.nanmean(data_2) - np.nanmean(data_1)])
                                best_cum_dist_labels.append(
                                    [g_n + '(' + str(best_counter) + ')'])
                                best_counter += 1
                            except:
                                print("\tSkipping invalid dataset.")
                        # Collect mean distribution differences
                        mean_diff_collection[tp_i] = dict()
                        mean_diff_collection[tp_i]['data'] = mean_diffs
                        mean_diff_collection[tp_i]['labels'] = cum_dist_labels
                        best_mean_diff_collection[tp_i] = dict()
                        best_mean_diff_collection[tp_i]['data'] = best_mean_diffs
                        best_mean_diff_collection[tp_i]['labels'] = best_cum_dist_labels
                        mean_diff_labels.append(taste_2 + ' - ' + taste_1)
                    # Plot box plots of mean correlation differences
                    ax_mean_diff[i_3].axhline(
                        0, label='_', alpha=0.2, color='k', linestyle='dashed')
                    ax_best_mean_diff[i_3].axhline(
                        0, label='_', alpha=0.2, color='k', linestyle='dashed')
                    points_boxplot = []
                    points_boxplot_best = []
                    for m_i in range(len(mean_diff_collection)):
                        points = np.array(mean_diff_collection[m_i]['data'])
                        if len(points) > 0:
                            if np.nanmax(points) > max_mean_diff_i:
                                max_mean_diff_i = np.nanmax(points)
                            if np.nanmin(points) < min_mean_diff_i:
                                min_mean_diff_i = np.nanmin(points)
                            ax_mean_diff[i_3].scatter(np.random.normal(
                                m_i+1, 0.04, size=len(points)), points, color='g', alpha=0.2)
                            ax_mean_diff[i_3].boxplot([points[~np.isnan(points)]], positions=[
                                                      m_i+1], sym='', \
                                    medianprops=dict(linestyle='-', color='blue'), \
                                        showcaps=True, showbox=True)
                        points_boxplot.append(list(points))
                    for m_i in range(len(best_mean_diff_collection)):
                        points_best = np.array(best_mean_diff_collection[m_i]['data'])
                        if len(points_best) > 0:
                            if np.nanmax(points_best) > max_best_mean_diff_i:
                                max_best_mean_diff_i = np.nanmax(points_best)
                            if np.nanmin(points_best) < min_best_mean_diff_i:
                                min_best_mean_diff_i = np.nanmin(points_best)
                            ax_best_mean_diff[i_3].scatter(np.random.normal(
                                m_i+1, 0.04, size=len(points_best)), points_best, \
                                    color='g', alpha=0.2)
                            ax_best_mean_diff[i_3].boxplot([points_best[~np.isnan(points_best)]], positions=[
                                                         m_i+1], sym='', \
                                    medianprops=dict(linestyle='-', color='blue'), \
                                        showcaps=True, showbox=True)
                        points_boxplot_best.append(list(points_best))
                    if len(points_boxplot) > 0:
                        if max_mean_diff < max_mean_diff_i:
                            max_mean_diff = max_mean_diff_i
                        if min_mean_diff > min_mean_diff_i:
                            min_mean_diff = min_mean_diff_i
                    if len(points_boxplot_best) > 0:
                        if max_best_mean_diff < max_best_mean_diff_i:
                            max_best_mean_diff = max_best_mean_diff_i
                        if min_best_mean_diff > min_best_mean_diff_i:
                            min_best_mean_diff = min_best_mean_diff_i
                    # Calculate significances
                    # __Significance from 0
                    sig_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot):
                        # First check if positive or negative distribution on average
                        # Then test significance accordingly
                        if np.mean(points_dist) > 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile <= 5:
                                sig_inds.extend([points_ind+1])
                        if np.mean(points_dist) < 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile >= 95:
                                sig_inds.extend([points_ind+1])
                    significance_storage[i_3]['sig_dists'] = sig_inds
                    # __All points above or below 0
                    greater_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot):
                        if len(np.where(np.array(points_dist) > 0)[0]) == len(points_dist):
                            greater_inds.extend([points_ind+1])
                    significance_storage[i_3]['greater_dists'] = greater_inds
                    less_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot):
                        if len(np.where(np.array(points_dist) < 0)[0]) == len(points_dist):
                            less_inds.extend([points_ind+1])
                    significance_storage[i_3]['less_dists'] = less_inds
                    sig_inds_best = []
                    for points_ind, points_dist in enumerate(points_boxplot_best):
                        # First check if positive or negative distribution on average
                        # Then test significance accordingly
                        if np.mean(points_dist) > 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile <= 5:
                                sig_inds_best.extend([points_ind+1])
                        if np.mean(points_dist) < 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile >= 95:
                                sig_inds_best.extend([points_ind+1])
                    best_significance_storage[i_3]['sig_dists'] = sig_inds_best
                    # __All points above or below 0
                    greater_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot_best):
                        if len(np.where(np.array(points_dist) > 0)[0]) == len(points_dist):
                            greater_inds.extend([points_ind+1])
                    best_significance_storage[i_3]['greater_dists'] = greater_inds
                    less_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot_best):
                        if len(np.where(np.array(points_dist) < 0)[0]) == len(points_dist):
                            less_inds.extend([points_ind+1])
                    best_significance_storage[i_3]['less_dists'] = less_inds
                    # __Pairwise significance
                    pair_diffs = list(combinations(np.arange(len(mean_diff_collection)),2))
                    step = max_mean_diff_i/10
                    for pair_i, pair in enumerate(pair_diffs):
                       data_1 = mean_diff_collection[pair[0]]['data']
                       data_2 = mean_diff_collection[pair[1]]['data']
                       if (len(data_1) > 0)*(len(data_2) > 0):
                            result = ks_2samp(data_1,data_2)
                            if result[1] <= 0.05:
                               marker='*'
                               ind_1 = pair[0] + 1
                               ind_2 = pair[1] + 1
                               significance_storage[i_3][pair_i] = dict()
                               significance_storage[i_3][pair_i]['ind_1'] = ind_1
                               significance_storage[i_3][pair_i]['ind_2'] = ind_2
                               significance_storage[i_3][pair_i]['marker'] = marker
                    best_pair_diffs = list(combinations(np.arange(len(best_mean_diff_collection)),2))
                    step = max_mean_diff_i/10
                    for pair_i, pair in enumerate(best_pair_diffs):
                       data_1 = best_mean_diff_collection[pair[0]]['data']
                       data_2 = best_mean_diff_collection[pair[1]]['data']
                       if (len(data_1) > 0)*(len(data_2) > 0):
                            result = ks_2samp(data_1,data_2)
                            if result[1] <= 0.05:
                               marker='*'
                               ind_1 = pair[0] + 1
                               ind_2 = pair[1] + 1
                               best_significance_storage[i_3][pair_i] = dict()
                               best_significance_storage[i_3][pair_i]['ind_1'] = ind_1
                               best_significance_storage[i_3][pair_i]['ind_2'] = ind_2
                               best_significance_storage[i_3][pair_i]['marker'] = marker
                    ax_mean_diff[i_3].set_xticks(
                        np.arange(1, len(mean_diff_collection)+1), mean_diff_labels, rotation=45)
                    ax_mean_diff[i_3].set_title(xlabel)
                    ax_mean_diff[i_3].set_ylabel('Mean Correlation Difference')
                    ax_best_mean_diff[i_3].set_xticks(
                        np.arange(1, len(best_mean_diff_collection)+1), mean_diff_labels, rotation=45)
                    ax_best_mean_diff[i_3].set_title(xlabel)
                    ax_best_mean_diff[i_3].set_ylabel(
                        'Mean Correlation Difference')
                # Update all plots with remaining values
                for i_3 in range(combo_lengths[2]):
                    # Add individual significance data
                    step = max_mean_diff/10
                    sig_height = max_mean_diff + step
                    sig_taste_data = significance_storage[i_3]
                    for key in list(sig_taste_data.keys()):
                        if key == 'sig_dists':  # Individual dist significance
                            dist_sig_data = sig_taste_data[key]
                            for sig_plot_ind in dist_sig_data:
                                ax_mean_diff[i_3].text(
                                    sig_plot_ind, sig_height, '*', horizontalalignment='center', \
                                        verticalalignment='center',color='g')
                        elif key == 'greater_dists':  
                            side_data = sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_mean_diff[i_3].text(
                                    sig_plot_ind+0.25, sig_height, '>0', horizontalalignment='center', \
                                        verticalalignment='center',color='b')
                        elif key == 'less_dists':  
                            side_data = sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_mean_diff[i_3].text(
                                    sig_plot_ind-0.25, sig_height, '<0', horizontalalignment='center', \
                                        verticalalignment='center',color='r')
                        else: # Pair data
                            ind_1 = sig_taste_data[key]['ind_1']
                            ind_2 = sig_taste_data[key]['ind_2']
                            marker = sig_taste_data[key]['marker']
                            ax_mean_diff[i_3].plot(
                                [ind_1, ind_2], [sig_height, sig_height], color='k', linestyle='solid')
                            ax_mean_diff[i_3].plot([ind_1, ind_1], [
                                                      sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                            ax_mean_diff[i_3].plot([ind_2, ind_2], [
                                                      sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                            ax_mean_diff[i_3].text(
                                ind_1 + (ind_2-ind_1)/2, sig_height + step/2, marker,\
                                    horizontalalignment='center', verticalalignment='center',\
                                        color='k')
                            sig_height += step
                    
                    # Add best significance data
                    step = max_best_mean_diff/10
                    best_sig_height = max_best_mean_diff + step
                    best_sig_taste_data = best_significance_storage[i_3]
                    for key in list(best_sig_taste_data.keys()):
                        if key == 'sig_dists':  # Individual dist significance
                            dist_sig_data = best_sig_taste_data[key]
                            for sig_plot_ind in dist_sig_data:
                                ax_best_mean_diff[i_3].text(
                                    sig_plot_ind, best_sig_height, '*', horizontalalignment='center', \
                                        verticalalignment='center',color='g')
                        elif key == 'greater_dists':  
                            side_data = best_sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_best_mean_diff[i_3].text(
                                    sig_plot_ind+0.25, best_sig_height, '>0', horizontalalignment='center', \
                                        verticalalignment='center',color='b')
                        elif key == 'less_dists':  
                            side_data = best_sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_best_mean_diff[i_3].text(
                                    sig_plot_ind-0.25, best_sig_height, '<0', horizontalalignment='center', \
                                        verticalalignment='center',color='r')
                        else: # Pair data
                            ind_1 = best_sig_taste_data[key]['ind_1']
                            ind_2 = best_sig_taste_data[key]['ind_2']
                            marker = best_sig_taste_data[key]['marker']
                            ax_best_mean_diff[i_3].plot(
                                [ind_1, ind_2], [best_sig_height, best_sig_height], color='k', linestyle='solid')
                            ax_best_mean_diff[i_3].plot([ind_1, ind_1], [
                                                      best_sig_height-step/2, best_sig_height+step/2], color='k', linestyle='solid')
                            ax_best_mean_diff[i_3].plot([ind_2, ind_2], [
                                                      best_sig_height-step/2, best_sig_height+step/2], color='k', linestyle='solid')
                            ax_best_mean_diff[i_3].text(
                                ind_1 + (ind_2-ind_1)/2, best_sig_height + step/2, marker,\
                                    horizontalalignment='center', verticalalignment='center',\
                                        color='k')
                            best_sig_height += step
                    
                    # Adjust y-limits
                    # ax_best_mean_diff[i_3].set_ylim(
                    #     [min_best_mean_diff - np.abs(min_best_mean_diff)/5, best_sig_height + best_sig_height/5])
                    
                # Finish plots with titles and save
                f_pop_vec_plot_name = combo_1.replace(
                    ' ', '_') + '_' + combo_2.replace(' ', '_')
                f_mean_diff.suptitle('Mean Correlation Difference: ' +
                                     combo_1.replace(' ', '_') + ' x ' + combo_2.replace(' ', '_'))
                f_mean_diff.tight_layout()
                f_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff.png')
                f_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff.svg')
                plt.close(f_mean_diff)
                f_best_mean_diff.suptitle('Mean Correlation Difference: ' +
                                     combo_1.replace(' ', '_') + ' x ' + combo_2.replace(' ', '_'))
                f_best_mean_diff.tight_layout()
                f_best_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff_best.png')
                f_best_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff_best.svg')
                plt.close(f_best_mean_diff)


def cross_epoch_diffs(corr_data, min_best_cutoff, save_dir, unique_given_names, unique_corr_names,
                      unique_segment_names, unique_taste_names):
    """This function collects statistics across different correlation types and
    plots them together
    INPUTS:
            - corr_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    corr_data[name] = dictionary of dataset data
                    corr_data[name]['corr_data'] = dict of length #correlation types
                    corr_data[name]['corr_data'][corr_name] = dict of length #segments
                    corr_data[name]['corr_data'][corr_name][seg_name] = dict of length #tastes
                    corr_data[name]['corr_data'][corr_name][seg_name][taste_name]['data'] = numpy 
                            array of population average vector correlations [num_dev x num_trials x num_epochs]
            - results_dir: directory to save the resulting plots
            - unique_corr_names: unique names of correlation analyses to compare
    OUTPUTS: plots and statistical significance tests
    """
    # Set parameters
    warnings.filterwarnings('ignore')
    bin_edges = np.linspace(0, 1, 1001)
    bin_x_vals = np.arange(0, 1, 1/1000)
    
    # Create further save dirs
    mean_diff_save = os.path.join(save_dir, 'Corr_Mean_Diffs')
    if not os.path.isdir(mean_diff_save):
        os.mkdir(mean_diff_save)

    class cross_epoch_attributes:
        def __init__(self, combo, names, i_1, i_2, i_3, unique_corr_names,
                     unique_segment_names, unique_taste_names):
            setattr(self, names[0], eval(combo[0])[i_1])
            setattr(self, names[1], eval(combo[1])[i_2])
            setattr(self, names[2], eval(combo[2])[i_3])

    # _____Reorganize data by unique correlation type_____
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, min_best_cutoff, 
                                                            unique_corr_names, unique_given_names,
                                                            unique_segment_names,unique_taste_names)

    # Plot all combinations
    unique_epochs = np.arange(max_epochs)
    characteristic_list = ['unique_corr_names',
                           'unique_segment_names', 'unique_taste_names']
    characteristic_dict = dict()
    for cl in characteristic_list:
        characteristic_dict[cl] = eval(cl)
    name_list = ['corr_name', 'seg_name', 'taste_name']
    # Get attribute pairs for plotting views
    all_combinations = list(combinations(characteristic_list, 2))
    all_combinations_full = []
    all_names_full = []
    for ac in all_combinations:
        ac_list = list(ac)
        missing = np.setdiff1d(characteristic_list, ac_list)
        full_combo = ac_list
        full_combo.extend(missing)
        all_combinations_full.append(full_combo)
        names_combo = [
            name_list[characteristic_list.index(c)] for c in full_combo]
        all_names_full.append(names_combo)
    # Get segment pairs for comparison
    epoch_combinations = list(combinations(unique_epochs, 2))

    for c_i in range(len(all_combinations_full)):
        combo = all_combinations_full[c_i]
        combo_lengths = [len(characteristic_dict[combo[i]])
                         for i in range(len(combo))]
        names = all_names_full[c_i]
        for i_1 in range(combo_lengths[0]):
            combo_1 = eval(combo[0])[i_1]
            if type(combo_1) == np.int64:
                combo_1 = "epoch_" + str(combo_1)
            for i_2 in range(combo_lengths[1]):
                combo_2 = eval(combo[1])[i_2]
                if type(combo_2) == np.int64:
                    combo_2 = "epoch_" + str(combo_2)
                f_mean_diff, ax_mean_diff = plt.subplots(
                    ncols=combo_lengths[2], figsize=(combo_lengths[2]*5, 5),\
                        sharex = True, sharey = True)
                f_best_mean_diff, ax_best_mean_diff = plt.subplots(
                    ncols=combo_lengths[2], figsize=(combo_lengths[2]*5, 5),\
                        sharex = True, sharey = True)
                max_mean_diff = -1
                min_mean_diff = 1
                max_best_mean_diff = -1
                min_best_mean_diff = 1
                significance_storage = dict()
                best_significance_storage = dict()
                
                for i_3 in range(combo_lengths[2]):
                    significance_storage[i_3] = dict()
                    best_significance_storage[i_3] = dict()
                    max_mean_diff_i = -1
                    min_mean_diff_i = 1
                    max_best_mean_diff_i = -1
                    min_best_mean_diff_i = 1
                    xlabel = eval(combo[2])[i_3]
                    if type(xlabel) == np.int64:
                        xlabel = "epoch_" + str(xlabel)
                    # Begin pulling data
                    att = cross_epoch_attributes(combo, names, i_1, i_2, i_3, unique_corr_names,
                                                 unique_segment_names, unique_taste_names)
                    corr_name = att.corr_name
                    seg_name = att.seg_name
                    taste_name = att.taste_name
                    # Pit segment pairs against each other
                    mean_diff_collection = dict()
                    best_mean_diff_collection = dict()
                    mean_diff_labels = []
                    for ep_i, ep in enumerate(epoch_combinations):
                        epoch_1 = ep[0]
                        epoch_2 = ep[1]
                        title = 'Epoch ' + \
                            str(epoch_2) + ' - Epoch ' + str(epoch_1)
                        mean_diffs = []
                        best_mean_diffs = []
                        cum_dist_labels = []
                        best_cum_dist_labels = []
                        counter = 0
                        best_counter = 0
                        for g_n in unique_given_names:
                            try:
                                data_1 = unique_data_dict[corr_name][g_n][seg_name][taste_name]['data'][:, epoch_1]
                                data_2 = unique_data_dict[corr_name][g_n][seg_name][taste_name]['data'][:, epoch_2]
                                mean_diffs.extend(
                                    [np.nanmean(data_2) - np.nanmean(data_1)])
                                cum_dist_labels.append(
                                    [g_n + '(' + str(counter) + ')'])
                                counter += 1
                            except:
                                print("\tSkipping invalid dataset.")
                            try:    
                                #Best Means
                                data_1 = unique_best_data_dict[corr_name][g_n][seg_name][taste_name][epoch_1]
                                data_2 = unique_best_data_dict[corr_name][g_n][seg_name][taste_name][epoch_2]
                                best_mean_diffs.extend(
                                    [np.nanmean(data_2) - np.nanmean(data_1)])
                                best_cum_dist_labels.append(
                                    [g_n + '(' + str(best_counter) + ')'])
                                best_counter += 1
                            except:
                                print("\tSkipping invalid dataset.")
                        # Collect mean distribution differences
                        mean_diff_collection[ep_i] = dict()
                        mean_diff_collection[ep_i]['data'] = mean_diffs
                        mean_diff_collection[ep_i]['labels'] = cum_dist_labels
                        best_mean_diff_collection[ep_i] = dict()
                        best_mean_diff_collection[ep_i]['data'] = best_mean_diffs
                        best_mean_diff_collection[ep_i]['labels'] = best_cum_dist_labels
                        mean_diff_labels.append(title)
                    # Plot box plots of mean correlation differences
                    ax_mean_diff[i_3].axhline(
                        0, label='_', alpha=0.2, color='k', linestyle='dashed')
                    ax_best_mean_diff[i_3].axhline(
                        0, label='_', alpha=0.2, color='k', linestyle='dashed')
                    points_boxplot = []
                    points_boxplot_best = []
                    for m_i in range(len(mean_diff_collection)):
                        points = np.array(mean_diff_collection[m_i]['data'])
                        if len(points) > 0:
                            if np.nanmax(points) > max_mean_diff_i:
                                max_mean_diff_i = np.nanmax(points)
                            if np.nanmin(points) < min_mean_diff_i:
                                min_mean_diff_i = np.nanmin(points)
                            ax_mean_diff[i_3].scatter(np.random.normal(
                                m_i+1, 0.04, size=len(points)), points, color='g', alpha=0.2)
                            ax_mean_diff[i_3].boxplot([points[~np.isnan(points)]], positions=[
                                                      m_i+1], sym='', medianprops=dict(linestyle='-', color='blue'), showcaps=True, showbox=True)
                        points_boxplot.append(list(points))
                    for m_i in range(len(best_mean_diff_collection)):
                        points_best = np.array(best_mean_diff_collection[m_i]['data'])
                        if len(points_best) > 0:
                            if np.nanmax(points_best) > max_best_mean_diff_i:
                                max_best_mean_diff_i = np.nanmax(points_best)
                            if np.nanmin(points_best) < min_best_mean_diff_i:
                                min_best_mean_diff_i = np.nanmin(points_best)
                            ax_best_mean_diff[i_3].scatter(np.random.normal(
                                m_i+1, 0.04, size=len(points_best)), points_best, color='g', alpha=0.2)
                            ax_best_mean_diff[i_3].boxplot([points_best[~np.isnan(points_best)]], positions=[
                                                         m_i+1], sym='', medianprops=dict(linestyle='-', color='blue'), showcaps=True, showbox=True)
                        points_boxplot_best.append(list(points_best))
                    if len(points_boxplot) > 0:
                        if max_mean_diff < max_mean_diff_i:
                            max_mean_diff = max_mean_diff_i
                        if min_mean_diff > min_mean_diff_i:
                            min_mean_diff = min_mean_diff_i
                    if len(points_boxplot_best) > 0:
                        if max_best_mean_diff < max_best_mean_diff_i:
                            max_best_mean_diff = max_best_mean_diff_i
                        if min_best_mean_diff > min_best_mean_diff_i:
                            min_best_mean_diff = min_best_mean_diff_i
                    # Calculate significances
                    # __Significance from 0
                    sig_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot):
                        # First check if positive or negative distribution on average
                        # Then test significance accordingly
                        if np.mean(points_dist) > 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile <= 5:
                                sig_inds.extend([points_ind+1])
                        if np.mean(points_dist) < 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile >= 95:
                                sig_inds.extend([points_ind+1])
                    significance_storage[i_3]['sig_dists'] = sig_inds
                    # __All points above or below 0
                    greater_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot):
                        if len(np.where(np.array(points_dist) > 0)[0]) == len(points_dist):
                            greater_inds.extend([points_ind+1])
                    significance_storage[i_3]['greater_dists'] = greater_inds
                    less_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot):
                        if len(np.where(np.array(points_dist) < 0)[0]) == len(points_dist):
                            less_inds.extend([points_ind+1])
                    significance_storage[i_3]['less_dists'] = less_inds
                    
                    sig_inds_best = []
                    for points_ind, points_dist in enumerate(points_boxplot_best):
                        # First check if positive or negative distribution on average
                        # Then test significance accordingly
                        if np.mean(points_dist) > 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile <= 5:
                                sig_inds_best.extend([points_ind+1])
                        if np.mean(points_dist) < 0:
                            zero_percentile = percentileofscore(points_dist, 0)
                            if zero_percentile >= 95:
                                sig_inds_best.extend([points_ind+1])
                    best_significance_storage[i_3]['sig_dists'] = sig_inds_best
                    # __All points above or below 0
                    greater_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot_best):
                        if len(np.where(np.array(points_dist) > 0)[0]) == len(points_dist):
                            greater_inds.extend([points_ind+1])
                    best_significance_storage[i_3]['greater_dists'] = greater_inds
                    less_inds = []
                    for points_ind, points_dist in enumerate(points_boxplot_best):
                        if len(np.where(np.array(points_dist) < 0)[0]) == len(points_dist):
                            less_inds.extend([points_ind+1])
                    best_significance_storage[i_3]['less_dists'] = less_inds
                    
                    # __Pairwise significance
                    pair_diffs = list(combinations(np.arange(len(mean_diff_collection)),2))
                    step = max_mean_diff_i/10
                    for pair_i, pair in enumerate(pair_diffs):
                       data_1 = mean_diff_collection[pair[0]]['data']
                       data_2 = mean_diff_collection[pair[1]]['data']
                       if (len(data_1) > 0)*(len(data_2) > 0):
                            result = ks_2samp(data_1,data_2)
                            if result[1] <= 0.05:
                               marker='*'
                               ind_1 = pair[0] + 1
                               ind_2 = pair[1] + 1
                               significance_storage[i_3][pair_i] = dict()
                               significance_storage[i_3][pair_i]['ind_1'] = ind_1
                               significance_storage[i_3][pair_i]['ind_2'] = ind_2
                               significance_storage[i_3][pair_i]['marker'] = marker
                    pair_diffs = list(combinations(np.arange(len(mean_diff_collection)),2))
                    step = max_mean_diff_i/10
                    for pair_i, pair in enumerate(pair_diffs):
                       data_1 = best_mean_diff_collection[pair[0]]['data']
                       data_2 = best_mean_diff_collection[pair[1]]['data']
                       if (len(data_1) > 0)*(len(data_2) > 0):
                            result = ks_2samp(data_1,data_2)
                            if result[1] <= 0.05:
                               marker='*'
                               ind_1 = pair[0] + 1
                               ind_2 = pair[1] + 1
                               best_significance_storage[i_3][pair_i] = dict()
                               best_significance_storage[i_3][pair_i]['ind_1'] = ind_1
                               best_significance_storage[i_3][pair_i]['ind_2'] = ind_2
                               best_significance_storage[i_3][pair_i]['marker'] = marker
                    ax_mean_diff[i_3].set_xticks(
                        np.arange(1, len(mean_diff_collection)+1), mean_diff_labels, rotation=45)
                    ax_mean_diff[i_3].set_title(xlabel)
                    ax_mean_diff[i_3].set_ylabel('Mean Correlation Difference')
                    ax_best_mean_diff[i_3].set_xticks(
                        np.arange(1, len(best_mean_diff_collection)+1), mean_diff_labels, rotation=45)
                    ax_best_mean_diff[i_3].set_title(xlabel)
                    ax_best_mean_diff[i_3].set_ylabel(
                        'Mean Correlation Difference')
                # Update all plots with remaining values
                for i_3 in range(combo_lengths[2]):
                    # Add significance data
                    step = max_mean_diff/10
                    sig_height = max_mean_diff + step
                    sig_taste_data = significance_storage[i_3]
                    for key in list(sig_taste_data.keys()):
                        if key == 'sig_dists':  # Individual dist significance
                            dist_sig_data = sig_taste_data[key]
                            for sig_plot_ind in dist_sig_data:
                                ax_mean_diff[i_3].text(
                                    sig_plot_ind, sig_height, '*', horizontalalignment='center', \
                                        verticalalignment='center',color='g')
                        elif key == 'greater_dists':  
                            side_data = sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_mean_diff[i_3].text(
                                    sig_plot_ind+0.25, sig_height, '>0', horizontalalignment='center', \
                                        verticalalignment='center',color='b')
                        elif key == 'less_dists':  
                            side_data = sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_mean_diff[i_3].text(
                                    sig_plot_ind-0.25, sig_height, '<0', horizontalalignment='center', \
                                        verticalalignment='center',color='r')
                        else: # Pair data
                            ind_1 = sig_taste_data[key]['ind_1']
                            ind_2 = sig_taste_data[key]['ind_2']
                            marker = sig_taste_data[key]['marker']
                            ax_mean_diff[i_3].plot(
                                [ind_1, ind_2], [sig_height, sig_height], color='k', linestyle='solid')
                            ax_mean_diff[i_3].plot([ind_1, ind_1], [
                                                      sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                            ax_mean_diff[i_3].plot([ind_2, ind_2], [
                                                      sig_height-step/2, sig_height+step/2], color='k', linestyle='solid')
                            ax_mean_diff[i_3].text(
                                ind_1 + (ind_2-ind_1)/2, sig_height + step/2, marker,\
                                    horizontalalignment='center', verticalalignment='center',\
                                        color='k')
                            sig_height += step
                    # Adjust y-limits
                    # ax_mean_diff[i_3].set_ylim(
                    #     [min_mean_diff - np.abs(min_mean_diff)/5, sig_height + sig_height/5])
                    
                    # Add best significance data
                    step = max_best_mean_diff/10
                    best_sig_height = max_best_mean_diff + step
                    best_sig_taste_data = best_significance_storage[i_3]
                    for key in list(best_sig_taste_data.keys()):
                        if key == 'sig_dists':  # Individual dist significance
                            dist_sig_data = best_sig_taste_data[key]
                            for sig_plot_ind in dist_sig_data:
                                ax_best_mean_diff[i_3].text(
                                    sig_plot_ind, best_sig_height, '*', horizontalalignment='center', \
                                        verticalalignment='center',color='g')
                        elif key == 'greater_dists':  
                            side_data = best_sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_best_mean_diff[i_3].text(
                                    sig_plot_ind+0.25, best_sig_height, '>0', horizontalalignment='center', \
                                        verticalalignment='center',color='b')
                        elif key == 'less_dists':  
                            side_data = best_sig_taste_data[key]
                            for sig_plot_ind in side_data:
                                ax_best_mean_diff[i_3].text(
                                    sig_plot_ind-0.25, best_sig_height, '<0', horizontalalignment='center', \
                                        verticalalignment='center',color='r')
                        else: # Pair data
                            ind_1 = best_sig_taste_data[key]['ind_1']
                            ind_2 = best_sig_taste_data[key]['ind_2']
                            marker = best_sig_taste_data[key]['marker']
                            ax_best_mean_diff[i_3].plot(
                                [ind_1, ind_2], [best_sig_height, best_sig_height], color='k', linestyle='solid')
                            ax_best_mean_diff[i_3].plot([ind_1, ind_1], [
                                                      best_sig_height-step/2, best_sig_height+step/2], color='k', linestyle='solid')
                            ax_best_mean_diff[i_3].plot([ind_2, ind_2], [
                                                      best_sig_height-step/2, best_sig_height+step/2], color='k', linestyle='solid')
                            ax_best_mean_diff[i_3].text(
                                ind_1 + (ind_2-ind_1)/2, best_sig_height + step/2, marker,\
                                    horizontalalignment='center', verticalalignment='center',\
                                        color='k')
                            best_sig_height += step
                            
                    # Adjust y-limits
                    # ax_best_mean_diff[i_3].set_ylim(
                    #     [min_best_mean_diff - np.abs(min_best_mean_diff)/5, best_sig_height + best_sig_height/5])
                    
                # Finish plots with titles and save
                f_pop_vec_plot_name = combo_1.replace(
                    ' ', '_') + '_' + combo_2.replace(' ', '_')
                
                f_mean_diff.suptitle('Mean Correlation Difference: ' +
                                     combo_1.replace(' ', '_') + ' x ' + combo_2.replace(' ', '_'))
                f_mean_diff.tight_layout()
                f_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff.png')
                f_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff.svg')
                plt.close(f_mean_diff)
                f_best_mean_diff.suptitle('Mean Correlation Difference: ' +
                                     combo_1.replace(' ', '_') + ' x ' + combo_2.replace(' ', '_'))
                f_best_mean_diff.tight_layout()
                f_best_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff_best.png')
                f_best_mean_diff.savefig(os.path.join(
                    mean_diff_save, f_pop_vec_plot_name) + '_mean_diff_best.svg')
                plt.close(f_best_mean_diff)

def combined_corr_by_segment_dist(corr_data, min_best_cutoff, save_dir, unique_given_names, unique_corr_names,
                        unique_segment_names, unique_taste_names):
    """This function collects statistics across different correlation types and
    plots them together by segment.
    INPUTS:
            - corr_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    corr_data[name] = dictionary of dataset data
                    corr_data[name]['corr_data'] = dict of length #correlation types
                    corr_data[name]['corr_data'][corr_name] = dict of length #segments
                    corr_data[name]['corr_data'][corr_name][seg_name] = dict of length #tastes
                    corr_data[name]['corr_data'][corr_name][seg_name][taste_name]['data'] = numpy 
                            array of population average vector correlations [num_dev x num_trials x num_epochs]
            - save_dir: directory to save the resulting plots
            - unique_given_names: unique names of datasets
            - unique_corr_names: unique names of correlation analysis types
            - unique_segment_names: unique names of experimental segments
            - unique_taste_names: unique names of tastants
    OUTPUTS: plots and statistical significance tests
    """
    warnings.filterwarnings('ignore')
    
    colors = ['green','royalblue','blueviolet','teal','deeppink', \
              'springgreen','turquoise', 'midnightblue', 'lightskyblue', \
              'palevioletred', 'darkslateblue']
    
    # Create further save dirs
    corr_dist_save = os.path.join(save_dir, 'Corr_Combined_Dist')
    if not os.path.isdir(corr_dist_save):
        os.mkdir(corr_dist_save)
    
    class cross_segment_attributes:
        def __init__(self, combo, names, i_1, i_2, i_3, unique_corr_names,
                     unique_taste_names, unique_epochs):
            setattr(self, names[0], eval(combo[0])[i_1])
            setattr(self, names[1], eval(combo[1])[i_2])
            setattr(self, names[2], eval(combo[2])[i_3])

    # _____Reorganize data by unique correlation type_____
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, min_best_cutoff,
                                                    unique_corr_names, unique_given_names,
                                                    unique_segment_names,unique_taste_names)
    
    # Plot all combinations
    unique_epochs = np.arange(max_epochs)
    characteristic_list = ['unique_corr_names',
                           'unique_taste_names', 'unique_epochs']
    characteristic_dict = dict()
    for cl in characteristic_list:
        characteristic_dict[cl] = eval(cl)
    name_list = ['corr_name', 'taste_name', 'e_i']
    # Get attribute pairs for plotting views
    all_combinations = list(combinations(characteristic_list, 2))
    all_combinations_full = []
    all_names_full = []
    for ac in all_combinations:
        ac_list = list(ac)
        missing = np.setdiff1d(characteristic_list, ac_list)
        full_combo = ac_list
        full_combo.extend(missing)
        all_combinations_full.append(full_combo)
        names_combo = [
            name_list[characteristic_list.index(c)] for c in full_combo]
        all_names_full.append(names_combo)
    # Get segment pairs for comparison
    segment_combinations = list(combinations(unique_segment_names, 2))

    for c_i in range(len(all_combinations_full)):
        combo = all_combinations_full[c_i]
        combo_lengths = [len(characteristic_dict[combo[i]])
                         for i in range(len(combo))]
        names = all_names_full[c_i]
        for i_1 in range(combo_lengths[0]):
            combo_1 = eval(combo[0])[i_1]
            if type(combo_1) == np.int64:
                combo_1 = "epoch_" + str(combo_1)
            for i_2 in range(combo_lengths[1]):
                combo_2 = eval(combo[1])[i_2]
                if type(combo_2) == np.int64:
                    combo_2 = "epoch_" + str(combo_2)
                f_cdf, ax_cdf = plt.subplots(
                    ncols=combo_lengths[2], nrows=1, \
                        figsize=(combo_lengths[2]*5, 5), \
                            sharex = True, sharey = True)
                f_pdf, ax_pdf = plt.subplots(
                    ncols=combo_lengths[2], nrows=1, \
                        figsize=(combo_lengths[2]*5, 5), \
                            sharex = True, sharey = True)
                f_cdf_best, ax_cdf_best = plt.subplots(
                    ncols=combo_lengths[2], nrows=1, \
                        figsize=(combo_lengths[2]*5, 5), \
                            sharex = True, sharey = True)
                f_pdf_best, ax_pdf_best = plt.subplots(
                    ncols=combo_lengths[2], nrows=1, \
                        figsize=(combo_lengths[2]*5, 5), \
                            sharex = True, sharey = True)
                
                corr_storage = dict()
                best_corr_storage = dict()
                for i_3 in range(combo_lengths[2]):
                    combo_3 = eval(combo[2])[i_3]
                    if type(combo_3) == np.int64:
                        combo_3 = "epoch_" + str(combo_3)
                    corr_storage[combo_3] = dict()
                    best_corr_storage[combo_3] = dict()
                    # Begin pulling data
                    att = cross_segment_attributes(combo, names, i_1, i_2, i_3, unique_corr_names,
                                                   unique_taste_names, unique_epochs)
                    corr_name = att.corr_name
                    taste_name = att.taste_name
                    e_i = att.e_i
                    for s_i, seg in enumerate(unique_segment_names):
                        corr_storage[combo_3][seg] = []
                        best_corr_storage[combo_3][seg] = []
                        for g_n in unique_given_names:
                            try:
                                #Regular data
                                data = unique_data_dict[corr_name][g_n][seg][taste_name]['data'][:, e_i]
                                corr_storage[combo_3][seg].extend(list(data))
                            except:
                                print("\tSkipping invalid dataset.")
                            try:
                                #Best data
                                data = unique_best_data_dict[corr_name][g_n][seg][taste_name][e_i]
                                best_corr_storage[combo_3][seg].extend(list(data))
                            except:
                                print("\tSkipping invalid dataset.")
                        true_vals = len(corr_storage[combo_3][seg])
                        best_vals = len(best_corr_storage[combo_3][seg])
                        cdf_bins = min(1000,max(int(true_vals/100),20))
                        pdf_bins = min(200,max(int(true_vals/1000),20))
                        cdf_best_bins = min(1000,max(int(best_vals/20),20))
                        pdf_best_bins = min(200,max(int(best_vals/50),20))
                        cdf_data = np.array(corr_storage[combo_3][seg])
                        nonnan_data = cdf_data[~np.isnan(cdf_data)]
                        ax_cdf[i_3].ecdf(nonnan_data,label=seg,color=colors[s_i])
                        # ax_cdf[i_3].hist(corr_storage[combo_3][seg],bins=cdf_bins,histtype='step',\
                        #          density=True,cumulative=True,label=seg,color=colors[s_i])
                        ax_pdf[i_3].hist(corr_storage[combo_3][seg],bins=pdf_bins,histtype='step',\
                                 density=True,cumulative=False,label=seg,color=colors[s_i])
                        cdf_data = np.array(best_corr_storage[combo_3][seg])
                        nonnan_data = cdf_data[~np.isnan(cdf_data)]
                        ax_cdf_best[i_3].ecdf(nonnan_data,\
                                              label=seg,color=colors[s_i])
                        # ax_cdf_best[i_3].hist(best_corr_storage[combo_3][seg],\
                        #         bins=cdf_best_bins,histtype='step',\
                        #             density=True,cumulative=True,label=seg,color=colors[s_i])
                        ax_pdf_best[i_3].hist(best_corr_storage[combo_3][seg],\
                                bins=pdf_best_bins,histtype='step',\
                                    density=True,cumulative=False,label=seg,color=colors[s_i])
                    #Now add significance
                    reg_sig = ''
                    best_sig = ''
                    for sp_i, s_pair in enumerate(segment_combinations):
                        seg_1 = s_pair[0]
                        seg_2 = s_pair[1]
                        pair_sig = seg_1 + ' x ' + seg_2
                        best_pair_sig = seg_1 + ' x ' + seg_2
                        #Start with regular corr significance
                        data_1 = np.array(corr_storage[combo_3][seg_1])
                        data_1 = data_1[~np.isnan(data_1)]
                        data_2 = np.array(corr_storage[combo_3][seg_2])
                        data_2 = data_2[~np.isnan(data_2)]
                        if (len(data_1) > 1)*(len(data_2) > 1):
                            ks_result = ks_2samp(data_1,data_2)
                            if ks_result[1] <= 0.05:
                                pair_sig = pair_sig + ' K.S.= *'
                            else:
                                pair_sig = pair_sig + ' K.S.= n.s.'
                            tt_result = ttest_ind(data_1,data_2)
                            if tt_result[1] <= 0.05:
                                pair_sig = pair_sig + ' T.T.= *'
                            else:
                                pair_sig = pair_sig + ' T.T.= n.s.'
                            pair_sig = pair_sig + '\n'
                            reg_sig = reg_sig + pair_sig
                        
                        #Now best corr significance
                        best_data_1 = np.array(best_corr_storage[combo_3][seg_1])
                        best_data_1 = best_data_1[~np.isnan(best_data_1)]
                        best_data_2 = np.array(best_corr_storage[combo_3][seg_2])
                        best_data_2 = best_data_2[~np.isnan(best_data_2)]
                        if (len(best_data_1) > 1)*(len(best_data_2) > 1):
                            ks_result = ks_2samp(best_data_1,best_data_2)
                            if ks_result[1] <= 0.05:
                                best_pair_sig = best_pair_sig + ' K.S.= *'
                            else:
                                best_pair_sig = best_pair_sig + ' K.S.= n.s.'
                            tt_result = ttest_ind(best_data_1,best_data_2)
                            if tt_result[1] <= 0.05:
                                best_pair_sig = best_pair_sig + ' T.T.= *'
                            else:
                                best_pair_sig = best_pair_sig + ' T.T.= n.s.'
                            best_pair_sig = best_pair_sig + '\n'
                            best_sig = best_sig + best_pair_sig
                    #Plot significance
                    ax_cdf[i_3].text(-0.5,0.5,reg_sig,horizontalalignment='left',\
                                     verticalalignment='top',color='k',\
                                     backgroundcolor='w',fontsize='x-small')
                    ax_cdf_best[i_3].text(-0.5,0.5,best_sig,horizontalalignment='left',\
                                     verticalalignment='top',color='k',\
                                     backgroundcolor='w',fontsize='x-small')
                    
                    #Add legends
                    ax_cdf[i_3].legend(loc='upper left')
                    ax_pdf[i_3].legend(loc='upper left')
                    ax_cdf_best[i_3].legend(loc='upper left')
                    ax_pdf_best[i_3].legend(loc='upper left')
                    #Add title and axes labels
                    ax_cdf[i_3].set_title(combo_3)
                    ax_pdf[i_3].set_title(combo_3)
                    ax_cdf_best[i_3].set_title(combo_3)
                    ax_pdf_best[i_3].set_title(combo_3)
                    ax_cdf[i_3].set_xlabel('Correlation')
                    ax_pdf[i_3].set_xlabel('Correlation')
                    ax_cdf_best[i_3].set_xlabel('Correlation')
                    ax_pdf_best[i_3].set_xlabel('Correlation')
                    ax_cdf[i_3].set_ylabel('Cumulative Density')
                    ax_pdf[i_3].set_ylabel('Density')
                    ax_cdf_best[i_3].set_ylabel('Cumulative Density')
                    ax_pdf_best[i_3].set_ylabel('Density')
                #Add super title
                analysis_name = combo_1 +'_x_' + combo_2
                f_cdf.suptitle(analysis_name)
                plt.tight_layout()
                f_cdf.savefig(os.path.join(corr_dist_save,analysis_name + '_cdf.png'))
                f_cdf.savefig(os.path.join(corr_dist_save,analysis_name + '_cdf.svg'))
                plt.setp(ax_cdf, xlim=[0.5,1], ylim=[0.5,1])
                f_cdf.savefig(os.path.join(corr_dist_save,analysis_name + '_cdf_zoom.png'))
                f_cdf.savefig(os.path.join(corr_dist_save,analysis_name + '_cdf_zoom.svg'))
                plt.close(f_cdf)
                
                f_pdf.suptitle(analysis_name)
                plt.tight_layout()
                f_pdf.savefig(os.path.join(corr_dist_save,analysis_name + '_pdf.png'))
                f_pdf.savefig(os.path.join(corr_dist_save,analysis_name + '_pdf.svg'))
                plt.close(f_pdf)
                
                f_cdf_best.suptitle(analysis_name + '_best')
                plt.tight_layout()
                f_cdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_cdf.png'))
                f_cdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_cdf.svg'))
                plt.setp(ax_cdf_best, xlim=[0.5,1], ylim=[0.5,1])
                f_cdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_cdf_zoom.png'))
                f_cdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_cdf_zoom.svg'))
                plt.close(f_cdf_best)
                
                f_pdf_best.suptitle(analysis_name + '_best')
                plt.tight_layout()
                f_pdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_pdf.png'))
                f_pdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_pdf.svg'))
                plt.close(f_pdf_best)

def combined_corr_by_taste_dist(corr_data, min_best_cutoff, save_dir, unique_given_names, unique_corr_names,
                      unique_segment_names, unique_taste_names):
    """This function collects statistics across different correlation types and
    plots them together
    INPUTS:
            - corr_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    corr_data[name] = dictionary of dataset data
                    corr_data[name]['corr_data'] = dict of length #correlation types
                    corr_data[name]['corr_data'][corr_name] = dict of length #segments
                    corr_data[name]['corr_data'][corr_name][seg_name] = dict of length #tastes
                    corr_data[name]['corr_data'][corr_name][seg_name][taste_name]['data'] = numpy 
                            array of population average vector correlations [num_dev x num_trials x num_epochs]
            - results_dir: directory to save the resulting plots
            - unique_corr_names: unique names of correlation analyses to compare
    OUTPUTS: plots and statistical significance tests
    """
    # Set parameters
    warnings.filterwarnings('ignore')
    
    colors = ['green','royalblue','blueviolet','teal','deeppink', \
              'springgreen','turquoise', 'midnightblue', 'lightskyblue', \
              'palevioletred', 'darkslateblue']
    
    # Create further save dirs
    corr_dist_save = os.path.join(save_dir, 'Corr_Combined_Dist')
    if not os.path.isdir(corr_dist_save):
        os.mkdir(corr_dist_save)

    class cross_taste_attributes:
        def __init__(self, combo, names, i_1, i_2, i_3, unique_corr_names,
                     unique_segment_names, unique_epochs):
            setattr(self, names[0], eval(combo[0])[i_1])
            setattr(self, names[1], eval(combo[1])[i_2])
            setattr(self, names[2], eval(combo[2])[i_3])

    # _____Reorganize data by unique correlation type_____
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, min_best_cutoff, 
                                                        unique_corr_names, unique_given_names,
                                                        unique_segment_names,unique_taste_names)

    # Plot all combinations
    unique_epochs = np.arange(max_epochs)
    characteristic_list = ['unique_corr_names',
                           'unique_segment_names', 'unique_epochs']
    characteristic_dict = dict()
    for cl in characteristic_list:
        characteristic_dict[cl] = eval(cl)
    name_list = ['corr_name', 'seg_name', 'e_i']
    # Get attribute pairs for plotting views
    all_combinations = list(combinations(characteristic_list, 2))
    all_combinations_full = []
    all_names_full = []
    for ac in all_combinations:
        ac_list = list(ac)
        missing = np.setdiff1d(characteristic_list, ac_list)
        full_combo = ac_list
        full_combo.extend(missing)
        all_combinations_full.append(full_combo)
        names_combo = [
            name_list[characteristic_list.index(c)] for c in full_combo]
        all_names_full.append(names_combo)
    # Get segment pairs for comparison
    taste_combinations = list(combinations(unique_taste_names, 2))

    for c_i in range(len(all_combinations_full)):
        combo = all_combinations_full[c_i]
        combo_lengths = [len(characteristic_dict[combo[i]])
                         for i in range(len(combo))]
        names = all_names_full[c_i]
        for i_1 in range(combo_lengths[0]):
            combo_1 = eval(combo[0])[i_1]
            if type(combo_1) == np.int64:
                combo_1 = "epoch_" + str(combo_1)
            for i_2 in range(combo_lengths[1]):
                combo_2 = eval(combo[1])[i_2]
                if type(combo_2) == np.int64:
                    combo_2 = "epoch_" + str(combo_2)
                    f_cdf, ax_cdf = plt.subplots(
                        ncols=combo_lengths[2], nrows=1, \
                            figsize=(combo_lengths[2]*5, 5), \
                                sharex = True, sharey = True)
                    f_pdf, ax_pdf = plt.subplots(
                        ncols=combo_lengths[2], nrows=1, \
                            figsize=(combo_lengths[2]*5, 5), \
                                sharex = True, sharey = True)
                    f_cdf_best, ax_cdf_best = plt.subplots(
                        ncols=combo_lengths[2], nrows=1, \
                            figsize=(combo_lengths[2]*5, 5), \
                                sharex = True, sharey = True)
                    f_pdf_best, ax_pdf_best = plt.subplots(
                        ncols=combo_lengths[2], nrows=1, \
                            figsize=(combo_lengths[2]*5, 5), \
                                sharex = True, sharey = True)
                    corr_storage = dict()
                    best_corr_storage = dict()
                    for i_3 in range(combo_lengths[2]):
                        combo_3 = eval(combo[2])[i_3]
                        if type(combo_3) == np.int64:
                            combo_3 = "epoch_" + str(combo_3)
                        corr_storage[combo_3] = dict()
                        best_corr_storage[combo_3] = dict()
                        # Begin pulling data
                        att = cross_taste_attributes(combo, names, i_1, i_2, i_3, unique_corr_names,
                                                       unique_segment_names, unique_epochs)
                        corr_name = att.corr_name
                        seg_name = att.seg_name
                        e_i = att.e_i
                        for t_i, taste_name in enumerate(unique_taste_names):
                            corr_storage[combo_3][taste_name] = []
                            best_corr_storage[combo_3][taste_name] = []
                            for g_n in unique_given_names:
                                try:
                                    #Regular data
                                    data = unique_data_dict[corr_name][g_n][seg_name][taste_name]['data'][:, e_i]
                                    corr_storage[combo_3][taste_name].extend(list(data))
                                except:
                                    print("\tSkipping invalid dataset.")
                                try:
                                    #Best data
                                    data = unique_best_data_dict[corr_name][g_n][seg_name][taste_name][e_i]
                                    best_corr_storage[combo_3][taste_name].extend(list(data))
                                except:
                                    print("\tSkipping invalid dataset.")
                            true_vals = len(corr_storage[combo_3][taste_name])
                            best_vals = len(best_corr_storage[combo_3][taste_name])
                            cdf_bins = min(1000,max(int(true_vals/100),20))
                            pdf_bins = min(200,max(int(true_vals/1000),20))
                            cdf_best_bins = min(1000,max(int(best_vals/20),20))
                            pdf_best_bins = min(200,max(int(best_vals/50),20))
                            cdf_data = np.array(corr_storage[combo_3][taste_name])
                            nonnan_data = cdf_data[~np.isnan(cdf_data)]
                            ax_cdf[i_3].ecdf(nonnan_data,\
                                             label=taste_name,color=colors[t_i])
                            # ax_cdf[i_3].hist(corr_storage[combo_3][taste_name],bins=cdf_bins,histtype='step',\
                            #          density=True,cumulative=True,label=taste_name,color=colors[t_i])
                            ax_pdf[i_3].hist(corr_storage[combo_3][taste_name],bins=pdf_bins,histtype='step',\
                                     density=True,cumulative=False,label=taste_name,color=colors[t_i])
                            cdf_data = np.array(best_corr_storage[combo_3][taste_name])
                            nonnan_data = cdf_data[~np.isnan(cdf_data)]
                            ax_cdf_best[i_3].ecdf(nonnan_data,\
                                                  label=taste_name,color=colors[t_i])
                            # ax_cdf_best[i_3].hist(best_corr_storage[combo_3][taste_name],\
                            #         bins=cdf_best_bins,histtype='step',\
                            #             density=True,cumulative=True,label=taste_name,color=colors[t_i])
                            ax_pdf_best[i_3].hist(best_corr_storage[combo_3][taste_name],\
                                    bins=pdf_best_bins,histtype='step',\
                                        density=True,cumulative=False,label=taste_name,color=colors[t_i])
                        #Now add significance
                        reg_sig = ''
                        best_sig = ''
                        for tp_i, t_pair in enumerate(taste_combinations):
                            taste_1 = t_pair[0]
                            taste_2 = t_pair[1]
                            pair_sig = taste_1 + ' x ' + taste_2
                            best_pair_sig = taste_1 + ' x ' + taste_2
                            #Start with regular corr significance
                            data_1 = np.array(corr_storage[combo_3][taste_1])
                            data_1 = data_1[~np.isnan(data_1)]
                            data_2 = np.array(corr_storage[combo_3][taste_2])
                            data_2 = data_2[~np.isnan(data_2)]
                            if (len(data_1) > 1)*(len(data_2) > 1):
                                ks_result = ks_2samp(data_1,data_2)
                                if ks_result[1] <= 0.05:
                                    pair_sig = pair_sig + ' K.S.= *'
                                else:
                                    pair_sig = pair_sig + ' K.S.= n.s.'
                                tt_result = ttest_ind(data_1,data_2)
                                if tt_result[1] <= 0.05:
                                    pair_sig = pair_sig + ' T.T.= *'
                                else:
                                    pair_sig = pair_sig + ' T.T.= n.s.'
                                pair_sig = pair_sig + '\n'
                                reg_sig = reg_sig + pair_sig
                            
                            #Now best corr significance
                            best_data_1 = np.array(best_corr_storage[combo_3][taste_1])
                            best_data_1 = best_data_1[~np.isnan(best_data_1)]
                            best_data_2 = np.array(best_corr_storage[combo_3][taste_2])
                            best_data_2 = best_data_2[~np.isnan(best_data_2)]
                            if (len(best_data_1) > 1)*(len(best_data_2) > 1):
                                ks_result = ks_2samp(best_data_1,best_data_2)
                                if ks_result[1] <= 0.05:
                                    best_pair_sig = best_pair_sig + ' K.S.= *'
                                else:
                                    best_pair_sig = best_pair_sig + ' K.S.= n.s.'
                                tt_result = ttest_ind(best_data_1,best_data_2)
                                if tt_result[1] <= 0.05:
                                    best_pair_sig = best_pair_sig + ' T.T.= *'
                                else:
                                    best_pair_sig = best_pair_sig + ' T.T.= n.s.'
                                best_pair_sig = best_pair_sig + '\n'
                                best_sig = best_sig + best_pair_sig
                        ax_cdf[i_3].text(-0.5,0.5,reg_sig,horizontalalignment='left',\
                                         verticalalignment='top',color='k',\
                                         backgroundcolor='w',fontsize='x-small')
                        ax_cdf_best[i_3].text(-0.5,0.5,best_sig,horizontalalignment='left',\
                                         verticalalignment='top',color='k',\
                                         backgroundcolor='w',fontsize='x-small')
                        #Add legends
                        ax_cdf[i_3].legend(loc='upper left')
                        ax_pdf[i_3].legend(loc='upper left')
                        ax_cdf_best[i_3].legend(loc='upper left')
                        ax_pdf_best[i_3].legend(loc='upper left')
                        #Add title and axes labels
                        ax_cdf[i_3].set_title(combo_3)
                        ax_pdf[i_3].set_title(combo_3)
                        ax_cdf_best[i_3].set_title(combo_3)
                        ax_pdf_best[i_3].set_title(combo_3)
                        ax_cdf[i_3].set_xlabel('Correlation')
                        ax_pdf[i_3].set_xlabel('Correlation')
                        ax_cdf_best[i_3].set_xlabel('Correlation')
                        ax_pdf_best[i_3].set_xlabel('Correlation')
                        ax_cdf[i_3].set_ylabel('Cumulative Density')
                        ax_pdf[i_3].set_ylabel('Density')
                        ax_cdf_best[i_3].set_ylabel('Cumulative Density')
                        ax_pdf_best[i_3].set_ylabel('Density')
                    #Add super title
                    analysis_name = combo_1 +'_x_' + combo_2
                    f_cdf.suptitle(analysis_name)
                    plt.tight_layout()
                    f_cdf.savefig(os.path.join(corr_dist_save,analysis_name + '_cdf.png'))
                    f_cdf.savefig(os.path.join(corr_dist_save,analysis_name + '_cdf.svg'))
                    plt.close(f_cdf)
                    f_pdf.suptitle(analysis_name)
                    plt.tight_layout()
                    f_pdf.savefig(os.path.join(corr_dist_save,analysis_name + '_pdf.png'))
                    f_pdf.savefig(os.path.join(corr_dist_save,analysis_name + '_pdf.svg'))
                    plt.close(f_pdf)
                    f_cdf_best.suptitle(analysis_name + '_best')
                    plt.tight_layout()
                    f_cdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_cdf.png'))
                    f_cdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_cdf.svg'))
                    plt.close(f_cdf_best)
                    f_pdf_best.suptitle(analysis_name + '_best')
                    plt.tight_layout()
                    f_pdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_pdf.png'))
                    f_pdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_pdf.svg'))
                    plt.close(f_pdf_best)

def combined_corr_by_epoch_dist(corr_data, min_best_cutoff, save_dir, unique_given_names, unique_corr_names,
                        unique_segment_names, unique_taste_names):
    """This function collects statistics across different correlation types and
    plots them together by segment.
    INPUTS:
            - corr_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    corr_data[name] = dictionary of dataset data
                    corr_data[name]['corr_data'] = dict of length #correlation types
                    corr_data[name]['corr_data'][corr_name] = dict of length #segments
                    corr_data[name]['corr_data'][corr_name][seg_name] = dict of length #tastes
                    corr_data[name]['corr_data'][corr_name][seg_name][taste_name]['data'] = numpy 
                            array of population average vector correlations [num_dev x num_trials x num_epochs]
            - save_dir: directory to save the resulting plots
            - unique_given_names: unique names of datasets
            - unique_corr_names: unique names of correlation analysis types
            - unique_segment_names: unique names of experimental segments
            - unique_taste_names: unique names of tastants
    OUTPUTS: plots and statistical significance tests
    """
    warnings.filterwarnings('ignore')
    
    colors = ['green','royalblue','blueviolet','teal','deeppink', \
              'springgreen','turquoise', 'midnightblue', 'lightskyblue', \
              'palevioletred', 'darkslateblue']
    
    # Create further save dirs
    corr_dist_save = os.path.join(save_dir, 'Corr_Combined_Dist')
    if not os.path.isdir(corr_dist_save):
        os.mkdir(corr_dist_save)
    
    class cross_epoch_attributes:
        def __init__(self, combo, names, i_1, i_2, i_3, unique_corr_names,
                     unique_segment_names, unique_taste_names):
            setattr(self, names[0], eval(combo[0])[i_1])
            setattr(self, names[1], eval(combo[1])[i_2])
            setattr(self, names[2], eval(combo[2])[i_3])

    # _____Reorganize data by unique correlation type_____
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, 
                                                   min_best_cutoff, unique_corr_names,
                                                   unique_given_names, unique_segment_names,
                                                   unique_taste_names)
    
    # Plot all combinations
    unique_epochs = np.arange(max_epochs)
    characteristic_list = ['unique_corr_names',
                           'unique_segment_names', 'unique_taste_names']
    characteristic_dict = dict()
    for cl in characteristic_list:
        characteristic_dict[cl] = eval(cl)
    name_list = ['corr_name', 'seg_name', 'taste_name']
    # Get attribute pairs for plotting views
    all_combinations = list(combinations(characteristic_list, 2))
    all_combinations_full = []
    all_names_full = []
    for ac in all_combinations:
        ac_list = list(ac)
        missing = np.setdiff1d(characteristic_list, ac_list)
        full_combo = ac_list
        full_combo.extend(missing)
        all_combinations_full.append(full_combo)
        names_combo = [
            name_list[characteristic_list.index(c)] for c in full_combo]
        all_names_full.append(names_combo)
    # Get segment pairs for comparison
    epoch_combinations = list(combinations(unique_epochs, 2))

    for c_i in range(len(all_combinations_full)):
        combo = all_combinations_full[c_i]
        combo_lengths = [len(characteristic_dict[combo[i]])
                         for i in range(len(combo))]
        names = all_names_full[c_i]
        for i_1 in range(combo_lengths[0]):
            combo_1 = eval(combo[0])[i_1]
            if type(combo_1) == np.int64:
                combo_1 = "epoch_" + str(combo_1)
            for i_2 in range(combo_lengths[1]):
                combo_2 = eval(combo[1])[i_2]
                if type(combo_2) == np.int64:
                    combo_2 = "epoch_" + str(combo_2)
                f_cdf, ax_cdf = plt.subplots(
                    ncols=combo_lengths[2], nrows=1, \
                        figsize=(combo_lengths[2]*5, 5), \
                            sharex = True, sharey = True)
                f_pdf, ax_pdf = plt.subplots(
                    ncols=combo_lengths[2], nrows=1, \
                        figsize=(combo_lengths[2]*5, 5), \
                            sharex = True, sharey = True)
                f_cdf_best, ax_cdf_best = plt.subplots(
                    ncols=combo_lengths[2], nrows=1, \
                        figsize=(combo_lengths[2]*5, 5), \
                            sharex = True, sharey = True)
                f_pdf_best, ax_pdf_best = plt.subplots(
                    ncols=combo_lengths[2], nrows=1, \
                        figsize=(combo_lengths[2]*5, 5), \
                            sharex = True, sharey = True)
                
                corr_storage = dict()
                best_corr_storage = dict()
                for i_3 in range(combo_lengths[2]):
                    combo_3 = eval(combo[2])[i_3]
                    if type(combo_3) == np.int64:
                        combo_3 = "epoch_" + str(combo_3)
                    corr_storage[combo_3] = dict()
                    best_corr_storage[combo_3] = dict()
                    # Begin pulling data
                    att = cross_epoch_attributes(combo, names, i_1, i_2, i_3, unique_corr_names,
                                                   unique_segment_names, unique_taste_names)
                    corr_name = att.corr_name
                    seg_name = att.seg_name
                    taste_name = att.taste_name
                    for e_i, ep in enumerate(unique_epochs):
                        epoch_name = 'Epoch ' + str(ep)
                        corr_storage[combo_3][ep] = []
                        best_corr_storage[combo_3][ep] = []
                        for g_n in unique_given_names:
                            try:
                                #Regular data
                                data = unique_data_dict[corr_name][g_n][seg_name][taste_name]['data'][:, ep]
                                corr_storage[combo_3][ep].extend(list(data))
                            except:
                                print("\tSkipping invalid dataset.")
                            try:
                                #Best data
                                data = unique_best_data_dict[corr_name][g_n][seg_name][taste_name][ep]
                                best_corr_storage[combo_3][ep].extend(list(data))
                            except:
                                print("\tSkipping invalid dataset.")
                        true_vals = len(corr_storage[combo_3][ep])
                        best_vals = len(best_corr_storage[combo_3][ep])
                        cdf_bins = min(1000,max(int(true_vals/100),20))
                        pdf_bins = min(200,max(int(true_vals/1000),20))
                        cdf_best_bins = min(1000,max(int(best_vals/20),20))
                        pdf_best_bins = min(200,max(int(best_vals/50),20))
                        cdf_data = np.array(corr_storage[combo_3][ep])
                        nonnan_data = cdf_data[~np.isnan(cdf_data)]
                        ax_cdf[i_3].ecdf(nonnan_data,label=epoch_name,\
                                         color=colors[e_i])
                        # ax_cdf[i_3].hist(corr_storage[combo_3][ep],bins=cdf_bins,histtype='step',\
                        #          density=True,cumulative=True,label=epoch_name,color=colors[e_i])
                        ax_pdf[i_3].hist(corr_storage[combo_3][ep],bins=pdf_bins,histtype='step',\
                                 density=True,cumulative=False,label=epoch_name,color=colors[e_i])
                        cdf_data = np.array(best_corr_storage[combo_3][ep])
                        nonnan_data = cdf_data[~np.isnan(cdf_data)]
                        ax_cdf_best[i_3].ecdf(cdf_data,\
                                              label=epoch_name,color=colors[e_i])
                        # ax_cdf_best[i_3].hist(best_corr_storage[combo_3][ep],\
                        #         bins=cdf_best_bins,histtype='step',\
                        #             density=True,cumulative=True,label=epoch_name,color=colors[e_i])
                        ax_pdf_best[i_3].hist(best_corr_storage[combo_3][ep],\
                                bins=pdf_best_bins,histtype='step',\
                                    density=True,cumulative=False,label=epoch_name,color=colors[e_i])
                    #Now add significance
                    reg_sig = ''
                    best_sig = ''
                    for ep_i, ep in enumerate(epoch_combinations):
                        epoch_1 = ep[0]
                        epoch_2 = ep[1]
                        pair_sig = 'Epoch ' + str(epoch_1) + ' x Epoch ' + str(epoch_2)
                        best_pair_sig = 'Epoch ' + str(epoch_1) + ' x Epoch ' + str(epoch_2)
                        #Start with regular corr significance
                        data_1 = np.array(corr_storage[combo_3][epoch_1])
                        data_1 = data_1[~np.isnan(data_1)]
                        data_2 = np.array(corr_storage[combo_3][epoch_2])
                        data_2 = data_2[~np.isnan(data_2)]
                        if (len(data_1) > 1)*(len(data_2) > 1):
                            ks_result = ks_2samp(data_1,data_2)
                            if ks_result[1] <= 0.05:
                                pair_sig = pair_sig + ' K.S.= *'
                            else:
                                pair_sig = pair_sig + ' K.S.= n.s.'
                            tt_result = ttest_ind(data_1,data_2)
                            if tt_result[1] <= 0.05:
                                pair_sig = pair_sig + ' T.T.= *'
                            else:
                                pair_sig = pair_sig + ' T.T.= n.s.'
                            pair_sig = pair_sig + '\n'
                            reg_sig = reg_sig + pair_sig
                        
                        #Now best corr significance
                        best_data_1 = np.array(best_corr_storage[combo_3][epoch_1])
                        best_data_1 = best_data_1[~np.isnan(best_data_1)]
                        best_data_2 = np.array(best_corr_storage[combo_3][epoch_2])
                        best_data_2 = best_data_2[~np.isnan(best_data_2)]
                        if (len(best_data_1) > 1)*(len(best_data_2) > 1):
                            ks_result = ks_2samp(best_data_1,best_data_2)
                            if ks_result[1] <= 0.05:
                                best_pair_sig = best_pair_sig + ' K.S.= *'
                            else:
                                best_pair_sig = best_pair_sig + ' K.S.= n.s.'
                            tt_result = ttest_ind(best_data_1,best_data_2)
                            if tt_result[1] <= 0.05:
                                best_pair_sig = best_pair_sig + ' T.T.= *'
                            else:
                                best_pair_sig = best_pair_sig + ' T.T.= n.s.'
                            best_pair_sig = best_pair_sig + '\n'
                            best_sig = best_sig + best_pair_sig
                    #Plot significance
                    ax_cdf[i_3].text(-0.5,0.5,reg_sig,horizontalalignment='left',\
                                     verticalalignment='top',color='k',\
                                     backgroundcolor='w',fontsize='x-small')
                    ax_cdf_best[i_3].text(-0.5,0.5,best_sig,horizontalalignment='left',\
                                     verticalalignment='top',color='k',\
                                     backgroundcolor='w',fontsize='x-small')
                    
                    #Add legends
                    ax_cdf[i_3].legend(loc='upper left')
                    ax_pdf[i_3].legend(loc='upper left')
                    ax_cdf_best[i_3].legend(loc='upper left')
                    ax_pdf_best[i_3].legend(loc='upper left')
                    #Add title and axes labels
                    ax_cdf[i_3].set_title(combo_3)
                    ax_pdf[i_3].set_title(combo_3)
                    ax_cdf_best[i_3].set_title(combo_3)
                    ax_pdf_best[i_3].set_title(combo_3)
                    ax_cdf[i_3].set_xlabel('Correlation')
                    ax_pdf[i_3].set_xlabel('Correlation')
                    ax_cdf_best[i_3].set_xlabel('Correlation')
                    ax_pdf_best[i_3].set_xlabel('Correlation')
                    ax_cdf[i_3].set_ylabel('Cumulative Density')
                    ax_pdf[i_3].set_ylabel('Density')
                    ax_cdf_best[i_3].set_ylabel('Cumulative Density')
                    ax_pdf_best[i_3].set_ylabel('Density')
                #Add super title
                analysis_name = combo_1 +'_x_' + combo_2
                f_cdf.suptitle(analysis_name)
                plt.tight_layout()
                f_cdf.savefig(os.path.join(corr_dist_save,analysis_name + '_cdf.png'))
                f_cdf.savefig(os.path.join(corr_dist_save,analysis_name + '_cdf.svg'))
                plt.setp(ax_cdf, xlim=[0.5,1], ylim=[0.5,1])
                f_cdf.savefig(os.path.join(corr_dist_save,analysis_name + '_cdf_zoom.png'))
                f_cdf.savefig(os.path.join(corr_dist_save,analysis_name + '_cdf_zoom.svg'))
                plt.close(f_cdf)
                
                f_pdf.suptitle(analysis_name)
                plt.tight_layout()
                f_pdf.savefig(os.path.join(corr_dist_save,analysis_name + '_pdf.png'))
                f_pdf.savefig(os.path.join(corr_dist_save,analysis_name + '_pdf.svg'))
                plt.close(f_pdf)
                
                f_cdf_best.suptitle(analysis_name + '_best')
                plt.tight_layout()
                f_cdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_cdf.png'))
                f_cdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_cdf.svg'))
                plt.setp(ax_cdf_best, xlim=[0.5,1], ylim=[0.5,1])
                f_cdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_cdf_zoom.png'))
                f_cdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_cdf_zoom.svg'))
                plt.close(f_cdf_best)
                
                f_pdf_best.suptitle(analysis_name + '_best')
                plt.tight_layout()
                f_pdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_pdf.png'))
                f_pdf_best.savefig(os.path.join(corr_dist_save,analysis_name + '_best_pdf.svg'))
                plt.close(f_pdf_best)

def reorg_data_dict(corr_data, min_best_cutoff, unique_corr_names, unique_given_names, unique_segment_names,
                    unique_taste_names):
    # _____Reorganize data by unique correlation type_____
    unique_data_dict = dict()
    for ucn in unique_corr_names:
        unique_data_dict[ucn] = dict()
        for udn in unique_given_names:
            unique_data_dict[ucn][udn] = dict()
            for usn in unique_segment_names:
                unique_data_dict[ucn][udn][usn] = dict()
                for utn in unique_taste_names:
                    unique_data_dict[ucn][udn][usn][utn] = dict()
    
    max_epochs = 0
    for d_i in corr_data:  # Each dataset
        dataset = corr_data[d_i]
        given_name = d_i
        dataset_corr_data = dataset['corr_data']
        corr_names = list(dataset_corr_data.keys())
        for cn_i in corr_names:
            corr_name = cn_i
            corr_dev_stats = dataset_corr_data[cn_i]
            seg_names = list(corr_dev_stats.keys())
            for seg_name in seg_names:
                taste_names = list(np.intersect1d(list(corr_dev_stats[seg_name].keys()),unique_taste_names))
                for taste_name in taste_names:
                    try:
                        data = corr_dev_stats[seg_name][taste_name]['data']
                        num_epochs = data.shape[-1]
                        if num_epochs > max_epochs:
                            max_epochs = num_epochs
                        unique_data_dict[corr_name][given_name][seg_name][taste_name]['data'] = data.reshape(
                            -1, data.shape[-1])
                    except:
                        unique_data_dict[corr_name][given_name][seg_name][taste_name]['data'] = np.array([])
    
    # _____Reorganize best corr data by unique correlation type_____
    unique_best_data_dict = dict()
    for ucn in unique_corr_names:
        unique_best_data_dict[ucn] = dict()
        for udn in unique_given_names:
            unique_best_data_dict[ucn][udn] = dict()
            for usn in unique_segment_names:
                unique_best_data_dict[ucn][udn][usn] = dict()
                for utn in unique_taste_names:
                    unique_best_data_dict[ucn][udn][usn][utn] = dict()
    
    for d_i in corr_data:  # Each dataset
        dataset = corr_data[d_i]
        given_name = d_i
        dataset_corr_data = dataset['corr_data']
        corr_names = list(dataset_corr_data.keys())
        for cn_i in corr_names:
            corr_name = cn_i
            corr_dev_stats = dataset_corr_data[cn_i]
            seg_names = list(corr_dev_stats.keys())
            for seg_name in seg_names:
                data = corr_dev_stats[seg_name]
                taste_names = list(np.intersect1d(list(corr_dev_stats[seg_name].keys()),unique_taste_names))
                try:
                    num_dev = data[taste_names[0]]['num_dev']
                    num_cp = data[taste_names[0]]['num_cp']
                    #Store the index of [0] the best taste and [1] the best epoch
                    #   Note, this is on-average the correlation is best across deliveries
                    best_data = data['best'] #num_dev x 2 (col 1 = taste, col2 = epoch)
                    #Calculate average corrs across deliveries
                    dev_t_e_corrs = np.zeros((num_dev,len(taste_names)-1,num_cp))
                    for t_i, taste in enumerate(taste_names[:-1]):
                        dev_corrs = data[taste]['data']
                        dev_corr_means = np.nanmean(dev_corrs,1)
                        dev_t_e_corrs[:,t_i,:] = np.squeeze(dev_corr_means)
                    dev_t_e_corrs_flat = np.reshape(dev_t_e_corrs,(num_dev,(len(taste_names)-1)*num_cp))
                    dev_corr_max = np.max(dev_t_e_corrs_flat,1)
                    min_best_cutoff = np.percentile(dev_corr_max,75)
                    rep_candidate_dev = np.where(dev_corr_max > min_best_cutoff)[0]
                    #Now store the best correlation values by taste and epoch
                    for t_i, taste in enumerate(taste_names):
                        for cp_i in range(num_cp):
                            corr_list = []
                            for dev_i in rep_candidate_dev:
                                corr_list.extend(data[taste]['data'][dev_i,:,cp_i])
                                #corr_list.append(np.nanmean(data[taste]['data'][dev_i,:,cp_i]))
                            unique_best_data_dict[cn_i][d_i][seg_name][taste][cp_i] = corr_list
                except: #No data
                    for t_i, taste in enumerate(taste_names):
                        for cp_i in range(max_epochs):
                            unique_best_data_dict[cn_i][d_i][seg_name][taste][cp_i] = []
                        
    return unique_data_dict, unique_best_data_dict, max_epochs

def cross_dataset_seg_compare_means(seg_data,unique_given_names,unique_analysis_names,
                              unique_segment_names,unique_bin_sizes,save_dir):
    """This function takes data across animals and compares across segments the
    distribution means by bin size
    INPUTS:
            - seg_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    seg_data[name] = dictionary of dataset data
                    seg_data[name]['seg_data'] = dict of length # analysis types
                    seg_data[name]['seg_data'][analysis_names] = dict of length #segments
                    seg_data[name]['seg_data'][analysis_names][seg_name] = dict of length #bins
                    seg_data[name]['seg_data'][analysis_names][seg_name][bin_size] = numpy array of analysis data for that bin size
            - unique_given_names: unique names of datasets
            - unique_analysis_names: unique names of analysis types
            - unique_segment_names: unique names of experimental segments
            - unique_bin_sizes: unique bin sizes from individual animal segment analyses
            - save_dir: where to save resulting plots
    OUTPUTS: plots and statistical significance tests comparing segments to each other
    """
    
    num_bins = len(unique_bin_sizes)
    subplot_square = np.ceil(np.sqrt(num_bins)).astype('int')
    subplot_inds_square = np.reshape(np.arange(subplot_square**2),(subplot_square,subplot_square))
    
    for analysis in unique_analysis_names:
        try:
            f_mean, ax_mean = plt.subplots(nrows = subplot_square, ncols = subplot_square, figsize = (3*subplot_square,3*subplot_square))
            for bs_i, bs in enumerate(unique_bin_sizes):
                bs_string = str(bs)
                bs_data_collection = np.nan*np.ones((len(unique_given_names),len(unique_segment_names)))
                for name_ind, name in enumerate(unique_given_names):
                    for seg_ind, seg_name in enumerate(unique_segment_names):
                        try:
                            animal_seg_data = seg_data[name]['seg_data'][analysis][seg_name][bs_string]
                            bs_data_collection[name_ind,seg_ind] = np.nanmean(animal_seg_data)
                        except:
                            print(seg_name + " data not found for " + analysis + ' ' + name)
                max_val = np.nanmax(bs_data_collection)*1.05
                min_val = np.nanmin(bs_data_collection)*0.95
                bs_i_square = np.where(subplot_inds_square == bs_i)
                ax_mean[bs_i_square[0][0],bs_i_square[1][0]].boxplot(bs_data_collection)
                ax_mean[bs_i_square[0][0],bs_i_square[1][0]].set_xticklabels(unique_segment_names,rotation=45)
                ax_mean[bs_i_square[0][0],bs_i_square[1][0]].set_title('Bin Size = ' + str(np.round(bs,2)))
                for seg_ind, seg_name in enumerate(unique_segment_names):
                    animal_points = bs_data_collection[:,seg_ind]
                    nonnan_points = animal_points[~np.isnan(animal_points)]
                    animal_x_jitter = 0.1*np.random.randn(len(nonnan_points))
                    ax_mean[bs_i_square[0][0],bs_i_square[1][0]].scatter((seg_ind+1)*np.ones(len(nonnan_points)) + animal_x_jitter, nonnan_points, alpha=0.3, color='g')
                #Now calculate pairwise significance
                seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
                for sp in seg_pairs:
                    seg_1 = sp[0]
                    seg_1_data = bs_data_collection[:,seg_1]
                    seg_2 = sp[1]
                    seg_2_data = bs_data_collection[:,seg_2]
                    ttest_result = ttest_ind(seg_1_data[~np.isnan(seg_1_data)],seg_2_data[~np.isnan(seg_2_data)])
                    if ttest_result[1] <= 0.05:
                        ax_mean[bs_i_square[0][0],bs_i_square[1][0]].plot([seg_1+1, seg_2+1],[max_val,max_val],color='k')
                        max_val = max_val*1.05
                        ax_mean[bs_i_square[0][0],bs_i_square[1][0]].scatter([seg_1+1+(seg_2-seg_1)/2],[max_val],marker='*',s=3,c='k')
                        max_val = max_val*1.05
                ax_mean[bs_i_square[0][0],bs_i_square[1][0]].set_ylim([min_val,max_val])
            plt.suptitle(analysis)
            plt.tight_layout()
            f_mean.savefig(os.path.join(save_dir,analysis + '_means.png'))
            f_mean.savefig(os.path.join(save_dir,analysis + '_means.svg'))
            plt.close(f_mean)
        except:
            #No bins in this data!
            f = plt.figure(figsize=(5,5))
            data_means = np.nan*np.ones((len(unique_given_names),len(unique_segment_names)))
            for name_ind, name in enumerate(unique_given_names):
                for seg_ind, seg_name in enumerate(unique_segment_names):
                    animal_seg_data = seg_data[name]['seg_data'][analysis][seg_name]
                    data_means[name_ind,seg_ind] = np.nanmean(animal_seg_data)
            max_val = np.nanmax(data_means)*1.05
            min_val = np.nanmin(data_means)*0.95
            plt.boxplot(data_means)
            plt.xticks(np.arange(len(unique_segment_names)) + 1,unique_segment_names,rotation=45)
            plt.title(analysis)
            for seg_ind, seg_name in enumerate(unique_segment_names):
                animal_points = data_means[:,seg_ind]
                nonnan_points = animal_points[~np.isnan(animal_points)]
                animal_x_jitter = 0.1*np.random.randn(len(nonnan_points))
                plt.scatter((seg_ind+1)*np.ones(len(nonnan_points)) + animal_x_jitter, nonnan_points, alpha=0.3, color='g')
            #Now calculate pairwise significance
            seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
            for sp in seg_pairs:
                seg_1 = sp[0]
                seg_1_data = bs_data_collection[:,seg_1]
                seg_2 = sp[1]
                seg_2_data = bs_data_collection[:,seg_2]
                ttest_result = ttest_ind(seg_1_data[~np.isnan(seg_1_data)],seg_2_data[~np.isnan(seg_2_data)])
                if ttest_result[1] <= 0.05:
                    plt.plot([seg_1+1, seg_2+1],[max_val,max_val],color='k')
                    max_val = max_val*1.05
                    plt.scatter([seg_1+1+(seg_2-seg_1)/2],[max_val],marker='*',size=3,color='k')
                    max_val = max_val*1.05
            plt.ylim([min_val,max_val])
            plt.tight_layout()
            f.savefig(os.path.join(save_dir,analysis + '_means.png'))
            f.savefig(os.path.join(save_dir,analysis + '_means.svg'))
            plt.close(f_mean)
            
def cross_dataset_seg_compare_combined_dist(seg_data,unique_given_names,unique_analysis_names,
                              unique_segment_names,unique_bin_sizes,save_dir):
    """This function takes data across animals and compares across segments the
    distribution means by bin size
    INPUTS:
            - seg_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    seg_data[name] = dictionary of dataset data
                    seg_data[name]['seg_data'] = dict of length # analysis types
                    seg_data[name]['seg_data'][analysis_names] = dict of length #segments
                    seg_data[name]['seg_data'][analysis_names][seg_name] = dict of length #bins
                    seg_data[name]['seg_data'][analysis_names][seg_name][bin_size] = numpy array of analysis data for that bin size
            - unique_given_names: unique names of datasets
            - unique_analysis_names: unique names of analysis types
            - unique_segment_names: unique names of experimental segments
            - unique_bin_sizes: unique bin sizes from individual animal segment analyses
            - save_dir: where to save resulting plots
    OUTPUTS: plots and statistical significance tests comparing segments to each other
    """
    
    num_bins = len(unique_bin_sizes)
    subplot_square = np.ceil(np.sqrt(num_bins)).astype('int')
    subplot_inds_square = np.reshape(np.arange(subplot_square**2),(subplot_square,subplot_square))
    seg_x_ticks = np.arange(1,len(unique_segment_names)+1)
    for analysis in unique_analysis_names:
        try:
            f_violin, ax_violin = plt.subplots(nrows = subplot_square, ncols = subplot_square, 
                                         figsize = (3*subplot_square,3*subplot_square),
                                         sharex = True, sharey = False)
            for bs_i, bs in enumerate(unique_bin_sizes):
                bs_i_square = np.where(subplot_inds_square == bs_i)
                bs_string = str(bs)
                min_val = np.inf
                max_val = -np.inf
                bs_data_collection = []
                for seg_ind, seg_name in enumerate(unique_segment_names):
                    seg_combined = []
                    for name_ind, name in enumerate(unique_given_names):
                        try:
                            animal_seg_data = seg_data[name]['seg_data'][analysis][seg_name][bs_string]
                            seg_combined.extend(list(animal_seg_data))
                        except:
                            print(seg_name + " data not found for " + analysis + ' ' + name)
                    seg_combined = np.array(seg_combined)
                    seg_combined = seg_combined[~np.isnan(seg_combined)]
                    if np.nanmin(seg_combined) < min_val:
                        min_val = np.nanmin(seg_combined)
                    if np.nanmax(seg_combined) > max_val:
                        max_val = np.nanmax(seg_combined)
                    bs_data_collection.append(seg_combined)
                ax_violin[bs_i_square[0][0],bs_i_square[1][0]].violinplot(bs_data_collection,
                                                            showmeans = True, points = 1000,
                                                            positions = seg_x_ticks)
                ax_violin[bs_i_square[0][0],bs_i_square[1][0]].set_xticks(seg_x_ticks, unique_segment_names)
                #Now calculate pairwise significance
                seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
                pair_sig = ''
                for sp in seg_pairs:
                    seg_1 = sp[0]
                    seg_1_data = bs_data_collection[seg_1]
                    seg_1_data = seg_1_data[~np.isnan(seg_1_data)]
                    seg_2 = sp[1]
                    seg_2_data = bs_data_collection[seg_2]
                    seg_2_data = seg_2_data[~np.isnan(seg_2_data)]
                    if (len(seg_1_data) > 0)*(len(seg_2_data) > 0):
                        test_result = mannwhitneyu(seg_1_data,seg_2_data)
                        if test_result[1] <= 0.05:
                            pair_sig = pair_sig + '\n' + str(sp) + ' MWU = *'
                ax_violin[bs_i_square[0][0],bs_i_square[1][0]].set_title('Bin Size = ' + str(np.round(bs,2)) + pair_sig)
            plt.suptitle(analysis)
            plt.tight_layout()
            f_violin.savefig(os.path.join(save_dir,analysis + '_joint_violins.png'))
            f_violin.savefig(os.path.join(save_dir,analysis + '_joint_violins.svg'))
            plt.close(f_violin)
        except:
            #No bins in this data!
            f = plt.figure(figsize=(5,5))
            data_collection = []
            for seg_ind, seg_name in enumerate(unique_segment_names):
                seg_combined = []
                for name_ind, name in enumerate(unique_given_names):
                    animal_seg_data = seg_data[name]['seg_data'][analysis][seg_name]
                    seg_combined.extend(list(animal_seg_data))
                seg_combined = np.array(seg_combined)
                seg_combined = seg_combined[~np.isnan(seg_combined)]
                data_collection.append(np.array(seg_combined))
            plt.violinplot(data_collection,showmeans = True, points = 1000,
                       positions = seg_x_ticks)
            plt.xticks(seg_x_ticks, unique_segment_names)
            #Now calculate pairwise significance
            seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
            pair_sig = ''
            for sp in seg_pairs:
                seg_1 = sp[0]
                seg_1_data = data_collection[seg_1]
                seg_1_data = seg_1_data[~np.isnan(seg_1_data)]
                seg_2 = sp[1]
                seg_2_data = data_collection[seg_2]
                seg_2_data = seg_2_data[~np.isnan(seg_2_data)]
                if (len(seg_1_data) > 0)*(len(seg_2_data) > 0):
                    test_result = mannwhitneyu(seg_1_data,seg_2_data)
                    if test_result[1] <= 0.05:
                        pair_sig = pair_sig + '\n' + str(sp) + ' MWU = *'
            plt.title(analysis + pair_sig)
            plt.tight_layout()
            f.savefig(os.path.join(save_dir,analysis + '_joint_violins.png'))
            f.savefig(os.path.join(save_dir,analysis + '_joint_violins.svg'))
            plt.close(f)
            
def cross_dataset_seg_compare_mean_diffs(seg_data,unique_given_names,unique_analysis_names,
                              unique_segment_names,unique_bin_sizes,plot_save_dir):
    """This function takes data across animals and compares across segments the
    differences in distribution means by bin size
    INPUTS:
            - seg_data: dictionary containing correlation data across conditions.
                    length = number of datasets
                    seg_data[name] = dictionary of dataset data
                    seg_data[name]['seg_data'] = dict of length # analysis types
                    seg_data[name]['seg_data'][analysis_names] = dict of length #segments
                    seg_data[name]['seg_data'][analysis_names][seg_name] = dict of length #bins
                    seg_data[name]['seg_data'][analysis_names][seg_name][bin_size] = numpy array of analysis data for that bin size
            - unique_given_names: unique names of datasets
            - unique_analysis_names: unique names of analysis types
            - unique_segment_names: unique names of experimental segments
            - unique_bin_sizes: unique bin sizes from individual animal segment analyses
            - plot_save_dir: where to save resulting plots
    OUTPUTS: plots and statistical significance tests comparing segments to each other
    """
    
    num_bins = len(unique_bin_sizes)
    subplot_square = np.ceil(np.sqrt(num_bins)).astype('int')
    subplot_inds_square = np.reshape(np.arange(subplot_square**2),(subplot_square,subplot_square))
    
    for analysis in unique_analysis_names:
        if not os.path.isdir(os.path.join(plot_save_dir,analysis)):
            os.mkdir(os.path.join(plot_save_dir,analysis))
        try:
            #First do each bin individually
            for bs_i, bs in enumerate(unique_bin_sizes):
                bs_string = str(bs)
                f_mean_diff = plt.figure(figsize = (10,7))
                #Collect distribution means
                bs_data_collection = np.nan*np.ones((len(unique_given_names),len(unique_segment_names)))
                for name_ind, name in enumerate(unique_given_names):
                    for seg_ind, seg_name in enumerate(unique_segment_names):
                        try:
                            animal_seg_data = seg_data[name]['seg_data'][analysis][seg_name][bs_string]
                            bs_data_collection[name_ind,seg_ind] = np.nanmean(animal_seg_data)
                        except:
                            print(seg_name + " data not found for " + analysis + ' ' + name)
                #Pairwise for an animal perform mean subtractions
                seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
                bs_diff_data = np.nan*np.ones((len(unique_given_names),len(seg_pairs)))
                bs_diff_labels = []
                for sp_i, sp in enumerate(seg_pairs):
                    seg_1 = sp[0]
                    seg_1_data = bs_data_collection[:,seg_1]
                    seg_1_name = unique_segment_names[seg_1]
                    seg_2 = sp[1]
                    seg_2_data = bs_data_collection[:,seg_2]
                    seg_2_name = unique_segment_names[seg_2]
                    bs_diff_data[:,sp_i] = seg_2_data - seg_1_data
                    bs_diff_labels.extend([seg_2_name + ' - ' + seg_1_name])
                #Plot the results
                max_val = np.nanmax(bs_diff_data)*1.1
                min_val = np.nanmin(bs_diff_data) - 0.05*np.abs(np.nanmin(bs_diff_data))
                plt.axhline(0,alpha=0.3,color='k',linestyle='dashed')
                plt.boxplot(bs_diff_data)
                plt.xticks(np.arange(len(bs_diff_labels))+1,bs_diff_labels,rotation=45)
                plt.title('Bin Size = ' + str(np.round(bs,2)))
                for diff_ind, diff_name in enumerate(bs_diff_labels):
                    animal_points = bs_diff_data[:,diff_ind]
                    nonnan_points = animal_points[~np.isnan(animal_points)]
                    animal_x_jitter = 0.1*np.random.randn(len(nonnan_points))
                    plt.scatter((diff_ind+1)*np.ones(len(nonnan_points)) + animal_x_jitter, nonnan_points, alpha=0.3, color='g')
                #Plot if the distribution is significantly above 0
                for diff_i in range(len(bs_diff_labels)):
                    zero_percentile = percentileofscore(bs_diff_data[:,diff_i],0)
                    if zero_percentile <= 5:
                        plt.scatter([diff_i + 1],[max_val],marker='*',color='k',s=12)
                #Now calculate pairwise significance
                diff_pairs = list(combinations(np.arange(len(bs_diff_labels)),2))
                for dp in diff_pairs:
                    diff_1 = dp[0]
                    diff_1_data = bs_diff_data[:,diff_1]
                    diff_2 = dp[1]
                    diff_2_data = bs_diff_data[:,diff_2]
                    ttest_result = ttest_ind(diff_1_data[~np.isnan(diff_1_data)],diff_2_data[~np.isnan(diff_2_data)])
                    if ttest_result[1] <= 0.05:
                        plt.plot([diff_1+1, diff_2+1],[max_val,max_val],color='r')
                        max_val = max_val*1.05
                        plt.scatter([diff_1+1+(diff_2-diff_1)/2],[max_val],marker='*',s=3,c='r')
                        max_val = max_val*1.05
                plt.ylim([min_val,max_val+1])
                plt.tight_layout()
                f_mean_diff.savefig(os.path.join(plot_save_dir,analysis,'mean_diffs_bin_' + str(bs_i) + '.png'))
                f_mean_diff.savefig(os.path.join(plot_save_dir,analysis,'mean_diffs_bin_' + str(bs_i) + '.svg'))
                plt.close(f_mean_diff)
            #Next combine all bin data into one distribution and calculate mean
            f_mean_diff = plt.figure(figsize = (7,4))
            #Collect distribution means
            bs_data_collection = np.nan*np.ones((len(unique_given_names),len(unique_segment_names)))
            for name_ind, name in enumerate(unique_given_names):
                for seg_ind, seg_name in enumerate(unique_segment_names):
                    all_bin_data = []
                    for bs_i, bs in enumerate(unique_bin_sizes):
                        bs_string = str(bs)
                        try:
                            animal_seg_data = seg_data[name]['seg_data'][analysis][seg_name][bs_string]
                            try:
                                all_bin_data.extend(list(animal_seg_data))
                            except:
                                all_bin_data.extend([animal_seg_data])
                        except:
                            print(seg_name + " data not found for " + analysis + ' ' + name + ' bin ' + bs_string)
                    bs_data_collection[name_ind,seg_ind] = np.nanmean(all_bin_data)
            #Pairwise for an animal perform mean subtractions
            seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
            bs_diff_data = np.nan*np.ones((len(unique_given_names),len(seg_pairs)))
            bs_diff_labels = []
            for sp_i, sp in enumerate(seg_pairs):
                seg_1 = sp[0]
                seg_1_data = bs_data_collection[:,seg_1]
                seg_1_name = unique_segment_names[seg_1]
                seg_2 = sp[1]
                seg_2_data = bs_data_collection[:,seg_2]
                seg_2_name = unique_segment_names[seg_2]
                bs_diff_data[:,sp_i] = seg_2_data - seg_1_data
                bs_diff_labels.extend([seg_2_name + ' - ' + seg_1_name])
            #Plot the results
            max_val = np.nanmax(bs_diff_data)*1.1
            min_val = np.nanmin(bs_diff_data) - 0.05*np.abs(np.nanmin(bs_diff_data))
            plt.axhline(0,alpha=0.3,color='k',linestyle='dashed')
            plt.boxplot(bs_diff_data)
            plt.xticks(np.arange(len(bs_diff_labels))+1,bs_diff_labels,rotation=45)
            plt.title('Mean ' + analysis + ' Across Bins')
            for diff_ind, diff_name in enumerate(bs_diff_labels):
                animal_points = bs_diff_data[:,diff_ind]
                nonnan_points = animal_points[~np.isnan(animal_points)]
                animal_x_jitter = 0.1*np.random.randn(len(nonnan_points))
                plt.scatter((diff_ind+1)*np.ones(len(nonnan_points)) + animal_x_jitter, nonnan_points, alpha=0.3, color='g')
            #Plot if the distribution is significantly above 0
            for diff_i in range(len(bs_diff_labels)):
                zero_percentile = percentileofscore(bs_diff_data[:,diff_i],0)
                if zero_percentile <= 5:
                    plt.scatter([diff_i + 1],[max_val],marker='*',color='k',s=12)
            #Now calculate pairwise significance
            diff_pairs = list(combinations(np.arange(len(bs_diff_labels)),2))
            for dp in diff_pairs:
                diff_1 = dp[0]
                diff_1_data = bs_diff_data[:,diff_1]
                diff_2 = dp[1]
                diff_2_data = bs_diff_data[:,diff_2]
                ttest_result = ttest_ind(diff_1_data[~np.isnan(diff_1_data)],diff_2_data[~np.isnan(diff_2_data)])
                if ttest_result[1] <= 0.05:
                    plt.plot([diff_1+1, diff_2+1],[max_val,max_val],color='r')
                    max_val = max_val*1.05
                    plt.scatter([diff_1+1+(diff_2-diff_1)/2],[max_val],marker='*',s=3,c='r')
                    max_val = max_val*1.05
            plt.ylim([min_val,max_val + 0.25*max_val])
            plt.tight_layout()
            f_mean_diff.savefig(os.path.join(plot_save_dir,analysis,'mean_diffs_cross_bin.png'))
            f_mean_diff.savefig(os.path.join(plot_save_dir,analysis,'mean_diffs_cross_bin.svg'))
            plt.close(f_mean_diff)
        except:
            #No bins in this data!
            f_mean_diff = plt.figure(figsize=(7,4))
            seg_mean_data = np.nan*np.ones((len(unique_given_names),len(unique_segment_names)))
            for name_ind, name in enumerate(unique_given_names):
                for seg_ind, seg_name in enumerate(unique_segment_names):
                    animal_seg_data = seg_data[name]['seg_data'][analysis][seg_name]
                    seg_mean_data[name_ind,seg_ind] = np.nanmean(animal_seg_data)
            #Now calculate pairwise differences
            seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
            bs_diff_data = np.nan*np.ones((len(unique_given_names),len(seg_pairs)))
            bs_diff_labels = []
            for sp_i, sp in enumerate(seg_pairs):
                seg_1 = sp[0]
                seg_1_data = seg_mean_data[:,seg_1]
                seg_1_name = unique_segment_names[seg_1]
                seg_2 = sp[1]
                seg_2_data = seg_mean_data[:,seg_2]
                seg_2_name = unique_segment_names[seg_2]
                bs_diff_data[:,sp_i] = seg_2_data - seg_1_data
                bs_diff_labels.extend([seg_2_name + ' - ' + seg_1_name])
            #Plot the results
            max_val = np.nanmax(bs_diff_data)*1.1
            min_val = np.nanmin(bs_diff_data) - 0.05*np.abs(np.nanmin(bs_diff_data))
            plt.axhline(0,alpha=0.3,color='k',linestyle='dashed')
            plt.boxplot(bs_diff_data)
            plt.xticks(np.arange(len(bs_diff_labels))+1,bs_diff_labels,rotation=45)
            plt.title('Mean ' + analysis + ' Across Bins')
            for diff_ind, diff_name in enumerate(bs_diff_labels):
                animal_points = bs_diff_data[:,diff_ind]
                nonnan_points = animal_points[~np.isnan(animal_points)]
                animal_x_jitter = 0.1*np.random.randn(len(nonnan_points))
                plt.scatter((diff_ind+1)*np.ones(len(nonnan_points)) + animal_x_jitter, nonnan_points, alpha=0.3, color='g')
            #Plot if the distribution is significantly above 0
            for diff_i in range(len(bs_diff_labels)):
                zero_percentile = percentileofscore(bs_diff_data[:,diff_i],0)
                if zero_percentile < 5:
                    plt.scatter([diff_i + 1],[max_val],marker='*',color='k',s=12)
            #Now calculate pairwise significance
            diff_pairs = list(combinations(np.arange(len(bs_diff_labels)),2))
            for dp in diff_pairs:
                diff_1 = dp[0]
                diff_1_data = bs_diff_data[:,diff_1]
                diff_2 = dp[1]
                diff_2_data = bs_diff_data[:,diff_2]
                ttest_result = ttest_ind(diff_1_data[~np.isnan(diff_1_data)],diff_2_data[~np.isnan(diff_2_data)])
                if ttest_result[1] <= 0.05:
                    plt.plot([diff_1+1, diff_2+1],[max_val,max_val],color='r')
                    max_val = max_val*1.05
                    plt.scatter([diff_1+1+(diff_2-diff_1)/2],[max_val],marker='*',s=3,c='r')
                    max_val = max_val*1.05
            plt.ylim([min_val,max_val + 0.25*max_val])
            plt.tight_layout()
            f_mean_diff.savefig(os.path.join(plot_save_dir,analysis,'mean_diffs_cross_bin.png'))
            f_mean_diff.savefig(os.path.join(plot_save_dir,analysis,'mean_diffs_cross_bin.svg'))
            plt.close(f_mean_diff)
        
def cross_dataset_pop_rate_taste_corr_plots(rate_corr_data, unique_given_names, 
                                            unique_corr_types, unique_segment_names, 
                                            unique_taste_names, results_dir):
    """This function is dedicated to plotting statistics across datasets concerning
    correlation between the population firing rate and binned taste correlation
    calculations
    INPUTS:
        - rate_corr_data: dictionary containing rate correlation data with the
            following structure:
                - rate_corr_data[name] = data for specific dataset name
                - rate_corr_data[name]['rate_corr_data'] = dictionary containing correlation data
                - rate_corr_data[name]['rate_corr_data'][corr_type] = data for specific correlation type
                - rate_corr_data[name]['rate_corr_data'][corr_type][seg_name] = data for specific segment
                - rate_corr_data[name]['rate_corr_data'][corr_type][seg_name][taste_name] = # deliveries x # epochs correlation array
        - unique_given_names: list of unique dataset names
        - unique_corr_types: list of unique correlation type names
        - unique_segment_names: list of unique segment names
        - unique_taste_names: list of unique taste names
        - results_dir: directory to save the results
    OUTPUTS:
        - plots with statistical tests of significance
    """    
    
    #Calculate and store means of distributions across animals
    mean_corr_dict = dict()
    for corr_type in unique_corr_types:
        mean_corr_dict[corr_type] = dict()
        for taste in unique_taste_names:
            mean_corr_dict[corr_type][taste] = dict()
            for seg_name in unique_segment_names:
                mean_corr_dict[corr_type][taste][seg_name] = dict()
            
    for corr_type in unique_corr_types:
        for taste_name in unique_taste_names:
                for seg_name in unique_segment_names:
                    seg_epoch_means = []
                    seg_means = []
                    for name in unique_given_names:
                        try:
                            corr_array = rate_corr_data[name]['rate_corr_data'][corr_type][seg_name][taste_name]
                            seg_epoch_means.append(list(np.nanmean(corr_array,0))) #Length = #epochs
                            seg_means.append(np.nanmean(corr_array))
                        except:
                            #print(name + ' does not contain ' + corr_type + ' segment ' + seg_name + ' taste ' + taste_name + ' data.')
                            #Use nan placeholders
                            seg_epoch_means.append([np.nan for e_i in range(3)])
                            seg_means.append([np.nan])
                    mean_corr_dict[corr_type][taste_name][seg_name]['by_epoch'] = np.array(seg_epoch_means)
                    mean_corr_dict[corr_type][taste_name][seg_name]['all'] = np.array(seg_means)

    taste_colors = ['g','b','purple','c','m','k']
    
    #Create trend plots by epoch
    for corr_type in unique_corr_types:
        f, ax = plt.subplots(nrows = len(unique_segment_names), ncols = len(unique_taste_names), \
                             figsize = (4*len(unique_taste_names),4*len(unique_segment_names)),\
                                 sharex=True,sharey=True)
        for s_i, s_name in enumerate(unique_segment_names):
            for t_i, t_name in enumerate(unique_taste_names):
                corr_array = mean_corr_dict[corr_type][t_name][s_name]['by_epoch'] #num animals x num epochs
                if len(corr_array) > 0:
                    corr_max = np.nanmax(corr_array)
                    corr_min = np.nanmin(corr_array)
                    ax[s_i,t_i].axhline(0,linestyle='solid',color='b',alpha=0.3)
                    ax[s_i,t_i].boxplot(corr_array)
                    x_vals = np.arange(np.shape(corr_array)[1])+1
                    for d_i in range(np.shape(corr_array)[0]):
                        animal_x_jitter = 0.1*np.random.randn(len(x_vals))
                        ax[s_i,t_i].scatter(x_vals + animal_x_jitter, corr_array[d_i,:], alpha=0.3, color='g')
                    ax[s_i,t_i].plot(x_vals,np.nanmean(corr_array,0),\
                                     label='Mean',linestyle='dashed',alpha=0.5,color='k')
                    ax[s_i,t_i].set_xticks(x_vals,np.array(['Presence','Identity','Palatability']))
                    if s_i == 0:
                        ax[s_i,t_i].set_title(t_name)
                    if t_i == 0:
                        ax[s_i,t_i].set_ylabel(s_name + '\nMean Correlation')
                    #Now calculate pairwise significance
                    epoch_pairs = list(combinations(np.arange(3),2))
                    for ep in epoch_pairs:
                        e_1 = ep[0]
                        e_1_data = corr_array[:,e_1]
                        e_2 = ep[1]
                        e_2_data = corr_array[:,e_2]
                        ttest_result = ttest_ind(e_1_data[~np.isnan(e_1_data)],e_2_data[~np.isnan(e_2_data)])
                        if ttest_result[1] <= 0.05:
                            ax[s_i,t_i].plot([e_1+1, e_2+1],[corr_max,corr_max],color='k')
                            corr_max = corr_max*1.05
                            ax[s_i,t_i].scatter([e_1+1+(e_2-e_1)/2],[corr_max],marker='*',s=3,c='k')
                            corr_max = corr_max*1.05
                    #Now calculate individual distribution significance > 0
                    for e_i in range(3):
                        zero_percentile = percentileofscore(corr_array[:,e_i],0)
                        if zero_percentile <= 5:
                            corr_max = corr_max*1.05
                            ax[s_i,t_i].text(e_i+1,corr_max,'*%',color='blue')
        f.suptitle(corr_type + ' Mean Correlation of Population Rate to Bin Taste Correlation')
        plt.tight_layout()
        f.savefig(os.path.join(results_dir,corr_type + '_mean_trends_by_epoch.png'))
        f.savefig(os.path.join(results_dir,corr_type + '_mean_trends_by_epoch.svg'))
        plt.close(f)
   
    #Create trend plots of overall means
    for corr_type in unique_corr_types:
        #By Segment
        f_seg, ax_seg = plt.subplots(ncols = len(unique_segment_names), figsize = (4*len(unique_taste_names),4))
        max_corr = 0
        min_corr = 1
        for s_i, s_name in enumerate(unique_segment_names):
            ax_seg[s_i].axhline(0,linestyle='solid',color='b',alpha=0.3)
            taste_data = []
            taste_inds = []
            seg_max_corr = 0
            seg_min_corr = 1
            for t_i, t_name in enumerate(unique_taste_names):
                corr_vec = mean_corr_dict[corr_type][t_name][s_name]['all']
                if len(corr_array) > 1:
                    corr_max = np.nanmax(corr_vec)
                    corr_min = np.nanmin(corr_vec)
                    ax_seg[s_i].scatter(np.random.normal(t_i+1, 0.04,size=len(corr_vec)),
                                  corr_vec, color='g',alpha=0.2)
                    ax_seg[s_i].boxplot([corr_vec],positions=[t_i+1], sym='', \
                                    meanline=True, medianprops=dict(linestyle='-', color='blue'), \
                                    showmeans=True, meanprops=dict(linestyle='-', color='red'), \
                                    showcaps=True, showbox=True)
                    if corr_max > seg_max_corr:
                        seg_max_corr = corr_max
                    if corr_min < seg_min_corr:
                        seg_min_corr = corr_min
                taste_data.append(corr_vec)
                taste_inds.extend([t_i])
            ax_seg[s_i].set_xticks(np.arange(1, len(unique_taste_names)+1), unique_taste_names)
            ax_seg[s_i].set_title(s_name)
            ax_seg[s_i].set_ylabel('Mean Correlation')
            ax_seg[s_i].set_xlabel('Taste')
            #Now calculate pairwise significance
            taste_pairs = list(combinations(np.array(taste_inds),2))
            for tp in taste_pairs:
                t_1 = tp[0]
                t_1_data = taste_data[t_1]
                t_1_data = t_1_data[~np.isnan(t_1_data)]
                t_2 = tp[1]
                t_2_data = taste_data[t_2]
                t_2_data = t_2_data[~np.isnan(t_2_data)]
                ttest_result = ttest_ind(t_1_data,t_2_data)
                if ttest_result[1] <= 0.05:
                    ax_seg[s_i].plot([t_1+1, t_2+1],[corr_max,corr_max],color='k')
                    seg_max_corr = seg_max_corr*1.05
                    ax_seg[s_i].scatter([t_1+1+(t_2-t_1)/2],[corr_max],marker='*',s=3,c='k')
                    seg_max_corr = seg_max_corr*1.05
                ks_result = ks_2samp(t_1_data,t_2_data)
                if ks_result[1] <= 0.05:
                    ax_seg[s_i].plot([t_1+1, t_2+1],[corr_max,corr_max],color='k')
                    seg_max_corr = seg_max_corr*1.05
                    ax_seg[s_i].text(t_1+1+(t_2-t_1)/2,corr_max, '* K',\
                            horizontalalignment='center', verticalalignment='center',\
                                color='r')
                    seg_max_corr = seg_max_corr*1.05 
            #Individual distribution significance > 0
            for t_i in range(len(unique_taste_names)):
                zero_percentile = percentileofscore(taste_data[t_i],0)
                if zero_percentile <= 5:
                    ax_seg[s_i].text(t_i+1,seg_max_corr,'*%',color='blue')
                    seg_max_corr = seg_max_corr*1.05 
                if len(np.where(taste_data[t_i] < 0)[0]) == 0:
                    ax_seg[s_i].text(t_i+1,seg_max_corr,'*>0',color='blue')
                    seg_max_corr = seg_max_corr*1.05 
            if seg_max_corr > max_corr:
                max_corr = seg_max_corr
            if seg_min_corr < min_corr:
                min_corr = seg_min_corr
        for s_i in range(len(unique_segment_names)):
            ax_seg[s_i].set_ylim([min_corr*1.05,max_corr*1.05])
        f_seg.suptitle(corr_type + ' Mean Correlation of Population Rate to Bin Taste Correlation')
        plt.tight_layout()
        f_seg.savefig(os.path.join(results_dir,corr_type + '_by_seg_mean_trends.png'))
        f_seg.savefig(os.path.join(results_dir,corr_type + '_by_seg_mean_trends.svg'))
        plt.close(f_seg)
        #By Taste
        f_taste, ax_taste = plt.subplots(ncols = len(unique_segment_names), figsize = (4*len(unique_taste_names),4))
        max_corr = 0
        min_corr = 1
        for t_i, t_name in enumerate(unique_taste_names):
            ax_taste[t_i].axhline(0,linestyle='solid',color='b',alpha=0.3)
            seg_data = []
            seg_inds = []
            taste_max_corr = 0
            taste_min_corr = 1
            for s_i, s_name in enumerate(unique_segment_names):
                corr_vec = mean_corr_dict[corr_type][t_name][s_name]['all']
                if len(corr_array) > 1:
                    corr_max = np.nanmax(corr_vec)
                    corr_min = np.nanmin(corr_vec)
                    ax_taste[t_i].scatter(np.random.normal(s_i+1, 0.04,size=len(corr_vec)),
                                  corr_vec, color='g',alpha=0.2)
                    ax_taste[t_i].boxplot([corr_vec],positions=[s_i+1], sym='', \
                                    meanline=True, medianprops=dict(linestyle='-', color='blue'), \
                                    showmeans=True, meanprops=dict(linestyle='-', color='red'), \
                                        showcaps=True, showbox=True)
                    if corr_max > taste_max_corr:
                        taste_max_corr = corr_max
                    if corr_min < taste_min_corr:
                        taste_min_corr = corr_min
                seg_data.append(corr_vec)
                seg_inds.extend([s_i])
            ax_taste[t_i].set_xticks(np.arange(1, len(unique_segment_names)+1), unique_segment_names)
            ax_taste[t_i].set_title(t_name)
            ax_taste[t_i].set_ylabel('Mean Correlation')
            ax_taste[t_i].set_xlabel('Segment')
            #Now calculate pairwise significance
            seg_pairs = list(combinations(np.array(seg_inds),2))
            for sp in seg_pairs:
                s_1 = sp[0]
                s_1_data = seg_data[s_1]
                s_1_data = s_1_data[~np.isnan(s_1_data)]
                s_2 = sp[1]
                s_2_data = seg_data[s_2]
                s_2_data = s_2_data[~np.isnan(s_2_data)]
                ttest_result = ttest_ind(s_1_data,s_2_data)
                if ttest_result[1] <= 0.05:
                    ax_taste[t_i].plot([s_1+1, s_2+1],[corr_max,corr_max],color='k')
                    taste_max_corr = taste_max_corr*1.05
                    ax_taste[t_i].text(s_1+1+(s_2-s_1)/2,corr_max, '* T',\
                            horizontalalignment='center', verticalalignment='center',\
                                color='k')
                    taste_max_corr = taste_max_corr*1.05
                ks_result = ks_2samp(s_1_data,s_2_data)
                if ks_result[1] <= 0.05:
                    ax_taste[t_i].plot([s_1+1, s_2+1],[corr_max,corr_max],color='k')
                    taste_max_corr = taste_max_corr*1.05
                    ax_taste[t_i].text(s_1+1+(s_2-s_1)/2,corr_max, '* K',\
                            horizontalalignment='center', verticalalignment='center',\
                                color='r')
                    taste_max_corr = taste_max_corr*1.05 
            #Individual distribution significance > 0
            for s_i in range(len(unique_segment_names)):
                zero_percentile = percentileofscore(seg_data[s_i],0)
                if zero_percentile <= 5:
                    ax_taste[t_i].text(s_i+1,taste_max_corr,'*%',color='blue')
                    taste_max_corr = taste_max_corr*1.05 
                if len(np.where(seg_data[s_i] < 0)[0]) == 0:
                    ax_taste[t_i].text(s_i+1,taste_max_corr,'*>0',color='blue')
                    taste_max_corr = taste_max_corr*1.05 
            if taste_max_corr > max_corr:
                max_corr = taste_max_corr
            if taste_min_corr < min_corr:
                min_corr = taste_min_corr
        for t_i in range(len(unique_taste_names)):
            ax_taste[t_i].set_ylim([min_corr*1.05,max_corr*1.05])
        f_taste.suptitle(corr_type + ' Mean Correlation of Population Rate to Bin Taste Correlation')
        plt.tight_layout()
        f_taste.savefig(os.path.join(results_dir,corr_type + '_by_taste_mean_trends.png'))
        f_taste.savefig(os.path.join(results_dir,corr_type + '_by_taste_mean_trends.svg'))
        plt.close(f_taste)
            
def cross_dataset_dev_stats_plots(dev_stats_data, unique_given_names, 
                                            unique_dev_stats_names, unique_segment_names, 
                                            results_dir):   
    """This function is dedicated to plotting deviation statistics across animals.
    INPUTS:
        - dev_stats_data: dictionary containing data regarding deviation statistics
            organized as follows:
                - dev_stats_data.keys() are the unique_given_names
                - dev_stats_data[name]['dev_stats'] = dict containing deviation stats
                - dev_stats_data[name]['dev_stats'].keys() are the unique_dev_stats_names
                - dev_stats_data[name]['dev_stats'][stats_name] = dict containing specific statistic results
                - dev_stats_data[name]['dev_stats'][stats_name].keys() are the unique_segment_names
                - dev_stats_data[name]['dev_stats'][stats_name][seg_name] = array of deviation statistics
        - unique_given_names: names of imported datasets in dict
        - unique_dev_stats_names: names of types of statistics
        - unique_segment_names: names of segments analyzed
        - results_dir: storage directory
    OUTPUTS:
        - Plots with statistical results
    """
    colors = ['green','royalblue','blueviolet','teal','deeppink', \
              'springgreen','turquoise', 'midnightblue', 'lightskyblue', \
              'palevioletred', 'darkslateblue']
    for dev_stat in unique_dev_stats_names:
        max_val = 0
        max_mean = 0
        cross_animal_means = []
        combined_animal_results = []
        for seg_name in unique_segment_names:
            combined_seg_results = []
            seg_animal_means = []
            for name in unique_given_names:
                dataset = dev_stats_data[name]['dev_stats'][dev_stat][seg_name]
                combined_seg_results.extend(list(dataset))
                seg_animal_means.extend([np.nanmean(dataset)])
            if max(np.array(combined_seg_results)) > max_val:
                max_val = max(np.array(combined_seg_results))
            if max(np.array(seg_animal_means)) > max_mean:
                max_mean = max(np.array(seg_animal_means))
            cross_animal_means.append(seg_animal_means)
            combined_animal_results.append(combined_seg_results)
        max_val = np.ceil(max_val).astype('int')
        max_mean = np.ceil(max_mean).astype('int')
        #Create figure of combined data
        f, ax = plt.subplots(nrows=1, ncols=2, figsize = (4*2,4))
        #plt.boxplot(combined_animal_results,labels=unique_segment_names)
        for s_i, seg_name in enumerate(unique_segment_names):
            cdf_data = np.array(combined_animal_results[s_i])
            nonnan_data = cdf_data[~np.isnan(cdf_data)]
            ax[0].ecdf(nonnan_data,label=seg_name,color=colors[s_i])
            # ax[0].hist(combined_animal_results[s_i],bins=min([1000,max_val]),histtype='step',\
            #          density=True,cumulative=True,label=seg_name,color=colors[s_i])
            ax[1].hist(combined_animal_results[s_i],bins=max_val,histtype='step',\
                     density=True,label='seg_name',color=colors[s_i])
            data_mean = np.nanmean(combined_animal_results[s_i])
            ax[1].axvline(data_mean,color=colors[s_i],\
                          label=seg_name + 'mean = ' + str(np.round(data_mean,2)))
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='upper right')
        ax[0].set_xlabel(dev_stat)
        ax[1].set_xlabel(dev_stat)
        ax[0].set_ylabel('Cumulative Fraction')
        ax[1].set_xlabel('Density')
        ax[0].set_title('Cumulative Distribution')
        ax[1].set_title('Density Distribution')
        seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
        pair_statistics = ('Segment 1').ljust(10,' ') + ' | ' + ('Segment 2').ljust(10,' ') + ' | ' + ('TTest').ljust(5,' ')
        for sp in seg_pairs:
            seg_1 = sp[0]
            seg_1_data = np.array(combined_animal_results[seg_1])
            seg_2 = sp[1]
            seg_2_data = np.array(combined_animal_results[seg_2])
            ttest_result = ttest_ind(seg_1_data[~np.isnan(seg_1_data)],seg_2_data[~np.isnan(seg_2_data)])
            if ttest_result[1] <= 0.05:
                pair_statistics = pair_statistics + '\n' + \
                    unique_segment_names[seg_1].ljust(10, ' ') + '   ' + \
                        unique_segment_names[seg_2].ljust(10, ' ') + '   ' + ('*').ljust(5,' ')
            else:
                pair_statistics = pair_statistics + '\n' + \
                    unique_segment_names[seg_1].ljust(10, ' ') + '   ' + \
                        unique_segment_names[seg_2].ljust(10, ' ') + '   ' + ('n.s.').ljust(5,' ')
        ax[0].text(max_val,0,pair_statistics,horizontalalignment='right',\
                 verticalalignment='bottom',fontsize=10,color='k',
                 bbox=dict(boxstyle="round",color="grey", alpha=0.5))
        plt.tight_layout()
        f.savefig(os.path.join(results_dir,dev_stat + '_distributions.png'))
        f.savefig(os.path.join(results_dir,dev_stat + '_distributions.svg'))
        plt.close(f)
        
        #Create figure of animal means
        f_means = plt.figure(figsize=(5,5))
        for n_i, name in enumerate(unique_given_names):
            animal_means = []
            for s_i, seg_name in enumerate(unique_segment_names):
                animal_means.extend([cross_animal_means[s_i][n_i]])
            plt.plot(np.arange(len(unique_segment_names)),animal_means,\
                     color=colors[n_i],alpha=0.5,label=name + ' Mean')
        all_animal_means = []
        all_animal_stds = []
        for s_i, seg_name in enumerate(unique_segment_names):
            all_animal_means.extend([np.nanmean(combined_animal_results[s_i])])
            all_animal_stds.extend([np.nanstd(combined_animal_results[s_i])])
        plt.plot(np.arange(len(unique_segment_names)),all_animal_means,\
                 color='k',linestyle='dashed',label='Mean')
        plt.fill_between(np.arange(len(unique_segment_names)),\
                         np.array(all_animal_means) - np.array(all_animal_stds),\
                        np.array(all_animal_means) + np.array(all_animal_stds),\
                        color='k',alpha=0.2,label='Std')
        plt.xticks(np.arange(len(unique_segment_names)),unique_segment_names)
        plt.legend()
        plt.ylim([0,max_mean + np.nanmax(all_animal_stds)])
        plt.title(dev_stat)
        f_means.savefig(os.path.join(results_dir,dev_stat + '_means.png'))
        f_means.savefig(os.path.join(results_dir,dev_stat + '_means.svg'))
        plt.close(f_means)
        
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
    
def cross_dataset_cp_plots(cp_data, unique_given_names, unique_taste_names, 
                           max_cp_counts, results_dir):   
    """This function is dedicated to plotting deviation statistics compared
    to null distributions across animals.
    INPUTS:
        - dev_null_data: dictionary containing data regarding deviation statistics
            organized as follows:
                - cp_data.keys() are the unique_given_names
                - cp_data[name]['cp_data'] = dict containing deviation stats
                - cp_data[name]['cp_data'].keys() are the unique_dev_stats_names
                - cp_data[name]['cp_data'][taste_name] = array of size # deliveries x # cp + 2 with specific changepoint times
        - unique_taste_names: names of tastes in dataset
        - max_cp_counts: maximum number of changepoints across datasets
        - results_dir: storage directory
    OUTPUTS:
        - Plots with statistical results
    """
    colors = ['green','royalblue','blueviolet','teal','deeppink', \
              'springgreen','turquoise', 'midnightblue', 'lightskyblue', \
              'palevioletred', 'darkslateblue']
    
    f_cp_dist, ax_cp_dist = plt.subplots(nrows = max_cp_counts, \
                                         ncols = len(unique_taste_names), \
                                         figsize = (8,8))
    all_cp_list = [[] for cp_i in range(max_cp_counts)]
    for t_i, taste_name in enumerate(unique_taste_names):
        #Collect cp data across animals
        cp_list = [[] for cp_i in range(max_cp_counts)]
        for name in unique_given_names:
            cp_array = cp_data[name]['cp_data'][taste_name]
            taste_cp_diff = np.diff(cp_array)
            taste_cp_realigned = np.cumsum(taste_cp_diff,1)
            for cp_i in range(np.shape(cp_array)[1]-2):
                cp_list[cp_i].extend(list(taste_cp_realigned[:,cp_i]))
                all_cp_list[cp_i].extend(list(taste_cp_realigned[:,cp_i]))
        #Plot histograms on appropriate axes
        for cp_i in range(max_cp_counts):
            cp_dist = cp_list[cp_i]
            dist_mean = np.nanmean(cp_dist)
            dist_median = np.median(cp_dist)
            ax_cp_dist[cp_i,t_i].hist(cp_dist,density=True,alpha=0.3,\
                                      color=colors[cp_i],label='_')
            ax_cp_dist[cp_i,t_i].axvline(dist_mean,label='Mean CP ' + str(cp_i+1) + ' = ' + str(np.round(dist_mean,2)), color=colors[cp_i])
            ax_cp_dist[cp_i,t_i].axvline(dist_median,label='Median CP ' + str(cp_i+1) + ' = ' + str(np.round(dist_median,2)), linestyle='dashed', color=colors[cp_i])
            ax_cp_dist[cp_i,t_i].set_title(taste_name + ' cp ' + str(cp_i))
            ax_cp_dist[cp_i,t_i].set_xlabel('Time After Taste (ms)')
            ax_cp_dist[cp_i,t_i].set_ylabel('Density')
            ax_cp_dist[cp_i,t_i].legend(loc='upper left',fontsize=8)
    plt.tight_layout()
    f_cp_dist.savefig(os.path.join(results_dir,'cp_dists.png'))
    f_cp_dist.savefig(os.path.join(results_dir,'cp_dists.svg'))
    plt.close(f_cp_dist)
        
    f_all_cp, ax_all_cp = plt.subplots(ncols = max_cp_counts, figsize=(4*max_cp_counts,4))
    for cp_i in range(max_cp_counts):
        cp_dist = all_cp_list[cp_i]
        dist_mean = np.nanmean(cp_dist)
        dist_median = np.median(cp_dist)
        ax_all_cp[cp_i].hist(cp_dist,density=True,alpha=0.3,\
                                  color=colors[cp_i],label='_')
        ax_all_cp[cp_i].axvline(dist_mean,label='Mean CP ' + str(cp_i+1) + ' = ' + str(np.round(dist_mean,2)), color=colors[cp_i])
        ax_all_cp[cp_i].axvline(dist_median,label='Median CP ' + str(cp_i+1) + ' = ' + str(np.round(dist_median,2)), linestyle='dashed', color=colors[cp_i])
        ax_all_cp[cp_i].set_title('All cp ' + str(cp_i))
        ax_all_cp[cp_i].set_xlabel('Time After Taste (ms)')
        ax_all_cp[cp_i].set_ylabel('Density')
        ax_all_cp[cp_i].legend(loc='upper left',fontsize=8)
    plt.tight_layout()
    f_all_cp.savefig(os.path.join(results_dir,'all_cp_combined.png'))
    f_all_cp.savefig(os.path.join(results_dir,'all_cp_combined.svg'))
    plt.close(f_all_cp)
        
def cross_dataset_dev_split_corr_plots(dev_split_corr_data, unique_given_names, 
                                 unique_epoch_pairs, unique_segment_names, 
                                 unique_taste_names, results_dir):
    """
    This function plots the results of split deviation event correlation data
    across multiple animals.
    """
    
    #Variables
    num_epoch_pairs = len(unique_epoch_pairs)
    num_segments = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    
    taste_pair_inds = list(combinations(np.arange(len(unique_taste_names)),2))
    epoch_pair_inds = list(combinations(np.arange(len(unique_epoch_pairs)), 2))
    segment_pair_inds = list(combinations(np.arange(len(unique_segment_names)), 2))
    
    #Plot epoch pair x segment the taste correlations against each other
    f_e_seg, ax_e_seg = plt.subplots(nrows = num_epoch_pairs, ncols = num_segments,
                                     sharex = True, sharey = True, figsize=(8,num_epoch_pairs*2))
    for ep_i, ep_str in enumerate(unique_epoch_pairs):
        for s_i, seg_name in enumerate(unique_segment_names):
            #Collect cmf data
            t_data = []
            for t_i, t_name in enumerate(unique_taste_names):
                t_data_combined = []
                for g_i, g_name in enumerate(unique_given_names):
                    try: #If data exists for this animal
                        corr_array = dev_split_corr_data[g_name]['corr_data'][seg_name][t_name]
                        ep_list = dev_split_corr_data[g_name]['corr_data'][seg_name]['epoch_pairs']
                        ep_ind = [i for i in range(len(ep_list)) if str(ep_list[i]) == ep_str]
                        t_data_combined.extend(list(corr_array[ep_ind[0],:]))
                    except:
                        t_data_combined.extend([])
                t_data.append(t_data_combined)
            #Plot cmf
            for td_i, td in enumerate(t_data):
                ax_e_seg[ep_i,s_i].ecdf(td[~np.isnan(td)],label=unique_taste_names[td_i])
                # ax_e_seg[ep_i,s_i].hist(td,density=True,cumulative=True,bins=1000,
                #                         histtype='step',label=unique_taste_names[td_i])
            if (ep_i == 0)*(s_i == 0):
                ax_e_seg[ep_i,s_i].legend(loc='upper left')
            if ep_i == 0:
                ax_e_seg[ep_i,s_i].set_title(seg_name)
            if ep_i == num_epoch_pairs-1:
                ax_e_seg[ep_i,s_i].set_xlabel('Pearson Correlation')
            if s_i == 0:
                ax_e_seg[ep_i,s_i].set_ylabel(ep_str)
            #Calculate pairwise significances
            t_sig_text = "Sig:\n"
            for tp_i, tp in enumerate(taste_pair_inds):
                t_1 = tp[0]
                t_2 = tp[1]
                try:
                    k_res = ks_2samp(t_data[t_1],t_data[t_2])
                    if k_res.pvalue < 0.05:
                        t_sig_text = t_sig_text + unique_taste_names[t_1] + \
                            "x" + unique_taste_names[t_2] + "\n"
                except: #Not enough data for KS test
                    t_sig_text = t_sig_text
            ax_e_seg[ep_i,s_i].text(-0.5,0.05,t_sig_text)
            
    plt.suptitle('CMF Split Dev Epoch Pair Corr')
    plt.tight_layout()
    f_e_seg.savefig(os.path.join(results_dir,'epoch_x_seg_taste_cmfs.png'))
    f_e_seg.savefig(os.path.join(results_dir,'epoch_x_seg_taste_cmfs.svg'))
    plt.close(f_e_seg)
    
    
    #Plot epoch pair x taste the segment correlations against each other
    f_e_taste, ax_e_taste = plt.subplots(nrows = num_epoch_pairs, ncols = num_tastes,
                                     sharex = True, sharey = True, figsize=(8,num_epoch_pairs*2))
    for ep_i, ep_str in enumerate(unique_epoch_pairs):
        for t_i, t_name in enumerate(unique_taste_names):
            #Collect cmf data
            s_data = []
            for s_i, seg_name in enumerate(unique_segment_names):
                s_data_combined = []
                for g_i, g_name in enumerate(unique_given_names):
                    try: #If data exists for this animal
                        corr_array = dev_split_corr_data[g_name]['corr_data'][seg_name][t_name]
                        ep_list = dev_split_corr_data[g_name]['corr_data'][seg_name]['epoch_pairs']
                        ep_ind = [i for i in range(len(ep_list)) if str(ep_list[i]) == ep_str]
                        s_data_combined.extend(list(corr_array[ep_ind[0],:]))
                    except:
                        s_data_combined.extend([])
                s_data.append(s_data_combined)
            #Plot cmf
            for sd_i, sd in enumerate(s_data):
                ax_e_taste[ep_i,t_i].ecdf(sd[~np.isnan(sd)],label=unique_segment_names[sd_i])
                # ax_e_taste[ep_i,t_i].hist(sd,density=True,cumulative=True,bins=1000,
                #                         histtype='step',label=unique_segment_names[sd_i])
            if (ep_i == 0)*(t_i == 0):
                ax_e_taste[ep_i,t_i].legend(loc='upper left')
            if ep_i == 0:
                ax_e_taste[ep_i,t_i].set_title(t_name)
            if ep_i == num_epoch_pairs-1:
                ax_e_taste[ep_i,t_i].set_xlabel('Pearson Correlation')
            if t_i == 0:
                ax_e_taste[ep_i,t_i].set_ylabel(ep_str)
            #Calculate pairwise significances
            s_sig_text = "Sig:\n"
            for sp_i, sp in enumerate(segment_pair_inds):
                s_1 = sp[0]
                s_2 = sp[1]
                try:
                    k_res = ks_2samp(s_data[s_1],s_data[s_2])
                    if k_res.pvalue < 0.05:
                        s_sig_text = s_sig_text + unique_segment_names[s_1] + \
                            "x" + unique_segment_names[s_2] + "\n"
                except: #Not enough data for KS test
                    s_sig_text = s_sig_text
            ax_e_taste[ep_i,t_i].text(-0.5,0.05,s_sig_text)
            
    plt.suptitle('CMF Split Dev Epoch Pair Corr')
    plt.tight_layout()
    f_e_taste.savefig(os.path.join(results_dir,'epoch_x_taste_seg_cmfs.png'))
    f_e_taste.savefig(os.path.join(results_dir,'epoch_x_taste_seg_cmfs.svg'))
    plt.close(f_e_taste)
    
    #Plot segment x taste grid of epoch pairs against each other
    f_seg_taste, ax_seg_taste = plt.subplots(nrows = num_segments, ncols = num_tastes,
                                     sharex = True, sharey = True, figsize=(8,8))
    for s_i, seg_name in enumerate(unique_segment_names):
        for t_i, t_name in enumerate(unique_taste_names):
            #Collect cmf data
            e_data = []
            for ep_i, ep in enumerate(unique_epoch_pairs):
                e_data_combined = []
                for g_i, g_name in enumerate(unique_given_names):
                    try: #If data exists for this animal
                        corr_array = dev_split_corr_data[g_name]['corr_data'][seg_name][t_name]
                        ep_list = dev_split_corr_data[g_name]['corr_data'][seg_name]['epoch_pairs']
                        ep_ind = [i for i in range(len(ep_list)) if str(ep_list[i]) == ep_str]
                        e_data_combined.extend(list(corr_array[ep_ind[0],:]))
                    except:
                        e_data_combined.extend([])
                e_data.append(e_data_combined)
            #Plot cmf
            for ed_i, ed in enumerate(e_data):
                ax_seg_taste[s_i,t_i].ecdf(ed[~np.isnan(ed)],label=unique_epoch_pairs[ed_i])
                # ax_seg_taste[s_i,t_i].hist(ed,density=True,cumulative=True,bins=1000,
                #                         histtype='step',label=unique_epoch_pairs[ed_i])
            if (s_i == 0)*(t_i == 0):
                ax_seg_taste[s_i,t_i].legend(loc='upper left')
            if s_i == 0:
                ax_seg_taste[s_i,t_i].set_title(t_name)
            if s_i == num_epoch_pairs-1:
                ax_seg_taste[s_i,t_i].set_xlabel('Pearson Correlation')
            if t_i == 0:
                ax_seg_taste[s_i,t_i].set_ylabel(seg_name)
            #Calculate pairwise significances
            e_sig_text = "Sig:\n"
            for ep_i, ep in enumerate(epoch_pair_inds):
                e_1 = ep[0]
                e_2 = ep[1]
                try:
                    k_res = ks_2samp(e_data[e_1],e_data[e_2])
                    if k_res.pvalue < 0.05:
                        e_sig_text = e_sig_text + unique_epoch_pairs[e_1] + \
                            "x" + unique_epoch_pairs[e_2] + "\n"
                except: #Not enough data for KS test
                    e_sig_text = e_sig_text
            ax_seg_taste[s_i,t_i].text(-0.5,0.05,e_sig_text)
            
    plt.suptitle('CMF Split Dev Epoch Pair Corr')
    plt.tight_layout()
    f_seg_taste.savefig(os.path.join(results_dir,'seg_x_taste_epoch_cmfs.png'))
    f_seg_taste.savefig(os.path.join(results_dir,'seg_x_taste_epoch_cmfs.svg'))
    plt.close(f_seg_taste)
    
def cross_dataset_dev_split_best_corr_plots(dev_split_corr_data, unique_given_names, 
                                 unique_epoch_pairs, unique_segment_names, 
                                 unique_taste_names, results_dir):
    """
    This function plots the results of split deviation event correlation data
    across multiple animals.
    """
    
    #Variables
    num_epoch_pairs = len(unique_epoch_pairs)
    num_segments = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    
    taste_pair_inds = list(combinations(np.arange(len(unique_taste_names)),2))
    epoch_pair_inds = list(combinations(np.arange(len(unique_epoch_pairs)), 2))
    segment_pair_inds = list(combinations(np.arange(len(unique_segment_names)), 2))
    
    #Plot epoch pair x segment the taste correlations against each other
    f_e_seg, ax_e_seg = plt.subplots(nrows = num_epoch_pairs, ncols = num_segments,
                                     sharex = True, sharey = True, figsize=(8,num_epoch_pairs*2))
    for ep_i, ep_str in enumerate(unique_epoch_pairs):
        for s_i, seg_name in enumerate(unique_segment_names):
            #Collect cmf data
            t_data = []
            for t_i, t_name in enumerate(unique_taste_names):
                t_data_combined = []
                for g_i, g_name in enumerate(unique_given_names):
                    try: #If data exists for this animal
                        corr_array = dev_split_corr_data[g_name]['corr_data'][seg_name][t_name]
                        corr_argmax = np.argmax(corr_array,0)
                        ep_list = dev_split_corr_data[g_name]['corr_data'][seg_name]['epoch_pairs']
                        ep_ind = [i for i in range(len(ep_list)) if str(ep_list[i]) == ep_str]
                        best_ep_ind = np.where(corr_argmax == ep_ind)[0]
                        t_data_combined.extend(list(corr_array[ep_ind[0],best_ep_ind]))
                    except:
                        t_data_combined.extend([])
                t_data.append(t_data_combined)
            #Plot cmf
            for td_i, td in enumerate(t_data):
                ax_e_seg[ep_i,s_i].ecdf(td[~np.isnan(td)],label=unique_taste_names[td_i])
                # ax_e_seg[ep_i,s_i].hist(td,density=True,cumulative=True,bins=1000,
                #                         histtype='step',label=unique_taste_names[td_i])
            if (ep_i == 0)*(s_i == 0):
                ax_e_seg[ep_i,s_i].legend(loc='upper left')
            if ep_i == 0:
                ax_e_seg[ep_i,s_i].set_title(seg_name)
            if ep_i == num_epoch_pairs-1:
                ax_e_seg[ep_i,s_i].set_xlabel('Pearson Correlation')
            if s_i == 0:
                ax_e_seg[ep_i,s_i].set_ylabel(ep_str)
            #Calculate pairwise significances
            t_sig_text = "Sig:\n"
            for tp_i, tp in enumerate(taste_pair_inds):
                t_1 = tp[0]
                t_2 = tp[1]
                try:
                    k_res = ks_2samp(t_data[t_1],t_data[t_2])
                    if k_res.pvalue < 0.05:
                        t_sig_text = t_sig_text + unique_taste_names[t_1] + \
                            "x" + unique_taste_names[t_2] + "\n"
                except: #Not enough data for KS test
                    t_sig_text = t_sig_text
            ax_e_seg[ep_i,s_i].text(-0.5,0.05,t_sig_text)
            
    plt.suptitle('CMF Split Dev Epoch Pair Corr')
    plt.tight_layout()
    f_e_seg.savefig(os.path.join(results_dir,'epoch_x_seg_taste_cmfs_best.png'))
    f_e_seg.savefig(os.path.join(results_dir,'epoch_x_seg_taste_cmfs_best.svg'))
    plt.close(f_e_seg)
    
    
    #Plot epoch pair x taste the segment correlations against each other
    f_e_taste, ax_e_taste = plt.subplots(nrows = num_epoch_pairs, ncols = num_tastes,
                                     sharex = True, sharey = True, figsize=(8,num_epoch_pairs*2))
    for ep_i, ep_str in enumerate(unique_epoch_pairs):
        for t_i, t_name in enumerate(unique_taste_names):
            #Collect cmf data
            s_data = []
            for s_i, seg_name in enumerate(unique_segment_names):
                s_data_combined = []
                for g_i, g_name in enumerate(unique_given_names):
                    try: #If data exists for this animal
                        corr_array = dev_split_corr_data[g_name]['corr_data'][seg_name][t_name]
                        corr_argmax = np.argmax(corr_array,0)
                        ep_list = dev_split_corr_data[g_name]['corr_data'][seg_name]['epoch_pairs']
                        ep_ind = [i for i in range(len(ep_list)) if str(ep_list[i]) == ep_str]
                        best_ep_ind = np.where(corr_argmax == ep_ind)[0]
                        s_data_combined.extend(list(corr_array[ep_ind[0],best_ep_ind]))
                    except:
                        s_data_combined.extend([])
                s_data.append(s_data_combined)
            #Plot cmf
            for sd_i, sd in enumerate(s_data):
                ax_e_taste[ep_i,t_i].ecdf(sd[~np.isnan(sd)],label=unique_segment_names[sd_i])
                # ax_e_taste[ep_i,t_i].hist(sd,density=True,cumulative=True,bins=1000,
                #                         histtype='step',label=unique_segment_names[sd_i])
            if (ep_i == 0)*(t_i == 0):
                ax_e_taste[ep_i,t_i].legend(loc='upper left')
            if ep_i == 0:
                ax_e_taste[ep_i,t_i].set_title(t_name)
            if ep_i == num_epoch_pairs-1:
                ax_e_taste[ep_i,t_i].set_xlabel('Pearson Correlation')
            if t_i == 0:
                ax_e_taste[ep_i,t_i].set_ylabel(ep_str)
            #Calculate pairwise significances
            s_sig_text = "Sig:\n"
            for sp_i, sp in enumerate(segment_pair_inds):
                s_1 = sp[0]
                s_2 = sp[1]
                try:
                    k_res = ks_2samp(s_data[s_1],s_data[s_2])
                    if k_res.pvalue < 0.05:
                        s_sig_text = s_sig_text + unique_segment_names[s_1] + \
                            "x" + unique_segment_names[s_2] + "\n"
                except: #Not enough data for KS test
                    s_sig_text = s_sig_text
            ax_e_taste[ep_i,t_i].text(-0.5,0.05,s_sig_text)
            
    plt.suptitle('CMF Split Dev Epoch Pair Corr')
    plt.tight_layout()
    f_e_taste.savefig(os.path.join(results_dir,'epoch_x_taste_seg_cmfs_best.png'))
    f_e_taste.savefig(os.path.join(results_dir,'epoch_x_taste_seg_cmfs_best.svg'))
    plt.close(f_e_taste)
    
    #Plot segment x taste grid of epoch pairs against each other
    f_seg_taste, ax_seg_taste = plt.subplots(nrows = num_segments, ncols = num_tastes,
                                     sharex = True, sharey = True, figsize=(8,8))
    for s_i, seg_name in enumerate(unique_segment_names):
        for t_i, t_name in enumerate(unique_taste_names):
            #Collect cmf data
            e_data = []
            for ep_i, ep in enumerate(unique_epoch_pairs):
                e_data_combined = []
                for g_i, g_name in enumerate(unique_given_names):
                    try: #If data exists for this animal
                        corr_array = dev_split_corr_data[g_name]['corr_data'][seg_name][t_name]
                        corr_argmax = np.argmax(corr_array,0)
                        ep_list = dev_split_corr_data[g_name]['corr_data'][seg_name]['epoch_pairs']
                        ep_ind = [i for i in range(len(ep_list)) if str(ep_list[i]) == ep_str]
                        best_ep_ind = np.where(corr_argmax == ep_ind)[0]
                        e_data_combined.extend(list(corr_array[ep_ind[0],best_ep_ind]))
                    except:
                        e_data_combined.extend([])
                e_data.append(e_data_combined)
            #Plot cmf
            for ed_i, ed in enumerate(e_data):
                ax_seg_taste[s_i,t_i].ecdf(ed[~np.isnan(ed)],label=unique_epoch_pairs[ed_i])
                # ax_seg_taste[s_i,t_i].hist(ed,density=True,cumulative=True,bins=1000,
                #                         histtype='step',label=unique_epoch_pairs[ed_i])
            if (s_i == 0)*(t_i == 0):
                ax_seg_taste[s_i,t_i].legend(loc='upper left')
            if s_i == 0:
                ax_seg_taste[s_i,t_i].set_title(t_name)
            if s_i == num_epoch_pairs-1:
                ax_seg_taste[s_i,t_i].set_xlabel('Pearson Correlation')
            if t_i == 0:
                ax_seg_taste[s_i,t_i].set_ylabel(seg_name)
            #Calculate pairwise significances
            e_sig_text = "Sig:\n"
            for ep_i, ep in enumerate(epoch_pair_inds):
                e_1 = ep[0]
                e_2 = ep[1]
                try:
                    k_res = ks_2samp(e_data[e_1],e_data[e_2])
                    if k_res.pvalue < 0.05:
                        e_sig_text = e_sig_text + unique_epoch_pairs[e_1] + \
                            "x" + unique_epoch_pairs[e_2] + "\n"
                except: #Not enough data for KS test
                    e_sig_text = e_sig_text
            ax_seg_taste[s_i,t_i].text(-0.5,0.05,e_sig_text)
            
    plt.suptitle('CMF Split Dev Epoch Pair Corr')
    plt.tight_layout()
    f_seg_taste.savefig(os.path.join(results_dir,'seg_x_taste_epoch_cmfs_best.png'))
    f_seg_taste.savefig(os.path.join(results_dir,'seg_x_taste_epoch_cmfs_best.svg'))
    plt.close(f_seg_taste)
    
def cross_dataset_sliding_decode_frac_plots(sliding_decode_data, unique_given_names,
                                            unique_segment_names, unique_epochs,
                                            unique_taste_names, unique_decode_types,
                                            results_dir):
    """Plot the sliding decode fractions by decode type"""
    #Variables
    num_segments = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    num_epochs = len(unique_epochs)
    cmap = colormaps['brg']
    segment_colors = cmap(np.linspace(0, 1, num_segments))
    
    f_frac, ax_frac = plt.subplots(nrows = 3, ncols = 1, sharey = True,
                                   figsize = (4,12))
    ax_frac[0].set_ylim([-0.1,1.1])
    #Is Taste Data + Plots
    dt = 'is_taste'
    labels = ['Taste', 'No Taste']
    is_taste_seg_data = [] #num seg x num anim x 2
    for s_i, s_name in enumerate(unique_segment_names):
        all_animal_data = []
        for name_i, name in enumerate(unique_given_names):
            an_seg_names = np.array(sliding_decode_data[name]['segment_names'])
            an_seg_inds = np.array(sliding_decode_data[name]['segments_to_analyze'])
            an_seg_names = an_seg_names[an_seg_inds]
            an_s_i = np.where(an_seg_names == s_name)[0]
            if len(an_s_i) > 0:
                decode_data_i = np.squeeze(sliding_decode_data[name]['frac_decode_data'][dt][an_s_i[0],:])
                all_animal_data.append(list(decode_data_i))
        is_taste_seg_data.append(np.array(all_animal_data))
    is_taste_seg_means = np.nanmean(is_taste_seg_data,1)
    is_taste_seg_stds = np.nanstd(is_taste_seg_data,1)
    for s_i, s_name in enumerate(unique_segment_names):
        ax_frac[0].plot(np.arange(2),is_taste_seg_means[s_i,:],
                        color=segment_colors[s_i,:],label=s_name)
        ax_frac[0].fill_between(np.arange(2),
                                is_taste_seg_means[s_i,:]+is_taste_seg_stds[s_i,:],
                                is_taste_seg_means[s_i,:]-is_taste_seg_stds[s_i,:],
                                color=segment_colors[s_i,:],alpha=0.2,label='_')
    ax_frac[0].legend(loc='upper left')
    ax_frac[0].set_title('Is Taste Fraction')
    ax_frac[0].set_xticks(np.arange(2),labels)
    
    dt = 'which_taste'
    which_taste_data = [] #num taste x num seg x num anim
    for t_i, t_name in enumerate(unique_taste_names):
        all_seg_data = []
        for s_i, s_name in enumerate(unique_segment_names):
            all_animal_data = []
            for name_i, name in enumerate(unique_given_names):
                an_seg_names = np.array(sliding_decode_data[name]['segment_names'])
                an_seg_inds = np.array(sliding_decode_data[name]['segments_to_analyze'])
                an_seg_names = an_seg_names[an_seg_inds]
                an_s_i = np.where(an_seg_names == s_name)[0]
                taste_names = sliding_decode_data[name]['dig_in_names']
                an_t_i = np.where(np.array(taste_names) == t_name)[0]
                if (len(an_t_i) > 0) and (len(an_s_i) > 0):
                    decode_data_i = sliding_decode_data[name]['frac_decode_data'][dt]
                    all_animal_data.append(decode_data_i[an_s_i[0],an_t_i[0]])
            all_seg_data.append(all_animal_data)
        which_taste_data.append(np.array(all_seg_data))
    which_taste_seg_means = np.nanmean(which_taste_data,-1) #num taste x num seg
    which_taste_seg_stds = np.nanstd(which_taste_data,-1) #num taste x num seg
    for s_i, s_name in enumerate(unique_segment_names):
        ax_frac[1].plot(np.arange(num_tastes),which_taste_seg_means[:,s_i],
                        color=segment_colors[s_i,:],label=s_name)
        ax_frac[1].fill_between(np.arange(num_tastes),
                                which_taste_seg_means[:,s_i]+which_taste_seg_stds[:,s_i],
                                which_taste_seg_means[:,s_i]-which_taste_seg_stds[:,s_i],
                                color=segment_colors[s_i,:],label='_',alpha=0.2)
    ax_frac[1].legend(loc='upper left')
    ax_frac[1].set_title('Which Taste Fraction')
    ax_frac[1].set_xticks(np.arange(num_tastes),unique_taste_names)
    
    dt = 'which_epoch'
    unique_epoch_names = ['Epoch ' + str(e_i) for e_i in unique_epochs]
    which_epoch_data = [] #num epoch x num seg x num animals
    for e_i in unique_epochs:
        all_seg_data = []
        for s_i, s_name in enumerate(unique_segment_names):
            all_animal_data = []
            for name_i, name in enumerate(unique_given_names):
                an_seg_names = np.array(sliding_decode_data[name]['segment_names'])
                an_seg_inds = np.array(sliding_decode_data[name]['segments_to_analyze'])
                an_seg_names = an_seg_names[an_seg_inds]
                an_s_i = np.where(an_seg_names == s_name)[0]
                an_epochs = sliding_decode_data[name]['epochs_to_analyze']
                an_e_i = np.where(np.array(an_epochs) == e_i)[0]
                if (len(an_e_i) > 0) and (len(an_s_i) > 0):
                    decode_data_i = sliding_decode_data[name]['frac_decode_data'][dt]
                    all_animal_data.append(decode_data_i[an_s_i[0],an_e_i[0]])
            all_seg_data.append(all_animal_data)
        which_epoch_data.append(np.array(all_seg_data))
    which_epoch_seg_means = np.nanmean(which_epoch_data,-1) #num epoch x num seg 
    which_epoch_seg_stds = np.nanstd(which_epoch_data,-1) #num epoch x num seg 
    for s_i, s_name in enumerate(unique_segment_names):
        ax_frac[2].plot(np.arange(num_epochs),which_epoch_seg_means[:,s_i],
                        color=segment_colors[s_i,:],label=s_name)
        ax_frac[2].fill_between(np.arange(num_epochs),
                                which_epoch_seg_means[:,s_i]+which_epoch_seg_stds[:,s_i],
                                which_epoch_seg_means[:,s_i]-which_epoch_seg_stds[:,s_i],
                                color=segment_colors[s_i,:],alpha=0.2,label='_')
    ax_frac[2].legend(loc='upper left')
    ax_frac[2].set_title('Which Epoch Fraction')
    ax_frac[2].set_xticks(np.arange(num_epochs),unique_epoch_names)
    
    f_frac.savefig(os.path.join(results_dir,'decode_fracs.png'))
    f_frac.savefig(os.path.join(results_dir,'decode_fracs.svg'))
    plt.close(f_frac)
    
def cross_dataset_sliding_decode_corr_plots(sliding_decode_data, unique_given_names,
                                            unique_segment_names, unique_epochs,
                                            unique_taste_names, unique_decode_types, 
                                            results_dir):
    """Plot the sliding decode correlation to population rate by decode type"""
    #Variables
    num_segments = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    num_epochs = len(unique_epochs)
    cmap = colormaps['brg']
    segment_colors = cmap(np.linspace(0, 1, num_segments))
    
    f_corr, ax_corr = plt.subplots(nrows = 3, ncols = 1, sharey = True,
                                   figsize = (4,12))
    ax_corr[0].set_ylim([-1.1,1.1])
    #Is Taste Data + Plots
    dt = 'is_taste'
    labels = ['Taste', 'No Taste']
    is_taste_seg_data = [] #num seg x num anim x 2
    for s_i, s_name in enumerate(unique_segment_names):
        all_animal_data = []
        for name_i, name in enumerate(unique_given_names):
            an_seg_names = np.array(sliding_decode_data[name]['segment_names'])
            an_seg_inds = np.array(sliding_decode_data[name]['segments_to_analyze'])
            an_seg_names = an_seg_names[an_seg_inds]
            an_s_i = np.where(an_seg_names == s_name)[0]
            if len(an_s_i) > 0:
                decode_data_i = np.squeeze(sliding_decode_data[name]['pop_corr_data'][dt][an_s_i[0],:])
                all_animal_data.append(list(decode_data_i))
        is_taste_seg_data.append(np.array(all_animal_data))
    is_taste_seg_means = np.nanmean(is_taste_seg_data,1) #num seg x 2
    is_taste_seg_stds = np.nanstd(is_taste_seg_data,1)  #num seg x 2
    ax_corr[0].axhline(0,linestyle='dashed',color='k',alpha=0.5)
    ax_corr[0].plot(np.arange(num_segments),is_taste_seg_means[:,0],
                        color='b',label='Mean')
    ax_corr[0].fill_between(np.arange(num_segments),
                            is_taste_seg_means[:,0]+is_taste_seg_stds[:,0],
                            is_taste_seg_means[:,0]-is_taste_seg_stds[:,0],
                            color='b',alpha=0.2,label='Std')
    ax_corr[0].legend(loc='upper left')
    ax_corr[0].set_title('Is Taste x Pop Rate')
    ax_corr[0].set_xticks(np.arange(num_segments),unique_segment_names)
    
    dt = 'which_taste'
    which_taste_data = [] #num taste x num seg x num anim
    for t_i, t_name in enumerate(unique_taste_names):
        all_seg_data = []
        for s_i, s_name in enumerate(unique_segment_names):
            all_animal_data = []
            for name_i, name in enumerate(unique_given_names):
                an_seg_names = np.array(sliding_decode_data[name]['segment_names'])
                an_seg_inds = np.array(sliding_decode_data[name]['segments_to_analyze'])
                an_seg_names = an_seg_names[an_seg_inds]
                an_s_i = np.where(an_seg_names == s_name)[0]
                taste_names = sliding_decode_data[name]['dig_in_names']
                an_t_i = np.where(np.array(taste_names) == t_name)[0]
                if (len(an_t_i) > 0) and (len(an_s_i) > 0):
                    decode_data_i = sliding_decode_data[name]['pop_corr_data'][dt]
                    all_animal_data.append(decode_data_i[an_s_i[0],an_t_i[0]])
            all_seg_data.append(all_animal_data)
        which_taste_data.append(np.array(all_seg_data))
    which_taste_seg_means = np.nanmean(which_taste_data,-1) #num taste x num seg
    which_taste_seg_stds = np.nanstd(which_taste_data,-1) #num taste x num seg
    ax_corr[1].axhline(0,linestyle='dashed',color='k',alpha=0.5)
    for s_i, s_name in enumerate(unique_segment_names):
        ax_corr[1].plot(np.arange(num_tastes),which_taste_seg_means[:,s_i],
                        color=segment_colors[s_i,:],label=s_name)
        ax_corr[1].fill_between(np.arange(num_tastes),
                                which_taste_seg_means[:,s_i]+which_taste_seg_stds[:,s_i],
                                which_taste_seg_means[:,s_i]-which_taste_seg_stds[:,s_i],
                                color=segment_colors[s_i,:],label='_',alpha=0.2)
    ax_corr[1].legend(loc='upper left')
    ax_corr[1].set_title('Which Taste x Pop Rate')
    ax_corr[1].set_xticks(np.arange(num_tastes),unique_taste_names)
    
    dt = 'which_epoch'
    unique_epoch_names = ['Epoch ' + str(e_i) for e_i in unique_epochs]
    which_epoch_data = [] #num epoch x num seg x num animals
    for e_i in unique_epochs:
        all_seg_data = []
        for s_i, s_name in enumerate(unique_segment_names):
            all_animal_data = []
            for name_i, name in enumerate(unique_given_names):
                an_seg_names = np.array(sliding_decode_data[name]['segment_names'])
                an_seg_inds = np.array(sliding_decode_data[name]['segments_to_analyze'])
                an_seg_names = an_seg_names[an_seg_inds]
                an_s_i = np.where(an_seg_names == s_name)[0]
                an_epochs = sliding_decode_data[name]['epochs_to_analyze']
                an_e_i = np.where(np.array(an_epochs) == e_i)[0]
                if (len(an_e_i) > 0) and (len(an_s_i) > 0):
                    decode_data_i = sliding_decode_data[name]['pop_corr_data'][dt]
                    all_animal_data.append(decode_data_i[an_s_i[0],an_e_i[0]])
            all_seg_data.append(all_animal_data)
        which_epoch_data.append(np.array(all_seg_data))
    which_epoch_seg_means = np.nanmean(which_epoch_data,-1) #num epoch x num seg 
    which_epoch_seg_stds = np.nanstd(which_epoch_data,-1) #num epoch x num seg 
    ax_corr[2].axhline(0,linestyle='dashed',color='k',alpha=0.5)
    for s_i, s_name in enumerate(unique_segment_names):
        ax_corr[2].plot(np.arange(num_epochs),which_epoch_seg_means[:,s_i],
                        color=segment_colors[s_i,:],label=s_name)
        ax_corr[2].fill_between(np.arange(num_epochs),
                                which_epoch_seg_means[:,s_i]+which_epoch_seg_stds[:,s_i],
                                which_epoch_seg_means[:,s_i]-which_epoch_seg_stds[:,s_i],
                                color=segment_colors[s_i,:],alpha=0.2,label='_')
    ax_corr[2].legend(loc='upper left')
    ax_corr[2].set_title('Which Epoch x Pop Rate')
    ax_corr[2].set_xticks(np.arange(num_epochs),unique_epoch_names)
    
    f_corr.savefig(os.path.join(results_dir,'decode_pop_corr.png'))
    f_corr.savefig(os.path.join(results_dir,'decode_pop_corr.svg'))
    plt.close(f_corr)
    
def cross_dataset_dev_decode_frac_plots(dev_decode_data, unique_given_names,
                                            unique_segment_names, unique_epochs,
                                            unique_taste_names, unique_decode_types,
                                            results_dir):
    """Plot the dev decode fractions by decode type"""
    #Variables
    num_anim = len(unique_given_names)
    num_segments = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    num_epochs = len(unique_epochs)
    cmap = colormaps['brg']
    segment_colors = cmap(np.linspace(0, 1, num_segments))
    cmap = colormaps['gist_rainbow']
    anim_colors = cmap(np.linspace(0, 1, num_anim))
    cmap = colormaps['turbo']
    epoch_colors = cmap(np.linspace(0, 1, num_epochs))
    seg_pairs = list(combinations(np.arange(num_segments), 2))
    epoch_pairs = list(combinations(np.arange(num_epochs), 2))
    
    f_frac, ax_frac = plt.subplots(nrows = 2 + num_tastes, ncols = 1, sharey = True,
                                   figsize = (4,4*(2+num_tastes)))
    f_box, ax_box = plt.subplots(nrows = 2, ncols = 1, sharey = True,
                                   figsize = (4,8))
    ax_frac[0].set_ylim([-0.1,1.1])
    ax_box[0].set_ylim([-0.1,1.1])
    #Is Taste Data + Plots
    dt = 'is_taste'
    labels = ['Taste', 'No Taste']
    is_taste_seg_data = [] #num seg x num anim x 2
    for s_i, s_name in enumerate(unique_segment_names):
        all_animal_data = []
        for name_i, name in enumerate(unique_given_names):
            try:
                decode_data_i = np.squeeze(dev_decode_data[name][dt][s_name])
                num_dev, _ = np.shape(decode_data_i)
                decode_argmax = np.squeeze(np.argmax(decode_data_i,1))
                no_taste_count = np.sum(decode_argmax)
                all_animal_data.append([(num_dev - no_taste_count)/num_dev,no_taste_count/num_dev])
            except:
                decode_data_i = []
        is_taste_seg_data.append(np.array(all_animal_data))
    #Pairwise TTests
    pair_sig = np.zeros(len(seg_pairs))
    for sp_i, sp in enumerate(seg_pairs):
        s_1 = sp[0]
        s_2 = sp[1]
        s_data1 = is_taste_seg_data[s_1][:,0]
        s_data2 = is_taste_seg_data[s_2][:,0]
        ks_res = ks_2samp(s_data1, s_data2)
        if ks_res.pvalue < 0.05:
            pair_sig[sp_i] = 1
    #Averages
    is_taste_seg_means = np.nanmean(is_taste_seg_data,1)
    is_taste_seg_stds = np.nanstd(is_taste_seg_data,1)
    for s_i, s_name in enumerate(unique_segment_names):
        box_data = is_taste_seg_data[s_i][:,0]
        ax_box[0].boxplot(box_data,positions=s_i*np.ones(1))
        box_x = (np.random.rand(len(box_data))-0.5)/10 + s_i
        for n_i in range(num_anim):
            ax_box[0].scatter(box_x[n_i],box_data[n_i],alpha=0.2,c=anim_colors[n_i,:])
        if np.sum(pair_sig) > 0:
            sig_inds = np.where(pair_sig == 1)[0]
            for sig_i in sig_inds:
                s_1 = seg_pairs[sig_i][0]
                s_2 = seg_pairs[sig_i][1]
                ax_box[0].plot([s_1,s_2],[1,1],color='k')
                ax_box[0].scatter((s_2-s_1)/2+s_1,1.05,marker='*',color='k')
        ax_frac[0].plot(np.arange(2),is_taste_seg_means[s_i,:],
                        color=segment_colors[s_i,:],label=s_name)
        ax_frac[0].fill_between(np.arange(2),
                                is_taste_seg_means[s_i,:]+is_taste_seg_stds[s_i,:],
                                is_taste_seg_means[s_i,:]-is_taste_seg_stds[s_i,:],
                                color=segment_colors[s_i,:],alpha=0.2,label='_')
    ax_frac[0].legend(loc='upper left')
    ax_box[0].set_xticks(np.arange(num_segments),unique_segment_names)
    ax_frac[0].set_title('Is Taste Fraction')
    ax_box[0].set_title('Is Taste Fraction')
    ax_frac[0].set_xticks(np.arange(2),labels)
    
    dt = 'which_taste'
    which_taste_data = [] #num taste x num seg x num anim
    for t_i, t_name in enumerate(unique_taste_names):
        all_seg_data = []
        for s_i, s_name in enumerate(unique_segment_names):
            all_animal_data = []
            for name_i, name in enumerate(unique_given_names):
                is_taste_inds = np.where(np.argmax(dev_decode_data[name]['is_taste'][s_name],1) == 0)[0]
                decode_data_i = np.squeeze(dev_decode_data[name][dt][s_name][is_taste_inds,:])
                num_dev_is_taste, _ = np.shape(decode_data_i)
                decode_argmax = np.squeeze(np.argmax(decode_data_i,1))
                taste_names = dev_decode_data[name]['dig_in_names']
                an_t_i = np.where(np.array(taste_names) == t_name)[0]
                if len(an_t_i) > 0:
                    num_taste_decode = len(np.where(decode_argmax == an_t_i)[0])
                    frac_taste_decode = num_taste_decode/num_dev_is_taste
                    all_animal_data.append(frac_taste_decode)
            all_seg_data.append(all_animal_data)
        which_taste_data.append(np.array(all_seg_data))
    #Pairwise TTests
    sac_ind = [i for i in range(num_tastes) if unique_taste_names[i] == 'saccharin'][0]
    pair_sig = np.zeros(len(seg_pairs))
    for sp_i, sp in enumerate(seg_pairs):
        s_1 = sp[0]
        s_2 = sp[1]
        s_data1 = which_taste_data[sac_ind][s_1,:]
        s_data2 = which_taste_data[sac_ind][s_2,:]
        ks_res = ks_2samp(s_data1, s_data2)
        if ks_res.pvalue < 0.05:
            pair_sig[sp_i] = 1
    which_taste_seg_means = np.nanmean(which_taste_data,-1) #num taste x num seg
    which_taste_seg_stds = np.nanstd(which_taste_data,-1) #num taste x num seg
    for s_i, s_name in enumerate(unique_segment_names):
        box_data = which_taste_data[sac_ind][s_i,:]
        ax_box[1].boxplot(box_data,positions=s_i*np.ones(1))
        box_x = (np.random.rand(len(box_data))-0.5)/10 + s_i
        for n_i in range(num_anim):
            ax_box[1].scatter(box_x[n_i],box_data[n_i],alpha=0.2,c=anim_colors[n_i,:])
        if np.sum(pair_sig) > 0:
            sig_inds = np.where(pair_sig == 1)[0]
            for sig_i in sig_inds:
                s_1 = seg_pairs[sig_i][0]
                s_2 = seg_pairs[sig_i][1]
                ax_box[1].plot([s_1,s_2],[1+0.1*sig_i,1+0.1*sig_i],color='k')
                ax_box[1].scatter((s_2-s_1)/2+s_1,1.05+0.1*sig_i,marker='*',\
                                  color='k')
        ax_frac[1].plot(np.arange(num_tastes),which_taste_seg_means[:,s_i],
                        color=segment_colors[s_i,:],label=s_name)
        ax_frac[1].fill_between(np.arange(num_tastes),
                                which_taste_seg_means[:,s_i]+which_taste_seg_stds[:,s_i],
                                which_taste_seg_means[:,s_i]-which_taste_seg_stds[:,s_i],
                                color=segment_colors[s_i,:],label='_',alpha=0.2)
    ax_frac[1].legend(loc='upper left')
    ax_box[1].set_xticks(np.arange(num_segments),unique_segment_names)
    ax_frac[1].set_title('Which Taste Fraction')
    ax_box[1].set_title('Saccharin Fraction')
    ax_frac[1].set_xticks(np.arange(num_tastes),unique_taste_names)
    plt.figure(f_frac)
    plt.tight_layout()
    f_frac.savefig(os.path.join(results_dir,'decode_fracs.png'))
    f_frac.savefig(os.path.join(results_dir,'decode_fracs.svg'))
    plt.close(f_frac)
    plt.figure(f_box)
    plt.tight_layout()
    f_box.savefig(os.path.join(results_dir,'decode_box.png'))
    f_box.savefig(os.path.join(results_dir,'decode_box.svg'))
    plt.close(f_box)
    
    f_box_te, ax_box_te = plt.subplots(nrows = num_tastes, ncols = 2, sharey = True,
                                   figsize = (4*num_tastes,8))
    unique_epoch_names = ['Epoch ' + str(e_i) for e_i in unique_epochs]
    which_taste_epoch_data = [] #num_taste x num seg x num epoch x num animals
    for t_i, t_name in enumerate(unique_taste_names):
        all_seg_data = []
        for s_i, s_name in enumerate(unique_segment_names):
            all_epoch_data = []
            for e_i in unique_epochs:
                all_animal_data = []
                for name_i, name in enumerate(unique_given_names):
                    is_taste_inds = np.where(np.argmax(dev_decode_data[name]['is_taste'][s_name],1) == 0)[0]
                    taste_names = dev_decode_data[name]['dig_in_names']
                    an_t_i = np.where(np.array(taste_names) == t_name)[0]
                    which_taste_inds = np.where(np.argmax(dev_decode_data[name]['which_taste'][s_name],1) == an_t_i)[0]
                    is_which_taste_inds = np.intersect1d(is_taste_inds, which_taste_inds)
                    num_is_which_taste = len(is_which_taste_inds)
                    decode_data_i = np.squeeze(dev_decode_data[name]['which_epoch'][s_name][is_which_taste_inds,:])
                    data_argmax = np.argmax(decode_data_i,1)
                    an_epochs = dev_decode_data[name]['epochs_to_analyze']
                    an_e_i = np.where(np.array(an_epochs) == e_i)[0]
                    if (len(an_e_i) > 0):
                        frac_epoch_decodes = len(np.where(data_argmax == an_e_i)[0])/num_is_which_taste
                        all_animal_data.append(frac_epoch_decodes)
                all_epoch_data.append(all_animal_data)
            all_seg_data.append(all_epoch_data)
        #Calculate significance between segments by epoch
        which_taste_epoch_data.append(np.array(all_seg_data))
    #Averages
    which_taste_epoch_seg_means = np.nanmean(which_taste_epoch_data,-1) #num_taste x num seg x num epoch
    which_taste_epoch_seg_stds = np.nanstd(which_taste_epoch_data,-1) #num_taste x num seg x num epoch
    for t_i, t_name in enumerate(unique_taste_names):
        taste_box_data = which_taste_epoch_data[t_i]
        #Plot grouped by segment with epochs next to each other
        taste_box_data_reshape = []
        taste_box_data_x_labels = []
        for s_i in range(num_segments):
            for e_i in range(num_epochs):
                taste_box_data_reshape.append(taste_box_data[s_i][e_i,:])
                taste_box_data_x_labels.append(unique_epoch_names[e_i])
        box_x_inds = np.arange(len(taste_box_data_reshape))
        ax_box_te[t_i,0].boxplot(taste_box_data_reshape,positions=box_x_inds)
        for x_i in box_x_inds:
            box_x = (np.random.rand(len(box_data))-0.5)/10 + x_i
            for n_i in range(num_anim):
                ax_box_te[t_i,0].scatter(box_x[n_i],taste_box_data_reshape[x_i][n_i],alpha=0.2,c=anim_colors[n_i,:])
        #Add significance bars
        for s_i in range(num_segments):
            #Perform epoch pair significance tests
            s_i_x_start = s_i*num_epochs
            s_i_x_end = s_i*num_epochs + (num_epochs-1)
            #Plot segment groups
            ax_box_te[t_i,0].plot([s_i_x_start,s_i_x_end],[0,0],\
                                  label=unique_segment_names[s_i],
                                  c = segment_colors[s_i,:])
            for ep_i, ep in enumerate(epoch_pairs):
                e_1 = ep[0]
                e_2 = ep[1]
                e_data1 = which_taste_epoch_data[t_i][s_i,e_1,:]
                e_data2 = which_taste_epoch_data[t_i][s_i,e_2,:]
                ks_res = ks_2samp(e_data1, e_data2)
                if ks_res.pvalue < 0.05:
                    x_1 = s_i_x_start+e_1
                    x_2 = s_i_x_start+e_2
                    ax_box_te[t_i,0].plot([x_1,x_2],[1+0.1*ep_i,1+0.1*ep_i],\
                                          color='k',label='_')
                    ax_box_te[t_i,0].scatter((x_2-x_1)/2+x_1,1.05+0.1*ep_i,\
                                      marker='*',color='k',label='_')
        ax_box_te[0,0].legend(loc='upper left')
        ax_box_te[t_i,0].set_xticks(box_x_inds,taste_box_data_x_labels,rotation=45)
        ax_box_te[t_i,0].set_title(t_name + ' segment grouped')
        #Plot grouped by epoch with segments next to each other
        taste_box_data_reshape = []
        taste_box_data_x_labels = []
        for e_i in range(num_epochs):
            for s_i in range(num_segments):
                taste_box_data_reshape.append(taste_box_data[s_i][e_i,:])
                taste_box_data_x_labels.append(unique_segment_names[s_i])
        box_x_inds = np.arange(len(taste_box_data_reshape))
        ax_box_te[t_i,1].boxplot(taste_box_data_reshape,positions=box_x_inds)
        for x_i in box_x_inds:
            box_x = (np.random.rand(len(box_data))-0.5)/10 + x_i
            for n_i in range(num_anim):
                ax_box_te[t_i,1].scatter(box_x[n_i],taste_box_data_reshape[x_i][n_i],\
                                         alpha=0.2,c=anim_colors[n_i,:])
        #Add significance bars
        for e_i in range(num_epochs):
            #Perform epoch pair significance tests
            e_i_x_start = e_i*num_segments
            e_i_x_end = e_i*num_segments + (num_segments-1)
            #Plot epoch groups
            ax_box_te[t_i,1].plot([e_i_x_start,e_i_x_end],[0,0],\
                                  label=unique_epoch_names[e_i],
                                  c = epoch_colors[e_i,:])
            for sp_i, sp in enumerate(seg_pairs):
                s_1 = sp[0]
                s_2 = sp[1]
                e_data1 = which_taste_epoch_data[t_i][s_1,e_i,:]
                e_data2 = which_taste_epoch_data[t_i][s_2,e_i,:]
                ks_res = ks_2samp(e_data1, e_data2)
                if ks_res.pvalue < 0.05:
                    x_1 = e_i_x_start+s_1
                    x_2 = e_i_x_end+s_2
                    ax_box_te[t_i,1].plot([x_1,x_2],[1+0.1*sp_i,1+0.1*sp_i],\
                                          color='k',label='_')
                    ax_box_te[t_i,1].scatter((x_2-x_1)/2+x_1,1.05+0.1*sp_i,\
                                      marker='*',color='k',label='_')
        ax_box_te[0,1].legend(loc='upper left')
        ax_box_te[t_i,1].set_xticks(box_x_inds,taste_box_data_x_labels,rotation=45)
        ax_box_te[t_i,1].set_title(t_name + ' epoch grouped')
    plt.suptitle('Epoch/Segment Grouped Decode Fractions')
    plt.tight_layout()
    f_box_te.savefig(os.path.join(results_dir,'decode_box_te.png'))
    f_box_te.savefig(os.path.join(results_dir,'decode_box_te.svg'))
    plt.close(f_box_te)
    
def cross_dataset_dev_split_decode_frac_plots(dev_split_decode_data, unique_given_names,
                                            unique_segment_names, unique_taste_names, 
                                            decode_types, results_dir):
    """Plot the dev decode fractions by decode type"""
    #Variables
    num_anim = len(unique_given_names)
    num_segments = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    cmap = colormaps['brg']
    segment_colors = cmap(np.linspace(0, 1, num_segments))
    cmap = colormaps['gist_rainbow']
    anim_colors = cmap(np.linspace(0, 1, num_anim))
    seg_pairs = list(combinations(np.arange(num_segments), 2))
    
    #Is-Taste Calcs
    dt = 'is_taste'
    is_taste_seg_data_1 = [] #num seg x num anim x 2 - split 1
    is_taste_seg_data_2 = [] #num seg x num anim x 2 - split 2
    same_is_taste_data = [] #num_seg x num_anim
    for s_i, s_name in enumerate(unique_segment_names):
        all_animal_1_data = []
        all_animal_2_data = []
        all_animal_same_data = []
        for name_i, name in enumerate(unique_given_names):
            try:
                #decode_data_i will be num_dev x 2 x 2 where the first column is the first '2' is the taste/no-taste and the second is which half
                decode_data_i = np.squeeze(dev_split_decode_data[name]['decode_data'][s_name][dt])
                num_dev, _, _ = np.shape(decode_data_i)
                split_1_argmax = np.squeeze(np.argmax(np.squeeze(decode_data_i[:,:,0]),1))
                split_2_argmax = np.squeeze(np.argmax(np.squeeze(decode_data_i[:,:,1]),1))
                no_taste_1_count = np.sum(split_1_argmax)
                no_taste_2_count = np.sum(split_2_argmax)
                all_animal_1_data.append([(num_dev - no_taste_1_count)/num_dev,no_taste_1_count/num_dev])
                all_animal_2_data.append([(num_dev - no_taste_2_count)/num_dev,no_taste_2_count/num_dev])
                same_count = len(np.where(split_1_argmax - split_2_argmax == 0)[0])
                all_animal_same_data.append([same_count/num_dev])
            except:
                decode_data_i = []
        is_taste_seg_data_1.append(np.array(all_animal_1_data))
        is_taste_seg_data_2.append(np.array(all_animal_2_data))
        same_is_taste_data.append(all_animal_same_data)
    same_is_taste_data = np.squeeze(same_is_taste_data) #num_seg x num_anim
    is_taste_seg_data_1 = np.squeeze(is_taste_seg_data_1) #num seg x num anim x 2
    is_taste_seg_data_2 = np.squeeze(is_taste_seg_data_2) #num seg x num anim x 2
    
    #Which-Taste and Epoch pair calcs
    same_taste_data = [] #num taste x num seg x num anim
    all_taste_ep_pair_frac = dict()
    max_cp = 0
    for t_i, t_name in enumerate(unique_taste_names):
        all_seg_same_taste_frac = []
        all_seg_animal_ep_pair_frac = dict()
        for s_i, s_name in enumerate(unique_segment_names):
            all_animal_same_taste_frac = []
            all_animal_ep_pair_frac = dict()
            for name_i, name in enumerate(unique_given_names):
                taste_names = dev_split_decode_data[name]['dig_in_names']
                an_t_i = np.where(np.array(taste_names) == t_name)[0]
                #Is taste indices
                decode_data_i = np.squeeze(dev_split_decode_data[name]['decode_data'][s_name]['is_taste']) #deviation x is taste x split index
                num_dev, _, _ = np.shape(decode_data_i)
                split_1_argmax = np.squeeze(np.argmax(np.squeeze(decode_data_i[:,:,0]),1))
                split_2_argmax = np.squeeze(np.argmax(np.squeeze(decode_data_i[:,:,1]),1))
                is_taste_inds_1 = np.where(split_1_argmax == 0)[0]
                is_taste_inds_2 = np.where(split_2_argmax == 0)[0]
                both_taste_inds = np.intersect1d(is_taste_inds_1,is_taste_inds_2)
                #Which taste decode fracs
                which_decode_data_i = np.squeeze(dev_split_decode_data[name]['decode_data'][s_name]['which_taste']) #deviation x which taste x split index
                #Individual split taste decode fracs
                split_1_which_argmax = np.squeeze(np.argmax(np.squeeze(which_decode_data_i[is_taste_inds_1,:,0]),1))
                split_2_which_argmax = np.squeeze(np.argmax(np.squeeze(which_decode_data_i[is_taste_inds_2,:,1]),1))
                split_1_taste_inds = np.where(split_1_which_argmax == an_t_i)[0]
                split_2_taste_inds = np.where(split_2_which_argmax == an_t_i)[0]
                #Both split taste decode fracs
                both_taste_inds = np.intersect1d(split_1_taste_inds,split_2_taste_inds)
                all_animal_same_taste_frac.append(len(both_taste_inds)/num_dev)
                #Which epoch data
                which_epoch_data = np.squeeze(dev_split_decode_data[name]['decode_data'][s_name]['which_epoch']) #deviation x num epochs x split index
                _, num_epoch, _ = np.shape(which_epoch_data)
                if num_epoch > max_cp:
                    max_cp = num_epoch
                epoch_pairs = list(itertools.combinations(np.arange(num_epoch),2))
                epoch_pairs.extend(list(itertools.combinations(reversed(np.arange(num_epoch)),2)))
                epoch_pairs.extend([(i,i) for i in range(num_epoch)])
                both_taste_epoch_prob_1 = np.squeeze(which_epoch_data[is_taste_inds_1[both_taste_inds],:,0]) #len(both_taste_inds) x num epochs
                both_taste_epoch_prob_2 = np.squeeze(which_epoch_data[is_taste_inds_1[both_taste_inds],:,1])
                if len(both_taste_inds) > 1:
                    both_taste_epoch_ind_1 = np.argmax(both_taste_epoch_prob_1,1)
                    both_taste_epoch_ind_2 = np.argmax(both_taste_epoch_prob_2,1)
                elif len(both_taste_inds) == 1:
                    both_taste_epoch_ind_1 = np.argmax(both_taste_epoch_prob_1)
                    both_taste_epoch_ind_2 = np.argmax(both_taste_epoch_prob_2)
                else:
                    both_taste_epoch_ind_1 = []
                    both_taste_epoch_ind_2 = []
                for ep in epoch_pairs:
                    ep_1 = ep[0]
                    ep_2 = ep[1]
                    ep_1_ind = np.where(both_taste_epoch_ind_1 == ep_1)[0]
                    ep_2_ind = np.where(both_taste_epoch_ind_2 == ep_2)[0]
                    pair_counts = len(np.intersect1d(ep_1_ind, ep_2_ind))
                    if name_i == 0:
                        all_animal_ep_pair_frac[str(ep)] = []
                    if len(both_taste_inds) > 0:
                        all_animal_ep_pair_frac[str(ep)].append(pair_counts/len(both_taste_inds))
                    else:
                        all_animal_ep_pair_frac[str(ep)].append(np.nan)
            all_seg_same_taste_frac.append(all_animal_same_taste_frac)
            all_seg_animal_ep_pair_frac[s_name] = all_animal_ep_pair_frac
        same_taste_data.append(np.array(all_seg_same_taste_frac))
        all_taste_ep_pair_frac[t_name] = all_seg_animal_ep_pair_frac
    
    #Create plots if is-taste, which-taste fracs
    f_box, ax_box = plt.subplots(nrows = 2, ncols = 1, figsize = (4,8))
    #    is-taste plots
    ax_box[0].set_ylim([-0.1,1.1])
    for s_i, s_name in enumerate(unique_segment_names):
        box_data_1 = list(is_taste_seg_data_1[s_i][:,0])
        box_data_2 = list(is_taste_seg_data_2[s_i][:,0])
        box_x = (np.random.rand(len(box_data_1))-0.5)/10
        for n_i in range(num_anim):
            ax_box[0].scatter(box_x[n_i]+2*s_i,box_data_1[n_i],alpha=0.2,c=anim_colors[n_i,:],label='_')
            ax_box[0].scatter(box_x[n_i]+2*s_i+1,box_data_2[n_i],alpha=0.2,c=anim_colors[n_i,:],label='_')
        ax_box[0].boxplot(box_data_1,positions=[s_i*2])
        ax_box[0].boxplot(box_data_2,positions=[s_i*2+1])
        ax_box[0].plot([s_i*2,s_i*2+1],[1,1],c=segment_colors[s_i,:],label=s_name)
    x_tick_labels = []
    for us_i, us_n in enumerate(unique_segment_names):
        x_tick_labels.append('\nSplit 1')
        x_tick_labels.append('\nSplit 2')
    ax_box[0].set_xticks(np.arange(2*num_segments),x_tick_labels)
    ax_box[0].set_title('Is Taste Fraction')
    ax_box[0].legend(loc='upper left')
    #    which-taste plots
    x_tick_labels = []
    max_frac = np.max(same_taste_data)
    for s_i, s_name in enumerate(unique_segment_names):
        for t_i, t_name in enumerate(unique_taste_names):
            box_data = same_taste_data[t_i][s_i,:]
            x_loc = s_i*num_tastes + t_i
            x_tick_labels.append(t_name)
            ax_box[1].boxplot(box_data,positions=x_loc*np.ones(1))
            box_x = (np.random.rand(len(box_data))-0.5)/10 + x_loc
            for n_i in range(num_anim):
                ax_box[1].scatter(box_x[n_i],box_data[n_i],alpha=0.2,c=anim_colors[n_i,:])
        ax_box[1].plot([s_i*num_tastes,s_i*num_tastes + (num_tastes-1)],\
                       [max_frac+0.1*max_frac,max_frac+0.1*max_frac],\
                       c=segment_colors[s_i,:],label=s_name)
    ax_box[1].set_xticks(np.arange(num_tastes*num_segments),x_tick_labels)
    ax_box[1].set_title('Same Taste Fractions')
    ax_box[1].legend(loc='upper left')
    plt.tight_layout()
    f_box.savefig(os.path.join(results_dir,'is_taste_which_taste_decode_box.png'))
    f_box.savefig(os.path.join(results_dir,'is_taste_which_taste_decode_box.svg'))
    plt.close(f_box)
    
    #Create plots for epoch pair fractions
    epoch_pairs = list(itertools.combinations(np.arange(max_cp),2))
    epoch_pairs.extend(list(itertools.combinations(reversed(np.arange(max_cp)),2)))
    epoch_pairs.extend([(i,i) for i in range(max_cp)])
    epoch_pairs_labels = [str(ep) for ep in epoch_pairs]
    f_ep_box, ax_ep_box = plt.subplots(nrows = len(unique_taste_names), \
                                       ncols = len(unique_segment_names), \
                                    figsize = (len(unique_segment_names)*3,\
                                               len(unique_taste_names)*3))
    f_ep_trend, ax_ep_trend = plt.subplots(nrows = len(unique_taste_names), \
                                       ncols = len(unique_segment_names), \
                                    figsize = (len(unique_segment_names)*3,\
                                               len(unique_taste_names)*3))
    f_ep_avg, ax_ep_avg = plt.subplots(nrows = len(unique_taste_names), \
                                       ncols = len(unique_segment_names), \
                                    figsize = (len(unique_segment_names)*3,\
                                               len(unique_taste_names)*3))
    for t_i, t_name in enumerate(unique_taste_names):
        all_seg_animal_ep_pair_frac = all_taste_ep_pair_frac[t_name]
        for s_i, s_name in enumerate(unique_segment_names):
            all_animal_ep_pair_frac = all_seg_animal_ep_pair_frac[s_name]
            all_ep_trends = np.zeros((len(epoch_pairs),num_anim))
            for ep_i, ep in enumerate(epoch_pairs):
                box_data = np.array(all_animal_ep_pair_frac[str(ep)])
                all_ep_trends[ep_i,:] = box_data
                box_x = ep_i + (np.random.rand(len(box_data))-0.5)/10
                for n_i in range(num_anim):
                    ax_ep_box[t_i,s_i].scatter(box_x[n_i],box_data[n_i],\
                                               alpha=0.2,c=anim_colors[n_i,:],\
                                                   label='_')
                ax_ep_box[t_i,s_i].boxplot(box_data[~np.isnan(box_data)],positions=[ep_i])
            for n_i in range(num_anim):
                ax_ep_trend[t_i,s_i].plot(np.arange(len(epoch_pairs)),\
                                          all_ep_trends[:,n_i],alpha=0.3,
                                            c=anim_colors[n_i,:])
            mean_trend = np.nanmean(all_ep_trends,1)
            std_trend = np.nanstd(all_ep_trends,1)
            ax_ep_trend[t_i,s_i].plot(np.arange(len(epoch_pairs)),\
                                      mean_trend,c='k',label='Mean')
            ax_ep_trend[t_i,s_i].fill_between(np.arange(len(epoch_pairs)),\
                                      mean_trend-std_trend,mean_trend+std_trend,\
                                          color='k',alpha=0.2,label='Std')
            ax_ep_avg[t_i,s_i].pie(mean_trend,labels=epoch_pairs_labels,\
                                   autopct='%1.1f%%')
            ax_ep_box[t_i,s_i].set_xticks(np.arange(len(epoch_pairs)),epoch_pairs_labels,\
                                          rotation=45)
            ax_ep_box[t_i,s_i].set_ylabel(t_name + '\nFraction')
            ax_ep_box[t_i,s_i].set_title(s_name)
            plt.tight_layout()
            ax_ep_trend[t_i,s_i].set_xticks(np.arange(len(epoch_pairs)),epoch_pairs_labels,\
                                          rotation=45)
            ax_ep_trend[t_i,s_i].set_ylabel(t_name + '\nFraction')
            ax_ep_trend[t_i,s_i].set_title(s_name)
            ax_ep_avg[t_i,s_i].set_title(s_name + ' ' + t_name)
            plt.tight_layout()
    f_ep_box.savefig(os.path.join(results_dir,'which_epoch_decode_box.png'))
    f_ep_box.savefig(os.path.join(results_dir,'which_epoch_decode_box.svg'))
    plt.close(f_ep_box)
    f_ep_trend.savefig(os.path.join(results_dir,'which_epoch_decode_trend.png'))
    f_ep_trend.savefig(os.path.join(results_dir,'which_epoch_decode_trend.svg'))
    plt.close(f_ep_trend)
    f_ep_avg.savefig(os.path.join(results_dir,'which_epoch_decode_svg_pie.png'))
    f_ep_avg.savefig(os.path.join(results_dir,'which_epoch_decode_svg_pie.svg'))
    plt.close(f_ep_avg)
                
    #Plot epoch pairs separately with segments compared
    ep_sqrt = np.ceil(np.sqrt(len(epoch_pairs))).astype('int')
    ep_ref = np.reshape(np.ceil(np.arange(ep_sqrt**2)).astype('int'),(ep_sqrt,ep_sqrt))
    f_ep_segcomp, ax_ep_segcomp = plt.subplots(nrows = ep_sqrt, ncols = ep_sqrt, \
                                    figsize = (ep_sqrt*4,ep_sqrt*4), sharey = True)
    for ep_i, ep in enumerate(epoch_pairs):
        ep_index = np.where(ep_ref == ep_i)
        ep_row = ep_index[0][0]
        ep_col = ep_index[1][0]
        ax_ep_segcomp[ep_row,ep_col].set_title(epoch_pairs_labels[ep_i])
        for t_i, t_name in enumerate(unique_taste_names):
            all_seg_animal_ep_pair_frac = all_taste_ep_pair_frac[t_name]
            all_ep_trends = np.zeros((num_segments,num_anim))
            for s_i, s_name in enumerate(unique_segment_names):
                all_animal_ep_pair_frac = np.array(all_seg_animal_ep_pair_frac[s_name][str(ep)])
                all_ep_trends[s_i,:] = all_animal_ep_pair_frac
            mean_vals = np.nanmean(all_ep_trends,1)
            std_vals = np.nanstd(all_ep_trends,1)
            ax_ep_segcomp[ep_row,ep_col].plot(np.arange(num_segments),\
                                              mean_vals,label=t_name)
            # ax_ep_segcomp[ep_row,ep_col].fill_between(np.arange(num_segments),\
            #                                   mean_vals+std_vals,\
            #                                   mean_vals-std_vals,\
            #                                   alpha=0.3,label='_')
        ax_ep_segcomp[ep_row,ep_col].set_xticks(np.arange(num_segments),\
                                                unique_segment_names)
    ax_ep_segcomp[0,0].legend(loc='upper left')
    plt.suptitle('Average Decoding Fractions Across Animals')
    plt.tight_layout()
    f_ep_segcomp.savefig(os.path.join(results_dir,'which_epoch_decode_avg.png'))
    f_ep_segcomp.savefig(os.path.join(results_dir,'which_epoch_decode_avg.svg'))
    plt.close(f_ep_segcomp)
    
def rate_plots(dev_corr_rate_dict, corr_cutoffs, unique_segment_names, 
               max_epochs, unique_taste_names, corr_name, corr_cutoff_save, 
               corr_cutoff_indiv_save):
    
    #Called from cross_dataset_dev_by_corr_cutoff()
    num_tastes = len(unique_taste_names)
    num_segs = len(unique_segment_names)
    cmap = colormaps['gist_rainbow']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    taste_pairs = list(combinations(np.arange(num_tastes), 2))
    zoom_inds = np.where(corr_cutoffs >= 0.25)[0]
    cutoff_zoom_ind = zoom_inds[0]
    
    f_cc_taste_rate, ax_cc_taste_rate = plt.subplots(nrows = len(unique_segment_names),\
                                           ncols = max_epochs, sharex = True,\
                                           sharey = True, figsize = (8,8))
    f_cc_taste_rate_zoom, ax_cc_taste_rate_zoom = plt.subplots(nrows = len(unique_segment_names),\
                                           ncols = max_epochs, sharex = True,\
                                           sharey = True, figsize = (8,8))
    f_cc_taste_rate_box, ax_cc_taste_rate_box = plt.subplots(nrows = len(unique_segment_names),\
                                           ncols = max_epochs, sharex = True,\
                                           sharey = True, figsize = (8,8))
    f_cc_null_diffs, ax_cc_null_diffs = plt.subplots(nrows = len(unique_segment_names),\
                                           ncols = max_epochs, sharex = True,\
                                           sharey = True, figsize = (8,8))
    f_cc_null_diffs_zoom, ax_cc_null_diffs_zoom = plt.subplots(nrows = len(unique_segment_names),\
                                           ncols = max_epochs, sharex = True,\
                                           sharey = True, figsize = (8,8))
    for s_i, seg_name in enumerate(unique_segment_names):
        for cp_i in range(max_epochs):
            all_null_taste_rates = [] #Only for true tastes
            #Plot individual animal points at cutoff value
            indiv_animal_at_cutoff = []
            taste_curves = []
            taste_names = []
            none_curve = []
            for t_i, taste in enumerate(unique_taste_names):
                taste_rates = dev_corr_rate_dict[seg_name][taste][cp_i]['true'] #num_anim x num_cutoffs
                num_anim, _ = np.shape(taste_rates)
                null_rates = dev_corr_rate_dict[seg_name][taste][cp_i]['null']
                indiv_animal_at_cutoff.append(taste_rates[:,cutoff_zoom_ind])
                taste_mean = np.nanmean(taste_rates,0)
                taste_std = np.nanstd(taste_rates,0)
                taste_min = taste_mean - taste_std
                taste_min[taste_min < 0] = 0
                if taste == 'none':
                    none_curve.append(taste_mean)
                else:
                    all_null_taste_rates.extend(list(null_rates))
                    taste_curves.append(taste_mean)
                    taste_names.append(taste)
                #Plot individually cutoff curves
                f_indiv = plt.figure(figsize=(5,5))
                for na_i in range(num_anim):
                    plt.plot(corr_cutoffs,taste_rates[na_i,:],color='b',alpha=0.2,\
                             label='_')
                plt.plot(corr_cutoffs,taste_mean,color='b',alpha=1,\
                         linestyle='dashed',label='Animal Avg')
                plt.fill_between(corr_cutoffs,taste_min,taste_mean+taste_std,color='b',
                                 alpha=0.1,label='Null Std')
                null_mean = np.nanmean(null_rates,0)
                null_std = np.nanstd(null_rates,0)
                null_min = null_mean-null_std
                null_min[null_min < 0] = 0
                plt.plot(corr_cutoffs,null_mean,color='k',alpha=1,\
                         linestyle='dashed',label='Null Avg')
                plt.fill_between(corr_cutoffs,null_min,null_mean+null_std,color='k',
                                 alpha=0.1,label='Null Std')
                plt.ylabel('Avg Rate (Hz)')
                plt.xlabel('Min Correlation Cutoff')
                plt.legend(loc='upper left')
                plt.title(seg_name + '\n' + taste + '\nEpoch ' + str(cp_i))
                plt.tight_layout()
                f_indiv.savefig(os.path.join(corr_cutoff_indiv_save,corr_name + '_' + seg_name + '_' + taste + '_Epoch_' + str(cp_i) + '_avg_rates.png'))
                f_indiv.savefig(os.path.join(corr_cutoff_indiv_save,corr_name + '_' + seg_name + '_' + taste + '_Epoch_' + str(cp_i) + '_avg_rates.svg'))
                plt.close(f_indiv)
                #Plot in joint plot cutoff curves
                ax_cc_taste_rate[s_i,cp_i].plot(corr_cutoffs,taste_mean,\
                                                color=taste_colors[t_i,:],\
                                                    alpha=1,label=taste)
                ax_cc_taste_rate_zoom[s_i,cp_i].plot(corr_cutoffs[zoom_inds],\
                                                     taste_mean[zoom_inds],\
                                                color=taste_colors[t_i,:],\
                                                    alpha=1,label=taste)
            all_null_taste_rates = np.array(all_null_taste_rates)
            null_mean = np.nanmean(all_null_taste_rates,0)
            null_std = np.nanstd(all_null_taste_rates,0)
            null_min = null_mean-null_std
            null_min[null_min<0] = 0
            ax_cc_taste_rate[s_i,cp_i].plot(corr_cutoffs,null_mean,\
                                            color='k',alpha=1,
                                            linestyle='dashed',label='Null')
            ax_cc_taste_rate[s_i,cp_i].fill_between(corr_cutoffs,null_min,\
                                                    null_mean+null_std,color='k',\
                                                        alpha=0.1,label='Null Std')
            ax_cc_taste_rate_zoom[s_i,cp_i].plot(corr_cutoffs[zoom_inds],null_mean[zoom_inds],\
                                            color='k',alpha=1,
                                            linestyle='dashed',label='Null')
            ax_cc_taste_rate_zoom[s_i,cp_i].fill_between(corr_cutoffs[zoom_inds],\
                                                         null_min[zoom_inds],\
                                                    null_mean[zoom_inds]+null_std[zoom_inds],\
                                                        color='k',alpha=0.1,\
                                                            label='Null Std')
            #Diff plot
            none_curve = np.array(none_curve).squeeze()
            ax_cc_null_diffs[s_i,cp_i].plot(corr_cutoffs,np.zeros(len(corr_cutoffs)),\
                                            color='k',alpha=0.5,linestyle='dashed')
            ax_cc_null_diffs_zoom[s_i,cp_i].plot(corr_cutoffs,np.zeros(len(corr_cutoffs)),\
                                            color='k',alpha=0.5,linestyle='dashed')
            for t_i, taste in enumerate(taste_names):
                ax_cc_null_diffs[s_i,cp_i].plot(corr_cutoffs,\
                                                np.array(taste_curves[t_i])-none_curve,\
                                                color=taste_colors[t_i,:],\
                                                    alpha=1,label=taste)
                ax_cc_null_diffs_zoom[s_i,cp_i].plot(corr_cutoffs[zoom_inds],\
                                                (np.array(taste_curves[t_i])-none_curve)[zoom_inds],\
                                                color=taste_colors[t_i,:],\
                                                    alpha=1,label=taste)
            if s_i == 0:
                ax_cc_taste_rate[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                ax_cc_taste_rate_zoom[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                ax_cc_taste_rate_box[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                ax_cc_null_diffs[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                ax_cc_null_diffs_zoom[s_i,cp_i].set_title('Epoch ' + str(cp_i))
            if cp_i == 0:
                ax_cc_taste_rate[s_i,cp_i].set_ylabel(seg_name + '\nRate (Hz)')
                ax_cc_taste_rate_zoom[s_i,cp_i].set_ylabel(seg_name + '\nRate (Hz)')
                ax_cc_taste_rate_box[s_i,cp_i].set_ylabel(seg_name + '\nRate (Hz)')
                ax_cc_null_diffs[s_i,cp_i].set_ylabel(seg_name + '\nRate Diff From None(Hz)')
                ax_cc_null_diffs_zoom[s_i,cp_i].set_ylabel(seg_name + '\nRate Diff From None(Hz)')
            if s_i == num_segs-1:
                ax_cc_taste_rate[s_i,cp_i].set_xlabel('Min. Correlation Cutoff')
                ax_cc_taste_rate_zoom[s_i,cp_i].set_xlabel('Min. Correlation Cutoff')
                ax_cc_null_diffs[s_i,cp_i].set_xlabel('Min. Correlation Cutoff')
                ax_cc_null_diffs_zoom[s_i,cp_i].set_xlabel('Min. Correlation Cutoff')
            #Animal dist pairwise sig
            sig_pair = []
            for tp_i, tp in enumerate(taste_pairs):
                stat = ttest_ind(indiv_animal_at_cutoff[tp[0]],indiv_animal_at_cutoff[tp[1]],\
                                 equal_var=False,nan_policy='omit')
                if stat[1]<=0.05:
                    sig_pair.append(tp)
            #Combined box plots
            ax_cc_taste_rate_box[s_i,cp_i].boxplot(indiv_animal_at_cutoff,showmeans=False,showfliers=False)
            for t_i, taste in enumerate(unique_taste_names):
                x_locs = t_i + 1 + 0.1*np.random.randn(len(indiv_animal_at_cutoff[t_i]))
                ax_cc_taste_rate_box[s_i,cp_i].scatter(x_locs,indiv_animal_at_cutoff[t_i],alpha=0.5,color='g')
            for sp_i, sp in enumerate(sig_pair):
                ax_cc_taste_rate_box[s_i,cp_i].plot([sp[0]+1,sp[1]+1],[-0.01*(sp_i+1),-0.01*(sp_i+1)],color='k',alpha=0.3)
                x_sig = (sp[0]+1)+(sp[1]-sp[0])/2
                ax_cc_taste_rate_box[s_i,cp_i].scatter(x_sig,-0.01*(sp_i+1),marker='*',color='k')
            ax_cc_taste_rate_box[s_i,cp_i].set_xticks(np.arange(num_tastes) + 1,unique_taste_names)
            #Indiv animal box plots
            f_indiv_animal = plt.figure(figsize=(5,5))
            plt.boxplot(indiv_animal_at_cutoff,showmeans=False,showfliers=False)
            for t_i, taste in enumerate(unique_taste_names):
                x_locs = t_i + 1 + 0.1*np.random.randn(len(indiv_animal_at_cutoff[t_i]))
                plt.scatter(x_locs,indiv_animal_at_cutoff[t_i],alpha=0.5,color='g')
            for sp_i, sp in enumerate(sig_pair):
                plt.plot([sp[0]+1,sp[1]+1],[-0.01*(sp_i+1),-0.01*(sp_i+1)],color='k',alpha=0.3)
                x_sig = (sp[0]+1)+(sp[1]-sp[0])/2
                plt.scatter(x_sig,-0.01*(sp_i+1),marker='*',color='k')
            plt.xticks(np.arange(num_tastes) + 1,unique_taste_names)
            plt.ylabel('Rate (Hz)')
            plt.title(seg_name + '\nEpoch ' + str(cp_i) + '\nAt Cutoff 0.25')
            plt.tight_layout()
            f_indiv_animal.savefig(os.path.join(corr_cutoff_indiv_save,corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + 'indiv_animal_rates.png'))
            f_indiv_animal.savefig(os.path.join(corr_cutoff_indiv_save,corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + 'indiv_animal_rates.svg'))
            plt.close(f_indiv_animal)
    ax_cc_taste_rate[0,0].legend(loc='upper left')
    plt.suptitle('Rate by Cutoff')
    plt.tight_layout()
    f_cc_taste_rate.savefig(os.path.join(corr_cutoff_save,corr_name + '_taste_rates.png'))
    f_cc_taste_rate.savefig(os.path.join(corr_cutoff_save,corr_name + '_taste_rates.svg'))
    plt.close(f_cc_taste_rate)
    ax_cc_taste_rate_zoom[0,0].legend(loc='upper left')
    ax_cc_taste_rate_zoom[0,0].set_xlim([0.25,1])
    plt.suptitle('Rate by Cutoff')
    plt.tight_layout()
    f_cc_taste_rate_zoom.savefig(os.path.join(corr_cutoff_save,corr_name + '_taste_zoom_rates.png'))
    f_cc_taste_rate_zoom.savefig(os.path.join(corr_cutoff_save,corr_name + '_taste_zoom_rates.svg'))
    plt.close(f_cc_taste_rate_zoom)
    plt.figure(f_cc_taste_rate_box)
    plt.suptitle('Animal Rates at 0.25 Corr')
    plt.tight_layout()
    f_cc_taste_rate_box.savefig(os.path.join(corr_cutoff_save,corr_name + '_taste_box_rates.png'))
    f_cc_taste_rate_box.savefig(os.path.join(corr_cutoff_save,corr_name + '_taste_box_rates.svg'))
    plt.close(f_cc_taste_rate_box)
    plt.figure(f_cc_null_diffs)
    ax_cc_null_diffs[0,0].legend(loc='upper left')
    plt.suptitle('Rate Difference from None by Cutoff')
    plt.tight_layout()
    f_cc_null_diffs.savefig(os.path.join(corr_cutoff_save,corr_name + '_taste_rate_diffs.png'))
    f_cc_null_diffs.savefig(os.path.join(corr_cutoff_save,corr_name + '_taste_rate_diffs.svg'))
    plt.close(f_cc_null_diffs)
    plt.figure(f_cc_null_diffs_zoom)
    ax_cc_null_diffs_zoom[0,0].legend(loc='upper left')
    plt.suptitle('Rate Difference from None by Cutoff')
    plt.tight_layout()
    f_cc_null_diffs_zoom.savefig(os.path.join(corr_cutoff_save,corr_name + '_taste_rate_diffs_zoom.png'))
    f_cc_null_diffs_zoom.savefig(os.path.join(corr_cutoff_save,corr_name + '_taste_rate_diffs_zoom.svg'))
    plt.close(f_cc_null_diffs_zoom)
    
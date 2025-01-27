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
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu, percentileofscore
from scipy.signal import savgol_filter


def cross_dataset_dev_freq(corr_data, unique_given_names, unique_corr_names,
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
                    #Now store the best correlation values by taste and epoch
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


def cross_segment_diffs(corr_data, save_dir, unique_given_names, unique_corr_names,
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
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, unique_corr_names, unique_given_names, unique_segment_names,
                                                   unique_taste_names)

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

def cross_taste_diffs(corr_data, save_dir, unique_given_names, unique_corr_names,
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
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, unique_corr_names, unique_given_names, unique_segment_names,
                                                                      unique_taste_names)

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


def cross_epoch_diffs(corr_data, save_dir, unique_given_names, unique_corr_names,
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
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, unique_corr_names, unique_given_names, unique_segment_names,
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

def combined_corr_by_segment_dist(corr_data, save_dir, unique_given_names, unique_corr_names,
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
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, unique_corr_names, unique_given_names, unique_segment_names,
                                                   unique_taste_names)
    
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
                        ax_cdf[i_3].hist(corr_storage[combo_3][seg],bins=cdf_bins,histtype='step',\
                                 density=True,cumulative=True,label=seg,color=colors[s_i])
                        ax_pdf[i_3].hist(corr_storage[combo_3][seg],bins=pdf_bins,histtype='step',\
                                 density=True,cumulative=False,label=seg,color=colors[s_i])
                        ax_cdf_best[i_3].hist(best_corr_storage[combo_3][seg],\
                                bins=cdf_best_bins,histtype='step',\
                                    density=True,cumulative=True,label=seg,color=colors[s_i])
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

def combined_corr_by_taste_dist(corr_data, save_dir, unique_given_names, unique_corr_names,
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
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, unique_corr_names, unique_given_names, unique_segment_names,
                                                                      unique_taste_names)

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
                            ax_cdf[i_3].hist(corr_storage[combo_3][taste_name],bins=cdf_bins,histtype='step',\
                                     density=True,cumulative=True,label=taste_name,color=colors[t_i])
                            ax_pdf[i_3].hist(corr_storage[combo_3][taste_name],bins=pdf_bins,histtype='step',\
                                     density=True,cumulative=False,label=taste_name,color=colors[t_i])
                            ax_cdf_best[i_3].hist(best_corr_storage[combo_3][taste_name],\
                                    bins=cdf_best_bins,histtype='step',\
                                        density=True,cumulative=True,label=taste_name,color=colors[t_i])
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

def combined_corr_by_epoch_dist(corr_data, save_dir, unique_given_names, unique_corr_names,
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
    unique_data_dict, unique_best_data_dict, max_epochs = reorg_data_dict(corr_data, unique_corr_names, unique_given_names, unique_segment_names,
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
                        ax_cdf[i_3].hist(corr_storage[combo_3][ep],bins=cdf_bins,histtype='step',\
                                 density=True,cumulative=True,label=epoch_name,color=colors[e_i])
                        ax_pdf[i_3].hist(corr_storage[combo_3][ep],bins=pdf_bins,histtype='step',\
                                 density=True,cumulative=False,label=epoch_name,color=colors[e_i])
                        ax_cdf_best[i_3].hist(best_corr_storage[combo_3][ep],\
                                bins=cdf_best_bins,histtype='step',\
                                    density=True,cumulative=True,label=epoch_name,color=colors[e_i])
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

def reorg_data_dict(corr_data, unique_corr_names, unique_given_names, unique_segment_names,
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
                    best_inds = corr_dev_stats[seg_name]['best']
                    #Now store the best correlation values by taste and epoch
                    for t_i, taste in enumerate(taste_names):
                        for cp_i in range(num_cp):
                            corr_list = []
                            dev_inds = np.where((best_inds[:,0] == t_i)*(best_inds[:,1] == cp_i))[0]
                            for dev_i in dev_inds:
                                corr_list.extend(data[taste]['data'][dev_i,:,cp_i])
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
            ax[0].hist(combined_animal_results[s_i],bins=min([1000,max_val]),histtype='step',\
                     density=True,cumulative=True,label=seg_name,color=colors[s_i])
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
                ax_e_seg[ep_i,s_i].hist(td,density=True,cumulative=True,bins=1000,
                                        histtype='step',label=unique_taste_names[td_i])
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
                ax_e_taste[ep_i,t_i].hist(sd,density=True,cumulative=True,bins=1000,
                                        histtype='step',label=unique_segment_names[sd_i])
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
                ax_seg_taste[s_i,t_i].hist(ed,density=True,cumulative=True,bins=1000,
                                        histtype='step',label=unique_epoch_pairs[ed_i])
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
                ax_e_seg[ep_i,s_i].hist(td,density=True,cumulative=True,bins=1000,
                                        histtype='step',label=unique_taste_names[td_i])
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
                ax_e_taste[ep_i,t_i].hist(sd,density=True,cumulative=True,bins=1000,
                                        histtype='step',label=unique_segment_names[sd_i])
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
                ax_seg_taste[s_i,t_i].hist(ed,density=True,cumulative=True,bins=1000,
                                        histtype='step',label=unique_epoch_pairs[ed_i])
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
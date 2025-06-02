#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:32:10 2025

@author: Hannah Germaine

A collection of functions for comparing different datasets against each other 
in their multi-day correlation and decoding data.
"""

import os
import warnings
import csv
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ks_2samp, ttest_ind
from functions.compare_conditions_funcs import int_input, bool_input, int_list_input

def compare_corr_data(corr_dict, null_corr_dict, multiday_data_dict, unique_given_names,
                      unique_corr_names, unique_segment_names, unique_taste_names, 
                      max_cp, save_dir):
    
    corr_results_save_dir = os.path.join(save_dir,'Correlations')
    if not os.path.isdir(corr_results_save_dir):
        os.mkdir(corr_results_save_dir)
        
    colors = ['red','orange','yellow','green','royalblue','purple', \
              'magenta','brown', 'cyan']
    corr_cutoffs = np.round(np.arange(0,1.01,0.01),2)   
        
    #Significance Test all taste pairs
    test_corr_dist_sig_test(corr_dict, null_corr_dict, unique_given_names, 
                             unique_corr_names, unique_segment_names, 
                             unique_taste_names, max_cp, corr_results_save_dir)
    
    #Calculate indices unique to each correlation combo
    all_corr_dicts, all_null_corr_dicts = calc_ind_dicts(corr_dict, null_corr_dict,
                                                unique_given_names, unique_corr_names, 
                                                unique_segment_names, unique_taste_names, 
                                                max_cp, corr_cutoffs)
    
    #Create plots
    plot_corr_cutoff_tastes(all_corr_dicts, all_null_corr_dicts, corr_dict, null_corr_dict,
                            multiday_data_dict, unique_given_names, unique_corr_names, 
                            unique_segment_names, unique_taste_names, 
                            max_cp, corr_cutoffs, colors, corr_results_save_dir)
    plot_corr_cutoff_epochs(all_corr_dicts, all_null_corr_dicts, corr_dict, null_corr_dict,
                            multiday_data_dict, unique_given_names, unique_corr_names, 
                            unique_segment_names, unique_taste_names, max_cp, 
                            corr_cutoffs, colors, corr_results_save_dir)
           
def test_corr_dist_sig_test(corr_dict, null_corr_dict, unique_given_names, 
                            unique_corr_names, unique_segment_names, 
                            unique_taste_names, max_cp, plot_save_dir, verbose=False):
    """
    This function tests pairs of deviation event x taste correlation 
    distributions against each other and against null data.

    Parameters
    ----------
    corr_dict : dictionary of deviation event correlation values for each animal.
    null_corr_dict : dictionary of null deviation event correlation values for each animal.
    unique_given_names : list of animal names
    unique_corr_names : list of correlation type names
    unique_segment_names : list of segment names
    unique_taste_names : list of taste names
    max_cp : maximum number of changepoints across datasets
    plot_save_dir : directory to save results
    verbose : boolean, optional, default = False. Provide error message if True.

    Returns
    -------
    None.
    
    Outputs
    -------
    .csv files of significance test results

    """
    taste_inds = np.arange(len(unique_taste_names))
    t_pairs = list(combinations(list(taste_inds),2))
    sig_save_dir = os.path.join(plot_save_dir,'Sig_Tests')
    if not os.path.isdir(sig_save_dir):
        os.mkdir(sig_save_dir)
    taste_comp_save_dir = os.path.join(sig_save_dir,'TastexTaste')
    if not os.path.isdir(taste_comp_save_dir):
        os.mkdir(taste_comp_save_dir)
    null_comp_save_dir = os.path.join(sig_save_dir,'TastexNull')
    if not os.path.isdir(null_comp_save_dir):
        os.mkdir(null_comp_save_dir)
    
    #Compare taste distributions against each other
    for corr_name in unique_corr_names:
        for seg_name in unique_segment_names:
            for cp_i in range(max_cp):
                #Check for previously saved results in csv
                csv_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + '_ks.csv'
                csv_1_exists = os.path.isfile(os.path.join(taste_comp_save_dir,csv_save_name))
                csv_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + '_tt.csv'
                csv_2_exists = os.path.isfile(os.path.join(taste_comp_save_dir,csv_save_name))
                if (csv_1_exists and csv_2_exists):
                    #re_run = bool_input("Re-run significance tests?")
                    re_run = 'n'
                else:
                    re_run = 'y'
                if re_run == 'y':
                    all_taste_dist = []
                    for t_i, taste in enumerate(unique_taste_names):
                        taste_dist = []
                        for g_n in unique_given_names:
                            try:
                                data = corr_dict[g_n][corr_name][seg_name]['all'][taste]['data']
                                data_cp, num_pts = np.shape(data)
                                num_dev = corr_dict[g_n][corr_name][seg_name]['all'][taste]['num_dev']
                                num_deliv = int(num_pts/num_dev)
                                data_reshape = np.reshape(data,(data_cp,num_deliv,num_dev))
                                deliv_means = np.squeeze(np.nanmean(data_reshape,1)) #cp x num_dev
                                if taste == 'none_0':
                                    cp_means = np.nanmean(deliv_means,0) #num_dev
                                    taste_dist.extend(list(cp_means))
                                else:
                                    taste_dist.extend(list(np.squeeze(deliv_means[cp_i,:])))
                            except:
                                errormsg = 'No data for ' + seg_name + ' Epoch ' + \
                                    str(cp_i) + ' ' + taste + ' animal ' + g_n
                                if verbose == True:
                                    print(errormsg)
                        all_taste_dist.append(taste_dist)
                    #Now calculate KS-2samp results
                    ks_results = []
                    tt_results = []
                    for tp in t_pairs:
                        tp_i1 = tp[0]
                        data_1 = all_taste_dist[tp_i1]
                        tp_i2 = tp[1]
                        data_2 = all_taste_dist[tp_i2]
                        if len(data_1) > 0:
                            if len(data_2) > 0:
                                ks_result = ks_2samp(data_1,data_2,alternative='two-sided')
                                if ks_result[1] <= 0.05:
                                    if np.nanmean(data_1) < np.nanmean(data_2):
                                        ks_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'*<'])
                                    else:
                                        ks_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'*>'])
                                else:
                                    ks_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'n.s.'])
                                tt_result = ttest_ind(data_1,data_2,alternative='two-sided')
                                if tt_result[1] <= 0.05:
                                    if np.nanmean(data_1) < np.nanmean(data_2):
                                        tt_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'*<'])
                                    else:
                                        tt_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'*>'])
                                else:
                                    tt_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'n.s.'])
                                
                    #Output results to csv
                    csv_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + '_ks.csv'
                    with open(os.path.join(taste_comp_save_dir,csv_save_name),'w',newline='') as file:
                        writer = csv.writer(file)
                        for row in ks_results:
                            writer.writerow(row)
                    csv_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + '_tt.csv'
                    with open(os.path.join(taste_comp_save_dir,csv_save_name),'w',newline='') as file:
                        writer = csv.writer(file)
                        for row in tt_results:
                            writer.writerow(row)
                        
    #Compare taste distributions against null
    for corr_name in unique_corr_names:
        for seg_name in unique_segment_names:
            for cp_i in range(max_cp):
                csv_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + '_null_ks.csv'
                csv_2_exists = os.path.isfile(os.path.join(null_comp_save_dir,csv_save_name))
                csv_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + '_null_tt.csv'
                csv_2_exists = os.path.isfile(os.path.join(null_comp_save_dir,csv_save_name))
                if (csv_1_exists and csv_2_exists):
                    #re_run = bool_input("Re-run significance tests?")
                    re_run = 'n'
                else:
                    re_run = 'y'
                if re_run == 'y':
                    ks_results = []
                    tt_results = []
                    for t_i, taste in enumerate(unique_taste_names):
                        taste_dist = []
                        null_taste_dist = []
                        for g_n in unique_given_names:
                            #Collect true data
                            try:
                                data = corr_dict[g_n][corr_name][seg_name]['all'][taste]['data']
                                data_cp, num_pts = np.shape(data)
                                num_dev = corr_dict[g_n][corr_name][seg_name]['all'][taste]['num_dev']
                                num_deliv = int(num_pts/num_dev)
                                data_reshape = np.reshape(data,(data_cp,num_deliv,num_dev))
                                deliv_means = np.squeeze(np.nanmean(data_reshape,1)) #cp x num_dev
                                if taste == 'none_0':
                                    cp_means = np.nanmean(deliv_means,0) #num_dev
                                    taste_dist.extend(list(cp_means))
                                else:
                                    taste_dist.extend(list(np.squeeze(deliv_means[cp_i,:])))
                            except:
                                errormsg = 'No data for ' + seg_name + ' Epoch ' + \
                                    str(cp_i) + ' ' + taste + ' animal ' + g_n
                                if verbose == True:
                                    print(errormsg)
                            #Collect null data
                            try:
                                null_data = null_corr_dict[g_n][seg_name][taste][corr_name] #already average across deliveries
                                if taste == 'none_0':
                                    null_cp_data = []
                                    for cp_i_null in range(max_cp):
                                        null_cp_data.extend(null_data[cp_i_null])
                                else:
                                    null_cp_data = null_data[cp_i]
                                null_taste_dist.extend(null_cp_data)
                            except:
                                errormsg = 'No null data for ' + seg_name + ' Epoch ' + \
                                    str(cp_i) + ' ' + taste + ' animal ' + g_n
                                if verbose == True:
                                    print(errormsg)
                        #Run significance tests
                        if len(taste_dist) > 0:
                            if len(null_taste_dist) > 0:
                                ks_result = ks_2samp(taste_dist,null_taste_dist,alternative='two-sided')
                                if ks_result[1] <= 0.05:
                                    if np.nanmean(taste_dist) < np.nanmean(null_taste_dist):
                                        ks_results.append([taste,'*<'])
                                    else:
                                        ks_results.append([taste,'*>'])
                                else:
                                    ks_results.append([taste,'n.s.'])
                                tt_result = ttest_ind(taste_dist,null_taste_dist,alternative='two-sided')
                                if tt_result[1] <= 0.05:
                                    if np.nanmean(taste_dist) < np.nanmean(null_taste_dist):
                                        tt_results.append([taste,'*<'])
                                    else:
                                        tt_results.append([taste,'*>'])
                                else:
                                    tt_results.append([taste,'n.s.'])
                                
                    #Output results to csv
                    csv_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + '_null_ks.csv'
                    with open(os.path.join(null_comp_save_dir,csv_save_name),'w',newline='') as file:
                        writer = csv.writer(file)
                        for row in ks_results:
                            writer.writerow(row)
                    csv_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + '_null_tt.csv'
                    with open(os.path.join(null_comp_save_dir,csv_save_name),'w',newline='') as file:
                        writer = csv.writer(file)
                        for row in tt_results:
                            writer.writerow(row)
        
def calc_ind_dicts(corr_dict, null_corr_dict, unique_given_names, 
                   unique_corr_names, unique_segment_names, unique_taste_names, 
                   max_cp, corr_cutoffs, verbose=False):
    """

    Parameters
    ----------
    corr_dict : dictionary of deviation event correlation values for each animal.
    null_corr_dict : dictionary of null deviation event correlation values for each animal.
    unique_given_names : list of animal names
    unique_corr_names : list of correlation type names
    unique_segment_names : list of segment names
    unique_taste_names : list of taste names
    max_cp : maximum number of changepoints across datasets
    corr_cutoffs : numpy array of correlation cutoff values from 0 to 1
    verbose : boolean, optional, default = False. Provide error message if True.

    Returns
    -------
    all_corr_dicts : dictionary of lists of deviation indices above each cutoff
    all_null_corr_dicts: dictionary of lists of null deviation indices above 
        each cutoff

    """
    
    #Collect all indices above a cutoff
    all_corr_dicts = dict()
    all_null_corr_dicts = dict()
    for corr_name in unique_corr_names:
        all_corr_dicts[corr_name] = dict()
        all_null_corr_dicts[corr_name] = dict()
        for s_i, seg_name in enumerate(unique_segment_names):
            all_corr_dicts[corr_name][seg_name] = dict()
            all_null_corr_dicts[corr_name][seg_name] = dict()
            for cp_i in range(max_cp):
                all_corr_dicts[corr_name][seg_name][cp_i] = dict()
                all_null_corr_dicts[corr_name][seg_name][cp_i] = dict()
                for t_i, taste in enumerate(unique_taste_names):
                    all_corr_dicts[corr_name][seg_name][cp_i][taste] = dict()
                    all_null_corr_dicts[corr_name][seg_name][cp_i][taste] = dict()
                    for gn_i, gn in enumerate(unique_given_names):
                        #Collect true data
                        try:
                            #Grab animal data
                            data = corr_dict[gn][corr_name][seg_name]['all'][taste]['data']
                            data_cp, num_pts = np.shape(data)
                            num_dev = corr_dict[gn][corr_name][seg_name]['all'][taste]['num_dev']
                            num_deliv = int(num_pts/num_dev)
                            data_reshape = np.reshape(data,(data_cp,num_deliv,num_dev))
                            deliv_means = np.squeeze(np.nanmean(data_reshape,1)) #num_cp x num_dev
                            #Collect indices by cutoff
                            if taste == 'none_0':
                                cp_means = np.nanmean(deliv_means,0)
                                all_corr_dicts[corr_name][seg_name][cp_i][taste][gn] = [np.where(cp_means >= cc)[0] for cc in corr_cutoffs]
                            else:
                                all_corr_dicts[corr_name][seg_name][cp_i][taste][gn] = [np.where(deliv_means[cp_i,:] >= cc)[0] for cc in corr_cutoffs]
                        except:
                            errormsg = 'Missing data for ' + corr_name + ' ' + \
                                seg_name + ' Epoch ' + str(cp_i) + ' ' + taste + \
                                    ' ' + gn
                            if verbose == True:
                                print(errormsg)
                        #Collect null data
                        try:
                            #Grab animal data
                            null_data = null_corr_dict[gn][seg_name][taste][corr_name] #already nanmean across deliveries
                            #Collect indices by cutoff
                            if taste == 'none_0':
                                null_cp_data = []
                                for cp_i_2 in range(max_cp):
                                    try:
                                        null_cp_data.extend(null_data[cp_i])
                                    except:
                                        null_cp_data.extend([])
                            else:
                                null_cp_data = null_data[cp_i]
                            all_null_corr_dicts[corr_name][seg_name][cp_i][taste][gn] = [np.where(null_cp_data >= cc)[0] for cc in corr_cutoffs]
                        except:
                            errormsg = 'Missing null data for ' + corr_name + ' ' + \
                                seg_name + ' Epoch ' + str(cp_i) + ' ' + taste + \
                                    ' ' + gn
                            if verbose == True:
                                print(errormsg)

    return all_corr_dicts, all_null_corr_dicts
        
def plot_corr_cutoff_tastes(all_corr_dicts, all_null_corr_dicts, corr_dict, 
                            null_corr_dict, multiday_data_dict, unique_given_names, 
                            unique_corr_names, unique_segment_names, unique_taste_names, 
                            max_cp, corr_cutoffs, colors, plot_save_dir, verbose = False):
    
    """Plot number of events above correlation cutoff for each taste"""
    
    warnings.filterwarnings('ignore')
    non_none_tastes = [taste for taste in unique_taste_names if taste != 'none_0']
    cc_0_25_ind = np.where(corr_cutoffs >= 0.25)[0][0]
    cc_top_third_ind = np.where(corr_cutoffs >= 1/3)[0][0]
    num_anim = len(unique_given_names)
    num_cutoff = len(corr_cutoffs)
    num_segs = len(unique_segment_names)
    num_tastes = len(non_none_tastes)
    indiv_anim_plot_save_dir = os.path.join(plot_save_dir,'Individual_Animals')
    if not os.path.isdir(indiv_anim_plot_save_dir):
        os.mkdir(indiv_anim_plot_save_dir)
    
    for corr_name in unique_corr_names:
        
        #Plot all corr values          
        all_dev_corr_inds = all_corr_dicts[corr_name]
        all_null_dev_corr_inds = all_null_corr_dicts[corr_name]
        f_rate, ax_rate = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        f_rate_zoom, ax_rate_zoom = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        f_rate_zoom_split_y, ax_rate_zoom_split_y = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = False)
        f_rate_box, ax_rate_box = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        for s_i, seg_name in enumerate(unique_segment_names):
            seg_indiv_dir = os.path.join(indiv_anim_plot_save_dir,seg_name)
            if not os.path.isdir(seg_indiv_dir):
                os.mkdir(seg_indiv_dir)
            for cp_i in range(max_cp):
                cp_indiv_dir = os.path.join(seg_indiv_dir,'Epoch_' + str(cp_i))
                if not os.path.isdir(cp_indiv_dir):
                    os.mkdir(cp_indiv_dir)
                taste_null_rates = []
                for t_i, taste in enumerate(non_none_tastes):
                    taste_indiv_dir = os.path.join(cp_indiv_dir,taste)
                    if not os.path.isdir(taste_indiv_dir):
                        os.mkdir(taste_indiv_dir)
                    animal_inds = all_dev_corr_inds[seg_name][cp_i][taste]
                    animal_null_inds = all_null_dev_corr_inds[seg_name][cp_i][taste]
                    animal_true_rates = np.zeros((num_anim,num_cutoff))
                    animal_null_rates = np.zeros((num_anim,num_cutoff))
                    for gn_i, g_n in enumerate(unique_given_names):
                        #Collect true data
                        animal_seg_times = multiday_data_dict[g_n]['segment_times']
                        animal_seg_names = multiday_data_dict[g_n]['segment_names']
                        a_s_ind = [i for i in range(len(animal_seg_names)) if animal_seg_names[i] == seg_name][0]
                        seg_len = (animal_seg_times[a_s_ind+1]-animal_seg_times[a_s_ind])*(1/1000) #seconds length
                        try:
                            num_inds = np.array([len(animal_inds[g_n][cc_i]) for cc_i in range(len(corr_cutoffs))])
                            animal_true_rates[gn_i,:] = num_inds/seg_len
                        except:
                            errormsg = g_n + ' does not have ' + seg_name + \
                                ' epoch ' + str(cp_i) + ' ' + taste + ' data.'
                            if verbose == True:
                                print(errormsg)
                        #Collect null data
                        try:
                            num_null_inds = np.array([len(animal_null_inds[g_n][cc_i]) for cc_i in range(len(corr_cutoffs))])
                            num_null_sets = null_corr_dict[g_n]['num_null']
                            animal_null_rates[gn_i,:] = num_null_inds/num_null_sets/seg_len
                        except:
                            errormsg = g_n + ' does not have ' + seg_name + \
                                ' epoch ' + str(cp_i) + ' ' + taste + ' data.'
                            if verbose == True:
                                print(errormsg)
                        #Plot individual animal rates in separate folder
                        f_anim = plt.figure()
                        plt.title(g_n)
                        plt.plot(corr_cutoffs, animal_true_rates[gn_i,:], label='True')
                        plt.plot(corr_cutoffs, animal_null_rates[gn_i,:], label='Null')
                        plt.ylabel('Rate (Hz)')
                        plt.xlabel('Pearson Correlation Cutoff')
                        plt.legend(loc='upper right')
                        f_anim.savefig(os.path.join(taste_indiv_dir,corr_name+'_'+g_n+'_rate_taste_by_cutoff.png'))
                        f_anim.savefig(os.path.join(taste_indiv_dir,corr_name+'_'+g_n+'_rate_taste_by_cutoff.svg'))
                        plt.close(f_anim)
                        f_anim_zoom = plt.figure()
                        plt.title(g_n)
                        plt.plot(corr_cutoffs[cc_0_25_ind:], animal_true_rates[gn_i,cc_0_25_ind:], label='True')
                        plt.plot(corr_cutoffs[cc_0_25_ind:], animal_null_rates[gn_i,cc_0_25_ind:], label='Null')
                        plt.ylabel('Rate (Hz)')
                        plt.xlabel('Pearson Correlation Cutoff')
                        plt.legend(loc='upper right')
                        f_anim_zoom.savefig(os.path.join(taste_indiv_dir,corr_name+'_'+g_n+'_rate_taste_by_cutoff_zoom.png'))
                        f_anim_zoom.savefig(os.path.join(taste_indiv_dir,corr_name+'_'+g_n+'_rate_taste_by_cutoff_zoom.svg'))
                        plt.close(f_anim_zoom)
                    #Average rates
                    anim_true_avg_rate = np.nanmean(animal_true_rates,0)
                    anim_null_avg_rate = np.nanmean(animal_null_rates,0)
                    taste_null_rates.append(list(anim_null_avg_rate))
                    ax_rate[s_i,cp_i].plot(corr_cutoffs,anim_true_avg_rate,label=taste)
                    ax_rate_zoom[s_i,cp_i].plot(corr_cutoffs[cc_top_third_ind:],anim_true_avg_rate[cc_top_third_ind:],label=taste)
                    ax_rate_zoom_split_y[s_i,cp_i].plot(corr_cutoffs[cc_top_third_ind:],anim_true_avg_rate[cc_top_third_ind:],label=taste)
                    #0.25 rates box plot
                    anim_25_rates = animal_true_rates[:,cc_0_25_ind].flatten()
                    ax_rate_box[s_i,cp_i].boxplot(list(anim_25_rates),
                                                  positions=[t_i+1],
                                                  showmeans=False,showfliers=False)
                    x_locs = t_i + 1 + 0.1*np.random.randn(num_anim)
                    ax_rate_box[s_i,cp_i].scatter(x_locs,anim_25_rates,alpha=0.5,color='g')
                    
                taste_null_avg_rate = np.nanmean(np.array(taste_null_rates),0)
                ax_rate[s_i,cp_i].plot(corr_cutoffs,taste_null_avg_rate,label='Null')
                ax_rate_zoom[s_i,cp_i].plot(corr_cutoffs[cc_top_third_ind:],taste_null_avg_rate[cc_top_third_ind:],label='Null')
                ax_rate_zoom_split_y[s_i,cp_i].plot(corr_cutoffs[cc_top_third_ind:],taste_null_avg_rate[cc_top_third_ind:],label='Null')
                if s_i == 0:
                    ax_rate[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                    ax_rate_zoom[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                    ax_rate_zoom_split_y[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                    ax_rate_box[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                if cp_i == 0:
                    ax_rate[s_i,cp_i].set_ylabel(seg_name + '\nRate (Hz)')
                    ax_rate_zoom[s_i,cp_i].set_ylabel(seg_name + '\nRate (Hz)')
                    ax_rate_zoom_split_y[s_i,cp_i].set_ylabel(seg_name + '\nRate (Hz)')
                    ax_rate_box[s_i,cp_i].set_ylabel(seg_name + '\nRate (Hz)')
                if s_i == num_segs-1:
                    ax_rate[s_i,cp_i].set_xlabel('Min. Correlation Cutoff')
                    ax_rate_zoom[s_i,cp_i].set_xlabel('Min. Correlation Cutoff')
                    ax_rate_zoom_split_y[s_i,cp_i].set_xlabel('Min. Correlation Cutoff')
                    ax_rate_box[s_i,cp_i].set_xlabel('Taste')
                    ax_rate_box[s_i,cp_i].set_xticks(np.arange(num_tastes)+1,non_none_tastes,
                                                     horizontalalignment='right',rotation = 45)
        plt.figure(f_rate)
        ax_rate[0,0].legend(loc='upper right')
        ax_rate[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Rate of Events Above Cutoff')
        plt.tight_layout()
        f_rate.savefig(os.path.join(plot_save_dir,corr_name+'_rate_taste_by_cutoff.png'))
        f_rate.savefig(os.path.join(plot_save_dir,corr_name+'_rate_taste_by_cutoff.svg'))
        plt.close(f_rate)
        plt.figure(f_rate_zoom)
        ax_rate_zoom[0,0].legend(loc='upper right')
        ax_rate_zoom[0,0].set_xticks(np.arange(0.25,1.25,0.25))
        plt.suptitle('Rate of Events Above Zoom Cutoff')
        plt.tight_layout()
        f_rate_zoom.savefig(os.path.join(plot_save_dir,corr_name+'_rate_taste_by_cutoff_zoom.png'))
        f_rate_zoom.savefig(os.path.join(plot_save_dir,corr_name+'_rate_taste_by_cutoff_zoom.svg'))
        plt.close(f_rate_zoom)
        plt.figure(f_rate_zoom_split_y)
        ax_rate_zoom_split_y[0,0].legend(loc='upper right')
        ax_rate_zoom_split_y[0,0].set_xticks(np.arange(0.25,1.25,0.25))
        plt.suptitle('Rate of Events Above Zoom Cutoff')
        plt.tight_layout()
        f_rate_zoom_split_y.savefig(os.path.join(plot_save_dir,corr_name+'_rate_taste_by_cutoff_split_y_zoom.png'))
        f_rate_zoom_split_y.savefig(os.path.join(plot_save_dir,corr_name+'_rate_taste_by_cutoff_split_y_zoom.svg'))
        plt.close(f_rate_zoom_split_y)
        plt.figure(f_rate_box)
        plt.suptitle('Rate of Events Above 0.25 Cutoff')
        plt.tight_layout()
        f_rate_box.savefig(os.path.join(plot_save_dir,corr_name+'_rate_taste_at_cutoff_box.png'))
        f_rate_box.savefig(os.path.join(plot_save_dir,corr_name+'_rate_taste_at_cutoff_box.svg'))
        plt.close(f_rate_box)
        
def plot_corr_cutoff_epochs(all_corr_dicts, all_null_corr_dicts, corr_dict, null_corr_dict,
                            multiday_data_dict, unique_given_names, unique_corr_names,
                            unique_segment_names, unique_taste_names, max_cp, 
                            corr_cutoffs, colors, plot_save_dir, verbose = False):
    
    """Plot number of events above correlation cutoff for each taste that are 
    unique to the taste and unique to a given epoch."""
    warnings.filterwarnings('ignore') 
    non_none_tastes = [taste for taste in unique_taste_names if taste != 'none_0']
    cc_0_25_ind = np.where(corr_cutoffs >= 0.25)[0][0]
    cc_top_third_ind = np.where(corr_cutoffs >= 1/3)[0][0]
    num_anim = len(unique_given_names)
    num_cutoff = len(corr_cutoffs)
    num_segs = len(unique_segment_names)
    num_tastes = len(non_none_tastes)
    indiv_anim_plot_save_dir = os.path.join(plot_save_dir,'Individual_Animals')
    if not os.path.isdir(indiv_anim_plot_save_dir):
        os.mkdir(indiv_anim_plot_save_dir)
    
    for corr_name in unique_corr_names:
        #Plot all corr values
        all_dev_corr_inds = all_corr_dicts[corr_name]
        all_null_dev_corr_inds = all_null_corr_dicts[corr_name]
        f_rate, ax_rate = plt.subplots(nrows = num_segs,\
                                       ncols = num_tastes, \
                                    figsize=(4*num_tastes,4*num_segs),\
                                    sharex = True, sharey = True)
        f_rate_zoom, ax_rate_zoom = plt.subplots(nrows = num_segs,\
                                       ncols = num_tastes,\
                                    figsize=(4*num_tastes,4*num_segs),\
                                    sharex = True, sharey = True)
        f_rate_zoom_split_y, ax_rate_zoom_split_y = plt.subplots(nrows = num_segs,\
                                       ncols = num_tastes,\
                                    figsize=(4*num_tastes,4*num_segs),\
                                    sharex = True, sharey = False)
        f_rate_box, ax_rate_box = plt.subplots(nrows = num_segs,\
                                       ncols = num_tastes,\
                                    figsize=(4*num_tastes,4*num_segs),\
                                    sharex = True, sharey = True)
        for s_i, seg_name in enumerate(unique_segment_names):
            seg_indiv_dir = os.path.join(indiv_anim_plot_save_dir,seg_name)
            if not os.path.isdir(seg_indiv_dir):
                os.mkdir(seg_indiv_dir)
            for t_i, taste in enumerate(non_none_tastes):
                taste_indiv_dir = os.path.join(seg_indiv_dir,taste)
                if not os.path.isdir(taste_indiv_dir):
                    os.mkdir(taste_indiv_dir)
                epoch_null_rates = []
                for cp_i in range(max_cp):
                    cp_indiv_dir = os.path.join(taste_indiv_dir,'Epoch_' + str(cp_i))
                    if not os.path.isdir(cp_indiv_dir):
                        os.mkdir(cp_indiv_dir)
                    animal_inds = all_dev_corr_inds[seg_name][cp_i][taste]
                    animal_null_inds = all_null_dev_corr_inds[seg_name][cp_i][taste]
                    animal_true_rates = np.zeros((num_anim,num_cutoff))
                    animal_null_rates = np.zeros((num_anim,num_cutoff))
                    for gn_i, g_n in enumerate(unique_given_names):
                        #Collect true data
                        animal_seg_times = multiday_data_dict[g_n]['segment_times']
                        animal_seg_names = multiday_data_dict[g_n]['segment_names']
                        a_s_ind = [i for i in range(len(animal_seg_names)) if animal_seg_names[i] == seg_name][0]
                        seg_len = (animal_seg_times[a_s_ind+1]-animal_seg_times[a_s_ind])*(1/1000) #seconds length
                        try:
                            num_inds = np.array([len(animal_inds[g_n][cc_i]) for cc_i in range(len(corr_cutoffs))])
                            animal_true_rates[gn_i,:] = num_inds/seg_len
                        except:
                            errormsg = g_n + ' does not have ' + seg_name + \
                                ' epoch ' + str(cp_i) + ' ' + taste + ' data.'
                            if verbose == True:
                                print(errormsg) 
                        #Collect null data
                        try:
                            num_null_inds = np.array([len(animal_null_inds[g_n][cc_i]) for cc_i in range(len(corr_cutoffs))])
                            num_null_sets = null_corr_dict[g_n]['num_null']
                            animal_null_rates[gn_i,:] = num_null_inds/num_null_sets/seg_len
                        except:
                            errormsg = g_n + ' does not have ' + seg_name + \
                                ' epoch ' + str(cp_i) + ' ' + taste + ' data.'
                            if verbose == True:
                                print(errormsg)
                    #Average rates
                    anim_true_avg_rate = np.nanmean(animal_true_rates,0)
                    anim_null_avg_rate = np.nanmean(animal_null_rates,0)
                    epoch_null_rates.append(list(anim_null_avg_rate))
                    ax_rate[s_i,t_i].plot(corr_cutoffs,anim_true_avg_rate,label='Epoch ' + str(cp_i))
                    ax_rate_zoom[s_i,t_i].plot(corr_cutoffs[cc_top_third_ind:],anim_true_avg_rate[cc_top_third_ind:],label='Epoch ' + str(cp_i))
                    ax_rate_zoom_split_y[s_i,t_i].plot(corr_cutoffs[cc_top_third_ind:],anim_true_avg_rate[cc_top_third_ind:],label='Epoch ' + str(cp_i))
                    #0.25 rates box plot
                    anim_25_rates = animal_true_rates[:,cc_0_25_ind].flatten()
                    ax_rate_box[s_i,t_i].boxplot(list(anim_25_rates),
                                                  positions=[cp_i+1],
                                                  showmeans=False,showfliers=False)
                    x_locs = cp_i + 1 + 0.1*np.random.randn(num_anim)
                    ax_rate_box[s_i,t_i].scatter(x_locs,anim_25_rates,alpha=0.5,color='g')
                epoch_null_avg_rate = np.nanmean(np.array(epoch_null_rates),0)
                ax_rate[s_i,t_i].plot(corr_cutoffs,epoch_null_avg_rate,label='Null')
                ax_rate_zoom[s_i,t_i].plot(corr_cutoffs[cc_top_third_ind:],epoch_null_avg_rate[cc_top_third_ind:],label='Null')
                ax_rate_zoom_split_y[s_i,t_i].plot(corr_cutoffs[cc_top_third_ind:],epoch_null_avg_rate[cc_top_third_ind:],label='Null')
                if s_i == 0:
                    ax_rate[s_i, t_i].set_title(taste)
                    ax_rate_zoom[s_i, t_i].set_title(taste)
                    ax_rate_zoom_split_y[s_i, t_i].set_title(taste)
                    ax_rate_box[s_i, t_i].set_title(taste)
                if s_i == len(unique_segment_names)-1:
                    ax_rate[s_i, t_i].set_xlabel('Correlation Cutoff')
                    ax_rate_zoom[s_i, t_i].set_xlabel('Correlation Cutoff')
                    ax_rate_zoom_split_y[s_i, t_i].set_xlabel('Correlation Cutoff')
                    ax_rate_box[s_i, t_i].set_xlabel('State')
                    ax_rate_box[s_i,t_i].set_xticks(np.arange(max_cp)+1)
                if t_i == 0:
                    ax_rate[s_i, t_i].set_ylabel(seg_name + '\Rate (Hz)')
                    ax_rate_zoom[s_i, t_i].set_ylabel(seg_name + '\Rate (Hz)')
                    ax_rate_zoom_split_y[s_i, t_i].set_ylabel(seg_name + '\Rate (Hz)')
                    ax_rate_box[s_i, t_i].set_ylabel(seg_name + '\Rate (Hz)')
        plt.figure(f_rate)
        ax_rate[0,0].legend(loc='upper right')
        ax_rate[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Rate of Events Above Cutoff')
        plt.tight_layout()
        f_rate.savefig(os.path.join(plot_save_dir,corr_name+'_rate_epoch_by_cutoff.png'))
        f_rate.savefig(os.path.join(plot_save_dir,corr_name+'_rate_epoch_by_cutoff.svg'))
        plt.close(f_rate)
        plt.figure(f_rate_zoom)
        ax_rate_zoom[0,0].legend(loc='upper right')
        ax_rate_zoom[0,0].set_xticks(np.arange(0.25,1.25,0.25))
        plt.suptitle('Rate of Events Above Cutoff')
        plt.tight_layout()
        f_rate_zoom.savefig(os.path.join(plot_save_dir,corr_name+'_rate_epoch_by_cutoff_zoom.png'))
        f_rate_zoom.savefig(os.path.join(plot_save_dir,corr_name+'_rate_epoch_by_cutoff_zoom.svg'))
        plt.close(f_rate_zoom)
        plt.figure(f_rate_zoom_split_y)
        ax_rate_zoom_split_y[0,0].legend(loc='upper right')
        ax_rate_zoom_split_y[0,0].set_xticks(np.arange(0.25,1.25,0.25))
        plt.suptitle('Rate of Events Above Cutoff')
        plt.tight_layout()
        f_rate_zoom_split_y.savefig(os.path.join(plot_save_dir,corr_name+'_rate_epoch_by_cutoff_split_y_zoom.png'))
        f_rate_zoom_split_y.savefig(os.path.join(plot_save_dir,corr_name+'_rate_epoch_by_cutoff_split_y_zoom.svg'))
        plt.close(f_rate_zoom_split_y)
        plt.figure(f_rate_box)
        plt.suptitle('Rate of Events Above 0.25 Cutoff')
        plt.tight_layout()
        f_rate_box.savefig(os.path.join(plot_save_dir,corr_name+'_rate_epoch_at_cutoff_box.png'))
        f_rate_box.savefig(os.path.join(plot_save_dir,corr_name+'_rate_epoch_at_cutoff_box.svg'))
        plt.close(f_rate_box)
        
        
def compare_decode_data(decode_dict, multiday_data_dict, unique_given_names,
                       unique_decode_names, unique_segment_names, 
                       unique_taste_names, max_cp, save_dir, verbose):
    
    decode_results_save_dir = os.path.join(save_dir,'Decodes')
    if not os.path.isdir(decode_results_save_dir):
        os.mkdir(decode_results_save_dir)
        
    #Plot cross-animal rates of decodes
    decode_rates_plots(decode_dict,unique_given_names,unique_decode_names,
                           unique_segment_names,unique_taste_names,
                           max_cp,decode_results_save_dir,verbose)
    
    
def decode_rates_plots(decode_dict,unique_given_names,unique_decode_names,
                       unique_segment_names,unique_taste_names,
                       max_cp,decode_results_save_dir,verbose=False):
    
    colors = ['red','orange','yellow','green','royalblue','purple', \
              'magenta','brown', 'cyan']
    num_anim = len(unique_given_names)
    num_seg = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    unique_segment_names = ['pre-taste','post-taste','sickness'] #manual order override
    
    for dt in unique_decode_names:
        
        #Is-Taste Decode Results
        is_taste_rates = [] #num seg x num anim
        f_box = plt.figure(figsize = (5,5))
        for s_i, sn in enumerate(unique_segment_names):
            seg_is_taste_rates = []
            for gn in unique_given_names:
                try:
                    is_taste_data = decode_dict[gn][dt][sn]['is_taste'] #num_dev x 2
                    num_dev, _ = np.shape(is_taste_data)
                    is_taste_max = np.argmax(is_taste_data,1)
                    num_is_taste = len(np.where(np.array(is_taste_max) == 0)[0])
                    seg_is_taste_rates.append(num_is_taste/num_dev)
                except:
                    seg_is_taste_rates.append(np.nan)
                    errormsg = "Is-Taste data does not exist for " + gn + " " + sn
                    if verbose == True:
                        print(errormsg)
            is_taste_rates.append(seg_is_taste_rates)
            plt.boxplot(seg_is_taste_rates,positions = [s_i])
            x_scat = s_i + np.random.rand(num_anim)/10
            plt.scatter(x_scat,seg_is_taste_rates,color='g',alpha=0.3)
        plt.xticks(np.arange(num_seg),unique_segment_names)
        plt.xlabel('Segment')
        #plt.ylim([0,1])
        plt.ylabel('Fraction of Deviation Events')
        plt.title('Fraction of Events Decoded as Taste')
        plt.tight_layout()
        f_box.savefig(os.path.join(decode_results_save_dir,dt + '_is_taste_rates_box.png'))
        f_box.savefig(os.path.join(decode_results_save_dir,dt + '_is_taste_rates_box.svg'))
        plt.close(f_box)
        
        f_line = plt.figure(figsize = (5,5))
        mean_fraction = np.nanmean(np.array(is_taste_rates),1)
        std_fraction = np.nanstd(np.array(is_taste_rates),1)
        plt.fill_between(np.arange(num_seg),mean_fraction-std_fraction,\
                         mean_fraction+std_fraction,color='k',alpha=0.2)
        plt.plot(mean_fraction,color='k')
        plt.xticks(np.arange(num_seg),unique_segment_names)
        plt.xlabel('Segment')
        #plt.ylim([0,1])
        plt.ylabel('Fraction of Deviation Events')
        plt.title('Fraction of Events Decoded as Taste')
        plt.tight_layout()
        f_line.savefig(os.path.join(decode_results_save_dir,dt + '_is_taste_rates_line.png'))
        f_line.savefig(os.path.join(decode_results_save_dir,dt + '_is_taste_rates_line.svg'))
        plt.close(f_line)
        
        #Which-Taste Decode Results
        sqrt_taste = np.ceil(np.sqrt(num_tastes)).astype('int')
        sqr_taste = sqrt_taste**2
        not_plot_box = np.setdiff1d(np.arange(sqr_taste),np.arange(num_tastes))
        taste_ind_ref = np.reshape(np.arange(sqrt_taste**2),(sqrt_taste,sqrt_taste))
        epoch_x_labels = np.reshape(np.array([np.arange(max_cp) for s_i in range(num_seg)]),(max_cp*num_seg))
        f_box, ax_box = plt.subplots(nrows = sqrt_taste, ncols = sqrt_taste,\
                                     figsize = (8,8), sharex = True, sharey = True)
        f_line, ax_line = plt.subplots(nrows = sqrt_taste, ncols = sqrt_taste,\
                                     figsize = (8,8), sharex = True, sharey = True)
        f_box_epoch, ax_box_epoch = plt.subplots(nrows = sqrt_taste, ncols = sqrt_taste,\
                                     figsize = (8,8), sharex = True)
        which_taste_rates = dict()
        for t_i, tn in enumerate(unique_taste_names):
            t_plot_ind = np.where(taste_ind_ref == t_i)
            ax_r = t_plot_ind[0][0]
            ax_c = t_plot_ind[1][0]
            which_taste_rates[tn] = dict()
            seg_data = []
            for s_i, sn in enumerate(unique_segment_names):
                which_taste_rates[tn][sn] = dict()
                which_taste_rates[tn][sn]['which_taste'] = []
                for cp_i in range(max_cp):
                    which_taste_rates[tn][sn][cp_i] = []
                for gn in unique_given_names:
                    try:
                        gn_tastes = decode_dict[gn][dt]['tastes']
                        gn_t_i = [i for i in range(len(gn_tastes)) if gn_tastes[i] == tn]
                        if len(gn_t_i) > 0:
                            #Which Taste Data
                            try:
                                which_taste_data = decode_dict[gn][dt][sn]['which_taste'][:,gn_t_i] #num_dev
                                num_dev = len(which_taste_data)
                                gn_dev_inds = np.where(np.array(which_taste_data) == 1)[0]
                                num_which_taste = len(gn_dev_inds)
                                which_taste_rates[tn][sn]['which_taste'].append(num_which_taste/num_dev)
                                try:
                                    #Which Epoch Data
                                    which_epoch_data = decode_dict[gn][dt][sn]['which_epoch'][gn_dev_inds,:] #num_gn_dev x num_cp
                                    which_epoch_counts = np.nansum(np.array(which_epoch_data),0)
                                    for cp_i in range(len(which_epoch_counts)):
                                        which_taste_rates[tn][sn][cp_i].append(which_epoch_counts[cp_i]/num_dev)
                                except:
                                    for cp_i in range(max_cp):
                                        which_taste_rates[tn][sn][cp_i].append(np.nan)
                                    errormsg = "Which-Epoch data does not exist for " + gn + " " + sn + " " + tn
                                    if verbose == True:
                                        print(errormsg)
                            except:
                                which_taste_rates[tn][sn]['which_taste'].append(np.nan)
                                errormsg = "Which-Taste data does not exist for " + gn + " " + sn + " " + tn
                                if verbose == True:
                                    print(errormsg)
                    except:
                        which_taste_rates[tn][sn]['which_taste'].append(np.nan)
                        for cp_i in range(max_cp):
                            which_taste_rates[tn][sn][cp_i].append(np.nan)
                        errormsg = "Missing data for "+ gn + " " + dt
                        if verbose == True:
                            print(errormsg)
                seg_data.append(which_taste_rates[tn][sn]['which_taste'])
                #Plot Which-Taste Data
                t_data = which_taste_rates[tn][sn]['which_taste']
                non_nan_data = np.array(t_data)
                non_nan_data = non_nan_data[~np.isnan(non_nan_data)]
                ax_box[ax_r,ax_c].boxplot(non_nan_data,positions=[s_i])
                x_scat = s_i + np.random.rand(len(t_data))/10
                ax_box[ax_r,ax_c].scatter(x_scat,t_data,color='g',alpha=0.3)
                #Plot Which-Epoch Data
                max_y = 0
                for cp_i in range(max_cp):
                    cp_data = which_taste_rates[tn][sn][cp_i]
                    non_nan_data = np.array(cp_data)
                    non_nan_data = non_nan_data[~np.isnan(non_nan_data)]
                    if len(non_nan_data) > 0:
                        if max(non_nan_data) > max_y:
                            max_y = max(non_nan_data)
                        data_x = s_i*max_cp + cp_i
                        ax_box_epoch[ax_r,ax_c].boxplot(non_nan_data,positions=[data_x])
                        x_scat = data_x + np.random.rand(len(cp_data))/10
                        ax_box_epoch[ax_r,ax_c].scatter(x_scat,cp_data,\
                                                  color='g',alpha=0.3)
                ax_box_epoch[ax_r,ax_c].plot(np.arange(s_i*max_cp,s_i*max_cp+max_cp),\
                                             (max_y+0.1*max_y)*np.ones(max_cp),label=sn,
                                             color = colors[s_i])
            ax_box[ax_r,ax_c].set_xticks(np.arange(num_seg),unique_segment_names,
                                         rotation=45)
            ax_box[ax_r,ax_c].set_title(tn)
            ax_box_epoch[ax_r,ax_c].set_xticks(np.arange(num_seg*max_cp),epoch_x_labels)
            ax_box_epoch[ax_r,ax_c].set_title(tn)
            mean_data = np.nanmean(np.array(seg_data),1)
            std_data = np.nanstd(np.array(seg_data),1)
            ax_line[ax_r,ax_c].plot(np.arange(num_seg),mean_data,color='k')
            ax_line[ax_r,ax_c].fill_between(np.arange(num_seg),mean_data+std_data,\
                                            mean_data-std_data,color='k',alpha=0.2)
            ax_line[ax_r,ax_c].set_xticks(np.arange(num_seg),unique_segment_names,
                                          rotation=45)
            ax_line[ax_r,ax_c].set_title(tn)
        ax_box_epoch[0,0].legend()
        for ax_i in range(sqrt_taste):
            ax_box[sqrt_taste-1,ax_i].set_xlabel('Segment')
            ax_line[sqrt_taste-1,ax_i].set_xlabel('Segment')
            ax_box_epoch[sqrt_taste-1,ax_i].set_xlabel('Segment')
            ax_box[ax_i,0].set_ylabel('Fraction of Events')
            ax_line[ax_i,0].set_ylabel('Fraction of Events')
            ax_box_epoch[ax_i,0].set_ylabel('Fraction of Events')
        for ax_i in not_plot_box:
            ax_r, ax_c = np.where(taste_ind_ref == ax_i)
            ax_box[ax_r[0],ax_c[0]].axis('off')
            ax_box_epoch[ax_r[0],ax_c[0]].axis('off')
            ax_line[ax_r[0],ax_c[0]].axis('off')
        plt.figure(f_box)
        plt.suptitle(dt + ' Taste Decode Rates')
        f_box.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_rates_box.png'))
        f_box.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_rates_box.svg'))
        plt.close(f_box)
        plt.figure(f_line)
        plt.suptitle(dt + ' Taste Decode Rates')
        f_line.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_rates_line.png'))
        f_line.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_rates_line.svg'))
        plt.close(f_line)
        plt.figure(f_box_epoch)
        plt.suptitle(dt + ' Epoch Decode Rates')
        f_box_epoch.savefig(os.path.join(decode_results_save_dir,dt + '_which_epoch_rates_box.png'))
        f_box_epoch.savefig(os.path.join(decode_results_save_dir,dt + '_which_epoch_rates_box.svg'))
        plt.close(f_box_epoch)
        
        #Create pie charts of the fraction of events decoded as each taste by epoch
        f_pie, ax_pie = plt.subplots(nrows = 1, ncols = num_seg,\
                                     figsize=(15,5))
        overall_taste_counts = np.zeros(len(unique_taste_names))
        for s_i, sn in enumerate(unique_segment_names):
            taste_counts = np.zeros(len(unique_taste_names))
            for t_i, tn in enumerate(unique_taste_names):
                for gn in unique_given_names:
                    try:
                        gn_tastes = decode_dict[gn][dt]['tastes']
                        gn_t_i = [i for i in range(len(gn_tastes)) if gn_tastes[i] == tn]
                        if len(gn_t_i) > 0:
                            #Which Taste Data
                            try:
                                which_taste_data = decode_dict[gn][dt][sn]['which_taste'][:,gn_t_i] #num_dev
                                num_which_taste = np.nansum(which_taste_data)
                                taste_counts[t_i] += num_which_taste
                            except:
                                errormsg = "Missing data for "+ gn + " " + dt
                    except:
                        errormsg = "Missing data for "+ gn + " " + dt
                        if verbose == True:
                            print(errormsg)
            pie_labels = [unique_taste_names[t_i] + ' ' + \
                                      str(np.round(100*taste_counts[t_i]/np.nansum(taste_counts),2)) \
                                          for t_i in range(len(unique_taste_names))]
            ax_pie[s_i].pie(taste_counts,labels=pie_labels,labeldistance=1.1)
            overall_taste_counts += taste_counts
            ax_pie[s_i].set_title(sn)
        plt.suptitle('Percent Decoded Taste by Segment')
        f_pie.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_by_seg_pie.png'))
        f_pie.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_by_seg_pie.svg'))
        plt.close(f_pie)
        
        f = plt.figure(figsize = (5,5))
        overall_pie_labels = [unique_taste_names[t_i] + ' ' + \
                                  str(np.round(100*overall_taste_counts[t_i]/np.nansum(overall_taste_counts),2)) \
                                      for t_i in range(len(unique_taste_names))]
        plt.pie(overall_taste_counts,labels=overall_pie_labels,labeldistance=1.1)
        plt.suptitle('Percent Decoded Taste Overall')
        plt.tight_layout()
        f.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_oveall_pie.png'))
        f.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_oveall_pie.svg'))
        plt.close(f)
        
def select_analysis_groups(unique_list):
    """
    This function allows the user to select which aspects of the data to use 
    in the analysis.
    INPUTS:
        - unique_list: list of unique aspects available
    RETURNS:
        - unique_list: list of unique aspects to use
    NOTE:
        - This function will continue to prompt the user for an answer until the 
		answer given is a list of integers.
    """
    
    unique_prompt = ''
    for un_i, un in enumerate(unique_list):
        unique_prompt += str(un_i) + ': ' + un + '\n'
    unique_prompt += 'Please provide a comma-separate list of indices to ' + \
        'use in this analysis: '
    ind_to_keep = int_list_input(unique_prompt)
    unique_list = [unique_list[i] for i in ind_to_keep]
    
    return unique_list
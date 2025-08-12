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
from scipy.stats import ks_2samp, ttest_ind, anderson, f_oneway
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
                    #Plot
                    f_test, ax_test = plt.subplots(ncols=len(t_pairs),figsize=(3*len(t_pairs),5))
                    ks_results = []
                    tt_results = []
                    for tp_i, tp in enumerate(t_pairs):
                        tp_i1 = tp[0]
                        data_1 = all_taste_dist[tp_i1]
                        tp_i2 = tp[1]
                        data_2 = all_taste_dist[tp_i2]
                        if len(data_1) > 0:
                            if len(data_2) > 0:
                                ks_result = ks_2samp(data_1,data_2,alternative='two-sided')
                                if ks_result[1] <= 0.05:
                                    if np.nanmean(data_1) < np.nanmean(data_2):
                                        ks_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'*<',ks_result[1]])
                                    else:
                                        ks_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'*>',ks_result[1]])
                                else:
                                    ks_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'n.s.',ks_result[1]])
                                tt_result = ttest_ind(data_1,data_2,alternative='two-sided')
                                if tt_result[1] <= 0.05:
                                    if np.nanmean(data_1) < np.nanmean(data_2):
                                        tt_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'*<',tt_result[1]])
                                    else:
                                        tt_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'*>',tt_result[1]])
                                else:
                                    tt_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'n.s.',tt_result[1]])
                                ax_test[tp_i].hist(data_1,bins=100,density=True,cumulative=True,\
                                         label=unique_taste_names[tp_i1],alpha=0.5,\
                                             histtype='step')
                                ax_test[tp_i].hist(data_2,bins=100,density=True,cumulative=True,\
                                         label=unique_taste_names[tp_i2],alpha=0.5,\
                                             histtype='step')
                                ax_test[tp_i].set_xlabel('Pearson Correlation')
                                ax_test[tp_i].set_ylabel('Cumulative Density')
                                ax_test[tp_i].legend(loc='upper right')
                                ax_test[tp_i].set_title = unique_taste_names[tp_i1] + ' x ' + unique_taste_names[tp_i2]
                    plt.tight_layout()
                    fig_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i)
                    f_test.savefig(os.path.join(taste_comp_save_dir,fig_save_name  + '.png'))
                    f_test.savefig(os.path.join(taste_comp_save_dir,fig_save_name  + '.svg'))
                    plt.close(f_test)
                    
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
                    #Plot
                    f_test, ax_test = plt.subplots(ncols=len(unique_taste_names),figsize=(3*len(t_pairs),5))
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
                                taste_dist.extend(list(np.squeeze(deliv_means[cp_i,:])))
                            except:
                                errormsg = 'No data for ' + seg_name + ' Epoch ' + \
                                    str(cp_i) + ' ' + taste + ' animal ' + g_n
                                if verbose == True:
                                    print(errormsg)
                            #Collect null data
                            num_null = null_corr_dict[g_n]['num_null']
                            try:
                                null_data = null_corr_dict[g_n][seg_name][taste][corr_name][cp_i] #already average across deliveries but list for all null
                                null_cp_data = []
                                for null_i in range(num_null):
                                    null_cp_data.extend(null_data[null_i])
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
                                        ks_results.append([taste,'*<',ks_result[1]])
                                    else:
                                        ks_results.append([taste,'*>',ks_result[1]])
                                else:
                                    ks_results.append([taste,'n.s.',ks_result[1]])
                                tt_result = ttest_ind(taste_dist,null_taste_dist,alternative='two-sided')
                                if tt_result[1] <= 0.05:
                                    if np.nanmean(taste_dist) < np.nanmean(null_taste_dist):
                                        tt_results.append([taste,'*<',tt_result[1]])
                                    else:
                                        tt_results.append([taste,'*>',tt_result[1]])
                                else:
                                    tt_results.append([taste,'n.s.',tt_result[1]])
                                ax_test[t_i].hist(taste_dist,bins=100,density=True,cumulative=True,\
                                         label=unique_taste_names[t_i],alpha=0.5,\
                                             histtype='step')
                                ax_test[t_i].hist(null_taste_dist,bins=100,density=True,cumulative=True,\
                                         label='Null',alpha=0.5,\
                                             histtype='step')
                                ax_test[t_i].set_xlabel('Pearson Correlation')
                                ax_test[t_i].set_ylabel('Cumulative Density')
                                ax_test[t_i].legend(loc='upper right')
                                ax_test[t_i].set_title = unique_taste_names[t_i] + ' x null'
                    plt.tight_layout()
                    fig_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i)
                    f_test.savefig(os.path.join(null_comp_save_dir,fig_save_name  + '.png'))
                    f_test.savefig(os.path.join(null_comp_save_dir,fig_save_name  + '.svg'))
                    plt.close(f_test)
                    
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
                        num_null = null_corr_dict[gn]['num_null']
                        try:
                            #Grab animal data
                            null_data = null_corr_dict[gn][seg_name][taste][corr_name][cp_i] #list of mean correlation across deliveries for all null
                            #Collect indices by cutoff
                            null_cp_data = []
                            for null_i in range(num_null):
                                try:
                                    null_cp_data.extend(null_data[null_i])
                                except:
                                    null_cp_data.extend([])
                            null_cp_data = np.array(null_cp_data)
                            all_null_corr_dicts[corr_name][seg_name][cp_i][taste][gn] = [len(np.where(null_cp_data >= cc)[0])/num_null for cc in corr_cutoffs]
                        except:
                            errormsg = 'Missing null data for ' + corr_name + ' ' + \
                                seg_name + ' Epoch ' + str(cp_i) + ' ' + taste + \
                                    ' ' + gn
                            if verbose == True:
                                print(errormsg)

    return all_corr_dicts, all_null_corr_dicts

def plot_corr_cutoff_tastes(all_corr_dicts, corr_dict, multiday_data_dict, unique_given_names, 
                            unique_corr_names, unique_segment_names, unique_taste_names, 
                            max_cp, corr_cutoffs, colors, plot_save_dir, verbose = False):
    
    """Plot number of events above correlation cutoff for each taste"""
    
    warnings.filterwarnings('ignore')
    #non_none_tastes = [taste for taste in unique_taste_names if taste != 'none_0']
    cc_0_25_ind = np.where(corr_cutoffs >= 0.25)[0][0]
    cc_top_third_ind = np.where(corr_cutoffs >= 1/3)[0][0]
    num_anim = len(unique_given_names)
    num_cutoff = len(corr_cutoffs)
    num_segs = len(unique_segment_names)
    #num_tastes = len(non_none_tastes)
    num_tastes = len(unique_taste_names)
    indiv_anim_plot_save_dir = os.path.join(plot_save_dir,'Individual_Animals')
    if not os.path.isdir(indiv_anim_plot_save_dir):
        os.mkdir(indiv_anim_plot_save_dir)
    
    for corr_name in unique_corr_names:
        
        #Plot all corr values          
        all_dev_corr_inds = all_corr_dicts[corr_name]
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
                for t_i, taste in enumerate(unique_taste_names):
                    taste_indiv_dir = os.path.join(cp_indiv_dir,taste)
                    if not os.path.isdir(taste_indiv_dir):
                        os.mkdir(taste_indiv_dir)
                    animal_inds = all_dev_corr_inds[seg_name][cp_i][taste]
                    animal_true_rates = np.zeros((num_anim,num_cutoff))
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
                        #Plot individual animal rates in separate folder
                        f_anim = plt.figure()
                        plt.title(g_n)
                        plt.plot(corr_cutoffs, animal_true_rates[gn_i,:], label='True')
                        plt.ylabel('Rate (Hz)')
                        plt.xlabel('Pearson Correlation Cutoff')
                        plt.legend(loc='upper right')
                        f_anim.savefig(os.path.join(taste_indiv_dir,corr_name+'_'+g_n+'_rate_taste_by_cutoff.png'))
                        f_anim.savefig(os.path.join(taste_indiv_dir,corr_name+'_'+g_n+'_rate_taste_by_cutoff.svg'))
                        plt.close(f_anim)
                        f_anim_zoom = plt.figure()
                        plt.title(g_n)
                        plt.plot(corr_cutoffs[cc_0_25_ind:], animal_true_rates[gn_i,cc_0_25_ind:], label='True')
                        plt.ylabel('Rate (Hz)')
                        plt.xlabel('Pearson Correlation Cutoff')
                        plt.legend(loc='upper right')
                        f_anim_zoom.savefig(os.path.join(taste_indiv_dir,corr_name+'_'+g_n+'_rate_taste_by_cutoff_zoom.png'))
                        f_anim_zoom.savefig(os.path.join(taste_indiv_dir,corr_name+'_'+g_n+'_rate_taste_by_cutoff_zoom.svg'))
                        plt.close(f_anim_zoom)
                    #Average rates
                    anim_true_avg_rate = np.nanmean(animal_true_rates,0)
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
                    ax_rate_box[s_i,cp_i].set_xticks(np.arange(num_tastes)+1,unique_taste_names,
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
        
def plot_corr_cutoff_epochs(all_corr_dicts, corr_dict, multiday_data_dict, 
                            unique_given_names, unique_corr_names,
                            unique_segment_names, unique_taste_names, max_cp, 
                            corr_cutoffs, colors, plot_save_dir, verbose = False):
    
    """Plot number of events above correlation cutoff for each taste that are 
    unique to the taste and unique to a given epoch."""
    warnings.filterwarnings('ignore') 
    # non_none_tastes = [taste for taste in unique_taste_names if taste != 'none_0']
    cc_0_25_ind = np.where(corr_cutoffs >= 0.25)[0][0]
    cc_top_third_ind = np.where(corr_cutoffs >= 1/3)[0][0]
    num_anim = len(unique_given_names)
    num_cutoff = len(corr_cutoffs)
    num_segs = len(unique_segment_names)
    # num_tastes = len(non_none_tastes)
    num_tastes = len(unique_taste_names)
    indiv_anim_plot_save_dir = os.path.join(plot_save_dir,'Individual_Animals')
    if not os.path.isdir(indiv_anim_plot_save_dir):
        os.mkdir(indiv_anim_plot_save_dir)
    
    for corr_name in unique_corr_names:
        #Plot all corr values
        all_dev_corr_inds = all_corr_dicts[corr_name]
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
            for t_i, taste in enumerate(unique_taste_names):
                taste_indiv_dir = os.path.join(seg_indiv_dir,taste)
                if not os.path.isdir(taste_indiv_dir):
                    os.mkdir(taste_indiv_dir)
                anim_avg_rates = []
                for cp_i in range(max_cp):
                    cp_indiv_dir = os.path.join(taste_indiv_dir,'Epoch_' + str(cp_i))
                    if not os.path.isdir(cp_indiv_dir):
                        os.mkdir(cp_indiv_dir)
                    animal_inds = all_dev_corr_inds[seg_name][cp_i][taste]
                    animal_true_rates = np.zeros((num_anim,num_cutoff))
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
                    #Average rates
                    anim_true_avg_rate = np.nanmean(animal_true_rates,0)
                    anim_avg_rates.append(list(anim_true_avg_rate))
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
        
def plot_corr_cutoff_boxes(all_corr_dicts, corr_dict, multiday_data_dict, unique_given_names, 
                            unique_corr_names, unique_segment_names, unique_taste_names, 
                            max_cp, corr_cutoffs, colors, plot_save_dir, verbose = False):
    
    "Plot boxplots of individual null and animal rates above given cutoff"
    
    warnings.filterwarnings('ignore')
    # non_none_tastes = [taste for taste in unique_taste_names if taste != 'none_0']
    num_anim = len(unique_given_names)
    # num_tastes = len(non_none_tastes)
    num_tastes = len(unique_taste_names)
    cutoffs = [0.10,0.20,0.30,0.40,0.50]
    cutoff_inds = [np.where(corr_cutoffs == cutoffs[i])[0][0] for i in range(len(cutoffs))]
    cutoff_folders = ['cutoff_0_10','cutoff_0_20','cutoff_0_30','cutoff_0_40','cutoff_0_50']
    
    for cf_i, cf in enumerate(cutoff_folders):
        
        if not os.path.isdir(os.path.join(plot_save_dir,cf)):
            os.mkdir(os.path.join(plot_save_dir,cf))
            
        c_val = cutoffs[cf_i]
        c_ind = cutoff_inds[cf_i]
        
        for corr_name in unique_corr_names:
            all_dev_corr_inds = all_corr_dicts[corr_name]
            
            f_box, ax_box = plt.subplots(nrows = len(unique_segment_names),\
                                           ncols = max_cp, figsize=(8,8),\
                                        sharex = True, sharey = True)
            for s_i, seg_name in enumerate(unique_segment_names):
                for cp_i in range(max_cp):
                    for t_i, taste in enumerate(unique_taste_names):
                        animal_inds = all_dev_corr_inds[seg_name][cp_i][taste]
                        anim_rates = []
                        for gn_i, g_n in enumerate(unique_given_names):
                            #Collect true data
                            animal_seg_times = multiday_data_dict[g_n]['segment_times']
                            animal_seg_names = multiday_data_dict[g_n]['segment_names']
                            a_s_ind = [i for i in range(len(animal_seg_names)) if animal_seg_names[i] == seg_name][0]
                            seg_len = (animal_seg_times[a_s_ind+1]-animal_seg_times[a_s_ind])*(1/1000) #seconds length
                            try:
                                true_rate = len(animal_inds[g_n][c_ind])/seg_len
                                anim_rates.append(true_rate)
                            except:
                                errormsg = g_n + ' does not have ' + seg_name + \
                                    ' epoch ' + str(cp_i) + ' ' + taste + ' data.'
                                if verbose == True:
                                    print(errormsg)
                        #Create box plot
                        anim_x = t_i + 0.1*np.random.randn(num_anim)
                        ax_box[s_i,cp_i].boxplot(anim_rates,positions=[t_i])
                        ax_box[s_i,cp_i].scatter(anim_x,anim_rates,color='g',\
                                                 alpha=0.5)
                    ax_box[s_i,cp_i].set_xticks(np.arange(num_tastes),unique_taste_names,rotation=45)
                    if s_i == 0:
                        ax_box[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                ax_box[s_i,0].set_ylabel(seg_name)
            plt.suptitle('Event Rates at Cutoff = ' + str(c_val))
            plt.tight_layout()
            f_box.savefig(os.path.join(os.path.join(plot_save_dir,cf),corr_name+'_rate_at_' + cf + '.png'))
            f_box.savefig(os.path.join(os.path.join(plot_save_dir,cf),corr_name+'_rate_at_' + cf + '.svg'))
            plt.close(f_box)
    
def plot_diff_from_pre_boxes(all_corr_dicts, all_null_corr_dicts, corr_dict, 
                            null_corr_dict, multiday_data_dict, unique_given_names, 
                            unique_corr_names, unique_segment_names, unique_taste_names, 
                            max_cp, corr_cutoffs, colors, plot_save_dir, verbose = False):
    """Subtract rates from pre-taste from later intervals"""
    
    warnings.filterwarnings('ignore')
    # non_none_tastes = [taste for taste in unique_taste_names if taste != 'none_0']
    num_anim = len(unique_given_names)
    # num_tastes = len(non_none_tastes)
    num_tastes = len(unique_taste_names)
    cutoffs = [0.10,0.20,0.30,0.40,0.50]
    cutoff_inds = [np.where(corr_cutoffs == cutoffs[i])[0][0] for i in range(len(cutoffs))]
    cutoff_folders = ['cutoff_0_10','cutoff_0_20','cutoff_0_30','cutoff_0_40','cutoff_0_50']
    not_pre = ['post-taste','sickness']
    pre = 'pre-taste'
    
    for cf_i, cf in enumerate(cutoff_folders):
        
        if not os.path.isdir(os.path.join(plot_save_dir,cf)):
            os.mkdir(os.path.join(plot_save_dir,cf))
            
        c_val = cutoffs[cf_i]
        c_ind = cutoff_inds[cf_i]
        
        for corr_name in unique_corr_names:
            
            all_dev_corr_inds = all_corr_dicts[corr_name]
            
            f_box, ax_box = plt.subplots(nrows = len(not_pre),\
                                           ncols = max_cp, figsize=(8,8),\
                                        sharex = True, sharey = True)
            for s_i, seg_name in enumerate(not_pre):
                for cp_i in range(max_cp):
                    x_tick_labels = []
                    for t_i, taste in enumerate(unique_taste_names):
                        pre_animal_inds = all_dev_corr_inds[pre][cp_i][taste]
                        animal_inds = all_dev_corr_inds[seg_name][cp_i][taste]
                        anim_rates = []
                        null_rates = []
                        for gn_i, g_n in enumerate(unique_given_names):
                            #Collect true data
                            animal_seg_times = multiday_data_dict[g_n]['segment_times']
                            animal_seg_names = multiday_data_dict[g_n]['segment_names']
                            a_s_ind = [i for i in range(len(animal_seg_names)) if animal_seg_names[i] == seg_name][0]
                            a_pre_ind = [i for i in range(len(animal_seg_names)) if animal_seg_names[i] == pre][0]
                            seg_len = (animal_seg_times[a_s_ind+1]-animal_seg_times[a_s_ind])*(1/1000) #seconds length
                            pre_len = (animal_seg_times[a_pre_ind+1]-animal_seg_times[a_pre_ind])*(1/1000) #seconds length
                            try:
                                true_rate = len(animal_inds[g_n][c_ind])/seg_len
                                pre_rate = len(pre_animal_inds[g_n][c_ind])/pre_len
                                anim_rates.append(true_rate-pre_rate)
                            except:
                                errormsg = g_n + ' does not have ' + seg_name + \
                                    ' epoch ' + str(cp_i) + ' ' + taste + ' data.'
                                if verbose == True:
                                    print(errormsg)
                            #Collect null data
                            num_null = null_corr_dict[g_n]['num_null']
                            try:
                                pre_null_data = null_corr_dict[g_n][pre][taste][corr_name][cp_i]
                                null_data = null_corr_dict[g_n][seg_name][taste][corr_name][cp_i]
                                anim_null_rates = []
                                for null_i in range(num_null):
                                    pre_null_rate = len(np.where(pre_null_data[null_i] > c_val)[0])/pre_len
                                    null_rate = len(np.where(null_data[null_i] > c_val)[0])/seg_len
                                    anim_null_rates.append(null_rate-pre_null_rate)
                                null_rates.append(np.nanmean(anim_null_rates))
                            except:
                                errormsg = g_n + ' does not have ' + seg_name + \
                                    ' epoch ' + str(cp_i) + ' ' + taste + ' data.'
                                if verbose == True:
                                    print(errormsg)
                        #Create box plot
                        y_max = np.max(anim_rates)
                        anim_x = t_i + 0.1*np.random.randn(num_anim)
                        ax_box[s_i,cp_i].axhline(0,linestyle='dashed',color='k',alpha=0.25)
                        ax_box[s_i,cp_i].boxplot(anim_rates,positions=[t_i])
                        ax_box[s_i,cp_i].scatter(anim_x,anim_rates,color='g',\
                                                 alpha=0.5)
                        x_tick_labels.append(taste)
                        # > 0 test
                        if np.percentile(anim_rates,5) > 0:
                            ax_box[s_i,cp_i].text(t_i,y_max+0.15*y_max,'>0')
                    ax_box[s_i,cp_i].set_xticks(np.arange(num_tastes),unique_taste_names,rotation=45)
                    if s_i == 0:
                        ax_box[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                ax_box[s_i,0].set_ylabel(seg_name)
            plt.suptitle('Pre-Taste Normalized Event Rates at Cutoff = ' + str(c_val))
            plt.tight_layout()
            f_box.savefig(os.path.join(os.path.join(plot_save_dir,cf),corr_name+'_pre_normalized_rate_at_' + cf + '.png'))
            f_box.savefig(os.path.join(os.path.join(plot_save_dir,cf),corr_name+'_pre_normalized_rate_at_' + cf + '.svg'))
            plt.close(f_box)
        
def compare_decode_data(decode_dict, multiday_data_dict, unique_given_names,
                       unique_decode_names, unique_decode_groups, unique_segment_names, 
                       unique_taste_names, max_cp, save_dir, verbose=False):
    
    decode_results_save_dir = os.path.join(save_dir,'Decodes')
    if not os.path.isdir(decode_results_save_dir):
        os.mkdir(decode_results_save_dir)
        
    #Plot cross-animal rates of decodes
    decode_rates_plots(decode_dict,multiday_data_dict,unique_given_names,\
                       unique_decode_names,unique_decode_groups,unique_segment_names,\
                           unique_taste_names, max_cp,decode_results_save_dir,verbose)
    
    
def decode_rates_plots(decode_dict, multiday_data_dict, unique_given_names, 
                       unique_decode_names, unique_decode_groups, unique_segment_names,
                       unique_taste_names, max_cp, decode_results_save_dir,
                       verbose=False):
    
    colors = ['red','orange','yellow','green','royalblue','purple', \
              'magenta','brown', 'cyan']
    num_anim = len(unique_given_names)
    num_seg = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    unique_segment_names = ['pre-taste','post-taste','sickness'] #manual order override
    for dt in unique_decode_names:
        
        dt_decode_groups = unique_decode_groups[dt]
        num_groups = len(unique_decode_groups[dt])
        
        #Decoder Accuracy
        plot_decoder_accuracy_stats(num_anim, num_groups, unique_given_names, \
                                        dt_decode_groups, decode_dict, dt, \
                                            decode_results_save_dir)
        
        #Deviation Decodes
        plot_deviation_decode_stats(unique_segment_names,unique_given_names,dt_decode_groups,\
                                        multiday_data_dict,decode_dict,dt,decode_results_save_dir)
        
        #Sliding Bin Decodes
        
def plot_decode_corr_combined_data():
    """Create plots of correlation distributions of the decoded events"""
    

def plot_decoder_accuracy_stats(num_anim, num_groups, unique_given_names, \
                                dt_decode_groups, decode_dict, dt, \
                                    decode_results_save_dir):
    """
    This function plots accuracy stats a few different ways.

    """
    #Overall accuracy rates
    accuracy_rates = np.zeros((num_anim,num_groups))
    for gn_i, gn in enumerate(unique_given_names):
        for dg_i, dg in enumerate(dt_decode_groups):
            try:
                group_ind = [i for i in range(len(dt_decode_groups)) if dt_decode_groups[i] == dg][0]
                group_indices = decode_dict[gn][dt]['group_dict'][dt_decode_groups[group_ind]]
                try:
                    decode_predictions = decode_dict[gn][dt]['Decoder_Accuracy']['nb_decode_predictions']
                    group_counts = 0
                    group_totals = 0
                    for gi in group_indices:
                        group_predictions = decode_predictions[str(gi[1])+','+str(gi[0])]
                        group_totals += np.shape(group_predictions)[0]
                        if group_ind == num_groups-1: #No taste control still successful if null prediction
                            group_counts += np.sum(group_predictions[:,group_ind:])
                        else:
                            group_counts += np.sum(group_predictions[:,group_ind])
                    accuracy_rates[gn_i,dg_i] = group_counts/group_totals
                except:
                    accuracy_rates[gn_i,dg_i] = np.nan
            except:
                accuracy_rates[gn_i,dg_i] = np.nan
    accuracy_means = np.nanmean(accuracy_rates,0)
    f_accuracy = plt.figure(figsize=(10,10))
    plt.axhline(100/(num_groups+1),linestyle='dashed',alpha=0.5,color='k')
    plt.boxplot(100*accuracy_rates,labels=dt_decode_groups,showmeans=True)
    for g_i in range(num_groups):
        plt.text(g_i+1,100*accuracy_means[g_i],\
                 str(np.round(100*accuracy_means[g_i],2)),\
                     va='top',ha='left')
    plt.xticks(rotation=45)
    plt.title(dt + '\nCross-Animal Decoder Accuracy')
    f_accuracy.savefig(os.path.join(decode_results_save_dir,dt+'_decoder_accuracy_box.png'))
    f_accuracy.savefig(os.path.join(decode_results_save_dir,dt+'_decoder_accuracy_box.svg'))
    plt.close(f_accuracy)
    
    #Histograms of accuracy confusion
    accuracy_confusion_rates = np.zeros((num_groups,num_anim,num_groups))
    for gn_i, gn in enumerate(unique_given_names):
        for dg1_i, dg1 in enumerate(dt_decode_groups): #Group we're testing the decode rates for
            gn_decode_groups = []
            for dg_i, dg_val in enumerate(dt_decode_groups):
                if dg_val.split('_')[0] == 'Nacl': #To handle the salt vs nacl issue
                    dg_val = 'Salt_' + dg_val.split('_')[1]
                    gn_decode_groups.append(dg_val)
                else:
                    gn_decode_groups.append(dg_val)
            decode_predictions = decode_dict[gn][dt]['Decoder_Accuracy']['nb_decode_predictions']
            group_ind = [i for i in range(len(gn_decode_groups)) if gn_decode_groups[i] == dg1][0]
            group_indices = decode_dict[gn][dt]['group_dict'][dt_decode_groups[group_ind]]
            for gi in group_indices:
                group_predictions = decode_predictions[str(gi[1])+','+str(gi[0])]
                group_predictions_join_null = group_predictions[:,:-1]
                group_predictions_join_null[:,-1] += group_predictions[:,-1]
                accuracy_confusion_rates[dg1_i,gn_i,:] = np.nansum(group_predictions_join_null,0)
    accuracy_sums = np.nansum(accuracy_confusion_rates,1)
    totals = (np.nansum(accuracy_sums,1)*np.ones(np.shape(accuracy_sums))).T
    accuracy_fracs = accuracy_sums/totals
    f_accuracy_matrix = plt.figure(figsize=(8,8))
    plt.imshow(accuracy_fracs)
    plt.colorbar()
    plt.xticks(np.arange(num_groups),dt_decode_groups,rotation=45)
    plt.xlabel('Decoder Readouts')
    plt.yticks(np.arange(num_groups),dt_decode_groups,rotation=45)
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    f_accuracy_matrix.savefig(os.path.join(decode_results_save_dir,dt+'_decoder_confusion_matrix.png'))
    f_accuracy_matrix.savefig(os.path.join(decode_results_save_dir,dt+'_decoder_confusion_matrix.svg'))
    plt.close(f_accuracy_matrix)

def plot_deviation_decode_stats(unique_segment_names,unique_given_names,dt_decode_groups,\
                                multiday_data_dict,decode_dict,dt,decode_results_save_dir):
    """
    This function plots deviation decode stats a few different ways.

    """
    #All events how they're decoded
    num_seg = len(unique_segment_names)
    num_anim = len(unique_given_names)
    num_groups = len(dt_decode_groups)
    dev_decode_counts = np.zeros((num_seg,num_anim,num_groups))
    for seg_i, seg_name in enumerate(unique_segment_names):
        for gn_i, gn in enumerate(unique_given_names):
            seg_index = [i for i in range(len(multiday_data_dict[gn]['segment_names'])) if multiday_data_dict[gn]['segment_names'][i] == seg_name][0]
            anim_decode_groups = list(decode_dict[gn][dt]['group_dict'].keys())
            anim_decode_data = decode_dict[gn][dt]['NB_Decoding']['segment_' + str(seg_index)]
            for dg_i, dg in enumerate(dt_decode_groups):
                try:
                    anim_dg_ind = [i for i in range(len(anim_decode_groups)) if anim_decode_groups[i] == dg][0]
                    if len(np.shape(anim_decode_data))>1:
                        argmax_decode_data = np.argmax(anim_decode_data,1)
                        num_decoded = len(np.where(argmax_decode_data == anim_dg_ind)[0])
                    else:
                        num_decoded = len(np.where(anim_decode_data == anim_dg_ind)[0])
                    dev_decode_counts[seg_i,gn_i,dg_i] = num_decoded
                except:
                    dev_decode_counts[seg_i,gn_i,dg_i] = np.nan
    dev_counts = np.nansum(dev_decode_counts,2)
    dev_decode_fracs = np.zeros(np.shape(dev_decode_counts))
    for seg_i in range(len(unique_segment_names)):
        dev_decode_fracs[seg_i,:,:] = np.squeeze(dev_decode_counts[seg_i,:,:])/(dev_counts[seg_i,:]*np.ones((num_groups,num_anim))).T
    #By animal fraction plots
    f_decode_frac, ax_decode_frac = plt.subplots(ncols = len(unique_segment_names),\
                                 sharex = True, sharey = True, figsize=(12,4))
    for seg_i, seg_name in enumerate(unique_segment_names):
        data = np.squeeze(dev_decode_fracs[seg_i,:,:])
        #Perform stat test of normality
        group_norm = np.nan*np.ones(num_groups)
        for g_i in range(num_groups):
            data_norm = anderson(data[:,g_i].squeeze(), dist='norm')
            critical_level_ind = np.where(data_norm.statistic > data_norm.critical_values)[0]
            if (len(critical_level_ind) > 0) and (critical_level_ind[-1] >= 2):
                group_norm[g_i] = 0
            else: #Normally distributed
                group_norm[g_i] = 1
        not_norm = np.where(group_norm == 0)[0]
        #Perform ANOVA
        anova_inputs = []
        for g_i in range(num_groups):
            anova_inputs.append(data[:,g_i])
        _, anova_p_value = f_oneway(*anova_inputs)
        data_mean = np.nanmean(data,0)
        #Plot
        ax_decode_frac[seg_i].boxplot(data,labels=dt_decode_groups)
        for dc_i in not_norm:
            ax_decode_frac[seg_i].text(dc_i+1,1.05,'* AD',ha = 'left',va = 'top')
        ax_decode_frac[seg_i].set_xticks(np.arange(len(dt_decode_groups))+1,\
                                         dt_decode_groups,rotation=45)
        if anova_p_value <= 0.05:
            ax_decode_frac[seg_i].set_title(seg_name + '\n*ANOVA')
        else:
            ax_decode_frac[seg_i].set_title(seg_name)
    ax_decode_frac[0].set_ylim([-0.1,1.1])
    ax_decode_frac[0].set_ylabel('Fraction of Deviation Events')
    plt.suptitle('Deviation Decode Fractions')
    plt.tight_layout()
    f_decode_frac.savefig(os.path.join(decode_results_save_dir,dt+'_by_animal_dev_decode_rates.png'))
    f_decode_frac.savefig(os.path.join(decode_results_save_dir,dt+'_by_animal_dev_decode_rates.svg'))
    plt.close(f_decode_frac)
    
    #Outlier removed mean pies
    f_decode_pie, ax_decode_pie = plt.subplots(ncols = len(unique_segment_names),\
                                 sharex = True, sharey = True, figsize=(12,4))
    for seg_i, seg_name in enumerate(unique_segment_names):
        data = np.squeeze(dev_decode_counts[seg_i,:,:])
        #Calculate data outliers and remove
        no_outlier_data = np.nan*np.ones(np.shape(data))
        data_counts = np.zeros(num_groups).astype('int')
        for g_i in range(num_groups):
            group_data = data[:,g_i]
            z_scores = np.abs((group_data - np.nanmean(group_data))/np.nanstd(group_data))
            outlier_inds = np.where(z_scores >= 2)[0]
            keep_inds = np.setdiff1d(np.arange(num_anim),outlier_inds)
            no_outlier_data[keep_inds,g_i] = data[keep_inds,g_i]
            data_counts[g_i] = len(keep_inds)
        data_mean = np.nanmean(no_outlier_data,0)
        data_percents = np.round(100*data_mean/np.nansum(data_mean),2)
        data_labels = [dt_decode_groups[i] + '\n' + str(data_percents[i]) + '%' + \
                       '\nn=' + str(data_counts[i]) for i in range(num_groups)]
        explode = [0.1*i for i in range(len(dt_decode_groups))]
        ax_decode_pie[seg_i].pie(data_mean,explode=explode,\
                                  labeldistance=1.15,labels=data_labels)
        ax_decode_frac[seg_i].set_title(seg_name)
    plt.suptitle('Across Animals Outlier-Removed Mean Decode Percents')
    plt.tight_layout()
    f_decode_pie.savefig(os.path.join(decode_results_save_dir,dt+'_no_outlier_dev_decode_rates_pie.png'))
    f_decode_pie.savefig(os.path.join(decode_results_save_dir,dt+'_no_outlier_dev_decode_rates_pie.svg'))
    plt.close(f_decode_pie)
        
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
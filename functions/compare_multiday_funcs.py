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
from scipy.stats import ks_2samp

def compare_corr_data(corr_dict, multiday_data_dict, unique_given_names,
                      unique_corr_names, unique_segment_names, unique_taste_names, 
                      max_cp, save_dir):
    
    corr_results_save_dir = os.path.join(save_dir,'Correlations')
    if not os.path.isdir(corr_results_save_dir):
        os.mkdir(corr_results_save_dir)
        
    colors = ['red','orange','yellow','green','royalblue','purple', \
              'magenta','brown', 'cyan']
    corr_cutoffs = np.round(np.arange(0,1.01,0.01),2)   
        
    #KS-Test all taste pairs
    test_corr_dist_ks_test(corr_dict, unique_given_names, unique_corr_names,
                             unique_segment_names, unique_taste_names, max_cp,
                             corr_results_save_dir)
    
    #Calculate indices unique to each correlation combo
    all_corr_dicts, unique_corr_dicts = calc_ind_dicts(corr_dict, unique_given_names, unique_corr_names,
                             unique_segment_names, unique_taste_names, max_cp,
                             corr_cutoffs)
    
    #Create plots
    plot_corr_cutoff_tastes(all_corr_dicts, unique_corr_dicts, corr_dict,
                                        unique_given_names, unique_corr_names, 
                                        unique_segment_names, unique_taste_names, 
                                        max_cp, corr_cutoffs, colors, 
                                        corr_results_save_dir)
    
    plot_corr_cutoff_epochs(all_corr_dicts, unique_corr_dicts, corr_dict,
                            unique_given_names, unique_corr_names, 
                            unique_segment_names, unique_taste_names, max_cp, 
                            corr_cutoffs, colors, corr_results_save_dir)
        
           
def test_corr_dist_ks_test(corr_dict, unique_given_names, unique_corr_names,
                         unique_segment_names, unique_taste_names, max_cp,
                         plot_save_dir,verbose=False):
    
    taste_inds = np.arange(len(unique_taste_names))
    t_pairs = list(combinations(list(taste_inds),2))
    
    #Compare taste distributions against each other
    for corr_name in unique_corr_names:
        for seg_name in unique_segment_names:
            for cp_i in range(max_cp):
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
                for tp in t_pairs:
                    tp_i1 = tp[0]
                    data_1 = all_taste_dist[tp_i1]
                    tp_i2 = tp[1]
                    data_2 = all_taste_dist[tp_i2]
                    if len(data_1) > 0:
                        if len(data_2) > 0:
                            result = ks_2samp(data_1,data_2,alternative='two-sided')
                            if result[1] <= 0.05:
                                if np.nanmean(data_1) < np.nanmean(data_2):
                                    ks_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'*<'])
                                else:
                                    ks_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'*>'])
                            else:
                                ks_results.append([unique_taste_names[tp_i1],unique_taste_names[tp_i2],'n.s.'])
                #Output results to csv
                csv_save_name = corr_name + '_' + seg_name + '_Epoch_' + str(cp_i) + '.csv'
                with open(os.path.join(plot_save_dir,csv_save_name),'w',newline='') as file:
                    writer = csv.writer(file)
                    for row in ks_results:
                        writer.writerow(row)
    
        
def calc_ind_dicts(corr_dict, unique_given_names, unique_corr_names,
                         unique_segment_names, unique_taste_names, max_cp,
                         corr_cutoffs, verbose=False):
    #Collect all indices above a cutoff
    all_corr_dicts = dict()
    for corr_name in unique_corr_names:
        all_dev_corr_inds = dict()
        for s_i, seg_name in enumerate(unique_segment_names):
            all_dev_corr_inds[seg_name] = dict()
            for cp_i in range(max_cp):
                all_dev_corr_inds[seg_name][cp_i] = dict()
                for t_i, taste in enumerate(unique_taste_names):
                    all_dev_corr_inds[seg_name][cp_i][taste] = dict()
                    for gn_i, gn in enumerate(unique_given_names):
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
                                all_dev_corr_inds[seg_name][cp_i][taste][gn] = [np.where(cp_means >= cc)[0] for cc in corr_cutoffs]
                            else:
                                all_dev_corr_inds[seg_name][cp_i][taste][gn] = [np.where(deliv_means[cp_i,:] >= cc)[0] for cc in corr_cutoffs]
                        except:
                            errormsg = 'Missing data for ' + corr_name + ' ' + \
                                seg_name + ' Epoch ' + str(cp_i) + ' ' + taste + \
                                    ' ' + gn
                            if verbose == True:
                                print(errormsg)
        all_corr_dicts[corr_name] = all_dev_corr_inds
    
    #Collect unique index counts above a given cutoff for a taste-epoch combination
    taste_epoch_pairs = [] #list of taste-epoch pairs
    for taste in unique_taste_names:
        if taste == 'none_0':
            taste_epoch_pairs.append(taste + '-0')
        else:
            for cp_i in range(max_cp):
                taste_epoch_pairs.append(taste + '-' + str(cp_i))
    unique_corr_dicts = dict() #Collect counts of unique indices across animals for different conditions
    for corr_name in unique_corr_names:
        unique_corr_dicts[corr_name] = dict()
        for s_i, seg_name in enumerate(unique_segment_names):
            unique_corr_dicts[corr_name][seg_name] = dict()
            for cc_i, cc in enumerate(corr_cutoffs):
                unique_corr_dicts[corr_name][seg_name][cc_i] = dict()
                for taste in unique_taste_names:
                    unique_corr_dicts[corr_name][seg_name][cc_i][taste] = dict()
                    for cp_i in range(max_cp):
                        unique_corr_dicts[corr_name][seg_name][cc_i][taste][cp_i] = 0
                for tep_i, tep in enumerate(taste_epoch_pairs):
                    taste = tep.split('-')[0]
                    cp_i = int(tep.split('-')[1])
                    for gn_i, gn in enumerate(unique_given_names):
                        try:
                            anim_tep_inds = np.unique(all_dev_corr_inds[seg_name][cp_i][taste][gn][cc_i])
                            remaining_tep = np.setdiff1d(taste_epoch_pairs,tep)
                            other_tep_inds = []
                            for tep2_i, tep2 in enumerate(remaining_tep):
                                taste_2 = tep2.split('-')[0]
                                cp_i_2 = int(tep2.split('-')[1])
                                try:
                                    other_tep_inds.extend(all_dev_corr_inds[seg_name][cp_i_2][taste_2][gn][cc_i])
                                except:
                                    other_tep_inds.extend([])
                            other_tep_inds = np.unique(np.array(other_tep_inds))
                            unique_anim_tep_inds = np.setdiff1d(np.array(anim_tep_inds),np.array(other_tep_inds))
                        except:
                            unique_anim_tep_inds = []
                        unique_corr_dicts[corr_name][seg_name][cc_i][taste][cp_i] += len(unique_anim_tep_inds)
        
    return all_corr_dicts, unique_corr_dicts
        
def plot_corr_cutoff_tastes(all_corr_dicts, unique_corr_dicts, corr_dict,
                                        unique_given_names, unique_corr_names,
                                        unique_segment_names, unique_taste_names, 
                                        max_cp, corr_cutoffs, colors, plot_save_dir,
                                        verbose = False):
    
    """Plot number of events above correlation cutoff for each taste"""
    
    warnings.filterwarnings('ignore')
    non_none_tastes = [taste for taste in unique_taste_names if taste != 'none_0']
    cc_0_25_ind = np.where(corr_cutoffs == 0.25)[0][0]
    
    for corr_name in unique_corr_names:
        
        #Plot all corr values          
        all_dev_corr_inds = all_corr_dicts[corr_name]
        f_frac, ax_frac = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        f_count, ax_count = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        f_rate, ax_rate = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        max_cc = 0
        zoom_y_count = 0
        zoom_y_frac = 0
        for s_i, seg_name in enumerate(unique_segment_names):
            for cp_i in range(max_cp):
                for t_i, taste in enumerate(unique_taste_names):
                    animal_inds = all_dev_corr_inds[seg_name][cp_i][taste]
                    animal_ind_counts = np.zeros(len(corr_cutoffs))
                    animal_dev_counts = np.zeros(len(corr_cutoffs))
                    for g_n in unique_given_names:
                        try:
                            num_dev = corr_dict[g_n][corr_name][seg_name]['all'][taste]['num_dev']
                            num_inds = np.array([len(animal_inds[g_n][cc_i]) for cc_i in range(len(corr_cutoffs))])
                            animal_ind_counts = animal_ind_counts + num_inds
                            animal_dev_counts = animal_dev_counts + num_dev*np.ones(len(corr_cutoffs))
                        except:
                            errormsg = g_n + ' does not have ' + seg_name + \
                                ' epoch ' + str(cp_i) + ' ' + taste + ' data.'
                            if verbose == True:
                                print(errormsg)
                    animal_fracs = animal_ind_counts/animal_dev_counts
                    #Plot counts and fractions
                    ax_frac[s_i, cp_i].plot(corr_cutoffs,animal_fracs,label=taste,
                                            color=colors[t_i])
                    ax_count[s_i, cp_i].plot(corr_cutoffs,animal_ind_counts,label=taste,
                                            color=colors[t_i])
                    #Calculate 0 dropoff
                    count_0 = np.where(animal_ind_counts == 0)[0]
                    if len(count_0) > 0:
                        if corr_cutoffs[count_0[0]] > max_cc:
                            max_cc = corr_cutoffs[count_0[0]]
                    #Update zoom y vals
                    if np.nanmax(animal_ind_counts[cc_0_25_ind:]) > zoom_y_count:
                        zoom_y_count = np.nanmax(animal_ind_counts[cc_0_25_ind:])
                    if np.nanmax(animal_fracs[cc_0_25_ind:]) > zoom_y_frac:
                        zoom_y_frac = np.nanmax(animal_fracs[cc_0_25_ind:])
                #More plot updates
                if s_i == 0:
                    ax_frac[s_i, cp_i].set_title('Epoch ' + str(cp_i))
                    ax_count[s_i, cp_i].set_title('Epoch ' + str(cp_i))
                if s_i == len(unique_segment_names)-1:
                    ax_frac[s_i, cp_i].set_xlabel('Correlation Cutoff')
                    ax_count[s_i, cp_i].set_xlabel('Correlation Cutoff')
                if cp_i == 0:
                    ax_frac[s_i, cp_i].set_ylabel(seg_name + '\nFraction of Events Above Cutoff')
                    ax_count[s_i, cp_i].set_ylabel(seg_name + '\nNumber of Events Above Cutoff')
        plt.figure(f_frac)
        ax_frac[0,0].legend(loc='upper left')
        ax_frac[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Fraction of Events Above Cutoff')
        plt.tight_layout()
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_taste_by_cutoff.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_taste_by_cutoff.svg'))
        ax_frac[0,0].set_xlim([0.25,max_cc])
        ax_frac[0,0].set_ylim([0,zoom_y_frac])
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_taste_by_cutoff_zoom.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_taste_by_cutoff_zoom.svg'))
        plt.close(f_frac)
        plt.figure(f_count)
        ax_count[0,0].legend(loc='upper left')
        ax_count[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Number of Events Above Cutoff')
        plt.tight_layout()
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_taste_by_cutoff.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_taste_by_cutoff.svg'))
        ax_count[0,0].set_xlim([0.25,max_cc])
        ax_count[0,0].set_ylim([0,zoom_y_count])
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_taste_by_cutoff_zoom.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_taste_by_cutoff_zoom.svg'))
        plt.close(f_count) 
        
        
        #Plot unique indices
        f_frac, ax_frac = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        f_count, ax_count = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        max_cc = 0
        zoom_y_count = 0
        zoom_y_frac = 0
        for s_i, seg_name in enumerate(unique_segment_names):
            for cp_i in range(max_cp):
                for t_i, taste in enumerate(non_none_tastes):
                    num_dev = 0
                    for gn in unique_given_names:
                        try:
                            num_dev += corr_dict[gn][corr_name][seg_name]['all'][taste]['num_dev']
                        except:
                            num_dev += 0
                    unique_counts = np.zeros(len(corr_cutoffs))
                    for cc_i, cc in enumerate(corr_cutoffs):
                        unique_counts[cc_i] = unique_corr_dicts[corr_name][seg_name][cc_i][taste][cp_i]
                    unique_fracs = unique_counts/num_dev
                    #Plot unique counts and fractions
                    ax_frac[s_i, cp_i].plot(corr_cutoffs,unique_fracs,label=taste + ' only',
                                            color=colors[t_i])
                    ax_count[s_i, cp_i].plot(corr_cutoffs,unique_counts,label=taste + ' only',
                                            color=colors[t_i])
                    #Calculate 0 dropoff
                    count_0 = np.where(unique_counts == 0)[0]
                    if len(count_0) > 0:
                        if corr_cutoffs[count_0[0]] > max_cc:
                            max_cc = corr_cutoffs[count_0[0]]
                    #Update y lim
                    if np.nanmax(unique_counts[cc_0_25_ind:]) > zoom_y_count:
                        zoom_y_count = np.nanmax(unique_counts[cc_0_25_ind:])
                    if np.nanmax(unique_fracs[cc_0_25_ind]) > zoom_y_frac:
                        zoom_y_frac = np.nanmax(unique_fracs[cc_0_25_ind])
                #More plot updates
                if s_i == 0:
                    ax_frac[s_i, cp_i].set_title('Epoch ' + str(cp_i))
                    ax_count[s_i, cp_i].set_title('Epoch ' + str(cp_i))
                if s_i == len(unique_segment_names)-1:
                    ax_frac[s_i, cp_i].set_xlabel('Correlation Cutoff')
                    ax_count[s_i, cp_i].set_xlabel('Correlation Cutoff')
                if cp_i == 0:
                    ax_frac[s_i, cp_i].set_ylabel(seg_name + '\nFraction of Events Above Cutoff')
                    ax_count[s_i, cp_i].set_ylabel(seg_name + '\nNumber of Events Above Cutoff')
        plt.figure(f_frac)
        ax_frac[0,0].legend(loc='upper left')
        ax_frac[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Fraction of Events Above Cutoff')
        plt.tight_layout()
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff.svg'))
        ax_frac[0,0].set_xlim([0.25,max_cc])
        ax_frac[0,0].set_ylim([0,zoom_y_frac])
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff_zoom.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff_zoom.svg'))
        plt.close(f_frac)
        plt.figure(f_count)
        ax_count[0,0].legend(loc='upper left')
        ax_count[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Number of Events Above Cutoff')
        plt.tight_layout()
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff.svg'))
        ax_count[0,0].set_xlim([0.25,max_cc])
        ax_count[0,0].set_ylim([0,zoom_y_count])
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff_zoom.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff_zoom.svg'))
        plt.close(f_count) 
        
def plot_corr_cutoff_epochs(all_corr_dicts, unique_corr_dicts, corr_dict, 
                                        unique_given_names, unique_corr_names,
                                        unique_segment_names, unique_taste_names, 
                                        max_cp, corr_cutoffs, colors, plot_save_dir,
                                        verbose = False):
    
    """Plot number of events above correlation cutoff for each taste that are 
    unique to the taste and unique to a given epoch."""
    warnings.filterwarnings('ignore') 
    non_none_tastes = [taste for taste in unique_taste_names if taste != 'none_0']
    cc_0_25_ind = np.where(corr_cutoffs == 0.25)[0][0]
    
    for corr_name in unique_corr_names:
        
        #Plot all corr values
        all_dev_corr_inds = all_corr_dicts[corr_name]
        f_frac, ax_frac = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = len(non_none_tastes), figsize=(4*len(non_none_tastes),4*len(unique_segment_names)),\
                                    sharex = True, sharey = True)
        f_count, ax_count = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = len(non_none_tastes), figsize=(4*len(non_none_tastes),4*len(unique_segment_names)),\
                                    sharex = True, sharey = True)
        max_cc = 0
        zoom_y_count = 0
        zoom_y_frac = 0
        for s_i, seg_name in enumerate(unique_segment_names):
            for t_i, taste in enumerate(non_none_tastes):
                for cp_i in range(max_cp):
                    animal_inds = all_dev_corr_inds[seg_name][cp_i][taste]
                    animal_ind_counts = np.zeros(len(corr_cutoffs))
                    animal_dev_counts = np.zeros(len(corr_cutoffs))
                    for g_n in unique_given_names:
                        try:
                            num_dev = corr_dict[g_n][corr_name][seg_name]['all'][taste]['num_dev']
                            num_inds = np.array([len(animal_inds[g_n][cc_i]) for cc_i in range(len(corr_cutoffs))])
                            animal_ind_counts = animal_ind_counts + num_inds
                            animal_dev_counts = animal_dev_counts + num_dev*np.ones(len(corr_cutoffs))
                        except:
                            errormsg = g_n + ' does not have ' + seg_name + \
                                ' epoch ' + str(cp_i) + ' ' + taste + ' data.'
                            if verbose == True:
                                print(errormsg) 
                    animal_fracs = animal_ind_counts/animal_dev_counts
                    #Plot unique counts and fractions
                    ax_frac[s_i, t_i].plot(corr_cutoffs,animal_fracs,label='Epoch ' + str(cp_i),
                                            color=colors[cp_i])
                    ax_count[s_i, t_i].plot(corr_cutoffs,animal_ind_counts,label='Epoch ' + str(cp_i),
                                            color=colors[cp_i])
                    #Calculate 0 dropoff
                    count_0 = np.where(animal_ind_counts == 0)[0]
                    if len(count_0) > 0:
                        if corr_cutoffs[count_0[0]] > max_cc:
                            max_cc = corr_cutoffs[count_0[0]]
                    #Update zoom y vals
                    if np.nanmax(animal_ind_counts[cc_0_25_ind:]) > zoom_y_count:
                        zoom_y_count = np.nanmax(animal_ind_counts[cc_0_25_ind:])
                    if np.nanmax(animal_fracs[cc_0_25_ind:]) > zoom_y_frac:
                        zoom_y_frac = np.nanmax(animal_fracs[cc_0_25_ind:])
                if s_i == 0:
                    ax_frac[s_i, t_i].set_title(taste)
                    ax_count[s_i, t_i].set_title(taste)
                if s_i == len(unique_segment_names)-1:
                    ax_frac[s_i, t_i].set_xlabel('Correlation Cutoff')
                    ax_count[s_i, t_i].set_xlabel('Correlation Cutoff')
                if cp_i == 0:
                    ax_frac[s_i, t_i].set_ylabel(seg_name + '\nFraction of Events Above Cutoff')
                    ax_count[s_i, t_i].set_ylabel(seg_name + '\nNumber of Events Above Cutoff')
        plt.figure(f_frac)
        ax_frac[0,0].legend(loc='upper left')
        ax_frac[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Fraction of Events Above Cutoff')
        plt.tight_layout()
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_epoch_by_cutoff.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_epoch_by_cutoff.svg'))
        ax_frac[0,0].set_xlim([0.25,max_cc])
        ax_frac[0,0].set_ylim([0,zoom_y_frac])
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_epoch_by_cutoff_zoom.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_epoch_by_cutoff_zoom.svg'))
        plt.close(f_frac)
        plt.figure(f_count)
        ax_count[0,0].legend(loc='upper left')
        ax_count[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Number of Events Above Cutoff')
        plt.tight_layout()
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_epoch_by_cutoff.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_epoch_by_cutoff.svg'))
        ax_count[0,0].set_xlim([0.25,max_cc])
        ax_count[0,0].set_ylim([0,zoom_y_count])
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_epoch_by_cutoff_zoom.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_epoch_by_cutoff_zoom.svg'))
        plt.close(f_count) 
        
        #Plot unique indices
        f_frac, ax_frac = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = len(non_none_tastes), figsize=(8,8),\
                                    sharex = True, sharey = True)
        f_count, ax_count = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = len(non_none_tastes), figsize=(8,8),\
                                    sharex = True, sharey = True)
        max_cc = 0
        zoom_y_count = 0
        zoom_y_frac = 0
        for s_i, seg_name in enumerate(unique_segment_names):
            for t_i, taste in enumerate(non_none_tastes):
                for cp_i in range(max_cp):
                    num_dev = 0
                    for gn in unique_given_names:
                        try:
                            num_dev += corr_dict[gn][corr_name][seg_name]['all'][taste]['num_dev']
                        except:
                            num_dev += 0
                    unique_counts = np.zeros(len(corr_cutoffs))
                    for cc_i, cc in enumerate(corr_cutoffs):
                        unique_counts[cc_i] = unique_corr_dicts[corr_name][seg_name][cc_i][taste][cp_i]
                    unique_fracs = unique_counts/num_dev
                    #Plot unique counts and fractions
                    ax_frac[s_i, t_i].plot(corr_cutoffs,unique_fracs,
                                           label='Epoch ' + str(cp_i) + ' only',
                                            color=colors[cp_i])
                    ax_count[s_i, t_i].plot(corr_cutoffs,unique_counts,
                                            label='Epoch ' + str(cp_i) + ' only',
                                            color=colors[cp_i])
                    #Calculate 0 dropoff
                    count_0 = np.where(unique_counts == 0)[0]
                    if len(count_0) > 0:
                        if corr_cutoffs[count_0[0]] > max_cc:
                            max_cc = corr_cutoffs[count_0[0]]
                    #Update y lim
                    if np.nanmax(unique_counts[cc_0_25_ind:]) > zoom_y_count:
                        zoom_y_count = np.nanmax(unique_counts[cc_0_25_ind:])
                    if np.nanmax(unique_fracs[cc_0_25_ind:]) > zoom_y_frac:
                        zoom_y_frac = np.nanmax(unique_fracs[cc_0_25_ind:])             
                #More plot updates
                if s_i == 0:
                    ax_frac[s_i, t_i].set_title(taste)
                    ax_count[s_i, t_i].set_title(taste)
                if s_i == len(unique_segment_names)-1:
                    ax_frac[s_i, t_i].set_xlabel('Correlation Cutoff')
                    ax_count[s_i, t_i].set_xlabel('Correlation Cutoff')
                if cp_i == 0:
                    ax_frac[s_i, t_i].set_ylabel(seg_name + '\nFraction of Events Above Cutoff')
                    ax_count[s_i, t_i].set_ylabel(seg_name + '\nNumber of Events Above Cutoff')
        plt.figure(f_frac)
        ax_frac[0,0].legend(loc='upper left')
        ax_frac[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Fraction of Events Above Cutoff')
        plt.tight_layout()
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff.svg'))
        ax_frac[0,0].set_xlim([0.25,max_cc])
        ax_frac[0,0].set_ylim([0,zoom_y_frac])
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff_zoom.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff_zoom.svg'))
        plt.close(f_frac)
        plt.figure(f_count)
        ax_count[0,0].legend(loc='upper left')
        ax_count[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Number of Events Above Cutoff')
        plt.tight_layout()
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff.svg'))
        ax_count[0,0].set_xlim([0.25,max_cc])
        ax_count[0,0].set_ylim([0,zoom_y_count])
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff_zoom.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff_zoom.svg'))
        plt.close(f_count)
        
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
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
                      unique_segment_names, unique_group_names, save_dir):
    
    corr_results_save_dir = os.path.join(save_dir,'Correlations')
    if not os.path.isdir(corr_results_save_dir):
        os.mkdir(corr_results_save_dir)
        
    colors = ['red','orange','yellow','green','royalblue','purple', \
              'magenta','brown', 'cyan']
    corr_cutoffs = np.round(np.arange(0,1.01,0.01),2)   
        
    #Plot all data distributions against each other
    plot_all_dist(corr_dict, unique_given_names, unique_segment_names, 
                  unique_group_names, corr_results_save_dir)
    
    #Significance Test all group pairs
    test_corr_dist_sig_test(corr_dict, unique_given_names, 
                             unique_segment_names, unique_group_names, 
                             corr_results_save_dir)
    
    #Calculate indices unique to each correlation combo
    all_corr_dicts = calc_ind_dicts(corr_dict, unique_given_names, 
                                unique_segment_names, unique_group_names,
                                corr_cutoffs)
    np.save(os.path.join(corr_results_save_dir,'corr_cutoff_dict.npy'),\
            all_corr_dicts,allow_pickle=True)
    
    #Create plots
    plot_corr_cutoff_groups(all_corr_dicts, corr_dict, multiday_data_dict, 
                            unique_given_names, unique_segment_names, unique_group_names,
                            corr_cutoffs, colors, corr_results_save_dir)
           
def plot_all_dist(corr_dict, unique_given_names, 
                         unique_segment_names, unique_group_names, 
                         corr_results_save_dir, verbose=False):
    
    num_seg = len(unique_segment_names)
    num_groups = len(unique_group_names)
    
    #By Segment
    f_seg_all, ax_seg_all = plt.subplots(nrows = 2, ncols=num_seg,\
                                         figsize=(num_seg*4,2*4))
    max_density = 0
    for s_i, seg_name in enumerate(unique_segment_names):
        for g_i, g_name in enumerate(unique_group_names):
            g_data = []
            for gn_i, gn in enumerate(unique_given_names):
                g_data.extend(list(np.nanmean(corr_dict[gn][seg_name][g_name]['corr_vals_by_response'],1)))
            g_data = np.array(g_data)
            g_data = g_data[~np.isnan(g_data)]
            #Plot pdf
            hist_val = ax_seg_all[0,s_i].hist(g_data,bins=100,density=True,histtype='step',\
                                 label=g_name)
            if np.nanmax(hist_val[0]) > max_density:
                max_density = np.nanmax(hist_val[0])
            #Plot cdf
            ax_seg_all[1,s_i].hist(g_data,bins=100,density=True,cumulative=True,\
                                   histtype='step',label=g_name)
            if s_i == 0:
                ax_seg_all[0,s_i].set_ylabel('Probability Density')
                ax_seg_all[1,s_i].set_ylabel('Cumulative Density')
                ax_seg_all[0,s_i].legend(loc='upper left')
        ax_seg_all[0,s_i].set_title(seg_name)
        ax_seg_all[1,s_i].set_ylim([0,1])
        ax_seg_all[1,s_i].set_xlabel('Pearson Correlation')
    for s_i in range(num_seg):
        ax_seg_all[0,s_i].set_ylim([0,max_density])
    plt.suptitle('All Distributions')
    plt.tight_layout()
    f_seg_all.savefig(os.path.join(corr_results_save_dir,'Cross_Animal_Segment_Distributions.png'))
    f_seg_all.savefig(os.path.join(corr_results_save_dir,'Cross_Animal_Segment_Distributions.svg'))
    plt.close(f_seg_all)
    
    #By Group
    f_group_all, ax_group_all = plt.subplots(nrows = 2, ncols=num_groups,\
                                         figsize=(num_groups*4,2*4))
    max_density = 0
    for g_i, g_name in enumerate(unique_group_names):
        for s_i, seg_name in enumerate(unique_segment_names):
            s_data = []
            for gn_i, gn in enumerate(unique_given_names):
                s_data.extend(list(np.nanmean(corr_dict[gn][seg_name][g_name]['corr_vals_by_response'],1)))
            s_data = np.array(s_data)
            s_data = s_data[~np.isnan(s_data)]
            #Plot pdf
            hist_val = ax_group_all[0,g_i].hist(s_data,bins=100,density=True,histtype='step',\
                                 label=seg_name)
            if np.nanmax(hist_val[0]) > max_density:
                max_density = np.nanmax(hist_val[0])
            #Plot cdf
            ax_group_all[1,g_i].hist(s_data,bins=100,density=True,cumulative=True,\
                                   histtype='step',label=seg_name)
            if g_i == 0:
                ax_group_all[0,g_i].set_ylabel('Probability Density')
                ax_group_all[1,g_i].set_ylabel('Cumulative Density')
                ax_group_all[0,g_i].legend(loc='upper left')
        ax_group_all[0,g_i].set_title(g_name)
        ax_group_all[1,g_i].set_ylim([0,1])
        ax_group_all[1,g_i].set_xlabel('Pearson Correlation')
    for g_i in range(num_groups):
        ax_group_all[0,g_i].set_ylim([0,max_density])
    plt.suptitle('All Distributions')
    plt.tight_layout()
    f_group_all.savefig(os.path.join(corr_results_save_dir,'Cross_Animal_Group_Distributions.png'))
    f_group_all.savefig(os.path.join(corr_results_save_dir,'Cross_Animal_Group_Distributions.svg'))
    plt.close(f_group_all)
    
def test_corr_dist_sig_test(corr_dict, unique_given_names, 
                         unique_segment_names, unique_group_names, 
                         corr_results_save_dir, verbose=False):
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
    
    group_inds = np.arange(len(unique_group_names))
    g_pairs = list(combinations(list(group_inds),2))
    num_seg = len(unique_segment_names)
    seg_inds = np.arange(num_seg)
    seg_pairs = list(combinations(list(seg_inds),2))
    sig_save_dir = os.path.join(corr_results_save_dir,'Sig_Tests')
    if not os.path.isdir(sig_save_dir):
        os.mkdir(sig_save_dir)
    
    #Compare groups by segment
    by_seg_dir = os.path.join(sig_save_dir,'By_Segment')
    if not os.path.isdir(by_seg_dir):
        os.mkdir(by_seg_dir)
    for s_i, seg_name in enumerate(unique_segment_names):
        g_pair_sig_text = ["G1, G2, dir, T, KS"]
        for g_1, g_2 in g_pairs:
            g_1_name = unique_group_names[g_1]
            g_2_name = unique_group_names[g_2]
            g_pair_sig = g_1_name + ',' + g_2_name + ','
            g_1_data = []
            g_2_data = []
            for gn in unique_given_names:
                g_1_data.extend(list(np.nanmean(corr_dict[gn][seg_name][g_1_name]['corr_vals_by_response'],1)))
                g_2_data.extend(list(np.nanmean(corr_dict[gn][seg_name][g_2_name]['corr_vals_by_response'],1)))
            g_1_data = np.array(g_1_data)
            g_1_data = g_1_data[~np.isnan(g_1_data)]
            g_2_data = np.array(g_2_data)
            g_2_data = g_2_data[~np.isnan(g_2_data)]
            ks_result = ks_2samp(g_1_data,g_2_data,alternative='two-sided').pvalue
            tt_result = ttest_ind(g_1_data,g_2_data,alternative='two-sided').pvalue
            if ks_result <= 0.05:
                if np.nanmean(g_1_data) < np.nanmean(g_2_data):
                    g_pair_sig += '<,'
                elif np.nanmean(g_1_data) > np.nanmean(g_2_data):
                    g_pair_sig += '>,'
            else:
                g_pair_sig += '=,'
            g_pair_sig += str(np.round(tt_result,4)) + ','
            g_pair_sig += str(np.round(ks_result,4))
            g_pair_sig_text.append(g_pair_sig)
            #Plot data
            f_pair_data, ax_pair_data = plt.subplots(ncols=2,figsize=(10,5))
            #Density PDF
            ax_pair_data[0].axvline(0,linestyle='dashed',alpha=0.5,color='k',label='_')
            ax_pair_data[0].hist(g_1_data,bins=100,density=True,histtype='step',\
                                 label=g_1_name)
            ax_pair_data[0].hist(g_2_data,bins=100,density=True,histtype='step',\
                                 label=g_2_name)
            ax_pair_data[0].set_xlim([-1,1])
            ax_pair_data[0].set_xlabel('Pearson Correlation')
            ax_pair_data[0].set_ylabel('Density')
            ax_pair_data[0].set_title('Probability Density')
            ax_pair_data[0].legend(loc='upper left')
            #CDF
            ax_pair_data[1].axvline(0,linestyle='dashed',alpha=0.5,color='k',label='_')
            ax_pair_data[1].hist(g_1_data,bins=1000,density=True,cumulative=True,\
                                 histtype='step',label=g_1_name)
            ax_pair_data[1].hist(g_2_data,bins=1000,density=True,cumulative=True,\
                                 histtype='step',label=g_2_name)
            ax_pair_data[1].set_xlim([-1,1])
            ax_pair_data[1].set_xlabel('Pearson Correlation')
            ax_pair_data[1].set_ylabel('Cumulative Density')
            ax_pair_data[1].set_title('Cumulative Density')
            #Finish plot
            title = g_1_name + ' x ' + g_2_name
            g_sig_rewrite = g_pair_sig.split(',')
            plt.suptitle(title + '\n' + 'Dir ' + g_sig_rewrite[2] + \
                         '; TTp = ' + g_sig_rewrite[3] + \
                             ' KSp = ' + g_sig_rewrite[4])
            plt.tight_layout()
            f_pair_data.savefig(os.path.join(by_seg_dir,seg_name + '_' + ('_').join(title.split(' ')) + '.png'))
            f_pair_data.savefig(os.path.join(by_seg_dir,seg_name + '_' + ('_').join(title.split(' ')) + '.svg'))
            plt.close(f_pair_data)
                
        #Store sig data to csv
        csv_save_name = seg_name + '_significance.csv'
        with open(os.path.join(by_seg_dir,csv_save_name),'w') as file:
            writer = csv.writer(file)
            for row in g_pair_sig_text:
                writer.writerow(row)
                        
    #Compare groups across segments
    cross_seg_dir = os.path.join(sig_save_dir,'Across_Segments')
    if not os.path.isdir(cross_seg_dir):
        os.mkdir(cross_seg_dir)
    for g_i, g_name in enumerate(unique_group_names):
        seg_sig_text = ["S1, S2, dir, T, KS"]
        for s_1, s_2 in seg_pairs:
            s_1_name = unique_segment_names[s_1]
            s_2_name = unique_segment_names[s_2]
            s_pair_sig = s_1_name + ',' + s_2_name + ','
            s_1_data = []
            s_2_data = []
            for gn in unique_given_names:
                s_1_data.extend(list(np.nanmean(corr_dict[gn][s_1_name][g_name]['corr_vals_by_response'],1)))
                s_2_data.extend(list(np.nanmean(corr_dict[gn][s_2_name][g_name]['corr_vals_by_response'],1)))
            s_1_data = np.array(s_1_data)
            s_1_data = s_1_data[~np.isnan(s_1_data)]
            s_2_data = np.array(s_2_data)
            s_2_data = s_2_data[~np.isnan(s_2_data)]
            ks_result = ks_2samp(s_1_data,s_2_data,alternative='two-sided').pvalue
            tt_result = ttest_ind(s_1_data,s_2_data,alternative='two-sided').pvalue
            if ks_result <= 0.05:
                if np.nanmean(s_1_data) < np.nanmean(s_2_data):
                    s_pair_sig += '<,'
                elif np.nanmean(s_1_data) > np.nanmean(s_2_data):
                    s_pair_sig += '>,'
            else:
                s_pair_sig += '=,'
            s_pair_sig += str(np.round(tt_result,4)) + ','
            s_pair_sig += str(np.round(ks_result,4))
            seg_sig_text.append(s_pair_sig)
            #Plot data
            f_pair_data, ax_pair_data = plt.subplots(ncols=2,figsize=(10,5))
            #Density PDF
            ax_pair_data[0].axvline(0,linestyle='dashed',alpha=0.5,color='k',label='_')
            ax_pair_data[0].hist(s_1_data,bins=100,density=True,histtype='step',\
                                 label=s_1_name)
            ax_pair_data[0].hist(s_2_data,bins=100,density=True,histtype='step',\
                                 label=s_2_name)
            ax_pair_data[0].set_xlim([-1,1])
            ax_pair_data[0].set_xlabel('Pearson Correlation')
            ax_pair_data[0].set_ylabel('Density')
            ax_pair_data[0].set_title('Probability Density')
            ax_pair_data[0].legend(loc='upper left')
            #CDF
            ax_pair_data[1].axvline(0,linestyle='dashed',alpha=0.5,color='k',label='_')
            ax_pair_data[1].hist(s_1_data,bins=1000,density=True,cumulative=True,\
                                 histtype='step',label=s_1_name)
            ax_pair_data[1].hist(s_2_data,bins=1000,density=True,cumulative=True,\
                                 histtype='step',label=s_2_name)
            ax_pair_data[1].set_xlim([-1,1])
            ax_pair_data[1].set_xlabel('Pearson Correlation')
            ax_pair_data[1].set_ylabel('Cumulative Density')
            ax_pair_data[1].set_title('Cumulative Density')
            #Finish plot
            title = s_1_name + ' x ' + s_2_name
            s_sig_rewrite = s_pair_sig.split(',')
            plt.suptitle(title + '\n' + 'Dir ' + s_sig_rewrite[2] + \
                         '; TTp = ' + s_sig_rewrite[3] + \
                             ' KSp = ' + s_sig_rewrite[4])
            plt.tight_layout()
            f_pair_data.savefig(os.path.join(cross_seg_dir,g_name + '_' + ('_').join(title.split(' ')) + '.png'))
            f_pair_data.savefig(os.path.join(cross_seg_dir,g_name + '_' + ('_').join(title.split(' ')) + '.svg'))
            plt.close(f_pair_data)
                
        #Store sig data to csv
        csv_save_name = ('_').join(g_name.split(' ')) + '_significance.csv'
        with open(os.path.join(cross_seg_dir,csv_save_name),'w') as file:
            writer = csv.writer(file)
            for row in seg_sig_text:
                writer.writerow(row)
                        
    
    
    #Compare groups against null
    
    
def calc_ind_dicts(corr_dict, unique_given_names, unique_segment_names, 
                   unique_group_names, corr_cutoffs, verbose=False):
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
    num_seg = len(unique_segment_names)
    num_groups = len(unique_group_names)
    
    #Collect all indices above a cutoff
    all_corr_dicts = dict()
    for s_i, seg_name in enumerate(unique_segment_names):
        all_corr_dicts[seg_name] = dict()
        for g_i, group_name in enumerate(unique_group_names):
            all_corr_dicts[seg_name][group_name] = dict()
            for gn_i, gn in enumerate(unique_given_names):
                #Collect true data
                try:
                    #Grab animal data
                    data = corr_dict[gn][seg_name][group_name]['corr_vals_by_response']
                    num_dev, num_deliv = np.shape(data)
                    avg_corr = np.nanmean(data,1)
                    #Collect indices by cutoff
                    all_corr_dicts[seg_name][group_name][gn] = [np.where(avg_corr >= cc)[0] for cc in corr_cutoffs]
                except:
                    errormsg = 'Missing data for ' + seg_name + ' ' + \
                        group_name + ' ' + gn
                    if verbose == True:
                        print(errormsg)

    return all_corr_dicts

def plot_corr_cutoff_groups(all_corr_dicts, corr_dict, multiday_data_dict, 
                        unique_given_names, unique_segment_names, unique_group_names,
                        corr_cutoffs, colors, corr_results_save_dir,
                        verbose = False):
    
    """Plot number of events above correlation cutoff for each taste"""
    
    warnings.filterwarnings('ignore')
    #non_none_tastes = [taste for taste in unique_taste_names if taste != 'none_0']
    cutoff_val = 0.5
    cc_0_5_ind = np.where(corr_cutoffs >= cutoff_val)[0][0]
    cc_top_half_ind = np.where(corr_cutoffs >= cutoff_val)[0][0]
    num_anim = len(unique_given_names)
    num_cutoff = len(corr_cutoffs)
    num_seg = len(unique_segment_names)
    num_groups = len(unique_group_names)
    indiv_anim_plot_save_dir = os.path.join(corr_results_save_dir,'Individual_Animals')
    if not os.path.isdir(indiv_anim_plot_save_dir):
        os.mkdir(indiv_anim_plot_save_dir)
    cross_anim_plot_save_dir = os.path.join(corr_results_save_dir,'All_Animals')
    if not os.path.isdir(cross_anim_plot_save_dir):
        os.mkdir(cross_anim_plot_save_dir)
    
    
    #Store data
    all_rates = np.nan*np.ones((num_seg,num_groups,num_anim,num_cutoff))
    
    #Plot all corr values          
    f_rate, ax_rate = plt.subplots(nrows = 1, ncols = num_seg, figsize=(num_seg*4,4),\
                                sharex = True, sharey = True)
    f_rate_zoom, ax_rate_zoom = plt.subplots(nrows = 1, ncols = num_seg, \
                                             figsize=(num_seg*4,4),\
                                                 sharex = True, sharey = True)
    f_rate_box, ax_rate_box = plt.subplots(nrows = 1, ncols = num_seg, \
                                           figsize=(num_seg*4,num_groups*4),\
                                               sharex = True, sharey = True)
    for s_i, seg_name in enumerate(unique_segment_names):
        seg_indiv_dir = os.path.join(indiv_anim_plot_save_dir,seg_name)
        if not os.path.isdir(seg_indiv_dir):
            os.mkdir(seg_indiv_dir)
        for gp_i, g_name in enumerate(unique_group_names):
            animal_inds = all_corr_dicts[seg_name][g_name]
            animal_true_rates = np.nan*np.ones((num_anim,num_cutoff))
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
                        ' ' + g_name + ' data.'
                    if verbose == True:
                        print(errormsg)
                #Plot individual animal rates in separate folder
                f_anim = plt.figure()
                plt.title(g_n)
                plt.plot(corr_cutoffs, animal_true_rates[gn_i,:], label='True')
                plt.ylabel('Rate (Hz)')
                plt.xlabel('Pearson Correlation Cutoff')
                plt.legend(loc='upper right')
                f_anim.savefig(os.path.join(seg_indiv_dir,g_name+'_'+g_n+'_rate_taste_by_cutoff.png'))
                f_anim.savefig(os.path.join(seg_indiv_dir,g_name+'_'+g_n+'_rate_taste_by_cutoff.svg'))
                plt.close(f_anim)
                f_anim_zoom = plt.figure()
                plt.title(g_n)
                plt.plot(corr_cutoffs[cc_0_5_ind:], animal_true_rates[gn_i,cc_0_5_ind:], label='True')
                plt.ylabel('Rate (Hz)')
                plt.xlabel('Pearson Correlation Cutoff')
                plt.legend(loc='upper right')
                f_anim_zoom.savefig(os.path.join(seg_indiv_dir,g_name+'_'+g_n+'_rate_taste_by_cutoff_zoom.png'))
                f_anim_zoom.savefig(os.path.join(seg_indiv_dir,g_name+'_'+g_n+'_rate_taste_by_cutoff_zoom.svg'))
                plt.close(f_anim_zoom)
            #Average rates
            anim_true_avg_rate = np.nanmean(animal_true_rates,0)
            all_rates[s_i,gp_i,:,:] = animal_true_rates
            ax_rate[s_i].plot(corr_cutoffs,anim_true_avg_rate,label=g_name)
            ax_rate_zoom[s_i].plot(corr_cutoffs[cc_top_half_ind:],anim_true_avg_rate[cc_top_half_ind:],label=g_name)
            #0.25 rates box plot
            anim_50_rates = animal_true_rates[:,cc_0_5_ind].flatten()
            ax_rate_box[s_i].boxplot(list(anim_50_rates),
                                          positions=[gp_i+1],
                                          showmeans=False,showfliers=False)
            x_locs = gp_i + 1 + 0.1*np.random.randn(num_anim)
            ax_rate_box[s_i].scatter(x_locs,anim_50_rates,alpha=0.5,color='g')
        ax_rate[s_i].set_title(seg_name)
        ax_rate_zoom[s_i].set_title(seg_name)
        ax_rate_box[s_i].set_title(seg_name)
        ax_rate[s_i].set_xlabel('Min. Correlation Cutoff')
        ax_rate_zoom[s_i].set_xlabel('Min. Correlation Cutoff')
        ax_rate_box[s_i].set_xlabel('Group')
        ax_rate_box[s_i].set_xticks(np.arange(num_groups)+1,unique_group_names,
                                         horizontalalignment='right',rotation = 45)
        if s_i == 0:
            ax_rate[s_i].set_ylabel('Rate (Hz)')
            ax_rate_zoom[s_i].set_ylabel('Rate (Hz)')
            ax_rate_box[s_i].set_ylabel('Rate (Hz)')
    
    np.save(os.path.join(corr_results_save_dir,'corr_cutoff_rates.npy'),all_rates,allow_pickle=True)
    
    plt.figure(f_rate)
    ax_rate[0].legend(loc='upper right')
    ax_rate[0].set_xticks(np.arange(0,1.25,0.25))
    plt.suptitle('Rate of Events Above Cutoff')
    plt.tight_layout()
    f_rate.savefig(os.path.join(cross_anim_plot_save_dir,'rate_group_by_cutoff.png'))
    f_rate.savefig(os.path.join(cross_anim_plot_save_dir,'rate_group_by_cutoff.svg'))
    plt.close(f_rate)
    plt.figure(f_rate_zoom)
    ax_rate_zoom[0].legend(loc='upper right')
    ax_rate_zoom[0].set_xticks(np.arange(cutoff_val,1.25,0.25))
    plt.suptitle('Rate of Events Above Zoom Cutoff')
    plt.tight_layout()
    f_rate_zoom.savefig(os.path.join(cross_anim_plot_save_dir,'rate_group_by_cutoff_zoom.png'))
    f_rate_zoom.savefig(os.path.join(cross_anim_plot_save_dir,'rate_group_by_cutoff_zoom.svg'))
    plt.close(f_rate_zoom)
    plt.figure(f_rate_box)
    plt.suptitle('Rate of Events Above 0.25 Cutoff')
    plt.tight_layout()
    f_rate_box.savefig(os.path.join(cross_anim_plot_save_dir,'rate_group_at_cutoff_box.png'))
    f_rate_box.savefig(os.path.join(cross_anim_plot_save_dir,'rate_group_at_cutoff_box.svg'))
    plt.close(f_rate_box)
    
def plot_corr_rate_diffs(all_rates, unique_given_names, unique_segment_names, 
                         unique_group_names, corr_cutoffs, 
                         cross_anim_plot_save_dir, verbose = False):
    
    num_seg = len(unique_segment_names)
    num_anim = len(unique_given_names)
    num_groups = len(unique_group_names)
    #all_rates shape is [num_seg,num_groups,num_anim,num_cutoff]
    
    cutoff_vals = [0.25,0.5,0.75]
    cutoff_inds = [np.where(corr_cutoffs >= cv)[0][0] for cv in cutoff_vals]
    cutoff_zip = list(zip(cutoff_vals, cutoff_inds))
    
    #Differences between groups at particular cutoff value
    seg_save_dir = os.path.join(cross_anim_plot_save_dir,'by_segment')
    if not os.path.isdir(seg_save_dir):
        os.mkdir(seg_save_dir)
    group_inds = np.arange(num_groups)
    g_pairs = list(combinations(list(group_inds),2))
    num_pairs = len(g_pairs)
    sqrt_pairs = np.ceil(np.sqrt(num_pairs)).astype('int')
    sqr_reference = np.reshape(np.arange(np.square(sqrt_pairs)),(sqrt_pairs,sqrt_pairs))
    for s_i, seg_name in enumerate(unique_segment_names):
        for cv, cv_i in cutoff_zip:
            sup_title = 'Cutoff Value = ' + str(cv)
            f_pair, ax_pair = plt.subplots(nrows = sqrt_pairs, ncols = sqrt_pairs, \
                                  sharey = True, figsize=(sqrt_pairs*4,sqrt_pairs*4))
            
            f_diff, ax_diff = plt.subplots(figsize = (10,5))
            diff_ind = 0
            diff_xtick_labels = []
            for pair_i, (g_1, g_2) in enumerate(g_pairs):
                g_1_name = unique_group_names[g_1]
                g_2_name = unique_group_names[g_2]
                pair_loc = np.where(sqr_reference == pair_i)
                r_i = pair_loc[0][0]
                c_i = pair_loc[1][0]
                #Get data at cutoff vals
                data_1 = np.squeeze(all_rates[s_i,g_1,:,cv_i])
                data_2 = np.squeeze(all_rates[s_i,g_2,:,cv_i])
                max_val = np.max([np.max(data_1),np.max(data_2)])
                #Plot the boxplots
                ax_pair[r_i,c_i].axhline(0,alpha=0.5,linestyle='dashed',color='k')
                ax_pair[r_i,c_i].boxplot([data_1,data_2], positions = [0,1])
                ax_pair[r_i,c_i].scatter(np.zeros(num_anim),data_1,alpha=0.5,\
                                         color='g')
                ax_pair[r_i,c_i].scatter(np.ones(num_anim),data_2,alpha=0.5,\
                                         color='g')
                ax_pair[r_i,c_i].set_xticks(np.arange(2),[g_1_name,g_2_name],\
                                            rotation=45)
                #Run a ttest
                ttpval = ttest_ind(data_1,data_2).pvalue
                if ttpval <= 0.05:
                    ax_pair[r_i,c_i].plot([0,1],[1.05*max_val,1.05*max_val],\
                                          color='k')
                    ax_pair[r_i,c_i].text(0.5,1.1*max_val,'*',color='k')
                    #Add to difference plot
                    ax_diff.boxplot([data_1 - data_2],positions = [diff_ind])
                    ax_diff.scatter(diff_ind*np.ones(num_anim),data_1 - data_2,\
                                    alpha=0.5,color='g')
                    diff_xtick_labels.extend([g_1_name + ' - ' + g_2_name])
                    diff_ind += 1
                ax_pair[r_i,c_i].set_title(g_1_name + ' x ' + g_2_name)
                if c_i == 0:
                    ax_pair[r_i,c_i].set_ylabel('Rate (Hz)')
            plt.figure(f_pair)
            plt.suptitle(sup_title)
            plt.tight_layout()
            f_pair.savefig(os.path.join(seg_save_dir,seg_name + \
                    '_rate_pair_' + ('_').join(str(cv).split('.')) + '.png'))
            f_pair.savefig(os.path.join(seg_save_dir,seg_name + \
                    '_rate_pair_' + ('_').join(str(cv).split('.')) + '.svg'))
            plt.close(f_pair)
            plt.figure(f_diff)
            ax_diff.set_xlabel('Rate Difference (Hz)')
            ax_diff.set_xticks(np.arange(diff_ind),diff_xtick_labels,\
                               rotation = 45)
            ax_diff.set_title('Significantly different pairs')
            plt.suptitle(sup_title)
            plt.tight_layout()
            f_diff.savefig(os.path.join(seg_save_dir,seg_name + \
                    '_rate_sig_diffs_' + ('_').join(str(cv).split('.')) + '.png'))
            f_diff.savefig(os.path.join(seg_save_dir,seg_name + \
                    '_rate_sig_diffs_' + ('_').join(str(cv).split('.')) + '.svg'))
            plt.close(f_diff)
    
    #Differences between segments
    group_save_dir = os.path.join(cross_anim_plot_save_dir,'by_group_pair')
    if not os.path.isdir(group_save_dir):
        os.mkdir(group_save_dir)
    seg_inds = np.arange(num_seg)
    seg_pairs = list(combinations(list(seg_inds),2))
    num_pairs = len(seg_pairs)
    sqrt_pairs = np.ceil(np.sqrt(num_pairs)).astype('int')
    sqr_reference = np.reshape(np.arange(np.square(sqrt_pairs)),(sqrt_pairs,sqrt_pairs))
    for g_i, g_name in enumerate(unique_group_names):
        for cv, cv_i in cutoff_zip:
            sup_title = 'Cutoff Value = ' + str(cv)
            f_pair, ax_pair = plt.subplots(nrows = sqrt_pairs, ncols = sqrt_pairs, \
                                  sharey = True, figsize=(sqrt_pairs*4,sqrt_pairs*4))
            
            f_diff, ax_diff = plt.subplots(figsize = (10,5))
            diff_ind = 0
            diff_xtick_labels = []
            for pair_i, (s_1, s_2) in enumerate(seg_pairs):
                seg_name_1 = unique_segment_names[s_1]
                seg_name_2 = unique_segment_names[s_2]
                pair_loc = np.where(sqr_reference == pair_i)
                r_i = pair_loc[0][0]
                c_i = pair_loc[1][0]
                #Get data at cutoff vals
                data_1 = np.squeeze(all_rates[s_1,g_i,:,cv_i])
                data_2 = np.squeeze(all_rates[s_2,g_i,:,cv_i])
                max_val = np.max([np.max(data_1),np.max(data_2)])
                #Plot the boxplots
                ax_pair[r_i,c_i].axhline(0,alpha=0.5,linestyle='dashed',color='k')
                ax_pair[r_i,c_i].boxplot([data_1,data_2], positions = [0,1])
                ax_pair[r_i,c_i].scatter(np.zeros(num_anim),data_1,alpha=0.5,\
                                         color='g')
                ax_pair[r_i,c_i].scatter(np.ones(num_anim),data_2,alpha=0.5,\
                                         color='g')
                ax_pair[r_i,c_i].set_xticks(np.arange(2),[seg_name_1,seg_name_2],\
                                            rotation=45)
                #Run a ttest
                ttpval = ttest_ind(data_1,data_2).pvalue
                if ttpval <= 0.05:
                    ax_pair[r_i,c_i].plot([0,1],[1.05*max_val,1.05*max_val],\
                                          color='k')
                    ax_pair[r_i,c_i].text(0.5,1.1*max_val,'*',color='k')
                    #Add to difference plot
                    ax_diff.boxplot([data_1 - data_2],positions = [diff_ind])
                    ax_diff.scatter(diff_ind*np.ones(num_anim),data_1 - data_2,\
                                    alpha=0.5,color='g')
                    diff_xtick_labels.extend([seg_name_1 + ' - ' + seg_name_2])
                    diff_ind += 1
                ax_pair[r_i,c_i].set_title(seg_name_1 + ' x ' + seg_name_2)
                if c_i == 0:
                    ax_pair[r_i,c_i].set_ylabel('Rate (Hz)')
            plt.figure(f_pair)
            plt.suptitle(sup_title)
            plt.tight_layout()
            f_pair.savefig(os.path.join(cross_anim_plot_save_dir,g_name + \
                    '_rate_pair_' + ('_').join(str(cv).split('.')) + '.png'))
            f_pair.savefig(os.path.join(cross_anim_plot_save_dir,g_name + \
                    '_rate_pair_' + ('_').join(str(cv).split('.')) + '.svg'))
            plt.close(f_pair)
            plt.figure(f_diff)
            ax_diff.set_xlabel('Rate Difference (Hz)')
            ax_diff.set_xticks(np.arange(diff_ind),diff_xtick_labels,\
                               rotation = 45)
            ax_diff.set_title('Significantly different pairs')
            plt.suptitle(sup_title)
            plt.tight_layout()
            f_diff.savefig(os.path.join(cross_anim_plot_save_dir,g_name + \
                    '_rate_sig_diffs_' + ('_').join(str(cv).split('.')) + '.png'))
            f_diff.savefig(os.path.join(cross_anim_plot_save_dir,g_name + \
                    '_rate_sig_diffs_' + ('_').join(str(cv).split('.')) + '.svg'))
            plt.close(f_diff)
    
    
        
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
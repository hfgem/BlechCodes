#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:43:21 2024

@author: Hannah Germaine

A collection of function dedicated to analyzing and plotting the results of a
sliding bin correlation analysis of rest intervals.
"""

import os
import itertools
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind

def slide_corr_vs_rate(corr_slide_stats,bin_times,bin_pop_fr,num_cp,plot_dir,save_dir,
                       segment_names,dig_in_names,segments_to_analyze=[]):
    """Calculate and plot the correlation between sliding bin correlation 
    calculations and the population rate"""
    
    num_tastes = len(dig_in_names)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(len(segment_names))
    num_segments = len(segments_to_analyze)
    
    max_corr_val = 0
    min_corr_val = 0
    if os.path.isfile(os.path.join(save_dir,'popfr_corr_storage.npy')):
        popfr_corr_storage = np.load(os.path.join(save_dir,'popfr_corr_storage.npy'),allow_pickle=True).item()
        #Plot
        f, ax = plt.subplots(nrows = num_segments, ncols = num_tastes, figsize = (4*num_tastes,4*num_segments))
        for s_i, s_ind in tqdm.tqdm(enumerate(segments_to_analyze)):
            seg_name = segment_names[s_ind]
            for t_i in range(num_tastes):
                taste_name = dig_in_names[t_i]
                popfr_corr_vals = popfr_corr_storage[seg_name][taste_name]
                if np.min(popfr_corr_vals) < min_corr_val:
                    min_corr_val = np.min(popfr_corr_vals)
                if np.max(popfr_corr_vals) > max_corr_val:
                    max_corr_val = np.max(popfr_corr_vals)
                #Plot correlation across time
                #mean_corr_by_changepoint = np.nanmean(popfr_corr_vals,0)
                #std_corr_by_changepoint = np.nanstd(popfr_corr_vals,0)
                #ax[s_i,t_i].plot(np.arange(num_cp),mean_corr_by_changepoint)
                #ax[s_i,t_i].fill_between(np.arange(num_cp),mean_corr_by_changepoint-std_corr_by_changepoint,mean_corr_by_changepoint+std_corr_by_changepoint,alpha=0.3)
                for cp_i in range(num_cp):
                    corr_dataset = popfr_corr_vals[:,cp_i]
                    ax[s_i,t_i].violinplot(list(corr_dataset[~np.isnan(corr_dataset)]),[cp_i],showmeans=True)
                ax[s_i,t_i].set_title(seg_name + ' x ' + taste_name)
                ax[s_i,t_i].set_ylabel('Pearson Correlation')
                ax[s_i,t_i].set_xlabel('Epoch')
        for s_i in range(len(segments_to_analyze)):
            for t_i in range(num_tastes):
                ax[s_i,t_i].set_ylim([min_corr_val + 0.1*min_corr_val,max_corr_val + 0.1*max_corr_val])
        #Finish figure and save
        plt.suptitle('Population Rate x Bin Correlation')
        plt.tight_layout()
        f.savefig(os.path.join(plot_dir,'pop_rate_x_bin_corr_epochs.png'))
        f.savefig(os.path.join(plot_dir,'pop_rate_x_bin_corr_epochs.svg'))
        plt.close(f)
    else:
        #Plot for each taste and segment the correlation by epoch
        f, ax = plt.subplots(nrows = num_segments, ncols = num_tastes, figsize = (4*num_tastes,4*num_segments))
        popfr_corr_storage = dict()
        for s_i, s_ind in tqdm.tqdm(enumerate(segments_to_analyze)):
            seg_name = segment_names[s_ind]
            seg_pop_fr = np.array(bin_pop_fr[s_i])
            popfr_corr_storage[seg_name] = dict()
            for t_i in range(num_tastes):
                taste_name = dig_in_names[t_i]
                st_corr = corr_slide_stats[seg_name][t_i]['pop_vec_data_storage']
                num_times, num_deliv, _ = np.shape(st_corr)
                #Calculate correlation between population fr and correlations to taste
                popfr_corr_vals = np.zeros((num_deliv,num_cp))
                popfr_inf_inds = np.where(np.isinf(seg_pop_fr))[0]
                popfr_nan_inds = np.where(np.isnan(seg_pop_fr))[0]
                for d_i in range(num_deliv):
                    for cp_i in range(num_cp):
                        cp_corr_data = np.array(st_corr[:,d_i,cp_i]).squeeze()
                        cp_inf_inds = np.where(np.isinf(cp_corr_data))[0]
                        cp_nan_inds = np.where(np.isnan(cp_corr_data))[0]
                        remaining_inds = np.setdiff1d(np.arange(len(seg_pop_fr)),np.concatenate((popfr_inf_inds,popfr_nan_inds,cp_inf_inds,cp_nan_inds)))
                        #Pearson corr calc
                        # cp_corr_mean = np.nanmean(cp_corr_data)
                        # popfr_mean = np.nanmean(seg_pop_fr)
                        # numerator = np.nansum((cp_corr_data - cp_corr_mean)*(seg_pop_fr - popfr_mean))
                        # denominator = np.sqrt(np.nansum((cp_corr_data - cp_corr_mean)**2)*np.nansum((seg_pop_fr - popfr_mean)**2))
                        # popfr_corr_vals[d_i,cp_i] = numerator/denominator
                        if len(remaining_inds) > 2:
                            corr_result = pearsonr(cp_corr_data[remaining_inds],seg_pop_fr[remaining_inds],alternative='two-sided')[0]
                        else:
                            corr_result = np.nan
                        popfr_corr_vals[d_i,cp_i] = corr_result
                popfr_corr_storage[seg_name][taste_name] = popfr_corr_vals
                if np.min(popfr_corr_vals) < min_corr_val:
                    min_corr_val = np.min(popfr_corr_vals)
                if np.max(popfr_corr_vals) > max_corr_val:
                    max_corr_val = np.max(popfr_corr_vals)
                for cp_i in range(num_cp):
                    corr_dataset = popfr_corr_vals[:,cp_i]
                    ax[s_i,t_i].violinplot(list(corr_dataset[~np.isnan(corr_dataset)]),[cp_i],showmeans=True)
                ax[s_i,t_i].set_title(seg_name + ' x ' + taste_name)
                ax[s_i,t_i].set_ylabel('Pearson Correlation')
                ax[s_i,t_i].set_xlabel('Epoch')
        for s_i in range(len(segments_to_analyze)):
            for t_i in range(num_tastes):
                ax[s_i,t_i].set_ylim([min_corr_val + 0.1*min_corr_val,max_corr_val + 0.1*max_corr_val])
        #Finish figure and save
        plt.suptitle('Population Rate x Bin Correlation')
        plt.tight_layout()
        f.savefig(os.path.join(plot_dir,'pop_rate_x_bin_corr_epochs.png'))
        f.savefig(os.path.join(plot_dir,'pop_rate_x_bin_corr_epochs.svg'))
        plt.close(f)
        #Save dictionary
        np.save(os.path.join(save_dir,'popfr_corr_storage.npy'),popfr_corr_storage,True)

    #Plot for each epoch and segment the correlation by taste
    f, ax = plt.subplots(nrows = num_segments, ncols = num_cp, figsize = (4*num_tastes,4*num_segments))
    for s_i, s_ind in tqdm.tqdm(enumerate(segments_to_analyze)):
        seg_name = segment_names[s_ind]
        seg_pop_fr = np.array(bin_pop_fr[s_i])
        for cp_i in range(num_cp):
            #Plot individual taste distributions
            for t_i in range(num_tastes):
                taste_name = dig_in_names[t_i]
                st_corr = corr_slide_stats[seg_name][t_i]['pop_vec_data_storage']
                num_times, num_deliv, num_cp = np.shape(st_corr)
                popfr_corr_vals = popfr_corr_storage[seg_name][taste_name][:,cp_i].squeeze()
                ax[s_i,cp_i].boxplot(popfr_corr_vals[~np.isnan(popfr_corr_vals)],positions = [t_i+1])
                #Calculate if distribution is significantly above 0
                fifth_percentile = np.percentile(popfr_corr_vals[~np.isnan(popfr_corr_vals)],5)
                if 0 < fifth_percentile:
                    ax[s_i,cp_i].scatter(t_i,max_corr_val,marker='*',color='k',s=5)
            #Plot pairwise significances
            taste_pairs = list(itertools.combinations(np.arange(num_tastes), 2))
            sig_max_corr_val = max_corr_val
            for tp_i in range(len(taste_pairs)):
                t_0 = taste_pairs[tp_i][0]
                t_0_name = dig_in_names[t_0]
                t_1 = taste_pairs[tp_i][1]
                t_1_name = dig_in_names[t_1]
                t_0_corr_vals = popfr_corr_storage[seg_name][t_0_name][:,cp_i].squeeze()
                t_1_corr_vals = popfr_corr_storage[seg_name][t_1_name][:,cp_i].squeeze()
                stat = ttest_ind(t_0_corr_vals[~np.isnan(t_0_corr_vals)],t_1_corr_vals[~np.isnan(t_1_corr_vals)])
                if stat[1] < 0.05:
                    sig_max_corr_val = sig_max_corr_val + 0.05*sig_max_corr_val
                    ax[s_i,cp_i].plot([t_0+1,t_1+1],[sig_max_corr_val,sig_max_corr_val])
                    sig_max_corr_val = sig_max_corr_val + 0.05*sig_max_corr_val
                    ax[s_i,cp_i].scatter(t_0+1 + (t_1-t_0)/2,sig_max_corr_val,marker='*',color='k')
            ax[s_i,cp_i].set_title(seg_name + ' x Epoch ' + str(cp_i))
            ax[s_i,cp_i].set_xticklabels(dig_in_names)
            ax[s_i,cp_i].set_ylabel('Pearson Correlations')
            ax[s_i,cp_i].set_xlabel('Taste')
            ax[s_i,cp_i].set_ylim([min_corr_val + 0.2*min_corr_val,max_corr_val + 0.2*max_corr_val])
            ax[s_i,cp_i].set_ylim([min_corr_val + 0.2*min_corr_val,max_corr_val + 0.2*max_corr_val])
    #Finish figure and save
    plt.suptitle('Population Rate x Bin Correlation')
    plt.tight_layout()
    f.savefig(os.path.join(plot_dir,'pop_rate_x_bin_corr_tastes.png'))
    f.savefig(os.path.join(plot_dir,'pop_rate_x_bin_corr_tastes.svg'))
    plt.close(f)
    
    #Plot the distribution means on the same axes as trends but by taste
    min_mean = 0
    max_mean = 0
    f, ax = plt.subplots(ncols=num_tastes,figsize=(4*num_tastes,4))
    for s_i, s_ind in tqdm.tqdm(enumerate(segments_to_analyze)):
        seg_name = segment_names[s_ind]
        seg_pop_fr = np.array(bin_pop_fr[s_i])
        for t_i in range(num_tastes):
            taste_name = dig_in_names[t_i]
            popfr_corr_vals = popfr_corr_storage[seg_name][taste_name] #num_deliv x num_cp
            mean_vals = np.nanmean(popfr_corr_vals,0)
            if np.min(mean_vals) < min_mean:
                min_mean = np.min(mean_vals)
            if np.max(mean_vals) > max_mean:
                max_mean = np.max(mean_vals)
            ax[t_i].plot(np.arange(num_cp),mean_vals,label=seg_name)
    for t_i in range(num_tastes):
        ax[t_i].legend(loc='upper left')
        ax[t_i].axhline(0,alpha=0.2,linestyle='dashed',color='k')
        ax[t_i].set_ylim([min_mean - np.abs(0.1*min_mean), max_mean + 0.1*max_mean])
        ax[t_i].set_xticks(np.arange(num_cp))
        ax[t_i].set_xlabel('Epoch Index')
        ax[t_i].set_ylabel('Mean Pearson Correlation')
        ax[t_i].set_title(dig_in_names[t_i])
    plt.suptitle('Mean Correlation of Population Rate to Bin Taste Correlation')
    plt.tight_layout()
    f.savefig(os.path.join(plot_dir,'mean_pop_rate_x_bin_corr_tastes.png'))
    f.savefig(os.path.join(plot_dir,'mean_pop_rate_x_bin_corr_tastes.svg'))
    plt.close(f)
    
    return popfr_corr_storage

def top_corr_rate_dist(corr_slide_stats,bin_times,bin_pop_fr,num_cp,plot_dir,
                       segment_names,dig_in_names,segments_to_analyze=[]):
    """This function pulls out the top 90th+ percentile of correlation value 
    bins across segments and looks at the population firing rate distribution.
    
    INPUTS:
        - 
    OUTPUTS:
        - 
    
    """
    
    num_tastes = len(dig_in_names)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(len(segment_names))
    num_segments = len(segments_to_analyze)
    
    for s_i, s_ind in tqdm.tqdm(enumerate(segments_to_analyze)):
        seg_name = segment_names[s_ind]
        seg_pop_fr = np.array(bin_pop_fr[s_i])
        popfr_corr_storage[seg_name] = dict()
        for t_i in range(num_tastes):
            taste_name = dig_in_names[t_i]
            st_corr = corr_slide_stats[seg_name][t_i]['pop_vec_data_storage']
            num_times, num_deliv, _ = np.shape(st_corr)
            
            


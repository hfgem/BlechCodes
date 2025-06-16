#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:59:01 2025

@author: Hannah Germaine

A collection of functions dedicated to testing deviations' correlations and
decodes against multiple days of taste responses.
"""

import os
import csv
import time
import tqdm
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import functions.decode_parallel as dp
from matplotlib import colormaps, cm
from scipy.stats import pearsonr, ks_2samp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as gmm
from multiprocess import Pool


def multiday_dev_analysis(save_dir,all_dig_in_names,tastant_fr_dist_pop,
                          taste_num_deliv,max_hz_pop,tastant_fr_dist_z_pop,
                          max_hz_z_pop,min_hz_z_pop,max_num_cp,segment_dev_rasters,
                          segment_dev_times,segment_dev_fr_vecs,segment_dev_fr_vecs_zscore,
                          segments_to_analyze, segment_times, segment_spike_times,
                          bin_dt,segment_names_to_analyze):
    """
    This function serves as the main function which calls all others for 
    analyses of deviation events from the train day in comparison to taste
    responses across days.
    """
    
    corr_dir = os.path.join(save_dir,'Correlations')
    if not os.path.isdir(corr_dir):
        os.mkdir(corr_dir)
        
    #Variables
    num_seg = len(segment_dev_fr_vecs)
    num_neur = len(segment_dev_fr_vecs[0][0])
    num_tastes = len(all_dig_in_names)
    
    #Now go through segments and their deviation events and compare
    for s_i in range(num_seg):
        seg_name = segment_names_to_analyze[s_i]
        dev_rast = segment_dev_rasters[s_i]
        dev_times = segment_dev_times[s_i]
        dev_fr_vecs = segment_dev_fr_vecs[s_i]
        dev_fr_vecs_z = segment_dev_fr_vecs_zscore[s_i]
        
        #Run correlation analyses
        correlate_dev_to_taste(num_neur,all_dig_in_names,tastant_fr_dist_pop,
                                    taste_num_deliv,max_hz_pop,max_num_cp,dev_rast,
                                    dev_times,dev_fr_vecs,seg_name,corr_dir)
        correlate_dev_to_taste_zscore(num_neur,all_dig_in_names,tastant_fr_dist_z_pop,
                                          taste_num_deliv,max_hz_z_pop,min_hz_z_pop,
                                          max_num_cp,dev_rast,dev_times,dev_fr_vecs_z,
                                          seg_name,corr_dir)
        
        
def multiday_null_dev_analysis(save_dir,all_dig_in_names,tastant_fr_dist_pop,
                          taste_num_deliv,max_hz_pop,tastant_fr_dist_z_pop,
                          max_hz_z_pop,min_hz_z_pop,max_num_cp,null_dev_rasters,
                          null_dev_times,null_segment_dev_fr_vecs,
                          null_segment_dev_fr_vecs_zscore,segments_to_analyze, 
                          segment_times_to_analyze_reshaped, 
                          segment_spike_times_to_analyze,
                          bin_dt,segment_names_to_analyze):
    """
    This function serves as the main function which calls all others for 
    analyses of deviation events from the train day in comparison to taste
    responses across days.
    """
    
    corr_dir = os.path.join(save_dir,'Correlations')
    if not os.path.isdir(corr_dir):
        os.mkdir(corr_dir)
    null_corr_dir = os.path.join(corr_dir,'Null')
    if not os.path.isdir(null_corr_dir):
        os.mkdir(null_corr_dir)
        
    #Variables
    num_null = len(null_dev_rasters)
    num_seg = len(null_segment_dev_fr_vecs[0])
    num_neur = len(null_segment_dev_fr_vecs[0][0][0])
    num_tastes = len(all_dig_in_names)
    
    #Now go through segments and their deviation events and compare
    print('Correlating Null Deviation Events')
    for null_i in tqdm.tqdm(range(num_null)):
        null_i_corr_dir = os.path.join(null_corr_dir,'null_' + str(null_i))
        if not os.path.isdir(null_i_corr_dir):
            os.mkdir(null_i_corr_dir)
        for s_ind, s_i in enumerate(segments_to_analyze):
            seg_name = segment_names_to_analyze[s_ind]
            dev_rast = null_dev_rasters[null_i][s_ind]
            dev_times = null_dev_times[null_i][s_ind]
            # dev_fr_vecs = null_segment_dev_fr_vecs[null_i][s_ind]
            dev_fr_vecs_z = null_segment_dev_fr_vecs_zscore[null_i][s_ind]
            
            #Run correlation analyses
            correlate_dev_to_taste_zscore(num_neur,all_dig_in_names,tastant_fr_dist_z_pop,
                                              taste_num_deliv,max_hz_z_pop,min_hz_z_pop,
                                              max_num_cp,dev_rast,dev_times,dev_fr_vecs_z,
                                              seg_name,null_i_corr_dir,False)
            
        
def correlate_dev_to_taste_zscore(num_neur,all_dig_in_names,tastant_fr_dist_z_pop,
                                  taste_num_deliv,max_hz_z_pop,min_hz_z_pop,
                                  max_num_cp,dev_rast,dev_times,dev_fr_vecs_z,
                                  seg_name,corr_dir,plot_flag = True):
    
    fr_z_dir = os.path.join(corr_dir,'zscore_fr_corrs')
    if not os.path.isdir(fr_z_dir):
        os.mkdir(fr_z_dir)
        
    dev_vec_mat = np.array(dev_fr_vecs_z)
    num_dev, _ = np.shape(dev_vec_mat)
    dev_num = dev_vec_mat - np.expand_dims(np.nanmean(dev_vec_mat,1),1)
    dev_denom = np.sum(dev_num**2,1)
    
    #Z-Scored correlations
    corr_z_dict = dict()
    avg_corr_z_array = np.nan*np.ones((num_dev,len(all_dig_in_names),max_num_cp))
    for t_i, t_name in enumerate(all_dig_in_names):
        corr_z_dict[t_i] = dict()
        corr_z_dict[t_i]['name'] = t_name
        corr_z_dict[t_i]['data'] = dict()
        for e_i in range(max_num_cp):
            #Gather taste fr vecs
            all_epoch_taste_vecs = []
            for d_i in range(int(taste_num_deliv[t_i])):
                all_epoch_taste_vecs.append(np.squeeze(
                    tastant_fr_dist_z_pop[t_i][d_i][e_i]))
            all_epoch_taste_vecs_array = np.array(all_epoch_taste_vecs) #num_deliv x num_neur
            
            #Run all pairwise correlations
            taste_num = all_epoch_taste_vecs_array - np.expand_dims(np.nanmean(
                all_epoch_taste_vecs_array,1),1)
            taste_denom = np.sum(taste_num**2,1)
            
            all_corr_vals = []
            corr_vals_by_deliv = np.zeros((num_dev,int(taste_num_deliv[t_i])))
            for d_i in range(int(taste_num_deliv[t_i])):
                p_num = np.sum(dev_num*(taste_num[d_i,:]*np.ones(np.shape(dev_num))),1)
                p_denom = np.sqrt(dev_denom*taste_denom[d_i])
                corr_vec = p_num/p_denom
                corr_vals_by_deliv[:,d_i] = corr_vec
                all_corr_vals.extend(list(corr_vec))
            avg_taste_epoch_corr = np.nanmean(corr_vals_by_deliv,1)
            avg_corr_z_array[:,t_i,e_i] = avg_taste_epoch_corr
              
            corr_z_dict[t_i]['data'][e_i] = all_corr_vals
            corr_z_dict[t_i]['num_dev'] = num_dev
            corr_z_dict[t_i]['taste_num'] = taste_num
            
    np.save(os.path.join(fr_z_dir,seg_name+'_corr_z_dict.npy'),corr_z_dict,allow_pickle=True)
    
    #Calculate best taste,epoch for each deviation
    best_dev_inds = np.zeros((num_dev,3))
    for dev_i in range(num_dev):
        max_corr = np.nanmax(np.squeeze(avg_corr_z_array[dev_i,:,:]))
        ind_1, ind_2 = np.where(np.squeeze(avg_corr_z_array[dev_i,:,:]) == np.nanmax(np.squeeze(avg_corr_z_array[dev_i,:,:])))
        best_dev_inds[dev_i,0] = ind_1[0]
        best_dev_inds[dev_i,1] = ind_2[0]
        best_dev_inds[dev_i,2] = max_corr
    np.save(os.path.join(fr_z_dir,seg_name+'_best_corr.npy'),best_dev_inds,allow_pickle=True)
    if not os.path.isfile(os.path.join(fr_z_dir,'all_taste_names.npy')):
        np.save(os.path.join(fr_z_dir,'all_taste_names.npy'),all_dig_in_names,allow_pickle=True)
            
    #Now plot
    if plot_flag == True:
        plot_corr_dist(fr_z_dir,corr_z_dict,all_dig_in_names,max_num_cp,seg_name)
    
def correlate_null_dev_to_taste_zscore(num_neur,all_dig_in_names,tastant_fr_dist_z_pop,
                                  taste_num_deliv,max_hz_z_pop,min_hz_z_pop,
                                  max_num_cp,dev_rast,dev_times,dev_fr_vecs_z,
                                  seg_name,corr_dir):
    """
    This function correlates z-scored deviation events pulled out of null 
    datasets to z-scored taste responses.

    Parameters
    ----------
    num_neur : integer
        Number of neurons in dataset.
    all_dig_in_names : list of strings
        Names of all digital inputs in dataset.
    tastant_fr_dist_z_pop : dictionary
        Contains organized z-scored taste response firing rate vectors.
    taste_num_deliv : list if integers
        Number of deliveries of each tastant.
    max_hz_z_pop : boolean
        Maximum population firing rate.
    min_hz_z_pop : boolean
        Minimum population firing rate.
    max_num_cp : integer
        Maximum number of epochs across datasets.
    dev_rast :
        TODO: remove this variable.
    dev_times : 
        TODO: remove this variable.
    dev_fr_vecs_z : list
        List of deviation firing rate vectors.
    seg_name : string
        Name of analyzed segment, for labelling purposes.
    corr_dir : string
        Directory to store results.

    Returns
    -------
    None.
    
    Outputs
    -------
    Dictionary of correlation values + plot call.

    """
    
    fr_z_dir = os.path.join(corr_dir,'zscore_fr_corrs')
    if not os.path.isdir(fr_z_dir):
        os.mkdir(fr_z_dir)
        
    dev_vec_mat = np.array(dev_fr_vecs_z)
    num_dev, _ = np.shape(dev_vec_mat)
    dev_num = dev_vec_mat - np.expand_dims(np.nanmean(dev_vec_mat,1),1)
    dev_denom = np.sum(dev_num**2,1)
    
    #Z-Scored correlations
    corr_z_dict = dict()
    avg_corr_z_array = np.nan*np.ones((num_dev,len(all_dig_in_names),max_num_cp))
    for t_i, t_name in enumerate(all_dig_in_names):
        corr_z_dict[t_i] = dict()
        corr_z_dict[t_i]['name'] = t_name
        corr_z_dict[t_i]['data'] = dict()
        for e_i in range(max_num_cp):
            #Gather taste fr vecs
            all_epoch_taste_vecs = []
            for d_i in range(int(taste_num_deliv[t_i])):
                all_epoch_taste_vecs.append(np.squeeze(
                    tastant_fr_dist_z_pop[t_i][d_i][e_i]))
            all_epoch_taste_vecs_array = np.array(all_epoch_taste_vecs) #num_deliv x num_neur
            
            #Run all pairwise correlations
            taste_num = all_epoch_taste_vecs_array - np.expand_dims(np.nanmean(
                all_epoch_taste_vecs_array,1),1)
            taste_denom = np.sum(taste_num**2,1)
            
            all_corr_vals = []
            corr_vals_by_deliv = np.zeros((num_dev,int(taste_num_deliv[t_i])))
            for d_i in range(int(taste_num_deliv[t_i])):
                p_num = np.sum(dev_num*(taste_num[d_i,:]*np.ones(np.shape(dev_num))),1)
                p_denom = np.sqrt(dev_denom*taste_denom[d_i])
                corr_vec = p_num/p_denom
                corr_vals_by_deliv[:,d_i] = corr_vec
                all_corr_vals.extend(list(corr_vec))
            avg_taste_epoch_corr = np.nanmean(corr_vals_by_deliv,1)
            avg_corr_z_array[:,t_i,e_i] = avg_taste_epoch_corr
              
            corr_z_dict[t_i]['data'][e_i] = all_corr_vals
            corr_z_dict[t_i]['num_dev'] = num_dev
            corr_z_dict[t_i]['taste_num'] = taste_num
            
    np.save(os.path.join(fr_z_dir,seg_name+'_corr_z_dict.npy'),corr_z_dict,allow_pickle=True)
    
    #Calculate best taste,epoch for each deviation
    best_dev_inds = np.zeros((num_dev,3))
    for dev_i in range(num_dev):
        max_corr = np.nanmax(np.squeeze(avg_corr_z_array[dev_i,:,:]))
        ind_1, ind_2 = np.where(np.squeeze(avg_corr_z_array[dev_i,:,:]) == np.nanmax(np.squeeze(avg_corr_z_array[dev_i,:,:])))
        best_dev_inds[dev_i,0] = ind_1[0]
        best_dev_inds[dev_i,1] = ind_2[0]
        best_dev_inds[dev_i,2] = max_corr
    np.save(os.path.join(fr_z_dir,seg_name+'_best_corr.npy'),best_dev_inds,allow_pickle=True)
    if not os.path.isfile(os.path.join(fr_z_dir,'all_taste_names.npy')):
        np.save(os.path.join(fr_z_dir,'all_taste_names.npy'),all_dig_in_names,allow_pickle=True)
            
    #Now plot
    plot_corr_dist(fr_z_dir,corr_z_dict,all_dig_in_names,max_num_cp,seg_name)
    
def plot_corr_dist(corr_save_dir,corr_dict,all_dig_in_names,max_num_cp,seg_name):
    
    num_tastes = len(all_dig_in_names)
    epoch_pairs = list(itertools.combinations(np.arange(max_num_cp),2))
    taste_pairs = list(itertools.combinations(np.arange(len(all_dig_in_names)),2))
    #Plot epochs against each other for each taste
    sqrt_taste = np.ceil(np.sqrt(num_tastes)).astype('int')
    taste_ind_grid = np.reshape(np.arange(sqrt_taste**2),(sqrt_taste,sqrt_taste))
    f_taste, ax_taste = plt.subplots(nrows = sqrt_taste, ncols = sqrt_taste, 
                                     sharex = True, sharey = True, figsize=(8,8))
    ax_taste[0,0].set_xlim([-1,1])
    ax_taste[0,0].set_ylim([0,1])
    for t_i, t_name in enumerate(all_dig_in_names):
        ax_inds = np.where(taste_ind_grid == t_i)
        r_i = ax_inds[0][0]
        c_i = ax_inds[1][0]
        epoch_corrs = []
        #Plot cumulative distributions
        for e_i in range(max_num_cp):
            t_e_corr_data = corr_dict[t_i]['data'][e_i]
            epoch_corrs.append(t_e_corr_data)
            ax_taste[r_i,c_i].hist(t_e_corr_data,bins=1000,density=True, 
                            cumulative=True,histtype='step',label='Epoch ' + str(e_i))
        ax_taste[r_i,c_i].set_title(t_name)
        if t_i == 0:
            ax_taste[r_i,c_i].legend(loc='upper left')
        if c_i == 0:
            ax_taste[r_i,c_i].set_ylabel('Cumulative Density')
        if r_i == sqrt_taste-1:
            ax_taste[r_i,c_i].set_xlabel('Pearson Correlation')
        #Calculate pairwise epoch significances
        ks_sig = np.zeros(len(epoch_pairs))
        ks_dir = np.zeros(len(epoch_pairs))
        for ep_ind, e_pair in enumerate(epoch_pairs):
            e_1 = e_pair[0]
            e_2 = e_pair[1]
            ks_res = ks_2samp(epoch_corrs[e_1],epoch_corrs[e_2])
            if ks_res.pvalue <= 0.05:
                ks_sig[ep_ind] = 1
                ks_dir[ep_ind] = ks_res.statistic_sign
        sig_text = 'Significant Pairs:'
        if np.sum(ks_sig) > 0:
            sig_inds = np.where(ks_sig > 0)[0]
            for sig_i in sig_inds:
                e_1 = epoch_pairs[sig_i][0]
                e_2 = epoch_pairs[sig_i][1]
                e_pair_text = 'Epoch ' + str(e_1)
                if ks_dir[sig_i] == 1:
                    e_pair_text += ' < Epoch ' + str(e_2) 
                if ks_dir[sig_i] == -1:
                    e_pair_text += ' > Epoch ' + str(e_2)
                sig_text += '\n' + e_pair_text
        ax_taste[r_i,c_i].text(-0.75,0.2,sig_text)
    f_taste.savefig(os.path.join(corr_save_dir,seg_name+'_taste_corr.png'))
    f_taste.savefig(os.path.join(corr_save_dir,seg_name+'_taste_corr.svg'))
    plt.close(f_taste)
    
    #Plot tastes against each other for each epoch
    f_epoch, ax_epoch = plt.subplots(nrows = 2, ncols = max_num_cp, 
                                     gridspec_kw={'height_ratios': [2, 3]},
                                     sharex = True, sharey = True, figsize=(8,8))
    ax_epoch[0,0].set_xlim([-1,1])
    ax_epoch[0,0].set_ylim([0,1])
    for e_i in range(max_num_cp):
        taste_corrs = []
        #Plot cumulative distributions
        for t_i in range(num_tastes):
            t_e_corr_data = corr_dict[t_i]['data'][e_i]
            taste_corrs.append(t_e_corr_data)
            ax_epoch[0,e_i].hist(t_e_corr_data,bins=1000,density=True, 
                            cumulative=True,histtype='step',label=all_dig_in_names[t_i])
        ax_epoch[0,e_i].set_title('Epoch ' + str(e_i))
        ax_epoch[0,e_i].set_xlabel('Pearson Correlation')
        if e_i == 0:
            ax_epoch[0,e_i].legend(loc='upper left')
            ax_epoch[0,e_i].set_ylabel('Cumulative Density')
        #Calculate Taste Mean Orders
        taste_corr_means = [np.nanmean(taste_corrs[tc_i]) for tc_i in range(num_tastes)]
        taste_mean_order = np.argsort(taste_corr_means)
        taste_mean_name_order = 'Mean Order: '
        len_mod = 0
        for tmo_i, tmo in enumerate(taste_mean_order):
            if np.floor(len(taste_mean_name_order)/20).astype('int') > len_mod:
                len_mod = np.floor(len(taste_mean_name_order)/20).astype('int')
                taste_mean_name_order += '\n'
            if tmo_i < len(taste_mean_order)-1:
                taste_mean_name_order += all_dig_in_names[tmo] + ' < '
            else:
                taste_mean_name_order += all_dig_in_names[tmo]
        ax_epoch[1,e_i].set_title(taste_mean_name_order)    
        #Calculate Taste Percentile Orders
        taste_corr_90_percentiles = [np.percentile(taste_corrs[tc_i],90) for tc_i in range(num_tastes)]
        taste_90_order = np.argsort(taste_corr_90_percentiles)
        taste_90_name_order = '90th Percentile Order \n (min to max):'
        for tmo in taste_90_order:
            taste_90_name_order += '\n' + all_dig_in_names[tmo]
        ax_epoch[1,e_i].text(-0.75,0.1,taste_90_name_order)
    plt.suptitle(seg_name + ' Dev Correlations')
    plt.tight_layout()
    f_epoch.savefig(os.path.join(corr_save_dir,seg_name+'_epoch_corr.png'))
    f_epoch.savefig(os.path.join(corr_save_dir,seg_name+'_epoch_corr.svg'))
    plt.close(f_epoch)
        #Calculate pairwise epoch significances
        # ks_sig = np.zeros(len(taste_pairs))
        # ks_dir = np.zeros(len(taste_pairs))
        # for tp_ind, t_pair in enumerate(taste_pairs):
        #     t_1 = t_pair[0]
        #     t_2 = t_pair[1]
        #     t_name_1 = all_dig_in_names[t_1]
        #     t_name_2 = all_dig_in_names[t_2]
        #     ks_res = ks_2samp(taste_corrs[t_1],taste_corrs[t_2])
        #     if ks_res.pvalue <= 0.05:
        #         ks_sig[tp_ind] = 1
        #         ks_dir[tp_ind] = ks_res.statistic_sign
        # if np.sum(ks_sig) > 0:
        #     sig_text = 'Significant Pairs:'
        #     sig_inds = np.where(ks_sig > 0)[0]
        #     for sig_i in sig_inds:
        #         t_1 = taste_pairs[sig_i][0]
        #         t_2 = taste_pairs[sig_i][1]
        #         t_pair_text = all_dig_in_names[t_1]
        #         if ks_dir[sig_i] == 1:
        #             t_pair_text += ' < ' + all_dig_in_names[t_2]
        #         if ks_dir[sig_i] == -1:
        #             t_pair_text += ' > ' + all_dig_in_names[t_2]
        #         sig_text += '\n' + t_pair_text
        #     ax_epoch[1,e_i].text(-0.75,0.1,sig_text)
    
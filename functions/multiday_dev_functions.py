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
import itertools
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
                          segment_dev_times,segment_dev_fr_vecs,
                          segment_dev_fr_vecs_zscore,segment_names_to_analyze):
    """
    This function serves as the main function which calls all others for 
    analyses of deviation events from the train day in comparison to taste
    responses across days.
    """
    
    corr_dir = os.path.join(save_dir,'Correlations')
    if not os.path.isdir(corr_dir):
        os.mkdir(corr_dir)
    decode_dir = os.path.join(save_dir,'Decodes')
    if not os.path.isdir(decode_dir):
        os.mkdir(decode_dir)
        
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
        
        #Run decode analyses
        decode_dev_stepwise(num_neur,all_dig_in_names,tastant_fr_dist_pop,
                                   taste_num_deliv,max_num_cp,dev_rast,
                                   dev_times,dev_fr_vecs,seg_name,s_i,decode_dir)
        decode_dev_zscore_stepwise(num_neur,all_dig_in_names,tastant_fr_dist_z_pop,
                                   taste_num_deliv,max_num_cp,dev_rast,dev_times,
                                   dev_fr_vecs_z,seg_name,s_i,decode_dir)
        
        
def correlate_dev_to_taste(num_neur,all_dig_in_names,tastant_fr_dist_pop,
                           taste_num_deliv,max_hz_pop,max_num_cp,dev_rast,
                           dev_times,dev_fr_vecs,seg_name,corr_dir):
    
    fr_dir = os.path.join(corr_dir,'fr_corrs')
    if not os.path.isdir(fr_dir):
        os.mkdir(fr_dir)
        
    dev_vec_mat = np.array(dev_fr_vecs) #num_dev x num_neur
    dev_num = dev_vec_mat - np.expand_dims(np.nanmean(dev_vec_mat,1),1)
    dev_denom = np.sum(dev_num**2,1)
    
    #Regular correlations
    corr_dict = dict()
    for t_i, t_name in enumerate(all_dig_in_names):
        corr_dict[t_i] = dict()
        corr_dict[t_i]['name'] = t_name
        corr_dict[t_i]['data'] = dict()
        for e_i in range(max_num_cp):
            #Gather taste fr vecs
            all_epoch_taste_vecs = []
            for d_i in range(int(taste_num_deliv[t_i])):
                all_epoch_taste_vecs.append(np.squeeze(
                    tastant_fr_dist_pop[t_i][d_i][e_i]))
            all_epoch_taste_vecs_array = np.array(all_epoch_taste_vecs) #num_deliv x num_neur
            
            #Run all pairwise correlations
            taste_num = all_epoch_taste_vecs_array - np.expand_dims(np.nanmean(
                all_epoch_taste_vecs_array,1),1)
            taste_denom = np.sum(taste_num**2,1)
            
            all_corr_vals = []
            for d_i in range(int(taste_num_deliv[t_i])):
                p_num = np.sum(dev_num*(taste_num[d_i,:]*np.ones(np.shape(dev_num))),1)
                p_denom = np.sqrt(dev_denom*taste_denom[d_i])
                all_corr_vals.extend(list(p_num/p_denom))
              
            corr_dict[t_i]['data'][e_i] = all_corr_vals
    
    np.save(os.path.join(fr_dir,seg_name+'_corr_dict.npy'),corr_dict,allow_pickle=True)
            
    #Now plot
    plot_corr_dist(fr_dir,corr_dict,all_dig_in_names,max_num_cp,seg_name)
        
def correlate_dev_to_taste_zscore(num_neur,all_dig_in_names,tastant_fr_dist_z_pop,
                                  taste_num_deliv,max_hz_z_pop,min_hz_z_pop,
                                  max_num_cp,dev_rast,dev_times,dev_fr_vecs_z,
                                  seg_name,corr_dir):
    
    fr_z_dir = os.path.join(corr_dir,'zscore_fr_corrs')
    if not os.path.isdir(fr_z_dir):
        os.mkdir(fr_z_dir)
        
    dev_vec_mat = np.array(dev_fr_vecs_z) #num_dev x num_neur
    dev_num = dev_vec_mat - np.expand_dims(np.nanmean(dev_vec_mat,1),1)
    dev_denom = np.sum(dev_num**2,1)
    
    #Z-Scored correlations
    corr_z_dict = dict()
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
            for d_i in range(int(taste_num_deliv[t_i])):
                p_num = np.sum(dev_num*(taste_num[d_i,:]*np.ones(np.shape(dev_num))),1)
                p_denom = np.sqrt(dev_denom*taste_denom[d_i])
                all_corr_vals.extend(list(p_num/p_denom))
              
            corr_z_dict[t_i]['data'][e_i] = all_corr_vals
    
    np.save(os.path.join(fr_z_dir,seg_name+'_corr_z_dict.npy'),corr_z_dict,allow_pickle=True)
            
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
        
def decode_dev_stepwise(num_neur,all_dig_in_names,tastant_fr_dist_pop,
                           taste_num_deliv,max_num_cp,dev_rast,
                           dev_times,dev_fr_vecs,seg_name,s_i,decode_dir):
    
    fr_dir = os.path.join(decode_dir,'fr_decodes')
    if not os.path.isdir(fr_dir):
        os.mkdir(fr_dir)
        
    # Variables
    num_tastes = len(all_dig_in_names)
    dev_vec_mat = np.array(dev_fr_vecs)
    num_dev, num_neur = np.shape(dev_vec_mat)
    num_cp = len(tastant_fr_dist_pop[0][0])
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, num_cp))
    cmap = colormaps['gist_rainbow']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    cmap = colormaps['seismic']
    is_taste_colors = cmap(np.linspace(0, 1, 3))
    epochs_to_analyze = np.arange(num_cp)
    none_ind = -1
    for adi_i, adi_name in enumerate(all_dig_in_names):
        if adi_name[:4] == 'none':
            none_ind = adi_i
    
    #Collect data to train decoders
    true_taste_names = []
    true_taste_train_data = [] #For PCA all combined true taste data
    none_data = []
    by_taste_train_data = [] #All tastes in separate sub-lists
    by_taste_by_epoch_train_data = [] #True taste epoch data of size (num tastes - 1) x num epochs
    for t_i in range(num_tastes):
        num_deliveries = len(tastant_fr_dist_pop[t_i])
        train_taste_data = []
        train_by_epoch_taste_data = []
        for e_ind, e_i in enumerate(epochs_to_analyze):
            epoch_taste_data = []
            for d_i in range(num_deliveries):
                try:
                    if np.shape(tastant_fr_dist_pop[t_i][d_i][e_i])[0] == num_neur:
                        train_taste_data.extend(
                            list(tastant_fr_dist_pop[t_i][d_i][e_i].T))
                        epoch_taste_data.extend(
                            list(tastant_fr_dist_pop[t_i][d_i][e_i].T))
                    else:
                        train_taste_data.extend(
                            list(tastant_fr_dist_pop[t_i][d_i][e_i]))
                        epoch_taste_data.extend(
                            list(tastant_fr_dist_pop[t_i][d_i][e_i]))
                except:
                    train_taste_data.extend([])
            train_by_epoch_taste_data.append(epoch_taste_data)
        by_taste_by_epoch_train_data.append(train_by_epoch_taste_data)
        if t_i == none_ind:
            none_data.extend(train_taste_data)
            neur_max = np.expand_dims(np.max(np.array(train_taste_data),0),1)
            none_data.extend(list((neur_max*np.random.rand(num_neur,100)).T)) #Fully randomized data
            none_data.extend(list(((neur_max/10)*np.random.rand(num_neur,100)).T)) #Low frequency randomized data
            for nd_i in range(10): #Single spike by neuron data
                none_data.extend(list((np.eye(num_neur)).T))
        else:
            true_taste_train_data.extend(train_taste_data)
            true_taste_names.append(all_dig_in_names[t_i])
        by_taste_train_data.append(train_taste_data)
    by_taste_counts = np.array([len(by_taste_train_data[t_i]) for t_i in range(num_tastes)])
    by_taste_prob = by_taste_counts/np.sum(by_taste_counts)
    by_taste_true_train_data = [by_taste_train_data[t_i] for t_i in range(num_tastes-1)]
    by_taste_true_counts = np.array([len(by_taste_true_train_data[t_i]) for t_i in range(num_tastes-1)])
    by_taste_true_prob = by_taste_true_counts/np.sum(by_taste_true_counts)
    
    by_taste_epoch_counts = np.array([np.array([len(by_taste_by_epoch_train_data[t_i][e_i]) for e_i in range(len(epochs_to_analyze))]) for t_i in range(num_tastes-1)])
    by_taste_epoch_prob = by_taste_epoch_counts/np.expand_dims(np.sum(by_taste_epoch_counts,1),1)
        
    none_v_true_data = []
    none_v_true_data.append(true_taste_train_data)
    none_v_true_data.append(none_data)
    none_v_true_labels = ['Taste','No Taste']
    none_v_true_counts = np.array([len(none_v_true_data[i]) for i in range(len(none_v_true_data))])
    none_v_true_prob = none_v_true_counts/np.sum(none_v_true_counts)
    
    #Run PCA transform only on non-z-scored data
    need_pca = 1
    #Taste-Based PCA
    taste_pca = PCA()
    taste_pca.fit(np.array(true_taste_train_data).T)
    exp_var = taste_pca.explained_variance_ratio_
    num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
    if num_components == 0:
        num_components = 3
    pca_reduce_taste = PCA(num_components)
    pca_reduce_taste.fit(np.array(true_taste_train_data))
        
    #Run GMM fits to distributions of taste/no-taste
    none_v_taste_gmm = dict()
    for t_i in range(2):
        taste_train_data = np.array(none_v_true_data[t_i])
        transformed_data = pca_reduce_taste.transform(taste_train_data)
        #Fit GMM
        gm = gmm(n_components=1, n_init=10).fit(
            transformed_data)
        none_v_taste_gmm[t_i] = gm
        
    #Run GMM fits to true taste epoch-combined data
    just_taste_gmm = dict()
    for t_i in range(len(by_taste_true_train_data)):
        taste_train_data = np.array(by_taste_true_train_data[t_i])
        transformed_data = pca_reduce_taste.transform(taste_train_data)
        #Fit GMM
        gm = gmm(n_components=1, n_init=10).fit(
            transformed_data)
        just_taste_gmm[t_i] = gm
        
    #Run GMM fits to taste epoch-separated data
    taste_epoch_gmm = dict()
    for t_i in range(len(by_taste_by_epoch_train_data)):
        taste_epoch_train_data = by_taste_by_epoch_train_data[t_i] #dictionary of len = num_cp
        taste_epoch_gmm[t_i] = dict()
        for e_ind, e_i in enumerate(epochs_to_analyze):
            epoch_train_data = np.array(taste_epoch_train_data[e_ind])
            transformed_data = pca_reduce_taste.transform(epoch_train_data)
            #Fit GMM
            gm = gmm(n_components=1, n_init=10).fit(
                transformed_data)
            taste_epoch_gmm[t_i][e_ind] = gm
            
    # Grab neuron firing rates in sliding bins
    try:
        dev_decode_is_taste_array = np.load(
            os.path.join(fr_dir,seg_name + \
                         '_deviations_is_taste.npy'))
        
        dev_decode_which_taste_array = np.load(
            os.path.join(fr_dir,seg_name + \
                         '_deviations_which_taste.npy'))
            
        dev_decode_epoch_array = np.load(
            os.path.join(fr_dir,seg_name + \
                         '_deviations_which_epoch.npy'))
            
        print('\t\t\t\t' + seg_name + ' Previously Decoded')
    except:
        print('\t\t\t\tDecoding ' + seg_name + ' Deviation Splits')
        
        dev_decode_is_taste_array = np.zeros(num_dev) #deviation x is taste
        dev_decode_which_taste_array = np.nan*np.ones((num_dev,num_tastes-1)) #deviation x which taste
        dev_decode_epoch_array = np.nan*np.ones((num_dev,num_cp)) #deviation x epoch
        
        #Run through each deviation event to decode 
        tic = time.time()
        
        dev_fr_pca = pca_reduce_taste.transform(dev_vec_mat)
        dev_fr_list = list(dev_fr_pca)
            
        # Pass inputs to parallel computation on probabilities
        inputs = zip(dev_fr_list, itertools.repeat(len(none_v_taste_gmm)),
                      itertools.repeat(none_v_taste_gmm), itertools.repeat(none_v_true_prob))
        pool = Pool(4)
        dev_decode_is_taste_prob = pool.map(
            dp.segment_taste_decode_dependent_parallelized, inputs)
        pool.close()
        dev_decode_prob_taste_array = np.squeeze(np.array(dev_decode_is_taste_prob))
        dev_is_taste_argmax = np.argmax(dev_decode_prob_taste_array,1)
        dev_is_taste_inds = np.where(np.array(dev_is_taste_argmax) == 0)[0]
        dev_decode_is_taste_array[dev_is_taste_inds] = 1
        
        if len(dev_is_taste_inds) > 0: #at least some devs decoded as fully taste

            #Now determine which taste
            inputs = zip(dev_fr_list, itertools.repeat(len(just_taste_gmm)),
                          itertools.repeat(just_taste_gmm), itertools.repeat(by_taste_true_prob))
            pool = Pool(4)
            dev_decode_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            dev_decode_array = np.squeeze(np.array(dev_decode_prob)) #num_dev x 2
            dev_which_taste_argmax = np.argmax(dev_decode_array,1)
            dev_decode_which_taste_array[dev_is_taste_inds,dev_which_taste_argmax[dev_is_taste_inds]] = 1
            
            #Now determine for that taste which epoch it is
            dev_taste_list = []
            num_gmm = []
            which_taste_epoch_gmm = []
            prob_list = []
            for dev_ind, dev_i in enumerate(dev_is_taste_inds):
                dev_taste_list.append(dev_fr_list[dev_i])
                num_gmm.extend([len(taste_epoch_gmm[dev_which_taste_argmax[dev_i]])])
                which_taste_epoch_gmm.append(taste_epoch_gmm[dev_which_taste_argmax[dev_i]])
                prob_list.append(by_taste_epoch_prob[dev_which_taste_argmax[dev_i],:])
                
            #Now determine which epoch of that taste
            inputs = zip(dev_taste_list, num_gmm, \
                         which_taste_epoch_gmm, prob_list)
            pool = Pool(4)
            dev_decode_epoch_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            dev_decode_epoch_prob_array = np.squeeze(np.array(dev_decode_epoch_prob)) #num_dev x num_cp
            epoch_argmax = np.argmax(dev_decode_epoch_prob_array,1)
            dev_decode_epoch_array[dev_is_taste_inds,epoch_argmax] = 1
            
        np.save(os.path.join(fr_dir,seg_name + \
                         '_deviations_is_taste.npy'),dev_decode_is_taste_array)
        
        np.save(os.path.join(fr_dir,seg_name + \
                         '_deviations_which_taste.npy'),
                dev_decode_which_taste_array)
            
        np.save(os.path.join(fr_dir,seg_name + \
                         '_deviations_which_epoch.npy'),
                dev_decode_epoch_array) 
        
        toc = time.time()
        print('\t\t\t\t\tTime to decode ' + seg_name + \
              ' deviation splits = ' + str(np.round((toc-tic)/60, 2)) + ' (min)')
        
    dev_is_taste_inds, dev_which_taste_argmax = np.where(dev_decode_which_taste_array == 1)
        
    #Plot outcomes
    print('\t\t\t\t\tPlotting outcomes now.')
    plot_save_dir = os.path.join(fr_dir,seg_name)
    if not os.path.isdir(plot_save_dir):
        os.mkdir(plot_save_dir)
        
    decode_pie_plots(dev_is_taste_inds,num_dev,num_cp,dev_which_taste_argmax,
                         dev_decode_epoch_array,true_taste_names,
                         plot_save_dir)
    
    
def decode_dev_zscore_stepwise(num_neur,all_dig_in_names,tastant_fr_dist_z_pop,
                           taste_num_deliv,max_num_cp,dev_rast,dev_times,
                           dev_fr_vecs_z,seg_name,s_i,decode_dir):
    
    fr_z_dir = os.path.join(decode_dir,'fr_zscore_decodes')
    if not os.path.isdir(fr_z_dir):
        os.mkdir(fr_z_dir)
        
    # Variables
    num_tastes = len(all_dig_in_names)
    dev_vec_mat = np.array(dev_fr_vecs_z)
    num_dev, num_neur = np.shape(dev_vec_mat)
    num_cp = len(tastant_fr_dist_z_pop[0][0])
    epochs_to_analyze = np.arange(num_cp)
    none_ind = -1
    for adi_i, adi_name in enumerate(all_dig_in_names):
        if adi_name[:4] == 'none':
            none_ind = adi_i
    
    #Collect data to train decoders
    true_taste_names = []
    true_taste_train_data = [] #For PCA all combined true taste data
    none_data = []
    by_taste_train_data = [] #All tastes in separate sub-lists
    by_taste_by_epoch_train_data = [] #True taste epoch data of size (num tastes - 1) x num epochs
    for t_i in range(num_tastes):
        num_deliveries = len(tastant_fr_dist_z_pop[t_i])
        train_taste_data = []
        train_by_epoch_taste_data = []
        for e_ind, e_i in enumerate(epochs_to_analyze):
            epoch_taste_data = []
            for d_i in range(num_deliveries):
                try:
                    if np.shape(tastant_fr_dist_z_pop[t_i][d_i][e_i])[0] == num_neur:
                        train_taste_data.extend(
                            list(tastant_fr_dist_z_pop[t_i][d_i][e_i].T))
                        epoch_taste_data.extend(
                            list(tastant_fr_dist_z_pop[t_i][d_i][e_i].T))
                    else:
                        train_taste_data.extend(
                            list(tastant_fr_dist_z_pop[t_i][d_i][e_i]))
                        epoch_taste_data.extend(
                            list(tastant_fr_dist_z_pop[t_i][d_i][e_i]))
                except:
                    train_taste_data.extend([])
            train_by_epoch_taste_data.append(epoch_taste_data)
        by_taste_by_epoch_train_data.append(train_by_epoch_taste_data)
        if t_i == none_ind:
            none_data.extend(train_taste_data)
            neur_max = np.expand_dims(np.max(np.array(train_taste_data),0),1)
            none_data.extend(list((neur_max*np.random.rand(num_neur,100)).T)) #Fully randomized data
            none_data.extend(list(((neur_max/10)*np.random.rand(num_neur,100)).T)) #Low frequency randomized data
            for nd_i in range(10): #Single spike by neuron data
                none_data.extend(list((np.eye(num_neur)).T))
        else:
            true_taste_train_data.extend(train_taste_data)
            true_taste_names.append(all_dig_in_names[t_i])
        by_taste_train_data.append(train_taste_data)
    by_taste_counts = np.array([len(by_taste_train_data[t_i]) for t_i in range(num_tastes)])
    by_taste_prob = by_taste_counts/np.sum(by_taste_counts)
    by_taste_true_train_data = [by_taste_train_data[t_i] for t_i in range(num_tastes-1)]
    by_taste_true_counts = np.array([len(by_taste_true_train_data[t_i]) for t_i in range(num_tastes-1)])
    by_taste_true_prob = by_taste_true_counts/np.sum(by_taste_true_counts)
    
    by_taste_epoch_counts = np.array([np.array([len(by_taste_by_epoch_train_data[t_i][e_i]) for e_i in range(len(epochs_to_analyze))]) for t_i in range(num_tastes-1)])
    by_taste_epoch_prob = by_taste_epoch_counts/np.expand_dims(np.sum(by_taste_epoch_counts,1),1)
        
    none_v_true_data = []
    none_v_true_data.append(true_taste_train_data)
    none_v_true_data.append(none_data)
    none_v_true_labels = ['Taste','No Taste']
    none_v_true_counts = np.array([len(none_v_true_data[i]) for i in range(len(none_v_true_data))])
    none_v_true_prob = none_v_true_counts/np.sum(none_v_true_counts)
        
    #Run GMM fits to distributions of taste/no-taste
    none_v_taste_gmm = dict()
    for t_i in range(2):
        taste_train_data = np.array(none_v_true_data[t_i])
        #Fit GMM
        gm = gmm(n_components=1, n_init=10).fit(
            taste_train_data)
        none_v_taste_gmm[t_i] = gm
        
    #Run GMM fits to true taste epoch-combined data
    just_taste_gmm = dict()
    for t_i in range(len(by_taste_true_train_data)):
        taste_train_data = np.array(by_taste_true_train_data[t_i])
        #Fit GMM
        gm = gmm(n_components=1, n_init=10).fit(
            taste_train_data)
        just_taste_gmm[t_i] = gm
        
    #Run GMM fits to taste epoch-separated data
    taste_epoch_gmm = dict()
    for t_i in range(len(by_taste_by_epoch_train_data)):
        taste_epoch_train_data = by_taste_by_epoch_train_data[t_i] #dictionary of len = num_cp
        taste_epoch_gmm[t_i] = dict()
        for e_ind, e_i in enumerate(epochs_to_analyze):
            epoch_train_data = np.array(taste_epoch_train_data[e_ind])
            #Fit GMM
            gm = gmm(n_components=1, n_init=10).fit(
                epoch_train_data)
            taste_epoch_gmm[t_i][e_ind] = gm
            
    # Grab neuron firing rates in sliding bins
    try:
        dev_decode_is_taste_array = np.load(
            os.path.join(fr_z_dir,seg_name + \
                         '_deviations_is_taste.npy'))
        
        dev_decode_which_taste_array = np.load(
            os.path.join(fr_z_dir,seg_name + \
                         '_deviations_which_taste.npy'))
            
        dev_decode_epoch_array = np.load(
            os.path.join(fr_z_dir,seg_name + \
                         '_deviations_which_epoch.npy'))
            
        print('\t\t\t\t' + seg_name + ' Previously Decoded')
    except:
        print('\t\t\t\tDecoding ' + seg_name + ' Deviation Splits')
        
        dev_decode_is_taste_array = np.zeros((num_dev,2)) #deviation x is taste
        dev_decode_which_taste_array = np.nan*np.ones((num_dev,num_tastes-1)) #deviation x which taste
        dev_decode_epoch_array = np.nan*np.ones((num_dev,num_cp)) #deviation x epoch
        
        #Run through each deviation event to decode 
        tic = time.time()
        
        dev_fr_list = list(dev_vec_mat)
            
        # Pass inputs to parallel computation on probabilities
        inputs = zip(dev_fr_list, itertools.repeat(len(none_v_taste_gmm)),
                      itertools.repeat(none_v_taste_gmm), itertools.repeat(none_v_true_prob))
        pool = Pool(4)
        dev_decode_is_taste_prob = pool.map(
            dp.segment_taste_decode_dependent_parallelized, inputs)
        pool.close()
        dev_decode_is_taste_array = np.squeeze(np.array(dev_decode_is_taste_prob))
        dev_is_taste_argmax = np.argmax(dev_decode_is_taste_array,1)
        dev_is_taste_inds = np.where(np.array(dev_is_taste_argmax) == 0)[0]
        
        if len(dev_is_taste_inds) > 0: #at least some devs decoded as fully taste

            #Now determine which taste
            inputs = zip(dev_fr_list, itertools.repeat(len(just_taste_gmm)),
                          itertools.repeat(just_taste_gmm), itertools.repeat(by_taste_true_prob))
            pool = Pool(4)
            dev_decode_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            dev_decode_array = np.squeeze(np.array(dev_decode_prob)) #num_dev x 2
            dev_which_taste_argmax = np.argmax(dev_decode_array,1)
            dev_decode_which_taste_array[dev_is_taste_inds,dev_which_taste_argmax[dev_is_taste_inds]] = 1
            
            #Now determine for that taste which epoch it is
            dev_taste_list = []
            num_gmm = []
            which_taste_epoch_gmm = []
            prob_list = []
            for dev_ind, dev_i in enumerate(dev_is_taste_inds):
                dev_taste_list.append(dev_fr_list[dev_i])
                num_gmm.extend([len(taste_epoch_gmm[dev_which_taste_argmax[dev_i]])])
                which_taste_epoch_gmm.append(taste_epoch_gmm[dev_which_taste_argmax[dev_i]])
                prob_list.append(by_taste_epoch_prob[dev_which_taste_argmax[dev_i],:])
                
            #Now determine which epoch of that taste
            inputs = zip(dev_taste_list, num_gmm, \
                         which_taste_epoch_gmm, prob_list)
            pool = Pool(4)
            dev_decode_epoch_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            dev_decode_epoch_prob_array = np.squeeze(np.array(dev_decode_epoch_prob)) #num_dev x num_cp
            epoch_argmax = np.argmax(dev_decode_epoch_prob_array,1)
            dev_decode_epoch_array[dev_is_taste_inds,epoch_argmax] = 1
            
        np.save(os.path.join(fr_z_dir,seg_name + \
                         '_deviations_is_taste.npy'),dev_decode_is_taste_array)
        
        np.save(os.path.join(fr_z_dir,seg_name + \
                         '_deviations_which_taste.npy'),
                dev_decode_which_taste_array)
            
        np.save(os.path.join(fr_z_dir,seg_name + \
                         '_deviations_which_epoch.npy'),
                dev_decode_epoch_array) 
        
        toc = time.time()
        print('\t\t\t\t\tTime to decode ' + seg_name + \
              ' deviation splits = ' + str(np.round((toc-tic)/60, 2)) + ' (min)')
    
    dev_is_taste_inds, dev_which_taste_argmax = np.where(dev_decode_which_taste_array == 1)
        
    #Plot outcomes
    print('\t\t\t\t\tPlotting outcomes now.')
    plot_save_dir = os.path.join(fr_z_dir,seg_name)
    if not os.path.isdir(plot_save_dir):
        os.mkdir(plot_save_dir)
        
    decode_pie_plots(dev_is_taste_inds,num_dev,num_cp,dev_which_taste_argmax,
                         dev_decode_epoch_array,true_taste_names,plot_save_dir)
    
def decode_pie_plots(dev_is_taste_inds,num_dev,num_cp,dev_which_taste_argmax,
                     dev_decode_epoch_array,true_taste_names,plot_save_dir):
    """Plot decoding results in pie charts!"""
    epoch_labels = ['Epoch ' + str(e_i) for e_i in np.arange(num_cp)]
    num_tastes = len(true_taste_names)
    
    #Pie chart of taste vs none decode fractions
    f_istaste_pie = plt.figure(figsize=(5,5))
    num_taste = len(dev_is_taste_inds)
    num_not_taste = num_dev - num_taste
    pie_fracs = [num_taste/num_dev, num_not_taste/num_dev]
    pie_labels = ['Is Taste\n' + str(np.round(100*pie_fracs[0],2)) + '%','Not Taste\n' + str(np.round(100*pie_fracs[1],2)) + '%']
    plt.pie(pie_fracs, labels=pie_labels, labeldistance = 1)
    plt.title('Dev Events Decoded as Taste')
    plt.tight_layout()
    f_istaste_pie.savefig(os.path.join(plot_save_dir,'is_taste_pie.png'))
    f_istaste_pie.savefig(os.path.join(plot_save_dir,'is_taste_pie.svg'))
    plt.close(f_istaste_pie)
    
    #Pie chart of which taste fractions from all dev events
    taste_counts = [len(np.where(dev_which_taste_argmax == t_i)[0]) for t_i in range(len(true_taste_names))]
    taste_fracs_all_dev = list(np.array(taste_counts)/num_dev)
    taste_fracs_all_dev.extend([num_not_taste/num_dev])
    taste_fracs_all_dev = np.array(taste_fracs_all_dev)
    remove_ind = np.where(taste_fracs_all_dev == 0)[0]
    keep_ind = np.setdiff1d(np.arange(len(taste_fracs_all_dev)),remove_ind)
    taste_names_with_none = []
    taste_names_with_none.extend(true_taste_names)
    taste_names_with_none.extend(['none'])
    taste_names_with_none = list(np.array(taste_names_with_none)[keep_ind])
    keep_taste_fracs = taste_fracs_all_dev[keep_ind]
    keep_percent_labels = [taste_names_with_none[k_i] + '\n' + str(np.round(100*keep_taste_fracs[k_i],2)) + '%' for k_i in range(len(keep_ind))]
    
    #Pie chart of which taste fractions from only taste decoded dev events
    f_whichtaste_pie = plt.figure(figsize = (8,8))
    plt.pie(keep_taste_fracs,labels=keep_percent_labels,
                             explode= np.arange(len(taste_names_with_none))/len(taste_names_with_none), 
                             labeldistance = 1)
    plt.title('Fraction of All Events')
    plt.tight_layout()
    f_whichtaste_pie.savefig(os.path.join(plot_save_dir,'which_taste_pie_all_events.png'))
    f_whichtaste_pie.savefig(os.path.join(plot_save_dir,'which_taste_pie_all_events.svg'))
    plt.close(f_whichtaste_pie)
    
    taste_fracs_just_taste = np.array(taste_counts)/num_taste
    remove_ind = np.where(taste_fracs_just_taste == 0)[0]
    keep_ind = np.setdiff1d(np.arange(len(taste_fracs_just_taste)),remove_ind)
    keep_labels = list(np.array(true_taste_names)[keep_ind])
    keep_fracs = taste_fracs_just_taste[keep_ind]
    keep_percent_labels = [keep_labels[k_i] + '\n' + str(np.round(100*keep_fracs[k_i],2)) + '%' for k_i in range(len(keep_ind))]
    f_whichtaste_pie_taste = plt.figure(figsize = (8,8))
    plt.pie(keep_fracs,labels=keep_percent_labels,
                             explode= np.arange(len(keep_ind))/len(keep_ind), 
                             labeldistance = 1)
    plt.title('Fraction of Taste Decoded Events')
    plt.tight_layout()
    f_whichtaste_pie_taste.savefig(os.path.join(plot_save_dir,'which_taste_pie_taste_events.png'))
    f_whichtaste_pie_taste.savefig(os.path.join(plot_save_dir,'which_taste_pie_taste_events.svg'))
    plt.close(f_whichtaste_pie_taste)
    
    #Pie chart of which epoch for each taste decoded from taste decoded dev events
    f_taste_epoch_pie, ax_taste_epoch_pie = plt.subplots(nrows = num_tastes, ncols = 1,
                                                        figsize = (num_tastes*2,5))
    for t_i, t_name in enumerate(true_taste_names):
        dev_this_taste_inds = np.where(dev_which_taste_argmax == t_i)[0]
        epoch_counts = []
        for e_i in range(num_cp):
            epoch_inds = np.where(dev_decode_epoch_array[dev_is_taste_inds[dev_this_taste_inds],:] == 1)[1]
            epoch_counts.append(len(np.where(epoch_inds == e_i)[0]))
        if np.sum(epoch_counts) > 0:
            epoch_frac = np.array(epoch_counts)/np.sum(epoch_counts)
            epoch_percent_labels = [epoch_labels[e_i] + '\n' + str(np.round(100*epoch_frac[e_i],2)) + '%' for e_i in range(num_cp)]
            ax_taste_epoch_pie[t_i].pie(np.array(epoch_counts)/len(dev_is_taste_inds),
                                                  labels=epoch_percent_labels, 
                                                  explode= np.arange(num_cp)/num_cp, 
                                                  labeldistance = 1)
            ax_taste_epoch_pie[t_i].set_title(t_name)
    plt.tight_layout()
    f_taste_epoch_pie.savefig(os.path.join(plot_save_dir,'which_epoch_pie.png'))
    f_taste_epoch_pie.savefig(os.path.join(plot_save_dir,'which_epoch_pie.svg'))
    plt.close(f_taste_epoch_pie)

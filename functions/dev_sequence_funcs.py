#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:14:16 2024

@author: Hannah Germaine

A collection of functions dedicated to testing deviations for sequential
activity.
"""

import os
import csv
import time
import itertools
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import functions.decode_parallel as dp
from scipy import stats
from matplotlib import colormaps
from multiprocess import Pool
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as gmm

def split_match_calc(num_neur, segment_dev_rasters,segment_zscore_means,segment_zscore_stds,
                   tastant_fr_dist_pop,tastant_fr_dist_z_pop,dig_in_names,segment_names,
                   num_null, save_dir, segments_to_analyze, epochs_to_analyze = []):
    """
    This function is dedicated to an analysis of whether, when a deviation event
    is split in half down the middle, the pair of sides looks similar to adjacent
    pairs of epochs for different tastes. To do so, this function calculates the
    firing rate vectors across the population for each event when split down 
    the middle, and then calculates the correlation between the resulting matrix
    and the taste epoch pair matrices. The function outputs the distributions 
    of these correlations into a plot.
    """
    
    # Save Dirs
    dist_dir = os.path.join(save_dir, 'dist_tests')
    if not os.path.isdir(dist_dir):
        os.mkdir(dist_dir)
    corr_dir = os.path.join(save_dir, 'corr_tests')
    if not os.path.isdir(corr_dir):
        os.mkdir(corr_dir)
    decode_dir = os.path.join(save_dir,'decode_splits')
    if not os.path.isdir(decode_dir):
        os.mkdir(decode_dir)
    non_z_decode_dir = os.path.join(decode_dir,'firing_rates')
    if not os.path.isdir(non_z_decode_dir):
        os.mkdir(non_z_decode_dir)
    z_decode_dir = os.path.join(decode_dir,'zscore_firing_rates')
    if not os.path.isdir(z_decode_dir):
        os.mkdir(z_decode_dir)
    null_decode_dir = os.path.join(non_z_decode_dir,'null_decodes')
    if not os.path.isdir(null_decode_dir):
        os.mkdir(null_decode_dir)
    null_z_decode_dir = os.path.join(z_decode_dir,'null_decodes')
    if not os.path.isdir(null_z_decode_dir):
        os.mkdir(null_z_decode_dir)
    
    # Variables
    num_tastes = len(dig_in_names)
    num_segments = len(segment_dev_rasters)
    num_taste_deliv = [len(tastant_fr_dist_pop[t_i]) for t_i in range(num_tastes)]
    max_num_cp = 0
    for t_i in range(num_tastes):
        for d_i in range(num_taste_deliv[t_i]):
            if len(tastant_fr_dist_pop[t_i][d_i]) > max_num_cp:
                max_num_cp = len(tastant_fr_dist_pop[t_i][d_i])
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes)) #+1 for the deviation distribution color
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(max_num_cp)
    
    taste_pairs = list(itertools.combinations(np.arange(num_tastes),2))
    taste_pair_names = []
    for tp_i, tp in enumerate(taste_pairs):
        taste_pair_names.append(dig_in_names[tp[0]] + ' v. ' + dig_in_names[tp[1]])
    epoch_splits = list(itertools.combinations(epochs_to_analyze, 2))
    epoch_pair_pairs = list(itertools.combinations(np.arange(len(epoch_splits)), 2))
    epoch_pair_pair_names = []
    for epp_i, epp_pair_i in enumerate(epoch_pair_pairs):
        epoch_pair_1 = epoch_splits[epp_pair_i[0]]
        epoch_pair_2 = epoch_splits[epp_pair_i[1]]
        epoch_pair_pair_names.append(str(epoch_pair_1) + ' v. ' + str(epoch_pair_2))
        
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, len(epoch_splits)))
    
    #Collect the firing rate pair matrices for the taste deliveries
    all_taste_fr_mats = [] #num tastes x num epoch pairs x num deliv x (num neur x 2)
    all_taste_fr_mats_zscore = [] #num tastes x num epoch pairs x num deliv x (num neur x 2)
    for t_i in range(num_tastes):
        t_pair_mats = []
        t_pair_z_mats = []
        for e_p_ind, e_pair in enumerate(epoch_splits):
            e_pair_mats = []
            e_pair_z_mats = []
            for d_i in range(num_taste_deliv[t_i]):
                try:
                    e_1_vec = tastant_fr_dist_pop[t_i][d_i][e_pair[0]]
                    e_2_vec = tastant_fr_dist_pop[t_i][d_i][e_pair[1]]
                    e_pair_mats.append(np.concatenate((e_1_vec,e_2_vec),1))
                    e_1_vec_z = tastant_fr_dist_z_pop[t_i][d_i][e_pair[0]].T
                    e_2_vec_z = tastant_fr_dist_z_pop[t_i][d_i][e_pair[1]].T
                    e_pair_z_mats.append(np.concatenate((e_1_vec_z,e_2_vec_z),1))
                except:
                    e_pair_mats.extend([])
            t_pair_mats.append(e_pair_mats)
            t_pair_z_mats.append(e_pair_z_mats)
        all_taste_fr_mats.append(t_pair_mats)
        all_taste_fr_mats_zscore.append(t_pair_z_mats)
        
    #Now go through segments and their deviation events and compare
    for seg_ind, s_i in enumerate(segments_to_analyze):
        seg_dev_rast = segment_dev_rasters[seg_ind]
        seg_z_mean = segment_zscore_means[seg_ind]
        seg_z_std = segment_zscore_stds[seg_ind]
        num_dev = len(seg_dev_rast)
        
        dev_mats = []
        dev_mats_z = []
        null_dev_dict = dict()
        null_dev_z_dict = dict()
        for null_i in range(num_null):
            null_dev_dict[null_i] = []
            null_dev_z_dict[null_i] = []
        for dev_i in range(num_dev):
            #Pull raster firing rate vectors
            dev_rast = seg_dev_rast[dev_i]
            num_spikes_per_neur = np.sum(dev_rast,1).astype('int')
            _, num_dt = np.shape(dev_rast)
            half_dt = np.ceil(num_dt/2).astype('int')
            first_half_rast = dev_rast[:,:half_dt]
            second_half_rast = dev_rast[:,-half_dt:]
            #Create fr vecs
            first_half_fr_vec = np.expand_dims(np.sum(first_half_rast,1)/(half_dt/1000),1) #In Hz
            second_half_fr_vec = np.expand_dims(np.sum(second_half_rast,1)/(half_dt/1000),1) #In Hz
            dev_mat = np.concatenate((first_half_fr_vec,second_half_fr_vec),1)
            dev_mats.append(dev_mat)
            #Create z-scored fr vecs
            first_half_fr_vec_z = (first_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
            second_half_fr_vec_z = (second_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
            dev_mat_z = np.concatenate((first_half_fr_vec_z,second_half_fr_vec_z),1)
            dev_mats_z.append(dev_mat_z)
            #Create null versions of the event
            for null_i in range(num_null):
                shuffle_rast = np.zeros(np.shape(dev_rast))
                for neur_i in range(num_neur):
                    new_spike_ind = random.sample(list(np.arange(num_dt)),num_spikes_per_neur[neur_i])
                    shuffle_rast[neur_i,new_spike_ind] = 1
                first_half_shuffle_rast = shuffle_rast[:,:half_dt]
                second_half_shuffle_rast = shuffle_rast[:,-half_dt:]
                #Create fr vecs
                first_half_fr_vec = np.expand_dims(np.sum(first_half_shuffle_rast,1)/(half_dt/1000),1) #In Hz
                second_half_fr_vec = np.expand_dims(np.sum(second_half_shuffle_rast,1)/(half_dt/1000),1) #In Hz
                shuffle_dev_mat = np.concatenate((first_half_fr_vec,second_half_fr_vec),1)
                #Create z-scored fr vecs
                first_half_fr_vec_z = (first_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
                second_half_fr_vec_z = (second_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
                shuffle_dev_mat_z = np.concatenate((first_half_fr_vec_z,second_half_fr_vec_z),1)
                null_dev_dict[null_i].append(shuffle_dev_mat)
                null_dev_z_dict[null_i].append(shuffle_dev_mat_z)
            
        dev_mats_array = np.array(dev_mats) #num dev x num neur x 2
        dev_mats_z_array = np.array(dev_mats_z) #num dev x num neur x 2
        for null_i in range(num_null):
            null_dev_dict[null_i] = np.array(null_dev_dict[null_i]) #num dev x num neur x 2
            null_dev_z_dict[null_i] = np.array(null_dev_z_dict[null_i]) #num dev x num neur x 2
            
        #For each taste and epoch pair, calculate the distribution of correlations
        #for the deviation event matrices to the taste response matrices
        # all_taste_dist_vals = [] #num tastes x num epoch pairs x num dev x num deliv
        # all_taste_dist_vals_z = [] #num tastes x num epoch pairs x num dev x num deliv
        # all_taste_avg_dist_vals = [] #num tastes x num epoch pairs x num dev
        # all_taste_avg_dist_vals_z = [] #num tastes x num epoch pairs x num dev
        # all_taste_avg_corr_vals = [] #num tastes x num epoch pairs x num dev x (2x2)
        # all_taste_avg_corr_vals_z = [] #num tastes x num epoch pairs x num dev x (2x2)
        # all_taste_avg_corr_sigs = [] #num tastes x num epoch pairs x num dev x (2x2)
        # all_taste_avg_corr_sigs_z = [] #num tastes x num epoch pairs x num dev x (2x2)
        # for t_i in range(num_tastes):
        #     taste_dist_vals = [] #num epoch pairs x num dev x num deliv
        #     taste_dist_vals_z = [] #num epoch pairs x num dev x num deliv
        #     taste_avg_dist_vals = [] #num epoch pairs x num dev
        #     taste_avg_dist_vals_z = [] #num epoch pairs x num dev
        #     taste_avg_corr_vals = [] #num epoch pairs x num dev x (2x2)
        #     taste_avg_corr_vals_z = [] #num epoch pairs x num dev x (2x2)
        #     taste_avg_corr_sigs = [] #num epoch pairs x num dev x (2x2)
        #     taste_avg_corr_sigs_z = [] #num epoch pairs x num dev x (2x2)
        #     for e_p_ind, e_pair in enumerate(epoch_splits):
        #         e_pair_name = 'Epoch ' + str(e_pair[0]) + ', Epoch ' + str(e_pair[1])
                
        #         taste_deliv_mat_array = np.array(all_taste_fr_mats[t_i][e_p_ind]) #num deliv x num neur x 2
        #         taste_deliv_mat_z_array = np.array(all_taste_fr_mats_zscore[t_i][e_p_ind]) #num deliv x num neur x 2
                
        #         all_dist_vals = [] #size num dev x num deliv
        #         all_dist_vals_z = [] #size num dev x num deliv
        #         avg_dist_vals = [] #size num dev
        #         avg_dist_vals_z = [] #size num dev
        #         avg_corr_vals = [] #size num dev x (2x2)
        #         avg_corr_vals_z = [] #size num dev x (2x2)
        #         avg_corr_sigs = [] #size num dev x (2x2)
        #         avg_corr_sigs_z = [] #size num dev x (2x2)
        #         for dev_i in range(num_dev):
        #             dev_mat = dev_mats_array[dev_i,:,:].squeeze()
        #             dev_mat_z = dev_mats_z_array[dev_i,:,:].squeeze()
                    
        #             dev_i_dists = [] #size num deliv
        #             dev_i_z_dists = [] #size num deliv
        #             for deliv_i in range(num_taste_deliv[t_i]):
        #                 try:
        #                     taste_mat = taste_deliv_mat_array[deliv_i,:,:].squeeze()
        #                     taste_mat_z = taste_deliv_mat_z_array[deliv_i,:,:].squeeze()
        #                     #Calculate euclidean distances
        #                     dev_i_dists.extend([np.sqrt(np.sum(np.square(dev_mat-taste_mat)))])
        #                     dev_i_z_dists.extend([np.sqrt(np.sum(np.square(dev_mat_z-taste_mat_z)))])
        #                 except: #Trial missing condition
        #                     dev_i_dists.extend([])
        #             all_dist_vals.append(dev_i_dists)
        #             all_dist_vals_z.append(dev_i_z_dists)
        #             #Calculate average taste templates
        #             taste_avg_mat = np.nanmean(taste_deliv_mat_array,0).squeeze()
        #             taste_z_avg_mat = np.nanmean(taste_deliv_mat_z_array,0).squeeze()
        #             #Now the average correlation to that taste's template
        #             dev_i_avg_dist = np.sqrt(np.sum(np.square(dev_mat-taste_avg_mat)))
        #             dev_i_avg_z_dist =np.sqrt(np.sum(np.square(dev_mat_z-taste_z_avg_mat)))
        #             avg_dist_vals.extend([dev_i_avg_dist])
        #             avg_dist_vals_z.extend([dev_i_avg_z_dist])
        #             #Calculate pairwise correlations to taste's template
        #             dev_i_corrs = np.zeros((2,2))
        #             dev_i_z_corrs = np.zeros((2,2))
        #             dev_i_sigs = np.zeros((2,2))
        #             dev_i_z_sigs = np.zeros((2,2))
        #             for dev_half in range(2):
        #                 for deliv_half in range(2):
        #                     corr_results = stats.pearsonr(dev_mat[:,dev_half],taste_avg_mat[:,deliv_half])
        #                     dev_i_corrs[dev_half,deliv_half] = corr_results[0]
        #                     dev_i_sigs[dev_half,deliv_half] = corr_results[1]
        #                     corr_results_z = stats.pearsonr(dev_mat_z[:,dev_half],taste_z_avg_mat[:,deliv_half])
        #                     dev_i_z_corrs[dev_half,deliv_half] = corr_results_z[0]
        #                     dev_i_z_sigs[dev_half,deliv_half] = corr_results_z[1]
        #             avg_corr_vals.append(dev_i_corrs)
        #             avg_corr_vals_z.append(dev_i_z_corrs)
        #             avg_corr_sigs.append(dev_i_sigs)
        #             avg_corr_sigs_z.append(dev_i_z_sigs)
                
        #         taste_dist_vals.append(all_dist_vals)
        #         taste_dist_vals_z.append(all_dist_vals_z)
        #         taste_avg_dist_vals.append(avg_dist_vals)
        #         taste_avg_dist_vals_z.append(avg_dist_vals_z)
        #         taste_avg_corr_vals.append(avg_corr_vals)
        #         taste_avg_corr_vals_z.append(avg_corr_vals_z)
        #         taste_avg_corr_sigs.append(avg_corr_sigs)
        #         taste_avg_corr_sigs_z.append(avg_corr_sigs_z)
            
        #     all_taste_dist_vals.append(taste_dist_vals)
        #     all_taste_dist_vals_z.append(taste_dist_vals_z)
        #     all_taste_avg_dist_vals.append(taste_avg_dist_vals)
        #     all_taste_avg_dist_vals_z.append(taste_avg_dist_vals_z)
        #     all_taste_avg_corr_vals.append(taste_avg_corr_vals)
        #     all_taste_avg_corr_vals_z.append(taste_avg_corr_vals_z)
        #     all_taste_avg_corr_sigs.append(taste_avg_corr_sigs)
        #     all_taste_avg_corr_sigs_z.append(taste_avg_corr_sigs_z)
            
        #Calculate significance of distance distributions
        
        # sig_test_dist_distributions(num_tastes, taste_pairs, taste_pair_names, \
        #                             epoch_splits, epoch_pair_pairs, epoch_pair_pair_names, \
        #                             dig_in_names, taste_colors, epoch_colors, segment_names, s_i, \
        #                             all_taste_avg_dist_vals, all_taste_avg_dist_vals_z, dist_dir)
            
        # #Plot the distance distributions and significantly close distances
        
        # plot_dist_distributions(num_tastes, epoch_splits, dig_in_names, taste_colors, \
        #                             epoch_colors, segment_names, s_i, all_taste_avg_dist_vals, \
        #                             all_taste_avg_dist_vals_z, dist_dir)
        
        # plot_sig_dist_cutoff_results(epoch_splits, all_taste_avg_dist_vals, \
        #                                 all_taste_avg_dist_vals_z, num_tastes, \
        #                                 dig_in_names, taste_colors, s_i, segment_names, \
        #                                 dist_dir)
            
        # #Look at correlation results
        # #Calculate significance of correlation distributions
        
        
        # #   How many of the p-values from the correlation are significant?
        
        
        # #   Is the on-diagonal correlation strength stronger than off-diagonal?
        # plot_corr_distributions(num_tastes, epoch_splits, dig_in_names, taste_colors, \
        #                             epoch_colors, segment_names, s_i, all_taste_avg_corr_vals, \
        #                             all_taste_avg_corr_vals_z, corr_dir)
        
        #Decode each deviation event split
        
        decode_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist_pop, 
                        dig_in_names, dev_mats_array, segment_names, s_i,
                        non_z_decode_dir, epochs_to_analyze)
        decode_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist_z_pop, 
                        dig_in_names, dev_mats_z_array, segment_names, s_i,
                        z_decode_dir, epochs_to_analyze)
        
        #Decode null distribution
        decode_null_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist_pop, 
                        dig_in_names, null_dev_dict, segment_names, s_i,
                        null_decode_dir, non_z_decode_dir, epochs_to_analyze)
        decode_null_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist_z_pop, 
                        dig_in_names, null_dev_z_dict, segment_names, s_i,
                        null_z_decode_dir, z_decode_dir, epochs_to_analyze)
            
        
                     
def decode_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist, 
                dig_in_names, dev_mats_array, segment_names, s_i,
                decode_dir, epochs_to_analyze=[]):
    """Decode taste from epoch-specific firing rates"""
    print('\t\tRunning Is-Taste-Which-Taste GMM Decoder')
    
    # Variables
    num_tastes = len(dig_in_names)
    num_dev, num_neur, num_splits = np.shape(dev_mats_array)
    num_cp = len(tastant_fr_dist[0][0])
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, num_cp))
    cmap = colormaps['gist_rainbow']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    cmap = colormaps['seismic']
    is_taste_colors = cmap(np.linspace(0, 1, 3))
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    epoch_splits = list(itertools.combinations(epochs_to_analyze, 2))
    epoch_splits.extend(list(itertools.combinations(np.fliplr(np.expand_dims(epochs_to_analyze,0)).squeeze(), 2)))
    epoch_splits.extend([(e_i,e_i) for e_i in epochs_to_analyze])
    epoch_split_inds = np.arange(len(epoch_splits))
    epoch_split_names = [str(ep) for ep in epoch_splits]
        
    #Collect data to train decoders
    true_taste_train_data = [] #For PCA all combined true taste data
    none_data = []
    by_taste_train_data = [] #All tastes in separate sub-lists
    by_taste_by_epoch_train_data = [] #True taste epoch data of size (num tastes - 1) x num epochs
    for t_i in range(num_tastes):
        num_deliveries = len(tastant_fr_dist[t_i])
        train_taste_data = []
        train_by_epoch_taste_data = []
        for e_ind, e_i in enumerate(epochs_to_analyze):
            epoch_taste_data = []
            for d_i in range(num_deliveries):
                try:
                    if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                        train_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i].T))
                        epoch_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i].T))
                    else:
                        train_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i]))
                        epoch_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i]))
                except:
                    train_taste_data.extend([])
            train_by_epoch_taste_data.append(epoch_taste_data)
        by_taste_by_epoch_train_data.append(train_by_epoch_taste_data)
        if t_i < num_tastes-1:
            true_taste_train_data.extend(train_taste_data)
        else:
            none_data.extend(train_taste_data)
            neur_max = np.expand_dims(np.max(np.array(train_taste_data),0),1)
            none_data.extend(list((neur_max*np.random.rand(num_neur,100)).T)) #Fully randomized data
            none_data.extend(list(((neur_max/10)*np.random.rand(num_neur,100)).T)) #Low frequency randomized data
            for nd_i in range(10): #Single spike by neuron data
                none_data.extend(list((np.eye(num_neur)).T))
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
    need_pca = 0
    by_taste_pca_reducers = dict()
    if np.min(np.array(true_taste_train_data)) >= 0:
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
        if need_pca == 1:
            transformed_data = pca_reduce_taste.transform(taste_train_data)
        else:
            transformed_data = taste_train_data
        #Fit GMM
        gm = gmm(n_components=1, n_init=10).fit(
            transformed_data)
        none_v_taste_gmm[t_i] = gm
        
    #Run GMM fits to true taste epoch-combined data
    just_taste_gmm = dict()
    for t_i in range(len(by_taste_true_train_data)):
        taste_train_data = np.array(by_taste_true_train_data[t_i])
        if need_pca == 1:
            transformed_data = pca_reduce_taste.transform(taste_train_data)
        else:
            transformed_data = taste_train_data
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
            if need_pca == 1:
                transformed_data = pca_reduce_taste.transform(epoch_train_data)
            else:
                transformed_data = epoch_train_data
            #Fit GMM
            gm = gmm(n_components=1, n_init=10).fit(
                transformed_data)
            taste_epoch_gmm[t_i][e_ind] = gm
            
       
    # If trial_start_frac > 0 use only trials after that threshold
    #trial_start_ind = np.floor(max_num_deliv*trial_start_frac).astype('int')
    
    # Segment-by-segment use deviation rasters and times to zoom in and test
    #	epoch-specific decoding of tastes. Add decoding of 50 ms on either
    #	side of the deviation event as well for context decoding.
    
    # Grab neuron firing rates in sliding bins
    try:
        dev_decode_is_taste_array = np.load(
            os.path.join(decode_dir,segment_names[s_i] + \
                         '_deviations_is_taste.npy'))
        
        dev_decode_array = np.load(
            os.path.join(decode_dir,segment_names[s_i] + \
                         '_deviations_which_taste.npy'))
            
        dev_decode_epoch_array = np.load(
            os.path.join(decode_dir,segment_names[s_i] + \
                         '_deviations_which_epoch.npy'))
            
        print('\t\t\t\t' + segment_names[s_i] + ' Previously Decoded')
    except:
        print('\t\t\t\tDecoding ' + segment_names[s_i] + ' Deviation Splits')
        
        dev_decode_is_taste_array = np.zeros((num_dev,2,num_splits)) #deviation x is taste x split index
        dev_decode_array = np.zeros((num_dev,num_tastes-1,num_splits)) #deviation x which taste x split index
        dev_decode_epoch_array = np.zeros((num_dev,len(epochs_to_analyze),num_splits)) #deviation x epoch x split index
        
        taste_decode = []
        epoch_orders = []
        
        #Run through each deviation event to decode 
        tic = time.time()
        
        dev_fr_list = []
        
        for dev_i in range(num_dev):
            #Converting to list for parallel processing
            dev_fr_mat = np.squeeze(dev_mats_array[dev_i,:,:]) #Shape num_neur x 2
            if need_pca == 1:    
                dev_fr_pca = pca_reduce_taste.transform(dev_fr_mat.T)
                list_dev_fr = list(dev_fr_pca)
            else:
                list_dev_fr = list(dev_fr_mat.T)
            dev_fr_list.extend(list_dev_fr)
            
        # Pass inputs to parallel computation on probabilities
        inputs = zip(dev_fr_list, itertools.repeat(len(none_v_taste_gmm)),
                      itertools.repeat(none_v_taste_gmm), itertools.repeat(none_v_true_prob))
        pool = Pool(4)
        dev_decode_is_taste_prob = pool.map(
            dp.segment_taste_decode_dependent_parallelized, inputs)
        pool.close()
        dev_decode_is_taste_prob_array = np.squeeze(np.array(dev_decode_is_taste_prob)).T #2 x num dev splits
        dev_is_taste_argmax = []
        for dev_i in range(num_dev):
            dev_decode_is_taste_array[dev_i,:,:] = dev_decode_is_taste_prob_array[:,dev_i*2:dev_i*2+1]
            is_taste_argmax = np.argmax(dev_decode_is_taste_array[dev_i,:,:].squeeze(),0)
            dev_is_taste_argmax.append(is_taste_argmax)
        dev_is_taste_argmax_sum = np.sum(np.array(dev_is_taste_argmax),1)
        dev_is_taste_inds = np.where(dev_is_taste_argmax_sum == 0)[0]
        
        if len(dev_is_taste_inds) > 0: #at least some devs decoded as fully taste

            #Now determine which taste
            inputs = zip(dev_fr_list, itertools.repeat(len(just_taste_gmm)),
                          itertools.repeat(just_taste_gmm), itertools.repeat(by_taste_true_prob))
            pool = Pool(4)
            dev_decode_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            dev_decode_prob_array = np.squeeze(np.array(dev_decode_prob)).T #2 x num dev splits
            dev_which_taste_argmax = []
            for dev_i in dev_is_taste_inds:
                dev_decode_array[dev_i,:,:] = dev_decode_prob_array[:,dev_i*2:dev_i*2+1]
                which_taste_argmax = np.argmax(dev_decode_array[dev_i,:,:],0)
                dev_which_taste_argmax.append(which_taste_argmax)
            dev_which_taste_argmax_array = np.array(dev_which_taste_argmax)
                
            
            same_taste_test_ind = dev_which_taste_argmax_array[:,0]
            same_taste_bool = [] #True or false for each event whether same across splits
            for i in range(len(same_taste_test_ind)):
                same_taste_bool.append(all([dev_which_taste_argmax_array[i,j] == same_taste_test_ind[i] for j in range(num_splits)]))
            same_taste_ind = np.where(same_taste_bool)[0]
            
            if np.sum(np.array(same_taste_bool).astype('int')) > 0: #There are events that have the same taste decoded in all splits
                taste_decode.extend([same_taste_ind[same_taste_bool]])    
                dev_same_taste_inds = dev_is_taste_inds[same_taste_ind]
                dev_same_taste_list = []
                num_gmm = []
                same_taste_epoch_gmm = []
                prob_list = []
                
                for dev_ind, dev_i in enumerate(dev_same_taste_inds):
                    for sp_i in range(num_splits):
                        dev_same_taste_list.extend(dev_fr_list[2*dev_i:2*dev_i+2])
                        num_gmm.extend([len(taste_epoch_gmm[same_taste_test_ind[dev_ind]])])
                        same_taste_epoch_gmm.append(taste_epoch_gmm[same_taste_test_ind[dev_ind]])
                        prob_list.append(by_taste_epoch_prob[same_taste_test_ind[dev_ind],:])
            
                #Now determine which epoch of that taste
                inputs = zip(dev_same_taste_list, num_gmm, \
                             same_taste_epoch_gmm, prob_list)
                pool = Pool(4)
                dev_decode_epoch_prob = pool.map(
                    dp.segment_taste_decode_dependent_parallelized, inputs)
                pool.close()
                dev_decode_epoch_prob_array = np.squeeze(np.array(dev_decode_epoch_prob)).T #3 x num dev splits
                for dev_ind, dev_i in tqdm.tqdm(enumerate(dev_same_taste_inds)):
                    dev_decode_epoch_array[dev_i,:,:] = dev_decode_epoch_prob_array[:,dev_ind*2:dev_ind*2+2].squeeze()
            
                    which_epoch_argmax = np.argmax(dev_decode_epoch_array[dev_i,:,:],0)
                    epoch_orders.append(which_epoch_argmax)
        
        # Save decoding probabilities        
        np.save(os.path.join(decode_dir,segment_names[s_i]
                             + '_deviations_is_taste.npy'), dev_decode_is_taste_array)
        np.save(os.path.join(decode_dir,segment_names[s_i]
                             + '_deviations_which_taste.npy'), dev_decode_array)
        np.save(os.path.join(decode_dir,segment_names[s_i]
                             + '_deviations_which_epoch.npy'), dev_decode_epoch_array)
        
        toc = time.time()
        print('\t\t\t\t\tTime to decode ' + segment_names[s_i] + \
              ' deviation splits = ' + str(np.round((toc-tic)/60, 2)) + ' (min)')
            
    # Plot outcomes
    print('\t\t\t\t\tPlotting outcomes now.')
    
    #Is-Taste Summaries
    dev_decode_is_taste_ind = np.argmax(dev_decode_is_taste_array,1) #Index for each dev event splits of the max decoded value
    dev_decode_same_diff_is_taste = np.sum(dev_decode_is_taste_ind,1)
    #    dev_decode_same_diff_is_taste will have 0 for all-splits decoded as taste, num_splits for all splits decoded as not taste
    frac_taste = (num_splits*np.ones(np.shape(dev_decode_same_diff_is_taste)) - dev_decode_same_diff_is_taste)/num_splits
    np.save(os.path.join(decode_dir,segment_names[s_i]+'_frac_taste.npy'),frac_taste)
    #    frac_taste converts dev_decode_same_diff_is_taste to what fraction of splits were decoded as taste for each dev event
    #    Plot the distribution of fraction of deviation event decoded as taste
    f_frac_is_taste, ax_frac_is_taste = plt.subplots(ncols=2, figsize=(10,5))
    ax_frac_is_taste[0].hist(frac_taste, bins=10, density=False, \
                             histtype='step')
    ax_frac_is_taste[0].set_xlabel('Fraction of Deviation Event Decoded as Taste')
    ax_frac_is_taste[0].set_ylabel('Number of Deviation Events')
    ax_frac_is_taste[0].set_title('Distribution of Taste Fractions')
    ax_frac_is_taste[1].hist(frac_taste, bins=10, density=True, \
                             cumulative = True, histtype='step')
    ax_frac_is_taste[1].set_xlabel('Fraction of Deviation Event Decoded as Taste')
    ax_frac_is_taste[1].set_ylabel('Cumulative Density of Deviation Events')
    ax_frac_is_taste[1].set_title('Cumulative Distribution of Taste Fractions')
    plt.suptitle('Distribution of Fraction of Deviation Event Decoded as Taste')
    plt.tight_layout()
    f_frac_is_taste.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_dev_is_taste.png'))
    f_frac_is_taste.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_dev_is_taste.svg'))
    plt.close(f_frac_is_taste)
    #    Plot pie chart of deviation events decoded fully as taste, fully as no-taste, and fractionally decoded
    taste_count = len(np.where(frac_taste == 1)[0])
    no_taste_count = len(np.where(frac_taste == 0)[0])
    frac_count = num_dev - (taste_count + no_taste_count)
    
    # Sample data
    count_data = {'Taste': taste_count, 'Fractional': frac_count, 'No Taste': no_taste_count}
    # Filter out zero values
    filtered_count_data = {k: v for k, v in count_data.items() if v > 0}
    explode_vals = [0.1*i for i in range(len(filtered_count_data))]
    
    f_frac_pie = plt.figure(figsize=(5,5))
    plt.pie(filtered_count_data.values(), labels = filtered_count_data.keys(), \
        explode = explode_vals, pctdistance=1.2, labeldistance = 1.5, \
            rotatelabels = True, autopct='%1.2f%%')
    plt.title('Percent of Deviation Events Split-Decoded as Taste')
    plt.tight_layout()
    f_frac_pie.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_dev_is_taste_pie.png'))
    f_frac_pie.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_dev_is_taste_pie.svg'))
    plt.close(f_frac_pie)
    
    #Same-Taste Summaries
    is_taste_inds = np.where(frac_taste == 1)[0]
    is_taste_which_taste_decode_probs = dev_decode_array[is_taste_inds,:,:] #len(is_taste_inds) x num_tastes-1 x num_splits
    which_taste_argmax = np.argmax(is_taste_which_taste_decode_probs,1) #len(is_taste_inds) x num_splits
    np.save(os.path.join(decode_dir,segment_names[s_i]+'_which_taste_argmax.npy'),which_taste_argmax)
    same_taste_bool = np.zeros(taste_count)
    for tc_i in range(taste_count):
        taste_0 = which_taste_argmax[tc_i,0]
        if all([i == taste_0 for i in which_taste_argmax[tc_i,:]]):
            same_taste_bool[tc_i] = 1
    same_taste_count = np.sum(same_taste_bool)
    diff_taste_count = taste_count - same_taste_count
    f_frac_pie = plt.figure(figsize=(5,5))
    plt.pie([same_taste_count, diff_taste_count], labels = ['Same Taste', 'Diff Taste'], \
        explode = [0,0.2], pctdistance=1.2, labeldistance = 1.5, \
            rotatelabels = False, autopct='%1.2f%%')
    plt.title('Percent of Deviation Events Split-Decoded as Same Taste')
    plt.tight_layout()
    f_frac_pie.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_dev_same_taste_pie.png'))
    f_frac_pie.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_dev_same_taste_pie.svg'))
    plt.close(f_frac_pie)
    
    #Which-Taste Summaries
    same_taste_ind = np.where(same_taste_bool == 1)[0]
    same_taste_which_taste_decode_probs = is_taste_which_taste_decode_probs[same_taste_ind,:,:] #same_taste_count x num_tastes-1 x num_splits
    same_taste_which_taste_argmax = which_taste_argmax[same_taste_ind,:]
    np.save(os.path.join(decode_dir,segment_names[s_i]+'_same_taste_which_taste_argmax.npy'),same_taste_which_taste_argmax)
    count_data = {}
    for t_i in range(num_tastes-1):
        count_data[dig_in_names[t_i]] = len(np.where(same_taste_which_taste_argmax == t_i)[0])
    # Filter out zero values
    filtered_count_data = {k: v for k, v in count_data.items() if v > 0}
    explode_vals = [0.1*i for i in range(len(filtered_count_data))]
    
    f_frac_pie = plt.figure(figsize=(5,5))
    plt.pie(count_data.values(), labels = count_data.keys(), \
        explode = explode_vals, pctdistance=1.2, labeldistance = 1.5, \
            rotatelabels = False, autopct='%1.2f%%')
    plt.title('Which Taste are Same-Taste Deviation Events')
    plt.tight_layout()
    f_frac_pie.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_dev_same_taste_which_taste_pie.png'))
    f_frac_pie.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_dev_same_taste_which_taste_pie.svg'))
    plt.close(f_frac_pie)
    
    #Which-Epoch Summaries
    is_taste_dev_decode_epoch_array = dev_decode_epoch_array[is_taste_inds,:,:]
    same_taste_dev_decode_epoch_array = is_taste_dev_decode_epoch_array[same_taste_ind,:,:]
    np.save(os.path.join(decode_dir,segment_names[s_i]+'_same_taste_dev_decode_epoch_array.npy'),same_taste_dev_decode_epoch_array)
    f_epoch_order_pie, ax_epoch_order_pie = plt.subplots(ncols = num_tastes-1, figsize=((num_tastes-1)*5,5))
    f_epoch_order_bar, ax_epoch_order_bar = plt.subplots(ncols = num_tastes-1, figsize=((num_tastes-1)*5,5))
    f_epoch_joint_bar = plt.figure(figsize=(5,5))
    for t_i in range(num_tastes-1):
        taste_name = dig_in_names[t_i]
        taste_decode_inds = np.where(same_taste_which_taste_argmax == t_i)[0]
        taste_decode_probs = is_taste_dev_decode_epoch_array[taste_decode_inds,:,:] #decoded as taste x num epochs x num splits
        epoch_decode_argmax = np.argmax(taste_decode_probs,1)
        epoch_order_dict = {}
        for ep_i, ep in enumerate(epoch_splits):
            match_1 = epoch_decode_argmax[:,0] == ep[0]
            match_2 = epoch_decode_argmax[:,1] == ep[1]
            epoch_order_dict[ep_i] = len(np.where(match_1*match_2)[0])
        # Filter out zero values
        filtered_count_data = {k: v for k, v in epoch_order_dict.items() if v > 0}
        filtered_labels = [str(epoch_splits[int(i)]) for i in filtered_count_data.keys()]
        #Pie Chart
        ax_epoch_order_pie[t_i].pie(filtered_count_data.values(), labels = filtered_labels, \
            pctdistance=1.2, labeldistance = 1.5, \
                rotatelabels = False, autopct='%1.2f%%')
        ax_epoch_order_pie[t_i].set_title(taste_name)
        #Histograms Split
        ax_epoch_order_bar[t_i].bar(filtered_labels,filtered_count_data.values())
        ax_epoch_order_bar[t_i].set_title(taste_name)
        #Histograms Combined
        plt.figure(f_epoch_joint_bar)
        plt.bar(epoch_split_inds+(0.25*t_i),epoch_order_dict.values(),\
                width=0.2,label=dig_in_names[t_i])
    plt.figure(f_epoch_order_pie)
    plt.tight_layout()
    f_epoch_order_pie.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_which_epoch_pie.png'))
    f_epoch_order_pie.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_which_epoch_pie.svg'))
    plt.close(f_epoch_order_pie)
    plt.figure(f_epoch_order_bar)
    plt.tight_layout()
    f_epoch_order_bar.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_which_epoch_bar.png'))
    f_epoch_order_bar.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_which_epoch_bar.svg'))
    plt.close(f_epoch_order_bar)
    plt.figure(f_epoch_joint_bar)
    plt.xticks(epoch_split_inds,epoch_split_names)
    plt.legend(loc='upper left')
    plt.tight_layout()
    f_epoch_joint_bar.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_which_epoch_bar_joint.png'))
    f_epoch_joint_bar.savefig(os.path.join(decode_dir,segment_names[s_i]
                         + '_frac_which_epoch_bar_joint.svg'))
    plt.close(f_epoch_joint_bar)
    
def decode_null_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist, 
                dig_in_names, null_dev_dict, segment_names, s_i,
                null_decode_dir, true_decode_dir, epochs_to_analyze=[]):
    """Decode taste from epoch-specific firing rates"""
    print('\t\tRunning Null Is-Taste-Which-Taste GMM Decoder')
    
    # Variables
    num_tastes = len(dig_in_names)
    num_null = len(null_dev_dict)
    num_dev, num_neur, num_splits = np.shape(null_dev_dict[0])
    num_cp = len(tastant_fr_dist[0][0])
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, num_cp))
    cmap = colormaps['gist_rainbow']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    cmap = colormaps['seismic']
    is_taste_colors = cmap(np.linspace(0, 1, 3))
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    epoch_splits = list(itertools.combinations(epochs_to_analyze, num_splits))
    epoch_splits.extend(list(itertools.combinations(np.fliplr(np.expand_dims(epochs_to_analyze,0)).squeeze(), 2)))
    epoch_splits.extend([(e_i,e_i) for e_i in epochs_to_analyze])
    epoch_split_inds = np.arange(len(epoch_splits))
    epoch_split_names = [str(ep) for ep in epoch_splits]
    epoch_split_plot_square = np.ceil(np.sqrt(len(epoch_splits))).astype('int')
    epoch_split_plot_square_reference = np.reshape(np.arange(epoch_split_plot_square**2),\
                                                   (epoch_split_plot_square,epoch_split_plot_square))
        
    #Collect data to train decoders
    true_taste_train_data = [] #For PCA all combined true taste data
    none_data = []
    by_taste_train_data = [] #All tastes in separate sub-lists
    by_taste_by_epoch_train_data = [] #True taste epoch data of size (num tastes - 1) x num epochs
    for t_i in range(num_tastes):
        num_deliveries = len(tastant_fr_dist[t_i])
        train_taste_data = []
        train_by_epoch_taste_data = []
        for e_ind, e_i in enumerate(epochs_to_analyze):
            epoch_taste_data = []
            for d_i in range(num_deliveries):
                try:
                    if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                        train_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i].T))
                        epoch_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i].T))
                    else:
                        train_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i]))
                        epoch_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i]))
                except:
                    train_taste_data.extend([])
            train_by_epoch_taste_data.append(epoch_taste_data)
        by_taste_by_epoch_train_data.append(train_by_epoch_taste_data)
        if t_i < num_tastes-1:
            true_taste_train_data.extend(train_taste_data)
        else:
            none_data.extend(train_taste_data)
            neur_max = np.expand_dims(np.max(np.array(train_taste_data),0),1)
            none_data.extend(list((neur_max*np.random.rand(num_neur,100)).T)) #Fully randomized data
            none_data.extend(list(((neur_max/10)*np.random.rand(num_neur,100)).T)) #Low frequency randomized data
            for nd_i in range(10): #Single spike by neuron data
                none_data.extend(list((np.eye(num_neur)).T))
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
    need_pca = 0
    by_taste_pca_reducers = dict()
    if np.min(np.array(true_taste_train_data)) >= 0:
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
        if need_pca == 1:
            transformed_data = pca_reduce_taste.transform(taste_train_data)
        else:
            transformed_data = taste_train_data
        #Fit GMM
        gm = gmm(n_components=1, n_init=10).fit(
            transformed_data)
        none_v_taste_gmm[t_i] = gm
        
    #Run GMM fits to true taste epoch-combined data
    just_taste_gmm = dict()
    for t_i in range(len(by_taste_true_train_data)):
        taste_train_data = np.array(by_taste_true_train_data[t_i])
        if need_pca == 1:
            transformed_data = pca_reduce_taste.transform(taste_train_data)
        else:
            transformed_data = taste_train_data
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
            if need_pca == 1:
                transformed_data = pca_reduce_taste.transform(epoch_train_data)
            else:
                transformed_data = epoch_train_data
            #Fit GMM
            gm = gmm(n_components=1, n_init=10).fit(
                transformed_data)
            taste_epoch_gmm[t_i][e_ind] = gm
    
    try:
        null_is_taste_counts = np.load(os.path.join(null_decode_dir,segment_names[s_i] + '_null_is_taste_counts.npy'))
        null_which_taste_counts = np.load(os.path.join(null_decode_dir,segment_names[s_i] + '_null_which_taste_counts.npy'))
        null_epoch_pair_counts = np.load(os.path.join(null_decode_dir,segment_names[s_i] + '_null_epoch_pair_counts.npy'))
            
        print('\t\t\t\t' + segment_names[s_i] + ' Null Splits Previously Decoded')
    except:
        print('\t\t\t\tDecoding ' + segment_names[s_i] + ' Null Deviation Splits')
        
        #Run through each deviation event to decode 
        tic = time.time()
        
        null_is_taste_counts = np.zeros(num_null)
        null_which_taste_counts = np.zeros((num_null,num_tastes-1))
        null_epoch_pair_counts = np.zeros((num_null,num_tastes-1,len(epoch_splits)))
        
        for null_i in range(num_null):
            
            null_fr_mats = null_dev_dict[null_i]
            
            dev_decode_is_taste_array = np.zeros((num_dev,2,num_splits)) #deviation x is taste x split index
            dev_decode_array = np.zeros((num_dev,num_tastes-1,num_splits)) #deviation x which taste x split index
            dev_decode_epoch_array = np.zeros((num_dev,len(epochs_to_analyze),num_splits)) #deviation x epoch x split index
            
            null_dev_fr_list = []
            for dev_i in range(num_dev):
                #Converting to list for parallel processing
                dev_fr_mat = np.squeeze(null_fr_mats[dev_i,:,:]) #Shape num_neur x 2
                if need_pca == 1:    
                    dev_fr_pca = pca_reduce_taste.transform(dev_fr_mat.T)
                    list_dev_fr = list(dev_fr_pca)
                else:
                    list_dev_fr = list(dev_fr_mat.T)
                null_dev_fr_list.extend(list_dev_fr)
            
            # Pass inputs to parallel computation on probabilities
            inputs = zip(null_dev_fr_list, itertools.repeat(len(none_v_taste_gmm)),
                          itertools.repeat(none_v_taste_gmm), itertools.repeat(none_v_true_prob))
            pool = Pool(4)
            dev_decode_is_taste_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            dev_decode_is_taste_prob_array = np.squeeze(np.array(dev_decode_is_taste_prob)) #(num dev splits*num_dev) x 2
            dev_decode_reshape_array = np.reshape(dev_decode_is_taste_prob_array,(num_dev,num_splits,2)) #num dev x num splits x 2
            
            #Now determine which taste
            inputs = zip(null_dev_fr_list, itertools.repeat(len(just_taste_gmm)),
                          itertools.repeat(just_taste_gmm), itertools.repeat(by_taste_true_prob))
            pool = Pool(4)
            dev_decode_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            dev_decode_prob_array = np.squeeze(np.array(dev_decode_prob)) #(num dev splits*num_dev) x 2
            dev_decode_prob_reshape_array = np.reshape(dev_decode_prob_array,(num_dev,num_splits,2)) #num dev x num splits x 2
            
            #Calculate all counts / store probabilities to arrays
            
            dev_is_taste_argmax = np.zeros((num_dev,num_splits))
            dev_which_taste_argmax = np.zeros((num_dev,num_splits))
            for dev_i in range(num_dev):
                #Is Taste
                dev_decode_is_taste_array[dev_i,:,:] = dev_decode_reshape_array[dev_i,:,:].T
                dev_is_taste_argmax[dev_i,:] = np.argmax(np.squeeze(dev_decode_is_taste_array[dev_i,:,:]),0)
                #Which Taste
                dev_decode_array[dev_i,:,:] = dev_decode_prob_reshape_array[dev_i,:,:].T
                dev_which_taste_argmax[dev_i,:] = np.argmax(np.squeeze(dev_decode_array[dev_i,:,:]),0)
            
            #Is Taste Counts
            taste_devs = np.where(np.sum(dev_is_taste_argmax,1) == 0)[0]
            null_is_taste_counts[null_i] = len(taste_devs)
            #Which Taste
            same_taste_dev_inds = []
            same_taste_dev_taste_inds = []
            for dev_i in range(num_dev):
                which_taste_argmax = dev_which_taste_argmax[dev_i,:]
                same_taste_ind = which_taste_argmax[0]
                same_taste_bool = all([i == same_taste_ind for i in which_taste_argmax])
                same_taste_dev_taste_inds.extend([same_taste_ind])
                if same_taste_bool == True: #The taste is the same across the splits
                    same_taste_dev_inds.extend([dev_i])
            is_taste_same_taste_inds = np.intersect1d(taste_devs,np.array(same_taste_dev_inds))
            for t_i in range(num_tastes-1):
                null_which_taste_counts[null_i,t_i] = len(np.where(np.array(same_taste_dev_taste_inds)[is_taste_same_taste_inds] == t_i)[0])
            
            #Now for those that are decoded as taste and same taste determine which epoch
            same_taste_inds = []
            num_options = []
            gmms_list = []
            epoch_prob_list = []
            for itst_i in is_taste_same_taste_inds:
                same_taste_ind = int(same_taste_dev_taste_inds[itst_i])
                for ns_i in range(num_splits):
                    same_taste_inds.extend([int(same_taste_ind)])
                    num_options.extend([len(taste_epoch_gmm[same_taste_ind])])
                    gmms_list.append(taste_epoch_gmm[same_taste_ind])
                    epoch_prob_list.append(by_taste_epoch_prob[same_taste_ind,:])
            
            #Now determine which epoch of that taste
            inputs = zip(null_dev_fr_list, num_options, gmms_list, epoch_prob_list)
            pool = Pool(4)
            dev_decode_epoch_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            dev_decode_epoch_prob_array = np.squeeze(np.array(dev_decode_epoch_prob)) #len(is_taste_same_taste_inds)*num_splits x num_cp
            dev_decode_prob_reshape_array = np.reshape(dev_decode_epoch_prob_array,(len(is_taste_same_taste_inds),num_splits,num_cp)) #num len(is_taste_same_taste_inds) x num splits x 2
            
            for t_i in range(num_tastes-1):
                this_taste_inds = np.where(np.array(same_taste_dev_taste_inds)[is_taste_same_taste_inds] == t_i)[0]
                epoch_split_counts = dict()
                for ep in epoch_split_names:
                    epoch_split_counts[ep] = 0
                for itst_i in this_taste_inds:
                    epoch_decodes_i = dev_decode_prob_reshape_array[itst_i,:,:].squeeze()
                    epoch_split_i = tuple(np.argmax(epoch_decodes_i,1))
                    epoch_split_counts[str(epoch_split_i)] += 1
                null_epoch_pair_counts[null_i,t_i,:] = np.array(list(epoch_split_counts.values()))
        
        np.save(os.path.join(null_decode_dir,segment_names[s_i] + '_null_is_taste_counts.npy'),null_is_taste_counts)
        np.save(os.path.join(null_decode_dir,segment_names[s_i] + '_null_which_taste_counts.npy'),null_which_taste_counts)
        np.save(os.path.join(null_decode_dir,segment_names[s_i] + '_null_epoch_pair_counts.npy'),null_epoch_pair_counts)
        
        toc = time.time()
        print('\t\t\t\t\tTime to decode all nulls = ' + str(np.round((toc-tic)/60, 2)) + ' (min)')
        
    # Plot outcomes
    print('\t\t\t\t\tPlotting outcomes now.')
    
    #Is-Taste Summaries
    frac_all_taste = null_is_taste_counts/num_dev
    true_frac_taste = np.load(os.path.join(true_decode_dir,segment_names[s_i]+'_frac_taste.npy'))
    true_taste_frac_val = len(np.where(true_frac_taste == 1)[0])/num_dev
    percentile_95 = np.percentile(frac_all_taste,95)
    percentile_5 = np.percentile(frac_all_taste,5)
    f_frac_all_taste = plt.figure(figsize=(5,5))
    plt.hist(frac_all_taste, label='Null Distribution')
    plt.axvline(true_taste_frac_val,label='True Data',color='r')
    if true_taste_frac_val > percentile_95:
        plt.title('Fraction of deviation events with \nall splits decoded as taste\n*0.95')
    elif true_taste_frac_val < percentile_5:
        plt.title('Fraction of deviation events with \nall splits decoded as taste\n*0.05')
    else:
        plt.title('Fraction of deviation events with \nall splits decoded as taste')
    plt.xlabel('Fraction')
    plt.ylabel('Number of Null Distributions')
    plt.legend(loc='upper left')
    plt.tight_layout()
    f_frac_all_taste.savefig(os.path.join(true_decode_dir,segment_names[s_i]
                         + '_frac_dev_all_taste_v_null.png'))
    f_frac_all_taste.savefig(os.path.join(true_decode_dir,segment_names[s_i]
                         + '_frac_dev_all_taste_v_null.svg'))
    plt.close(f_frac_all_taste)
    
    #Which Taste Summaries
    which_taste_true_argmax = np.load(os.path.join(true_decode_dir,segment_names[s_i]+'_which_taste_argmax.npy'))
    same_taste_bool_true = np.zeros(len(which_taste_true_argmax))
    for tc_i in range(len(which_taste_true_argmax)):
        taste_0 = which_taste_true_argmax[tc_i,0]
        if all([i == taste_0 for i in which_taste_true_argmax[tc_i,:]]):
            same_taste_bool_true[tc_i] = 1
    same_taste_inds = np.where(same_taste_bool_true == 1)[0]
    true_taste_inds = which_taste_true_argmax[same_taste_inds,0]
    true_taste_counts = [len(np.where(true_taste_inds == t_i)[0]) for t_i in range(num_tastes-1)]
    true_taste_fractions = np.array(true_taste_counts)/num_dev
    null_taste_fractions = null_which_taste_counts/num_dev
    true_taste_only_taste_fractions = np.array(true_taste_counts)/len(same_taste_inds)
    null_taste_only_taste_fractions = null_which_taste_counts/np.expand_dims(np.sum(null_which_taste_counts,1),1)
    
    f_which_taste, ax_which_taste = plt.subplots(nrows=2, ncols=num_tastes-1, 
                                                 sharex = 'col', sharey = 'row',
                                                 figsize=(5,5))
    for t_i in range(num_tastes-1):
        null_vals = null_taste_fractions[:,t_i]
        ax_which_taste[0,t_i].hist(null_vals,label='Null Distribution')
        true_val = true_taste_fractions[t_i]
        ax_which_taste[0,t_i].axvline(true_val,label='True Data',color='r')
        
        percentile_95 = np.percentile(null_vals,95)
        percentile_5 = np.percentile(null_vals,5)
        if true_val > percentile_95:
            ax_which_taste[0,t_i].set_title(dig_in_names[t_i] + ' *.95')
        elif true_val < percentile_5:
            ax_which_taste[0,t_i].set_title(dig_in_names[t_i] + ' *.05')
        else:
            ax_which_taste[0,t_i].set_title(dig_in_names[t_i])
        
        ax_which_taste[0,t_i].set_xlabel('Fraction of All\nDeviation Events')
        plt.tight_layout()
        null_vals = null_taste_only_taste_fractions[:,t_i]
        ax_which_taste[1,t_i].hist(null_vals,label='Null Distribution')
        true_val = true_taste_only_taste_fractions[t_i]
        ax_which_taste[1,t_i].axvline(true_val,label='True Data',color='r')
        ax_which_taste[1,t_i].set_xlabel('Fraction of Taste Only\nDeviation Events')
        percentile_95 = np.percentile(null_vals,95)
        percentile_5 = np.percentile(null_vals,5)
        if true_val > percentile_95:
            ax_which_taste[1,t_i].set_title(dig_in_names[t_i] + ' *.95')
        elif true_val < percentile_5:
            ax_which_taste[1,t_i].set_title(dig_in_names[t_i] + ' *.05')
        else:
            ax_which_taste[1,t_i].set_title(dig_in_names[t_i])
        plt.tight_layout()
        if t_i == 0:
            ax_which_taste[0,t_i].legend(loc='upper right')
            ax_which_taste[0,t_i].set_ylabel('# Null Distributions')
            ax_which_taste[1,t_i].set_ylabel('# Null Distributions')
    f_which_taste.savefig(os.path.join(true_decode_dir,segment_names[s_i]
                         + '_frac_dev_same_taste_v_null.png'))
    f_which_taste.savefig(os.path.join(true_decode_dir,segment_names[s_i]
                         + '_frac_dev_same_taste_v_null.svg'))
    plt.close(f_which_taste)
    
    #Epoch Order Summaries
    same_taste_dev_decode_epoch_array = np.load(os.path.join(true_decode_dir,segment_names[s_i]+'_same_taste_dev_decode_epoch_array.npy'))
    for t_i in range(num_tastes-1):
        #True Data Epoch Order Counts
        same_taste_which_taste_inds = np.where(true_taste_inds == t_i)[0]
        taste_name = dig_in_names[t_i]
        taste_decode_probs = same_taste_dev_decode_epoch_array[same_taste_which_taste_inds,:,:] #decoded as taste x num epochs x num splits
        epoch_decode_argmax = np.argmax(taste_decode_probs,1)
        epoch_order_dict = {}
        for ep_i, ep in enumerate(epoch_splits):
            all_matches = []
            for split_i in range(num_splits):
                all_matches.append(epoch_decode_argmax[:,split_i] == ep[split_i])
            match_mult = np.prod(all_matches,0)
            epoch_order_dict[ep_i] = len(np.where(match_mult)[0])
        
        #Null Epoch Order Counts
        null_epoch_order_dict = {}
        for ep_i, ep in enumerate(epoch_splits):
            null_epoch_order_dict[ep_i] = []
            for null_i in range(num_null):
                null_epoch_order_dict[ep_i].extend([null_epoch_pair_counts[null_i][t_i][ep_i]])
                
        #Plot against each other
        f_epoch_order, ax_epoch_order = plt.subplots(nrows = epoch_split_plot_square, \
                                                     ncols = epoch_split_plot_square, \
                                                         figsize = (8,8))
        for ep_i, ep_name in enumerate(epoch_split_names):
            #Axis location
            ep_r, ep_c = np.where(epoch_split_plot_square_reference == ep_i)
            ep_r = ep_r[0]
            ep_c = ep_c[0]
            #Plot hist vs vertline
            null_vals = null_epoch_order_dict[ep_i]
            ax_epoch_order[ep_r,ep_c].hist(null_vals,\
                                           label='Null Distribution')
            true_val = epoch_order_dict[ep_i]
            ax_epoch_order[ep_r,ep_c].axvline(true_val,\
                                              label='Null Distribution',\
                                                  color='r')
            percentile_95 = np.percentile(null_vals,95)
            percentile_5 = np.percentile(null_vals,5)
            if true_val > percentile_95:
                ax_epoch_order[ep_r,ep_c].set_title(ep_name + ' *.95')
            elif true_val < percentile_5:
                ax_epoch_order[ep_r,ep_c].set_title(ep_name + ' *.05')
            else:
                ax_epoch_order[ep_r,ep_c].set_title(ep_name)
            if ep_c == 0:
                ax_epoch_order[ep_r,ep_c].set_ylabel('# Null Distributions')
            if ep_r == epoch_split_plot_square-1:
                ax_epoch_order[ep_r,ep_c].set_xlabel('# Occurrences')
            if (ep_r == 0)*(ep_c == 0):
                ax_epoch_order[ep_r,ep_c].legend(loc='upper left')
        plt.suptitle(taste_name + '\nEpoch Pair Data Comp')
        plt.tight_layout()
        f_epoch_order.savefig(os.path.join(true_decode_dir,segment_names[s_i]
                             + '_' + taste_name + '_frac_dev_epoch_order_v_null.png'))
        f_epoch_order.savefig(os.path.join(true_decode_dir,segment_names[s_i]
                             + '_' + taste_name + '_frac_dev_epoch_order_v_null.svg'))
        plt.close(f_epoch_order)
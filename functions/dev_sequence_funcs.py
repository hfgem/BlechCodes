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
from scipy.stats import f
from matplotlib import colormaps
from multiprocess import Pool
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as gmm

def split_match_calc(num_neur,segment_dev_rasters,segment_zscore_means,segment_zscore_stds,
                   tastant_raster_dict,tastant_fr_dist_pop,tastant_fr_dist_z_pop,
                   dig_in_names,segment_names,num_null,save_dir,
                   segments_to_analyze,epochs_to_analyze = []):
    """
    This function is dedicated to an analysis of whether, when a deviation event
    is split in half down the middle, the pair of sides looks similar to adjacent
    pairs of epochs for different tastes. To do so, this function calculates the
    firing rate vectors across the population for each event when split down 
    the middle, and then calculates the correlation between the resulting matrix
    and the taste epoch pair matrices. The function outputs the distributions 
    of these correlations into a plot.
    """
    
    #Split Dirs
    # dist_dir = os.path.join(save_dir, 'dist_tests')
    # if not os.path.isdir(dist_dir):
    #     os.mkdir(dist_dir)
    # corr_dir = os.path.join(save_dir, 'corr_tests')
    # if not os.path.isdir(corr_dir):
    #     os.mkdir(corr_dir)
    decode_split_dir = os.path.join(save_dir,'decode_splits')
    if not os.path.isdir(decode_split_dir):
        os.mkdir(decode_split_dir)
    non_z_decode_split_dir = os.path.join(decode_split_dir,'firing_rates')
    if not os.path.isdir(non_z_decode_split_dir):
        os.mkdir(non_z_decode_split_dir)
    z_decode_dir = os.path.join(decode_split_dir,'zscore_firing_rates')
    if not os.path.isdir(z_decode_dir):
        os.mkdir(z_decode_dir)
    null_decode_dir = os.path.join(non_z_decode_split_dir,'null_decodes_win_neur')
    if not os.path.isdir(null_decode_dir):
        os.mkdir(null_decode_dir)
    null_z_decode_dir = os.path.join(z_decode_dir,'null_decodes_win_neur')
    if not os.path.isdir(null_z_decode_dir):
        os.mkdir(null_z_decode_dir)
    null_decode_dir_2 = os.path.join(non_z_decode_split_dir,'null_decodes_across_neur')
    if not os.path.isdir(null_decode_dir_2):
        os.mkdir(null_decode_dir_2)
    null_z_decode_dir_2 = os.path.join(z_decode_dir,'null_decodes_across_neur')
    if not os.path.isdir(null_z_decode_dir_2):
        os.mkdir(null_z_decode_dir_2)
        
    # Sequence Dirs
    sequence_dir = os.path.join(save_dir,'sequence_tests')
    if not os.path.isdir(sequence_dir):
        os.mkdir(sequence_dir)
    null_sequence_dir = os.path.join(sequence_dir,'null_sequences_win_neur')
    if not os.path.isdir(null_sequence_dir):
        os.mkdir(null_sequence_dir)
    null_sequence_dir_2 = os.path.join(sequence_dir,'null_sequences_across_neur')
    if not os.path.isdir(null_sequence_dir_2):
        os.mkdir(null_sequence_dir_2)        
    
    # Variables
    bin_dt = 10 #Sequence binning size
    num_tastes = len(dig_in_names)
    num_taste_deliv = [len(tastant_fr_dist_pop[t_i]) for t_i in range(num_tastes)]
    max_num_cp = 0
    for t_i in range(num_tastes):
        for d_i in range(num_taste_deliv[t_i]):
            if len(tastant_fr_dist_pop[t_i][d_i]) > max_num_cp:
                max_num_cp = len(tastant_fr_dist_pop[t_i][d_i])
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(max_num_cp)
    
    taste_pairs = list(itertools.combinations(np.arange(num_tastes),2))
    taste_pair_names = []
    for tp_i, tp in enumerate(taste_pairs):
        taste_pair_names.append(dig_in_names[tp[0]] + ' v. ' + dig_in_names[tp[1]])
        
    #Calculate rank order sequences for taste responses by epoch
    taste_seqs_dict, avg_taste_seqs_dict = calc_tastant_seq(tastant_raster_dict, bin_dt)
    
    #Now go through segments and their deviation events and compare
    for seg_ind, s_i in enumerate(segments_to_analyze):
        seg_dev_rast = segment_dev_rasters[seg_ind]
        seg_z_mean = segment_zscore_means[seg_ind]
        seg_z_std = segment_zscore_stds[seg_ind]
        num_dev = len(seg_dev_rast)
        
        #Split in half calcs
        create_splits_run_calcs(num_null, num_dev, seg_dev_rast, seg_z_mean, 
                                seg_z_std, num_neur, dig_in_names, segment_names, 
                                s_i, epochs_to_analyze, tastant_fr_dist_pop, 
                                tastant_fr_dist_z_pop, z_decode_dir, 
                                null_z_decode_dir, null_z_decode_dir_2)
        
        #Sequence calcs
        create_sequence_run_calcs(num_null, num_dev, seg_dev_rast, seg_z_mean, 
                                seg_z_std, num_neur, dig_in_names, segment_names, 
                                s_i, bin_dt, epochs_to_analyze, tastant_raster_dict,
                                taste_seqs_dict, avg_taste_seqs_dict, sequence_dir, 
                                null_sequence_dir, null_sequence_dir_2)
        
def create_splits_run_calcs(num_null, num_dev, seg_dev_rast, seg_z_mean, seg_z_std, 
                            num_neur, dig_in_names, segment_names, s_i, 
                            epochs_to_analyze, tastant_fr_dist_pop, tastant_fr_dist_z_pop,
                            z_decode_dir, null_z_decode_dir, null_z_decode_dir_2):
    dev_mats = []
    dev_mats_z = []
    null_dev_dict = dict()
    null_dev_z_dict = dict()
    null_dev_dict_2 = dict()
    null_dev_z_dict_2 = dict()
    for null_i in range(num_null):
        null_dev_dict[null_i] = []
        null_dev_z_dict[null_i] = []
        null_dev_dict_2[null_i] = []
        null_dev_z_dict_2[null_i] = []
    for dev_i in range(num_dev):
        #Pull raster for firing rate vectors
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
            #Shuffle within-neuron spike times
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
            #Shuffle across-neuron spike times
            shuffle_rast_2 = np.zeros(np.shape(dev_rast))
            new_neuron_order = random.sample(list(np.arange(num_neur)),num_neur)
            for nn_ind, nn in enumerate(new_neuron_order):
                shuffle_rast_2[nn_ind,:] = shuffle_rast[nn,:]
            first_half_shuffle_rast_2 = shuffle_rast_2[:,:half_dt]
            second_half_shuffle_rast_2 = shuffle_rast_2[:,-half_dt:]
            #Create fr vecs
            first_half_fr_vec_2 = np.expand_dims(np.sum(first_half_shuffle_rast_2,1)/(half_dt/1000),1) #In Hz
            second_half_fr_vec_2 = np.expand_dims(np.sum(second_half_shuffle_rast_2,1)/(half_dt/1000),1) #In Hz
            shuffle_dev_mat_2 = np.concatenate((first_half_fr_vec_2,second_half_fr_vec_2),1)
            #Create z-scored fr vecs
            first_half_fr_vec_z_2 = (first_half_fr_vec_2 - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
            second_half_fr_vec_z_2 = (second_half_fr_vec_2 - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
            shuffle_dev_mat_z_2 = np.concatenate((first_half_fr_vec_z_2,second_half_fr_vec_z_2),1)
            null_dev_dict_2[null_i].append(shuffle_dev_mat_2)
            null_dev_z_dict_2[null_i].append(shuffle_dev_mat_z_2)      
        
    dev_mats_array = np.array(dev_mats) #num dev x num neur x 2
    dev_mats_z_array = np.array(dev_mats_z) #num dev x num neur x 2
    for null_i in range(num_null):
        null_dev_dict[null_i] = np.array(null_dev_dict[null_i]) #num dev x num neur x 2
        null_dev_z_dict[null_i] = np.array(null_dev_z_dict[null_i]) #num dev x num neur x 2
        null_dev_dict_2[null_i] = np.array(null_dev_dict_2[null_i]) #num dev x num neur x 2
        null_dev_z_dict_2[null_i] = np.array(null_dev_z_dict_2[null_i]) #num dev x num neur x 2
        
    #Decode each deviation event split
    
    # decode_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist_pop, 
    #                 dig_in_names, dev_mats_array, segment_names, s_i,
    #                 non_z_decode_dir, epochs_to_analyze)
    decode_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist_z_pop, 
                    dig_in_names, dev_mats_z_array, segment_names, s_i,
                    z_decode_dir, epochs_to_analyze)
    
    #Run decoded splits significance tests
    decode_splits_significance_tests(dig_in_names, dev_mats_z_array, segment_names, 
                                         s_i, z_decode_dir, epochs_to_analyze)
    
    #Decode null distribution
    # decode_null_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist_pop, 
    #                 dig_in_names, null_dev_dict, segment_names, s_i,
    #                 null_decode_dir, non_z_decode_dir, epochs_to_analyze)
    decode_null_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist_z_pop, 
                    dig_in_names, null_dev_z_dict, segment_names, s_i,
                    null_z_decode_dir, z_decode_dir, epochs_to_analyze)
    # decode_null_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist_pop, 
    #                 dig_in_names, null_dev_dict_2, segment_names, s_i,
    #                 null_decode_dir_2, non_z_decode_dir, epochs_to_analyze)
    decode_null_deviation_splits_is_taste_which_taste_which_epoch(tastant_fr_dist_z_pop, 
                    dig_in_names, null_dev_z_dict_2, segment_names, s_i,
                    null_z_decode_dir_2, z_decode_dir, epochs_to_analyze)
                     
def calc_tastant_seq(tastant_raster_dict, bin_dt):
    """This function is dedicated to calculating the rank sequences in tastant
    delivery responses"""
    #Gather variables
    num_tastes = len(tastant_raster_dict)
    num_deliv_per_taste = np.zeros(num_tastes).astype('int')
    max_num_cp = 0
    for t_i in range(num_tastes):
        num_deliv_per_taste[t_i] = len(tastant_raster_dict[t_i])
        for d_i in range(num_deliv_per_taste[t_i]):
            if len(tastant_raster_dict[t_i][d_i]) > max_num_cp:
                max_num_cp = len(tastant_raster_dict[t_i][d_i])
    
    #Organize taste responses into rank sequences based on maximal firing 
    #location of each neuron within an epoch of response
    taste_seqs_dict = dict()
    avg_taste_seqs_dict = dict()
    for t_i in range(num_tastes):
        taste_seqs_dict[t_i] = dict()
        avg_taste_seqs_dict[t_i] = []
        for cp_i in range(max_num_cp):
            taste_seqs_dict[t_i][cp_i] = []
            for d_i in range(num_deliv_per_taste[t_i].astype('int')):
                try:
                    taste_rast = tastant_raster_dict[t_i][d_i][cp_i]
                    num_spikes_per_neur = np.sum(taste_rast,1).astype('int')
                    num_neur, num_dt = np.shape(taste_rast)
                    bin_starts = np.arange(num_dt-bin_dt)
                    taste_bin_counts = np.zeros((num_neur,len(bin_starts)))
                    for bs in bin_starts:
                        taste_bin_counts[:,bs] = np.sum(taste_rast[:,bs:bs+bin_dt],1)
                    max_counts = np.max(taste_bin_counts,1)
                    max_ind = calc_max_ind(num_neur, taste_bin_counts, max_counts)
                    neur_ord = np.argsort(max_ind, kind='stable')
                    taste_seqs_dict[t_i][cp_i].append(neur_ord)
                except:
                    print("Missing data: taste " + str(t_i) + " epoch " + str(cp_i) + " delivery " + str(d_i))
            deliv_seq_array = np.array(taste_seqs_dict[t_i][cp_i])
            mean_rank = np.mean(deliv_seq_array,0)
            avg_taste_seqs_dict[t_i].append(np.argsort(mean_rank))
    
    return taste_seqs_dict, avg_taste_seqs_dict

def create_sequence_run_calcs(num_null, num_dev, seg_dev_rast, seg_z_mean, 
                        seg_z_std, num_neur, dig_in_names, segment_names, 
                        s_i, bin_dt, epochs_to_analyze, tastant_raster_dict, 
                        taste_seqs_dict, avg_taste_seqs_dict, sequence_dir, 
                        null_sequence_dir, null_sequence_dir_2):
    try: #Import of pre-calculated sequences
        dev_seqs_array = np.load(os.path.join(sequence_dir,segment_names[s_i] + '_dev_seqs_array.npy'),allow_pickle=True)
        null_dev_dict = np.load(os.path.join(null_sequence_dir,segment_names[s_i] + '_null_dev_dict.npy'),allow_pickle=True).item()
        null_dev_dict_2 = np.load(os.path.join(null_sequence_dir_2,segment_names[s_i] + '_null_dev_dict.npy'),allow_pickle=True).item()
        print("\t\t\t\t\tImported previously calculated sequences for " + segment_names[s_i])
    except: #Create sequences
        tic = time.time()
        dev_seqs = []
        null_dev_dict = dict()
        null_dev_dict_2 = dict()
        for null_i in range(num_null):
            null_dev_dict[null_i] = []
            null_dev_dict_2[null_i] = []
            
        #Calculate deviation sequences
        for dev_i in range(num_dev):
            #Pull raster and count up firing within bins
            dev_rast = seg_dev_rast[dev_i]
            num_spikes_per_neur = np.sum(dev_rast,1).astype('int')
            _, num_dt = np.shape(dev_rast)
            bin_starts = np.arange(num_dt-bin_dt)
            dev_bin_counts = np.zeros((num_neur,len(bin_starts)))
            for bs in bin_starts:
                dev_bin_counts[:,bs] = np.sum(dev_rast[:,bs:bs+bin_dt],1)
            max_counts = np.max(dev_bin_counts,1)
            #Neuron Ordering: note that if multiple neurons fire the same number 
            #of times, it will order them by index 
            max_ind = calc_max_ind(num_neur, dev_bin_counts, max_counts)
            neur_ord = np.argsort(max_ind, kind='stable')
            dev_seqs.append(neur_ord)
            
            #Create null versions of the event
            for null_i in range(num_null):
                #Shuffle within-neuron spike times
                shuffle_rast = np.zeros(np.shape(dev_rast))
                for neur_i in range(num_neur):
                    new_spike_ind = random.sample(list(np.arange(num_dt)),num_spikes_per_neur[neur_i])
                    shuffle_rast[neur_i,new_spike_ind] = 1
                dev_bin_counts = np.zeros((num_neur,len(bin_starts)))
                for bs in bin_starts:
                    dev_bin_counts[:,bs] = np.sum(shuffle_rast[:,bs:bs+bin_dt],1)
                max_counts = np.max(dev_bin_counts,1)
                max_ind = calc_max_ind(num_neur, dev_bin_counts, max_counts)
                neur_ord = np.argsort(max_ind, kind='stable')
                null_dev_dict[null_i].append(neur_ord)
                #Shuffle across-neuron spike times
                shuffle_rast_2 = np.zeros(np.shape(dev_rast))
                new_neuron_order = random.sample(list(np.arange(num_neur)),num_neur)
                for nn_ind, nn in enumerate(new_neuron_order):
                    shuffle_rast_2[nn_ind,:] = shuffle_rast[nn,:]
                dev_bin_counts = np.zeros((num_neur,len(bin_starts)))
                for bs in bin_starts:
                    dev_bin_counts[:,bs] = np.sum(shuffle_rast_2[:,bs:bs+bin_dt],1)
                max_counts = np.max(dev_bin_counts,1)
                max_ind = calc_max_ind(num_neur, dev_bin_counts, max_counts)
                neur_ord = np.argsort(max_ind, kind='stable')
                null_dev_dict_2[null_i].append(neur_ord)
                
        dev_seqs_array = np.array(dev_seqs) #num dev x num neur
        for null_i in range(num_null):
            null_dev_dict[null_i] = np.array(null_dev_dict[null_i]) #num dev x num neur x 2
            null_dev_dict_2[null_i] = np.array(null_dev_dict_2[null_i]) #num dev x num neur x 2
        toc = time.time()
        print('\t\t\t\t\tTime to calculate sequences for ' + segment_names[s_i] + \
              ' = ' + str(np.round((toc-tic)/60, 2)) + ' (min)')
        np.save(os.path.join(sequence_dir,segment_names[s_i] + '_dev_seqs_array.npy'),dev_seqs_array,allow_pickle=True)
        np.save(os.path.join(null_sequence_dir,segment_names[s_i] + '_null_dev_dict.npy'),null_dev_dict,allow_pickle=True)
        np.save(os.path.join(null_sequence_dir_2,segment_names[s_i] + '_null_dev_dict.npy'),null_dev_dict_2,allow_pickle=True)
        
    #Run rank order correlations for null data and true data and pull out 
    #significant events
    dev_rank_order_tests(dev_seqs_array, null_dev_dict, avg_taste_seqs_dict, 
                             dig_in_names, segment_names[s_i], sequence_dir, null_sequence_dir)
    
    dev_rank_order_tests(dev_seqs_array, null_dev_dict_2, avg_taste_seqs_dict, 
                             dig_in_names, segment_names[s_i], sequence_dir, null_sequence_dir_2)
    
def calc_max_ind(num_neur, dev_bin_counts, max_counts):
    """This function calculates the index for each neuron where its maximal
    firing occurs within a deviation event"""
    max_ind = np.zeros(num_neur)
    for n_i in range(num_neur):
        max_locs = np.where(dev_bin_counts[n_i,:] == max_counts[n_i])[0]
        if len(max_locs) > 1:
            diff_max_locs = np.diff(max_locs)
            mult_max_locs = np.where(diff_max_locs > 1)[0] + 1
            if len(mult_max_locs) > 0:
                mult_max_locs = np.concatenate((np.zeros(1),mult_max_locs,(len(max_locs)-1)*np.ones(1))).astype('int')
                #Multiple peak firing locations
                mean_parts = []
                for part_i in range(len(mult_max_locs)-1):
                    ind_1 = mult_max_locs[part_i]
                    ind_2 = mult_max_locs[part_i+1]
                    mean_parts.extend([np.ceil(max_locs[ind_1] + (max_locs[ind_2] - max_locs[ind_1])/2).astype('int')])
                max_ind[n_i] = np.nanmean(mean_parts).astype('int')
            else:
                mean_max_loc = np.ceil(max_locs[0] + (max_locs[-1] - max_locs[0])/2).astype('int')
                max_ind[n_i] = mean_max_loc #Split the difference
        else:
            max_ind[n_i] = max_locs
    return max_ind

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
            
            if len(same_taste_ind) > 0: #There are events that have the same taste decoded in all splits
                taste_decode.extend([same_taste_ind])    
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
    f_frac_all_taste.savefig(os.path.join(null_decode_dir,segment_names[s_i]
                         + '_frac_dev_all_taste_v_null.png'))
    f_frac_all_taste.savefig(os.path.join(null_decode_dir,segment_names[s_i]
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
        null_vals_no_nan = null_vals[~np.isnan(null_vals)]
        true_val = true_taste_fractions[t_i]
        if len(null_vals_no_nan) > 0:
            ax_which_taste[0,t_i].hist(null_vals_no_nan,label='Null Distribution')
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
        else:
            ax_which_taste[0,t_i].axvline(true_val,label='True Data',color='r')
            ax_which_taste[0,t_i].set_title(dig_in_names[t_i] + ' *.95')
        plt.tight_layout()
        null_vals = null_taste_only_taste_fractions[:,t_i]
        null_vals_no_nan = null_vals[~np.isnan(null_vals)]
        true_val = true_taste_only_taste_fractions[t_i]
        if len(null_vals_no_nan) > 0:
            ax_which_taste[1,t_i].hist(null_vals_no_nan,label='Null Distribution')
            ax_which_taste[1,t_i].axvline(true_val,label='True Data',color='r')
            ax_which_taste[1,t_i].set_xlabel('Fraction of Taste Only\nDeviation Events')
            percentile_95 = np.percentile(null_vals_no_nan,95)
            percentile_5 = np.percentile(null_vals_no_nan,5)
            if true_val > percentile_95:
                ax_which_taste[1,t_i].set_title(dig_in_names[t_i] + ' *.95')
            elif true_val < percentile_5:
                ax_which_taste[1,t_i].set_title(dig_in_names[t_i] + ' *.05')
            else:
                ax_which_taste[1,t_i].set_title(dig_in_names[t_i])
        else:
            ax_which_taste[1,t_i].axvline(true_val,label='True Data',color='r')
            ax_which_taste[1,t_i].set_xlabel('Fraction of Taste Only\nDeviation Events')
            ax_which_taste[1,t_i].set_title(dig_in_names[t_i] + ' *.95')
        plt.tight_layout()
        if t_i == 0:
            ax_which_taste[0,t_i].legend(loc='upper right')
            ax_which_taste[0,t_i].set_ylabel('# Null Distributions')
            ax_which_taste[1,t_i].set_ylabel('# Null Distributions')
    f_which_taste.savefig(os.path.join(null_decode_dir,segment_names[s_i]
                         + '_frac_dev_same_taste_v_null.png'))
    f_which_taste.savefig(os.path.join(null_decode_dir,segment_names[s_i]
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
        f_epoch_order.savefig(os.path.join(null_decode_dir,segment_names[s_i]
                             + '_' + taste_name + '_frac_dev_epoch_order_v_null.png'))
        f_epoch_order.savefig(os.path.join(null_decode_dir,segment_names[s_i]
                             + '_' + taste_name + '_frac_dev_epoch_order_v_null.svg'))
        plt.close(f_epoch_order)
        
def decode_splits_significance_tests(dig_in_names, dev_mats_array, segment_names, 
                                     s_i, splits_decode_dir, epochs_to_analyze=[]):
    """Decode taste from epoch-specific firing rates"""
    print('\t\tRunning ' + segment_names[s_i] + ' Decode Split Significance Tests')
    
    # Variables
    num_tastes = len(dig_in_names)
    num_dev, num_neur, num_splits = np.shape(dev_mats_array)
    split_pairs = list(itertools.combinations(np.arange(num_splits), 2))    
    sqrt_neur = np.ceil(np.sqrt(num_neur)).astype('int')
    neur_inds = np.reshape(np.concatenate((np.arange(num_neur),-1*np.ones(sqrt_neur**2 - num_neur))).squeeze(),
                           (sqrt_neur,sqrt_neur))
    
    #Significance test fr distribution differences against each other
    #for two halves
    sig_storage = np.zeros((len(split_pairs),3,2 + num_tastes-1)) #Rows: Hotelling's T-Squared, F-statistic, p-val x Cols: is-taste, same-taste, which-taste
    
    dev_decode_is_taste_array = np.load(
        os.path.join(splits_decode_dir,segment_names[s_i] + \
                     '_deviations_is_taste.npy'))
    dev_is_taste_argmax = []
    for dev_i in range(num_dev):
        is_taste_argmax = np.argmax(dev_decode_is_taste_array[dev_i,:,:].squeeze(),0)
        dev_is_taste_argmax.append(is_taste_argmax)
    dev_is_taste_argmax_sum = np.sum(np.array(dev_is_taste_argmax),1)
    dev_is_taste_inds = np.where(dev_is_taste_argmax_sum == 0)[0]
    if len(dev_is_taste_inds) > 0: #at least some devs decoded as fully taste
        dev_is_taste_splits = [dev_mats_array[dev_is_taste_inds,:,sp_i].squeeze() for sp_i in range(num_splits)]
        for sp_ind, sp in enumerate(split_pairs):
            vals_1 = dev_is_taste_splits[sp[0]]
            vals_2 = dev_is_taste_splits[sp[1]]
            # Z-score the data to fall on either side of 0
            all_vals = np.concatenate((vals_1,vals_2),axis=0) #(number of splits x 2) x num_neur
            neur_mean = np.nanmean(all_vals,0)
            neur_std = np.nanstd(all_vals,0)
            vals_1_z = (vals_1 - np.ones(np.shape(vals_1))*neur_mean)/(np.ones(np.shape(vals_1))*neur_std)
            vals_2_z = (vals_2 - np.ones(np.shape(vals_2))*neur_mean)/(np.ones(np.shape(vals_2))*neur_std)
            nan_inds = np.where(neur_std == 0)[0]
            non_nan_inds = np.setdiff1d(np.arange(num_neur),nan_inds)
            vals_1_z = vals_1_z[:,non_nan_inds]
            vals_2_z = vals_2_z[:,non_nan_inds]
            # Perform the Hotelling's t-squared test
            try:
                T2, F_hot, p_value = hotelling_t2(vals_1_z, vals_2_z)
                sig_storage[sp_ind,:,0] = [T2, F_hot, p_value]
                title = "Hotelling's T-squared: " + str(np.round(T2,2)) + \
                    "\nF-statistic: " + str(np.round(F_hot,3)) + \
                        "\np-value: " + str(np.round(p_value,4))
            except:
                sig_storage[sp_ind,:,0] = [np.nan, np.nan, np.nan]
                title = "Not enough samples for hotelling's t"
            
            f_plot, ax = plt.subplots(nrows = sqrt_neur, ncols = sqrt_neur,
                                 figsize = (8,8))
            for n_i in range(len(non_nan_inds)):
                ax_ind = np.where(neur_inds == n_i)
                ax[ax_ind[0][0],ax_ind[1][0]].hist(vals_1_z[:,n_i],alpha=0.2,label='Split ' + str(sp[0]))
                ax[ax_ind[0][0],ax_ind[1][0]].hist(vals_2_z[:,n_i],alpha=0.2,label='Split ' + str(sp[1]))
                ax[ax_ind[0][0],ax_ind[1][0]].set_title('Neuron ' + str(n_i))
                ax[ax_ind[0][0],ax_ind[1][0]].set_xlabel('Firing Rate (Hz)')
            ax[0,0].legend()
            plt.suptitle(title)
            plt.tight_layout()
            f_plot.savefig(os.path.join(splits_decode_dir,segment_names[s_i] + \
                         '_' + str(sp[0]) + '_vs_' + str(sp[1]) + \
                             '_deviations_is_taste_split_sig.png'))
            f_plot.savefig(os.path.join(splits_decode_dir,segment_names[s_i] + \
                         '_' + str(sp[0]) + '_vs_' + str(sp[1]) + \
                             '_deviations_is_taste_split_sig.svg'))
            plt.close(f_plot)
            
        #Now look at sig for those that are the same taste
        dev_decode_array = np.load(
        os.path.join(splits_decode_dir,segment_names[s_i] + \
                     '_deviations_which_taste.npy'))
        dev_which_taste_argmax = []
        for dev_i in dev_is_taste_inds:
            which_taste_argmax = np.argmax(dev_decode_array[dev_i,:,:],0)
            dev_which_taste_argmax.append(which_taste_argmax)
        dev_which_taste_argmax_array = np.array(dev_which_taste_argmax)
        same_taste_test_ind = dev_which_taste_argmax_array[:,0]
        same_taste_bool = [] #True or false for each event whether same across splits
        for i in range(len(same_taste_test_ind)):
            same_taste_bool.append(all([dev_which_taste_argmax_array[i,j] == same_taste_test_ind[i] for j in range(num_splits)]))
        same_taste_ind = np.where(same_taste_bool)[0]
        if len(same_taste_ind) > 0: #There are events that have the same taste decoded in all splits
            dev_same_taste_inds = dev_is_taste_inds[same_taste_ind]
            dev_same_taste_splits = [dev_mats_array[dev_same_taste_inds,:,sp_i].squeeze() for sp_i in range(num_splits)]
            for sp_ind, sp in enumerate(split_pairs):
                vals_1 = dev_same_taste_splits[sp[0]]
                vals_2 = dev_same_taste_splits[sp[1]]
                # Z-score the data to fall on either side of 0
                all_vals = np.concatenate((vals_1,vals_2),axis=0) #(number of splits x 2) x num_neur
                neur_mean = np.nanmean(all_vals,0)
                neur_std = np.nanstd(all_vals,0)
                vals_1_z = (vals_1 - np.ones(np.shape(vals_1))*neur_mean)/(np.ones(np.shape(vals_1))*neur_std)
                vals_2_z = (vals_2 - np.ones(np.shape(vals_2))*neur_mean)/(np.ones(np.shape(vals_2))*neur_std)
                nan_inds = np.where(neur_std == 0)[0]
                non_nan_inds = np.setdiff1d(np.arange(num_neur),nan_inds)
                vals_1_z = vals_1_z[:,non_nan_inds]
                vals_2_z = vals_2_z[:,non_nan_inds]
                # Perform the Hotelling's t-squared test
                try:
                    T2, F_hot, p_value = hotelling_t2(vals_1_z, vals_2_z)
                    sig_storage[sp_ind,:,1] = [T2, F_hot, p_value]
                    title = "Hotelling's T-squared: " + str(np.round(T2,2)) + \
                        "\nF-statistic: " + str(np.round(F_hot,3)) + \
                            "\np-value: " + str(np.round(p_value,4))
                except:
                    sig_storage[sp_ind,:,1] = [np.nan, np.nan, np.nan]
                    title = "Not enough samples for hotelling's t"
                
                f_plot, ax = plt.subplots(nrows = sqrt_neur, ncols = sqrt_neur,
                                     figsize = (8,8))
                for n_i in range(len(non_nan_inds)):
                    ax_ind = np.where(neur_inds == n_i)
                    ax[ax_ind[0][0],ax_ind[1][0]].hist(vals_1_z[:,n_i],alpha=0.2,label='Split ' + str(sp[0]))
                    ax[ax_ind[0][0],ax_ind[1][0]].hist(vals_2_z[:,n_i],alpha=0.2,label='Split ' + str(sp[1]))
                    ax[ax_ind[0][0],ax_ind[1][0]].set_title('Neuron ' + str(n_i))
                    ax[ax_ind[0][0],ax_ind[1][0]].set_xlabel('Firing Rate (Hz)')
                ax[0,0].legend()
                plt.suptitle(title)
                plt.tight_layout()
                f_plot.savefig(os.path.join(splits_decode_dir,segment_names[s_i] + \
                             '_' + str(sp[0]) + '_vs_' + str(sp[1]) + \
                                 '_deviations_same_taste_split_sig.png'))
                f_plot.savefig(os.path.join(splits_decode_dir,segment_names[s_i] + \
                             '_' + str(sp[0]) + '_vs_' + str(sp[1]) + \
                                 '_deviations_same_taste_split_sig.svg'))
                plt.close(f_plot)
                
            #Now split it up by which taste the splits are decoded as
            for t_i in range(num_tastes-1):
                dev_which_taste_inds = dev_is_taste_inds[np.where(same_taste_test_ind[same_taste_ind] == t_i)[0]]
                dev_which_taste_splits = [dev_mats_array[dev_which_taste_inds,:,sp_i].squeeze() for sp_i in range(num_splits)]
                for sp_ind, sp in enumerate(split_pairs):
                    vals_1 = dev_which_taste_splits[sp[0]]
                    vals_2 = dev_which_taste_splits[sp[1]]
                    # Z-score the data to fall on either side of 0
                    all_vals = np.concatenate((vals_1,vals_2),axis=0) #(number of splits x 2) x num_neur
                    neur_mean = np.nanmean(all_vals,0)
                    neur_std = np.nanstd(all_vals,0)
                    vals_1_z = (vals_1 - np.ones(np.shape(vals_1))*neur_mean)/(np.ones(np.shape(vals_1))*neur_std)
                    vals_2_z = (vals_2 - np.ones(np.shape(vals_2))*neur_mean)/(np.ones(np.shape(vals_2))*neur_std)
                    nan_inds = np.where(neur_std == 0)[0]
                    non_nan_inds = np.setdiff1d(np.arange(num_neur),nan_inds)
                    vals_1_z = vals_1_z[:,non_nan_inds]
                    vals_2_z = vals_2_z[:,non_nan_inds]
                    # Perform the Hotelling's t-squared test
                    try:
                        T2, F_hot, p_value = hotelling_t2(vals_1_z, vals_2_z)
                        sig_storage[sp_ind,:,2+t_i] = [T2, F_hot, p_value]
                        title = "Hotelling's T-squared: " + str(np.round(T2,2)) + \
                            "\nF-statistic: " + str(np.round(F_hot,3)) + \
                                "\np-value: " + str(np.round(p_value,4))
                    except:
                        sig_storage[sp_ind,:,2+t_i] = [np.nan, np.nan, np.nan]
                        title = "Not enough samples for hotelling's t"
                    
                    f_plot, ax = plt.subplots(nrows = sqrt_neur, ncols = sqrt_neur,
                                         figsize = (8,8))
                    for n_i in range(len(non_nan_inds)):
                        ax_ind = np.where(neur_inds == n_i)
                        ax[ax_ind[0][0],ax_ind[1][0]].hist(vals_1_z[:,n_i],alpha=0.2,label='Split ' + str(sp[0]))
                        ax[ax_ind[0][0],ax_ind[1][0]].hist(vals_2_z[:,n_i],alpha=0.2,label='Split ' + str(sp[1]))
                        ax[ax_ind[0][0],ax_ind[1][0]].set_title('Neuron ' + str(n_i))
                        ax[ax_ind[0][0],ax_ind[1][0]].set_xlabel('Firing Rate (Hz)')
                    ax[0,0].legend()
                    plt.suptitle(title)
                    plt.tight_layout()
                    f_plot.savefig(os.path.join(splits_decode_dir,segment_names[s_i] + \
                                 '_' + str(sp[0]) + '_vs_' + str(sp[1]) + \
                                     '_' + dig_in_names[t_i] + \
                                         '_deviations_which_taste_split_sig.png'))
                    f_plot.savefig(os.path.join(splits_decode_dir,segment_names[s_i] + \
                                 '_' + str(sp[0]) + '_vs_' + str(sp[1]) + \
                                     '_' + dig_in_names[t_i] + \
                                         '_deviations_which_taste_split_sig.svg'))
                    plt.close(f_plot)
    
    for sp_ind, sp in enumerate(split_pairs):
        sig_csv_file = os.path.join(splits_decode_dir,'split_sig_hotellings_' + str(sp[0]) + '_x_' + str(sp[1]) + '.csv')
        if os.path.isfile(sig_csv_file):
            with open(sig_csv_file, 'a') as f:
                write = csv.writer(f, delimiter=',')
                hot_t_list = [segment_names[s_i], 'Hotellings T']
                hot_t_list.extend([sig_storage[sp_ind,0,i] for i in range(2+num_tastes-1)])
                write.writerow(hot_t_list)
                f_stat_list = [segment_names[s_i], 'F-Stat']
                f_stat_list.extend([sig_storage[sp_ind,1,i] for i in range(2+num_tastes-1)])
                write.writerow(f_stat_list)
                p_val_list = [segment_names[s_i], 'p-val']
                p_val_list.extend([sig_storage[sp_ind,2,i] for i in range(2+num_tastes-1)])
                write.writerow(p_val_list)
        else:
            with open(sig_csv_file, 'w') as f:
                write = csv.writer(f, delimiter=',')
                title_list = ['Seg Name', 'Stat Name', 'Is-Taste', 'Same-Taste']
                for t_i in range(num_tastes-1):
                    title_list.extend([dig_in_names[t_i]])
                write.writerow(title_list)
                hot_t_list = [segment_names[s_i], 'Hotellings T']
                hot_t_list.extend([sig_storage[sp_ind,0,i] for i in range(2+num_tastes-1)])
                write.writerow(hot_t_list)
                f_stat_list = [segment_names[s_i], 'F-Stat']
                f_stat_list.extend([sig_storage[sp_ind,1,i] for i in range(2+num_tastes-1)])
                write.writerow(f_stat_list)
                p_val_list = [segment_names[s_i], 'p-val']
                p_val_list.extend([sig_storage[sp_ind,2,i] for i in range(2+num_tastes-1)])
                write.writerow(p_val_list)
                
def hotelling_t2(X, Y):
    """
    Perform Hotelling's T-squared test on two samples.
    """
    n1, p = X.shape
    n2, _ = Y.shape

    X_bar = np.mean(X, axis=0)
    Y_bar = np.mean(Y, axis=0)

    S_pooled = ((n1 - 1) * np.cov(X.T) + (n2 - 1) * np.cov(Y.T)) / (n1 + n2 - 2)

    T2 = ((n1 * n2) / (n1 + n2)) * (X_bar - Y_bar) @ np.linalg.inv(S_pooled) @ (X_bar - Y_bar).T

    F = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * T2

    p_value = 1 - f.cdf(F, p, n1 + n2 - p - 1)

    return T2, F, p_value

def dev_rank_order_tests(dev_seqs_array, null_dev_dict, avg_taste_seqs_dict, 
                         dig_in_names, seg_name, sequence_dir, null_sequence_dir):
    """This function tests whether neurons tend to have a particular 
    rank order of firing within a deviation event"""
    
    #Gather variables
    num_tastes = len(avg_taste_seqs_dict)
    max_num_cp = 0
    for t_i in range(num_tastes):
        if len(avg_taste_seqs_dict[t_i]) > max_num_cp:
            max_num_cp = len(avg_taste_seqs_dict[t_i])
    num_dev, num_neur = np.shape(dev_seqs_array)
    num_null = len(null_dev_dict)
    
    #Run through and calculate Spearman's correlation distributions
    spearmans_dict = dict()
    for t_i in range(num_tastes):
        spearmans_dict[t_i] = dict()
        for cp_i in range(max_num_cp):
            spearmans_dict[t_i][cp_i] = dict()
            #Calculate deviation correlations
            true_corrs = np.zeros(num_dev)
            for dev_i in range(num_dev):
                res = stats.spearmanr(dev_seqs_array[dev_i,:], avg_taste_seqs_dict[t_i][cp_i], axis=0, nan_policy='omit', alternative='two-sided')
                true_corrs[dev_i] = res[0]
            spearmans_dict[t_i][cp_i]['true'] = true_corrs
            #Calculate null correlations
            null_corrs = np.zeros((num_null, num_dev))
            for null_i in range(num_null):
                null_devs = null_dev_dict[null_i]
                for dev_i in range(num_dev):
                    res = stats.spearmanr(null_devs[dev_i,:], avg_taste_seqs_dict[t_i][cp_i], axis=0, nan_policy='omit', alternative='two-sided')
                    null_corrs[null_i,dev_i] = res[0]
            spearmans_dict[t_i][cp_i]['null'] = null_corrs
            #Since these are rank order corrs we'll absolute value them and calculate the 95th percentile for null
            percentile_95 = np.percentile(np.abs(null_corrs.flatten()),95)
            spearmans_dict[t_i][cp_i]['percentile_95'] = percentile_95
            sig_true_corrs = np.where(np.abs(true_corrs) > percentile_95)[0]
            spearmans_dict[t_i][cp_i]['sig_true_inds'] = sig_true_corrs
    
    #Now create plots of distributions and statistics
    
    #Indiv Distributions True vs Null
    f_dist_pdf, ax_dist_pdf = plt.subplots(nrows = num_tastes, ncols = max_num_cp, figsize = (8,8),
                                   sharex = True, sharey = True)
    f_dist_cdf, ax_dist_cdf = plt.subplots(nrows = num_tastes, ncols = max_num_cp, figsize = (8,8),
                                   sharex = True, sharey = True)
    for t_i in range(num_tastes):
        for cp_i in range(max_num_cp):
            ax_dist_pdf[t_i,cp_i].hist(np.abs(spearmans_dict[t_i][cp_i]['null'].flatten()),
                                       bins = 50, density=True,
                                       histtype='step',label='Null')
            ax_dist_cdf[t_i,cp_i].hist(np.abs(spearmans_dict[t_i][cp_i]['null'].flatten()),
                                       bins = 100, density=True, cumulative=True,
                                       histtype='step',label='Null')
            ax_dist_pdf[t_i,cp_i].hist(np.abs(spearmans_dict[t_i][cp_i]['true']),
                                       bins = 50, density=True,
                                       histtype='step',label='True')
            ax_dist_cdf[t_i,cp_i].hist(np.abs(spearmans_dict[t_i][cp_i]['true']),
                                       bins = 100, density=True, cumulative=True,
                                       histtype='step',label='True')
            ax_dist_pdf[t_i,cp_i].axvline(spearmans_dict[t_i][cp_i]['percentile_95'],
                                          color='r',label='Null 95th Percentile')
            ax_dist_cdf[t_i,cp_i].axvline(spearmans_dict[t_i][cp_i]['percentile_95'],
                                          color='r',label='Null 95th Percentile')
            num_sig = len(spearmans_dict[t_i][cp_i]['sig_true_inds'])
            title = str(np.round(100*num_sig/num_dev,2)) + ' Percent Significant'
            if t_i == 0:
                ax_dist_pdf[t_i,cp_i].set_title('Epoch ' + str(cp_i) + '\n' + title)
                ax_dist_cdf[t_i,cp_i].set_title('Epoch ' + str(cp_i) + '\n' + title)
            else:
                ax_dist_pdf[t_i,cp_i].set_title(title)
                ax_dist_cdf[t_i,cp_i].set_title(title)
            if cp_i == 0:
                ax_dist_pdf[t_i,cp_i].set_ylabel(dig_in_names[t_i] + '\nDensity')
                ax_dist_cdf[t_i,cp_i].set_ylabel(dig_in_names[t_i] + '\nCumulative Density')
            if t_i == num_tastes-1:
                ax_dist_pdf[t_i,cp_i].set_xlabel('Spearman Correlation')
                ax_dist_cdf[t_i,cp_i].set_xlabel('Spearman Correlation')
            if (t_i == 0)*(cp_i == 0):
                ax_dist_pdf[t_i,cp_i].legend(loc = 'upper left')
                ax_dist_cdf[t_i,cp_i].legend(loc = 'upper left')
    f_dist_pdf.suptitle(seg_name + ' Probability Density Distributions')
    plt.tight_layout()
    f_dist_pdf.savefig(os.path.join(null_sequence_dir,seg_name+'_taste_epoch_spearman_pdfs.png'))
    f_dist_pdf.savefig(os.path.join(null_sequence_dir,seg_name+'_taste_epoch_spearman_pdfs.svg'))
    plt.close(f_dist_pdf)
    f_dist_cdf.suptitle(seg_name + ' Cumulative Probability Density Distributions')
    plt.tight_layout()
    f_dist_cdf.savefig(os.path.join(null_sequence_dir,seg_name+'_taste_epoch_spearman_cdfs.png'))
    f_dist_cdf.savefig(os.path.join(null_sequence_dir,seg_name+'_taste_epoch_spearman_cdfs.svg'))
    plt.close(f_dist_cdf)
    
    #Taste-v-taste 
    taste_pairs = list(itertools.combinations(np.arange(num_tastes),2))
    taste_pair_names = []
    for tp_i, tp in enumerate(taste_pairs):
        taste_pair_names.append(dig_in_names[tp[0]] + ' v. ' + dig_in_names[tp[1]])
    f_taste_cdf, ax_taste_cdf = plt.subplots(ncols=max_num_cp, figsize=(8,5))
    for cp_i in range(max_num_cp):
        ks_datasets = []
        for t_i in range(num_tastes):
            taste_data = np.abs(spearmans_dict[t_i][cp_i]['true'].flatten())
            ks_datasets.append(taste_data[~np.isnan(taste_data)])
            ax_taste_cdf[cp_i].hist(taste_data, bins = 50, density=True, cumulative=True,
                                       histtype='step',label=dig_in_names[t_i])
        if cp_i == 0:
            ax_taste_cdf[cp_i].legend(loc='upper left')
        ax_taste_cdf[cp_i].set_title('Epoch ' + str(cp_i))
        #Significance tests
        ks_sig = np.zeros(len(taste_pairs))
        ks_dir = np.zeros(len(taste_pairs))
        for p_i, p_vals in enumerate(taste_pairs):
            ks_res = stats.ks_2samp(ks_datasets[p_vals[0]], ks_datasets[p_vals[1]], \
                           alternative='two-sided')
            if ks_res.pvalue <= 0.05:
                ks_sig[p_i] = 1
                ks_dir[p_i] = ks_res.statistic_sign
        sig_text = 'Significant Pairs:'
        if np.sum(ks_sig) > 0:
            sig_inds = np.where(ks_sig > 0)[0]
            for sig_i in sig_inds:
                if ks_dir[sig_i] == 1:
                    sig_text += '\n' + taste_pair_names[sig_i] + ' < '
                if ks_dir[sig_i] == -1:
                    sig_text += '\n' + taste_pair_names[sig_i] + ' > '
        ax_taste_cdf[cp_i].text(0,0.2,sig_text)
    f_taste_cdf.suptitle(seg_name)
    plt.tight_layout()
    f_taste_cdf.savefig(os.path.join(null_sequence_dir,seg_name+'_taste_compare_true_spearman_cdfs.png'))
    f_taste_cdf.savefig(os.path.join(null_sequence_dir,seg_name+'_taste_compare_true_spearman_cdfs.svg'))
    plt.close(f_taste_cdf)
    
    #Epoch-v-epoch
    epoch_pairs = list(itertools.combinations(np.arange(max_num_cp),2))
    epoch_pair_names = []
    for ep_i, ep in enumerate(epoch_pairs):
        epoch_pair_names.append('Epoch ' + str(ep[0]) + ' v. ' + 'Epoch ' + str(ep[1]))
    f_epoch_cdf, ax_epoch_cdf = plt.subplots(ncols=max_num_cp, figsize=(8,5))
    for t_i in range(num_tastes):
        ks_datasets = []
        for cp_i in range(max_num_cp):
            epoch_data = np.abs(spearmans_dict[t_i][cp_i]['true'].flatten())
            ks_datasets.append(epoch_data[~np.isnan(epoch_data)])
            ax_epoch_cdf[t_i].hist(epoch_data, bins = 50, density=True, cumulative=True,
                                       histtype='step',label='Epoch ' + str(cp_i))
        if t_i == 0:
            ax_epoch_cdf[t_i].legend(loc='upper left')
        ax_epoch_cdf[t_i].set_title(dig_in_names[t_i])
        #Significance tests
        ks_sig = np.zeros(len(epoch_pairs))
        ks_dir = np.zeros(len(epoch_pairs))
        for p_i, p_vals in enumerate(epoch_pairs):
            ks_res = stats.ks_2samp(ks_datasets[p_vals[0]], ks_datasets[p_vals[1]], \
                           alternative='two-sided')
            if ks_res.pvalue <= 0.05:
                ks_sig[p_i] = 1
                ks_dir[p_i] = ks_res.statistic_sign
        sig_text = 'Significant Pairs:'
        if np.sum(ks_sig) > 0:
            sig_inds = np.where(ks_sig > 0)[0]
            for sig_i in sig_inds:
                if ks_dir[sig_i] == 1:
                    sig_text += '\n' + epoch_pair_names[sig_i] + ' < '
                if ks_dir[sig_i] == -1:
                    sig_text += '\n' + epoch_pair_names[sig_i] + ' > '
        ax_epoch_cdf[t_i].text(0,0.2,sig_text)
    f_epoch_cdf.suptitle(seg_name)
    plt.tight_layout()
    f_epoch_cdf.savefig(os.path.join(null_sequence_dir,seg_name+'_epoch_compare_true_spearman_cdfs.png'))
    f_epoch_cdf.savefig(os.path.join(null_sequence_dir,seg_name+'_epoch_compare_true_spearman_cdfs.svg'))
    plt.close(f_epoch_cdf)
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:57:12 2024

@author: hannahgermaine

File dedicated to functions related to decoding of tastes where neurons are
treated as dependent
"""

import tqdm
import os
import itertools
import time
import numpy as np
#file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
# os.chdir(file_path)
import matplotlib.pyplot as plt
from matplotlib import colormaps
from multiprocess import Pool
from sklearn.mixture import GaussianMixture as gmm
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import functions.decode_parallel as dp
from sklearn import svm
from random import sample
from scipy.stats import pearsonr

def taste_fr_dist(num_neur, tastant_spike_times, cp_raster_inds, fr_bins,
                  start_dig_in_times, pre_taste_dt, post_taste_dt, trial_start_frac=0):
    """Calculate the multidimensional distributions of firing rates maintaining
    dependencies between neurons"""

    num_tastes = len(tastant_spike_times)
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    max_num_deliv = 0  # Find the maximum number of deliveries across tastants
    for t_i in range(num_tastes):
        num_deliv = len(tastant_spike_times[t_i])
        if num_deliv > max_num_deliv:
            max_num_deliv = num_deliv
    del t_i, num_deliv

    # If trial_start_frac > 0 use only trials after that threshold
    trial_start_ind = np.floor(max_num_deliv*trial_start_frac).astype('int')
    new_max_num_deliv = (max_num_deliv - trial_start_ind).astype('int')

    deliv_taste_index = []
    taste_num_deliv = np.zeros(num_tastes)
    for t_i in range(num_tastes):
        num_deliv = len(tastant_spike_times[t_i][trial_start_ind:])
        deliv_taste_index.extend(list((t_i*np.ones(num_deliv)).astype('int')))
        taste_num_deliv[t_i] = num_deliv
    del t_i, num_deliv

    # Set up storage dictionary of results
    tastant_fr_dist = dict()  # Population firing rate distributions by epoch
    for t_i in range(num_tastes):
        tastant_fr_dist[t_i] = dict()
        for d_i in range(new_max_num_deliv):
            tastant_fr_dist[t_i][d_i-trial_start_ind] = dict()
            for cp_i in range(num_cp):
                tastant_fr_dist[t_i][d_i-trial_start_ind][cp_i] = dict()

    max_hz = 0
    for t_i in range(num_tastes):
        num_deliv = int(taste_num_deliv[t_i])
        taste_cp = cp_raster_inds[t_i]
        for d_i in range(num_deliv):  # index for that taste
            if d_i >= trial_start_ind:
                # grab spiking information
                # length num_neur list of lists
                raster_times = tastant_spike_times[t_i][d_i]
                start_taste_i = start_dig_in_times[t_i][d_i]
                deliv_cp = taste_cp[d_i, :] - pre_taste_dt
                # Calculate binned firing rate vectors
                for cp_i in range(num_cp):
                    # population changepoints
                    start_epoch = int(deliv_cp[cp_i])
                    end_epoch = int(deliv_cp[cp_i+1])
                    sdi = start_taste_i + start_epoch
                    epoch_len = end_epoch - start_epoch
                    if epoch_len > 0:
                        td_i_bin = np.zeros((num_neur, epoch_len+1))
                        for n_i in range(num_neur):
                            n_i_spike_times = np.array(
                                raster_times[n_i] - sdi).astype('int')
                            keep_spike_times = n_i_spike_times[np.where(
                                (0 <= n_i_spike_times)*(epoch_len >= n_i_spike_times))[0]]
                            td_i_bin[n_i, keep_spike_times] = 1
                        # Calculate the firing rate vectors for these bins
                        all_tb_fr = []
                        for fr_bin_size in fr_bins:
                            # Convert to  milliseconds
                            fr_half_bin = np.ceil(
                                fr_bin_size*500).astype('int')
                            quart_bin = np.ceil(fr_half_bin/2).astype('int')
                            fr_bin_dt = np.ceil(fr_half_bin*2).astype('int')
                            new_time_bins = np.arange(
                                fr_half_bin, epoch_len-fr_half_bin, quart_bin)
                            if len(new_time_bins) < 1:
                                tb_fr = list(np.sum(td_i_bin,1)/(np.shape(td_i_bin)[1]/1000))
                                all_tb_fr.extend([tb_fr])
                            else:
                                # Calculate the firing rate vectors for these bins
                                tb_fr = np.zeros((num_neur, len(new_time_bins)))
                                for tb_i, tb in enumerate(new_time_bins):
                                    tb_fr[:, tb_i] = np.sum(
                                        td_i_bin[:, tb-fr_half_bin:tb+fr_half_bin], 1)/(fr_bin_dt/1000)
                                all_tb_fr.extend(list(tb_fr.T))
                        all_tb_fr = np.array(all_tb_fr).T
                        # Store the firing rate vectors
                        tastant_fr_dist[t_i][d_i -
                                             trial_start_ind][cp_i] = all_tb_fr
                        # Store maximum firing rate
                        if np.max(all_tb_fr) > max_hz:
                            max_hz = np.max(all_tb_fr)

    return tastant_fr_dist, taste_num_deliv, max_hz


def taste_fr_dist_zscore(num_neur, tastant_spike_times, segment_spike_times,
                         segment_names, segment_times, cp_raster_inds, fr_bins,
                         start_dig_in_times, pre_taste_dt, post_taste_dt, bin_dt, trial_start_frac=0):
    """This function calculates spike count distributions for each neuron for
    each taste delivery for each epoch"""

    num_tastes = len(tastant_spike_times)
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    half_bin = np.floor(bin_dt/2).astype('int')
    max_num_deliv = 0  # Find the maximum number of deliveries across tastants
    for t_i in range(num_tastes):
        num_deliv = len(tastant_spike_times[t_i])
        if num_deliv > max_num_deliv:
            max_num_deliv = num_deliv
    del t_i, num_deliv

    # If trial_start_frac > 0 use only trials after that threshold
    trial_start_ind = np.floor(max_num_deliv*trial_start_frac).astype('int')
    new_max_num_deliv = (max_num_deliv - trial_start_ind).astype('int')

    deliv_taste_index = []
    taste_num_deliv = np.zeros(num_tastes)
    for t_i in range(num_tastes):
        num_deliv = len(tastant_spike_times[t_i][trial_start_ind:])
        deliv_taste_index.extend(list((t_i*np.ones(num_deliv)).astype('int')))
        taste_num_deliv[t_i] = num_deliv
    del t_i, num_deliv

    s_i_taste = np.nan*np.ones(1)
    for s_i in range(len(segment_names)):
        if segment_names[s_i].lower() == 'taste':
            s_i_taste[0] = s_i

    if not np.isnan(s_i_taste[0]):
        s_i = int(s_i_taste[0])
        seg_start = int(segment_times[s_i])
        seg_end = int(segment_times[s_i+1])
        seg_len = seg_end - seg_start
        time_bin_starts = np.arange(
            seg_start+half_bin, seg_end-half_bin, bin_dt)
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = (np.array(
                segment_spike_times_s_i[n_i]) - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        tb_fr = np.zeros((num_neur, len(time_bin_starts)))
        for tb_i, tb in enumerate(time_bin_starts):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_bin:tb+half_bin-seg_start], 1)/(2*half_bin*(1/1000))
        mean_fr = np.mean(tb_fr, 1)
        std_fr = np.std(tb_fr, 1)
    else:
        mean_fr = np.zeros(num_neur)
        std_fr = np.zeros(num_neur)

    # Determine the spike fr distributions for each neuron for each taste
    #print("\tPulling spike fr distributions by taste by neuron")
    tastant_fr_dist = dict()  # Population firing rate distributions by epoch
    for t_i in range(num_tastes):
        tastant_fr_dist[t_i] = dict()
        for d_i in range(new_max_num_deliv):
            if d_i >= trial_start_ind:
                tastant_fr_dist[t_i][d_i-trial_start_ind] = dict()
                for cp_i in range(num_cp):
                    tastant_fr_dist[t_i][d_i-trial_start_ind][cp_i] = dict()
    # ____
    max_hz = 0
    min_hz = 0
    for t_i in range(num_tastes):
        num_deliv = (taste_num_deliv[t_i]).astype('int')
        taste_cp = cp_raster_inds[t_i]
        for d_i in range(num_deliv):  # index for that taste
            if d_i >= trial_start_ind:
                raster_times = tastant_spike_times[t_i][d_i]
                start_taste_i = start_dig_in_times[t_i][d_i]
                deliv_cp = taste_cp[d_i, :] - pre_taste_dt
                # Bin the average firing rates following taste delivery start
                times_post_taste = [(np.array(raster_times[n_i])[np.where((raster_times[n_i] >= start_taste_i)*(
                    raster_times[n_i] < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int') for n_i in range(num_neur)]
                bin_post_taste = np.zeros((num_neur, post_taste_dt))
                for n_i in range(num_neur):
                    bin_post_taste[n_i, times_post_taste[n_i]] += 1
                for cp_i in range(num_cp):
                    # population changepoints
                    start_epoch = int(deliv_cp[cp_i])
                    end_epoch = int(deliv_cp[cp_i+1])
                    sdi = start_taste_i + start_epoch
                    epoch_len = end_epoch - start_epoch
                    if epoch_len > 0:
                        td_i_bin = np.zeros((num_neur, epoch_len+1))
                        for n_i in range(num_neur):
                            n_i_spike_times = np.array(
                                raster_times[n_i] - sdi).astype('int')
                            keep_spike_times = n_i_spike_times[np.where(
                                (0 <= n_i_spike_times)*(epoch_len >= n_i_spike_times))[0]]
                            td_i_bin[n_i, keep_spike_times] = 1
                        all_tb_fr = []
                        for fr_bin_size in fr_bins:
                            # Convert to  milliseconds
                            fr_half_bin = np.ceil(
                                fr_bin_size*500).astype('int')
                            quart_bin = np.ceil(fr_half_bin/2).astype('int')
                            fr_bin_dt = np.ceil(fr_half_bin*2).astype('int')
                            new_time_bins = np.arange(
                                fr_half_bin, epoch_len-fr_half_bin, quart_bin)
                            # Calculate the firing rate vectors for these bins
                            if len(new_time_bins) < 1:
                                tb_fr = list(np.sum(td_i_bin,1)/(np.shape(td_i_bin)[1]/1000))
                                all_tb_fr.extend([tb_fr])
                            else:
                                tb_fr = np.zeros((num_neur, len(new_time_bins)))
                                for tb_i, tb in enumerate(new_time_bins):
                                    tb_fr[:, tb_i] = np.sum(
                                        td_i_bin[:, tb-fr_half_bin:tb+fr_half_bin], 1)/(fr_bin_dt/1000)
                                all_tb_fr.extend(list(tb_fr.T))
                        all_tb_fr = np.array(all_tb_fr).T
                        # Z-score firing rates
                        bst_hz_z = (
                            np.array(all_tb_fr) - np.expand_dims(mean_fr, 1))/np.expand_dims(std_fr, 1)
                        # num_neurxall_n
                        tastant_fr_dist[t_i][d_i -
                                             trial_start_ind][cp_i] = np.array(bst_hz_z).T
                        if np.max(bst_hz_z) > max_hz:
                            max_hz = np.max(bst_hz_z)
                        if np.min(bst_hz_z) < min_hz:
                            min_hz = np.min(bst_hz_z)

    return tastant_fr_dist, taste_num_deliv, max_hz, min_hz

def decode_deviations(tastant_fr_dist, segment_spike_times, dig_in_names, 
                  segment_times, segment_names, start_dig_in_times, taste_num_deliv,
                  segment_dev_times, segment_dev_fr_vecs, bin_dt, 
                  save_dir, z_score = False, epochs_to_analyze=[], segments_to_analyze=[]):
    """Decode taste from epoch-specific firing rates"""
    print('\t\tRunning Is-Taste-Which-Taste GMM Decoder')
    
    decode_save_dir = os.path.join(save_dir,'Is_Taste_Which_Taste')
    if not os.path.isdir(decode_save_dir):
        os.mkdir(decode_save_dir)
    
    # Variables
    num_tastes = len(start_dig_in_times)
    num_neur = len(segment_spike_times[0])
    num_cp = len(tastant_fr_dist[0][0])
    num_segments = len(segment_spike_times)
    #p_taste = taste_num_deliv/np.sum(taste_num_deliv)  # P(taste)
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, num_cp))
    cmap = colormaps['gist_rainbow']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    cmap = colormaps['seismic']
    is_taste_colors = cmap(np.linspace(0, 1, 2))
    dev_buffer = 50
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
        
    seg_names = list(np.array(segment_names)[segments_to_analyze])
    epoch_names = ['Epoch ' + str(e_i) for e_i in range(num_cp)]
       
    #Create null dataset from shuffled rest spikes
    shuffled_fr_vecs, segment_spike_times_bin, \
        seg_means, seg_stds = create_null_decode_dataset(segments_to_analyze, \
                                    segment_times, segment_spike_times, \
                                    num_neur, bin_dt, z_score)
    
    #Train decoder
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
            if z_score == True:
                neur_max = np.expand_dims(np.max(np.abs(np.array(train_taste_data)),0),1)
                none_data.extend(list((neur_max*np.random.randn(num_neur,100)).T)) #Fully randomized data
                none_data.extend(list(((neur_max/10)*np.random.randn(num_neur,100)).T)) #Low frequency randomized data
            else:
                neur_max = np.expand_dims(np.max(np.array(train_taste_data),0),1)
                none_data.extend(list((neur_max*np.random.rand(num_neur,100)).T)) #Fully randomized data
                none_data.extend(list(((neur_max/10)*np.random.rand(num_neur,100)).T)) #Low frequency randomized data
            none_data.extend(shuffled_fr_vecs)
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
    
    #Store segment by segment fraction of decodes
    seg_is_taste_frac = np.nan*np.ones((len(segments_to_analyze),2))
    seg_which_taste_frac = np.nan*np.ones((len(segments_to_analyze),2))
    seg_which_epoch_frac = np.nan*np.ones((len(segments_to_analyze),num_cp))
    
    # Segment-by-segment use deviation rasters and times to zoom in and test
    #	epoch-specific decoding of tastes. Add decoding of 50 ms on either
    #	side of the deviation event as well for context decoding.
    for seg_ind, s_i in enumerate(segments_to_analyze):
        # Create plot save dir
        seg_decode_save_dir = os.path.join(decode_save_dir,
            'segment_' + str(s_i) + '/')
        if not os.path.isdir(seg_decode_save_dir):
            os.mkdir(seg_decode_save_dir)
        
        #Import existing data
        try:
            decode_is_taste_prob_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_deviations_is_taste.npy'))
            pre_dev_decode_is_taste_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_pre_deviations_is_taste.npy'))
            post_dev_decode_is_taste_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_post_deviations_is_taste.npy'))
            decode_which_taste_prob_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_deviations_which_taste.npy'))
            pre_dev_decode_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_pre_deviations_which_taste.npy'))
            post_dev_decode_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_post_deviations_which_taste.npy')) 
            decode_epoch_prob_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_deviations_which_epoch.npy'))
                
            print('\t\t\t\tSegment ' + str(s_i) + ' Previously Decoded')
        except:
            print('\t\t\t\tDecoding Segment ' + str(s_i) + ' Deviations')
            tic = time.time()
            
            seg_dev_fr_mat = np.array(segment_dev_fr_vecs[seg_ind]).T
            seg_dev_times = segment_dev_times[seg_ind]
            _, num_dev = np.shape(seg_dev_times)
            segment_spike_times_s_i_bin = segment_spike_times_bin[seg_ind]
            if z_score == True:
                mean_fr = seg_means[seg_ind]
                std_fr = seg_stds[seg_ind]
            
            decode_is_taste_prob_array = np.nan*np.ones((num_dev,2)) #deviation x is taste
            decode_which_taste_prob_array = np.nan*np.ones((num_dev,num_tastes-1)) #deviation x which taste
            decode_epoch_prob_array = np.nan*np.ones((num_dev,num_cp)) #deviation x epoch
            
            #Run through each deviation event to decode 
            
            #Pull pre-dev bin frs
            pre_dev_fr_mat = []
            for dev_i in range(num_dev):
                pre_dev_end = seg_dev_times[0,dev_i]
                pre_dev_start = np.max(pre_dev_end-dev_buffer,0)
                pre_dev_len = pre_dev_end - pre_dev_start
                pre_dev_bin = segment_spike_times_s_i_bin[:,pre_dev_start:pre_dev_end]
                pre_dev_fr = np.sum(pre_dev_bin,1)/(pre_dev_len/1000)
                if z_score == True:
                    pre_dev_fr = (pre_dev_fr - mean_fr)/std_fr
                pre_dev_fr_mat.append(list(pre_dev_fr))
            pre_dev_fr_mat = np.array(pre_dev_fr_mat)
            
            #Pull post-dev bin frs
            post_dev_fr_mat = []
            for dev_i in range(num_dev):
                post_dev_start = seg_dev_times[1,dev_i]
                post_dev_end = np.min([pre_dev_end+dev_buffer,seg_len])
                post_dev_len = post_dev_end - post_dev_start
                post_dev_bin = segment_spike_times_s_i_bin[:,post_dev_start:post_dev_end]
                post_dev_fr = np.sum(post_dev_bin,1)/(post_dev_len/1000)
                if z_score == True:
                    post_dev_fr = (post_dev_fr - mean_fr)/std_fr
                post_dev_fr_mat.append(list(post_dev_fr))
            post_dev_fr_mat = np.array(post_dev_fr_mat)
            
            #Converting to list for parallel processing
            if need_pca == 1:    
                dev_fr_pca = pca_reduce_taste.transform(seg_dev_fr_mat.T)
                list_dev_fr = list(dev_fr_pca)
                pre_dev_fr_pca = pca_reduce_taste.transform(pre_dev_fr_mat)
                list_pre_dev_fr = list(pre_dev_fr_pca)
                post_dev_fr_pca = pca_reduce_taste.transform(post_dev_fr_mat)
                list_post_dev_fr = list(post_dev_fr_pca)
            else:
                list_dev_fr = list(seg_dev_fr_mat.T)
                list_pre_dev_fr = list(pre_dev_fr_mat)
                list_post_dev_fr = list(post_dev_fr_mat)
            del seg_dev_fr_mat
            
            # Pass inputs to parallel computation on probabilities
            tic = time.time()
            #Deviation Bins
            inputs = zip(list_dev_fr, itertools.repeat(len(none_v_taste_gmm)),
                          itertools.repeat(none_v_taste_gmm), itertools.repeat(none_v_true_prob))
            pool = Pool(4)
            dev_decode_is_taste_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            decode_is_taste_prob_array = np.squeeze(np.array(dev_decode_is_taste_prob)) #num_dev x 2
            dev_is_taste_argmax = np.squeeze(np.argmax(decode_is_taste_prob_array,1)) #num_dev length indices
            dev_is_taste_inds = np.where(dev_is_taste_argmax == 0)[0]
            #Pre-Deviation Bins
            inputs = zip(list_pre_dev_fr, itertools.repeat(len(none_v_taste_gmm)),
                          itertools.repeat(none_v_taste_gmm), itertools.repeat(none_v_true_prob))
            pool = Pool(4)
            pre_dev_decode_is_taste_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            pre_dev_decode_is_taste_array = np.squeeze(np.array(pre_dev_decode_is_taste_prob)) #num_dev x 2
            #Post-Deviation Bins
            inputs = zip(list_post_dev_fr, itertools.repeat(len(none_v_taste_gmm)),
                          itertools.repeat(none_v_taste_gmm), itertools.repeat(none_v_true_prob))
            pool = Pool(4)
            post_dev_decode_is_taste_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            post_dev_decode_is_taste_array = np.squeeze(np.array(post_dev_decode_is_taste_prob)) #num_dev x 2
            
            # Save decoding probabilities
            np.save(os.path.join(decode_save_dir,'segment_' +
                    str(s_i) + '_deviations_is_taste.npy'), decode_is_taste_prob_array)
            np.save(os.path.join(decode_save_dir,'segment_' +
                    str(s_i) + '_pre_deviations_is_taste.npy'), pre_dev_decode_is_taste_array)
            np.save(os.path.join(decode_save_dir,'segment_' +
                    str(s_i) + '_post_deviations_is_taste.npy'), post_dev_decode_is_taste_array)
            
            #Now determine which taste
            if len(dev_is_taste_inds) > 0: #at least some devs decoded as fully taste
                
                #Deviation Bins
                inputs = zip(list_dev_fr, itertools.repeat(len(just_taste_gmm)),
                              itertools.repeat(just_taste_gmm), itertools.repeat(by_taste_true_prob))
                pool = Pool(4)
                dev_decode_prob = pool.map(
                    dp.segment_taste_decode_dependent_parallelized, inputs)
                pool.close()
                dev_decode_array = np.squeeze(np.array(dev_decode_prob)) #num_devx2
                decode_which_taste_prob_array[dev_is_taste_inds,:] = dev_decode_array[dev_is_taste_inds,:]
                dev_which_taste_argmax_array = np.argmax(dev_decode_array,1)
                
                #Pre-Deviation Bins
                inputs = zip(list_pre_dev_fr, itertools.repeat(len(just_taste_gmm)),
                              itertools.repeat(just_taste_gmm), itertools.repeat(by_taste_true_prob))
                pool = Pool(4)
                pre_dev_decode_prob = pool.map(
                    dp.segment_taste_decode_dependent_parallelized, inputs)
                pool.close()
                pre_dev_decode_array = np.squeeze(np.array(pre_dev_decode_prob)) #num_devx2
                #Post-Deviation Bins
                inputs = zip(list_post_dev_fr, itertools.repeat(len(just_taste_gmm)),
                              itertools.repeat(just_taste_gmm), itertools.repeat(by_taste_true_prob))
                pool = Pool(4)
                post_dev_decode_prob = pool.map(
                    dp.segment_taste_decode_dependent_parallelized, inputs)
                pool.close()
                post_dev_decode_array = np.squeeze(np.array(post_dev_decode_prob)) #num_devx2
                
                # Save decoding probabilities
                np.save(os.path.join(decode_save_dir,'segment_' +
                        str(s_i) + '_deviations_which_taste.npy'), dev_decode_array)
                np.save(os.path.join(decode_save_dir,'segment_' +
                        str(s_i) + '_pre_deviations_which_taste.npy'), pre_dev_decode_array)
                np.save(os.path.join(decode_save_dir,'segment_' +
                        str(s_i) + '_post_deviations_which_taste.npy'), post_dev_decode_array)
                
                #Now determine which epoch for those that are decoded as taste
                dev_taste_list = []
                num_gmm = []
                same_taste_epoch_gmm = []
                prob_list = []
                for dev_ind, dev_i in enumerate(dev_is_taste_inds):
                    dev_taste_list.append(list_dev_fr[dev_i])
                    num_gmm.extend([len(taste_epoch_gmm[dev_which_taste_argmax_array[dev_ind]])])
                    same_taste_epoch_gmm.append(taste_epoch_gmm[dev_which_taste_argmax_array[dev_ind]])
                    prob_list.append(by_taste_epoch_prob[dev_which_taste_argmax_array[dev_ind],:])
                    
                #Now determine which epoch of that taste
                inputs = zip(dev_taste_list, num_gmm, same_taste_epoch_gmm, prob_list)
                pool = Pool(4)
                dev_decode_epoch_prob = pool.map(
                    dp.segment_taste_decode_dependent_parallelized, inputs)
                pool.close()
                decode_epoch_prob_array[dev_is_taste_inds,:] = np.squeeze(np.array(dev_decode_epoch_prob)) #num dev x 3
                np.save(os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                                 '_deviations_which_epoch.npy'),decode_epoch_prob_array)
                
            toc = time.time()
            print('\t\t\t\t\tTime to decode = ' +
                  str(np.round((toc-tic)/60, 2)) + ' (min)')
        
        #Create plots
        num_dev,_ = np.shape(decode_is_taste_prob_array)
        decode_is_taste_argmax = np.squeeze(np.argmax(decode_is_taste_prob_array,1))
        is_taste_inds = np.where(decode_is_taste_argmax == 0)[0]
        seg_is_taste_frac[seg_ind,:] = [len(is_taste_inds)/num_dev,1-len(is_taste_inds)/num_dev]
        if len(is_taste_inds) > 0:
            decode_which_taste_argmax = np.squeeze(np.argmax(decode_which_taste_prob_array[is_taste_inds,:],1))
            decode_which_epoch_argmax = np.squeeze(np.argmax(decode_epoch_prob_array[is_taste_inds,:],1))
            #Calculate fraction of taste decodes for this segment
            decode_which_taste_frac = []
            for t_i in range(num_tastes-1):
                taste_dev_vec = np.zeros(num_dev)
                taste_inds = is_taste_inds[np.where(decode_which_taste_argmax == t_i)[0]]
                decode_which_taste_frac.append(len(taste_inds)/len(is_taste_inds))
            seg_which_taste_frac[seg_ind,:] = np.array(decode_which_taste_frac)
            #Calculate fraction of epoch decodes for this segment
            decode_which_epoch_frac = []
            for e_i in range(num_cp):
                epoch_bin_vec = np.zeros(num_dev)
                epoch_inds = is_taste_inds[np.where(decode_which_epoch_argmax == e_i)[0]]
                epoch_bin_vec[epoch_inds] = 1
                decode_which_epoch_frac.append(len(epoch_inds)/len(is_taste_inds))
            seg_which_epoch_frac[seg_ind,:] = np.array(decode_which_epoch_frac)
            
    #Plot fractions of decodes by segment in line graphs
    f_frac, ax_frac = plt.subplots(nrows = 3, sharey = True, figsize = (4,8))
    #Frac is-taste
    ax_frac[0].set_ylim([-0.1,1.1])
    ax_frac[0].axhline(0.5,linestyle='dashed',c='k',alpha=0.2,label='_')
    for s_ind, s_i in enumerate(segments_to_analyze):
        ax_frac[0].plot(np.arange(2),seg_is_taste_frac[s_ind,:],
                        label=seg_names[s_ind])
    ax_frac[0].set_xticks(np.arange(2),['Taste','No Taste'])
    ax_frac[0].legend(loc='upper left')
    ax_frac[0].set_title('Is Taste Decode Fractions')
    #Frac which-taste
    ax_frac[1].axhline(1/(num_tastes-1),linestyle='dashed',c='k',alpha=0.2,label='_')
    for s_ind, s_i in enumerate(segments_to_analyze):
        ax_frac[1].plot(np.arange(num_tastes-1),seg_which_taste_frac[s_ind,:],
                        label=seg_names[s_ind])
    ax_frac[1].set_xticks(np.arange(num_tastes-1),dig_in_names[:-1])
    ax_frac[1].legend(loc='upper left')
    ax_frac[1].set_title('Which Taste Decode Fractions')
    #Frac which-epoch
    ax_frac[2].axhline(1/num_cp,linestyle='dashed',c='k',alpha=0.2,label='_')
    for s_ind, s_i in enumerate(segments_to_analyze):
        ax_frac[2].plot(np.arange(num_cp),seg_which_epoch_frac[s_ind,:],
                        label=seg_names[s_ind])
    ax_frac[2].set_xticks(np.arange(num_cp),epoch_names)
    ax_frac[2].legend(loc='upper left')
    ax_frac[2].set_title('Which Epoch Decode Fractions')
    plt.tight_layout()
    f_frac.savefig(os.path.join(decode_save_dir,'frac_decode_plot.png'))
    f_frac.savefig(os.path.join(decode_save_dir,'frac_decode_plot.svg'))
    plt.close(f_frac)
        
def decode_sliding_bins(tastant_fr_dist, segment_spike_times, dig_in_names, 
                  palatable_dig_inds, segment_times, segment_names, 
                  start_dig_in_times, taste_num_deliv,
                  segment_dev_times, segment_dev_fr_vecs, bin_dt, 
                  group_list, group_names, non_none_tastes, decode_dir, 
                  z_score = False, epochs_to_analyze=[], segments_to_analyze=[]):
    """Decode taste in sliding bins of rest intervals"""
    
    print('\t\tRunning Sliding Bin Is-Taste-Which-Taste GMM Decoder')
    
    # Variables
    num_tastes = len(start_dig_in_times)
    num_neur = len(segment_spike_times[0])
    # num_cp = len(tastant_fr_dist[0][0])
    num_segments = len(segment_spike_times)
    
    #Bin size for sliding bin decoding
    half_bin = 25
    bin_size = half_bin*2
    
    # if len(epochs_to_analyze) == 0:
    #     epochs_to_analyze = np.arange(num_cp)
    epochs_to_analyze = np.array([0,1,2])
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
        
    seg_names = list(np.array(segment_names)[segments_to_analyze])
        
    #Create null dataset from shuffled rest spikes
    shuffled_fr_vecs, segment_spike_times_bin, \
        seg_means, seg_stds = create_null_decode_dataset(segments_to_analyze, \
                                    segment_times, segment_spike_times, \
                                    num_neur, bin_dt, z_score)
        
    #Create training groups of firing rate vectors
    grouped_train_data = [] #Using group_list above, create training groups
    grouped_train_counts = [] #Number of values in the group
    grouped_train_names = []
    for g_i, g_list in enumerate(group_list):
        group_data_collection = []
        for (e_i,t_i) in g_list:
            for d_i in range(int(taste_num_deliv[t_i])):
                try:
                    if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                        group_data_collection.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i].T))
                    else:
                        group_data_collection.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i]))
                except:
                    group_data_collection.extend([])
        if len(group_data_collection) > 0:
            grouped_train_data.append(group_data_collection)
            grouped_train_counts.append(len(group_data_collection))
            grouped_train_names.append(group_names[g_i])
    #Now add the generated null data
    avg_count = int(np.ceil(np.nanmean(grouped_train_counts)))
    null_inds_to_use = sample(list(np.arange(len(shuffled_fr_vecs))),avg_count)
    grouped_train_data.append(np.array(shuffled_fr_vecs)[null_inds_to_use,:])
    grouped_train_counts.append(avg_count)
    grouped_train_names.append('Null')
    group_prob = np.array(grouped_train_counts) / np.sum(np.array(grouped_train_counts)) 
    num_groups = len(grouped_train_names)
    
    #Run PCA transform only on non-z-scored data
    if z_score == True:
        pca_reduce_taste = train_taste_PCA(num_tastes,num_neur,epochs_to_analyze,\
                                           taste_num_deliv,tastant_fr_dist)
    
    #Run GMM fits to distributions of different groups
    group_gmms = dict()
    for g_i, g_data in enumerate(grouped_train_data):
        train_data = np.array(g_data)
        if z_score == False:
            transformed_data = pca_reduce_taste.transform(train_data)
        else:
            transformed_data = train_data
        #Fit GMM
        gm = gmm(n_components=1, n_init=10).fit(
            transformed_data)
        group_gmms[g_i] = gm
    
    #Store segment by segment correlation between pop rate and decode group
    seg_group_rate_corr = np.nan*np.ones((len(segments_to_analyze),num_groups))
    
    #Store segment by segment fraction of decodes
    seg_group_frac = np.nan*np.ones((len(segments_to_analyze),num_groups))
        
    #Run through each segment and decode bins of activity
    for seg_ind, s_i in enumerate(segments_to_analyze):
        # Create segment save dir
        seg_decode_save_dir = os.path.join(decode_dir,
            'segment_' + str(s_i) + '/')
        if not os.path.isdir(seg_decode_save_dir):
            os.mkdir(seg_decode_save_dir)
            
        # Get segment variables
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = seg_end - seg_start# in dt = ms
        segment_spike_times_s_i_bin = segment_spike_times_bin[seg_ind]
        if z_score == True:
            mean_fr = seg_means[seg_ind]
            std_fr = seg_stds[seg_ind]
        
        #Convert binary spikes to binned fr vecs
        segment_slide_bins = np.arange(half_bin,seg_len,half_bin)
        segment_binned_fr = np.zeros((num_neur,len(segment_slide_bins)))
        segment_binned_pop_fr = np.zeros(len(segment_slide_bins)) #For comparison with pop fr later
        for ssb_i, ssb in enumerate(segment_slide_bins):
            ssb_s = ssb_i - half_bin
            ssb_e = ssb_i + half_bin
            bin_fr = np.sum(segment_spike_times_s_i_bin[:,ssb_s:ssb_e],1)/(bin_size/1000) #Hz converted
            if z_score == True:
                segment_binned_fr[:,ssb_i] = (bin_fr - mean_fr)/std_fr
            else:
                segment_binned_fr[:,ssb_i] = bin_fr
            segment_binned_pop_fr[ssb_i] = (np.sum(segment_spike_times_s_i_bin[:,ssb_s:ssb_e])/num_neur)/(bin_size/1000) #Hz converted
        _, num_bin = np.shape(segment_binned_fr)
        
        # Grab neuron firing rates in sliding bins
        try:
            decode_group_prob_array = np.load(
                os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                             '_sliding_group.npy'))
        except:
            print('\t\t\t\tDecoding Segment ' + str(s_i) + ' Bins')
            tic = time.time()
            
            #Run through each bin to decode 
            #Converting to list for parallel processing
            if z_score == False:    
                seg_fr_pca = pca_reduce_taste.transform(segment_binned_fr.T)
                seg_fr_list = list(seg_fr_pca)
            else:
                seg_fr_list = list(segment_binned_fr.T)
            
            # Pass inputs to parallel computation on probabilities
            inputs = zip(seg_fr_list, itertools.repeat(len(group_gmms)),
                          itertools.repeat(group_gmms), itertools.repeat(group_prob))
            pool = Pool(4)
            decode_group_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            decode_group_prob_array = np.squeeze(np.array(decode_group_prob)) #num_bin x num_groups
                
            np.save(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                                 '_sliding_group.npy'),decode_group_prob_array)
            
            toc = time.time()
            print('\t\t\t\t\tTime to decode = ' +
                  str(np.round((toc-tic)/60, 2)) + ' (min)')
        
        group_argmax = np.squeeze(np.argmax(decode_group_prob_array,1)) #num_dev length indices
        
        #Save decode fractions
        hist_vals = np.histogram(group_argmax,bins=num_groups)
        seg_group_frac[seg_ind,:] = hist_vals[0]/np.nansum(hist_vals[0])
        
        #Save correlation to pop rate
        for g_ind in range(num_groups):
            ga_i = np.where(group_argmax == g_ind)[0]
            bin_group = np.zeros(num_bin)
            bin_group[ga_i] = 1
            pcorr = pearsonr(bin_group,segment_binned_pop_fr)
            seg_group_rate_corr[seg_ind,g_ind] = pcorr[0]
            
        #Save a histogram of group decodes
        f = plt.figure(figsize=(5,5))
        plt.hist(group_argmax)
        plt.xticks(np.arange(num_groups),grouped_train_names,rotation=45)
        plt.title('Segment ' + str(s_i) + ' group decode counts')
        plt.tight_layout()
        f.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                             '_sliding_group_decode_hist.svg'))
        f.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                             '_sliding_group_decode_hist.png'))
        plt.close(f)
                    
    #Save cross-segment data
    np.save(os.path.join(decode_dir,'seg_group_rate_corr.npy'), seg_group_rate_corr, allow_pickle=True) 
    np.save(os.path.join(decode_dir,'seg_group_frac.npy'), seg_group_frac, allow_pickle=True) 
    
    #PLOTS
    #Plot decode trends
    plot_decode_trends(num_groups, grouped_train_names, segments_to_analyze,
                           seg_group_frac, seg_names, decode_dir)
    
    #Plot population rate correlation results
    f_box_corr = plt.figure(figsize=(5,5))
    plt.axhline(0,linestyle='dashed',alpha=0.2,color='k')
    max_y = 0
    min_y = 0
    for g_i, g_name in enumerate(grouped_train_names):
        plt.scatter(g_i*np.ones(len(segments_to_analyze)),seg_group_rate_corr[:,g_i],\
                    color='g',alpha=0.5)
        group_corr_mean = np.nanmean(seg_group_rate_corr[:,g_i])
        plt.scatter([g_i],group_corr_mean,
                    color='k',alpha=1,label='average corr')
        max_ind = np.argmax(np.abs(seg_group_rate_corr[:,g_i]))
        max_val = seg_group_rate_corr[max_ind,g_i]
        max_sign = np.sign(max_val)
        plt.text(g_i,max_val+max_sign*0.025,str(np.round(group_corr_mean,2)),\
                 ha='left',va='center')
        if max_val+max_sign*0.025 > max_y:
            max_y = max_val+max_sign*0.025
        if max_val+max_sign*0.025 < min_y:
            min_y = max_val+max_sign*0.025
    plt.xticks(np.arange(num_groups),grouped_train_names,rotation=45)
    plt.ylim([min_y+0.1*min_y,max_y+0.1*max_y])
    plt.title('Group Decoding x Population Rate Correlation')
    plt.tight_layout()
    f_box_corr.savefig(os.path.join(decode_dir,'seg_group_rate_corr.png'))
    f_box_corr.savefig(os.path.join(decode_dir,'seg_group_rate_corr.svg'))
    plt.close(f_box_corr)

def decoder_accuracy_tests(tastant_fr_dist, segment_spike_times, 
                dig_in_names, segment_times, segment_names, start_dig_in_times, 
                taste_num_deliv, bin_dt, group_list, group_names, non_none_tastes, 
                decode_dir, z_score = False, 
                epochs_to_analyze=[], segments_to_analyze=[]):
    """
    This function runs decoder accuracy tests via LOO train-test method.

    Parameters
    ----------
    tastant_fr_dist : dict
        Dictionary of firing rate vectors by taste.
    segment_spike_times : list
        List of list with times of individual neurons spiking during each segment.
    dig_in_names : list
        List of taste names.
    segment_times : numpy array
        Array of start and end times (ms) for each segment.
    segment_names : list
        List of names of each segment.
    start_dig_in_times : list
        List of lists with time of tastant delivery start.
    taste_num_deliv : numpy array
        Array of number of total deliveries of each tastant.
    bin_dt : TYPE
        DESCRIPTION.
    group_list : TYPE
        DESCRIPTION.
    group_names : TYPE
        DESCRIPTION.
    non_none_tastes : TYPE
        DESCRIPTION.
    decode_dir : TYPE
        DESCRIPTION.
    z_score : TYPE, optional
        DESCRIPTION. The default is False.
    epochs_to_analyze : TYPE, optional
        DESCRIPTION. The default is [].
    segments_to_analyze : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    None.

    """
    decode_accuracy_save_dir = os.path.join(decode_dir,'Decoder_Accuracy')
    if not os.path.isdir(decode_accuracy_save_dir):
        os.mkdir(decode_accuracy_save_dir)
        
    # Variables
    num_tastes = len(start_dig_in_times)
    num_neur = len(segment_spike_times[0])
    # num_cp = len(tastant_fr_dist[0][0])
    num_segments = len(segment_spike_times)
    # if len(epochs_to_analyze) == 0:
    #     epochs_to_analyze = np.arange(num_cp)
    epochs_to_analyze = np.array([0,1,2])
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
    num_cp = len(epochs_to_analyze)
        
    #Create null dataset from shuffled rest spikes
    shuffled_fr_vecs, segment_spike_times_bin, \
        seg_means, seg_stds = create_null_decode_dataset(segments_to_analyze, \
                                    segment_times, segment_spike_times, \
                                    num_neur, bin_dt, z_score)
    
    #Store taste x delivery x epoch decoding success rates
    decode_success_bool = dict()
    for t_i in range(num_tastes):
        decode_success_bool[t_i] = np.zeros((int(taste_num_deliv[t_i]),num_cp))
    
    #Create a LOO list
    total_num_deliv = int(np.sum(taste_num_deliv))
    loo_taste_index = []
    loo_deliv_index = []
    for t_i in range(num_tastes):  # Only perform on actual tastes
        num_deliv = int(taste_num_deliv[t_i])
        loo_taste_index.extend(list((t_i*np.ones(num_deliv)).astype('int')))
        loo_deliv_index.extend(list((np.arange(num_deliv)).astype('int')))
    del t_i, num_deliv
    
    tic = time.time()
    for loo_i in tqdm.tqdm(range(total_num_deliv)):
        loo_t_i = loo_taste_index[loo_i]
        loo_d_i = loo_deliv_index[loo_i]
        for loo_e_i, loo_e in enumerate(epochs_to_analyze):
            
            test_data = [np.squeeze(tastant_fr_dist[loo_t_i][loo_d_i][loo_e])]
            
            test_group_ind = -1
            for g_i, g_list in enumerate(group_list):
                for g_e, g_t in g_list:
                    if (g_e == loo_e) and (g_t == loo_t_i):
                        test_group_ind = g_i
            
            #Create training groups of firing rate vectors
            grouped_train_data = [] #Using group_list above, create training groups
            grouped_train_counts = [] #Number of values in the group
            grouped_train_names = []
            for g_i, g_list in enumerate(group_list):
                group_data_collection = []
                for (e_i,t_i) in g_list:
                    for d_i in range(int(taste_num_deliv[t_i])):
                        #Skip the loo value
                        if (e_i == loo_e) and (t_i == loo_t_i) and (d_i == loo_d_i):
                            group_data_collection.extend([])
                        else:
                            try:
                                if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                                    group_data_collection.extend(
                                        list(tastant_fr_dist[t_i][d_i][e_i].T))
                                else:
                                    group_data_collection.extend(
                                        list(tastant_fr_dist[t_i][d_i][e_i]))
                            except:
                                group_data_collection.extend([])
                if len(group_data_collection) > 0:
                    grouped_train_data.append(group_data_collection)
                    grouped_train_counts.append(len(group_data_collection))
                    grouped_train_names.append(group_names[g_i])
            #Now add the generated null data
            avg_count = int(np.ceil(np.nanmean(grouped_train_counts)))
            null_inds_to_use = sample(list(np.arange(len(shuffled_fr_vecs))),avg_count)
            grouped_train_data.append(np.array(shuffled_fr_vecs)[null_inds_to_use,:])
            grouped_train_counts.append(avg_count)
            grouped_train_names.append('Null')
            group_prob = np.array(grouped_train_counts) / np.sum(np.array(grouped_train_counts)) 
            num_groups = len(grouped_train_names)
            
            #Run PCA transform only on non-z-scored data
            if z_score == True:
                pca_reduce_taste = train_taste_PCA(num_tastes,num_neur,epochs_to_analyze,\
                                                   taste_num_deliv,tastant_fr_dist)
            
            #Run GMM fits to distributions of different groups
            group_gmms = dict()
            for g_i, g_data in enumerate(grouped_train_data):
                train_data = np.array(g_data)
                if z_score == False:
                    transformed_data = pca_reduce_taste.transform(train_data)
                else:
                    transformed_data = train_data
                #Fit GMM
                gm = gmm(n_components=1, n_init=10).fit(
                    transformed_data)
                group_gmms[g_i] = gm
                    
            #Run decoders on the taste response
            inputs = zip(test_data, itertools.repeat(len(group_gmms)),
                          itertools.repeat(group_gmms), itertools.repeat(group_prob))
            pool = Pool(4)
            dev_decode_is_taste_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            decode_argmax = np.argmax(np.array(dev_decode_is_taste_prob[0]))
            if decode_argmax == test_group_ind:
                decode_success_bool[loo_t_i][loo_d_i,loo_e_i] = 1
    toc = time.time()
    print('\tTime to run decode accuracy tests = ' +
                  str(np.round((toc-tic)/60, 2)) + ' (min)')
    np.save(os.path.join(decode_accuracy_save_dir,'decoder_accuracy_dict.npy'),\
            decode_success_bool,allow_pickle=True)
        
    #PLOTS
    chance_rate = 100/num_groups
    
    #Overall Taste Accuracy
    decode_success_rates_taste = []
    for t_i in range(num_tastes):
        decode_success_rates_taste.append(100*np.sum(decode_success_bool[t_i])/(taste_num_deliv[t_i]*num_cp))
    f_bar = plt.figure(figsize=(5,5))
    plt.axhline(chance_rate,linestyle='dashed',alpha=0.2,color='k')
    plt.bar(np.arange(num_tastes),decode_success_rates_taste)
    plt.xticks(np.arange(num_tastes),dig_in_names,rotation=45)
    plt.xlabel('Taste')
    plt.ylabel('Percent Accurately Decoded')
    plt.title('Overall Taste Accuracy')
    plt.tight_layout()
    f_bar.savefig(os.path.join(decode_accuracy_save_dir,'overall_taste_accuracy.png'))
    f_bar.savefig(os.path.join(decode_accuracy_save_dir,'overall_taste_accuracy.svg'))
    plt.close(f_bar)
    
    #Accuracy by Epoch
    decode_success_rates_epoch = []
    for e_ind, e_i in enumerate(epochs_to_analyze):
        taste_epoch_rates = []
        for t_i in range(num_tastes):
            taste_epoch_rates.append(100*np.sum(decode_success_bool[t_i][:,e_ind])/(taste_num_deliv[t_i]))
        decode_success_rates_epoch.append(taste_epoch_rates)
    decode_success_rates_epoch = np.array(decode_success_rates_epoch)
    f_epoch = plt.figure(figsize=(5,5))
    plt.axhline(chance_rate,linestyle='dashed',alpha=0.2,color='k')
    for t_i, t_name in enumerate(dig_in_names):
        plt.plot(decode_success_rates_epoch[:,t_i],label=t_name)
    plt.legend(loc='upper left')
    plt.xticks(np.arange(num_cp),epochs_to_analyze)
    plt.xlabel('Epoch')
    plt.ylabel('Percent Accurately Decoded')
    plt.title('By Epoch Taste Accuracy')
    plt.tight_layout()
    f_epoch.savefig(os.path.join(decode_accuracy_save_dir,'by_epoch_taste_accuracy.png'))
    f_epoch.savefig(os.path.join(decode_accuracy_save_dir,'by_epoch_taste_accuracy.svg'))
    plt.close(f_epoch)
    
def create_null_decode_dataset(segments_to_analyze, segment_times, segment_spike_times,
                               num_neur, bin_dt, z_score = False):
    """
    This function creates a null dataset of firing rate vectors from shuffled 
    segment spike times to use in training a decoder.

    Parameters
    ----------
    segments_to_analyze : list
        list of which segment indices to use in analysis.
    segment_times : list
        list of start and end times of segments.
    segment_spike_times : list
        list of lists containing neuron spike times by segment.
    num_neur : int
        number of neurons in dataset.
    bin_dt : bool
        bin size in timesteps to use for z-scoring data.
    z_score : bool, optional
        boolean indicating whether the data should be z-scored. The default is False.

    Returns
    -------
    shuffled_fr_vecs : list
        firing rate vectors of null dataset by segment.
    segment_spike_times_bin : list
        list of boolean arrays containing spike times marked by 1s.
    seg_means : list
        list of mean neuron firing rate vectors by segment.
    seg_stds : list
        list of std neuron firing rate vectors by segment.
    """
    shuffled_fr_vecs = []
    segment_spike_times_bin = []
    seg_means = []
    seg_stds = []
    for seg_i, s_i in enumerate(segments_to_analyze):
        # Get segment variables
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = segment_times[s_i+1] - segment_times[s_i]  # in dt = ms
        # Binerize Segment Spike Times
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = np.array(
                segment_spike_times_s_i[n_i] - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        segment_spike_times_bin.append(segment_spike_times_s_i_bin)
        if z_score == True:
            # Calculate mean and std of binned segment spikes for z-scoring
            z_time_bins = np.arange(0,seg_len-bin_dt,bin_dt)
            seg_fr = np.zeros((num_neur,len(z_time_bins))) #Hz
            for bdt_i, bdt in enumerate(z_time_bins):
                seg_fr[:,bdt_i] = np.sum(segment_spike_times_s_i_bin[:,bdt:bdt+bin_dt],1)/(bin_dt/1000)
            mean_fr = np.nanmean(seg_fr,1)
            seg_means.append(mean_fr)
            std_fr = np.nanstd(seg_fr,1)
            seg_stds.append(std_fr)
        # Binerize Shuffled Segment Spike Times
        segment_spike_times_s_i_shuffle = [sample(list(np.arange(seg_len)),len(segment_spike_times[s_i][n_i])) for n_i in range(num_neur)]
        segment_spike_times_s_i_shuffle_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = np.array(
                segment_spike_times_s_i_shuffle[n_i]).astype('int')
            segment_spike_times_s_i_shuffle_bin[n_i, n_i_spike_times] = 1
        #Create fr vecs
        fr_vec_widths = [sample(list(np.arange(50,1000)),1)[0] for i in range(500)]
        fr_vec_starts = sample(list(np.arange(1000,seg_len-1000)),500)
        for fr_i, fr_s in enumerate(fr_vec_starts):
            fr_w = fr_vec_widths[fr_i]
            fr_vec = np.sum(segment_spike_times_s_i_shuffle_bin[:,fr_s:fr_s+fr_w],1)/(fr_w/1000)
            if z_score == True:
                shuffled_fr_vecs.append(list((fr_vec-mean_fr)/std_fr))
            else:
                shuffled_fr_vecs.append(list(fr_vec))
    
    return shuffled_fr_vecs, segment_spike_times_bin, seg_means, seg_stds

def decode_groupings(epochs_to_analyze,dig_in_names,palatable_dig_inds,non_none_tastes):
    """
    Create fr vector grouping instructions: list of epoch,taste pairs.
    
    Parameters
    ----------
    epochs_to_analyze: list
        list of indices of epochs to be analyzed.
    dig_in_names: list
        list of strings of taste names
    palatable_dig_inds: list
        list of indices of tastes that are palatable
    non_none_tastes: list of taste names that are not "none"
    
    Returns
    -------
    group_list: list
        list of lists containing tuples (e_i,t_i) of which epoch and taste index
        belongs to a decoding group.
    group_names: list
        list of strings naming the decoding groups.
    """
    group_list = []
    group_list_names = []
    none_group = []
    palatable_group = []
    unpalatable_group = []
    for e_ind, e_i in enumerate(epochs_to_analyze):
        epoch_group = []
        epoch_names = []
        for t_ind, t_name in enumerate(dig_in_names):
            #First check if this is none
            if np.setdiff1d([t_name],non_none_tastes).size > 0:
                none_group.append((e_i,t_ind))
            #Next check if it should be grouped
            else:
                if e_i == 0:
                    epoch_group.append((e_i, t_ind))
                    epoch_names.append((e_i, t_name))
                elif e_i == 1:
                    group_list.append([(e_i,t_ind)])
                    group_list_names.append([(e_i,t_name)])
                else:
                    #Check if palatable or unpalatable
                    if (np.intersect1d(palatable_dig_inds,t_name)).size == 0: #unpalatable
                        unpalatable_group.append((e_i,t_ind))
                    else:
                        palatable_group.append((e_i,t_ind))
        if len(epoch_group) > 0:
            group_list.append(epoch_group)
            group_list_names.append([(e_i,'all')])
    group_list.append(palatable_group)
    group_list_names.append(['Palatable'])
    group_list.append(unpalatable_group)
    group_list_names.append(['Unpalatable'])
    group_list.append(none_group)
    group_list_names.append(['None'])
    
    #Prompt the user to name each group
    group_names = []
    for gl_i, gl in enumerate(group_list_names):
        print("\n")
        print(gl)
        gl_name = input("How should the above group be colloquially named? ")
        group_names.append(gl_name)
    
    return group_list, group_names

def train_taste_PCA(num_tastes,num_neur, epochs_to_analyze,taste_num_deliv,tastant_fr_dist):
    """
    Train a PCA reducer on taste response data.

    Parameters
    ----------
    num_tastes : int
        number of tastes to analyze.
    num_neur : int
        number of neurons to analyze.
    epochs_to_analyze : list
        list of epoch indices to analyze.
    taste_num_deliv : list
        list of number of deliveries by taste.
    tastant_fr_dist : list
        list of lists containing firing rate vectors broken down by taste,
        delivery, and epoch.

    Returns
    -------
    pca_reduce_taste : pca model
        a fit PCA model that can be used in dimensionality reduction.

    """
    
    true_taste_train_data = [] #For PCA all combined true taste data
    for t_i in range(num_tastes):
        train_taste_data = []
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for d_i in range(int(taste_num_deliv[t_i])):
                try:
                    if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                        train_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i].T))
                    else:
                        train_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i]))
                except:
                    train_taste_data.extend([])
        if t_i < num_tastes-1:
            true_taste_train_data.extend(train_taste_data)
    #Taste-Based PCA
    taste_pca = PCA()
    taste_pca.fit(np.array(true_taste_train_data).T)
    exp_var = taste_pca.explained_variance_ratio_
    num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
    if num_components == 0:
        num_components = 3
    pca_reduce_taste = PCA(num_components)
    pca_reduce_taste.fit(np.array(true_taste_train_data))
    
    return pca_reduce_taste

def plot_decode_trends(num_groups, grouped_train_names, segments_to_analyze,
                       seg_group_frac, seg_names, decode_dir):
    """
    This function generates plots of decoding rates

    Parameters
    ----------
    num_groups : int
        Number of decode groups in dataset.
    grouped_train_names : list
        List of strings of the group names.
    segments_to_analyze : list
        List of segment indices to analyze.
    seg_group_frac : numpy array
        Array of size len(segments_to_analyze) x num_groups containing fraction
        of decodes of each group within the segment.
    seg_names : list
        List of segment names.
    decode_dir : string
        Directory to save results.

    Returns
    -------
    Plots stored to directory 'decode_dir'.

    """
    #Plot decode fraction trends
    f_trends, ax_trends = plt.subplots(nrows = 1, ncols = num_groups, \
                                       sharex = True, figsize = (num_groups*4,4))
    for g_ind, g_name in enumerate(grouped_train_names):
        ax_trends[g_ind].plot(np.arange(len(segments_to_analyze)),seg_group_frac[:,g_ind])
        ax_trends[g_ind].set_xticks(np.arange(len(segments_to_analyze)),seg_names)
        ax_trends[g_ind].set_xlabel('Segment')
        ax_trends[g_ind].set_ylabel('Decode Fraction')
        ax_trends[g_ind].set_title(g_name)
    plt.suptitle('Decoding Fraction Across Segments')
    plt.tight_layout()
    f_trends.savefig(os.path.join(decode_dir,'seg_group_frac_trends.png'))
    f_trends.savefig(os.path.join(decode_dir,'seg_group_frac_trends.svg'))
    plt.close(f_trends)
    
    f_trends_combined = plt.figure(figsize=(5,5))
    for g_ind, g_name in enumerate(grouped_train_names):
        plt.plot(np.arange(len(segments_to_analyze)),seg_group_frac[:,g_ind],\
                 label=g_name)
    plt.legend(loc='upper left')
    plt.xticks(np.arange(len(segments_to_analyze)),seg_names)
    plt.xlabel('Segment')
    plt.ylabel('Decode Fraction')
    plt.suptitle('Decoding Fraction Across Segments')
    plt.tight_layout()
    f_trends_combined.savefig(os.path.join(decode_dir,'seg_group_frac_trends_combined.png'))
    f_trends_combined.savefig(os.path.join(decode_dir,'seg_group_frac_trends_combined.svg'))
    plt.close(f_trends_combined)
    
    f_trends_pie, ax_trends_pie = plt.subplots(nrows = 1, ncols = len(segments_to_analyze), \
                                       figsize = (len(segments_to_analyze)*4,4))
    for s_ind, s_i in enumerate(segments_to_analyze):
        pie_labels = [g_name + '\n' + str(np.round(100*seg_group_frac[s_ind,g_i],2)) + \
                      '%' for g_i, g_name in enumerate(grouped_train_names)]
        ax_trends_pie[s_ind].pie(seg_group_frac[s_ind,:],labels=pie_labels)
        ax_trends_pie[s_ind].set_title(seg_names[s_ind])
    plt.title('Decode Fractions by Segment')
    plt.tight_layout()
    f_trends_pie.savefig(os.path.join(decode_dir,'seg_group_frac_trends_pie.png'))
    f_trends_pie.savefig(os.path.join(decode_dir,'seg_group_frac_trends_pie.svg'))
    plt.close(f_trends_pie)
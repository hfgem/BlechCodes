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
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = seg_end - seg_start
        time_bin_starts = np.arange(
            seg_start+half_bin, seg_end-half_bin, bin_dt)
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = np.array(
                segment_spike_times_s_i[n_i] - seg_start).astype('int')
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


def decode_epochs(tastant_fr_dist, segment_spike_times, post_taste_dt,
                  e_skip_dt, e_len_dt, dig_in_names, segment_times,
                  segment_names, start_dig_in_times, taste_num_deliv,
                  taste_select_epoch, max_hz, save_dir,
                  neuron_count_thresh, decode_prob_cutoff, z_score = False,
                  epochs_to_analyze=[], segments_to_analyze=[]):
    """Decode taste from epoch-specific firing rates"""
    print('\t\tRunning GMM Decoder')
    # Variables
    num_tastes = len(start_dig_in_times)
    num_neur = len(segment_spike_times[0])
    num_cp = len(tastant_fr_dist[0][0])
    num_segments = len(segment_spike_times)
    p_taste = taste_num_deliv/np.sum(taste_num_deliv)  # P(taste)
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    half_decode_bin_dt = np.ceil(e_len_dt/2).astype('int')
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, len(epochs_to_analyze)))

    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
        
    # If trial_start_frac > 0 use only trials after that threshold
    #trial_start_ind = np.floor(max_num_deliv*trial_start_frac).astype('int')
    for e_ind, e_i in enumerate(epochs_to_analyze):  # By epoch conduct decoding
        print('\t\t\tDecoding Epoch ' + str(e_i))
        
        epoch_decode_save_dir = os.path.join(
            save_dir, 'decode_prob_epoch_' + str(e_i))
        if not os.path.isdir(epoch_decode_save_dir):
            os.mkdir(epoch_decode_save_dir)

        taste_select_neur = np.where(taste_select_epoch[e_i, :] == 1)[0]
        
        # Collect fr of each population for each taste
        train_data = []
        true_taste_train_data = [] #Only true tastes - excluding "none"
        for t_i in range(num_tastes):
            train_taste_data = []
            taste_num_deliv = len(tastant_fr_dist[t_i])
            for d_i in range(taste_num_deliv):
                if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                    train_taste_data.extend(
                        list(tastant_fr_dist[t_i][d_i][e_i].T))
                else:
                    train_taste_data.extend(
                        list(tastant_fr_dist[t_i][d_i][e_i]))
            if t_i < num_tastes-1:
                true_taste_train_data.extend(train_taste_data)
            else: #None condition - augment with randomized data in neuron firing range + single spike examples
                neur_max = np.expand_dims(np.max(np.array(train_taste_data),0),1)
                train_taste_data.extend(list((neur_max*np.random.rand(num_neur,100)).T))
                train_taste_data.extend(list(((neur_max/10)*np.random.rand(num_neur,100)).T))
                for eye_i in range(50):
                    train_taste_data.extend(list(neur_max*np.eye(num_neur).T))
                train_taste_data.extend(list(np.zeros((num_neur,100)).T))
            train_data.append(np.array(train_taste_data))
        train_counts = np.array([len(train_data[t_i]) for t_i in range(len(train_data))])
        p_taste_train_counts = train_counts/np.sum(train_counts)
        
        # Run PCA transform only on non-z-scored data
        if np.min(true_taste_train_data) >= 0:
            pca = PCA()
            pca.fit(np.array(true_taste_train_data).T)
            exp_var = pca.explained_variance_ratio_
            num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
            if num_components == 0:
                num_components = 3
            pca_reduce = PCA(num_components)
            pca_reduce.fit(np.array(true_taste_train_data))

        # Fit a Gaussian mixture model with the number of dimensions = number of neurons
        all_taste_gmm = dict()
        for t_i in range(num_tastes):
            train_taste_data = train_data[t_i]
            if np.min(true_taste_train_data) >= 0:
                # ___PCA Transformed Data
                transformed_test_taste_data = pca_reduce.transform(
                    np.array(train_taste_data))
            else:
                # ___True Data
                transformed_test_taste_data = np.array(
                    train_taste_data)
            gm = gmm(n_components=1, n_init=10).fit(
                transformed_test_taste_data)
            # Insert here a line of fitting the Gamma-MM
            all_taste_gmm[t_i] = gm

        # Segment-by-segment use full taste decoding times to zoom in and test
        #	epoch-specific and smaller interval
        for s_i in segments_to_analyze:
            # Get segment variables
            seg_start = segment_times[s_i]
            seg_end = segment_times[s_i+1]
            seg_len = segment_times[s_i+1] - segment_times[s_i]  # in dt = ms
            new_time_bins = np.arange(
                seg_start+half_decode_bin_dt, seg_end-half_decode_bin_dt, e_skip_dt)
            # Now pull epoch-specific probabilities
            seg_decode_epoch_prob = np.zeros((num_tastes, seg_len))
            # Start with assumption of "none" taste at all times
            seg_decode_epoch_prob[-1, :] = 1
            # Binerize Spike Times
            segment_spike_times_s_i = segment_spike_times[s_i]
            segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
            for n_i in taste_select_neur:
                n_i_spike_times = np.array(
                    segment_spike_times_s_i[n_i] - seg_start).astype('int')
                segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
            
            # Grab neuron firing rates in sliding bins
            try:
                seg_decode_epoch_prob = np.load(
                    os.path.join(epoch_decode_save_dir,'segment_' + str(s_i) + '.npy'))
                tb_fr = np.load(os.path.join(epoch_decode_save_dir,
                                'segment_' + str(s_i) + '_tb_fr.npy'))
                print('\t\t\t\tSegment ' + str(s_i) + ' Previously Decoded')
            except:
                print('\t\t\t\tDecoding Segment ' + str(s_i))
                # Perform parallel computation for each time bin
                print('\t\t\t\t\tCalculate firing rates for time bins')
                try:
                    tb_fr = np.load(os.path.join(epoch_decode_save_dir,
                                    'segment_' + str(s_i) + '_tb_fr.npy'))
                except:
                    if len(new_time_bins) > 1:
                        tb_fr = np.zeros((num_neur, len(new_time_bins)))
                        for tb_i, tb in enumerate(new_time_bins):
                            tb_fr[:, tb_i] = np.sum(
                                segment_spike_times_s_i_bin[:, tb-seg_start-half_decode_bin_dt:tb+half_decode_bin_dt-seg_start], 1)/(int(2*half_decode_bin_dt)/1000)
                        np.save(os.path.join(epoch_decode_save_dir,'segment_' +
                                str(s_i) + '_tb_fr.npy'), tb_fr)
                    else:
                        tb_fr = np.expand_dims(np.sum(segment_spike_times_s_i_bin,1)/((seg_len+1)/1000),1)
                    del tb_i, tb
                #Z-scoring
                if z_score == True:
                    #Calculate mean and std for binned spikes
                    tb_fr_mean = np.mean(tb_fr,1)
                    tb_fr_std = np.std(tb_fr,1)
                    #Convert tb_fr to z-score
                    tb_fr = np.divide(tb_fr - np.expand_dims(tb_fr_mean,1),np.expand_dims(tb_fr_std,1))
                #Converting to list for parallel processing
                if np.min(true_taste_train_data) >= 0:    
                    tb_fr_pca = pca_reduce.transform(tb_fr.T)
                    list_tb_fr = list(tb_fr_pca)
                else:
                    list_tb_fr = list(tb_fr)
                del tb_fr
                
                # Pass inputs to parallel computation on probabilities
                inputs = zip(list_tb_fr, itertools.repeat(num_tastes),
                              itertools.repeat(all_taste_gmm), itertools.repeat(p_taste))
                tic = time.time()
                pool = Pool(4)
                tb_decode_prob = pool.map(
                    dp.segment_taste_decode_dependent_parallelized, inputs)
                pool.close()
                toc = time.time()
                print('\t\t\t\t\tTime to decode = ' +
                      str(np.round((toc-tic)/60, 2)) + ' (min)')
                tb_decode_array = np.squeeze(np.array(tb_decode_prob)).T
                # The whole skip interval should have the same decode probability
                for s_dt in range(e_skip_dt):
                    seg_decode_epoch_prob[:, new_time_bins -
                                          seg_start + s_dt] = tb_decode_array
                # Save decoding probabilities
                np.save(os.path.join(epoch_decode_save_dir,'segment_' +
                        str(s_i) + '.npy'), seg_decode_epoch_prob)
            # Create plots
            seg_decode_save_dir = os.path.join(epoch_decode_save_dir,
                'segment_' + str(s_i) + '/')
            if not os.path.isdir(seg_decode_save_dir):
                os.mkdir(seg_decode_save_dir)

            seg_decode_epoch_prob_nonan = np.zeros(
                np.shape(seg_decode_epoch_prob))
            seg_decode_epoch_prob_nonan[:] = seg_decode_epoch_prob[:]
            seg_decode_epoch_prob_nonan[np.isnan(
                seg_decode_epoch_prob_nonan)] = 0
            seg_decode_missing = np.where(np.sum(seg_decode_epoch_prob_nonan,0) == 0)[0]
            seg_decode_epoch_prob_nonan[-1,seg_decode_missing] = 1 #Make it a no-taste timepoint
            seg_decode_epoch_taste_ind = np.argmax(seg_decode_epoch_prob, 0)
            
            # Updated decoding based on threshold
            seg_decode_epoch_taste_bin = np.zeros(
                np.shape(seg_decode_epoch_prob))
            for t_i in range(num_tastes):
                if t_i < num_tastes-1:
                    taste_max_bin = (seg_decode_epoch_taste_ind == t_i).astype('int')
                    taste_bin_cutoff = (seg_decode_epoch_prob[t_i,:] >= decode_prob_cutoff).astype('int')
                    taste_bin = taste_max_bin*taste_bin_cutoff
                    # To ensure starts and ends of bins align
                    taste_bin[0] = 0
                    taste_bin[-1] = 0
                    # Calculate decoding periods
                    diff_decoded_taste = np.diff(taste_bin)
                    start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
                    end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
                    num_decoded = len(start_decoded)
                    # Calculate number of neurons in each period
                    num_neur_decoded = np.zeros(num_decoded)
                    for nd_i in range(num_decoded):
                        d_start = start_decoded[nd_i]
                        d_end = end_decoded[nd_i]
                        for n_i in range(num_neur):
                            if len(np.where(segment_spike_times_s_i_bin[n_i, d_start:d_end])[0]) > 0:
                                num_neur_decoded[nd_i] += 1
                    # Now cut at threshold and only keep matching decoded intervals
                    decode_ind = np.where(
                        num_neur_decoded > neuron_count_thresh)[0]
                    for db in decode_ind:
                        s_db = start_decoded[db]
                        e_db = end_decoded[db]
                        seg_decode_epoch_taste_bin[t_i, s_db:e_db] = 1
                else: #No taste everywhere the other taste is not marked
                    where_decoded_taste = (np.sum(seg_decode_epoch_taste_bin,0) > 0).astype('int')
                    where_not_decoded_taste = np.ones(len(where_decoded_taste)) - where_decoded_taste
                    seg_decode_epoch_taste_bin[t_i,:] = where_not_decoded_taste
            # Line plot
            f1 = plt.figure()
            for t_i in range(num_tastes):
                plt.plot(np.arange(seg_start, seg_end)/1000/60, \
                         seg_decode_epoch_prob_nonan[t_i,:], \
                             color=taste_colors[t_i,:], label=dig_in_names[t_i])
                plt.fill_between(np.arange(seg_start, seg_end)/1000/60, \
                                 seg_decode_epoch_taste_bin[t_i, :], alpha=0.2 \
                                 ,color=taste_colors[t_i,:], label='_')
            plt.legend(loc='right')
            plt.ylabel('Decoding Fraction')
            plt.xlabel('Time (min)')
            plt.title('Segment ' + str(s_i))
            f1.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + '.png'))
            f1.savefig(os.path.join(seg_decode_save_dir, 'segment_' + str(s_i) + '.svg'))
            plt.close(f1)
            # Imshow
            f2 = plt.figure()
            plt.imshow(seg_decode_epoch_prob_nonan,
                       aspect='auto', interpolation='none')
            x_ticks = np.ceil(np.linspace(
                0, len(new_time_bins)-1, 10)).astype('int')
            x_tick_labels = np.round(new_time_bins[x_ticks]/1000/60, 2)
            plt.xticks(x_ticks, x_tick_labels)
            y_ticks = np.arange(len(dig_in_names))
            plt.yticks(y_ticks, dig_in_names)
            plt.ylabel('Decoding Fraction')
            plt.xlabel('Time (min)')
            plt.title('Segment ' + str(s_i))
            f2.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + '_im.png'))
            f2.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + '_im.svg'))
            plt.close(f2)
            # Fraction of occurrences
            f3 = plt.figure()
            plt.pie(np.sum(seg_decode_epoch_taste_bin, 1)/np.sum(seg_decode_epoch_taste_bin),
                    labels=['water', 'saccharin', 'none'], autopct='%1.1f%%', pctdistance=1.5,
                    colors=taste_colors)
            plt.title('Segment ' + str(s_i))
            f3.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + '_pie.png'))
            f3.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + '_pie.svg'))
            plt.close(f3)
            
def decode_deviations_epochs(tastant_fr_dist, segment_spike_times, post_taste_dt,
                  e_skip_dt, e_len_dt, dig_in_names, segment_times,
                  segment_names, start_dig_in_times, taste_num_deliv,
                  segment_dev_rasters, segment_dev_times, taste_select_epoch,
                  max_hz, save_dir, neuron_count_thresh, decode_prob_cutoff, 
                  z_score = False, epochs_to_analyze=[], segments_to_analyze=[]):
    """Decode taste from epoch-specific firing rates"""
    print('\t\tRunning GMM Decoder')
    # Variables
    num_tastes = len(start_dig_in_times)
    num_neur = len(segment_spike_times[0])
    num_cp = len(tastant_fr_dist[0][0])
    num_segments = len(segment_spike_times)
    p_taste = taste_num_deliv/np.sum(taste_num_deliv)  # P(taste)
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    half_decode_bin_dt = np.ceil(e_len_dt/2).astype('int')
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, len(epochs_to_analyze)))

    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
        
    # If trial_start_frac > 0 use only trials after that threshold
    #trial_start_ind = np.floor(max_num_deliv*trial_start_frac).astype('int')
    for e_ind, e_i in enumerate(epochs_to_analyze):  # By epoch conduct decoding
        print('\t\t\tDecoding Epoch ' + str(e_i))
        
        epoch_decode_save_dir = os.path.join(
            save_dir, 'decode_prob_epoch_' + str(e_i))
        if not os.path.isdir(epoch_decode_save_dir):
            os.mkdir(epoch_decode_save_dir)

        taste_select_neur = np.where(taste_select_epoch[e_i, :] == 1)[0]
        
        # Collect fr of each population for each taste
        train_data = []
        true_taste_train_data = [] #Only true tastes - excluding "none"
        for t_i in range(num_tastes):
            train_taste_data = []
            taste_num_deliv = len(tastant_fr_dist[t_i])
            for d_i in range(taste_num_deliv):
                if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                    train_taste_data.extend(
                        list(tastant_fr_dist[t_i][d_i][e_i].T))
                else:
                    train_taste_data.extend(
                        list(tastant_fr_dist[t_i][d_i][e_i]))
            if t_i < num_tastes-1:
                true_taste_train_data.extend(train_taste_data)
            else: #None condition - augment with randomized data in neuron firing range + single spike examples
                neur_max = np.expand_dims(np.max(np.array(train_taste_data),0),1)
                train_taste_data.extend(list((neur_max*np.random.rand(num_neur,100)).T))
                train_taste_data.extend(list(((neur_max/10)*np.random.rand(num_neur,100)).T))
                for eye_i in range(50):
                    train_taste_data.extend(list(neur_max*np.eye(num_neur).T))
                train_taste_data.extend(list(np.zeros((num_neur,100)).T))
            train_data.append(np.array(train_taste_data))
        train_counts = np.array([len(train_data[t_i]) for t_i in range(len(train_data))])
        p_taste_train_counts = train_counts/np.sum(train_counts)
        
        # Run PCA transform only on non-z-scored data
        if np.min(true_taste_train_data) >= 0:
            pca = PCA()
            pca.fit(np.array(true_taste_train_data).T)
            exp_var = pca.explained_variance_ratio_
            num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
            if num_components == 0:
                num_components = 3
            pca_reduce = PCA(num_components)
            pca_reduce.fit(np.array(true_taste_train_data))

        # Fit a Gaussian mixture model with the number of dimensions = number of neurons
        all_taste_gmm = dict()
        for t_i in range(num_tastes):
            train_taste_data = train_data[t_i]
            if np.min(true_taste_train_data) >= 0:
                # ___PCA Transformed Data
                transformed_test_taste_data = pca_reduce.transform(
                    np.array(train_taste_data))
            else:
                # ___True Data
                transformed_test_taste_data = np.array(
                    train_taste_data)
            gm = gmm(n_components=1, n_init=10).fit(
                transformed_test_taste_data)
            # Insert here a line of fitting the Gamma-MM
            all_taste_gmm[t_i] = gm

        # Segment-by-segment use full taste decoding times to zoom in and test
        #	epoch-specific and smaller interval
        for s_i in segments_to_analyze:
            # Get segment variables
            seg_start = segment_times[s_i]
            seg_end = segment_times[s_i+1]
            seg_len = segment_times[s_i+1] - segment_times[s_i]  # in dt = ms
            new_time_bins = np.arange(
                seg_start+half_decode_bin_dt, seg_end-half_decode_bin_dt, e_skip_dt)
            # Now pull epoch-specific probabilities
            seg_decode_epoch_prob = np.zeros((num_tastes, seg_len))
            # Start with assumption of "none" taste at all times
            seg_decode_epoch_prob[-1, :] = 1
            # Binerize Spike Times
            segment_spike_times_s_i = segment_spike_times[s_i]
            segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
            for n_i in taste_select_neur:
                n_i_spike_times = np.array(
                    segment_spike_times_s_i[n_i] - seg_start).astype('int')
                segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
            
            # Grab neuron firing rates in sliding bins
            try:
                seg_decode_epoch_prob = np.load(
                    os.path.join(epoch_decode_save_dir,'segment_' + str(s_i) + '.npy'))
                tb_fr = np.load(os.path.join(epoch_decode_save_dir,
                                'segment_' + str(s_i) + '_tb_fr.npy'))
                print('\t\t\t\tSegment ' + str(s_i) + ' Previously Decoded')
            except:
                print('\t\t\t\tDecoding Segment ' + str(s_i))
                # Perform parallel computation for each time bin
                print('\t\t\t\t\tCalculate firing rates for time bins')
                try:
                    tb_fr = np.load(os.path.join(epoch_decode_save_dir,
                                    'segment_' + str(s_i) + '_tb_fr.npy'))
                except:
                    if len(new_time_bins) > 1:
                        tb_fr = np.zeros((num_neur, len(new_time_bins)))
                        for tb_i, tb in enumerate(new_time_bins):
                            tb_fr[:, tb_i] = np.sum(
                                segment_spike_times_s_i_bin[:, tb-seg_start-half_decode_bin_dt:tb+half_decode_bin_dt-seg_start], 1)/(int(2*half_decode_bin_dt)/1000)
                        np.save(os.path.join(epoch_decode_save_dir,'segment_' +
                                str(s_i) + '_tb_fr.npy'), tb_fr)
                    else:
                        tb_fr = np.expand_dims(np.sum(segment_spike_times_s_i_bin,1)/((seg_len+1)/1000),1)
                    del tb_i, tb
                #Z-scoring
                if z_score == True:
                    #Calculate mean and std for binned spikes
                    tb_fr_mean = np.mean(tb_fr,1)
                    tb_fr_std = np.std(tb_fr,1)
                    #Convert tb_fr to z-score
                    tb_fr = np.divide(tb_fr - np.expand_dims(tb_fr_mean,1),np.expand_dims(tb_fr_std,1))
                #Converting to list for parallel processing
                if np.min(true_taste_train_data) >= 0:    
                    tb_fr_pca = pca_reduce.transform(tb_fr.T)
                    list_tb_fr = list(tb_fr_pca)
                else:
                    list_tb_fr = list(tb_fr)
                del tb_fr
                
                # Pass inputs to parallel computation on probabilities
                inputs = zip(list_tb_fr, itertools.repeat(num_tastes),
                              itertools.repeat(all_taste_gmm), itertools.repeat(p_taste))
                tic = time.time()
                pool = Pool(4)
                tb_decode_prob = pool.map(
                    dp.segment_taste_decode_dependent_parallelized, inputs)
                pool.close()
                toc = time.time()
                print('\t\t\t\t\tTime to decode = ' +
                      str(np.round((toc-tic)/60, 2)) + ' (min)')
                tb_decode_array = np.squeeze(np.array(tb_decode_prob)).T
                # The whole skip interval should have the same decode probability
                for s_dt in range(e_skip_dt):
                    seg_decode_epoch_prob[:, new_time_bins -
                                          seg_start + s_dt] = tb_decode_array
                # Save decoding probabilities
                np.save(os.path.join(epoch_decode_save_dir,'segment_' +
                        str(s_i) + '.npy'), seg_decode_epoch_prob)
            # Create plots
            seg_decode_save_dir = os.path.join(epoch_decode_save_dir,
                'segment_' + str(s_i) + '/')
            if not os.path.isdir(seg_decode_save_dir):
                os.mkdir(seg_decode_save_dir)

            seg_decode_epoch_prob_nonan = np.zeros(
                np.shape(seg_decode_epoch_prob))
            seg_decode_epoch_prob_nonan[:] = seg_decode_epoch_prob[:]
            seg_decode_epoch_prob_nonan[np.isnan(
                seg_decode_epoch_prob_nonan)] = 0
            seg_decode_missing = np.where(np.sum(seg_decode_epoch_prob_nonan,0) == 0)[0]
            seg_decode_epoch_prob_nonan[-1,seg_decode_missing] = 1 #Make it a no-taste timepoint
            seg_decode_epoch_taste_ind = np.argmax(seg_decode_epoch_prob, 0)
            
            # Updated decoding based on threshold
            seg_decode_epoch_taste_bin = np.zeros(
                np.shape(seg_decode_epoch_prob))
            for t_i in range(num_tastes):
                if t_i < num_tastes-1:
                    taste_max_bin = (seg_decode_epoch_taste_ind == t_i).astype('int')
                    taste_bin_cutoff = (seg_decode_epoch_prob[t_i,:] >= decode_prob_cutoff).astype('int')
                    taste_bin = taste_max_bin*taste_bin_cutoff
                    # To ensure starts and ends of bins align
                    taste_bin[0] = 0
                    taste_bin[-1] = 0
                    # Calculate decoding periods
                    diff_decoded_taste = np.diff(taste_bin)
                    start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
                    end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
                    num_decoded = len(start_decoded)
                    # Calculate number of neurons in each period
                    num_neur_decoded = np.zeros(num_decoded)
                    for nd_i in range(num_decoded):
                        d_start = start_decoded[nd_i]
                        d_end = end_decoded[nd_i]
                        for n_i in range(num_neur):
                            if len(np.where(segment_spike_times_s_i_bin[n_i, d_start:d_end])[0]) > 0:
                                num_neur_decoded[nd_i] += 1
                    # Now cut at threshold and only keep matching decoded intervals
                    decode_ind = np.where(
                        num_neur_decoded > neuron_count_thresh)[0]
                    for db in decode_ind:
                        s_db = start_decoded[db]
                        e_db = end_decoded[db]
                        seg_decode_epoch_taste_bin[t_i, s_db:e_db] = 1
                else: #No taste everywhere the other taste is not marked
                    where_decoded_taste = (np.sum(seg_decode_epoch_taste_bin,0) > 0).astype('int')
                    where_not_decoded_taste = np.ones(len(where_decoded_taste)) - where_decoded_taste
                    seg_decode_epoch_taste_bin[t_i,:] = where_not_decoded_taste
            # Line plot
            f1 = plt.figure()
            for t_i in range(num_tastes):
                plt.plot(np.arange(seg_start, seg_end)/1000/60, \
                         seg_decode_epoch_prob_nonan[t_i,:], \
                             color=taste_colors[t_i,:], label=dig_in_names[t_i])
                plt.fill_between(np.arange(seg_start, seg_end)/1000/60, \
                                 seg_decode_epoch_taste_bin[t_i, :], alpha=0.2 \
                                 ,color=taste_colors[t_i,:], label='_')
            plt.legend(loc='right')
            plt.ylabel('Decoding Fraction')
            plt.xlabel('Time (min)')
            plt.title('Segment ' + str(s_i))
            f1.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + '.png'))
            f1.savefig(os.path.join(seg_decode_save_dir, 'segment_' + str(s_i) + '.svg'))
            plt.close(f1)
            # Imshow
            f2 = plt.figure()
            plt.imshow(seg_decode_epoch_prob_nonan,
                       aspect='auto', interpolation='none')
            x_ticks = np.ceil(np.linspace(
                0, len(new_time_bins)-1, 10)).astype('int')
            x_tick_labels = np.round(new_time_bins[x_ticks]/1000/60, 2)
            plt.xticks(x_ticks, x_tick_labels)
            y_ticks = np.arange(len(dig_in_names))
            plt.yticks(y_ticks, dig_in_names)
            plt.ylabel('Decoding Fraction')
            plt.xlabel('Time (min)')
            plt.title('Segment ' + str(s_i))
            f2.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + '_im.png'))
            f2.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + '_im.svg'))
            plt.close(f2)
            # Fraction of occurrences
            f3 = plt.figure()
            plt.pie(np.sum(seg_decode_epoch_taste_bin, 1)/np.sum(seg_decode_epoch_taste_bin),
                    labels=['water', 'saccharin', 'none'], autopct='%1.1f%%', pctdistance=1.5,
                    colors=taste_colors)
            plt.title('Segment ' + str(s_i))
            f3.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + '_pie.png'))
            f3.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + '_pie.svg'))
            plt.close(f3)


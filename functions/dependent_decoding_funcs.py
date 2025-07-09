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
import random
import numpy as np
#file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
# os.chdir(file_path)
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches
from multiprocess import Pool
from sklearn.mixture import GaussianMixture as gmm
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import functions.decode_parallel as dp
import functions.plot_dev_decoding_funcs as pddf
from sklearn import svm
from random import sample
from scipy.stats import pearsonr, ks_2samp

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

def decode_deviations(tastant_fr_dist, tastant_spike_times, segment_spike_times, dig_in_names, 
                  segment_times, segment_names, start_dig_in_times, taste_num_deliv,
                  segment_dev_times, dev_vecs, bin_dt, 
                  group_list, group_names, non_none_tastes, decode_dir, 
                  z_score = False, segments_to_analyze=[]):
    """Decode taste from epoch-specific firing rates"""
    print('\t\tRunning NB Decoder')
    
    decode_save_dir = os.path.join(decode_dir,'NB_Decoding')
    if not os.path.isdir(decode_save_dir):
        os.mkdir(decode_save_dir)
    
    # Variables
    num_tastes = len(start_dig_in_times)
    num_neur = len(segment_spike_times[0])
    num_segments = len(segment_spike_times)
    if not group_names[-1] == 'Null Data':
        num_groups = len(group_names) + 1
        group_names.append('Null Data')
    else:
        num_groups = len(group_names)
    cmap = colormaps['jet']
    group_colors = cmap(np.linspace(0, 1, num_groups))
    dev_buffer = 50
    
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
        
    seg_names = list(np.array(segment_names)[segments_to_analyze])
       
    #Grab all taste-epoch pairs in the training groups for testing
    taste_epoch_pairs = []
    for gl_i, gl in enumerate(group_list):
        for gp_i, gp in enumerate(gl):
            taste_epoch_pairs.append([gp[1],gp[0]])
    
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
    num_groups = len(grouped_train_names)
    
    #Create categorical NB dataset
    categorical_train_data = []
    categorical_train_y = []
    for g_i, g_data in enumerate(grouped_train_data):
        categorical_train_data.extend(g_data)
        categorical_train_y.extend(g_i*np.ones(len(g_data)).astype('int'))
    categorical_train_data = np.array(categorical_train_data)
    categorical_train_y = np.array(categorical_train_y)
    
    #Plot PCA-reduced training data
    #PCA
    pca_reduce_plot = PCA(2)
    pca_data = pca_reduce_plot.fit_transform(np.array(categorical_train_data))
    f_pca, ax_pca = plt.subplots(nrows = 1, ncols = 2, 
                                 sharex = True, sharey = True,
                                 figsize=(10,5))
    for g_i, g_name in enumerate(grouped_train_names):
        group_where = np.where(categorical_train_y == g_i)[0]
        if len(group_where) > 0:
            ax_pca[0].scatter(pca_data[group_where,0],pca_data[group_where,1],\
                        color = group_colors[g_i,:],\
                        alpha=0.5,label=g_name)
    ax_pca[0].set_title('Individual Responses')
    for g_i, g_name in enumerate(grouped_train_names):
        group_where = np.where(categorical_train_y == g_i)[0]
        if len(group_where) > 0:
            mean_x = np.nanmean(pca_data[group_where,0])
            x_std = np.nanstd(pca_data[group_where,0])
            mean_y = np.nanmean(pca_data[group_where,1])
            y_std = np.nanstd(pca_data[group_where,1])
            m, c = np.polyfit(pca_data[group_where,0], pca_data[group_where,1], 1)
            angle = np.degrees(np.arctan(m))
            oval = patches.Ellipse(
                    (mean_x, mean_y),
                    2*x_std,
                    2*y_std,
                    angle,
                    facecolor=group_colors[g_i,:],
                    alpha=0.3
                    )
            ax_pca[1].add_patch(oval)
            ax_pca[1].scatter(mean_x,mean_y,\
                        color = group_colors[g_i,:],\
                        alpha=1,label=g_name)
    ax_pca[1].set_title('Average Locations')
    ax_pca[1].legend(loc='lower right')
    plt.suptitle('PCA Projection of Training Groups')
    plt.tight_layout()
    f_pca.savefig(os.path.join(decode_save_dir,'pca_data_distribution.png'))
    f_pca.savefig(os.path.join(decode_save_dir,'pca_data_distribution.svg'))
    plt.close(f_pca)
    
    #Run PCA transform only on non-z-scored data
    if z_score == True:
        train_data = categorical_train_data
    else:
        pca_reduce_taste = train_taste_PCA(num_neur,taste_epoch_pairs,\
                                           non_none_tastes,dig_in_names,
                                           taste_num_deliv,tastant_fr_dist)
        train_data = pca_reduce_taste.transform(categorical_train_data)
   
    #Fit NB
    nb = GaussianNB()
    nb.fit(train_data, categorical_train_y) 
       
    #Store segment by segment fraction of decodes
    seg_decode_counts = np.nan*np.ones((len(segments_to_analyze),num_groups))
    seg_decode_frac = np.nan*np.ones((len(segments_to_analyze),num_groups))
    seg_decode_counts_pre = np.nan*np.ones((len(segments_to_analyze),num_groups))
    seg_decode_frac_pre = np.nan*np.ones((len(segments_to_analyze),num_groups))
    seg_decode_counts_post = np.nan*np.ones((len(segments_to_analyze),num_groups))
    seg_decode_frac_post = np.nan*np.ones((len(segments_to_analyze),num_groups))
    
    # Segment-by-segment use deviation rasters and times to zoom in and test
    #	decoding of tastes. Add decoding of 50 ms on either
    #	side of the deviation event as well for context decoding.
    for seg_ind, s_i in enumerate(segments_to_analyze):
        # Create plot save dir
        seg_decode_save_dir = os.path.join(decode_save_dir,
            'segment_' + str(s_i) + '/')
        if not os.path.isdir(seg_decode_save_dir):
            os.mkdir(seg_decode_save_dir)
        
        # Get segment variables
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = seg_end - seg_start# in dt = ms
        seg_dev_fr_mat = np.array(dev_vecs[seg_ind]).T
        seg_dev_times = segment_dev_times[seg_ind]
        _, num_dev = np.shape(seg_dev_times)
        segment_spike_times_s_i_bin = segment_spike_times_bin[seg_ind]
        if z_score == True:
            mean_fr = seg_means[seg_ind]
            std_fr = seg_stds[seg_ind]
        
        #Import existing data
        try:
            dev_decode_prob_array = np.load( os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                          '_deviation_decodes.npy'))
            pre_dev_decode_prob_array = np.load(
                os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                             '_pre_deviations_decodes.npy'))
            post_dev_decode_prob_array = np.load(
                os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                             '_post_deviations_decodes.npy'))
                
            print('\t\t\t\tSegment ' + str(s_i) + ' Previously Decoded')
        except:
            print('\t\t\t\tDecoding Segment ' + str(s_i) + ' Deviations')
            tic = time.time()
            
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
            if z_score == False:    
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
            
            #NB
            tic = time.time()
            #Dev categorical
            dev_decode_prob_array = nb.predict_proba(list_dev_fr)
            #Pre-dev categorical
            pre_dev_decode_prob_array = nb.predict_proba(list_pre_dev_fr)
            #Post-dev categorical
            post_dev_decode_prob_array = nb.predict_proba(list_post_dev_fr)
            
            # Save decoding probabilities
            np.save( os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                          '_deviation_decodes.npy'),dev_decode_prob_array)
            np.save(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                             '_pre_deviations_decodes.npy'), \
                    pre_dev_decode_prob_array)
            np.save(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                             '_post_deviations_decodes.npy'),post_dev_decode_prob_array)
                
            toc = time.time()
            print('\t\t\t\t\tTime to decode = ' +
                  str(np.round((toc-tic)/60, 2)) + ' (min)')
        
        #Store segment fraction of decodes
        dev_decode_argmax = np.argmax(dev_decode_prob_array,1)
        hist_vals = np.histogram(dev_decode_argmax,bins=np.arange(num_groups+1))
        seg_decode_counts[seg_ind,:] = hist_vals[0]
        seg_decode_frac[seg_ind,:] = hist_vals[0]/num_dev
        
        pre_dev_decode_argmax = np.argmax(pre_dev_decode_prob_array,1)
        pre_hist_vals = np.histogram(pre_dev_decode_argmax,bins=np.arange(num_groups+1))
        seg_decode_counts_pre[seg_ind,:] = pre_hist_vals[0]
        seg_decode_frac_pre[seg_ind,:] = pre_hist_vals[0]/num_dev
        
        post_dev_decode_argmax = np.argmax(post_dev_decode_prob_array,1)
        post_hist_vals = np.histogram(post_dev_decode_argmax,bins=np.arange(num_groups+1))
        seg_decode_counts_post[seg_ind,:] = post_hist_vals[0]
        seg_decode_frac_post[seg_ind,:] = post_hist_vals[0]/num_dev
        
    #Create trend plots
    pddf.plot_decode_trends(num_groups, grouped_train_names, segments_to_analyze,
                           seg_decode_frac, seg_decode_counts, seg_names, 
                           decode_save_dir)
    pre_save_dir = os.path.join(decode_save_dir,'pre_dev')
    if not os.path.isdir(pre_save_dir):
        os.mkdir(pre_save_dir)
    pddf.plot_decode_trends(num_groups, grouped_train_names, segments_to_analyze,
                           seg_decode_frac_pre, seg_decode_counts_pre, 
                           seg_names, pre_save_dir)
    
    
    post_save_dir = os.path.join(decode_save_dir,'post_dev')
    if not os.path.isdir(post_save_dir):
        os.mkdir(post_save_dir)
    pddf.plot_decode_trends(num_groups, grouped_train_names, segments_to_analyze,
                           seg_decode_frac_post, seg_decode_counts_post, 
                           seg_names, post_save_dir)
    
    #Create individual decode plots
    pddf.plot_decoded(tastant_fr_dist, tastant_spike_times, segment_spike_times, 
                      dig_in_names, segment_times, segment_names, start_dig_in_times, 
                      taste_num_deliv, segment_dev_times, dev_vecs, bin_dt, 
                      num_groups, grouped_train_names, grouped_train_data, 
                      non_none_tastes, decode_save_dir, 
                      z_score, segments_to_analyze = segments_to_analyze)
    
def decode_sliding_bins(tastant_fr_dist, segment_spike_times, dig_in_names, 
                  segment_times, segment_names, start_dig_in_times, taste_num_deliv,
                  bin_dt, group_list, group_names, non_none_tastes, decode_dir, 
                  z_score = False, segments_to_analyze=[]):
    
    print('\t\tRunning Sliding Bin NB Decoder')
    decode_save_dir = os.path.join(decode_dir,'Sliding_Decoding')
    if not os.path.isdir(decode_save_dir):
        os.mkdir(decode_save_dir)
    
    # Variables
    num_tastes = len(start_dig_in_times)
    num_neur = len(segment_spike_times[0])
    num_segments = len(segment_spike_times)
    if not group_names[-1] == 'Null Data':
        num_groups = len(group_names) + 1
        group_names.append('Null Data')
    else:
        num_groups = len(group_names)
    cmap = colormaps['jet']
    group_colors = cmap(np.linspace(0, 1, num_groups))
    dev_buffer = 50
    
    #Bin size for sliding bin decoding
    half_bin = 25
    bin_size = half_bin*2
    
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
        
    seg_names = list(np.array(segment_names)[segments_to_analyze])
        
    #Grab all taste-epoch pairs in the training groups for testing
    taste_epoch_pairs = []
    for gl_i, gl in enumerate(group_list):
        for gp_i, gp in enumerate(gl):
            taste_epoch_pairs.append([gp[1],gp[0]])
    
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
    num_groups = len(grouped_train_names)
    
    #Create categorical NB dataset
    categorical_train_data = []
    categorical_train_y = []
    for g_i, g_data in enumerate(grouped_train_data):
        categorical_train_data.extend(g_data)
        categorical_train_y.extend(g_i*np.ones(len(g_data)))
    
    #Run PCA transform only on non-z-scored data
    if z_score == True:
        train_data = categorical_train_data
    else:
        pca_reduce_taste = train_taste_PCA(num_neur,taste_epoch_pairs,\
                                           non_none_tastes,dig_in_names,
                                           taste_num_deliv,tastant_fr_dist)
        train_data = pca_reduce_taste.transform(categorical_train_data)
   
    #Fit NB
    nb = GaussianNB()
    nb.fit(train_data, categorical_train_y) 
    
    #Store segment by segment correlation between pop rate and decode group
    seg_group_rate_corr = np.nan*np.ones((len(segments_to_analyze),num_groups))
    
    #Store segment by segment fraction of decodes
    seg_group_counts = np.nan*np.ones((len(segments_to_analyze),num_groups))
    seg_group_frac = np.nan*np.ones((len(segments_to_analyze),num_groups))
    
    seg_lengths = []
        
    #Run through each segment and decode bins of activity
    for seg_ind, s_i in enumerate(segments_to_analyze):
        # Create segment save dir
        seg_decode_save_dir = os.path.join(decode_save_dir,
            'segment_' + str(s_i) + '/')
        if not os.path.isdir(seg_decode_save_dir):
            os.mkdir(seg_decode_save_dir)
            
        # Get segment variables
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = seg_end - seg_start# in dt = ms
        seg_lengths.append(seg_len/1000)
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
            seg_decode_argmax_array = np.load(
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
            seg_decode_argmax_array = nb.predict(seg_fr_list)
            np.save(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                                 '_sliding_group.npy'),seg_decode_argmax_array)
            
            toc = time.time()
            print('\t\t\t\t\tTime to decode = ' +
                  str(np.round((toc-tic)/60, 2)) + ' (min)')
        
        #Save decode fractions
        hist_vals = np.histogram(seg_decode_argmax_array,bins=np.arange(num_groups+1))
        seg_group_counts[seg_ind,:] = hist_vals[0]
        seg_group_frac[seg_ind,:] = hist_vals[0]/np.nansum(hist_vals[0])
        
        #Save correlation to pop rate
        for g_ind in range(num_groups):
            ga_i = np.where(seg_decode_argmax_array == g_ind)[0]
            bin_group = np.zeros(num_bin)
            bin_group[ga_i] = 1
            pcorr = pearsonr(bin_group,segment_binned_pop_fr)
            seg_group_rate_corr[seg_ind,g_ind] = pcorr[0]
            
        #Save a histogram of group decodes
        f = plt.figure(figsize=(5,5))
        hist_vals = np.histogram(seg_decode_argmax_array,bins=np.arange(num_groups+1))
        plt.bar(np.arange(num_groups),hist_vals[0]/(seg_len/1000))
        plt.xticks(np.arange(num_groups),grouped_train_names,rotation=45)
        plt.ylabel('Bin Decode Rate (Hz)')
        plt.title('Segment ' + str(s_i) + ' group decode counts')
        plt.tight_layout()
        f.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                             '_sliding_group_decode_hist.svg'))
        f.savefig(os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                             '_sliding_group_decode_hist.png'))
        plt.close(f)
                    
    #Save cross-segment data
    np.save(os.path.join(decode_save_dir,'seg_group_rate_corr.npy'), seg_group_rate_corr, allow_pickle=True) 
    np.save(os.path.join(decode_save_dir,'seg_group_frac.npy'), seg_group_frac, allow_pickle=True) 
    
    #PLOTS
    #Plot decode trends
    pddf.plot_decode_trends(num_groups, grouped_train_names, segments_to_analyze,
                           seg_group_frac, seg_group_counts, seg_names, decode_save_dir)
    
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
    f_box_corr.savefig(os.path.join(decode_save_dir,'seg_group_rate_corr.png'))
    f_box_corr.savefig(os.path.join(decode_save_dir,'seg_group_rate_corr.svg'))
    plt.close(f_box_corr)

def decoder_accuracy_tests(tastant_fr_dist, segment_spike_times, 
                dig_in_names, segment_times, segment_names, start_dig_in_times, 
                taste_num_deliv, group_list, group_names, non_none_tastes, 
                decode_dir, bin_dt, z_score = False, 
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
    group_list : list
        List of tuples describing group membership through (epoch, taste) indices.
    group_names : list
        List of group names.
    non_none_tastes : list
        Taste names that are not none.
    decode_dir : string
        String containing directory to save results to.
    z_score : boolean, optional
        Whether the data is z-scored or not. The default is False.
    epochs_to_analyze : list, optional
        List of which epoch indices to analyze. The default is [].
    segments_to_analyze : list, optional
        List of which segment indices to analyze. The default is [].

    Returns
    -------
    Outputs plots and accuracy results.
    """
    
    decode_accuracy_save_dir = os.path.join(decode_dir,'Decoder_Accuracy')
    if not os.path.isdir(decode_accuracy_save_dir):
        os.mkdir(decode_accuracy_save_dir)
        
    # Variables
    num_tastes = len(dig_in_names)
    num_neur = len(segment_spike_times[0])
    num_segments = len(segment_spike_times)
    if not group_names[-1] == 'Null Data':
        num_groups = len(group_names) + 1
        group_names.append('Null Data')
    else:
        num_groups = len(group_names)
    epochs_to_analyze = np.array([0,1,2])
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
    num_cp = len(epochs_to_analyze)
    cmap = colormaps['gist_rainbow']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    cmap = colormaps['jet']
    group_colors = cmap(np.linspace(0, 1, num_groups))
        
    #Create null dataset from shuffled rest spikes
    shuffled_fr_vecs, segment_spike_times_bin, \
        seg_means, seg_stds = create_null_decode_dataset(segments_to_analyze, \
                                    segment_times, segment_spike_times, \
                                    num_neur, bin_dt, z_score)
    
    #Grab all taste-epoch pairs in the training groups for testing
    taste_epoch_pairs = []
    for gl_i, gl in enumerate(group_list):
        for gp_i, gp in enumerate(gl):
            taste_epoch_pairs.append([gp[1],gp[0]])
    
    #Create a LOO list
    total_num_deliv = int(np.sum(taste_num_deliv))
    loo_taste_index = []
    loo_deliv_index = []
    loo_epoch_index = []
    loo_category = []
    for t_i, e_i in taste_epoch_pairs:  # Only perform on actual tastes
        for d_i in range(int(taste_num_deliv[t_i])):
            loo_taste_index.append(t_i)
            loo_deliv_index.append(d_i)
            loo_epoch_index.append(e_i)
            test_group_ind = -1
            for g_i, g_list in enumerate(group_list):
                for g_e, g_t in g_list:
                    if (g_e == e_i) and (g_t == t_i):
                        test_group_ind = g_i
            loo_category.append(test_group_ind)
    del t_i, d_i, e_i, g_i
    loo_category = np.array(loo_category)
    total_loo = len(loo_taste_index)
    
    try:
        nb_decoder_accuracy_dict = np.load(os.path.join(decode_accuracy_save_dir,'nb_decoder_accuracy_dict.npy'),\
                allow_pickle=True).item()
        nb_decode_predictions = np.load(os.path.join(decode_accuracy_save_dir,'nb_decode_predictions.npy'),\
                allow_pickle=True).item()
    except:
        #Store taste x delivery x epoch decoding success rates
        nb_decoder_accuracy_dict = dict()
        for t_i, e_i in taste_epoch_pairs:
            nb_decoder_accuracy_dict[str(t_i) + ',' + str(e_i)] = np.zeros(int(taste_num_deliv[t_i]))
        nb_decode_predictions = dict()
        for t_i, e_i in taste_epoch_pairs:
            nb_decode_predictions[str(t_i) + ',' + str(e_i)] = np.zeros((int(taste_num_deliv[t_i]),num_groups))
        
        tic = time.time()
        for loo_i in tqdm.tqdm(range(total_loo)):
            loo_t_i = loo_taste_index[loo_i]
            loo_d_i = loo_deliv_index[loo_i]
            loo_e_i = loo_epoch_index[loo_i]
            loo_g_i = loo_category[loo_i]
            test_data = [np.squeeze(tastant_fr_dist[loo_t_i][loo_d_i][loo_e_i])]
            
            #Create training groups of firing rate vectors
            grouped_train_data = [] #Using group_list above, create training groups
            grouped_train_counts = [] #Number of values in the group
            grouped_train_names = []
            for g_i, g_list in enumerate(group_list):
                group_data_collection = []
                for (e_i,t_i) in g_list:
                    for d_i in range(int(taste_num_deliv[t_i])):
                        #Skip the loo value
                        if (e_i == loo_e_i) and (t_i == loo_t_i) and (d_i == loo_d_i):
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
            # group_prob = np.array(grouped_train_counts) / np.sum(np.array(grouped_train_counts)) 
            num_groups = len(grouped_train_names)
            
            #Create categorical NB dataset
            categorical_train_data = []
            categorical_train_y = []
            for g_i, g_data in enumerate(grouped_train_data):
                categorical_train_data.extend(g_data)
                categorical_train_y.extend(g_i*np.ones(len(g_data)))
            categorical_train_data = np.array(categorical_train_data)
            categorical_train_y = np.array(categorical_train_y)
            
            #Plot reduced data if loo_i == 0
            if loo_i == 0:
                #PCA
                pca_reduce_plot = PCA(2)
                pca_data = pca_reduce_plot.fit_transform(np.array(categorical_train_data))
                f_pca, ax_pca = plt.subplots(nrows = 1, ncols = 2, 
                                             sharex = True, sharey = True,
                                             figsize=(10,5))
                for g_i, g_name in enumerate(grouped_train_names):
                    group_where = np.where(categorical_train_y == g_i)[0]
                    ax_pca[0].scatter(pca_data[group_where,0],pca_data[group_where,1],\
                                color = group_colors[g_i,:],\
                                alpha=0.5,label=g_name)
                ax_pca[0].set_title('Individual Responses')
                for g_i, g_name in enumerate(grouped_train_names):
                    group_where = np.where(categorical_train_y == g_i)[0]
                    mean_x = np.nanmean(pca_data[group_where,0])
                    x_std = np.nanstd(pca_data[group_where,0])
                    mean_y = np.nanmean(pca_data[group_where,1])
                    y_std = np.nanstd(pca_data[group_where,1])
                    m, c = np.polyfit(pca_data[group_where,0], pca_data[group_where,1], 1)
                    angle = np.degrees(np.arctan(m))
                    oval = patches.Ellipse(
                            (mean_x, mean_y),
                            2*x_std,
                            2*y_std,
                            angle,
                            facecolor=group_colors[g_i,:],
                            alpha=0.3
                            )
                    ax_pca[1].add_patch(oval)
                    ax_pca[1].scatter(mean_x,mean_y,\
                                color = group_colors[g_i,:],\
                                alpha=1,label=g_name)
                ax_pca[1].set_title('Average Locations')
                ax_pca[1].legend(loc='lower right')
                plt.suptitle('PCA Projection of Training Groups')
                plt.tight_layout()
                f_pca.savefig(os.path.join(decode_accuracy_save_dir,'pca_data_distribution.png'))
                f_pca.savefig(os.path.join(decode_accuracy_save_dir,'pca_data_distribution.svg'))
                plt.close(f_pca)
            
            #Run PCA transform only on non-z-scored data
            if z_score == True:
                train_data = categorical_train_data
            else:
                pca_reduce_taste = train_taste_PCA(num_neur,taste_epoch_pairs,\
                                                   non_none_tastes,dig_in_names,
                                                   taste_num_deliv,tastant_fr_dist)
                train_data = pca_reduce_taste.transform(categorical_train_data)
                
            #Fit NB
            nb = GaussianNB()
            nb.fit(train_data, categorical_train_y)
            
            #Run decoder on the taste response
            #NB
            nb_prob = nb.predict_proba(test_data)
            nb_argmax = int(np.argmax(nb_prob))
            nb_decode_predictions[str(loo_t_i) + ',' + str(loo_e_i)][loo_d_i,:] = nb_prob
            if loo_t_i == num_tastes-1: #No taste control
                if (nb_argmax == loo_g_i) or (nb_argmax == num_groups-1):
                    nb_decoder_accuracy_dict[str(loo_t_i) + ',' + str(loo_e_i)][loo_d_i] = 1
            else:
                if nb_argmax == loo_g_i: #Correct group decoded
                    nb_decoder_accuracy_dict[str(loo_t_i) + ',' + str(loo_e_i)][loo_d_i] = 1                
                
        toc = time.time()
        print('\tTime to run decode accuracy tests = ' +
                      str(np.round((toc-tic)/60, 2)) + ' (min)')
        np.save(os.path.join(decode_accuracy_save_dir,'nb_decoder_accuracy_dict.npy'),\
                nb_decoder_accuracy_dict,allow_pickle=True)
        np.save(os.path.join(decode_accuracy_save_dir,'nb_decode_predictions.npy'),\
                nb_decode_predictions,allow_pickle=True)
    
    #PLOTS
    chance_rate = 100/num_groups
    
    #Overall Taste Accuracy
    nb_decode_success_counts_taste = np.zeros(num_tastes)
    nb_decode_total_counts_taste = np.zeros(num_tastes)
    for t_i, e_i in taste_epoch_pairs:
        nb_decode_success_counts_taste[t_i] += np.sum(nb_decoder_accuracy_dict[str(t_i) + ',' + str(e_i)])
        nb_decode_total_counts_taste[t_i] += taste_num_deliv[t_i]
    nb_decode_success_rates_taste = 100*(nb_decode_success_counts_taste/nb_decode_total_counts_taste)
    f_bar = plt.figure(figsize=(5,5))
    plt.axhline(chance_rate,linestyle='dashed',alpha=0.2,color='k')
    plt.bar(np.arange(num_tastes),nb_decode_success_rates_taste)
    plt.xticks(np.arange(num_tastes),dig_in_names,rotation=45)
    plt.ylim([0,100])
    plt.xlabel('Taste')
    plt.ylabel('Percent Accurately Decoded')
    plt.title('Overall Taste Accuracy')
    plt.tight_layout()
    f_bar.savefig(os.path.join(decode_accuracy_save_dir,'nb_overall_taste_accuracy.png'))
    f_bar.savefig(os.path.join(decode_accuracy_save_dir,'nb_overall_taste_accuracy.svg'))
    plt.close(f_bar)
    
    #Accuracy by Epoch
    unique_tastes = np.unique([t_i for t_i, e_i in taste_epoch_pairs])
    nb_decode_success_counts_epoch = np.zeros((len(unique_tastes),len(epochs_to_analyze)))
    nb_decode_total_counts_epoch = np.zeros((len(unique_tastes),len(epochs_to_analyze)))
    for t_i, e_i in taste_epoch_pairs:
        t_i_ind = np.where(unique_tastes == t_i)[0]
        nb_decode_success_counts_epoch[t_i_ind,e_i] += np.sum(nb_decoder_accuracy_dict[str(t_i) + ',' + str(e_i)])
        nb_decode_total_counts_epoch[t_i_ind,e_i] += taste_num_deliv[t_i]
    nb_decode_success_rates_epoch = 100*(nb_decode_success_counts_epoch/nb_decode_total_counts_epoch)
    nb_decode_success_rates_epoch = np.array(nb_decode_success_rates_epoch)
    f_epoch = plt.figure(figsize=(5,5))
    plt.axhline(0.8,linestyle='dashed',alpha=0.2,color='b')
    plt.axhline(chance_rate,linestyle='dashed',alpha=0.2,color='k')
    for t_i in unique_tastes:
        t_i_ind = np.where(unique_tastes == t_i)[0]
        t_name = dig_in_names[t_i]
        plt.scatter(np.arange(len(epochs_to_analyze)) + 0.05*t_i, nb_decode_success_rates_epoch[t_i_ind,:],label=t_name,
                        color=taste_colors[t_i,:])
    plt.legend(loc='upper left')
    plt.xticks(np.arange(num_cp),epochs_to_analyze)
    plt.ylim([0,100])
    plt.xlabel('Epoch')
    plt.ylabel('Percent Accurately Decoded')
    plt.title('By Epoch Taste Accuracy')
    plt.tight_layout()
    f_epoch.savefig(os.path.join(decode_accuracy_save_dir,'nb_by_epoch_taste_accuracy.png'))
    f_epoch.savefig(os.path.join(decode_accuracy_save_dir,'nb_by_epoch_taste_accuracy.svg'))
    plt.close(f_epoch)
    
    #Accuracy breakdown by group
    grid_side = np.ceil(np.sqrt(num_groups)).astype('int')
    grid_inds = np.reshape(np.arange(grid_side**2),(grid_side,grid_side))
    f_breakdown_nb, ax_breakdown_nb = plt.subplots(nrows = grid_side, \
                                                   ncols = grid_side, \
                                                   sharey = True, \
                                                   figsize=(10,10))
    for gn_i, gn in enumerate(group_names[:-1]):
        r_i, c_i = np.where(grid_inds == gn_i)
        loo_category_inds = np.where(loo_category == gn_i)[0]
        nb_loo_category_classifications = []
        for loo_i in loo_category_inds:
            loo_t_i = loo_taste_index[loo_i]
            loo_d_i = loo_deliv_index[loo_i]
            loo_e_i = loo_epoch_index[loo_i]
            nb_loo_category_classifications.append(np.argmax(nb_decode_predictions[str(loo_t_i) + ',' + str(loo_e_i)][loo_d_i,:]))
        hist_vals = np.histogram(nb_loo_category_classifications,bins = np.arange(num_groups+1))
        ax_breakdown_nb[r_i[0],c_i[0]].bar(np.arange(num_groups),100*hist_vals[0]/np.nansum(hist_vals[0]))
        ax_breakdown_nb[r_i[0],c_i[0]].set_xticks(np.arange(num_groups)+0.5,\
                                                  group_names,\
                                                      ha="right", rotation=45)
        ax_breakdown_nb[r_i[0],c_i[0]].set_title(gn)
    for gn_i in np.setdiff1d(np.arange(grid_side**2).astype('int'),np.arange(num_groups-1).astype('int')):
        r_i, c_i = np.where(grid_inds == gn_i)
        ax_breakdown_nb[r_i[0],c_i[0]].remove()
    ax_breakdown_nb[0,0].set_ylim([0,100])
    ax_breakdown_nb[0,0].set_ylabel('Percent Decoded')
    plt.figure(f_breakdown_nb)
    plt.suptitle('NB by-group breakdown')
    plt.tight_layout()
    f_breakdown_nb.savefig(os.path.join(decode_accuracy_save_dir,'nb_by_group_decodes.png'))
    f_breakdown_nb.savefig(os.path.join(decode_accuracy_save_dir,'nb_by_group_decodes.svg'))
    plt.close(f_breakdown_nb)
    
    #Probability breakdown by group
    grid_side = np.ceil(np.sqrt(num_groups)).astype('int')
    grid_inds = np.reshape(np.arange(grid_side**2),(grid_side,grid_side))
    f_breakdown_nb, ax_breakdown_nb = plt.subplots(nrows = grid_side, \
                                                   ncols = grid_side, \
                                                   sharey = True, \
                                                   figsize=(10,10))
    for gn_i, gn in enumerate(group_names[:-1]):
        r_i, c_i = np.where(grid_inds == gn_i)
        loo_category_inds = np.where(loo_category == gn_i)[0]
        nb_loo_category_probabilities = []
        for loo_i in loo_category_inds:
            loo_t_i = loo_taste_index[loo_i]
            loo_d_i = loo_deliv_index[loo_i]
            loo_e_i = loo_epoch_index[loo_i]
            nb_loo_category_probabilities.append(nb_decode_predictions[str(loo_t_i) + ',' + str(loo_e_i)][loo_d_i,:])
        nb_loo_category_probabilities = np.array(nb_loo_category_probabilities)
        mean_probabilities = np.nanmean(nb_loo_category_probabilities,0)
        for cat_i in range(num_groups):
            ax_breakdown_nb[r_i[0],c_i[0]].scatter(cat_i*np.ones(len(loo_category_inds)),\
                                                   nb_loo_category_probabilities[:,cat_i],\
                                                       alpha=0.2,color='g')
            ax_breakdown_nb[r_i[0],c_i[0]].scatter(cat_i,mean_probabilities[cat_i],\
                                                   alpha=1,color='g',s=80)
        ax_breakdown_nb[r_i[0],c_i[0]].axhline(1/num_groups,alpha=0.2,color='k',linestyle='dashed')
        ax_breakdown_nb[r_i[0],c_i[0]].set_xticks(np.arange(num_groups),\
                                                  group_names,\
                                                  ha="right", rotation=45)
        ax_breakdown_nb[r_i[0],c_i[0]].set_title(gn)
    for gn_i in np.setdiff1d(np.arange(grid_side**2).astype('int'),np.arange(num_groups-1).astype('int')):
        r_i, c_i = np.where(grid_inds == gn_i)
        ax_breakdown_nb[r_i[0],c_i[0]].remove()
    ax_breakdown_nb[0,0].set_ylim([0,1])
    ax_breakdown_nb[0,0].set_ylabel('Probability Decoded')
    plt.figure(f_breakdown_nb)
    plt.suptitle('NB by-group decode probabilities')
    plt.tight_layout()
    f_breakdown_nb.savefig(os.path.join(decode_accuracy_save_dir,'nb_by_group_prob_decodes.png'))
    f_breakdown_nb.savefig(os.path.join(decode_accuracy_save_dir,'nb_by_group_prob_decodes.svg'))
    plt.close(f_breakdown_nb)
    
    f_breakdown_nb_combined = plt.figure(figsize=(10,10))
    nb_loo_category_classifications = []
    for gn_i, gn in enumerate(group_names[:-1]):
        r_i, c_i = np.where(grid_inds == gn_i)
        loo_category_inds = np.where(loo_category == gn_i)[0]
        for loo_i in loo_category_inds:
            loo_t_i = loo_taste_index[loo_i]
            loo_d_i = loo_deliv_index[loo_i]
            loo_e_i = loo_epoch_index[loo_i]
            nb_loo_category_classifications.append(np.argmax(nb_decode_predictions[str(loo_t_i) + ',' + str(loo_e_i)][loo_d_i,:]))
    hist_vals = np.histogram(nb_loo_category_classifications,bins = np.arange(num_groups+1))
    plt.bar(np.arange(num_groups),100*hist_vals[0]/np.nansum(hist_vals[0]))
    plt.xticks(np.arange(num_groups)+0.5,group_names,ha="right",rotation=45)
    plt.title('Overall Decoded Category Rates')
    plt.ylim([0,100])
    plt.ylabel('Percent Decoded')
    plt.tight_layout()
    f_breakdown_nb_combined.savefig(os.path.join(decode_accuracy_save_dir,'nb_all_group_decode_rates.png'))
    f_breakdown_nb_combined.savefig(os.path.join(decode_accuracy_save_dir,'nb_all_group_decode_rates.svg'))
    plt.close(f_breakdown_nb_combined)
    
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

def decode_groupings(epochs_to_analyze,dig_in_names,non_none_tastes):
    """
    Create fr vector grouping instructions: list of epoch,taste pairs.
    
    Parameters
    ----------
    epochs_to_analyze: list
        list of indices of epochs to be analyzed.
    dig_in_names: list
        list of strings of taste names
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
    group_names = []
    none_group = []
    palatability_group = []
    for e_ind, e_i in enumerate(epochs_to_analyze):
        epoch_group = []
        epoch_names = []
        for t_ind, t_name in enumerate(dig_in_names):
            if np.setdiff1d([t_name],non_none_tastes).size > 0: #None
                none_group.append((e_i,t_ind))
            else:
                #Combine presence data
                if e_i == 0:
                    epoch_group.append((e_i, t_ind))
                    epoch_names.append((e_i, t_name))
                #Separate identity data
                if e_i == 1:
                    group_list.append([(e_i, t_ind)])
                    group_list_names.append([(e_i,t_name)])
                    group_names.append(t_name.capitalize() + ' Identity')
                #Separate palatability data
                if e_i == 2:
                    palatability_group.append((e_i,t_ind))
        if len(epoch_group) > 0:
            group_list.append(epoch_group)
            group_list_names.append([(e_i,'all')])
            group_names.append('Presence')
    group_list.append(palatability_group)
    group_list_names.append(['Palatability'])
    group_names.append('Palatability')
    group_list.append(none_group)
    group_list_names.append(['None'])
    group_names.append('No Taste Control')
    
    final_group_list = []
    final_group_names = []
    for g_i in range(len(group_list)):
        if len(group_list[g_i]) > 0:
            final_group_list.append(group_list[g_i])
            final_group_names.append(group_names[g_i])
    
    return final_group_list, final_group_names

def multiday_decode_groupings(epochs_to_analyze,all_dig_in_names,non_none_tastes):
    """
    Create fr vector grouping instructions: list of epoch,taste pairs.
    
    Parameters
    ----------
    epochs_to_analyze: list
        list of indices of epochs to be analyzed.
    all_dig_in_names: list
        list of strings of taste names across days
    non_none_tastes: list of taste names that are not "none"
    
    Returns
    -------
    group_list: list
        list of lists containing tuples (e_i,t_i) of which epoch and taste index
        belongs to a decoding group.
    group_names: list
        list of strings naming the decoding groups.
    """
    #Pull out unique taste names
    dig_in_first_names = [dn.split('_')[0] for dn in all_dig_in_names]
    unique_dig_in_names = list(np.unique(dig_in_first_names))
    dig_in_day_1_names = [dn.split('_')[0] for dn in all_dig_in_names if int(dn.split('_')[1]) == 0]
    dig_in_later_day_unique = list(np.setdiff1d(unique_dig_in_names,dig_in_day_1_names))
    
    group_list = []
    group_list_names = []
    group_names = []
    none_group = []
    identity_group = []
    palatability_group = []
    
    for e_ind, e_i in enumerate(epochs_to_analyze):
        epoch_group = []
        epoch_names = []
        for t_ind, t_name in enumerate(all_dig_in_names):
            if int(t_name.split('_')[1]) == 0: #Day 1 data
                if np.setdiff1d([t_name],non_none_tastes).size > 0: #None
                    none_group.append((e_i,t_ind))
                else:
                    #Combine presence data
                    if e_i == 0:
                        epoch_group.append((e_i, t_ind))
                        epoch_names.append((e_i, t_name))
                    #Separate identity data
                    if e_i == 1:
                        identity_group.append((e_i, t_ind))
                    #Separate palatability data
                    if e_i == 2:
                        palatability_group.append((e_i,t_ind))
            else: #Day 2+ taste
                if np.intersect1d(dig_in_later_day_unique,t_name.split('_')[0]).size > 0: #Unique to next day taste
                    if e_i == 1:
                        identity_group.append((e_i, t_ind))
        if len(epoch_group) > 0:
            group_list.append(epoch_group)
            group_list_names.append([(e_i,'all')])
            group_names.append('Presence')
    group_list.append(identity_group)
    group_list_names.append(['Identity'])
    group_names.append('Identity')
    group_list.append(palatability_group)
    group_list_names.append(['Palatability'])
    group_names.append('Palatability')
    group_list.append(none_group)
    group_list_names.append(['None'])
    group_names.append('No Taste Control')
    
    final_group_list = []
    final_group_names = []
    for g_i in range(len(group_list)):
        if len(group_list[g_i]) > 0:
            final_group_list.append(group_list[g_i])
            final_group_names.append(group_names[g_i])
    
    return final_group_list, final_group_names

def multiday_decode_groupings_split_identity(epochs_to_analyze,all_dig_in_names,non_none_tastes):
    """
    Create fr vector grouping instructions: list of epoch,taste pairs.
    
    Parameters
    ----------
    epochs_to_analyze: list
        list of indices of epochs to be analyzed.
    all_dig_in_names: list
        list of strings of taste names across days
    non_none_tastes: list of taste names that are not "none"
    
    Returns
    -------
    group_list: list
        list of lists containing tuples (e_i,t_i) of which epoch and taste index
        belongs to a decoding group.
    group_names: list
        list of strings naming the decoding groups.
    """
    #Pull out unique taste names
    dig_in_first_names = [dn.split('_')[0] for dn in all_dig_in_names]
    unique_dig_in_names = list(np.unique(dig_in_first_names))
    dig_in_day_1_names = [dn.split('_')[0] for dn in all_dig_in_names if int(dn.split('_')[1]) == 0]
    dig_in_later_day_unique = list(np.setdiff1d(unique_dig_in_names,dig_in_day_1_names))
    
    group_list = []
    group_list_names = []
    group_names = []
    none_group = []
    identity_group = []
    palatability_group = []
    
    for e_ind, e_i in enumerate(epochs_to_analyze):
        epoch_group = []
        epoch_names = []
        for t_ind, t_name in enumerate(all_dig_in_names):
            if int(t_name.split('_')[1]) == 0: #Day 1 data
                if np.setdiff1d([t_name],non_none_tastes).size > 0: #None
                    none_group.append((e_i,t_ind))
                else:
                    #Combine presence data
                    if e_i == 0:
                        epoch_group.append((e_i, t_ind))
                        epoch_names.append((e_i, t_name))
                    #Separate identity data
                    if e_i == 1:
                        group_list.append([(e_i, t_ind)])
                        group_list_names.append([(e_i,t_name)])
                        group_names.append(t_name.capitalize() + ' Identity')
                    #Separate palatability data
                    if e_i == 2:
                        palatability_group.append((e_i,t_ind))
            else: #Day 2+ taste
                if np.intersect1d(dig_in_later_day_unique,t_name.split('_')[0]).size > 0: #Unique to next day taste
                    if e_i == 1:
                        group_list.append([(e_i, t_ind)])
                        group_list_names.append([(e_i,t_name)])
                        group_names.append(t_name.capitalize() + ' Identity')
        if len(epoch_group) > 0:
            group_list.append(epoch_group)
            group_list_names.append([(e_i,'all')])
            group_names.append('Presence')
    group_list.append(identity_group)
    group_list_names.append(['Identity'])
    group_names.append('Identity')
    group_list.append(palatability_group)
    group_list_names.append(['Palatability'])
    group_names.append('Palatability')
    group_list.append(none_group)
    group_list_names.append(['None'])
    group_names.append('No Taste Control')
    
    final_group_list = []
    final_group_names = []
    for g_i in range(len(group_list)):
        if len(group_list[g_i]) > 0:
            final_group_list.append(group_list[g_i])
            final_group_names.append(group_names[g_i])
    
    return final_group_list, final_group_names

def train_taste_PCA(num_neur,taste_epoch_pairs,non_none_tastes,dig_in_names,
                    taste_num_deliv,tastant_fr_dist):
    """
    Train a PCA reducer on taste response data.

    Parameters
    ----------
    taste_epoch_pairs : list
        list of [taste,epoch] pair lists to analyze.
    num_neur : int
        number of neurons to analyze.
    non_none_tastes: list
        list of taste names that are not a "none" taste
    dig_in_names: list
        list of all taste names to reference true taste indices
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
    for t_i, e_i in taste_epoch_pairs:
        t_name = dig_in_names[t_i]
        if np.intersect1d(non_none_tastes,[t_name]).size > 0:
            for d_i in range(int(taste_num_deliv[t_i])):
                try:
                    if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                        true_taste_train_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i].T))
                    else:
                        true_taste_train_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i]))
                except:
                    true_taste_train_data.extend([])
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

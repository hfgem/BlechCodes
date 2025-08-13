#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:55:59 2025

@author: hannahgermaine

File dedicated to functions related to LSTM decoding of tastes where responses
are timeseries of firing rates.
"""

import os
import tqdm
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from matplotlib import colormaps

def create_taste_matrices(num_neur, tastant_spike_times, segment_spike_times,
                         segment_names, segment_times, cp_raster_inds, fr_bins,
                         start_dig_in_times, pre_taste_dt, post_taste_dt, 
                         all_dig_in_names, num_bins, z_bin_dt, start_bins=0):
    """Function to take spike times following taste delivery and create 
    matrices of timeseries firing trajectories"""
    
    num_tastes = len(tastant_spike_times)
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    bin_starts = np.linspace(start_bins,post_taste_dt,num_bins+1)
    half_z_bin = np.floor(z_bin_dt/2).astype('int')
    max_num_deliv = 0  # Find the maximum number of deliveries across tastants
    for t_i in range(num_tastes):
        num_deliv = len(tastant_spike_times[t_i])
        if num_deliv > max_num_deliv:
            max_num_deliv = num_deliv
    del t_i, num_deliv
    
    #Create storage matrices and outputs
    taste_unique_categories = list(all_dig_in_names)
    taste_unique_categories.append('Null')
    taste_matrices = []
    taste_labels = []
    null_matrices = []
    null_labels = []
    #Get taste segment z-score info
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
            seg_start+half_z_bin, seg_end-half_z_bin, bin_dt)
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = (np.array(
                segment_spike_times_s_i[n_i]) - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        tb_fr = np.zeros((num_neur, len(time_bin_starts)))
        for tb_i, tb in enumerate(time_bin_starts):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_z_bin:tb+half_z_bin-seg_start], 1)/(2*half_z_bin*(1/1000))
        mean_fr = np.mean(tb_fr, 1)
        std_fr = np.std(tb_fr, 1)
    else:
        mean_fr = np.zeros(num_neur)
        std_fr = np.zeros(num_neur)
    
    #Generate response matrices
    for t_i in range(num_tastes):
        num_deliv = (taste_num_deliv[t_i]).astype('int')
        taste_cp = cp_raster_inds[t_i]
        for d_i in range(num_deliv):  # index for that taste
            if d_i >= trial_start_ind:
                raster_times = tastant_spike_times[t_i][d_i]
                start_taste_i = start_dig_in_times[t_i][d_i]
                deliv_cp = taste_cp[d_i, :] - pre_taste_dt
                # Binerize the activity following taste delivery start
                times_post_taste = [(np.array(raster_times[n_i])[np.where((raster_times[n_i] >= start_taste_i)*(
                    raster_times[n_i] < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int') for n_i in range(num_neur)]
                bin_post_taste = np.zeros((num_neur, post_taste_dt))
                for n_i in range(num_neur):
                    bin_post_taste[n_i, times_post_taste[n_i]] += 1
                #Calculate binned firing rate matrix
                fr_mat = np.zeros((num_neur,num_bins))
                for bin_i in range(num_bins):
                    bs_i = bin_starts[bin_i]
                    be_i = bin_starts[bin_i+1]
                    b_len = (be_i - bs_i)/1000
                    fr_mat[:,bin_i] = np.sum(bin_post_taste[:,bs_i:be_i],1)/b_len
                #Convert to z-scored matrix
                fr_z_mat = (fr_mat - np.expand_dims(mean_fr,1))/np.expand_dims(std_fr,1)
                taste_matrices.append(fr_z_mat)
                taste_labels.append(t_i)
                #Generate null taste response matrix - same firing count across interval, but shuffled times
                rand_times_post_taste = [np.random.randint(0,post_taste_dt,len(tpt)) for tpt in times_post_taste]
                rand_bin_post_taste = np.zeros((num_neur, post_taste_dt))
                for n_i in range(num_neur):
                    rand_bin_post_taste[n_i, rand_times_post_taste[n_i]] += 1
                rand_fr_mat = np.zeros((num_neur,num_bins))
                for bin_i in range(num_bins):
                    bs_i = bin_starts[bin_i]
                    be_i = bin_starts[bin_i+1]
                    b_len = (be_i - bs_i)/1000
                    fr_mat[:,bin_i] = np.sum(bin_post_taste[:,bs_i:be_i],1)/b_len
                rand_fr_z_mat = (fr_mat - np.expand_dims(mean_fr,1))/np.expand_dims(std_fr,1)
                null_matrices.append(fr_z_mat)
                null_labels.append(len(taste_unique_categories)-1)
    
    #Sample null dataset to a subset that match the true taste delivery counts
    #Combine into one dataset
    rand_null_ind = np.random.randint(0,len(null_matrices),max_num_deliv)
    training_matrices = []
    training_labels = []
    training_matrices.extend(taste_matrices)
    training_labels.extend(taste_labels)
    training_matrices.extend([null_matrices[rni] for rni in rand_null_ind])
    training_labels.extend([null_labels[rni] for rni in rand_null_ind])
    
    return taste_unique_categories, training_matrices, training_labels
    
def create_dev_matrices(segment_spike_times, segments_to_analyze, 
                        start_end_times, deviations, z_bin_dt, num_bins):
    """Function to take spike times during deviation events and create 
    matrices of timeseries firing trajectories the same size as taste trajectories"""
    
    half_z_bin = np.floor(z_bin_dt/2).astype('int')
    dev_matrices = []
    
    for s_ind, s_i in enumerate(segments_to_analyze):
        seg_dev_matrices = []
        
        seg_spikes = spike_times[s_i]
        seg_start = int(start_end_times[s_i][0])
        seg_end = int(start_end_times[s_i][1])
        seg_len = seg_end - seg_start
        num_neur = len(seg_spikes)
        spikes_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            neur_spikes = np.array(seg_spikes[n_i]).astype(
                'int') - seg_start
            spikes_bin[n_i, neur_spikes] = 1
        # Calculate z-score mean and std
        time_bin_starts = np.arange(
            seg_start+half_z_bin, seg_end-half_z_bin, bin_dt)
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = (np.array(
                segment_spike_times_s_i[n_i]) - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        tb_fr = np.zeros((num_neur, len(time_bin_starts)))
        for tb_i, tb in enumerate(time_bin_starts):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_z_bin:tb+half_z_bin-seg_start], 1)/(2*half_z_bin*(1/1000))
        mean_fr = np.mean(tb_fr, 1)
        std_fr = np.std(tb_fr, 1)
        
        seg_fr = np.zeros(np.shape(spikes_bin))
        for tb_i in range(num_dt - z_bin_dt):
            seg_fr[:, tb_i] = np.sum(
                spikes_bin[:, tb_i:tb_i+z_bin_dt], 1)/z_bin
        mean_fr = np.nanmean(seg_fr, 1)
        std_fr = np.nanstd(seg_fr, 1)
        zscore_means.append(mean_fr)
        zscore_stds.append(std_fr)
        #Now pull deviation matrices
        seg_dev = deviations[s_ind]
        seg_dev[0] = 0
        seg_dev[-1] = 0
        change_inds = np.diff(seg_dev)
        start_dev_bouts = np.where(change_inds == 1)[0] + 1
        end_dev_bouts = np.where(change_inds == -1)[0]
        for b_i in range(len(start_dev_bouts)):
            dev_s_i = start_dev_bouts[b_i]
            dev_e_i = end_dev_bouts[b_i]
            dev_len = dev_e_i - dev_s_i
            
            dev_rast_i = spikes_bin[:, dev_s_i:dev_e_i]
            
            bin_starts = np.ceil(np.linspace(0,dev_len,num_bins+2)).astype('int')
            
            dev_fr_mat = np.zeros((num_neur,num_bins))
            for nb_i in range(num_bins):
                bs_i = bin_starts[nb_i]
                be_i = bin_starts[nb_i+2]
                dev_fr_mat[:,nb_i] = np.sum(dev_rast_i[:,bs_i:be_i],1)/((be_i-bs_i)/1000)
            z_dev_fr_mat = (dev_fr_mat - np.expand_dims(mean_fr,1))/np.expand_dims(std_fr,1)
            seg_dev_matrices.append(z_dev_fr_mat)
        dev_matrices.append(seg_dev_matrices)
        
    return dev_matrices
        
            
def lstm_cross_validation():
    """Function to perform training and cross-validation of a LSTM model using
    taste response firing trajectories to determine best model size"""
    
    latent_dim_sizes = np.arange(20,150,10)
    k_folds = 5
    
    fold_accuracies = dict() #For each size return fold matrix
    for latent_dim in latent_dim_sizes:
        
    
def fit_model(input_data,latent_dim):
    
    model = _get_lstm_model(input_shape,latent_dim,num_classes)
    model.fit(X, [Y,_,_], epochs = 20, batch_size = 40) #y1 is the one-hot, y2 is hidden state, y3 is cell state
    
    #Get hidden states of trained model
    
    
def lstm_dev_prediction():
    """Function to use the trained LSTM model to make classification of deviation
    events into one of the taste response categories."""
    
    
def _get_lstm_model(input_shape, latent_dim, num_classes):
    """Function to define and return an LSTM model for training/prediction."""
    
    inputs = layer.Input(shape=input_shape)
    lstm_outputs, state_h, state_c = layers.LSTM(latent_dim, activation='relu',
                                                 dropout=0.1,return_state=True)(inputs)
    predictions = layers.Dense(num_classes, activation='softmax')(lstm_outputs)
    model = Model(inputs=inputs, outputs=[predictions,state_h,state_c])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model


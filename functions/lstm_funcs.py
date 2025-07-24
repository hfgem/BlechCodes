#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:03:43 2025

@author: hannahgermaine
"""

import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import combinations
from random import sample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



def prep_lstm_data(fr_dict,day_1_tastes,all_dig_in_names):
    """Convert the data into a useable format for training and testing the LSTM"""
    
    print("--- Preparing taste data for LSTM training ---")
    #Pull out day 1 tastes
    
    data_list = []
    data_labels = []
    for dn_i, dn in enumerate(all_dig_in_names):
        taste, day = dn.split('_')
        day = int(day)
        if len(np.intersect1d(day_1_tastes,taste)): #Day 1 taste
            if day == 0:
                data = fr_dict[day][taste]
                data_list.extend(data)
                data_labels.extend([dn for i in range(len(data))])
        else: #Not a Day 1 taste
            if day > 0: #Confirm that it's not a day 1 taste
                data = fr_dict[day][taste]
                data_list.extend(data)
                data_labels.extend([dn for i in range(len(data))])
    data_array = np.array(data_list)
    
    unique_labels = np.unique(data_labels)
    label_inds = np.arange(len(unique_labels))
    data_inds = []
    for dl in data_labels:
        data_inds.extend(np.where(unique_labels == dl)[0])
    data_inds = np.array(data_inds)
        
    return data_array, data_labels, data_inds
    
def run_model_tests(data_array,data_inds):
    """Test different size LSTM models"""
    print("--- Testing different LSTM model sizes ---")
    
    num_label = len(np.unique(data_inds))
    activ_f = tf.keras.layers.ELU(alpha=2)
    
    sizes_to_test = np.arange(60,220,20).astype('int')
    sizes_tuples_to_test = list(combinations(sizes_to_test,2))
    mean_size_accuracy = np.zeros((len(sizes_tuples_to_test),num_label))
    for s_ind, sp in tqdm.tqdm(enumerate(sizes_tuples_to_test)):
        mean_size_accuracy[s_ind,:] = run_fold_tests(data_array, data_inds, \
                                                     sp, activ_f)
    overall_mean = np.nanmean(mean_size_accuracy,1)
    overall_std = np.nanstd(mean_size_accuracy,1)
    best_ind = np.argmax(overall_mean - overall_std)
    best_lstm_size = sizes_tuples_to_test[best_ind]
    return mean_size_accuracy[best_ind,:], best_lstm_size
    
def run_fold_tests(data_array, data_inds, sp, activ_f):
    """Run k-fold cross validation"""
    
    data_len, num_neur, num_bins = np.shape(data_array)
    num_label = int(len(np.unique(data_inds)))
    n_folds = 10
    
    #K-fold train/test
    randomize_inds = sample(list(np.arange(data_len)),data_len)
    randomize_data = data_array[randomize_inds,:,:]
    randomize_labels = data_inds[randomize_inds]
    fold_inds = np.ceil(np.linspace(0,data_len,n_folds+1)).astype('int')
    fold_inds[-1] = data_len
    
    #Store fold accuracies by label
    fold_accuracy = np.nan*np.ones((n_folds,num_label))
    for iter_i in range(n_folds):
        #Split data into train and test
        test_inds = np.arange(fold_inds[iter_i],fold_inds[iter_i+1])
        test_data = randomize_data[test_inds,:,:]
        test_labels = randomize_labels[test_inds]
        train_inds = np.setdiff1d(np.arange(data_len),test_inds)
        train_data = randomize_data[train_inds,:,:]
        train_labels = randomize_labels[train_inds]
        train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=num_label)
        
        fold_accuracy[iter_i,:] = run_lstm(sp,train_data,\
                                        train_labels_one_hot,test_data,\
                                            test_labels,num_label,activ_f)
    
    mean_accuracy = np.nanmean(fold_accuracy,0)
    
    return mean_accuracy
    
def run_lstm(size_pair,train_data,train_labels_one_hot,\
             test_data,test_labels,num_label,activ_f):
    """Run the LSTM given inputs"""
    
    _, interp_len, n_components = np.shape(train_data)
    
    
    #Define LSTM model
    model = Sequential([
        LSTM(int(size_pair[0]), input_shape = (interp_len,n_components), activation=activ_f, return_sequences = True), # LSTM layer with 'size' units
        LSTM(int(size_pair[1]), activation=activ_f), # LSTM layer with 'size' units
        Dense(units=num_label, activation='softmax') # Output layer for predicting 
        ])
    #Compile
    model.compile(optimizer='adam',loss='categorical_crossentropy',\
                  metrics=['categorical_accuracy'])
    #Fit
    model.fit(train_data,train_labels_one_hot,epochs=10,batch_size=20,verbose=0)
    #Predict
    predictions = model.predict(test_data,verbose=0)
    predict_inds = np.argmax(predictions,1)
    #Calculate prediction accuracy by label
    label_accuracy = np.zeros(num_label)
    for l_i in range(num_label):
        test_l_inds = np.where(test_labels == l_i)[0]
        if len(test_l_inds) > 0:
            predict_count = len(np.where(predict_inds[test_l_inds] == l_i)[0])
            label_accuracy[l_i] = predict_count/len(test_l_inds)
            
    return label_accuracy

def create_dev_rasters(num_seg, spike_times, start_end_times, deviations, \
                       z_bin, no_z = False):
    """Create deviation rasters and firing rate vectors"""
    #These rasters and vectors will include 50 ms before the event and 50 ms after the event
    z_bin_dt = np.ceil(z_bin*1000).astype('int')
    buffer = 50 #time before and after each event to use
    
    dev_rasters = dict()
    dev_times = dict()
    dev_fr_vecs = dict()
    dev_fr_vecs_zscore = dict() # Includes pre-interval for z-scoring
    zscore_means = []
    zscore_stds = []
    for s_i in range(num_seg):
        seg_spikes = spike_times[s_i]
        num_neur = len(seg_spikes)
        num_dt = int(start_end_times[s_i][1] - start_end_times[s_i][0] + 1)
        spikes_bin = np.zeros((num_neur, num_dt))
        for n_i in range(num_neur):
            neur_spikes = np.array(seg_spikes[n_i]).astype(
                'int') - int(start_end_times[s_i][0])
            spikes_bin[n_i, neur_spikes] = 1
        if not no_z:
            # Calculate z-score mean and std
            seg_fr = np.zeros(np.shape(spikes_bin))
            for tb_i in range(num_dt - z_bin_dt):
                seg_fr[:, tb_i] = np.sum(
                    spikes_bin[:, tb_i:tb_i+z_bin_dt], 1)/z_bin
            mean_fr = np.nanmean(seg_fr, 1)
            std_fr = np.nanstd(seg_fr, 1)
            zscore_means.append(mean_fr)
            zscore_stds.append(std_fr)
        #Now pull rasters and firing rate vectors
        seg_rast = []
        seg_times = []
        seg_dev_fr = []
        seg_dev_fr_z = []
        seg_dev = deviations[s_i] #Binary vector of deviation indications
        seg_dev[0:buffer] = 0 # remove all those too early to calculate a z-score
        seg_dev[-buffer:] = 0 # remove all those too late to calculate a z-score
        change_inds = np.diff(seg_dev)
        start_dev_bouts = np.where(change_inds == 1)[0] + 1
        end_dev_bouts = np.where(change_inds == -1)[0]
        for b_i in range(len(start_dev_bouts)):
            dev_s_i = start_dev_bouts[b_i] - buffer
            dev_e_i = end_dev_bouts[b_i] + buffer
            dev_rast_i = spikes_bin[:, dev_s_i:dev_e_i]
            dev_fr = np.sum(dev_rast_i, 1)/((dev_e_i-dev_s_i)/1000)
            seg_rast.append(dev_rast_i)
            seg_times.append([dev_s_i,dev_e_i])
            seg_dev_fr.append(dev_fr)
            if not no_z:
                dev_fr_z = (dev_fr - mean_fr)/std_fr
                seg_dev_fr_z.append(dev_fr_z)
        #Store to dict
        dev_rasters[s_i] = seg_rast
        dev_times[s_i] = seg_times
        dev_fr_vecs[s_i] = seg_dev_fr
        dev_fr_vecs_zscore[s_i] = seg_dev_fr_z
        
    return dev_rasters, dev_times, dev_fr_vecs, dev_fr_vecs_zscore, zscore_means, zscore_std
    
def prep_lstm_dev_data():
    """Prepare deviation events for the LSTM"""
    
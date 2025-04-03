#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 12:53:43 2025

@author: Hannah Germaine

This file contains functions for training and testing and then using a neural
network to classify deviation events as particular taste responses.
"""

import os
import random
import warnings
import tqdm
import numpy as np
import tensorflow as tf

def run_nn_pipeline(save_dir,all_dig_in_names,tastant_fr_dist_pop,
                    taste_num_deliv,max_hz_pop,tastant_fr_dist_z_pop,
                    max_hz_z_pop,min_hz_z_pop,max_num_cp,segment_dev_rasters,
                    segment_dev_times,segment_dev_fr_vecs,segment_dev_fr_vecs_zscore,
                    segments_to_analyze, segment_times, segment_spike_times,
                    bin_dt,segment_names_to_analyze):
    
    #Variables/Filepaths
    train_dir = os.path.join(save_dir,'NN_Training')
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    result_dir = os.path.join(save_dir,'NN_Results')
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        
    #Variables
    num_seg = len(segment_dev_fr_vecs)
    num_neur = len(segment_dev_fr_vecs[0][0])
    num_tastes = len(all_dig_in_names)
    
    #Run training and testing of nn using taste responses
    train_test_nn(train_dir,all_dig_in_names,taste_num_deliv,
                      tastant_fr_dist_z_pop,max_num_cp)
    
    #Run classification of deviation events using best nn structure
    
    
    
def train_test_nn(train_dir, all_dig_in_names,taste_num_deliv,
                  tastant_fr_dist,max_num_cp,bin_dt,
                  segments_to_analyze, segment_times, segment_spike_times,
                  is_z = False):
    
    warnings.filterwarnings
    
    #Variable setup
    no_taste_ind = [t_i for t_i in range(len(all_dig_in_names)) if all_dig_in_names[t_i].split('_')[0] == 'none'][0]
    shuffle_count = np.ceil(np.nanmean(np.array(taste_num_deliv))/len(segments_to_analyze)).astype('int')
    num_neur = len(np.squeeze(np.array(tastant_fr_dist[0][0][0])))
    label_strings = []
    label_t_i_inds = []
    label_cp_i_inds = []
    for t_i in range(len(all_dig_in_names)):
        if t_i == no_taste_ind:
            label_strings.append(str(t_i) + '_0')
            label_t_i_inds.append(t_i)
            label_cp_i_inds.append(0)
        else:
            for cp_i in range(max_num_cp):
                label_strings.append(str(t_i) + '_' + str(cp_i))
                label_t_i_inds.append(t_i)
                label_cp_i_inds.append(cp_i)
    label_t_i_inds = np.array(label_t_i_inds)
    label_cp_i_inds = np.array(label_cp_i_inds)
    num_labels = len(label_strings)
    label_inds = np.arange(num_labels)
    label_vec = np.eye(num_labels)
    
    #Collect true data into response and label sets
    responses = []
    labels = [] #basis vectors with 1 at label ind and 0 elsewhere
    label_indices = [] #index of label
    for t_i, taste in enumerate(all_dig_in_names):
        for d_i in range(int(np.ceil(taste_num_deliv[t_i]))):
            for cp_i in range(max_num_cp):
                try:
                    trial_data = np.squeeze(np.array(tastant_fr_dist[t_i][d_i][cp_i]))
                    if t_i == no_taste_ind:
                        responses.append(trial_data)
                        label_i = np.where((label_t_i_inds == t_i)*(label_cp_i_inds == 0))[0][0]
                    else:
                        responses.append(trial_data)
                        label_i = np.where((label_t_i_inds == t_i)*(label_cp_i_inds == cp_i))[0][0]
                    labels.append(label_vec[label_i])
                    label_indices.append(label_i)
                except:
                    skip = 1
    #Augment no-taste data
    label_i = np.where((label_t_i_inds == no_taste_ind)*(label_cp_i_inds == 0))[0][0]
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
        # Calculate mean and std of binned segment spikes for z-scoring
        if is_z == True:
            z_time_bins = np.arange(0,seg_len-bin_dt,bin_dt)
            seg_fr = np.zeros((num_neur,len(z_time_bins))) #Hz
            for bdt_i, bdt in enumerate(z_time_bins):
                seg_fr[:,bdt_i] = np.sum(segment_spike_times_s_i_bin[:,bdt:bdt+bin_dt],1)/(bin_dt/1000)
            mean_fr = np.nanmean(seg_fr,1)
            std_fr = np.nanstd(seg_fr,1)
        # Binerize Shuffled Segment Spike Times
        segment_spike_times_s_i_shuffle = [random.sample(list(np.arange(seg_len)),len(segment_spike_times[s_i][n_i])) for n_i in range(num_neur)]
        segment_spike_times_s_i_shuffle_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = np.array(
                segment_spike_times_s_i_shuffle[n_i]).astype('int')
            segment_spike_times_s_i_shuffle_bin[n_i, n_i_spike_times] = 1
        #Create fr vecs
        fr_vec_widths = random.sample(list(np.arange(250,800)),shuffle_count)
        fr_vec_starts = random.sample(list(np.arange(800,seg_len-800)),shuffle_count)
        for fr_i, fr_s in enumerate(fr_vec_starts):
            fr_w = fr_vec_widths[fr_i]
            fr_vec = np.sum(segment_spike_times_s_i_shuffle_bin[:,fr_s:fr_s+fr_w],1)/(fr_w/1000)
            if is_z == True:
                trial_data = np.squeeze((fr_vec-mean_fr)/std_fr)
            else:
                trial_data = np.squeeze(fr_vec)
            responses.append(trial_data)
            labels.append(label_vec[label_i])
            label_indices.append(label_i)
    label_indices = np.array(label_indices)
    labels = np.array(labels)
    responses = np.array(responses)
    
    #Fitting params
    dense_layer_sizes = np.arange(100,1100,100)
    num_iter = 10
    num_data_points = len(labels)
    train_prop = 0.75
    shuffle_buffer = 100
    batch_size = 250
    
    print("Train/Testing Single Layer NN")
    mean_train_accuracy_single_layer, mean_test_accuracy_single_layer \
        = single_layer_nn(dense_layer_sizes,num_iter,num_data_points,train_prop,
                        shuffle_buffer,batch_size,responses,labels,no_taste_ind,train_dir)
    
    print("Train/Testing Double Layer NN")
    mean_train_accuracy_double_layer, mean_test_accuracy_double_layer \
        = double_layer_nn(dense_layer_sizes,num_iter,num_data_points,train_prop,
                        shuffle_buffer,batch_size,responses,labels,train_dir)

def single_layer_nn(dense_layer_sizes,num_iter,num_data_points,train_prop,
                    shuffle_buffer,batch_size,responses,labels,no_taste_ind,save_dir):
    #Run nn optimization
    dls_train_accuracy = np.zeros((num_iter,len(dense_layer_sizes)))
    dls_test_accuracy = np.zeros((num_iter,len(dense_layer_sizes)))
    for n_i in tqdm.tqdm(range(num_iter)):
        #Create train/test sets
        train_inds = random.sample(list(np.arange(num_data_points)),\
                                   np.ceil(num_data_points*train_prop).astype('int'))
        test_inds = list(np.setdiff1d(np.arange(num_data_points),train_inds))
        train_data = responses[train_inds]
        train_labels = labels[train_inds]
        # none_train_inds = [i for i in range(len(train_labels)) if np.where(train_labels[i])[0][0] == no_taste_ind]
        test_data = responses[test_inds]
        test_labels = labels[test_inds]
        # none_test_inds = [i for i in range(len(test_labels)) if np.where(test_labels[i])[0][0] == no_taste_ind]
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
        train_dataset = train_dataset.shuffle(shuffle_buffer).batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        #Using the same data, test different inner dense layer sizes
        for dls_i, dls in enumerate(dense_layer_sizes):
            #Set up model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(num_neur),
                tf.keras.layers.Dense(dls,activation='tanh'),
                tf.keras.layers.Dense(num_labels)
            ])
            #Compile model
            model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss='mse',
                          metrics=['accuracy'])
            #Train model
            train_hist = model.fit(train_dataset, epochs=100, verbose=0)
            dls_train_accuracy[n_i,dls_i] = train_hist.history['accuracy'][-1]
            #Test model
            results = model.evaluate(test_dataset, verbose=0)
            dls_test_accuracy[n_i,dls_i] = results[1]
            
    mean_train_accuracy = np.nanmean(dls_train_accuracy,0)
    mean_test_accuracy = np.nanmean(dls_test_accuracy,0)
    
    f = plt.figure(figsize=(5,5))
    plt.plot(dense_layer_sizes,mean_train_accuracy,label='Mean Train Accuracy')
    plt.plot(dense_layer_sizes,mean_test_accuracy,label='Mean Test Accuracy')
    title_text = 'Single Layer NN'
    plt.legend(loc='upper left')
    plt.title(title_text)
    plt.xlabel('Dense Layer Size')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    f.savefig(os.path.join(save_dir,('_').join(title_text.split(' '))+'.png'))
    f.savefig(os.path.join(save_dir,('_').join(title_text.split(' '))+'.svg'))
    plt.close(f)
    
    return mean_train_accuracy, mean_test_accuracy

def double_layer_nn(dense_layer_sizes,num_iter,num_data_points,train_prop,
                    shuffle_buffer,batch_size,responses,labels,save_dir):
    #Run nn optimization
    dls_train_accuracy = np.zeros((num_iter,len(dense_layer_sizes),len(dense_layer_sizes)))
    dls_test_accuracy = np.zeros((num_iter,len(dense_layer_sizes),len(dense_layer_sizes)))
    for n_i in tqdm.tqdm(range(num_iter)):
        #Create train/test sets
        train_inds = random.sample(list(np.arange(num_data_points)),\
                                   np.ceil(num_data_points*train_prop).astype('int'))
        test_inds = list(np.setdiff1d(np.arange(num_data_points),train_inds))
        train_data = responses[train_inds]
        train_labels = labels[train_inds]
        test_data = responses[test_inds]
        test_labels = labels[test_inds]
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
        train_dataset = train_dataset.shuffle(shuffle_buffer).batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        #Using the same data, test different inner dense layer sizes
        for dls1_i, dls1 in tqdm.tqdm(enumerate(dense_layer_sizes)):
            for dls2_i, dls2 in enumerate(dense_layer_sizes):
                #Set up model
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(num_neur),
                    tf.keras.layers.Dense(dls1,activation='tanh'),
                    tf.keras.layers.Dense(dls2,activation='tanh'),
                    tf.keras.layers.Dense(num_labels)
                ])
                #Compile model
                model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,
                              momentum=0.9),
                              loss='mse',
                              metrics=['accuracy'])
                #Train model
                train_hist = model.fit(train_dataset, epochs=100, verbose=0)
                dls_train_accuracy[n_i,dls1_i,dls2_i] = train_hist.history['accuracy'][-1]
                #Test model
                results = model.evaluate(test_dataset, verbose=0)
                dls_test_accuracy[n_i,dls1_i,dls2_i] = results[1]
    mean_train_accuracy = np.nanmean(dls_train_accuracy,0)
    mean_test_accuracy = np.nanmean(dls_test_accuracy,0)
    
    f, ax = plt.subplots(nrows=2,ncols=2,height_ratios=[4,1],figsize=(8,5))
    im = ax[0,0].imshow(mean_train_accuracy,aspect='auto')
    ax[0,0].set_title('Mean Train Accuracy')
    ax[0,0].set_xlabel('Dense Layer Size')
    ax[0,0].set_xticks(np.arange(len(dense_layer_sizes)),dense_layer_sizes)
    ax[0,0].set_ylabel('Dense Layer Size')
    ax[0,0].set_yticks(np.arange(len(dense_layer_sizes)),dense_layer_sizes)
    f.colorbar(im, cax=ax[1,0], orientation='horizontal')
    im2 = ax[0,1].imshow(mean_test_accuracy,aspect='auto')
    ax[0,1].set_title('Mean Test Accuracy')
    ax[0,1].set_xlabel('Dense Layer Size')
    ax[0,1].set_xticks(np.arange(len(dense_layer_sizes)),dense_layer_sizes)
    ax[0,1].set_ylabel('Dense Layer Size')
    ax[0,1].set_yticks(np.arange(len(dense_layer_sizes)),dense_layer_sizes)
    f.colorbar(im2, cax=ax[1,1], orientation='horizontal')
    title_text = 'Double Layer NN'
    plt.suptitle(title_text)
    plt.tight_layout()
    f.savefig(os.path.join(save_dir,('_').join(title_text.split(' '))+'.png'))
    f.savefig(os.path.join(save_dir,('_').join(title_text.split(' '))+'.svg'))
    plt.close(f)
    
    return mean_train_accuracy, mean_test_accuracy


def classify_dev_nn():
    #Do stuff
    a = 1
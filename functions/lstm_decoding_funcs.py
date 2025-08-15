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
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import curve_fit
from matplotlib import colormaps
os.environ["OMP_NUM_THREADS"] = "4"

def create_taste_matrices(day_vars, all_dig_in_names, num_bins, z_bin_dt, start_bins=0):
    """Function to take spike times following taste delivery and create 
    matrices of timeseries firing trajectories"""
    
    print("\n--- Creating Taste Matrices ---")
    half_z_bin = np.floor(z_bin_dt/2).astype('int')
    
    #Create storage matrices and outputs
    taste_unique_categories = list(all_dig_in_names)
    training_matrices = []
    training_labels = []
    for t_i, t_name in tqdm.tqdm(enumerate(all_dig_in_names)):
        taste, day = t_name.split('_')
        day = int(day)
        t_i_day = np.where(np.array(day_vars[day]['dig_in_names']) == taste)[0][0]
        
        segment_names = day_vars[day]['segment_names']
        segment_times = day_vars[day]['segment_times']
        segment_spike_times = day_vars[day]['segment_spike_times']
        tastant_spike_times = day_vars[day]['tastant_spike_times']
        num_neur = len(tastant_spike_times[0][0])
        start_dig_in_times = day_vars[day]['start_dig_in_times']
        pre_taste_dt = day_vars[day]['pre_taste_dt']
        post_taste_dt = day_vars[day]['post_taste_dt']
        bin_starts = np.ceil(np.linspace(start_bins,post_taste_dt,num_bins+1)).astype('int')
        num_deliv = len(tastant_spike_times[t_i_day])
        
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
                seg_start+half_z_bin, seg_end-half_z_bin, z_bin_dt)
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
        for d_i in range(num_deliv):  # index for that taste
            raster_times = tastant_spike_times[t_i_day][d_i]
            start_taste_i = start_dig_in_times[t_i_day][d_i]
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
            training_matrices.append(fr_z_mat)
            training_labels.append(t_i)
    
    return taste_unique_categories, training_matrices, training_labels
    
def create_dev_matrices(day_vars, deviations, z_bin_dt, num_bins):
    """Function to take spike times during deviation events and create 
    matrices of timeseries firing trajectories the same size as taste trajectories"""
    
    print("\n--- Creating Deviation Matrices ---")
    segment_spike_times = day_vars[0]['segment_spike_times']
    segments_to_analyze = day_vars[0]['segments_to_analyze']
    start_end_times = np.array(day_vars[0]['segment_times_reshaped'])[segments_to_analyze]
    
    half_z_bin = np.floor(z_bin_dt/2).astype('int')
    dev_matrices = dict()
    
    for s_ind, s_i in tqdm.tqdm(enumerate(segments_to_analyze)):
        seg_dev_matrices = []
        
        seg_spikes = segment_spike_times[s_i]
        seg_start = int(start_end_times[s_ind][0])
        seg_end = int(start_end_times[s_ind][1])
        seg_len = seg_end - seg_start
        num_neur = len(seg_spikes)
        spikes_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            neur_spikes = np.array(seg_spikes[n_i]).astype(
                'int') - seg_start
            spikes_bin[n_i, neur_spikes] = 1
        # Calculate z-score mean and std
        time_bin_starts = np.arange(
            seg_start+half_z_bin, seg_end-half_z_bin, z_bin_dt)
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = (np.array(
                segment_spike_times_s_i[n_i]) - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        num_dt = len(time_bin_starts)
        tb_fr = np.zeros((num_neur, num_dt))
        for tb_i, tb in enumerate(time_bin_starts):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_z_bin:tb+half_z_bin-seg_start], 1)/(2*half_z_bin*(1/1000))
        mean_fr = np.mean(tb_fr, 1)
        std_fr = np.std(tb_fr, 1)
        
        seg_fr = np.zeros(np.shape(spikes_bin))
        for tb_i in range(num_dt - z_bin_dt):
            seg_fr[:, tb_i] = np.sum(
                spikes_bin[:, tb_i:tb_i+z_bin_dt], 1)/(z_bin_dt/1000)
        mean_fr = np.nanmean(seg_fr, 1)
        std_fr = np.nanstd(seg_fr, 1)
        
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
        dev_matrices[s_ind] = np.array(seg_dev_matrices)
        
    return dev_matrices

def lstm_cross_validation(training_matrices,training_labels,\
                          taste_unique_categories,savedir):
    """Function to perform training and cross-validation of a LSTM model using
    taste response firing trajectories to determine best model size"""
    
    latent_dim_sizes = np.arange(20,150,10)
    num_classes = len(np.unique(training_labels))
    ex_per_class = [len(np.where(np.array(training_labels) == i)[0]) for i in range(num_classes)]
    min_count = np.min(ex_per_class)
    
    X = np.array(training_matrices)
    Y = np.array(tf.one_hot(training_labels, num_classes))
    
    num_samples, timesteps, features = X.shape
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_dict = dict() #For each size return fold matrix
    for l_i, latent_dim in enumerate(latent_dim_sizes):
        fold_dict[l_i] = dict()
        fold_dict[l_i]["latent_dim"] = latent_dim
        fold_dict[l_i]["taste_unique_categories"] = taste_unique_categories
        
        histories = []                # To store training history (loss, accuracy) per fold
        val_accuracy_per_fold = []     # To store final validation loss and accuracy per fold
        val_loss_per_fold = []
        prediction_probabilities = np.zeros((num_samples,num_classes))
        state_cs = np.zeros((num_samples,latent_dim))
        
        for fold, (train_index, val_index) in enumerate(kf.split(X,training_labels)):
            # Split data into training and validation sets for the current fold
            #Ensure shuffling of categories
            train_index_rand = train_index[np.random.choice(np.arange(len(train_index)),len(train_index))]
            val_index_rand = val_index[np.random.choice(np.arange(len(val_index)),len(val_index))]
            
            train_data, test_data = X[train_index_rand], X[val_index_rand]
            train_cat, test_cat = Y[train_index_rand,:], Y[val_index_rand,:]
            
            history, val_loss, val_accuracy, predictions, state_c = fit_model(train_data,\
                                                    train_cat,test_data,\
                                                    test_cat,num_classes,\
                                                    latent_dim,fold,savedir)
            histories.append(history.history)
            val_accuracy_per_fold.append(val_accuracy)
            val_loss_per_fold.append(val_loss)
            prediction_probabilities[val_index_rand,:] = predictions
            state_cs[val_index_rand,:] = state_c
            
        fold_dict[l_i]["histories"] = histories
        fold_dict[l_i]["val_accuracy_per_fold"] = val_accuracy_per_fold
        fold_dict[l_i]["val_loss_per_fold"] = val_loss_per_fold
        fold_dict[l_i]["predictions"] = prediction_probabilities
        fold_dict[l_i]["true_labels"] = Y
        
        argmax_predict = np.argmax(prediction_probabilities,1)
        predict_onehot = np.array(tf.one_hot(argmax_predict, np.shape(prediction_probabilities)[1]))
        
        #Plot predictions
        f, ax = plt.subplots(ncols=3)
        ax[0].imshow(Y,aspect='auto')
        ax[0].set_title('Categories')
        ax[0].set_xticks(np.arange(num_classes),taste_unique_categories,
                         rotation=45)
        ax[1].imshow(prediction_probabilities,aspect='auto')
        ax[1].set_title('Predictions')
        ax[1].set_xticks(np.arange(num_classes),taste_unique_categories,
                         rotation=45)
        ax[2].imshow(predict_onehot,aspect='auto')
        ax[2].set_title('One-hot predictions')
        ax[2].set_xticks(np.arange(num_classes),taste_unique_categories,
                         rotation=45)
        plt.tight_layout()
        f.savefig(os.path.join(savedir,'latent_' + str(latent_dim) + '_predictions.png'))
        f.savefig(os.path.join(savedir,'latent_' + str(latent_dim) + '_predictions.svg'))
        plt.close(f)
        
        #Plot hidden states
        avg_state = []
        for class_i in range(num_classes):
            class_inds = np.where(np.array(training_labels) == class_i)[0]
            avg_state.append(np.nanmean(state_cs[class_inds,:],0))
            
        f_state, ax_state = plt.subplots(ncols=2)
        ax_state[0].imshow(state_cs,aspect='auto')
        ax_state[0].set_title('Test LSTM Hidden State')
        for class_i, class_n in enumerate(taste_unique_categories):
            ax_state[1].plot(np.arange(latent_dim),avg_state[class_i],\
                             label=class_n)
        ax_state[1].legend(loc='upper left')
        ax_state[1].set_title('Avg LSTM State C')
        plt.tight_layout()
        f_state.savefig(os.path.join(savedir,'latent_' + str(latent_dim) + '_state_c.png'))
        f_state.savefig(os.path.join(savedir,'latent_' + str(latent_dim) + '_state_c.svg'))
        plt.close(f_state)
        
    np.save(os.path.join(savedir,'fold_dict.npy'),fold_dict,allow_pickle=True)
    
def fit_model(train_data,train_cat,test_data,test_cat,num_classes,latent_dim,fold,savedir):
    """Function to fit model"""
    
    model = _get_lstm_model(np.shape(train_data[0]),latent_dim,num_classes)
    #Print model summary
    #model.summary()
    
    history = model.fit(train_data, train_cat, epochs = 20, batch_size = 40,\
                        validation_data = (test_data,test_cat),\
                            verbose=0)
    
    val_loss, val_accuracy = model.evaluate(test_data, test_cat, verbose=0)
    
    lstm_output_extractor_model = Model(inputs=model.input,
                                            outputs=model.get_layer('lstm_layer').output)
    lstm_outputs, state_h, state_c = lstm_output_extractor_model.predict(test_data)
    
    predictions = model.predict(test_data)
    
    return history, val_loss, val_accuracy, predictions, state_c
    
def _get_lstm_model(input_shape, latent_dim, num_classes):
    """Function to define and return an LSTM model for training/prediction."""
    
    inputs = layers.Input(shape=input_shape)
    lstm_outputs, state_h, state_c = layers.LSTM(units = int(latent_dim), 
                                                 dropout=0.1,return_state=True,\
                                                name='lstm_layer')(inputs)
    predictions = layers.Dense(units = int(num_classes), activation='softmax',\
                               name='dense_layer')(lstm_outputs)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

def get_best_size(fold_dict, savedir):
    """Calculate best model size"""
    
    num_tested = len(fold_dict)
    class_names = fold_dict[0]["taste_unique_categories"]
    num_classes = len(class_names)
    true_taste_inds = [i for i in range(num_classes) if class_names[i].split('_')[0] != 'none']
    tested_latent_dim = np.array([fold_dict[l_i]["latent_dim"] for l_i in range(num_tested)])
    
    accuracy = np.nan*np.ones((num_tested,num_classes))
    strong_accuracy = np.nan*np.ones((num_tested,num_classes))
    confusion_matrices = np.nan*np.ones((num_tested,num_classes,num_classes))
    
    for l_i, latent_dim in enumerate(tested_latent_dim):
        predictions = fold_dict[l_i]["predictions"]
        true_labels = fold_dict[l_i]["true_labels"]
        true_inds = np.where(true_labels == 1)[1]
        argmax_predictions = np.argmax(predictions,1)
        matching_predictions = np.where(true_inds == argmax_predictions)[0]
        for c_i in range(num_classes):
            #Calculate accuracy of prediction
            class_inds = np.where(true_inds == c_i)[0]
            predicted_inds = np.intersect1d(class_inds,matching_predictions)
            accuracy[l_i,c_i] = len(predicted_inds)/len(class_inds)
            #Calculate strong accuracy of prediction
            strong_accuracy[l_i,c_i] = len(np.where(predictions[predicted_inds,c_i] >= 0.25)[0])/len(class_inds)
            #Confusion matrix population
            predicted_classes = argmax_predictions[class_inds]
            confusion_matrices[l_i,c_i,:] = np.array([len(np.where(predicted_classes == c_i2)[0])/len(class_inds) for c_i2 in range(num_classes)])
    
    average_accuracy = np.nanmean(accuracy,1)
    average_strong_accuracy = np.nanmean(strong_accuracy,1)
    average_true_taste_accuracy = np.nanmean(accuracy[:,true_taste_inds],1)
    average_strong_true_taste_accuracy = np.nanmean(strong_accuracy[:,true_taste_inds],1)
    
    #Plot results
    f_accuracy, ax_accuracy = plt.subplots(nrows = 2, ncols = 2, figsize = (7.5,7.5))
    #Plot accuracy
    img = ax_accuracy[0,0].imshow(accuracy,aspect='auto',cmap='viridis')
    ax_accuracy[0,0].set_xticks(np.arange(num_classes),class_names,rotation=45)
    ax_accuracy[0,0].set_yticks(np.arange(num_tested),tested_latent_dim)
    ax_accuracy[0,0].set_ylabel('Latent Dim')
    img.set_clim(0, 1)
    ax_accuracy[0,0].set_title('Accurate Predictions')
    plt.colorbar(mappable=img,ax=ax_accuracy[0,0])
    #Plot strong accuracy
    img = ax_accuracy[0,1].imshow(strong_accuracy,aspect='auto',cmap='viridis')
    ax_accuracy[0,1].set_xticks(np.arange(num_classes),class_names,rotation=45)
    ax_accuracy[0,1].set_yticks(np.arange(num_tested),tested_latent_dim)
    img.set_clim(0, 1)
    ax_accuracy[0,1].set_title('Accurate Predictions w Probability >= 0.25')
    plt.colorbar(mappable=img,ax=ax_accuracy[0,1])
    #Plot average accuracy by size
    img = ax_accuracy[1,0].imshow(accuracy - strong_accuracy,aspect='auto',cmap='viridis')
    ax_accuracy[1,0].set_xticks(np.arange(num_classes),class_names,rotation=45)
    ax_accuracy[1,0].set_yticks(np.arange(num_tested),tested_latent_dim)
    img.set_clim(0, 1)
    ax_accuracy[1,0].set_title('All - Strong')
    plt.colorbar(mappable=img,ax=ax_accuracy[1,0])
    #Plot average strong accuracy by size
    ax_accuracy[1,1].plot(tested_latent_dim,average_accuracy,label='Average Accuracy')
    ax_accuracy[1,1].plot(tested_latent_dim,average_strong_accuracy,label='Average Strong Accuracy')
    ax_accuracy[1,1].plot(tested_latent_dim,average_true_taste_accuracy,label='Average True Accuracy')
    ax_accuracy[1,1].plot(tested_latent_dim,average_strong_true_taste_accuracy,label='Average Strong True Accuracy')
    ax_accuracy[1,1].set_ylim([0,1])
    ax_accuracy[1,1].set_ylabel('Average Accuracy')
    ax_accuracy[1,1].set_xlabel('Latent Dim')
    ax_accuracy[1,1].legend(loc='upper right')
    #Finish and save
    plt.tight_layout()
    f_accuracy.savefig(os.path.join(savedir,'accuracy_plots.png'))
    f_accuracy.savefig(os.path.join(savedir,'accuracy_plots.svg'))
    plt.close(f_accuracy)
    
    #Fit choose best size based on accuracy - std accuracy scoring
    f_log = plt.figure(figsize=(5,5))
    all_y = []
    all_x = []
    for t_i in true_taste_inds:
        all_y.extend(list(strong_accuracy[:,t_i]))
        all_x.extend(list(tested_latent_dim))
    plt.scatter(all_x,all_y,color='g',alpha=0.5,label='True Taste Accuracies')
    score = average_strong_true_taste_accuracy - np.nanstd(strong_accuracy[:,true_taste_inds],1)
    plt.plot(tested_latent_dim,score,linestyle='dashed',color='b',label='Score Curve')
    # try:
    #     params, covariance = curve_fit(shifted_log_func, all_x, all_y)
    #     a_fit, b_fit, c_fit = params
    #     log_y = shifted_log_func(tested_latent_dim, a_fit, b_fit, c_fit)
    #     plt.plot(tested_latent_dim,log_y,linestyle='dashed',color='k',label='Log Fit')
    #     #Calculate elbow
    #     deriv_1 = np.diff(log_y)/np.diff(tested_latent_dim)
    #     deriv_2 = np.diff(deriv_1)
    #     m = (deriv_2[-1] - deriv_2[0])/(len(deriv_2)-1)
    #     line = m*np.arange(len(deriv_2)) + deriv_2[0]
    #     best_ind = np.argmax(deriv_2 - line) + 1
    #     best_latent_dim = tested_latent_dim[best_ind]
    # except: #A log can't be fit
    #     #Calculate best score by taking the mean accuracy and subtracting the std
    #     best_ind = np.argmax(score)
    #     best_latent_dim = tested_latent_dim[best_ind]
    best_ind = np.argmax(score)
    best_latent_dim = tested_latent_dim[best_ind]
    plt.axvline(best_latent_dim,label='Best Size = ' + str(best_latent_dim),\
                color='r',linestyle='dashed')#Finish plot
    plt.ylabel('Strong Accuracy')
    plt.xlabel('Latent Dim')
    plt.legend(loc='upper left')
    plt.title('Calculated Best Latent Dim')
    plt.tight_layout()
    f_log.savefig(os.path.join(savedir,'best_latent_dim.png'))
    f_log.savefig(os.path.join(savedir,'best_latent_dim.svg'))
    plt.close(f_log)
    
    return best_latent_dim
    
def shifted_log_func(x, a, b, c):
        return a * np.log(x + c) + b

def lstm_dev_decoding(dev_matrices, training_matrices, training_labels,\
                      latent_dim, taste_unique_categories, savedir):
    """Function to run the best model on classifying the deviation events"""
    
    training_labels = np.array(training_labels)
    num_classes = len(np.unique(training_labels))
    class_inds = [np.where(training_labels == i)[0] for i in range(num_classes)]     
    ex_per_class = [len(ci) for ci in class_inds]
    min_count = np.min(ex_per_class)
    
    #Equalize the data to be the same number of samples per class
    keep_class_inds = []
    for i in range(class_inds):
        keep_class_inds.extend(list(class_inds[i][np.random.sample(class_inds[i],min_count)]))
    
    X = np.array(training_matrices)[keep_class_inds]
    train_labels_balanced = training_labels[keep_class_inds]
    Y = np.array(tf.one_hot(train_labels_balanced, num_classes))
    
    num_samples, timesteps, features = X.shape
    
    history = model.fit(X, Y, epochs = 20, batch_size = 40,\
                            verbose=0)
    
    lstm_output_extractor_model = Model(inputs=model.input,
                                            outputs=model.get_layer('lstm_layer').output)
    lstm_outputs, state_h, state_c = lstm_output_extractor_model.predict(np.array(dev_matrices))
    
    predictions = model.predict(np.array(dev_matrices))
    
    return predictions
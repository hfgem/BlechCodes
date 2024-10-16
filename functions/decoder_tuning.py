#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:38:30 2024

@author: hannahgermaine

File dedicated to functions related to testing bayesian decoder parameters for
best decoder outcomes.
"""

import os
import itertools
import csv
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from multiprocess import Pool

from sklearn.mixture import GaussianMixture as gmm
#from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
#from random import choices, sample
import functions.decode_parallel as dp
from sklearn import svm


def test_decoder_params(dig_in_names, start_dig_in_times, num_neur, tastant_spike_times,
                        tastant_fr_dist, cp_raster_inds, pre_taste_dt, post_taste_dt,
                        epochs_to_analyze, taste_select_neur, e_skip_dt, e_len_dt, save_dir):
    """This function tests different decoder types to determine
    the best combination to use in replay decoding
    INPUTS:
            - dig_in_names: name of each taste deliveres
            - start_dig_in_times: start times of tastant deliveries

    OUTPUTS:
            - best_components: list of epoch-specific best number of components for gmm
    """

    print('\tRunning Decoder Tests First.')

    # Get trial indices for train/test sets
    num_tastes = len(tastant_spike_times)
    all_trial_inds = []
    for t_i in range(num_tastes):
        taste_trials = len(tastant_spike_times[t_i])
        all_trial_inds.append(list(np.arange(taste_trials)))

    del t_i, taste_trials

    # Plot distributions treated in different ways
    plot_distributions(start_dig_in_times, tastant_fr_dist, epochs_to_analyze,
                        num_neur, dig_in_names, save_dir)

    # Run decoder through training and jackknife testing to determine success rates
    gmm_success_rates, gmm_success_rates_by_taste = run_decoder(num_neur, start_dig_in_times,
                                                                tastant_fr_dist, all_trial_inds,
                                                                tastant_spike_times, cp_raster_inds,
                                                                pre_taste_dt, e_len_dt, e_skip_dt,
                                                                dig_in_names, save_dir,
                                                                epochs_to_analyze)

    # Run Naive Bayes decoder to test
    nb_success_rates, nb_success_rates_by_taste = naive_bayes_decoding(num_neur, tastant_spike_times,
                                                                        cp_raster_inds, tastant_fr_dist,
                                                                        all_trial_inds, dig_in_names,
                                                                        start_dig_in_times, pre_taste_dt,
                                                                        post_taste_dt, e_skip_dt, e_len_dt,
                                                                        save_dir, epochs_to_analyze)
    
    # Run SVM classifier to test
    svm_success_rates, svm_success_rates_by_taste = svm_classification(num_neur, tastant_spike_times, cp_raster_inds,
                                                                       tastant_fr_dist, all_trial_inds, dig_in_names,
                                                                       start_dig_in_times, pre_taste_dt, post_taste_dt,
                                                                       e_skip_dt, e_len_dt, save_dir, epochs_to_analyze)
    # Both Models Plot
    plot_all_results(epochs_to_analyze, gmm_success_rates, nb_success_rates, svm_success_rates,
                     num_tastes, save_dir)


def plot_distributions(start_dig_in_times, tastant_fr_dist, epochs_to_analyze,
                       num_neur, dig_in_names, save_dir):
    """This function plots the firing rate distributions across tastes as is,
    z-scored, and PCA'ed.
    INPUTS:
            - 
    OUTPUTS:
            - 
    """
    dist_save = os.path.join(save_dir, 'FR_Distributions')
    if not os.path.isdir(dist_save):
        os.mkdir(dist_save)
        
    fr_save = os.path.join(dist_save,'Raw_FR')
    if not os.path.isdir(fr_save):
        os.mkdir(fr_save)

    num_tastes = len(start_dig_in_times)
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    neur_sqrt = np.ceil(np.sqrt(num_neur)).astype('int')
    square_num = neur_sqrt**2
    #neur_map = np.reshape(np.arange(square_num), (neur_sqrt, neur_sqrt))

    for e_ind, e_i in enumerate(epochs_to_analyze):
        file_exists = os.path.isfile(os.path.join(
            dist_save, 'PCA_FR_distributions_'+str(e_i)+'.png'))
        if not file_exists:
            taste_data = []
            all_data = [] #Only true tastes - none is left out to differentiate more
            all_data_labels = [] #Only true tastes - none is left out to differentiate more
            max_fr = 0
            for t_i in range(num_tastes):
                train_taste_data = []
                taste_num_deliv = len(tastant_fr_dist[t_i])
                for d_i in range(taste_num_deliv):
                    try:
                        train_taste_data.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i].T))
                    except:
                        train_taste_data.extend([])
                taste_data.append(np.array(train_taste_data))
                if len(train_taste_data) > 0:
                    if t_i < num_tastes-1:
                        if np.max(train_taste_data) > max_fr:
                            max_fr = np.max(train_taste_data)
                        all_data.extend(train_taste_data)
                        all_data_labels.extend(
                            list(t_i*np.ones(len(train_taste_data))))
            
            # Plot Neuron Firing Rates
            f_true, ax_true = plt.subplots(nrows = int(neur_sqrt),ncols = int(neur_sqrt), 
                                           figsize=(5*int(neur_sqrt), int(neur_sqrt)*5),
                                           sharex = True)
            neur_map = np.reshape(np.arange(square_num),(neur_sqrt,neur_sqrt))
            for n_i in range(num_neur):
                neur_row, neur_col = np.argwhere(neur_map == n_i)[0]
                ax_true[neur_row,neur_col].set_title('Neuron ' + str(n_i))
                ax_true[neur_row,neur_col].set_xlabel('FR')
                ax_true[neur_row,neur_col].set_ylabel('Probability')
                for t_i in range(num_tastes):
                    neur_data = taste_data[t_i][n_i,:]
                    ax_true[neur_row,neur_col].hist(neur_data, 10, density=True, histtype='bar', alpha=1/num_tastes, \
                            label=dig_in_names[t_i], color=taste_colors[t_i, :])
                if n_i == 0:
                    ax_true[neur_row,neur_col].legend(loc='upper right')
            f_true.tight_layout()
            f_true.savefig(os.path.join(
                fr_save, 'FR_distributions_'+str(e_ind)+'.png'))
            f_true.savefig(os.path.join(
                fr_save, 'FR_distributions_'+str(e_ind)+'.svg'))
            plt.close(f_true)
            
            # Scatter Plot Pairs of Neuron Firing Rates
            for n_1 in range(num_neur-1):
                for n_2 in np.arange(n_1+1,num_neur):
                    f_pair = plt.figure(figsize=(5,5))
                    for t_i in range(num_tastes):
                        neur_data1 = taste_data[t_i][n_1,:]
                        neur_data2 = taste_data[t_i][n_2,:]
                        plt.scatter(neur_data1,neur_data2,alpha=1/num_tastes, \
                                    label=dig_in_names[t_i], color=taste_colors[t_i, :])
                    plt.xlabel('Neuron ' + str(n_1))
                    plt.ylabel('Neuron ' + str(n_2))
                    plt.legend(loc='upper right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        fr_save, 'FR_n_'+str(n_1)+'_n_'+str(n_2)+'.png'))
            
            # Run PCA transform
            pca = PCA()
            pca.fit(np.array(all_data))
            exp_var = pca.explained_variance_ratio_
            num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
            if num_components == 0:
                num_components = 3
            pca_reduce = PCA(num_components)
            pca_reduce.fit(np.array(all_data))
            all_transformed = pca_reduce.transform(np.array(all_data))
            min_pca = np.min(all_transformed)
            max_pca = np.max(all_transformed)
            comp_sqrt = np.ceil(np.sqrt(num_components)).astype('int')
            comp_square_num = comp_sqrt**2
            comp_map = np.reshape(np.arange(comp_square_num),
                                  (comp_sqrt, comp_sqrt))
            
            #Plot PCA Results
            f_pca, ax_pca = plt.subplots(nrows=int(comp_sqrt), ncols=int(comp_sqrt), figsize=(
                5*int(comp_sqrt), int(comp_sqrt)*5))  # PCA reduced firing rates
            f_2pc, ax_2pc = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
            f_2pc_mean, ax_2pc_mean = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
            for t_i in range(num_tastes):
                transformed_data = pca_reduce.transform(
                    np.array(taste_data[t_i]))
                #Plot component histograms
                for c_i in range(num_components):
                    comp_row, comp_col = np.argwhere(comp_map == c_i)[0]
                    ax_pca[comp_row, comp_col].hist(transformed_data[:, c_i], np.linspace(
                        min_pca, max_pca, 100), density=True, histtype='bar', alpha=1/num_tastes, \
                            label=dig_in_names[t_i], color=taste_colors[t_i, :])
                    ax_pca[comp_row, comp_col].set_xlabel('PCA(FR)')
                    ax_pca[comp_row, comp_col].set_ylabel('Probability')
                    ax_pca[comp_row, comp_col].set_title(
                        'Component ' + str(c_i))
                    if c_i == 0:
                        ax_pca[comp_row, comp_col].legend(loc='upper right')
                #Plot components 1-v-2
                ax_2pc.scatter(transformed_data[:,0],transformed_data[:,1],\
                                  label=dig_in_names[t_i], color=taste_colors[t_i, :],\
                                      alpha=0.3)
                ax_2pc_mean.scatter(np.nanmean(transformed_data[:,0]),np.nanmean(transformed_data[:,1]),\
                                  label=dig_in_names[t_i], color=taste_colors[t_i, :],\
                                      alpha=0.5)
                ax_2pc.legend(loc='upper right')
                ax_2pc_mean.legend(loc='upper right')
            ax_2pc.set_xlabel('Component 0')
            ax_2pc.set_ylabel('Component 1')
            ax_2pc.set_title('Epoch ' + str(e_ind))
            ax_2pc_mean.set_xlabel('Component 0 Mean')
            ax_2pc_mean.set_ylabel('Component 1 Mean')
            ax_2pc_mean.set_title('Epoch ' + str(e_ind))
            f_pca.tight_layout()
            f_pca.savefig(os.path.join(
                dist_save, 'PCA_FR_distributions_'+str(e_ind)+'.png'))
            f_pca.savefig(os.path.join(
                dist_save, 'PCA_FR_distributions_'+str(e_ind)+'.svg'))
            plt.close(f_pca)
            f_2pc.tight_layout()
            f_2pc.savefig(os.path.join(
                dist_save, 'PCA_FR_2comp_scatter_'+str(e_ind)+'.png'))
            f_2pc.savefig(os.path.join(
                dist_save, 'PCA_FR_2comp_scatter_'+str(e_ind)+'.svg'))
            plt.close(f_2pc)
            f_2pc_mean.tight_layout()
            f_2pc_mean.savefig(os.path.join(
                dist_save, 'PCA_FR_2comp_scatter_'+str(e_ind)+'_means.png'))
            f_2pc_mean.savefig(os.path.join(
                dist_save, 'PCA_FR_2comp_scatter_'+str(e_ind)+'_means.svg'))
            plt.close(f_2pc_mean)
            

def run_2step_epoch_taste_decoder(num_neur, start_dig_in_times, tastant_fr_dist, all_trial_inds,
                                  tastant_spike_times, cp_raster_inds,
                                  pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names,
                                  save_dir, epochs_to_analyze=[]):
    """This function runs a two-step decoding where first the epoch is decoded
    and then the taste
    INPUTS:
            - num_neur: number of neurons in dataset
            - start_dig_in_times: times of taste deliveries
            - tastant_fr_dist: firing rate distribution to fit over (train set)
            - all_trial_inds: indices of all trials used in testing the fit
            - tastant_spike_times: spike times for each tastant delivery
            - cp_raster_inds: changepoint times for all taste deliveries
            - pre_taste_dt: ms before taste delivery in cp_raster_inds
            - e_len_dt: decoding chunk length
            - e_skip_dt: decoding skip length
            - dig_in_names: taste names
            - save_dir: directory where to save results
            - epochs_to_analyze: array of which epochs to analyze
    OUTPUTS:
            - Plots of decoder results on individual trials as well as overall success
                    metrics.
            - 
    """

    # Variables
    num_tastes = len(start_dig_in_times)
    num_cp = len(tastant_fr_dist[0][0])
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))

    # Jackknife decoding total number of trials
    total_trials = np.sum([len(all_trial_inds[t_i])
                          for t_i in range(num_tastes)])
    total_trial_inds = np.arange(total_trials)
    all_trial_taste_inds = np.array(
        [t_i*np.ones(len(all_trial_inds[t_i])) for t_i in range(num_tastes)]).flatten()
    all_trial_delivery_inds = np.array(
        [all_trial_inds[t_i] for t_i in range(num_tastes)]).flatten()

    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    else:
        num_cp = len(epochs_to_analyze)
    p_epoch = np.ones(num_cp)/num_cp
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, len(epochs_to_analyze)))

    # Save dir
    decoder_save_dir = os.path.join(save_dir, 'GMM_Sequential_Decoder_Tests')
    if not os.path.isdir(decoder_save_dir):
        os.mkdir(decoder_save_dir)

    print('\t\t\t\tPerforming LOO Decoding')
    # Fraction of the trial decoded as each taste for each epoch
    trial_decode_storage = np.zeros((total_trials, num_cp, num_tastes))
    # Binary storage of successful decodes (max fraction of trial = taste delivered)
    trial_success_storage = np.zeros((total_trials, num_cp))

    trial_decodes = os.path.join(decoder_save_dir, 'Individual_Trials')
    if not os.path.isdir(trial_decodes):
        os.mkdir(trial_decodes)

    # Which trial is being left out for decoding
    for l_o_ind in tqdm.tqdm(total_trial_inds):
        l_o_taste_ind = all_trial_taste_inds[l_o_ind].astype(
            'int')  # Taste of left out trial
        l_o_delivery_ind = all_trial_delivery_inds[l_o_ind].astype(
            'int')  # Delivery index of left out trial

        p_taste = np.zeros(num_tastes)
        for t_i in range(num_tastes):
            if t_i != l_o_taste_ind:
                p_taste[t_i] = len(all_trial_inds[t_i]) / \
                    (len(total_trial_inds)-1)
            else:
                p_taste[t_i] = (len(all_trial_inds[t_i])-1) / \
                    (len(total_trial_inds)-1)

        # Collect the training data by epoch and by taste
        epoch_train_data = dict()
        epoch_train_labels = dict()
        for e_ind, e_i in enumerate(epochs_to_analyze):
            epoch_train_data[e_i] = []
            epoch_train_labels[e_i] = []
            for t_i in range(num_tastes):
                train_taste_data = []
                taste_num_deliv = len(tastant_fr_dist[t_i])
                for d_i in range(taste_num_deliv):
                    if (d_i == l_o_delivery_ind) and (t_i == l_o_taste_ind):
                        # This is the Leave-One-Out trial so do nothing
                        train_taste_data.extend([])
                    else:
                        if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                            train_taste_data.extend(
                                list(tastant_fr_dist[t_i][d_i][e_i].T))
                        else:
                            train_taste_data.extend(
                                list(tastant_fr_dist[t_i][d_i][e_i]))
                epoch_train_labels[e_i].extend(
                    list((t_i*np.ones(len(train_taste_data))).astype('int')))
                epoch_train_data[e_i].extend(train_taste_data)

        all_epoch_train_data = []
        all_epoch_train_labels = []
        for e_i in epochs_to_analyze:
            all_epoch_train_data.extend(list(np.array(epoch_train_data[e_i])))
            all_epoch_train_labels.extend(
                list(e_i*np.ones(len(epoch_train_data[e_i]))))
        all_epoch_train_data = np.array(all_epoch_train_data)
        all_epoch_train_labels = np.array(all_epoch_train_labels)
        # First PCA transform the epoch train data
        # Run PCA transform only on non-z-scored data
        taste_pca = dict()
        if np.min(all_epoch_train_data) >= 0:
            pca_epochs = PCA()
            pca_epochs.fit(np.array(all_epoch_train_data).T)
            exp_var = pca_epochs.explained_variance_ratio_
            num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
            pca_reduce_epochs = PCA(num_components)
            pca_reduce_epochs.fit(np.array(all_epoch_train_data))
            for e_ind, e_i in enumerate(epochs_to_analyze):
                all_taste_epoch_i_train_data = epoch_train_data[e_i]
                pca_epoch_i = PCA()
                pca_epoch_i.fit(np.array(all_taste_epoch_i_train_data))
                exp_var = pca_epoch_i.explained_variance_ratio_
                num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
                pca_reduce_tastes = PCA(num_components)
                pca_reduce_tastes.fit(np.array(all_taste_epoch_i_train_data))
                taste_pca[e_i] = pca_reduce_tastes

        # Fit Gaussian mixture models to epochs then tastes
        all_gmm = dict()
        epoch_gmm = dict()
        for e_ind, e_i in enumerate(epochs_to_analyze):
            all_gmm[e_i] = dict()
            train_epoch_data = epoch_train_data[e_i]
            if np.min(all_epoch_train_data) >= 0:
                # ___PCA Transformed Data
                transformed_test_epoch_data = pca_reduce_epochs.transform(
                    np.array(train_epoch_data))
            else:
                # ___True Data
                transformed_test_epoch_data = np.array(train_epoch_data)
            gm = gmm(n_components=1, n_init=10).fit(
                transformed_test_epoch_data)
            all_gmm[e_i]['all'] = gm
            all_gmm[e_i]['taste'] = dict()
            epoch_gmm[e_i] = gm
            # Now do by taste for this epoch
            pca_reduce_tastes = taste_pca[e_ind]
            all_taste_train_data = epoch_train_data[e_i]
            all_taste_train_labels = np.array(epoch_train_labels[e_i])
            for t_i in range(num_tastes):
                train_taste_data = np.array(all_taste_train_data)[
                    all_taste_train_labels == t_i]
                if np.min(all_epoch_train_data) >= 0:
                    transformed_test_taste_data = pca_reduce_tastes.transform(
                        np.array(train_taste_data))
                else:
                    transformed_test_taste_data = np.array(train_taste_data).T
                gm = gmm(n_components=1, n_init=10).fit(
                    transformed_test_taste_data)
                all_gmm[e_i]['taste'][t_i] = gm

        for loo_e_ind, loo_e_i in enumerate(epochs_to_analyze):
            # Grab trial firing rate data
            t_cp_rast = cp_raster_inds[l_o_taste_ind]
            taste_start_dig_in = start_dig_in_times[l_o_taste_ind]
            deliv_cp = t_cp_rast[l_o_delivery_ind, :] - pre_taste_dt
            sdi = np.ceil(
                taste_start_dig_in[l_o_delivery_ind] + deliv_cp[loo_e_i]).astype('int')
            edi = np.ceil(
                taste_start_dig_in[l_o_delivery_ind] + deliv_cp[loo_e_i+1]).astype('int')
            data_len = np.ceil(edi - sdi).astype('int')
            new_time_bins = np.arange(250, data_len-250, 50)
            # ___Grab neuron firing rates in sliding bins
            td_i_bin = np.zeros((num_neur, data_len+1))
            for n_i in range(num_neur):
                n_i_spike_times = np.array(
                    tastant_spike_times[l_o_taste_ind][l_o_delivery_ind][n_i] - sdi).astype('int')
                keep_spike_times = n_i_spike_times[np.where(
                    (0 <= n_i_spike_times)*(data_len >= n_i_spike_times))[0]]
                td_i_bin[n_i, keep_spike_times] = 1
            # Calculate the firing rate vectors for these bins
            tb_fr = np.zeros((num_neur, len(new_time_bins)))
            for tb_i, tb in enumerate(new_time_bins):
                tb_fr[:, tb_i] = np.sum(
                    td_i_bin[:, tb-250:tb+250], 1)/(500/1000)

            # ___Decode epochs for given firing rates___
            # First transform data by epoch
            if np.min(all_epoch_train_data) >= 0:
                # PCA transform fr
                try:
                    tb_fr_pca_epoch = pca_reduce_epochs.transform(tb_fr.T)
                except:
                    tb_fr_pca_epoch = pca_reduce_epochs.transform(tb_fr)
                list_tb_fr_epoch = list(tb_fr_pca_epoch)
            else:
                list_tb_fr_epoch = list(tb_fr.T)

            # First decode the epoch probabilities
            inputs = zip(list_tb_fr_epoch, itertools.repeat(num_cp),
                         itertools.repeat(epoch_gmm), itertools.repeat(p_epoch))
            pool = Pool(4)
            tb_decode_epoch_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            tb_decode_epoch_array = np.squeeze(
                np.array(tb_decode_epoch_prob)).T

            f_loo, ax_loo = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))
            f_loo.suptitle('Taste ' + dig_in_names[l_o_taste_ind] + ' Delivery ' + str(
                l_o_delivery_ind) + ' Epoch ' + str(loo_e_i))
            # ___Plot epoch decode results
            for e_i_plot in range(num_cp):
                ax_loo[0].plot(new_time_bins+deliv_cp[loo_e_i], tb_decode_epoch_array[e_i_plot, :],
                               label=str(epochs_to_analyze[e_i_plot]), color=epoch_colors[e_i_plot])
                ax_loo[0].fill_between(new_time_bins+deliv_cp[loo_e_i], tb_decode_epoch_array[e_i_plot,
                                       :], color=epoch_colors[e_i_plot], alpha=0.5, label='_')
            ax_loo[0].set_ylabel('P(Epoch)')
            ax_loo[0].set_ylim([-0.1, 1.1])
            ax_loo[0].set_xlabel('Time (ms')
            ax_loo[0].legend(loc='upper right')

            # Calculate the maximally decoded epoch
            epoch_max_inds = np.argmax(tb_decode_epoch_array, 0)
            epoch_decode_fracs = [len(np.where(epoch_max_inds == e_i_decode)[
                                      0])/len(new_time_bins) for e_i_decode in range(num_cp)]
            decoded_epoch = np.argmax(epoch_decode_fracs)
            pca_reduce_tastes = taste_pca[decoded_epoch]
            ax_loo[0].set_title("Max decoded epoch " + str(decoded_epoch))

            # ___Continue to decode tastes for decoded epoch___
            # Transform data by taste
            if np.min(all_epoch_train_data) >= 0:
                # PCA transform fr
                try:
                    tb_fr_pca_taste = pca_reduce_tastes.transform(tb_fr.T)
                except:
                    tb_fr_pca_taste = pca_reduce_tastes.transform(tb_fr)
                list_tb_fr_taste = list(tb_fr_pca_taste)
            else:
                list_tb_fr_taste = list(tb_fr.T)

            # Now decode the taste probabilities
            inputs = zip(list_tb_fr_taste, itertools.repeat(num_tastes),
                         itertools.repeat(all_gmm[decoded_epoch]['taste']), itertools.repeat(p_taste))
            pool = Pool(4)
            tb_decode_taste_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            tb_decode_taste_array = np.squeeze(
                np.array(tb_decode_taste_prob)).T

            # ___Plot taste decode results
            for t_i_plot in range(num_tastes):
                ax_loo[1].plot(new_time_bins+deliv_cp[loo_e_i], tb_decode_taste_array[t_i_plot,
                               :], label=dig_in_names[t_i_plot], color=taste_colors[t_i_plot])
                ax_loo[1].fill_between(new_time_bins+deliv_cp[loo_e_i], tb_decode_taste_array[t_i_plot,
                                       :], color=taste_colors[t_i_plot], alpha=0.5, label='_')
            ax_loo[1].set_ylabel('P(Epoch)')
            ax_loo[1].set_ylim([-0.1, 1.1])
            ax_loo[1].set_xlabel('Time (ms')
            ax_loo[1].legend(loc='upper right')

            # Calculate the maximally decoded epoch
            taste_max_inds = np.argmax(tb_decode_taste_array, 0)
            taste_decode_fracs = [len(np.where(taste_max_inds == t_i_decode)[
                                      0])/len(new_time_bins) for t_i_decode in range(num_tastes)]
            decoded_taste = np.argmax(taste_decode_fracs)
            ax_loo[1].set_title("Max decoded taste " +
                                dig_in_names[decoded_taste])

            if decoded_taste == l_o_taste_ind:
                trial_success_storage[l_o_ind, loo_e_ind] = 1
            trial_decode_storage[l_o_ind, loo_e_ind, :] = taste_decode_fracs

            # Save decoding figure
            plt.tight_layout()
            f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' + str(l_o_taste_ind) +
                          '_delivery_' + str(l_o_delivery_ind) + '_epoch_' + str(loo_e_i) + '.png'))
            f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' + str(l_o_taste_ind) +
                          '_delivery_' + str(l_o_delivery_ind) + '_epoch_' + str(loo_e_i) + '.svg'))
            plt.close(f_loo)


def run_2step_istaste_whichtaste_decoder(num_neur, start_dig_in_times, tastant_fr_dist, all_trial_inds,
                                         tastant_spike_times, cp_raster_inds,
                                         pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names,
                                         save_dir, epochs_to_analyze=[]):
    """This function runs a two-step decoding where first the epoch is decoded
    and then the taste
    INPUTS:
            - num_neur: number of neurons in dataset
            - start_dig_in_times: times of taste deliveries
            - tastant_fr_dist: firing rate distribution to fit over (train set)
            - all_trial_inds: indices of all trials used in testing the fit
            - tastant_spike_times: spike times for each tastant delivery
            - cp_raster_inds: changepoint times for all taste deliveries
            - pre_taste_dt: ms before taste delivery in cp_raster_inds
            - e_len_dt: decoding chunk length
            - e_skip_dt: decoding skip length
            - dig_in_names: taste names
            - save_dir: directory where to save results
            - epochs_to_analyze: array of which epochs to analyze
    OUTPUTS:
            - Plots of decoder results on individual trials as well as overall success
                    metrics.
            - 
    """

    # Variables
    num_tastes = len(start_dig_in_times)
    num_cp = len(tastant_fr_dist[0][0])
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))

    # Jackknife decoding total number of trials
    total_trials = np.sum([len(all_trial_inds[t_i])
                          for t_i in range(num_tastes)])
    total_trial_inds = np.arange(total_trials)
    all_trial_taste_inds = np.array(
        [t_i*np.ones(len(all_trial_inds[t_i])) for t_i in range(num_tastes)]).flatten()
    all_trial_delivery_inds = np.array(
        [all_trial_inds[t_i] for t_i in range(num_tastes)]).flatten()

    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    else:
        num_cp = len(epochs_to_analyze)

    cmap = colormaps['RdBu']
    if_taste_colors = cmap(np.linspace(0, 1, 2))

    # Save dir
    decoder_save_dir = os.path.join(
        save_dir, 'GMM_Sequential_Taste_Decoder_Tests')
    if not os.path.isdir(decoder_save_dir):
        os.mkdir(decoder_save_dir)

    print('\t\t\t\tPerforming LOO Decoding')
    # Binary storage of successful decodes (max fraction of trial = taste delivered)
    trial_success_storage = np.zeros(total_trials)
    istaste_success_storage = np.zeros(total_trials)

    trial_decodes = os.path.join(decoder_save_dir, 'Individual_Trials')
    if not os.path.isdir(trial_decodes):
        os.mkdir(trial_decodes)

    # Which trial is being left out for decoding
    for l_o_ind in tqdm.tqdm(total_trial_inds):
        l_o_taste_ind = all_trial_taste_inds[l_o_ind].astype(
            'int')  # Taste of left out trial
        l_o_delivery_ind = all_trial_delivery_inds[l_o_ind].astype(
            'int')  # Delivery index of left out trial

        p_taste = np.zeros(num_tastes)
        for t_i in range(num_tastes):
            if t_i != l_o_taste_ind:
                p_taste[t_i] = len(all_trial_inds[t_i]) / \
                    (len(total_trial_inds)-1)
            else:
                p_taste[t_i] = (len(all_trial_inds[t_i])-1) / \
                    (len(total_trial_inds)-1)

        # Collect the training data by epoch and by taste
        taste_train_data = dict()
        taste_train_labels = dict()
        taste_epoch_labels = dict()
        for t_i in range(num_tastes):
            all_taste_data = []
            all_epoch_labels = []
            taste_num_deliv = len(tastant_fr_dist[t_i])
            for d_i in range(taste_num_deliv):
                if not ((d_i == l_o_delivery_ind) and (t_i == l_o_taste_ind)):
                    for e_i in epochs_to_analyze:
                        if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                            all_taste_data.extend(
                                list(tastant_fr_dist[t_i][d_i][e_i].T))
                            all_epoch_labels.extend(
                                e_i*np.ones(len(list(tastant_fr_dist[t_i][d_i][e_i].T))))
                        else:
                            all_taste_data.extend(
                                list(tastant_fr_dist[t_i][d_i][e_i]))
                            all_epoch_labels.extend(
                                e_i*np.ones(len(list(tastant_fr_dist[t_i][d_i][e_i]))))
            taste_train_labels[t_i] = list(
                (t_i*np.ones(len(all_taste_data))).astype('int'))
            taste_train_data[t_i] = all_taste_data
            taste_epoch_labels[t_i] = all_epoch_labels

        del t_i, all_taste_data, all_epoch_labels, taste_num_deliv, d_i, e_i

        all_taste_train_data = []
        all_taste_train_labels = []
        only_taste_train_data = []
        only_taste_train_labels = []
        all_taste_train_names = ['No Taste', 'Taste']
        for t_i in range(num_tastes):
            all_taste_train_data.extend(list(taste_train_data[t_i]))
            if t_i == num_tastes - 1:
                all_taste_train_labels.extend(
                    np.zeros(len(taste_train_data[t_i])))  # No taste == 0
            else:
                all_taste_train_labels.extend(
                    np.ones(len(taste_train_data[t_i])))  # Taste == 1
                only_taste_train_data.extend(list(taste_train_data[t_i]))
                only_taste_train_labels.extend(
                    t_i*np.ones(len(taste_train_data[t_i])))
        all_taste_train_data = np.array(all_taste_train_data)
        all_taste_train_labels = np.array(all_taste_train_labels)
        only_taste_train_data = np.array(only_taste_train_data)
        only_taste_train_labels = np.array(only_taste_train_labels)
        # First PCA transform the taste train data
        # Run PCA transform only on non-z-scored data
        if np.min(all_taste_train_data) >= 0:
            pca_all_tastes = PCA()
            pca_all_tastes.fit(np.array(all_taste_train_data).T)
            exp_var = pca_all_tastes.explained_variance_ratio_
            num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
            pca_reduce_all_tastes = PCA(num_components)
            pca_reduce_all_tastes.fit(np.array(all_taste_train_data))
        if np.min(only_taste_train_data) >= 0:
            pca_only_tastes = PCA()
            pca_only_tastes.fit(np.array(only_taste_train_data).T)
            exp_var = pca_only_tastes.explained_variance_ratio_
            num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
            pca_reduce_only_tastes = PCA(num_components)
            pca_reduce_only_tastes.fit(np.array(only_taste_train_data))

        # Fit Gaussian mixture models to taste or no taste
        all_taste_gmm = dict()
        num_all_taste = np.zeros(2)
        for t_i in range(2):  # 0 is no taste, 1 is taste
            train_data = all_taste_train_data[all_taste_train_labels == t_i]
            num_all_taste[t_i] = len(train_data)
            if np.min(train_data) >= 0:
                # ___PCA Transformed Data
                transformed_train_data = pca_reduce_all_tastes.transform(
                    np.array(train_data))
            else:
                # ___True Data
                transformed_train_data = np.array(train_data)
            gm = gmm(n_components=1, n_init=10).fit(transformed_train_data)
            all_taste_gmm[t_i] = gm
        p_all_taste = num_all_taste/np.sum(num_all_taste)
        # Fit Gaussian mixture models to which taste
        only_taste_gmm = dict()
        num_only_taste = np.zeros(num_tastes-1)
        for t_i in range(num_tastes-1):
            train_data = only_taste_train_data[only_taste_train_labels == t_i]
            num_only_taste[t_i] = len(train_data)
            if np.min(train_data) >= 0:
                # ___PCA Transformed Data
                transformed_train_data = pca_reduce_only_tastes.transform(
                    np.array(train_data))
            else:
                # ___True Data
                transformed_train_data = np.array(train_data)
            gm = gmm(n_components=1, n_init=10).fit(transformed_train_data)
            only_taste_gmm[t_i] = gm
        p_only_taste = num_only_taste/np.sum(num_only_taste)

        # Grab trial firing rate data
        t_cp_rast = cp_raster_inds[l_o_taste_ind]
        taste_start_dig_in = start_dig_in_times[l_o_taste_ind]
        deliv_cp = t_cp_rast[l_o_delivery_ind, :] - pre_taste_dt
        sdi = np.ceil(taste_start_dig_in[l_o_delivery_ind]).astype('int')
        edi = np.ceil(
            taste_start_dig_in[l_o_delivery_ind] + deliv_cp[-1]).astype('int')
        data_len = np.ceil(edi - sdi).astype('int')
        new_time_bins = np.arange(50, data_len-50, 50)
        # ___Grab neuron firing rates in sliding bins
        td_i_bin = np.zeros((num_neur, data_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = np.array(
                tastant_spike_times[l_o_taste_ind][l_o_delivery_ind][n_i] - sdi).astype('int')
            keep_spike_times = n_i_spike_times[np.where(
                (0 <= n_i_spike_times)*(data_len >= n_i_spike_times))[0]]
            td_i_bin[n_i, keep_spike_times] = 1
        # Calculate the firing rate vectors for these bins
        tb_fr = np.zeros((num_neur, len(new_time_bins)))
        for tb_i, tb in enumerate(new_time_bins):
            tb_fr[:, tb_i] = np.sum(td_i_bin[:, tb-50:tb+50], 1)/(100/1000)

        # ___Decode taste or not for given firing rates___
        # First transform data
        if np.min(all_taste_train_data) >= 0:
            # PCA transform fr
            try:
                tb_fr_pca_all_tastes = pca_reduce_all_tastes.transform(tb_fr.T)
            except:
                tb_fr_pca_all_tastes = pca_reduce_all_tastes.transform(tb_fr)
            list_tb_fr_all_tastes = list(tb_fr_pca_all_tastes)
        else:
            list_tb_fr_all_tastes = list(tb_fr.T)

        # First decode whether a taste is present
        inputs = zip(list_tb_fr_all_tastes, itertools.repeat(2),
                     itertools.repeat(all_taste_gmm), itertools.repeat(p_all_taste))
        pool = Pool(4)
        tb_decode_all_taste_prob = pool.map(
            dp.segment_taste_decode_dependent_parallelized, inputs)
        pool.close()
        tb_decode_all_taste_array = np.squeeze(
            np.array(tb_decode_all_taste_prob)).T

        f_loo, ax_loo = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))
        f_loo.suptitle(
            'Taste ' + dig_in_names[l_o_taste_ind] + ' Delivery ' + str(l_o_delivery_ind))
        # ___Plot epoch decode results
        for t_i_plot in range(2):
            ax_loo[0].plot(new_time_bins, tb_decode_all_taste_array[t_i_plot, :],
                           label=all_taste_train_names[t_i_plot], color=if_taste_colors[t_i_plot])
            ax_loo[0].fill_between(new_time_bins, tb_decode_all_taste_array[t_i_plot, :],
                                   color=if_taste_colors[t_i_plot], alpha=0.5, label='_')
        ax_loo[0].set_ylabel('P(Is Taste)')
        ax_loo[0].set_ylim([-0.1, 1.1])
        ax_loo[0].set_xlabel('Time (ms')
        ax_loo[0].legend(loc='upper right')

        # Calculate if taste maximally decoded
        iftaste_max_inds = np.argmax(tb_decode_all_taste_array, 0)
        iftaste_decode_fracs = [len(np.where(iftaste_max_inds == i_decode)[
                                    0])/len(new_time_bins) for i_decode in range(2)]
        decoded_iftaste = np.argmax(iftaste_decode_fracs).astype('int')
        ax_loo[0].set_title(
            "Max decoded " + all_taste_train_names[decoded_iftaste])

        # ___Continue to decode which taste if taste___
        if decoded_iftaste == 1:
            if l_o_taste_ind != num_tastes - 1:  # Successfully decoded that it's a taste
                istaste_success_storage[l_o_ind] = 1

            # Transform data by taste
            if np.min(only_taste_train_data) >= 0:
                # PCA transform fr
                try:
                    tb_fr_pca_taste = pca_reduce_only_tastes.transform(tb_fr.T)
                except:
                    tb_fr_pca_taste = pca_reduce_only_tastes.transform(tb_fr)
                list_tb_fr_taste = list(tb_fr_pca_taste)
            else:
                list_tb_fr_taste = list(tb_fr.T)

            # Now decode the taste probabilities
            inputs = zip(list_tb_fr_taste, itertools.repeat(num_tastes-1),
                         itertools.repeat(only_taste_gmm), itertools.repeat(p_only_taste))
            pool = Pool(4)
            tb_decode_taste_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            tb_decode_taste_array = np.squeeze(
                np.array(tb_decode_taste_prob)).T

            # ___Plot taste decode results
            for t_i_plot in range(num_tastes-1):
                ax_loo[1].plot(new_time_bins, tb_decode_taste_array[t_i_plot, :],
                               label=dig_in_names[t_i_plot], color=taste_colors[t_i_plot])
                ax_loo[1].fill_between(new_time_bins, tb_decode_taste_array[t_i_plot, :],
                                       color=taste_colors[t_i_plot], alpha=0.5, label='_')
            ax_loo[1].set_ylabel('P(Taste)')
            ax_loo[1].set_ylim([-0.1, 1.1])
            ax_loo[1].set_xlabel('Time (ms')
            ax_loo[1].legend(loc='upper right')

            # Calculate the maximally decoded taste
            taste_max_inds = np.argmax(tb_decode_taste_array, 0)
            taste_decode_fracs = [len(np.where(taste_max_inds == t_i_decode)[
                                      0])/len(new_time_bins) for t_i_decode in range(num_tastes-1)]
            decoded_taste = np.argmax(taste_decode_fracs)
            ax_loo[1].set_title("Max decoded taste " +
                                dig_in_names[decoded_taste])

            if decoded_taste == l_o_taste_ind:
                trial_success_storage[l_o_ind] = 1
        else:  # Check if it's a no taste trial
            if l_o_taste_ind == num_tastes - 1:
                trial_success_storage[l_o_ind] = 1
                istaste_success_storage[l_o_ind] = 1

        # Save decoding figure
        plt.tight_layout()
        f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' +
                      str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.png'))
        f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' +
                      str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.svg'))
        plt.close(f_loo)

    # Calculate success percents
    whichtaste_success = np.nanmean(trial_success_storage)
    istaste_success = np.nanmean(istaste_success_storage)


def run_decoder(num_neur, start_dig_in_times, tastant_fr_dist, all_trial_inds,
                tastant_spike_times, cp_raster_inds,
                pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names,
                save_dir, epochs_to_analyze=[]):
    """This function runs a decoder with a given set of parameters and returns
    the decoding probabilities of taste delivery periods
    INPUTS:
            - num_neur: number of neurons in dataset
            - start_dig_in_times: times of taste deliveries
            - tastant_fr_dist: firing rate distribution to fit over (train set)
            - all_trial_inds: indices of all trials used in testing the fit
            - tastant_spike_times: spike times for each tastant delivery
            - cp_raster_inds: changepoint times for all taste deliveries
            - pre_taste_dt: ms before taste delivery in cp_raster_inds
            - e_len_dt: decoding chunk length
            - e_skip_dt: decoding skip length
            - dig_in_names: taste names
            - save_dir: directory where to save results
            - epochs_to_analyze: array of which epochs to analyze
    OUTPUTS:
            - Plots of decoder results on individual trials as well as overall success
                    metrics.
            - epoch_success_storage: vector of length number of epochs containing success
                    percentages overall.
            - epoch_success_by_taste: array of size num_epochs x num_tastes containing
                    success percentages by decoded taste by epoch.
    """
    print("\t\tTesting GMM Decoder.")
    # TODO: Handle taste selective neurons
    # Variables
    num_tastes = len(start_dig_in_times)
    num_cp = len(tastant_fr_dist[0][0])
    p_taste = np.ones(num_tastes)/num_tastes  # P(taste)
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    half_decode_bin_dt = np.ceil(e_len_dt/2).astype('int')

    # Jackknife decoding total number of trials
    total_trials = np.sum([len(all_trial_inds[t_i])
                          for t_i in range(num_tastes)])
    total_trial_inds = np.arange(total_trials)
    all_trial_taste_inds = []
    for t_i in range(num_tastes):
        all_trial_taste_inds.extend(list(t_i*np.ones(len(all_trial_inds[t_i]))))
    all_trial_delivery_inds = []
    for t_i in range(num_tastes):
        all_trial_delivery_inds.extend(list(all_trial_inds[t_i]))

    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, len(epochs_to_analyze)))

    # Save dir
    decoder_save_dir = os.path.join(save_dir, 'GMM_Decoder_Tests')
    if not os.path.isdir(decoder_save_dir):
        os.mkdir(decoder_save_dir)

    epoch_success_storage = np.zeros(len(epochs_to_analyze))
    epoch_decode_storage = []

    for e_ind, e_i in enumerate(epochs_to_analyze):  # By epoch conduct decoding
        print('\t\t\tDecoding Epoch ' + str(e_i))

        epoch_decode_save_dir = os.path.join(
            decoder_save_dir, 'decode_prob_epoch_' + str(e_i))
        if not os.path.isdir(epoch_decode_save_dir):
            os.mkdir(epoch_decode_save_dir)

        trial_decodes = os.path.join(
            epoch_decode_save_dir, 'Individual_Trials')
        if not os.path.isdir(trial_decodes):
            os.mkdir(trial_decodes)

        try:  # Try to import the decoding results
            trial_success_storage = []
            with open(os.path.join(epoch_decode_save_dir, 'success_by_trial.csv'), newline='') as successtrialfile:
                filereader = csv.reader(
                    successtrialfile, delimiter=',', quotechar='|')
                for row in filereader:
                    trial_success_storage.append(np.array(row).astype('float'))
            trial_success_storage = np.array(trial_success_storage).squeeze()

            trial_decode_storage = []
            with open(os.path.join(epoch_decode_save_dir, 'mean_taste_decode_components.csv'), newline='') as decodefile:
                filereader = csv.reader(
                    decodefile, delimiter=',', quotechar='|')
                for row in filereader:
                    trial_decode_storage.append(np.array(row).astype('float'))
            trial_decode_storage = np.array(trial_decode_storage).squeeze()

            epoch_decode_storage.append(trial_decode_storage)

            # Calculate overall decoding success by component count
            taste_success_percent = np.round(
                100*np.nanmean(trial_success_storage), 2)
            epoch_success_storage[e_ind] = taste_success_percent

        except:  # Run decoding

            # Fraction of the trial decoded as each taste for each component count
            trial_decode_storage = np.zeros((total_trials, num_tastes))
            # Binary storage of successful decodes (max fraction of trial = taste delivered)
            trial_success_storage = np.zeros(total_trials)

            print('\t\t\t\tPerforming LOO Decoding')
            # Which trial is being left out for decoding
            for l_o_ind in tqdm.tqdm(total_trial_inds):
                l_o_taste_ind = all_trial_taste_inds[l_o_ind].astype(
                    'int')  # Taste of left out trial
                l_o_delivery_ind = all_trial_delivery_inds[l_o_ind].astype(
                    'int')  # Delivery index of left out trial

                # Run gmm distribution fits to fr of each population for each taste
                train_data = []
                all_train_data = [] #Only true tastes - excluding "none"
                #taste_bic_scores = np.zeros((len(component_counts),num_tastes))
                for t_i in range(num_tastes):
                    train_taste_data = []
                    taste_num_deliv = len(tastant_fr_dist[t_i])
                    for d_i in all_trial_inds[t_i]:
                        if (d_i == l_o_delivery_ind) and (t_i == l_o_taste_ind):
                            # This is the Leave-One-Out trial so do nothing
                            train_taste_data.extend([])
                        else:
                            if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                                train_taste_data.extend(
                                    list(tastant_fr_dist[t_i][d_i][e_i].T))
                            else:
                                train_taste_data.extend(
                                    list(tastant_fr_dist[t_i][d_i][e_i]))
                    train_data.append(np.array(train_taste_data))
                    if t_i < num_tastes-1:
                        all_train_data.extend(train_taste_data)
                    
                # Run PCA transform only on non-z-scored data
                if np.min(all_train_data) >= 0:
                    pca = PCA()
                    pca.fit(np.array(all_train_data).T)
                    exp_var = pca.explained_variance_ratio_
                    num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
                    if num_components == 0:
                        num_components = 3
                    pca_reduce = PCA(num_components)
                    pca_reduce.fit(np.array(all_train_data))

                # Grab trial firing rate data
                t_cp_rast = cp_raster_inds[l_o_taste_ind]
                taste_start_dig_in = start_dig_in_times[l_o_taste_ind]
                deliv_cp = t_cp_rast[l_o_delivery_ind, :] - pre_taste_dt
                sdi = np.ceil(
                    taste_start_dig_in[l_o_delivery_ind] + deliv_cp[e_i]).astype('int')
                edi = np.ceil(
                    taste_start_dig_in[l_o_delivery_ind] + deliv_cp[e_i+1]).astype('int')
                data_len = np.ceil(edi - sdi).astype('int')
                new_time_bins = np.arange(half_decode_bin_dt, data_len-half_decode_bin_dt, e_skip_dt)
                # ___Grab neuron firing rates in sliding bins
                td_i_bin = np.zeros((num_neur, data_len+1))
                for n_i in range(num_neur):
                    n_i_spike_times = np.array(
                        tastant_spike_times[l_o_taste_ind][l_o_delivery_ind][n_i] - sdi).astype('int')
                    keep_spike_times = n_i_spike_times[np.where(
                        (0 <= n_i_spike_times)*(data_len >= n_i_spike_times))[0]]
                    td_i_bin[n_i, keep_spike_times] = 1
                if len(new_time_bins) > 1:
                    # Calculate the firing rate vectors for these bins
                    tb_fr = np.zeros((num_neur, len(new_time_bins)))
                    for tb_i, tb in enumerate(new_time_bins):
                        tb_fr[:, tb_i] = np.sum(
                            td_i_bin[:, tb-half_decode_bin_dt:tb+half_decode_bin_dt], 1)/(int(half_decode_bin_dt*2)/1000)
                else:
                    tb_fr = np.expand_dims(np.sum(td_i_bin,1)/((data_len+1)/1000),1)
                    
                if np.min(all_train_data) >= 0: #If it's not z-scored PCA to whiten
                     # PCA transform fr
                    try:
                        tb_fr_pca = pca_reduce.transform(tb_fr.T)
                    except:
                        tb_fr_pca = pca_reduce.transform(tb_fr)
                    list_tb_fr = list(tb_fr_pca)
                else: #If z-scored, train directly on data
                    list_tb_fr = list(tb_fr.T)
                     
                f_loo = plt.figure(figsize=(5, 5))
                plt.suptitle(
                    'Taste ' + dig_in_names[l_o_taste_ind] + ' Delivery ' + str(l_o_delivery_ind))
                # Fit a Gaussian mixture model with the number of dimensions = number of neurons
                all_taste_gmm = dict()
                for t_i in range(num_tastes):
                    train_taste_data = train_data[t_i]
                    if np.min(all_train_data) >= 0:
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

                # Calculate decoding probabilities for given jackknifed trial

                # Type 1: Bins of firing rates across the epoch of response
                # ___Pass inputs to parallel computation on probabilities
                inputs = zip(list_tb_fr, itertools.repeat(num_tastes),
                             itertools.repeat(all_taste_gmm), itertools.repeat(p_taste))
                pool = Pool(4)
                tb_decode_prob = pool.map(
                    dp.segment_taste_decode_dependent_parallelized, inputs)
                pool.close()
                tb_decode_array = np.squeeze(np.array(tb_decode_prob)).T
                # ___Plot decode results
                if len(new_time_bins) > 1:
                    for t_i_plot in range(num_tastes):
                        plt.plot(new_time_bins+deliv_cp[e_i], tb_decode_array[t_i_plot, :],
                                 label=dig_in_names[t_i_plot], color=taste_colors[t_i_plot])
                        plt.fill_between(
                            new_time_bins+deliv_cp[e_i], tb_decode_array[t_i_plot, :], color=taste_colors[t_i_plot], alpha=0.5, label='_')
                    plt.xlabel('Time (ms)')
                else:
                    for t_i_plot in range(num_tastes):
                        plt.axhline(tb_decode_array[t_i_plot],
                                 label=dig_in_names[t_i_plot], color=taste_colors[t_i_plot])
                        plt.fill_between([0,1],[0,0],[tb_decode_array[t_i_plot],tb_decode_array[t_i_plot]],
                                         color=taste_colors[t_i_plot],alpha=0.5,label='_')
                plt.ylabel('P(Taste)')
                plt.ylim([-0.1, 1.1])
                
                plt.legend(loc='upper right')
                # ___Calculate the average fraction of the epoch that was decoded as each taste and store
                if len(new_time_bins) > 0:
                    taste_max_inds = np.argmax(tb_decode_array, 0)
                    taste_decode_fracs = [len(np.where(taste_max_inds == t_i_decode)[
                                              0])/len(new_time_bins) for t_i_decode in range(num_tastes)]
                else:
                    taste_decode_fracs = list(tb_decode_array)
                trial_decode_storage[l_o_ind, :] = taste_decode_fracs
                # ___Calculate the fraction of time in the epoch of each taste being best
                best_taste = np.where(
                    taste_decode_fracs == np.max(taste_decode_fracs))[0]
                if len(best_taste) == 1:
                    if best_taste == l_o_taste_ind:
                        trial_success_storage[l_o_ind] = 1
                else:
                    # Taste is one of the predicted tastes in a "tie"
                    if len(np.where(best_taste == l_o_taste_ind)[0]) > 0:
                        trial_success_storage[l_o_ind] = 1

                # Save decoding figure
                plt.tight_layout()
                f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' +
                              str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.png'))
                f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' +
                              str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.svg'))
                plt.close(f_loo)

            # Once all trials are decoded, save decoding success results
            np.savetxt(os.path.join(epoch_decode_save_dir,
                       'success_by_trial.csv'), trial_success_storage, delimiter=',')
            np.savetxt(os.path.join(epoch_decode_save_dir,
                       'mean_taste_decode_components.csv'), trial_decode_storage, delimiter=',')
            epoch_decode_storage.append(trial_decode_storage)

            # Calculate overall decoding success by component count
            taste_success_percent = np.round(
                100*np.nanmean(trial_success_storage), 2)
            epoch_success_storage[e_ind] = taste_success_percent

    # Plot the overall success results for different component counts across epochs
    f_epochs = plt.figure(figsize=(5, 5))
    plt.bar(np.arange(len(epochs_to_analyze)), epoch_success_storage)
    epoch_labels = ['Epoch ' + str(e_i) for e_i in epochs_to_analyze]
    plt.xticks(np.arange(len(epochs_to_analyze)), labels=epoch_labels)
    plt.ylim([0, 100])
    plt.axhline(100/num_tastes, linestyle='dashed',
                color='k', alpha=0.75, label='Chance')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.title('Decoding Success')
    f_epochs.savefig(os.path.join(decoder_save_dir, 'gmm_success.png'))
    f_epochs.savefig(os.path.join(decoder_save_dir, 'gmm_success.svg'))
    plt.close(f_epochs)

    # Plot the by-taste success results
    # true taste indices == all_trial_taste_inds
    # decode percents == epoch_decode_storage
    f_percents = plt.figure(figsize=(5, 5))
    epoch_success_by_taste = np.zeros((len(epochs_to_analyze), num_tastes))
    for e_ind, e_i in enumerate(epochs_to_analyze):
        epoch_decode_percents = epoch_decode_storage[e_ind]
        success_by_taste = np.zeros(num_tastes)
        for t_i in range(num_tastes):
            taste_trials = np.where(np.array(all_trial_taste_inds) == t_i)[0]
            taste_trial_results_bin = np.zeros(len(taste_trials))
            for tt_ind, tt_i in enumerate(taste_trials):
                trial_decode_results = epoch_decode_percents[tt_i, :]
                best_taste = np.where(
                    trial_decode_results == np.max(trial_decode_results))[0]
                if len(best_taste) == 1:
                    if best_taste[0] == t_i:
                        taste_trial_results_bin[tt_ind] = 1
                else:
                    # Taste is one of the predicted tastes in a "tie"
                    if len(np.where(best_taste == t_i)[0]) > 0:
                        taste_trial_results_bin[tt_ind] = 1
            success_by_taste[t_i] = 100*np.nanmean(taste_trial_results_bin)
        epoch_success_by_taste[e_ind, :] = success_by_taste
        plt.scatter(np.arange(num_tastes), success_by_taste,
                    label='Epoch ' + str(e_i), color=epoch_colors[e_ind, :])
        plt.plot(np.arange(num_tastes), success_by_taste, label='_',
                 color=epoch_colors[e_ind, :], linestyle='dashed', alpha=0.75)
    np.savetxt(os.path.join(decoder_save_dir, 'epoch_success_by_taste.csv'),
               epoch_success_by_taste, delimiter=',')
    plt.axhline(100/num_tastes, label='Chance',
                color='k', linestyle='dashed', alpha=0.75)
    plt.legend(loc='lower left')
    plt.xlabel('Taste')
    plt.xticks(np.arange(num_tastes), dig_in_names)
    plt.ylabel('Percent')
    plt.title('Decoding Success by Taste')
    f_percents.savefig(os.path.join(
        decoder_save_dir, 'gmm_success_by_taste.png'))
    f_percents.savefig(os.path.join(
        decoder_save_dir, 'gmm_success_by_taste.svg'))
    plt.close(f_percents)

    return epoch_success_storage, epoch_success_by_taste

def decoding_all_combined(num_neur, start_dig_in_times, tastant_fr_dist, all_trial_inds,
                tastant_spike_times, cp_raster_inds,
                pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names,
                save_dir, epochs_to_analyze=[]):
    """This function trains a decoder with a given set of parameters to 
    distinguish all taste-epoch firing rate distributions.
    INPUTS:
            - num_neur: number of neurons in dataset
            - start_dig_in_times: times of taste deliveries
            - tastant_fr_dist: firing rate distribution to fit over (train set)
            - all_trial_inds: indices of all trials used in testing the fit
            - tastant_spike_times: spike times for each tastant delivery
            - cp_raster_inds: changepoint times for all taste deliveries
            - pre_taste_dt: ms before taste delivery in cp_raster_inds
            - e_len_dt: decoding chunk length
            - e_skip_dt: decoding skip length
            - dig_in_names: taste names
            - save_dir: directory where to save results
            - epochs_to_analyze: array of which epochs to analyze
    OUTPUTS:
            - Plots of decoder results on individual trials as well as overall success
                    metrics.
            - epoch_success_storage: vector of length number of epochs containing success
                    percentages overall.
            - epoch_success_by_taste: array of size num_epochs x num_tastes containing
                    success percentages by decoded taste by epoch.
    """
    print("\t\tTesting GMM Taste-Epoch Category Decoder.")
    # Variables
    num_tastes = len(start_dig_in_times)
    num_cp = len(tastant_fr_dist[0][0])
    num_epochs_to_analyze = len(epochs_to_analyze)
    num_cat = num_tastes*num_epochs_to_analyze
    train_epochs = []
    train_tastes = []
    cat_names = []
    for e_i in range(num_epochs_to_analyze):
        for t_i in range(num_tastes):
            train_epochs.extend([e_i])
            train_tastes.extend([t_i])
            cat_names.extend([dig_in_names[t_i] + ' ' + str(e_i)])
    
    #p_taste = np.ones(num_tastes)/num_tastes  # P(taste)
    cmap = colormaps['jet']
    t_e_colors = cmap(np.linspace(0,1,num_cat))
    p_t_e = np.ones(num_cat)/(num_cat) #P(taste-epoch)
    cmap = colormaps['twilight_shifted']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, len(epochs_to_analyze)))
    half_decode_bin_dt = np.ceil(e_len_dt/2).astype('int')

    # Jackknife decoding total number of trials
    taste_trial_counts = [len(all_trial_inds[t_i])
                          for t_i in range(num_tastes)]
    total_trials = np.sum([len(all_trial_inds[t_i])
                          for t_i in range(num_tastes)])
    total_trial_inds = np.arange(total_trials)
    all_trial_taste_inds = np.array(
        [t_i*np.ones(len(all_trial_inds[t_i])) for t_i in range(num_tastes)]).flatten()
    all_trial_delivery_inds = np.array(
        [all_trial_inds[t_i] for t_i in range(num_tastes)]).flatten()

    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    
    # Save dir
    decoder_save_dir = os.path.join(save_dir, 'GMM_All_Cat_Decoder_Tests')
    if not os.path.isdir(decoder_save_dir):
        os.mkdir(decoder_save_dir)

    trial_decodes = os.path.join(
        decoder_save_dir, 'Individual_Trials')
    if not os.path.isdir(trial_decodes):
        os.mkdir(trial_decodes)
        
    try:  # Try to import the decoding results
        trial_success_storage = []
        with open(os.path.join(decoder_save_dir, 'success_by_trial.csv'), newline='') as successtrialfile:
            filereader = csv.reader(
                successtrialfile, delimiter=',', quotechar='|')
            for row in filereader:
                trial_success_storage.append(np.array(row).astype('float'))
        trial_success_storage = np.array(trial_success_storage).squeeze()
        
        trial_decode_storage = []
        with open(os.path.join(decoder_save_dir, 'mean_cat_decode_components.csv'), newline='') as decodefile:
            filereader = csv.reader(
                decodefile, delimiter=',', quotechar='|')
            for row in filereader:
                trial_decode_storage.append(np.array(row).astype('float'))
        trial_decode_storage = np.array(trial_decode_storage).squeeze()

        trial_taste_decode_storage = []
        with open(os.path.join(decoder_save_dir, 'mean_taste_decode_components.csv'), newline='') as decodefile:
            filereader = csv.reader(
                decodefile, delimiter=',', quotechar='|')
            for row in filereader:
                trial_taste_decode_storage.append(np.array(row).astype('float'))
        trial_taste_decode_storage = np.array(trial_taste_decode_storage).squeeze()

    except:  # Run decoding
        # Fraction of the trial decoded as each taste for each component count
        trial_decode_storage = np.zeros((total_trials, num_cat))
        trial_taste_decode_storage = np.zeros((total_trials,num_tastes))
        # Binary storage of successful decodes (max fraction of trial = taste delivered)
        trial_success_storage = np.zeros(total_trials)

        print('\t\t\t\tPerforming LOO Decoding')
        # Which trial is being left out for decoding
        for l_o_ind in tqdm.tqdm(total_trial_inds):
            l_o_taste_ind = all_trial_taste_inds[l_o_ind].astype(
                'int')  # Taste of left out trial
            l_o_delivery_ind = all_trial_delivery_inds[l_o_ind].astype(
                'int')  # Delivery index of left out trial

            # Run gmm distribution fits to fr of each population for each taste
            train_data = []
            all_train_data = []
            for tr_e_i in epochs_to_analyze:
                for tr_t_i in range(num_tastes):
                    train_taste_data = []
                    taste_num_deliv = len(tastant_fr_dist[tr_t_i])
                    for tr_d_i in range(taste_num_deliv):
                        if (tr_d_i == l_o_delivery_ind) and (tr_t_i == l_o_taste_ind):
                            # This is the Leave-One-Out trial so do nothing
                            train_taste_data.extend([])
                        else:
                            if np.shape(tastant_fr_dist[tr_t_i][tr_d_i][tr_e_i])[0] == num_neur:
                                train_taste_data.extend(
                                    list(tastant_fr_dist[tr_t_i][tr_d_i][tr_e_i].T))
                            else:
                                train_taste_data.extend(
                                    list(tastant_fr_dist[tr_t_i][tr_d_i][tr_e_i]))
                    train_data.append(np.array(train_taste_data))
                    if tr_t_i < num_tastes-1:
                        all_train_data.extend(train_taste_data)
            
            # Run PCA transform only on non-z-scored data
            if np.min(all_train_data) >= 0:
                pca = PCA()
                pca.fit(np.array(all_train_data).T)
                exp_var = pca.explained_variance_ratio_
                num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
                if num_components == 0:
                    num_components = 3
                pca_reduce = PCA(num_components)
                pca_reduce.fit(np.array(all_train_data))
            
            # Grab trial firing rate data
            t_cp_rast = cp_raster_inds[l_o_taste_ind]
            taste_start_dig_in = start_dig_in_times[l_o_taste_ind]
            deliv_cp = t_cp_rast[l_o_delivery_ind, :] - pre_taste_dt
            sdi = np.ceil(
                taste_start_dig_in[l_o_delivery_ind] + deliv_cp[0]).astype('int')
            edi = np.ceil(
                taste_start_dig_in[l_o_delivery_ind] + deliv_cp[-1]).astype('int')
            data_len = np.ceil(edi - sdi).astype('int')
            new_time_bins = np.arange(half_decode_bin_dt, data_len-half_decode_bin_dt, e_skip_dt)
            # ___Grab neuron firing rates in sliding bins
            td_i_bin = np.zeros((num_neur, data_len+1))
            for n_i in range(num_neur):
                n_i_spike_times = np.array(
                    tastant_spike_times[l_o_taste_ind][l_o_delivery_ind][n_i] - sdi).astype('int')
                keep_spike_times = n_i_spike_times[np.where(
                    (0 <= n_i_spike_times)*(data_len >= n_i_spike_times))[0]]
                td_i_bin[n_i, keep_spike_times] = 1
            # Calculate the firing rate vectors for these bins
            tb_fr = np.zeros((num_neur, len(new_time_bins)))
            for tb_i, tb in enumerate(new_time_bins):
                tb_fr[:, tb_i] = np.sum(
                    td_i_bin[:, tb-half_decode_bin_dt:tb+half_decode_bin_dt], 1)/(int(half_decode_bin_dt*2)/1000)

            if np.min(all_train_data) >= 0: #If it's not z-scored PCA to whiten
                # PCA transform fr
                try:
                    tb_fr_pca = pca_reduce.transform(tb_fr.T)
                except:
                    tb_fr_pca = pca_reduce.transform(tb_fr)
                list_tb_fr = list(tb_fr_pca)
            else: #If z-scored, train directly on data
                list_tb_fr = list(tb_fr.T)

            f_loo = plt.figure(figsize=(5, 5))
            plt.suptitle(
                'Taste ' + dig_in_names[l_o_taste_ind] + ' Delivery ' + str(l_o_delivery_ind))
            # Fit a Gaussian mixture model with the number of dimensions = number of neurons
            all_taste_gmm = dict()
            for tr_i in range(len(train_data)):
                tr_t_i = train_tastes[tr_i]
                tr_e_i = train_epochs[tr_i]
                train_taste_data = train_data[tr_i]
                if np.min(all_train_data) >= 0:
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
                all_taste_gmm[tr_i] = gm
            
            # Calculate decoding probabilities for given jackknifed trial

            # Type 1: Bins of firing rates across the epoch of response
            # ___Pass inputs to parallel computation on probabilities
            inputs = zip(list_tb_fr, itertools.repeat(num_cat),
                         itertools.repeat(all_taste_gmm), itertools.repeat(p_t_e))
            pool = Pool(4)
            tb_decode_prob = pool.map(
                dp.segment_taste_decode_dependent_parallelized, inputs)
            pool.close()
            tb_decode_array = np.squeeze(np.array(tb_decode_prob)).T
            # ___Plot decode results
            for cat_i in range(num_cat):
                plt.plot(new_time_bins+deliv_cp[0], tb_decode_array[cat_i, :],
                         label=cat_names[cat_i], color=t_e_colors[cat_i])
                plt.fill_between(
                    new_time_bins+deliv_cp[0], tb_decode_array[cat_i, :], color=t_e_colors[cat_i], alpha=0.5, label='_')
            plt.ylabel('P(Category)')
            plt.ylim([-0.1, 1.1])
            plt.xlabel('Time (ms)')
            plt.legend(loc='upper right')
            # ___Calculate the average fraction of the true category that was decoded correctly and store
            cat_max_inds = np.argmax(tb_decode_array, 0)
            cat_decode_fracs = [len(np.where(cat_max_inds == cat_i_decode)[
                                      0])/len(new_time_bins) for cat_i_decode in range(num_cat)]
            taste_decode_fracs = []
            for taste_i in range(num_tastes):
                taste_decode_fracs.append(np.sum(np.array(cat_decode_fracs)[np.where(np.array(train_tastes) == taste_i)[0]]))
            trial_decode_storage[l_o_ind, :] = cat_decode_fracs
            trial_taste_decode_storage[l_o_ind,:] = taste_decode_fracs
            # ___Determine if in the trial the delivered taste is best
            best_taste = np.argmax(taste_decode_fracs)
            if (best_taste == l_o_taste_ind):
                trial_success_storage[l_o_ind] = 1
            # Save decoding figure
            plt.tight_layout()
            f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' +
                          str(l_o_taste_ind) + 'epoch_' + str(e_i) + '_delivery_' +
                          str(l_o_delivery_ind) + '.png'))
            f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' +
                          str(l_o_taste_ind) + 'epoch_' + str(e_i) + '_delivery_' + 
                          str(l_o_delivery_ind) + '.svg'))
            plt.close(f_loo)
        
        # Once all trials are decoded, save decoding success results
        np.savetxt(os.path.join(decoder_save_dir,
                   'success_by_trial.csv'), trial_success_storage, delimiter=',')
        np.savetxt(os.path.join(decoder_save_dir,
                   'mean_cat_decode_components.csv'), trial_decode_storage, delimiter=',')
        np.savetxt(os.path.join(decoder_save_dir,
                   'mean_taste_decode_components.csv'), trial_taste_decode_storage, delimiter=',')
        
    # Calculate overall decoding success by component count and taste
    taste_success_percent = np.round(100*np.nanmean(trial_success_storage),2)
        
    #Plot the success by taste
    taste_success = np.zeros(num_tastes)
    
    
    # Plot the overall success results for different component counts
    f_epochs = plt.figure(figsize=(5, 5))
    plt.bar(np.arange(num_tastes), taste_success_percent)
    plt.xticks(np.arange(num_tastes), labels=dig_in_names)
    plt.ylim([0, 100])
    plt.axhline(100/num_tastes, linestyle='dashed',
                color='k', alpha=0.75, label='Chance')
    plt.legend()
    plt.xlabel('Taste')
    plt.ylabel('Percent')
    plt.title('Decoding Success')
    f_epochs.savefig(os.path.join(decoder_save_dir, 'gmm_success.png'))
    f_epochs.savefig(os.path.join(decoder_save_dir, 'gmm_success.svg'))
    plt.close(f_epochs)

    return taste_success_percent
    

def naive_bayes_decoding(num_neur, tastant_spike_times, cp_raster_inds,
                         tastant_fr_dist, all_trial_inds, dig_in_names,
                         start_dig_in_times, pre_taste_dt, post_taste_dt,
                         e_skip_dt, e_len_dt, save_dir, epochs_to_analyze=[]):
    """This function trains a Gaussian Naive Bayes decoder to decode different 
    taste epochs from activity.
    INPUTS:
            - num_neur: number of neurons in dataset
            - tastant_spike_times: spike times for each tastant delivery
            - cp_raster_inds: changepoint times for all taste deliveries
            - tastant_fr_dist: firing rate distribution to fit over (train set)
            - all_trial_inds: indices of all trials used in testing the fit
            - dig_in_names: taste names
            - start_dig_in_times: start of each tastant delivery
            - pre_taste_dt: ms before taste delivery in cp_raster_inds
            - post_taste_dt: ms after taste delivery in cp_raster_inds
            - save_dir: directory where to save results
            - epochs_to_analyze: array of which epochs to analyze
    OUTPUTS:
            - Plots of decoder results on individual trials as well as overall success
                    metrics.
            - epoch_success_storage: vector of length number of epochs containing success
                    percentages overall.
            - epoch_success_by_taste: array of size num_epochs x num_tastes containing
                    success percentages by decoded taste by epoch.
    """

    print("\t\tTesting NB Decoder.")
    # TODO: Handle taste selective neurons

    # Variables
    num_tastes = len(start_dig_in_times)
    num_cp = len(tastant_fr_dist[0][0])
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    half_len = np.ceil(e_len_dt/2).astype('int')

    # Jackknife decoding total number of trials
    total_trials = np.sum([len(all_trial_inds[t_i])
                          for t_i in range(num_tastes)])
    total_trial_inds = np.arange(total_trials)
    all_trial_taste_inds = []
    for t_i in range(num_tastes):
        all_trial_taste_inds.extend(list(t_i*np.ones(len(all_trial_inds[t_i]))))
    all_trial_delivery_inds = []
    for t_i in range(num_tastes):
        all_trial_delivery_inds.extend(list(all_trial_inds[t_i]))

    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, len(epochs_to_analyze)))

    bayes_storage = os.path.join(save_dir, 'Naive_Bayes_Decoder_Tests')
    if not os.path.isdir(bayes_storage):
        os.mkdir(bayes_storage)

    epoch_success_storage = np.zeros(len(epochs_to_analyze))
    epoch_decode_storage = []

    for e_ind, e_i in enumerate(epochs_to_analyze):  # By epoch conduct decoding
        print('\t\t\tDecoding Epoch ' + str(e_i))

        epoch_decode_save_dir = os.path.join(
            bayes_storage, 'decode_prob_epoch_' + str(e_i))
        if not os.path.isdir(epoch_decode_save_dir):
            os.mkdir(epoch_decode_save_dir)

        trial_decodes = os.path.join(
            epoch_decode_save_dir, 'Individual_Trials')
        if not os.path.isdir(trial_decodes):
            os.mkdir(trial_decodes)

        try:  # Try to import the decoding results
            trial_success_storage = []
            with open(os.path.join(epoch_decode_save_dir, 'success_by_trial.csv'), newline='') as successtrialfile:
                filereader = csv.reader(
                    successtrialfile, delimiter=',', quotechar='|')
                for row in filereader:
                    trial_success_storage.append(np.array(row).astype('float'))
            trial_success_storage = np.array(trial_success_storage).squeeze()

            trial_decode_storage = []
            with open(os.path.join(epoch_decode_save_dir, 'mean_taste_decode_components.csv'), newline='') as decodefile:
                filereader = csv.reader(
                    decodefile, delimiter=',', quotechar='|')
                for row in filereader:
                    trial_decode_storage.append(np.array(row).astype('float'))
            trial_decode_storage = np.array(trial_decode_storage).squeeze()

            epoch_decode_storage.append(trial_decode_storage)

            # Calculate overall decoding success by component count
            taste_success_percent = np.round(
                100*np.nanmean(trial_success_storage), 2)
            epoch_success_storage[e_ind] = taste_success_percent

        except:  # Run decoding

            # Fraction of the trial decoded as each taste for each component count
            trial_decode_storage = np.zeros((total_trials, num_tastes))
            # Binary storage of successful decodes (max fraction of trial = taste delivered)
            trial_success_storage = np.zeros(total_trials)

            print('\t\tPerforming LOO Decoding')

            # Which trial is being left out for decoding
            for l_o_ind in tqdm.tqdm(total_trial_inds):
                l_o_taste_ind = all_trial_taste_inds[l_o_ind].astype(
                    'int')  # Taste of left out trial
                l_o_delivery_ind = all_trial_delivery_inds[l_o_ind].astype(
                    'int')  # Delivery index of left out trial

                # Collect trial data for decoder
                taste_state_inds = []  # matching of index
                taste_state_labels = []  # matching of label
                train_fr_data = []  # firing rate vector storage
                # firing rate vector labelled indices (from taste_state_inds)
                train_fr_labels = []
                for t_i in range(num_tastes):
                    t_name = dig_in_names[t_i]
                    # Store the current iteration label and index
                    taste_state_labels.extend([t_name + '_' + str(e_i)])
                    taste_state_inds.extend([t_i])
                    # Store firing rate vectors for each train set delivery
                    for d_i, trial_ind in enumerate(all_trial_inds[t_i]):
                        if (d_i == l_o_delivery_ind) and (t_i == l_o_taste_ind):
                            train_fr_data.extend([])  # Basically do nothing
                        else:
                            tb_fr = tastant_fr_dist[t_i][d_i][e_i]
                            list_tb_fr = list(tb_fr.T)
                            train_fr_data.extend(list_tb_fr)
                            bst_hz_labels = list(t_i*np.ones(len(list_tb_fr)))
                            train_fr_labels.extend(bst_hz_labels)

                # Train a Bayesian decoder on all trials but the left out one
                gnb = GaussianNB()
                gnb.fit(np.array(train_fr_data), np.array(train_fr_labels))

                # Now perform decoding of the left out trial with the decoder
                taste_cp = cp_raster_inds[l_o_taste_ind]
                # length num_neur list of lists
                start_taste_i = start_dig_in_times[l_o_taste_ind][l_o_delivery_ind]
                deliv_cp = taste_cp[l_o_delivery_ind, :] - pre_taste_dt
                start_epoch = int(deliv_cp[e_i])
                end_epoch = int(deliv_cp[e_i+1])
                sdi = start_taste_i + start_epoch
                epoch_len = end_epoch - start_epoch
                if epoch_len > 0:
                    # Decode 50 ms bins, skip ahead 25 ms
                    new_time_bins = np.arange(half_len, epoch_len-half_len, half_len)
                    f_loo = plt.figure(figsize=(5, 5))
                    plt.suptitle(
                        'Taste ' + dig_in_names[l_o_taste_ind] + ' Delivery ' + str(l_o_delivery_ind))

                    # ___Grab neuron firing rates in sliding bins
                    td_i_bin = np.zeros((num_neur, epoch_len+1))
                    for n_i in range(num_neur):
                        n_i_spike_times = np.array(
                            tastant_spike_times[l_o_taste_ind][l_o_delivery_ind][n_i] - sdi).astype('int')
                        keep_spike_times = n_i_spike_times[np.where(
                            (0 <= n_i_spike_times)*(epoch_len >= n_i_spike_times))[0]]
                        td_i_bin[n_i, keep_spike_times] = 1
                    if len(new_time_bins) > 0:
                        # Calculate the firing rate vectors for these bins
                        tb_fr = np.zeros((num_neur, len(new_time_bins)))
                        for tb_i, tb in enumerate(new_time_bins):
                            tb_fr[:, tb_i] = np.sum(
                                td_i_bin[:, tb-half_len:tb+half_len], 1)/(half_len*2/1000)
                    else:
                        tb_fr = np.expand_dims(np.sum(td_i_bin,1)/((epoch_len+1)/1000),1)
                    list_tb_fr = list(tb_fr.T)
                    # Predict the results
                    deliv_test_predictions = gnb.predict_proba(list_tb_fr)
                    if len(new_time_bins) > 0:
                        taste_max_inds = np.argmax(deliv_test_predictions, 1)
                        taste_decode_fracs = [len(np.where(taste_max_inds == t_i_decode)[
                                                  0])/len(new_time_bins) for t_i_decode in range(num_tastes)]
                    else:
                        taste_decode_fracs = list(deliv_test_predictions[0])
                    trial_decode_storage[l_o_ind, :] = taste_decode_fracs
                    # ___Plot decode results
                    if len(new_time_bins) > 0:
                        for t_i_plot in range(num_tastes):
                            plt.plot(new_time_bins+deliv_cp[e_i], deliv_test_predictions[:, t_i_plot],
                                     label=dig_in_names[t_i_plot], color=taste_colors[t_i_plot])
                            plt.fill_between(
                                new_time_bins+deliv_cp[e_i], deliv_test_predictions[:, t_i_plot], color=taste_colors[t_i_plot], alpha=0.5, label='_')
                    else:
                        for t_i_plot in range(num_tastes):
                            plt.axhline(deliv_test_predictions[0][t_i_plot],
                                     label=dig_in_names[t_i_plot], color=taste_colors[t_i_plot])
                            plt.fill_between([0,1],[0,0],[deliv_test_predictions[0][t_i_plot],deliv_test_predictions[0][t_i_plot]],
                                             color=taste_colors[t_i_plot],alpha=0.5,label='_')
                    plt.ylabel('P(Taste)')
                    plt.ylim([-0.1, 1.1])
                    plt.xlabel('Time (ms)')
                    plt.legend(loc='upper right')
                    # ___Calculate the fraction of time in the epoch of each taste being best
                    best_taste = np.argmax(taste_decode_fracs)
                    if best_taste == l_o_taste_ind:
                        trial_success_storage[l_o_ind] = 1

                    # Save decoding figure
                    plt.tight_layout()
                    f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' + str(
                        l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.png'))
                    f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' + str(
                        l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.svg'))
                    plt.close(f_loo)

            # Once all trials are decoded, save decoding success results
            np.savetxt(os.path.join(epoch_decode_save_dir,
                       'success_by_trial.csv'), trial_success_storage, delimiter=',')
            np.savetxt(os.path.join(epoch_decode_save_dir,
                       'mean_taste_decode_components.csv'), trial_decode_storage, delimiter=',')
            epoch_decode_storage.append(trial_decode_storage)

            # Calculate overall decoding success by component count
            taste_success_percent = np.round(
                100*np.nanmean(trial_success_storage), 2)
            epoch_success_storage[e_ind] = taste_success_percent

    # Plot the success results for different component counts across epochs
    f_epochs = plt.figure(figsize=(5, 5))
    plt.bar(np.arange(len(epochs_to_analyze)), epoch_success_storage)
    epoch_labels = ['Epoch ' + str(e_i) for e_i in epochs_to_analyze]
    plt.xticks(np.arange(len(epochs_to_analyze)), labels=epoch_labels)
    plt.ylim([0, 100])
    plt.axhline(100/num_tastes, linestyle='dashed',
                color='k', alpha=0.75, label='Chance')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.title('Decoding Success')
    f_epochs.savefig(os.path.join(bayes_storage, 'nb_success.png'))
    f_epochs.savefig(os.path.join(bayes_storage, 'nb_success.svg'))
    plt.close(f_epochs)

    # Plot the by-taste success results
    f_percents = plt.figure(figsize=(5, 5))
    epoch_success_by_taste = np.zeros((len(epochs_to_analyze), num_tastes))
    for e_ind, e_i in enumerate(epochs_to_analyze):
        epoch_decode_percents = epoch_decode_storage[e_ind]
        success_by_taste = np.zeros(num_tastes)
        for t_i in range(num_tastes):
            taste_trials = np.where(np.array(all_trial_taste_inds) == t_i)[0]
            taste_trial_results_bin = np.zeros(len(taste_trials))
            for tt_ind, tt_i in enumerate(taste_trials):
                trial_decode_results = epoch_decode_percents[tt_i, :]
                best_taste = np.where(
                    trial_decode_results == np.max(trial_decode_results))[0]
                if len(best_taste) == 1:
                    if best_taste[0] == t_i:
                        taste_trial_results_bin[tt_ind] = 1
                else:
                    # Taste is one of the predicted tastes in a "tie"
                    if len(np.where(best_taste == t_i)[0]) > 0:
                        taste_trial_results_bin[tt_ind] = 1
            success_by_taste[t_i] = 100*np.nanmean(taste_trial_results_bin)
        epoch_success_by_taste[e_ind, :] = success_by_taste
        plt.scatter(np.arange(num_tastes), success_by_taste,
                    label='Epoch ' + str(e_i), color=epoch_colors[e_ind, :])
        plt.plot(np.arange(num_tastes), success_by_taste, label='_',
                 color=epoch_colors[e_ind, :], linestyle='dashed', alpha=0.75)
    np.savetxt(os.path.join(bayes_storage, 'epoch_success_by_taste.csv'),
               epoch_success_by_taste, delimiter=',')
    plt.axhline(100/num_tastes, label='Chance',
                color='k', linestyle='dashed', alpha=0.75)
    plt.legend(loc='lower left')
    plt.xlabel('Taste')
    plt.xticks(np.arange(num_tastes), dig_in_names)
    plt.ylabel('Percent')
    plt.title('Decoding Success by Taste')
    f_percents.savefig(os.path.join(bayes_storage, 'nb_success_by_taste.png'))
    f_percents.savefig(os.path.join(bayes_storage, 'nb_success_by_taste.svg'))
    plt.close(f_percents)

    return epoch_success_storage, epoch_success_by_taste

def svm_classification(num_neur, tastant_spike_times, cp_raster_inds,
                         tastant_fr_dist, all_trial_inds, dig_in_names,
                         start_dig_in_times, pre_taste_dt, post_taste_dt,
                         e_skip_dt, e_len_dt, save_dir, epochs_to_analyze=[]):
    """This function trains an SVM to classify different tastes from firing rates.
    It is run in a LOO fashion to classify one left out delivery trial based on the
    fit for all the others.
    INPUTS:
            - num_neur: number of neurons in dataset
            - tastant_spike_times: spike times for each tastant delivery
            - cp_raster_inds: changepoint times for all taste deliveries
            - tastant_fr_dist: firing rate distribution to fit over (train set)
            - all_trial_inds: indices of all trials used in testing the fit
            - dig_in_names: taste names
            - start_dig_in_times: start of each tastant delivery
            - pre_taste_dt: ms before taste delivery in cp_raster_inds
            - post_taste_dt: ms after taste delivery in cp_raster_inds
            - save_dir: directory where to save results
            - epochs_to_analyze: array of which epochs to analyze
    OUTPUTS:
            - Plots of decoder results on individual trials as well as overall success
                    metrics.
            - epoch_success_storage: vector of length number of epochs containing success
                    percentages overall.
            - epoch_success_by_taste: array of size num_epochs x num_tastes containing
                    success percentages by decoded taste by epoch.
    """
    
    print("\t\tTesting SVM.")
    
    # Variables
    num_tastes = len(start_dig_in_times)
    num_cp = len(tastant_fr_dist[0][0])
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    half_len = np.ceil(e_len_dt/2).astype('int')

    # Jackknife decoding total number of trials
    total_trials = np.sum([len(all_trial_inds[t_i])
                          for t_i in range(num_tastes)])
    total_trial_inds = np.arange(total_trials)
    all_trial_taste_inds = []
    for t_i in range(num_tastes):
        all_trial_taste_inds.extend(list(t_i*np.ones(len(all_trial_inds[t_i]))))
    all_trial_delivery_inds = []
    for t_i in range(num_tastes):
        all_trial_delivery_inds.extend(list(all_trial_inds[t_i]))

    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, len(epochs_to_analyze)))

    svm_storage = os.path.join(save_dir, 'SVM_Tests')
    if not os.path.isdir(svm_storage):
        os.mkdir(svm_storage)
        
    epoch_success_storage = np.zeros(len(epochs_to_analyze))
    epoch_decode_storage = []

    for e_ind, e_i in enumerate(epochs_to_analyze):  # By epoch conduct decoding
        print('\t\t\tDecoding Epoch ' + str(e_i))

        epoch_decode_save_dir = os.path.join(
            svm_storage, 'classify_epoch_' + str(e_i))
        if not os.path.isdir(epoch_decode_save_dir):
            os.mkdir(epoch_decode_save_dir)
            
        trial_decodes = os.path.join(
            epoch_decode_save_dir, 'Individual_Trials')
        if not os.path.isdir(trial_decodes):
            os.mkdir(trial_decodes)
            
        try:  # Try to import the previous results
            trial_success_storage = []
            with open(os.path.join(epoch_decode_save_dir, 'success_by_trial.csv'), newline='') as successtrialfile:
                filereader = csv.reader(
                    successtrialfile, delimiter=',', quotechar='|')
                for row in filereader:
                    trial_success_storage.append(np.array(row).astype('float'))
            trial_success_storage = np.array(trial_success_storage).squeeze()

            trial_decode_storage = []
            with open(os.path.join(epoch_decode_save_dir, 'mean_taste_decode_components.csv'), newline='') as decodefile:
                filereader = csv.reader(
                    decodefile, delimiter=',', quotechar='|')
                for row in filereader:
                    trial_decode_storage.append(np.array(row).astype('float'))
            trial_decode_storage = np.array(trial_decode_storage).squeeze()

            epoch_decode_storage.append(trial_decode_storage)

            # Calculate overall decoding success by component count
            taste_success_percent = np.round(
                100*np.nanmean(trial_success_storage), 2)
            epoch_success_storage[e_ind] = taste_success_percent
            
        except: # Run the svm classification
        
            # Fraction of the trial decoded as each taste for each component count
            trial_decode_storage = np.zeros((total_trials, num_tastes))
            # Binary storage of successful decodes (max fraction of trial = taste delivered)
            trial_success_storage = np.zeros(total_trials)

            print('\t\tPerforming LOO Classification')
            
            # Which trial is being left out for decoding
            for l_o_ind in tqdm.tqdm(total_trial_inds):
                l_o_taste_ind = all_trial_taste_inds[l_o_ind].astype(
                    'int')  # Taste of left out trial
                l_o_delivery_ind = all_trial_delivery_inds[l_o_ind].astype(
                    'int')  # Delivery index of left out trial

                # Collect trial data for decoder
                taste_state_inds = []  # matching of index
                taste_state_labels = []  # matching of label
                train_fr_data = []  # firing rate vector storage
                # firing rate vector labelled indices (from taste_state_inds)
                train_fr_labels = []
                for t_i in range(num_tastes):
                    t_name = dig_in_names[t_i]
                    # Store the current iteration label and index
                    taste_state_labels.extend([t_name + '_' + str(e_i)])
                    taste_state_inds.extend([t_i])
                    # Store firing rate vectors for each train set delivery
                    for d_i, trial_ind in enumerate(all_trial_inds[t_i]):
                        if (d_i == l_o_delivery_ind) and (t_i == l_o_taste_ind):
                            train_fr_data.extend([])  # Basically do nothing
                        else:
                            tb_fr = tastant_fr_dist[t_i][d_i][e_i]
                            list_tb_fr = list(tb_fr.T)
                            train_fr_data.extend(list_tb_fr)
                            bst_hz_labels = list(t_i*np.ones(len(list_tb_fr)))
                            train_fr_labels.extend(bst_hz_labels)
                            
                # Train an SVM on all trials but left out
                clf = svm.SVC()
                clf.fit(train_fr_data, train_fr_labels)
                
                # Now perform decoding of the left out trial with the decoder
                taste_cp = cp_raster_inds[l_o_taste_ind]
                # length num_neur list of lists
                start_taste_i = start_dig_in_times[l_o_taste_ind][l_o_delivery_ind]
                deliv_cp = taste_cp[l_o_delivery_ind, :] - pre_taste_dt
                start_epoch = int(deliv_cp[e_i])
                end_epoch = int(deliv_cp[e_i+1])
                sdi = start_taste_i + start_epoch
                epoch_len = end_epoch - start_epoch
                if epoch_len > 0:
                    # Decode 50 ms bins, skip ahead 25 ms
                    new_time_bins = np.arange(half_len, epoch_len-half_len, half_len)
                    f_loo = plt.figure(figsize=(5, 5))
                    plt.suptitle(
                        'Taste ' + dig_in_names[l_o_taste_ind] + ' Delivery ' + str(l_o_delivery_ind))

                    # ___Grab neuron firing rates in sliding bins
                    td_i_bin = np.zeros((num_neur, epoch_len+1))
                    for n_i in range(num_neur):
                        n_i_spike_times = np.array(
                            tastant_spike_times[l_o_taste_ind][l_o_delivery_ind][n_i] - sdi).astype('int')
                        keep_spike_times = n_i_spike_times[np.where(
                            (0 <= n_i_spike_times)*(epoch_len >= n_i_spike_times))[0]]
                        td_i_bin[n_i, keep_spike_times] = 1
                    if len(new_time_bins) > 0:
                        # Calculate the firing rate vectors for these bins
                        tb_fr = np.zeros((num_neur, len(new_time_bins)))
                        for tb_i, tb in enumerate(new_time_bins):
                            tb_fr[:, tb_i] = np.sum(
                                td_i_bin[:, tb-half_len:tb+half_len], 1)/(half_len*2/1000)
                    else:
                        tb_fr = np.expand_dims(np.sum(td_i_bin,1)/((epoch_len+1)/1000),1)
                    list_tb_fr = list(tb_fr.T)
                    # Predict the results
                    predicted_classification = clf.predict(list_tb_fr)
                    if len(new_time_bins) > 0:
                        taste_decode_fracs = [len(np.where(predicted_classification == t_i_decode)[
                                                  0])/len(new_time_bins) for t_i_decode in range(num_tastes)]
                    else:
                        taste_decode_fracs = [len(np.where(predicted_classification == t_i_decode)[
                                                  0]) for t_i_decode in range(num_tastes)] #should be binary unit vector
                    trial_decode_storage[l_o_ind, :] = taste_decode_fracs
                    # ___Plot classification results
                    for t_i_plot in range(num_tastes):
                        plt.axhline(taste_decode_fracs[t_i_plot],
                                 label=dig_in_names[t_i_plot], color=taste_colors[t_i_plot])
                        plt.fill_between([0,1],[0,0],[taste_decode_fracs[t_i_plot],taste_decode_fracs[t_i_plot]],
                                         color=taste_colors[t_i_plot],alpha=0.5,label='_')
                    plt.ylabel('Fraction Classified')
                    plt.ylim([-0.1, 1.1])
                    plt.legend(loc='upper right')
                    # ___Store trial success
                    best_taste = np.argmax(taste_decode_fracs)
                    if best_taste == l_o_taste_ind:
                        trial_success_storage[l_o_ind] = 1
                            
                    # Save decoding figure
                    plt.tight_layout()
                    f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' + str(
                        l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.png'))
                    f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' + str(
                        l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.svg'))
                    plt.close(f_loo)
                    
            # Once all trials are decoded, save decoding success results
            np.savetxt(os.path.join(epoch_decode_save_dir,
                       'success_by_trial.csv'), trial_success_storage, delimiter=',')
            np.savetxt(os.path.join(epoch_decode_save_dir,
                       'mean_taste_decode_components.csv'), trial_decode_storage, delimiter=',')
            epoch_decode_storage.append(trial_decode_storage)

            # Calculate overall decoding success by component count
            taste_success_percent = np.round(
                100*np.nanmean(trial_success_storage), 2)
            epoch_success_storage[e_ind] = taste_success_percent
            
    # Plot the success results for different component counts across epochs
    f_epochs = plt.figure(figsize=(5, 5))
    plt.bar(np.arange(len(epochs_to_analyze)), epoch_success_storage)
    epoch_labels = ['Epoch ' + str(e_i) for e_i in epochs_to_analyze]
    plt.xticks(np.arange(len(epochs_to_analyze)), labels=epoch_labels)
    plt.ylim([0, 100])
    plt.axhline(100/num_tastes, linestyle='dashed',
                color='k', alpha=0.75, label='Chance')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.title('Decoding Success')
    f_epochs.savefig(os.path.join(svm_storage, 'svm_success.png'))
    f_epochs.savefig(os.path.join(svm_storage, 'svm_success.svg'))
    plt.close(f_epochs)

    # Plot the by-taste success results
    f_percents = plt.figure(figsize=(5, 5))
    epoch_success_by_taste = np.zeros((len(epochs_to_analyze), num_tastes))
    for e_ind, e_i in enumerate(epochs_to_analyze):
        epoch_decode_percents = epoch_decode_storage[e_ind]
        success_by_taste = np.zeros(num_tastes)
        for t_i in range(num_tastes):
            taste_trials = np.where(np.array(all_trial_taste_inds) == t_i)[0]
            taste_trial_results_bin = np.zeros(len(taste_trials))
            for tt_ind, tt_i in enumerate(taste_trials):
                trial_decode_results = epoch_decode_percents[tt_i, :]
                best_taste = np.where(
                    trial_decode_results == np.max(trial_decode_results))[0]
                if len(best_taste) == 1:
                    if best_taste[0] == t_i:
                        taste_trial_results_bin[tt_ind] = 1
                else:
                    # Taste is one of the predicted tastes in a "tie"
                    if len(np.where(best_taste == t_i)[0]) > 0:
                        taste_trial_results_bin[tt_ind] = 1
            success_by_taste[t_i] = 100*np.nanmean(taste_trial_results_bin)
        epoch_success_by_taste[e_ind, :] = success_by_taste
        plt.scatter(np.arange(num_tastes), success_by_taste,
                    label='Epoch ' + str(e_i), color=epoch_colors[e_ind, :])
        plt.plot(np.arange(num_tastes), success_by_taste, label='_',
                 color=epoch_colors[e_ind, :], linestyle='dashed', alpha=0.75)
    np.savetxt(os.path.join(svm_storage, 'epoch_success_by_taste.csv'),
               epoch_success_by_taste, delimiter=',')
    plt.axhline(100/num_tastes, label='Chance',
                color='k', linestyle='dashed', alpha=0.75)
    plt.legend(loc='lower left')
    plt.xlabel('Taste')
    plt.xticks(np.arange(num_tastes), dig_in_names)
    plt.ylabel('Percent')
    plt.title('Decoding Success by Taste')
    f_percents.savefig(os.path.join(svm_storage, 'svm_success_by_taste.png'))
    f_percents.savefig(os.path.join(svm_storage, 'svm_success_by_taste.svg'))
    plt.close(f_percents)
    
    return epoch_success_storage, epoch_success_by_taste

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will calculate the BIC score"""
    return -estimator.bic(X)


def plot_gmm_bic_scores(taste_bic_scores, component_counts, e_i,
                        dig_in_names, save_dir):
    """This function plots decoder BIC scores for each number of components
    and returns the number of components that, on average, provides the lowest
    BIC score when fit to half the taste deliveries.
    INPUTS:
            - taste_bic_scores: array of [num_components x num_tastes] with BIC scores
            - component_counts: number of components for each bic score result
            - e_i: index of epoch being tested
            - dig_in_names: names of each taste
            - save_dir: where to save results/figures
    OUTPUTS:
            - best_component_count: the component count that on average provides
                    the lowest BIC score across tastes
    """
    _, num_tastes = np.shape(taste_bic_scores)

    # Plot the BIC scores by taste
    f = plt.figure(figsize=(8, 8))
    for t_i in range(num_tastes):
        plt.plot(component_counts,
                 taste_bic_scores[:, t_i], label=dig_in_names[t_i], alpha=0.5)
    plt.plot(component_counts, np.nanmean(taste_bic_scores, 1),
             color='k', linestyle='dashed', alpha=1, label='Mean')
    plt.plot(component_counts, np.nanmean(
        taste_bic_scores[:, :-1], 1), color='k', linestyle='dotted', alpha=1, label='True Taste Mean')
    plt.legend()
    plt.title('GMM Fit BIC Scores')
    plt.xlabel('# Components')
    plt.ylabel('BIC Score')
    plt.tight_layout()
    f.savefig(os.path.join(save_dir, 'gmm_fit_bic_scores.png'))
    f.savefig(os.path.join(save_dir, 'gmm_fit_bic_scores.svg'))
    plt.close(f)

    # The best number of components is that which has the lowest average BIC for the true tastes
    best_component_count = component_counts[np.argmin(
        np.nanmean(taste_bic_scores[:, :-1], 1))]

    return best_component_count


def plot_all_results(epochs_to_analyze, gmm_success_rates, nb_success_rates, 
                     svm_success_rates, num_tastes, save_dir):
    """This function plots the results of both GMM and NB decoder tests on one
    set of axes.
    INPUTS:
            - epochs_to_analyze: which epochs were analyzed
            - gmm_success_rates: vector of success by epoch using gmm
            - nb_success_rates: vector of success by epoch using nb
            - svm_success_rates: vector of success by epoch using svm
            - num_tastes: number of tastes
            - save_dir: where to save plots
    OUTPUTS: Figure with model results.
    """

    cmap = colormaps['cool']
    model_colors = cmap(np.linspace(0, 1, 3))

    model_results_comb = plt.figure(figsize=(8, 8))
    num_epochs = len(epochs_to_analyze)
    plt.plot(np.arange(num_epochs), gmm_success_rates,
             label='GMM', color=model_colors[0, :])
    plt.plot(np.arange(num_epochs), nb_success_rates,
             label='NB', color=model_colors[1, :])
    plt.plot(np.arange(num_epochs), svm_success_rates,
             label='SVM', color=model_colors[2, :])
    plt.axhline(100/num_tastes, label='Chance',
                linestyle='dashed', color='k', alpha=0.75)
    gmm_avg_success = np.nanmean(gmm_success_rates)
    plt.axhline(gmm_avg_success, label='GMM Mean',
                linestyle='dashed', alpha=0.75, color=model_colors[0, :])
    nb_avg_success = np.nanmean(nb_success_rates)
    plt.axhline(nb_avg_success, label='NB Mean', linestyle='dashed',
                alpha=0.75, color=model_colors[1, :])
    svm_avg_success = np.nanmean(svm_success_rates)
    plt.axhline(svm_avg_success, label='SVM Mean', linestyle='dashed',
                alpha=0.75, color=model_colors[2, :])
    plt.xticks(np.arange(len(epochs_to_analyze)), epochs_to_analyze)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Percent of Trials Successfully Decoded')
    plt.title('Model Success by Epoch')
    model_results_comb.savefig(os.path.join(
        save_dir, 'Decoder_Success_Results.png'))
    model_results_comb.savefig(os.path.join(
        save_dir, 'Decoder_Success_Results.svg'))
    plt.close(model_results_comb)

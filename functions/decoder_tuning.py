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
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.naive_bayes import GaussianNB
#from random import choices, sample
import functions.decode_parallel as dp
from sklearn import svm


def test_decoder_params(dig_in_names, start_dig_in_times, num_neur, tastant_spike_times,
                        tastant_fr_dist, cp_raster_inds, pre_taste_dt, post_taste_dt,
                        epochs_to_analyze, taste_select_neur, e_skip_dt, e_len_dt, 
                        max_hz, save_dir):
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
                                                                dig_in_names, max_hz, save_dir,
                                                                epochs_to_analyze)

    # # Run Naive Bayes decoder to test
    # nb_success_rates, nb_success_rates_by_taste = naive_bayes_decoding(num_neur, tastant_spike_times,
    #                                                                     cp_raster_inds, tastant_fr_dist,
    #                                                                     all_trial_inds, dig_in_names,
    #                                                                     start_dig_in_times, pre_taste_dt,
    #                                                                     post_taste_dt, e_skip_dt, e_len_dt,
    #                                                                     max_hz, save_dir, epochs_to_analyze)
    
    # # Run SVM classifier to test
    # svm_success_rates, svm_success_rates_by_taste = svm_classification(num_neur, tastant_spike_times, cp_raster_inds,
    #                                                                    tastant_fr_dist, all_trial_inds, dig_in_names,
    #                                                                    start_dig_in_times, pre_taste_dt, post_taste_dt,
    #                                                                    e_skip_dt, e_len_dt, max_hz, 
    #                                                                    save_dir, epochs_to_analyze)
    # # Both Models Plot
    # plot_all_results(epochs_to_analyze, gmm_success_rates, nb_success_rates, svm_success_rates,
    #                  num_tastes, save_dir)


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
            taste_data_combined = []
            taste_data_labels = []
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
                taste_data_combined.extend(train_taste_data)
                taste_data_labels.extend(list(t_i*np.ones(len(train_taste_data))))
                if len(train_taste_data) > 0:
                    if t_i < num_tastes-1:
                        if np.max(train_taste_data) > max_fr:
                            max_fr = np.max(train_taste_data)
                        all_data.extend(train_taste_data)
                        all_data_labels.extend(
                            list(t_i*np.ones(len(train_taste_data))))
            
            # # Plot Neuron Firing Rates
            # f_true, ax_true = plt.subplots(nrows = int(neur_sqrt),ncols = int(neur_sqrt), 
            #                                figsize=(5*int(neur_sqrt), int(neur_sqrt)*5),
            #                                sharex = True)
            # neur_map = np.reshape(np.arange(square_num),(neur_sqrt,neur_sqrt))
            # for n_i in range(num_neur):
            #     neur_row, neur_col = np.argwhere(neur_map == n_i)[0]
            #     ax_true[neur_row,neur_col].set_title('Neuron ' + str(n_i))
            #     ax_true[neur_row,neur_col].set_xlabel('FR')
            #     ax_true[neur_row,neur_col].set_ylabel('Probability')
            #     for t_i in range(num_tastes):
            #         neur_data = taste_data[t_i][n_i,:]
            #         ax_true[neur_row,neur_col].hist(neur_data, 10, density=True, histtype='bar', alpha=1/num_tastes, \
            #                 label=dig_in_names[t_i], color=taste_colors[t_i, :])
            #     if n_i == 0:
            #         ax_true[neur_row,neur_col].legend(loc='upper right')
            # f_true.tight_layout()
            # f_true.savefig(os.path.join(
            #     fr_save, 'FR_distributions_'+str(e_ind)+'.png'))
            # f_true.savefig(os.path.join(
            #     fr_save, 'FR_distributions_'+str(e_ind)+'.svg'))
            # plt.close(f_true)
            
            # # Scatter Plot Pairs of Neuron Firing Rates
            # for n_1 in range(num_neur-1):
            #     for n_2 in np.arange(n_1+1,num_neur):
            #         f_pair = plt.figure(figsize=(5,5))
            #         for t_i in range(num_tastes):
            #             neur_data1 = taste_data[t_i][n_1,:]
            #             neur_data2 = taste_data[t_i][n_2,:]
            #             plt.scatter(neur_data1,neur_data2,alpha=1/num_tastes, \
            #                         label=dig_in_names[t_i], color=taste_colors[t_i, :])
            #         plt.xlabel('Neuron ' + str(n_1))
            #         plt.ylabel('Neuron ' + str(n_2))
            #         plt.legend(loc='upper right')
            #         plt.tight_layout()
            #         plt.savefig(os.path.join(
            #             fr_save, 'FR_n_'+str(n_1)+'_n_'+str(n_2)+'.png'))
            #         plt.close(f_pair)
            
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
            

def run_decoder(num_neur, start_dig_in_times, tastant_fr_dist, all_trial_inds,
                tastant_spike_times, cp_raster_inds,
                pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names,
                max_hz, save_dir, epochs_to_analyze=[]):
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
            - max_hz: maximum firing rate in taste data
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
                train_data_combined = []
                train_data_labels = []
                true_taste_train_data = [] #Only true tastes - excluding "none"
                true_taste_train_labels = []
                #taste_bic_scores = np.zeros((len(component_counts),num_tastes))
                for t_i in range(num_tastes):
                    train_taste_data = []
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
                    if t_i < num_tastes-1:
                        true_taste_train_data.extend(train_taste_data)
                        true_taste_train_labels.extend(list(t_i*np.ones(len(train_taste_data))))
                    else: #None condition - augment with randomized data
                        neur_max = np.expand_dims(np.max(np.array(train_taste_data),0),1)
                        train_taste_data.extend(list((neur_max*np.random.rand(num_neur,100)).T))
                        train_taste_data.extend(list(((neur_max/10)*np.random.rand(num_neur,100)).T))
                        train_taste_data.extend(list((np.eye(num_neur)).T))
                    train_data.append(np.array(train_taste_data))
                    train_data_combined.extend(train_taste_data)
                    train_data_labels.extend(list(t_i*np.ones(len(train_taste_data))))
                    
                # Run PCA transform only on non-z-scored data
                if np.min(true_taste_train_data) >= 0:
                    #PCA
                    pca = PCA()
                    pca.fit(np.array(true_taste_train_data).T)
                    exp_var = pca.explained_variance_ratio_
                    num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
                    if num_components == 0:
                        num_components = 3
                    pca_reduce = PCA(num_components)
                    pca_reduce.fit(np.array(true_taste_train_data))

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
                    
                if np.min(true_taste_train_data) >= 0: #If it's not z-scored PCA to whiten
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
   

def naive_bayes_decoding(num_neur, tastant_spike_times, cp_raster_inds,
                         tastant_fr_dist, all_trial_inds, dig_in_names,
                         start_dig_in_times, pre_taste_dt, post_taste_dt,
                         e_skip_dt, e_len_dt, max_hz, save_dir, epochs_to_analyze=[]):
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
                            list_tb_fr = []
                            for tbfr_i in range(np.shape(tb_fr)[1]):
                                list_tb_fr.append(list(tb_fr[:,tbfr_i]))
                            if t_i < num_tastes - 1:
                                train_fr_data.extend(list_tb_fr)
                            else: #None condition - augment with randomized data
                                for a_i in range(100): #100 augmented sets spanning the full FR range
                                    list_tb_fr.append(list(max_hz*np.random.rand(num_neur)))
                                for a_i in range(100): #100 augmented sets spanning the low FR range
                                    list_tb_fr.append(list((max_hz/10)*np.random.rand(num_neur)))
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
                         e_skip_dt, e_len_dt, max_hz, save_dir, epochs_to_analyze=[]):
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
                            list_tb_fr = []
                            for tbfr_i in range(np.shape(tb_fr)[1]):
                                list_tb_fr.append(list(tb_fr[:,tbfr_i]))
                            if t_i < num_tastes - 1:
                                train_fr_data.extend(list_tb_fr)
                            else: #None condition - augment with randomized data
                                for a_i in range(100): #100 augmented sets spanning the full FR range
                                    list_tb_fr.append(list(max_hz*np.random.rand(num_neur)))
                                for a_i in range(100): #100 augmented sets spanning the low FR range
                                    list_tb_fr.append(list((max_hz/10)*np.random.rand(num_neur)))
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

def multistep_epoch_decoder(num_neur, start_dig_in_times, tastant_fr_dist, 
                all_trial_inds, tastant_spike_times, cp_raster_inds,
                pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names,
                max_hz, save_dir, epochs_to_analyze=[]):
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
            - max_hz: maximum firing rate in taste data
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
    print("\t\tTesting multi-step GMM Decoder.")
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

    # Save dirs
    decoder_save_dir = os.path.join(save_dir, 'Multistep_Epoch_GMM_Decoder_Tests')
    if not os.path.isdir(decoder_save_dir):
        os.mkdir(decoder_save_dir)
        
    trial_decodes = os.path.join(
        decoder_save_dir, 'Individual_Trials')
    if not os.path.isdir(trial_decodes):
        os.mkdir(trial_decodes)
        
    try:  # Try to import the decoding results
        trial_epoch_success_storage = []
        with open(os.path.join(decoder_save_dir, 'trial_epoch_success_storage.csv'), newline='') as successtrialfile:
            filereader = csv.reader(
                successtrialfile, delimiter=',', quotechar='|')
            for row in filereader:
                trial_epoch_success_storage.append(np.array(row).astype('float'))
        trial_epoch_success_storage = np.array(trial_epoch_success_storage).squeeze()
        
        print("\nEpoch Success:")
        print(np.sum(trial_epoch_success_storage)/(num_cp*total_trials))
        print("\nBy-Epoch Success:")
        print(np.sum(trial_epoch_success_storage,0)/(total_trials))
        
        trial_final_success_storage = []
        with open(os.path.join(decoder_save_dir, 'trial_final_success_storage.csv'), newline='') as successtrialfile:
            filereader = csv.reader(
                successtrialfile, delimiter=',', quotechar='|')
            for row in filereader:
                trial_final_success_storage.append(np.array(row).astype('float'))
        trial_final_success_storage = np.array(trial_final_success_storage).squeeze()
        
        print("\nTaste Success:")
        print(np.sum(trial_final_success_storage)/(num_cp*total_trials))
        print("\nBy-Epoch Success:")
        print(np.sum(trial_final_success_storage,0)/(total_trials))

    except:  # Run decoding
    
        # Fraction of the trial decoded as each taste for each component count
        trial_decode_storage = np.zeros((total_trials, num_cp, num_tastes))
        # Binary storage of successful decodes (max fraction of trial = taste delivered)
        trial_epoch_success_storage = np.zeros((total_trials, num_cp)) #is the decoded epoch correct?
        trial_final_success_storage = np.zeros((total_trials, num_cp)) #after the taste decode is the taste correct?
        
        print('\t\t\t\tPerforming LOO Decoding')
        # Which trial is being left out for decoding
        for l_o_ind in tqdm.tqdm(total_trial_inds):
            l_o_taste_ind = all_trial_taste_inds[l_o_ind].astype(
                'int')  # Taste of left out trial
            l_o_delivery_ind = all_trial_delivery_inds[l_o_ind].astype(
                'int')  # Delivery index of left out trial
            
            #Collect firing rate data for GMM Fits
            true_epoch_train_data = []
            true_epoch_train_data_labels = []
            none_train_data = []
            by_epoch_true_taste_train_data = [] #num epochs x num tastes x num deliveries
            by_epoch_all_taste_train_data = [] #num epochs x num tastes x num deliveries
            by_epoch_all_taste_prob = []
            for e_i in epochs_to_analyze:
                train_epoch_data = []
                true_taste_train_data = []
                true_taste_train_data_counts = []
                train_none_data = []
                for t_i in range(num_tastes):
                    train_taste_data = []
                    for d_i in all_trial_inds[t_i]:
                        if (d_i == l_o_delivery_ind) and (t_i == l_o_taste_ind):
                            # This is the Leave-One-Out trial so do nothing
                            train_taste_data.extend([])
                        else:
                            if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                                train_taste_data.extend(
                                    list(tastant_fr_dist[t_i][d_i][e_i].T))
                                if t_i < num_tastes-1:
                                    train_epoch_data.extend(
                                        list(tastant_fr_dist[t_i][d_i][e_i].T))
                                else:
                                    train_none_data.extend(
                                        list(tastant_fr_dist[t_i][d_i][e_i].T))
                            else:
                                train_taste_data.extend(
                                    list(tastant_fr_dist[t_i][d_i][e_i]))
                                if t_i < num_tastes-1:
                                    train_epoch_data.extend(
                                        list(tastant_fr_dist[t_i][d_i][e_i]))
                                else:
                                    train_none_data.extend(
                                        list(tastant_fr_dist[t_i][d_i][e_i]))
                    if t_i < num_tastes-1:
                        true_taste_train_data.append(train_taste_data)
                        true_taste_train_data_counts.extend([len(train_taste_data)])
                    else: #None condition - augment with randomized data
                        neur_max = np.expand_dims(np.max(np.array(train_taste_data),0),1)
                        train_none_data.extend(list((neur_max*np.random.rand(num_neur,100)).T))
                        train_none_data.extend(list(((neur_max/10)*np.random.rand(num_neur,100)).T))
                        train_none_data.extend(list((np.eye(num_neur)).T))
                by_epoch_true_taste_train_data.append(true_taste_train_data)
                all_taste_train_data = []
                all_taste_train_data.extend(true_taste_train_data)
                all_taste_train_data.extend([train_none_data])
                all_taste_counts = [len(all_taste_train_data[t_i]) for t_i in range(len(all_taste_train_data))]
                all_taste_prob = np.array(all_taste_counts)/np.sum(np.array(all_taste_counts))
                by_epoch_all_taste_train_data.append(all_taste_train_data)
                by_epoch_all_taste_prob.append(all_taste_prob)
                true_epoch_train_data.append(train_epoch_data)
                true_epoch_train_data_labels.append(list(e_i*np.ones(len(train_epoch_data))))
                none_train_data.extend(train_none_data)
            
            true_epoch_train_data_concat = []
            for e_i in range(len(true_epoch_train_data)):
                true_epoch_train_data_concat.extend(true_epoch_train_data[e_i])
            
            all_epoch_train_data = []
            all_epoch_train_data_counts = []
            all_epoch_train_labels = ['Epoch ' + str(e_i) for e_i in epochs_to_analyze]
            all_epoch_train_data.extend(true_epoch_train_data)
            all_epoch_train_data_counts.extend([len(true_epoch_train_data[t_i]) for t_i in range(len(true_epoch_train_data))])
            # all_epoch_train_data.append(none_train_data)
            # all_epoch_train_data_counts.extend([len(none_train_data)])
            # all_epoch_train_labels.extend(['None'])
            p_epoch = np.array(all_epoch_train_data_counts)/np.sum(np.array(all_epoch_train_data_counts))
            
            # Run PCA transform only on non-z-scored data
            need_pca = 0
            by_epoch_pca_reducers = dict()
            if np.min(np.array(true_epoch_train_data_concat)) >= 0:
                need_pca = 1
                #Epoch-Based PCA
                epoch_pca = PCA()
                epoch_pca.fit(np.array(true_epoch_train_data_concat).T)
                exp_var = epoch_pca.explained_variance_ratio_
                num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
                if num_components == 0:
                    num_components = 3
                pca_reduce_epoch = PCA(num_components)
                pca_reduce_epoch.fit(np.array(true_epoch_train_data_concat))
                
                #Within epoch taste-based PCA
                for e_i in range(len(true_epoch_train_data)):
                    true_taste_train_data_concat = []
                    for t_i in range(num_tastes-1):
                        true_taste_train_data_concat.extend(by_epoch_true_taste_train_data[e_i][t_i])
                    taste_epoch_pca = PCA()
                    taste_epoch_pca.fit(np.array(true_taste_train_data_concat).T)
                    exp_var = epoch_pca.explained_variance_ratio_
                    num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
                    if num_components == 0:
                        num_components = 3
                    pca_reduce = PCA(num_components)
                    pca_reduce.fit(np.array(true_taste_train_data_concat))
                    by_epoch_pca_reducers[e_i] = pca_reduce
            
            # Run GMM Fits to distributions of each epoch firing rates
            all_epoch_gmm = dict()
            if l_o_ind == 0:
                f_epoch_pca, ax_epoch_pca = plt.subplots(ncols=3)
            for e_i in range(len(all_epoch_train_data)): #Note this will be number of epochs + 1 for no taste
                epoch_train_data = np.array(all_epoch_train_data[e_i])
                if need_pca == 1:
                    #PCA Transformed Data
                    transformed_data = pca_reduce_epoch.transform(epoch_train_data)
                else:
                    transformed_data = epoch_train_data
                #Plot transformed data
                if l_o_ind == 0:
                    epoch_label = all_epoch_train_labels[e_i]
                    ax_epoch_pca[0].scatter(transformed_data[:,0],transformed_data[:,1],\
                                            alpha=0.3, label=epoch_label)
                    ax_epoch_pca[0].legend()
                    ax_epoch_pca[0].set_title('PCA 0 x 1')
                    ax_epoch_pca[1].scatter(transformed_data[:,1],transformed_data[:,2],\
                                            alpha=0.3, label=epoch_label)
                    ax_epoch_pca[1].legend()
                    ax_epoch_pca[1].set_title('PCA 1 x 2')
                    ax_epoch_pca[2].scatter(transformed_data[:,0],transformed_data[:,2],\
                                            alpha=0.3, label=epoch_label)
                    ax_epoch_pca[2].legend()
                    ax_epoch_pca[2].set_title('PCA 0 x 2')
                #Fit GMM
                gm = gmm(n_components=1, n_init=10).fit(
                    transformed_data)
                all_epoch_gmm[e_i] = gm
            if l_o_ind == 0:
                plt.tight_layout()
                f_epoch_pca.savefig(os.path.join(decoder_save_dir,'epoch_pca.png'))
                f_epoch_pca.savefig(os.path.join(decoder_save_dir,'epoch_pca.svg'))
                plt.close(f_epoch_pca)
                
            # Run GMM Fits to distributions of each taste firing rates within epochs
            by_epoch_taste_gmm = dict()
            for e_i in range(len(by_epoch_all_taste_train_data)):
                all_taste_train_data = by_epoch_all_taste_train_data[e_i]
                all_taste_gmm = dict()
                if l_o_ind == 0:
                    f_taste_pca, ax_taste_pca = plt.subplots(ncols=3)
                for t_i in range(len(all_taste_train_data)):
                    taste_train_data = np.array(all_taste_train_data[t_i])
                    if need_pca == 1:
                        #PCA Transformed Data
                        transformed_data = pca_reduce_epoch.transform(taste_train_data)
                    else:
                        transformed_data = taste_train_data
                    #Plot
                    if l_o_ind == 0:
                        ax_taste_pca[0].scatter(transformed_data[:,0],transformed_data[:,1],\
                                                alpha=0.3, label=dig_in_names[t_i])
                        ax_taste_pca[0].legend()
                        ax_taste_pca[0].set_title('PCA 0 x 1')
                        ax_taste_pca[1].scatter(transformed_data[:,1],transformed_data[:,2],\
                                                alpha=0.3, label=dig_in_names[t_i])
                        ax_taste_pca[1].legend()
                        ax_taste_pca[1].set_title('PCA 1 x 2')
                        ax_taste_pca[2].scatter(transformed_data[:,0],transformed_data[:,2],\
                                                alpha=0.3, label=dig_in_names[t_i])
                        ax_taste_pca[2].legend()
                        ax_taste_pca[2].set_title('PCA 0 x 2')
                    #Fit GMM
                    gm = gmm(n_components=1, n_init=10).fit(
                        transformed_data)
                    all_taste_gmm[t_i] = gm
                by_epoch_taste_gmm[e_i] = all_taste_gmm
                if l_o_ind == 0:
                    plt.suptitle('Epoch ' + str(e_i))
                    plt.tight_layout()
                    f_taste_pca.savefig(os.path.join(decoder_save_dir,'epoch_' + str(e_i) + '_pca.png'))
                    f_taste_pca.savefig(os.path.join(decoder_save_dir,'epoch_' + str(e_i) + '_pca.svg'))
                    plt.close(f_taste_pca)
            
            # Grab trial firing rate data
            for e_ind, e_i in enumerate(epochs_to_analyze):
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
                    
                if need_pca == 1: #If it's not z-scored PCA to whiten
                     # PCA transform fr
                    try:
                        tb_fr_pca = pca_reduce.transform(tb_fr.T)
                    except:
                        tb_fr_pca = pca_reduce.transform(tb_fr)
                    list_tb_fr = list(tb_fr_pca)
                else: #If z-scored, train directly on data
                    list_tb_fr = list(tb_fr.T)
                    
                #Run decoders in order of epoch first and then taste
                #    epoch decoding
                inputs = zip(list_tb_fr, itertools.repeat(len(all_epoch_gmm)),
                             itertools.repeat(all_epoch_gmm), itertools.repeat(p_epoch))
                pool = Pool(4)
                tb_epoch_decode_prob = pool.map(
                    dp.segment_taste_decode_dependent_parallelized, inputs)
                pool.close()
                tb_epoch_decode_array = np.squeeze(np.array(tb_epoch_decode_prob)).T
                
                epoch_ind = np.argmax(tb_epoch_decode_array)
                if epoch_ind == e_ind:
                    trial_epoch_success_storage[l_o_ind,e_ind] = 1
                elif epoch_ind == len(epochs_to_analyze): #None condition
                    if l_o_taste_ind == num_tastes-1:
                        trial_epoch_success_storage[l_o_ind,e_ind] = 1
                 
                #    taste decoding
                if epoch_ind < len(epochs_to_analyze): #There is a true epoch selected and not "none"
                    if need_pca == 1: #If it's not z-scored PCA to whiten
                         # PCA transform fr
                        try:
                            tb_fr_pca = by_epoch_pca_reducers[epoch_ind].transform(tb_fr.T)
                        except:
                            tb_fr_pca = by_epoch_pca_reducers[epoch_ind].transform(tb_fr)
                        list_tb_fr = list(tb_fr_pca)
                    else: #If z-scored, train directly on data
                        list_tb_fr = list(tb_fr.T)
                
                    inputs = zip(list_tb_fr, itertools.repeat(num_tastes),
                                 itertools.repeat(by_epoch_taste_gmm[epoch_ind]), itertools.repeat(by_epoch_all_taste_prob[epoch_ind]))
                    pool = Pool(4)
                    tb_taste_decode_prob = pool.map(
                        dp.segment_taste_decode_dependent_parallelized, inputs)
                    pool.close()
                    tb_taste_decode_array = np.squeeze(np.array(tb_taste_decode_prob)).T
                else:
                    tb_taste_decode_array = np.zeros(num_tastes)
                    tb_taste_decode_array[-1] = 1
                
                #Plot individual trial decoding results
                f_loo = plt.figure(figsize=(5, 5))
                plt.suptitle(
                    'Taste ' + dig_in_names[l_o_taste_ind] + ' Delivery ' + str(l_o_delivery_ind) + \
                        '\nTrue Epoch: ' + str(e_ind) + ' Decoded Epoch: ' + str(epoch_ind))
                if len(new_time_bins) > 1:
                    for t_i_plot in range(num_tastes):
                        plt.plot(new_time_bins+deliv_cp[e_i], tb_taste_decode_array[t_i_plot, :],
                                 label=dig_in_names[t_i_plot], color=taste_colors[t_i_plot])
                        plt.fill_between(
                            new_time_bins+deliv_cp[e_i], tb_taste_decode_array[t_i_plot, :], color=taste_colors[t_i_plot], alpha=0.5, label='_')
                    plt.xlabel('Time (ms)')
                else:
                    for t_i_plot in range(num_tastes):
                        plt.axhline(tb_taste_decode_array[t_i_plot],
                                 label=dig_in_names[t_i_plot], color=taste_colors[t_i_plot])
                        plt.fill_between([0,1],[0,0],[tb_taste_decode_array[t_i_plot],tb_taste_decode_array[t_i_plot]],
                                         color=taste_colors[t_i_plot],alpha=0.5,label='_')
                plt.ylabel('P(Taste)')
                plt.ylim([-0.1, 1.1])
                plt.legend(loc='upper right')
                plt.tight_layout()
                f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' +
                              str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '_epoch_' + str(e_i) + '.png'))
                f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' +
                              str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '_epoch_' + str(e_i) + '.svg'))
                plt.close(f_loo)
                
                # ___Calculate the average fraction of the epoch that was decoded as each taste and store
                if len(new_time_bins) > 0:
                    taste_max_inds = np.argmax(tb_taste_decode_array, 0)
                    taste_decode_fracs = [len(np.where(taste_max_inds == t_i_decode)[
                                              0])/len(new_time_bins) for t_i_decode in range(num_tastes)]
                else:
                    taste_decode_fracs = list(tb_taste_decode_array)
                trial_decode_storage[l_o_ind, e_ind, :] = taste_decode_fracs
                # ___Calculate the fraction of time in the epoch of each taste being best
                best_taste = np.where(
                    taste_decode_fracs == np.max(taste_decode_fracs))[0]
                if len(best_taste) == 1:
                    if best_taste == l_o_taste_ind:
                        trial_final_success_storage[l_o_ind, e_ind] = 1
                else:
                    # Taste is one of the predicted tastes in a "tie"
                    if len(np.where(best_taste == l_o_taste_ind)[0]) > 0:
                        trial_final_success_storage[l_o_ind, e_ind] = 1
        
        # Once all trials are decoded, save decoding success results
        np.save(os.path.join(decoder_save_dir,'trial_decode_storage.npy'),trial_decode_storage,allow_pickle=True)
        np.savetxt(os.path.join(decoder_save_dir,
                   'trial_epoch_success_storage.csv'), trial_epoch_success_storage, delimiter=',')
        np.savetxt(os.path.join(decoder_save_dir,
                   'trial_final_success_storage.csv'), trial_final_success_storage, delimiter=',')
        
        #Print overall results
        print("\nEpoch Success:")
        print(np.sum(trial_epoch_success_storage)/(num_cp*total_trials))
        print("\nBy-Epoch Success:")
        print(np.sum(trial_epoch_success_storage,0)/(total_trials))
        
        print("\nTaste Success:")
        print(np.sum(trial_final_success_storage)/(num_cp*total_trials))
        print("\nBy-Epoch Success:")
        print(np.sum(trial_final_success_storage,0)/(total_trials))
        
    
def multistep_taste_decoder(num_neur, start_dig_in_times, tastant_fr_dist, 
                all_trial_inds, tastant_spike_times, cp_raster_inds,
                pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names,
                max_hz, save_dir, epochs_to_analyze=[]):
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
            - max_hz: maximum firing rate in taste data
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
    print("\t\tTesting multi-step GMM Decoder.")
    # Variables
    num_tastes = len(start_dig_in_times)
    num_cp = len(tastant_fr_dist[0][0])
    p_taste = np.ones(num_tastes)/num_tastes  # P(taste)
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, num_tastes))
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

    # Save dirs
    decoder_save_dir = os.path.join(save_dir, 'Multistep_Taste_GMM_Decoder_Tests')
    if not os.path.isdir(decoder_save_dir):
        os.mkdir(decoder_save_dir)
        
    trial_decodes = os.path.join(
        decoder_save_dir, 'Individual_Trials')
    if not os.path.isdir(trial_decodes):
        os.mkdir(trial_decodes)
        
    try:  # Try to import the decoding results
        print("Import statements here.")
        
        trial_epoch_success_storage = []
        with open(os.path.join(decoder_save_dir, 'trial_epoch_success_storage.csv'), newline='') as successtrialfile:
            filereader = csv.reader(
                successtrialfile, delimiter=',', quotechar='|')
            for row in filereader:
                trial_epoch_success_storage.append(np.array(row).astype('float'))
        trial_epoch_success_storage = np.array(trial_epoch_success_storage).squeeze()
        
        trial_taste_success_storage = []
        with open(os.path.join(decoder_save_dir, 'trial_taste_success_storage.csv'), newline='') as successtrialfile:
            filereader = csv.reader(
                successtrialfile, delimiter=',', quotechar='|')
            for row in filereader:
                trial_taste_success_storage.append(np.array(row).astype('float'))
        trial_taste_success_storage = np.array(trial_taste_success_storage).squeeze()
        
        #Print success
        overall_taste_success = np.sum(trial_taste_success_storage)/(total_trials*num_cp)
        by_epoch_taste_success = np.sum(trial_taste_success_storage,0)/(total_trials)
        print("\nOverall success in taste decoding:")
        print(overall_taste_success)
        print("By-epoch success in taste decoding:")
        print(by_epoch_taste_success)
        
        overall_epoch_success = np.sum(trial_epoch_success_storage)/(total_trials*num_cp)
        by_epoch_epoch_success = np.sum(trial_epoch_success_storage,0)/(total_trials)
        print("\nOverall success in epoch decoding:")
        print(overall_epoch_success)
        print("By-epoch success in epoch decoding:")
        print(by_epoch_epoch_success)
        
    except: # Run decoding
        #Probabilities of decoding for each trial of which epoch it came from
        #following determination of a taste
        trial_epoch_decode_storage = np.zeros((total_trials, num_cp, num_cp))
        #Probabilities of decoding for each trial which taste it was
        trial_taste_decode_storage = np.zeros((total_trials, num_cp, num_tastes))
        #Binary storage of whether the taste was correctly decoded from that trial's epoch of activity
        trial_taste_success_storage = np.zeros((total_trials, num_cp))
        #Binary storage of whether the epoch was correctly decoded following taste designation
        trial_epoch_success_storage = np.zeros((total_trials, num_cp))
        
        print('\t\t\t\tPerforming LOO Decoding')
        # Which trial is being left out for decoding
        for l_o_ind in tqdm.tqdm(total_trial_inds):
            l_o_taste_ind = all_trial_taste_inds[l_o_ind].astype(
                'int')  # Taste of left out trial
            l_o_delivery_ind = all_trial_delivery_inds[l_o_ind].astype(
                'int')  # Delivery index of left out trial
            
            #Collect firing rate data for GMM Fits
            true_taste_train_data = []
            true_taste_train_data_labels = []
            none_train_data = []
            by_taste_all_epoch_train_data = []
            by_taste_all_epoch_prob = []
            for t_i in range(num_tastes):
                train_true_taste_data = []
                train_taste_epoch_data = []
                train_taste_epoch_counts = []
                for e_i in epochs_to_analyze:
                    train_epoch_data = []
                    for d_i in all_trial_inds[t_i]:
                        if (d_i == l_o_delivery_ind) and (t_i == l_o_taste_ind):
                            # This is the Leave-One-Out trial so do nothing
                            train_epoch_data.extend([])
                        else:
                            if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                                train_epoch_data.extend(
                                    list(tastant_fr_dist[t_i][d_i][e_i].T))
                                if t_i < num_tastes-1:
                                    train_true_taste_data.extend(
                                        list(tastant_fr_dist[t_i][d_i][e_i].T))
                                else:
                                    none_train_data.extend(
                                        list(tastant_fr_dist[t_i][d_i][e_i].T))
                            else:
                                train_epoch_data.extend(
                                    list(tastant_fr_dist[t_i][d_i][e_i]))
                                if t_i < num_tastes-1:
                                    train_true_taste_data.extend(
                                        list(tastant_fr_dist[t_i][d_i][e_i]))
                                else:
                                    none_train_data.extend(
                                        list(tastant_fr_dist[t_i][d_i][e_i]))
                    train_taste_epoch_data.append(train_epoch_data)
                    train_taste_epoch_counts.extend([len(train_epoch_data)])
                true_taste_train_data.extend(train_true_taste_data)
                true_taste_train_data_labels.extend(list(t_i*np.ones(len(train_true_taste_data))))
                by_taste_all_epoch_train_data.append(train_taste_epoch_data)
                by_taste_all_epoch_prob.append(np.array(train_taste_epoch_counts)/np.sum(np.array(train_taste_epoch_counts)))
            
            #combine none and true taste for gmm fitting and predictions
            all_taste_train_data = []
            all_taste_train_labels = []
            for t_i in range(num_tastes-1):
                true_taste_inds = np.where(np.array(true_taste_train_data_labels) == t_i)[0]
                all_taste_train_data.append(np.array(true_taste_train_data)[true_taste_inds])
            all_taste_train_data.append(np.array(none_train_data))
            count_all_taste = [len(all_taste_train_data[t_i]) for t_i in range(num_tastes)]
            p_all_taste = np.array(count_all_taste)/np.sum(np.array(count_all_taste))
            
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
                
                #Within taste epoch-based PCA
                for t_i in range(num_tastes-1):
                    epoch_train_data_concat = []
                    for e_i in range(num_cp):
                        epoch_train_data_concat.extend(by_taste_all_epoch_train_data[t_i][e_i])
                    epoch_taste_pca = PCA()
                    epoch_taste_pca.fit(np.array(epoch_train_data_concat).T)
                    exp_var = epoch_taste_pca.explained_variance_ratio_
                    num_components = np.where(np.cumsum(exp_var) >= 0.9)[0][0]
                    if num_components == 0:
                        num_components = 3
                    pca_reduce = PCA(num_components)
                    pca_reduce.fit(np.array(epoch_train_data_concat))
                    by_taste_pca_reducers[t_i] = pca_reduce
            
            #Run GMM fits to distributions of each taste
            #Also run GMM fits to epoch distributions within each taste except none
            all_taste_gmm = dict()
            by_taste_epoch_gmm = dict()
            if l_o_ind == 0:
                f_taste_pca, ax_taste_pca = plt.subplots(ncols = 3)
            for t_i in range(num_tastes):
                taste_train_data = np.array(all_taste_train_data[t_i])
                if need_pca == 1:
                    transformed_data = pca_reduce_taste.transform(taste_train_data)
                else:
                    transformed_data = taste_train_data
                #Plot transformed data
                if l_o_ind == 0:
                    taste_label = dig_in_names[t_i]
                    ax_taste_pca[0].scatter(transformed_data[:,0],transformed_data[:,1],\
                                            alpha=0.3, label=taste_label)
                    ax_taste_pca[0].legend()
                    ax_taste_pca[0].set_title('PCA 0 x 1')
                    ax_taste_pca[1].scatter(transformed_data[:,1],transformed_data[:,2],\
                                            alpha=0.3, label=taste_label)
                    ax_taste_pca[1].legend()
                    ax_taste_pca[1].set_title('PCA 1 x 2')
                    ax_taste_pca[2].scatter(transformed_data[:,0],transformed_data[:,2],\
                                            alpha=0.3, label=taste_label)
                    ax_taste_pca[2].legend()
                    ax_taste_pca[2].set_title('PCA 0 x 2')
                #Fit GMM
                gm = gmm(n_components=1, n_init=10).fit(
                    transformed_data)
                all_taste_gmm[t_i] = gm
                
                if t_i < num_tastes-1:
                    by_taste_epoch_gmm[t_i] = dict()
                    if l_o_ind == 0:
                        f_epoch_pca, ax_epoch_pca = plt.subplots(ncols=3)
                    for e_i in range(num_cp):
                        taste_epoch_train_data = np.array(by_taste_all_epoch_train_data[t_i][e_i])
                        if need_pca == 1:
                            transformed_data = by_taste_pca_reducers[t_i].transform(taste_epoch_train_data)
                        else:
                            transformed_data = taste_epoch_train_data
                        if l_o_ind == 0:
                            epoch_label = "Epoch " + str(e_i)
                            ax_epoch_pca[0].scatter(transformed_data[:,0],transformed_data[:,1],\
                                                    alpha=0.3, label=epoch_label)
                            ax_epoch_pca[0].legend()
                            ax_epoch_pca[0].set_title('PCA 0 x 1')
                            ax_epoch_pca[1].scatter(transformed_data[:,1],transformed_data[:,2],\
                                                    alpha=0.3, label=epoch_label)
                            ax_epoch_pca[1].legend()
                            ax_epoch_pca[1].set_title('PCA 1 x 2')
                            ax_epoch_pca[2].scatter(transformed_data[:,0],transformed_data[:,2],\
                                                    alpha=0.3, label=epoch_label)
                            ax_epoch_pca[2].legend()
                            ax_epoch_pca[2].set_title('PCA 0 x 2')
                        #Fit GMM
                        gm = gmm(n_components=1, n_init=10).fit(
                            transformed_data)
                        by_taste_epoch_gmm[t_i][e_i] = gm
                    if l_o_ind == 0:
                        plt.suptitle(dig_in_names[t_i])
                        plt.tight_layout()
                        f_epoch_pca.savefig(os.path.join(decoder_save_dir,dig_in_names[t_i] + '_pca.png'))
                        f_epoch_pca.savefig(os.path.join(decoder_save_dir,dig_in_names[t_i] + '_pca.svg'))
                        plt.close(f_epoch_pca)
                        
            if l_o_ind == 0:
                plt.tight_layout()
                f_taste_pca.savefig(os.path.join(decoder_save_dir,'taste_pca.png'))
                f_taste_pca.savefig(os.path.join(decoder_save_dir,'taste_pca.svg'))
                plt.close(f_taste_pca)
                
            #Grab trial firing rate data
            for e_ind, e_i in enumerate(epochs_to_analyze):
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
                    
                if need_pca == 1: #If it's not z-scored PCA to whiten
                    try:
                        tb_fr_pca = pca_reduce_taste.transform(tb_fr.T)
                    except:
                        tb_fr_pca = pca_reduce_taste.transform(tb_fr)
                    list_tb_fr = list(tb_fr_pca)
                else:
                    list_tb_fr = list(tb_fr.T)
                
                #Run decoders in order of epoch first and then taste
                #    taste decoding
                inputs = zip(list_tb_fr, itertools.repeat(len(all_taste_gmm)),
                             itertools.repeat(all_taste_gmm), itertools.repeat(p_all_taste))
                pool = Pool(4)
                tb_taste_decode_prob = pool.map(
                    dp.segment_taste_decode_dependent_parallelized, inputs)
                pool.close()
                tb_taste_decode_array = np.squeeze(np.array(tb_taste_decode_prob)).T
                trial_taste_decode_storage[l_o_ind,e_ind,:] = tb_taste_decode_array
                
                taste_ind = np.argmax(tb_taste_decode_array)
                if taste_ind < num_tastes-1: #Taste condition
                    if taste_ind == l_o_taste_ind: #Correctly decoded?
                        trial_taste_success_storage[l_o_ind,e_ind] = 1
                    
                    if need_pca == 1: #If it's not z-scored PCA to whiten
                        try:
                            tb_fr_pca = by_taste_pca_reducers[taste_ind].transform(tb_fr.T)
                        except:
                            tb_fr_pca = by_taste_pca_reducers[taste_ind].transform(tb_fr)
                        list_tb_fr = list(tb_fr_pca)
                    else:
                        list_tb_fr = list(tb_fr.T)
                    
                    #Now decode epoch
                    inputs = zip(list_tb_fr, itertools.repeat(num_cp),
                                 itertools.repeat(by_taste_epoch_gmm[taste_ind]), itertools.repeat(by_taste_all_epoch_prob[taste_ind]))
                    pool = Pool(4)
                    tb_epoch_decode_prob = pool.map(
                        dp.segment_taste_decode_dependent_parallelized, inputs)
                    pool.close()
                    tb_epoch_decode_array = np.squeeze(np.array(tb_epoch_decode_prob)).T
                    trial_epoch_decode_storage[l_o_ind,e_ind,:] = tb_epoch_decode_array
                    
                    epoch_ind = np.argmax(tb_epoch_decode_array)
                    if epoch_ind == e_ind:
                        trial_epoch_success_storage[l_o_ind,e_ind] = 1
                    
                elif taste_ind == num_tastes-1: #None condition
                    if l_o_taste_ind == num_tastes-1:
                        trial_taste_success_storage[l_o_ind,e_ind] = 1
                        trial_epoch_success_storage[l_o_ind,e_ind] = 1
                    epoch_ind = -1
                    tb_epoch_decode_array = np.ones(num_cp)/num_cp #Basically doesn't matter
                    
                #Plot individual trial decoding results
                f_loo = plt.figure(figsize=(5, 5))
                plt.suptitle(
                    'Taste ' + dig_in_names[l_o_taste_ind] + ' Delivery ' + str(l_o_delivery_ind) + ' Epoch ' + str(e_ind) +\
                        '\nDecoded Taste: ' + dig_in_names[taste_ind] + ' Decoded Epoch: ' + str(epoch_ind))
                if len(np.shape(tb_epoch_decode_array)) > 1:
                    for e_i_plot in range(num_cp):
                        plt.plot(new_time_bins+deliv_cp[e_i], tb_epoch_decode_array[e_i_plot, :], \
                                 label="Epoch " + str(e_i_plot), color=epoch_colors[e_i_plot])
                        plt.fill_between(
                            new_time_bins+deliv_cp[e_i], tb_epoch_decode_array[e_i_plot, :], \
                                color=epoch_colors[e_i_plot], alpha=0.5, label='_')
                    plt.xlabel('Time (ms)')
                else:
                    for e_i_plot in range(num_cp):
                        plt.axhline(tb_epoch_decode_array[e_i_plot],
                                 label="Epoch " + str(e_i_plot), color=epoch_colors[e_i_plot])
                        plt.fill_between([0,1],[0,0],[tb_epoch_decode_array[e_i_plot],tb_epoch_decode_array[e_i_plot]],
                                         color=epoch_colors[e_i_plot],alpha=0.5,label='_')
                plt.ylabel('P(Epoch)')
                plt.ylim([-0.1, 1.1])
                plt.legend(loc='upper right')
                plt.tight_layout()
                f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' +
                              str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '_epoch_' + str(e_i) + '.png'))
                f_loo.savefig(os.path.join(trial_decodes, 'decoding_results_taste_' +
                              str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '_epoch_' + str(e_i) + '.svg'))
                plt.close(f_loo)
        
        #Save results
        np.save(os.path.join(decoder_save_dir,'trial_epoch_decode_storage.npy'),trial_epoch_decode_storage,allow_pickle=True)
        np.save(os.path.join(decoder_save_dir,'trial_taste_decode_storage.npy'),trial_taste_decode_storage,allow_pickle=True)
        np.savetxt(os.path.join(decoder_save_dir,
                   'trial_taste_success_storage.csv'), trial_taste_success_storage, delimiter=',')
        np.savetxt(os.path.join(decoder_save_dir,
                   'trial_epoch_success_storage.csv'), trial_epoch_success_storage, delimiter=',')
        
        
        #Print success
        overall_taste_success = np.sum(trial_taste_success_storage)/(total_trials*num_cp)
        by_epoch_taste_success = np.sum(trial_taste_success_storage,0)/(total_trials)
        print("\nOverall success in taste decoding:")
        print(overall_taste_success)
        print("By-epoch success in taste decoding:")
        print(by_epoch_taste_success)
        
        overall_epoch_success = np.sum(trial_epoch_success_storage)/(total_trials*num_cp)
        by_epoch_epoch_success = np.sum(trial_epoch_success_storage,0)/(total_trials)
        print("\nOverall success in epoch decoding:")
        print(overall_epoch_success)
        print("By-epoch success in epoch decoding:")
        print(by_epoch_epoch_success)
                
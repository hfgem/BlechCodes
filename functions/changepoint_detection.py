#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:01:49 2023

@author: Hannah Germaine

This is a collection of functions for performing changepoint detection on raster matrices
"""

import tqdm
import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def calc_cp_iter_pop(tastant_spike_times, cp_bin, num_cp, start_dig_in_times,
                     end_dig_in_times, before_taste, after_taste,
                     dig_in_names, taste_cp_save_dir):
    num_tastes = len(tastant_spike_times)
    taste_cp = []
    for t_i in range(num_tastes):
        print('Calculating changepoints for taste ' + dig_in_names[t_i])
        dig_in_name = dig_in_names[t_i]
        num_deliv = len(tastant_spike_times[t_i])
        num_neur = len(tastant_spike_times[t_i][0])
        deliv_cp = np.zeros((num_deliv, num_cp+2))
        deliv_st = dict()
        # Calculate changepoints for the population for each tastant delivery
        for d_i in tqdm.tqdm(range(num_deliv)):
            start_deliv = start_dig_in_times[t_i][d_i]
            end_deliv = end_dig_in_times[t_i][d_i]
            taste_deliv_len = end_deliv-start_deliv
            bin_length = before_taste+taste_deliv_len+after_taste+1
            # Create the binary spike matrix
            neur_st = dict()
            deliv_bin_full = np.zeros((num_neur, bin_length))
            for n_i in range(num_neur):
                spike_ind = tastant_spike_times[t_i][d_i][n_i]
                converted_ind = list(
                    (np.array(spike_ind) - start_deliv + before_taste).astype('int'))
                try:
                    cur_st = deliv_st[d_i]
                    cur_st.append(converted_ind)
                    neur_st[n_i] = cur_st
                except:
                    neur_st[n_i] = [converted_ind]
                deliv_bin_full[n_i, converted_ind] = 1
            # Crop spike matrix to just taste interval
            bin_length -= before_taste
            deliv_bin = deliv_bin_full[:, before_taste:]
            # Run through each timepoint starting at the minimum changepoint bin
            # size to the length of the segment - the changepoint bin size and
            # calculate the proxy for a changepoint between two Poisson processes.
            # Do this iteratively in intervals defined by the previous
            # changepoint locations.
            peak_inds = []  # Storage for where peak indices might be
            # Relative - each time divided by the max likelihood of that iteration
            peak_likelihoods = []
            for iter_i in range(num_cp):
                if len(peak_inds) > 0:
                    sorted_peak_inds = np.sort(np.array(peak_inds))
                    sub_bins = [0]
                    sub_bins.extend(list(sorted_peak_inds))
                    sub_bins.extend([bin_length-cp_bin])
                else:
                    sub_bins = [0, bin_length - cp_bin]
                for b_i in range(len(sub_bins)-1):
                    sub_bin_len = sub_bins[b_i+1] - sub_bins[b_i]
                    sub_bin_spikes = np.sum(
                        deliv_bin[:, sub_bins[b_i]:sub_bins[b_i+1]]).astype('int')
                    if (sub_bin_len > cp_bin) & (sub_bin_spikes > 0):
                        cp_likelihood_bin_pop = np.zeros(
                            (num_neur, bin_length))
                        for n_i in range(num_neur):
                            for time_i in np.arange(sub_bins[b_i], sub_bins[b_i+1]):
                                N_1 = np.sum(
                                    deliv_bin[n_i, sub_bins[b_i]:time_i])
                                N_2 = np.sum(
                                    deliv_bin[n_i, time_i+1:sub_bins[b_i+1]])
                                T_1 = time_i - sub_bins[b_i]
                                T_2 = sub_bins[b_i+1] - time_i
                                cp_like_calc = (
                                    ((N_1/T_1)**N_1)*((N_2/T_2)**N_2))/(((N_1+N_2)/(T_1+T_2))**(N_1+N_2))
                                # /np.max(cp_like_calc)
                                cp_likelihood_bin_pop[n_i,
                                                      time_i] = cp_like_calc
                        cp_likelihood_bin = np.prod(
                            cp_likelihood_bin_pop, axis=0)
                        bin_peak_inds = find_peaks(
                            cp_likelihood_bin[cp_bin:], distance=cp_bin)[0] + cp_bin
                        if len(bin_peak_inds) > 0:
                            sorted_peak_ind = bin_peak_inds[np.argsort(
                                cp_likelihood_bin[bin_peak_inds])]
                            if iter_i > 0:
                                for spi in range(len(sorted_peak_ind)):
                                    max_peak_ind = sorted_peak_ind[-(spi+1)]
                                    # /np.max(cp_likelihood_bin)
                                    max_peak_likelihood = cp_likelihood_bin[max_peak_ind]
                                    close_existing_peaks = np.where(
                                        np.abs(np.array(peak_inds) - max_peak_ind) < cp_bin)[0]
                                    if len(close_existing_peaks) == 0:
                                        peak_inds.extend([max_peak_ind])
                                        peak_likelihoods.extend(
                                            [max_peak_likelihood])
                                        break
                            elif iter_i == 0:
                                max_peak_ind = sorted_peak_ind[-1]
                                # /np.max(cp_likelihood_bin)
                                max_peak_likelihood = cp_likelihood_bin[max_peak_ind]
                                peak_inds.extend([max_peak_ind])
                                peak_likelihoods.extend([max_peak_likelihood])
            # Now select the best num_cp peaks
            peak_inds = np.array(peak_inds) + before_taste
            peak_likelihoods = np.array(peak_likelihoods)
            best_peak_inds = np.zeros(num_cp+2)
            best_peak_inds[0] = before_taste
            selected_peak_inds = np.sort(
                np.array(peak_inds)[np.argsort(peak_likelihoods[-num_cp:])])
            if len(selected_peak_inds) < num_cp:
                remaining_cp = num_cp - len(selected_peak_inds)
                selected_peak_inds = np.concatenate(
                    (selected_peak_inds, (bin_length-1)*np.ones(remaining_cp)))
            best_peak_inds[1:-1] = selected_peak_inds
            best_peak_inds[-1] = after_taste + before_taste
            deliv_cp[d_i, :] = best_peak_inds.astype('int')
            raster_dir = os.path.join(taste_cp_save_dir,'rasters')
            if os.path.isdir(raster_dir) == False:
                os.mkdir(raster_dir)
            plot_cp_rasters_pop(deliv_bin_full, best_peak_inds,
                                d_i, before_taste, dig_in_name, raster_dir)
            fr_dir = os.path.join(taste_cp_save_dir,'fr_plots')
            if os.path.isdir(fr_dir) == False:
                os.mkdir(fr_dir)
            plot_cp_fr_changes(deliv_bin, best_peak_inds,
                                d_i, before_taste, dig_in_name, fr_dir)
        taste_cp.append(deliv_cp)

    return taste_cp

def plot_cp_rasters_pop(deliv_bin, best_peak_inds, d_i, before_taste, dig_in_name, save_dir):
    """This function creates plots of taste response rasters with vertical 
    delineations of where the changepoints were calculated"""
    num_neur, num_dt = np.shape(deliv_bin)
    deliv_st = [np.where(deliv_bin[n_i, :] > 0)[0] -
                before_taste for n_i in range(num_neur)]
    fig_t = plt.figure(figsize=(5, 5))
    plt.eventplot(deliv_st)
    for cp_i in best_peak_inds:
        plt.axvline(cp_i-before_taste, color='r')
    plt.title(dig_in_name + ' delivery ' + str(d_i))
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    fig_name = dig_in_name + '_deliv_' + str(d_i)
    fig_t.savefig(os.path.join(save_dir,fig_name + '.png'))
    fig_t.savefig(os.path.join(save_dir,fig_name + '.svg'))
    plt.close(fig_t)
    
def plot_cp_fr_changes(deliv_bin, best_peak_inds, d_i, before_taste, dig_in_name, save_dir):
    """This function creates firing rate plots of taste responses with the
    delineations of where the changepoints were calculated"""
    num_neur, num_dt = np.shape(deliv_bin)
    
    fig_t, ax = plt.subplots(nrows = num_neur, ncols = 1, figsize = (5,15), \
                             sharex = True)
    fig_name = dig_in_name + '_deliv_' + str(d_i)
    
    #Plots of each neuron's fr across time with changepoints
    half_bin_size = 100
    bin_size = half_bin_size*2
    fr_mat = np.zeros((num_neur, num_dt - bin_size))
    x_vals = np.arange(half_bin_size,num_dt-half_bin_size)
    for t_i in x_vals:
        fr_mat[:,t_i-half_bin_size] = np.sum(deliv_bin[:,t_i-half_bin_size:t_i+half_bin_size],1)/(bin_size/1000) #in Hz
    for n_i in range(num_neur):
        ax[n_i].plot(x_vals/1000,fr_mat[n_i,:])
        for cp_i in best_peak_inds:
            ax[n_i].axvline((cp_i-before_taste)/1000)
        ax[n_i].set_xlabel('Time (s)')
        
    fig_t.savefig(os.path.join(save_dir,fig_name + '.png'))
    fig_t.savefig(os.path.join(save_dir,fig_name + '.svg'))
    plt.close(fig_t)
    
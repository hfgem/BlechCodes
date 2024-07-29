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

def calc_cp_bayes(tastant_spike_times, cp_bin, num_cp, start_dig_in_times,
                  end_dig_in_times, before_taste, after_taste,
                  dig_in_names, taste_cp_save_dir):
    neur_save_dir = taste_cp_save_dir + 'neur/'
    if os.path.isdir(neur_save_dir) == False:
        os.mkdir(neur_save_dir)
    deliv_save_dir = taste_cp_save_dir + 'deliv/'
    if os.path.isdir(deliv_save_dir) == False:
        os.mkdir(deliv_save_dir)
    num_tastes = len(tastant_spike_times)
    num_deliv = len(tastant_spike_times[0])
    num_neur = len(tastant_spike_times[0][0])
    taste_cp = []
    for t_i in range(num_tastes):
        print('Calculating changepoints for taste ' + dig_in_names[t_i])
        neur_deliv_cp = np.zeros((num_deliv, num_neur, num_cp+1))
        deliv_st = dict()
        # Calculate changepoints for each neuron for each tastant delivery
        for n_i in tqdm.tqdm(range(num_neur)):
            neur_deliv_st = []
            # Collects for 1 neuron the cp likelihood for each delivery index
            neur_cp_likelihood = dict()
            bin_length_collect = []
            for d_i in range(num_deliv):
                start_deliv = start_dig_in_times[t_i][d_i]
                end_deliv = end_dig_in_times[t_i][d_i]
                taste_deliv_len = end_deliv-start_deliv
                spike_ind = tastant_spike_times[t_i][d_i][n_i]
                bin_length = before_taste+taste_deliv_len+after_taste+1
                bin_length_collect.extend([bin_length])
                # Create the binary spike matrix
                deliv_bin = np.zeros(bin_length)
                converted_ind = list(
                    (np.array(spike_ind) - start_deliv + before_taste).astype('int'))
                try:
                    cur_st = deliv_st[d_i]
                    cur_st.append(converted_ind)
                    deliv_st[d_i] = cur_st
                except:
                    deliv_st[d_i] = [converted_ind]
                neur_deliv_st.append(converted_ind)
                deliv_bin[converted_ind] = 1
                # Run through each timepoint starting at the minimum changepoint bin
                # size to the length of the segment - the changepoint bin size and
                # calculate the proxy for a changepoint between two Poisson processes
                cp_likelihood_d_i = np.zeros(bin_length)
                for time_i in np.arange(cp_bin, bin_length-cp_bin):
                    #N_1 = np.sum(deliv_bin[:time_i])
                    N_1 = np.sum(deliv_bin[time_i-cp_bin:time_i])
                    #N_2 = np.sum(deliv_bin[time_i:])
                    N_2 = np.sum(deliv_bin[time_i:time_i+cp_bin])
                    # /((N_1+N_2)/(2*cp_bin))**(N_1+N_2)
                    cp_likelihood_d_i[time_i] = (
                        ((N_1/cp_bin)**N_1)*((N_2/cp_bin)**N_2))
                    # cp_likelihood_d_i[time_i] = (((N_1/time_i)**N_1)*((N_2/(bin_length-time_i))**N_2))#/((N_1+N_2)/(2*bin_length))**(N_1+N_2)
                peak_inds = find_peaks(
                    cp_likelihood_d_i[cp_bin:], distance=cp_bin)[0]
                peak_inds = peak_inds[peak_inds > before_taste] + cp_bin
                ordered_peak_ind = np.argsort(cp_likelihood_d_i[peak_inds])
                best_peak_inds = np.zeros(num_cp+1)
                best_peak_inds[0] = before_taste
                found_inds = peak_inds[np.sort(ordered_peak_ind[0:num_cp])]
                if len(found_inds) < num_cp:
                    diff_len_cp = num_cp - len(found_inds)
                    best_peak_inds[1:len(found_inds)+1] = found_inds
                    best_peak_inds[len(found_inds) +
                                   1:] = (bin_length-1)*np.ones(diff_len_cp)
                else:
                    best_peak_inds[1:num_cp+1] = found_inds
                neur_cp_likelihood[d_i] = list(best_peak_inds.astype('int'))
            # _____Look at the average cp likelihood for this one neuron across deliveries_____
            neur_cp_likelihood_list = []
            neur_cp_likelihood_bin = np.zeros(
                (num_deliv, np.max(bin_length_collect)))
            for key in neur_cp_likelihood.keys():
                neur_cp_likelihood_list.append(list(neur_cp_likelihood[key]))
                neur_cp_likelihood_bin[int(key), list(
                    neur_cp_likelihood[key])] = 1
            # _____Plot changepoints across deliveries for one neuron_____
            plot_cp_rasters_neur(neur_deliv_st, neur_cp_likelihood_list, before_taste,
                                 dig_in_names[t_i], n_i, num_deliv, num_cp, neur_save_dir)
            for key in neur_cp_likelihood.keys():
                neur_deliv_cp[int(key), n_i, :] = neur_cp_likelihood[key]
        # _____Plot changepoints for each delivery across the population_____
        plot_cp_rasters_deliv(deliv_st, neur_deliv_cp,
                              before_taste, dig_in_names[t_i], deliv_save_dir)
        # Store results for tastant
        taste_cp.append(neur_deliv_cp)

    return taste_cp

# TODO: Use the cp framework to write an HMM approach that outputs the same data structure as other cp calc functions


def calc_cp_iter(tastant_spike_times, cp_bin, num_cp, start_dig_in_times,
                 end_dig_in_times, before_taste, after_taste,
                 dig_in_names, taste_cp_save_dir):
    # Neuron changepoint save directory
    neur_save_dir = taste_cp_save_dir + 'neur/'
    if os.path.isdir(neur_save_dir) == False:
        os.mkdir(neur_save_dir)
    # Delivery changepoint save directory
    deliv_save_dir = taste_cp_save_dir + 'deliv/'
    if os.path.isdir(deliv_save_dir) == False:
        os.mkdir(deliv_save_dir)
    # Grab parameters
    num_tastes = len(tastant_spike_times)
    # Calculate changepoints by taste
    taste_cp = []
    for t_i in range(num_tastes):
        print('Calculating changepoints for taste ' + dig_in_names[t_i])
        num_deliv = len(tastant_spike_times[t_i])
        num_neur = len(tastant_spike_times[t_i][0])
        neur_deliv_cp = np.zeros(
            (num_deliv, num_neur, num_cp+1))  # Store changepoints
        deliv_st = dict()  # Delivery spike times storage
        # Calculate changepoints for each neuron for each tastant delivery
        for n_i in tqdm.tqdm(range(num_neur)):
            neur_deliv_st = []
            # Collects for 1 neuron the cp likelihood for each delivery index
            neur_cp_likelihood = dict()
            bin_length_collect = []
            for d_i in range(num_deliv):
                start_deliv = start_dig_in_times[t_i][d_i]
                end_deliv = end_dig_in_times[t_i][d_i]
                taste_deliv_len = end_deliv-start_deliv
                spike_ind = tastant_spike_times[t_i][d_i][n_i]
                bin_length = before_taste+taste_deliv_len+after_taste+1
                bin_length_collect.extend([bin_length])
                # Create the binary spike matrix
                deliv_bin = np.zeros(bin_length)
                converted_ind = list(
                    (np.array(spike_ind) - start_deliv + before_taste).astype('int'))
                try:
                    cur_st = deliv_st[d_i]
                    cur_st.append(converted_ind)
                    deliv_st[d_i] = cur_st
                except:
                    deliv_st[d_i] = [converted_ind]
                neur_deliv_st.append(converted_ind)
                deliv_bin[converted_ind] = 1
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
                        sub_bins = [cp_bin]
                        sub_bins.extend(list(sorted_peak_inds))
                        sub_bins.extend([bin_length-cp_bin])
                    else:
                        sub_bins = [cp_bin, bin_length - cp_bin]
                    for b_i in range(len(sub_bins)-1):
                        sub_bin_len = sub_bins[b_i+1] - sub_bins[b_i]
                        sub_bin_spikes = np.sum(
                            deliv_bin[sub_bins[b_i]:sub_bins[b_i+1]])
                        if (sub_bin_len > cp_bin) & (sub_bin_spikes > 0):
                            cp_likelihood_bin = np.zeros(bin_length)
                            for time_i in np.arange(sub_bins[b_i]+cp_bin, sub_bins[b_i+1]):
                                N_1 = np.sum(deliv_bin[sub_bins[b_i]:time_i])
                                N_2 = np.sum(
                                    deliv_bin[time_i+1:sub_bins[b_i+1]])
                                T_1 = time_i - sub_bins[b_i]
                                T_2 = sub_bins[b_i+1] - time_i
                                cp_likelihood_bin[time_i] = (
                                    ((N_1/T_1)**N_1)*((N_2/T_2)**N_2))/(((N_1+N_2)/(T_1+T_2))**(N_1+N_2))
                            bin_peak_inds = find_peaks(
                                cp_likelihood_bin[cp_bin:], distance=cp_bin)[0]
                            bin_peak_inds = bin_peak_inds[bin_peak_inds >
                                                          before_taste] + cp_bin
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
                                    peak_likelihoods.extend(
                                        [max_peak_likelihood])
                # Now select the first num_cp peaks
                best_peak_inds = np.zeros(num_cp+1)
                best_peak_inds[0] = before_taste
                selected_peak_inds = np.sort(np.array(peak_inds))[:num_cp]
                if len(selected_peak_inds) < num_cp:
                    remaining_cp = num_cp - len(selected_peak_inds)
                    selected_peak_inds = np.concatenate(
                        (selected_peak_inds, (bin_length-1)*np.ones(remaining_cp)))
                best_peak_inds[1:] = selected_peak_inds
                neur_cp_likelihood[d_i] = list(best_peak_inds.astype('int'))
            # _____Look at the average cp likelihood for this one neuron across deliveries_____
            neur_cp_likelihood_list = []
            neur_cp_likelihood_bin = np.zeros(
                (num_deliv, np.max(bin_length_collect)))
            for key in neur_cp_likelihood.keys():
                neur_cp_likelihood_list.append(list(neur_cp_likelihood[key]))
                neur_cp_likelihood_bin[int(key), list(
                    neur_cp_likelihood[key])] = 1
            # _____Plot changepoints across deliveries for one neuron_____
            plot_cp_rasters_neur(neur_deliv_st, neur_cp_likelihood_list, before_taste,
                                 dig_in_names[t_i], n_i, num_deliv, num_cp, neur_save_dir)
            for key in neur_cp_likelihood.keys():
                neur_deliv_cp[int(key), n_i, :] = neur_cp_likelihood[key]
        # _____Plot changepoints for each delivery across the population_____
        plot_cp_rasters_deliv(deliv_st, neur_deliv_cp,
                              before_taste, dig_in_names[t_i], deliv_save_dir)
        # Store results for tastant
        taste_cp.append(neur_deliv_cp)

    return taste_cp


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
            plot_cp_rasters_pop(deliv_bin_full, best_peak_inds,
                                d_i, before_taste, dig_in_name, taste_cp_save_dir)
        taste_cp.append(deliv_cp)

    return taste_cp


def plot_cp_rasters_neur(neur_deliv_st, neur_cp_likelihood_list, before_taste, dig_in_name, n_i, num_deliv, num_cp, taste_cp_save_dir):
    # Delivery aligned
    fig = plt.figure(figsize=(5, 10))
    plt.eventplot(neur_deliv_st, colors='k', alpha=0.5)
    plt.eventplot(neur_cp_likelihood_list, colors='r')
    plt.axvline(before_taste, color='b')
    plt.title(dig_in_name + ' neuron ' + str(n_i) +
              ' raster aligned by taste deliv')
    plt.xlabel('Time (ms)')
    plt.ylabel('Delivery Index')
    fig_name = dig_in_name + '_neur_' + str(n_i) + '_rast_aligned_taste'
    fig.savefig(taste_cp_save_dir + fig_name + '.png')
    fig.savefig(taste_cp_save_dir + fig_name + '.svg')
    plt.close(fig)
    # CP aligned
    # for cp_i in range(num_cp):
    #	fig = plt.figure(figsize=(5,10))
    #	realigned_deliv_st = []
    #	realigned_cp = []
    #	for d_i in range(num_deliv):
    #		realign_d_i = np.array(neur_deliv_st[d_i]) - neur_cp_likelihood_list[d_i][cp_i]
    #		realigned_deliv_st.append(list(realign_d_i))
    #		realigned_cp.append(list(np.array(neur_cp_likelihood_list[d_i]) - neur_cp_likelihood_list[d_i][cp_i]))
    #	plt.eventplot(realigned_deliv_st,colors='k',alpha=0.5)
    #	plt.eventplot(realigned_cp,colors='r')
    #	plt.title(dig_in_name + ' neuron '+ str(n_i) + ' raster aligned by cp ' + str(cp_i))
    #	plt.xlabel('Aligned Index')
    #	plt.ylabel('Delivery Index')
    #	fig_name = dig_in_name + '_neur_' + str(n_i) + '_rast_aligned_cp_' + str(cp_i)
    #	fig.savefig(taste_cp_save_dir + fig_name + '.png')
    #	fig.savefig(taste_cp_save_dir + fig_name + '.svg')
    #	plt.close(fig)


def plot_cp_rasters_deliv(deliv_st, neur_deliv_cp, before_taste, dig_in_name, taste_cp_save_dir):
    num_deliv, num_neur, num_cp = np.shape(neur_deliv_cp)
    for d_i in range(num_deliv):
        # Delivery aligned
        fig = plt.figure(figsize=(5, 5))
        spike_times = deliv_st[d_i]
        plt.eventplot(spike_times, colors='b', alpha=0.5)
        plt.eventplot(neur_deliv_cp[d_i, :, :], colors='r')
        plt.title(dig_in_name + ' delivery ' + str(d_i))
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        fig_name = dig_in_name + '_deliv_' + str(d_i)
        fig.savefig(taste_cp_save_dir + fig_name + '.png')
        fig.savefig(taste_cp_save_dir + fig_name + '.svg')
        plt.close(fig)


def plot_cp_rasters_pop(deliv_bin, deliv_cp, d_i, before_taste, dig_in_name, taste_cp_save_dir):
    num_neur, num_dt = np.shape(deliv_bin)
    deliv_st = [np.where(deliv_bin[n_i, :] > 0)[0] -
                before_taste for n_i in range(num_neur)]
    fig_t = plt.figure(figsize=(5, 5))
    plt.eventplot(deliv_st)
    for cp_i in deliv_cp:
        plt.axvline(cp_i-before_taste, color='r')
    plt.title(dig_in_name + ' delivery ' + str(d_i))
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    fig_name = dig_in_name + '_deliv_' + str(d_i)
    fig_t.savefig(taste_cp_save_dir + fig_name + '.png')
    fig_t.savefig(taste_cp_save_dir + fig_name + '.svg')
    plt.close(fig_t)

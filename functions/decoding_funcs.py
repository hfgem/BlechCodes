#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:12:51 2023

@author: Hannah Germaine
A collection of decoding functions used across analyses.
"""

import os
import warnings
import json
import tqdm
import numpy as np

#file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
# os.chdir(file_path)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate
from scipy.stats import pearsonr
from random import sample


def taste_decoding_cp(tastant_spike_times, pop_taste_cp_raster_inds,
                      start_dig_in_times, end_dig_in_times, dig_in_names,
                      num_neur, num_cp, num_tastes, pre_taste_dt, post_taste_dt, save_dir):
    """Use Bayesian theory to decode tastes from activity and determine which 
    neurons are taste selective. The functions uses a "leave-one-out" method of
    decoding, where the single delivery to be decoded is left out of the fit 
    distributions in the probability calculations.

    Note: the last taste in the 
    tastant_spike_times, etc... variables is always 'none'.

    This function uses the population changepoint times to bin the spike counts for the 
    distribution."""
    warnings.filterwarnings('ignore')

    max_num_deliv = 0  # Find the maximum number of deliveries across tastants
    taste_num_deliv = np.zeros(num_tastes).astype('int')
    deliv_taste_index = []
    for t_i in range(num_tastes):  # Only perform on actual tastes
        num_deliv = len(tastant_spike_times[t_i])
        deliv_taste_index.extend(list((t_i*np.ones(num_deliv)).astype('int')))
        taste_num_deliv[t_i] = num_deliv
        if num_deliv > max_num_deliv:
            max_num_deliv = num_deliv
    total_num_deliv = np.sum(taste_num_deliv)
    del t_i, num_deliv

    # Determine the firing rate distributions for each neuron for each taste for each cp for each delivery
    # To use in LOO Decoding
    try:
        with open(os.path.join(save_dir, 'tastant_epoch_delivery.json'), 'r') as fp:
            tastant_epoch_delivery = json.load(fp)
        print("\tImported previously calculated firing rate distributions")
        max_hz_cp = 0
        for cp_i in range(num_cp):
            for t_i in range(num_tastes):
                for n_i in range(num_neur):
                    for d_i in range(max_num_deliv):
                        if np.nanmax(tastant_epoch_delivery[str(cp_i)][str(t_i)][str(n_i)][str(d_i)]['full']) > max_hz_cp:
                            max_hz_cp = np.nanmax(tastant_epoch_delivery[str(
                                cp_i)][str(t_i)][str(n_i)][str(d_i)]['full'])
    except:
        print("\tNow calculating firing rate distributions")
        tastant_epoch_delivery, max_hz_cp = fr_dist_calculator(num_cp, num_tastes, num_neur, max_num_deliv, taste_num_deliv,
                                                               pop_taste_cp_raster_inds, tastant_spike_times, start_dig_in_times,
                                                               pre_taste_dt, post_taste_dt)
        # Save the dictionary of firing rates so you don't have to calculate it again in the future
        with open(os.path.join(save_dir, 'tastant_epoch_delivery.json'), 'w') as fp:
            json.dump(tastant_epoch_delivery, fp)
        # Reload the data
        with open(os.path.join(save_dir, 'tastant_epoch_delivery.json'), 'r') as fp:
            tastant_epoch_delivery = json.load(fp)

    max_hz_cp = np.ceil(max_hz_cp).astype('int')
    # Perform "Leave-One-Out" decoding: one delivery is left out of the distributions
    # and then "decoded" probabilistically based on the distribution formed by
    # the other deliveries
    # mark with a 1 if successfully decoded
    taste_select_success_epoch = np.zeros(
        (num_cp, num_neur, num_tastes, max_num_deliv))
    p_taste_epoch = np.zeros(
        (num_neur, num_tastes, max_num_deliv, num_cp))  # by epoch
    loo_distribution_save_dir = save_dir + 'LOO_Distributions/'
    if os.path.isdir(loo_distribution_save_dir) == False:
        os.mkdir(loo_distribution_save_dir)
    print("\tNow performing leave-one-out calculations of decoding.")
    # d_i_o is the left out delivery
    for d_i_o in tqdm.tqdm(range(total_num_deliv)):

        # _____Calculate By Epoch_____
        hist_bins_cp = np.arange(0, max_hz_cp+1).astype('int')
        x_vals_cp = hist_bins_cp[:-1] + np.diff(hist_bins_cp)/2

        d_i, t_i, p_taste_fr_cp_neur, taste_success_fr_cp_neur = loo_epoch_decode(num_cp, num_tastes, num_neur, max_num_deliv,
                                                                                  tastant_epoch_delivery, max_hz_cp, x_vals_cp, hist_bins_cp, dig_in_names[
                                                                                      :-1], d_i_o,
                                                                                  loo_distribution_save_dir, deliv_taste_index, taste_num_deliv)
        p_taste_epoch[:, :, d_i, :] = p_taste_fr_cp_neur
        taste_select_success_epoch[:, :, t_i, d_i] = taste_success_fr_cp_neur

        del hist_bins_cp, x_vals_cp, d_i, t_i, p_taste_fr_cp_neur, taste_success_fr_cp_neur
    print('\n')
    # Now calculate the probability of successfully decoding as the fraction of deliveries successful
    # num cp x num neur x num tastes
    taste_select_prob_epoch = np.sum(
        taste_select_success_epoch, axis=3)/taste_num_deliv

    return p_taste_epoch, taste_select_prob_epoch


def fr_dist_calculator(num_cp, num_tastes, num_neur, max_num_deliv, taste_num_deliv,
                       pop_taste_cp_raster_inds, tastant_spike_times, start_dig_in_times,
                       pre_taste_dt, post_taste_dt):
    """Calculates firing rate distributions for all conditions to use in LOO decoding"""
    tastant_epoch_delivery = dict()  # Create a nested dictionary for storage of firing rates
    for cp_i in range(num_cp):
        tastant_epoch_delivery[cp_i] = dict()
        for t_i in range(num_tastes):
            tastant_epoch_delivery[cp_i][t_i] = dict()
            for n_i in range(num_neur):
                tastant_epoch_delivery[cp_i][t_i][n_i] = dict()
                for d_i in range(max_num_deliv):
                    tastant_epoch_delivery[cp_i][t_i][n_i][d_i] = dict()
    max_hz_cp = 0
    for cp_i in range(num_cp):
        for t_i in range(num_tastes):
            print('\t\tChangepoint ' + str(cp_i) + " Taste " + str(t_i))
            # Grab taste-related variables
            #taste_d_i = np.sum(taste_num_deliv[:t_i])
            #num_deliv = taste_num_deliv[t_i]
            taste_cp_pop = pop_taste_cp_raster_inds[t_i]
            t_i_spike_times = tastant_spike_times[t_i]
            t_i_dig_in_times = start_dig_in_times[t_i]
            for n_i in tqdm.tqdm(range(num_neur)):
                for d_i in range(max_num_deliv):
                    try: #If number of deliveries reaches that amount for the taste
                        raster_times = t_i_spike_times[d_i][n_i]
                        start_taste_i = t_i_dig_in_times[d_i]
                        deliv_cp_pop = taste_cp_pop[d_i, :] - pre_taste_dt
                        # Binerize the firing following taste delivery start
                        times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(
                            raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
                        bin_post_taste = np.zeros(post_taste_dt)
                        bin_post_taste[times_post_taste] += 1
                        # Grab changepoints
                        start_epoch = int(deliv_cp_pop[cp_i])
                        end_epoch = int(deliv_cp_pop[cp_i+1])
                        epoch_len = end_epoch - start_epoch
                        # Set bin sizes to use in breaking up the epoch for firing rates
                        # 50 ms increments up to full epoch size
                        bin_sizes = np.arange(50, epoch_len, 50)
                        if bin_sizes[-1] != epoch_len:
                            bin_sizes = np.concatenate(
                                (bin_sizes, epoch_len*np.ones(1)))
                        bin_sizes = bin_sizes.astype('int')
                        all_my_fr = []
                        # Get firing rates for each bin size and concatenate
                        for b_i in bin_sizes:
                            all_my_fr.extend(
                                [np.sum(bin_post_taste[i:i+b_i])/(b_i/1000) for i in range(end_epoch-b_i+1)])
                        tastant_epoch_delivery[cp_i][t_i][n_i][d_i]['full'] = all_my_fr
                        # Get singular firing rate for the entire epoch for decoding
                        tastant_epoch_delivery[cp_i][t_i][n_i][d_i]['true'] = np.sum(
                            bin_post_taste[start_epoch:end_epoch])/(epoch_len/1000)
                        # Calculate and update maximum firing rate
                        if np.nanmax(all_my_fr) > max_hz_cp:
                            max_hz_cp = np.nanmax(all_my_fr)
                    except:
                        tastant_epoch_delivery[cp_i][t_i][n_i][d_i]['full'] = []
                        tastant_epoch_delivery[cp_i][t_i][n_i][d_i]['true'] = np.nan

    return tastant_epoch_delivery, max_hz_cp


def loo_epoch_decode(num_cp, num_tastes, num_neur, max_num_deliv, tastant_epoch_delivery,
                     max_hz_cp, x_vals_cp, hist_bins_cp, dig_in_names, d_i_o,
                     save_dir, deliv_taste_index, taste_num_deliv):

    p_taste_fr_cp_neur = np.zeros((num_neur, num_tastes, num_cp))
    taste_success_fr_cp_neur = np.zeros((num_cp, num_neur))

    # Calculate which taste and delivery d_i_o is:
    t_i_true = deliv_taste_index[d_i_o]
    if t_i_true > 0:
        d_i_true = d_i_o - np.cumsum(taste_num_deliv)[t_i_true-1]
    else:
        d_i_true = d_i_o

    for cp_i in range(num_cp):
        # Fit the firing rate distributions for each neuron for each taste (use gamma distribution) and plot
        #print("\tFitting firing rate distributions by taste by neuron")
        p_fr_taste = np.zeros((num_tastes, num_neur, len(hist_bins_cp)))
        for t_i in range(num_tastes):
            for n_i in range(num_neur):
                full_data = []
                for d_i in range(max_num_deliv):
                    if t_i == t_i_true:
                        if d_i != d_i_true:
                            d_i_data = np.array(tastant_epoch_delivery[str(
                                cp_i)][str(t_i)][str(n_i)][str(d_i)]['full'])
                            d_i_data = d_i_data[~np.isnan(d_i_data)]
                            if len(d_i_data) > 0:
                                full_data.extend(list(d_i_data))
                    else:
                        d_i_data = np.array(tastant_epoch_delivery[str(
                            cp_i)][str(t_i)][str(n_i)][str(d_i)]['full'])
                        d_i_data = d_i_data[~np.isnan(d_i_data)]
                        if len(d_i_data) > 0:
                            full_data.extend(list(d_i_data))
                full_data_array = np.array(full_data)
                max_fr_data = np.nanmax(full_data_array)
                bin_centers = np.concatenate(
                    (np.linspace(0, np.max(max_fr_data), 20), (max_hz_cp+1)*np.ones(1)))
                bin_width = np.diff(bin_centers)[0]
                bin_edges = np.concatenate(
                    (bin_centers - bin_width, (bin_centers[-1] + np.diff(bin_centers)[-1])*np.ones(1)))
                fit_data = np.histogram(
                    full_data_array, density=True, bins=bin_edges)
                new_fit = interpolate.interp1d(
                    bin_centers, fit_data[0], kind='linear')
                filtered_data = new_fit(hist_bins_cp)
                # return to a probability density
                filtered_data = filtered_data/np.sum(filtered_data)
                p_fr_taste[t_i, n_i, :] = filtered_data
            del n_i, full_data, bin_centers, bin_edges, fit_data, new_fit, filtered_data

        if d_i_o == 0:
            # Plot the taste distributions against each other
            fig_t, ax_t = plt.subplots(
                nrows=num_neur, ncols=1, sharex=True, figsize=(5, num_neur))
            for n_i in range(num_neur):
                if n_i == 0:
                    ax_t[n_i].plot((p_fr_taste[:, n_i, :]).T,
                                   label=dig_in_names)
                    ax_t[n_i].legend()
                else:
                    ax_t[n_i].plot((p_fr_taste[:, n_i, :]).T)
            ax_t[num_neur-1].set_xlabel('Firing Rate (Hz)')
            fig_t.supylabel('Probability')
            plt.suptitle('LOO Delivery ' + str(d_i_o))
            fig_t.tight_layout()
            fig_t.savefig(save_dir + 'loo_' + str(d_i_o) +
                          '_epoch_' + str(cp_i) + '.png')
            fig_t.savefig(save_dir + 'loo_' + str(d_i_o) +
                          '_epoch_' + str(cp_i) + '.svg')
            plt.close(fig_t)

        # Calculate the taste probabilities by neuron by delivery
        #print("\tCalculating probability of successful decoding")
        # For each neuron, for each taste and its delivery, determine the probability of each of the tastes being that delivery
        # p(taste|fr) = [p(fr|taste)xp(taste)]/p(fr)
        loo_taste_num_deliv = np.zeros(np.shape(taste_num_deliv))
        loo_taste_num_deliv[:] = taste_num_deliv[:]
        loo_taste_num_deliv[t_i_true] -= 1
        p_taste = loo_taste_num_deliv/np.sum(loo_taste_num_deliv)
        for n_i in range(num_neur):
            # Calculate the probability of each taste for each epoch
            fr = tastant_epoch_delivery[str(cp_i)][str(
                t_i_true)][str(n_i)][str(d_i_true)]['true']
            # compare each taste against the true taste data
            for t_i_2 in range(num_tastes):
                closest_x = np.argmin(np.abs(x_vals_cp - fr))
                p_fr = np.nansum(np.squeeze(
                    p_fr_taste[:, n_i, closest_x]))/num_tastes
                p_taste_fr_cp_neur[n_i, t_i_2, cp_i] = (
                    p_fr_taste[t_i_2, n_i, closest_x]*p_taste[t_i_2])/p_fr
            # Since the probability of each taste is calculated, now we determine
            #	if the highest probability taste aligns with the truly delivered taste
            if t_i_true == np.argmax(p_taste_fr_cp_neur[n_i, :, cp_i]):
                taste_success_fr_cp_neur[cp_i, n_i] = 1

    return d_i_true, t_i_true, p_taste_fr_cp_neur, taste_success_fr_cp_neur


def taste_fr_dist(num_neur, num_cp, tastant_spike_times,
                  taste_cp_raster_inds, pop_taste_cp_raster_inds,
                  start_dig_in_times, pre_taste_dt, post_taste_dt):
    """This function calculates fr distributions for each neuron for
    each taste delivery for each epoch"""

    num_tastes = len(tastant_spike_times)

    max_num_deliv = 0  # Find the maximum number of deliveries across tastants
    taste_num_deliv = np.zeros(num_tastes).astype('int')
    deliv_taste_index = []
    for t_i in range(num_tastes):
        num_deliv = len(tastant_spike_times[t_i])
        deliv_taste_index.extend(list((t_i*np.ones(num_deliv)).astype('int')))
        taste_num_deliv[t_i] = num_deliv
        if num_deliv > max_num_deliv:
            max_num_deliv = num_deliv
    del t_i, num_deliv

    # Determine the spike fr distributions for each neuron for each taste
    #print("\tPulling spike fr distributions by taste by neuron")
    tastant_fr_dist_pop = dict()  # Population firing rate distributions by epoch
    for t_i in range(num_tastes):
        tastant_fr_dist_pop[t_i] = dict()
        for n_i in range(num_neur):
            tastant_fr_dist_pop[t_i][n_i] = dict()
            for d_i in range(max_num_deliv):
                tastant_fr_dist_pop[t_i][n_i][d_i] = dict()
                for cp_i in range(num_cp):
                    tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = dict()
    # ____
    max_hz_pop = 0
    for t_i in range(num_tastes):
        num_deliv = taste_num_deliv[t_i]
        taste_cp_pop = pop_taste_cp_raster_inds[t_i]
        for n_i in range(num_neur):
            for d_i in range(num_deliv):  # index for that taste
                raster_times = tastant_spike_times[t_i][d_i][n_i]
                start_taste_i = start_dig_in_times[t_i][d_i]
                deliv_cp_pop = taste_cp_pop[d_i, :] - pre_taste_dt
                # Bin the average firing rates following taste delivery start
                times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(
                    raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
                bin_post_taste = np.zeros(post_taste_dt)
                bin_post_taste[times_post_taste] += 1
                for cp_i in range(num_cp):
                    # population changepoints
                    start_epoch = int(deliv_cp_pop[cp_i])
                    end_epoch = int(deliv_cp_pop[cp_i+1])
                    epoch_len = end_epoch - start_epoch
                    # Set bin sizes to use in breaking up the epoch for firing rates
                    # 50 ms increments up to full epoch size
                    bin_sizes = np.arange(50, epoch_len, 50)
                    if bin_sizes[-1] != epoch_len:
                        bin_sizes = np.concatenate(
                            (bin_sizes, epoch_len*np.ones(1)))
                    bin_sizes = bin_sizes.astype('int')
                    all_hz_bst = []
                    # Get firing rates for each bin size and concatenate
                    for b_i in bin_sizes:
                        all_hz_bst.extend(
                            [np.sum(bin_post_taste[i:i+b_i])/(b_i/1000) for i in range(end_epoch-b_i+1)])
                    tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = all_hz_bst
                    if np.nanmax(all_hz_bst) > max_hz_pop:
                        max_hz_pop = np.nanmax(all_hz_bst)
                del cp_i, start_epoch, end_epoch
                # ___
    del t_i, num_deliv, n_i, d_i, raster_times, start_taste_i, times_post_taste, bin_post_taste

    return tastant_fr_dist_pop, max_hz_pop, taste_num_deliv


def taste_fr_dist_zscore(num_neur, num_cp, tastant_spike_times, segment_spike_times,
                         segment_names, segment_times, taste_cp_raster_inds, pop_taste_cp_raster_inds,
                         start_dig_in_times, pre_taste_dt, post_taste_dt, bin_dt):
    """This function calculates z-scored firing rate distributions for each neuron for
    each taste delivery for each epoch"""

    num_tastes = len(tastant_spike_times)
    half_bin = np.floor(bin_dt/2).astype('int')

    max_num_deliv = 0  # Find the maximum number of deliveries across tastants
    taste_num_deliv = np.zeros(num_tastes).astype('int')
    deliv_taste_index = []
    for t_i in range(num_tastes):
        num_deliv = len(tastant_spike_times[t_i])
        deliv_taste_index.extend(list((t_i*np.ones(num_deliv)).astype('int')))
        taste_num_deliv[t_i] = num_deliv
        if num_deliv > max_num_deliv:
            max_num_deliv = num_deliv
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
        for tb_i, tb in enumerate(tqdm.tqdm(time_bin_starts)):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_bin:tb+half_bin-seg_start], 1)/(2*half_bin*(1/1000))
        mean_fr = np.mean(tb_fr, 1)
        std_fr = np.std(tb_fr, 1)
    else:
        mean_fr = np.zeros(num_neur)
        std_fr = np.zeros(num_neur)

    # Determine the spike fr distributions for each neuron for each taste
    #print("\tPulling spike fr distributions by taste by neuron")
    tastant_fr_dist_pop = dict()  # Population firing rate distributions by epoch
    for t_i in range(num_tastes):
        tastant_fr_dist_pop[t_i] = dict()
        for n_i in range(num_neur):
            tastant_fr_dist_pop[t_i][n_i] = dict()
            for d_i in range(max_num_deliv):
                tastant_fr_dist_pop[t_i][n_i][d_i] = dict()
                for cp_i in range(num_cp):
                    tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = dict()
    # ____
    max_hz_pop = 0
    min_hz_pop = 0
    for t_i in range(num_tastes):
        num_deliv = taste_num_deliv[t_i]
        taste_cp_pop = pop_taste_cp_raster_inds[t_i]
        for n_i in range(num_neur):
            for d_i in range(num_deliv):  # index for that taste
                raster_times = tastant_spike_times[t_i][d_i][n_i]
                start_taste_i = start_dig_in_times[t_i][d_i]
                deliv_cp_pop = taste_cp_pop[d_i, :] - pre_taste_dt
                # Bin the average firing rates following taste delivery start
                times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(
                    raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
                bin_post_taste = np.zeros(post_taste_dt)
                bin_post_taste[times_post_taste] += 1
                for cp_i in range(num_cp):
                    # population changepoints
                    start_epoch = int(deliv_cp_pop[cp_i])
                    end_epoch = int(deliv_cp_pop[cp_i+1])
                    epoch_len = end_epoch - start_epoch
                    # Set bin sizes to use in breaking up the epoch for firing rates
                    # 50 ms increments up to full epoch size
                    bin_sizes = np.arange(50, epoch_len, 50)
                    if bin_sizes[-1] != epoch_len:
                        bin_sizes = np.concatenate(
                            (bin_sizes, epoch_len*np.ones(1)))
                    bin_sizes = bin_sizes.astype('int')
                    all_my_fr = []
                    # Get firing rates for each bin size and concatenate
                    for b_i in bin_sizes:
                        bst_hz = [np.sum(bin_post_taste[i:i+b_i])/(b_i/1000)
                                  for i in range(end_epoch-b_i+1)]
                        bst_hz_z = (np.array(bst_hz) -
                                    mean_fr[n_i])/std_fr[n_i]
                        all_my_fr.extend(list(bst_hz_z))
                    tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = all_my_fr
                    if np.nanmax(bst_hz_z) > max_hz_pop:
                        max_hz_pop = np.nanmax(bst_hz_z)
                    if np.nanmin(bst_hz_z) < min_hz_pop:
                        min_hz_pop = np.nanmin(bst_hz_z)
                del cp_i, start_epoch, end_epoch, bst_hz, bst_hz_z

    del t_i, num_deliv, n_i, d_i, raster_times, start_taste_i, times_post_taste, bin_post_taste

    return tastant_fr_dist_pop, max_hz_pop, min_hz_pop, taste_num_deliv


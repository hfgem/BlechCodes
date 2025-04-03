#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:46:03 2023

@author: Hannah Germaine
Set of miscellaneous functions to support analyses in analyze_states.py.
"""

import time
import tables
import tqdm
import os
import csv
import random
import numpy as np
import functions.load_intan_rhd_format.load_intan_rhd_format as rhd
import functions.data_processing as dp
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import find_peaks, savgol_filter


def add_no_taste(start_dig_in_times, end_dig_in_times, dig_in_names):

    # Add to the tastants / dig_ins a "no taste delivered" control
    all_start_times = []
    all_end_times = []
    num_deliv = []
    for t_i in range(len(start_dig_in_times)):  # for each taste
        all_start_times.extend(start_dig_in_times[t_i])
        num_deliv.extend([len(start_dig_in_times[t_i])])
        all_end_times.extend(end_dig_in_times[t_i])

    all_start_times = np.array(all_start_times)
    all_end_times = np.array(all_end_times)
    num_none = np.ceil(np.mean(np.array(num_deliv))).astype('int')
    time_before_vec = np.random.random_integers(10000, 20000, size=(
        num_none))  # Number of seconds before taste delivery to grab
    dig_in_len_vec = all_end_times - all_start_times
    none_len_vec = np.array(random.sample(list(dig_in_len_vec), num_none))
    none_start_times = np.array(random.sample(
        list(all_start_times), num_none)) - time_before_vec
    none_end_times = none_start_times + none_len_vec

    dig_in_names.extend(['none'])
    start_dig_in_times.append(list(none_start_times))
    end_dig_in_times.append(list(none_end_times))

    num_tastes = len(start_dig_in_times)

    return dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes


def calc_segment_spike_times(segment_times, spike_times, num_neur):
    segment_spike_times = []
    for s_i in tqdm.tqdm(range(len(segment_times)-1)):
        min_time = segment_times[s_i]  # in ms
        max_time = segment_times[s_i+1]  # in ms
        s_t = [list(np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= min_time)
                    * (np.array(spike_times[i]) <= max_time))[0]]) for i in range(num_neur)]
        segment_spike_times.append(s_t)

    return segment_spike_times


def calc_tastant_spike_times(segment_times, spike_times, start_dig_in_times, end_dig_in_times, pre_taste, post_taste, num_tastes, num_neur):
    tastant_spike_times = []
    pre_taste_dt = int(np.ceil(pre_taste*(1000/1)))  # Convert to ms timescale
    post_taste_dt = int(np.ceil(post_taste*(1000/1))
                        )  # Convert to ms timescale
    for t_i in tqdm.tqdm(range(num_tastes)):
        t_start = start_dig_in_times[t_i]
        t_end = end_dig_in_times[t_i]
        t_st = []
        for t_d_i in range(len(t_start)):
            start_i = int(max(t_start[t_d_i] - pre_taste_dt, 0))
            end_i = int(min(t_end[t_d_i] + post_taste_dt,
                        segment_times[-1]*1000))
            # Grab spike times into one list
            s_t = [list(np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= start_i)
                        * (np.array(spike_times[i]) <= end_i))[0]]) for i in range(num_neur)]
            t_st.append(s_t)
        tastant_spike_times.append(t_st)

    return tastant_spike_times

def taste_responsivity_PSTH(PSTH_times, PSTH_taste_deliv_times, tastant_PSTH):
    """A test of whether or not each neuron is taste responsive by looking at
    PSTH activity before and after taste delivery and calculating if there is a 
    significant change for each delivery - probability of taste responsivity 
    is then the fraction of deliveries where there was a significant response"""

    num_tastes = len(PSTH_taste_deliv_times)
    taste_responsivity_probability = []
    for t_i in range(num_tastes):
        [num_deliv, num_neur, len_PSTH] = np.shape(tastant_PSTH[t_i])
        taste_responsive_neur = np.zeros((num_neur, num_deliv))
        for n_i in range(num_neur):
            start_taste_i = np.where(
                PSTH_times[t_i] == PSTH_taste_deliv_times[t_i][0])[0][0]
            end_taste_i = np.where(
                PSTH_times[t_i] == PSTH_taste_deliv_times[t_i][1])[0][0]
            for d_i in range(num_deliv):
                pre_taste_PSTH_vals = tastant_PSTH[t_i][d_i,
                                                        n_i, :start_taste_i]
                post_taste_PSTH_vals = tastant_PSTH[t_i][d_i,
                                                         n_i, end_taste_i:end_taste_i+start_taste_i]
                p_val, _ = stats.ks_2samp(
                    pre_taste_PSTH_vals, post_taste_PSTH_vals)
                if p_val < 0.05:
                    taste_responsive_neur[n_i, d_i] = 1
        taste_responsivity_probability.append(
            np.sum(taste_responsive_neur, 1)/num_deliv)

    return taste_responsivity_probability


def taste_responsivity_raster(tastant_spike_times, start_dig_in_times, end_dig_in_times, num_neur, pre_taste_dt):
    """A test of whether or not each neuron is taste responsive by looking at
    spike activity before and after taste delivery and calculating if there is 
    a significant change for each delivery - probability of taste responsivity 
    is then the fraction of deliveries where there was a significant response"""
    bin_sum_dt = 100  # in ms = dt
    num_tastes = len(tastant_spike_times)
    colors = cm.cool(np.arange(num_tastes-1)/(num_tastes-1))
    taste_responsivity_probability = []
    taste_responsivity_binary = np.array([True for n_i in range(num_neur)])
    for t_i in range(num_tastes-1):  # Last taste is always "none" and shouldn't be counted
        num_deliv = len(tastant_spike_times[t_i])
        taste_responsive_neur = np.zeros((num_neur, num_deliv))
        for n_i in tqdm.tqdm(range(num_neur)):
            for d_i in range(num_deliv):
                raster_times = tastant_spike_times[t_i][d_i][n_i]
                start_taste_i = start_dig_in_times[t_i][d_i]
                end_taste_i = end_dig_in_times[t_i][d_i]
                times_pre_taste = (np.array(raster_times)[np.where(raster_times < start_taste_i)[
                                   0]] - (start_taste_i - pre_taste_dt)).astype('int')
                bin_pre_taste = np.zeros(pre_taste_dt)
                bin_pre_taste[times_pre_taste] += 1
                times_post_taste = (np.array(raster_times)[np.where((raster_times > end_taste_i)*(
                    raster_times < end_taste_i + pre_taste_dt))[0]] - end_taste_i).astype('int')
                bin_post_taste = np.zeros(pre_taste_dt)
                bin_post_taste[times_post_taste] += 1
                pre_taste_spike_nums = [sum(
                    bin_pre_taste[b_i:b_i+bin_sum_dt]) for b_i in range(pre_taste_dt - bin_sum_dt)]
                post_taste_spike_nums = [sum(
                    bin_post_taste[b_i:b_i+bin_sum_dt]) for b_i in range(pre_taste_dt - bin_sum_dt)]
                # Since these are stochastic samples, assuming a mean fr pre and post, we can use the Mann-Whitney-U Test
                try:
                    _, p_val = stats.mannwhitneyu(
                        pre_taste_spike_nums, post_taste_spike_nums)
                    if p_val < 0.05:
                        taste_responsive_neur[n_i, d_i] = 1
                except:
                    pass
        taste_responsivity_probability.append(
            np.sum(taste_responsive_neur, 1)/num_deliv)
        taste_responsivity_binary *= (np.sum(taste_responsive_neur,
                                      1)/num_deliv > 1/2)
    for t_i in range(num_tastes-1):
        plt.plot(
            taste_responsivity_probability[t_i], color=colors[t_i], label=str(t_i))
    plt.legend()

    return taste_responsivity_probability, taste_responsivity_binary


def taste_discriminability_test(post_taste_dt, num_tastes, tastant_spike_times,
                                num_neur, start_dig_in_times, bin_size, discrim_save_dir):
    # Pull spike rasters to use in ANOVA
    taste_fr_data = []
    for t_i in range(num_tastes):
        t_st = tastant_spike_times[t_i]
        num_deliv = len(t_st)
        deliv_rasters = np.zeros((num_deliv, num_neur, post_taste_dt+1))
        for d_i in range(num_deliv):
            st_d_i = start_dig_in_times[t_i][d_i]
            for n_i in range(num_neur):
                st_n_d_i = (t_st[d_i][n_i] - st_d_i).astype('int')
                st_n_d_i = st_n_d_i[st_n_d_i >= 0]
                st_n_d_i = st_n_d_i[st_n_d_i < post_taste_dt]
                deliv_rasters[d_i, n_i, st_n_d_i] = 1
        taste_fr_data.append(deliv_rasters)
    # Run ANOVA on time bins of taste response
    x_gauss = np.arange(bin_size)
    interval_gauss = np.exp(-(x_gauss - bin_size/2)**2/(2*(bin_size/6)**2))
    anova_results_all = np.zeros(
        (num_neur, post_taste_dt))  # including "none" taste
    anova_results_true = np.zeros(
        (num_neur, post_taste_dt))  # only true tastes
    for b_i in np.arange(post_taste_dt):
        start_bin = np.max([np.floor(b_i - bin_size/2).astype('int'), 0])
        start_bin_diff = np.abs(np.floor(b_i - bin_size/2).astype('int')-0)
        end_bin = np.min(
            [post_taste_dt, np.floor(b_i + bin_size/2).astype('int')])
        end_bin_diff = np.abs(
            post_taste_dt-np.floor(b_i + bin_size/2).astype('int'))
        if start_bin == 0:
            gauss_cut = interval_gauss[start_bin_diff:]
        elif end_bin == post_taste_dt:
            if end_bin_diff > 0:
                gauss_cut = interval_gauss[:-1*end_bin_diff]
            else:
                gauss_cut = interval_gauss
        else:
            gauss_cut = interval_gauss
        for n_i in range(num_neur):
            t_fr = []
            for t_i in range(num_tastes):
                rast_data = np.squeeze(
                    taste_fr_data[t_i][:, n_i, start_bin:end_bin])
                fr_data = np.sum(rast_data, 1) / \
                    (bin_size/1000)  # converted to Hz
                t_fr.append(list(fr_data))
            eval_string = 'stats.f_oneway('
            for t_i in range(num_tastes):
                eval_string += 't_fr[' + str(t_i) + ']'
                if t_i < num_tastes-1:
                    eval_string += ','
                else:
                    eval_string += ')'
            a_stat, a_pval = eval(eval_string)
            if a_pval <= 0.05:
                anova_results_all[n_i, start_bin:end_bin] += gauss_cut
            eval_string = 'stats.f_oneway('
            for t_i in range(num_tastes-1):
                eval_string += 't_fr[' + str(t_i) + ']'
                if t_i < num_tastes-2:
                    eval_string += ','
                else:
                    eval_string += ')'
            a_stat, a_pval = eval(eval_string)
            if a_pval <= 0.05:
                anova_results_true[n_i, start_bin:end_bin] += gauss_cut

    # Plot the anova significant difference results
    f = plt.figure(figsize=(5, 5))
    plt.imshow(anova_results_all, aspect='auto')
    plt.title('Anova of all tastes in sliding bins')
    plt.xlabel('Time post-taste delivery (ms)')
    plt.ylabel('Neuron Index')
    plt.tight_layout()
    f.savefig(os.path.join(discrim_save_dir, 'anova_all.png'))
    f.savefig(os.path.join(discrim_save_dir, 'anova_all.svg'))
    plt.close(f)
    f = plt.figure(figsize=(5, 5))
    plt.imshow(anova_results_true, aspect='auto')
    plt.title('Anova of true tastes in sliding bins')
    plt.xlabel('Time post-taste delivery (ms)')
    plt.ylabel('Neuron Index')
    plt.tight_layout()
    f.savefig(os.path.join(discrim_save_dir, 'anova_true.png'))
    f.savefig(os.path.join(discrim_save_dir, 'anova_true.svg'))
    plt.close(f)
    f, ax = plt.subplots(2, 1, figsize=(5, 5))
    ax[0].plot(np.sum(anova_results_all, 0))
    ax[0].set_title('Summed ANOVA All')
    ax[1].plot(np.sum(anova_results_true, 0))
    ax[1].set_title('Summed ANOVA True')
    plt.tight_layout()
    f.savefig(os.path.join(discrim_save_dir, 'summed_anovas.png'))
    f.savefig(os.path.join(discrim_save_dir, 'summed_anovas.svg'))
    plt.close(f)

    # Now pull intervals of taste discriminability based on true
    discrim_true = np.sum(anova_results_true, 0)
    smooth_discrim = savgol_filter(discrim_true, int(bin_size/2), 1)
    #smooth_discrim2 = savgol_filter(smooth_discrim, int(bin_size/2), 1)
    [peaks, _] = find_peaks(smooth_discrim, distance=bin_size, rel_height=1)
    [troughs, _] = find_peaks(-smooth_discrim, distance=bin_size, rel_height=1)
    # Make sure there's a trough location between all consecutive peaks
    for p_i in range(len(peaks)-1):
        trough_ind = np.where(
            (troughs < peaks[p_i+1])*(troughs > peaks[p_i]))[0]
        if len(trough_ind) == 0:
            troughs = np.sort(np.concatenate(
                (troughs, ((peaks[p_i] + (peaks[p_i+1] - peaks[p_i])/2)*np.ones(1)).astype('int'))))
    # Make sure there's a peak location between all consecutive troughs
    for t_i in range(len(troughs)-1):
        peak_ind = np.where((peaks < troughs[t_i+1])*(peaks > troughs[t_i]))[0]
        if len(peak_ind) == 0:
            peaks = np.sort(np.concatenate(
                (peaks, ((troughs[t_i] + (troughs[t_i+1] - troughs[t_i])/2)*np.ones(1)).astype('int'))))
    # Add a trouch at 0
    if troughs[0] > peaks[0]:
        troughs = np.concatenate((np.zeros(1), troughs))
    # Add an ending trough
    if troughs[-1] < peaks[-1]:
        troughs = np.concatenate((troughs, post_taste_dt*np.ones(1)))
    # Now pull where the epochs start and end using the troughs
    peak_epochs = troughs.astype('int')
    discriminable_segments = np.zeros(post_taste_dt)
    for p_i in range(len(peaks)):
        discriminable_segments[peak_epochs[p_i]:peak_epochs[p_i+1]] = p_i+1

    f, ax = plt.subplots(3, 1, figsize=(5, 5))
    ax[0].plot(discrim_true)
    ax[0].set_title('Summed ANOVA True')
    ax[1].plot(smooth_discrim)
    for p_i in peaks:
        ax[1].axvline(p_i, linestyle='dashed', alpha=0.5, color='b')
    for t_i in troughs:
        ax[1].axvline(t_i, linestyle='dashed', alpha=0.5, color='r')
    ax[1].set_title('Smoothed ANOVA True')
    ax[2].plot(discriminable_segments)
    ax[2].set_title('Segmented Smoothed ANOVA True')
    plt.tight_layout()
    f.savefig(os.path.join(discrim_save_dir, 'discriminable_intervals.png'))
    f.savefig(os.path.join(discrim_save_dir, 'discriminable_intervals.svg'))
    plt.close(f)

    # Determine which neurons are most discriminative in each interval
    discrim_neur = np.zeros((len(peaks), num_neur))
    for e_i in range(len(peaks)):
        best_neur = np.where(
            np.sum(anova_results_true[:, peak_epochs[e_i]:peak_epochs[e_i+1]], 1) > 0)[0]
        discrim_neur[e_i, best_neur] = 1
    f = plt.figure(figsize=(5, 5))
    plt.imshow(discrim_neur, aspect='auto')
    plt.xlabel('Neuron Index')
    plt.ylabel('Discriminability Epoch')
    plt.tight_layout()
    f.savefig(os.path.join(discrim_save_dir, 'discriminable_neurons.png'))
    f.savefig(os.path.join(discrim_save_dir, 'discriminable_neurons.svg'))
    plt.close(f)

    return anova_results_all, anova_results_true, peak_epochs, discrim_neur


def taste_response_rasters(num_tastes, num_neur, tastant_spike_times,
                           start_dig_in_times, cp_raster_inds, pre_taste_dt):
    """This function pulls the taste response rasters, by epoch, into dicts"""
    #Set up storage dictionary
    num_tastes = len(tastant_spike_times)
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    max_num_deliv = 0  # Find the maximum number of deliveries across tastants
    for t_i in range(num_tastes):
        num_deliv = len(tastant_spike_times[t_i])
        if num_deliv > max_num_deliv:
            max_num_deliv = num_deliv
    del t_i, num_deliv
    
    taste_num_deliv = np.zeros(num_tastes)
    for t_i in range(num_tastes):
        num_deliv = len(tastant_spike_times[t_i][:])
        taste_num_deliv[t_i] = num_deliv
    del t_i, num_deliv
    
    tastant_raster_dict = dict() #Store the rasters by epoch
    for t_i in range(num_tastes):
        tastant_raster_dict[t_i] = dict()
        for d_i in range(max_num_deliv):
            tastant_raster_dict[t_i][d_i] = dict()
    
    for t_i in range(num_tastes):
        num_deliv = int(taste_num_deliv[t_i])
        taste_cp = cp_raster_inds[t_i]
        for d_i in range(num_deliv):
            raster_times = tastant_spike_times[t_i][d_i]
            start_taste_i = start_dig_in_times[t_i][d_i]
            deliv_cp = taste_cp[d_i, :] - pre_taste_dt
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
                tastant_raster_dict[t_i][d_i][cp_i] = td_i_bin
    
    return tastant_raster_dict

def get_bin_activity(segment_times_reshaped, segment_spike_times, bin_size, 
                     segments_to_analyze = [], no_z = False):
    """Pull firing rate vectors (regular and z-scored) for sliding bins across 
    rest intervals and return along with bin start times"""
    
    num_segments = len(segment_spike_times)
    half_bin = np.ceil(bin_size/2).astype(int)

    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
    else:
        num_segments = len(segments_to_analyze)
    
    bin_times = []
    bin_pop_fr = dict()
    bin_fr_vecs = dict()
    bin_fr_vecs_zscore = dict()
    for s_i, s_ind in tqdm.tqdm(enumerate(segments_to_analyze)):
        seg_times = segment_times_reshaped[s_ind]
        seg_len = np.ceil(seg_times[1] - seg_times[0] + 1).astype('int')
        seg_st = segment_spike_times[s_ind] 
        num_neur = len(seg_st)
        #Pull binary raster of segment
        bin_spike_storage = np.zeros((num_neur,seg_len))
        for n_i in range(num_neur):
            neur_st = (np.array(seg_st[n_i]) - seg_times[0]).astype('int')
            bin_spike_storage[n_i,neur_st] = 1
        del neur_st, seg_st
        #Grab neuron firing rates across bins in segment
        seg_fr_vecs = []
        seg_pop_fr = []
        seg_bin_times = np.arange(half_bin,seg_len-half_bin)
        for st_i, st_ind in enumerate(seg_bin_times):
            seg_fr_vecs.append(list(np.sum(bin_spike_storage[:,st_ind-half_bin:st_ind+half_bin],axis=1)/(2*half_bin/1000))) #In Hz
            seg_pop_fr.extend([np.sum(bin_spike_storage[:,st_ind-half_bin:st_ind+half_bin])/(2*half_bin/1000)/num_neur]) #In Hz
        #Calculate z-scored fr vecs
        mean_fr = np.mean(np.array(seg_fr_vecs),1)
        std_fr = np.std(np.array(seg_fr_vecs),1)
        seg_fr_vecs_z = list((np.array(seg_fr_vecs)-np.expand_dims(mean_fr,1))/np.expand_dims(std_fr,1))
        #Store
        bin_times.append(seg_times)
        bin_pop_fr[s_i] = seg_pop_fr
        bin_fr_vecs[s_i] = seg_fr_vecs
        bin_fr_vecs_zscore[s_i] = seg_fr_vecs_z
    
    return bin_times, bin_pop_fr, bin_fr_vecs, bin_fr_vecs_zscore
    

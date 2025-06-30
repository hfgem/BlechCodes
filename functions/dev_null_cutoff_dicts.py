#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 16:04:51 2025

@author: hannahgermaine
"""

def dev_null_dict_creation(num_null, null_dev_lengths, null_dev_neuron_counts,
                           null_dev_spike_counts, null_dev_frs, true_dev_neuron_counts,
                           true_dev_spike_counts, true_dev_lengths, true_dev_frs,
                           neur_count_dict, neur_spike_dict, neur_fr_dict,
                           neur_len_dict, seg_name):
    
    # Neuron count data
    null_max_neur_count = np.max(
        [np.max(null_dev_neuron_counts[null_i]) for null_i in range(num_null)])
    max_neur_count = int(
        np.max([np.max(null_max_neur_count), np.max(true_dev_neuron_counts)]))
    neur_x_vals = np.arange(10, max_neur_count)
    true_neur_x_val_counts = np.nan*np.ones(np.shape(neur_x_vals))
    null_neur_x_val_counts_all = []
    null_neur_x_val_counts_mean = np.nan * \
        np.ones(np.shape(neur_x_vals))
    null_neur_x_val_counts_std = np.nan * \
        np.ones(np.shape(neur_x_vals))
    for n_cut_i, n_cut in enumerate(neur_x_vals):
        true_neur_x_val_counts[n_cut_i] = np.sum(
            (np.array(true_dev_neuron_counts) > n_cut).astype('int'))
        null_neur_x_val_counts = []
        for null_i in range(num_null):
            null_neur_x_val_counts.append(
                np.sum((np.array(null_dev_neuron_counts[null_i]) > n_cut).astype('int')))
        null_neur_x_val_counts_all.append(null_neur_x_val_counts)
        null_neur_x_val_counts_mean[n_cut_i] = np.nanmean(
            null_neur_x_val_counts)
        null_neur_x_val_counts_std[n_cut_i] = np.nanstd(
            null_neur_x_val_counts)
        # Plot the individual distribution
        dpf.plot_dev_x_null_single_dist(
            null_neur_x_val_counts, true_neur_x_val_counts[n_cut_i], 'neur_count_cutoff_' + str(n_cut), seg_fig_save_dir)
    neur_count_dict[seg_name + '_true'] = [list(neur_x_vals),
                                           list(true_neur_x_val_counts)]
    neur_count_dict[seg_name + '_null'] = [list(neur_x_vals),
                                           list(
                                               null_neur_x_val_counts_mean),
                                           list(null_neur_x_val_counts_std)]
    # Calculate percentiles
    percentiles = []  # Calculate percentile of true data point in null data distribution
    for n_cut in neur_x_vals:
        try:
            percentiles.extend([round(stats.percentileofscore(
                null_neur_x_val_counts_all[n_cut-1], true_neur_x_val_counts[n_cut-1]), 2)])
        except:
            percentiles.extend([100])
    neur_count_dict[seg_name +
                    '_percentile'] = [list(neur_x_vals), percentiles]

    # Spike count data
    null_max_neur_spikes = np.max(
        [np.max(null_dev_spike_counts[null_i]) for null_i in range(num_null)])
    max_spike_count = int(
        np.max([np.max(null_max_neur_spikes), np.max(true_dev_spike_counts)]))
    spike_x_vals = np.arange(1, max_spike_count)
    true_neur_x_val_spikes = np.nan*np.ones(np.shape(spike_x_vals))
    null_neur_x_val_spikes_all = []
    null_neur_x_val_spikes_mean = np.nan * \
        np.ones(np.shape(spike_x_vals))
    null_neur_x_val_spikes_std = np.nan * \
        np.ones(np.shape(spike_x_vals))
    for s_cut_i, s_cut in enumerate(spike_x_vals):
        true_neur_x_val_spikes[s_cut_i] = np.sum(
            (np.array(true_dev_spike_counts) > s_cut).astype('int'))
        null_neur_x_val_spikes = []
        for null_i in range(num_null):
            null_neur_x_val_spikes.append(
                np.sum((np.array(null_dev_spike_counts[null_i]) > s_cut).astype('int')))
        null_neur_x_val_spikes_all.append(null_neur_x_val_spikes)
        null_neur_x_val_spikes_mean[s_cut_i] = np.nanmean(
            null_neur_x_val_spikes)
        null_neur_x_val_spikes_std[s_cut_i] = np.nanstd(
            null_neur_x_val_spikes)
        # Plot the individual distribution
        dpf.plot_dev_x_null_single_dist(
            null_neur_x_val_spikes, true_neur_x_val_spikes[s_cut_i], 'spike_count_cutoff_' + str(s_cut), seg_fig_save_dir)
    neur_spike_dict[seg_name + '_true'] = [list(spike_x_vals),
                                           list(true_neur_x_val_spikes)]
    neur_spike_dict[seg_name + '_null'] = [list(spike_x_vals),
                                           list(
                                               null_neur_x_val_spikes_mean),
                                           list(null_neur_x_val_spikes_std)]
    percentiles = []  # Calculate percentile of true data point in null data distribution
    for s_cut in spike_x_vals:
        try:
            percentiles.extend([round(stats.percentileofscore(
                null_neur_x_val_spikes_all[s_cut-1], true_neur_x_val_spikes[s_cut-1]), 2)])
        except:
            percentiles.extend([100])
    neur_spike_dict[seg_name +
                    '_percentile'] = [list(spike_x_vals), percentiles]

    # Burst length data
    null_max_neur_len = np.max(
        [np.max(null_dev_lengths[null_i]) for null_i in range(num_null)])
    max_len = int(
        np.max([np.max(null_max_neur_len), np.max(true_dev_lengths)]))
    len_x_vals = np.arange(min_dev_size, max_len)
    true_neur_x_val_lengths = np.nan*np.ones(np.shape(len_x_vals))
    null_neur_x_val_lengths_all = []
    null_neur_x_val_lengths_mean = np.nan * \
        np.ones(np.shape(len_x_vals))
    null_neur_x_val_lengths_std = np.nan * \
        np.ones(np.shape(len_x_vals))
    for l_cut_i, l_cut in enumerate(len_x_vals):
        true_neur_x_val_lengths[l_cut_i] = np.sum(
            (np.array(true_dev_lengths) > l_cut).astype('int'))
        null_neur_x_val_lengths = []
        for null_i in range(num_null):
            null_neur_x_val_lengths.append(
                np.sum((np.array(null_dev_lengths[null_i]) > l_cut).astype('int')))
        null_neur_x_val_lengths_all.append(null_neur_x_val_lengths)
        null_neur_x_val_lengths_mean[l_cut_i] = np.nanmean(
            null_neur_x_val_lengths)
        null_neur_x_val_lengths_std[l_cut_i] = np.nanstd(
            null_neur_x_val_lengths)
        # Plot the individual distribution
        dpf.plot_dev_x_null_single_dist(
            null_neur_x_val_lengths, true_neur_x_val_lengths[l_cut_i], 'length_cutoff_' + str(l_cut), seg_fig_save_dir)
    neur_len_dict[seg_name + '_true'] = [list(len_x_vals),
                                         list(true_neur_x_val_lengths)]
    neur_len_dict[seg_name + '_null'] = [list(len_x_vals),
                                         list(
                                             null_neur_x_val_lengths_mean),
                                         list(null_neur_x_val_lengths_std)]
    percentiles = []  # Calculate percentile of true data point in null data distribution
    for l_cut in len_x_vals:
        try:
            percentiles.extend([round(stats.percentileofscore(
                null_neur_x_val_lengths_all[l_cut-1], true_neur_x_val_lengths[l_cut-1]), 2)])
        except:
            percentiles.extend([100])
    neur_len_dict[seg_name +
                  '_percentile'] = [list(len_x_vals), percentiles]
    
    #FR data
    null_max_fr = np.max([np.max(null_dev_frs[null_i]) for null_i in range(num_null)])
    max_fr = np.max([null_max_fr,np.max(true_dev_frs)])
    fr_x_vals = np.arange(np.ceil(max_fr).astype('int'))
    true_fr_x_val_counts = np.nan*np.ones(len(fr_x_vals))
    null_fr_x_val_counts_all = []
    null_fr_x_val_counts_mean = np.nan*np.ones(len(fr_x_vals))
    null_fr_x_val_counts_std = np.nan*np.ones(len(fr_x_vals))
    for n_cut_i, n_cut in enumerate(fr_x_vals):
        true_fr_x_val_counts[n_cut_i] = np.sum(
            (np.array(true_dev_frs) > n_cut).astype('int'))
        null_fr_x_val_counts = []
        for null_i in range(num_null):
            null_fr_x_val_counts.append(
                np.sum((np.array(null_dev_frs[null_i]) > n_cut).astype('int')))
        null_fr_x_val_counts_all.append(null_fr_x_val_counts)
        null_fr_x_val_counts_mean[n_cut_i] = np.nanmean(
            null_fr_x_val_counts)
        null_fr_x_val_counts_std[n_cut_i] = np.nanstd(
            null_fr_x_val_counts)
        # Plot the individual distribution
        dpf.plot_dev_x_null_single_dist(
            null_fr_x_val_counts, true_fr_x_val_counts[n_cut_i], 'dev_fr_cutoff_' + str(n_cut), seg_fig_save_dir)
    neur_fr_dict[seg_name + '_true'] = [list(fr_x_vals),
                                           list(true_fr_x_val_counts)]
    neur_fr_dict[seg_name + '_null'] = [list(fr_x_vals),
                                           list(
                                               null_fr_x_val_counts_mean),
                                           list(null_fr_x_val_counts_std)]
    # Calculate percentiles
    percentiles = []  # Calculate percentile of true data point in null data distribution
    for n_cut in fr_x_vals:
        try:
            percentiles.extend([round(stats.percentileofscore(
                null_fr_x_val_counts_all[n_cut-1], true_fr_x_val_counts[n_cut-1]), 2)])
        except:
            percentiles.extend([100])
    neur_fr_dict[seg_name +
                    '_percentile'] = [list(fr_x_vals), percentiles]

    return neur_count_dict, neur_spike_dict, neur_fr_dict, neur_len_dict
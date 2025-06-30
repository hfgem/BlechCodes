#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:10:12 2023

@author: Hannah Germaine
Functions to plot deviation stats and the like
"""
import os
import tqdm
import warnings
import itertools
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import ks_2samp, kruskal, pearsonr
from random import sample
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn import svm


def plot_dev_rasters(segment_deviations, segment_spike_times, segment_dev_times,
                     segment_times_reshaped, pre_taste, post_taste, min_dev_size,
                     segment_names, dev_dir, max_plot=50):
    num_segments = len(segment_names)
    dev_buffer = 100  # ms before and after a deviation to plot
    half_min_dev_size = int(np.ceil(min_dev_size/2))
    dev_rates = np.zeros(num_segments)
    for s_i in range(num_segments):
        print("\t\tPlotting deviations in segment " + segment_names[s_i])
        filepath = dev_dir + segment_names[s_i] + '/'
        indiv_dev_filepath = filepath + 'indiv_dev/'
        if os.path.isdir(indiv_dev_filepath) == False:
            os.mkdir(indiv_dev_filepath)
            # Plot when deviations occur
            f = plt.figure(figsize=(5, 5))
            plt.plot(segment_deviations[s_i])
            plt.title('Segment ' + segment_names[s_i] + ' deviations')
            x_ticks = plt.xticks()[0]
            x_tick_labels = [np.round(x_ticks[i]/1000/60, 2)
                             for i in range(len(x_ticks))]
            plt.xticks(x_ticks, x_tick_labels)
            plt.xlabel('Time (min)')
            plt.yticks([0, 1], ['No Dev', 'Dev'])
            plt.tight_layout()
            fig_name = filepath + 'all_deviations'
            f.savefig(fig_name + '.png')
            f.savefig(fig_name + '.svg')
            plt.close(f)
            # Plot individual segments with pre and post time
            segment_rasters = segment_spike_times[s_i]
            segment_times = segment_dev_times[s_i] + \
                segment_times_reshaped[s_i][0]
            segment_length = segment_times_reshaped[s_i][1] - \
                segment_times_reshaped[s_i][0]
            num_neur = len(segment_rasters)
            num_deviations = len(segment_times[0, :])
            dev_rates[s_i] = num_deviations / \
                segment_length*(1000/1)  # Converted to Hz
            plot_dev_indices = sample(
                list(np.arange(num_deviations)), max_plot)
            for dev_i in tqdm.tqdm(plot_dev_indices):
                dev_times = segment_times[:, dev_i]
                dev_start = int(dev_times[0])
                dev_len = dev_times[1] - dev_start
                dev_rast_ind = []
                raster_len = np.ceil(2*dev_buffer + dev_len).astype('int')
                dev_binary = np.zeros((num_neur, raster_len))
                for n_i in range(num_neur):
                    segment_neur_rast = np.array(segment_rasters[n_i])
                    seg_dev_ind = np.where((segment_neur_rast > dev_start - dev_buffer)*(
                        segment_neur_rast < dev_times[1] + dev_buffer))[0]
                    seg_dev_rast_ind = segment_neur_rast[seg_dev_ind]
                    seg_dev_rast_ind_shift = (
                        seg_dev_rast_ind - dev_start + dev_buffer).astype('int')
                    dev_binary[n_i, seg_dev_rast_ind_shift] = 1
                    dev_rast_ind.append(seg_dev_rast_ind)
                # Create firing rates matrix
                firing_rate_vec = np.zeros(raster_len)
                for t_i in range(raster_len):
                    min_t_i = max(t_i-half_min_dev_size, 0)
                    max_t_i = min(t_i+half_min_dev_size, raster_len)
                    firing_rate_vec[t_i] = np.mean(
                        np.sum(dev_binary[:, min_t_i:max_t_i], 1)/(half_min_dev_size*2/1000))
                rate_x_tick_labels = np.arange(-1 *
                                               dev_buffer, dev_len + dev_buffer)
                # Now plot the rasters with firing rate deviations
                f1, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(
                    5, 5), gridspec_kw=dict(height_ratios=[2, 1]))
                # Deviation Raster Plot
                adjusted_dev_rast_ind = [
                    list(np.array(dev_rast_ind[n_i]) - dev_start) for n_i in range(num_neur)]
                ax1[0].eventplot(adjusted_dev_rast_ind, colors='b', alpha=0.5)
                ax1[0].axvline(0)
                ax1[0].axvline(dev_len)
                ax1[0].set_ylabel('Neuron Index')
                ax1[0].set_title('Deviation ' + str(dev_i))
                # Deviation population activity plot
                ax1[1].plot(rate_x_tick_labels, firing_rate_vec)
                ax1[1].axvline(0)
                ax1[1].axvline(dev_len)
                ax1[1].set_xlabel('Time (ms)')
                ax1[1].set_ylabel('Population rate (Hz)')
                fig_name = indiv_dev_filepath + 'dev_' + str(dev_i)
                f1.savefig(fig_name + '.png')
                f1.savefig(fig_name + '.svg')
                plt.close(f1)
    f = plt.figure()
    plt.scatter(np.arange(num_segments), dev_rates)
    plt.plot(np.arange(num_segments), dev_rates)
    plt.xticks(np.arange(num_segments), segment_names)
    for dr_i in range(num_segments):
        plt.annotate(str(round(dev_rates[dr_i], 2)), (dr_i, dev_rates[dr_i]))
    plt.title('Deviation Rates')
    plt.xlabel('Segment')
    plt.ylabel('Rate (Hz)')
    f.savefig(os.path.join(dev_dir, 'Deviation_Rates.png'))
    f.savefig(os.path.join(dev_dir, 'Deviation_Rates.svg'))
    plt.close(f)


def plot_dev_stats(data, data_name, save_dir, x_label=[], y_label=[]):
    """General function to plot given statistics"""
    plt.figure(figsize=(5, 5))
    # Plot the trend
    plt.subplot(2, 1, 1)
    plt.plot(data)
    if len(x_label) > 0:
        plt.xlabel(x_label)
    else:
        plt.xlabel('Deviation Index')
    if len(y_label) > 0:
        plt.ylabel(y_label)
    else:
        plt.ylabel(data_name)
    plt.title(data_name + ' trend')
    # Plot the histogram
    plt.subplot(2, 1, 2)
    plt.hist(data)
    if len(y_label) > 0:
        plt.xlabel(y_label)
    else:
        plt.xlabel(data_name)
    plt.ylabel('Number of Occurrences')
    plt.title(data_name + ' histogram')
    plt.tight_layout()
    im_name = ('_').join(data_name.split(' '))
    plt.savefig(save_dir + im_name + '.png')
    plt.savefig(save_dir + im_name + '.svg')
    plt.close()


def plot_dev_stats_dict(dict_data, iteration_names, data_name, save_dir, x_label, y_label):
    labels, data = [*zip(*dict_data.items())]
    # Calculate pairwise significant differences
    x_ticks = np.arange(1, len(labels)+1)
    pairs = list(itertools.combinations(x_ticks, 2))
    pair_sig = np.zeros(len(pairs))
    # cross-group stat sig
    args = [d for d in data]
    try:
        kw_stat, kw_p_val = kruskal(*args, nan_policy='omit')
    except:
        kw_p_val = 1
    # pairwise stat sig
    for pair_i in range(len(pairs)):
        pair = pairs[pair_i]
        ks_pval = ks_2samp(data[pair[0]-1], data[pair[1]-1])[1]
        if ks_pval < 0.05:
            pair_sig[pair_i] = 1

    # Plot distributions as box and whisker plots comparing across iterations
    bw_fig = plt.figure(figsize=(5, 5))
    plt.boxplot(data)
    plt.xticks(x_ticks, labels=iteration_names)
    y_ticks = plt.yticks()[0]
    y_max = np.max(y_ticks)
    y_range = y_max - np.min(y_ticks)
    x_mean = np.mean(x_ticks)
    if kw_p_val <= 0.05:
        plt.plot([x_ticks[0], x_ticks[-1]], [y_max+0.05 *
                 y_range, y_max+0.05*y_range], color='k')
        plt.scatter(x_mean, y_max+0.1*y_range, marker='*', color='k')
    jitter_vals = np.linspace(0.9*y_max, y_max, len(pairs))
    step = np.mean(np.diff(jitter_vals))
    for pair_i in range(len(pairs)):
        pair = pairs[pair_i]
        if pair_sig[pair_i] == 1:
            plt.plot([pair[0], pair[1]], [jitter_vals[pair_i],
                     jitter_vals[pair_i]], color='k', linestyle='dashed')
            plt.scatter((pair[0]+pair[1])/2,
                        jitter_vals[pair_i]+step, marker='*', color='k')
    plt.title(data_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    im_name = ('_').join(data_name.split(' '))
    plt.savefig(save_dir + im_name + '_box.png')
    plt.savefig(save_dir + im_name + '_box.svg')
    plt.close(bw_fig)
    # Plot distributions as violin plots
    violin_fig = plt.figure(figsize=(5, 5))
    plt.violinplot(data, positions=np.arange(1, len(labels)+1))
    plt.xticks(range(1, len(labels)+1), labels=iteration_names)
    y_ticks = plt.yticks()[0]
    y_max = np.max(y_ticks)
    y_range = y_max - np.min(y_ticks)
    x_mean = np.mean(x_ticks)
    if kw_p_val <= 0.05:
        plt.plot([x_ticks[0], x_ticks[-1]], [y_max+0.05 *
                 y_range, y_max+0.05*y_range], color='k')
        plt.scatter(x_mean, y_max+0.1*y_range, marker='*', color='k')
    jitter_vals = np.linspace(0.9*y_max, y_max, len(pairs))
    step = np.mean(np.diff(jitter_vals))
    for pair_i in range(len(pairs)):
        pair = pairs[pair_i]
        if pair_sig[pair_i] == 1:
            plt.plot([pair[0], pair[1]], [jitter_vals[pair_i],
                     jitter_vals[pair_i]], color='k', linestyle='dashed')
            plt.scatter((pair[0]+pair[1])/2,
                        jitter_vals[pair_i]+step, marker='*', color='k')
    plt.title(data_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    im_name = ('_').join(data_name.split(' '))
    plt.savefig(save_dir + im_name + '_violin.png')
    plt.savefig(save_dir + im_name + '_violin.svg')
    plt.close(violin_fig)
    # Plot distributions as PDFs
    pdf_fig = plt.figure(figsize=(3, 3))
    for d_i in range(len(data)):
        plt.hist(data[d_i], label=iteration_names[d_i],
                 density=True, histtype='step')
    plt.legend()
    plt.title(data_name)
    plt.xlabel(y_label)
    plt.ylabel('Probability')
    plt.tight_layout()
    im_name = ('_').join(data_name.split(' '))
    plt.savefig(save_dir + im_name + '_pdf.png')
    plt.savefig(save_dir + im_name + '_pdf.svg')
    plt.close(pdf_fig)
    # Plot distributions as CDFs
    cdf_fig = plt.figure(figsize=(3, 3))
    for d_i in range(len(data)):
        max_bin = max(np.max(data[d_i]), 1)
        plt.hist(data[d_i], bins=np.arange(0, max_bin), label=iteration_names[d_i],
                 density=True, cumulative=True, histtype='step')
    plt.legend()
    plt.title(data_name)
    plt.xlabel(y_label)
    plt.ylabel('Probability')
    plt.tight_layout()
    im_name = ('_').join(data_name.split(' '))
    plt.savefig(save_dir + im_name + '_cdf.png')
    plt.savefig(save_dir + im_name + '_cdf.svg')
    plt.close(cdf_fig)


def plot_null_v_true_stats(true_data, null_data, data_name, save_dir, x_label=[]):
    """General function to plot given null and true statistics
    true_data is given as a numpy array
    null_data is given as a dictionary with keys = null index and values = numpy arrays
    """
    plt.figure(figsize=(5, 5))
    null_vals = []
    null_x_vals = []
    for key in null_data.keys():
        null_vals.extend(list(null_data[key]))
        null_x_vals.extend([int(key)])
    mean_null_vals = np.mean(null_vals)
    std_null_vals = np.std(null_vals)
    # Plot the histograms
    plt.subplot(3, 1, 1)
    plt.hist(true_data, bins=25, color='b', alpha=0.4, label='True')
    plt.hist(null_vals, bins=25, color='g', alpha=0.4, label='Null')
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('Number of Occurrences')
    plt.title(data_name + ' histogram')
    # Plot the probability distribution functions
    plt.subplot(3, 1, 2)
    plt.hist(true_data, bins=25, density=True,
             histtype='step', color='b', label='True')
    plt.hist(null_vals, bins=25, density=True,
             histtype='step', color='g', label='Null')
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('PDF')
    plt.title(data_name + ' PDF')
    # Plot the cumulative distribution functions
    plt.subplot(3, 1, 3)
    true_sort = np.sort(true_data)
    true_unique = np.unique(true_sort)
    cmf_true = np.array([np.sum((true_sort <= u_val).astype(
        'int'))/len(true_sort) for u_val in true_unique])
    null_sort = np.sort(null_vals)
    null_unique = np.unique(null_sort)
    cmf_null = np.array([np.sum((null_sort <= u_val).astype(
        'int'))/len(null_sort) for u_val in null_unique])
    plt.plot(true_unique, cmf_true, color='b', label='True')
    plt.plot(null_unique, cmf_null, color='g', label='Null')
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('CDF')
    plt.title(data_name + ' CDF')
    plt.tight_layout()
    im_name = ('_').join(data_name.split(' '))
    plt.savefig(save_dir + im_name + '_truexnull.png')
    plt.savefig(save_dir + im_name + '_truexnull.svg')
    plt.close()


def plot_stats(dev_stats, segment_names, dig_in_names, save_dir, dist_name,
               neuron_indices, segments_to_analyze):
    """This function takes in deviation correlations and plots the distributions.
    Outputs are saved as .png and .svg files.

    neuron_indices should be binary and shaped num_neur x num_cp
    """
    warnings.filterwarnings('ignore')
    # Grab parameters
    num_tastes = len(dig_in_names)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(len(segment_names))
    for s_i in segments_to_analyze:  # Loop through each segment
        print("\t\tBeginning plot calcs for segment " + str(s_i))
        seg_name = segment_names[s_i]
        seg_stats = dev_stats[seg_name]
        for t_i in range(num_tastes):  # Loop through each taste
            print("\t\t\tTaste #" + str(t_i + 1))
            taste_stats = seg_stats[t_i]
            # _____Population Vector CP Calculations_____
            # Import correlation numpy array
            pop_vec_data_storage = taste_stats['pop_vec_data_storage']
            data_shape = np.shape(pop_vec_data_storage)
            if len(data_shape) == 3:
                num_cp = data_shape[2]
            # Plot the individual neuron distribution for each changepoint index
            f = plt.figure(figsize=(5, 5))
            cp_data = []
            plt.subplot(2, 1, 1)
            for c_p in range(num_cp):
                all_dist_cp = (pop_vec_data_storage[:, :, c_p]).flatten()
                all_dist_cp = all_dist_cp[~np.isnan(all_dist_cp)]
                cp_data.append(all_dist_cp)
                plt.hist(all_dist_cp, density=True, cumulative=False,
                         histtype='step', label='Epoch ' + str(c_p))
            plt.xlabel(dist_name)
            plt.legend()
            plt.title('Probability Mass Function - ' + dist_name)
            plt.subplot(2, 1, 2)
            for c_p in range(num_cp):
                all_dist_cp = cp_data[c_p]
                all_dist_cp = all_dist_cp[~np.isnan(all_dist_cp)]
                plt.hist(all_dist_cp, bins=1000, density=True,
                         cumulative=True, histtype='step', label='Epoch ' + str(c_p))
            plt.xlabel(dist_name)
            plt.legend()
            plt.title('Cumulative Mass Function - ' + dist_name)
            plt.suptitle(dist_name + ' distributions for \nsegment ' +
                         segment_names[s_i] + ', taste ' + dig_in_names[t_i])
            plt.tight_layout()
            filename = save_dir + \
                segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec'
            f.savefig(filename + '.png')
            f.savefig(filename + '.svg')
            plt.close(f)
            if dist_name == 'Correlation':
                # Plot the individual neuron distribution for each changepoint index
                f = plt.figure(figsize=(5, 5))
                cp_data = []
                plt.subplot(2, 1, 1)
                min_y = 1
                min_x = 1
                max_x = 0
                for c_p in range(num_cp):
                    all_dist_cp = (pop_vec_data_storage[:, :, c_p]).flatten()
                    all_dist_cp = all_dist_cp[~np.isnan(all_dist_cp)]
                    cp_data.append(all_dist_cp)
                    hist_vals = plt.hist(
                        all_dist_cp, density=True, cumulative=False, histtype='step', label='Epoch ' + str(c_p))
                    max_x_val = np.max(np.abs(hist_vals[1]))
                    half_max_x = np.floor(max_x_val/2)
                    bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                    bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                    if bin_min_y < min_y:
                        min_y = bin_min_y
                    if half_max_x < min_x:
                        min_x = half_max_x
                    if max_x_val > max_x:
                        max_x = max_x_val
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Probability Mass Function - ' + dist_name)
                plt.subplot(2, 1, 2)
                min_y = 1
                min_x = 1
                max_x = 0
                for c_p in range(num_cp):
                    all_dist_cp = cp_data[c_p]
                    all_dist_cp = all_dist_cp[~np.isnan(all_dist_cp)]
                    hist_vals = plt.hist(all_dist_cp, bins=1000, density=True,
                                         cumulative=True, histtype='step', label='Epoch ' + str(c_p))
                    max_x_val = np.max(np.abs(hist_vals[1]))
                    half_max_x = np.floor(max_x_val/2)
                    bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                    bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                    if bin_min_y < min_y:
                        min_y = bin_min_y
                    if half_max_x < min_x:
                        min_x = half_max_x
                    if max_x_val > max_x:
                        max_x = max_x_val
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Cumulative Mass Function - ' + dist_name)
                plt.suptitle(dist_name + ' distributions for \nsegment ' +
                             segment_names[s_i] + ', taste ' + dig_in_names[t_i])
                plt.tight_layout()
                filename = save_dir + \
                    segment_names[s_i] + '_' + \
                    dig_in_names[t_i] + '_pop_vec_zoom'
                f.savefig(filename + '.png')
                f.savefig(filename + '.svg')
                plt.close(f)
                # Plot the individual neuron distribution for each changepoint index
                f = plt.figure(figsize=(5, 5))
                cp_data = []
                plt.subplot(2, 1, 1)
                for c_p in range(num_cp):
                    all_dist_cp = (pop_vec_data_storage[:, :, c_p]).flatten()
                    all_dist_cp = all_dist_cp[~np.isnan(all_dist_cp)]
                    cp_data.append(all_dist_cp)
                    plt.hist(all_dist_cp, density=True, log=True, cumulative=False,
                             histtype='step', label='Epoch ' + str(c_p))
                plt.xlabel(dist_name)
                plt.xlim([0.5, 1])
                plt.legend()
                plt.title('Probability Mass Function - ' + dist_name)
                plt.subplot(2, 1, 2)
                min_y = 1
                min_x = 1
                max_x = 0
                for c_p in range(num_cp):
                    all_dist_cp = cp_data[c_p]
                    all_dist_cp = all_dist_cp[~np.isnan(all_dist_cp)]
                    hist_vals = plt.hist(all_dist_cp, bins=1000, density=True, log=True,
                                         cumulative=True, histtype='step', label='Epoch ' + str(c_p))
                    max_x_val = np.max(np.abs(hist_vals[1]))
                    half_max_x = np.floor(max_x_val/2)
                    bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                    bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                    if bin_min_y < min_y:
                        min_y = bin_min_y
                    if half_max_x < min_x:
                        min_x = half_max_x
                    if max_x_val > max_x:
                        max_x = max_x_val
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Cumulative Mass Function - ' + dist_name)
                plt.suptitle(dist_name + ' distributions for \nsegment ' +
                             segment_names[s_i] + ', taste ' + dig_in_names[t_i])
                plt.tight_layout()
                filename = save_dir + \
                    segment_names[s_i] + '_' + \
                    dig_in_names[t_i] + '_pop_vec_log_zoom'
                f.savefig(filename + '.png')
                f.savefig(filename + '.svg')
                plt.close(f)


def plot_combined_stats(dev_stats, segment_names, dig_in_names, save_dir,
                        dist_name, neuron_indices, segments_to_analyze=[]):
    """This function takes in deviation rasters, tastant delivery spikes, and
    changepoint indices to calculate correlations of each deviation to each 
    changepoint interval. Outputs are saved .npy files with name indicating
    segment and taste containing matrices of shape [num_dev, num_deliv, num_neur, num_cp]
    with the distances stored.

    neuron_indices should be binary and shaped num_neur x num_cp
    """
    warnings.filterwarnings('ignore')
    # Grab parameters
    num_tastes = len(dig_in_names)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(segment_names)
    num_segments = len(segments_to_analyze)
    segment_names = np.array(segment_names)[segments_to_analyze]

    # Define storage
    segment_pop_vec_data = []  # segments x tastes x cp from fr population vector
    for s_i, s_ind in enumerate(segments_to_analyze):  # Loop through each segment
        seg_name = segment_names[s_i]
        seg_stats = dev_stats[seg_name]
        print("\t\tBeginning combined plot calcs for segment " + seg_name)
        taste_pop_vec_data = []
        for t_i in range(num_tastes):  # Loop through each taste
            print("\t\t\tTaste #" + str(t_i + 1))
            taste_stats = seg_stats[t_i]
            # Import distance numpy array
            pop_vec_data_storage = taste_stats['pop_vec_data_storage']
            num_dev, num_deliv, num_cp = np.shape(pop_vec_data_storage)
            cp_data_pop_vec = []
            for c_p in range(num_cp):
                all_pop_vec_cp = (pop_vec_data_storage[:, :, c_p]).flatten()
                cp_data_pop_vec.append(all_pop_vec_cp)
            taste_pop_vec_data.append(cp_data_pop_vec)
        segment_pop_vec_data.append(taste_pop_vec_data)
        # Plot taste data against each other
        for c_p in range(num_cp):
            # _____Population Vector Data
            # Plot data across all neurons
            f0 = plt.figure(figsize=(5, 5))
            plt.subplot(2, 1, 1)
            for t_i in range(num_tastes):
                try:
                    data = taste_pop_vec_data[t_i][c_p]
                    plt.hist(data[~np.isnan(data)], density=True, cumulative=False,
                             histtype='step', label='Taste ' + dig_in_names[t_i])
                except:
                    pass
            plt.xlabel(dist_name)
            plt.legend()
            plt.title('Probability Mass Function - ' + dist_name)
            plt.subplot(2, 1, 2)
            for t_i in range(num_tastes):
                try:
                    data = taste_pop_vec_data[t_i][c_p]
                    plt.hist(data[~np.isnan(data)], bins=1000, density=True,
                             cumulative=True, histtype='step', label='Taste ' + dig_in_names[t_i])
                except:
                    pass
            plt.xlabel(dist_name)
            plt.legend()
            plt.title('Cumulative Mass Function - ' + dist_name)
            plt.suptitle('Population Distributions for \n' +
                         segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
            plt.tight_layout()
            filename = save_dir + \
                segment_names[s_i] + '_epoch' + str(c_p) + '_pop_vec'
            f0.savefig(filename + '.png')
            f0.savefig(filename + '.svg')
            plt.close(f0)
            # Zoom to correlations > 0.5
            if dist_name == 'Correlation':
                # Plot data across all neurons
                f1 = plt.figure(figsize=(5, 5))
                plt.subplot(2, 1, 1)
                min_y = 1
                min_x = 1
                max_x = 0
                for t_i in range(num_tastes):
                    try:
                        data = taste_pop_vec_data[t_i][c_p]
                        hist_vals = plt.hist(data[~np.isnan(
                            data)], density=True, cumulative=False, histtype='step', label='Taste ' + dig_in_names[t_i])
                        max_x_val = np.max(np.abs(hist_vals[1]))
                        half_max_x = np.floor(max_x_val/2)
                        bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                        bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                        if bin_min_y < min_y:
                            min_y = bin_min_y
                        if half_max_x < min_x:
                            min_x = half_max_x
                        if max_x_val > max_x:
                            max_x = max_x_val
                    except:
                        pass
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Probability Mass Function - ' + dist_name)
                plt.subplot(2, 1, 2)
                min_y = 1
                min_x = 1
                max_x = 0
                for t_i in range(num_tastes):
                    try:
                        data = taste_pop_vec_data[t_i][c_p]
                        hist_vals = plt.hist(data[~np.isnan(
                            data)], bins=1000, density=True, cumulative=True, histtype='step', label='Taste ' + dig_in_names[t_i])
                        max_x_val = np.max(np.abs(hist_vals[1]))
                        half_max_x = np.floor(max_x_val/2)
                        bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                        bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                        if bin_min_y < min_y:
                            min_y = bin_min_y
                        if half_max_x < min_x:
                            min_x = half_max_x
                        if max_x_val > max_x:
                            max_x = max_x_val
                    except:
                        pass
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Cumulative Mass Function - ' + dist_name)
                plt.suptitle('Population Distributions for \n' +
                             segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
                plt.tight_layout()
                filename = save_dir + \
                    segment_names[s_i] + '_epoch' + str(c_p) + '_zoom_pop_vec'
                f1.savefig(filename + '.png')
                f1.savefig(filename + '.svg')
                plt.close(f1)
                # Plot data across all neurons with log
                f1 = plt.figure(figsize=(5, 5))
                plt.subplot(2, 1, 1)
                min_y = 1
                min_x = 1
                max_x = 0
                for t_i in range(num_tastes):
                    try:
                        data = taste_pop_vec_data[t_i][c_p]
                        hist_vals = plt.hist(data[~np.isnan(
                            data)], density=True, log=True, cumulative=False, histtype='step', label='Taste ' + dig_in_names[t_i])
                        max_x_val = np.max(np.abs(hist_vals[1]))
                        half_max_x = np.floor(max_x_val/2)
                        bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                        bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                        if bin_min_y < min_y:
                            min_y = bin_min_y
                        if half_max_x < min_x:
                            min_x = half_max_x
                        if max_x_val > max_x:
                            max_x = max_x_val
                    except:
                        pass
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Probability Mass Function - ' + dist_name)
                plt.subplot(2, 1, 2)
                min_y = 1
                min_x = 1
                max_x = 0
                for t_i in range(num_tastes):
                    try:
                        data = taste_pop_vec_data[t_i][c_p]
                        hist_vals = plt.hist(data[~np.isnan(data)], bins=1000, density=True, log=True,
                                             cumulative=True, histtype='step', label='Taste ' + dig_in_names[t_i])
                        max_x_val = np.max(np.abs(hist_vals[1]))
                        half_max_x = np.floor(max_x_val/2)
                        bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                        bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                        if bin_min_y < min_y:
                            min_y = bin_min_y
                        if half_max_x < min_x:
                            min_x = half_max_x
                        if max_x_val > max_x:
                            max_x = max_x_val
                    except:
                        pass
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Cumulative Mass Function - ' + dist_name)
                plt.suptitle('Population Distributions for \n' +
                             segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
                plt.tight_layout()
                filename = save_dir + \
                    segment_names[s_i] + '_epoch' + \
                    str(c_p) + '_log_zoom_pop_vec'
                f1.savefig(filename + '.png')
                f1.savefig(filename + '.svg')
                plt.close(f1)

    for t_i in range(num_tastes):  # Loop through each taste
        # _____Population Vec Data_____
        for c_p in range(num_cp):
            f2 = plt.figure(figsize=(5, 5))
            plt.subplot(2, 1, 1)
            for s_i in range(num_segments):
                try:
                    data = segment_pop_vec_data[s_i][t_i][c_p]
                    plt.hist(data[~np.isnan(data)], density=True, log=True, cumulative=False,
                             histtype='step', label='Segment ' + segment_names[s_i])
                except:
                    pass
            plt.xlabel(dist_name)
            plt.legend()
            plt.title('Probability Mass Function - ' + dist_name)
            plt.subplot(2, 1, 2)
            for s_i in range(num_segments):
                try:
                    data = segment_pop_vec_data[s_i][t_i][c_p]
                    plt.hist(data[~np.isnan(data)], bins=1000, density=True, log=True,
                             cumulative=True, histtype='step', label='Segment ' + segment_names[s_i])
                except:
                    pass
            plt.xlabel(dist_name)
            plt.legend()
            plt.title('Cumulative Mass Function - ' + dist_name)
            plt.suptitle('Population Avg Distributions for \n' +
                         segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
            plt.tight_layout()
            filename = save_dir + \
                dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_vec'
            f2.savefig(filename + '.png')
            f2.savefig(filename + '.svg')
            plt.close(f2)
            # Zoom to correlations > 0.5
            if dist_name == 'Correlation':
                f3 = plt.figure(figsize=(5, 5))
                plt.subplot(2, 1, 1)
                min_y = 1
                min_x = 1
                max_x = 0
                for s_i in range(num_segments):
                    try:
                        data = segment_pop_vec_data[s_i][t_i][c_p]
                        hist_vals = plt.hist(data[~np.isnan(
                            data)], density=True, cumulative=False, histtype='step', label='Segment ' + segment_names[s_i])
                        max_x_val = np.max(np.abs(hist_vals[1]))
                        half_max_x = np.floor(max_x_val/2)
                        bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                        bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                        if bin_min_y < min_y:
                            min_y = bin_min_y
                        if half_max_x < min_x:
                            min_x = half_max_x
                        if max_x_val > max_x:
                            max_x = max_x_val
                    except:
                        pass
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Probability Mass Function - ' + dist_name)
                plt.subplot(2, 1, 2)
                min_y = 1
                min_x = 1
                max_x = 0
                for s_i in range(num_segments):
                    try:
                        data = segment_pop_vec_data[s_i][t_i][c_p]
                        hist_vals = plt.hist(data[~np.isnan(
                            data)], bins=1000, density=True, cumulative=True, histtype='step', label='Segment ' + segment_names[s_i])
                        max_x_val = np.max(np.abs(hist_vals[1]))
                        half_max_x = np.floor(max_x_val/2)
                        bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                        bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                        if bin_min_y < min_y:
                            min_y = bin_min_y
                        if half_max_x < min_x:
                            min_x = half_max_x
                        if max_x_val > max_x:
                            max_x = max_x_val
                    except:
                        pass
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Cumulative Mass Function - ' + dist_name)
                plt.suptitle('Population Avg Distributions for \n' +
                             segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
                plt.tight_layout()
                filename = save_dir + \
                    dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_vec_zoom'
                f3.savefig(filename + '.png')
                f3.savefig(filename + '.svg')
                plt.close(f3)
                # Log
                f3 = plt.figure(figsize=(5, 5))
                plt.subplot(2, 1, 1)
                min_y = 1
                min_x = 1
                max_x = 0
                for s_i in range(num_segments):
                    try:
                        data = segment_pop_vec_data[s_i][t_i][c_p]
                        hist_vals = plt.hist(data[~np.isnan(
                            data)], density=True, log=True, cumulative=False, histtype='step', label='Segment ' + segment_names[s_i])
                        max_x_val = np.max(np.abs(hist_vals[1]))
                        half_max_x = np.floor(max_x_val/2)
                        bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                        bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                        if bin_min_y < min_y:
                            min_y = bin_min_y
                        if half_max_x < min_x:
                            min_x = half_max_x
                        if max_x_val > max_x:
                            max_x = max_x_val
                    except:
                        pass
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Probability Mass Function - ' + dist_name)
                plt.subplot(2, 1, 2)
                min_y = 1
                min_x = 1
                max_x = 0
                for s_i in range(num_segments):
                    try:
                        data = segment_pop_vec_data[s_i][t_i][c_p]
                        hist_vals = plt.hist(data[~np.isnan(data)], bins=1000, density=True, log=True,
                                             cumulative=True, histtype='step', label='Segment ' + segment_names[s_i])
                        max_x_val = np.max(np.abs(hist_vals[1]))
                        half_max_x = np.floor(max_x_val/2)
                        bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
                        bin_min_y = hist_vals[0][np.max(bin_05-1, 0)]
                        if bin_min_y < min_y:
                            min_y = bin_min_y
                        if half_max_x < min_x:
                            min_x = half_max_x
                        if max_x_val > max_x:
                            max_x = max_x_val
                    except:
                        pass
                if min_y == 1:
                    min_y = 0.5
                if max_x_val < 0.5:
                    plt.xlim([min_x, max_x])
                else:
                    plt.xlim([0.5, 1])
                plt.xlabel(dist_name)
                plt.ylim([min_y, 1])
                plt.legend()
                plt.title('Cumulative Mass Function - ' + dist_name)
                plt.suptitle('Population Avg Distributions for \n' +
                             segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
                plt.tight_layout()
                filename = save_dir + \
                    dig_in_names[t_i] + '_epoch' + \
                    str(c_p) + '_pop_vec_log_zoom'
                f3.savefig(filename + '.png')
                f3.savefig(filename + '.svg')
                plt.close(f3)

    return segment_pop_vec_data


def plot_dev_x_null_single_dist(null_data, true_val, plot_name, save_dir):
    """This function plots the individual null distribution with a vertical
    line of the true data point"""

    f = plt.figure(figsize=(5, 5))
    plt.hist(null_data, color='k', alpha=0.5, label='Null Data')
    plt.axvline(true_val, color='r', label='True Count')
    f.savefig(os.path.join(save_dir, plot_name + '.png'))
    f.savefig(os.path.join(save_dir, plot_name + '.svg'))
    plt.close(f)

def best_corr_calc_plot(dig_in_names, epochs_to_analyze, segments_to_analyze,
                        segment_names, segment_times_reshaped, segment_dev_times,
                        dev_dir, min_dev_size, segment_spike_times, corr_data_dir,
                        cp, tastant_spike_times, start_dig_in_times, end_dig_in_times,
                        pre_taste, post_taste, num_neur, save_dir, no_indiv_plot = False):
    """This function calculates which deviation is correlated most to which
    condition and plots"""

    num_tastes = len(dig_in_names)
    pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
    post_taste_dt = np.ceil(post_taste*1000).astype('int')
    half_min_dev_size = int(np.ceil(min_dev_size/2))
    dev_buffer = 100 #ms before and after to plot
    num_deliv_plot = 2 #Number of example taste delivery responses to plot
    
    if no_indiv_plot == False:
        # Pull taste delivery fr vecs and rasters
        all_taste_deliv_fr_vec = []
        all_taste_deliv_rasters = dict()
        for t_i in range(num_tastes):
            taste_cp_pop = cp[t_i]
            taste_spikes = tastant_spike_times[t_i]
            # Note, num_cp = num epochs + 1 with the first value the taste delivery index
            num_deliv, num_cp = np.shape(taste_cp_pop)
            #Set up dictionary of rasters
            all_taste_deliv_rasters[t_i] = dict()
            for cp_i in range(num_cp):
                all_taste_deliv_rasters[t_i][cp_i] = []
            taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i]
                                [deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
            deliv_adjustment = [start_dig_in_times[t_i][deliv_i] +
                                pre_taste_dt for deliv_i in range(num_deliv)]
            # Calculate delivery firing rate vectors
            taste_deliv_fr_vec = np.zeros((num_deliv, num_cp-1, num_neur))
            for deliv_i in range(num_deliv):
                # Pull delivery raster
                deliv_rast = np.zeros((num_neur, taste_deliv_len[deliv_i]))
                for n_i in range(num_neur):
                    n_st = taste_spikes[deliv_i][n_i]
                    if len(n_st) >= 1:
                        if len(n_st) > 1:
                            neur_deliv_st = list(np.array(n_st).astype(
                                'int') - deliv_adjustment[deliv_i])
                        else:
                            neur_deliv_st = int(
                                n_st[0]) - deliv_adjustment[deliv_i]
                        deliv_rast[n_i, neur_deliv_st] = 1
                deliv_fr_vec = np.zeros((num_cp-1, num_neur))
                for cp_i in range(num_cp-1):
                    cp_vals = (taste_cp_pop[deliv_i, cp_i:cp_i+2]).astype('int')
                    epoch_len = cp_vals[1] - cp_vals[0]
                    #Pull out cp raster
                    all_taste_deliv_rasters[t_i][cp_i].append(deliv_rast[:, cp_vals[0]:cp_vals[1]])
                    # Pull out the delivery cp fr vector
                    deliv_vec = np.sum(
                        deliv_rast[:, cp_vals[0]:cp_vals[1]], 1)/(epoch_len/1000)  # in Hz
                    deliv_fr_vec[cp_i, :] = deliv_vec
                taste_deliv_fr_vec[deliv_i, :, :] = deliv_fr_vec
            del taste_cp_pop, taste_spikes, num_deliv, num_cp, taste_deliv_len
            del deliv_adjustment, deliv_i, deliv_rast, n_i, n_st, neur_deliv_st
            del deliv_fr_vec, cp_i, cp_vals, epoch_len, deliv_vec
            all_taste_deliv_fr_vec.append(taste_deliv_fr_vec)

    best_corr_labels = []
    best_corr_tastes = []
    best_corr_epochs = []
    for t_i in range(num_tastes):
        for e_ind, e_i in enumerate(epochs_to_analyze):
            best_corr_labels.append(dig_in_names[t_i] + '_' + str(e_i))
            best_corr_tastes.extend([t_i])
            best_corr_epochs.extend([e_ind])

    #label_colors = cm.plasma(np.linspace(0, 1, len(best_corr_labels)))
    taste_colors = cm.jet(np.linspace(0, 1, num_tastes))
    epoch_colors = cm.cool(np.linspace(0, 1, len(epochs_to_analyze)))
    segment_colors = cm.cividis(np.linspace(0, 1, len(segments_to_analyze)))

    seg_data = dict()

    # Calculate which deviation event is correlated to which taste/epoch most
    f_frac, ax_frac = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    for s_ind, s_i in enumerate(segments_to_analyze):  # By rest interval
        # Store deviation events by sorted categories
        dev_label_ind = []
        dev_taste_epoch_label = []
        dev_sorting = dict()
        for l_ind, l_name in enumerate(best_corr_labels):
            dev_sorting[l_name] = dict()
            dev_sorting[l_name]['taste'] = best_corr_tastes[l_ind]
            dev_sorting[l_name]['epoch'] = best_corr_epochs[l_ind]
            dev_sorting[l_name]['num_dev'] = 0
            dev_sorting[l_name]['start_times'] = []
            dev_sorting[l_name]['end_times'] = []
            dev_sorting[l_name]['corr_vals'] = []

        # Get deviation data
        seg_dev_times = segment_dev_times[s_ind]
        start_dev_bouts = seg_dev_times[0, :].flatten()
        end_dev_bouts = seg_dev_times[1, :].flatten()
        num_dev = len(start_dev_bouts)

        # Import deviation correlation data
        num_trial_per_taste = []
        all_taste_corr = []  # num tastes x num dev x num trials x num cp
        mean_taste_corr = []  # num tastes x num dev x num cp
        for t_i in range(num_tastes):
            t_data = np.load(os.path.join(
                corr_data_dir, segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec.npy'))
            [_, num_trials_i, _] = np.shape(t_data)
            num_trial_per_taste.append([num_trials_i])
            all_taste_corr.append(t_data)
            mean_taste_corr.append(np.nanmean(t_data, 1))

        # Sort the deviation events by best correlation
        num_plot = 25
        for d_i in range(num_dev):
            corr_vals = []
            for t_i in range(num_tastes):
                for e_ind, e_i in enumerate(epochs_to_analyze):
                    corr_vals.append([mean_taste_corr[t_i][d_i][e_i]])
            best_ind = np.argmax(corr_vals)
            dev_label_ind.extend([best_ind])
            l_name = best_corr_labels[best_ind]
            taste_ind = best_corr_tastes[best_ind]
            epoch_ind = best_corr_epochs[best_ind]
            dev_taste_epoch_label.append([taste_ind,epoch_ind])
            #Save to dictionary
            dev_sorting[l_name]['num_dev'] += 1
            dev_sorting[l_name]['start_times'].extend([start_dev_bouts[d_i]])
            dev_sorting[l_name]['end_times'].extend([end_dev_bouts[d_i]])
            dev_sorting[l_name]['corr_vals'].extend([corr_vals[best_ind]])
            if no_indiv_plot == False:
                if num_plot < 50:
                    if l_name == 'saccharin_1':
                        #Pull the segment spike times for the deviation event + buffer
                        dev_times = segment_dev_times[s_ind][:,d_i]
                        seg_start = segment_times_reshaped[s_ind,0]
                        dev_start = int(dev_times[0])
                        dev_len = dev_times[1] - dev_start
                        dev_rast_ind = []
                        raster_len = np.ceil(2*dev_buffer + dev_len + 1).astype('int')
                        dev_binary = np.zeros((num_neur, raster_len))
                        dev_rast_times = []
                        for n_i in range(num_neur):
                            neur_spike_times = np.array(segment_spike_times[s_i][n_i]) - seg_start
                            seg_spike_ind = np.where((neur_spike_times >= (dev_start - dev_buffer))*\
                                                     (neur_spike_times <= (dev_start + dev_len + dev_buffer)))[0]
                            if len(seg_spike_ind) > 0:
                                seg_spike_times = np.sort(neur_spike_times[seg_spike_ind])
                                reshape_spike_times = (seg_spike_times - (dev_start - dev_buffer)).astype('int')
                                dev_rast_times.append(list(reshape_spike_times))
                                dev_binary[n_i,reshape_spike_times] = 1
                            else:
                                dev_rast_times.append([])
                        del neur_spike_times, seg_spike_ind
                        #Create firing rates vector
                        firing_rate_vec = np.zeros(raster_len)
                        for rt_i in range(raster_len):
                            min_t_i = max(rt_i-half_min_dev_size, 0)
                            max_t_i = min(rt_i+half_min_dev_size, raster_len)
                            firing_rate_vec[rt_i] = np.mean(
                                np.sum(dev_binary[:, min_t_i:max_t_i], 1)/(half_min_dev_size*2/1000))
                        rate_x_tick_labels = np.arange(dev_len + 2*dev_buffer + 1)
                        #Create a plot of the deviation event and example taste event raster
                        #As well as the firing rate vectors for both
                        num_taste_deliv, _, _ = np.shape(all_taste_deliv_fr_vec[taste_ind])
                        plot_true_indices = sample(
                            list(np.arange(num_taste_deliv)), num_deliv_plot)
                        dev_fig, dev_ax = plt.subplots(nrows=num_deliv_plot+2,ncols=2,figsize=(10,(num_deliv_plot+2)*4))
                        #Collect the deviation and taste rasters
                        _, dev_rast_len = np.shape(dev_binary)
                        dev_fr_vec = np.sum(dev_binary,1)/(np.shape(dev_binary)[1]/1000)
                        all_taste_rast_len = []
                        for ax_ind, pti in enumerate(plot_true_indices):
                            taste_cp_rast = all_taste_deliv_rasters[taste_ind][epoch_ind][pti]
                            all_taste_rast_len.extend([np.shape(taste_cp_rast[1])])
                        #Calculate the longest length raster to make the raster eventplots equal size
                        max_len = np.max([np.max(all_taste_rast_len),dev_rast_len])
                        #Plot the taste and epoch rasters
                        dev_ax[0,0].eventplot(dev_rast_times)
                        dev_ax[0,0].axvline(dev_buffer)
                        dev_ax[0,0].axvline(dev_buffer+dev_len)
                        dev_ax[0,0].axvline(max_len,color='k')
                        dev_ax[0,0].set_title('Deviation Raster')
                        #Plot the population rate for the deviation event
                        dev_ax[0,1].plot(rate_x_tick_labels,firing_rate_vec)
                        dev_ax[0,1].axvline(dev_buffer)
                        dev_ax[0,1].axvline(dev_buffer+dev_len)
                        dev_ax[0,1].axvline(max_len,color='k')
                        #Plot the taste responses
                        for ax_ind, pti in enumerate(plot_true_indices):
                            #Pull raster for example taste delivery
                            taste_cp_rast = all_taste_deliv_rasters[taste_ind][epoch_ind][pti]
                            taste_rast_times = []
                            for n_i in range(num_neur):
                                taste_rast_times.append(list(np.where(taste_cp_rast[n_i,:] == 1)[0]))
                            #Calculate firing rate vector for delivery
                            taste_fr_vec = np.sum(taste_cp_rast,1)/(all_taste_rast_len[ax_ind][0]/1000)
                            #Plot raster and vec
                            dev_ax[ax_ind+1,0].eventplot(taste_rast_times)
                            dev_ax[ax_ind+1,0].axvline(max_len,color='k')
                            dev_ax[ax_ind+1,0].set_title('Ex Taste Raster ' + str(pti))
                            dev_ax[ax_ind+1,1].imshow(np.expand_dims(taste_fr_vec,0),cmap='viridis')
                            dev_ax[ax_ind+1,1].set_title('Taste Response Firing Rate Vector')
                        #Plot the deviation firing rate vec as an imshow
                        dev_ax[num_deliv_plot+1,0].imshow(np.expand_dims(dev_fr_vec,0),cmap='viridis')
                        dev_ax[num_deliv_plot+1,0].set_title('Deviation Firing Rate Vector')
                        #Plot the average taste firing rate vec as an imshow
                        all_taste_fr_vec = all_taste_deliv_fr_vec[taste_ind][:,epoch_ind,:].squeeze()
                        dev_ax[num_deliv_plot+1,1].imshow(np.expand_dims(np.nanmean(all_taste_fr_vec,0),0),cmap='viridis')
                        dev_ax[num_deliv_plot+1,1].set_title('Average Taste Response Firing Rate Vector')
                        #Add suptitle and clean up to save
                        dev_fig.suptitle('Segment ' + str(s_i) + '\nDev ' + str(d_i) + '\nBest Fit ' + l_name)
                        plt.tight_layout()
                        dev_fig.savefig(os.path.join(save_dir, 'seg_' + str(s_i) + '_dev_'+str(d_i)+'_'+l_name+'.png'))
                        dev_fig.savefig(os.path.join(save_dir, 'seg_' + str(s_i) + '_dev_'+str(d_i)+'_'+l_name+'.svg'))
                        plt.close(dev_fig)
                        num_plot += 1
        #Save label indices for later easy use
        dev_taste_epoch_label_array = np.array(dev_taste_epoch_label)
        np.save(os.path.join(corr_data_dir,segment_names[s_i] + '_best_taste_epoch_array.npy'),dev_taste_epoch_label_array)
        
        # Calculate the fractions
        count_per_label = np.zeros(len(best_corr_labels))
        count_per_taste = np.zeros(num_tastes)
        count_per_epoch = np.zeros(len(epochs_to_analyze))
        for l_ind, l_name in enumerate(best_corr_labels):
            count_per_label[l_ind] = dev_sorting[l_name]['num_dev']
            taste_ind = dev_sorting[l_name]['taste']
            epoch_ind = dev_sorting[l_name]['epoch']
            count_per_taste[taste_ind] += dev_sorting[l_name]['num_dev']
            count_per_epoch[epoch_ind] += dev_sorting[l_name]['num_dev']
        fraction_per_label = count_per_label/np.sum(count_per_label)
        fraction_per_taste = count_per_taste/np.sum(count_per_taste)
        fraction_per_epoch = count_per_epoch/np.sum(count_per_epoch)

        ax_frac[0].plot(np.arange(len(best_corr_labels)),
                        fraction_per_label*100, label=segment_names[s_i])
        ax_frac[1].plot(np.arange(num_tastes),
                        fraction_per_taste*100, label=segment_names[s_i])
        ax_frac[2].plot(np.arange(len(epochs_to_analyze)),
                        fraction_per_epoch*100, label=segment_names[s_i])

        seg_data[s_i] = dev_sorting
    
    ax_frac[0].set_ylabel('Percent of Deviation Events')
    ax_frac[0].set_xticks(np.arange(len(best_corr_labels)),
                          best_corr_labels, rotation=45)
    ax_frac[0].legend(loc='upper right')
    ax_frac[1].set_xticks(np.arange(num_tastes), dig_in_names)
    ax_frac[1].set_ylabel('Percent of Deviation Events')
    ax_frac[2].set_xticks(np.arange(len(epochs_to_analyze)), [
                          'Epoch ' + str(e_i) for e_i in epochs_to_analyze])
    ax_frac[2].set_ylabel('Percent of Deviation Events')
    f_frac.suptitle('Deviation Correlation Percents')
    plt.tight_layout()
    # Save figure
    f_frac.savefig(os.path.join(save_dir, 'best_corr_percents.png'))
    f_frac.savefig(os.path.join(save_dir, 'best_corr_percents.svg'))
    plt.close(f_frac)

    # Plot the cumulative distribution functions by segment and the density differences
    density_x_vals = np.arange(0, 1.1, 0.025)
    density_bins = np.concatenate((density_x_vals - 0.05,1.125*np.ones(1)))
    taste_pairs = list(combinations(np.arange(num_tastes), 2))
    epoch_pairs = list(combinations(np.arange(len(epochs_to_analyze)), 2))

    f_cum, ax_cum = plt.subplots(
        nrows=len(segments_to_analyze), ncols=2, figsize=(8, 8))
    f_dens, ax_dens = plt.subplots(
        nrows=len(segments_to_analyze), ncols=2, figsize=(8, 8))
    for s_ind, s_i in enumerate(segments_to_analyze):  # By rest interval
        dev_sorting = seg_data[s_i]
        # Taste data
        taste_density = []
        for t_i in range(num_tastes):
            taste_corr = []
            for l_ind, l_name in enumerate(best_corr_labels):
                if dev_sorting[l_name]['taste'] == t_i:
                    taste_corr.extend(dev_sorting[l_name]['corr_vals'])
            taste_corr = np.array(taste_corr).squeeze()
            density_hist_data = np.histogram(
                taste_corr, bins=density_bins, density=True)
            try: #There are actually taste correlations to plot
                ax_cum[s_ind, 0].hist(taste_corr, bins=density_bins, histtype='step', density=True,
                                      cumulative=True, color=taste_colors[t_i, :], label=dig_in_names[t_i])
                ax_dens[s_ind, 0].plot(
                    density_x_vals, density_hist_data[0], color=taste_colors[t_i, :], label=dig_in_names[t_i])
                taste_density.append(density_hist_data[0])
            except:
                taste_density.append([])
            ax_cum[s_ind, 0].set_ylim([-0.05, 1.05])
            ax_cum[s_ind, 0].set_xlim([-0.05, 1.05])
            #ax_dens[s_ind, 0].set_ylim([-0.05, 1.05])
            ax_dens[s_ind, 0].set_xlim([-0.05, 1.05])
        # Epoch data
        epoch_density = []
        for e_ind, e_i in enumerate(epochs_to_analyze):
            epoch_corr = []
            for l_ind, l_name in enumerate(best_corr_labels):
                if dev_sorting[l_name]['epoch'] == e_ind:
                    epoch_corr.extend(dev_sorting[l_name]['corr_vals'])
            epoch_corr = np.array(epoch_corr).squeeze()
            density_hist_data = np.histogram(
                epoch_corr, bins=density_bins, density=True)
            epoch_density.append(density_hist_data[0])
            ax_cum[s_ind, 1].hist(epoch_corr, bins=density_bins, histtype='step', density=True,
                                  cumulative=True, color=epoch_colors[e_ind, :], label='Epoch ' + str(e_i))
            ax_cum[s_ind, 1].set_ylim([-0.05, 1.05])
            ax_cum[s_ind, 1].set_xlim([-0.05, 1.05])
            ax_dens[s_ind, 1].plot(
                density_x_vals, density_hist_data[0], color=epoch_colors[e_ind, :], label='Epoch ' + str(e_i))
            #ax_dens[s_ind, 1].set_ylim([-0.05, 1.05])
            ax_dens[s_ind, 1].set_xlim([-0.05, 1.05])
        # Plot cleanups
        ax_cum[s_ind, 0].set_ylabel(segment_names[s_i])
        ax_dens[s_ind, 0].set_ylabel(segment_names[s_i])
    # Cumulative plot cleanups
    ax_cum[0, 0].set_title('Taste Correlations')
    ax_cum[0, 1].set_title('Epoch Correlations')
    ax_cum[0, 0].legend()
    ax_cum[0, 1].legend()
    ax_cum[-1, 0].set_xlabel('Correlation')
    ax_cum[-1, 1].set_xlabel('Correlation')
    f_cum.suptitle('Cumulative Correlations of Best Events')
    plt.tight_layout()
    f_cum.savefig(os.path.join(save_dir, 'best_corr_cum_hist.png'))
    f_cum.savefig(os.path.join(save_dir, 'best_corr_cum_hist.svg'))
    plt.close(f_cum)
    # Density plot cleanups
    ax_dens[0, 0].set_title('Taste Correlations')
    ax_dens[0, 1].set_title('Epoch Correlations')
    ax_dens[0, 0].legend()
    ax_dens[0, 1].legend()
    ax_dens[-1, 0].set_xlabel('Correlation')
    ax_dens[-1, 1].set_xlabel('Correlation')
    f_dens.suptitle('Normalized Density Correlations of Best Events')
    plt.tight_layout()
    f_dens.savefig(os.path.join(save_dir, 'best_corr_norm_dens_hist.png'))
    f_dens.savefig(os.path.join(save_dir, 'best_corr_norm_dens_hist.svg'))
    plt.close(f_dens)
    
def sig_count_plot(sig_dev_counts, segments_to_analyze, segment_names,
                   dig_in_names, save_dir):
    """
    Basic function to plot the number of significant deviations by condition.
    Called from / data passed from dev_funcs.py function 
    "calculate_significant_dev()".

    Returns
    -------
    None.

    """
    num_seg = len(segments_to_analyze)
    num_tastes = len(dig_in_names)
    
    max_num = 0
    max_num_cp = 0
    #By segment
    f,ax = plt.subplots(ncols = num_seg, figsize=(4*num_seg,4))
    for s_i in range(num_seg):
        for t_i in range(num_tastes):
            ax[s_i].plot(sig_dev_counts[s_i][t_i],label=dig_in_names[t_i])
            if max(sig_dev_counts[s_i][t_i]) > max_num:
                max_num = max(sig_dev_counts[s_i][t_i])
            if len(sig_dev_counts[s_i][t_i]) > max_num_cp:
                max_num_cp = len(sig_dev_counts[s_i][t_i])
        ax[s_i].set_title(segment_names[segments_to_analyze[s_i]])
        ax[s_i].set_xlabel('Epoch Index')
        ax[s_i].set_ylabel('# Significant Deviation Events')
    for s_i in range(num_seg):
        ax[s_i].set_ylim([-1,max_num + np.ceil(0.05*max_num).astype('int')])
        ax[s_i].legend(loc='upper left')
    plt.tight_layout()
    f.savefig(os.path.join(save_dir,'sig_dev_event_counts_segment.png'))
    f.savefig(os.path.join(save_dir,'sig_dev_event_counts_segment.svg'))
    plt.close(f)
    
    #By taste
    f,ax = plt.subplots(ncols = num_tastes, figsize=(4*num_tastes,4))
    for t_i in range(num_tastes):
        for s_i in range(num_seg):
            ax[t_i].plot(sig_dev_counts[s_i][t_i],label=segment_names[segments_to_analyze[s_i]])
        ax[t_i].set_title(dig_in_names[t_i])
        ax[t_i].set_xlabel('Epoch Index')
        ax[t_i].set_ylabel('# Significant Deviation Events')
        ax[t_i].set_ylim([-1,max_num + np.ceil(0.05*max_num).astype('int')])
        ax[t_i].legend(loc='upper left')
    plt.tight_layout()
    f.savefig(os.path.join(save_dir,'sig_dev_event_counts_taste.png'))
    f.savefig(os.path.join(save_dir,'sig_dev_event_counts_taste.svg'))
    plt.close(f)
    
def sig_val_plot(sig_dev,segments_to_analyze,segment_names,dig_in_names,save_dir):
    """
    This function plots the distributions of correlation values of events that 
    are significant across conditions.
    
    INPUTS:
        - sig_dev: dictionary generated by functions/dev_funcs.py function 
            'calculate_significant_dev()' that contains times and correlation
            values of significant deviation events across conditions.
        - segments_to_analyze: which segments are being analyzed
        - segment_names: names of segments
        - dig_in_names: names of tastes
        - save_dir: where to save plots
    OUTPUTS:
        - plots of distributions of correlation values across segment, taste, 
            and epoch for events that are deemed significant.
    """
    
    num_seg = len(segments_to_analyze)
    num_tastes = len(dig_in_names)
    
    max_num_cp = 0
    max_count = 0
    #By segment
    f,ax = plt.subplots(nrows= num_tastes, ncols = num_seg, figsize=(4*num_seg,4*num_tastes))
    for s_i in range(num_seg):
        for t_i in range(num_tastes):
            num_cp = len(sig_dev[s_i]['taste_sig'][t_i]['cp_sig'])
            if num_cp > max_num_cp:
                max_num_cp = num_cp
            for cp_i in range(num_cp):
                corr_vals = sig_dev[s_i]['taste_sig'][t_i]['cp_sig'][cp_i]['dev_corrs']
                hist_results = ax[t_i,s_i].hist(corr_vals,bins=np.arange(-0.05,1.05,0.05),
                                 alpha=0.3,label='Epoch ' + str(cp_i))
                if max(hist_results[0]) > max_count:
                    max_count = max(hist_results[0])
    for s_i in range(num_seg):
        for t_i in range(num_tastes):
            ax[t_i,s_i].set_title(segment_names[segments_to_analyze[s_i]] + ' x ' + dig_in_names[t_i])
            ax[t_i,s_i].set_xlabel('Correlation')
            ax[t_i,s_i].set_ylabel('# Deviation Events')
            ax[t_i,s_i].set_xlim([0,1])
            ax[t_i,s_i].set_ylim([0,max_count + np.ceil(0.05*max_count).astype('int')])
            ax[t_i,s_i].legend(loc='upper left')
    plt.tight_layout()
    f.savefig(os.path.join(save_dir,'sig_dev_corr_hists.png'))
    f.savefig(os.path.join(save_dir,'sig_dev_corr_hists.svg'))
    plt.close(f)
    
def plot_dev_dim_reduced(dev_vec, tastant_fr_dist, segment_names, 
                         dig_in_names, segments_to_analyze, 
                         plot_save_dir):
    """
    Plot the deviation events in reduced dimensions
    """
    
    colors = ['forestgreen','lime','mediumseagreen','royalblue','navy','blue',
              'palevioletred','red','maroon']
    seg_colors = ['darkorange','cyan','magenta']
    epoch_colors = ['forestgreen','royalblue','maroon']
    num_neur = len(dev_vec[0][0])
    num_tastes = len(dig_in_names)
    num_cp = len(tastant_fr_dist[0][0])
    X = []
    y = []
    for s_ind, s_i in enumerate(segments_to_analyze):
        #Collect data for SVM
        seg_dev = dev_vec[s_ind]
        num_dev = len(seg_dev)
        X.extend(seg_dev)
        y.extend(list(s_ind*np.ones(num_dev)))
        #Perform PCA dim reduction for individual segment devs
        pca = PCA(2)
        pca.fit(np.array(seg_dev))
        transformed_devs = pca.transform(seg_dev)
        f_pca_tastes, ax_pca_tastes = plt.subplots(ncols = num_cp, \
                                     sharex = True, sharey = True, \
                                         figsize=(10,4))
        f_pca_epochs, ax_pca_epochs = plt.subplots(ncols = num_tastes, \
                                     sharex = True, sharey = True, \
                                         figsize=(10,4))
        for cp_i in range(num_cp):
            ax_pca_tastes[cp_i].set_title('Epoch ' + str(cp_i))
        for t_i, taste in enumerate(dig_in_names):
            ax_pca_epochs[t_i].set_title(taste)
            ax_pca_epochs[t_i].scatter(transformed_devs[:,0],transformed_devs[:,1],alpha=0.1,\
                        color='k',label='Deviations')
            t_data = tastant_fr_dist[t_i]
            for cp_i in range(num_cp):
                t_cp_data =np.squeeze(np.array([t_data[t][cp_i] for t in t_data]))
                t_cp_mean = np.expand_dims(np.nanmean(t_cp_data,0),0)
                t_transform = pca.transform(t_cp_mean)
                if t_i == 0:
                    ax_pca_tastes[cp_i].scatter(transformed_devs[:,0],transformed_devs[:,1],alpha=0.1,\
                                color='k',label='Deviations')
                    
                ax_pca_tastes[cp_i].scatter(t_transform[:,0],t_transform[:,1],alpha=1,\
                            color=epoch_colors[t_i],label=taste)
                ax_pca_epochs[t_i].scatter(t_transform[:,0],t_transform[:,1],alpha=1,\
                            color=epoch_colors[cp_i],label='Epoch ' + str(cp_i))
        plt.figure(f_pca_epochs)
        ax_pca_epochs[0].legend(loc='upper left')
        plt.suptitle(segment_names[s_i])
        plt.tight_layout()
        f_pca_epochs.savefig(os.path.join(plot_save_dir,segment_names[s_i] + '_epochs_pca.png'))
        f_pca_epochs.savefig(os.path.join(plot_save_dir,segment_names[s_i] + '_epochs_pca.svg'))
        plt.close(f_pca_epochs)
        plt.figure(f_pca_tastes)
        ax_pca_tastes[0].legend(loc='upper left')
        plt.suptitle(segment_names[s_i])
        plt.tight_layout()
        f_pca_tastes.savefig(os.path.join(plot_save_dir,segment_names[s_i] + '_tastes_pca.png'))
        f_pca_tastes.savefig(os.path.join(plot_save_dir,segment_names[s_i] + '_tastes_pca.svg'))
        plt.close(f_pca_tastes)
    
    #Fit SVM to dev
    #Rescale values using z-scoring
    X = np.array(X)
    y = np.array(y)    
    X_norm = (X - np.expand_dims(np.nanmean(X,1),1))/np.expand_dims(np.nanstd(X,1),1)
    svm_class = svm.SVC(kernel='linear')
    svm_class.fit(X_norm,y)
    w = svm_class.coef_[0] #weights of classifier normal vector
    w_norm = np.linalg.norm(w)
    X_projected = w@X_norm.T/w_norm**2
    #Calculate orthogonal vectors with significantly different distributions for 2D plot
    sig_u = [] #significant vector storage
    u_p = [] #p-vals of significance
    for i in range(100):
        inds_to_use = random.sample(list(np.arange(num_neur)),2)
        u = np.zeros(num_neur)
        u[inds_to_use[0]] = -1*w[inds_to_use[1]]
        u[inds_to_use[1]] = w[inds_to_use[0]]
        u_norm = np.linalg.norm(u)
        u_proj = u@X_norm.T/u_norm**2
        sp_1_u_proj = u_proj[:num_dev]
        sp_2_u_proj = u_proj[num_dev:]
        ks_stats = ks_2samp(sp_1_u_proj,sp_2_u_proj)
        if ks_stats.pvalue <= 0.05:
            sig_u.append(u)
            u_p.append(ks_stats.pvalue)
    if len(u_p) > 0:
        min_p = np.argmin(u_p)
        u = sig_u[min_p]
        u_norm = np.linalg.norm(u)
        X_orth_projected = u@X_norm.T/u_norm**2
        for s_ind, s_i in enumerate(segments_to_analyze):
            f_svm = plt.figure(figsize=(5,5))
            s_data = np.where(y == s_ind)[0]
            plt.scatter(X_projected[s_data],X_orth_projected[s_data],alpha=0.1,\
                        color='k',label='Deviations')
            for t_i, taste in enumerate(dig_in_names):
                t_data = tastant_fr_dist[t_i]
                if taste == 'none':
                    none_data = []
                    for cp_i in range(num_cp):
                        none_data.extend([t_data[t][cp_i] for t in t_data])
                    none_data = np.squeeze(np.array(none_data))
                    t_cp_mean = np.expand_dims(np.nanmean(none_data,0),0)
                    t_mean_norm = (t_cp_mean - np.nanmean(t_cp_mean))/np.nanstd(t_cp_mean)
                    t_projected = w@t_mean_norm.T/w_norm**2
                    t_orth_projected = u@t_mean_norm.T/u_norm**2
                    plt.scatter(t_projected,t_orth_projected,color=colors[t_i*num_cp + cp_i],
                                label=taste)
                else:
                    for cp_i in range(num_cp):
                        t_cp_data = np.squeeze(np.array([t_data[t][cp_i] for t in t_data]))
                        t_cp_mean = np.expand_dims(np.nanmean(t_cp_data,0),0)
                        t_mean_norm = (t_cp_mean - np.nanmean(t_cp_mean))/np.nanstd(t_cp_mean)
                        t_projected = w@t_mean_norm.T/w_norm**2
                        t_orth_projected = u@t_mean_norm.T/u_norm**2
                        plt.scatter(t_projected,t_orth_projected,color=colors[t_i*num_cp + cp_i],
                                    label=taste + ' epoch ' + str(cp_i))
            plt.legend()
            plt.title(segment_names[s_i])
            plt.tight_layout()
            f_svm.savefig(os.path.join(plot_save_dir,segment_names[s_i] + '_svm_proj.png'))
            f_svm.savefig(os.path.join(plot_save_dir,segment_names[s_i] + '_svm_proj.svg'))
            plt.close(f_svm)
        
    
def plot_dev_x_dev_corr(dev_vec, segment_names, segments_to_analyze, plot_save_dir):
    """
    Plot the distribution of correlations of deviation events to each other
    compared with average corr of taste responses with each other.
    """
    
    f_corr, ax_corr = plt.subplots(ncols=len(segments_to_analyze),sharex = True,
                                   sharey = True, figsize=(len(segments_to_analyze)*4,4))
    for s_ind, s_i in enumerate(segments_to_analyze):
        seg_dev = dev_vec[s_ind]
        pair_inds = list(combinations(np.arange(len(seg_dev)),2))
        corr_vals = []
        for pi in tqdm.tqdm(pair_inds):
            corr_vals.append(pearsonr(seg_dev[pi[0]],seg_dev[pi[1]])[0])
        ax_corr[s_ind].hist(corr_vals)
        ax_corr[s_ind].set_title(segment_names[s_i])
    plt.tight_layout()
    f_corr.savefig(os.path.join(plot_save_dir,'dev_x_dev_corr.svg'))
    f_corr.savefig(os.path.join(plot_save_dir,'dev_x_dev_corr.png'))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:35:22 2023

@author: hannahgermaine

This is a collection of functions for calculating and analyzing general 
cross-segment activity changes.
"""
import os
import json
import tqdm
import itertools
import random
import tables
from numba import jit
import scipy.stats as stats
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
os.environ["OMP_NUM_THREADS"] = "4"


def bin_spike_counts(save_dir, segment_spike_times, segment_names, segment_times):
    """This function calculates the number of spikes across bins in the recording
    and compares the distributions between/across segments to determine if overall
    firing has changed
    INPUTS:
            - save_dir: directory to save results
            - segment_spike_times: [num_segments x num_neur] nested list with indices of spikes in a ms timescale dataset (each index is a ms)
            - segment_names: the name of each segment
            - segment_times: the ms time delineations of each segment
    """
    # Create save dir for individual distributions
    dist_save_dir = save_dir + 'indiv_distributions/'
    if os.path.isdir(dist_save_dir) == False:
        os.mkdir(dist_save_dir)
        
    if not os.path.isfile(os.path.join(dist_save_dir,'segment_fano_factors.npy')):
        # First calculate individual distributions, plot them, and save
        segment_neur_counts = dict()
        segment_counts = dict()
        segment_frs = dict()
        segment_isis = dict()
        segment_fano_factors = dict()
        # Get distributions for different bin sizes
        bin_sizes = np.arange(0.05, 1.05, 0.05)
        print("\nCalculating distributions for different bin sizes.")
        for s_i in tqdm.tqdm(range(len(segment_names))):
            segment_spikes = segment_spike_times[s_i]
            num_neur = len(segment_spikes)
            segment_start_time = segment_times[s_i]
            segment_end_time = segment_times[s_i+1]
            segment_len = int(segment_end_time-segment_start_time)
            # Convert to a binary spike matrix
            bin_spike = np.zeros((num_neur, segment_len+1))
            for n_i in range(num_neur):
                spike_indices = (
                    np.array(segment_spikes[n_i]) - segment_start_time).astype('int')
                bin_spike[n_i, spike_indices] = 1
            # Calculate count, fr, isi, and fano factor distributions
            neur_count_results = dict(calculate_spike_neuron_distribution(
                bin_spike, bin_sizes[i]) for i in range(len(bin_sizes)))
            count_results = dict(calculate_spike_count_distribution(
                bin_spike, bin_sizes[i]) for i in range(len(bin_sizes)))
            fr_results = dict(calculate_fr_distribution(
                bin_spike, bin_sizes[i]) for i in range(len(bin_sizes)))
            isi_results = calculate_isi_distribution(bin_spike)
            fano_results = calculate_fano_factors(count_results)
            # Save results to master dictionary
            segment_neur_counts.update({segment_names[s_i]: neur_count_results})
            segment_counts.update({segment_names[s_i]: count_results})
            segment_frs.update({segment_names[s_i]: fr_results})
            segment_isis.update({segment_names[s_i]: isi_results})
            segment_fano_factors.update({segment_names[s_i]: fano_results})
            # Plot spike distributions
            # plot_distributions(
            #     neur_count_results, segment_names[s_i], 'Neuron Spike Counts', figure_save_dir)
            # plot_distributions(
            #     count_results, segment_names[s_i], 'Spike Counts', figure_save_dir)
            # plot_distributions(
            #     fr_results, segment_names[s_i], 'Firing Rates', figure_save_dir)
            # plot_distributions(isi_results, 'ISI Distributions' + ' CV = ' + str(
            #     round(np.std(isi_results)/np.mean(isi_results), 2)), 'ISIs (s)', figure_save_dir)
        np.save(os.path.join(dist_save_dir,'segment_neur_counts.npy'),segment_neur_counts,allow_pickle=True)
        np.save(os.path.join(dist_save_dir,'segment_counts.npy'),segment_counts,allow_pickle=True)
        np.save(os.path.join(dist_save_dir,'segment_frs.npy'),segment_frs,allow_pickle=True)
        np.save(os.path.join(dist_save_dir,'segment_isis.npy'),segment_isis,allow_pickle=True)
        np.save(os.path.join(dist_save_dir,'segment_fano_factors.npy'),segment_fano_factors,allow_pickle=True)
        
    # Use the T-Test to calculate if segment distributions are different
    # Create save dir for T-Test pair results
    figure_save_dir = save_dir + 'pair_t_tests/'
    if os.path.isdir(figure_save_dir) == False:
        os.mkdir(figure_save_dir)
        print("\nCalculating T-Test for pairs of segments:")
        s_i_pairs = list(itertools.combinations(segment_names, 2))
        print("\tNeurons Spiking Count distributions:")
        # First calculating for spike counts
        neur_count_save_dir = figure_save_dir + 'neur_spike_counts/'
        dist_name = 'Neuron Spike Count'
        segment_pair_count_calculations = T_test_pipeline(
            neur_count_save_dir, dist_name, s_i_pairs, segment_neur_counts)
        print("\tCount distributions:")
        # First calculating for spike counts
        count_save_dir = figure_save_dir + 'spike_counts/'
        dist_name = 'Spike Count'
        segment_pair_count_calculations = T_test_pipeline(
            count_save_dir, dist_name, s_i_pairs, segment_counts)
        # Second calculating for firing rates (Hz)
        print("\tFiring rate distributions:")
        fr_save_dir = figure_save_dir + 'frs/'
        dist_name = 'Firing Rate'
        segment_pair_fr_calculations = T_test_pipeline(
            fr_save_dir, dist_name, s_i_pairs, segment_frs)
        # Third calculating for ISIs
        #print("\tISI and Fano Factor Plots:")
        #isi_fano_save_dir = figure_save_dir + 'isis_fano/'
        #if os.path.isdir(isi_fano_save_dir) == False:
        #    os.mkdir(isi_fano_save_dir)
        #single_trend_plots(segment_fano_factors, 'Fano Factor', isi_fano_save_dir)
        #single_trend_plots(segment_isis, 'ISIs', isi_fano_save_dir)


@jit(forceobj=True)
def calculate_spike_neuron_distribution(spike_times, bin_size):
    """This function calculates the spike count distribution for a given dataset
    and given bin sizes
    INPUTS:
            - spike_times: binary matrix of num_neur x num_time (in ms bins) with 1s where a neuron fires
            - bin_size: width (in seconds) of bins to calculate the number of spikes in
    """
    bin_dt = int(bin_size*1000)
    bin_borders = np.arange(0, len(spike_times[0, :]), bin_dt)
    bin_neur_counts = np.zeros(len(bin_borders)-1)
    for b_i in range(len(bin_borders)-1):
        bin_neur_counts[b_i] = int(
            np.sum(np.sum(spike_times[:, bin_borders[b_i]:bin_borders[b_i+1]], 1) > 0))

    return str(bin_size), bin_neur_counts


@jit(forceobj=True)
def calculate_spike_count_distribution(spike_times, bin_size):
    """This function calculates the spike count distribution for a given dataset
    and given bin sizes
    INPUTS:
            - spike_times: binary matrix of num_neur x num_time (in ms bins) with 1s where a neuron fires
            - bin_size: width (in seconds) of bins to calculate the number of spikes in
    """
    bin_dt = int(bin_size*1000)
    bin_borders = np.arange(0, len(spike_times[0, :]), bin_dt)
    bin_counts = np.zeros(len(bin_borders)-1)
    for b_i in range(len(bin_borders)-1):
        bin_counts[b_i] = np.sum(
            spike_times[:, bin_borders[b_i]:bin_borders[b_i+1]])

    return str(bin_size), bin_counts


@jit(forceobj=True)
def calculate_fr_distribution(spike_times, bin_size):
    """This function calculates the firing rate distribution for a given dataset
    and given bin sizes
    INPUTS:
            - spike_times: binary matrix of num_neur x num_time (in ms bins) with 1s where a neuron fires
            - bin_size: width (in seconds) of bins to calculate the number of spikes in
    """
    num_neur, _ = np.shape(spike_times)
    bin_dt = int(bin_size*1000)
    bin_borders = np.arange(0, len(spike_times[0, :]), bin_dt)
    bin_frs = np.zeros(len(bin_borders)-1)
    for b_i in range(len(bin_borders)-1):
        bin_frs[b_i] = np.sum(
            spike_times[:, bin_borders[b_i]:bin_borders[b_i+1]])/num_neur/bin_size

    return str(bin_size), bin_frs


@jit(forceobj=True)
def calculate_isi_distribution(spike_times):
    """This function calculates the inter-spike-interval distribution for a given dataset
    and given bin sizes
    INPUTS:
            - spike_times: binary matrix of num_neur x num_time (in ms bins) with 1s where a neuron fires
            - bin_size: width (in seconds) of bins to calculate the number of spikes in
    """
    all_isis = []
    num_neur, _ = np.shape(spike_times)
    for n_i in range(num_neur):
        spike_indices = np.where(spike_times[n_i,:] > 0)[0]
        bin_isis = np.diff(spike_indices)/1000  # Converted to seconds
        all_isis.extend(list(bin_isis))

    return np.array(all_isis)


@jit(forceobj=True)
def calculate_fano_factors(count_results):
    """Calculates the fano factor by bin size by taking an input of the spike count
    dictionary generated by calculate_spike_count_distribution()"""

    fano_results = dict()
    for b_sz in count_results:
        c_vals = count_results[b_sz]
        var_val = np.var(c_vals)
        mean_val = np.mean(c_vals)
        fano_results.update({b_sz: var_val/mean_val})

    return fano_results


@jit(forceobj=True)
def plot_distributions(results, title, dist_name, save_location):
    """This function plots the given distributions on the same axes for easy
    comparison. Given distributions must be input as a dictionary with name and
    values to be easily plotted together
    INPUTS:
            - results: dictionary of distribution input and np.array of results
            - title: title for the plot of which distribution is being plotted
            - dist_name: what is the distribution of? ex.) "Spike Counts"
            - save_location: folder in which to save the plot
    """
    fig = plt.figure(figsize=(5, 5))
    if type(results) == dict:
        for key in results:
            plt.hist(results[key], label=str(key), bins=50, alpha=0.5)
        if len(results) <= 10:
            plt.legend()
    else:
        plt.hist(results, bins=min(len(results)/10, 50))
    plt.tight_layout()
    plt.xlabel(dist_name)
    plt.ylabel('Occurrences')
    plt.title(title)
    im_name = ('_').join(title.split(' '))
    plt.savefig(save_location + im_name + '_' +
                ('_').join(dist_name.split(' ')) + '.png')
    plt.savefig(save_location + im_name + '_' +
                ('_').join(dist_name.split(' ')) + '.svg')
    plt.close()


@jit(forceobj=True)
def T_test_pipeline(t_save_dir, dist_name, s_i_pairs, data_dict):
    """This function runs data through a 2-sample KS test pipline of calculations
    and plots
    INPUTS:
            - ks_save_dir: directory to save given results
            - dist_name: the name of the distribution being tested
            - s_i_pairs: names of segments in pair
            - data_dict: dictionary of distribution values to be tested
    """
    if os.path.isdir(t_save_dir) == False:
        os.mkdir(t_save_dir)
    segment_pair_calculations = dict()
    for pair_i in tqdm.tqdm(s_i_pairs):
        seg_1 = pair_i[0]
        seg_2 = pair_i[1]
        print("\t\t" + seg_1 + " vs " + seg_2)
        seg_1_data = data_dict[seg_1]
        seg_2_data = data_dict[seg_2]
        pair_results = T_test_distributions(
            dist_name, seg_1, seg_2, seg_1_data, seg_2_data, t_save_dir)
        segment_pair_calculations.update({seg_1+"_"+seg_2: pair_results})

    return segment_pair_calculations


@jit(forceobj=True)
def T_test_distributions(dist_name, name_1, name_2, dict_1, dict_2, fig_save_dir):
    """This function performs a two-sample KS-test on a given pair of values:
    INPUTS:
            - dist_name: what distribution is being tested
            - name_1: the name of what the first dictionary refers to
            - name_2: the name of what the second dictionary refers to
            - dict_1, dict_2: two dictionaries with matching keys - the matching keys' values 
                    will be compared against each other with a KS-test.
            - fig_save_dir: directory to store plots
    OUTPUTS:
            - a dictionary with a tuple of whether the pair rejects the null distribution
            (1 for rejects 0 for not) and the KS-test results for each matched key.
            - a plot of the ks-test p-values, mean distribution values, and std distribution values
    """
    results = dict()
    results_minus_0 = dict()
    bin_size = []
    means = []
    means_minus_0 = []
    stds = []
    stds_minus_0 = []
    for key in dict_1:
        bin_size.extend([float(key)])
        values_1 = dict_1[key]
        values_2 = dict_2[key]
        values_1_no_0 = values_1[np.where(values_1 > 0)]
        values_2_no_0 = values_2[np.where(values_2 > 0)]
        means.append([np.mean(values_1), np.mean(values_2)])
        means_minus_0.append([np.mean(values_1_no_0), np.mean(values_2_no_0)])
        stds.append([np.std(values_1), np.std(values_2)])
        stds_minus_0.append([np.std(values_1_no_0), np.std(values_2_no_0)])
        tresult = stats.ttest_ind(values_1, values_2)
        results.update({key: tresult})
        tresult_minus_0 = stats.ttest_ind(values_1_no_0, values_2_no_0)
        results_minus_0.update({key: tresult_minus_0})

    # Plot the KS-Test results as bin size increases
    T_test_plots(dist_name, name_1, name_2, results,
                  bin_size, means, stds, fig_save_dir)

    # Plot the KS-Test results as bin size increases without 0 bins included
    T_test_plots(dist_name, name_1, name_2, results_minus_0, bin_size,
                  means_minus_0, stds_minus_0, fig_save_dir, name_modifier='_no_0')

    return results


@jit(forceobj=True)
def T_test_plots(dist_name, name_1, name_2, results, bin_size, means, stds, fig_save_dir, name_modifier=''):
    """This function plots the results from the KS-test calculator"""
    # Plot the T-Test results as bin size increases
    fig = plt.figure(figsize=(6, 10))
    # Same/Diff subplot
    key_val_pairs = []
    for key in results:
        tresult = results[key]
        same = 0
        if tresult[1] < 0.05:
            same = 1
        key_val_pairs.append([float(key), same])
    plt.subplot(3, 2, 1)
    plt.plot(np.array(key_val_pairs).T[0, :], np.array(key_val_pairs).T[1, :])
    plt.yticks([0, 1], ['Same', 'Different'], fontsize=10)
    plt.title('Same or Different', fontsize=10)
    plt.xlabel('Bin Size (s)', fontsize=10)
    # P-values subplot
    p_val_pairs = []
    for key in results:
        tresult = results[key]
        p_val_pairs.append([float(key), tresult[1]])
    plt.subplot(3, 2, 2)
    plt.plot(np.array(p_val_pairs).T[0, :], np.array(p_val_pairs).T[1, :])
    plt.axhline(0.05, linestyle='-', alpha=0.5, color='r')
    plt.title('P-Values by Bin Size', fontsize=10)
    plt.xlabel('Bin Size (s)', fontsize=10)
    plt.ylabel('P-Value', fontsize=10)
    # Means subplot
    plt.subplot(3, 2, 3)
    plt.plot(bin_size, np.array(means)[:, 0], label=name_1)
    plt.plot(bin_size, np.array(means)[:, 1], label=name_2)
    plt.legend()
    plt.title('Mean ' + dist_name + ' by Bin Size', fontsize=10)
    plt.xlabel('Bin Size (s)', fontsize=10)
    plt.ylabel('Mean ' + dist_name, fontsize=10)
    # Stds subplot
    plt.subplot(3, 2, 4)
    plt.plot(bin_size, np.array(stds)[:, 0], label=name_1)
    plt.plot(bin_size, np.array(stds)[:, 1], label=name_2)
    plt.legend()
    plt.title('Std. ' + dist_name + ' by Bin Size', fontsize=10)
    plt.xlabel('Bin Size (s)', fontsize=10)
    plt.ylabel('Std. ' + dist_name, fontsize=10)
    # CVs subplot
    plt.subplot(3, 2, 5)
    plt.plot(bin_size, np.array(stds)[:, 0] /
             np.array(means)[:, 0], label=name_1)
    plt.plot(bin_size, np.array(stds)[:, 1] /
             np.array(means)[:, 1], label=name_2)
    plt.legend()
    plt.title('CV ' + dist_name + ' by Bin Size', fontsize=10)
    plt.xlabel('Bin Size (s)', fontsize=10)
    plt.ylabel('CV ' + dist_name, fontsize=10)
    plt.suptitle(name_1 + " vs " + name_2, fontsize=12)
    plt.tight_layout()
    im_name = ('_').join((name_1+"_vs_"+name_2+name_modifier).split(' '))
    plt.savefig(fig_save_dir + im_name + '.png')
    plt.savefig(fig_save_dir + im_name + '.svg')
    plt.close()


@jit(forceobj=True)
def single_trend_plots(segment_trends, trend_name, fig_save_dir):
    """This function is specific to plotting single trends per segment.
    INPUTS:
            - segment_trends: dictionary of key = segment, values = 1-D array
            - trend_name: name of trend to use in title and storage
            - fig_save_dir: directory to store figure
    """
    num_seg = len(segment_trends)
    fig = plt.figure(figsize=(num_seg, 2*num_seg))
    plot_ind = 0
    for seg_key in segment_trends:
        plot_ind += 1
        seg_value_dict = segment_trends[seg_key]
        try:
            bins = []
            values = []
            for key in seg_value_dict:
                bins.append(float(key))
                values.append(seg_value_dict[key]/1000)  # Converted to s
            plt.subplot(num_seg, 1, plot_ind)
            plt.plot(bins, values)
            plt.xlabel("Bin Size (s)")
            plt.ylabel(trend_name)
            plt.title(seg_key)
        except:
            plt.subplot(num_seg, 1, plot_ind)
            plt.hist(seg_value_dict,bins=100,density=True)
            plt.axvline(np.mean(seg_value_dict),color='k',label='Mean = ' + str(np.round(np.mean(seg_value_dict),2)))
            plt.legend()
            plt.xlabel(trend_name)
            plt.ylabel('Density')
            plt.title(seg_key)
    plt.suptitle(trend_name + " Plots")
    plt.tight_layout()
    plt.savefig(fig_save_dir + ('_').join(trend_name.split(' ')) + '.png')
    plt.close()

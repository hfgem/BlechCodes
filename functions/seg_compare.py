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
    bin_dt = int(bin_size*1000)
    bin_borders = np.arange(0, len(spike_times[0, :]), bin_dt)
    bin_frs = np.zeros(len(bin_borders)-1)
    for b_i in range(len(bin_borders)-1):
        bin_frs[b_i] = np.sum(
            spike_times[:, bin_borders[b_i]:bin_borders[b_i+1]])/bin_size

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


def bin_neur_spike_counts(save_dir, segment_spike_times, segment_names, segment_times, num_thresh, bin_size):
    """This function calculates the number of neurons spiking in a bin for each
    segment, and compares different related calculations across segments"""

    bin_dt = int(bin_size*1000)

    # Create HDF5 to save results
    hdf5_name = 'thresh_data.h5'
    hf5_dir = save_dir + hdf5_name
    hf5 = tables.open_file(hf5_dir, 'w', title=hf5_dir[-1])
    hf5.create_group('/', 'true_calcs')
    hf5.create_group('/', 'null_calcs')
    hf5.create_group('/', 'settings')
    atom = tables.FloatAtom()
    hf5.create_earray('/settings', 'bin_size', atom, (0,))
    exec("hf5.root.settings.bin_size.append(np.expand_dims(bin_size,0))")
    hf5.create_earray('/settings', 'num_thresh', atom, (0,))
    exec("hf5.root.settings.num_thresh.append(num_thresh)")
    hf5.close()
    print('Created nodes in HF5')

    print("Calculating High Neuron Spike Count Bins")
    # Get neur_count_results dictionary
    try:
        # Load json data
        neur_count_results = json.load(save_dir + "neur_count_results.json")
    except:
        neur_count_results = high_bins(
            segment_spike_times, segment_times, bin_size, num_thresh)
        # Save count dict - NEEDS WORK
        # for s_i in range(len(segment_names)):
        #	json_file = json.dumps(neur_count_results[s_i])
        #	f = open(save_dir + "neur_count_results" + ('_').join(segment_names[s_i].split(' ')) + ".json","w")
        #	f.write(json_file)
        #	f.close()
    # Get list results
    try:
        # Load hf5 data
        hf5 = tables.open_file(hf5_dir, 'r+', title=hf5_dir[-1])
        neur_count_seg = []
        for s_i in range(len(segment_names)):
            seg_name = ('_').join(('_').join(
                segment_names[s_i].split(' ')).split('-'))
            data_imported = exec(
                "hf5.root.true_calcs.neur_counts." + seg_name + "[:]")
            neur_count_seg.append(data_imported)
        hf5.close()
    except:
        # Reformat count results for storage
        neur_count_seg = []
        for s_i in range(len(segment_names)):
            seg_counts = neur_count_results[s_i]
            reformat_seg_counts = []
            for key in seg_counts:
                reformat_seg_counts.append(seg_counts[key])
            reformat_seg_counts = np.array(reformat_seg_counts)
            neur_count_seg.append(reformat_seg_counts)
        # Save count results
        hf5 = tables.open_file(hf5_dir, 'r+', title=hf5_dir[-1])
        atom = tables.FloatAtom()
        hf5.create_group('/true_calcs', 'neur_counts')
        for s_i in range(len(segment_names)):
            seg_name = ('_').join(('_').join(
                segment_names[s_i].split(' ')).split('-'))
            hf5.create_earray('/true_calcs/neur_counts', seg_name,
                              atom, (0,)+np.shape(neur_count_seg[s_i]))
            exec("hf5.root.true_calcs.neur_counts."+seg_name +
                 ".append(np.expand_dims(neur_count_seg[s_i],0))")
        hf5.close()

    num_null = 50
    print("Calculating Null Distribution Spike Count Bins")
    null_neur_count_results_separate = null_high_bins(
        num_null, segment_spike_times, segment_times, bin_size, num_thresh)
    null_neur_count_results = null_results_recombined(
        null_neur_count_results_separate, segment_times)
    # TO DO: Add storage of null results

    print("Calculating Bin Start/End Times")
    neur_bout_times = high_bin_times(segment_names, neur_count_results, bin_dt)
    null_neur_bout_times_separate = [high_bin_times(
        segment_names, null_neur_count_results_separate[n_n], bin_dt) for n_n in range(num_null)]
    # Reformat bout time results
    neur_bout_seg_thresh = []
    neur_bout_seg = []
    for s_i in range(len(segment_names)):
        seg_bouts = neur_bout_times[s_i]
        reformat_seg_thresh = []
        reformat_seg_bouts = []
        for key in seg_bouts:
            reformat_seg_thresh.extend([int(key)])
            bout_arrays = seg_bouts[key]
            reformat_seg_bouts.append(np.concatenate(
                (np.expand_dims(bout_arrays[0], 0), np.expand_dims(bout_arrays[1], 0))))
        neur_bout_seg.append(reformat_seg_bouts)
        neur_bout_seg_thresh.append(reformat_seg_thresh)
        del seg_bouts, reformat_seg_bouts, key, bout_arrays

    # Save bout times
    hf5 = tables.open_file(hf5_dir, 'r+', title=hf5_dir[-1])
    atom = tables.FloatAtom()
    hf5.create_group('/true_calcs', 'neur_bout_times')
    for s_i in range(len(segment_names)):
        seg_name = ('_').join(('_').join(
            segment_names[s_i].split(' ')).split('-'))
        hf5.create_group('/true_calcs/neur_bout_times/', seg_name)
        for n_t in range(len(neur_bout_seg_thresh[s_i])):
            hf5.create_earray('/true_calcs/neur_bout_times/'+seg_name, 'thresh_'+str(
                neur_bout_seg_thresh[s_i][n_t]), atom, (0,)+np.shape(neur_bout_seg[s_i][n_t]))
            exec("hf5.root.true_calcs.neur_bout_times."+seg_name+".thresh_" +
                 str(neur_bout_seg_thresh[s_i][n_t])+".append(np.expand_dims(neur_bout_seg[s_i][n_t],0))")
    hf5.close()
    # TO DO: Add storage of null bout times?

    print("Calculating High Neuron Bin Lengths")
    neur_bout_lengths = high_bin_lengths(
        segment_names, neur_count_results, bin_dt)
    null_neur_bout_lengths = high_bin_lengths(
        segment_names, null_neur_count_results, bin_dt)
    null_neur_bout_lengths_separate = [high_bin_lengths(
        segment_names, null_neur_count_results_separate[n_n], bin_dt) for n_n in range(num_null)]

    print("Plotting Neuron Bin Length Distributions")
    plot_name = 'True Data'
    plot_indiv_seg_high_length_dist(
        save_dir, plot_name, segment_names, neur_count_results, bin_dt)
    plot_name = 'Null Data'
    plot_indiv_seg_high_length_dist(
        save_dir, plot_name, segment_names, null_neur_count_results, bin_dt)

    print("Plotting Neuron Bin Length Distributions x Segments")
    plot_name = 'True Data'
    plot_cross_seg_length_dist(
        save_dir, plot_name, num_thresh, segment_names, neur_bout_lengths)
    plot_name = 'Example Null Data'
    plot_cross_seg_length_dist(
        save_dir, plot_name, num_thresh, segment_names, null_neur_bout_lengths_separate[0])

    print("Plotting Bout Count Trends x Segments")
    plot_name = 'True Data'
    plot_bout_count_trends(save_dir, plot_name, segment_names,
                           segment_times, neur_bout_lengths)
    plot_name = 'Example Null Data'
    plot_bout_count_trends(save_dir, plot_name, segment_names,
                           segment_times, null_neur_bout_lengths_separate[0])

    print("Plotting Bout Count Trends True x Null")
    plot_name = 'True x Null Data'
    plot_bout_count_trends_truexnull(save_dir, plot_name, segment_names,
                                     segment_times, neur_bout_lengths, null_neur_bout_lengths_separate)

    return neur_bout_seg_thresh, neur_bout_seg


def null_results_recombined(null_results, segment_times):
    # Reformat null results into same format as regular results
    num_seg = len(segment_times) - 1
    num_null = len(null_results)
    null_results_combined = []
    for s_i in range(num_seg):
        seg_dict = dict()
        for n_n in range(num_null):
            new_vals = null_results[n_n][s_i]
            if n_n == 0:
                for key in new_vals:
                    seg_dict.update({key: new_vals[key]})
            else:
                for key in new_vals:
                    current_vals = seg_dict[key]
                    seg_dict.update(
                        {key: np.concatenate((current_vals, new_vals[key]))})
        null_results_combined.append(seg_dict)
    del s_i, seg_dict, n_n, new_vals, key, current_vals

    return null_results_combined


@jit(forceobj=True)
def high_bin_times(segment_names, count_results, bin_dt):
    segment_bout_times = []
    for s_i in range(len(segment_names)):
        seg_name = ('_').join(segment_names[s_i].split(' '))
        bout_times_dict = dict()
        for key in count_results[s_i]:
            high_times = np.where(count_results[s_i][key] > 0)[0]
            if len(high_times) > 1:
                time_diffs = np.diff(high_times)
                bout_start_ind = np.concatenate(
                    (np.zeros(1), np.where(time_diffs > 1)[0] + 1)).astype('int')
                bout_end_ind = np.concatenate(
                    (bout_start_ind[1:] - 1, len(high_times)*np.ones(1)-1)).astype('int')
                bout_times_dict.update(
                    {key: [high_times[bout_start_ind], high_times[bout_end_ind]+bin_dt]})
        # Store bout lengths
        segment_bout_times.append(bout_times_dict)

    return segment_bout_times


@jit(forceobj=True)
def high_bin_lengths(segment_names, count_results, bin_dt):
    segment_bout_lengths = []
    for s_i in range(len(segment_names)):
        seg_name = ('_').join(segment_names[s_i].split(' '))
        bout_length_dict = dict()
        for key in count_results[s_i]:
            high_times = np.where(count_results[s_i][key] > 0)[0]
            if len(high_times) > 1:
                time_diffs = np.diff(high_times)
                bout_start_ind = np.concatenate(
                    (np.zeros(1), np.where(time_diffs > 1)[0] + 1)).astype('int')
                bout_end_ind = np.concatenate(
                    (bout_start_ind[1:] - 1, len(high_times)*np.ones(1)-1)).astype('int')
                bout_lengths = (
                    high_times[bout_end_ind] - high_times[bout_start_ind] + bin_dt)/1000  # In seconds
                bout_length_dict.update({key: bout_lengths})
        # Store bout lengths
        segment_bout_lengths.append(bout_length_dict)

    return segment_bout_lengths


def plot_indiv_seg_high_length_dist(save_dir, plot_name, segment_names, segment_bout_lengths, bin_dt):
    for s_i in range(len(segment_names)):
        bout_length_dict = segment_bout_lengths[s_i]
        seg_name = ('_').join(segment_names[s_i].split(' '))
        seg_thresh_bin_save_dir = save_dir + seg_name + '/'
        if os.path.isdir(seg_thresh_bin_save_dir) == False:
            os.mkdir(seg_thresh_bin_save_dir)
        plot_save_dir = seg_thresh_bin_save_dir + \
            ('_').join(plot_name.split(' ')) + '/'
        if os.path.isdir(plot_save_dir) == False:
            os.mkdir(plot_save_dir)
        for key in bout_length_dict:
            bout_lengths = bout_length_dict[key]
            if len(bout_lengths) > 1:
                # Generate individual histograms of bin lengths
                plt.figure(figsize=(5, 5))
                plt.hist(bout_lengths)
                plt.xlabel('Bout Length (s)')
                plt.ylabel('Number of Instances')
                plt.title('Bout Lengths for > ' + key + ' Neurons Firing' +
                          '\n' + 'Total Number of Bins = ' + str(len(bout_lengths)))
                plt.tight_layout()
                im_name = 'thresh_' + key + '_neur'
                plt.savefig(plot_save_dir + im_name + '.png')
                plt.savefig(plot_save_dir + im_name + '.svg')
                plt.close()
        # Generate combined histogram of bin lengths
        plt.figure(figsize=(5, 5))
        for key in bout_length_dict:
            plt.hist(bout_length_dict[key], alpha=0.4, label='>' + key +
                     ' neurons; ' + str(len(bout_length_dict[key])) + ' bouts')
        plt.legend()
        plt.xlabel('Bout Length (s)')
        plt.ylabel('Number of Instances')
        plt.title('Bout Lengths For Different Neuron Firing Cutoffs')
        plt.tight_layout()
        im_name = 'combined_thresh' + '_' + ('_').join(plot_name.split(' '))
        plt.savefig(seg_thresh_bin_save_dir + im_name + '.png')
        plt.savefig(seg_thresh_bin_save_dir + im_name + '.svg')
        plt.close()
    del s_i, seg_name, seg_thresh_bin_save_dir, bout_length_dict, key, bout_lengths, im_name

    return segment_bout_lengths


def plot_cross_seg_length_dist(save_dir, plot_name, num_thresh, segment_names, bout_lengths):
    # Plot the bout length distributions by cutoff trend across segments
    for n_i in num_thresh:
        try:
            plt.figure(figsize=(5, 5))
            for s_i in range(len(segment_names)):
                plt.hist(bout_lengths[s_i][str(n_i)],
                         alpha=0.4, label=segment_names[s_i])
            plt.legend()
            plt.xlabel('Bout Length (s)')
            plt.ylabel('Number of Instances')
            plt.title('Bout Lengths for > ' + str(n_i) + ' Neurons Firing')
            plt.tight_layout()
            im_name = 'thresh_' + str(n_i) + '_lengths' + \
                '_' + ('_').join(plot_name.split(' '))
            plt.savefig(save_dir + im_name + '.png')
            plt.savefig(save_dir + im_name + '.svg')
            plt.close()
        except:
            print("Could not plot for num = " + str(n_i))
    del n_i, s_i, im_name


def plot_bout_count_trends(save_dir, plot_name, segment_names, segment_times, bout_lengths):
    """This function plots the number of bouts by cutoff for each segment"""
    # Plot the number of bouts by cutoff trend for each segment
    plt.figure(figsize=(5, 5))
    for s_i in range(len(segment_names)):
        segment_bouts = bout_lengths[s_i]
        segment_bout_counts = []
        for key in segment_bouts:
            segment_bout_counts.append([int(key), len(segment_bouts[key])])
        segment_bout_counts = np.array(segment_bout_counts).T
        plt.plot(
            segment_bout_counts[0, :], segment_bout_counts[1, :], label=segment_names[s_i])
    plt.legend()
    plt.xlabel('Neuron Count Cutoff')
    plt.ylabel('Number of Bouts')
    plt.title('Number of Bouts by Cutoff per Segment')
    plt.tight_layout()
    im_name = 'bouts_by_cutoff' + '_' + ('_').join(plot_name.split(' '))
    plt.savefig(save_dir + im_name + '.png')
    plt.savefig(save_dir + im_name + '.svg')
    plt.close()
    del s_i, segment_bouts, segment_bout_counts, key, im_name
    # Plot the number of bouts by cutoff trend for each segment normalized by length
    plt.figure(figsize=(5, 5))
    for s_i in range(len(segment_names)):
        segment_bouts = bout_lengths[s_i]
        segment_length = (segment_times[s_i+1] - segment_times[s_i])/1000
        segment_bout_counts = []
        for key in segment_bouts:
            segment_bout_counts.append(
                [int(key), len(segment_bouts[key])/segment_length])
        segment_bout_counts = np.array(segment_bout_counts).T
        plt.plot(
            segment_bout_counts[0, :], segment_bout_counts[1, :], label=segment_names[s_i])
    plt.legend()
    plt.xlabel('Neuron Count Cutoff')
    plt.ylabel('Bouts/Second')
    plt.title('Bouts/Second by Cutoff per Segment')
    plt.tight_layout()
    im_name = 'bouts_per_second_by_cutoff' + \
        '_' + ('_').join(plot_name.split(' '))
    plt.savefig(save_dir + im_name + '.png')
    plt.savefig(save_dir + im_name + '.svg')
    plt.close()
    del s_i, segment_bouts, segment_length, segment_bout_counts, key, im_name


def plot_bout_count_trends_truexnull(save_dir, plot_name, segment_names, segment_times, true_bout_lengths, null_bout_lengths):
    """This function plots the number of bouts by cutoff for each segment"""
    # Set up plot colors
    num_segments = len(segment_names)
    num_null = len(null_bout_lengths)
    cm_subsection = np.linspace(0, 1, num_segments)
    # Color maps for each segment
    cmap = [cm.gist_rainbow(x) for x in cm_subsection]
    # Save calculations for separate plots
    true_bout_counts = []
    null_bout_counts_mean = []
    null_bout_counts_std = []
    # Plot the number of bouts by cutoff trend for each segment
    plt.figure(figsize=(5, 5))
    for s_i in range(num_segments):
        # Segment length in seconds for normalization
        segment_length = (segment_times[s_i+1] - segment_times[s_i])/1000
        # Calculate and plot true segment counts
        true_segment_bouts = true_bout_lengths[s_i]
        true_segment_bout_counts = []
        for key in true_segment_bouts:
            true_segment_bout_counts.append(
                [int(key), len(true_segment_bouts[key])])
        true_segment_bout_counts = np.array(true_segment_bout_counts).T
        true_bout_counts.append(true_segment_bout_counts)
        plt.plot(true_segment_bout_counts[0, :], true_segment_bout_counts[1, :] /
                 segment_length, color=cmap[s_i], label='True ' + segment_names[s_i])
        # Calculate and plot null segment count distributions
        null_segment_bout_counts = dict()
        for n_n in range(num_null):
            for key in null_bout_lengths[n_n][s_i]:
                try:
                    existing_vals = null_segment_bout_counts[key]
                    new_vals = np.array([len(null_bout_lengths[n_n][s_i])])
                    null_segment_bout_counts.update(
                        {key: np.concatenate((existing_vals, new_vals))})
                except:
                    new_vals = np.array([len(null_bout_lengths[n_n][s_i])])
                    null_segment_bout_counts.update({key: new_vals})
        null_segment_bout_counts_mean = []
        null_segment_bout_counts_std = []
        for key in null_segment_bout_counts:
            null_segment_bout_counts_mean.append(
                [int(key), np.mean(null_segment_bout_counts[key])])
            null_segment_bout_counts_std.append(
                [int(key), np.std(null_segment_bout_counts[key])])
        null_segment_bout_counts_mean = np.array(
            null_segment_bout_counts_mean).T
        null_bout_counts_mean.append(null_segment_bout_counts_mean)
        null_segment_bout_counts_std = np.array(null_segment_bout_counts_std).T
        null_bout_counts_std.append(null_segment_bout_counts_std)
        plt.plot(null_segment_bout_counts_mean[0, :], null_segment_bout_counts_mean[1, :] /
                 segment_length, color=cmap[s_i], linestyle='--', label='Null mean ' + segment_names[s_i])
        plt.fill_between(null_segment_bout_counts_std[0, :],
                         (null_segment_bout_counts_mean[1, :] -
                          null_segment_bout_counts_std[1, :])/segment_length,
                         (null_segment_bout_counts_mean[1, :] +
                          null_segment_bout_counts_std[1, :])/segment_length,
                         alpha=0.4, color=cmap[s_i], label='Null std ' + segment_names[s_i])
    plt.legend()
    plt.xlabel('Neuron Count Cutoff')
    plt.ylabel('Number of Bouts/Second')
    plt.title('Number of Bouts/Second by Cutoff per Segment')
    plt.tight_layout()
    im_name = 'norm_bouts_by_cutoff' + '_' + ('_').join(plot_name.split(' '))
    plt.savefig(save_dir + im_name + '.png')
    plt.savefig(save_dir + im_name + '.svg')
    plt.close()
    # Now plot for individual segments
    for s_i in range(num_segments):
        # Segment length in seconds for normalization
        segment_length = (segment_times[s_i+1] - segment_times[s_i])/1000
        seg_name = segment_names[s_i]
        seg_thresh_bin_save_dir = save_dir + seg_name + '/'
        if os.path.isdir(seg_thresh_bin_save_dir) == False:
            os.mkdir(seg_thresh_bin_save_dir)
        plot_save_dir = seg_thresh_bin_save_dir + \
            ('_').join(plot_name.split(' ')) + '/'
        if os.path.isdir(plot_save_dir) == False:
            os.mkdir(plot_save_dir)
        plt.figure(figsize=(5, 5))
        plt.plot(true_bout_counts[s_i][0, :], true_bout_counts[s_i][1, :] /
                 segment_length, color=cmap[s_i], label='True ' + segment_names[s_i])
        plt.plot(null_bout_counts_mean[s_i][0, :], null_bout_counts_mean[s_i][1, :] /
                 segment_length, color=cmap[s_i], linestyle='--', label='Null mean ')
        plt.fill_between(null_segment_bout_counts_std[0, :],
                         (null_bout_counts_mean[s_i][1, :] -
                          null_bout_counts_std[s_i][1, :])/segment_length,
                         (null_bout_counts_mean[s_i][1, :] +
                          null_bout_counts_std[s_i][1, :])/segment_length,
                         alpha=0.4, color=cmap[s_i], label='Null std ')
        plt.legend()
        plt.xlabel('Neuron Count Cutoff')
        plt.ylabel('Number of Bouts/Second')
        plt.title('Number of Bouts/Second by Cutoff')
        plt.tight_layout()
        im_name = 'norm_bouts_by_cutoff' + \
            '_' + ('_').join(seg_name.split(' '))
        plt.savefig(save_dir + im_name + '.png')
        plt.savefig(save_dir + im_name + '.svg')
        plt.close()
    del s_i, true_segment_bouts, true_segment_bout_counts, \
        null_segment_bout_counts, n_n, key, im_name, existing_vals, new_vals, \
        null_segment_bout_counts_mean, null_segment_bout_counts_std


def seg_compare_null():
    """This function calculates whether given segments of the experiment are different
    from each other, by comparing each segment to a null distrbution which shuffles
    time bins from all given segments randomly"""
    print("Do something")


def null_shuffle():
    """This function creates a shuffled dataset to analyze"""
    print("Do something")

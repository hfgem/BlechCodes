#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:20:24 2024

@author: Hannah Germaine

This is the fifth step of the analysis pipeline: statistics of the recording
are compared against null distributions.
"""

import functions.null_distributions as nd
import os
import tqdm
import gzip
import itertools
import json
import scipy.stats as stats
from multiprocessing import Pool
import numpy as np

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)


class run_data_null_analysis():

    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.gather_variables()
        self.compare_true_null()
        self.plot_true_null()

    def gather_variables(self,):
        # Directories
        self.bin_dir = self.metadata['dir_name'] + 'thresholded_statistics/'
        if os.path.isdir(self.bin_dir) == False:
            os.mkdir(self.bin_dir)
        self.null_dir = self.metadata['dir_name'] + 'null_data/'
        if os.path.isdir(self.null_dir) == False:
            os.mkdir(self.null_dir)
        # Variables
        self.num_neur = self.data_dict['num_neur']
        self.segment_names = self.data_dict['segment_names']
        self.num_segments = len(self.segment_names)
        self.pre_taste = self.metadata['params_dict']['pre_taste']
        self.post_taste = self.metadata['params_dict']['post_taste']
        self.segments_to_analyze = self.metadata['params_dict']['segments_to_analyze']
        self.segment_spike_times = self.data_dict['segment_spike_times']
        self.num_null = self.metadata['params_dict']['num_null']
        self.count_cutoff = np.arange(1, self.num_neur)
        self.bin_size = self.metadata['params_dict']['compare_null_params']['bin_size']
        lag_min = self.metadata['params_dict']['compare_null_params']['lag_min']
        lag_max = self.metadata['params_dict']['compare_null_params']['lag_max']
        self.lag_vals = np.arange(lag_min, lag_max).astype('int')
        self.segment_times = self.data_dict['segment_times']

    def compare_true_null(self,):
        try:  # Import complete calculated dictionaries if they exist
            filepath = self.bin_dir + 'neur_count_dict.npy'
            neur_count_dict = np.load(filepath, allow_pickle=True).item()
            filepath = self.bin_dir + 'neur_spike_dict.npy'
            neur_spike_dict = np.load(filepath, allow_pickle=True).item()
            for s_i in self.segments_to_analyze:
                seg_name = self.segment_names[s_i]
                neur_true_count_data = self.neur_count_dict[seg_name + '_true']
                neur_null_count_data = self.neur_count_dict[seg_name + '_null']
                percentile_count_data = self.neur_count_dict[seg_name + '_percentile']
                neur_true_spike_data = self.neur_spike_dict[seg_name + '_true']
                neur_null_spike_data = self.neur_spike_dict[seg_name + '_null']
                percentile_spike_data = self.neur_spike_dict[seg_name + '_percentile']
            del seg_name, neur_true_count_data, neur_null_count_data, percentile_count_data, neur_true_spike_data, neur_null_spike_data, percentile_spike_data
            print('\tImported complete thresholded datasets into memory')
        except:  # Calculate dictionaries
            print("\tImporting/Calculating Null and True Data Stats")
            try:
                filepath = self.bin_dir + 'neur_count_dict.npy'
                neur_count_dict = np.load(filepath, allow_pickle=True).item()
            except:
                neur_count_dict = dict()
            try:
                filepath = self.bin_dir + 'neur_spike_dict.npy'
                neur_spike_dict = np.load(filepath, allow_pickle=True).item()
            except:
                neur_spike_dict = dict()
            for s_i in tqdm.tqdm(self.segments_to_analyze):
                seg_name = self.segment_names[s_i]
                try:
	                neur_true_spike_data = self.neur_spike_dict[seg_name + '_true']
	                neur_null_spike_data = self.neur_spike_dict[seg_name + '_null']
	                percentile_spike_data = self.neur_spike_dict[seg_name + '_percentile']
	                neur_true_count_data = self.neur_count_dict[seg_name + '_true']
	                neur_null_count_data = self.neur_count_dict[seg_name + '_null']
	                percentile_count_data = self.neur_count_dict[seg_name + '_percentile']
                except:
					# Gather data / parameters
                    print(
                        '\t\tNow Calculating Null and True Statistics for Segment ' + seg_name)
                    segment_spikes = self.segment_spike_times[s_i]
                    segment_start_time = self.segment_times[s_i]
                    segment_end_time = self.segment_times[s_i+1]
                    # Segment save dir
                    seg_null_dir = self.null_dir + \
                        self.segment_names[s_i] + '/'
                    if os.path.isdir(seg_null_dir) == False:
                        os.mkdir(seg_null_dir)
                    # Get null spike data
                    null_bin_spikes = self.get_null_spikes(seg_null_dir, segment_spikes,
                                                           segment_start_time, segment_end_time)
                    # Get true spike data
                    true_bin_spikes = self.get_true_spikes(segment_spikes,
                                                           segment_end_time, segment_start_time)
                    # Get stats
                    true_neur_counts, true_spike_counts, null_neur_counts, null_spike_counts = self.get_stats(null_bin_spikes,
                                                                                                              true_bin_spikes, segment_start_time,
                                                                                                              segment_end_time)
                    # Store the neuron count data
                    true_x_vals, true_neur_count_array, null_x_vals, mean_null_neur_counts, std_null_neur_counts, percentiles = self.summarize_data(
                        true_neur_counts, null_neur_counts)
                    neur_count_dict[seg_name + '_true'] = [list(true_x_vals),
                                                           list(true_neur_count_array)]
                    neur_count_dict[seg_name + '_null'] = [list(null_x_vals),
                                                           list(
                                                               mean_null_neur_counts),
                                                           list(std_null_neur_counts)]
                    neur_count_dict[seg_name +
                                    '_percentile'] = [list(true_x_vals), percentiles]
                    # Store the neuron spike count data
                    true_x_vals, true_spike_count_array, null_x_vals, mean_null_spike_counts, std_null_spike_counts, percentiles = self.summarize_data(
                        true_spike_counts, null_spike_counts)
                    neur_spike_dict[seg_name + '_true'] = [list(true_x_vals),
                                                           list(true_spike_count_array)]
                    neur_spike_dict[seg_name + '_null'] = [list(null_x_vals),
                                                           list(
                                                               mean_null_spike_counts),
                                                           list(std_null_spike_counts)]
                    neur_spike_dict[seg_name +
                                    '_percentile'] = [list(true_x_vals), percentiles]
                    # Save the dictionaries in the current state
                    filepath = self.bin_dir + 'neur_count_dict.npy'
                    np.save(filepath, neur_count_dict)
                    self.neur_count_dict = neur_count_dict
                    filepath = self.bin_dir + 'neur_spike_dict.npy'
                    np.save(filepath, neur_spike_dict)
                    self.neur_spike_dict = neur_spike_dict
                    # Clear memory
                    del segment_spikes, segment_start_time, segment_end_time, seg_null_dir, null_bin_spikes, true_bin_spikes
                    del true_neur_counts, true_spike_counts, null_neur_counts, null_spike_counts
                    del true_x_vals, true_neur_count_array, null_x_vals, mean_null_neur_counts, std_null_neur_counts, percentiles
                    del true_spike_count_array, mean_null_spike_counts, std_null_spike_counts
            # Save the dictionaries in completion
            filepath = self.bin_dir + 'neur_count_dict.npy'
            np.save(filepath, neur_count_dict)
            self.neur_count_dict = neur_count_dict
            filepath = self.bin_dir + 'neur_spike_dict.npy'
            np.save(filepath, neur_spike_dict)
            self.neur_spike_dict = neur_spike_dict

    def get_null_spikes(self, seg_null_dir, segment_spikes,
                        segment_start_time, segment_end_time):
        """Import or generate the null datasets"""
        try:
            filepath = seg_null_dir + 'null_' + str(0) + '.json'
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)
            print('\tNow importing null dataset into memory')
            null_segment_spikes = []
            for n_i in tqdm.tqdm(range(self.num_null)):
                filepath = seg_null_dir + 'null_' + str(n_i) + '.json'
                with gzip.GzipFile(filepath, mode="r") as f:
                    json_bytes = f.read()
                    json_str = json_bytes.decode('utf-8')
                    data = json.loads(json_str)
                    null_segment_spikes.append(data)
        except:
            # First create a null distribution set
            print('\tBeginning null distribution creation')
            with Pool(processes=4) as pool:  # start 4 worker processes
                pool.map(nd.run_null_create_parallelized, zip(np.arange(self.num_null),
                                                              itertools.repeat(
                                                                  segment_spikes),
                                                              itertools.repeat(
                                                                  segment_start_time),
                                                              itertools.repeat(
                                                                  segment_end_time),
                                                              itertools.repeat(seg_null_dir)))
            null_segment_spikes = []
            print('\tNow importing null dataset into memory')
            for n_i in tqdm.tqdm(range(self.num_null)):
                filepath = seg_null_dir + 'null_' + str(n_i) + '.json'
                with gzip.GzipFile(filepath, mode="r") as f:
                    json_bytes = f.read()
                    json_str = json_bytes.decode('utf-8')
                    data = json.loads(json_str)
                    null_segment_spikes.append(data)
        # _____Convert null data to binary spike matrix_____
        null_bin_spikes = []
        for n_n in range(self.num_null):
            null_bin_spike = np.zeros(
                (self.num_neur, segment_end_time-segment_start_time+1))
            null_spikes = null_segment_spikes[n_n]
            for n_i in range(self.num_neur):
                spike_indices = (np.where((null_spikes[n_i] >= segment_start_time)*(
                    null_spikes[n_i] <= segment_end_time))[0]).astype('int')
                null_bin_spike[n_i, spike_indices] = 1
            null_bin_spikes.append(null_bin_spike)

        return null_bin_spikes

    def get_true_spikes(self, segment_spikes, segment_end_time, segment_start_time):
        bin_spike = np.zeros(
            (self.num_neur, segment_end_time-segment_start_time+1))
        for n_i in range(self.num_neur):
            spike_indices = (
                np.array(segment_spikes[n_i]) - segment_start_time).astype('int')
            bin_spike[n_i, spike_indices] = 1
        return bin_spike

    def get_stats(self, null_bin_spikes, true_bin_spikes, segment_start_time, segment_end_time):
        print('\tCalculating Neuron and Spike Statistics')
        true_neur_counts, true_spike_counts = nd.high_bins(
            [true_bin_spikes, segment_start_time, segment_end_time, self.bin_size, self.count_cutoff])
        null_neur_counts, null_spike_counts = self.get_null_stats(
            segment_start_time, segment_end_time, null_bin_spikes)

        return true_neur_counts, true_spike_counts, null_neur_counts, null_spike_counts

    def get_null_stats(self, segment_start_time, segment_end_time, null_bin_spikes):
        null_neur_counts = dict()
        null_spike_counts = dict()
        # Run nd.high_bins() to get null neuron counts and spike counts
        chunk_inds = np.linspace(0, self.num_null, 10).astype(
            'int')  # chunk it to keep memory usage lower
        results_counts = []
        print('\t\tCalculating bins for null distributions')
        for c_i in tqdm.tqdm(range(len(chunk_inds)-1)):
            null_bin_spike_chunk = null_bin_spikes[chunk_inds[c_i]                                                   :chunk_inds[c_i+1]]
            with Pool(processes=4) as pool:  # start 4 worker processes
                results_chunk_counts = pool.map(nd.high_bins, zip(null_bin_spike_chunk,
                                                                  itertools.repeat(
                                                                      segment_start_time),
                                                                  itertools.repeat(
                                                                      segment_end_time),
                                                                  itertools.repeat(
                                                                      self.bin_size),
                                                                  itertools.repeat(self.count_cutoff)))
                results_counts.extend(results_chunk_counts)
        for n_n in range(self.num_null):
            null_neur_count = results_counts[n_n][0]
            null_spike_count = results_counts[n_n][1]
            for key in null_neur_count.keys():
                if key in null_neur_counts.keys():
                    null_neur_counts[key].append(null_neur_count[key])
                else:
                    null_neur_counts[key] = [null_neur_count[key]]
            for key in null_spike_count.keys():
                if key in null_spike_counts.keys():
                    null_spike_counts[key].append(null_spike_count[key])
                else:
                    null_spike_counts[key] = [null_spike_count[key]]
        return null_neur_counts, null_spike_counts

    def summarize_data(self, true_counts, null_counts):
        # Neuron count
        true_x_vals = np.array([(np.ceil(float(key))).astype('int')
                               for key in true_counts.keys()])
        true_count_array = np.array([true_counts[key]
                                    for key in true_counts.keys()])
        null_x_vals = np.array([(np.ceil(float(key))).astype('int')
                               for key in null_counts.keys()])
        mean_null_counts = np.array(
            [np.mean(null_counts[key]) for key in null_counts.keys()])
        std_null_counts = np.array([np.std(null_counts[key])
                                   for key in null_counts.keys()])
        percentiles = []  # Calculate percentile of true data point in null data distribution
        for key in true_counts.keys():
            try:
                percentiles.extend(
                    [round(stats.percentileofscore(null_counts[key], true_counts[key]), 2)])
            except:
                percentiles.extend([100])

        return true_x_vals, true_count_array, null_x_vals, mean_null_counts, std_null_counts, percentiles

    def plot_true_null(self,):
        # _____Plotting_____
        neur_true_count_x = []
        neur_true_count_vals = []
        neur_null_count_x = []
        neur_null_count_mean = []
        neur_null_count_std = []
        neur_true_spike_x = []
        neur_true_spike_vals = []
        neur_null_spike_x = []
        neur_null_spike_mean = []
        neur_null_spike_std = []
        neur_true_rate_x = []
        neur_true_rate_vals = []
        neur_null_rate_x = []
        neur_null_rate_mean = []
        neur_null_rate_std = []
        for s_i in tqdm.tqdm(self.segments_to_analyze):
            seg_name = self.segment_names[s_i]
            segment_start_time = self.segment_times[s_i]
            segment_end_time = self.segment_times[s_i+1]
            segment_length = segment_end_time - segment_start_time
            neur_true_count_data = self.neur_count_dict[seg_name + '_true']
            neur_null_count_data = self.neur_count_dict[seg_name + '_null']
            percentile_count_data = self.neur_count_dict[seg_name + '_percentile']
            neur_true_spike_data = self.neur_spike_dict[seg_name + '_true']
            neur_null_spike_data = self.neur_spike_dict[seg_name + '_null']
            percentile_spike_data = self.neur_spike_dict[seg_name + '_percentile']
            # Plot the neuron count data
            # Normalizing the number of bins to number of bins / second
            norm_val = segment_length/1000
            nd.plot_indiv_truexnull(np.array(neur_true_count_data[0]), np.array(neur_null_count_data[0]), np.array(neur_true_count_data[1]), np.array(neur_null_count_data[1]),
                                    np.array(neur_null_count_data[2]), segment_length, norm_val, self.bin_dir, 'Neuron Counts', seg_name, np.array(percentile_count_data[1]))
            neur_true_count_x.append(np.array(neur_true_count_data[0]))
            neur_null_count_x.append(np.array(neur_null_count_data[0]))
            neur_true_count_vals.append(np.array(neur_true_count_data[1]))
            neur_null_count_mean.append(np.array(neur_null_count_data[1]))
            neur_null_count_std.append(np.array(neur_null_count_data[2]))
            # Plot the spike count data
            nd.plot_indiv_truexnull(np.array(neur_true_spike_data[0]), np.array(neur_null_spike_data[0]), np.array(neur_true_spike_data[1]), np.array(neur_null_spike_data[1]),
                                    np.array(neur_null_spike_data[2]), np.array(segment_length), norm_val, self.bin_dir, 'Spike Counts', seg_name, np.array(percentile_spike_data[1]))
            neur_true_spike_x.append(np.array(neur_true_spike_data[0]))
            neur_null_spike_x.append(np.array(neur_null_spike_data[0]))
            neur_true_spike_vals.append(np.array(neur_true_spike_data[1]))
            neur_null_spike_mean.append(np.array(neur_null_spike_data[1]))
            neur_null_spike_std.append(np.array(neur_null_spike_data[2]))
            # Store the bouts/second data
            neur_true_rate_x.append(np.array(neur_true_count_data[0])/norm_val)
            neur_null_rate_x.append(np.array(neur_null_count_data[0])/norm_val)
            neur_true_rate_vals.append(
                np.array(neur_true_count_data[1])/norm_val)
            neur_null_rate_mean.append(
                np.array(neur_null_count_data[1])/norm_val)
            neur_null_rate_std.append(
                np.array(neur_null_count_data[2])/norm_val)
        # Plot all neuron count data
        nd.plot_all_truexnull(neur_true_count_x, neur_null_count_x, neur_true_count_vals, neur_null_count_mean,
                              neur_null_count_std, norm_val, self.bin_dir, 'Neuron Counts', list(np.array(self.segment_names)[self.segments_to_analyze]))
        # Plot all spike count data
        nd.plot_all_truexnull(neur_true_spike_x, neur_null_spike_x, neur_true_spike_vals, neur_null_spike_mean,
                              neur_null_spike_std, norm_val, self.bin_dir, 'Spike Counts', list(np.array(self.segment_names)[self.segments_to_analyze]))

        # Plot all bouts/second data
        nd.plot_all_truexnull(neur_true_rate_x, neur_null_rate_x, neur_true_rate_vals, neur_null_rate_mean,
                              neur_null_rate_std, norm_val, self.bin_dir, 'Bouts per Second', list(np.array(self.segment_names)[self.segments_to_analyze]))

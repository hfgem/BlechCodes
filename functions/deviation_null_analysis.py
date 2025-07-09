#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:28:37 2024

@author: Hannah Germaine

This is the fourth step of the analysis pipeline: deviation events are compared
to null shuffled distributions deviation events.
"""

import os
import sys
import warnings
import tqdm
import gzip
import itertools
import json
from multiprocessing import Pool
import numpy as np
import scipy.stats as stats

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.null_distributions as nd
import functions.dev_plot_funcs as dpf
import functions.dev_funcs as df
import functions.dev_null_cutoff_dicts as dncd

warnings.filterwarnings("ignore")


class run_deviation_null_analysis():

    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.gather_variables()
        self.import_deviations()
        self.gen_null_distributions()
        self.convert_to_rasters()
        self.calc_statistics()
        self.plot_statistics()

    def gather_variables(self,):
        self.num_neur = self.data_dict['num_neur']
        self.pre_taste = self.metadata['params_dict']['pre_taste']
        self.post_taste = self.metadata['params_dict']['post_taste']
        self.segments_to_analyze = self.metadata['params_dict']['segments_to_analyze']
        self.segment_spike_times = self.data_dict['segment_spike_times']
        self.segment_names = self.data_dict['segment_names']
        self.num_segments = len(self.segment_names)
        self.segment_times = self.data_dict['segment_times']
        self.segment_times_reshaped = [
            [self.segment_times[i], self.segment_times[i+1]] for i in range(self.num_segments)]
        self.local_size = self.metadata['params_dict']['local_size']
        self.min_dev_size = self.metadata['params_dict']['min_dev_size']
        self.num_null = self.metadata['params_dict']['num_null']
        self.max_plot = self.metadata['params_dict']['max_plot']
        self.count_cutoff = np.arange(1, self.num_neur)
        self.bin_size = self.metadata['params_dict']['compare_null_params']['bin_size']
        self.dev_dir = self.metadata['dir_name'] + 'Deviations/'
        self.null_dir = self.metadata['dir_name'] + 'null_data/'
        if os.path.isdir(self.null_dir) == False:
            os.mkdir(self.null_dir)
        self.bin_dir = self.dev_dir + 'null_x_true_deviations/'
        if os.path.isdir(self.bin_dir) == False:
            os.mkdir(self.bin_dir)

    def import_deviations(self,):
        try:  # test if the data exists by trying to import the last
            filepath = self.dev_dir + \
                self.segment_names[self.segments_to_analyze[-1]
                                   ] + '/deviations.json'
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)

            print("\tNow importing calculated deviations")
            segment_deviations = []
            for s_i in tqdm.tqdm(self.segments_to_analyze):
                filepath = self.dev_dir + \
                    self.segment_names[s_i] + '/deviations.json'
                with gzip.GzipFile(filepath, mode="r") as f:
                    json_bytes = f.read()
                    json_str = json_bytes.decode('utf-8')
                    data = json.loads(json_str)
                    segment_deviations.append(data)
            self.segment_deviations = segment_deviations
        except:
            print("ERROR! ERROR! ERROR!")
            print("Deviations were not calculated previously as expected.")
            print("Something went wrong in the analysis pipeline.")
            print("Please try reverting your analysis_state_tracker.csv to 1 to rerun.")
            print("If issues persist, contact Hannah.")
            sys.exit()

    def gen_null_distributions(self,):
        # _____Generate null datasets_____
        for s_i in self.segments_to_analyze:
            seg_null_dir = self.null_dir + self.segment_names[s_i] + '/'
            if os.path.isdir(seg_null_dir) == False:
                os.mkdir(seg_null_dir)
            try:
                filepath = seg_null_dir + 'null_' + \
                    str(self.num_null-1) + '.json'
                with gzip.GzipFile(filepath, mode="r") as f:
                    json_bytes = f.read()
                    json_str = json_bytes.decode('utf-8')
                    null_segment_spike_times = json.loads(json_str)
                print('\t' + self.segment_names[s_i] +
                      ' null distributions previously created')
            except:
                # First create a null distribution set
                print('\tNow creating ' +
                      self.segment_names[s_i] + ' null distributions')
                with Pool(processes=4) as pool:  # start 4 worker processes
                    pool.map(nd.run_null_create_parallelized, zip(np.arange(self.num_null),
                                                                  itertools.repeat(
                                                                      self.segment_spike_times[s_i]),
                                                                  itertools.repeat(
                                                                      self.segment_times[0]),
                                                                  itertools.repeat(
                                                                      self.segment_times[-1]),
                                                                  itertools.repeat(seg_null_dir)))
                pool.close()
        print('\tCalculating null distribution spike times')
        # _____Grab null dataset spike times_____
        all_null_segment_spike_times = []
        for null_i in range(self.num_null):
            null_segment_spike_times = []
            
            for s_i in self.segments_to_analyze:
                seg_null_dir = self.null_dir + self.segment_names[s_i] + '/'
                # Import the null distribution into memory
                filepath = seg_null_dir + 'null_' + str(null_i) + '.json'
                with gzip.GzipFile(filepath, mode="r") as f:
                    json_bytes = f.read()
                    json_str = json_bytes.decode('utf-8')
                    data = json.loads(json_str)
    
                seg_null_dir = self.null_dir + self.segment_names[s_i] + '/'

                seg_start = self.segment_times_reshaped[s_i][0]
                seg_end = self.segment_times_reshaped[s_i][1]
                null_seg_st = []
                for n_i in range(self.num_neur):
                    seg_spike_inds = np.where(
                        (data[n_i] >= seg_start)*(data[n_i] <= seg_end))[0]
                    null_seg_st.append(
                        list(np.array(data[n_i])[seg_spike_inds]))
                null_segment_spike_times.append(null_seg_st)
            all_null_segment_spike_times.append(null_segment_spike_times)
        self.all_null_segment_spike_times = all_null_segment_spike_times

        # _____Import or calculate null deviations for all segments_____
        try:  # test if the data exists by trying to import the last
            filepath = self.dev_dir + 'null_data/' + \
                self.segment_names[self.segments_to_analyze[-1]] + \
                '/null_'+str(self.num_null - 1)+'_deviations.json'
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)
        except:
            print("\nNow calculating deviations")
            for null_i in range(self.num_null):
                try:  # Not to have to restart if deviation calculation was interrupted partway
                    last_seg = self.segment_names[self.segments_to_analyze[-1]]
                    filepath = self.dev_dir + 'null_data/' + last_seg + \
                        '/null_'+str(null_i)+'_deviations.json'
                    with gzip.GzipFile(filepath, mode="r") as f:
                        json_bytes = f.read()
                        json_str = json_bytes.decode('utf-8')
                        data = json.loads(json_str)
                    print("\t\tNull " + str(null_i) +
                          " Deviations Previously Calculated.")
                    # Puts the onus on the user to delete the null deviations if they want them completely recalculated
                except:
                    print("\tCreating Null " + str(null_i))
                    seg_dirs = []
                    # Import the null distribution into memory
                    filepath = seg_null_dir + 'null_' + str(null_i) + '.json'
                    with gzip.GzipFile(filepath, mode="r") as f:
                        json_bytes = f.read()
                        json_str = json_bytes.decode('utf-8')
                        data = json.loads(json_str)

                    for s_i in self.segments_to_analyze:
                        # create storage directory for null deviation data
                        if os.path.isdir(self.dev_dir + 'null_data/') == False:
                            os.mkdir(self.dev_dir + 'null_data/')
                        seg_dir = self.dev_dir + 'null_data/' + \
                            self.segment_names[s_i] + '/'
                        if os.path.isdir(seg_dir) == False:
                            os.mkdir(seg_dir)
                        seg_dir = self.dev_dir + 'null_data/' + \
                            self.segment_names[s_i] + \
                            '/null_' + str(null_i) + '_'
                        seg_dirs.append(seg_dir)

                    null_segment_spike_times = all_null_segment_spike_times[null_i]
                    segment_times_reshaped = [
                        self.segment_times_reshaped[i] for i in self.segments_to_analyze]
                    with Pool(processes=4) as pool:  # start 4 worker processes
                        pool.map(df.run_dev_pull_parallelized, zip(null_segment_spike_times,
                                                                   itertools.repeat(
                                                                       self.local_size),
                                                                   itertools.repeat(
                                                                       self.min_dev_size),
                                                                   segment_times_reshaped,
                                                                   seg_dirs))
                    pool.close()

        print("\tNow importing calculated null deviations")
        all_null_deviations = []
        for null_i in tqdm.tqdm(range(self.num_null)):
            null_segment_deviations = []
            for s_i in self.segments_to_analyze:
                filepath = self.dev_dir + 'null_data/' + \
                    self.segment_names[s_i] + '/null_' + \
                    str(null_i) + '_deviations.json'
                with gzip.GzipFile(filepath, mode="r") as f:
                    json_bytes = f.read()
                    json_str = json_bytes.decode('utf-8')
                    data = json.loads(json_str)
                    null_segment_deviations.append(data)
            all_null_deviations.append(null_segment_deviations)
        del null_i, null_segment_deviations, s_i, filepath, json_bytes, json_str, data

        self.all_null_deviations = all_null_deviations

    def convert_to_rasters(self,):
        # Calculate segment deviation spikes
        print("\tNow pulling true deviation rasters")
        num_seg = len(self.segments_to_analyze)
        seg_spike_times = [self.segment_spike_times[i]
                           for i in self.segments_to_analyze]
        seg_times_reshaped = np.array(self.segment_times_reshaped)[
            self.segments_to_analyze, :]
        z_bin = self.metadata['params_dict']['z_bin']
        segment_dev_rasters, segment_dev_times, _, _, _, _ = df.create_dev_rasters(num_seg,
                                                                             seg_spike_times,
                                                                             seg_times_reshaped,
                                                                             self.segment_deviations, z_bin)
        self.segment_dev_rasters = segment_dev_rasters
        self.segment_dev_times = segment_dev_times

        # Calculate segment deviation spikes
        print("\tNow pulling null deviation rasters")
        null_dev_rasters = []
        null_dev_times = []
        for null_i in tqdm.tqdm(range(self.num_null)):
            null_segment_deviations = self.all_null_deviations[null_i]
            null_segment_spike_times = self.all_null_segment_spike_times[null_i]
            null_segment_dev_rasters_i, null_segment_dev_times_i, _, _, _, _ = df.create_dev_rasters(num_seg,
                                                                                               null_segment_spike_times,
                                                                                               seg_times_reshaped,
                                                                                               null_segment_deviations, z_bin)
            null_dev_rasters.append(null_segment_dev_rasters_i)
            null_dev_times.append(null_segment_dev_times_i)

        self.__dict__.pop('all_null_deviations', None)
        self.null_dev_rasters = null_dev_rasters
        self.null_dev_times = null_dev_times

    def calc_statistics(self,):
        try:  # Import calculated dictionaries if they exist
            neur_count_dict = np.load(os.path.join(self.bin_dir,'neur_count_dict.npy'), allow_pickle=True).item()
            neur_spike_dict = np.load(os.path.join(self.bin_dir,'neur_spike_dict.npy'), allow_pickle=True).item()
            neur_spike_dict = np.load(os.path.join(self.bin_dir,'neur_len_dict.npy'), allow_pickle=True).item()
            neur_fr_dict =np.load(os.path.join(self.bin_dir,'neur_fr_dict.npy'), allow_pickle=True).item()
            print('\tTruexNull deviation datasets previously calculated.')
        except:  # Calculate dictionaries
            print('\tCalculating Deviation Statistics')
            neur_count_dict = dict()
            neur_spike_dict = dict()
            neur_fr_dict = dict()
            neur_len_dict = dict()
            for s_ind, s_i in tqdm.tqdm(enumerate(self.segments_to_analyze)):
                # Gather data / parameters
                seg_name = self.segment_names[s_i]
                # Create segment save dir for figures
                seg_fig_save_dir = os.path.join(self.bin_dir, seg_name)
                if not os.path.isdir(seg_fig_save_dir):
                    os.mkdir(seg_fig_save_dir)
                # _____Gather null data deviation event stats_____
                null_dev_lengths = []
                null_dev_neuron_counts = []
                null_dev_spike_counts = []
                null_dev_frs = []
                for null_i in range(self.num_null):
                    all_rast = self.null_dev_rasters[null_i][s_ind]
                    null_i_num_neur, null_i_num_spikes, all_len = df.calculate_dev_null_stats(
                        all_rast, self.null_dev_times[null_i][s_ind])
                    null_dev_neuron_counts.append(null_i_num_neur)
                    null_dev_spike_counts.append(null_i_num_spikes)
                    null_dev_lengths.append(all_len)
                    null_dev_frs.append(np.array(null_i_num_spikes)/(all_len/1000)/self.num_neur)
                # _____Gather true data deviation event stats_____
                all_rast = self.segment_dev_rasters[s_ind]
                true_dev_neuron_counts, true_dev_spike_counts, true_dev_lengths = df.calculate_dev_null_stats(
                    all_rast, self.segment_dev_times[s_ind])
                true_dev_frs = np.array(true_dev_spike_counts)/(true_dev_lengths/1000)/self.num_neur
                # _____Gather data as dictionary of number of events as a function of cutoff
                neur_count_dict, neur_spike_dict, neur_fr_dict, neur_len_dict = \
                    dncd.dev_null_dict_creation(self.num_null, null_dev_lengths, null_dev_neuron_counts,
                                               null_dev_spike_counts, null_dev_frs, true_dev_neuron_counts,
                                               true_dev_spike_counts, true_dev_lengths, true_dev_frs,
                                               neur_count_dict, neur_spike_dict, neur_fr_dict,
                                               neur_len_dict, seg_name, self.min_dev_size, seg_fig_save_dir)

            # Save the dictionaries
            np.save(os.path.join(self.bin_dir,'neur_count_dict.npy'), neur_count_dict)
            np.save(os.path.join(self.bin_dir,'neur_spike_dict.npy'), neur_spike_dict)
            np.save(os.path.join(self.bin_dir,'neur_len_dict.npy'), neur_len_dict)
            np.save(os.path.join(self.bin_dir,'neur_fr_dict.npy'), neur_fr_dict)

    def plot_statistics(self,):
        print('\tPlotting deviation statistics datasets')
        filepath = self.bin_dir + 'neur_count_dict.npy'
        neur_count_dict = np.load(filepath, allow_pickle=True).item()
        filepath = self.bin_dir + 'neur_spike_dict.npy'
        neur_spike_dict = np.load(filepath, allow_pickle=True).item()
        filepath = self.bin_dir + 'neur_len_dict.npy'
        neur_len_dict = np.load(filepath, allow_pickle=True).item()

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
        neur_true_len_x = []
        neur_true_len_vals = []
        neur_null_len_x = []
        neur_null_len_mean = []
        neur_null_len_std = []
        for s_ind, s_i in tqdm.tqdm(enumerate(self.segments_to_analyze)):
            seg_name = self.segment_names[s_i]
            segment_start_time = self.segment_times[s_i]
            segment_end_time = self.segment_times[s_i+1]
            segment_length = segment_end_time - segment_start_time
            neur_true_count_data = neur_count_dict[seg_name + '_true']
            neur_null_count_data = neur_count_dict[seg_name + '_null']
            percentile_count_data = neur_count_dict[seg_name + '_percentile']
            neur_true_spike_data = neur_spike_dict[seg_name + '_true']
            neur_null_spike_data = neur_spike_dict[seg_name + '_null']
            percentile_spike_data = neur_spike_dict[seg_name + '_percentile']
            neur_true_len_data = neur_len_dict[seg_name + '_true']
            neur_null_len_data = neur_len_dict[seg_name + '_null']
            percentile_len_data = neur_len_dict[seg_name + '_percentile']
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
            # Plot the length data
            nd.plot_indiv_truexnull(np.array(neur_true_len_data[0]), np.array(neur_null_len_data[0]), np.array(neur_true_len_data[1]), np.array(neur_null_len_data[1]),
                                    np.array(neur_null_len_data[2]), np.array(segment_length), norm_val, self.bin_dir, 'Lengths', seg_name, np.array(percentile_len_data[1]))
            neur_true_len_x.append(np.array(neur_true_len_data[0]))
            neur_null_len_x.append(np.array(neur_null_len_data[0]))
            neur_true_len_vals.append(np.array(neur_true_len_data[1]))
            neur_null_len_mean.append(np.array(neur_null_len_data[1]))
            neur_null_len_std.append(np.array(neur_null_len_data[2]))
        # Plot all neuron count data
        nd.plot_all_truexnull(neur_true_count_x, neur_null_count_x, neur_true_count_vals, neur_null_count_mean,
                              neur_null_count_std, norm_val, self.bin_dir, 'Neuron Counts', list(np.array(self.segment_names)[self.segments_to_analyze]))
        # Plot all spike count data
        nd.plot_all_truexnull(neur_true_spike_x, neur_null_spike_x, neur_true_spike_vals, neur_null_spike_mean,
                              neur_null_spike_std, norm_val, self.bin_dir, 'Spike Counts', list(np.array(self.segment_names)[self.segments_to_analyze]))

        # Plot all length data
        nd.plot_all_truexnull(neur_true_len_x, neur_null_len_x, neur_true_len_vals, neur_null_len_mean,
                              neur_null_len_std, norm_val, self.bin_dir, 'Lengths', list(np.array(self.segment_names)[self.segments_to_analyze]))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:55:12 2024

@author: Hannah Germaine

In this step of the analysis pipeline, a Bayesian decoder is trained on taste
responses and then run on the deviation events calculated earlier in the 
pipeline.
"""

import os
import tqdm
import gzip
import json
import numpy as np

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.decoding_funcs as df
import functions.dependent_decoding_funcs as ddf
import functions.dev_funcs as dev_f
import functions.hdf5_handling as hf5

class run_deviation_dependent_bayes():

    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.gather_variables()
        self.import_deviations()
        self.pull_fr_dist()
        self.decode_all_neurons()

    def gather_variables(self,):
        # Directories
        self.hdf5_dir = self.metadata['hdf5_dir']
        self.bayes_dir = self.metadata['dir_name'] + \
            'Burst_Dependent_Decoding/'
        if os.path.isdir(self.bayes_dir) == False:
            os.mkdir(self.bayes_dir)
        self.dev_dir = self.metadata['dir_name'] + 'Deviations/'
        # General Params/Variables
        self.num_neur = self.data_dict['num_neur']
        self.pre_taste = self.metadata['params_dict']['pre_taste']
        self.post_taste = self.metadata['params_dict']['post_taste']
        self.pre_taste_dt = np.ceil(self.pre_taste*1000).astype('int')
        self.post_taste_dt = np.ceil(self.post_taste*1000).astype('int')
        self.segments_to_analyze = self.metadata['params_dict']['segments_to_analyze']
        self.epochs_to_analyze = self.metadata['params_dict']['epochs_to_analyze']
        self.segment_names = self.data_dict['segment_names']
        self.num_segments = len(self.segment_names)
        self.segment_spike_times = self.data_dict['segment_spike_times']
        self.segment_times = self.data_dict['segment_times']
        self.segment_times_reshaped = [
            [self.segment_times[i], self.segment_times[i+1]] for i in range(self.num_segments)]
        # Remember this imported value is 1 less than the number of epochs
        self.num_cp = self.metadata['params_dict']['num_cp'] + 1
        self.tastant_spike_times = self.data_dict['tastant_spike_times']
        self.start_dig_in_times = self.data_dict['start_dig_in_times']
        self.end_dig_in_times = self.data_dict['end_dig_in_times']
        self.dig_in_names = self.data_dict['dig_in_names']
        self.num_tastes = len(self.dig_in_names)
        self.min_dev_size = self.metadata['params_dict']['min_dev_size']
        # Decoding Params/Variables
        self.e_skip_time = self.metadata['params_dict']['bayes_params']['e_skip_time']
        self.e_len_time = self.metadata['params_dict']['bayes_params']['e_len_time']
        self.e_skip_dt = np.ceil(self.e_skip_time*1000).astype('int')
        self.e_len_dt = np.ceil(self.e_len_time*1000).astype('int')
        self.neuron_count_thresh = self.metadata['params_dict']['bayes_params']['neuron_count_thresh']
        self.max_decode = self.metadata['params_dict']['bayes_params']['max_decode']
        self.seg_stat_bin = self.metadata['params_dict']['bayes_params']['seg_stat_bin']
        self.trial_start_frac = self.metadata['params_dict']['bayes_params']['trial_start_frac']
        self.decode_prob_cutoff = self.metadata['params_dict']['bayes_params']['decode_prob_cutoff']
        self.bin_time = self.metadata['params_dict']['bayes_params']['bin_time']
        self.bin_dt = np.ceil(self.bin_time*1000).astype('int')
        # Import changepoint data
        pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
            self.hdf5_dir, 'changepoint_data', 'pop_taste_cp_raster_inds')
        self.pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
        num_pt_cp = self.num_cp + 2
        # Import taste selectivity data
        try:
            select_neur = hf5.pull_data_from_hdf5(
                self.hdf5_dir, 'taste_selectivity', 'taste_select_neur_epoch_bin')[0]
            self.select_neur = select_neur
        except:
            print("\tNo taste selectivity data found. Skipping.")
        # Import discriminability data
        peak_epochs = np.squeeze(hf5.pull_data_from_hdf5(
            self.hdf5_dir, 'taste_discriminability', 'peak_epochs'))
        discrim_neur = np.squeeze(hf5.pull_data_from_hdf5(
            self.hdf5_dir, 'taste_discriminability', 'discrim_neur'))
        # Convert discriminatory neuron changepoint data into pop_taste_cp_raster_inds shape
        # TODO: Add a flag for a user to select whether to use discriminatory neurons or selective neurons
        num_discrim_cp = np.shape(peak_epochs)[0]
        discrim_cp_raster_inds = []
        for t_i in range(len(self.dig_in_names)):
            t_cp_vec = np.ones(
                (np.shape(pop_taste_cp_raster_inds[t_i])[0], num_discrim_cp+1))
            t_cp_vec = (peak_epochs[:num_pt_cp] +
                        int(self.pre_taste*1000))*t_cp_vec
            discrim_cp_raster_inds.append(t_cp_vec)
        self.discrim_cp_raster_inds = discrim_cp_raster_inds
        self.discrim_neur = discrim_neur

    def import_deviations(self,):
        print("\tNow importing calculated deviations")
        segment_deviations = []
        for s_i in tqdm.tqdm(range(self.num_segments)):
            filepath = self.dev_dir + \
                self.segment_names[s_i] + '/deviations.json'
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)
                segment_deviations.append(data)

        print("\tNow pulling true deviation rasters")
        segment_dev_rasters, segment_dev_times, _ = dev_f.create_dev_rasters(self.num_segments, self.segment_spike_times,
                                                                             np.array(self.segment_times_reshaped), self.segment_deviations, self.pre_taste)

        self.segment_dev_rasters = segment_dev_rasters
        self.segment_dev_times = segment_dev_times

    def pull_fr_dist(self,):
        print("\tPulling FR Distributions")
        tastant_fr_dist_pop, taste_num_deliv, max_hz_pop = ddf.taste_fr_dist(self.num_neur,
                                                                             self.num_cp, self.tastant_spike_times,
                                                                             self.pop_taste_cp_raster_inds,
                                                                             self.start_dig_in_times, self.pre_taste_dt,
                                                                             self.post_taste_dt, self.trial_start_frac)
        self.tastant_fr_dist_pop = tastant_fr_dist_pop
        self.taste_num_deliv = taste_num_deliv
        self.max_hz_pop = max_hz_pop
        tastant_fr_dist_z_pop, _, max_hz_z_pop, min_hz_z_pop = ddf.taste_fr_dist_zscore(self.num_neur,
                                                                                        self.num_cp, self.tastant_spike_times,
                                                                                        self.segment_spike_times, self.segment_names,
                                                                                        self.segment_times, self.pop_taste_cp_raster_inds,
                                                                                        self.start_dig_in_times, self.pre_taste_dt,
                                                                                        self.post_taste_dt, self.bin_dt, self.trial_start_frac)
        self.tastant_fr_dist_z_pop = tastant_fr_dist_z_pop
        self.max_hz_z_pop = max_hz_z_pop
        self.min_hz_z_pop = min_hz_z_pop

    def decode_all_neurons(self,):
        print("\tDecoding all neurons")
        decode_dir = self.bayes_dir + 'All_Neurons/'
        if os.path.isdir(decode_dir) == False:
            os.mkdir(decode_dir)
        self.decode_dir = decode_dir

        taste_select_neur = np.ones(np.shape(self.discrim_cp_raster_inds))
        self.taste_select_neur = taste_select_neur

        ddf.decode_epochs(self.tastant_fr_dist_pop, self.segment_spike_times,
                          self.post_taste_dt, self.e_skip_dt, self.e_len_dt,
                          self.dig_in_names, self.segment_times, self.segment_names,
                          self.start_dig_in_times, self.taste_num_deliv, self.select_neur,
                          self.max_hz_pop, decode_dir, self.neuron_count_thresh,
                          self.trial_start_frac, self.epochs_to_analyze, self.segments_to_analyze)

        self.plot_decoded_data()

    def decode_taste_selective_neurons(self,):
        print("\tDecoding taste selective neurons")
        decode_dir = self.bayes_dir + 'Taste_Selective/'
        if os.path.isdir(decode_dir) == False:
            os.mkdir(decode_dir)
        self.decode_dir = decode_dir

        taste_select_neur = self.discrim_neur
        self.taste_select_neur = taste_select_neur

        ddf.decode_epochs(self.tastant_fr_dist_pop, self.segment_spike_times,
                          self.post_taste_dt, self.e_skip_dt, self.e_len_dt,
                          self.dig_in_names, self.segment_times, self.segment_names,
                          self.start_dig_in_times, self.taste_num_deliv, self.select_neur,
                          self.max_hz_pop, decode_dir, self.neuron_count_thresh,
                          self.trial_start_frac, self.epochs_to_analyze, self.segments_to_analyze)

    def plot_decoded_data(self,):
        print("\t\tPlotting Decoded Results")
        df.plot_decoded(self.tastant_fr_dist_pop, self.num_tastes, self.num_neur,
                        self.num_cp, self.segment_spike_times, self.tastant_spike_times,
                        self.start_dig_in_times, self.end_dig_in_times, self.post_taste_dt,
                        self.pre_taste_dt, self.discrim_cp_raster_inds, self.dig_in_names,
                        self.segment_times, self.segment_names, self.taste_num_deliv,
                        self.taste_select_neur, self.decode_dir, self.max_decode, self.max_hz_pop,
                        self.seg_stat_bin, self.neuron_count_thresh, self.trial_start_frac,
                        self.epochs_to_analyze, self.segments_to_analyze, self.decode_prob_cutoff)

        print("\t\tPlotting Results as a Function of Average Decoding Probability")
        df.plot_decoded_func_p(self.tastant_fr_dist_pop, self.num_tastes, self.num_neur,
                               self.num_cp, self.segment_spike_times, self.tastant_spike_times,
                               self.start_dig_in_times, self.end_dig_in_times, self.post_taste_dt,
                               self.discrim_cp_raster_inds, self.e_skip_dt, self.e_len_dt,
                               self.dig_in_names, self.segment_times, self.segment_names,
                               self.taste_num_deliv, self.taste_select_neur, self.decode_dir,
                               self.max_decode, self.max_hz_pop, self.seg_stat_bin,
                               self.epochs_to_analyze, self.segments_to_analyze)

        print("Plotting Results as a Function of Co-Active Neurons")
        df.plot_decoded_func_n(self.tastant_fr_dist_pop, self.num_tastes, self.num_neur,
                               self.num_cp, self.segment_spike_times, self.tastant_spike_times,
                               self.start_dig_in_times, self.end_dig_in_times, self.post_taste_dt,
                               self.discrim_cp_raster_inds, self.e_skip_dt, self.e_len_dt,
                               self.dig_in_names, self.segment_times, self.segment_names,
                               self.taste_num_deliv, self.taste_select_neur, self.decode_dir,
                               self.max_decode, self.max_hz_pop, self.seg_stat_bin,
                               self.epochs_to_analyze, self.segments_to_analyze)

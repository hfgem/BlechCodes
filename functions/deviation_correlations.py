#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:03:33 2024

@author: Hannah Germaine
"""
import os
import json
import gzip
import tqdm
import numpy as np

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.dev_plot_funcs as dpf
import functions.dev_funcs as df
import functions.hdf5_handling as hf5

class run_deviation_correlations():

    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.gather_variables()
        self.import_deviations_and_cp()
        print('\n')
        self.calculate_correlations_all()
        # self.calculate_correlations_selective()
        # self.calculate_correlations_all_zscore()
		# self.calculate_correlations_selective_zscore()

    def gather_variables(self,):
        # Directories
        self.dev_dir = self.metadata['dir_name'] + 'Deviations/'
        self.hdf5_dir = self.metadata['hdf5_dir']
        self.comp_dir = self.metadata['dir_name'] + 'dev_x_taste/'
        if os.path.isdir(self.comp_dir) == False:
            os.mkdir(self.comp_dir)
        self.corr_dir = self.comp_dir + 'corr/'
        if os.path.isdir(self.corr_dir) == False:
            os.mkdir(self.corr_dir)
        # Params/Variables
        self.num_neur = self.data_dict['num_neur']
        self.pre_taste = self.metadata['params_dict']['pre_taste']
        self.post_taste = self.metadata['params_dict']['post_taste']
        self.segments_to_analyze = self.metadata['params_dict']['segments_to_analyze']
        self.epochs_to_analyze = self.metadata['params_dict']['epochs_to_analyze']
        self.segment_names = self.data_dict['segment_names']
        self.num_segments = len(self.segment_names)
        self.segment_spike_times = self.data_dict['segment_spike_times']
        self.segment_times = self.data_dict['segment_times']
        self.segment_times_reshaped = [
            [self.segment_times[i], self.segment_times[i+1]] for i in range(self.num_segments)]
        # Remember this is 1 less than the number of epochs
        self.num_cp = self.metadata['params_dict']['num_cp']
        self.tastant_spike_times = self.data_dict['tastant_spike_times']
        self.start_dig_in_times = self.data_dict['start_dig_in_times']
        self.end_dig_in_times = self.data_dict['end_dig_in_times']
        self.dig_in_names = self.data_dict['dig_in_names']
        self.z_bin = self.metadata['params_dict']['z_bin']

    def import_deviations_and_cp(self,):
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
        print("\tNow pulling true deviation rasters")
        num_segments = len(self.segments_to_analyze)
        segment_spike_times_reshaped = [self.segment_spike_times[i]
                               for i in self.segments_to_analyze]
        segment_times_reshaped = np.array(
            [self.segment_times_reshaped[i] for i in self.segments_to_analyze])
        segment_dev_rasters, segment_dev_times, segment_dev_vec, segment_dev_vec_zscore = df.create_dev_rasters(num_segments,
                                                                                                                segment_spike_times_reshaped,
                                                                                                                segment_times_reshaped,
                                                                                                                segment_deviations, self.z_bin)
        self.segment_dev_rasters = segment_dev_rasters
        self.segment_dev_times = segment_dev_times
        self.segment_dev_vec = segment_dev_vec
        self.segment_dev_vec_zscore = segment_dev_vec_zscore
        print("\tNow pulling changepoints")
        # Import changepoint data
        data_group_name = 'changepoint_data'
        pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
            self.hdf5_dir, data_group_name, 'pop_taste_cp_raster_inds')
        self.pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
        num_pt_cp = self.num_cp + 2
        # Import discriminability data
        data_group_name = 'taste_discriminability'
        peak_epochs = np.squeeze(hf5.pull_data_from_hdf5(
            self.hdf5_dir, data_group_name, 'peak_epochs'))
        discrim_neur = np.squeeze(hf5.pull_data_from_hdf5(
            self.hdf5_dir, data_group_name, 'discrim_neur'))
        # Convert discriminatory neuron data into pop_taste_cp_raster_inds shape
        # TODO: Test this first, then if going with this rework functions to fit instead!
        #num_discrim_cp = np.shape(discrim_neur)[0]
        #discrim_cp_raster_inds = []
        # for t_i in range(len(self.dig_in_names)):
        #     t_cp_vec = np.ones(
        #         (np.shape(pop_taste_cp_raster_inds[t_i])[0], num_discrim_cp))
        #     t_cp_vec = (peak_epochs[:num_pt_cp] +
        #                 int(self.pre_taste*1000))*t_cp_vec
        #     discrim_cp_raster_inds.append(t_cp_vec)
        #self.num_discrim_cp = len(peak_epochs)
        self.discrim_neur = discrim_neur
        #self.discrim_cp_raster_inds = discrim_cp_raster_inds

    def calculate_correlations_all(self,):
        print("\tCalculate correlations for all neurons")
        # Create storage directory
        self.current_corr_dir = self.corr_dir + 'all_neur/'
        if os.path.isdir(self.current_corr_dir) == False:
            os.mkdir(self.current_corr_dir)
        #self.neuron_keep_indices = np.ones((self.num_neur,self.num_cp+1))
        self.neuron_keep_indices = np.ones(np.shape(self.discrim_neur))
        # Calculate correlations
        df.calculate_vec_correlations(self.num_neur, self.segment_dev_vec, self.tastant_spike_times,
                                      self.start_dig_in_times, self.end_dig_in_times, self.segment_names,
                                      self.dig_in_names, self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
                                      self.current_corr_dir, self.neuron_keep_indices, self.segments_to_analyze)  # For all neurons in dataset
        # Calculate significant events
        sig_dev, sig_dev_counts = df.calculate_significant_dev(self.segment_dev_times, 
                                                               self.segment_times, self.dig_in_names,
                                                               self.segment_names, self.current_corr_dir,
                                                               self.segments_to_analyze)
        # Now plot and calculate significance!
        self.calculate_plot_corr_stats()
        self.calculate_significance()
        self.best_corr()

    def calculate_correlations_selective(self,):
        print("\tCalculate correlations for taste selective neurons only")
        # Import taste selectivity data
        #data_group_name = 'taste_selectivity'
        #taste_select_neur_epoch_bin = hf5.pull_data_from_hdf5(self.hdf5_dir,data_group_name,'taste_select_neur_epoch_bin')[0]
        #self.neuron_keep_indices = taste_select_neur_epoch_bin.T
        self.neuron_keep_indices = self.discrim_neur
        # Create storage directory
        self.current_corr_dir = self.corr_dir + 'taste_select_neur/'
        if os.path.isdir(self.current_corr_dir) == False:
            os.mkdir(self.current_corr_dir)
        # Calculate correlations
        df.calculate_vec_correlations(self.num_neur, self.segment_dev_vec, self.tastant_spike_times,
                                      self.start_dig_in_times, self.end_dig_in_times, self.segment_names,
                                      self.dig_in_names, self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
                                      self.current_corr_dir, self.neuron_keep_indices, self.segments_to_analyze)  # For all neurons in dataset
        # Now plot and calculate significance!
        self.calculate_plot_corr_stats()
        self.calculate_significance()
        self.best_corr()

    def calculate_correlations_all_zscore(self,):
        print("\tCalculate correlations for all neurons z-scored")
        # Create storage directory
        self.current_corr_dir = self.corr_dir + 'all_neur_zscore/'
        if os.path.isdir(self.current_corr_dir) == False:
            os.mkdir(self.current_corr_dir)
        self.neuron_keep_indices = np.ones(np.shape(self.discrim_neur))
        # Calculate correlations
        df.calculate_vec_correlations_zscore(self.num_neur, self.z_bin, self.segment_dev_vec_zscore, self.tastant_spike_times,
                                             self.segment_times, self.segment_spike_times, self.start_dig_in_times, self.end_dig_in_times,
                                             self.segment_names, self.dig_in_names, self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
                                             self.current_corr_dir, self.neuron_keep_indices, self.segments_to_analyze)
        # Calculate significant events
        sig_dev, sig_dev_counts = df.calculate_significant_dev(self.segment_dev_times, 
                                                               self.segment_times, self.dig_in_names,
                                                               self.segment_names, self.current_corr_dir,
                                                               self.segments_to_analyze)
        # Now plot and calculate significance!
        self.calculate_plot_corr_stats()
        self.calculate_significance()
        self.best_corr()

    def calculate_correlations_selective_zscore(self,):
        print("\tCalculate correlations for taste selective neurons z-scored")
        # Create storage directory
        self.current_corr_dir = self.corr_dir + 'taste_select_neur_zscore/'
        if os.path.isdir(self.current_corr_dir) == False:
            os.mkdir(self.current_corr_dir)
        # Import taste selectivity data
        #data_group_name = 'taste_selectivity'
        #taste_select_neur_epoch_bin = hf5.pull_data_from_hdf5(self.hdf5_dir,data_group_name,'taste_select_neur_epoch_bin')[0]
        #self.neuron_keep_indices = taste_select_neur_epoch_bin.T
        self.neuron_keep_indices = self.discrim_neur
        # Calculate correlations
        df.calculate_vec_correlations_zscore(self.num_neur, self.z_bin, self.segment_dev_vec_zscore, self.tastant_spike_times,
                                             self.segment_times, self.segment_spike_times, self.start_dig_in_times, self.end_dig_in_times,
                                             self.segment_names, self.dig_in_names, self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
                                             self.current_corr_dir, self.neuron_keep_indices, self.segments_to_analyze)
        # Now plot and calculate significance!
        self.calculate_plot_corr_stats()
        self.calculate_significance()
        self.best_corr()

    def calculate_plot_corr_stats(self,):
        # Plot dir setup
        plot_dir = self.current_corr_dir + 'plots/'
        if os.path.isdir(plot_dir) == False:
            os.mkdir(plot_dir)
        self.plot_dir = plot_dir
        # Calculate stats
        print("\tCalculating Correlation Statistics")
        corr_dev_stats = df.pull_corr_dev_stats(
            self.segment_names, self.dig_in_names, self.current_corr_dir, self.segments_to_analyze)
        print("\tPlotting Correlation Statistics")
        dpf.plot_stats(corr_dev_stats, self.segment_names, self.dig_in_names, self.plot_dir,
                       'Correlation', self.neuron_keep_indices, self.segments_to_analyze)
        print("\tPlotting Combined Correlation Statistics")
        segment_pop_vec_data = dpf.plot_combined_stats(corr_dev_stats, self.segment_names, self.dig_in_names,
                                                       self.plot_dir, 'Correlation', self.neuron_keep_indices, self.segments_to_analyze)
        self.segment_pop_vec_data = segment_pop_vec_data
        df.top_dev_corr_bins(corr_dev_stats, self.segment_names, self.dig_in_names,
                             self.plot_dir, self.neuron_keep_indices, self.segments_to_analyze)

    def calculate_significance(self,):
        print("\tCalculate statistical significance between correlation distributions.")
        self.current_stats_dir = self.current_corr_dir + 'stats/'
        if os.path.isdir(self.current_stats_dir) == False:
            os.mkdir(self.current_stats_dir)

        # KS-test
        df.stat_significance(self.segment_pop_vec_data, self.segment_names, self.dig_in_names,
                             self.current_stats_dir, 'population_vec_correlation', self.segments_to_analyze)

        # T-test less
        df.stat_significance_ttest_less(self.segment_pop_vec_data, self.segment_names,
                                        self.dig_in_names, self.current_stats_dir,
                                        'population_vec_correlation_ttest_less', self.segments_to_analyze)

        # T-test more
        df.stat_significance_ttest_more(self.segment_pop_vec_data, self.segment_names,
                                        self.dig_in_names, self.current_stats_dir,
                                        'population_vec_correlation_ttest_more', self.segments_to_analyze)

        # Mean compare
        df.mean_compare(self.segment_pop_vec_data, self.segment_names, self.dig_in_names,
                        self.current_stats_dir, 'population_vec_mean_difference', self.segments_to_analyze)

    def best_corr(self,):
        print("\tDetermine best correlation per deviation and plot stats.")
        self.best_dir = self.current_corr_dir + 'best/'
        if os.path.isdir(self.best_dir) == False:
            os.mkdir(self.best_dir)

        dpf.best_corr_calc_plot(self.dig_in_names, self.epochs_to_analyze,
                                self.segments_to_analyze, self.segment_names,
                                self.segment_dev_times, self.dev_dir,
                                self.current_corr_dir, self.best_dir)

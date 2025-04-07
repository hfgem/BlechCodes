#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:22:00 2024

@author: Hannah Germaine

This code runs an analysis of sliding bins and their correlations to taste 
responses.
"""

import os

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import numpy as np
import functions.analysis_funcs as af
import functions.dev_funcs as df
import functions.hdf5_handling as hf5
import functions.dev_plot_funcs as dpf
import functions.slide_plot_funcs as spf

class run_sliding_correlations():
    
    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.gather_variables()
        self.calculate_bin_data()
        # self.calculate_correlations_all()
        self.calculate_correlations_zscore()
        
    def gather_variables(self,):
        # Directories
        self.slide_dir = self.metadata['dir_name'] + 'Sliding_Correlations/'
        if os.path.isdir(self.slide_dir) == False:
            os.mkdir(self.slide_dir)
        self.hdf5_dir = self.metadata['hdf5_dir']
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
        self.num_cp = self.metadata['params_dict']['num_cp'] + 1
        self.tastant_spike_times = self.data_dict['tastant_spike_times']
        self.start_dig_in_times = self.data_dict['start_dig_in_times']
        self.end_dig_in_times = self.data_dict['end_dig_in_times']
        self.dig_in_names = self.data_dict['dig_in_names']
        self.bin_size = self.metadata['params_dict']['min_dev_size'] #Use the same minimal size as deviation events
        self.z_bin = self.metadata['params_dict']['z_bin']
        data_group_name = 'changepoint_data'
        self.pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
            self.hdf5_dir, data_group_name, 'pop_taste_cp_raster_inds')
        data_group_name = 'taste_discriminability'
        self.discrim_neur = np.squeeze(hf5.pull_data_from_hdf5(
            self.hdf5_dir, data_group_name, 'discrim_neur'))
       
        
    def calculate_bin_data(self,):
        print("\tNow calculating binned activity")
        bin_times, bin_pop_fr, bin_fr_vecs, bin_fr_vecs_zscore = af.get_bin_activity(self.segment_times_reshaped,
                                                                    self.segment_spike_times, self.bin_size, 
                                                                    self.segments_to_analyze, False)
        self.bin_times = bin_times
        self.bin_pop_fr = bin_pop_fr
        self.bin_fr_vecs = bin_fr_vecs
        self.bin_fr_vecs_zscore = bin_fr_vecs_zscore
        
    def calculate_correlations_all(self,):
        self.corr_dir = os.path.join(self.slide_dir,'all_neur')
        if os.path.isdir(self.corr_dir) == False:
            os.mkdir(self.corr_dir)
        
        neuron_keep_indices = np.ones(np.shape(self.discrim_neur))
        self.neuron_keep_indices = neuron_keep_indices
        
        df.calculate_vec_correlations(self.num_neur, self.bin_fr_vecs, self.tastant_spike_times,
                                      self.start_dig_in_times, self.end_dig_in_times, self.segment_names,
                                      self.dig_in_names, self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
                                      self.corr_dir, self.neuron_keep_indices, self.segments_to_analyze)  # For all neurons in dataset
        
        #Create plots
        self.generate_distribution_plots()
        self.calculate_corr_to_pop_rate()
        
    def calculate_correlations_zscore(self,):
        self.corr_dir = os.path.join(self.slide_dir,'all_neur_zscore')
        if os.path.isdir(self.corr_dir) == False:
            os.mkdir(self.corr_dir)
            
        neuron_keep_indices = np.ones(np.shape(self.discrim_neur))
        self.neuron_keep_indices = neuron_keep_indices
        
        df.calculate_vec_correlations_zscore(self.num_neur, self.z_bin, self.bin_fr_vecs_zscore, self.bin_pop_fr, self.tastant_spike_times,
                                             self.segment_times, self.segment_spike_times, self.start_dig_in_times, self.end_dig_in_times,
                                             self.segment_names, self.dig_in_names, self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
                                             self.corr_dir, self.neuron_keep_indices, self.segments_to_analyze)
        
        #Create plots
        self.generate_distribution_plots()
        self.calculate_corr_to_pop_rate()
        
    def generate_distribution_plots(self,):
        """Generate cumulative and density distribution plots of the 
        correlation distributions"""
        # Plot dir setup
        plot_dir = os.path.join(self.corr_dir,'plots/')
        if os.path.isdir(plot_dir) == False:
            os.mkdir(plot_dir)
        self.plot_dir = plot_dir
        # Pull statistics into dictionary and plot
        corr_slide_stats = df.pull_corr_dev_stats(
            self.segment_names, self.dig_in_names, self.corr_dir, 
            self.segments_to_analyze, False)
        self.corr_slide_stats = corr_slide_stats
        print("\tPlotting Correlation Statistics")
        dpf.plot_stats(corr_slide_stats, self.segment_names, self.dig_in_names, self.plot_dir,
                       'Correlation', self.neuron_keep_indices, self.segments_to_analyze)
        segment_pop_vec_data = dpf.plot_combined_stats(corr_slide_stats, self.segment_names, self.dig_in_names,
                                                       self.plot_dir, 'Correlation', self.neuron_keep_indices, self.segments_to_analyze)
        self.segment_pop_vec_data = segment_pop_vec_data
        
    
    def calculate_corr_to_pop_rate(self,):
        """Calculate how correlated bin correlations to taste are with the 
        population firing rate in that bin"""
        #Correlation calculations and plots
        spf.slide_corr_vs_rate(self.corr_slide_stats,self.bin_times,self.bin_pop_fr,
                               self.num_cp,self.plot_dir,self.corr_dir,
                               self.segment_names,self.dig_in_names,
                               self.segments_to_analyze)
        #90th-Percentile Correlations and the related pop rates
        spf.top_corr_rate_dist(self.corr_slide_stats,self.bin_times,self.bin_pop_fr,
                               self.num_cp,self.plot_dir,self.corr_dir,
                               self.segment_names,self.dig_in_names,
                               self.segments_to_analyze)

    def calculate_dev_overlap(self,):
        """Calculate how highly-correlated bins overlap with calculated 
        deviation events"""
    
    
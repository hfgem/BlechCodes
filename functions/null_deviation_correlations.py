#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:30:38 2024

@author: Hannah Germaine

This set of functions deals with calculating correlation values for null 
distributions - to be used in significance tests of true data correlations.
"""

import os
import sys
import warnings

import tqdm
import gzip
import json
import numpy as np

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.dev_funcs as df
import functions.dev_plot_funcs as dpf
import functions.hdf5_handling as hf5


warnings.filterwarnings("ignore")

class run_null_deviation_correlations():
    
    def __init__(self,args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.gather_variables()
        self.import_null_deviations()
        self.convert_null_dev_to_rasters()
        self.calculate_correlations_all_null()
        self.calculate_correlations_zscore_null()
        
    def gather_variables(self,):
        #These directories should already exist
        self.hdf5_dir = self.metadata['hdf5_dir']
        self.null_dir = self.metadata['dir_name'] + 'null_data/'
        self.dev_dir = self.metadata['dir_name'] + 'Deviations/'
        self.comp_dir = self.metadata['dir_name'] + 'dev_x_taste/'
        if os.path.isdir(self.comp_dir) == False:
            os.mkdir(self.comp_dir)
        self.corr_dir = self.comp_dir + 'corr/'
        if os.path.isdir(self.corr_dir) == False:
            os.mkdir(self.corr_dir)
        
        self.num_neur = self.data_dict['num_neur']
        self.tastant_spike_times = self.data_dict['tastant_spike_times']
        self.start_dig_in_times = self.data_dict['start_dig_in_times']
        self.end_dig_in_times = self.data_dict['end_dig_in_times']
        self.dig_in_names = self.data_dict['dig_in_names']
        self.segment_names = self.data_dict['segment_names']
        self.num_segments = len(self.segment_names)
        self.pre_taste = self.metadata['params_dict']['pre_taste']
        self.post_taste = self.metadata['params_dict']['post_taste']
        # Import changepoint data
        self.num_cp = self.metadata['params_dict']['num_cp']+ 1
        data_group_name = 'changepoint_data'
        pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
            self.hdf5_dir, data_group_name, 'pop_taste_cp_raster_inds')
        self.pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
        data_group_name = 'taste_discriminability'
        discrim_neur = np.squeeze(hf5.pull_data_from_hdf5(
            self.hdf5_dir, data_group_name, 'discrim_neur'))
        self.discrim_neur = discrim_neur
        self.num_null = self.metadata['params_dict']['num_null']
        self.segments_to_analyze = self.metadata['params_dict']['segments_to_analyze']
        self.segment_names = self.data_dict['segment_names']
        self.segment_spike_times = self.data_dict['segment_spike_times']
        self.segment_times = self.data_dict['segment_times']
        self.segment_times_reshaped = [
            [self.segment_times[i], self.segment_times[i+1]] for i in range(self.num_segments)]
        self.z_bin = self.metadata['params_dict']['z_bin']
        
    def import_null_deviations(self,):
        try:  # test if the data exists by trying to import the last from each segment
            null_i = self.num_null - 1
            for s_i in tqdm.tqdm(self.segments_to_analyze):
                filepath = self.dev_dir + 'null_data/' + \
                     self.segment_names[s_i] + '/null_' + \
                         str(null_i) + '_deviations.json'
                with gzip.GzipFile(filepath, mode="r") as f:
                    json_bytes = f.read()
                    json_str = json_bytes.decode('utf-8')
                    data = json.loads(json_str)
        except:
            print("ERROR! ERROR! ERROR!")
            print("Null deviations were not calculated previously as expected.")
            print("Something went wrong in the analysis pipeline.")
            print("Please try reverting your analysis_state_tracker.csv to 2 to rerun.")
            print("If issues persist, contact Hannah.")
            sys.exit()
            
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
        
    def convert_null_dev_to_rasters(self,):
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
        
        print("\tNow pulling null deviation rasters")
        num_seg = len(self.segments_to_analyze)
        seg_times_reshaped = np.array(self.segment_times_reshaped)[
            self.segments_to_analyze, :]
        
        null_dev_vecs = []
        null_dev_vecs_zscore = []
        for s_i in range(num_seg):
            null_dev_vecs.append([])
            null_dev_vecs_zscore.append([])
        for null_i in tqdm.tqdm(range(self.num_null)):
            null_segment_deviations = self.all_null_deviations[null_i]
            null_segment_spike_times = self.all_null_segment_spike_times[null_i]
            _, _, null_segment_dev_vecs_i, null_segment_dev_vecs_zscore_i, _, _ = df.create_dev_rasters(num_seg,
                                                                     null_segment_spike_times,
                                                                     seg_times_reshaped,
                                                                     null_segment_deviations,
                                                                     self.z_bin, no_z = False)
            #Compiled all into a single segment group, rather than keeping separated by null dist
            for s_i in range(num_seg):
                null_dev_vecs[s_i].extend(null_segment_dev_vecs_i[s_i])
                null_dev_vecs_zscore[s_i].extend(null_segment_dev_vecs_zscore_i[s_i])

        self.__dict__.pop('all_null_deviations', None)
        self.null_dev_vecs = null_dev_vecs
        self.null_dev_vecs_zscore = null_dev_vecs_zscore
    
    def calculate_correlations_all_null(self,):
        print('\tCalculating null correlation distributions')
        if os.path.isdir(self.corr_dir + 'all_neur/') == False:
            os.mkdir(self.corr_dir + 'all_neur/')
        self.current_corr_dir = self.corr_dir + 'all_neur/' + 'null/'
        if os.path.isdir(self.current_corr_dir) == False:
            os.mkdir(self.current_corr_dir)
        self.neuron_keep_indices = np.ones(np.shape(self.discrim_neur))
        # Calculate correlations
        df.calculate_vec_correlations(self.num_neur, self.null_dev_vecs, self.tastant_spike_times,
                                      self.start_dig_in_times, self.end_dig_in_times, self.segment_names,
                                      self.dig_in_names, self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
                                      self.current_corr_dir, self.neuron_keep_indices, self.segments_to_analyze)  # For all neurons in dataset
        # Now plot and calculate significance!
        self.stats_plots()
        
    def calculate_correlations_zscore_null(self,):
        print('\tCalculating null correlation distributions')
        if os.path.isdir(self.corr_dir + 'all_neur_zscore/') == False:
            os.mkdir(self.corr_dir + 'all_neur_zscore/')
        self.current_corr_dir = self.corr_dir + 'all_neur_zscore/' + 'null/'
        if os.path.isdir(self.current_corr_dir) == False:
            os.mkdir(self.current_corr_dir)
        self.neuron_keep_indices = np.ones(np.shape(self.discrim_neur))
        # Calculate correlations
        df.calculate_vec_correlations_zscore(self.num_neur, self.z_bin, self.null_dev_vecs_zscore, self.tastant_spike_times,
                                      self.segment_times, self.segment_spike_times, self.start_dig_in_times, self.end_dig_in_times,
                                      self.segment_names, self.dig_in_names, self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
                                      self.current_corr_dir, self.neuron_keep_indices, self.segments_to_analyze)  # For all neurons in dataset
        # Now plot and calculate significance!
        self.stats_plots()
    
    def stats_plots(self,):
        # Plot dir setup
        print('\tPlotting null correlation distributions')
        plot_dir = self.current_corr_dir + 'plots/'
        if os.path.isdir(plot_dir) == False:
            os.mkdir(plot_dir)
        self.plot_dir = plot_dir
        corr_dev_stats = df.pull_corr_dev_stats(
            self.segment_names, self.dig_in_names, self.current_corr_dir, self.segments_to_analyze)
        dpf.plot_stats(corr_dev_stats, self.segment_names, self.dig_in_names, self.plot_dir,
                       'Correlation', self.neuron_keep_indices, self.segments_to_analyze)
        segment_pop_vec_data = dpf.plot_combined_stats(corr_dev_stats, self.segment_names, self.dig_in_names,
                                                       self.plot_dir, 'Correlation', self.neuron_keep_indices, self.segments_to_analyze)
        null_corr_percentiles = df.null_dev_corr_90_percentiles(corr_dev_stats, self.segment_names, self.dig_in_names, 
                                         self.current_corr_dir, self.segments_to_analyze)
    
    
    
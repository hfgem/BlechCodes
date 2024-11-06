#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:06:35 2024

@author: Hannah Germaine

In this step of the analysis pipeline, deviation events are split in half
and tested for sequential changes in firing rate and taste sequence similarity.
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
import functions.plot_dev_decoding_funcs as pddf
import functions.dev_funcs as dev_f
import functions.hdf5_handling as hf5

class run_deviation_sequence_analysis():
    
    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.gather_variables()
        self.import_deviations()
        self.pull_fr_dist()
        self.analyze_sequences()
        
    def gather_variables(self,):
        # Directories
        self.hdf5_dir = self.metadata['hdf5_dir']
        self.seq_dir = self.metadata['dir_name'] + \
            'Deviation_Sequence_Analysis/'
        if os.path.isdir(self.seq_dir) == False:
            os.mkdir(self.seq_dir)
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
        # Import changepoint data
        pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
            self.hdf5_dir, 'changepoint_data', 'pop_taste_cp_raster_inds')
        self.pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
        num_pt_cp = self.num_cp + 2
        
    def import_deviations(self,):
        print("\tNow importing calculated deviations")
        
        num_seg_to_analyze = len(self.segments_to_analyze)
        segment_names_to_analyze = [self.segment_names[i] for i in self.segments_to_analyze]
        segment_times_to_analyze_reshaped = [
            [self.segment_times[i], self.segment_times[i+1]] for i in self.segments_to_analyze]
        segment_spike_times_to_analyze = [self.segment_spike_times[i] for i in self.segments_to_analyze]
        
        segment_deviations = []
        for s_i in tqdm.tqdm(range(num_seg_to_analyze)):
            filepath = self.dev_dir + \
                segment_names_to_analyze[s_i] + '/deviations.json'
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)
                segment_deviations.append(data)

        print("\tNow pulling true deviation rasters")
        segment_dev_rasters, segment_dev_times, segment_dev_fr_vecs, \
            segment_dev_fr_vecs_zscore, zscore_means, zscore_stds \
                = dev_f.create_dev_rasters(num_seg_to_analyze,
                            segment_spike_times_to_analyze, 
                            np.array(segment_times_to_analyze_reshaped),
                            segment_deviations, self.pre_taste)

        self.segment_dev_rasters = segment_dev_rasters
        self.segment_dev_times = segment_dev_times
        self.segment_dev_fr_vecs = segment_dev_fr_vecs
        self.segment_dev_fr_vecs_zscore = segment_dev_fr_vecs_zscore
        
    def pull_fr_dist(self,):
        print("\tPulling FR Distributions")
        tastant_fr_dist_pop, taste_num_deliv, max_hz_pop = ddf.taste_fr_dist(self.num_neur, self.tastant_spike_times,
                                                                        	 self.pop_taste_cp_raster_inds, self.bayes_fr_bins,
                                                                        	 self.start_dig_in_times, self.pre_taste_dt,
                                                                        	 self.post_taste_dt, self.trial_start_frac)
        self.tastant_fr_dist_pop = tastant_fr_dist_pop
        self.taste_num_deliv = taste_num_deliv
        self.max_hz_pop = max_hz_pop
        tastant_fr_dist_z_pop, taste_num_deliv, max_hz_z_pop, min_hz_z_pop = ddf.taste_fr_dist_zscore(self.num_neur, self.tastant_spike_times,
                                                                                                	  self.segment_spike_times, self.segment_names,
                                                                                                	  self.segment_times, self.pop_taste_cp_raster_inds,
                                                                                                	  self.bayes_fr_bins, self.start_dig_in_times, self.pre_taste_dt,
                                                                                                	  self.post_taste_dt, self.bin_dt, self.trial_start_frac)
        self.tastant_fr_dist_z_pop = tastant_fr_dist_z_pop
        self.max_hz_z_pop = max_hz_z_pop
        self.min_hz_z_pop = min_hz_z_pop
        
    def analyze_sequences(self,):
        print("Analyzing deviation sequences")
        
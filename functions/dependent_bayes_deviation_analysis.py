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
from utils.input_funcs import *
import functions.dependent_decoding_funcs as ddf
import functions.plot_dev_decoding_funcs as pddf
import functions.dev_funcs as dev_f
import functions.hdf5_handling as hf5

class run_deviation_dependent_bayes():

    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.gather_variables()
        self.import_deviations()
        self.pull_fr_dist()
        self.decode_groups()
        self.decode_zscored()
        #self.decode_nonzscored()

    def gather_variables(self,):
        # Directories
        self.hdf5_dir = self.metadata['hdf5_dir']
        self.slide_decode_dir = self.metadata['dir_name'] + \
            'Sliding_Decoding/'
        if os.path.isdir(self.slide_decode_dir) == False:
            os.mkdir(self.slide_decode_dir)
        self.bayes_dir = self.metadata['dir_name'] + \
            'Deviation_Dependent_Decoding/'
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
        #Ask for user input on which dig-ins are palatable
        print("\nUSER INPUT REQUESTED: Mark which tastants are palatable.\n")
        self.palatable_dig_inds = select_analysis_groups(self.dig_in_names)
        self.num_tastes = len(self.dig_in_names)
        self.min_dev_size = self.metadata['params_dict']['min_dev_size']
        # Decoding Params/Variables
        self.e_skip_time = self.metadata['params_dict']['bayes_params']['e_skip_time']
        self.e_skip_dt = np.ceil(self.e_skip_time*1000).astype('int')
        self.taste_e_len_time = self.metadata['params_dict']['bayes_params']['taste_e_len_time']
        self.taste_e_len_dt = np.ceil(self.taste_e_len_time*1000).astype('int')
        self.seg_e_len_time = self.metadata['params_dict']['bayes_params']['seg_e_len_time']
        self.seg_e_len_dt = np.ceil(self.seg_e_len_time*1000).astype('int') 
        self.bayes_fr_bins = self.metadata['params_dict']['bayes_params']['fr_bins']
        self.neuron_count_thresh = self.metadata['params_dict']['bayes_params']['neuron_count_thresh']
        self.max_decode = self.metadata['params_dict']['bayes_params']['max_decode']
        self.seg_stat_bin = self.metadata['params_dict']['bayes_params']['seg_stat_bin']
        self.trial_start_frac = self.metadata['params_dict']['bayes_params']['trial_start_frac']
        self.decode_prob_cutoff = self.metadata['params_dict']['bayes_params']['decode_prob_cutoff']
        self.bin_time = self.metadata['params_dict']['bayes_params']['z_score_bin_time']
        self.bin_dt = np.ceil(self.bin_time*1000).astype('int')
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
            segment_dev_fr_vecs_zscore, _, _ = dev_f.create_dev_rasters(num_seg_to_analyze, 
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
        
    def decode_groups(self,):
        print("Determine decoding groups")
        #Create fr vector grouping instructions: list of epoch,taste pairs
        non_none_tastes = [taste for taste in dig_in_names if taste[:4] != 'none']
        self.non_none_tastes = non_none_tastes
        group_list, group_names = ddf.decode_groupings(self.epochs_to_analyze,
                                                       self.dig_in_names,
                                                       self.palatable_dig_inds,
                                                       self.non_none_tastes)
        self.group_list = group_list
        self.group_names = group_names
        
    def decode_zscored(self,):
        print("\tRun z-scored data decoder pipeline")
        decode_dir = self.slide_decode_dir + 'All_Neurons_Z_Scored/'
        if os.path.isdir(decode_dir) == False:
            os.mkdir(decode_dir)
        self.decode_dir = decode_dir
        self.z_score = True
        self.tastant_fr_dist = self.tastant_fr_dist_z_pop
        self.dev_vecs = self.segment_dev_fr_vecs_zscore
        self.decoder_accuracy_tests()
        self.decode_sliding_bin()
        self.decode_dev()
        
    def decode_nonzscored(self,):
        print("\tRun non-z-scored data decoder pipeline")
        decode_dir = os.path.join(self.slide_decode_dir,'All_Neurons_Z_Scored')
        if os.path.isdir(decode_dir) == False:
            os.mkdir(decode_dir)
        self.decode_dir = decode_dir
        self.z_score = False
        self.tastant_fr_dist = self.tastant_fr_dist_pop
        self.dev_vecs = self.segment_dev_fr_vecs
        self.decoder_accuracy_tests()
        self.decode_sliding_bin()
        self.decode_dev()
        
    def decoder_accuracy_tests(self,):
        print("\t\tRunning decoder accuracy tests.")
        
        ddf.decoder_accuracy_tests(self.tastant_fr_dist, self.segment_spike_times, 
                        self.dig_in_names, self.segment_times, self.segment_names, 
                        self.start_dig_in_times, self.taste_num_deliv,
                        self.group_list, self.group_names, self.non_none_tastes, 
                        self.decode_dir, self.bin_dt, self.z_score, 
                        self.epochs_to_analyze, self.segments_to_analyze)
        

    def decode_sliding_bin(self,):
        print("\t\tDecoding sliding bins of rest intervals.")

        ddf.decode_sliding_bins(self.tastant_fr_dist, self.segment_spike_times, 
                                self.dig_in_names, self.palatable_dig_inds,
                                self.segment_times, self.segment_names, 
                                self.start_dig_in_times, self.taste_num_deliv, 
                                self.segment_dev_times, self.dev_vecs, 
                                self.bin_dt, self.group_list, self.group_names, 
                                self.non_none_tastes, self.decode_dir, self.z_score, 
                                self.epochs_to_analyze, self.segments_to_analyze)
        
    def decode_dev(self,):
        print("\t\tDecoding deviation events.")
        
        ddf.decode_deviations(self.tastant_fr_dist_z_pop, self.segment_spike_times,
                                     self.dig_in_names, self.segment_times, 
                                     self.segment_names, self.start_dig_in_times, 
                                     self.taste_num_deliv, self.segment_dev_times,
                                     self.segment_dev_fr_vecs_zscore, self.bin_dt, 
                                     self.decode_dir, True, 
                                     self.epochs_to_analyze, self.segments_to_analyze)
        self.plot_decoded_data()


    def plot_decoded_data(self,):
        print("\t\tPlotting Decoded Results")
        
        pddf.plot_is_taste_which_taste_decoded(self.num_tastes, self.num_neur, 
                                               self.segment_spike_times, self.tastant_spike_times,
                                               self.start_dig_in_times, self.post_taste_dt, 
                                               self.pre_taste_dt, self.pop_taste_cp_raster_inds, 
                                               self.bin_dt, self.dig_in_names, self.segment_times,
                                               self.segment_names, self.decode_dir, self.max_hz_pop,
                                               self.segment_dev_times, self.segment_dev_fr_vecs, 
                                               self.segment_dev_fr_vecs_zscore, self.neuron_count_thresh, 
                                               self.seg_e_len_dt, self.trial_start_frac,
                                               self.epochs_to_analyze, self.segments_to_analyze, 
                                               self.decode_prob_cutoff)
    
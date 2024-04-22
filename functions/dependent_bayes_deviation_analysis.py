#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:55:12 2024

@author: Hannah Germaine

In this step of the analysis pipeline, a Bayesian decoder is trained on taste
responses and then run on the deviation events calculated earlier in the 
pipeline.
"""

import os, tqdm, gzip, json

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import numpy as np
import functions.analysis_funcs as af
import functions.dev_funcs as dev_f
import functions.dependent_decoding_funcs as ddf

class run_deviation_dependent_bayes():
	
	def __init__(self,args):
		self.metadata = args[0]
		self.data_dict = args[1]
		self.gather_variables()
		self.import_deviations()
		self.pull_fr_dist()
		
	def gather_variables(self,):
		#Directories
		self.hdf5_dir = self.metadata['hdf5_dir']
		self.bayes_dir = self.metadata['dir_name'] + 'Burst_Dependent_Decoding/'
		if os.path.isdir(self.bayes_dir) == False:
			os.mkdir(self.bayes_dir)
		self.dev_dir = self.metadata['dir_name'] + 'Deviations/'
		#General Params/Variables
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
		self.segment_times_reshaped = [[self.segment_times[i],self.segment_times[i+1]] for i in range(self.num_segments)]
		self.num_cp = self.metadata['params_dict']['num_cp'] + 1 #Remember this imported value is 1 less than the number of epochs
		self.tastant_spike_times = self.data_dict['tastant_spike_times']
		self.start_dig_in_times = self.data_dict['start_dig_in_times']
		self.end_dig_in_times = self.data_dict['end_dig_in_times']
		self.dig_in_names = self.data_dict['dig_in_names']
		self.num_tastes = len(self.dig_in_names)
		self.min_dev_size = self.metadata['params_dict']['min_dev_size']
		#Decoding Params/Variables
		self.skip_time = self.metadata['params_dict']['bayes_params']['skip_time']
		self.skip_dt = np.ceil(self.skip_time*1000).astype('int')
		self.neuron_count_thresh = self.metadata['params_dict']['bayes_params']['neuron_count_thresh']
		self.max_decode = self.metadata['params_dict']['bayes_params']['max_decode']
		self.seg_stat_bin = self.metadata['params_dict']['bayes_params']['seg_stat_bin']
		self.trial_start_frac = self.metadata['params_dict']['bayes_params']['trial_start_frac']
		self.decode_prob_cutoff = self.metadata['params_dict']['bayes_params']['decode_prob_cutoff']
		self.bin_time = self.metadata['params_dict']['bayes_params']['bin_time']
		self.bin_dt = np.ceil(self.bin_time*1000).astype('int')
		pop_taste_cp_raster_inds = af.pull_data_from_hdf5(self.hdf5_dir, 'changepoint_data', 'pop_taste_cp_raster_inds')
		self.pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
		
	def import_deviations(self,):
		print("\tNow importing calculated deviations")
		segment_deviations = []
		for s_i in tqdm.tqdm(range(self.num_segments)):
			filepath = self.dev_dir + self.segment_names[s_i] + '/deviations.json'
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
		tastant_fr_dist_z_pop, taste_num_deliv, max_hz_z_pop, min_hz_z_pop = ddf.taste_fr_dist_zscore(self.num_neur,
	                                                                                                  self.num_cp, self.tastant_spike_times,
	                                                                                                  self.segment_spike_times, self.segment_names,
	                                                                                                  self.segment_times, self.pop_taste_cp_raster_inds,
	                                                                                                  self.start_dig_in_times, self.pre_taste_dt,
	                                                                                                  self.post_taste_dt, self.bin_dt, self.trial_start_frac)
		self.tastant_fr_dist_z_pop = tastant_fr_dist_z_pop
		self.taste_num_deliv = taste_num_deliv
		self.max_hz_z_pop = max_hz_z_pop
		self.min_hz_z_pop = min_hz_z_pop
		
	def decode_all_neurons(self,):
		print("\tDecoding all neurons")
		
		
		
		
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:25:10 2024

@author: Hannah Germaine

In this step of the analysis pipeline, a Bayesian decoder is trained on taste
responses and then run on sweeping bins of the recorded activity to determine
where taste-like activity occurs.
"""

import os

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import numpy as np
import functions.analysis_funcs as af
import functions.dependent_decoding_funcs as ddf
import functions.decoding_funcs as df

class run_dependent_bayes():
	
	def __init__(self,args):
		self.metadata = args[0]
		self.data_dict = args[1]
		self.gather_variables()
		self.pull_fr_dist()
		self.decode_all_neurons()
		self.decode_selective_neurons()
		self.decode_all_neurons_zscored()
		self.decode_selective_neurons_zscored()
		
	def gather_variables(self,):
		#Directories
		self.hdf5_dir = self.metadata['hdf5_dir']
		self.bayes_dir = self.metadata['dir_name'] + 'Bayes_Dependent_Decoding/'
		if os.path.isdir(self.bayes_dir) == False:
			os.mkdir(self.bayes_dir)
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
		#Bayes Params/Variabls
		self.skip_time = self.metadata['params_dict']['bayes_params']['skip_time']
		self.skip_dt = np.ceil(self.skip_time*1000).astype('int')
		self.e_skip_time = self.metadata['params_dict']['bayes_params']['e_skip_time']
		self.e_skip_dt = np.ceil(self.e_skip_time*1000).astype('int')
		self.e_len_time = self.metadata['params_dict']['bayes_params']['e_len_time']
		self.e_len_dt = np.ceil(self.e_len_time*1000).astype('int')
		self.neuron_count_thresh = self.metadata['params_dict']['bayes_params']['neuron_count_thresh']
		self.max_decode = self.metadata['params_dict']['bayes_params']['max_decode']
		self.seg_stat_bin = self.metadata['params_dict']['bayes_params']['seg_stat_bin']
		self.trial_start_frac = self.metadata['params_dict']['bayes_params']['trial_start_frac']
		self.decode_prob_cutoff = self.metadata['params_dict']['bayes_params']['decode_prob_cutoff']
		self.bin_time = self.metadata['params_dict']['bayes_params']['bin_time']
		self.bin_dt = np.ceil(self.bin_time*1000).astype('int')
		pop_taste_cp_raster_inds = af.pull_data_from_hdf5(self.hdf5_dir, 'changepoint_data', 'pop_taste_cp_raster_inds')
		self.pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
	   
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
		self.decode_dir = self.bayes_dir + 'All_Neurons/'
		if os.path.isdir(self.decode_dir) == False:
			os.mkdir(self.decode_dir)
		self.cur_dist = self.tastant_fr_dist_pop
		self.select_neur = np.ones((self.num_cp, self.num_neur))
		
		ddf.decode_epochs(self.cur_dist, self.segment_spike_times, self.post_taste_dt,
	                      self.skip_dt, self.e_skip_dt, self.e_len_dt, self.dig_in_names, self.segment_times,
	                      self.segment_names, self.start_dig_in_times, self.taste_num_deliv,
	                      self.select_neur, self.max_hz_pop, self.decode_dir,
	                      self.neuron_count_thresh, self.trial_start_frac, self.epochs_to_analyze,
	                      self.segments_to_analyze)
		
		self.plot_decode_results()
		
	def decode_selective_neurons(self,):
		print("\tDecoding taste selective neurons")
		
		try:
			self.select_neur = af.pull_data_from_hdf5(self.hdf5_dir, 'taste_selectivity', 'taste_select_neur_epoch_bin')[0]
		except:
			print("ERROR: No taste selective data found.")
			quit()
		
		self.decode_dir = self.bayes_dir + 'Taste_Selective/'
		if os.path.isdir(self.decode_dir) == False:
			os.mkdir(self.decode_dir)
		self.cur_dist = self.tastant_fr_dist_pop
		
		ddf.decode_epochs(self.cur_dist, self.segment_spike_times, self.post_taste_dt,
	                      self.skip_dt, self.e_skip_dt, self.e_len_dt, self.dig_in_names, self.segment_times,
	                      self.segment_names, self.start_dig_in_times, self.taste_num_deliv,
	                      self.select_neur, self.max_hz_pop, self.decode_dir,
	                      self.neuron_count_thresh, self.trial_start_frac, self.epochs_to_analyze,
	                      self.segments_to_analyze)
		
		self.plot_decode_results()
		
	def decode_all_neurons_zscored(self,):
		print("\tDecoding all neurons")
		self.decode_dir = self.bayes_dir + 'All_Neurons_ZScored/'
		if os.path.isdir(self.decode_dir) == False:
			os.mkdir(self.decode_dir)
		self.cur_dist = self.tastant_fr_dist_z_pop
		self.select_neur = np.ones((self.num_cp, self.num_neur))
		
		ddf.decode_epochs_zscore(self.cur_dist, self.segment_spike_times, self.post_taste_dt,
	                      self.skip_dt, self.e_skip_dt, self.e_len_dt, self.dig_in_names, self.segment_times,
						  self.bin_dt, self.segment_names, self.start_dig_in_times, self.taste_num_deliv,
	                      self.select_neur, self.max_hz_z_pop, self.decode_dir,
	                      self.neuron_count_thresh, self.trial_start_frac, self.epochs_to_analyze,
	                      self.segments_to_analyze)
		
		self.plot_decode_results()
		
	def decode_selective_neurons_zscored(self,):
		print("\tDecoding taste selective neurons")
		
		self.select_neur = af.pull_data_from_hdf5(self.hdf5_dir, 'taste_selectivity', 'taste_select_neur_epoch_bin')[0]
		
		self.decode_dir = self.bayes_dir + 'Taste_Selective/'
		if os.path.isdir(self.decode_dir) == False:
			os.mkdir(self.decode_dir)
		self.cur_dist = self.tastant_fr_dist_z_pop
		
		ddf.decode_epochs_zscore(self.cur_dist, self.segment_spike_times, self.post_taste_dt,
	                      self.skip_dt, self.e_skip_dt, self.e_len_dt, self.dig_in_names, self.segment_times,
						  self.bin_dt, self.segment_names, self.start_dig_in_times, self.taste_num_deliv,
	                      self.select_neur, self.max_hz_z_pop, self.decode_dir,
	                      self.neuron_count_thresh, self.trial_start_frac, self.epochs_to_analyze,
	                      self.segments_to_analyze)
		
		self.plot_decode_results()
		
	def plot_decode_results(self,):
		print("\t\tPlotting results")
		df.plot_decoded(self.cur_dist, self.num_tastes, self.num_neur, self.num_cp,
	                    self.segment_spike_times, self.tastant_spike_times, self.start_dig_in_times, 
						self.end_dig_in_times, self.post_taste_dt, self.pre_taste_dt,
	                    self.pop_taste_cp_raster_inds, self.dig_in_names, self.segment_times,
	                    self.segment_names, self.taste_num_deliv, self.select_neur,
	                    self.decode_dir, self.max_decode, self.max_hz_pop, self.seg_stat_bin,
	                    self.neuron_count_thresh, self.trial_start_frac, self.epochs_to_analyze,
	                    self.segments_to_analyze, self.decode_prob_cutoff)
		df.plot_decoded_func_p(self.cur_dist, self.num_tastes, self.num_neur, 
						 self.num_cp, self.segment_spike_times, self.tastant_spike_times,
						 self.start_dig_in_times, self.end_dig_in_times, self.post_taste_dt, 
						 self.pop_taste_cp_raster_inds, self.e_skip_dt, self.e_len_dt, 
						 self.dig_in_names, self.segment_times, self.segment_names, 
						 self.taste_num_deliv, self.select_neur, self.decode_dir, 
						 self.max_decode, self.max_hz_pop, self.seg_stat_bin,
						 self.epochs_to_analyze, self.segments_to_analyze)
		df.plot_decoded_func_n(self.cur_dist, self.num_tastes, self.num_neur, self.num_cp, 
						 self.segment_spike_times, self.tastant_spike_times, self.start_dig_in_times, 
						 self.end_dig_in_times, self.post_taste_dt, self.pop_taste_cp_raster_inds,
						 self.e_skip_dt, self.e_len_dt, self.dig_in_names, self.segment_times,
						 self.segment_names, self.taste_num_deliv, self.select_neur,
						 self.decode_dir, self.max_decode, self.max_hz_pop, self.seg_stat_bin,
						 self.epochs_to_analyze, self.segments_to_analyze)
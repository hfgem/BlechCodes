#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:03:33 2024

@author: Hannah Germaine
"""
import os,json,gzip,tqdm

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import numpy as np
import functions.analysis_funcs as af
import functions.dev_funcs as df
import functions.dev_plot_funcs as dpf
import functions.hdf5_handling as hf5

class run_deviation_correlations():
	
	def __init__(self,args):
		self.metadata = args[0]
		self.data_dict = args[1]
		self.gather_variables()
		self.import_deviations_and_cp()
		self.calculate_correlations_all()
		self.calculate_correlations_selective()
		self.calculate_correlations_all_zscore()
		
		
	def gather_variables(self,):
		#Directories
		self.dev_dir = self.metadata['dir_name'] + 'Deviations/'
		self.hdf5_dir = self.metadata['hdf5_dir']
		self.comp_dir = self.metadata['dir_name'] + 'dev_x_taste/'
		if os.path.isdir(self.comp_dir) == False:
			os.mkdir(self.comp_dir)
		self.corr_dir = self.comp_dir + 'corr/'
		if os.path.isdir(self.corr_dir) == False:
			os.mkdir(self.corr_dir)
		#Params/Variables
		self.num_neur = self.data_dict['num_neur']
		self.pre_taste = self.metadata['params_dict']['pre_taste']
		self.post_taste = self.metadata['params_dict']['post_taste']
		self.segments_to_analyze = self.metadata['params_dict']['segments_to_analyze']
		self.segment_names = self.data_dict['segment_names']
		self.num_segments = len(self.segment_names)
		self.segment_spike_times = self.data_dict['segment_spike_times']
		self.segment_times = self.data_dict['segment_times']
		self.segment_times_reshaped = [[self.segment_times[i],self.segment_times[i+1]] for i in range(self.num_segments)]
		self.num_cp = self.metadata['params_dict']['num_cp'] #Remember this is 1 less than the number of epochs
		self.tastant_spike_times = self.data_dict['tastant_spike_times']
		self.start_dig_in_times = self.data_dict['start_dig_in_times']
		self.end_dig_in_times = self.data_dict['end_dig_in_times']
		self.dig_in_names = self.data_dict['dig_in_names']
		
	def import_deviations_and_cp(self,):
		print("Now importing calculated deviations")
		segment_deviations = []
		for s_i in tqdm.tqdm(self.segments_to_analyze):
			filepath = self.dev_dir + self.segment_names[s_i] + '/deviations.json'
			with gzip.GzipFile(filepath, mode="r") as f:
				json_bytes = f.read()
				json_str = json_bytes.decode('utf-8')			
				data = json.loads(json_str) 
				segment_deviations.append(data)
		print("Now pulling true deviation rasters")
		num_segments = len(self.segments_to_analyze)
		segment_spike_times = [self.segment_spike_times[i] for i in self.segments_to_analyze]
		segment_times_reshaped = np.array([self.segment_times_reshaped[i] for i in self.segments_to_analyze])
		segment_dev_rasters, segment_dev_times, segment_dev_rasters_zscore = df.create_dev_rasters(num_segments, 
																	segment_spike_times,
																	segment_times_reshaped,
																	segment_deviations,
																	self.pre_taste)
		self.segment_dev_rasters = segment_dev_rasters
		self.segment_dev_times = segment_dev_times
		self.segment_dev_rasters_zscore = segment_dev_rasters_zscore
		print("Now pulling changepoints")
		#Import changepoint data
		data_group_name = 'changepoint_data'
		pop_taste_cp_raster_inds = af.pull_data_from_hdf5(self.hdf5_dir,data_group_name,'pop_taste_cp_raster_inds')
		self.pop_taste_cp_raster_inds = pop_taste_cp_raster_inds
		
	def calculate_correlations_all(self,):
		print("\tCalculate correlations for all neurons")
		#Create storage directory
		self.current_corr_dir = self.corr_dir + 'all_neur/'
		if os.path.isdir(self.current_corr_dir) == False:
			os.mkdir(self.current_corr_dir)
		self.neuron_keep_indices = np.ones((self.num_neur,self.num_cp+1))
		#Calculate correlations
		df.calculate_vec_correlations(self.segment_dev_rasters, self.tastant_spike_times,
								   self.start_dig_in_times, self.end_dig_in_times, self.segment_names, 
								   self.dig_in_names, self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds, 
								   self.current_corr_dir, self.neuron_keep_indices, self.segments_to_analyze) #For all neurons in dataset
		#Now plot and calculate significance!
		self.calculate_plot_corr_stats()
		self.calculate_significance()
		
	def calculate_correlations_selective(self,):
		print("\tCalculate correlations for taste selective neurons only")
		#Import taste selectivity data
		data_group_name = 'taste_selectivity'
		taste_select_neur_epoch_bin = af.pull_data_from_hdf5(self.hdf5_dir,data_group_name,'taste_select_neur_epoch_bin')[0]
		self.neuron_keep_indices = taste_select_neur_epoch_bin.T
		#Create storage directory
		self.current_corr_dir = self.corr_dir + 'taste_select_neur/'
		if os.path.isdir(self.current_corr_dir) == False:
			os.mkdir(self.current_corr_dir)
		#Calculate correlations
		df.calculate_vec_correlations(self.segment_dev_rasters, self.tastant_spike_times,
								   self.start_dig_in_times, self.end_dig_in_times, self.segment_names, 
								   self.dig_in_names, self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
								   self.current_corr_dir, self.neuron_keep_indices, self.segments_to_analyze) #For all neurons in dataset
		#Now plot and calculate significance!
		self.calculate_plot_corr_stats()
		self.calculate_significance()
		
	def calculate_correlations_all_zscore(self,):
		print("\tCalculate correlations for all neurons z-scored")
		#Create storage directory
		self.current_corr_dir = self.corr_dir + 'all_neur_zscore/'
		if os.path.isdir(self.current_corr_dir) == False:
			os.mkdir(self.current_corr_dir)
		self.neuron_keep_indices = np.ones((self.num_neur,self.num_cp+1))
		#Calculate correlations
		df.calculate_vec_correlations_zscore(self.segment_dev_rasters_zscore, self.tastant_spike_times,
								   self.start_dig_in_times, self.end_dig_in_times, self.segment_names, self.dig_in_names,
								   self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
								   self.current_corr_dir, self.neuron_keep_indices, self.segments_to_analyze)
		#Now plot and calculate significance!
		self.calculate_plot_corr_stats()
		self.calculate_significance()
		
	def calculate_correlations_selective_zscore(self,):
		print("\tCalculate correlations for taste selective neurons z-scored")
		#Create storage directory
		self.current_corr_dir = self.corr_dir + 'taste_select_neur_zscore/'
		if os.path.isdir(self.current_corr_dir) == False:
			os.mkdir(self.current_corr_dir)
		#Import taste selectivity data
		data_group_name = 'taste_selectivity'
		taste_select_neur_epoch_bin = af.pull_data_from_hdf5(self.hdf5_dir,data_group_name,'taste_select_neur_epoch_bin')[0]
		self.neuron_keep_indices = taste_select_neur_epoch_bin.T
		#Calculate correlations
		df.calculate_vec_correlations_zscore(self.segment_dev_rasters_zscore, self.tastant_spike_times,
								   self.start_dig_in_times, self.end_dig_in_times, self.segment_names, self.dig_in_names,
								   self.pre_taste, self.post_taste, self.pop_taste_cp_raster_inds,
								   self.current_corr_dir, self.neuron_keep_indices, self.segments_to_analyze)
		#Now plot and calculate significance!
		self.calculate_plot_corr_stats()
		self.calculate_significance()
		
	def calculate_plot_corr_stats(self,):
		#Plot dir setup
		plot_dir = self.current_corr_dir + 'plots/'
		if os.path.isdir(plot_dir) == False:
			os.mkdir(plot_dir)
		self.plot_dir = plot_dir
		#Calculate stats
		print("\tCalculating Correlation Statistics")
		corr_dev_stats = df.pull_corr_dev_stats(self.segment_names, self.dig_in_names, self.current_corr_dir, self.segments_to_analyze)
		print("Plotting Correlation Statistics")
		dpf.plot_stats(corr_dev_stats, self.segment_names, self.dig_in_names, self.plot_dir, 
				 'Correlation', self.neuron_keep_indices, self.segments_to_analyze)
		print("Plotting Combined Correlation Statistics")
		segment_pop_vec_data = dpf.plot_combined_stats(corr_dev_stats, self.segment_names, self.dig_in_names, \
													self.plot_dir, 'Correlation',self.neuron_keep_indices,self.segments_to_analyze)
		self.segment_pop_vec_data = segment_pop_vec_data
		df.top_dev_corr_bins(corr_dev_stats,self.segment_names,self.dig_in_names,self.plot_dir,self.neuron_keep_indices,self.segments_to_analyze)
		
	def calculate_significance(self,):
		print("\tCalculate statistical significance between correlation distributions.")
		self.current_stats_dir = self.current_corr_dir + 'stats/'
		if os.path.isdir(self.current_stats_dir) == False:
			os.mkdir(self.current_stats_dir)
		
		#KS-test
		df.stat_significance(self.segment_pop_vec_data, self.segment_names, self.dig_in_names, \
						  self.current_stats_dir, 'population_vec_correlation', self.segments_to_analyze)
		
		#T-test less
		df.stat_significance_ttest_less(self.segment_pop_vec_data, self.segment_names, \
									self. dig_in_names, self.current_stats_dir, 
									 'population_vec_correlation_ttest_less', self.segments_to_analyze)
		
		#T-test more
		df.stat_significance_ttest_more(self.segment_pop_vec_data, self.segment_names, \
									 self.dig_in_names, self.current_stats_dir, 
									 'population_vec_correlation_ttest_more', self.segments_to_analyze)
		
		#Mean compare
		df.mean_compare(self.segment_pop_vec_data, self.segment_names, self.dig_in_names, self.current_stats_dir, 'population_vec_mean_difference', self.segments_to_analyze)
		
	
	
		
		
		
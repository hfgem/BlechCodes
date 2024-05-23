#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:18:06 2024

@author: Hannah Germaine

Functions to support compare_conditions.py in running cross-dataset analyses.
"""

import os, warnings, easygui, pickle

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import numpy as np
warnings.filterwarnings("ignore")
import functions.analysis_funcs as af
import functions.dev_funcs as df
import functions.compare_conditions_funcs as ccf
import functions.compare_datasets_funcs as cdf

class run_compare_conditions_analysis():
	
	def __init__(self,args):
		self.all_data_dict = args[0]
		self.corr_dir = args[1]
		if len(self.corr_dir) > 0:
			self.import_corr()
		else:
			self.gather_data()
		self.find_groupings()
		self.plot_results()
		
	def import_corr(self,):
		"""Import previously saved correlation data"""
		dict_save_dir = os.path.join(self.corr_dir,'corr_data.pkl')
		file = open(dict_save_dir, 'rb')
		corr_data = pickle.load(file)
		file.close()
		self.corr_data = corr_data
		self.results_dir = self.corr_dir
		
	def gather_data(self,):
		"""Import the relevant data from each dataset to be analyzed. This 
		includes the number of neurons, segments to analyze, segment names, 
		segment start and end times, taste dig in names, and the correlation
		data for all neurons and taste-selective neurons"""
		
		num_datasets = len(self.all_data_dict)
		dataset_names = list(self.all_data_dict.keys())
		corr_data = dict()
		for n_i in range(num_datasets):
			data_name = dataset_names[n_i]
			data_dict = self.all_data_dict[data_name]['data']
			metadata = self.all_data_dict[data_name]['metadata']
			data_save_dir = data_dict['data_path']
			dev_corr_save_dir = os.path.join(data_save_dir,'dev_x_taste','corr')
			num_corr_types = os.listdir(dev_corr_save_dir)
			corr_data[data_name] = dict()
			corr_data[data_name]['num_neur'] = data_dict['num_neur']
			segments_to_analyze = metadata['params_dict']['segments_to_analyze']
			corr_data[data_name]['segments_to_analyze'] = segments_to_analyze
			corr_data[data_name]['segment_names'] = data_dict['segment_names']
			segment_times = data_dict['segment_times']
			num_segments = len(corr_data[data_name]['segment_names'])
			corr_data[data_name]['segment_times_reshaped'] = [[segment_times[i],segment_times[i+1]] for i in range(num_segments)]
			dig_in_names = data_dict['dig_in_names']
			corr_data[data_name]['dig_in_names'] = dig_in_names
			corr_data[data_name]['corr_data'] = dict()
			for nct_i in range(len(num_corr_types)):
				nct = num_corr_types[nct_i]
				result_dir = os.path.join(dev_corr_save_dir,nct)
				corr_data[data_name]['corr_data'][nct] = dict()
				for s_i in segments_to_analyze:
					segment_stats = dict()
					seg_name = corr_data[data_name]['segment_names'][s_i]
					corr_data[data_name]['corr_data'][nct][seg_name] = dict()
					for t_i in range(len(dig_in_names)):
						taste_name = dig_in_names[t_i]
						corr_data[data_name]['corr_data'][nct][seg_name][taste_name] = dict()
						try:
							filename_pop_vec = os.path.join(result_dir,seg_name + '_' + taste_name + '_pop_vec.npy')
							data = np.load(filename_pop_vec)
							corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['data'] = data
							num_dev, num_deliv, num_cp = np.shape(data)
							corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['num_dev'] = num_dev
							corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['num_deliv'] = num_deliv
							corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['num_cp'] = num_cp
						except:
							print("No data in directory " + result_dir)
		self.corr_data = corr_data	
		#Save the combined dataset somewhere...
		#_____Analysis Storage Directory_____
		print('Please select a directory to save all results from this set of analyses.')
		#DO NOT use the same directory as where the correlation analysis results are stored
		results_dir = easygui.diropenbox(title='Please select the storage folder.')
		#Save the dictionary of data
		dict_save_dir = os.path.join(results_dir,'corr_data.pkl')
		f = open(dict_save_dir,"wb")
		pickle.dump(corr_data,f)
		self.results_dir = results_dir

	def find_groupings(self,):
		"""Across the different datasets, get the unique data names/indices,
		correlation combinations and names/indices, unique segment names/indices,
		and unique taste names/indices to align datasets to each other in these
		different groups."""
		
		num_datasets = len(self.corr_data)
		corr_data = self.corr_data
		unique_given_names = list(corr_data.keys())
		unique_given_indices = np.sort(np.unique(unique_given_names, return_index=True)[1])
		unique_given_names = [unique_given_names[i] for i in unique_given_indices]
		unique_corr_names = np.array([list(corr_data[name]['corr_data'].keys()) for name in unique_given_names]).flatten() #How many types of correlation analyses
		unique_corr_indices = np.sort(np.unique(unique_corr_names, return_index=True)[1])
		unique_corr_names = [unique_corr_names[i] for i in unique_corr_indices]
		unique_segment_names = []
		unique_taste_names = []
		for name in unique_given_names:
			for corr_name in unique_corr_names:
				try:
					seg_names = list(corr_data[name]['corr_data'][corr_name].keys())
					unique_segment_names.extend(seg_names)
					for seg_name in seg_names:
						taste_names = list(corr_data[name]['corr_data'][corr_name][seg_name].keys())
						unique_taste_names.extend(taste_names)
				except:
					print(name + " does not have correlation data for " + corr_name)
		unique_segment_indices = np.sort(np.unique(unique_segment_names, return_index=True)[1])
		unique_segment_names = [unique_segment_names[i] for i in unique_segment_indices]
		unique_taste_indices = np.sort(np.unique(unique_taste_names, return_index=True)[1])
		unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
		
		self.unique_given_names = unique_given_names
		self.unique_corr_names = unique_corr_names
		self.unique_segment_names = unique_segment_names
		self.unique_taste_names = unique_taste_names
	
	def plot_results(self,):
		num_cond = len(self.corr_data)
		results_dir = self.results_dir
		
		print("Beginning Plots.")
		if num_cond > 1:
			#Cross-Dataset: different given names on the same axes
			#____Deviation Event Frequencies____
			dev_freq_dir = os.path.join(results_dir,'dev_frequency_plots')
			if os.path.isdir(dev_freq_dir) == False:
				 os.mkdir(dev_freq_dir)
			print("\tCalculating Cross-Taste Deviation Frequencies")
			taste_dev_freq_dir = os.path.join(dev_freq_dir,'cross_tastes')
			if os.path.isdir(taste_dev_freq_dir) == False:
				 os.mkdir(taste_dev_freq_dir)
			cdf.cross_dataset_dev_freq_taste(self.corr_data,self.unique_given_names,\
							self.unique_corr_names,self.unique_segment_names,\
								self.unique_taste_names,taste_dev_freq_dir)
			print("\tCalculating Cross-Segment Deviation Frequencies")
			seg_dev_freq_dir = os.path.join(dev_freq_dir,'cross_segments')
			if os.path.isdir(seg_dev_freq_dir) == False:
				 os.mkdir(seg_dev_freq_dir)
			cdf.cross_dataset_dev_freq_seg(self.corr_data,self.unique_given_names,\
							self.unique_corr_names,self.unique_segment_names,\
								self.unique_taste_names,seg_dev_freq_dir)
			#____Correlation Distributions____
			cross_segment_dir = os.path.join(results_dir,'cross_segment_plots')
			if os.path.isdir(cross_segment_dir) == False:
				 os.mkdir(cross_segment_dir)
			print("\tComparing Segments")
			cdf.cross_segment_diffs(self.corr_data,cross_segment_dir,self.unique_given_names,\
							self.unique_corr_names,self.unique_segment_names,self.unique_taste_names)
			cross_taste_dir = os.path.join(results_dir,'cross_taste_plots')
			if os.path.isdir(cross_taste_dir) == False:
				 os.mkdir(cross_taste_dir)
			print("\tComparing Tastes")
			cdf.cross_taste_diffs(self.corr_data,cross_taste_dir,self.unique_given_names,\
						  self.unique_corr_names,self.unique_segment_names,self.unique_taste_names)
			cross_epoch_dir = os.path.join(results_dir,'cross_epoch_plots')
			if os.path.isdir(cross_epoch_dir) == False:
				os.mkdir(cross_epoch_dir)
			print("\tComparing Epochs")
			cdf.cross_epoch_diffs(self.corr_data,cross_epoch_dir,self.unique_given_names,\
						 self.unique_corr_names,self.unique_segment_names,self.unique_taste_names)
		else:
			#Cross-Corr: all neur, taste selective, all neuron z-score, and taste selective z-score on same axes
			cross_corr_dir = os.path.join(results_dir,'cross_corr_plots')
			if os.path.isdir(cross_corr_dir) == False:
				os.mkdir(cross_corr_dir)
			print("\tCross Condition Plots.")
			ccf.cross_corr_name(self.corr_data,cross_corr_dir,self.unique_given_names,\
					   self.unique_corr_names,self.unique_segment_names,self.unique_taste_names)
			
			#Cross-Segment: different segments on the same axes
			cross_segment_dir = os.path.join(results_dir,'cross_segment_plots')
			if os.path.isdir(cross_segment_dir) == False:
				os.mkdir(cross_segment_dir)
			print("\tCross Segment Plots.")
			ccf.cross_segment(self.corr_data,cross_segment_dir,self.unique_given_names,\
					 self.unique_corr_names,self.unique_segment_names,self.unique_taste_names)
			
			#Cross-Taste: different tastes on the same axes
			cross_taste_dir = os.path.join(results_dir,'cross_taste_plots')
			if os.path.isdir(cross_taste_dir) == False:
				os.mkdir(cross_taste_dir)
			print("\tCross Taste Plots.")
			ccf.cross_taste(self.corr_data,cross_taste_dir,self.unique_given_names,\
				   self.unique_corr_names,self.unique_segment_names,self.unique_taste_names)
			
			#Cross-Epoch: different epochs on the same axes
			cross_epoch_dir = os.path.join(results_dir,'cross_epoch_plots')
			if os.path.isdir(cross_epoch_dir) == False:
				os.mkdir(cross_epoch_dir)
			print("\tCross Epoch Plots.")
			ccf.cross_epoch(self.corr_data,cross_epoch_dir,self.unique_given_names,\
				   self.unique_corr_names,self.unique_segment_names,self.unique_taste_names)
		
		print("Done.")
		
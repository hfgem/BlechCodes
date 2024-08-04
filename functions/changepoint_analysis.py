#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:37:28 2024

@author: Hannah Germaine

This is the second step of the analysis pipeline: changepoint detection and related
"""
import os
import numpy as np

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.analysis_funcs as af
import functions.hdf5_handling as hf5
import functions.changepoint_detection as cd
import functions.plot_funcs as pf
import functions.decoding_funcs as df

class run_changepoint_detection():
	
	def __init__(self,args):
		self.metadata = args[0]
		self.data_dict = args[1]
		self.get_changepoints()
		self.test_taste_similarity()
		self.test_taste_discriminability()
		self.test_neuron_taste_selectivity()
		
	def get_changepoints(self,):
		cp_bin = self.metadata['params_dict']['cp_bin']
		num_cp = self.metadata['params_dict']['num_cp']
		before_taste = np.ceil(self.metadata['params_dict']['pre_taste']*1000).astype('int') #Milliseconds before taste delivery to plot
		after_taste = np.ceil(self.metadata['params_dict']['post_taste']*1000).astype('int') #Milliseconds after taste delivery to plot
		hdf5_dir = self.metadata['hdf5_dir']
		tastant_spike_times = self.data_dict['tastant_spike_times']
		start_dig_in_times = self.data_dict['start_dig_in_times']
		end_dig_in_times = self.data_dict['end_dig_in_times']
		dig_in_names = self.data_dict['dig_in_names']
		
		#Set storage directory
		cp_save_dir = self.metadata['dir_name'] + 'Changepoint_Calculations/'
		if os.path.isdir(cp_save_dir) == False:
			os.mkdir(cp_save_dir)
		
		#_____All data_____
		taste_cp_save_dir = cp_save_dir + 'All_Taste_CPs/'
		if os.path.isdir(taste_cp_save_dir) == False:
			os.mkdir(taste_cp_save_dir)
		data_group_name = 'changepoint_data'
		#Raster Poisson Bayes Changepoint Calcs Indiv Neurons
		try:
# 			taste_cp_raster_inds = hf5.pull_data_from_hdf5(hdf5_dir,data_group_name,'taste_cp_raster_inds')
			pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(hdf5_dir,data_group_name,'pop_taste_cp_raster_inds')
		except:	
# 			taste_cp_raster_save_dir = taste_cp_save_dir + 'neur/'
# 			if os.path.isdir(taste_cp_raster_save_dir) == False:
# 				os.mkdir(taste_cp_raster_save_dir)
# 			taste_cp_raster_inds = cd.calc_cp_iter(tastant_spike_times,cp_bin,num_cp,start_dig_in_times,
# 						  end_dig_in_times,before_taste,after_taste,
# 						  dig_in_names,taste_cp_raster_save_dir)
# 			hf5.add_data_to_hdf5(hdf5_dir,data_group_name,'taste_cp_raster_inds',taste_cp_raster_inds)
# 			
			taste_cp_raster_pop_save_dir = taste_cp_save_dir + 'pop/'
			if os.path.isdir(taste_cp_raster_pop_save_dir) == False:
				os.mkdir(taste_cp_raster_pop_save_dir)
			pop_taste_cp_raster_inds = cd.calc_cp_iter_pop(tastant_spike_times,cp_bin,num_cp,start_dig_in_times,
						  end_dig_in_times,before_taste,after_taste,
						  dig_in_names,taste_cp_raster_pop_save_dir)
			hf5.add_data_to_hdf5(hdf5_dir,data_group_name,'pop_taste_cp_raster_inds',pop_taste_cp_raster_inds)
		self.pop_taste_cp_raster_inds = pop_taste_cp_raster_inds

	def test_taste_similarity(self,):
		hdf5_dir = self.metadata['hdf5_dir']
		num_tastes = self.data_dict['num_tastes']
		num_cp = self.metadata['params_dict']['num_cp']
		num_neur = self.data_dict['num_neur']
		num_segments = len(self.data_dict['segment_names'])
		tastant_spike_times = self.data_dict['tastant_spike_times']
		start_dig_in_times = self.data_dict['start_dig_in_times']
		end_dig_in_times = self.data_dict['end_dig_in_times']
		pop_taste_cp_raster_inds = self.pop_taste_cp_raster_inds
		post_taste_dt = int(np.ceil(self.metadata['params_dict']['post_taste']*(1000/1)))
		dig_in_names = self.data_dict['dig_in_names']
		taste_epoch_save_dir = self.metadata['dir_name'] + 'Taste_Delivery_Similarity/'
		if os.path.isdir(taste_epoch_save_dir) == False:
			os.mkdir(taste_epoch_save_dir)
		data_group_name = 'taste_similarity_data'
		try:
			epoch_trial_out_of_bounds = hf5.pull_data_from_hdf5(hdf5_dir,data_group_name,'epoch_out_of_bounds')
		except:
			epoch_trial_out_of_bounds = pf.taste_response_similarity_plots(num_tastes,num_cp,num_neur,num_segments,
											tastant_spike_times,start_dig_in_times,
											end_dig_in_times,pop_taste_cp_raster_inds,
											post_taste_dt,dig_in_names,taste_epoch_save_dir)
			hf5.add_data_to_hdf5(hdf5_dir,data_group_name,'epoch_out_of_bounds',epoch_trial_out_of_bounds)
				
	
	def test_taste_discriminability(self,):
		"""Run an ANOVA on time and taste to determine neuron-by-neuron taste
		discriminability across time"""
		#Grab variables
		hdf5_dir = self.metadata['hdf5_dir']
		start_dig_in_times = self.data_dict['start_dig_in_times']
		tastant_spike_times = self.data_dict['tastant_spike_times']
		num_tastes = self.data_dict['num_tastes']
		num_neur = self.data_dict['num_neur']
		post_taste_dt = int(np.ceil(self.metadata['params_dict']['post_taste']*(1000/1)))
		bin_size = self.metadata['params_dict']['cp_bin'] #ms to slide across - aligned with minimal state size
		
		#Set Storage Directory
		discrim_save_dir = self.metadata['dir_name'] + 'Taste_Delivery_Discriminability/'
		if os.path.isdir(discrim_save_dir) == False:
			os.mkdir(discrim_save_dir)
		#Get results
		data_group_name = 'taste_discriminability'
		try:
			anova_results_all = hf5.pull_data_from_hdf5(hdf5_dir,data_group_name,'anova_results_all')
			anova_results_true = hf5.pull_data_from_hdf5(hdf5_dir,data_group_name,'anova_results_true')
			peak_epochs = hf5.pull_data_from_hdf5(hdf5_dir,data_group_name,'peak_epochs')
			discrim_neur = hf5.pull_data_from_hdf5(hdf5_dir,data_group_name,'discrim_neur')
		except:
			anova_results_all, anova_results_true, peak_epochs, discrim_neur = af.taste_discriminability_test(post_taste_dt,
																		 num_tastes,tastant_spike_times,
																		 num_neur,start_dig_in_times,
																		 bin_size,discrim_save_dir)
			hf5.add_data_to_hdf5(hdf5_dir,data_group_name,'anova_results_all',anova_results_all)
			hf5.add_data_to_hdf5(hdf5_dir,data_group_name,'anova_results_true',anova_results_true)
			hf5.add_data_to_hdf5(hdf5_dir,data_group_name,'peak_epochs',peak_epochs)
			hf5.add_data_to_hdf5(hdf5_dir,data_group_name,'discrim_neur',discrim_neur)
	
	def test_neuron_taste_selectivity(self,):
		hdf5_dir = self.metadata['hdf5_dir']
		num_tastes = self.data_dict['num_tastes']
		tastant_spike_times = self.data_dict['tastant_spike_times']
		pop_taste_cp_raster_inds = self.pop_taste_cp_raster_inds
		start_dig_in_times = self.data_dict['start_dig_in_times']
		end_dig_in_times = self.data_dict['end_dig_in_times']
		dig_in_names = self.data_dict['dig_in_names']
		num_cp = self.metadata['params_dict']['num_cp'] + 1
		num_neur = self.data_dict['num_neur']
		pre_taste_dt = int(np.ceil(self.metadata['params_dict']['pre_taste']*(1000/1)))
		post_taste_dt = int(np.ceil(self.metadata['params_dict']['post_taste']*(1000/1)))
		
		data_group_name = 'taste_selectivity'
		
		decoding_save_dir = self.metadata['dir_name'] + 'Taste_Selectivity/'
		if os.path.isdir(decoding_save_dir) == False:
			os.mkdir(decoding_save_dir)
			
		#_____Calculate taste decoding probabilities and success probabilities_____
		try:
			taste_select_prob_epoch = hf5.pull_data_from_hdf5(hdf5_dir,data_group_name,'taste_select_prob_epoch')[0]
			p_taste_epoch = hf5.pull_data_from_hdf5(hdf5_dir,data_group_name,'p_taste_epoch')[0]
			taste_select_neur_epoch_bin = hf5.pull_data_from_hdf5(hdf5_dir,data_group_name,'taste_select_neur_epoch_bin')[0]
		except:
			print("\tUsing population changepoint indices to calculate taste selectivity by epoch.")
			p_taste_epoch, taste_select_prob_epoch = df.taste_decoding_cp(tastant_spike_times,\
														   pop_taste_cp_raster_inds,start_dig_in_times,end_dig_in_times,dig_in_names, \
															   num_neur,num_cp,num_tastes-1,pre_taste_dt,post_taste_dt,decoding_save_dir)
			#_____Calculate binary matrices of taste selective neurons / taste selective neurons by epoch_____
			#On average, does the neuron decode neurons more often than chance?
			taste_select_neur_epoch_count = np.zeros((num_cp,num_neur))
			for t_i in range(num_tastes-1):
				select_neur = (taste_select_prob_epoch[:,:,t_i] > 1/(num_tastes-1)).astype('int')
				taste_select_neur_epoch_count += select_neur
			taste_select_neur_epoch_bin = taste_select_neur_epoch_count > 1
			
			#Save
			hf5.add_data_to_hdf5(hdf5_dir,data_group_name,'p_taste_epoch',p_taste_epoch)
			hf5.add_data_to_hdf5(hdf5_dir,data_group_name,'taste_select_prob_epoch',taste_select_prob_epoch)
			hf5.add_data_to_hdf5(hdf5_dir,data_group_name,'taste_select_neur_epoch_bin',taste_select_neur_epoch_bin)
	
			
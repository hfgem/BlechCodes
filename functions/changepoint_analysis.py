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
import functions.changepoint_detection as cd

class run_changepoint_detection():
	
	def __init__(self,args):
		self.metadata = args[0]
		self.data_dict = args[1]
		self.get_changepoints()
		self.test_taste_similarity()
		
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
			taste_cp_raster_inds = af.pull_data_from_hdf5(hdf5_dir,data_group_name,'taste_cp_raster_inds')
			pop_taste_cp_raster_inds = af.pull_data_from_hdf5(hdf5_dir,data_group_name,'pop_taste_cp_raster_inds')
		except:	
			taste_cp_raster_save_dir = taste_cp_save_dir + 'neur/'
			if os.path.isdir(taste_cp_raster_save_dir) == False:
				os.mkdir(taste_cp_raster_save_dir)
			taste_cp_raster_inds = cd.calc_cp_iter(tastant_spike_times,cp_bin,num_cp,start_dig_in_times,
						  end_dig_in_times,before_taste,after_taste,
						  dig_in_names,taste_cp_raster_save_dir)
			af.add_data_to_hdf5(hdf5_dir,data_group_name,'taste_cp_raster_inds',taste_cp_raster_inds)
			
			taste_cp_raster_pop_save_dir = taste_cp_save_dir + 'pop/'
			if os.path.isdir(taste_cp_raster_pop_save_dir) == False:
				os.mkdir(taste_cp_raster_pop_save_dir)
			pop_taste_cp_raster_inds = cd.calc_cp_iter_pop(tastant_spike_times,cp_bin,num_cp,start_dig_in_times,
						  end_dig_in_times,before_taste,after_taste,
						  dig_in_names,taste_cp_raster_pop_save_dir)
			af.add_data_to_hdf5(hdf5_dir,data_group_name,'pop_taste_cp_raster_inds',pop_taste_cp_raster_inds)

	def test_taste_similarity(self,):
		
			
			
			
			
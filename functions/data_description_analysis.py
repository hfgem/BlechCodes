#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:44:41 2024

@author: Hannah Germaine

This is the first step of the analysis pipeline: PSTHs, Raster Plots, etc... are analyzed here
"""

import os
import numpy as np

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)
import functions.analysis_funcs as af
import functions.plot_funcs as pf
import functions.seg_compare as sc

class run_data_description_analysis():
	
	def __init__(self,args):
		self.metadata = args[0]
		self.data_dict = args[1]
		self.get_spike_time_datasets()
		self.get_psth_raster()
		self.seg_compare()
		
		
	def get_spike_time_datasets(self,):
	
		#_____Pull out spike times for all tastes (and no taste)_____
		segment_spike_times = af.calc_segment_spike_times(self.data_dict['segment_times'],self.data_dict['spike_times'],self.data_dict['num_neur'])
		tastant_spike_times = af.calc_tastant_spike_times(self.data_dict['segment_times'],self.data_dict['spike_times'],
														  self.data_dict['start_dig_in_times'],self.data_dict['end_dig_in_times'],
														  self.data_dict['pre_taste'],self.data_dict['post_taste'],self.data_dict['num_tastes'],self.data_dict['num_neur'])
		
		self.data_dict['segment_spike_times'] = segment_spike_times
		self.data_dict['tastant_spike_times'] = tastant_spike_times
		
	def get_psth_raster(self,):
		hdf5_dir = self.metadata['hdf5_dir']
		pre_taste_dt = int(np.ceil(self.data_dict['pre_taste']*(1000/1))) #Convert to ms timescale
		post_taste_dt = int(np.ceil(self.data_dict['post_taste']*(1000/1))) #Convert to ms timescale
		bin_width = 0.25 #Gaussian convolution kernel width in seconds
		bin_step = 25 #Step size in ms to take in PSTH calculation
		data_group_name = 'PSTH_data'
		try:
			tastant_PSTH = af.pull_data_from_hdf5(hdf5_dir,data_group_name,'tastant_PSTH')
			PSTH_times = af.pull_data_from_hdf5(hdf5_dir,data_group_name,'PSTH_times')
			PSTH_taste_deliv_times = af.pull_data_from_hdf5(hdf5_dir,data_group_name,'PSTH_taste_deliv_times')
			avg_tastant_PSTH = af.pull_data_from_hdf5(hdf5_dir,data_group_name,'avg_tastant_PSTH')
			print("PSTH data imported")
		except:
			"Calculating and plotting raster and PSTH data"
			data_save_dir = self.metadata['dir_name']
			start_dig_in_times = self.data_dict['start_dig_in_times']
			end_dig_in_times = self.data_dict['end_dig_in_times']
			segment_names = self.data_dict['segment_names']
			segment_times = self.data_dict['segment_times']
			segment_spike_times = self.data_dict['segment_spike_times']
			tastant_spike_times = self.data_dict['tastant_spike_times']
			pre_taste_dt = self.data_dict['pre_taste_dt']
			post_taste_dt = self.data_dict['post_taste_dt']
			num_neur = self.data_dict['num_neur']
			num_tastes = self.data_dict['num_tastes']
			dig_in_names = self.data_dict['dig_in_names']
			pf.raster_plots(data_save_dir, self.data_dict['dig_in_names'], start_dig_in_times, end_dig_in_times, 
							segment_names, segment_times, segment_spike_times,
							tastant_spike_times, pre_taste_dt, post_taste_dt,
							num_neur, num_tastes)
			PSTH_times, PSTH_taste_deliv_times, tastant_PSTH, avg_tastant_PSTH = pf.PSTH_plots(data_save_dir, num_tastes,
																							   num_neur, dig_in_names, 
																							   start_dig_in_times, end_dig_in_times, 
																							   pre_taste_dt, post_taste_dt, 
																							   segment_times, segment_spike_times,
																							   bin_width, bin_step)
			af.add_data_to_hdf5(hdf5_dir,data_group_name,'tastant_PSTH',tastant_PSTH)
			af.add_data_to_hdf5(hdf5_dir,data_group_name,'PSTH_times',PSTH_times)
			af.add_data_to_hdf5(hdf5_dir,data_group_name,'PSTH_taste_deliv_times',PSTH_taste_deliv_times)
			af.add_data_to_hdf5(hdf5_dir,data_group_name,'avg_tastant_PSTH',avg_tastant_PSTH)
			
	def seg_compare(self,):
		data_save_dir = self.metadata['dir_name']
		
		#_____Grab and plot firing rate distributions and comparisons (by segment)_____
		sc_save_dir = data_save_dir + 'Segment_Comparison/'
		if os.path.isdir(sc_save_dir) == False:
			os.mkdir(sc_save_dir)
			
		#All data
		all_sc_save_dir = sc_save_dir + 'All/'
		if os.path.isdir(all_sc_save_dir) == False:
			os.mkdir(all_sc_save_dir)
			sc.bin_spike_counts(all_sc_save_dir,self.data_dict['segment_spike_times'],self.data_dict['segment_names'],self.data_dict['segment_times'])
			
	
	
		
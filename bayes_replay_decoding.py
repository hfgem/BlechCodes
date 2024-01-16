#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 09:22:42 2024

@author: hannahgermaine

This code is written to perform Bayesian decoding of taste information outside 
of taste delivery intervals to determine potential replay events. The first 
pass will use a larger bin size, to hone in on regions of potential replay, and
the second pass will go into those regions and test smaller bins.

Assumes analyze_states.py was run first.
"""

if __name__ == '__main__':
	
	import os, tqdm
	file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
	os.chdir(file_path)
	import numpy as np
	import functions.hdf5_handling as hf5
	import functions.analysis_funcs as af
	import functions.decoding_funcs as df
	
	#_____Get the directory of the hdf5 file_____
	sorted_dir, segment_dir, cleaned_dir = hf5.sorted_data_import() #Program will automatically quit if file not found in given folder
	fig_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'
	print('\nData Directory:')
	print(fig_save_dir)
	
	#_____Import data_____
	#todo: update intan rhd file import code to accept directory input
	num_neur, all_waveforms, spike_times, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes = af.import_data(sorted_dir, segment_dir, fig_save_dir)
	
	#_____Calculate spike time datasets_____
	pre_taste = 0.5 #Seconds before tastant delivery to store
	post_taste = 2 #Seconds after tastant delivery to store
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int') #Milliseconds before taste delivery to plot
	post_taste_dt = np.ceil(post_taste*1000).astype('int') #Milliseconds after taste delivery to plot
	
	#_____Add "no taste" control segments to the dataset_____
	if dig_in_names[-1] != 'none':
		dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes = af.add_no_taste(start_dig_in_times, end_dig_in_times, post_taste, dig_in_names)

	segment_spike_times = af.calc_segment_spike_times(segment_times,spike_times,num_neur)
	tastant_spike_times = af.calc_tastant_spike_times(segment_times,spike_times,
													  start_dig_in_times,end_dig_in_times,
													  pre_taste,post_taste,num_tastes,num_neur)
	num_segments = len(segment_spike_times)
	segment_times_reshaped = [[segment_times[i],segment_times[i+1]] for i in range(num_segments)]
	
	#Raster Poisson Bayes Changepoint Calcs Indiv Neurons
	data_group_name = 'changepoint_data'
	taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_cp_raster_inds')
	pop_taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'pop_taste_cp_raster_inds')
	num_cp = np.shape(taste_cp_raster_inds[0])[-1] - 1
	
	bayes_dir = fig_save_dir + 'Bayes_Decoding/'
	if os.path.isdir(bayes_dir) == False:
		os.mkdir(bayes_dir)
	
	#_____DECODE ALL NEURONS_____
	print("\nDecoding using all neurons.\n")
	
	bayes_dir_all = bayes_dir + 'All_Neurons/'
	if os.path.isdir(bayes_dir_all) == False:
		os.mkdir(bayes_dir_all)
	
	#Get FR Distributions
	taste_select = np.ones(num_neur) #stand in to use full population
	taste_select_epoch = np.ones((num_cp,num_neur)) #stand in to use full population
	full_taste_fr_dist, tastant_fr_dist, tastant_fr_dist_pop, taste_num_deliv = df.taste_fr_dist(num_neur,
																							  num_cp,tastant_spike_times,
																							  taste_cp_raster_inds,pop_taste_cp_raster_inds,
																							  start_dig_in_times, pre_taste_dt, post_taste_dt)
	
	#Decode by segment for a sliding post-taste bin size first
	#___Phase 1: Decode using full taste response___
	skip_time = 0.05 #Seconds to skip forward in sliding bin
	skip_dt = np.ceil(skip_time*1000).astype('int')
	df.decode_phase_1(full_taste_fr_dist,segment_spike_times,post_taste_dt,
				   skip_dt,dig_in_names,segment_times,segment_names,
				   start_dig_in_times,taste_num_deliv,taste_select,bayes_dir_all)
#%%
	#___Phase 2: Decode using epoch-specific responses___
	e_skip_time = 0.01 #Seconds to skip forward in sliding bin
	e_skip_dt = np.ceil(e_skip_time*1000).astype('int')
	e_len_time = 0.05 #Seconds to decode
	e_len_dt = np.ceil(e_len_time*1000).astype('int')
	df.decode_phase_2(tastant_fr_dist,segment_spike_times,post_taste_dt,
					   skip_dt,e_skip_dt,e_len_dt,dig_in_names,segment_times,
					   segment_names,start_dig_in_times,taste_num_deliv,
					   taste_select_epoch,bayes_dir_all)

#%%	
	#_____DECODE TASTE SELECTIVE NEURONS_____
	print("\nNow decoding using only taste selective neurons.\n")
	
	data_group_name = 'taste_selectivity'
	try:
		taste_select_neur_bin = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_neur_bin')[0]
		taste_select_neur_epoch_bin = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_neur_epoch_bin')[0]
		#TODO: Pass these to the functions below and use them to reduce which neurons are being used in the calculations
	except:
		print("ERROR: No taste selective data.")
		quit()
	
	bayes_dir_select = bayes_dir + 'Taste_Selective/'
	if os.path.isdir(bayes_dir_select) == False:
		os.mkdir(bayes_dir_select)
	
	#Get FR Distributions
	full_taste_fr_dist, tastant_fr_dist, tastant_fr_dist_pop, taste_num_deliv = df.taste_fr_dist(num_neur,
																							  num_cp,tastant_spike_times,
																							  taste_cp_raster_inds,pop_taste_cp_raster_inds,
																							  start_dig_in_times, pre_taste_dt, post_taste_dt)
	
	#Decode by segment for a sliding post-taste bin size first
	#___Phase 1: Decode using full taste response___
	skip_time = 0.05 #Seconds to skip forward in sliding bin
	skip_dt = np.ceil(skip_time*1000).astype('int')
	df.decode_phase_1(full_taste_fr_dist,segment_spike_times,post_taste_dt,
				   skip_dt,dig_in_names,segment_times,segment_names,
				   start_dig_in_times,taste_num_deliv,taste_select_neur_bin,bayes_dir_all)

	#___Phase 2: Decode using epoch-specific responses___
	e_skip_time = 0.01 #Seconds to skip forward in sliding bin
	e_skip_dt = np.ceil(e_skip_time*1000).astype('int')
	e_len_time = 0.05 #Seconds to decode
	e_len_dt = np.ceil(e_len_time*1000).astype('int')
	df.decode_phase_2(tastant_fr_dist,segment_spike_times,post_taste_dt,
					   skip_dt,e_skip_dt,e_len_dt,dig_in_names,segment_times,
					   segment_names,start_dig_in_times,taste_num_deliv,
					   taste_select_neur_epoch_bin,bayes_dir_all)
	
	
	
	
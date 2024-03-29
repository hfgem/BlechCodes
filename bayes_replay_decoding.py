#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 09:22:42 2024

@author: hannahgermaine

This code is written to perform Bayesian decoding of taste information outside 
of taste delivery intervals to determine potential replay events. This code
assumes neurons are independent.

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
		
	#%% Set decoding variables
	
	#If first decoding full taste response
	use_full = 0 #Decode the full taste response first or not
	skip_time = 0.05 #Seconds to skip forward in sliding bin
	skip_dt = np.ceil(skip_time*1000).astype('int')
	
	#For epoch decoding
	e_skip_time = 0.01 #Seconds to skip forward in sliding bin
	e_skip_dt = np.ceil(e_skip_time*1000).astype('int')
	e_len_time = 0.05 #Seconds to decode
	e_len_dt = np.ceil(e_len_time*1000).astype('int')
	
	#Decoding settings
	neuron_count_thresh = 1/3 #Fraction of total population that must be active to consider a decoding event
	max_decode = 50 #number of example decodes to plot
	seg_stat_bin = 60*1000 #ms to bin segment for decoding counts in bins
	
	#Z-scoring settings
	bin_time = 0.1 #Seconds to skip forward in calculating firing rates
	bin_dt = np.ceil(bin_time*1000).astype('int')
	
	#%% Get FR distributions
	
	tastant_fr_dist_pop, max_hz_pop, taste_num_deliv = df.taste_fr_dist(num_neur,num_cp,tastant_spike_times,
														 taste_cp_raster_inds,pop_taste_cp_raster_inds,
														 start_dig_in_times, pre_taste_dt, post_taste_dt)
	
	tastant_fr_dist_pop_z, max_hz_pop, min_hz_pop, taste_num_deliv_z = df.taste_fr_dist_zscore(num_neur,num_cp,tastant_spike_times,
																segment_spike_times,segment_names,segment_times,
																taste_cp_raster_inds,pop_taste_cp_raster_inds,
																start_dig_in_times, pre_taste_dt, post_taste_dt, bin_dt)
	
	
	#%%
	
	#_____DECODE ALL NEURONS_____
	print("\nDecoding using all neurons.\n")
	
	bayes_dir_all = bayes_dir + 'All_Neurons/'
	if os.path.isdir(bayes_dir_all) == False:
		os.mkdir(bayes_dir_all)
	
	#Get FR Distributions
	taste_select = np.ones(num_neur) #stand in to use full population
	taste_select_epoch = np.ones((num_cp,num_neur)) #stand in to use full population
	
	#___Decode using epoch-specific responses___
	df.decode_epochs(tastant_fr_dist_pop,segment_spike_times,post_taste_dt,
					   skip_dt,e_skip_dt,e_len_dt,dig_in_names,segment_times,
					   segment_names,start_dig_in_times,taste_num_deliv,
					   taste_select_epoch,use_full,max_hz_pop,bayes_dir_all)
	
	
	#___Plot Results___
	df.plot_decoded(tastant_fr_dist_pop,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
					 e_skip_dt,e_len_dt,dig_in_names,segment_times,
					 segment_names,taste_num_deliv,taste_select_epoch,
					 use_full,bayes_dir_all,max_decode,max_hz_pop,seg_stat_bin,neuron_count_thresh)
	
	df.plot_decoded_func_p(tastant_fr_dist_pop,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
					 e_skip_dt,e_len_dt,dig_in_names,segment_times,
					 segment_names,taste_num_deliv,taste_select_epoch,
					 use_full,bayes_dir_all,max_decode,max_hz_pop,seg_stat_bin)
	
	df.plot_decoded_func_n(tastant_fr_dist_pop,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
					 e_skip_dt,e_len_dt,dig_in_names,segment_times,
					 segment_names,taste_num_deliv,taste_select_epoch,
					 use_full,bayes_dir_all,max_decode,max_hz_pop,seg_stat_bin)

#%%	
	#_____DECODE TASTE SELECTIVE NEURONS_____
	print("\nNow decoding using only taste selective neurons.\n")
	
	data_group_name = 'taste_selectivity'
	try:
		taste_select_neur_bin = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_neur_bin')[0]
		taste_select_neur_epoch_bin = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_neur_epoch_bin')[0]
	except:
		print("ERROR: No taste selective data.")
		quit()
	
	bayes_dir_select = bayes_dir + 'Taste_Selective/'
	if os.path.isdir(bayes_dir_select) == False:
		os.mkdir(bayes_dir_select)
	
	df.decode_epochs(tastant_fr_dist_pop,segment_spike_times,post_taste_dt,
					   skip_dt,e_skip_dt,e_len_dt,dig_in_names,segment_times,
					   segment_names,start_dig_in_times,taste_num_deliv,
					   taste_select_neur_epoch_bin,use_full,max_hz_pop,bayes_dir_select)
	
	#___Plot Results___
					
	df.plot_decoded(tastant_fr_dist_pop,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					     start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
						  e_skip_dt,e_len_dt,dig_in_names,segment_times,
						   segment_names,taste_num_deliv,taste_select_epoch,
						    use_full,bayes_dir_select,max_decode,max_hz_pop,seg_stat_bin,neuron_count_thresh)
	
	df.plot_decoded_func_p(tastant_fr_dist_pop,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
					 e_skip_dt,e_len_dt,dig_in_names,segment_times,
					 segment_names,taste_num_deliv,taste_select_epoch,
					 use_full,bayes_dir_select,max_decode,max_hz_pop,seg_stat_bin)
	
	df.plot_decoded_func_n(tastant_fr_dist_pop,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
					 e_skip_dt,e_len_dt,dig_in_names,segment_times,
					 segment_names,taste_num_deliv,taste_select_epoch,
					 use_full,bayes_dir_select,max_decode,max_hz_pop,seg_stat_bin)
	
#%%
	#_____DECODE ALL NEURONS Z-SCORED_____
	print("\nNow decoding using all neurons z-scored.\n")
	
	bayes_dir_all_z = bayes_dir + 'All_Neurons_ZScored/'
	if os.path.isdir(bayes_dir_all_z) == False:
		os.mkdir(bayes_dir_all_z)
	
	#Get FZ-Scored R Distributions
	taste_select = np.ones(num_neur) #stand in to use full population
	taste_select_epoch = np.ones((num_cp,num_neur)) #stand in to use full population
	
	df.decode_epochs_zscore(tastant_fr_dist_pop_z,segment_spike_times,post_taste_dt,
					   skip_dt,e_skip_dt,e_len_dt,dig_in_names,segment_times,bin_dt,
					   segment_names,start_dig_in_times,taste_num_deliv_z,
					   taste_select_epoch,use_full,max_hz_pop,min_hz_pop,bayes_dir_all_z)
	
	tastant_fr_dist_pop_z, max_hz_pop, min_hz_pop, taste_num_deliv_z
	
	#___Plot Results___
	df.plot_decoded(tastant_fr_dist_pop_z,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					     start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
						  e_skip_dt,e_len_dt,dig_in_names,segment_times,
						   segment_names,taste_num_deliv_z,taste_select_epoch,
						    use_full,bayes_dir_all_z,max_decode,max_hz_pop,seg_stat_bin,neuron_count_thresh)
	
	df.plot_decoded_func_p(tastant_fr_dist_pop_z,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
					 e_skip_dt,e_len_dt,dig_in_names,segment_times,
					 segment_names,taste_num_deliv_z,taste_select_epoch,
					 use_full,bayes_dir_all_z,max_decode,max_hz_pop,seg_stat_bin)
	
	df.plot_decoded_func_n(tastant_fr_dist_pop_z,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
					 e_skip_dt,e_len_dt,dig_in_names,segment_times,
					 segment_names,taste_num_deliv_z,taste_select_epoch,
					 use_full,bayes_dir_all_z,max_decode,max_hz_pop,seg_stat_bin)
	
	
#%%
	#_____DECODE TASTE SELECTIVE NEURONS Z-SCORED_____
	print("\nNow decoding using only taste selective neurons z-scored.\n")
	
	data_group_name = 'taste_selectivity'
	try:
		taste_select_neur_bin = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_neur_bin')[0]
		taste_select_neur_epoch_bin = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_neur_epoch_bin')[0]
	except:
		print("ERROR: No taste selective data.")
		quit()
	
	bayes_dir_select_z = bayes_dir + 'Taste_Selective_ZScored/'
	if os.path.isdir(bayes_dir_select_z) == False:
		os.mkdir(bayes_dir_select_z)
	
	df.decode_epochs_zscore(tastant_fr_dist_pop_z,segment_spike_times,post_taste_dt,
					   skip_dt,e_skip_dt,e_len_dt,dig_in_names,segment_times,bin_dt,
					   segment_names,start_dig_in_times,taste_num_deliv_z,
					   taste_select_neur_epoch_bin,use_full,max_hz_pop,min_hz_pop,bayes_dir_select_z)
	
	#___Plot Results___
	df.plot_decoded(tastant_fr_dist_pop_z,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					     start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
						  e_skip_dt,e_len_dt,dig_in_names,segment_times,
						   segment_names,taste_num_deliv_z,taste_select_epoch,
						    use_full,bayes_dir_select_z,max_decode,max_hz_pop,seg_stat_bin,neuron_count_thresh)
	
	df.plot_decoded_func_p(tastant_fr_dist_pop_z,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
					 e_skip_dt,e_len_dt,dig_in_names,segment_times,
					 segment_names,taste_num_deliv_z,taste_select_epoch,
					 use_full,bayes_dir_select_z,max_decode,max_hz_pop,seg_stat_bin)
	
	df.plot_decoded_func_n(tastant_fr_dist_pop_z,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
					 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
					 e_skip_dt,e_len_dt,dig_in_names,segment_times,
					 segment_names,taste_num_deliv_z,taste_select_epoch,
					 use_full,bayes_dir_select_z,max_decode,max_hz_pop,seg_stat_bin)
	
	
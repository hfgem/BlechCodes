#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:52:05 2023

@author: Hannah Germaine
This set of code calculates deviations in activity from a local mean. Deviations
are stored in .json files and then imported for comparison with taste segments.
"""

import os,json,gzip,itertools,tqdm
import numpy as np
import functions.analysis_funcs as af
import functions.dev_funcs as df
import functions.hdf5_handling as hf5
from multiprocessing import Pool
		
if __name__ == '__main__':
	
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
	
	#_____Add "no taste" control segments to the dataset_____
	if dig_in_names[-1] != 'none':
		dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes = af.add_no_taste(start_dig_in_times, end_dig_in_times, post_taste, dig_in_names)

	segment_spike_times = af.calc_segment_spike_times(segment_times,spike_times,num_neur)
	tastant_spike_times = af.calc_tastant_spike_times(segment_times,spike_times,
													  start_dig_in_times,end_dig_in_times,
													  pre_taste,post_taste,num_tastes,num_neur)
	
	#_____Calculate deviations for each segment_____
	num_segments = len(segment_spike_times)
	segment_times_reshaped = [[segment_times[i],segment_times[i+1]] for i in range(num_segments)]
	local_size = 60*1000 #local bin size to compare deviations (in number of ms = dt)
	min_dev_size = 50 #minimum bin size for a deviation (in number of ms = dt)
	
	#Create deviation storage directory
	dev_dir = fig_save_dir + 'Deviations/'
	if os.path.isdir(dev_dir) == False:
		os.mkdir(dev_dir)
	
	#_____Import or calculate deviations for all segments_____
	"""Deviations are calculated by (1) finding the prominence of min_dev_size 
	bin firing rates compared to firing rates from a local window of size local_size,
	(2) calculating the 90th percentile of positive prominence values, and (3) 
	pulling out those bins of time where the activity is above the 90th percentile
	prominence"""
	try: #test if the data exists by trying to import the last 
		filepath = dev_dir + segment_names[-1] + '/deviations.json'
		with gzip.GzipFile(filepath, mode="r") as f:
			json_bytes = f.read()
			json_str = json_bytes.decode('utf-8')
			data = json.loads(json_str)
	except:
		seg_dirs = []
		for s_i in range(num_segments):
			#create storage directory
			seg_dir = dev_dir + segment_names[s_i] + '/'
			if os.path.isdir(seg_dir) == False:
				os.mkdir(seg_dir)
			seg_dirs.append(seg_dir)
		print("Now calculating deviations")
		with Pool(processes=4) as pool:  # start 4 worker processes
			pool.map(df.run_dev_pull_parallelized, zip(segment_spike_times,
											 itertools.repeat(local_size),
											 itertools.repeat(min_dev_size),
											 segment_times_reshaped,
											 seg_dirs))
	print("Now importing calculated deviations")
	segment_deviations = []
	for s_i in tqdm.tqdm(range(num_segments)):
		filepath = dev_dir + segment_names[s_i] + '/deviations.json'
		with gzip.GzipFile(filepath, mode="r") as f:
			json_bytes = f.read()
			json_str = json_bytes.decode('utf-8')            
			data = json.loads(json_str) 
			segment_deviations.append(data)
	
	#_____Pull rasters of deviations and plot_____
	#Calculate segment deviation spikes
	print("Now pulling true deviation rasters")
	segment_dev_rasters, segment_dev_times = df.create_dev_rasters(num_segments, segment_spike_times, 
						   np.array(segment_times_reshaped), segment_deviations)
		
	#Plot deviations
	print("Now plotting deviations")
	df.plot_dev_rasters(segment_deviations,segment_spike_times,segment_dev_times,segment_times_reshaped,pre_taste,post_taste,segment_names,dev_dir)
	
	#_____Calculate segment deviation statistics - length,IDI_____
	print("Now calculating and plotting true deviation statistics")
	segment_length_dict, segment_IDI_dict, segment_num_spike_dict, segment_num_neur_dict = df.calculate_dev_stats(segment_dev_rasters,segment_dev_times,segment_names,dev_dir)
	
	#_____Import supporting data for analyses_____
	#Import taste responsivity data
	#data_group_name = 'taste_responsivity'
	#taste_responsive_ind = (af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_responsive_ind')[0]).astype('int')
	#most_taste_responsive_ind = (af.pull_data_from_hdf5(sorted_dir,data_group_name,'most_taste_responsive_ind')[0]).astype('int')	
	
	#Import changepoint data
	data_group_name = 'changepoint_data'
	taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_cp_raster_inds')
	pop_taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'pop_taste_cp_raster_inds')
	
	#Generate comparable binary matrix for regular data
	num_cp = np.shape(taste_cp_raster_inds[0])[-1] - 1
	for t_i in range(num_tastes-1):
		if np.shape(taste_cp_raster_inds[t_i+1])[-1] - 1 < num_cp:
			num_cp = np.shape(taste_cp_raster_inds[t_i+1])[-1] - 1
	all_neur_binary = np.ones((num_neur,num_cp))
	
	#_____Calculate correlation between taste and deviation rasters for individual neurons_____
	#Create directory to store analysis results
	comp_dir = fig_save_dir + 'dev_x_taste/'
	if os.path.isdir(comp_dir) == False:
		os.mkdir(comp_dir)
		
	#Calculate correlation of true data deviation rasters with taste response rasters
	corr_dir = comp_dir + 'corr/'
	if os.path.isdir(corr_dir) == False:
		os.mkdir(corr_dir)
	df.calculate_correlations(segment_dev_rasters, tastant_spike_times,
							   start_dig_in_times, end_dig_in_times, segment_names, 
							   dig_in_names, pre_taste, post_taste, taste_cp_raster_inds, 
							   pop_taste_cp_raster_inds, corr_dir) #For all neurons in dataset
	corr_dev_stats = df.pull_corr_dev_stats(segment_names, dig_in_names, corr_dir)
	
	#_____Plot and statistically evaluate individual neuron correlation data_____
	#__________For all neurons__________
	all_neur_corr_dir = corr_dir + 'all_neur/'
	if os.path.isdir(all_neur_corr_dir) == False:
		os.mkdir(all_neur_corr_dir)
	df.plot_stats(corr_dev_stats, segment_names, dig_in_names, all_neur_corr_dir, 'Correlation',all_neur_binary)
	segment_corr_data, segment_corr_data_avg = df.plot_combined_stats(corr_dev_stats, segment_names, dig_in_names, all_neur_corr_dir, 'Correlation',all_neur_binary)
	df.top_dev_corr_bins(corr_dev_stats,segment_names,dig_in_names,all_neur_corr_dir,all_neur_binary)
	#Calculate pairwise significance
	#KS-test
	df.stat_significance(segment_corr_data, segment_names, dig_in_names, all_neur_corr_dir, 'neuron_correlation')
	df.stat_significance(segment_corr_data_avg, segment_names, dig_in_names, all_neur_corr_dir, 'population_avg_correlation')
	#T-test
	df.stat_significance_ttest(segment_corr_data, segment_names, dig_in_names, all_neur_corr_dir, 'neuron_correlation_ttest_less')
	df.stat_significance_ttest(segment_corr_data_avg, segment_names, dig_in_names, all_neur_corr_dir, 'population_avg_correlation_ttest_less')
	#Mean compare
	df.mean_compare(segment_corr_data, segment_names, dig_in_names, all_neur_corr_dir, 'neuron_mean_difference')
	df.mean_compare(segment_corr_data_avg, segment_names, dig_in_names, all_neur_corr_dir, 'population_avg_mean_difference')
	
	#__________For taste selective neurons__________
	
	#Import taste selectivity data
	data_group_name = 'taste_selectivity'
	taste_response_prob = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_response_prob')[0]
	taste_select_prob = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob')[0]
	taste_response_prob_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_response_prob_epoch')[0]
	taste_select_prob_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob_epoch')[0]
	
	taste_select_corr_dir = corr_dir + 'taste_select_neur/'
	if os.path.isdir(taste_select_corr_dir) == False:
		os.mkdir(taste_select_corr_dir)
	df.plot_stats(corr_dev_stats,segment_names,dig_in_names,taste_select_corr_dir,'Correlation',most_taste_selective_binary)
	segment_corr_data, segment_corr_data_avg = df.plot_combined_stats(corr_dev_stats, segment_names, dig_in_names, taste_select_corr_dir, 'Correlation',most_taste_selective_binary)
	df.top_dev_corr_bins(corr_dev_stats,segment_names,dig_in_names,taste_select_corr_dir,most_taste_selective_binary)
	#Calculate pairwise significance
	#KS-test
	df.stat_significance(segment_corr_data, segment_names, dig_in_names, taste_select_corr_dir, 'neuron_correlation')
	df.stat_significance(segment_corr_data_avg, segment_names, dig_in_names, taste_select_corr_dir, 'population_avg_correlation')
	#T-test
	df.stat_significance_ttest(segment_corr_data, segment_names, dig_in_names, taste_select_corr_dir, 'neuron_correlation_ttest')
	df.stat_significance_ttest(segment_corr_data_avg, segment_names, dig_in_names, taste_select_corr_dir, 'population_avg_correlation_ttest')
	#Mean compare
	df.mean_compare(segment_corr_data, segment_names, dig_in_names, taste_select_corr_dir, 'neuron_mean_difference')
	df.mean_compare(segment_corr_data_avg, segment_names, dig_in_names, taste_select_corr_dir, 'population_avg_mean_difference')
	
	#_____Calculate correlation between taste and deviation rasters for population_____
	#Calculate population correlation of true data deviation rasters with taste response rasters'
	corr_pop_dir = comp_dir + 'corr_pop/'
	if os.path.isdir(corr_pop_dir) == False:
		os.mkdir(corr_pop_dir)
	df.calculate_correlations_pop(segment_dev_rasters, tastant_spike_times,
							   start_dig_in_times, end_dig_in_times, segment_names, dig_in_names,
							   pre_taste, post_taste, taste_cp_raster_inds, corr_pop_dir,
							   neuron_keep_indices=[])
	all_neur_pop_corr_dir = corr_dir + 'all_neur/'
	if os.path.isdir(all_neur_pop_corr_dir) == False:
		os.mkdir(all_neur_pop_corr_dir)
	corr_dev_stats_pop = df.pull_corr_dev_stats_pop(segment_names, dig_in_names, corr_pop_dir)
	segment_pop_corr_data, segment_pop_corr_data_avg = df.plot_combined_stats_pop(corr_dev_stats_pop, segment_names, dig_in_names, all_neur_pop_corr_dir, 
							'Correlation')
	
	
	df.stat_significance(segment_corr_data, segment_names, dig_in_names, taste_select_corr_dir, 'neuron_correlation')
	
#%%
	#Calculate distance of true data deviation rasters from taste response rasters
	#dist_dir = comp_dir + 'dist/' #Create distance directory if doesn't exist
	#if os.path.isdir(dist_dir) == False:
	#	os.mkdir(dist_dir)
	#df.calculate_distances(segment_dev_rasters, tastant_spike_times,
	#						   start_dig_in_times, end_dig_in_times, segment_names,
	#						   dig_in_names, pre_taste, post_taste, 
	#						   taste_cp_raster_inds, dist_dir) #for all neurons in dataset
	#Plot distance calculations
	#dist_dev_stats = df.pull_corr_dev_stats(segment_names, dig_in_names, dist_dir)
	#df.plot_stats(dist_dev_stats, segment_names, dig_in_names, dist_dir, 'Distance')
	#segment_dist_data = df.plot_combined_stats(dist_dev_stats, segment_names, dig_in_names, dist_dir, 'Distance')
	#df.stat_significance(segment_corr_data, segment_names, dig_in_names, dist_dir, 'Distance')
	
	#Import null datasets for deviation analyses
	#null_dir = fig_save_dir + 'null_data/' #This should exist from compare_null.py - make sure that was run before running this script or it'll throw an error!
	#null_dev_dir = dev_dir + 'null_data/' #Create null deviation storage directory
	#if os.path.isdir(null_dev_dir) == False:
	#	os.mkdir(null_dev_dir)
	#for s_i in range(num_segments):
	#	print(segment_names[s_i] + ' TruexNull Statistics:')
	#	print("\tCalculating null distribution deviations for segment " + segment_names[s_i])
	#	seg_null_dir = null_dir + segment_names[s_i] + '/'
	#	null_files = os.listdir(seg_null_dir)
	#	num_null = 0
	#	null_names = []
	#	null_dev_save_dirs = []
	#	for n_f in null_files:
	#		if n_f[-4:] == 'json':
	#			num_null += 1
	#		null_names.append(str(num_null-1))
	#		seg_dirs = []
	#		null_dev_save_dir = null_dev_dir + segment_names[s_i]
	#		if os.path.isdir(null_dev_save_dir) == False:
	#			os.mkdir(null_dev_save_dir)
	#		null_dev_save_dirs.append(null_dev_save_dir + '/null_' + str(num_null-1) + '_')	
	#	null_segment_spikes = []
	#	for n_i in range(num_null):
	#		filepath = seg_null_dir + 'null_' + str(n_i) + '.json'
	#		with gzip.GzipFile(filepath, mode="r") as f:
	#			json_bytes = f.read()
	#			json_str = json_bytes.decode('utf-8')            
	#			data = json.loads(json_str) 
	#			null_segment_spikes.append(data)
	#	try:
	#		filepath = null_dev_save_dirs[-1] + 'deviations.json'
	#		with gzip.GzipFile(filepath, mode="r") as f:
	#			json_bytes = f.read()
	#			json_str = json_bytes.decode('utf-8')            
	#			data = json.loads(json_str)
	#	except:
	#		with Pool(processes=4) as pool: # start 4 worker processes
	#			pool.map(df.run_dev_pull_parallelized,zip(null_segment_spikes, 
	#											 itertools.repeat(local_size), 
	#											 itertools.repeat(min_dev_size),
	#											 itertools.repeat(segment_times_reshaped[s_i]),
	#											 null_dev_save_dirs))
	#	null_deviations = []
	#	for n_i in tqdm.tqdm(range(num_null)):
	#		filepath = null_dev_save_dirs[n_i] + 'deviations.json'
	#		with gzip.GzipFile(filepath, mode="r") as f:
	#			json_bytes = f.read()
	#			json_str = json_bytes.decode('utf-8')            
	#			data = json.loads(json_str) 
	#			null_deviations.append(data)
	#			
	#	null_segment_times = segment_times[s_i:s_i+2] * np.ones((num_null,2))
	#			
	#	#Calculate segment deviation spikes
	#	print("\tNow pulling null deviation rasters for segment " + segment_names[s_i])
	#	null_dev_rasters, null_dev_times = df.create_dev_rasters(num_null, null_segment_spikes, 
	#						   null_segment_times, null_deviations)
	#		
	#	#Calculate segment deviation statistics - length,IDI,counts
	#	print("\tNow calculating null deviation statistics for segment " + segment_names[s_i])
	#	null_length_dict, null_IDI_dict, null_num_spike_dict, null_num_neur_dict = df.calculate_dev_stats(null_dev_rasters,null_dev_times,null_names,null_dev_dir)
	#	
	#	#Plot null vs true statistics
	#	print("\tNow plotting true x null statistics for segment " + segment_names[s_i])
	#	df.plot_null_v_true_stats(segment_length_dict[s_i],null_length_dict,
	#						segment_names[s_i] + ' deviation lengths',dev_dir,x_label='length (ms)')
	#	df.plot_null_v_true_stats(segment_IDI_dict[s_i],null_IDI_dict,
	#						segment_names[s_i] + ' inter-deviation-intervals (IDIs)',dev_dir,x_label='time (ms)')
	#	df.plot_null_v_true_stats(segment_num_spike_dict[s_i],null_num_spike_dict,
	#						segment_names[s_i] + ' deviation spike counts',dev_dir,x_label='spike count')
	#	df.plot_null_v_true_stats(segment_num_neur_dict[s_i],null_num_neur_dict,
	#						segment_names[s_i] + ' deviation neuron counts',dev_dir,x_label='neuron count')
	#	
	#	#Calculate null correlations with taste responses
		
		
	#For each set of changepoint options calculate the correlation of segment devs with each epoch
	
	
	#For each set of changepoint options calculate the correlation of null segment devs with each epoch
	
	
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
		
	#Create folder to store correlation results
	corr_dir = comp_dir + 'corr/'
	if os.path.isdir(corr_dir) == False:
		os.mkdir(corr_dir)
#%%	
	#_____Calculate, plot, and statistically evaluate individual neuron correlation data_____
	
	#__________For all neurons__________
	all_neur_corr_dir = corr_dir + 'all_neur/'
	if os.path.isdir(all_neur_corr_dir) == False:
		os.mkdir(all_neur_corr_dir)
		
	df.calculate_correlations(segment_dev_rasters, tastant_spike_times,
							   start_dig_in_times, end_dig_in_times, segment_names, 
							   dig_in_names, pre_taste, post_taste, taste_cp_raster_inds, 
							   pop_taste_cp_raster_inds, all_neur_corr_dir) #For all neurons in dataset
	df.calculate_vec_correlations(segment_dev_rasters, tastant_spike_times,
							   start_dig_in_times, end_dig_in_times, segment_names, 
							   dig_in_names, pre_taste, post_taste, taste_cp_raster_inds, 
							   pop_taste_cp_raster_inds, all_neur_corr_dir) #For all neurons in dataset
	corr_dev_stats = df.pull_corr_dev_stats(segment_names, dig_in_names, all_neur_corr_dir)
	all_neur_plot_dir = all_neur_corr_dir + 'plots/'
	if os.path.isdir(all_neur_plot_dir) == False:
		os.mkdir(all_neur_plot_dir)
	df.plot_stats(corr_dev_stats, segment_names, dig_in_names, all_neur_plot_dir, 'Correlation',all_neur_binary)
	segment_corr_data, segment_corr_data_avg, segment_corr_pop_data, segment_pop_vec_data = df.plot_combined_stats(corr_dev_stats, \
																								segment_names, dig_in_names, all_neur_plot_dir, \
																								'Correlation',all_neur_binary)
	df.top_dev_corr_bins(corr_dev_stats,segment_names,dig_in_names,all_neur_plot_dir,all_neur_binary)
	
	#Calculate pairwise significance
	all_neur_stats_dir = all_neur_corr_dir + 'stats/'
	if os.path.isdir(all_neur_stats_dir) == False:
		os.mkdir(all_neur_stats_dir)
	
	#KS-test
	df.stat_significance(segment_corr_data, segment_names, dig_in_names, all_neur_stats_dir, 'neuron_correlation')
	df.stat_significance(segment_corr_data_avg, segment_names, dig_in_names, all_neur_stats_dir, 'population_avg_correlation')
	df.stat_significance(segment_corr_pop_data, segment_names, dig_in_names, all_neur_stats_dir, 'population_correlation')
	df.stat_significance(segment_pop_vec_data, segment_names, dig_in_names, all_neur_stats_dir, 'population_vec_correlation')
	
	#T-test less
	df.stat_significance_ttest_less(segment_corr_data, segment_names, dig_in_names, all_neur_stats_dir, 'neuron_correlation_ttest_less')
	df.stat_significance_ttest_less(segment_corr_data_avg, segment_names, dig_in_names, all_neur_stats_dir, 'population_avg_correlation_ttest_less')
	df.stat_significance_ttest_less(segment_corr_pop_data, segment_names, dig_in_names, all_neur_stats_dir, 'population_correlation_ttest_less')
	df.stat_significance_ttest_less(segment_pop_vec_data, segment_names, dig_in_names, all_neur_stats_dir, 'population_vec_correlation_ttest_less')
	
	#T-test more
	df.stat_significance_ttest_more(segment_corr_data, segment_names, dig_in_names, all_neur_stats_dir, 'neuron_correlation_ttest_more')
	df.stat_significance_ttest_more(segment_corr_data_avg, segment_names, dig_in_names, all_neur_stats_dir, 'population_avg_correlation_ttest_more')
	df.stat_significance_ttest_more(segment_corr_pop_data, segment_names, dig_in_names, all_neur_stats_dir, 'population_correlation_ttest_more')
	df.stat_significance_ttest_more(segment_pop_vec_data, segment_names, dig_in_names, all_neur_stats_dir, 'population_vec_correlation_ttest_more')
	
	#Mean compare
	df.mean_compare(segment_corr_data, segment_names, dig_in_names, all_neur_stats_dir, 'neuron_mean_difference')
	df.mean_compare(segment_corr_data_avg, segment_names, dig_in_names, all_neur_stats_dir, 'population_avg_mean_difference')
	df.mean_compare(segment_corr_pop_data, segment_names, dig_in_names, all_neur_stats_dir, 'population_mean_difference')
	df.mean_compare(segment_pop_vec_data, segment_names, dig_in_names, all_neur_stats_dir, 'population_vec_mean_difference')

#%%	
	#__________For taste selective neurons__________
	
	#Import taste selectivity data
	data_group_name = 'taste_selectivity'
	#taste_response_prob = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_response_prob')[0]
	#taste_select_prob = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob')[0]
	taste_response_prob_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_response_prob_epoch')[0]
	taste_select_prob_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob_epoch')[0]
	
	taste_select_corr_dir = corr_dir + 'taste_select_neur/'
	if os.path.isdir(taste_select_corr_dir) == False:
		os.mkdir(taste_select_corr_dir)
	
	df.calculate_correlations(segment_dev_rasters, tastant_spike_times,
							   start_dig_in_times, end_dig_in_times, segment_names, 
							   dig_in_names, pre_taste, post_taste, taste_cp_raster_inds, 
							   pop_taste_cp_raster_inds, taste_select_corr_dir, taste_select_prob_epoch)
	df.calculate_vec_correlations(segment_dev_rasters, tastant_spike_times,
							   start_dig_in_times, end_dig_in_times, segment_names, 
							   dig_in_names, pre_taste, post_taste, taste_cp_raster_inds, 
							   pop_taste_cp_raster_inds, taste_select_corr_dir, taste_select_prob_epoch) #For all neurons in dataset
	corr_dev_stats = df.pull_corr_dev_stats(segment_names, dig_in_names, taste_select_corr_dir)
	
	taste_select_neur_plot_dir = taste_select_corr_dir + 'plots/'
	if os.path.isdir(taste_select_neur_plot_dir) == False:
		os.mkdir(taste_select_neur_plot_dir)
	df.plot_stats(corr_dev_stats, segment_names, dig_in_names, taste_select_neur_plot_dir, 'Correlation',all_neur_binary)
	segment_corr_data, segment_corr_data_avg, segment_corr_pop_data, segment_pop_vec_data = df.plot_combined_stats(corr_dev_stats, \
																								segment_names, dig_in_names, taste_select_neur_plot_dir, \
																								'Correlation',all_neur_binary)
	df.top_dev_corr_bins(corr_dev_stats,segment_names,dig_in_names,taste_select_neur_plot_dir,all_neur_binary)
	
	#Calculate pairwise significance
	taste_select_neur_stats_dir = taste_select_corr_dir + 'stats/'
	if os.path.isdir(taste_select_neur_stats_dir) == False:
		os.mkdir(taste_select_neur_stats_dir)
	
	#KS-test
	df.stat_significance(segment_corr_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'neuron_correlation')
	df.stat_significance(segment_corr_data_avg, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_avg_correlation')
	df.stat_significance(segment_corr_pop_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_correlation')
	df.stat_significance(segment_pop_vec_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_vec_correlation')
	
	#T-test less
	df.stat_significance_ttest_less(segment_corr_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'neuron_correlation_ttest_less')
	df.stat_significance_ttest_less(segment_corr_data_avg, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_avg_correlation_ttest_less')
	df.stat_significance_ttest_less(segment_corr_pop_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_correlation_ttest_less')
	df.stat_significance_ttest_less(segment_pop_vec_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_vec_correlation_ttest_less')
	
	#T-test more
	df.stat_significance_ttest_more(segment_corr_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'neuron_correlation_ttest_more')
	df.stat_significance_ttest_more(segment_corr_data_avg, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_avg_correlation_ttest_more')
	df.stat_significance_ttest_more(segment_corr_pop_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_correlation_ttest_more')
	df.stat_significance_ttest_more(segment_pop_vec_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_vec_correlation_ttest_more')
	
	#Mean compare
	df.mean_compare(segment_corr_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'neuron_mean_difference')
	df.mean_compare(segment_corr_data_avg, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_avg_mean_difference')
	df.mean_compare(segment_corr_pop_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_mean_difference')
	df.mean_compare(segment_pop_vec_data, segment_names, dig_in_names, taste_select_neur_stats_dir, 'population_vec_mean_difference')

	
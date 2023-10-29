#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:58:10 2023

@author: Hannah Germaine
Calculate the correlation results for the taste selective population of neurons.
Must be run following find_deviations.py
"""
		
if __name__ == '__main__':

	import os,json,gzip,itertools,tqdm
	import numpy as np
	import functions.analysis_funcs as af
	import functions.dev_funcs as df
	import functions.dev_plot_funcs as dpf
	import functions.hdf5_handling as hf5
	from multiprocessing import Pool
	import functions.corr_dist_calc_parallel as cdcp
	import functions.corr_dist_calc_parallel_pop as cdcpp
	
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
	#pre_taste will be used for z-scoring as well
	z_bin = 0.05 #Seconds bin for z-scoring
	
	#_____Add "no taste" control segments to the dataset_____
	if dig_in_names[-1] != 'none':
		dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes = af.add_no_taste(start_dig_in_times, end_dig_in_times, post_taste, dig_in_names)

	segment_spike_times = af.calc_segment_spike_times(segment_times,spike_times,num_neur)
	tastant_spike_times = af.calc_tastant_spike_times(segment_times,spike_times,
													  start_dig_in_times,end_dig_in_times,
													  pre_taste,post_taste,num_tastes,num_neur)
	num_segments = len(segment_spike_times)
	segment_times_reshaped = [[segment_times[i],segment_times[i+1]] for i in range(num_segments)]
	
	#Deviation storage directory
	dev_dir = fig_save_dir + 'Deviations/'
	
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
	_, segment_dev_times, segment_dev_rasters_zscore = df.create_dev_rasters(num_segments, segment_spike_times, 
						   np.array(segment_times_reshaped), segment_deviations, pre_taste)
	
	#Import changepoint data
	data_group_name = 'changepoint_data'
	taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_cp_raster_inds')
	pop_taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'pop_taste_cp_raster_inds')
	
	#Generate comparable binary matrix for regular data
	num_cp = np.shape(taste_cp_raster_inds[0])[-1] - 1
	for t_i in range(num_tastes-1):
		if np.shape(taste_cp_raster_inds[t_i+1])[-1] - 1 < num_cp:
			num_cp = np.shape(taste_cp_raster_inds[t_i+1])[-1] - 1
	neuron_keep_indices = np.ones((num_neur,num_cp))

	#_____Calculate correlation between taste and deviation rasters for individual neurons_____
	#Create directory to store analysis results
	comp_dir = fig_save_dir + 'dev_x_taste/'
	if os.path.isdir(comp_dir) == False:
		os.mkdir(comp_dir)
		
	#Create folder to store correlation results
	corr_dir = comp_dir + 'corr/'
	if os.path.isdir(corr_dir) == False:
		os.mkdir(corr_dir)
	
	#Import taste selectivity data
	data_group_name = 'taste_selectivity'
	#taste_response_prob = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_response_prob')[0]
	#taste_select_prob = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob')[0]
	#taste_response_prob_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_response_prob_epoch')[0]
	taste_select_prob_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob_epoch')[0]
	
	taste_select_z_corr_dir = corr_dir + 'taste_select_neur_zscore/'
	if os.path.isdir(taste_select_z_corr_dir) == False:
		os.mkdir(taste_select_z_corr_dir)
	
	df.calculate_correlations_zscore(segment_dev_rasters_zscore, tastant_spike_times,
							   start_dig_in_times, end_dig_in_times, segment_names, 
							   dig_in_names, pre_taste, post_taste, taste_cp_raster_inds, 
							   pop_taste_cp_raster_inds, taste_select_z_corr_dir, taste_select_prob_epoch) #For all neurons in dataset
	df.calculate_vec_correlations(segment_dev_rasters_zscore, tastant_spike_times,
							   start_dig_in_times, end_dig_in_times, segment_names, 
							   dig_in_names, pre_taste, post_taste, taste_cp_raster_inds, 
							   pop_taste_cp_raster_inds, taste_select_z_corr_dir, taste_select_prob_epoch) #For all neurons in dataset
	
	corr_dev_stats = df.pull_corr_dev_stats(segment_names, dig_in_names, taste_select_z_corr_dir)
	
	taste_select_neur_z_plot_dir = taste_select_z_corr_dir + 'plots/'
	if os.path.isdir(taste_select_neur_z_plot_dir) == False:
		os.mkdir(taste_select_neur_z_plot_dir)
	dpf.plot_stats(corr_dev_stats, segment_names, dig_in_names, taste_select_neur_z_plot_dir, 'Correlation',taste_select_prob_epoch)
	segment_corr_data, segment_corr_data_avg, segment_corr_pop_data, segment_pop_vec_data = dpf.plot_combined_stats(corr_dev_stats, \
																								segment_names, dig_in_names, taste_select_neur_z_plot_dir, \
																								'Correlation',taste_select_prob_epoch)
	df.top_dev_corr_bins(corr_dev_stats,segment_names,dig_in_names,taste_select_neur_z_plot_dir,taste_select_prob_epoch)
	
	#Calculate pairwise significance
	taste_select_z_stats_dir = taste_select_z_corr_dir + 'stats/'
	if os.path.isdir(taste_select_z_stats_dir) == False:
		os.mkdir(taste_select_z_stats_dir)
	
	#KS-test
	df.stat_significance(segment_corr_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'neuron_correlation')
	df.stat_significance(segment_corr_data_avg, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_avg_correlation')
	df.stat_significance(segment_corr_pop_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_correlation')
	df.stat_significance(segment_pop_vec_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_vec_correlation')
	
	#T-test less
	df.stat_significance_ttest_less(segment_corr_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'neuron_correlation_ttest_less')
	df.stat_significance_ttest_less(segment_corr_data_avg, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_avg_correlation_ttest_less')
	df.stat_significance_ttest_less(segment_corr_pop_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_correlation_ttest_less')
	df.stat_significance_ttest_less(segment_pop_vec_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_vec_correlation_ttest_less')
	
	#T-test more
	df.stat_significance_ttest_more(segment_corr_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'neuron_correlation_ttest_more')
	df.stat_significance_ttest_more(segment_corr_data_avg, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_avg_correlation_ttest_more')
	df.stat_significance_ttest_more(segment_corr_pop_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_correlation_ttest_more')
	df.stat_significance_ttest_more(segment_pop_vec_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_vec_correlation_ttest_more')
	
	#Mean compare
	df.mean_compare(segment_corr_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'neuron_mean_difference')
	df.mean_compare(segment_corr_data_avg, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_avg_mean_difference')
	df.mean_compare(segment_corr_pop_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_mean_difference')
	df.mean_compare(segment_pop_vec_data, segment_names, dig_in_names, taste_select_z_stats_dir, 'population_vec_mean_difference')
	
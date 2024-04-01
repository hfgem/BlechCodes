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
import functions.dev_plot_funcs as dpf
import functions.hdf5_handling as hf5
from multiprocessing import Pool
		
if __name__ == '__main__':
	
# 	#_____Get the directory of the hdf5 file_____
# 	sorted_dir, segment_dir, cleaned_dir = hf5.sorted_data_import() #Program will automatically quit if file not found in given folder
# 	fig_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'
# 	print('\nData Directory:')
# 	print(fig_save_dir)

# 	#_____Import data_____
# 	#todo: update intan rhd file import code to accept directory input
# 	num_neur, all_waveforms, spike_times, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes = af.import_data(sorted_dir, segment_dir, fig_save_dir)
# 	
# 	#_____Calculate spike time datasets_____
# 	pre_taste = 0.5 #Seconds before tastant delivery to store
# 	post_taste = 2 #Seconds after tastant delivery to store
# 	#pre_taste will also be used for the z-scoring of both deliveries and deviation bins
# 	#the firing rate binning will be used for z-scoring as well
# 	
# 	#Analysis/plotting params
# 	segments_to_analyze = np.array([0, 2, 4])
# 	max_plot = 50
# 	
# 	#_____Add "no taste" control segments to the dataset_____
# 	if dig_in_names[-1] != 'none':
# 		dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes = af.add_no_taste(start_dig_in_times, end_dig_in_times, post_taste, dig_in_names)

# 	segment_spike_times = af.calc_segment_spike_times(segment_times,spike_times,num_neur)
# 	tastant_spike_times = af.calc_tastant_spike_times(segment_times,spike_times,
# 													  start_dig_in_times,end_dig_in_times,
# 													  pre_taste,post_taste,num_tastes,num_neur)
	
# 	#_____Calculate deviations for each segment_____
# 	num_segments = len(segment_spike_times)
# 	segment_times_reshaped = [[segment_times[i],segment_times[i+1]] for i in range(num_segments)]
# 	local_size = 60*1000 #local bin size to compare deviations (in number of ms = dt)
# 	min_dev_size = 20 #minimum bin size for a deviation (in number of ms = dt)
# 	
# 	#Create deviation storage directory
# 	dev_dir = fig_save_dir + 'Deviations/'
# 	if os.path.isdir(dev_dir) == False:
# 		os.mkdir(dev_dir)
# 	
# 	#_____Import or calculate deviations for all segments_____
# 	"""Deviations are calculated by (1) finding the prominence of min_dev_size 
# 	bin firing rates compared to firing rates from a local window of size local_size,
# 	(2) calculating the 90th percentile of positive prominence values, and (3) 
# 	pulling out those bins of time where the activity is above the 90th percentile
# 	prominence"""
# 	try: #test if the data exists by trying to import the last 
# 		filepath = dev_dir + segment_names[-1] + '/deviations.json'
# 		with gzip.GzipFile(filepath, mode="r") as f:
# 			json_bytes = f.read()
# 			json_str = json_bytes.decode('utf-8')
# 			data = json.loads(json_str)
# 	except:
# 		seg_dirs = []
# 		for s_i in range(num_segments):
# 			#create storage directory
# 			seg_dir = dev_dir + segment_names[s_i] + '/'
# 			if os.path.isdir(seg_dir) == False:
# 				os.mkdir(seg_dir)
# 			seg_dirs.append(seg_dir)
# 		print("\nNow calculating deviations")
# 		with Pool(processes=4) as pool:  # start 4 worker processes
# 			pool.map(df.run_dev_pull_parallelized, zip(segment_spike_times,
# 											 itertools.repeat(local_size),
# 											 itertools.repeat(min_dev_size),
# 											 segment_times_reshaped,
# 											 seg_dirs))
# 		pool.close()
#%%

# 	print("Now importing calculated deviations")
# 	segment_deviations = []
# 	for s_i in tqdm.tqdm(range(num_segments)):
# 		filepath = dev_dir + segment_names[s_i] + '/deviations.json'
# 		with gzip.GzipFile(filepath, mode="r") as f:
# 			json_bytes = f.read()
# 			json_str = json_bytes.decode('utf-8')			
# 			data = json.loads(json_str) 
# 			segment_deviations.append(data)
# 	
# 	#_____Pull rasters of deviations and plot_____
# 	#Calculate segment deviation spikes
# 	print("Now pulling true deviation rasters")
# 	segment_dev_rasters, segment_dev_times, segment_dev_rasters_zscore = df.create_dev_rasters(num_segments, segment_spike_times, 
# 						   np.array(segment_times_reshaped), segment_deviations, pre_taste)
# 		
# 	#Plot deviations
# 	print("Now plotting deviations")
# 	dpf.plot_dev_rasters(segment_deviations,segment_spike_times,segment_dev_times,
# 					  segment_times_reshaped,pre_taste,post_taste,min_dev_size,
# 					  segment_names,dev_dir,segments_to_analyze,max_plot)
# 	
# 	#_____Calculate segment deviation statistics - length,IDI_____
# 	print("Now calculating and plotting true deviation statistics")
# 	segment_length_dict, segment_IDI_dict, segment_num_spike_dict, segment_num_neur_dict = df.calculate_dev_stats(segment_dev_rasters,
# 																					   segment_dev_times,segment_names,dev_dir,segments_to_analyze)

				
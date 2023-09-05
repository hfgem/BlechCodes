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
	
	data_group_name = 'PSTH_data'
	tastant_PSTH = af.pull_data_from_hdf5(sorted_dir,data_group_name,'tastant_PSTH')
	PSTH_times = af.pull_data_from_hdf5(sorted_dir,data_group_name,'PSTH_times')
	PSTH_taste_deliv_times = af.pull_data_from_hdf5(sorted_dir,data_group_name,'PSTH_taste_deliv_times')
	avg_tastant_PSTH = af.pull_data_from_hdf5(sorted_dir,data_group_name,'avg_tastant_PSTH')
	
	#_____Calculate spike time datasets_____
	pre_taste = 0.5 #Seconds before tastant delivery to store
	post_taste = 2 #Seconds after tastant delivery to store

	segment_spike_times = af.calc_segment_spike_times(segment_times,spike_times,num_neur)
	
	#_____Calculate deviations for each segment_____
	num_segments = len(segment_spike_times)
	segment_times_reshaped = [[segment_times[i],segment_times[i+1]] for i in range(num_segments)]
	local_size = 60*1000 #local bin size to compare deviations (in number of ms = dt)
	min_dev_size = 50 #minimum bin size for a deviation (in number of ms = dt)
	
	#Create deviation storage directory
	dev_dir = fig_save_dir + 'Deviations/'
	if os.path.isdir(dev_dir) == False:
		os.mkdir(dev_dir)
	
	#First import or calculate deviations for all segments
	"""Deviations are calculated by (1) finding the prominence of min_dev_size 
	bin firing rates compared to firing rates from a local window of size local_size,
	(2) calculating the 90th percentile of positive prominence values, and (3) 
	pulling out those bins of time where the activity is above the 90th percentile
	prominence"""
	try:
		filepath = dev_dir + segment_names[0] + '/deviations.json'
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
		with Pool(processes=4) as pool: # start 4 worker processes
			pool.map(df.run_dev_pull_parallelized,zip(segment_spike_times, 
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
	
	#Calculate segment deviation spikes
	print("Now pulling true deviation rasters")
	segment_dev_rasters, segment_dev_times = df.create_dev_rasters(num_segments, segment_spike_times, 
						   segment_times, segment_deviations)
		
	#Calculate segment deviation statistics - length,IDI
	print("Now calculating true deviation statistics")
	df.calculate_dev_stats(segment_dev_rasters,segment_dev_times,segment_names,dev_dir)
	
	#Import null datasets and calculate deviation bins + their statistics
	null_dir = fig_save_dir + 'null_data/' #This should exist from compare_null.py - make sure that was run before running this script or it'll throw an error!
	null_dev_dir = dev_dir + 'null_data/' #Create null deviation storage directory
	if os.path.isdir(null_dev_dir) == False:
		os.mkdir(null_dev_dir)
	for s_i in range(num_segments):
		seg_null_dir = null_dir + segment_names[s_i] + '/'
		null_files = os.listdir(seg_null_dir)
		num_null = 0
		null_names = []
		null_dev_save_dirs = []
		for n_f in null_files:
			if n_f[-4:] == 'json':
				num_null += 1
			null_names.append(str(num_null-1))
			seg_dirs = []
			null_dev_save_dir = null_dev_dir + segment_names[s_i]
			if os.path.isdir(null_dev_save_dir) == False:
				os.mkdir(null_dev_save_dir)
			null_dev_save_dirs.append(null_dev_save_dir + '/null_' + str(num_null-1))	
		null_segment_spikes = []
		for n_i in range(num_null):
			filepath = seg_null_dir + 'null_' + str(n_i) + '.json'
			with gzip.GzipFile(filepath, mode="r") as f:
				json_bytes = f.read()
				json_str = json_bytes.decode('utf-8')            
				data = json.loads(json_str) 
				null_segment_spikes.append(data)
		with Pool(processes=4) as pool: # start 4 worker processes
			pool.map(df.run_dev_pull_parallelized,zip(null_segment_spikes, 
											 itertools.repeat(local_size), 
											 itertools.repeat(min_dev_size),
											 segment_times_reshaped,
											 null_dev_save_dirs))
		null_deviations = []
		for n_i in tqdm.tqdm(range(num_null)):
			filepath = dev_dir + segment_names[s_i] + '/deviations.json'
			with gzip.GzipFile(filepath, mode="r") as f:
				json_bytes = f.read()
				json_str = json_bytes.decode('utf-8')            
				data = json.loads(json_str) 
				segment_deviations.append(data)
	
	#Import taste responsivity data
	data_group_name = 'taste_responsivity'
	taste_responsivity_probability = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_responsivity_probability')
	taste_responsivity_binary = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_responsivity_binary')
	
	#Next import changepoint times
	data_group_name = 'changepoint_data'
	taste_cp_PSTH_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_cp_PSTH_inds')
	taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_cp_raster_inds')
	taste_resp_taste_cp_PSTH_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_resp_taste_cp_PSTH_inds')
	taste_resp_taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_resp_taste_cp_raster_inds')
	most_taste_resp_taste_cp_PSTH_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'most_taste_resp_taste_cp_PSTH_inds')
	most_taste_resp_taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'most_taste_resp_taste_cp_raster_inds')
	
	#For each set of changepoint options calculate the correlation of segment devs with each epoch
	
	
	#For each set of changepoint options calculate the correlation of null segment devs with each epoch
	
	
	
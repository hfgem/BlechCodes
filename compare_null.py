#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:35:02 2023

@author: Hannah Germaine
Script to generate null distributions and compare them against the true data
"""

import itertools, tqdm
from multiprocessing import Pool
import os, json, gzip
file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
os.chdir(file_path)
import numpy as np
import scipy.stats as stats
from numba import jit
import functions.analysis_funcs as af
import functions.hdf5_handling as hf5
import functions.seg_compare as sc
from functions.null_distributions import run_null_create_parallelized

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

	segment_spike_times = af.calc_segment_spike_times(segment_times,spike_times,num_neur)
	
	#_____Generate null distributions for each segment_____
	num_segments = len(segment_spike_times)
	num_null = 50
	
	#Compare true and null
	count_cutoff = np.arange(np.ceil(num_neur/3).astype('int'),num_neur) #Calculate bins with these numbers of neurons firing together
	bin_size = 0.05 #Bin size for neuron cutoff

	#Null dir
	null_dir = fig_save_dir + 'null_data/'
	if os.path.isdir(null_dir) == False:
		os.mkdir(null_dir)
	
	#Figure save dir
	bin_dir = fig_save_dir + 'thresholded_deviations/'
	if os.path.isdir(bin_dir) == False:
		os.mkdir(bin_dir)
		
	try: #Import calculated dictionaries if they exist
		filepath = bin_dir + 'neur_count_dict.npy'
		neur_count_dict = np.load(filepath, allow_pickle=True).item()
		filepath = bin_dir + 'neur_spike_dict.npy'
		neur_spike_dict = np.load(filepath, allow_pickle=True).item()
		print('\t Imported thresholded datasets into memory')
			
	except: #Calculate dictionaries
		neur_count_dict = dict()
		neur_spike_dict = dict()
		#For each segment separately
		for s_i in tqdm.tqdm(range(num_segments)):
			seg_name = segment_names[s_i]
			print('\t Now Generating Null Distributions for Segment ' + seg_name)
			segment_spikes = segment_spike_times[s_i]
			segment_start_time = segment_times[s_i]
			segment_end_time = segment_times[s_i+1]
			segment_length = segment_end_time - segment_start_time
			#Segment save dir
			seg_null_dir = null_dir + segment_names[s_i] + '/'
			if os.path.isdir(seg_null_dir) == False:
				os.mkdir(seg_null_dir)
			#First test to see if null dataset is already stored in memory
			try:
				filepath = seg_null_dir + 'null_' + str(0) + '.json'
				with gzip.GzipFile(filepath, mode="r") as f:
					json_bytes = f.read()
					json_str = json_bytes.decode('utf-8')            
					data = json.loads(json_str)
				print('\t Now importing null dataset into memory')
				null_segment_spikes = []
				for n_i in tqdm.tqdm(range(num_null)):
					filepath = seg_null_dir + 'null_' + str(n_i) + '.json'
					with gzip.GzipFile(filepath, mode="r") as f:
						json_bytes = f.read()
						json_str = json_bytes.decode('utf-8')            
						data = json.loads(json_str) 
						null_segment_spikes.append(data)
			except:
				#First create a null distribution set
				with Pool(processes=4) as pool: # start 4 worker processes
					pool.map(run_null_create_parallelized,zip(np.arange(num_null), 
													 itertools.repeat(segment_spikes), 
													 itertools.repeat(segment_start_time), 
													 itertools.repeat(segment_end_time), 
													 itertools.repeat(seg_null_dir)))
				print('\t Now importing null dataset into memory')
				null_segment_spikes = []
				for n_i in tqdm.tqdm(range(num_null)):
					filepath = seg_null_dir + 'null_' + str(n_i) + '.json'
					with gzip.GzipFile(filepath, mode="r") as f:
						json_bytes = f.read()
						json_str = json_bytes.decode('utf-8')            
						data = json.loads(json_str) 
						null_segment_spikes.append(data)
			#Convert null data to binary spike matrix
			null_bin_spikes = []
			for n_n in range(num_null):
				null_spikes = null_segment_spikes[n_n]
				null_bin_spike = np.zeros((num_neur,segment_end_time-segment_start_time+1))
				for n_i in range(num_neur):
					spike_indices = (np.array(null_spikes[n_i]) - segment_start_time).astype('int')
					null_bin_spike[n_i,spike_indices] = 1
				null_bin_spikes.append(null_bin_spike)
			print('\t Now comparing spiking neuron count distributions')
			#Convert real data to a binary spike matrix
			bin_spike = np.zeros((num_neur,segment_end_time-segment_start_time+1))
			for n_i in range(num_neur):
				spike_indices = (np.array(segment_spikes[n_i]) - segment_start_time).astype('int')
				bin_spike[n_i,spike_indices] = 1
			true_neur_counts, true_spike_counts = sc.high_bins(bin_spike,segment_start_time,segment_end_time,bin_size,count_cutoff)
			null_neur_counts = dict()
			null_spike_counts = dict()
			for n_n in tqdm.tqdm(range(num_null)):
				null_neur_count, null_spike_count = sc.high_bins(null_bin_spikes[n_n],segment_start_time,segment_end_time,bin_size,count_cutoff)
				for key in null_neur_count.keys():
					if key in null_neur_counts.keys():
						null_neur_counts[key].append(null_neur_count[key])
					else:
						null_neur_counts[key] = [null_neur_count[key]]
				for key in null_spike_count.keys():
					if key in null_spike_counts.keys():
						null_spike_counts[key].append(null_spike_count[key])
					else:
						null_spike_counts[key] = [null_spike_count[key]]
			#Calculate the neuron count data
			true_x_vals = np.array([(np.ceil(float(key))).astype('int') for key in true_neur_counts.keys()])
			true_neur_count_array = np.array([true_neur_counts[key] for key in true_neur_counts.keys()])
			neur_count_dict[seg_name + '_true'] =  [list(true_x_vals),
										   list(true_neur_count_array)]
			null_x_vals = np.array([(np.ceil(float(key))).astype('int') for key in null_neur_counts.keys()])
			mean_null_neur_counts = np.array([np.mean(null_neur_counts[key]) for key in null_neur_counts.keys()])
			std_null_neur_counts = np.array([np.std(null_neur_counts[key]) for key in null_neur_counts.keys()])
			neur_count_dict[seg_name + '_null'] =  [list(null_x_vals),
										   list(mean_null_neur_counts),
										   list(std_null_neur_counts)]
			#Plot the neuron count data
			sc.plot_high_bins_truexnull(true_x_vals,null_x_vals,true_neur_count_array,mean_null_neur_counts,
							   std_null_neur_counts,segment_length,bin_dir,'Neuron Counts',seg_name)
			#Calculate percentile of true data point in null data distribution
			percentiles = []
			for key in true_neur_counts.keys():
				try:
					percentiles.extend([stats.percentileofscore(null_neur_counts[key],true_neur_counts[key])])
				except:
					percentiles.extend([100])
			neur_count_dict[seg_name + '_percentile'] =  [list(true_x_vals),percentiles]
			#Calculate the spike count data
			true_x_vals = np.array([(np.ceil(float(key))).astype('int') for key in true_spike_counts.keys()])
			true_spike_count_array = np.array([true_spike_counts[key] for key in true_spike_counts.keys()])
			neur_spike_dict[seg_name + '_true'] =  [list(true_x_vals),
										   list(true_spike_count_array)]
			null_x_vals = np.array([(np.ceil(float(key))).astype('int') for key in null_spike_counts.keys()])
			mean_null_spike_counts = np.array([np.mean(null_spike_counts[key]) for key in null_spike_counts.keys()])
			std_null_spike_counts = np.array([np.std(null_spike_counts[key]) for key in null_spike_counts.keys()])
			neur_spike_dict[seg_name + '_null'] =  [list(null_x_vals),
										   list(mean_null_spike_counts),
										   list(std_null_spike_counts)]
			#Plot the spike count data
			sc.plot_high_bins_truexnull(true_x_vals,null_x_vals,true_spike_count_array,mean_null_spike_counts,
							   std_null_spike_counts,segment_length,bin_dir,'Spike Counts',seg_name)
			#Calculate percentile of true data point in null data distribution
			percentiles = []
			for key in true_spike_counts.keys():
				try:
					percentiles.extend([stats.percentileofscore(null_spike_counts[key],true_spike_counts[key])])
				except:
					percentiles.extend([100])
			neur_count_dict[seg_name + '_percentile'] =  [list(true_x_vals),percentiles]
		
		#Save the dictionaries
		filepath = bin_dir + 'neur_count_dict.npy'
		np.save(filepath, neur_count_dict) 
		filepath = bin_dir + 'neur_spike_dict.npy'
		np.save(filepath, neur_spike_dict) 

		
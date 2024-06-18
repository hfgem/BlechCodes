#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:42:59 2024

@author: hannahgermaine

Compare true deviation events against null deviation events
"""

from multiprocessing import Pool
import os, json, gzip, itertools, tqdm
file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
os.chdir(file_path)
import numpy as np
import scipy.stats as stats
import functions.analysis_funcs as af
import functions.hdf5_handling as hf5
import functions.null_distributions as nd
import functions.dev_funcs as df

if __name__ == '__main__':
	
# 	#_____Get the directory of the hdf5 file_____
# 	sorted_dir, segment_dir, cleaned_dir = hf5.sorted_data_import() #Program will automatically quit if file not found in given folder
# 	fig_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'
# 	print('\nData Directory:')
# 	print(fig_save_dir)

# 	#_____Import data_____
# 	#todo: update intan rhd file import code to accept directory input
# 	num_neur, all_waveforms, spike_times, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes = af.import_data(sorted_dir, segment_dir, fig_save_dir)

# 	#_____Calculate spike time datasets_____
# 	pre_taste = 0.5 #Seconds before tastant delivery to store
# 	post_taste = 2 #Seconds after tastant delivery to store
# 	segments_to_analyze = np.array([0, 2, 4])

# 	#_____Add "no taste" control segments to the dataset_____
# 	if dig_in_names[-1] != 'none':
# 		dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes = af.add_no_taste(start_dig_in_times, end_dig_in_times, post_taste, dig_in_names)

# 	segment_spike_times = af.calc_segment_spike_times(segment_times,spike_times,num_neur)
# 	
# 	#_____Calculate deviations for each segment_____
# 	num_segments = len(segment_spike_times)
# 	segment_times_reshaped = [[segment_times[i],segment_times[i+1]] for i in range(num_segments)]
# 	local_size = 60*1000 #local bin size to compare deviations (in number of ms = dt)
# 	min_dev_size = 50 #minimum bin size for a deviation (in number of ms = dt)
# 	
# 	#_____Generate null distributions for each segment_____
# 	num_null = 50
# 	
# 	max_plot = 50
# 	
# 	#Compare true and null
# 	count_cutoff = np.arange(1,num_neur) #Calculate bins with these numbers of neurons firing together
# 	bin_size = 0.05 #Bin size for neuron cutoff
# 	lag_vals = np.arange(2,100).astype('int')

# 	#Create deviation storage directory
# 	dev_dir = fig_save_dir + 'Deviations/'
# 	if os.path.isdir(dev_dir) == False:
# 		os.mkdir(dev_dir)

# 	#Null dir
# 	null_dir = fig_save_dir + 'null_data/'
# 	if os.path.isdir(null_dir) == False:
# 		os.mkdir(null_dir)
# 	
# 	#Figure save dir
# 	bin_dir = fig_save_dir + 'thresholded_deviations/'
# 	if os.path.isdir(bin_dir) == False:
# 		os.mkdir(bin_dir)	
#%%	
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
# 		
# 	#_____Import or calculate null deviations for all segments_____
# 	try: #test if the data exists by trying to import the last 
# 		filepath = dev_dir + 'null_data/' + segment_names[-1] + '/null_0_deviations.json'
# 		with gzip.GzipFile(filepath, mode="r") as f:
# 			json_bytes = f.read()
# 			json_str = json_bytes.decode('utf-8')
# 			data = json.loads(json_str)
# 	except:
# 		print("\nNow calculating deviations")
# 		for null_i in range(num_null):
# 			seg_dirs = []
# 			for s_i in range(num_segments):
# 				#create storage directory
# 				seg_dir = dev_dir + 'null_data/' + segment_names[s_i] + '/'
# 				if os.path.isdir(seg_dir) == False:
# 					os.mkdir(seg_dir)
# 				seg_dir = dev_dir + 'null_data/' + segment_names[s_i] + '/null_' + str(null_i) + '_'
# 				seg_dirs.append(seg_dir)
# 			with Pool(processes=4) as pool:  # start 4 worker processes
# 				pool.map(df.run_dev_pull_parallelized, zip(segment_spike_times,
# 												 itertools.repeat(local_size),
# 												 itertools.repeat(min_dev_size),
# 												 segment_times_reshaped,
# 												 seg_dirs))
# 			pool.close()
# 	
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
# 	print("Now importing calculated null deviations")
# 	all_null_deviations = []
# 	for null_i in tqdm.tqdm(range(num_null)):
# 		null_segment_deviations = []
# 		for s_i in range(num_segments):
# 			filepath = dev_dir + 'null_data/' + segment_names[s_i] + '/null_' + str(null_i) + '_deviations.json'
# 			with gzip.GzipFile(filepath, mode="r") as f:
# 				json_bytes = f.read()
# 				json_str = json_bytes.decode('utf-8')			
# 				data = json.loads(json_str) 
# 				null_segment_deviations.append(data)
# 		all_null_deviations.append(null_segment_deviations)
# 	del null_i, null_segment_deviations, s_i, filepath, json_bytes, json_str, data
# 	
# 	#Calculate segment deviation spikes
# 	print("Now pulling true deviation rasters")
# 	segment_dev_rasters, segment_dev_times, segment_dev_rasters_zscore = df.create_dev_rasters(num_segments, segment_spike_times, 
# 						   np.array(segment_times_reshaped), segment_deviations, pre_taste)
# 		
# 	#Calculate segment deviation spikes
# 	print("Now pulling null deviation rasters")
# 	null_dev_rasters = []
# 	null_dev_times = []
# 	null_segment_dev_rasters_zscore = []
# 	for null_i in tqdm.tqdm(range(num_null)):
# 		null_segment_deviations = all_null_deviations[null_i]
# 		null_segment_dev_rasters_i, null_segment_dev_times_i, null_segment_dev_rasters_zscore_i = df.create_dev_rasters(num_segments, segment_spike_times, 
# 							   np.array(segment_times_reshaped), null_segment_deviations, pre_taste)
# 		null_dev_rasters.append(null_segment_dev_rasters_i)
# 		null_dev_times.append(null_segment_dev_times_i)
# 		null_segment_dev_rasters_zscore.append(null_segment_dev_rasters_zscore_i)
# 		
# 	del all_null_deviations
		
#%%
# 	#_____Calculate Burst Event Statistics_____
# 	try: #Import calculated dictionaries if they exist
# 		filepath = bin_dir + 'neur_count_dict.npy'
# 		neur_count_dict = np.load(filepath, allow_pickle=True).item()
# 		filepath = bin_dir + 'neur_spike_dict.npy'
# 		neur_spike_dict = np.load(filepath, allow_pickle=True).item()
# 		print('\t Imported thresholded datasets into memory')
# 	except: #Calculate dictionaries
# 		neur_count_dict = dict()
# 		neur_spike_dict = dict()
# 		neur_len_dict = dict()
# 		for s_i in tqdm.tqdm(segments_to_analyze):
# 			#Gather data / parameters
# 			seg_name = segment_names[s_i]
# 			segment_spikes = segment_spike_times[s_i]
# 			segment_start_time = segment_times[s_i]
# 			segment_end_time = segment_times[s_i+1]
# 			segment_length = segment_end_time - segment_start_time
# 			#CALCULATE STATS
# 			#_____Gather null data deviation event stats_____
# 			null_dev_lengths = []
# 			null_dev_neuron_counts = []
# 			null_dev_spike_counts = []
# 			for null_i in range(num_null):
# 				all_rast = null_dev_rasters[null_i][s_i]
# 				null_i_num_neur = []
# 				null_i_num_spikes = []
# 				for nr in range(len(all_rast)):
# 					num_spikes_n_i = np.sum(all_rast[nr],1)
# 					num_spikes_i = np.sum(num_spikes_n_i)
# 					null_i_num_spikes.append(num_spikes_i)
# 					num_neur_i = np.sum((num_spikes_n_i > 0).astype('int'))
# 					null_i_num_neur.append(num_neur_i)
# 				all_len = null_dev_times[null_i][s_i][1,:] - null_dev_times[null_i][s_i][0,:]
# 				null_dev_neuron_counts.append(null_i_num_neur)
# 				null_dev_spike_counts.append(null_i_num_spikes)
# 				null_dev_lengths.append(all_len)
# 			
# 			#_____Gather true data deviation event stats_____
# 			true_dev_neuron_counts = []
# 			true_dev_spike_counts = []
# 			all_rast = segment_dev_rasters[s_i]
# 			for nr in range(len(all_rast)):
# 				num_spikes_n_i = np.sum(all_rast[nr],1)
# 				num_spikes_i = np.sum(num_spikes_n_i)
# 				true_dev_spike_counts.append(num_spikes_i)
# 				num_neur_i = np.sum((num_spikes_n_i > 0).astype('int'))
# 				true_dev_neuron_counts.append(num_neur_i)
# 			true_dev_lengths = segment_dev_times[s_i][1,:] - segment_dev_times[s_i][0,:]
# 			#_____Gather data as dictionary of number of events as a function of cutoff
# 			#Neuron count data
# 			null_max_neur_count = np.max([np.max(null_dev_neuron_counts[null_i]) for null_i in range(num_null)])
# 			max_neur_count = int(np.max([np.max(null_max_neur_count),np.max(true_dev_neuron_counts)]))
# 			neur_x_vals = np.arange(10,max_neur_count)
# 			true_neur_x_val_counts = np.zeros(np.shape(neur_x_vals))
# 			null_neur_x_val_counts_all = []
# 			null_neur_x_val_counts_mean = np.zeros(np.shape(neur_x_vals))
# 			null_neur_x_val_counts_std = np.zeros(np.shape(neur_x_vals))
# 			for n_cut_i, n_cut in enumerate(neur_x_vals):
# 				true_neur_x_val_counts[n_cut_i] = np.sum((np.array(true_dev_neuron_counts) > n_cut).astype('int'))
# 				null_neur_x_val_counts = []
# 				for null_i in range(num_null):
# 					null_neur_x_val_counts.append(np.sum((np.array(null_dev_neuron_counts[null_i]) > n_cut).astype('int')))
# 				null_neur_x_val_counts_all.append(null_neur_x_val_counts)
# 				null_neur_x_val_counts_mean[n_cut_i] = np.mean(null_neur_x_val_counts)
# 				null_neur_x_val_counts_std[n_cut_i] = np.std(null_neur_x_val_counts)
# 			neur_count_dict[seg_name + '_true'] =  [list(neur_x_vals),
# 										   list(true_neur_x_val_counts)]
# 			neur_count_dict[seg_name + '_null'] =  [list(neur_x_vals),
# 										   list(null_neur_x_val_counts_mean),
# 										   list(null_neur_x_val_counts_std)]
# 			percentiles = [] #Calculate percentile of true data point in null data distribution
# 			for n_cut in neur_x_vals:
# 				try:
# 					percentiles.extend([round(stats.percentileofscore(null_neur_x_val_counts_all[n_cut-1],true_neur_x_val_counts[n_cut-1]),2)])
# 				except:
# 					percentiles.extend([100])
# 			neur_count_dict[seg_name + '_percentile'] =  [list(neur_x_vals),percentiles]
# 			
# 			#Spike count data
# 			null_max_neur_spikes = np.max([np.max(null_dev_spike_counts[null_i]) for null_i in range(num_null)])
# 			max_spike_count = int(np.max([np.max(null_max_neur_spikes),np.max(true_dev_spike_counts)]))
# 			spike_x_vals = np.arange(1,max_spike_count)
# 			true_neur_x_val_spikes = np.zeros(np.shape(spike_x_vals))
# 			null_neur_x_val_spikes_all = []
# 			null_neur_x_val_spikes_mean = np.zeros(np.shape(spike_x_vals))
# 			null_neur_x_val_spikes_std = np.zeros(np.shape(spike_x_vals))
# 			for s_cut_i, s_cut in enumerate(spike_x_vals):
# 				true_neur_x_val_spikes[s_cut_i] = np.sum((np.array(true_dev_spike_counts) > s_cut).astype('int'))
# 				null_neur_x_val_spikes = []
# 				for null_i in range(num_null):
# 					null_neur_x_val_spikes.append(np.sum((np.array(null_dev_spike_counts[null_i]) > s_cut).astype('int')))
# 				null_neur_x_val_spikes_all.append(null_neur_x_val_spikes)
# 				null_neur_x_val_spikes_mean[s_cut_i] = np.mean(null_neur_x_val_spikes)
# 				null_neur_x_val_spikes_std[s_cut_i] = np.std(null_neur_x_val_spikes)
# 			neur_spike_dict[seg_name + '_true'] =  [list(spike_x_vals),
# 										   list(true_neur_x_val_spikes)]
# 			neur_spike_dict[seg_name + '_null'] =  [list(spike_x_vals),
# 										   list(null_neur_x_val_spikes_mean),
# 										   list(null_neur_x_val_spikes_std)]
# 			percentiles = [] #Calculate percentile of true data point in null data distribution
# 			for s_cut in spike_x_vals:
# 				try:
# 					percentiles.extend([round(stats.percentileofscore(null_neur_x_val_spikes_all[s_cut-1],true_neur_x_val_spikes[s_cut-1]),2)])
# 				except:
# 					percentiles.extend([100])
# 			neur_spike_dict[seg_name + '_percentile'] =  [list(spike_x_vals),percentiles]
# 			
# 			
# 			#Burst length data
# 			null_max_neur_len = np.max([np.max(null_dev_lengths[null_i]) for null_i in range(num_null)])
# 			max_len = int(np.max([np.max(null_max_neur_len),np.max(true_dev_lengths)]))
# 			len_x_vals = np.arange(1,max_len)
# 			true_neur_x_val_lengths = np.zeros(np.shape(len_x_vals))
# 			null_neur_x_val_lengths_all = []
# 			null_neur_x_val_lengths_mean = np.zeros(np.shape(len_x_vals))
# 			null_neur_x_val_lengths_std = np.zeros(np.shape(len_x_vals))
# 			for l_cut_i, l_cut in enumerate(len_x_vals):
# 				true_neur_x_val_lengths[l_cut_i] = np.sum((np.array(true_dev_lengths) > l_cut).astype('int'))
# 				null_neur_x_val_lengths = []
# 				for null_i in range(num_null):
# 					null_neur_x_val_lengths.append(np.sum((np.array(null_dev_lengths[null_i]) > l_cut).astype('int')))
# 				null_neur_x_val_lengths_all.append(null_neur_x_val_lengths)
# 				null_neur_x_val_lengths_mean[l_cut_i] = np.mean(null_neur_x_val_lengths)
# 				null_neur_x_val_lengths_std[l_cut_i] = np.std(null_neur_x_val_lengths)
# 			neur_len_dict[seg_name + '_true'] =  [list(len_x_vals),
# 										   list(true_neur_x_val_lengths)]
# 			neur_len_dict[seg_name + '_null'] =  [list(len_x_vals),
# 										   list(null_neur_x_val_lengths_mean),
# 										   list(null_neur_x_val_lengths_std)]
# 			percentiles = [] #Calculate percentile of true data point in null data distribution
# 			for l_cut in len_x_vals:
# 				try:
# 					percentiles.extend([round(stats.percentileofscore(null_neur_x_val_lengths_all[l_cut-1],true_neur_x_val_lengths[l_cut-1]),2)])
# 				except:
# 					percentiles.extend([100])
# 			neur_len_dict[seg_name + '_percentile'] =  [list(len_x_vals),percentiles]
# 		
# 		#Save the dictionaries
# 		filepath = bin_dir + 'neur_count_dict.npy'
# 		np.save(filepath, neur_count_dict)
# 		filepath = bin_dir + 'neur_spike_dict.npy'
# 		np.save(filepath, neur_spike_dict) 
# 		filepath = bin_dir + 'neur_len_dict.npy'
# 		np.save(filepath, neur_len_dict)

# 	#_____Plotting_____
# 	neur_true_count_x = []
# 	neur_true_count_vals = []
# 	neur_null_count_x = []
# 	neur_null_count_mean = []
# 	neur_null_count_std = []
# 	neur_true_spike_x = []
# 	neur_true_spike_vals = []
# 	neur_null_spike_x = []
# 	neur_null_spike_mean = []
# 	neur_null_spike_std = []
# 	neur_true_len_x = []
# 	neur_true_len_vals = []
# 	neur_null_len_x = []
# 	neur_null_len_mean = []
# 	neur_null_len_std = []
# 	for s_i in tqdm.tqdm(segments_to_analyze):
# 		seg_name = segment_names[s_i]
# 		segment_start_time = segment_times[s_i]
# 		segment_end_time = segment_times[s_i+1]
# 		segment_length = segment_end_time - segment_start_time
# 		neur_true_count_data = neur_count_dict[seg_name + '_true']
# 		neur_null_count_data = neur_count_dict[seg_name + '_null']
# 		percentile_count_data = neur_count_dict[seg_name + '_percentile']
# 		neur_true_spike_data = neur_spike_dict[seg_name + '_true']
# 		neur_null_spike_data = neur_spike_dict[seg_name + '_null']
# 		percentile_spike_data = neur_spike_dict[seg_name + '_percentile']
# 		neur_true_len_data = neur_len_dict[seg_name + '_true']
# 		neur_null_len_data = neur_len_dict[seg_name + '_null']
# 		percentile_len_data = neur_len_dict[seg_name + '_percentile']
# 		#Plot the neuron count data
# 		norm_val = segment_length/1000 #Normalizing the number of bins to number of bins / second
# 		nd.plot_indiv_truexnull(np.array(neur_true_count_data[0]),np.array(neur_null_count_data[0]),np.array(neur_true_count_data[1]),np.array(neur_null_count_data[1]),
# 						   np.array(neur_null_count_data[2]),segment_length,norm_val,bin_dir,'Neuron Counts',seg_name,np.array(percentile_count_data[1]))	
# 		neur_true_count_x.append(np.array(neur_true_count_data[0]))
# 		neur_null_count_x.append(np.array(neur_null_count_data[0]))
# 		neur_true_count_vals.append(np.array(neur_true_count_data[1]))
# 		neur_null_count_mean.append(np.array(neur_null_count_data[1]))
# 		neur_null_count_std.append(np.array(neur_null_count_data[2]))
# 		#Plot the spike count data
# 		nd.plot_indiv_truexnull(np.array(neur_true_spike_data[0]),np.array(neur_null_spike_data[0]),np.array(neur_true_spike_data[1]),np.array(neur_null_spike_data[1]),
# 						   np.array(neur_null_spike_data[2]),np.array(segment_length),norm_val,bin_dir,'Spike Counts',seg_name,np.array(percentile_spike_data[1]))
# 		neur_true_spike_x.append(np.array(neur_true_spike_data[0]))
# 		neur_null_spike_x.append(np.array(neur_null_spike_data[0]))
# 		neur_true_spike_vals.append(np.array(neur_true_spike_data[1]))
# 		neur_null_spike_mean.append(np.array(neur_null_spike_data[1]))
# 		neur_null_spike_std.append(np.array(neur_null_spike_data[2]))
# 		#Plot the length data
# 		nd.plot_indiv_truexnull(np.array(neur_true_len_data[0]),np.array(neur_null_len_data[0]),np.array(neur_true_len_data[1]),np.array(neur_null_len_data[1]),
# 						   np.array(neur_null_len_data[2]),np.array(segment_length),norm_val,bin_dir,'Spike Counts',seg_name,np.array(percentile_len_data[1]))
# 		neur_true_len_x.append(np.array(neur_true_len_data[0]))
# 		neur_null_len_x.append(np.array(neur_null_len_data[0]))
# 		neur_true_len_vals.append(np.array(neur_true_len_data[1]))
# 		neur_null_len_mean.append(np.array(neur_null_len_data[1]))
# 		neur_null_len_std.append(np.array(neur_null_len_data[2]))
# 	#Plot all neuron count data
# 	nd.plot_all_truexnull(neur_true_count_x,neur_null_count_x,neur_true_count_vals,neur_null_count_mean,
# 									 neur_null_count_std,norm_val,bin_dir,'Neuron Counts',list(np.array(segment_names)[segments_to_analyze]))
# 	#Plot all spike count data
# 	nd.plot_all_truexnull(neur_true_spike_x,neur_null_spike_x,neur_true_spike_vals,neur_null_spike_mean,
# 									 neur_null_spike_std,norm_val,bin_dir,'Spike Counts',list(np.array(segment_names)[segments_to_analyze]))
# 	
# 	#Plot all length data
# 	nd.plot_all_truexnull(neur_true_len_x,neur_null_len_x,neur_true_len_vals,neur_null_len_mean,
# 									 neur_null_len_std,norm_val,bin_dir,'Lengths',list(np.array(segment_names)[segments_to_analyze]))
# 	
# 	
# 	
	
	
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:35:02 2023

@author: Hannah Germaine
Script to generate null distributions and compare them against the true data
"""

# import itertools, tqdm
# from multiprocessing import Pool
# import os, json, gzip
# file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
# os.chdir(file_path)
# import numpy as np
# import scipy.stats as stats
# import functions.analysis_funcs as af
# import functions.hdf5_handling as hf5
# import functions.null_distributions as nd

# if __name__ == '__main__':
# 	
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

# 	segment_spike_times = af.calc_segment_spike_times(segment_times,spike_times,num_neur)
# 	
# 	#_____Generate null distributions for each segment_____
# 	num_segments = len(segment_spike_times)
# 	num_null = 50
# 	
# 	#Compare true and null
# 	count_cutoff = np.arange(1,num_neur) #Calculate bins with these numbers of neurons firing together
# 	bin_size = 0.05 #Bin size for neuron cutoff
# 	lag_vals = np.arange(2,100).astype('int')

# 	#Null dir
# 	null_dir = fig_save_dir + 'null_data/'
# 	if os.path.isdir(null_dir) == False:
# 		os.mkdir(null_dir)
# 	
# 	#Figure save dir
# 	bin_dir = fig_save_dir + 'thresholded_statistics/'
# 	if os.path.isdir(bin_dir) == False:
# 		os.mkdir(bin_dir)
# 	
# 	try: #Import calculated dictionaries if they exist
# 		filepath = bin_dir + 'neur_count_dict.npy'
# 		neur_count_dict = np.load(filepath, allow_pickle=True).item()
# 		filepath = bin_dir + 'neur_spike_dict.npy'
# 		neur_spike_dict = np.load(filepath, allow_pickle=True).item()
# 		#filepath = bin_dir + 'neur_autocorr_dict.npy'
# 		#neur_autocorr_dict = np.load(filepath, allow_pickle=True).item()
# 		print('\t Imported thresholded datasets into memory')
# 	except: #Calculate dictionaries
# 		neur_count_dict = dict()
# 		neur_spike_dict = dict()
# 		#neur_autocorr_dict = dict()
# 		#For each segment separately
# 		for s_i in tqdm.tqdm(range(num_segments)):
# 			#Gather data / parameters
# 			seg_name = segment_names[s_i]
# 			print('\t Now Generating Null Distributions for Segment ' + seg_name)
# 			segment_spikes = segment_spike_times[s_i]
# 			segment_start_time = segment_times[s_i]
# 			segment_end_time = segment_times[s_i+1]
# 			segment_length = segment_end_time - segment_start_time
# 			#Segment save dir
# 			seg_null_dir = null_dir + segment_names[s_i] + '/'
# 			if os.path.isdir(seg_null_dir) == False:
# 				os.mkdir(seg_null_dir)
# 			#_____Import or generate null dataset_____
# 			try:
# 				filepath = seg_null_dir + 'null_' + str(0) + '.json'
# 				with gzip.GzipFile(filepath, mode="r") as f:
# 					json_bytes = f.read()
# 					json_str = json_bytes.decode('utf-8')            
# 					data = json.loads(json_str)
# 				print('\t Now importing null dataset into memory')
# 				null_segment_spikes = []
# 				for n_i in tqdm.tqdm(range(num_null)):
# 					filepath = seg_null_dir + 'null_' + str(n_i) + '.json'
# 					with gzip.GzipFile(filepath, mode="r") as f:
# 						json_bytes = f.read()
# 						json_str = json_bytes.decode('utf-8')            
# 						data = json.loads(json_str) 
# 						null_segment_spikes.append(data)
# 			except:
# 				#First create a null distribution set
# 				with Pool(processes=4) as pool: # start 4 worker processes
# 					pool.map(nd.run_null_create_parallelized,zip(np.arange(num_null), 
# 													 itertools.repeat(segment_spikes), 
# 													 itertools.repeat(segment_start_time), 
# 													 itertools.repeat(segment_end_time), 
# 													 itertools.repeat(seg_null_dir)))
# 				print('\t Now importing null dataset into memory')
# 				null_segment_spikes = []
# 				for n_i in range(num_null):
# 					filepath = seg_null_dir + 'null_' + str(n_i) + '.json'
# 					with gzip.GzipFile(filepath, mode="r") as f:
# 						json_bytes = f.read()
# 						json_str = json_bytes.decode('utf-8')            
# 						data = json.loads(json_str) 
# 						null_segment_spikes.append(data)
# 			#_____Convert null data to binary spike matrix_____
# 			null_bin_spikes = []
# 			for n_n in range(num_null):
# 				null_spikes = null_segment_spikes[n_n]
# 				null_bin_spike = np.zeros((num_neur,segment_end_time-segment_start_time+1))
# 				for n_i in range(num_neur):
# 					spike_indices = (np.array(null_spikes[n_i]) - segment_start_time).astype('int')
# 					null_bin_spike[n_i,spike_indices] = 1
# 				null_bin_spikes.append(null_bin_spike)
# 			print('\t Now calculating count, spike, and autocorrelation distributions')
# 			#_____Convert real data to a binary spike matrix_____
# 			bin_spike = np.zeros((num_neur,segment_end_time-segment_start_time+1))
# 			for n_i in range(num_neur):
# 				spike_indices = (np.array(segment_spikes[n_i]) - segment_start_time).astype('int')
# 				bin_spike[n_i,spike_indices] = 1
			#_____Calculate statistics of true and null datasets_____
# 			true_neur_counts, true_spike_counts = nd.high_bins([bin_spike,segment_start_time,segment_end_time,bin_size,count_cutoff])
# 			#true_autocorr = nd.auto_corr([bin_spike,segment_start_time,segment_end_time,lag_vals])
# 			null_neur_counts = dict()
# 			null_spike_counts = dict()
# 			#null_autocorrs = dict()
# 			#Run nd.high_bins() to get null neuron counts and spike counts
# 			chunk_inds = np.linspace(0,num_null,10).astype('int') #chunk it to keep memory usage lower
# 			results_counts = []
# 			print('\t\tCalculating bins for null distributions')
# 			for c_i in tqdm.tqdm(range(len(chunk_inds)-1)):
# 				null_bin_spike_chunk = null_bin_spikes[chunk_inds[c_i]:chunk_inds[c_i+1]]
# 				with Pool(processes=4) as pool: # start 4 worker processes
# 					results_chunk_counts = pool.map(nd.high_bins,zip(null_bin_spike_chunk, 
# 													 itertools.repeat(segment_start_time), 
# 													 itertools.repeat(segment_end_time), 
# 													 itertools.repeat(bin_size), 
# 													 itertools.repeat(count_cutoff)))
# 					results_counts.extend(results_chunk_counts)
# 			#Run nd.auto_corr() to get null autocorrelations
# 			#with Pool(processes=4) as pool: # start 4 worker processes
# 			#	results_autocorr = pool.map(nd.auto_corr,zip(null_bin_spikes,
# 			#									 itertools.repeat(segment_start_time),
# 			#									 itertools.repeat(segment_end_time),
# 			#									itertools.repeat(lag_vals)))
# 			for n_n in range(num_null):
# 				null_neur_count = results_counts[n_n][0]
# 				null_spike_count = results_counts[n_n][1]
# 			#	null_autocorr = results_autocorr[n_n]
# 				for key in null_neur_count.keys():
# 					if key in null_neur_counts.keys():
# 						null_neur_counts[key].append(null_neur_count[key])
# 					else:
# 						null_neur_counts[key] = [null_neur_count[key]]
# 				for key in null_spike_count.keys():
# 					if key in null_spike_counts.keys():
# 						null_spike_counts[key].append(null_spike_count[key])
# 					else:
# 						null_spike_counts[key] = [null_spike_count[key]]
# 			#	for lag_i in range(len(lag_vals)): #key in null_autocorr.keys():
			#		lag_val = int(lag_vals[lag_i])
			#		if lag_val in null_autocorrs.keys():
			#			null_autocorrs[lag_val].append(null_autocorr[lag_i])
			#		else:
			#			null_autocorrs[lag_val] = [null_autocorr[lag_i]]
			#_____Savethe neuron count data_____
# 			true_x_vals = np.array([(np.ceil(float(key))).astype('int') for key in true_neur_counts.keys()])
# 			true_neur_count_array = np.array([true_neur_counts[key] for key in true_neur_counts.keys()])
# 			neur_count_dict[seg_name + '_true'] =  [list(true_x_vals),
# 										   list(true_neur_count_array)]
# 			null_x_vals = np.array([(np.ceil(float(key))).astype('int') for key in null_neur_counts.keys()])
# 			mean_null_neur_counts = np.array([np.mean(null_neur_counts[key]) for key in null_neur_counts.keys()])
# 			std_null_neur_counts = np.array([np.std(null_neur_counts[key]) for key in null_neur_counts.keys()])
# 			neur_count_dict[seg_name + '_null'] =  [list(null_x_vals),
# 										   list(mean_null_neur_counts),
# 										   list(std_null_neur_counts)]
# 			percentiles = [] #Calculate percentile of true data point in null data distribution
# 			for key in true_neur_counts.keys():
# 				try:
# 					percentiles.extend([round(stats.percentileofscore(null_neur_counts[key],true_neur_counts[key]),2)])
# 				except:
# 					percentiles.extend([100])
# 			neur_count_dict[seg_name + '_percentile'] =  [list(true_x_vals),percentiles]
# 			#_____Save the spike count data_____
# 			true_x_vals = np.array([(np.ceil(float(key))).astype('int') for key in true_spike_counts.keys()])
# 			true_spike_count_array = np.array([true_spike_counts[key] for key in true_spike_counts.keys()])
# 			neur_spike_dict[seg_name + '_true'] =  [list(true_x_vals),
# 										   list(true_spike_count_array)]
# 			null_x_vals = np.array([(np.ceil(float(key))).astype('int') for key in null_spike_counts.keys()])
# 			mean_null_spike_counts = np.array([np.mean(null_spike_counts[key]) for key in null_spike_counts.keys()])
# 			std_null_spike_counts = np.array([np.std(null_spike_counts[key]) for key in null_spike_counts.keys()])
# 			neur_spike_dict[seg_name + '_null'] =  [list(null_x_vals),
# 										   list(mean_null_spike_counts),
# 										   list(std_null_spike_counts)]
# 			percentiles = [] #Calculate percentile of true data point in null data distribution
# 			for key in true_spike_counts.keys():
# 				try:
# 					percentiles.extend([round(stats.percentileofscore(null_spike_counts[key],true_spike_counts[key]),2)])
# 				except:
# 					percentiles.extend([100])
# 			neur_spike_dict[seg_name + '_percentile'] =  [list(true_x_vals),percentiles]
			#_____Save autocorrelation data_____
			#true_x_vals = np.array([(np.ceil(float(key))).astype('int') for key in true_autocorr.keys()])
			#true_autocorr_array = np.array([true_autocorr[key] for key in true_autocorr.keys()])
			#neur_autocorr_dict[seg_name + '_true'] =  [list(true_x_vals),
			#							   list(true_autocorr_array)]
			#null_x_vals = np.array([(np.ceil(float(key))).astype('int') for key in null_autocorrs.keys()])
			#mean_null_autocorrs = np.array([np.mean(null_autocorrs[key]) for key in null_autocorrs.keys()])
			#std_null_autocorrs = np.array([np.std(null_autocorrs[key]) for key in null_autocorrs.keys()])
			#neur_autocorr_dict[seg_name + '_null'] =  [list(null_x_vals),
			#							   list(mean_null_autocorrs),
			#							   list(std_null_autocorrs)]
			#percentiles = [] #Calculate percentile of true data point in null data distribution
			#for key in true_autocorr.keys():
			#	try:
			#		percentiles.extend([round(stats.percentileofscore(null_autocorrs[key],true_autocorr[key]),2)])
			#	except:
			#		percentiles.extend([100])
			#neur_autocorr_dict[seg_name + '_percentile'] =  [list(true_x_vals),percentiles]
			
# 		
# 		#Save the dictionaries
# 		filepath = bin_dir + 'neur_count_dict.npy'
# 		np.save(filepath, neur_count_dict)
# 		filepath = bin_dir + 'neur_spike_dict.npy'
# 		np.save(filepath, neur_spike_dict) 
# 		#filepath = bin_dir + 'neur_autocorr_dict.npy'
# 		#np.save(filepath, neur_autocorr_dict)

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
# 	neur_true_rate_x = []
# 	neur_true_rate_vals = []
# 	neur_null_rate_x = []
# 	neur_null_rate_mean = []
# 	neur_null_rate_std = []
# 	#neur_true_autocorr_x = []
# 	#neur_true_autocorr_vals = []
# 	#neur_null_autocorr_x = []
# 	#neur_null_autocorr_mean = []
# 	#neur_null_autocorr_std = []
# 	#for s_i in tqdm.tqdm(range(num_segments)):
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
# 		#neur_true_autocorr_data = neur_autocorr_dict[seg_name + '_true']
# 		#neur_null_autocorr_data = neur_autocorr_dict[seg_name + '_null']
# 		#percentile_autocorr_data = neur_autocorr_dict[seg_name + '_percentile']
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
# 		#Store the bouts/second data
# 		neur_true_rate_x.append(np.array(neur_true_count_data[0])/norm_val)
# 		neur_null_rate_x.append(np.array(neur_null_count_data[0])/norm_val)
# 		neur_true_rate_vals.append(np.array(neur_true_count_data[1])/norm_val)
# 		neur_null_rate_mean.append(np.array(neur_null_count_data[1])/norm_val)
# 		neur_null_rate_std.append(np.array(neur_null_count_data[2])/norm_val)
# 		#Plot the autocorrelation data
# 		#nd.plot_indiv_truexnull(np.array(neur_true_autocorr_data[0]),np.array(neur_null_autocorr_data[0]),np.array(neur_true_autocorr_data[1]),np.array(neur_null_autocorr_data[1]),
# 		#				   np.array(neur_null_autocorr_data[2]),np.array(segment_length),norm_val,bin_dir,'Autocorrelation',seg_name,np.array(percentile_autocorr_data[1]))
# 		#neur_true_autocorr_x.append(np.array(neur_true_autocorr_data[0]))
# 		#neur_null_autocorr_x.append(np.array(neur_null_autocorr_data[0]))
# 		#neur_true_autocorr_vals.append(np.array(neur_true_autocorr_data[1]))
# 		#neur_null_autocorr_mean.append(np.array(neur_null_autocorr_data[1]))
# 		#neur_null_autocorr_std.append(np.array(neur_null_autocorr_data[2]))
# 	#Plot all neuron count data
# 	nd.plot_all_truexnull(neur_true_count_x,neur_null_count_x,neur_true_count_vals,neur_null_count_mean,
# 									 neur_null_count_std,norm_val,bin_dir,'Neuron Counts',list(np.array(segment_names)[segments_to_analyze]))
# 	#Plot all spike count data
# 	nd.plot_all_truexnull(neur_true_spike_x,neur_null_spike_x,neur_true_spike_vals,neur_null_spike_mean,
# 									 neur_null_spike_std,norm_val,bin_dir,'Spike Counts',list(np.array(segment_names)[segments_to_analyze]))
# 	
# 	#Plot all bouts/second data
# 	nd.plot_all_truexnull(neur_true_rate_x,neur_null_rate_x,neur_true_rate_vals,neur_null_rate_mean,
# 									 neur_null_rate_std,norm_val,bin_dir,'Bouts per Second',list(np.array(segment_names)[segments_to_analyze]))
# 	#Plot all autocorrelation data
# 	#nd.plot_all_truexnull(neur_true_autocorr_x,neur_null_autocorr_x,neur_true_autocorr_vals,neur_null_autocorr_mean,
# 	#								 neur_null_autocorr_std,norm_val,bin_dir,'Autocorrelation Lags',segment_names)
# 	
# 	
	
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:24:49 2023

@author: Hannah Germaine

A collection of functions used by find_deviations.py to pull, reformat, analyze,
etc... the deviations in true and null datasets.
"""

import os, json, gzip, tqdm, itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, ks_2samp, ttest_ind, kruskal
#file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
#os.chdir(file_path)
import functions.corr_dist_calc_parallel as cdcp
import functions.corr_dist_calc_parallel_pop as cdcpp
import functions.corr_dist_calc_parallel_zscore as cdcpz
import functions.corr_dist_calc_parallel_pop_zscore as cdcppz
import functions.dev_plot_funcs as dpf
from multiprocess import Pool
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

def run_dev_pull_parallelized(inputs):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	"""
	spikes = inputs[0]  # list of length num_neur with each neur having a list of spike times
	local_size = inputs[1]  # how many bins are the local activity?
	# what is the minimum number of bins for a deviation?
	min_dev_size = inputs[2]
	segment_times_i = inputs[3]  # start and end times of segment
	save_dir = inputs[4]  # directory to save data
	# calculate the deviations
	num_neur = len(spikes)
	num_dt = segment_times_i[1] - segment_times_i[0] + 1
	spikes_bin = np.zeros((num_neur, num_dt))
	for n_i in range(num_neur):
		n_spikes = np.array(spikes[n_i]).astype('int') - int(segment_times_i[0])
		spikes_bin[n_i, n_spikes] = 1
	spike_sum = np.sum(spikes_bin, 0)
	half_min_dev_size = int(np.ceil(min_dev_size/2))
	half_local_size = int(np.ceil(local_size/2))
	#Spike sum reshape
	spike_sum_reshape = np.zeros((num_dt,half_min_dev_size*2))
	for i_s in np.arange(half_min_dev_size, num_dt-half_min_dev_size):
		spike_sum_reshape[i_s,:] = spike_sum[i_s - half_min_dev_size:i_s + half_min_dev_size]
	fr_calc = np.sum(spike_sum_reshape,1)/(2*half_min_dev_size)
	# Find where the firing rate is above 3std from the mean
	local_mean_fr = np.zeros(num_dt)
	local_std_fr = np.zeros(num_dt)
	for i_s in tqdm.tqdm(np.arange(half_min_dev_size,num_dt-half_min_dev_size)):
		min_ind = max(i_s - half_local_size,0)
		max_ind = min(num_dt,i_s+half_local_size)
		local_mean_fr[i_s] = np.mean(fr_calc[min_ind:max_ind])
		local_std_fr[i_s] = np.std(fr_calc[min_ind:max_ind])
	peak_fr_ind = np.where(fr_calc >= local_mean_fr + 3*local_std_fr)[0]
	deviations = np.zeros(num_dt)
	for t_i in peak_fr_ind:
		deviations[t_i - half_min_dev_size:t_i + half_min_dev_size] = 1
	# store each in a json
	json_str = json.dumps(list(deviations))
	json_bytes = json_str.encode()
	filepath = save_dir + 'deviations.json'
	with gzip.GzipFile(filepath, mode="w") as f:
		f.write(json_bytes)

def create_dev_rasters(num_iterations, spike_times,
					   start_end_times, deviations, pre_z):
	"""This function takes the spike times and creates binary matrices of 
	rasters of spiking"""
	dev_rasters = []
	dev_times = []
	dev_rasters_zscore = [] #Includes pre-interval for z-scoring
	pre_z_dt = int(pre_z*1000) #in timesteps
	for ind in range(num_iterations):
		seg_spikes = spike_times[ind]
		num_neur = len(seg_spikes)
		num_dt = int(start_end_times[ind, 1] - start_end_times[ind, 0] + 1)
		spikes_bin = np.zeros((num_neur, num_dt))
		for n_i in range(num_neur):
			neur_spikes = np.array(seg_spikes[n_i]).astype(
				'int') - int(start_end_times[ind, 0])
			spikes_bin[n_i, neur_spikes] = 1
		seg_rast = []
		seg_rast_zscore = []
		ind_dev = deviations[ind]
		ind_dev[0] = 0
		ind_dev[-1] = 0
		change_inds = np.diff(deviations[ind])
		start_dev_bouts = np.where(change_inds == 1)[0]
		end_dev_bouts = np.where(change_inds == -1)[0]
		#remove all those too early to calculate a z-score in the future
		keep_vals = np.where(start_dev_bouts >= pre_z_dt+1)[0]
		start_dev_bouts = start_dev_bouts[keep_vals]
		end_dev_bouts = end_dev_bouts[keep_vals]
		if len(start_dev_bouts) > len(end_dev_bouts):
			end_dev_bouts = np.append(end_dev_bouts,num_dt)
		if len(end_dev_bouts) > len(start_dev_bouts):
			start_dev_bouts = np.insert(start_dev_bouts,0,0)
		bout_times = np.concatenate(
			(np.expand_dims(start_dev_bouts, 0), np.expand_dims(end_dev_bouts, 0)))
		for b_i in range(len(start_dev_bouts)):
			seg_rast.append(spikes_bin[:, start_dev_bouts[b_i]:end_dev_bouts[b_i]])
			seg_rast_zscore.append(spikes_bin[:, start_dev_bouts[b_i] - pre_z_dt:end_dev_bouts[b_i]])
			
		dev_rasters.append(seg_rast)
		dev_times.append(bout_times)
		dev_rasters_zscore.append(seg_rast_zscore)

	return dev_rasters, dev_times, dev_rasters_zscore

def calculate_dev_stats(rasters, times, iteration_names, save_dir, iterations_to_analyze=[]):
	"""This function calculates deviation statistics - and plots them - including:
			- deviation lengths
			- inter-deviation-intervals (IDIs)
			- number of spikes / deviation
			- number of neurons spiking / deviation
	"""

	num_iterations = len(rasters)
	length_dict = dict()
	IDI_dict = dict()
	num_spike_dict = dict()
	num_neur_dict = dict()
	
	if len(iterations_to_analyze) == 0:
		iterations_to_analyze = np.arange(num_iterations)
		
	iteration_names_to_analyze = list(np.array(iteration_names)[iterations_to_analyze])

	for it in tqdm.tqdm(iterations_to_analyze):
		iter_name = iteration_names[it]
		# Gather data
		iter_rasters = rasters[it]
		bout_times = times[it]
		# Calculate segment lengths
		seg_lengths = bout_times[1, :] - bout_times[0, :]
		length_dict[it] = seg_lengths
		data_name = iter_name + ' deviation lengths'
		dpf.plot_dev_stats(seg_lengths, data_name, save_dir,
					   x_label='deviation index', y_label='length (ms)')
		# Calculate IDIs
		seg_IDIs = bout_times[1, 1:] - bout_times[0, :-1]
		IDI_dict[it] = seg_IDIs
		data_name = iter_name + ' inter-deviation-intervals'
		dpf.plot_dev_stats(seg_IDIs, data_name, save_dir,
					   x_label='distance index', y_label='length (ms)')
		# Calculate number of spikes
		seg_spike_num = [np.sum(np.sum(iter_rasters[r_i]))
						 for r_i in range(len(iter_rasters))]
		num_spike_dict[it] = seg_spike_num
		data_name = iter_name + ' total spike count'
		dpf.plot_dev_stats(seg_spike_num, data_name, save_dir,
					   x_label='deviation index', y_label='# spikes')
		# Calculate number of neurons spiking
		seg_neur_num = [np.sum(np.sum(iter_rasters[r_i], 1) > 0)
						for r_i in range(len(iter_rasters))]
		num_neur_dict[it] = seg_neur_num
		data_name = iter_name + ' total neuron count'
		dpf.plot_dev_stats(seg_neur_num, data_name, save_dir,
					   x_label='deviation index', y_label='# neurons')
		
	#Now plot stats across iterations
	dpf.plot_dev_stats_dict(length_dict, iteration_names_to_analyze, 'Deviation Lengths', save_dir, 'Segment', 'Length (ms)')
	dpf.plot_dev_stats_dict(IDI_dict, iteration_names_to_analyze, 'Inter-Deviation-Intervals', save_dir, 'Segment', 'Length (ms)')
	dpf.plot_dev_stats_dict(num_spike_dict, iteration_names_to_analyze, 'Total Spike Count', save_dir, 'Segment', '# Spikes')
	dpf.plot_dev_stats_dict(num_neur_dict, iteration_names_to_analyze, 'Total Neuron Count', save_dir, 'Segment', '# Neurons')

	return length_dict, IDI_dict, num_spike_dict, num_neur_dict

def calculate_dev_null_stats(all_rast,dev_times):
	"""This function calculates statistics of deviation events for comparison 
	between true data and null distributions"""
	
	num_neur = []
	num_spikes = []
	for nr in range(len(all_rast)):
		num_spikes_n_i = np.sum(all_rast[nr],1)
		num_spikes_i = np.sum(num_spikes_n_i)
		num_spikes.append(num_spikes_i)
		num_neur_i = np.sum((num_spikes_n_i > 0).astype('int'))
		num_neur.append(num_neur_i)
	all_len = dev_times[1,:] - dev_times[0,:]
	
	return num_neur, num_spikes, all_len

def calculate_correlations(segment_dev_rasters, tastant_spike_times,
						   start_dig_in_times, end_dig_in_times, segment_names, dig_in_names,
						   pre_taste, post_taste, taste_cp_raster_inds, pop_taste_cp_raster_inds,
						   save_dir, neuron_keep_indices=[]):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval"""

	#Grab parameters
	fr_bin = 25 #ms to bin together for number of spikes 'fr'
	num_tastes = len(start_dig_in_times)
	num_segments = len(segment_dev_rasters)
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
	post_taste_dt = np.ceil(post_taste*1000).astype('int')

	for s_i in range(num_segments):  #Loop through each segment
		print("Beginning timeseries correlation calcs for segment " + str(s_i))
		#Gather segment data
		seg_rast = segment_dev_rasters[s_i]
		num_dev = len(seg_rast)
			
		for t_i in range(num_tastes):  #Loop through each taste
			#Find the number of neurons
			if np.shape(neuron_keep_indices)[0] == 0:
				total_num_neur = np.shape(seg_rast[0])[0]
				taste_keep_ind = np.arange(total_num_neur)
			else:
				total_num_neur = np.sum(neuron_keep_indices[:,t_i]).astype('int')
				taste_keep_ind = (np.where(((neuron_keep_indices[:,t_i]).astype('int')).flatten())[0]).astype('int')
			#Try to import previously stored data if exists
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
			filename_loaded = 0
			filename_pop = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop.npy'
			filename_pop_loaded = 0
			try:
				neuron_corr_storage = np.load(filename)
				filename_loaded = 1
			except:
				print("Individual Neuron Timeseries Correlations Need to Be Calculated")
			try:
				neuron_vec_corr_storage = np.load(filename_pop)
				filename_pop_loaded = 1
			except:
				print("Population Timeseries Correlations Need to Be Calculated")
			if filename_loaded*filename_pop_loaded == 0:
				print("\tCalculating Taste #" + str(t_i + 1))
				taste_cp = taste_cp_raster_inds[t_i][:, taste_keep_ind, :]
				taste_cp_pop = pop_taste_cp_raster_inds[t_i]
				taste_spikes = tastant_spike_times[t_i]
				#Note, num_cp = num_cp+1 with the first value the taste delivery index
				num_deliv, _, num_cp = np.shape(taste_cp)
				taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i][deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
				deliv_adjustment = [start_dig_in_times[t_i][deliv_i] + pre_taste_dt for deliv_i in range(num_deliv)]
				#Store the correlation results in a numpy array
				neuron_corr_storage = np.nan*np.ones((num_dev, num_deliv, total_num_neur, num_cp-1))
				neuron_pop_corr_storage = np.nan*np.ones((num_dev, num_deliv, num_cp-1))
				for dev_i in tqdm.tqdm(range(num_dev)): #Loop through all deviations
					dev_rast = seg_rast[dev_i][taste_keep_ind,:]
					dev_len = np.shape(dev_rast)[1]
					end_ind = np.arange(fr_bin,fr_bin+dev_len)
					end_ind[end_ind > dev_len] = dev_len
					#TODO: test gaussian convolution instead of binning
					dev_rast_binned = np.zeros(np.shape(dev_rast)) #timeseries information kept
					for start_ind in range(dev_len):
						dev_rast_binned[:,start_ind] = np.sum(dev_rast[:,start_ind:end_ind[start_ind]],1)
					#Individual neuron changepoints
					if filename_loaded == 0:
						inputs = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
						 itertools.repeat(np.arange(0,total_num_neur)), itertools.repeat(taste_cp), \
							 deliv_adjustment, itertools.repeat(dev_rast_binned), itertools.repeat(fr_bin))
						pool = Pool(4)
						deliv_corr_storage = pool.map(cdcp.deliv_corr_parallelized, inputs)
						pool.close()
						neuron_corr_storage[dev_i,:,:,:] = np.array(deliv_corr_storage)
					#Population changepoints
					if filename_pop_loaded == 0:
						pool = Pool(4)
						inputs_pop = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
							itertools.repeat(np.arange(0,total_num_neur)), itertools.repeat(taste_cp_pop), \
							deliv_adjustment, itertools.repeat(dev_rast_binned), itertools.repeat(fr_bin))
						deliv_vec_corr_storage = pool.map(cdcpp.deliv_corr_population_parallelized, inputs_pop)
						pool.close()
						neuron_pop_corr_storage[dev_i,:,:] = np.array(deliv_vec_corr_storage)
				#Save to a numpy array
				if filename_loaded == 0:
					np.save(filename,neuron_corr_storage)
				if filename_pop_loaded == 0:
					np.save(filename_pop,neuron_pop_corr_storage)


def calculate_vec_correlations(segment_dev_rasters, tastant_spike_times,
						   start_dig_in_times, end_dig_in_times, segment_names, dig_in_names,
						   pre_taste, post_taste, taste_cp_raster_inds, pop_taste_cp_raster_inds,
						   save_dir, neuron_keep_indices=[], segments_to_analyze=[]):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval"""
	#Grab parameters
	num_tastes = len(start_dig_in_times)
	num_segments = len(segment_dev_rasters)
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
	post_taste_dt = np.ceil(post_taste*1000).astype('int')
	
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(num_segments)
	
	for s_i in segments_to_analyze:  #Loop through each segment
		print("Beginning population vector correlation calcs for segment " + str(s_i))
		#Gather segment data
		seg_rast = segment_dev_rasters[s_i]
		num_dev = len(seg_rast)
			
		for t_i in range(num_tastes):  #Loop through each taste
			#Set storage directory and check if data previously stored
			filename_pop_vec = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec.npy'
			filename_pop_vec_loaded = 0
			try:
				neuron_pop_vec_corr_storage = np.load(filename_pop_vec)
				filename_pop_vec_loaded = 1
			except:
				print("Vector correlations not calculated for taste " + str(t_i))
			if filename_pop_vec_loaded == 0:
				print("\tCalculating correlations for taste " + str(t_i))
				taste_cp_pop = pop_taste_cp_raster_inds[t_i]
				taste_spikes = tastant_spike_times[t_i]
				#Note, num_cp = num_cp+1 with the first value the taste delivery index
				num_deliv, num_cp = np.shape(taste_cp_pop)
				taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i][deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
				deliv_adjustment = [start_dig_in_times[t_i][deliv_i] + pre_taste_dt for deliv_i in range(num_deliv)]
				#Store the correlation results in a numpy array
				neuron_pop_vec_corr_storage = np.nan*np.ones((num_dev, num_deliv, num_cp-1))
				for cp_i in range(num_cp-1):
					print('\t\tChangepoint ' + str(cp_i))
					#Find the number of neurons
					if np.shape(neuron_keep_indices)[0] == 0:
						total_num_neur = np.shape(seg_rast[0])[0]
						taste_keep_ind = np.arange(total_num_neur)
					else:
						#neuron_keep_indices = taste_select_neur_epoch_bin = num_cp x num_neur
						total_num_neur = np.sum(neuron_keep_indices[:,cp_i]).astype('int')
						taste_keep_ind = (np.where(((neuron_keep_indices[:,cp_i]).astype('int')).flatten())[0]).astype('int')
					#Loop through all deviations
					for dev_i in tqdm.tqdm(range(num_dev)): 
						dev_rast = seg_rast[dev_i][taste_keep_ind,:]
						dev_len = np.shape(dev_rast)[1]
						dev_vec = np.sum(dev_rast,1)/(dev_len/1000) #in Hz
						#Population fr vector changepoints
						pool = Pool(4)
						inputs_pop = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
							itertools.repeat(taste_keep_ind), itertools.repeat(taste_cp_pop), \
							deliv_adjustment, itertools.repeat(dev_vec), itertools.repeat(cp_i))
						deliv_vec_corr_storage = pool.map(cdcpp.deliv_corr_population_vec_parallelized, inputs_pop)
						pool.close()
						neuron_pop_vec_corr_storage[dev_i,:,cp_i] = np.array(deliv_vec_corr_storage)
					np.save(filename_pop_vec,neuron_pop_vec_corr_storage)
				np.save(filename_pop_vec,neuron_pop_vec_corr_storage)

def calculate_vec_correlations_zscore(segment_dev_rasters_zscore, tastant_spike_times,
						   start_dig_in_times, end_dig_in_times, segment_names, dig_in_names,
						   pre_taste, post_taste, taste_cp_raster_inds, pop_taste_cp_raster_inds,
						   save_dir, neuron_keep_indices=[], segments_to_analyze=[]):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval"""
	#Grab parameters
	num_tastes = len(start_dig_in_times)
	num_segments = len(segment_dev_rasters_zscore)
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
	post_taste_dt = np.ceil(post_taste*1000).astype('int')
	
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(num_segments)

	for s_i in segments_to_analyze:  #Loop through each segment
		print("Beginning population vector correlation calcs for segment " + str(s_i))
		#Gather segment data
		seg_rast = segment_dev_rasters_zscore[s_i]
		num_dev = len(seg_rast)
			
		for t_i in range(num_tastes):  #Loop through each taste
			#Set storage directory and check if data previously stored
			filename_pop_vec = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec.npy'
			filename_pop_vec_loaded = 0
			try:
				neuron_pop_vec_corr_storage = np.load(filename_pop_vec)
				filename_pop_vec_loaded = 1
			except:
				print("Vector correlations not calculated for taste " + str(t_i))
			if filename_pop_vec_loaded == 0:
				print("\tCalculating Taste #" + str(t_i + 1))
				taste_cp_pop = pop_taste_cp_raster_inds[t_i]
				taste_spikes = tastant_spike_times[t_i]
				#Note, num_cp = num_cp+1 with the first value the taste delivery index
				num_deliv, num_cp = np.shape(taste_cp_pop)
				taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i][deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
				deliv_adjustment = [start_dig_in_times[t_i][deliv_i] + pre_taste_dt for deliv_i in range(num_deliv)]
				#Store the correlation results in a numpy array
				neuron_pop_vec_corr_storage = np.nan*np.ones((num_dev, num_deliv, num_cp-1))
				for cp_i in tqdm.tqdm(range(num_cp-1)):
					#Find the number of neurons
					if np.shape(neuron_keep_indices)[0] == 0:
						total_num_neur = np.shape(seg_rast[0])[0]
						taste_keep_ind = np.arange(total_num_neur)
					else:
						#neuron_keep_indices = taste_select_neur_epoch_bin = num_cp x num_neur
						total_num_neur = np.sum(neuron_keep_indices[cp_i,:]).astype('int')
						taste_keep_ind = (np.where(((neuron_keep_indices[cp_i,:]).astype('int')).flatten())[0]).astype('int')
					for dev_i in tqdm.tqdm(range(num_dev)): #Loop through all deviations
						dev_rast = seg_rast[dev_i][taste_keep_ind,:]
						dev_z_mean = np.expand_dims(np.mean(dev_rast[:,:pre_taste_dt],axis=1),1)
						dev_z_std = np.expand_dims(np.std(dev_rast[:pre_taste_dt],axis=1),1)
						dev_z_std[dev_z_std == 0] = 1
						dev_zscored = (dev_rast - dev_z_mean)/dev_z_std
						dev_zscored = dev_zscored[:,pre_taste_dt:]
						
						dev_len = np.shape(dev_zscored)[1]
						dev_vec = np.mean(dev_zscored,axis=1)/(dev_len/1000) #in Hz the average z-scored firing rate for the deviation
						#Population fr vector changepoints
						pool = Pool(4)
						inputs_pop = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
							itertools.repeat(taste_keep_ind), itertools.repeat(taste_cp_pop), \
							deliv_adjustment, itertools.repeat(dev_vec), itertools.repeat(cp_i))
						deliv_vec_corr_storage = pool.map(cdcppz.deliv_corr_population_vec_parallelized, inputs_pop)
						pool.close()
						neuron_pop_vec_corr_storage[dev_i,:,cp_i] = np.array(deliv_vec_corr_storage)
				np.save(filename_pop_vec,neuron_pop_vec_corr_storage)
				
				
def pull_corr_dev_stats(segment_names, dig_in_names, save_dir, segments_to_analyze=[]):
	"""For each epoch and each segment pull out the top 10 most correlated deviation 
	bins and plot side-by-side with the epoch they are correlated with"""
	
	#Grab parameters
	dev_stats = dict()
	num_tastes = len(dig_in_names)
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(len(segment_names))
	num_segments = len(segments_to_analyze)
		
	for s_i in segments_to_analyze:  #Loop through each segment
		segment_stats = dict()
		seg_name = segment_names[s_i]
		for t_i in range(num_tastes):  #Loop through each taste
			#Import distance numpy array
			filename_pop_vec = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec.npy'
			population_vec_data_storage = np.load(filename_pop_vec)
			#Calculate statistics
			data_dict = dict()
			data_dict['segment'] = seg_name
			data_dict['taste'] = dig_in_names[t_i]
			num_dev, num_deliv, num_cp = np.shape(population_vec_data_storage)
			data_dict['num_dev'] = num_dev
			data_dict['pop_vec_data_storage'] = np.abs(population_vec_data_storage)
			segment_stats[t_i] = data_dict
		dev_stats[seg_name] = segment_stats

	return dev_stats


def stat_significance(segment_data, segment_names, dig_in_names, save_dir, dist_name, segments_to_analyze=[]):
	
	#Grab parameters
	#segment_data shape = segments x tastes x cp
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(len(segment_names))
	num_segments = len(segments_to_analyze)
	segment_names = np.array(segment_names)[segments_to_analyze]
	num_tastes = len(segment_data[0])
	num_cp = len(segment_data[0][0])
	
	#Calculate statistical significance of pairs
	#Are the correlation distributions significantly different across pairs?
	
	#All pair combinations
	a = [list(segments_to_analyze),list(np.arange(num_tastes)),list(np.arange(num_cp))]
	data_combinations = list(itertools.product(*a))
	pair_combinations = list(itertools.combinations(data_combinations,2))
	
	#Pair combination significance storage
	save_file = save_dir + dist_name + '_significance.txt'
	pair_significances = np.zeros(len(pair_combinations))
	pair_significance_statements = []
	
	print("Calculating Significance for All Combinations")
	for p_i in tqdm.tqdm(range(len(pair_combinations))):
		try:
			ind_1 = pair_combinations[p_i][0]
			ind_2 = pair_combinations[p_i][1]
			data_1 = np.abs(segment_data[ind_1[0]][ind_1[1]][ind_1[2]])
			data_2 = np.abs(segment_data[ind_2[0]][ind_2[1]][ind_2[2]])
			result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
			if result[1] < 0.05:
				pair_significances[p_i] = 1
				statement = segment_names[ind_1[0]] + '_' + dig_in_names[ind_1[1]] + '_epoch' + str(ind_1[2]) + \
					  ' vs ' + segment_names[ind_2[0]] + '_' + dig_in_names[ind_2[1]] + '_epoch' + str(ind_2[2]) + \
						  ' = significantly different with p-val = ' + str(result[1])
			else:
				statement = segment_names[ind_1[0]] + '_' + dig_in_names[ind_1[1]] + '_epoch' + str(ind_1[2]) + \
					  ' vs ' + segment_names[ind_2[0]] + '_' + dig_in_names[ind_2[1]] + '_epoch' + str(ind_2[2]) + \
						  ' = not significantly different with p-val = ' + str(result[1])
			pair_significance_statements.append(statement)
		except:
			pass
		
	with open(save_file, 'w') as f:
		for line in pair_significance_statements:
			f.write(line)
			f.write('\n')
			
def stat_significance_ttest_less(segment_data, segment_names, dig_in_names, 
								 save_dir, dist_name, segments_to_analyze = []):
	
	#Grab parameters
	#segment_data shape = segments x tastes x cp
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(len(segment_names))
	num_segments = len(segments_to_analyze)
	segment_names = np.array(segment_names)[segments_to_analyze]
	num_tastes = len(segment_data[0])
	num_cp = len(segment_data[0][0])
	
	#Calculate statistical significance of pairs
	#Are the correlation distributions significantly different across pairs?
	
	#All pair combinations
	a = [list(segments_to_analyze),list(np.arange(num_tastes)),list(np.arange(num_cp))]
	data_combinations = list(itertools.product(*a))
	pair_combinations = list(itertools.combinations(data_combinations,2))
	
	#Pair combination significance storage
	save_file = save_dir + dist_name + '_significance.txt'
	pair_significance_statements = []
	
	print("Calculating Significance for All Combinations")
	for p_i in tqdm.tqdm(range(len(pair_combinations))):
		try:
			ind_1 = pair_combinations[p_i][0]
			ind_2 = pair_combinations[p_i][1]
			data_1 = np.abs(segment_data[ind_1[0]][ind_1[1]][ind_1[2]])
			data_2 = np.abs(segment_data[ind_2[0]][ind_2[1]][ind_2[2]])
			result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
			if result[1] < 0.05:
				statement = segment_names[ind_1[0]] + '_' + dig_in_names[ind_1[1]] + '_epoch' + str(ind_1[2]) + \
					' vs ' + segment_names[ind_2[0]] + '_' + dig_in_names[ind_2[1]] + '_epoch' + str(ind_2[2]) + \
						' = significantly different with p-val = ' + str(result[1])
			else:
				statement = segment_names[ind_1[0]] + '_' + dig_in_names[ind_1[1]] + '_epoch' + str(ind_1[2]) + \
					' vs ' + segment_names[ind_2[0]] + '_' + dig_in_names[ind_2[1]] + '_epoch' + str(ind_2[2]) + \
						' = not significantly different with p-val = ' + str(result[1])
			pair_significance_statements.append(statement)
		except:
			pass
		
	with open(save_file, 'w') as f:
		for line in pair_significance_statements:
			f.write(line)
			f.write('\n')
			
def stat_significance_ttest_more(segment_data, segment_names, dig_in_names, 
								 save_dir, dist_name, segments_to_analyze = []):
	
	#Grab parameters
	#segment_data shape = segments x tastes x cp
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(len(segment_names))
	num_segments = len(segments_to_analyze)
	segment_names = np.array(segment_names)[segments_to_analyze]
	num_tastes = len(segment_data[0])
	num_cp = len(segment_data[0][0])
	
	#Calculate statistical significance of pairs
	#Are the correlation distributions significantly different across pairs?
	
	#All pair combinations
	a = [list(segments_to_analyze),list(np.arange(num_tastes)),list(np.arange(num_cp))]
	data_combinations = list(itertools.product(*a))
	pair_combinations = list(itertools.combinations(data_combinations,2))
	
	#Pair combination significance storage
	save_file = save_dir + dist_name + '_significance.txt'
	pair_significance_statements = []
	
	print("Calculating Significance for All Combinations")
	for p_i in tqdm.tqdm(range(len(pair_combinations))):
		try:
			ind_1 = pair_combinations[p_i][0]
			ind_2 = pair_combinations[p_i][1]
			data_1 = np.abs(segment_data[ind_1[0]][ind_1[1]][ind_1[2]])
			data_2 = np.abs(segment_data[ind_2[0]][ind_2[1]][ind_2[2]])
			result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='more')
			if result[1] < 0.05:
				statement = segment_names[ind_1[0]] + '_' + dig_in_names[ind_1[1]] + '_epoch' + str(ind_1[2]) + \
					' vs ' + segment_names[ind_2[0]] + '_' + dig_in_names[ind_2[1]] + '_epoch' + str(ind_2[2]) + \
						' = significantly different with p-val = ' + str(result[1])
			else:
				statement = segment_names[ind_1[0]] + '_' + dig_in_names[ind_1[1]] + '_epoch' + str(ind_1[2]) + \
					' vs ' + segment_names[ind_2[0]] + '_' + dig_in_names[ind_2[1]] + '_epoch' + str(ind_2[2]) + \
						' = not significantly different with p-val = ' + str(result[1])
			pair_significance_statements.append(statement)
		except:
			pass
		
	with open(save_file, 'w') as f:
		for line in pair_significance_statements:
			f.write(line)
			f.write('\n')

			
def mean_compare(segment_data, segment_names, dig_in_names, save_dir, 
				 dist_name, segments_to_analyze = []):
	
	#Grab parameters
	#segment_data shape = segments x tastes x cp
	
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(len(segment_names))
	num_segments = len(segments_to_analyze)
	segment_names = np.array(segment_names)[segments_to_analyze]
	num_tastes = len(segment_data[0])
	num_cp = len(segment_data[0][0])
	
	#Calculate mean comparison of pairs
	
	#All pair combinations
	a = [list(segments_to_analyze),list(np.arange(num_tastes)),list(np.arange(num_cp))]
	data_combinations = list(itertools.product(*a))
	pair_combinations = list(itertools.combinations(data_combinations,2))
	
	#Pair combination significance storage
	save_file = save_dir + dist_name + '.txt'
	pair_mean_statements = []
	
	print("Calculating Significance for All Combinations")
	for p_i in tqdm.tqdm(range(len(pair_combinations))):
		try:
			ind_1 = pair_combinations[p_i][0]
			ind_2 = pair_combinations[p_i][1]
			data_1 = np.abs(segment_data[ind_1[0]][ind_1[1]][ind_1[2]])
			data_2 = np.abs(segment_data[ind_2[0]][ind_2[1]][ind_2[2]])
			result =  (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
			if result == 1:
				statement = segment_names[ind_1[0]] + '_' + dig_in_names[ind_1[1]] + '_epoch' + str(ind_1[2]) + \
						' < ' + segment_names[ind_2[0]] + '_' + dig_in_names[ind_2[1]] + '_epoch' + str(ind_2[2])
			if result == 0:
				statement = segment_names[ind_1[0]] + '_' + dig_in_names[ind_1[1]] + '_epoch' + str(ind_1[2]) + \
						' > ' + segment_names[ind_2[0]] + '_' + dig_in_names[ind_2[1]] + '_epoch' + str(ind_2[2])

			pair_mean_statements.append(statement)
		except:
			pass
		
	with open(save_file, 'w') as f:
		for line in pair_mean_statements:
			f.write(line)
			f.write('\n')

			
def top_dev_corr_bins(dev_stats,segment_names,dig_in_names,save_dir,
					  neuron_indices,segments_to_analyze=[]):
	"""Calculate which deviation index is most correlated with which taste 
	delivery and which epoch and store to a text file.
	
	neuron_indices should be binary and shaped num_neur x num_cp
	"""
	
	#Grab parameters
	num_tastes = len(dig_in_names)
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(len(segment_names))
	num_segments = len(segments_to_analyze)
	segment_names = np.array(segment_names)[segments_to_analyze]
	
	#Define storage
	for s_i, s_true_i in enumerate(segments_to_analyze):  #Loop through each segment
		seg_name = segment_names[s_i]
		seg_stats = dev_stats[seg_name]
		print("Beginning calcs for segment " + str(s_true_i))
		for t_i in range(num_tastes):  #Loop through each taste
			pop_vec_save_file = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_top_corr_combos_pop_vec.txt'
			corr_pop_vec_data = []
			print("\tTaste #" + str(t_i + 1))
			taste_stats = seg_stats[t_i]
			#Import distance numpy array
			pop_vec_data_storage = taste_stats['pop_vec_data_storage']
			#num_dev, num_deliv, total_num_neur, num_cp = np.shape(neuron_data_storage)
			num_dev, num_deliv, num_cp = np.shape(pop_vec_data_storage)
			top_99_percentile_pop_vec = np.percentile((pop_vec_data_storage[~np.isnan(pop_vec_data_storage)]).flatten(),99)
			for dev_i in range(num_dev):
				pop_vec_data = pop_vec_data_storage[dev_i,:,:]
				[pop_vec_deliv_i,pop_vec_cp_i] = np.where(pop_vec_data >= top_99_percentile_pop_vec)
				if len(pop_vec_deliv_i) > 0:
					for d_i in range(len(pop_vec_deliv_i)):
						dev_pop_cp_corr_val = pop_vec_data[pop_vec_deliv_i[d_i],pop_vec_cp_i[d_i]]
						statement = 'dev-' + str(dev_i) + '; epoch-' + str(pop_vec_cp_i[d_i]) + '; deliv-' + str(pop_vec_deliv_i[d_i]) + '; corr-' + str(dev_pop_cp_corr_val)
						corr_pop_vec_data.append(statement)
			#Save to file population vector statements
			with open(pop_vec_save_file, 'w') as f:
				for line in corr_pop_vec_data:
					f.write(line)
					f.write('\n')
			




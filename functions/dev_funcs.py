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
from scipy.stats import pearsonr
import functions.corr_dist_calc_parallel as cdcp
from multiprocessing import Pool

def run_dev_pull_parallelized(inputs):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	"""
	spikes = inputs[0]  # list of length num_neur with each neur having a list of spike times
	local_size = inputs[1]  # how many bins are the local activity?
	# what is the minimum number of bins for a deviation?
	min_dev_size = inputs[2]
	segment_times = inputs[3]  # start and end times of segment
	save_dir = inputs[4]  # directory to save data
	# calculate the deviations
	num_neur = len(spikes)
	num_dt = segment_times[1] - segment_times[0] + 1
	spikes_bin = np.zeros((num_neur, num_dt))
	for n_i in range(num_neur):
		n_spikes = np.array(spikes[n_i]).astype('int') - int(segment_times[0])
		spikes_bin[n_i, n_spikes] = 1
	spike_sum = np.sum(spikes_bin, 0)
	half_min_dev_size = int(np.ceil(min_dev_size/2))
	half_local_size = int(np.ceil(local_size/2))
	fr_calc = np.array([np.sum(spike_sum[i_s - half_min_dev_size:i_s + half_min_dev_size]) /
					   min_dev_size for i_s in np.arange(half_local_size, num_dt-half_local_size)])  # in spikes per min_dev_size
	local_fr_calc = np.array([np.sum(spike_sum[l_s - half_local_size:l_s + half_local_size]) /
							 local_size for l_s in np.arange(half_local_size, num_dt-half_local_size)])  # in spikes per min_dev_size
	peak_fr_ind = find_peaks(fr_calc - local_fr_calc, height=0)[0]
	# Find where the prominence is above the 90th percentile of prominence values
	peak_fr_prominence = fr_calc[peak_fr_ind] - local_fr_calc[peak_fr_ind]
	top_90_prominence = np.percentile(peak_fr_prominence, 90)
	top_90_prominence_ind = peak_fr_ind[np.where(
		peak_fr_prominence >= top_90_prominence)[0]]
	true_ind = top_90_prominence_ind + half_local_size
	deviations = np.zeros(num_dt)
	for t_i in true_ind:
		deviations[t_i - half_min_dev_size:t_i + half_min_dev_size] = 1
	# store each in a json
	json_str = json.dumps(list(deviations))
	json_bytes = json_str.encode()
	filepath = save_dir + 'deviations.json'
	with gzip.GzipFile(filepath, mode="w") as f:
		f.write(json_bytes)


def create_dev_rasters(num_iterations, spike_times,
					   start_end_times, deviations):
	"""This function takes the spike times and creates binary matrices of 
	rasters of spiking"""
	dev_rasters = []
	dev_times = []
	for ind in tqdm.tqdm(range(num_iterations)):
		seg_spikes = spike_times[ind]
		num_neur = len(seg_spikes)
		num_dt = int(start_end_times[ind, 1] - start_end_times[ind, 0] + 1)
		spikes_bin = np.zeros((num_neur, num_dt))
		for n_i in range(num_neur):
			neur_spikes = np.array(seg_spikes[n_i]).astype(
				'int') - int(start_end_times[ind, 0])
			spikes_bin[n_i, neur_spikes] = 1
		seg_rast = []
		change_inds = np.diff(deviations[ind])
		start_dev_bouts = np.where(change_inds == 1)[0]
		end_dev_bouts = np.where(change_inds == -1)[0]
		bout_times = np.concatenate(
			(np.expand_dims(start_dev_bouts, 0), np.expand_dims(end_dev_bouts, 0)))
		for b_i in range(len(start_dev_bouts)):
			seg_rast.append(
				spikes_bin[:, start_dev_bouts[b_i]:end_dev_bouts[b_i]])
		dev_rasters.append(seg_rast)
		dev_times.append(bout_times)

	return dev_rasters, dev_times


def calculate_dev_stats(rasters, times, iteration_names, save_dir):
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

	for it in tqdm.tqdm(range(num_iterations)):
		iter_name = iteration_names[it]
		# Gather data
		iter_rasters = rasters[it]
		bout_times = times[it]
		# Calculate segment lengths
		seg_lengths = bout_times[1, :] - bout_times[0, :]
		length_dict[it] = seg_lengths
		data_name = iter_name + ' deviation lengths'
		plot_dev_stats(seg_lengths, data_name, save_dir,
					   x_label='deviation index', y_label='length (ms)')
		# Calculate IDIs
		seg_IDIs = bout_times[1, 1:] - bout_times[0, :-1]
		IDI_dict[it] = seg_IDIs
		data_name = iter_name + ' inter-deviation-intervals'
		plot_dev_stats(seg_IDIs, data_name, save_dir,
					   x_label='distance index', y_label='length (ms)')
		# Calculate number of spikes
		seg_spike_num = [np.sum(np.sum(iter_rasters[r_i]))
						 for r_i in range(len(iter_rasters))]
		num_spike_dict[it] = seg_spike_num
		data_name = iter_name + ' total spike count'
		plot_dev_stats(seg_spike_num, data_name, save_dir,
					   x_label='deviation index', y_label='# spikes')
		# Calculate number of neurons spiking
		seg_neur_num = [np.sum(np.sum(iter_rasters[r_i], 1) > 0)
						for r_i in range(len(iter_rasters))]
		num_neur_dict[it] = seg_neur_num
		data_name = iter_name + ' total neuron count'
		plot_dev_stats(seg_neur_num, data_name, save_dir,
					   x_label='deviation index', y_label='# neurons')

	return length_dict, IDI_dict, num_spike_dict, num_neur_dict


def plot_dev_stats(data, data_name, save_dir, x_label=[], y_label=[]):
	"""General function to plot given statistics"""
	plt.figure(figsize=(5, 5))
	# Plot the trend
	plt.subplot(2, 1, 1)
	plt.plot(data)
	if len(x_label) > 0:
		plt.xlabel(x_label)
	else:
		plt.xlabel('Deviation Index')
	if len(y_label) > 0:
		plt.ylabel(y_label)
	else:
		plt.ylabel(data_name)
	plt.title(data_name + ' trend')
	# Plot the histogram
	plt.subplot(2, 1, 2)
	plt.hist(data)
	if len(y_label) > 0:
		plt.xlabel(y_label)
	else:
		plt.xlabel(data_name)
	plt.ylabel('Number of Occurrences')
	plt.title(data_name + ' histogram')
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '.png')
	plt.savefig(save_dir + im_name + '.svg')
	plt.close()


def plot_null_v_true_stats(true_data, null_data, data_name, save_dir, x_label=[]):
	"""General function to plot given null and true statistics
	true_data is given as a numpy array
	null_data is given as a dictionary with keys = null index and values = numpy arrays
	"""
	plt.figure(figsize=(5, 5))
	null_vals = []
	null_x_vals = []
	for key in null_data.keys():
		null_vals.extend(list(null_data[key]))
		null_x_vals.extend([int(key)])
	mean_null_vals = np.mean(null_vals)
	std_null_vals = np.std(null_vals)
	# Plot the histograms
	plt.subplot(3, 1, 1)
	plt.hist(true_data, bins=25, color='b', alpha=0.4, label='True')
	plt.hist(null_vals, bins=25, color='g', alpha=0.4, label='Null')
	plt.legend()
	plt.xlabel(x_label)
	plt.ylabel('Number of Occurrences')
	plt.title(data_name + ' histogram')
	# Plot the probability distribution functions
	plt.subplot(3, 1, 2)
	plt.hist(true_data, bins=25, density=True,
			 histtype='step', color='b', label='True')
	plt.hist(null_vals, bins=25, density=True,
			 histtype='step', color='g', label='Null')
	plt.legend()
	plt.xlabel(x_label)
	plt.ylabel('PDF')
	plt.title(data_name + ' PDF')
	# Plot the cumulative distribution functions
	plt.subplot(3, 1, 3)
	plt.hist(true_data, bins=25, density=True, cumulative=True,
			 histtype='step', color='b', label='True')
	plt.hist(null_vals, bins=25, density=True, cumulative=True,
			 histtype='step', color='g', label='Null')
	plt.legend()
	plt.xlabel(x_label)
	plt.ylabel('CDF')
	plt.title(data_name + ' CDF')
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '_truexnull.png')
	plt.savefig(save_dir + im_name + '_truexnull.svg')
	plt.close()


def calculate_correlations(segment_dev_rasters, tastant_spike_times,
						   start_dig_in_times, end_dig_in_times, segment_names, dig_in_names,
						   pre_taste, post_taste, taste_cp_raster_inds, save_dir,
						   neuron_keep_indices=[]):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval"""

	#Grab parameters
	fr_bin = 50 #ms to bin together for number of spikes 'fr'
	num_tastes = len(start_dig_in_times)
	num_segments = len(segment_dev_rasters)
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
	post_taste_dt = np.ceil(post_taste*1000).astype('int')

	#segment_correlation_data = []
	segment_distance_data = []
	for s_i in range(num_segments):  #Loop through each segment
		print("Beginning correlation calcs for segment " + str(s_i))
		#Gather segment data
		seg_rast = segment_dev_rasters[s_i]
		num_dev = len(seg_rast)
		if len(neuron_keep_indices) == 0:
			total_num_neur = np.shape(seg_rast[0])[0]
			neuron_keep_indices = np.arange(total_num_neur)
		else:
			total_num_neur = len(neuron_keep_indices)
		#taste_correlation_data = []
		taste_distance_data = []
		for t_i in range(num_tastes):  #Loop through each taste
			print("\tTaste #" + str(t_i + 1))
			taste_cp = taste_cp_raster_inds[t_i][:, neuron_keep_indices, :]
			taste_spikes = tastant_spike_times[t_i]
			#Note, num_cp = num_cp+1 with the first value the taste delivery index
			num_deliv, _, num_cp = np.shape(taste_cp)
			taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i][deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
			deliv_adjustment = [start_dig_in_times[t_i][deliv_i] + pre_taste_dt for deliv_i in range(num_deliv)]
			num_deliv, _, num_cp = np.shape(taste_cp)
			#Store the correlation results in a numpy array
			neuron_corr_storage = np.zeros((num_dev, num_deliv, total_num_neur, num_cp-1))
			for dev_i in tqdm.tqdm(range(num_dev)): #Loop through all deviations
				dev_rast = seg_rast[dev_i][neuron_keep_indices,:]
				dev_len = np.shape(dev_rast)[1]
				end_ind = np.arange(fr_bin,fr_bin+dev_len)
				end_ind[end_ind > dev_len] = dev_len
				dev_rast_binned = np.zeros(np.shape(dev_rast))
				for start_ind in range(dev_len):
					dev_rast_binned[:,start_ind] = np.sum(dev_rast[:,start_ind:end_ind[start_ind]],1)
				
				inputs = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
				 itertools.repeat(neuron_keep_indices), itertools.repeat(taste_cp), \
					 deliv_adjustment, itertools.repeat(dev_rast_binned), itertools.repeat(fr_bin))
				pool = Pool(4)
				deliv_corr_storage = pool.map(cdcp.deliv_corr_parallelized, inputs)
				neuron_corr_storage[dev_i,:,:,:] = np.array(deliv_corr_storage)
			
			#Save to a numpy array
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
			np.save(filename,neuron_corr_storage)
		#segment_correlation_data.append(taste_correlation_data)	
		segment_distance_data.append(taste_distance_data)
	

def calculate_distances(segment_dev_rasters, tastant_spike_times,
						   start_dig_in_times, end_dig_in_times, segment_names,
						   dig_in_names, pre_taste, post_taste, taste_cp_raster_inds, 
						   save_dir,neuron_keep_indices=[]):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval. Outputs are saved .npy files with name indicating
	segment and taste containing matrices of shape [num_dev, num_deliv, num_neur, num_cp]
	with the distances stored."""
	
	#Grab parameters
	fr_bin = 10 #ms to bin together for number of spikes 'fr'
	num_tastes = len(start_dig_in_times)
	num_segments = len(segment_dev_rasters)
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
	post_taste_dt = np.ceil(post_taste*1000).astype('int')

	for s_i in range(num_segments):  #Loop through each segment
		print("Beginning distance calcs for segment " + str(s_i))
		#Gather segment data
		seg_rast = segment_dev_rasters[s_i]
		num_dev = len(seg_rast)
		if len(neuron_keep_indices) == 0:
			total_num_neur = np.shape(seg_rast[0])[0]
			neuron_keep_indices = np.arange(total_num_neur)
		else:
			total_num_neur = len(neuron_keep_indices)
		for t_i in range(num_tastes):  #Loop through each taste
			print("\tTaste #" + str(t_i + 1))
			taste_cp = taste_cp_raster_inds[t_i][:, neuron_keep_indices, :]
			taste_spikes = tastant_spike_times[t_i]
			#Note, num_cp = num_cp+1 with the first value the taste delivery index
			num_deliv, _, num_cp = np.shape(taste_cp)
			taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i][deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
			deliv_adjustment = [start_dig_in_times[t_i][deliv_i] + pre_taste_dt for deliv_i in range(num_deliv)]
			#Store the correlation results in a numpy array
			neuron_distance_storage = np.zeros((num_dev, num_deliv, total_num_neur, num_cp-1))
			for dev_i in tqdm.tqdm(range(num_dev)): #Loop through all deviations
				dev_rast = seg_rast[dev_i][neuron_keep_indices,:]
				dev_len = np.shape(dev_rast)[1]
				end_ind = np.arange(fr_bin,fr_bin+dev_len)
				end_ind[end_ind > dev_len] = dev_len
				dev_rast_binned = np.zeros(np.shape(dev_rast))
				for start_ind in range(dev_len):
					dev_rast_binned[:,start_ind] = np.sum(dev_rast[:,start_ind:end_ind[start_ind]],1)
				
				inputs = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
				 itertools.repeat(neuron_keep_indices), itertools.repeat(taste_cp), \
					 deliv_adjustment, itertools.repeat(dev_rast_binned), itertools.repeat(fr_bin))
				pool = Pool(4)
				deliv_distance_storage = pool.map(cdcp.deliv_dist_parallelized, inputs)
				neuron_distance_storage[dev_i,:,:,:] = np.array(deliv_distance_storage)
			#Save to a numpy array
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
			np.save(filename,neuron_distance_storage)

def plot_stats(segment_names, dig_in_names, pre_taste, post_taste, taste_cp_raster_inds, 
						   save_dir,dist_name):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval. Outputs are saved .npy files with name indicating
	segment and taste containing matrices of shape [num_dev, num_deliv, num_neur, num_cp]
	with the distances stored."""
	
	#Grab parameters
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	for s_i in range(num_segments):  #Loop through each segment
		print("Beginning distance calcs for segment " + str(s_i))
		for t_i in range(num_tastes):  #Loop through each taste
			print("\tTaste #" + str(t_i + 1))
			#Import distance numpy array
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
			neuron_distance_storage = np.load(filename)
			num_dev, num_deliv, total_num_neur, num_cp = np.shape(neuron_distance_storage)
			#Plot the distribution of distances for each changepoint index
			f = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for c_p in range(num_cp):
				all_dist_cp = (neuron_distance_storage[:,:,:,c_p]).flatten()
				plt.hist(all_dist_cp[all_dist_cp!=0],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_dist_cp = (neuron_distance_storage[:,:,:,c_p]).flatten()
				plt.hist(all_dist_cp[all_dist_cp!=0],density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle(dist_name + ' distributions for segment ' + segment_names[s_i] + ' taste ' + dig_in_names[t_i])
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i]
			plt.savefig(f,filename + '.png')
			plt.savefig(f,filename + '.svg')
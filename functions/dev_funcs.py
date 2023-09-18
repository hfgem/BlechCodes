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
from scipy.stats import pearsonr, ks_2samp
import functions.corr_dist_calc_parallel as cdcp
from multiprocessing import Pool
from sklearn.decomposition import PCA

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
	# Find where the firing rate is above 3std from the mean
	fr_calc = np.array([np.sum(spike_sum[i_s - half_min_dev_size:i_s + half_min_dev_size]) /
					   min_dev_size for i_s in np.arange(half_min_dev_size, num_dt-half_min_dev_size)])  # in spikes per min_dev_size
	local_fr_mean = []
	local_fr_std = []
	for i_s in np.arange(half_min_dev_size,num_dt-half_min_dev_size):
		min_ind = max(i_s - half_local_size,0)
		max_ind = min(num_dt-half_min_dev_size,i_s+half_local_size)
		local_fr_mean.append(np.mean(fr_calc[min_ind:max_ind]))
		local_fr_std.append(np.std(fr_calc[min_ind:max_ind]))
	local_fr_mean = np.array(local_fr_mean)
	local_fr_std = np.array(local_fr_std)
	peak_fr_ind = np.where(fr_calc >= local_fr_mean + 3*local_fr_std)[0]
	true_ind = peak_fr_ind + half_min_dev_size
	# Find where the prominence is above the 90th percentile of prominence values
	#fr_calc = np.array([np.sum(spike_sum[i_s - half_min_dev_size:i_s + half_min_dev_size]) /
	#				   min_dev_size for i_s in np.arange(half_local_size, num_dt-half_local_size)])  # in spikes per min_dev_size
	#local_fr_calc = np.array([np.sum(spike_sum[l_s - half_local_size:l_s + half_local_size]) /
	#						 local_size for l_s in np.arange(half_local_size, num_dt-half_local_size)])  # in spikes per min_dev_size
	#peak_fr_ind = find_peaks(fr_calc - local_fr_calc, height=0)[0]
	#peak_fr_prominence = fr_calc[peak_fr_ind] - local_fr_calc[peak_fr_ind]
	#top_90_prominence = np.percentile(peak_fr_prominence, 90)
	#top_90_prominence_ind = peak_fr_ind[np.where(
	#	peak_fr_prominence >= top_90_prominence)[0]]
	#true_ind = top_90_prominence_ind + half_local_size
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


def plot_dev_rasters(segment_deviations,segment_spike_times,segment_dev_times,
					 segment_times_reshaped,pre_taste,post_taste,segment_names,dev_dir):
	num_segments = len(segment_names)
	for s_i in range(num_segments):
		print("Plotting deviations in segment " + segment_names[s_i])
		filepath = dev_dir + segment_names[s_i] + '/'
		indiv_dev_filepath = filepath + 'indiv_dev/'
		if os.path.isdir(indiv_dev_filepath) == False:
			os.mkdir(indiv_dev_filepath)
		#Plot when deviations occur
		f = plt.figure(figsize=(5,5))
		plt.plot(segment_deviations[s_i])
		plt.title('Segment ' + segment_names[s_i] + ' deviations')
		x_ticks = plt.xticks()[0]
		x_tick_labels = [np.round(x_ticks[i]/1000/60,2) for i in range(len(x_ticks))]
		plt.xticks(x_ticks,x_tick_labels)
		plt.xlabel('Time (min)')
		plt.yticks([0,1],['No Dev','Dev'])
		plt.tight_layout()
		fig_name = filepath + 'all_deviations'
		f.savefig(fig_name + '.png')
		f.savefig(fig_name + '.svg')
		plt.close(f)
		#Plot individual segments with pre and post time
		segment_rasters = segment_spike_times[s_i]
		segment_times = segment_dev_times[s_i] + segment_times_reshaped[s_i][0]
		num_neur = len(segment_rasters)
		num_deviations = len(segment_times[0,:])
		for dev_i in tqdm.tqdm(range(num_deviations)):
			dev_times = segment_times[:,dev_i]
			dev_buffer = dev_times[1] - dev_times[0]
			dev_rast_ind = []
			for n_i in range(num_neur):
				seg_dev_ind = np.where((segment_rasters[n_i] > dev_times[0] - dev_buffer)*(segment_rasters[n_i] < dev_times[1] + dev_buffer))[0]
				dev_rast_ind.append(np.array(segment_rasters[n_i])[seg_dev_ind])
			f1 = plt.figure(figsize=(5,5))
			plt.eventplot(dev_rast_ind,colors='b',alpha=0.5)
			plt.axvline(dev_times[0])
			plt.axvline(dev_times[1])
			x_ticks = plt.xticks()[0]
			if dev_times[0]/1000 > 99:
				x_tick_labels = [np.round(x_ticks[i]/1000/60,2) for i in range(len(x_ticks))]
				plt.xticks(x_ticks,x_tick_labels)
				plt.xlabel('Time (m)')
			else:
				x_tick_labels = [np.round(x_ticks[i]/1000,2) for i in range(len(x_ticks))]
				plt.xticks(x_ticks,x_tick_labels)
				plt.xlabel('Time (s)')
			plt.ylabel('Neuron Index')
			plt.title('Deviation ' + str(dev_i))
			fig_name = indiv_dev_filepath + 'dev_' + str(dev_i)
			f1.savefig(fig_name + '.png')
			f1.savefig(fig_name + '.svg')
			plt.close(f1)
		

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
	fr_bin = 25 #ms to bin together for number of spikes 'fr'
	num_tastes = len(start_dig_in_times)
	num_segments = len(segment_dev_rasters)
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
	post_taste_dt = np.ceil(post_taste*1000).astype('int')

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
			
		for t_i in range(num_tastes):  #Loop through each taste
		
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
			try:
				neuron_corr_storage = np.load(filename)
				print("\tTaste #" + str(t_i + 1) + 'previously calculated')
			except:
				print("\tCalculating Taste #" + str(t_i + 1))
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
					pool.close()
					neuron_corr_storage[dev_i,:,:,:] = np.array(deliv_corr_storage)
				
				#Save to a numpy array
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
				np.save(filename,neuron_corr_storage)


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
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
			try:
				neuron_corr_storage = np.load(filename)
				print("\tTaste #" + str(t_i + 1) + ' previously calculated')
			except:
				print("\tCalculating Taste #" + str(t_i + 1))
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
					pool.close()
					neuron_distance_storage[dev_i,:,:,:] = np.array(deliv_distance_storage)
				#Save to a numpy array
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
				np.save(filename,neuron_distance_storage)


def pull_corr_dev_stats(segment_names, dig_in_names, save_dir):
	"""For each epoch and each segment pull out the top 10 most correlated deviation 
	bins and plot side-by-side with the epoch they are correlated with"""
	
	#Grab parameters
	dev_stats = dict()
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	for s_i in range(num_segments):  #Loop through each segment
		segment_stats = dict()
		for t_i in range(num_tastes):  #Loop through each taste
			#Import distance numpy array
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
			neuron_data_storage = np.load(filename)
			#Calculate statistics
			data_dict = dict()
			data_dict['segment'] = segment_names[s_i]
			data_dict['taste'] = dig_in_names[t_i]
			num_dev, num_deliv, total_num_neur, num_cp = np.shape(neuron_data_storage)
			data_dict['num_dev'] = num_dev
			data_dict['neuron_data_storage'] = neuron_data_storage
			segment_stats[t_i] = data_dict
		dev_stats[s_i] = segment_stats

	return dev_stats

def plot_stats(dev_stats, segment_names, dig_in_names, save_dir, dist_name, 
			   neuron_indices):
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
		seg_stats = dev_stats[s_i]
		for t_i in range(num_tastes):  #Loop through each taste
			print("\tTaste #" + str(t_i + 1))
			taste_stats = seg_stats[t_i]
			#Import distance numpy array
			neuron_data_storage = taste_stats['neuron_data_storage']
			num_dev, num_deliv, total_num_neur, num_cp = np.shape(neuron_data_storage)
			#Plot the individual neuron distribution for each changepoint index
			f = plt.figure(figsize=(5,5))
			cp_data = []
			plt.subplot(2,1,1)
			for c_p in range(num_cp):
				all_dist_cp = (neuron_data_storage[:,:,neuron_indices,c_p]).flatten()
				cp_data.append(all_dist_cp)
				plt.hist(all_dist_cp[all_dist_cp != 0],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_dist_cp = cp_data[c_p]
				plt.hist(all_dist_cp[all_dist_cp != 0],density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i]
			f.savefig(filename + '.png')
			f.savefig(filename + '.svg')
			plt.close(f)
			if dist_name == 'Correlation':
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				for c_p in range(num_cp):
					all_dist_cp = (neuron_data_storage[:,:,neuron_indices,c_p]).flatten()
					cp_data.append(all_dist_cp)
					plt.hist(all_dist_cp[all_dist_cp >= 0.5],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					plt.hist(all_dist_cp[all_dist_cp >= 0.5],density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				for c_p in range(num_cp):
					all_dist_cp = (neuron_data_storage[:,:,neuron_indices,c_p]).flatten()
					cp_data.append(all_dist_cp)
					plt.hist(all_dist_cp[all_dist_cp >= 0.5],density=True,log=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					plt.hist(all_dist_cp[all_dist_cp >= 0.5],density=True,log=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_log_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
			#Plot the population average distribution for each changepoint index
			f1 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			avg_cp_data = []
			for c_p in range(num_cp):
				all_avg_dist_cp = (np.nanmean(neuron_data_storage[:,:,neuron_indices,c_p],2)).flatten()
				avg_cp_data.append(all_avg_dist_cp)
				plt.hist(all_avg_dist_cp[all_avg_dist_cp != 0],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel('Average ' + dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_avg_dist_cp = avg_cp_data[c_p]
				plt.hist(all_avg_dist_cp[all_avg_dist_cp != 0],density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel('Average ' + dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle(dist_name + ' avg population distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_avg_pop'
			f1.savefig(filename + '.png')
			f1.savefig(filename + '.svg')
			plt.close(f1)
			if dist_name == 'Correlation':
				#Zoomed to high correlation
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				avg_cp_data = []
				for c_p in range(num_cp):
					all_avg_dist_cp = (np.nanmean(neuron_data_storage[:,:,neuron_indices,c_p],2)).flatten()
					avg_cp_data.append(all_avg_dist_cp)
					plt.hist(all_avg_dist_cp[all_avg_dist_cp >= 0.5],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel('Average ' + dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for c_p in range(num_cp):
					all_avg_dist_cp = avg_cp_data[c_p]
					plt.hist(all_avg_dist_cp[all_avg_dist_cp >= 0.5],density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel('Average ' + dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' avg population distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_avg_pop_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Now with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				avg_cp_data = []
				for c_p in range(num_cp):
					all_avg_dist_cp = (np.nanmean(neuron_data_storage[:,:,neuron_indices,c_p],2)).flatten()
					avg_cp_data.append(all_avg_dist_cp)
					plt.hist(all_avg_dist_cp[all_avg_dist_cp >= 0.5],density=True,log=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel('Average ' + dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for c_p in range(num_cp):
					all_avg_dist_cp = avg_cp_data[c_p]
					plt.hist(all_avg_dist_cp[all_avg_dist_cp >= 0.5],density=True,log=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel('Average ' + dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' avg population distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_avg_pop_log_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
			
			
def plot_combined_stats(dev_stats, segment_names, dig_in_names, save_dir, 
						dist_name, neuron_indices):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval. Outputs are saved .npy files with name indicating
	segment and taste containing matrices of shape [num_dev, num_deliv, num_neur, num_cp]
	with the distances stored."""
	
	#Grab parameters
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	
	#Define storage
	segment_data = [] #segments x tastes x cp
	segment_data_avg = [] #segments x tastes x cp avged across neurons in the deviation
	for s_i in range(num_segments):  #Loop through each segment
		seg_stats = dev_stats[s_i]
		print("Beginning calcs for segment " + str(s_i))
		taste_data = [] #tastes x cp
		taste_data_avg = []
		for t_i in range(num_tastes):  #Loop through each taste
			print("\tTaste #" + str(t_i + 1))
			taste_stats = seg_stats[t_i]
			#Import distance numpy array
			neuron_data_storage = taste_stats['neuron_data_storage']
			num_dev, num_deliv, total_num_neur, num_cp = np.shape(neuron_data_storage)
			cp_data = []
			cp_data_avg = []
			for c_p in range(num_cp):
				all_dist_cp = (neuron_data_storage[:,:,neuron_indices,c_p]).flatten()
				cp_data.append(all_dist_cp)
				avg_dist_cp = np.nanmean(neuron_data_storage[:,:,neuron_indices,c_p],2).flatten()
				cp_data_avg.append(avg_dist_cp)
			taste_data.append(cp_data)
			taste_data_avg.append(cp_data_avg)
		segment_data.append(taste_data)
		segment_data_avg.append(taste_data_avg)
		#Plot taste data against each other
		for c_p in range(num_cp):
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_data[t_i][c_p]
					plt.hist(data[data != 0],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_data[t_i][c_p]
					plt.hist(data[data != 0],density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p)
			f0.savefig(filename + '.png')
			f0.savefig(filename + '.svg')
			plt.close(f0)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				#Plot data across all neurons
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Plot data across all neurons with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_log_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
		#Now plot population averages
		for c_p in range(num_cp):
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_data_avg[t_i][c_p]
					plt.hist(data[data != 0],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_data_avg[t_i][c_p]
					plt.hist(data[data != 0],density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_avg_pop'
			f0.savefig(filename + '.png')
			f0.savefig(filename + '.svg')
			plt.close(f0)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				#Plot data across all neurons
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p)  + '_avg_pop' + '_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Plot data across all neurons with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p)  + '_avg_pop' + '_log_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
			
	for t_i in range(num_tastes): #Loop through each taste
		#Plot segment data against each other by epoch
		for c_p in range(num_cp):
			f2 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for s_i in range(num_segments):
				try:
					data = segment_data_avg[s_i][t_i][c_p]
					plt.hist(data[data != 0],density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for s_i in range(num_segments):
				try:
					data = segment_data_avg[s_i][t_i][c_p]
					plt.hist(data[data != 0],density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_avg'
			f2.savefig(filename + '.png')
			f2.savefig(filename + '.svg')
			plt.close(f2)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						plt.hist(data[data > 0.5],density=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						plt.hist(data[data > 0.5],density=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_avg_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
				#Log
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						plt.hist(data[data > 0.5],density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						plt.hist(data[data > 0.5],density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_avg_log_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
	
		
	return segment_data, segment_data_avg
		

def stat_significance(segment_data, segment_names, dig_in_names, save_dir, dist_name):
	
	#Grab parameters
	#segment_data shape = segments x tastes x cp
	num_segments = len(segment_data)
	num_tastes = len(segment_data[0])
	num_cp = len(segment_data[0][0])
	
	#Calculate statistical significance of pairs
	#Are the correlation distributions significantly different across pairs?
	
	#All pair combinations
	a = [list(np.arange(num_segments)),list(np.arange(num_tastes)),list(np.arange(num_cp))]
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
			data_1 = segment_data[ind_1[0]][ind_1[1]][ind_1[2]]
			data_2 = segment_data[ind_2[0]][ind_2[1]][ind_2[2]]
			result = ks_2samp(data_1,data_2)
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
			
			
def top_dev_corr_bins(dev_stats,segment_names,dig_in_names,save_dir):
	"""Calculate which deviation index is most correlated with which taste 
	delivery and which epoch and store to a text file."""
	
	#Grab parameters
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	
	
	#Define storage
	for s_i in range(num_segments):  #Loop through each segment
		seg_stats = dev_stats[s_i]
		print("Beginning calcs for segment " + str(s_i))
		for t_i in range(num_tastes):  #Loop through each taste
			save_file = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_top_corr_combos.txt'
			corr_data = []
			print("\tTaste #" + str(t_i + 1))
			taste_stats = seg_stats[t_i]
			#Import distance numpy array
			neuron_data_storage = taste_stats['neuron_data_storage']
			num_dev, num_deliv, total_num_neur, num_cp = np.shape(neuron_data_storage)
			#Calculate, for each deviation bin, which taste delivery and cp it correlates with most
			all_dev_data = np.nanmean(neuron_data_storage,2) #num_dev x num_deliv x num_cp 
			top_99_percentile = np.percentile(all_dev_data[~np.isnan(all_dev_data)].flatten(),99)
			for dev_i in range(num_dev):
				dev_data = all_dev_data[dev_i,:,:]  #num_deliv x num_cp
				[deliv_i,cp_i] = np.where(dev_data >= top_99_percentile)
				if len(deliv_i) > 0:
					for d_i in range(len(deliv_i)):
						dev_cp_corr_val = dev_data[deliv_i[d_i],cp_i[d_i]]
						statement = 'dev-' + str(dev_i) + '; epoch-' + str(cp_i[d_i]) + '; deliv-' + str(deliv_i[d_i]) + '; corr-' + str(dev_cp_corr_val)
						corr_data.append(statement)
			#Save to file
			with open(save_file, 'w') as f:
				for line in corr_data:
					f.write(line)
					f.write('\n')
	




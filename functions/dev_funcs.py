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
import functions.corr_dist_calc_parallel as cdcp
import functions.corr_dist_calc_parallel_pop as cdcpp
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
		if len(start_dev_bouts) > len(end_dev_bouts):
			end_dev_bouts = np.append(end_dev_bouts,num_dt)
		if len(end_dev_bouts) > len(start_dev_bouts):
			start_dev_bouts = np.insert(start_dev_bouts,0,0)
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
	dev_buffer = 100 #ms before and after a deviation to plot
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
			dev_start = int(dev_times[0])
			dev_len = dev_times[1] - dev_start
			dev_rast_ind = []
			raster_len = 2*dev_buffer + dev_len
			dev_binary = np.zeros((num_neur,raster_len))
			for n_i in range(num_neur):
				segment_neur_rast = np.array(segment_rasters[n_i])
				seg_dev_ind = np.where((segment_neur_rast > dev_start - dev_buffer)*(segment_neur_rast < dev_times[1] + dev_buffer))[0]
				seg_dev_rast_ind = segment_neur_rast[seg_dev_ind]
				seg_dev_rast_ind_shift = (seg_dev_rast_ind - dev_start + dev_buffer).astype('int')
				dev_binary[n_i,seg_dev_rast_ind_shift] = 1
				dev_rast_ind.append(seg_dev_rast_ind)
			#Create firing rates matrix
			firing_rate_vec = np.zeros(raster_len)
			for t_i in range(raster_len):
				min_t_i = max(t_i-25,0)
				max_t_i = min(t_i+25,raster_len)
				firing_rate_vec[t_i] = np.mean(np.sum(dev_binary[:,min_t_i:max_t_i],1)/(50/1000))
			rate_x_tick_labels = np.arange(-1*dev_buffer,dev_len + dev_buffer)
			#Now plot the rasters with firing rate deviations
			f1, ax1 = plt.subplots(nrows=2,ncols=1,figsize=(5,5),gridspec_kw=dict(height_ratios=[2,1]))
			#Deviation Raster Plot
			adjusted_dev_rast_ind = [list(np.array(dev_rast_ind[n_i]) - dev_start) for n_i in range(num_neur)]
			ax1[0].eventplot(adjusted_dev_rast_ind,colors='b',alpha=0.5)
			ax1[0].axvline(0)
			ax1[0].axvline(dev_len)
			ax1[0].set_ylabel('Neuron Index')
			ax1[0].set_title('Deviation ' + str(dev_i))
			#Deviation population activity plot
			ax1[1].plot(rate_x_tick_labels,firing_rate_vec)
			ax1[1].axvline(0)
			ax1[1].axvline(dev_len)
			ax1[1].set_xlabel('Time (ms)')
			ax1[1].set_ylabel('Population rate (Hz)')
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
		
	#Now plot stats across iterations
	plot_dev_stats_dict(length_dict, iteration_names, 'Deviation Lengths', save_dir, 'Segment', 'Length (ms)')
	plot_dev_stats_dict(IDI_dict, iteration_names, 'Inter-Deviation-Intervals', save_dir, 'Segment', 'Length (ms)')
	plot_dev_stats_dict(num_spike_dict, iteration_names, 'Total Spike Count', save_dir, 'Segment', '# Spikes')
	plot_dev_stats_dict(num_neur_dict, iteration_names, 'Total Neuron Count', save_dir, 'Segment', '# Neurons')

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


def plot_dev_stats_dict(dict_data, iteration_names, data_name, save_dir, x_label, y_label):
	labels, data = [*zip(*dict_data.items())]
	#Calculate pairwise significant differences
	x_ticks = np.arange(1,len(labels)+1)
	pairs = list(itertools.combinations(x_ticks,2))
	pair_sig = np.zeros(len(pairs))
	#cross-group stat sig
	args = [d for d in data]
	try:
		kw_stat, kw_p_val = kruskal(*args,nan_policy='omit')
	except:
		kw_p_val = 1
	#pairwise stat sig
	for pair_i in range(len(pairs)):
		pair = pairs[pair_i]
		ks_pval = ks_2samp(data[pair[0]-1],data[pair[1]-1])[1]
		if ks_pval < 0.05:
			pair_sig[pair_i] = 1
	
	#Plot distributions as box and whisker plots comparing across iterations
	bw_fig = plt.figure(figsize=(5,5))
	plt.boxplot(data)
	plt.xticks(x_ticks,labels=iteration_names)
	y_ticks = plt.yticks()[0]
	y_max = np.max(y_ticks)
	y_range = y_max - np.min(y_ticks)
	x_mean = np.mean(x_ticks)
	if kw_p_val <= 0.05:
		plt.plot([x_ticks[0],x_ticks[-1]],[y_max+0.05*y_range,y_max+0.05*y_range],color='k')
		plt.scatter(x_mean,y_max+0.1*y_range,marker='*',color='k')
	jitter_vals = np.linspace(0.9*y_max,y_max,len(pairs))
	step = np.mean(np.diff(jitter_vals))
	for pair_i in range(len(pairs)):
		pair = pairs[pair_i]
		if pair_sig[pair_i] == 1:
			plt.plot([pair[0],pair[1]],[jitter_vals[pair_i],jitter_vals[pair_i]],color='k',linestyle='dashed')
			plt.scatter((pair[0]+pair[1])/2,jitter_vals[pair_i]+step,marker='*',color='k')
	plt.title(data_name)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '_box.png')
	plt.savefig(save_dir + im_name + '_box.svg')
	plt.close(bw_fig)
	#Plot distributions as violin plots
	violin_fig = plt.figure(figsize=(5,5))
	plt.violinplot(data,positions=np.arange(1,len(labels)+1))
	plt.xticks(range(1,len(labels)+1),labels=iteration_names)
	y_ticks = plt.yticks()[0]
	y_max = np.max(y_ticks)
	y_range = y_max - np.min(y_ticks)
	x_mean = np.mean(x_ticks)
	if kw_p_val <= 0.05:
		plt.plot([x_ticks[0],x_ticks[-1]],[y_max+0.05*y_range,y_max+0.05*y_range],color='k')
		plt.scatter(x_mean,y_max+0.1*y_range,marker='*',color='k')
	jitter_vals = np.linspace(0.9*y_max,y_max,len(pairs))
	step = np.mean(np.diff(jitter_vals))
	for pair_i in range(len(pairs)):
		pair = pairs[pair_i]
		if pair_sig[pair_i] == 1:
			plt.plot([pair[0],pair[1]],[jitter_vals[pair_i],jitter_vals[pair_i]],color='k',linestyle='dashed')
			plt.scatter((pair[0]+pair[1])/2,jitter_vals[pair_i]+step,marker='*',color='k')
	plt.title(data_name)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '_violin.png')
	plt.savefig(save_dir + im_name + '_violin.svg')
	plt.close(violin_fig)
	#Plot distributions as PDFs
	pdf_fig = plt.figure(figsize=(3,3))
	for d_i in range(len(data)):
		plt.hist(data[d_i],label=iteration_names[d_i],density=True,histtype='step')
	plt.legend()
	plt.title(data_name)
	plt.xlabel(y_label)
	plt.ylabel('Probability')
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '_pdf.png')
	plt.savefig(save_dir + im_name + '_pdf.svg')
	plt.close(pdf_fig)
	#Plot distributions as CDFs
	cdf_fig = plt.figure(figsize=(3,3))
	for d_i in range(len(data)):
		plt.hist(data[d_i],bins=np.arange(0,np.max(data[d_i])),label=iteration_names[d_i],density=True,cumulative=True,histtype='step')
	plt.legend()
	plt.title(data_name)
	plt.xlabel(y_label)
	plt.ylabel('Probability')
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '_cdf.png')
	plt.savefig(save_dir + im_name + '_cdf.svg')
	plt.close(cdf_fig)


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
	true_sort = np.sort(true_data)
	true_unique = np.unique(true_sort)
	cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
	null_sort = np.sort(null_vals)
	null_unique = np.unique(null_sort)
	cmf_null = np.array([np.sum((null_sort <= u_val).astype('int'))/len(null_sort) for u_val in null_unique])
	plt.plot(true_unique,cmf_true, color='b', label='True')
	plt.plot(null_unique,cmf_null, color='g', label='Null')
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
				neuron_keep_indices = np.arange(total_num_neur)
			else:
				total_num_neur = np.sum(neuron_keep_indices[:,t_i]).astype('int')
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
				taste_neuron_keep_indices = np.where(neuron_keep_indices[:,t_i])[0]
				taste_cp = taste_cp_raster_inds[t_i][:, taste_neuron_keep_indices, :]
				taste_cp_pop = pop_taste_cp_raster_inds[t_i]
				taste_spikes = tastant_spike_times[t_i]
				#Note, num_cp = num_cp+1 with the first value the taste delivery index
				num_deliv, _, num_cp = np.shape(taste_cp)
				taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i][deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
				deliv_adjustment = [start_dig_in_times[t_i][deliv_i] + pre_taste_dt for deliv_i in range(num_deliv)]
				num_deliv, _, num_cp = np.shape(taste_cp)
				#Store the correlation results in a numpy array
				neuron_corr_storage = np.nan*np.ones((num_dev, num_deliv, total_num_neur, num_cp-1))
				neuron_pop_corr_storage = np.nan*np.ones((num_dev, num_deliv, num_cp-1))
				for dev_i in tqdm.tqdm(range(num_dev)): #Loop through all deviations
					dev_rast = seg_rast[dev_i][taste_neuron_keep_indices,:]
					dev_len = np.shape(dev_rast)[1]
					end_ind = np.arange(fr_bin,fr_bin+dev_len)
					end_ind[end_ind > dev_len] = dev_len
					#TODO: test gaussian convolution instead of binning
					dev_rast_binned = np.zeros(np.shape(dev_rast)) #timeseries information kept
					for start_ind in range(dev_len):
						dev_rast_binned[:,start_ind] = np.sum(dev_rast[:,start_ind:end_ind[start_ind]],1)
					dev_rast_vec = np.sum(dev_rast,1)/(dev_len/1000) #Converted to Hz
					#Individual neuron changepoints
					if filename_loaded == 0:
						inputs = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
						 itertools.repeat(np.arange(0,len(taste_neuron_keep_indices))), itertools.repeat(taste_cp), \
							 deliv_adjustment, itertools.repeat(dev_rast_binned), itertools.repeat(fr_bin))
						pool = Pool(4)
						deliv_corr_storage = pool.map(cdcp.deliv_corr_parallelized, inputs)
						pool.close()
						neuron_corr_storage[dev_i,:,:,:] = np.array(deliv_corr_storage)
					#Population changepoints
					if filename_pop_loaded == 0:
						pool = Pool(4)
						inputs_pop = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
							itertools.repeat(np.arange(0,len(taste_neuron_keep_indices))), itertools.repeat(taste_cp_pop), \
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
						   save_dir, neuron_keep_indices=[]):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval"""

	#Grab parameters
	num_tastes = len(start_dig_in_times)
	num_segments = len(segment_dev_rasters)
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
	post_taste_dt = np.ceil(post_taste*1000).astype('int')

	for s_i in range(num_segments):  #Loop through each segment
		print("Beginning population vector correlation calcs for segment " + str(s_i))
		#Gather segment data
		seg_rast = segment_dev_rasters[s_i]
		num_dev = len(seg_rast)
			
		for t_i in range(num_tastes):  #Loop through each taste
			#Find the number of neurons
			if np.shape(neuron_keep_indices)[0] == 0:
				total_num_neur = np.shape(seg_rast[0])[0]
				neuron_keep_indices = np.arange(total_num_neur)
			else:
				total_num_neur = np.sum(neuron_keep_indices[:,t_i]).astype('int')
		
			#Set storage directory and check if data previously stored
			filename_pop_vec = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec.npy'
			filename_pop_vec_loaded = 0
			try:
				neuron_pop_vec_corr_storage = np.load(filename_pop_vec_loaded)
				filename_pop_vec_loaded = 1
			except:
				pass
			if filename_pop_vec_loaded == 0:
				print("\tCalculating Taste #" + str(t_i + 1))
				taste_cp = taste_cp_raster_inds[t_i][:, neuron_keep_indices, :]
				taste_cp_pop = pop_taste_cp_raster_inds[t_i]
				taste_spikes = tastant_spike_times[t_i]
				#Note, num_cp = num_cp+1 with the first value the taste delivery index
				num_deliv, _, num_cp = np.shape(taste_cp)
				taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i][deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
				deliv_adjustment = [start_dig_in_times[t_i][deliv_i] + pre_taste_dt for deliv_i in range(num_deliv)]
				num_deliv, _, num_cp = np.shape(taste_cp)
				#Store the correlation results in a numpy array
				neuron_pop_vec_corr_storage = np.nan*np.ones((num_dev, num_deliv, num_cp-1))
				for dev_i in tqdm.tqdm(range(num_dev)): #Loop through all deviations
					dev_rast = seg_rast[dev_i][neuron_keep_indices,:]
					dev_len = np.shape(dev_rast)[1]
					dev_vec = np.sum(dev_rast,1)/(dev_len/1000) #in Hz
					#Population fr vector changepoints
					pool = Pool(4)
					inputs_pop = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
						itertools.repeat(neuron_keep_indices), itertools.repeat(taste_cp_pop), \
						deliv_adjustment, itertools.repeat(dev_vec))
					deliv_vec_corr_storage = pool.map(cdcpp.deliv_corr_population_vec_parallelized, inputs_pop)
					pool.close()
					neuron_pop_vec_corr_storage[dev_i,:,:] = np.array(deliv_vec_corr_storage)
				np.save(filename_pop_vec,neuron_pop_vec_corr_storage)


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
			filename_pop = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop.npy'
			filename_pop_vec = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec.npy'
			neuron_data_storage = np.load(filename)
			population_data_storage = np.load(filename_pop)
			population_vec_data_storage = np.load(filename_pop_vec)
			#Calculate statistics
			data_dict = dict()
			data_dict['segment'] = segment_names[s_i]
			data_dict['taste'] = dig_in_names[t_i]
			num_dev, num_deliv, total_num_neur, num_cp = np.shape(neuron_data_storage)
			data_dict['num_dev'] = num_dev
			data_dict['neuron_data_storage'] = neuron_data_storage
			data_dict['pop_data_storage'] = population_data_storage
			data_dict['pop_vec_data_storage'] = population_vec_data_storage
			segment_stats[t_i] = data_dict
		dev_stats[s_i] = segment_stats

	return dev_stats


def plot_stats(dev_stats, segment_names, dig_in_names, save_dir, dist_name, 
			   neuron_indices):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval. Outputs are saved .npy files with name indicating
	segment and taste containing matrices of shape [num_dev, num_deliv, num_neur, num_cp]
	with the distances stored.
	
	neuron_indices should be binary and shaped num_neur x num_cp
	"""
	
	#Grab parameters
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	for s_i in range(num_segments):  #Loop through each segment
		print("Beginning plot calcs for segment " + str(s_i))
		seg_stats = dev_stats[s_i]
		for t_i in range(num_tastes):  #Loop through each taste
			print("\tTaste #" + str(t_i + 1))
			taste_stats = seg_stats[t_i]
			#_____Individual Neuron CP Calculations_____
			#Import correlation numpy array
			neuron_data_storage = taste_stats['neuron_data_storage']
			data_shape = np.shape(neuron_data_storage)
			if len(data_shape) == 4:
				num_dev = data_shape[0]
				num_deliv = data_shape[1]
				total_num_neur = data_shape[2]
				num_cp = data_shape[3]
			elif len(data_shape) == 3:
				num_dev = data_shape[0]
				num_deliv = data_shape[1]
				num_cp = data_shape[2]
			
			#Plot the individual neuron distribution for each changepoint index
			f = plt.figure(figsize=(5,5))
			cp_data = []
			plt.subplot(2,1,1)
			for c_p in range(num_cp):
				all_dist_cp = (neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p]).flatten()
				cp_data.append(all_dist_cp)
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_dist_cp = cp_data[c_p]
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
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
				min_y = 1
				for c_p in range(num_cp):
					all_dist_cp = (neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p]).flatten()
					cp_data.append(all_dist_cp)
					hist_vals = plt.hist(all_dist_cp,density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
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
					all_dist_cp = (neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p]).flatten()
					cp_data.append(all_dist_cp)
					plt.hist(all_dist_cp,density=True,log=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
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
				all_avg_dist_cp = (np.nanmean(neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p],2)).flatten()
				avg_cp_data.append(all_avg_dist_cp)
				plt.hist(all_avg_dist_cp[~np.isnan(all_avg_dist_cp)],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel('Average ' + dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_avg_dist_cp = avg_cp_data[c_p]
				plt.hist(all_avg_dist_cp[~np.isnan(all_avg_dist_cp)],bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
				#plt.ecdf(all_avg_dist_cp,label='Epoch ' + str(c_p))
				#true_sort = np.sort(all_avg_dist_cp)
				#true_unique = np.unique(true_sort)
				#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
				#plt.plot(true_unique,cmf_true,label='Epoch ' + str(c_p))
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
					all_avg_dist_cp = (np.nanmean(neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p],2)).flatten()
					avg_cp_data.append(all_avg_dist_cp)
					plt.hist(all_avg_dist_cp[all_avg_dist_cp >= 0.5],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel('Average ' + dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for c_p in range(num_cp):
					all_avg_dist_cp = avg_cp_data[c_p]
					hist_vals = plt.hist(all_avg_dist_cp,bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.ylim([min_y,1])
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
					all_avg_dist_cp = (np.nanmean(neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p],2)).flatten()
					avg_cp_data.append(all_avg_dist_cp)
					plt.hist(all_avg_dist_cp[all_avg_dist_cp >= 0.5],density=True,log=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel('Average ' + dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for c_p in range(num_cp):
					all_avg_dist_cp = avg_cp_data[c_p]
					hist_vals = plt.hist(all_avg_dist_cp,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.ylim([min_y,1])
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
			
			#_____Population CP Calculations_____
			#Import correlation numpy array
			pop_data_storage = taste_stats['pop_data_storage']
			data_shape = np.shape(pop_data_storage)
			if len(data_shape) == 3:
				num_dev = data_shape[0]
				num_deliv = data_shape[1]
				num_cp = data_shape[2]
			
			#Plot the individual neuron distribution for each changepoint index
			f = plt.figure(figsize=(5,5))
			cp_data = []
			plt.subplot(2,1,1)
			for c_p in range(num_cp):
				all_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
				cp_data.append(all_dist_cp)
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_dist_cp = cp_data[c_p]
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop'
			f.savefig(filename + '.png')
			f.savefig(filename + '.svg')
			plt.close(f)
			if dist_name == 'Correlation':
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				min_y = 1
				for c_p in range(num_cp):
					all_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
					cp_data.append(all_dist_cp)
					hist_vals = plt.hist(all_dist_cp,density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				for c_p in range(num_cp):
					all_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
					cp_data.append(all_dist_cp)
					plt.hist(all_dist_cp,density=True,log=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_log_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
				
			#_____Population Vector CP Calculations_____
			#Import correlation numpy array
			pop_vec_data_storage = taste_stats['pop_vec_data_storage']
			data_shape = np.shape(pop_vec_data_storage)
			if len(data_shape) == 3:
				num_dev = data_shape[0]
				num_deliv = data_shape[1]
				num_cp = data_shape[2]
			#Plot the individual neuron distribution for each changepoint index
			f = plt.figure(figsize=(5,5))
			cp_data = []
			plt.subplot(2,1,1)
			for c_p in range(num_cp):
				all_dist_cp = (pop_vec_data_storage[:,:,c_p]).flatten()
				cp_data.append(all_dist_cp)
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_dist_cp = cp_data[c_p]
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec'
			f.savefig(filename + '.png')
			f.savefig(filename + '.svg')
			plt.close(f)
			if dist_name == 'Correlation':
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				min_y = 1
				for c_p in range(num_cp):
					all_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
					cp_data.append(all_dist_cp)
					hist_vals = plt.hist(all_dist_cp,density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				for c_p in range(num_cp):
					all_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
					cp_data.append(all_dist_cp)
					plt.hist(all_dist_cp,density=True,log=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec_log_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
			

def plot_combined_stats(dev_stats, segment_names, dig_in_names, save_dir, 
						dist_name, neuron_indices):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval. Outputs are saved .npy files with name indicating
	segment and taste containing matrices of shape [num_dev, num_deliv, num_neur, num_cp]
	with the distances stored.
	
	neuron_indices should be binary and shaped num_neur x num_cp
	"""
	
	#Grab parameters
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	
	#Define storage
	segment_data = [] #segments x tastes x cp
	segment_pop_data = [] #segments x tastes x cp from fr population matrix
	segment_data_avg = [] #segments x tastes x cp avged across neurons in the deviation
	segment_pop_vec_data = [] #segments x tastes x cp from fr population vector
	for s_i in range(num_segments):  #Loop through each segment
		seg_stats = dev_stats[s_i]
		print("Beginning combined plot calcs for segment " + str(s_i))
		taste_data = [] #tastes x cp
		taste_pop_data = [] #tastes x cp
		taste_data_avg = []
		taste_pop_vec_data = []
		for t_i in range(num_tastes):  #Loop through each taste
			print("\tTaste #" + str(t_i + 1))
			taste_stats = seg_stats[t_i]
			#Import distance numpy array
			neuron_data_storage = taste_stats['neuron_data_storage']
			pop_data_storage = taste_stats['pop_data_storage']
			pop_vec_data_storage = taste_stats['pop_vec_data_storage']
			num_dev, num_deliv, total_num_neur, num_cp = np.shape(neuron_data_storage)
			cp_data = []
			cp_data_pop = []
			cp_data_avg = []
			cp_data_pop_vec = []
			for c_p in range(num_cp):
				all_dist_cp = (neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p]).flatten()
				all_pop_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
				cp_data.append(all_dist_cp)
				cp_data_pop.append(all_pop_dist_cp)
				avg_dist_cp = np.nanmean(neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p],2).flatten()
				cp_data_avg.append(avg_dist_cp)
				all_pop_vec_cp = (pop_vec_data_storage[:,:,c_p]).flatten()
				cp_data_pop_vec.append(all_pop_vec_cp)
			taste_data.append(cp_data)
			taste_pop_data.append(cp_data_pop)
			taste_data_avg.append(cp_data_avg)
			taste_pop_vec_data.append(cp_data_pop_vec)
		segment_data.append(taste_data)
		segment_pop_data.append(taste_pop_data)
		segment_data_avg.append(taste_data_avg)
		segment_pop_vec_data.append(taste_pop_vec_data)
		#Plot taste data against each other
		for c_p in range(num_cp):
			#_____Individual Neurons_____
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
					#true_sort = np.sort(data)
					#true_unique = np.unique(true_sort)
					#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
					#plt.plot(true_unique,cmf_true,label='Taste ' + dig_in_names[t_i])
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
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
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
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_log_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
			#_____Population Data_____
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_pop_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_pop_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_pop'
			f0.savefig(filename + '.png')
			f0.savefig(filename + '.svg')
			plt.close(f0)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				#Plot data across all neurons
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_pop_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_pop_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_zoom_pop'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Plot data across all neurons with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_pop_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_pop_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_log_zoom_pop'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
			#_____Population Vector Data
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_pop_vec_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_pop_vec_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_pop_vec'
			f0.savefig(filename + '.png')
			f0.savefig(filename + '.svg')
			plt.close(f0)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				#Plot data across all neurons
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_pop_vec_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_pop_vec_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_zoom_pop_vec'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Plot data across all neurons with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_pop_vec_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_pop_vec_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_log_zoom_pop_vec'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
		#Now plot individual neuron data population averages
		for c_p in range(num_cp):
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_data_avg[t_i][c_p]
					plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_data_avg[t_i][c_p]
					plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Averaged Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_avg_neur'
			f0.savefig(filename + '.png')
			f0.savefig(filename + '.svg')
			plt.close(f0)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				#Plot data across all neurons
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Averaged Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p)  + '_avg_neur_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Plot data across all neurons with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Averaged Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p)  + '_avg_neur_log_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
			
	for t_i in range(num_tastes): #Loop through each taste
		#Plot segment data against each other by epoch
		#_____Individual Neuron Averages_____
		for c_p in range(num_cp):
			f2 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for s_i in range(num_segments):
				try:
					data = segment_data_avg[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for s_i in range(num_segments):
				try:
					data = segment_data_avg[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_neur_avg'
			f2.savefig(filename + '.png')
			f2.savefig(filename + '.svg')
			plt.close(f2)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_neur_avg_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
				#Log
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_neur_avg_log_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
				
		#_____Population Data_____
		for c_p in range(num_cp):
			f2 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for s_i in range(num_segments):
				try:
					data = segment_pop_data[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for s_i in range(num_segments):
				try:
					data = segment_pop_data[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop'
			f2.savefig(filename + '.png')
			f2.savefig(filename + '.svg')
			plt.close(f2)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_pop_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_pop_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
				#Log
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_pop_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_pop_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_log_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
		
		#_____Population Vec Data_____
		for c_p in range(num_cp):
			f2 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for s_i in range(num_segments):
				try:
					data = segment_pop_vec_data[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for s_i in range(num_segments):
				try:
					data = segment_pop_vec_data[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_vec'
			f2.savefig(filename + '.png')
			f2.savefig(filename + '.svg')
			plt.close(f2)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_pop_vec_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_pop_vec_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_vec_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
				#Log
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_pop_vec_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				for s_i in range(num_segments):
					try:
						data = segment_pop_vec_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_vec_log_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)
		
	return segment_data, segment_data_avg, segment_pop_data, segment_pop_vec_data
		

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
			

def stat_significance_ttest_less(segment_data, segment_names, dig_in_names, save_dir, dist_name):
	
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
	pair_significance_statements = []
	
	print("Calculating Significance for All Combinations")
	for p_i in tqdm.tqdm(range(len(pair_combinations))):
		try:
			ind_1 = pair_combinations[p_i][0]
			ind_2 = pair_combinations[p_i][1]
			data_1 = segment_data[ind_1[0]][ind_1[1]][ind_1[2]]
			data_2 = segment_data[ind_2[0]][ind_2[1]][ind_2[2]]
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
			
def stat_significance_ttest_more(segment_data, segment_names, dig_in_names, save_dir, dist_name):
	
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
	pair_significance_statements = []
	
	print("Calculating Significance for All Combinations")
	for p_i in tqdm.tqdm(range(len(pair_combinations))):
		try:
			ind_1 = pair_combinations[p_i][0]
			ind_2 = pair_combinations[p_i][1]
			data_1 = segment_data[ind_1[0]][ind_1[1]][ind_1[2]]
			data_2 = segment_data[ind_2[0]][ind_2[1]][ind_2[2]]
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

			
def mean_compare(segment_data, segment_names, dig_in_names, save_dir, dist_name):
	
	#Grab parameters
	#segment_data shape = segments x tastes x cp
	num_segments = len(segment_data)
	num_tastes = len(segment_data[0])
	num_cp = len(segment_data[0][0])
	
	#Calculate mean comparison of pairs
	
	#All pair combinations
	a = [list(np.arange(num_segments)),list(np.arange(num_tastes)),list(np.arange(num_cp))]
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
			data_1 = segment_data[ind_1[0]][ind_1[1]][ind_1[2]]
			data_2 = segment_data[ind_2[0]][ind_2[1]][ind_2[2]]
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

			
def top_dev_corr_bins(dev_stats,segment_names,dig_in_names,save_dir,neuron_indices):
	"""Calculate which deviation index is most correlated with which taste 
	delivery and which epoch and store to a text file.
	
	neuron_indices should be binary and shaped num_neur x num_cp
	"""
	
	#Grab parameters
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	
	#Define storage
	for s_i in range(num_segments):  #Loop through each segment
		seg_stats = dev_stats[s_i]
		print("Beginning calcs for segment " + str(s_i))
		for t_i in range(num_tastes):  #Loop through each taste
			save_file = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_top_corr_combos_neur_avg.txt'
			pop_save_file = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_top_corr_combos_pop.txt'
			pop_vec_save_file = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_top_corr_combos_pop_vec.txt'
			corr_data = []
			corr_pop_data = []
			corr_pop_vec_data = []
			print("\tTaste #" + str(t_i + 1))
			taste_stats = seg_stats[t_i]
			#Import distance numpy array
			neuron_data_storage = taste_stats['neuron_data_storage']
			pop_data_storage = taste_stats['pop_data_storage']
			pop_vec_data_storage = taste_stats['pop_vec_data_storage']
			num_dev, num_deliv, total_num_neur, num_cp = np.shape(neuron_data_storage)
			#Calculate, for each deviation bin, which taste delivery and cp it correlates with most
			all_dev_data = np.zeros((num_dev,num_deliv,num_cp))
			for c_p in range(num_cp):
				all_dev_data[:,:,c_p] = np.nanmean(neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p],2) #num_dev x num_deliv x num_cp 
			top_99_percentile = np.percentile(all_dev_data[~np.isnan(all_dev_data)].flatten(),99)
			top_99_percentile_pop = np.percentile(pop_data_storage[~np.isnan(pop_data_storage)][:,:,c_p].flatten(),99)
			top_99_percentile_pop_vec = np.percentile(pop_vec_data_storage[~np.isnan(pop_vec_data_storage)][:,:,c_p].flatten(),99)
			for dev_i in range(num_dev):
				dev_data = all_dev_data[dev_i,:,:]  #num_deliv x num_cp
				[deliv_i,cp_i] = np.where(dev_data >= top_99_percentile)
				pop_dev_data = pop_data_storage[dev_i,:,:]
				[pop_deliv_i,pop_cp_i] = np.where(pop_dev_data >= top_99_percentile_pop)
				pop_vec_data = pop_vec_data_storage[dev_i,:,:]
				[pop_vec_deliv_i,pop_vec_cp_i] = np.where(pop_vec_data >= top_99_percentile_pop_vec)
				if len(deliv_i) > 0:
					for d_i in range(len(deliv_i)):
						dev_cp_corr_val = dev_data[deliv_i[d_i],cp_i[d_i]]
						statement = 'dev-' + str(dev_i) + '; epoch-' + str(cp_i[d_i]) + '; deliv-' + str(deliv_i[d_i]) + '; corr-' + str(dev_cp_corr_val)
						corr_data.append(statement)
				if len(pop_deliv_i) > 0:
					for d_i in range(len(pop_deliv_i)):
						dev_pop_cp_corr_val = pop_dev_data[pop_deliv_i[d_i],pop_cp_i[d_i]]
						statement = 'dev-' + str(dev_i) + '; epoch-' + str(cp_i[d_i]) + '; deliv-' + str(deliv_i[d_i]) + '; corr-' + str(dev_pop_cp_corr_val)
						corr_pop_data.append(statement)
				if len(pop_vec_deliv_i) > 0:
					for d_i in range(len(pop_deliv_i)):
						dev_pop_cp_corr_val = pop_dev_data[pop_vec_deliv_i[d_i],pop_vec_cp_i[d_i]]
						statement = 'dev-' + str(dev_i) + '; epoch-' + str(cp_i[d_i]) + '; deliv-' + str(deliv_i[d_i]) + '; corr-' + str(dev_pop_cp_corr_val)
						corr_pop_vec_data.append(statement)
			#Save to file neuron average statements
			with open(save_file, 'w') as f:
				for line in corr_data:
					f.write(line)
					f.write('\n')
			#Save to file population statements
			with open(pop_save_file, 'w') as f:
				for line in corr_pop_data:
					f.write(line)
					f.write('\n')
			#Save to file population vector statements
			with open(pop_vec_save_file, 'w') as f:
				for line in corr_pop_vec_data:
					f.write(line)
					f.write('\n')
			




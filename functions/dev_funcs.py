#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:24:49 2023

@author: Hannah Germaine

A collection of functions used by find_deviations.py to pull, reformat, analyze,
etc... the deviations in true and null datasets.
"""

import os, json, gzip
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.signal import find_peaks

def run_dev_pull_parallelized(inputs):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	"""
	spikes = inputs[0] #list of length num_neur with each neur having a list of spike times
	local_size = inputs[1] #how many bins are the local activity?
	min_dev_size = inputs[2] #what is the minimum number of bins for a deviation?
	segment_times = inputs[3] #start and end times of segment
	save_dir = inputs[4] #directory to save data
	#calculate the deviations
	num_neur = len(spikes)
	num_dt = segment_times[1] - segment_times[0] + 1
	spikes_bin = np.zeros((num_neur,num_dt))
	for n_i in range(num_neur):
		n_spikes = np.array(spikes[n_i]).astype('int') - int(segment_times[0])
		spikes_bin[n_i,n_spikes] = 1
	spike_sum = np.sum(spikes_bin,0)
	half_min_dev_size = int(np.ceil(min_dev_size/2))
	half_local_size = int(np.ceil(local_size/2))
	fr_calc = np.array([np.sum(spike_sum[i_s - half_min_dev_size:i_s + half_min_dev_size])/min_dev_size for i_s in np.arange(half_local_size,num_dt-half_local_size)]) #in spikes per min_dev_size
	local_fr_calc = np.array([np.sum(spike_sum[l_s - half_local_size:l_s + half_local_size])/local_size for l_s in np.arange(half_local_size,num_dt-half_local_size)]) #in spikes per min_dev_size
	peak_fr_ind = find_peaks(fr_calc - local_fr_calc,height=0)[0]
	#Find where the prominence is above the 90th percentile of prominence values
	peak_fr_prominence = fr_calc[peak_fr_ind] - local_fr_calc[peak_fr_ind]
	top_90_prominence = np.percentile(peak_fr_prominence,90)
	top_90_prominence_ind = peak_fr_ind[np.where(peak_fr_prominence >= top_90_prominence)[0]]
	true_ind = top_90_prominence_ind + half_local_size
	deviations = np.zeros(num_dt)
	for t_i in true_ind:
		deviations[t_i - half_min_dev_size:t_i + half_min_dev_size] = 1
	#store each in a json
	json_str = json.dumps(list(deviations))
	json_bytes = json_str.encode()
	filepath = save_dir + 'deviations.json'
	with gzip.GzipFile(filepath, mode="w") as f:
		f.write(json_bytes)
		
		
def create_dev_rasters(num_segments, segment_spike_times, 
					   segment_times, segment_deviations):
	"""This function takes the spike times and creates binary matrices of 
	rasters of spiking"""
	segment_dev_rasters = []
	segment_dev_times = []
	for s_i in range(num_segments):
		seg_spikes = segment_spike_times[s_i]
		num_neur = len(seg_spikes)
		num_dt = segment_times[s_i+1] - segment_times[s_i] + 1
		spikes_bin = np.zeros((num_neur,num_dt))
		for n_i in range(num_neur):
			neur_spikes = np.array(seg_spikes[n_i]).astype('int') - segment_times[s_i]
			spikes_bin[n_i,neur_spikes] = 1
		seg_rast = []
		change_inds = np.diff(segment_deviations[s_i])
		start_dev_bouts = np.where(change_inds == 1)[0]
		end_dev_bouts = np.where(change_inds == -1)[0]
		bout_times = np.concatenate((np.expand_dims(start_dev_bouts,0),np.expand_dims(end_dev_bouts,0)))
		for b_i in range(len(start_dev_bouts)):
			seg_rast.append(spikes_bin[:,start_dev_bouts[b_i]:end_dev_bouts[b_i]])
		segment_dev_rasters.append(seg_rast)
		segment_dev_times.append(bout_times)
		
	return segment_dev_rasters, segment_dev_times


def calculate_dev_stats(segment_dev_rasters,segment_dev_times,segment_names,save_dir):
	"""This function calculates deviation statistics - and plots them - including:
		- deviation lengths
		- inter-deviation-intervals (IDIs)
		- number of spikes / deviation
		- number of neurons spiking / deviation
	"""
	
	num_segments = len(segment_dev_rasters)
	length_dict = dict()
	IDI_dict = dict()
	num_spike_dict = dict()
	num_neur_dict = dict()
	
	for s_i in range(num_segments):
		seg_name = segment_names[s_i]
		#Gather data
		seg_rasters = segment_dev_rasters[s_i]
		seg_bout_times = segment_dev_times[s_i]
		#Calculate segment lengths
		seg_lengths = seg_bout_times[1,:] - seg_bout_times[0,:]
		length_dict[s_i] = seg_lengths
		data_name = seg_name + ' deviation lengths'
		plot_dev_stats(seg_lengths,data_name,save_dir,x_label='deviation index',y_label='length (ms)')
		#Calculate IDIs
		seg_IDIs = seg_bout_times[1,1:] - seg_bout_times[0,:-1]
		IDI_dict[s_i] = seg_IDIs
		data_name = seg_name + ' inter-deviation-intervals'
		plot_dev_stats(seg_IDIs,data_name,save_dir,x_label='distance index',y_label='length (ms)')
		#Calculate number of spikes
		seg_spike_num = [np.sum(np.sum(seg_rasters[r_i])) for r_i in range(len(seg_rasters))]
		num_spike_dict[s_i] = seg_spike_num
		data_name = seg_name + ' total spike count'
		plot_dev_stats(seg_spike_num,data_name,save_dir,x_label='deviation index',y_label='# spikes')
		#Calculate number of neurons spiking
		seg_neur_num = [np.sum(np.sum(seg_rasters[r_i],1)>0) for r_i in range(len(seg_rasters))]
		num_neur_dict[s_i] = seg_neur_num
		data_name = seg_name + ' total neuron count'
		plot_dev_stats(seg_neur_num,data_name,save_dir,x_label='deviation index',y_label='# neurons')
		
def plot_dev_stats(data,data_name,save_dir,x_label=[],y_label=[]):
	"""General function to plot given statistics"""
	plt.figure(figsize = (5,5))
	#Plot the trend
	plt.subplot(2,1,1)
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
	#Plot the histogram
	plt.subplot(2,1,2)
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
	
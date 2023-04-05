#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:56:14 2023

@author: Hannah Germaine

This is a collection of functions for plotting deviation bins
(To eventually replace parts of "plot_funcs.py")
"""

import tqdm, os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_deviations(dev_save_dir, num_neur, segment_names, segment_times, dev_thresh,
					segment_devs, segment_bouts, segment_bout_lengths, 
					segment_ibis, segment_spike_times, num_null_sets,
					null_segment_dev_counts, null_segment_dev_ibis,
					null_segment_dev_bout_len, fig_buffer_size):
	"""Master function to generate all deviation plots desired"""
	num_segments = len(segment_names)
	
	#First plot when deviations occur in each segment
	#segment_dev_frac_ind are those that are above the dev_thresh cutoff
	deviation_times(num_segments,num_neur,segment_names,dev_save_dir,segment_devs,dev_thresh)
	
	#Next plot bout lengths and inter-bout intervals and trends across segments
	segment_bout_stats_plots(num_segments,segment_names,segment_bout_lengths,segment_ibis,dev_save_dir)
	
	#Zoom in and plot rasters for the deviations
	dev_bin_plots(dev_save_dir,segment_names,segment_times,
						      segment_spike_times,num_neur,fig_buffer_size,
							  segment_bouts)
	
	#Plot true deviations against null distribution
	null_v_true_dev_plots(dev_save_dir,segment_names,segment_bouts,segment_bout_lengths,segment_ibis,
					   num_null_sets,null_segment_dev_counts,null_segment_dev_ibis,null_segment_dev_bout_len)
	
def deviation_times(num_segments,num_neur,segment_names,dev_save_dir,segment_devs,dev_thresh):
	"""
	This function plots the times in each segment that a deviation is tracked.
	INPUTS:
		- num_segments: number of segments in experiment
		- num_neur: number of neurons in data
		- sampling_rate: sampling rate of recording
		- segment_names: names of different segments
		- dev_save_dir: where to save plots
		- segment_devs: indices and deviation fractions for each segment
		- dev_thresh: threshold for a good deviation to plot
	OUTPUTS:
		- png and svg plots for each segment with 1 when a deviation occurs and 0 otherwise.
	"""
	#Now plot deviations
	print("Beginning Deviation Plots.")
	for i in tqdm.tqdm(range(num_segments)):
		print("\n\tPlotting deviations for segment " + segment_names[i])
		seg_dev_save_dir = dev_save_dir + ('_').join(segment_names[i].split(' ')) + '/'
		if os.path.isdir(seg_dev_save_dir) == False:
			os.mkdir(seg_dev_save_dir)
		dev_times = segment_devs[i][0]*(1/1000) #Converted to seconds
		fig_i = plt.figure(figsize=((max(dev_times)-min(dev_times))*(1/60),10))
		plt.plot(dev_times,segment_devs[i][1])
		plt.xlabel('Time (s)')
		plt.ylabel('Deviation Fraction')
		im_name = (' ').join(segment_names[i].split('_'))
		plt.title(im_name + ' deviation fractions')
		save_name = ('_').join(segment_names[i].split(' ')) + '_devs'
		fig_i.savefig(seg_dev_save_dir + save_name + '.png')
		fig_i.savefig(seg_dev_save_dir + save_name + '.svg')
		plt.close(fig_i)

def segment_bout_stats_plots(num_segments,segment_names,segment_bout_lengths,segment_ibis,dev_save_dir):
	"""This function plots the bout lengths and IBIs
	INPUTS:
		- segment_bout_lengths: 
		- segment_ibis:
	OUTPUTS:
		Histogram plots of each segment's bout lengths and inter-bout-intervals
	"""
	mean_segment_len = []
	std_segment_len = []
	mean_segment_ibis = []
	std_segment_ibis = []
	
	for i in tqdm.tqdm(range(num_segments)):
		seg_dev_save_dir = dev_save_dir + ('_').join(segment_names[i].split(' ')) + '/'
		if os.path.isdir(seg_dev_save_dir) == False:
			os.mkdir(seg_dev_save_dir)
		segment_len = segment_bout_lengths[i]
		mean_segment_len.extend([np.mean(segment_len)])
		std_segment_len.extend([np.std(segment_len)])
		segment_ibi = segment_ibis[i]
		mean_segment_ibis.extend([np.mean(segment_ibi)])
		std_segment_ibis.extend([np.std(segment_ibi)])
		#Plot individual histograms
		fig_i = plt.figure(figsize = (10,10))
		plt.subplot(1,2,1)
		plt.hist(segment_len)
		plt.title('Bout lengths (s) histogram')
		plt.xlabel('Bout length (s)')
		plt.ylabel('Counts')
		plt.subplot(1,2,2)
		plt.hist(segment_ibi)
		plt.title('Inter-bout-intervals (s) histogram')
		plt.xlabel('IBI (s)')
		plt.ylabel('Counts')
		save_name = ('_').join(segment_names[i].split(' ')) + '_dev_hist.png'
		fig_i.savefig(seg_dev_save_dir + save_name)
		plt.close(fig_i)
	mean_segment_len = np.array(mean_segment_len)
	std_segment_len = np.array(std_segment_len)
	mean_segment_ibis = np.array(mean_segment_ibis)
	std_segment_ibis = np.array(std_segment_ibis)
	#Plot mean and standard deviations across segments
	fig_i = plt.figure(figsize = (10,10))
	#cm_subsection = np.linspace(0,1,num_neur)
	#cmap = [cm.gist_rainbow(x) for x in cm_subsection]
	plt.subplot(1,2,1)
	plt.plot(segment_names,mean_segment_len,color='k',label='Mean Across Neurons')
	plt.plot(segment_names,mean_segment_len + std_segment_len,alpha=0.25,color='k',linestyle='-',label='Mean Across Neurons')
	plt.plot(segment_names,mean_segment_len - std_segment_len,alpha=0.25,color='k',linestyle='-',label='Mean Across Neurons')
	plt.legend()
	plt.title('Mean Bout Lengths by Segment')
	plt.xlabel('Experimental Segment')
	plt.ylabel('Mean Bout Length (s)')
	plt.subplot(1,2,2)
	plt.plot(segment_names,mean_segment_ibis,color='k',label='Mean Across Neurons')
	plt.plot(segment_names,mean_segment_ibis + std_segment_ibis,alpha=0.25,color='k',linestyle='-',label='Mean Across Neurons')
	plt.plot(segment_names,mean_segment_ibis - std_segment_ibis,alpha=0.25,color='k',linestyle='-',label='Mean Across Neurons')
	plt.legend()
	plt.title('Mean Inter-Bout-Intervals by Segment')
	plt.xlabel('Experimental Segment')
	plt.ylabel('Mean Inter-Bout-Interval (s)')
	fig_i.tight_layout()
	save_name = 'cross-segment_bout_stats.png'
	fig_i.savefig(dev_save_dir + save_name)
	plt.close(fig_i)

def dev_bin_plots(fig_save_dir,segment_names,segment_times,
					      segment_spike_times,num_neur,fig_buffer_size,
						  segment_bouts):
	"""This function creates visualizations of bins with high deviation from local mean
	INPUTS:
		- fig_save_dir: directory to save visualizations
		- segment_names: names of different experiment segments
		- segment_times: time indices of different segment starts/ends
		- segment_spike_times: when spikes occur in each segment
		- num_neur: the number of neurons
		- fig_buffer_size: how much (in seconds) to plot before and after a deviation event
		- segment_bouts: bouts of time in which segments occur
	OUTPUTS:
		- 
	"""
	print("\nBeginning individual deviation segment plots.")
	#Create save directory
	dev_save_dir = fig_save_dir
	if os.path.isdir(dev_save_dir) == False:
		os.mkdir(dev_save_dir)
	#Convert the bin size from time to samples
	num_segments = len(segment_names)
	local_bin_dt = int(np.ceil(fig_buffer_size*1000)) #in ms timescale
	half_local_bin_dt = int(np.ceil(local_bin_dt/2))
	#Run through deviation times by segment and plot rasters
	for s_i in tqdm.tqdm(range(num_segments)):
		print("\nGrabbing spike rasters for segment " + segment_names[s_i])
		seg_dev_save_dir = dev_save_dir + ('_').join(segment_names[s_i].split(' ')) + '/'
		if os.path.isdir(seg_dev_save_dir) == False:
			os.mkdir(seg_dev_save_dir)
		seg_rast_save_dir = seg_dev_save_dir + 'dev_rasters/'
		if os.path.isdir(seg_rast_save_dir) == False:
			os.mkdir(seg_rast_save_dir)
		segment_dev_start_times = segment_bouts[s_i][:,0]
		segment_dev_end_times = segment_bouts[s_i][:,1]
		segment_spikes = [np.array(segment_spike_times[s_i][n_i]) for n_i in range(num_neur)]
		min_seg_time = segment_times[s_i]
		max_seg_time = segment_times[s_i+1]
		for d_i in range(len(segment_dev_start_times)):
			min_time = max(segment_dev_start_times[d_i] - half_local_bin_dt,min_seg_time)
			max_time = min(segment_dev_end_times[d_i] + half_local_bin_dt,max_seg_time)
			s_t = []
			for n_i in range(num_neur):
				try:
					s_t.append(list(segment_spikes[n_i][np.where((segment_spikes[n_i] >= min_time)*(segment_spikes[n_i] <= max_time))[0]]))
				except:
					print(segment_spikes[n_i])
			s_t_time = [list(np.array(s_t[i])*(1/1000)) for i in range(len(s_t))] #In seconds
			#Plot segment deviation raster
			plt.figure(figsize=(10,num_neur))
			plt.xlabel('Time (s)')
			plt.ylabel('Neuron Index')
			plt.axvline(segment_dev_start_times[d_i]*(1/1000),color='r',alpha=0.4)
			plt.axvline(segment_dev_end_times[d_i]*(1/1000),color='r',alpha=0.4)
			plt.eventplot(s_t_time,color='b')
			plt.title('Deviation ' + str(d_i))
			plt.tight_layout()
			im_name = 'dev_' + str(d_i)
			plt.savefig(seg_rast_save_dir + im_name + '.png')
			plt.savefig(seg_rast_save_dir + im_name + '.svg')
			plt.close()

def null_v_true_dev_plots(dev_save_dir,segment_names,segment_bouts,segment_bout_lengths,segment_ibis,num_null_sets,null_segment_dev_counts,null_segment_dev_ibis,null_segment_dev_bout_len):
	"""This function plots histograms of null distribution values and 95th percentile cutoffs against true deviation values
	INPUTS:
		- fig_save_dir
		- segment_names
		- segment_bouts
		- segment_bout_lengths
		- segment_ibis
		- num_null_sets
		- null_segment_dev_counts
		- null_segment_dev_ibis
		- null_segment_dev_bout_len
	OUTPUTS:
		- Plots
	"""
	num_segments = len(segment_names)
	#Create save directory
	if os.path.isdir(dev_save_dir) == False:
		os.mkdir(dev_save_dir)
	#Go through each segment
	for s_i in range(num_segments):
		print("\tPlotting null distributions for segment " + segment_names[s_i])
		#Create Save Directory
		seg_dev_save_dir = dev_save_dir + ('_').join(segment_names[s_i].split(' ')) + '/'
		if os.path.isdir(seg_dev_save_dir) == False:
			os.mkdir(seg_dev_save_dir)
		seg_null_hist_save_dir = seg_dev_save_dir + 'dev_histograms/'
		if os.path.isdir(seg_null_hist_save_dir) == False:
			os.mkdir(seg_null_hist_save_dir)
		#Histogram of bout length
		fig_i = plt.figure(figsize=(5,5))
		seg_true_dev_bout_lens = segment_bout_lengths[s_i]
		seg_null_dev_bout_lens = null_segment_dev_bout_len[s_i]
		seg_null_dev_bout_lens_flat = []
		for s_n_i in range(len(seg_null_dev_bout_lens)):
			seg_null_dev_bout_lens_flat.extend(seg_null_dev_bout_lens[s_n_i])
		im_name = (' ').join(segment_names[s_i].split('_'))
		plt.subplot(1,2,1)
		plt.hist(seg_null_dev_bout_lens_flat,bins=20,alpha=0.5,color='blue',label='Null Data')
		plt.axvline(np.mean(seg_true_dev_bout_lens),color='orange',label='Mean of True Data')
		plt.legend()
		plt.title(im_name + ' Null Distribution')
		plt.xlabel('Deviation Bout Length (s)')
		plt.ylabel('Number of Instances')
		plt.subplot(1,2,2)
		plt.hist(seg_true_dev_bout_lens,bins=20,alpha=0.5,color='orange',label='True Data')
		plt.axvline(np.mean(seg_true_dev_bout_lens),color='orange',label='Mean of True Data')
		plt.legend()
		plt.title(im_name + ' True Distribution')
		plt.xlabel('Deviation Bout Length (s)')
		plt.ylabel('Number of Instances')
		plt.title(im_name + ' deviation lengths x null distribution')
		plt.tight_layout()
		save_name = ('_').join(segment_names[s_i].split(' ')) + '_null_v_true_lens'
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.png')
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.svg')
		plt.close(fig_i)
		#Histogram of IBIs
		fig_i = plt.figure(figsize=(5,5))
		seg_true_dev_ibis = segment_ibis[s_i]
		seg_null_dev_bout_ibis = null_segment_dev_ibis[s_i]
		seg_null_dev_bout_ibis_flat = []
		for s_n_i in range(len(seg_null_dev_bout_ibis)):
			seg_null_dev_bout_lens_flat.extend(seg_null_dev_bout_ibis[s_n_i])
		im_name = (' ').join(segment_names[s_i].split('_'))
		plt.subplot(1,2,1)
		plt.hist(seg_null_dev_bout_ibis_flat,bins=20,alpha=0.5,color='blue',label='Null Data')
		plt.axvline(np.mean(seg_true_dev_ibis),color='orange',label='Mean of True Data')
		plt.legend()
		plt.title(im_name + ' Null Distribution')
		plt.xlabel('Deviation Inter-Bout-Interals (IBIs) (s)')
		plt.ylabel('Number of Instances')
		plt.subplot(1,2,2)
		plt.hist(seg_true_dev_ibis,bins=20,alpha=0.5,color='orange',label='True Data')
		plt.axvline(np.mean(seg_true_dev_ibis),color='orange',label='Mean of True Data')
		plt.legend()
		plt.title(im_name + ' True Distribution')
		plt.xlabel('Deviation Inter-Bout-Interals (IBIs) (s)')
		plt.ylabel('Number of Instances')
		plt.suptitle(im_name + ' Deviation IBIs x Null Distribution')
		plt.tight_layout()
		save_name = ('_').join(segment_names[s_i].split(' ')) + '_null_v_true_ibis'
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.png')
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.svg')
		plt.close(fig_i)
		#Histogram of deviation counts
		fig_i = plt.figure(figsize=(5,5))
		seg_true_dev_count = len(segment_bout_lengths[s_i])
		seg_null_dev_counts = null_segment_dev_counts[s_i]
		im_name = (' ').join(segment_names[s_i].split('_'))
		plt.hist(seg_null_dev_counts,bins=20,alpha=0.5,color='blue',label='Null Data')
		plt.axvline(seg_true_dev_count,color='orange',label='True Count')
		plt.legend()
		plt.xlabel('Deviation Counts')
		plt.ylabel('Number of Instances')
		plt.title(im_name + ' deviation counts x null distribution')
		plt.tight_layout()
		save_name = ('_').join(segment_names[s_i].split(' ')) + '_null_v_true_counts'
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.png')
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.svg')
		plt.close(fig_i)
		
	#Now compare segment deviations against each other
	print("\tPlotting null distributions for all segments combined")
	cm_subsection = np.linspace(0,1,num_segments)
	cmap = [cm.gist_rainbow(x) for x in cm_subsection] #Color maps for each segment
	
	#Bout lengths
	mean_vals = []
	fig_lens = plt.figure(figsize=(5,5))
	for s_i in range(num_segments):
		segment_name = (' ').join(segment_names[s_i].split('_'))
		seg_true_dev_bout_lens = segment_bout_lengths[s_i]
		mean_true = np.mean(seg_true_dev_bout_lens)
		mean_vals.extend([mean_true])
		seg_null_dev_bout_lens = null_segment_dev_bout_len[s_i]
		seg_null_dev_bout_lens_flat = []
		for s_n_i in range(len(seg_null_dev_bout_lens)):
			seg_null_dev_bout_lens_flat.extend(seg_null_dev_bout_lens[s_n_i])
		plt.hist(seg_null_dev_bout_lens_flat,bins=20,color=cmap[s_i],alpha=0.5,label=segment_name + ' null')
		plt.axvline(mean_true,color=cmap[s_i],label=segment_name + ' mean')
	plt.xlim((0,max(mean_vals) + 0.25))
	plt.legend()
	plt.xlabel('Deviation Length (s)')
	plt.ylabel('Number of Instances')
	plt.title('cross-segment deviation lengths x null distribution')
	plt.tight_layout()
	save_name ='all_seg_null_v_true_lengths'
	fig_lens.savefig(dev_save_dir + save_name + '.png')
	fig_lens.savefig(dev_save_dir + save_name + '.svg')
	plt.close(fig_lens)
	
	#Bout lengths
	mean_vals = []
	fig_lens = plt.figure(figsize=(5,5))
	for s_i in range(num_segments):
		segment_name = (' ').join(segment_names[s_i].split('_'))
		seg_true_dev_bout_lens = segment_bout_lengths[s_i]
		mean_true = np.mean(seg_true_dev_bout_lens)
		mean_vals.extend([mean_true])
		plt.hist(seg_true_dev_bout_lens,bins=20,color=cmap[s_i],alpha=0.5,label=segment_name + ' distribution')
		plt.axvline(mean_true,color=cmap[s_i],label=segment_name + ' mean')
	plt.xlim((0,max(mean_vals) + 0.25))
	plt.legend()
	plt.xlabel('Deviation Length (s)')
	plt.ylabel('Number of Instances')
	plt.title('cross-segment deviation lengths')
	plt.tight_layout()
	save_name ='all_seg_true_lengths'
	fig_lens.savefig(dev_save_dir + save_name + '.png')
	fig_lens.savefig(dev_save_dir + save_name + '.svg')
	plt.close(fig_lens)
	
	#Bout ibis
	mean_vals = []
	fig_ibis = plt.figure(figsize=(5,5))
	for s_i in range(num_segments):
		segment_name = (' ').join(segment_names[s_i].split('_'))
		seg_true_dev_ibis = segment_ibis[s_i]
		mean_true = np.mean(seg_true_dev_ibis)
		mean_vals.extend([mean_true])
		seg_null_dev_bout_ibis = null_segment_dev_ibis[s_i]
		seg_null_dev_bout_ibis_flat = []
		for s_n_i in range(len(seg_null_dev_bout_ibis)):
			seg_null_dev_bout_lens_flat.extend(seg_null_dev_bout_ibis[s_n_i])
		plt.hist(seg_null_dev_bout_ibis_flat,bins=20,color=cmap[s_i],alpha=0.5,label=segment_name + ' null')
		plt.axvline(mean_true,color=cmap[s_i],label=segment_name + ' mean')
	#plt.xlim((0,max(mean_vals) + 0.25))
	plt.legend()
	plt.xlabel('Deviation IBI (s)')
	plt.ylabel('Number of Instances')
	plt.title('cross-segment deviation ibis x null distribution')
	plt.tight_layout()
	save_name ='all_seg_null_v_true_ibis'
	fig_ibis.savefig(dev_save_dir + save_name + '.png')
	fig_ibis.savefig(dev_save_dir + save_name + '.svg')
	plt.close(fig_ibis)
	
	#Bout ibis
	mean_vals = []
	fig_ibis = plt.figure(figsize=(5,5))
	for s_i in range(num_segments):
		segment_name = (' ').join(segment_names[s_i].split('_'))
		seg_true_dev_ibis = segment_ibis[s_i]
		mean_true = np.mean(seg_true_dev_ibis)
		mean_vals.extend([mean_true])
		plt.hist(seg_true_dev_ibis,bins=20,color=cmap[s_i],alpha=0.5,label=segment_name + ' distribution')
		plt.axvline(mean_true,color=cmap[s_i],label=segment_name + ' mean')
	#plt.xlim((0,max(mean_vals) + 0.25))
	plt.legend()
	plt.xlabel('Deviation IBI (s)')
	plt.ylabel('Number of Instances')
	plt.title('cross-segment deviation ibis')
	plt.tight_layout()
	save_name ='all_seg_true_ibis'
	fig_ibis.savefig(dev_save_dir + save_name + '.png')
	fig_ibis.savefig(dev_save_dir + save_name + '.svg')
	plt.close(fig_ibis)
	
	#Bout counts
	true_counts = []
	fig_counts = plt.figure(figsize=(5,5))
	for s_i in range(num_segments):
		segment_name = (' ').join(segment_names[s_i].split('_'))
		seg_true_dev_count = len(segment_bout_lengths[s_i])
		seg_null_dev_counts = null_segment_dev_counts[s_i]
		true_counts.extend([seg_true_dev_count])
		plt.hist(seg_null_dev_counts,bins=20,color=cmap[s_i],alpha=0.5,label=segment_name + ' null')
		plt.axvline(seg_true_dev_count,color=cmap[s_i],label=segment_name + ' true')
	plt.legend()
	plt.xlabel('Deviation Count')
	plt.ylabel('Number of Instances')
	plt.title('cross-segment deviation counts x null distribution')
	plt.tight_layout()
	save_name ='all_seg_null_v_true_counts'
	fig_counts.savefig(dev_save_dir + save_name + '.png')
	fig_counts.savefig(dev_save_dir + save_name + '.svg')
	plt.close(fig_counts)

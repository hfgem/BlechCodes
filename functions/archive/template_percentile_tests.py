#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:14:20 2022

@author: hannahgermaine

Template-matching test: a test of different template distance percentiles for
cutoff of noise, and the resulting clustering.
WARNING: This function assumes the spike sorting has already been performed on
the dataset, and it will import spike times accordingly. It will FAIL if the
data was not previously sorted.
"""

import functions.spike_sort as ss
import numpy as np
import tables, tqdm, csv

percentiles = np.arange(0,90,10)
percentiles[0] = 1

def run_template_test(data_dir):
	"""This function performs clustering spike sorting to separate out spikes
	from the dataset"""
	
	#Import data
	hf5 = tables.open_file(data_dir, 'r+', title = data_dir[-1])
	data = hf5.root.clean_data[0,:,:]
	sampling_rate = hf5.root.sampling_rate[0]
	segment_times = hf5.root.segment_times[:]
	segment_names = [hf5.root.segment_names[i].decode('UTF-8') for i in range(len(hf5.root.segment_names))]
	#Need to pull the times of different data segments to improve plotting
	hf5.close()
	del hf5
	downsamp_dir = ('_').join(data_dir.split('_')[:-1])+'_downsampled.h5'
	#Import downsampled dig-in data
	hf5 = tables.open_file(downsamp_dir, 'r+', title = downsamp_dir[-1])
	dig_ins = hf5.root.dig_ins.dig_ins[0]
	hf5.close()
	
	dir_save = ('/').join(data_dir.split('/')[:-1]) + '/sort_results/'
	
	pull_noise_cluster_data(data,sampling_rate,dir_save,percentiles,dig_ins,
						 segment_times,segment_names)
	
def pull_noise_cluster_data(data,sampling_rate,dir_save,percentiles,dig_ins,
							segment_times,segment_names):
	"""This function pulls the indices of spikes belonging to different 
	clusters kept after noise sorting, and runs them through template matching 
	with different percentile values"""
	
	#Grab relevant parameters
	num_neur, num_time = np.shape(data)
	num_pts_left = int(np.round(sampling_rate*(1/1000)))
	num_pts_right = int(np.round(sampling_rate*(1.5/1000)))
	axis_labels = np.arange(-num_pts_left,num_pts_right)
	viol_1 = sampling_rate*(1/1000)
	viol_2 = sampling_rate*(2/1000)
	final_clust_num = 5 #Number of clusters to use in final clustering. Consistent across all units.
	re_sort = 'n' #We don't want to re-sort the noise data
	clust_num = 8
	
	#Grab dig in times
	all_dig_ins = np.sum(dig_ins,0)
	dig_times = list(np.where(all_dig_ins > 0)[0])
	dig_diff = list(np.where(np.array(dig_times)[1:-1]-np.array(dig_times)[0:-2]>1)[0])
	dig_in_times = []
	dig_in_times.extend([0])
	dig_in_times.extend(dig_diff)
	dig_in_times = list(np.array(dig_times)[dig_in_times])
	
	#Directory for sorted data
	sort_hf5_name = dir_save.split('/')[-1].split('.')[0].split('_')[0] + '_sort.h5'
	sort_hf5_dir = dir_save + sort_hf5_name
	
	#Import all predicted spike indices
	print("Importing initial spike time data.")
	peak_indices = ss.potential_spike_times(data, sampling_rate, dir_save)
	for i in tqdm.tqdm(range(num_neur)):
		print("\n Testing channel #" + str(i))
		data_copy = np.array(data[i,:])
		#Grab peaks
		peak_ind = peak_indices[i] #Peak indices in original recording length
		left_peak_ind = np.array(peak_ind) - num_pts_left
		right_peak_ind = np.array(peak_ind) + num_pts_right
		left_peak_comp = np.zeros((len(left_peak_ind),2))
		right_peak_comp = np.zeros((len(left_peak_ind),2))
		left_peak_comp[:,0] = left_peak_ind
		right_peak_comp[:,0] = right_peak_ind
		right_peak_comp[:,1] = num_time
		p_i_l = np.max(left_peak_comp,axis=1).astype(int)
		p_i_r = np.min(right_peak_comp,axis=1).astype(int)
		del left_peak_ind, right_peak_ind, left_peak_comp, right_peak_comp
		data_chunk_lengths = p_i_r - p_i_l
		too_short = np.where(data_chunk_lengths < num_pts_left + num_pts_right)[0]
		keep_ind = np.setdiff1d(np.arange(len(p_i_l)),too_short)
		all_spikes = [list(data_copy[p_i_l[ind]:p_i_r[ind]]) for ind in keep_ind]
		all_peaks = list(np.array(peak_ind)[keep_ind]) #Peak indices in original recording length
		del p_i_l, p_i_r, data_chunk_lengths, too_short, keep_ind
		sort_ind = ss.spike_clust(all_spikes, all_peaks, 
									 clust_num, i, dir_save, axis_labels, 
									 viol_1, viol_2, 'noise_removal', segment_times,
									 segment_names, dig_in_times, re_sort)
		sorted_peak_ind = [list(np.array(all_peaks)[sort_ind[i]]) for i in range(len(sort_ind))]
		for p_i in tqdm.tqdm(range(len(percentiles))):
			good_spikes = []
			good_ind = []
			print("\t Performing Template Matching to Further Clean")
			#FUTURE IMPROVEMENT NOTE: Add csv storage of indices for further speediness if re-processing in future
			for g_i in range(len(sort_ind)):
				print("\t Template Matching Sorted Group " + str(g_i))
				s_i = sorted_peak_ind[g_i] #Original indices
				s_s_i = sort_ind[g_i] #Sorted set indices
				sort_spikes = np.array(all_spikes)[s_s_i]
				g_spikes, g_ind = ss.spike_template_sort(sort_spikes,sampling_rate,num_pts_left,num_pts_right,10)
				good_spikes.extend(g_spikes) #Store the good spike profiles
				good_ind.extend(list(np.array(s_i)[g_ind])) #Store the original indices
			del g_i, s_i
			print("\t Performing Clustering of Remaining Waveforms (Second Pass)")
			
		
		print("\t Importing post-noise-sorting data")
		noise_sort_neur_dir = dir_save + 'unit_' + str(i) + '/noise_removal/'
		neuron_spikes_csv = noise_sort_neur_dir + 'neuron_spike_ind.csv'
		with open(neuron_spikes_csv, newline='') as f:
			reader = csv.reader(f)
			neuron_spikes_list = list(reader)






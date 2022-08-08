#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 05:50:05 2022

@author: hannahgermaine
This set of functions pulls spikes out of cleaned and ICA sorted data.
"""

import numpy as np
from scipy.signal import butter, lfilter, find_peaks 
from sklearn.cluster import KMeans
import tables, tqdm, os, random
import functions.hdf5_handling as h5
import matplotlib.pyplot as plt

def run_spike_sort(data,sampling_rate,dir_save):
	"""This function performs clustering spike sorting to separate out spikes
	from the dataset"""
	
	#Grab relevant parameters
	num_neur, num_time = np.shape(data)
	min_dist_btwn_peaks = int(np.round(sampling_rate*(2/1000)))
	num_pts_left = int(np.round(sampling_rate*(1/1000)))
	num_pts_right = int(np.round(sampling_rate*(1.5/1000)))
	axis_labels = np.arange(-num_pts_left,num_pts_right)
	total_pts = num_pts_left + num_pts_right
	viol_1 = sampling_rate*(1/1000)
	viol_2 = sampling_rate*(2/1000)
	
	#Create directory for sorted data
	if os.path.isdir(dir_save) == False:
		os.mkdir(dir_save)
	sort_hf5_name = dir_save.split('/')[-1].split('.')[0].split('_')[0] + '_sort.h5'
	sort_hf5_dir = dir_save + sort_hf5_name
	
	#Get the number of clusters to use in spike sorting
	print("Beginning Spike Sorting")
	clust_num = 0
	clust_loop = 1
	while clust_loop == 1:
		cluster_num = input("Please enter the number of clusters you'd like to use for each signal (> 1): ")
		try:
			clust_num = int(cluster_num)
			clust_loop = 0
		except:
			print("ERROR: Please enter a valid integer.")
	
	#Pull spikes from data	
	print("Now beginning spike sorting.")
	separated_spikes = []
	separated_spikes_bin = []
	for i in tqdm.tqdm(range(num_neur)):
		print("\n Sorting channel #" + str(i))
		data_copy = data[i,:]
		#Grab peaks
		positive_peaks_data = find_peaks(data_copy,
						  distance=min_dist_btwn_peaks)[0]
		negative_peaks_data = find_peaks(-1*data_copy,
						  distance=min_dist_btwn_peaks)[0]
		#Pull spike profiles
		print("Pulling Spike Profiles.")
		pos_spikes = []
		for p_i in positive_peaks_data:
			p_i_l = int(max(p_i - num_pts_left,0))
			p_i_r = int(min(p_i + num_pts_right,num_time))
			points = np.arange(p_i_l,p_i_r)
			if len(points) < total_pts:
				missing_len = int(total_pts - len(points))
				data_chunk = list(data[i,points])
				data_chunk.extend([0 for k in range(0,missing_len)])
			else:
				data_chunk = list(data[i,points])
			pos_spikes.append(data_chunk)
		neg_spikes = []
		for p_i in negative_peaks_data:
			p_i_l = int(max(p_i - num_pts_left,0))
			p_i_r = int(min(p_i + num_pts_right,num_time))
			points = np.arange(p_i_l,p_i_r)
			if len(points) < total_pts:
				missing_len = int(total_pts - len(points))
				data_chunk = list(data[i,points])
				data_chunk.extend([0 for k in range(0,missing_len)])
			else:
				data_chunk = list(data[i,points])
			neg_spikes.append(data_chunk)
		#Perform clustering on pulled spikes
		print("Positive Spikes First.")
		neuron_pos_spike_ind = spike_clust(pos_spikes, positive_peaks_data, clust_num, i, 
							   dir_save, axis_labels, viol_1, viol_2, 'pos')
		neuron_spikes = np.zeros((len(neuron_pos_spike_ind),num_time)) #Vector of spike waveforms
		neuron_spikes_bin = np.zeros((len(neuron_pos_spike_ind),num_time)) #Binary vector of spike times
		for nsi in range(len(neuron_pos_spike_ind)):
			for pi in neuron_pos_spike_ind[nsi,:]:
				p_i_l = int(max(p_i - num_pts_left,0))
				p_i_r = int(min(p_i + num_pts_right,num_time))
				points = np.arange(p_i_l,p_i_r)
				neuron_spikes[nsi,points] = data[i,points]
				neuron_spikes_bin[nsi,pi] = 1
		separated_spikes.extend(neuron_spikes)
		separated_spikes_bin.extend(neuron_spikes_bin)
		print("Negative Spikes Second.")
		neuron_neg_spike_ind = spike_clust(neg_spikes, negative_peaks_data, clust_num, i, 
							   dir_save, axis_labels, viol_1, viol_2, 'neg')
		neuron_spikes = np.zeros((len(neuron_neg_spike_ind),num_time)) #Vector of spike waveforms
		neuron_spikes_bin = np.zeros((len(neuron_neg_spike_ind),num_time)) #Binary vector of spike times
		for nsi in range(len(neuron_neg_spike_ind)):
			for pi in neuron_neg_spike_ind[nsi,:]:
				p_i_l = int(max(p_i - num_pts_left,0))
				p_i_r = int(min(p_i + num_pts_right,num_time))
				points = np.arange(p_i_l,p_i_r)
				neuron_spikes[nsi,points] = data[i,points]
				neuron_spikes_bin[nsi,pi] = 1
		separated_spikes.extend(neuron_spikes)
		separated_spikes_bin.extend(neuron_spikes_bin)
	
	return sort_hf5_dir, separated_spikes

def run_ica_spike_sort(ICA_h5_dir):
	"""This function pulls data from the ICA hf5 file, finds the ICA data 
	peaks, and performs clustering spike sorting to separate out true peaks"""
	
	#Import ICA weights and cleaned data
	hf5 = tables.open_file(ICA_h5_dir, 'r+', title = ICA_h5_dir[-1])
	ICA_weights = hf5.root.ica_weights[0,:,:]
	clean_data = hf5.root.cleaned_data[0,:,:]
	sampling_rate = hf5.root.sampling_rate[0]
	hf5.close()
	
	#Create directory for sorted data
	sort_data_dir = ('/').join(ICA_h5_dir.split('/')[:-2]) + '/sort_results/'
	
	#Convert data to ICA components
	components = np.matmul(ICA_weights,clean_data)
	del clean_data	
	
	#Pull spikes from components	
	sort_hf5_dir, separated_spikes = run_spike_sort(components,sampling_rate,sort_data_dir)
		
	return sort_hf5_dir, separated_spikes

def spike_clust(spikes, peak_indices, clust_num, i, sort_data_dir, axis_labels, 
				viol_1, viol_2, type_spike):
	"""This function performs clustering on spikes pulled from each component.
	Inputs:
		spikes = list of spike samples num_spikes x length_spike
		peak_indices = indices of each spike
		clust_num = number of clusters for clustering
		i = index of component being clustered
		sort_data_dir = directory to store images in
		axis_labels = x-labels for plotting spike samples
		viol_1 = number of indices btwn spikes for 1 ms violation
		viol_2 = number of indices btwn spikes for 2 ms violation
		type_spike = type of peak (pos or neg)
	Outputs:
		neuron_spike_ind = indices of spikes selected as true."""
	
	print("Performing K-Means clustering of data.")
	
	#Create storage folder
	sort_neur_dir = sort_data_dir + 'unit_' + str(i) + '/'
	if os.path.isdir(sort_neur_dir) == False:
		os.mkdir(sort_neur_dir)
	sort_neur_type_dir = sort_neur_dir + type_spike + '/'
	if os.path.isdir(sort_neur_type_dir) == False:
		os.mkdir(sort_neur_type_dir)
		
	#Set parameters
	viol_2_cutoff = 1.5 #Maximum allowed violation percentage for 2 ms
	viol_1_cutoff = 0.1 #Maximum allowed violation percentage for 1 ms
	num_vis = 1000
	
	#Perform kmeans clustering
	kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(spikes)
	centers = kmeans.cluster_centers_
	center_dists = np.zeros([clust_num,clust_num])
	for j1 in range(clust_num):
		for j2 in range(clust_num-1):
			center_dists[j1,j2] = np.sqrt((centers[j1][0] - centers[j2][0])**2 + (centers[j1][1] - centers[j2][1])**2)
	cluster_inertia = kmeans.inertia_
	#Pull spikes by label and ask for user input of whether it's a spike or not
	labels = kmeans.labels_
	violations = []
	any_good = 0
	for li in range(clust_num):
		ind_labelled = list(np.where(labels == li)[0])
		spikes_labelled = [spikes[il] for il in ind_labelled]
		#Check for violations first
		peak_ind = [peak_indices[i] for i in ind_labelled]
		peak_diff = list(np.subtract(peak_ind[1:-1],peak_ind[0:-2]))
		viol_1_times = len(np.where(peak_diff <= viol_1)[0])
		viol_2_times = len(np.where(peak_diff <= viol_2)[0])
		viol_1_percent = round(viol_1_times/len(peak_diff)*100,2)
		viol_2_percent = round(viol_2_times/len(peak_diff)*100,2)
		violations.append([viol_1_percent,viol_2_percent])
		if viol_2_percent < viol_2_cutoff:
			if viol_1_percent < viol_1_cutoff:
				any_good += 1
				print("Cluster " + str(li) + " passed violation cutoffs. Now plotting.")
				#Select sub-population of spikes to plot as an example
				plot_ind = [random.randint(0,len(peak_ind)) for i in range(num_vis)] #Pick 100 random waveforms to plot
				#Plot spikes overlayed
				plt.figure(figsize=(30,20))
				plt.subplot(1,2,1)
				for si in plot_ind:
					plt.plot(axis_labels,spikes_labelled[si],alpha=0.5)
				plt.ylabel('mV')
				plt.title('Cluster ' + str(li) + ' x' + str(num_vis) + ' Waveforms')
				plt.subplot(1,2,2)
				plt.errorbar(axis_labels,np.mean(spikes_labelled,0),yerr=np.std(spikes_labelled,0),xerr=None)
				plt.title('Cluster' + str(li) + ' Average Waveform + Error')
				plt.savefig(sort_neur_type_dir + 'waveforms_' + str(li) + '.png', dpi=100)
	neuron_spike_ind = []
	if any_good > 0:
		print("Please navigate to the directory " + sort_neur_dir)
		print("Inspect the output visuals of spike clusters, and decide which you'd like to keep.")
		keep_loop = 1
		while keep_loop == 1:
			keep_any = input("Would you like to keep any of the clusters as spikes (y/n)? ")
			if keep_any != 'y' and keep_any != 'n':
				print("Error, please enter a valid value.")
			else:
				keep_loop = 0
		if keep_any == 'y':	
			print("Please enter a comma-separated list of indices you'd like to keep (ex. 0,4,6)")
			ind_good = input("Keep-indices: ").split(',')
	
		try:
			ind_good = [int(ind_good[i]) for i in range(len(ind_good))]
			comb_loop = 1
			while comb_loop == 1:
				combine_spikes = input("Do all of these spikes come from the same neuron (y/n)? ")
				if combine_spikes != 'y' and combine_spikes != 'n':
					print("Error, please enter a valid value.")
				else:
					comb_loop = 0
			for ig in ind_good:
				ind_labelled = list(np.where(labels == ig)[0])
				peak_ind = [peak_indices[i] for i in ind_labelled]
				if combine_spikes == 'y':
					neuron_spike_ind.extend(peak_ind)
				else:
					neuron_spike_ind.append(peak_ind)
		except:
			print("No spikes selected.")
	else:
		print("No good clusters.")
		
	return neuron_spike_ind
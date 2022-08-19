#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 05:50:05 2022

@author: hannahgermaine
This set of functions pulls spikes out of cleaned and ICA sorted data.
"""

import numpy as np
from scipy.signal import find_peaks 
from sklearn.cluster import KMeans
import tables, tqdm, os, random, csv
import matplotlib.pyplot as plt
import functions.spike_templates as st

def run_spike_sort(data_dir):
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
	
	dir_save = ('/').join(data_dir.split('/')[:-1]) + '/sort_results/'
	
	sort_hf5_dir, separated_spikes = spike_sort(data,sampling_rate,dir_save,
											 segment_times,segment_names)
	

def run_ica_spike_sort(ICA_h5_dir):
	"""This function pulls data from the ICA hf5 file, finds the ICA data 
	peaks, and performs clustering spike sorting to separate out true peaks"""
	
	#Import ICA weights and cleaned data
	hf5 = tables.open_file(ICA_h5_dir, 'r+', title = ICA_h5_dir[-1])
	ICA_weights = hf5.root.ica_weights[0,:,:]
	clean_data = hf5.root.cleaned_data[0,:,:]
	sampling_rate = hf5.root.sampling_rate[0]
	hf5.close()
	del hf5
	
	#Create directory for sorted data
	sort_data_dir = ('/').join(ICA_h5_dir.split('/')[:-2]) + '/sort_results/'
	
	#Convert data to ICA components
	components = np.matmul(ICA_weights,clean_data)
	del clean_data	
	
	#Pull spikes from components	
	sort_hf5_dir, separated_spikes = run_spike_sort(components,sampling_rate,sort_data_dir)
		
	return sort_hf5_dir, separated_spikes

def spike_sort(data,sampling_rate,dir_save,segment_times,segment_names):
	"""This function performs clustering spike sorting to separate out spikes
	from the dataset"""
	
	#Grab relevant parameters
	num_neur, num_time = np.shape(data)
	min_dist_btwn_peaks = int(np.round(sampling_rate*(0.5/1000)))
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
		print("Think of the number of clusters you'd like to use for sorting.")
		cluster_num = input("Please enter the number you'd like to use (> 1): ")
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
		#Remove any positive peaks that are too close to negative peaks
		all_peaks = np.unique(np.concatenate((positive_peaks_data,negative_peaks_data)))
		all_peaks_diff = all_peaks[1:-1] - all_peaks[0:-2]
		too_close_peaks = np.where(all_peaks_diff < min_dist_btwn_peaks)[0]
		too_close_ind = np.unique(np.concatenate((too_close_peaks,too_close_peaks+1)))
		positive_peaks_data = np.setdiff1d(positive_peaks_data,too_close_ind)
		del all_peaks_diff, too_close_peaks, too_close_ind
		#Pull spike profiles
		print("\t Pulling Positive Spike Profiles.")
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
		print("\t Pulling Negative Spike Profiles.")
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
		del p_i, p_i_l, p_i_r, points, missing_len, data_chunk
		all_spikes = []
		all_spikes.extend(pos_spikes)
		all_spikes.extend(neg_spikes)
		good_spikes, peaks_data = spike_template_sort(all_spikes,sampling_rate,num_pts_left,num_pts_right)
		#Perform clustering on template selected spikes
		neuron_spike_ind = spike_clust(good_spikes, peaks_data, 
									 clust_num, i, dir_save, axis_labels, 
									 viol_1, viol_2, 'all', segment_times,
									 segment_names)
		ns_size_1 = np.shape(neuron_spike_ind)[0]
		try:
			ns_size_2 = np.shape(neuron_spike_ind[0])[0]
			neuron_spikes = np.zeros((ns_size_1,num_time)) #Vector of spike waveforms
			neuron_spikes_bin = np.zeros((ns_size_1,num_time)) #Binary vector of spike times
			for nsi in range(ns_size_1):
				for pi in neuron_spike_ind[nsi]:
					p_i_l = int(max(pi - num_pts_left,0))
					p_i_r = int(min(pi + num_pts_right,num_time))
					points = np.arange(p_i_l,p_i_r)
					neuron_spikes[nsi,points] = data[i,points]
					neuron_spikes_bin[nsi,pi] = 1
		except:
			neuron_spikes = np.zeros((num_time)) #Vector of spike waveforms
			neuron_spikes_bin = np.zeros((num_time)) #Binary vector of spike times
			for pi in neuron_spike_ind:
				p_i_l = int(max(pi - num_pts_left,0))
				p_i_r = int(min(pi + num_pts_right,num_time))
				points = np.arange(p_i_l,p_i_r)
				neuron_spikes[points] = data[i,points]
				neuron_spikes_bin[pi] = 1
		separated_spikes.extend(neuron_spikes)
		separated_spikes_bin.extend(neuron_spikes_bin)
	
	return sort_hf5_dir, separated_spikes

def spike_clust(spikes, peak_indices, clust_num, i, sort_data_dir, axis_labels, 
				viol_1, viol_2, type_spike, segment_times, segment_names):
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
	
	print("\t Performing K-Means clustering of data.")
	
	#Create storage folder
	sort_neur_dir = sort_data_dir + 'unit_' + str(i) + '/'
	if os.path.isdir(sort_neur_dir) == False:
		os.mkdir(sort_neur_dir)
	sort_neur_type_dir = sort_neur_dir + type_spike + '/'
	if os.path.isdir(sort_neur_type_dir) == False:
		os.mkdir(sort_neur_type_dir)
	sort_neur_type_csv = sort_neur_type_dir + 'neuron_spike_ind.csv'
	re_sort = 'n'
	if os.path.isfile(sort_neur_type_csv) == False:
		re_sort = 'y'
	else:
		#MODIFY TO SEARCH FOR A CSV OF SPIKE INDICES
		print('\t The ' + type_spike + ' spikes of this unit have previously been sorted.')
		re_sort = input('Would you like to re-sort [y/n]? ')
	
	if re_sort == 'y':
		#Set parameters
		viol_2_cutoff = 2 #Maximum allowed violation percentage for 2 ms
		viol_1_cutoff = 1 #Maximum allowed violation percentage for 1 ms
		num_vis = 500 #Number of waveforms to visualize for example plot
		
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
			#TEMPORARY CODE CHANGES BELOW
			#Just to see what clusters look like, commenting violation cutoffs for now
			if viol_2_percent < 100: #viol_2_cutoff:
				if viol_1_percent < 100: #viol_1_cutoff:
					any_good += 1
					#Adding in pass test in meantime
					pass_val = (viol_2_percent < viol_2_cutoff) and (viol_1_percent < viol_1_cutoff)
					if pass_val == True:
						print("\t \t Cluster " + str(li) + " passed violation cutoffs. Now plotting.")
					else:
						print("\t \t Cluster " + str(li) + " did not pass violation cutoffs. Now plotting.")
					#Select sub-population of spikes to plot as an example
					plot_num_vis = min(num_vis,len(spikes_labelled)-1)
					plot_ind = [random.randint(0,len(peak_ind)) for i in range(plot_num_vis)] #Pick 100 random waveforms to plot
					#Plot spikes overlayed
					fig = plt.figure(figsize=(30,20))
					plt.subplot(2,2,1)
					for si in plot_ind:
						try:
							plt.plot(axis_labels,spikes_labelled[si],'-b',alpha=0.2)
						except:
							print("\t \t Error: Skipped plotting a waveform.")
					plt.ylabel('mV')
					plt.title('Cluster ' + str(li) + ' x' + str(plot_num_vis) + ' Waveforms')
					plt.subplot(2,2,2)
					plt.errorbar(axis_labels,np.mean(spikes_labelled,0),yerr=np.std(spikes_labelled,0),xerr=None)
					plt.title('Cluster ' + str(li) + ' Average Waveform + Error')
					plt.subplot(2,2,3)
					#Find ISI distribution and plot
					plt.hist(peak_diff,bins=min(50,round(len(peak_diff)/10)))
					plt.title('Cluster ' + str(li) + ' ISI Distribution')
					#Histogram of time of spike occurrence
					plt.subplot(2,2,4)
					plt.hist(peak_ind,bins=min(50,round(len(peak_ind)/10)))
					[plt.axvline(segment_times[i],label=segment_names[i]) for i in range(len(segment_names))]
					plt.legend()
					plt.title('Cluster ' + str(li) + ' Spike Time Histogram')
					plt.suptitle('Number of Waveforms = ' + str(len(spikes_labelled)) + '\n 1 ms violation percent = ' + str(viol_1_percent) + '\n 2 ms violation percent = ' + str(viol_2_percent))
					fig.savefig(sort_neur_type_dir + 'waveforms_' + str(li) + '.png', dpi=100)
					plt.close(fig)
		neuron_spike_ind = []
		if any_good > 0:
			print("\t Please navigate to the directory " + sort_neur_dir)
			print("\t Inspect the output visuals of spike clusters, and decide which you'd like to keep.")
			keep_loop = 1
			while keep_loop == 1:
				keep_any = input("\t Would you like to keep any of the clusters as spikes (y/n)? ")
				if keep_any != 'y' and keep_any != 'n':
					print("\t Error, please enter a valid value.")
				else:
					keep_loop = 0
			if keep_any == 'y':	
				print("\t Please enter a comma-separated list of indices you'd like to keep (ex. 0,4,6)")
				ind_good = input("\t Keep-indices: ").split(',')
		
			try:
				ind_good = [int(ind_good[i]) for i in range(len(ind_good))]
				comb_loop = 1
				while comb_loop == 1:
					combine_spikes = input("\t Do all of these spikes come from the same neuron (y/n)? ")
					if combine_spikes != 'y' and combine_spikes != 'n':
						print("\t Error, please enter a valid value.")
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
				print("\t No spikes selected.")
		else:
			print("\t No good clusters.")
		#Save to CSV
		with open(sort_neur_type_csv, 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows(neuron_spike_ind)
	else:
		#ADD CODE TO PULL SPIKE INDICES FROM CSV
		with open(sort_neur_type_csv, newline='') as f:
			reader = csv.reader(f)
			neuron_spike_ind = list(reader)
		
	return neuron_spike_ind

def spike_template_sort(all_spikes,sampling_rate,num_pts_left,num_pts_right):
	"""This function performs template-matching to pull out potential spikes."""
	
	#Grab templates
	print("Performing Template Comparison.")
	max_volt = np.max(np.abs(all_spikes),1)
	max_volt = np.expand_dims(max_volt,1)
	norm_spikes = np.divide(all_spikes,max_volt) #Normalize the data
	#Grab the number of peaks > 2 std per snippet
	data_std = np.std(norm_spikes,1)
	num_peaks_pos = np.array([len(find_peaks(all_spikes[i],data_std[i])[0]) for i in range(len(all_spikes))])
	num_peaks_neg = np.array([len(find_peaks(-1*all_spikes[i],data_std[i])[0]) for i in range(len(all_spikes))])
	num_peaks = num_peaks_pos + num_peaks_neg
	#Grab templates of spikes
	spike_templates = st.generate_templates(sampling_rate,num_pts_left,num_pts_right)
	num_types = np.shape(spike_templates)[0]
	good_ind = []
	for i in tqdm.tqdm(range(num_types)):
		#Distance from template
		a = np.multiply(spike_templates[i,:],np.zeros(np.shape(all_spikes)))
		dist = np.sqrt(np.sum((norm_spikes - a)**2,1))
# 		dist_cutoff = np.nanpercentile(dist,10)
# 		g_i = list(np.where(dist < dist_cutoff)[0])
# 		good_ind.extend(g_i)
		#Cross correlation with template
		cross_corr = [np.correlate(norm_spikes[j],spike_templates[i])[0] for j in range(len(norm_spikes))]
# 		top_cross_corr = np.nanpercentile(cross_corr,95)
# 		bottom_cross_corr = np.nanpercentile(cross_corr,5)
# 		c_i_bottom = list(np.where(cross_corr < bottom_cross_corr)[0])
# 		c_i_top = list(np.where(cross_corr > top_cross_corr)[0])
# 		c_i = []
# 		c_i.extend(c_i_bottom)
# 		c_i.extend(c_i_top)
# 		good_ind.extend(c_i)
		#Combination value: dist*(1/abs(corr))*peak_count
		corr_conversion = np.divide(1,np.abs(cross_corr))
		similarity_index = np.multiply(np.multiply(dist,corr_conversion),num_peaks)
		sim_cutoff = np.nanpercentile(similarity_index,5)
		g_i = list(np.where(similarity_index < sim_cutoff)[0])
		good_ind.extend(g_i)
# 	del i, a, g_i, max_volt, norm_spikes
	good_ind = np.unique(good_ind)
	potential_spikes = np.array(all_spikes)[good_ind]
	potential_spikes = [list(potential_spikes[i,:]) for i in range(len(potential_spikes))]
	
	#Plots for checking
# 	axis_labels = np.arange(-num_pts_left,num_pts_right)
# 	bad_ind = np.setdiff1d(np.arange(len(all_spikes)),good_ind)
# 	num_vis = 10
# 	samp_bad = [random.randint(0,len(bad_ind)) for i in range(num_vis)]
# 	samp_good = [random.randint(0,len(good_ind)) for i in range(num_vis)]
# 	fig = plt.figure(figsize=(10,10))
# 	for i in range(num_vis):
# 		plt.subplot(num_vis,2,(2*i)+1)
# 		plt.plot(axis_labels,all_spikes[samp_good[i]])
# 		plt.title('Good Example')
# 		plt.subplot(num_vis,2,(2*i)+2)
# 		plt.plot(axis_labels,all_spikes[samp_bad[i]])
# 		plt.title('Bad Example')
# 	plt.tight_layout()
	
	return potential_spikes, good_ind
	
	
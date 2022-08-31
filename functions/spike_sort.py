#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 05:50:05 2022

@author: hannahgermaine
This set of functions pulls spikes out of cleaned and ICA sorted data.
"""

import numpy as np
import scipy.stats as ss
from scipy.signal import find_peaks, correlate, convolve2d
from sklearn.cluster import KMeans
import tables, tqdm, os, random, csv, time, sys
import matplotlib.pyplot as plt
from numba import jit


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
	downsamp_dir = ('_').join(data_dir.split('_')[:-1])+'_downsampled.h5'
	#Import downsampled dig-in data
	hf5 = tables.open_file(downsamp_dir, 'r+', title = downsamp_dir[-1])
	dig_ins = hf5.root.dig_ins.dig_ins[0]
	hf5.close()
	
	dir_save = ('/').join(data_dir.split('/')[:-1]) + '/sort_results/'
	
	sort_hf5_dir, separated_spikes = spike_sort(data,sampling_rate,dir_save,
											 segment_times,segment_names,dig_ins)
	

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

def potential_spike_times(data,sampling_rate,dir_save):
	"""Function to grab potential spike times for further analysis. Peaks 
	outside 1 absolute deviation and 1 ms to the left, and 1.5 ms to the right 
	around them are kept, while the rest are scrubbed."""
	
	get_ind = 'n'
	init_times_csv = dir_save + 'init_times.csv'
	if os.path.isfile(init_times_csv) == False:
		get_ind = 'y'
	else:
		print('\t Initial spike times previously pulled.')
		
	if get_ind == 'y':	
		num_neur, num_time = np.shape(data)
		#Grab mean and std
		std_dev = np.std(data,1)
		print("Searching for potential spike indices")
		peak_ind = []
		for i in range(num_neur):
			data_copy = data[i,:]
			#Start with positive peaks
			positive_peaks_data = find_peaks(data_copy,height=1*std_dev[i])[0]#,
							  #distance=min_dist_btwn_peaks)[0]
			negative_peaks_data = find_peaks(-1*data_copy,height=1*std_dev[i])[0]#,
							  #distance=min_dist_btwn_peaks)[0]
			#Remove any positive peaks that are too close to negative peaks
			all_peaks = np.unique(np.concatenate((positive_peaks_data,negative_peaks_data)))
			all_peaks_diff = all_peaks[1:-1] - all_peaks[0:-2]
			too_close_peaks = np.where(all_peaks_diff < sampling_rate/1000)[0]
			too_close_ind = np.unique(np.concatenate((too_close_peaks,too_close_peaks+1)))
			positive_peaks_data = np.setdiff1d(positive_peaks_data,too_close_ind)
			peak_indices = []
			peak_indices.extend(positive_peaks_data)
			peak_indices.extend(negative_peaks_data)
			peak_ind.append(peak_indices)
		#Save results to .csv
		with open(init_times_csv, 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows(peak_ind)
	else:
		print('\t Importing spike times.')
		with open(init_times_csv, newline='') as f:
			reader = csv.reader(f)
			peak_ind_csv = list(reader)
		peak_ind = []
		for i_c in range(len(peak_ind_csv)):
			str_list = peak_ind_csv[i_c]
			int_list = [int(str_list[i]) for i in range(len(str_list))]
			peak_ind.append(int_list)
	
	return peak_ind

def spike_sort(data,sampling_rate,dir_save,segment_times,segment_names,dig_ins):
	"""This function performs clustering spike sorting to separate out spikes
	from the dataset
	INPUTS:
		-data = array of num_neur x num_time size containing the cleaned data
		-sampling_rate = integer of the sampling rate (Hz)
		-dir_save = where to save outputs
		-segment_times = times of different segments in the data
		-segment_names = names of the different segments
		-dig_ins = array of num_dig x num_time with 1s wherever a tastant 
					was being delivered"""
	
	#Grab relevant parameters
	num_neur, num_time = np.shape(data)
	#min_dist_btwn_peaks = 3 #in sampling rate steps
	num_pts_left = int(np.round(sampling_rate*(1/1000)))
	num_pts_right = int(np.round(sampling_rate*(1.5/1000)))
	axis_labels = np.arange(-num_pts_left,num_pts_right)
	#total_pts = num_pts_left + num_pts_right
	viol_1 = sampling_rate*(1/1000)
	viol_2 = sampling_rate*(2/1000)
	
	#Grab dig in times
	all_dig_ins = np.sum(dig_ins,0)
	dig_times = list(np.where(all_dig_ins > 0)[0])
	dig_diff = list(np.where(np.array(dig_times)[1:-1]-np.array(dig_times)[0:-2]>1)[0])
	dig_in_times = []
	dig_in_times.extend([0])
	dig_in_times.extend(dig_diff)
	dig_in_times = list(np.array(dig_times)[dig_in_times])
	
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
		print("\n INPUT REQUESTED: Think of the number of clusters you'd like to use for initial sorting (Removal of noise).")
		cluster_num = input("Please enter the number you'd like to use (> 1): ")
		try:
			clust_num = int(cluster_num)
			clust_loop = 0
		except:
			print("ERROR: Please enter a valid integer.")
	del cluster_num, clust_loop
	
	clust_num_fin = 0
	clust_loop = 1
	while clust_loop == 1:
		print("\n INPUT REQUESTED: Think of the number of clusters you'd like to use for final sorting. I recommended that it's less than the initial value.")
		cluster_num_fin = input("Please enter the number you'd like to use (> 1): ")
		try:
			clust_num_fin = int(cluster_num_fin)
			clust_loop = 0
		except:
			print("ERROR: Please enter a valid integer.")
	del cluster_num_fin, clust_loop
	
	#Pull spikes from data	
	print("\n Now beginning spike sorting.")
	separated_spikes = []
	separated_spikes_bin = []
	peak_indices = potential_spike_times(data, sampling_rate, dir_save)
	for i in tqdm.tqdm(range(num_neur)):
		print("\n Sorting channel #" + str(i))
		#First check for final sort and ask if want to keep
		keep_final = 0
		final_sort_neur_dir = dir_save + 'unit_' + str(i) + '/final/'
		neuron_spikes_csv = final_sort_neur_dir + 'neuron_spikes.csv'
		neuron_spikes_bin_csv = final_sort_neur_dir + 'neuron_spikes_bin.csv'
		if os.path.isfile(neuron_spikes_csv):
			print("\t INPUT REQUESTED: Sorting previously performed.")
			sort_loop = 1
			while sort_loop == 1:
				resort_channel = input("\t Would you like to re-sort [y/n]? ")
				if resort_channel != 'y' and resort_channel != 'n':
					print("\t Incorrect entry.")
				elif resort_channel == 'n':
					keep_final = 1
					sort_loop = 0
				elif resort_channel == 'y':
					sort_loop = 0
		tic = time.time()
		if keep_final == 0:
			#If no final sort or don't want to keep, then run through protocol
			data_copy = np.array(data[i,:])
			#Grab peaks
			peak_ind = peak_indices[i] #Peak indices in original recording length
			#Pull spike profiles
			print("\t Pulling Spike Profiles.")
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
			all_spikes = np.zeros((len(keep_ind),num_pts_left+num_pts_right))
			for k_i in tqdm.tqdm(range(len(keep_ind))):
				ind = keep_ind[k_i]
				all_spikes[k_i,:] = data_copy[p_i_l[ind]:p_i_r[ind]]
			all_spikes = list(all_spikes)
			all_peaks = list(np.array(peak_ind)[keep_ind]) #Peak indices in original recording length
			del p_i_l, p_i_r, data_chunk_lengths, too_short, keep_ind
			#Cluster all spikes first to get rid of noise
			print("\t Performing Clustering to Remove Noise (First Pass)")
			sort_ind = spike_clust(all_spikes, all_peaks, 
										 clust_num, i, dir_save, axis_labels, 
										 viol_1, viol_2, 'noise_removal', segment_times,
										 segment_names, dig_in_times)
			sorted_peak_ind = [list(np.array(all_peaks)[sort_ind[i]]) for i in range(len(sort_ind))]
			good_spikes = []
			good_ind = []
			print("\t Performing Template Matching to Further Clean")
			#FUTURE IMPROVEMENT NOTE: Add csv storage of indices for further speediness if re-processing in future
			for g_i in range(len(sort_ind)):
				print("\t Template Matching Sorted Group " + str(g_i))
				s_i = sorted_peak_ind[g_i] #Original indices
				s_s_i = sort_ind[g_i] #Sorted set indices
				sort_spikes = np.array(all_spikes)[s_s_i]
				g_spikes, g_ind = spike_template_sort(sort_spikes,sampling_rate,num_pts_left,num_pts_right,10)
				good_spikes.extend(g_spikes) #Store the good spike profiles
				good_ind.extend(list(np.array(s_i)[g_ind])) #Store the original indices
			del g_i, s_i
			print("\t Performing Clustering of Remaining Waveforms (Second Pass)")
			
			sort_ind_2 = spike_clust(good_spikes, good_ind, 
										 clust_num_fin, i, dir_save, axis_labels, 
										 viol_1, viol_2, 'final', segment_times,
										 segment_names, dig_in_times)
			
			#Save sorted spike indices and profiles
			final_spikes = []
			final_ind = []
			for g_i in range(len(sort_ind_2)):
				s_i = sort_ind_2[g_i]
				spikes_i = [list(good_spikes[s_i_i]) for s_i_i in s_i]
				final_spikes.append(spikes_i)
				final_ind.append(list(np.array(good_ind)[s_i]))
			num_neur_sort = len(final_ind)
			if num_neur_sort > 0:
				neuron_spikes = np.zeros((num_neur_sort,num_time))
				neuron_spikes_bin = np.zeros((num_neur_sort,num_time))
				for n_i in range(num_neur_sort):
					for pi in range(len(final_ind[n_i])):
						p_i = final_ind[n_i][pi]
						p_i_l = p_i - num_pts_left
						p_i_r = p_i + num_pts_right
						points = np.arange(p_i_l,p_i_r)
						neuron_spikes[n_i,points] = final_spikes[n_i][pi]
						neuron_spikes_bin[n_i,p_i] = 1
				#Save results to csv
				print("\t Saving final results to CSV.")
				with open(neuron_spikes_csv, 'w') as f:
					# using csv.writer method from CSV package
					write = csv.writer(f)
					write.writerows(neuron_spikes)
				with open(neuron_spikes_bin_csv, 'w') as f:
					# using csv.writer method from CSV package
					write = csv.writer(f)
					write.writerows(neuron_spikes_bin)
			else:
				print("\t No neurons found.")
		else:
			print("\t Importing previously sorted data.")
			with open(neuron_spikes_csv, newline='') as f:
				reader = csv.reader(f)
				neuron_spikes_list = list(reader)
			neuron_spikes = []
			for i_c in range(len(neuron_spikes_list)):
				str_list = neuron_spikes_list[i_c]
				float_list = [float(str_list[i]) for i in range(len(str_list))]
				neuron_spikes.append(float_list)
			with open(neuron_spikes_bin_csv, newline='') as f:
				reader = csv.reader(f)
				neuron_spikes_bin_list = list(reader)
			neuron_spikes_bin = []
			for i_c in range(len(neuron_spikes_bin_list)):
				str_list = neuron_spikes_bin_list[i_c]
				int_list = [round(float(str_list[i])) for i in range(len(str_list))]
				neuron_spikes_bin.append(int_list)
		
		#Store in larger matrix
		separated_spikes.extend(list(neuron_spikes))
		separated_spikes_bin.extend(list(neuron_spikes_bin))
			
		toc = time.time()
		print(" Time to sort channel " + str(i) + " = " + str(round((toc - tic)/60)) + " minutes")	
			
		print("\n CHECKPOINT REACHED: You don't have to sort all neurons right now.")
		cont_loop = 1
		while cont_loop == 1:
			cont_units = input("INPUT REQUESTED: Would you like to continue sorting [y/n]? ")
			if cont_units != 'y' and cont_units != 'n':
				print("Incorrect input.")
			elif cont_units == 'n':
				cont_loop = 0
				sys.exit()
			elif cont_units == 'y':
				cont_loop = 0
	
	return sort_hf5_dir, separated_spikes

def spike_clust(spikes, peak_indices, clust_num, i, sort_data_dir, axis_labels, 
				viol_1, viol_2, type_spike, segment_times, segment_names, 
				dig_in_times,re_sort='y'):
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
		segment_times = time of different experiment segments
		segment_names = names of different segments
		dig_in_times = times of tastant delivery
		re_sort = whether to re-sort if data has been previously sorted.
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
	if os.path.isfile(sort_neur_type_csv) == False:
		re_sort = 'y'
	else:
		if re_sort != 'n':
			#MODIFY TO SEARCH FOR A CSV OF SPIKE INDICES
			print('\t The ' + type_spike + ' sorting has been previously performed.')
			re_sort = input('\t Would you like to re-sort [y/n]? ')
	
	if re_sort == 'y':
		#Set parameters
		viol_2_cutoff = 10 #Maximum allowed violation percentage for 2 ms
		viol_1_cutoff = 5 #Maximum allowed violation percentage for 1 ms
		num_vis = 500 #Number of waveforms to visualize for example plot
		
		#Perform kmeans clustering
		rand_ind = np.random.randint(len(spikes),size=(100000,))
		rand_spikes = list(np.array(spikes)[rand_ind])
		print('\t Performing fitting.')
		kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(rand_spikes)
		print('\t Performing label prediction.')
		labels = kmeans.predict(spikes)
		print('\t Now testing/plotting clusters.')
		violations = []
		any_good = 0
		for li in range(clust_num):
			ind_labelled = np.where(labels == li)[0]
			spikes_labelled = np.array(spikes)[ind_labelled]
			#Check for violations first
			peak_ind = np.unique(np.array(peak_indices)[ind_labelled])
			peak_diff = np.subtract(peak_ind[1:-1],peak_ind[0:-2])
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
					plot_ind = np.random.randint(0,len(peak_ind),size=(plot_num_vis,)) #Pick 500 random waveforms to plot
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
					spike_bit = spikes_labelled[np.random.randint(0,len(spikes_labelled),size=(10000,))]
					mean_bit = np.mean(spike_bit,axis=0)
					std_bit = np.std(spike_bit,axis=0)
					plt.errorbar(axis_labels,mean_bit,yerr=std_bit,xerr=None)
					plt.title('Cluster ' + str(li) + ' Average Waveform + Std Range')
					plt.subplot(2,2,3)
					#Find ISI distribution and plot
					plt.hist(peak_diff,bins=min(100,round(len(peak_diff)/10)))
					plt.title('Cluster ' + str(li) + ' ISI Distribution (zoomed to < 100)')
					#Histogram of time of spike occurrence
					plt.subplot(2,2,4)
					plt.hist(peak_ind,bins=min(100,round(len(peak_ind)/10)))
					[plt.axvline(segment_times[i],label=segment_names[i]) for i in range(len(segment_names))]
					[plt.axvline(dig_in_times[i],c='g') for i in range(len(dig_in_times))]
					plt.legend()
					plt.title('Cluster ' + str(li) + ' Spike Time Histogram')
					plt.suptitle('Number of Waveforms = ' + str(len(spikes_labelled)) + '\n 1 ms violation percent = ' + str(viol_1_percent) + '\n 2 ms violation percent = ' + str(viol_2_percent))
					fig.savefig(sort_neur_type_dir + 'waveforms_' + str(li) + '.png', dpi=100)
					plt.close(fig)
		neuron_spike_ind = []
		if any_good > 0:
			print("\n \t INPUT REQUESTED: Please navigate to the directory " + sort_neur_type_dir)
			print("\t Inspect the output visuals of spike clusters, and decide which you'd like to keep.")
			keep_loop = 1
			while keep_loop == 1:
				keep_any = input("\t Would you like to keep any of the clusters as spikes (y/n)? ")
				if keep_any != 'y' and keep_any != 'n':
					print("\t Error, please enter a valid value.")
				else:
					keep_loop = 0
			if keep_any == 'y':	
				print("\n \t INPUT REQUESTED: Please enter a comma-separated list of indices you'd like to keep (ex. 0,4,6)")
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
					peak_ind = np.where(labels == ig)[0]
					if combine_spikes == 'y':
						if len(ind_good) == 1:
							neuron_spike_ind.extend([peak_ind])
						else:
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
		with open(sort_neur_type_csv, newline='') as f:
			reader = csv.reader(f)
			neuron_spike_ind_csv = list(reader)
		neuron_spike_ind = []
		for i_c in range(len(neuron_spike_ind_csv)):
			str_list = neuron_spike_ind_csv[i_c]
			int_list = [int(str_list[i]) for i in range(len(str_list))]
			neuron_spike_ind.append(int_list)
			
	return neuron_spike_ind

@jit(forceobj=True)
def spike_template_sort(all_spikes,sampling_rate,num_pts_left,num_pts_right,cut_percentile):
	"""This function performs template-matching to pull out potential spikes.
	INPUTS:
		- all_spikes
		- sampling_rate
		- num_pts_left
		- num_pts_right
		- cut_percentile
	OUTPUTS:
		- potential_spikes
		- good_ind
	"""
	#Grab templates
	print("\t Preparing Data for Template Matching")
	peak_val = np.abs(all_spikes[:,num_pts_left])
	peak_val = np.expand_dims(peak_val,1)
	norm_spikes = np.divide(all_spikes,peak_val) #Normalize the data
	#Grab templates of spikes
	spike_templates = generate_templates(sampling_rate,num_pts_left,num_pts_right)
	num_types = np.shape(spike_templates)[0]
	good_ind = []
	print("\t Performing Template Comparison.")
	for i in range(num_types):
		#Template correlation
		spike_mat = np.multiply(np.ones(np.shape(norm_spikes)),spike_templates[i,:])
		dist = np.sqrt(np.sum(np.square(np.subtract(norm_spikes,spike_mat)),1))
		num_peaks = [len(find_peaks(norm_spikes[s],0.5)[0]) + len(find_peaks(-1*norm_spikes[s],0.5)[0]) for s in range(len(norm_spikes))]
		score = dist*num_peaks
		percentile = np.percentile(score,cut_percentile)
# 		score_chop = dist[np.where(score < percentile)[0]]
# 		percentile = np.percentile(score_chop,10)
		good_i = np.where(score < percentile)[0]
		good_ind.extend(list(good_i))
	good_ind = list(np.unique(good_ind))
	potential_spikes = [all_spikes[g_i] for g_i in good_ind]
	
#	#Plots for checking
#  	axis_labels = np.arange(-num_pts_left,num_pts_right)
#  	bad_ind = np.setdiff1d(np.arange(len(all_spikes)),good_ind)
#  	num_vis = 10
#  	samp_bad = [random.randint(0,len(bad_ind)) for i in range(num_vis)]
#  	samp_good = [random.randint(0,len(good_ind)) for i in range(num_vis)]
#  	fig = plt.figure(figsize=(10,10))
#  	for i in range(num_vis):
# 		 plt.subplot(num_vis,2,(2*i)+1)
# 		 plt.plot(axis_labels,norm_spikes[samp_good[i]])
# 		 plt.title('Good Example')
# 		 plt.subplot(num_vis,2,(2*i)+2)
# 		 plt.plot(axis_labels,norm_spikes[samp_bad[i]])
# 		 plt.title('Bad Example')
#  	plt.tight_layout()
	
	return potential_spikes, good_ind
	
@jit(forceobj=True)
def generate_templates(sampling_rate,num_pts_left,num_pts_right):
	"""This function generates 3 template vectors of neurons with a peak 
	centered between num_pts_left and num_pts_right."""
	
	x_points = np.arange(-num_pts_left,num_pts_right)
	templates = np.zeros((3,len(x_points)))
	
	fast_spike_width = sampling_rate*(1/1000)
	sd = fast_spike_width/12
	
	pos_spike = ss.norm.pdf(x_points, 0, sd)
	max_pos_spike = max(abs(pos_spike))
	pos_spike = pos_spike/max_pos_spike + 0.01*np.random.randn(len(pos_spike))
	fast_spike = -1*pos_spike
	reg_spike_bit = ss.gamma.pdf(np.arange(fast_spike_width),5)
	peak_reg = find_peaks(reg_spike_bit)[0][0]
	reg_spike = np.concatenate((0.01*np.random.randn(num_pts_left-peak_reg),-1*reg_spike_bit),axis=0)
	len_diff = len(x_points) - len(reg_spike)
	reg_spike = np.concatenate((reg_spike,0.01*np.random.randn(len_diff)))
	max_reg_spike = max(abs(reg_spike))
	reg_spike = reg_spike/max_reg_spike
	
	templates[0,:] = pos_spike
	templates[1,:] = fast_spike
	templates[2,:] = reg_spike
	
 	# fig = plt.figure()
 	# plt.subplot(3,1,1)
 	# plt.plot(x_points,pos_spike)
 	# plt.title('Positive Spike')
 	# plt.subplot(3,1,2)
 	# plt.plot(x_points,fast_spike)
 	# plt.title('Fast Spike')
 	# plt.subplot(3,1,3)
 	# plt.plot(x_points,reg_spike)
 	# plt.title('Regular Spike')
 	# plt.tight_layout()
	
	return templates

def test_collisions():
	"""This function tests the final selected neurons for collisions across 
	all units"""
	
	
	


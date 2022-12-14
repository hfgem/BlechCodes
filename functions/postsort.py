#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:11:45 2022

@author: hannahgermaine

This code is written to perform post-sorting functions such as collisions 
testing and re-combination of oversplit neurons.
"""
import tables, tqdm, os, csv, itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from numba import jit

def run_postsort(datadir):
	"""This function serves to run through the post-sorting flow of importing
	individual neuron waveforms, performing collision tests, and recombining 
	those neurons that are oversplit.
	INPUTS:
		- datadir = directory of all of the data (level containing sort_results folder)
		
	"""
	#Get directories / names
	downsampled_dir = datadir + '/' + datadir.split('/')[-1] + '_downsampled.h5'
	sort_data_dir = datadir + '/sort_results/'
	sort_hf5_name = datadir.split('/')[-1].split('.')[0].split('_')[0] + '_sort.h5'
	sort_hf5_dir = sort_data_dir + sort_hf5_name
	
	#Import data info
	downsampled_dir_info = tables.open_file(downsampled_dir,'r',title=downsampled_dir[-1])
	num_new_time = np.shape(downsampled_dir_info.root.electrode_array.data)[-1]
	sampling_rate = downsampled_dir_info.root.sampling_rate[0]
	segment_names = downsampled_dir_info.root.experiment_components.segment_names[0]
	segment_times = downsampled_dir_info.root.experiment_components.segment_times[0]
	dig_in_names = [downsampled_dir_info.root.dig_ins.dig_in_names[i].decode('UTF-8') for i in range(len(downsampled_dir_info.root.dig_ins.dig_in_names))]
	dig_ins = downsampled_dir_info.root.dig_ins.dig_ins[0]
	downsampled_dir_info.close()
	
	#Import sorted data
	separated_spikes_ind, separated_spikes_wav, separated_spikes_stats = import_sorted(datadir,sort_data_dir,sort_hf5_dir)
	
	#Perform collision tests of sorted data
	collision_results_dir = sort_data_dir + 'collision_results/'
	if os.path.isdir(collision_results_dir) == False:
		os.mkdir(collision_results_dir)
	remove_ind = test_collisions(sampling_rate,separated_spikes_ind,separated_spikes_wav,collision_results_dir)
	
	#Remove data selected above
	for r_i in remove_ind.sort(reverse=True):
		del separated_spikes_ind[r_i]
		del separated_spikes_wav[r_i]
		try:
			del separated_spikes_stats[r_i+1]
		except:
			print("No spike stats to remove")
	
	#Save new sort data in new .h5 file with ending "_repacked.h5"
	
	
	
	
def import_sorted(dir_save,sort_data_dir,sort_hf5_dir):
	"""This function imports the already sorted data into arrays + the sorting
	statistics
	INPUTS:
		- dir_save: the directory of all data
		- sort_data_dir: the directory of the sort results ('.../sort_results')
		- sort_hf5_dir: the directory of the dataset results (_sort.h5)
	OUTPUTS:
		- separated_spikes_ind: a list of lists containing the spike times for each unit
		- sort_stats: an array of averaged sorting statistics for each unit
	"""
	
	separated_spikes_ind = []
	separated_spikes_wav = []
	separated_spikes_stats = [['Electrode Index','Neuron Index','1 ms Violations','2 ms Violations','Average Firing Rate']]
	sort_hf5 = tables.open_file(sort_hf5_dir, 'r', title = sort_hf5_dir[-1])
	sorted_units_node = sort_hf5.get_node('/sorted_units')
	i = -1 #Counter for total number of neurons
	for s_n in sorted_units_node:
		i += 1
		num_sub_u = len([w_n for w_n in s_n.times]) #Number neurons in electrode
		#For each electrode pull out unit spike times
		for n_u in range(num_sub_u):
			unit_times = eval('s_n.times.neuron_' + str(n_u) + '[0]').round().astype(int)
			separated_spikes_ind.append([unit_times])
			unit_wavs = eval('s_n.waveforms.neuron_' + str(n_u) + '[0]').round().astype(float)
			separated_spikes_wav.append([unit_wavs])
		del n_u, unit_times, unit_wavs
		#For each electrode pull out the sort statistics
		sort_neur_dir = sort_data_dir + 'unit_' + str(num_sub_u) + '/'
		save_folder = sort_neur_dir + 'final_results/'
		#Import grouped stats from CSV
		new_sort_neur_stats_csv = save_folder + 'sort_stats.csv'
		if os.path.isdir(new_sort_neur_stats_csv) == 'True':
			with open(new_sort_neur_stats_csv, newline='') as f:
				reader = csv.reader(f)
				sort_stats_list = list(reader)
			for i_s in range(len(sort_stats_list) - 1):
				stat_row = sort_stats_list[i_s + 1]
				stat_row_float = [i,i_s + 1]
				stat_row_float.extend([float(stat_row[i]) for i in range(len(stat_row) - 1)])
				if float(stat_row[-1]) == 1:
					separated_spikes_stats.append(stat_row_float)
			del reader, sort_stats_list, i_s, stat_row, stat_row_float
		del num_sub_u, sort_neur_dir, save_folder, new_sort_neur_stats_csv
	sort_hf5.close()
	
	return separated_spikes_ind, separated_spikes_wav, separated_spikes_stats


def test_collisions(sampling_rate,spike_times,spike_wavs,dir_save):
	"""This function tests the final selected neurons for collisions across 
	all units. It performs pairwise tests and looks for spike times within 3 
	time bins, totalling the number of overlaps / average number of spikes 
	between the two neurons. If the percentage is over 50, the pair is flagged 
	and the user can determine which to remove based on the statistics.
	INPUTS:
		- sampling_rate = sampling rate of data
		- spike_times = list of lists containing indices of spike times
		- spike_wavs = list of lists containing waveforms
		- spike_stats = list of lists containing neuron stats
		- dir_save = directory to store collision results (.../collision_results/)
	OUTPUTS:
		- spikes_ind = new list of lists with individual neuron spike times
		- spikes_wav = new list of lists with individual neuron waveforms
		- spikes_stats = new list of lists with individual neuron spike stats
	"""
	num_neur_plot = 500
	num_neur = len(spike_times)
	all_pairs = list(itertools.combinations(np.arange(0,num_neur),2))
	blur_ind = round((0.5/1000)*sampling_rate) #0.5 ms collision window
	collision_cutoff = 50 #Percent cutoff for collisions
	colorCodes = np.array([[0,1,0],[0,0,1]]) #Colors for plotting collision rasters
	
	print("\t Testing all units pairwise.")
	collisions_detected = 0
	collision_percents = np.zeros((num_neur,num_neur)) #row, column value = % of row index that collided with column index
	for i in tqdm.tqdm(range(len(all_pairs))):
		ind_1 = all_pairs[i][0]
		ind_2 = all_pairs[i][1]
		spike_1_list = spike_times[ind_1][0]
		spike_2_list = spike_times[ind_2][0]
		
		#Calculate overlaps
		spike_1_overlaps, spike_2_overlaps = collision_func(spike_1_list,spike_2_list,blur_ind)
		
		col_perc_1 = np.round(100*spike_1_overlaps/len(spike_1_list),2)
		collision_percents[ind_1,ind_2] = col_perc_1
		col_perc_2 = np.round(100*spike_2_overlaps/len(spike_2_list),2)
		collision_percents[ind_2,ind_1] = col_perc_2
		if (col_perc_1 >= collision_cutoff) or (col_perc_2 >= collision_cutoff):
			collisions_detected += 1
			spikes_combined = [spike_1_list,spike_2_list]
			spikes_1_count = len(spike_1_list)
			spikes_2_count = len(spike_2_list)
			#Create a figure of the spike rasters together and save
			fig = plt.figure(figsize=(20,20))
			plt.subplot(2,2,1)
			spike_1_wavs = spike_wavs[ind_1][0]
			mean_bit = np.mean(spike_1_wavs,axis=0)
			std_bit = np.std(spike_1_wavs,axis=0)
			plt.plot(mean_bit,'-b',alpha = 1)
			plt.plot(mean_bit + std_bit,'-b',alpha = 0.5)
			plt.plot(mean_bit - std_bit,'-b',alpha = 0.5)
			plt.xlabel('Time (samples)')
			plt.ylabel('mV')
			plt.title('Unit ' + str(ind_1))
			plt.subplot(2,2,2)
			spike_2_wavs = spike_wavs[ind_2][0]
			mean_bit = np.mean(spike_2_wavs,axis=0)
			std_bit = np.std(spike_2_wavs,axis=0)
			plt.plot(mean_bit,'-b',alpha = 1)
			plt.plot(mean_bit + std_bit,'-b',alpha = 0.5)
			plt.plot(mean_bit - std_bit,'-b',alpha = 0.5)
			plt.xlabel('Time (samples)')
			plt.ylabel('mV')
			plt.title('Unit ' + str(ind_2))
			plt.subplot(2,2,3)
			plt.eventplot(spikes_combined,colors=colorCodes)
			plt.ylabel('Neuron')
			plt.xlabel('Spike')
			#Final title stuff
			line_1 = 'Unit ' + str(ind_1) + ' vs. Unit ' + str(ind_2)
			line_2 = 'Collision Percents = ' + str(col_perc_1) + ' and ' + str(col_perc_2)
			line_3 = 'U' + str(ind_1) + ' counts = ' + str(spikes_1_count)
			line_4 = 'U' + str(ind_2) + ' counts = ' + str(spikes_2_count)
			plt.suptitle(line_1 + '\n' + line_2 + '\n' + line_3 + '\n' + line_4,fontsize=20)
			plt.savefig(dir_save + 'unit_' + str(ind_1) + '_v_unit_' + str(ind_2) + '.png',dpi=100)
			plt.close(fig)
	
	#Create figure of collision percents and store
	fig = plt.figure(figsize=(20,20))
	plt.imshow(collision_percents)
	plt.colorbar()
	plt.title('Collision Percents for All Pairs')
	plt.savefig(dir_save + 'collision_percents.png',dpi=100)
	plt.close(fig)
	
	if collisions_detected > 0:
		print('\n INPUT REQUESTED: Collision plots have been made and stored in ' + dir_save)
		remove_ind = input('Please provide the indices you would like to remove (comma-separated ex. 0,1,2): ').split(',')
		remove_ind = [int(remove_ind[i]) for i in range(len(remove_ind))]
	else:
		remove_ind = []

	return remove_ind

def combine_spikes(sorted_spike_inds, sorted_spike_wavs, num_neur_sort):
	"""Function to recombine over-split spikes following sorting
	INPUTS:
		- sorted_spike_inds: list of lists with the spike time index of each
			waveform for a grouping
		- sorted_spike_wavs: list of lists with each list containing the 
			waveforms for a grouping
		- num_neur_sort: number of sorted waveform groupings
	OUTPUTS:
		- new_sorted_spike_inds: new list of lists with re-grouped spike indices
		- new_sorted_spike_wavs: new list of lists with re-grouped spike waveforms
	
	
	________
	WARNING: NOT READY YET! NEEDS MORE WORK!
	________
	"""
	#First find average waveform of each grouping
	mean_waveforms = []
	for n_i in range(num_neur_sort):
		all_wav = sorted_spike_wavs[n_i]
		mean_wav = np.mean(np.array(all_wav),0)
		mean_waveforms.append(list(mean_wav))
	#Next calculate the distance between all pairs of waveforms
	distances = []
	for n_i in range(num_neur_sort):
		wav_i = np.array(mean_waveforms[n_i])
		dist_to_i = []
		for n_j in range(num_neur_sort - 1):
			wav_j = np.array(mean_waveforms[n_j+1])
			dist = np.sqrt(np.sum((wav_i - wav_j)**2))
			dist_to_i.extend([float(dist)])
		distances.append(np.argmin(dist_to_i))
	
	#Now cluster
	clust_nums = np.arange(3,8)
	silh_scores = np.zeros(len(clust_nums))
	for clust_num in clust_nums:
		labels = KMeans(n_clusters=clust_num, random_state=np.random.randint(100)).fit_predict(distances[:,2])
		slh_vals = silhouette_samples(distances[:,2],labels)
		slh_avg = np.mean(slh_vals)
	clust_num = clust_nums[np.where(silh_scores == np.max(silh_scores))[0]][0]
	labels = KMeans(n_clusters=clust_num, random_state=np.random.randint(100)).fit_predict(distances[:,2])
	new_sorted_spike_inds = []
	new_sorted_spike_wavs = []
	for l_i in range(clust_num):
		ind_l = np.where(labels == l_i)[0]
		first_ind = np.array(distances)[ind_l,0]
		second_ind = np.array(distances)[ind_l,1]
		all_ind = np.unique(np.concatenate((first_ind,second_ind)))
		comb_ind = []
		comb_wav = []
		for c_i in all_ind:
			comb_ind.extend(sorted_spike_inds[c_i])
			comb_wav.extend(sorted_spike_wavs[c_i])
		new_sorted_spike_inds.append(comb_ind)
		new_sorted_spike_wavs.append(comb_wav)
	return new_sorted_spike_inds, new_sorted_spike_wavs


@jit(nogil = True)
def collision_func(spike_1_list,spike_2_list,blur_ind):
	"""Numba compiled function to compute overlaps for 2 neurons
	INPUTS:
		-spike_1_list: list of spike indices for neuron 1
		-spike_2_list: list of spike indices for neuron 2
		-blur_ind: number of indices within which to test for collisions
	OUTPUTS:
		-spike_1_overlaps: number of spikes fired by neuron 1 overlapping with neuron 2 spikes
		-spike_2_overlaps: number of spikes fired by neuron 2 overlapping with neuron 1 spikes
	"""
	spike_1_overlaps = 0
	spike_2_overlaps = 0
	for s_1 in spike_1_list:
		for s_2 in spike_2_list:
			if abs(s_1 - s_2) <= blur_ind:
				spike_1_overlaps += 1
				spike_2_overlaps += 1
		
	return spike_1_overlaps, spike_2_overlaps

def save_sort_hdf5():
	"""This function takes the final sort results after collision tests and 
	creates a new .h5 file with each neuron separately stored and re-numbered.
	Any additional necessary data is also stored in this file for use in analyses.
	Formatting:
		- sorted_units: folder contains unit_xxx (ex. unit_000, unit_001, etc...)
			sub_folders with times and waveforms arrays
		- raw_emg: folder with raw emg data stored in matrix
		- digital_in: folder contains arrays dig_in_x (ex. dig_in_0, dig_in_1, ...)
			of binary vectors containing times of dig in delivery as well as array
			dig_in_names containing the names of each dig_in
		- sampling_rate: array with single value of sampling rate
		- sort_settings: array with settings used in sorting the data
	INPUTS:
		- 
	"""
	
	
	
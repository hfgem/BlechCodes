#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 05:50:05 2022

@author: hannahgermaine
This set of functions pulls spikes out of cleaned data.
"""

import numpy as np
import functions.spike_clust as sc 
import functions.spike_template as st
from scipy.signal import find_peaks
import tables, tqdm, os, csv, time, sys


def run_spike_sort(data_dir):
	"""This function performs clustering spike sorting to separate out spikes
	from the dataset"""
	
	#Import data
	hf5 = tables.open_file(data_dir, 'r', title = data_dir[-1])
	data = hf5.root.clean_data[0,:,:]
	num_units, num_time = np.shape(data)
	sampling_rate = hf5.root.sampling_rate[0]
	segment_times = hf5.root.segment_times[:]
	segment_names = [hf5.root.segment_names[i].decode('UTF-8') for i in range(len(hf5.root.segment_names))]
	#Need to pull the times of different data segments to improve plotting
	hf5.close()
	del hf5
	downsamp_dir = ('_').join(data_dir.split('_')[:-1])+'_downsampled.h5'
	#Import downsampled dig-in data
	hf5 = tables.open_file(downsamp_dir, 'r', title = downsamp_dir[-1])
	dig_ins = hf5.root.dig_ins.dig_ins[0]
	dig_in_names = [hf5.root.dig_ins.dig_in_names[i].decode('UTF-8') for i in range(len(hf5.root.dig_ins.dig_in_names))]
	hf5.close()
	
	#Create directory for sorted data
	dir_save = ('/').join(data_dir.split('/')[:-1]) + '/sort_results/'
	if os.path.isdir(dir_save) == False:
		os.mkdir(dir_save)
	#Create .h5 file for storage of results
	sort_hf5_name = dir_save.split('/')[-3].split('.')[0].split('_')[0] + '_sort.h5'
	sort_hf5_dir = dir_save + sort_hf5_name
	if os.path.isfile(sort_hf5_dir) == False:
		sort_hf5 = tables.open_file(sort_hf5_dir, 'w', title = sort_hf5_dir[-1])
		sort_hf5.create_group('/','sorted_units')
		sort_hf5.close()
	
	#Perform sorting
	spike_sort(data,sampling_rate,dir_save,segment_times,
											 segment_names,dig_ins,
											 dig_in_names,sort_hf5_dir)
	del data
	
# 	#Perform compilation of all sorted spikes into a binary matrix
# 	separated_spikes_ind, sort_stats = import_sorted(dir_save,sort_hf5_dir)
# 	
# 	#Perform collision tests of imported data and remove until no collisions reported
# 	print("Now performing collision tests until no significant collisions are reported.")
# 	collision_loop = 1
# 	while collision_loop == 1:
# 		remove_ind = test_collisions(separated_spikes_ind,dir_save)
# 		if len(remove_ind) > 0:
# 			remove_ind.sort(reverse=True)
# 			for r_i in remove_ind:
# 				try:
# 					#First clean the spikes list
# 					del separated_spikes_ind[r_i]
# 				except:
# 					print("Removal of index skipped - index out of range.")
# 				try:
# 					#Second clean the sort statistics
# 					del sort_stats[r_i]
# 				except:
# 					print("Removal of index skipped - index out of range.")
# 		else:
# 			collision_loop = 0
# 			
# 	#Resave binary spikes as a numpy array
# 	print("Reshaping data for storage.")
# 	num_final_neur = len(separated_spikes_bin)
# 	num_time = len(separated_spikes_bin[0])
# 	spike_raster = np.zeros((num_final_neur,num_time))
# 	for n_i in tqdm.tqdm(range(num_final_neur)):
# 		spike_raster[n_i,:] = separated_spikes_bin[n_i]
# 	
# 	#Save spikes to an HDF5 file
# 	final_h5_dir = ('_').join(data_dir.split('_')[:-1])+'_sorted_results.h5'
# 	h5.save_sorted_spikes(final_h5_dir,spike_raster,sort_stats,sampling_rate,
# 					   segment_times,segment_names,dig_ins,dig_in_names)
	
	print('\n DONE SPIKE SORTING!')

def potential_spike_times(data,sampling_rate,dir_save,peak_thresh):
	"""Function to grab potential spike times for further analysis. Peaks 
	outside 1 absolute deviation and 1 ms to the left, and 1.5 ms to the right 
	around them are kept, while the rest are scrubbed.
	INPUTS:
		- data = one channel's worth of data (vector)
		- sampling_rate = smapling rate of data
		- dir_save = channel's save folder
		- peak_thresh = standard deviation threshold"""
	
	get_ind = 'n'
	init_times_csv = dir_save + 'init_times.csv'
	if os.path.isfile(init_times_csv) == False:
		get_ind = 'y'
		if os.path.isdir(dir_save) == False:
			os.mkdir(dir_save)
	else:
		print('\t Initial spike times previously pulled.')
		
	if get_ind == 'y':	
		#Grab mean and std
		std_dev = np.std(data)
		print("Searching for potential spike indices")
		peak_ind = find_peaks(-1*data,height=peak_thresh*std_dev)[0]
		#Save results to .csv
		with open(init_times_csv, 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([peak_ind])
	else:
		print('\t Importing spike times.')
		with open(init_times_csv, newline='') as f:
			reader = csv.reader(f)
			peak_ind_csv = list(reader)
		str_list = peak_ind_csv[0]
		peak_ind = [int(str_list[i]) for i in range(len(str_list))]
	
	return peak_ind

def spike_sort(data,sampling_rate,dir_save,segment_times,segment_names,
			   dig_ins,dig_in_names,sort_hf5_dir):
	"""This function performs clustering spike sorting to separate out spikes
	from the dataset
	INPUTS:
		-data = array of num_neur x num_time size containing the cleaned data
		-sampling_rate = integer of the sampling rate (Hz)
		-dir_save = where to save outputs
		-segment_times = times of different segments in the data
		-segment_names = names of the different segments
		-dig_ins = array of num_dig x num_time with 1s wherever a tastant 
					was being delivered
		-dig_in_names = array of names of each dig in used
		-sort_hf5_dir = hf5 file directory where the sort results should be saved
	"""	
	
	#Grab relevant parameters
	num_neur, num_time = np.shape(data)
	#min_dist_btwn_peaks = 3 #in sampling rate steps
	ms_left = 1
	ms_right = 1.5
	num_pts_left = int(np.round(sampling_rate*(ms_left/1000)))
	num_pts_right = int(np.round(sampling_rate*(ms_right/1000)))
	step_size = (ms_left + ms_right)/(num_pts_left+num_pts_right)
	axis_labels = np.round(np.arange(-ms_left,ms_right,step_size),2)
	axis_labels[num_pts_left] = 0
	#total_pts = num_pts_left + num_pts_right
	peak_thresh = 2 #Standard deviations from mean to cut off for peak finding
	threshold_percentile = 30 #Percentile to cut off in template-matching
	viol_1_percent = 0.5 #1 ms violations percent cutoff
	viol_2_percent = 2 #2 ms violations percent cutoff
	noise_clust = 6 #Set number of clusters for initial noise clustering
	clust_min = 4 #Minimum number of clusters to test in automated clustering
	clust_max = 8 #Maximum number of clusters to test + 1
	pre_clust = 1 #Whether to do an initial clustering step (1) or not (0)
	user_input = 0 #Whether a user manually selects final clusters and combines them
	#Ask for user input on type of clustering to perform
	clust_type, wav_type, comb_type, autosort = sort_settings(dir_save)
	
	#Grab dig in times for each tastant separately - grabs last index of delivery
	dig_times = [list(np.where(dig_ins[i] > 0)[0]) for i in range(len(dig_in_names))]
	dig_diff = [list(np.where(np.diff(dig_times[i])>1)[0] - 1) for i in range(len(dig_in_names))]
	dig_in_times = []
	for i in range(len(dig_in_names)):
		dig_in_vals = dig_diff[i]
		dig_in_vals.extend([len(dig_times)])
		dig_in_ind = list(np.array(dig_times[i])[dig_in_vals])
		dig_in_times.append(dig_in_ind)
	start_dig_diff = [list(np.where(np.diff(dig_times[i])>1)[0] + 1) for i in range(len(dig_in_names))]
	start_dig_in_times = []
	for i in range(len(dig_in_names)):
		dig_in_vals = [0]
		dig_in_vals.extend(start_dig_diff[i])
		dig_in_ind = list(np.array(dig_times[i])[dig_in_vals])
		start_dig_in_times.append(dig_in_ind)
	#number of samples tastant delivery length
	dig_in_lens = np.mean((np.array(dig_in_times) - np.array(start_dig_in_times))[:,2:-2],1)
	
	#Create .csv file name for storage of completed units
	sorted_units_csv = dir_save + 'sorted_units.csv'
	
	
	#First check if all units had previously been sorted
	prev_sorted = 0
	if os.path.isfile(sorted_units_csv) == True:
		with open(sorted_units_csv, 'r') as f:
			reader = csv.reader(f)
			sorted_units_list = list(reader)
			sorted_units_ind = [int(sorted_units_list[i][0]) for i in range(len(sorted_units_list))]
		sorted_units_unique = np.unique(sorted_units_ind)
		diff_units = np.setdiff1d(np.arange(num_neur),sorted_units_unique)
		if len(diff_units) == 0:
			prev_sorted = 1
	keep_final = 0
	if prev_sorted == 1:
		sort_loop = 1
		while sort_loop == 1:
			print('This data has been completely sorted before.')
			resort_channel = input("INPUT REQUESTED: Would you like to re-sort [y/n]? ")
			if resort_channel != 'y' and resort_channel != 'n':
				print("\t Incorrect entry.")
			elif resort_channel == 'n':
				keep_final = 1
				sort_loop = 0
			elif resort_channel == 'y':
				sort_loop = 0
	
	#Pull spikes from data	
	if keep_final == 0:
		
		#Get the number of clusters to use in spike sorting
		print("Beginning Spike Sorting")
		
		for i in tqdm.tqdm(range(num_neur)):
			print("\n Sorting channel #" + str(i))
			#First check for final sort and ask if want to keep
			keep_final = 0
			unit_dir = dir_save + 'unit_' + str(i) + '/'
			continue_no_resort = 0
			
			if os.path.isfile(sorted_units_csv) == True:
				with open(sorted_units_csv, 'r') as f:
					reader = csv.reader(f)
					sorted_units_list = list(reader)
					sorted_units_ind = [int(sorted_units_list[i][0]) for i in range(len(sorted_units_list))]
				try:
					in_list = sorted_units_ind.index(i)
				except:
					in_list = -1
				if in_list >= 0:
					print("\t Channel previously sorted.")
					sort_loop = 1
					while sort_loop == 1:
						resort_channel = input("\t INPUT REQUESTED: Would you like to re-sort [y/n]? ")
						if resort_channel != 'y' and resort_channel != 'n':
							print("\t Incorrect entry.")
						elif resort_channel == 'n':
							keep_final = 1
							sort_loop = 0
							continue_no_resort = 1
						elif resort_channel == 'y':
							sort_loop = 0
					del sort_loop, resort_channel
				del in_list, reader, sorted_units_list, sorted_units_ind
			tic = time.time()
			if keep_final == 0:
				#If no final sort or don't want to keep, then run through protocol
				data_copy = np.array(data[i,:])
				#Grab peaks
				peak_ind = potential_spike_times(data_copy,sampling_rate,unit_dir,peak_thresh)
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
				del p_i_l, p_i_r, data_chunk_lengths, too_short, keep_ind, data_copy
				if pre_clust == 1:
					#Cluster all spikes first to get rid of noise
					print("\n \t Performing Clustering to Remove Noise (First Pass)")
					sorted_peak_ind, waveform_ind  = sc.cluster(all_spikes, all_peaks, i, 
												 dir_save, axis_labels, 'noise_removal',
												 segment_times, segment_names, dig_in_lens, dig_in_times,
												 dig_in_names, sampling_rate, viol_1_percent,
												 viol_2_percent, noise_clust, clust_min, clust_max,
												 clust_type, wav_type, user_input)
					good_spikes = []
					good_ind = [] #List of lists with good indices in groupings
					good_all_spikes_ind = [] #indices aligned with "all_spikes"
					print("\t Performing Template Matching to Further Clean")
					#FUTURE IMPROVEMENT NOTE: Add csv storage of indices for further speediness if re-processing in future
					for g_i in range(len(sorted_peak_ind)):
						print("\t Template Matching Sorted Group " + str(g_i))
						s_i = sorted_peak_ind[g_i] #Original indices
						p_i = waveform_ind[g_i]
						sort_spikes = np.array(all_spikes)[p_i]
						g_spikes, g_ind = st.spike_template_sort(sort_spikes,sampling_rate,
											   num_pts_left,num_pts_right,
											   threshold_percentile,unit_dir,g_i)
						if comb_type == 'sep':
							#If separately clustering 
							good_spikes.extend(g_spikes) #Store the good spike profiles
							s_ind = [list(np.array(s_i)[g_ind[g_ii]]) for g_ii in range(len(g_ind))]
							p_ind = [list(np.array(p_i)[g_ind[g_ii]]) for g_ii in range(len(g_ind))]
						else:
							#If combining template results before final clustering
							g_spikes_comb = []
							[g_spikes_comb.extend(list(g_s)) for g_s in g_spikes]
							g_spikes_comb = np.array(g_spikes_comb)
							good_spikes.extend([g_spikes_comb]) #Store the good spike profiles
							s_ind = []
							for g_ii in range(len(g_ind)):
								s_ind.extend(list(np.array(s_i)[g_ind[g_ii]]))
							del g_ii
							p_ind = []
							for g_ii in range(len(g_ind)):
								p_ind.extend(list(np.array(p_i)[g_ind[g_ii]]))
							del g_ii
						good_ind.extend([s_ind]) #Store the original indices
						good_all_spikes_ind.extend([p_ind])
					del g_i, s_i, p_i, sort_spikes, g_spikes, g_ind, s_ind, p_ind
				else:
					print("\n \t Performing Template Matching To Sort Data")
					good_spikes = []
					good_ind = [] #List of lists with good indices in groupings
					s_i = all_peaks
					p_i = list(np.arange(len(all_peaks)))
					good_all_spikes_ind = [] #indices aligned with "all_spikes"
					for g_i in range(6):
						print("\t Template Matching Sorted Group " + str(g_i))
						sort_spikes = np.array(all_spikes)[p_i]
						g_spikes, g_ind = st.spike_template_sort(sort_spikes,sampling_rate,
											   num_pts_left,num_pts_right,
											   threshold_percentile,unit_dir,g_i)
						if comb_type == 'sep':
							#If separately clustering 
							good_spikes.extend(g_spikes) #Store the good spike profiles
							s_ind = [list(np.array(s_i)[g_ind[g_ii]]) for g_ii in range(len(g_ind))]
							p_ind = [list(np.array(p_i)[g_ind[g_ii]]) for g_ii in range(len(g_ind))]
						else:
							#If combining template results before final clustering
							g_spikes_comb = []
							[g_spikes_comb.extend(list(g_s)) for g_s in g_spikes]
							g_spikes_comb = np.array(g_spikes_comb)
							good_spikes.extend([g_spikes_comb]) #Store the good spike profiles
							s_ind = []
							for g_ii in range(len(g_ind)):
								s_ind.extend(list(np.array(s_i)[g_ind[g_ii]]))
							del g_ii
							p_ind = []
							for g_ii in range(len(g_ind)):
								p_ind.extend(list(np.array(p_i)[g_ind[g_ii]]))
							del g_ii
						s_i = list(np.setdiff1d(s_i,np.array(s_ind).flatten()))
						p_i = list(np.setdiff1d(p_i,np.array(p_ind).flatten()))
						good_ind.extend([s_ind]) #Store the original indices
						good_all_spikes_ind.extend([p_ind])
				print("\t Performing Clustering of Remaining Waveforms (Second Pass)")
				sorted_spike_inds = [] #grouped indices of spike clusters
				sorted_wav_inds = [] #grouped indices of spike waveforms from "all_spikes"
				for g_i in range(len(good_ind)): #Run through each set of potential clusters and perform cleanup clustering
					print("\t Sorting Template Matched Group " + str(g_i))
					sort_ind_2, waveform_ind_2  = sc.cluster(good_spikes[g_i], good_ind[g_i], i, 
												 dir_save, axis_labels, 'final/unit_' + str(g_i),
												 segment_times, segment_names, dig_in_lens, dig_in_times,
												 dig_in_names, sampling_rate, viol_1_percent,
												 viol_2_percent, noise_clust, clust_min, clust_max, 
												 clust_type, wav_type, user_input)
					good_as_ind = good_all_spikes_ind[g_i]
					sorted_spike_inds.extend(sort_ind_2)
					for w_i in range(len(waveform_ind_2)):
						sorted_wav_inds.append(list(np.array(good_as_ind)[waveform_ind_2[w_i]]))
				#Save sorted spike waveforms
				num_neur_sort = len(sorted_spike_inds)
				sorted_spike_wavs = []
				for g_i in range(num_neur_sort):
					s_i = sorted_wav_inds[g_i]
					spikes_i = [list(all_spikes[s_ii]) for s_ii in s_i]
					sorted_spike_wavs.append(list(spikes_i))
				#IN PROGRESS COMPONENT:
				#________________
				#Recombine spikes that belong together
				#new_sorted_spike_inds, new_sorted_spike_wavs = combine_spikes(sorted_spike_inds, sorted_spike_wavs, num_neur_sort)
				#________________
				#Save sort results
				if num_neur_sort > 0:
					#Save results
					print("\t Saving final results to .h5 file.")
					sort_hf5 = tables.open_file(sort_hf5_dir, 'r+', title = sort_hf5_dir[-1])
					existing_nodes = [int(i.__str__().split('_')[-1].split(' ')[0]) for i in sort_hf5.list_nodes('/sorted_units',classname='Group')]
					try:
						existing_nodes.index(i)
						already_stored = 1
					except:
						already_stored = 0
					if already_stored == 1:
						#Remove the existing node to be able to save anew
						exec('sort_hf5.root.sorted_units.unit_'+str(i)+'._f_remove(recursive=True,force=True)')
					atom = tables.FloatAtom()
					u_int = str(i)
					sort_hf5.create_group('/sorted_units', f'unit_{u_int}')
					sort_hf5.create_group(f'/sorted_units/unit_{u_int}','waveforms')
					sort_hf5.create_group(f'/sorted_units/unit_{u_int}','times')
					for s_w in range(len(sorted_spike_wavs)):				
						sort_hf5.create_earray(f'/sorted_units/unit_{u_int}/waveforms','neuron_' + str(s_w),atom,(0,)+np.shape(sorted_spike_wavs[s_w]))
						sort_hf5.create_earray(f'/sorted_units/unit_{u_int}/times','neuron_' + str(s_w),atom,(0,)+np.shape(sorted_spike_inds[s_w]))
						spike_wavs_expanded = np.expand_dims(sorted_spike_wavs[s_w],0)
						exec('sort_hf5.root.sorted_units.unit_'+str(i)+'.waveforms.neuron_'+str(s_w)+'.append(spike_wavs_expanded)')
						spike_inds_expanded = np.expand_dims(sorted_spike_inds[s_w],0)
						exec('sort_hf5.root.sorted_units.unit_'+str(i)+'.times.neuron_'+str(s_w)+'.append(spike_inds_expanded)')
					sort_hf5.close()
					del already_stored, atom, u_int, s_w, spike_wavs_expanded, spike_inds_expanded
					#Save unit index to sort csv
					if os.path.isfile(sorted_units_csv) == False:
						with open(sorted_units_csv, 'w') as f:
							write = csv.writer(f)
							write.writerows([[i]])
					else:
						with open(sorted_units_csv, 'a') as f:
							write = csv.writer(f)
							write.writerows([[i]])
					del write
				else:
					print("\t No neurons found.")
					#Save unit index to sort csv
					if os.path.isfile(sorted_units_csv) == False:
						with open(sorted_units_csv, 'w') as f:
							write = csv.writer(f)
							write.writerows([[i]])
					else:
						with open(sorted_units_csv, 'a') as f:
							write = csv.writer(f)
							write.writerows([[i]])
					del write
			toc = time.time()
			print(" Time to sort channel " + str(i) + " = " + str(round((toc - tic)/60)) + " minutes")	
			if (i < num_neur - 1) & (autosort == 0):
				if continue_no_resort == 0:
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

def sort_settings(dir_save):
	"""Function to prompt the user for settings to use in 
	sorting / re-load previously selected settings
	Inputs:
		- dir_save: for storage and upload of settings
	Outputs:
		- clust_type = the type of clustering algorithm to use: 'gmm' or 'kmeans'
		- wav_type = the type of waveform to use in clustering: 'full' or 'red'
		- comb_type = how to pass template-matching results to clustering: 'comb' or 'sep'
	"""
	
	#Check if settings were previously saved
	sort_settings_csv = dir_save + 'sort_settings.csv'
	file_exists = 0
	keep_file = 0
	if os.path.isfile(sort_settings_csv) == True:
		file_exists = 1
		keep_loop = 1
		while keep_loop == 1:
			print("\n Sort settings for clustering type, waveform type, and post-template-matching handling already exist.")
			keep_val = input("\n INPUT REQUESTED: Would you like to re-use the same settings [y,n]? ")
			if keep_val != 'y' and keep_val != 'n':
				print("Incorrect entry. Try again.")
				keep_loop = 1
			elif keep_val == 'y':
				keep_file = 1
				keep_loop = 0
			elif keep_val == 'n':
				keep_file = 0
				keep_loop = 0
	
	if file_exists*keep_file == 1:
		#Import the existing settings
		with open(sort_settings_csv,newline='') as f:
			reader = csv.reader(f)
			sort_settings_list = list(reader)
		clust_type = sort_settings_list[0][0]
		wav_type = sort_settings_list[1][0]
		comb_type = sort_settings_list[2][0]
	else:
		#Ask for user input on which clustering algorithm to use
		clust_loop = 1
		while clust_loop == 1:
			print('\n \n Clustering can be performed with GMMs or KMeans. Which algorithm would you like to use?')
			try:
				clust_type = int(input("INPUT REQUESTED: Enter 1 for gmm, 2 for kmeans: "))
				if clust_type != 1 and clust_type != 2:
					print("\t Incorrect entry.")
				elif clust_type == 1:
					clust_type = 'gmm'
					clust_loop = 0
				elif clust_type == 2:
					clust_type = 'kmeans'
					clust_loop = 0
			except:
				print("Error. Try again.")
		#Ask for user input on what data to cluster
		#'full' = full waveform, 'red' = reduced waveform
		wav_loop = 1
		while wav_loop == 1:
			print('\n Clustering can be performed on full waveforms or reduced via PCA. Which would you like to use?')
			try:
				wav_num = int(input("INPUT REQUESTED: Enter 1 for full waveform, 2 for PCA reduced: "))
				if wav_num != 1 and wav_num != 2:
					print("\t Incorrect entry.")
				elif wav_num == 1:
					wav_type = 'full'
					wav_loop = 0
				elif wav_num == 2:
					wav_type = 'red'
					wav_loop = 0
			except:
				print("Error. Try again.")
		#Ask for user input on whether to recombine template results
		#'comb' = combined, 'sep' = separate
		comb_loop = 1
		while comb_loop == 1:
			print('\n Template matching results can be recombined or kept separate. Which would you like to use?')
			try:
				comb_num = int(input("INPUT REQUESTED: Enter 1 for combined, 2 for separate: "))
				if comb_num != 1 and comb_num != 2:
					print("\t Incorrect entry.")
				elif comb_num == 1:
					comb_type = 'comb'
					comb_loop = 0
				elif comb_num == 2:
					comb_type = 'sep'
					comb_loop = 0
			except:
				print("Error. Try again.")
			
		#Ask for user input on whether to autosort or ask for the user to 
		#approve whether to move on to the next neuron" 1 = auto, 0 = manual
		autosort = 0
		auto_loop = 1
		while auto_loop == 1:
			print('\n Sorting can automatically cycle through all neurons or require manual input to continue. Which would you prefer?')
			try:
				autosort = int(input("INPUT REQUESTED: Enter 1 for automatically continuing, 0 for manually continuing: "))
				if autosort != 1 and autosort != 2:
					print("\t Incorrect entry.")
				else:
					auto_loop = 0
			except:
				print("Error. Try again.")
		
		result_vals = [clust_type, wav_type, comb_type, autosort]
		
		for i in result_vals:
			is_file = os.path.isfile(sort_settings_csv)
			if is_file == False:
				with open(sort_settings_csv, 'w') as f:
					write = csv.writer(f)
					write.writerows([[i]])
			else:
				with open(sort_settings_csv, 'a') as f:
					write = csv.writer(f)
					write.writerows([[i]])
	
	return clust_type, wav_type, comb_type, autosort


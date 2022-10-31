#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 05:50:05 2022

@author: hannahgermaine
This set of functions pulls spikes out of cleaned and ICA sorted data.
"""

import numpy as np
import scipy.stats as ss
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import rfft, fftfreq
import tables, tqdm, os, csv, time, sys, itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import functions.hdf5_handling as h5


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
	
	#Perform compilation of all sorted spikes into a binary matrix
	separated_spikes_bin, sort_stats = import_sorted(num_units,dir_save,sort_hf5_dir)
	
	#Perform collision tests of imported data and remove until no collisions reported
	print("Now performing collision tests until no significant collisions are reported.")
	collision_loop = 1
	while collision_loop == 1:
		remove_ind = test_collisions(separated_spikes_bin,dir_save)
		if len(remove_ind) > 0:
			remove_ind.sort(reverse=True)
			for r_i in remove_ind:
				try:
					#First clean the spikes list
					del separated_spikes_bin[r_i]
				except:
					print("Removal of index skipped - index out of range.")
				try:
					#Second clean the sort statistics
					del sort_stats[r_i]
				except:
					print("Removal of index skipped - index out of range.")
		else:
			collision_loop = 0
			
	#Resave binary spikes as a numpy array
	print("Reshaping data for storage.")
	num_final_neur = len(separated_spikes_bin)
	num_time = len(separated_spikes_bin[0])
	spike_raster = np.zeros((num_final_neur,num_time))
	for n_i in tqdm.tqdm(range(num_final_neur)):
		spike_raster[n_i,:] = separated_spikes_bin[n_i]
	
	#Save spikes to an HDF5 file
	final_h5_dir = ('_').join(data_dir.split('_')[:-1])+'_sorted_results.h5'
	h5.save_sorted_spikes(final_h5_dir,spike_raster,sort_stats,sampling_rate,
					   segment_times,segment_names,dig_ins,dig_in_names)
	
	print('\n DONE SPIKE SORTING!')

def potential_spike_times(data,sampling_rate,dir_save):
	"""Function to grab potential spike times for further analysis. Peaks 
	outside 1 absolute deviation and 1 ms to the left, and 1.5 ms to the right 
	around them are kept, while the rest are scrubbed.
	INPUTS:
		- data = one channel's worth of data (vector)
		- sampling_rate = smapling rate of data
		- dir_save = channel's save folder"""
	
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
		peak_ind = find_peaks(-1*data,height=1*std_dev)[0]
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
		-dig_in_names = array of names of each dig in used"""	
	
	#Grab relevant parameters
	num_neur, num_time = np.shape(data)
	#min_dist_btwn_peaks = 3 #in sampling rate steps
	num_pts_left = int(np.round(sampling_rate*(1/1000)))
	num_pts_right = int(np.round(sampling_rate*(1.5/1000)))
	axis_labels = np.arange(-num_pts_left,num_pts_right)
	#total_pts = num_pts_left + num_pts_right
	viol_1 = sampling_rate*(1/1000)
	viol_2 = sampling_rate*(2/1000)
	threshold_percentile = 25
	
	#Grab dig in times for each tastant separately
	dig_times = [list(np.where(dig_ins[i] > 0)[0]) for i in range(len(dig_in_names))]
	dig_diff = [list(np.where(np.diff(dig_times[i])>1)[0] + 1) for i in range(len(dig_in_names))]
	dig_in_times = []
	for i in range(len(dig_in_names)):
		dig_in_vals = [0]
		dig_in_vals.extend(dig_diff[i])
		dig_in_ind = list(np.array(dig_times[i])[dig_in_vals])
		dig_in_times.append(dig_in_ind)
	
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
		clust_num = 0
		clust_loop = 1
		while clust_loop == 1:
			print("\n INPUT REQUESTED: Think of the number of clusters you'd like to use for initial sorting (removal of noise).")
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
			print("\n INPUT REQUESTED: Think of the number of clusters you'd like to use for final sorting (after template-matching).")
			cluster_num_fin = input("Please enter the number you'd like to use (> 1): ")
			try:
				clust_num_fin = int(cluster_num_fin)
				clust_loop = 0
			except:
				print("ERROR: Please enter a valid integer.")
		del cluster_num_fin, clust_loop
		
		print("\n Now beginning spike sorting.")
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
			tic = time.time()
			if keep_final == 0:
				#If no final sort or don't want to keep, then run through protocol
				data_copy = np.array(data[i,:])
				#Grab peaks
				peak_ind = potential_spike_times(data_copy,sampling_rate,unit_dir)
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
				sorted_peak_ind, waveform_ind = spike_clust(all_spikes, all_peaks, 
											 clust_num, i, dir_save, axis_labels, 
											 viol_1, viol_2, 'noise_removal', segment_times,
											 segment_names, dig_in_times, dig_in_names,
											 sampling_rate, re_sort='y')
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
					g_spikes, g_ind = spike_template_sort(sort_spikes,sampling_rate,
										   num_pts_left,num_pts_right,
										   threshold_percentile,unit_dir,g_i)
					good_spikes.append(g_spikes) #Store the good spike profiles
					good_ind.append(list(np.array(s_i)[g_ind])) #Store the original indices
					good_all_spikes_ind.append(list(np.array(p_i)[g_ind]))
				del g_i, s_i
				print("\t Performing Clustering of Remaining Waveforms (Second Pass)")
				sorted_spike_inds = [] #grouped indices of spike clusters
				sorted_wav_inds = [] #grouped indices of spike waveforms from "all_spikes"
				for g_i in range(len(good_ind)): #Run through each set of potential clusters and perform cleanup clustering
					print("\t Sorting Template Matched Group " + str(g_i))
					sort_ind_2, waveform_ind_2 = spike_clust(good_spikes[g_i], good_ind[g_i], 
											 clust_num_fin, i, dir_save, axis_labels, 
											 viol_1, viol_2, 'final_' + str(g_i), segment_times,
											 segment_names, dig_in_times, dig_in_names,
											 sampling_rate)
					good_as_ind = good_all_spikes_ind[g_i]
					sorted_spike_inds.extend(sort_ind_2)
					for w_i in range(len(waveform_ind_2)):
						sorted_wav_inds.append(list(np.array(good_as_ind)[waveform_ind_2[w_i]]))
				
				#Save sorted spike indices and profiles
				num_neur_sort = len(sorted_spike_inds)
				sorted_spike_wavs = []
				for g_i in range(num_neur_sort):
					s_i = sorted_wav_inds[g_i]
					spikes_i = [list(all_spikes[s_ii]) for s_ii in s_i]
					sorted_spike_wavs.append(list(spikes_i))
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
					#Save unit index to sort csv
					if os.path.isfile(sorted_units_csv) == False:
						with open(sorted_units_csv, 'w') as f:
							write = csv.writer(f)
							write.writerows([[i]])
					else:
						with open(sorted_units_csv, 'a') as f:
							write = csv.writer(f)
							write.writerows([[i]])
				else:
					print("\t No neurons found.")
					
			toc = time.time()
			print(" Time to sort channel " + str(i) + " = " + str(round((toc - tic)/60)) + " minutes")	
			if i < num_neur - 1:
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
		

def spike_clust(spikes, peak_indices, clust_num, i, sort_data_dir, axis_labels, 
				viol_1, viol_2, type_spike, segment_times, segment_names, 
				dig_in_times,dig_in_names,sampling_rate,re_sort='y'):
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
		dig_in_times = array times of tastant delivery - each tastant separately
		dig_in_names = names of different tastants
		sampling_rate = number of samples per second
		re_sort = whether to re-sort if data has been previously sorted.
	Outputs:
		neuron_spike_ind = indices of spikes selected as true - the indices
						reflect the parsed data indices.
		waveform_ind = indices of waveforms, aka index in the list 'spikes'"""
	
	#Create storage folder
	sort_neur_dir = sort_data_dir + 'unit_' + str(i) + '/'
	if os.path.isdir(sort_neur_dir) == False:
		os.mkdir(sort_neur_dir)
	sort_neur_type_dir = sort_neur_dir + type_spike + '/'
	if os.path.isdir(sort_neur_type_dir) == False:
		os.mkdir(sort_neur_type_dir)
	sort_neur_stats_csv = sort_neur_type_dir + 'sort_stats.csv'
	sort_neur_ind_csv = sort_neur_type_dir + 'neuron_spike_ind.csv'
	sort_neur_wav_csv = sort_neur_type_dir + 'neuron_spike_wav.csv'
	if os.path.isfile(sort_neur_ind_csv) == False:
		re_sort = 'y'
	else:
		if re_sort != 'n':
			#MODIFY TO SEARCH FOR A CSV OF SPIKE INDICES
			print('\t INPUT REQUESTED: The ' + type_spike + ' sorting has been previously performed.')
			re_sort = input('\t Would you like to re-sort [y/n]? ')
	
	if re_sort == 'y':
		#Set parameters
		viol_2_cutoff = 2 #Maximum allowed violation percentage for 2 ms
		viol_1_cutoff = 1 #Maximum allowed violation percentage for 1 ms
		num_vis = 500 #Number of waveforms to visualize for example plot
		all_dig_in_times = np.unique(np.array(dig_in_times).flatten())
		PSTH_left_ms = 500
		PSTH_right_ms = 2000
		center = np.where(axis_labels == 0)[0]
		
 		#Project data to lower dimensions
		print("\t Projecting data to lower dimensions")
		pca = PCA(n_components = 3)
		spikes_pca = pca.fit_transform(spikes)
		#Grab spike amplitude and energy of spike to add to spikes_pca
		amp_vals = np.array([np.abs(spikes[i][center]) for i in range(len(spikes))])
		energy_vals = np.expand_dims(np.sum(np.square(spikes),1),1)
		
		#Perform kmeans clustering on downsampled data
		print("\t Performing clustering of data.")
		rand_ind = np.random.randint(len(spikes),size=(100000,))
		rand_spikes = list(np.array(spikes)[rand_ind])
		print('\t Performing fitting.')
		kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(rand_spikes)
# 		gm = GaussianMixture(n_components=clust_num).fit(rand_spikes)
		print('\t Performing label prediction.')
		labels = kmeans.predict(spikes)
		centers = kmeans.cluster_centers_
		center_pca = pca.fit_transform(centers)
		#std_centers = []
		#for li in range(clust_num):
		#	ind_labelled = np.where(labels == li)[0]
		#	labelled_spikes_array = np.array(spikes)[ind_labelled]
		#	dist = np.sqrt(np.sum((labelled_spikes_array - np.multiply(np.ones(np.shape(labelled_spikes_array)),centers[li]))**2,1))
		#	std_centers.append(np.std(dist))
# 		labels = gm.predict(spikes)
		print('\t Now testing/plotting clusters.')
		violations = []
		any_good = 0
		possible_colors = ['b','g','r','c','m','k','y','brown','pink','olive','gray'] #Colors for plotting different tastant deliveries
		clust_stats = np.zeros((clust_num,5))
		#Create cluster projection plot
		clust_fig = plt.figure(figsize=(15,15))
		ax = clust_fig.add_subplot(111, projection='3d')
		ax2 = clust_fig.add_subplot(333)
		ax3 = clust_fig.add_subplot(331)
		ax4 = clust_fig.add_subplot(339, projection='3d')
		for li in range(clust_num):
			ind_labelled = np.where(labels == li)[0]
			pca_labelled = spikes_pca[ind_labelled]
			pca_labelled_2 = spikes_pca[np.where(labels == clust_num - li)[0]]
			#3D plot
			ax.scatter(pca_labelled[:,0],pca_labelled[:,1],pca_labelled[:,2],
				  c=possible_colors[li],label='cluster '+str(li))
			#2D plot top
			ax2.scatter(pca_labelled[:,0],pca_labelled[:,1],
			   c=possible_colors[li],label='cluster '+str(li),alpha=0.2)
			#2D plot bottom
			ax3.scatter(pca_labelled_2[:,0],pca_labelled_2[:,1],
			   c=possible_colors[clust_num - li],label='cluster '+str(clust_num - li),alpha=0.2)
			#Centroid 3D plot
			ax4.scatter(center_pca[li,0],center_pca[li,1],center_pca[li,2],
			   c = possible_colors[li])
		ax.legend(loc='lower left')
		clust_fig.savefig(sort_neur_type_dir + 'cluster_projections.png', dpi=100)
		plt.close(clust_fig)
		#Create waveform/histogram/PSTH plots
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
			avg_fr = round(len(peak_ind)/(segment_times[-1])*sampling_rate,2) #in Hz
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
					
					#Pull PSTH data by tastant
					PSTH_left = -1*round((1/1000)*sampling_rate*PSTH_left_ms)
					PSTH_right = round((1/1000)*sampling_rate*PSTH_right_ms)
					PSTH_width = PSTH_right - PSTH_left
					PSTH_avg_taste = []
					for t_i in range(len(dig_in_times)): #By tastant
						spike_raster = np.zeros((len(dig_in_times[t_i]),PSTH_width))
						for d_i in range(len(dig_in_times[t_i])): #By delivery
							d_time = dig_in_times[t_i][d_i] #Time of delivery
							spike_inds_left = np.where(peak_ind - d_time > PSTH_left)[0]
							spike_inds_right = np.where(peak_ind - d_time < PSTH_right)[0]
							overlap_inds = np.intersect1d(spike_inds_left,spike_inds_right)
							spike_inds_overlap = peak_ind[overlap_inds] - d_time + PSTH_left
							spike_raster[d_i][spike_inds_overlap] = 1
						#The following bin size and step size come from Sadacca et al. 2016 
						bin_ms = 250 #Number of ms for binning the spike counts
						bin_size = round((bin_ms/1000)*sampling_rate)
						bin_step_size_ms = 10 #Number of ms for sliding the bin over
						bin_step_size = round((bin_step_size_ms/1000)*sampling_rate)
						PSTH_x_labels = np.arange(-PSTH_left_ms,PSTH_right_ms,bin_step_size_ms)
						PSTH_mat = np.zeros((len(dig_in_times[t_i]),len(PSTH_x_labels)))
						for b_i in range(len(PSTH_x_labels)):
							PSTH_mat[:,b_i] = np.sum(spike_raster[:,b_i*bin_step_size:(b_i*bin_step_size)+bin_size],1)
						PSTH_avg = (np.mean(PSTH_mat,0)/bin_ms)*1000 #Converted to Hz
						PSTH_avg_taste.append(PSTH_avg)
					#CREATE FIGURE
					fig = plt.figure(figsize=(30,20))
					#Plot average waveform
					plt.subplot(3,2,1)
					spike_bit = spikes_labelled[np.random.randint(0,len(spikes_labelled),size=(10000,))]
					mean_bit = np.mean(spike_bit,axis=0)
					std_bit = np.std(spike_bit,axis=0)
					plt.errorbar(axis_labels,mean_bit,yerr=std_bit,xerr=None)
					plt.title('Cluster ' + str(li) + ' Average Waveform + Std Range')
					#Plot spike overlay
					plt.subplot(3,2,2)
					for si in plot_ind:
						try:
							plt.plot(axis_labels,spikes_labelled[si],'-b',alpha=0.1)
						except:
							print("\t \t Error: Skipped plotting a waveform.")
					plt.ylabel('mV')
					plt.title('Cluster ' + str(li) + ' x' + str(plot_num_vis) + ' Waveforms')
					#Find ISI distribution and plot
					plt.subplot(3,2,3)
					plt.hist(peak_diff[np.where(peak_diff < sampling_rate)[0]],bins=min(100,round(len(peak_diff)/10)))
					plt.title('Cluster ' + str(li) + ' ISI Distribution')
					#Histogram of time of spike occurrence
					plt.subplot(3,2,4)
					plt.hist(peak_ind,bins=min(100,round(len(peak_ind)/10)))
					[plt.axvline(segment_times[i],label=segment_names[i],alpha=0.2,c=possible_colors[i]) for i in range(len(segment_names))]
					plt.legend()
					plt.title('Cluster ' + str(li) + ' Spike Time Histogram')
					#Histogram of time of spike occurrence zoomed to taste delivery
					#plt.subplot(3,2,5)
					#plt.hist(peak_ind,bins=2*len(all_dig_in_times))
					#for d_i in range(len(dig_in_names)):
					#	[plt.axvline(dig_in_times[d_i][i],c=possible_colors[d_i],alpha=0.2, label=dig_in_names[d_i]) for i in range(len(dig_in_times[d_i]))]
					#plt.xlim((min(all_dig_in_times)- sampling_rate,max(all_dig_in_times) + sampling_rate))
					#plt.title('Cluster ' + str(li) + ' Spike Time Histogram - Taste Interval')
					#Fourier Transform of Average Waveform For Cluster
					fourier = rfft(mean_bit)
					freqs = fftfreq(len(mean_bit), d=1/sampling_rate)
					fourier_peaks = find_peaks(fourier)[0]
					peak_freqs = freqs[fourier_peaks]
					peak_freqs = peak_freqs[peak_freqs>0]
					plt.plot(freqs,fourier)
					for p_f in range(len(peak_freqs)):
						plt.axvline(peak_freqs[p_f],color=possible_colors[p_f],label=str(round(peak_freqs[p_f],2)))
					plt.xlim((0,max(freqs)))
					plt.legend()
					plt.xlabel('Frequency (Hz)')
					plt.title('Fourier Transform of Mean Waveform')
					#PSTH figure
					plt.subplot(3,2,6)
# 					for p_i in range(len(dig_in_times)): #Plot individual instances
# 						plt.plot(PSTH_x_labels,PSTH_mat[p_i,:],alpha=0.1)
					for d_i in range(len(dig_in_times)):
						plt.plot(PSTH_x_labels,PSTH_avg_taste[d_i],c=possible_colors[d_i],label=dig_in_names[d_i])
					plt.legend()
					plt.axvline(0,c='k')
					plt.xlabel('Milliseconds from delivery')
					plt.ylabel('Average firing rate (Hz)')
					plt.title('Cluster ' + str(li) + ' Average PSTH')
					#Title and save figure
					line_1 = 'Number of Waveforms = ' + str(len(spikes_labelled))
					line_2 = '1 ms violation percent = ' + str(viol_1_percent)
					line_3 = '2 ms violation percent = ' + str(viol_2_percent)
					line_4 = 'Average firing rate = ' + str(avg_fr)
					plt.suptitle(line_1 + '\n' + line_2 + ' ; ' + line_3 + '\n' + line_4,fontsize=24)
					fig.savefig(sort_neur_type_dir + 'waveforms_' + str(li) + '.png', dpi=100)
					plt.close(fig)
					clust_stats[li,0:4] = np.array([len(spikes_labelled),viol_1_percent,viol_2_percent,avg_fr])
		neuron_spike_ind = []
		neuron_waveform_ind = []
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
####THERE IS AN ERROR IN THIS TRY STATEMENT THAT NEEDS TO BE FIXED####
			try:
				ind_good = [int(ind_good[i]) for i in range(len(ind_good))]
				clust_stats[np.array(ind_good),4] = 1
				combine_spikes = 'n'
				comb_loop = 1
				while comb_loop == 1:
					if len(ind_good) > 1:
						combine_spikes = input("\t Do any of these spikes come from the same neuron (y/n)? ")
						if combine_spikes != 'y' and combine_spikes != 'n':
							print("\t Error, please enter a valid value.")
						else:
							#Find if there are any that need to be combined into 1
							comb_loop = 0
							which_comb_loop = 1
							which_comb = []
							if combine_spikes == 'y':
								 while which_comb_loop == 1:
									 which_together = input("\t Which indices belong together [comma separated list]? ").split(',')
									 try:
										  together_ind = [int(which_together[i]) for i in range(len(which_together))]
										  which_comb.append(together_ind)
										  cont_loop_2 = 1
										  while cont_loop_2 == 1:
											  continue_statement = input("\t Are there more indices which belong together (y/n)? ")
											  if continue_statement != 'y' and continue_statement != 'n':
												  print("\t Error, try again.")
											  elif continue_statement == 'y':
												  cont_loop_2 = 0
											  else:
												  cont_loop_2 = 0
												  which_comb_loop = 0
									 except:
										 print("Error, try again.")
								 all_to_combine = []
								 for c_i in range(len(which_comb)):
									 all_to_combine.extend(which_comb[c_i])
								 all_to_combine = np.array(all_to_combine)
								 not_to_combine = np.setdiff1d(np.array(ind_good),all_to_combine)
								 ind_good = []
								 if len(not_to_combine) > 0:
									  ind_good.append(list(not_to_combine))
								 for w_i in range(len(which_comb)):
									 ind_good.append(which_comb[w_i])
					else:
 						combine_spikes = 'y'
 						comb_loop = 0
				for ig in ind_good:
					if np.size(ig) > 1:
						peak_ind = []
						wav_ind = []
						for ind_g in ig:
							wav_ind.extend(list(np.where(labels == ind_g)[0]))
							peak_ind.extend(list(np.array(peak_indices)[np.where(labels == ind_g)[0]]))
						neuron_spike_ind.append(peak_ind)
						neuron_waveform_ind.append(wav_ind)
					else:
						wav_ind = []
						wav_ind.extend(list(np.where(labels == ig)[0]))
						neuron_waveform_ind.append(wav_ind)
						peak_ind = []
						peak_ind.extend(list(np.array(peak_indices)[np.where(labels == ig)[0]]))
						neuron_spike_ind.append(peak_ind)
			except:
				print("\t No spikes selected.")
		else:
			print("\t No good clusters.")
		#Save to CSV spike indices
		with open(sort_neur_ind_csv, 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows(neuron_spike_ind)
		#Save to CSV waveform indices (index within group)
		with open(sort_neur_wav_csv, 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows(neuron_waveform_ind)
		#Save stats to CSV
		with open(sort_neur_stats_csv, 'w') as f:
			write = csv.writer(f,delimiter=',')
			write.writerows([['Number of Spikes','1 ms Violations','2 ms Violations','Average Firing Rate','Good']])
			write.writerows(clust_stats)
	else:
		print("\t Importing previously sorted data.")
		with open(sort_neur_ind_csv, newline='') as f:
			reader = csv.reader(f)
			neuron_spike_ind_csv = list(reader)
		neuron_spike_ind = []
		for i_c in range(len(neuron_spike_ind_csv)):
			str_list = neuron_spike_ind_csv[i_c]
			int_list = [int(str_list[i]) for i in range(len(str_list))]
			neuron_spike_ind.append(int_list)
		with open(sort_neur_wav_csv, newline='') as f:
			reader = csv.reader(f)
			neuron_spike_wav_csv = list(reader)
		neuron_waveform_ind = []
		for i_c in range(len(neuron_spike_wav_csv)):
			str_list = neuron_spike_wav_csv[i_c]
			int_list = [int(str_list[i]) for i in range(len(str_list))]
			neuron_waveform_ind.append(int_list)
			
	return neuron_spike_ind, neuron_waveform_ind

@jit(forceobj=True)
def spike_template_sort(all_spikes,sampling_rate,num_pts_left,num_pts_right,
						cut_percentile,unit_dir,clust_ind):
	"""This function performs template-matching to pull out potential spikes.
	INPUTS:
		- all_spikes
		- sampling_rate
		- num_pts_left
		- num_pts_right
		- cut_percentile
		- unit_dir - directory of unit's storage data
		- clust_ind
	OUTPUTS:
		- potential_spikes
		- good_ind
	"""
	template_dir = unit_dir + 'template_matching/'
	if os.path.isdir(template_dir) == False:
		os.mkdir(template_dir)
	
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
	#Plot a histogram of the scores and save to the tampleate_matching dir
	fig = plt.figure(figsize=(20,20))
	for i in range(num_types):
		#Template correlation
		spike_mat = np.multiply(np.ones(np.shape(norm_spikes)),spike_templates[i,:])
		dist = np.sqrt(np.sum(np.square(np.subtract(norm_spikes,spike_mat)),1))
		num_peaks = [len(find_peaks(norm_spikes[s],0.5)[0]) + len(find_peaks(-1*norm_spikes[s],0.5)[0]) for s in range(len(norm_spikes))]
		score = dist*num_peaks
		percentile = np.percentile(score,cut_percentile)
		#Plot the score results and determined cutoff together
		plt.subplot(2,num_types,i + 1)
		hist_counts = plt.hist(score,100)
		hist_peaks = find_peaks(hist_counts[0])
		cutoff_val = min(hist_counts[1][hist_peaks[0][0]+1],percentile)
		plt.axvline(cutoff_val,color='r',linestyle='dashed')
		plt.axvline(percentile,color='g',linestyle='dashed')
		plt.xlabel('Score = distance*peak_count')
		plt.ylabel('Number of occurrences')
		plt.title('Scores in comparison to template #' + str(i))
		plt.subplot(2,num_types,i + 1 + num_types)
		plt.plot(spike_templates[i,:])
		plt.title('Template #' + str(i))
		#good_i = np.where(score < percentile)[0]
		good_i = np.where(score < cutoff_val)[0]
		good_ind.extend(list(good_i))
	fig.savefig(template_dir + 'template_matching_results_cluster' + str(clust_ind) + '.png',dpi=100)
	plt.close(fig)
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

def import_sorted(num_units,dir_save,sort_hf5_dir):
	"""This function imports the already sorted data into arrays + the sorting
	statistics"""
	
	sort_hf5 = tables.open_file(sort_hf5_dir, 'r', title = sort_hf5_dir[-1])
	separated_spikes_bin = []
	print("\t Importing sorted data.")
	for i in tqdm.tqdm(range(num_units)):
		neuron_spikes = sort_hf5.get_node('/sorted_spikes_bin/unit_'+str(i))
		neuron_spikes_bin = neuron_spikes[0]
		del neuron_spikes
		for j in range(len(neuron_spikes_bin)):
			separated_spikes_bin.append(neuron_spikes_bin[j])
		del neuron_spikes_bin
	sort_hf5.close()
	
	sort_stats = []
	for i in tqdm.tqdm(range(num_units)):
		final_sort_neur_dir = dir_save + 'unit_' + str(i) + '/final/'
		sort_stats_csv = final_sort_neur_dir + 'sort_stats.csv'
		#Import sort statistics
		with open(sort_stats_csv,newline='') as f:
			reader = csv.reader(f)
			sort_stats_list = list(reader)
		for i_s in range(len(sort_stats_list) - 1):
			stat_row = sort_stats_list[i_s + 1]
			stat_row_float = [i,i_s+1]
			stat_row_float.extend([float(stat_row[i]) for i in range(len(stat_row) - 1)])
			if float(stat_row[-1]) == 1:
				sort_stats.append(stat_row_float)
	sort_stats = np.array(sort_stats)
	
	return separated_spikes_bin, sort_stats

def test_collisions(spike_raster,dir_save):
	"""This function tests the final selected neurons for collisions across 
	all units. It performs pairwise tests and looks for spike times within 3 
	time bins, totalling the number of overlaps / average number of spikes 
	between the two neurons. If the percentage is over 50, the pair is flagged 
	and the user can determine which to remove based on the statistics.
	INPUTS:
		- spike_raster = list of numpy arrays, where each array is the binary 
						spiking of a neuron
		- dir_save = directory to store collision results
	"""
	num_neur = len(spike_raster)
	all_pairs = list(itertools.combinations(np.arange(0,num_neur),2))
	blur_ind = 3
	collision_cutoff = 50 #Percent cutoff for collisions
	colorCodes = np.array([[0,1,0],[0,0,1]]) #Colors for plotting collision rasters
	
	collision_folder = dir_save + 'collisions/'
	if os.path.isdir(collision_folder) == False:
		os.mkdir(collision_folder)
	
	print("\t Testing all units pairwise.")
	collisions_detected = 0
	for i in tqdm.tqdm(range(len(all_pairs))):
		ind_1 = all_pairs[i][0]
		ind_2 = all_pairs[i][1]
		spikes_1 = spike_raster[ind_1]
		spikes_2 = spike_raster[ind_2]
		#Blur spike times to blur_ind bins in either direction
		spikes_1_blur = np.zeros(np.shape(spikes_1))
		spikes_1_blur += spikes_1
		spikes_2_blur = np.zeros(np.shape(spikes_2))
		spikes_2_blur += spikes_2
		num_spikes_1 = np.sum(spikes_1)
		num_spikes_2 = np.sum(spikes_2)
		avg_num_spikes = round((num_spikes_1 + num_spikes_2)/2)
		for i in range(blur_ind):
			spikes_1_blur[0:-1*(i+1)] += spikes_1[i+1:]
			spikes_1_blur[i+1:] += spikes_1[0:-1*(i+1)]
			spikes_2_blur[0:-1*(i+1)] += spikes_2[i+1:]
			spikes_2_blur[i+1:] += spikes_2[0:-1*(i+1)]
		#Multiply the vectors together to find overlaps
		collisions = np.multiply(spikes_1_blur,spikes_2_blur)
		collisions_count = len(np.where(np.diff(collisions) > 1)[0]) + 1
		collision_percent = round(100*collisions_count/avg_num_spikes)
		if collision_percent >= collision_cutoff:
			collisions_detected += 1
			spikes_combined = []
			spikes_combined.append(np.where(spikes_1 > 0)[0])
			spikes_combined.append(np.where(spikes_2 > 0)[0])
			spikes_1_count = np.sum(spikes_1)
			spikes_2_count = np.sum(spikes_2)
			#Create a figure of the spike rasters together and save
			fig = plt.figure(figsize=(20,20))
			plt.eventplot(spikes_combined,colors=colorCodes)
			line_1 = 'Unit ' + str(ind_1) + ' vs. Unit ' + str(ind_2)
			line_2 = 'Collision Percent = ' + str(collision_percent)
			line_3 = 'U' + str(ind_1) + ' counts = ' + str(spikes_1_count)
			line_4 = 'U' + str(ind_2) + ' counts = ' + str(spikes_2_count)
			plt.ylabel('Neuron')
			plt.xlabel('Spike')
			plt.suptitle(line_1 + '\n' + line_2 + '\n' + line_3 + '\n' + line_4,fontsize=20)
			plt.savefig(collision_folder + 'unit_' + str(ind_1) + '_v_unit_' + str(ind_2) + '.png',dpi=100)
			plt.close(fig)
	
	if collisions_detected > 0:
		print('\n INPUT REQUESTED: Collision plots have been made and stored in ' + collision_folder)
		remove_ind = input('Please provide the indices you would like to remove (comma-separated ex. 0,1,2): ').split(',')
		remove_ind = [int(remove_ind[i]) for i in range(len(remove_ind))]
	else:
		remove_ind = []

	return remove_ind

def re_cluster():
	"""This function allows for re-clustering of previously clustered data by 
	importing the final-clustering .csv results ('neuron_spikes.csv' and 
	'neuron_spikes_bin.csv'), recombining all the previously okayed spikes,
	and re-clustering with double the number of clusters of previously approved
	clusters"""
	
# def run_ica_spike_sort(ICA_h5_dir):
# 	"""This function pulls data from the ICA hf5 file, finds the ICA data 
# 	peaks, and performs clustering spike sorting to separate out true peaks"""
# 	
# 	#Import ICA weights and cleaned data
# 	hf5 = tables.open_file(ICA_h5_dir, 'r+', title = ICA_h5_dir[-1])
# 	ICA_weights = hf5.root.ica_weights[0,:,:]
# 	clean_data = hf5.root.cleaned_data[0,:,:]
# 	sampling_rate = hf5.root.sampling_rate[0]
# 	hf5.close()
# 	del hf5
# 	
# 	#Create directory for sorted data
# 	sort_data_dir = ('/').join(ICA_h5_dir.split('/')[:-2]) + '/sort_results/'
# 	
# 	#Convert data to ICA components
# 	components = np.matmul(ICA_weights,clean_data)
# 	del clean_data	
# 	
# 	#Pull spikes from components	
# 	sort_hf5_dir, separated_spikes = run_spike_sort(components,sampling_rate,sort_data_dir)
# 		
# 	return sort_hf5_dir, separated_spikes

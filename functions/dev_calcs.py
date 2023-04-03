#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:06:17 2023

@author: hannahgermaine

This is a collection of functions for calculating and analyzing deviation bins
"""

import os, tqdm, tables, random
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
from joblib import Parallel, delayed

def FR_dev_calcs(fig_save_dir,segment_names,segment_times,segment_spike_times,
				 num_neur,num_tastes,local_bin_size,deviation_bin_size,dev_thresh,
				 std_cutoff,fig_buffer_size,partic_neur_cutoff,num_null_sets):
	"""This is the master function that runs all the other calculator functions"""
	print("\nBeginning firing rate deviation calculations.")
	
	#Create results save directory
	dev_save_dir = fig_save_dir + 'deviations/'
	if os.path.isdir(dev_save_dir) == False:
		os.mkdir(dev_save_dir)
		
	#Create .h5 save directory
	hdf5_name = 'deviation_results.h5'
	hf5_dir = dev_save_dir + hdf5_name
	
	#Import/save settings
	local_bin_size,deviation_bin_size,dev_thresh,std_cutoff,partic_neur_cutoff,any_change = import_settings(hf5_dir,local_bin_size,deviation_bin_size,dev_thresh,std_cutoff,partic_neur_cutoff)
	
	#Convert the bin sizes from time to samples
	num_segments = len(segment_names)
	
	#Calculate deviation bins
	segment_devs, segment_dev_frac_ind = dev_calcs(hf5_dir,num_neur,num_segments,segment_names,segment_times,segment_spike_times,deviation_bin_size,local_bin_size,std_cutoff,partic_neur_cutoff,any_change)
	
	#Calculate deviation bouts
	segment_bouts, segment_bout_lengths, segment_ibis = deviation_bout_ibi_calc(hf5_dir,num_segments,segment_names,segment_times,num_neur,dev_save_dir,segment_devs,deviation_bin_size,any_change)
		
	#Calculate mean/std of bout data
	mean_segment_bout_lengths,std_segment_bout_lengths,mean_segment_ibis,std_segment_ibis,num_dev_per_seg,dev_per_seg_freq = mean_std_bout_ibi_calc(num_segments,num_neur,segment_names,segment_times,dev_save_dir,segment_bout_lengths,segment_ibis)
	
	#Calculate null distribution deviation bins
	null_segment_dev_counts,null_segment_dev_ibis,null_segment_dev_bout_len = null_dev_calc(hf5_dir,num_segments,num_neur,segment_names,segment_times,num_null_sets,segment_spike_times,deviation_bin_size,local_bin_size,std_cutoff,dev_thresh,partic_neur_cutoff)
	
	return dev_save_dir,segment_devs,segment_dev_frac_ind,segment_bouts,segment_bout_lengths,segment_ibis,mean_segment_bout_lengths,std_segment_bout_lengths,mean_segment_ibis,std_segment_ibis,num_dev_per_seg,dev_per_seg_freq,null_segment_dev_counts,null_segment_dev_ibis,null_segment_dev_bout_len
	
def import_settings(hf5_dir,local_bin_size,deviation_bin_size,dev_thresh,std_cutoff,partic_neur_cutoff):
	try:
		hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
		local_bin_size_import = hf5.root.settings.local_bin_size[0]
		deviation_bin_size_import = hf5.root.settings.deviation_bin_size[0]
		dev_thresh_import = hf5.root.settings.dev_thresh[0]
		std_cutoff_import = hf5.root.settings.std_cutoff[0]
		partic_neur_cutoff_import = hf5.root.settings.partic_neur_cutoff[0]
		hf5.close()
		
		#Overwrite values based on user input
		local_bin_size, change = keep_func('Local bin size',local_bin_size_import,local_bin_size)
		deviation_bin_size, change1 = keep_func('Deviation bin size',deviation_bin_size_import,deviation_bin_size)
		dev_thresh, change2 = keep_func('Deviation threshold',dev_thresh_import,dev_thresh)
		std_cutoff, change3 = keep_func('Std. cutoff',std_cutoff_import,std_cutoff)
		partic_neur_cutoff, change4 = keep_func('Frac. participating neurons',partic_neur_cutoff_import,partic_neur_cutoff)
		
		any_change = np.sum((change,change1,change2,change3,change4)) > 0
		
		#Save the new values in the HDF5 file
		hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
		hf5.root.settings.local_bin_size[0] = local_bin_size
		hf5.root.settings.deviation_bin_size[0] = deviation_bin_size
		hf5.root.settings.dev_thresh[0] = dev_thresh
		hf5.root.settings.std_cutoff[0] = std_cutoff
		hf5.root.settings.partic_neur_cutoff[0] = partic_neur_cutoff
		hf5.close()
	except:
		hf5 = tables.open_file(hf5_dir, 'w', title = hf5_dir[-1])
		hf5.create_group('/', 'true_calcs')
		hf5.create_group('/', 'null_calcs')
		hf5.create_group('/', 'settings')
		atom = tables.FloatAtom()
		hf5.create_earray('/settings','local_bin_size',atom,(0,))
		exec("hf5.root.settings.local_bin_size.append(np.expand_dims(local_bin_size,0))")
		hf5.create_earray('/settings','deviation_bin_size',atom,(0,))
		exec("hf5.root.settings.deviation_bin_size.append(np.expand_dims(deviation_bin_size,0))")
		hf5.create_earray('/settings','dev_thresh',atom,(0,))
		exec("hf5.root.settings.dev_thresh.append(np.expand_dims(dev_thresh,0))")
		hf5.create_earray('/settings','std_cutoff',atom,(0,))
		exec("hf5.root.settings.std_cutoff.append(np.expand_dims(std_cutoff,0))")
		hf5.create_earray('/settings','partic_neur_cutoff',atom,(0,))
		exec("hf5.root.settings.partic_neur_cutoff.append(np.expand_dims(partic_neur_cutoff,0))")
		hf5.close()
		print('Created nodes in HF5')
		any_change = False
		
	return local_bin_size,deviation_bin_size,dev_thresh,std_cutoff,partic_neur_cutoff,any_change
	
def keep_func(val_name,import_val,set_val):
	"""This function asks the user which value to keep out of two"""
	change = 0
	if import_val != set_val:
		import_loop = 1
		while import_loop == 1:
			print(val_name + " is unequal between imported data and analysis settings.")
			print("Imported value = " + str(import_val) + " ; setting value = " + str(set_val))
			keep_import = input("Would you like to keep the imported value? (y/n)")
			if (keep_import != 'y') and (keep_import != 'n'):
				print("Incorrect entry, please try again.")
			else:
				import_loop = 0
		if keep_import == 'y':
			set_val = import_val
			change = 1
	return set_val, change

def dev_calcs(hf5_dir,num_neur,num_segments,segment_names,segment_times,segment_spike_times,deviation_bin_size,local_bin_size,std_cutoff,partic_neur_cutoff,any_change):
	"""This function calculates the bins which deviate per segment"""
	#First try importing previously stored data
	calc_new = 0
	if any_change == False:
		try:
			hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
			#segment_devs import
			segment_names_import = []
			segment_devs_vals = []
			segment_devs_times = []
			saved_nodes = hf5.list_nodes('/true_calcs/segment_devs')
			for s_i in saved_nodes:
	 			data_name = s_i.name
	 			data_seg = ('-').join(data_name.split('_')[0:-1])
	 			data_type = data_name.split('_')[-1]
	 			if data_type == 'devs':
					 segment_devs_vals.append(s_i[0,:])
	 			elif data_type == 'times':
					 segment_devs_times.append(s_i[0,:])
	 			index_match = segment_names.index(data_seg)
	 			try:
					 segment_names_import.index(index_match)
	 			except:
					 segment_names_import.extend([index_match])
			segment_devs = []
			for ind in range(len(segment_names_import)):
	 			ind_loc = segment_names_import.index(ind)
	 			segment_dev_bit = [segment_devs_times[ind_loc].astype('int'),segment_devs_vals[ind_loc]]
	 			segment_devs.append(segment_dev_bit)
			#segment_dev_frac_ind import
			segment_names_import = []
			segment_devs_fracs = []
			segment_devs_times = []
			saved_nodes = hf5.list_nodes('/true_calcs/segment_dev_frac_ind')
			for s_i in saved_nodes:
	 			data_name = s_i.name
	 			data_seg = ('-').join(data_name.split('_')[0:-1])
	 			data_type = data_name.split('_')[-1]
	 			if data_type == 'fracs':
					 segment_devs_fracs.append(s_i[0,:])
	 			elif data_type == 'times':
					 segment_devs_times.append(s_i[0,:])
	 			index_match = segment_names.index(data_seg)
	 			try:
					 segment_names_import.index(index_match)
	 			except:
					 segment_names_import.extend([index_match])
			segment_dev_frac_ind = []
			for ind in range(len(segment_names_import)):
	 			ind_loc = segment_names_import.index(ind)
	 			segment_dev_frac_bit = [segment_devs_fracs[ind_loc],segment_devs_times[ind_loc].astype('int')]
	 			segment_dev_frac_ind.append(segment_dev_frac_bit)		
			hf5.close()	
			
			#Save results to .h5
			print("Imported previously saved deviation calculations.")
		except:
			print("No previous data to import. Beginning calculations.")
			calc_new = 1
	else:
		calc_new = 1
		
	if calc_new == 1:
		#Parameters
		dev_bin_dt = int(np.ceil(deviation_bin_size*1000))
		half_dev_bin_dt = int(np.ceil(dev_bin_dt/2))
		local_bin_dt = int(np.ceil(local_bin_size*1000))
		half_local_bin_dt = int(np.ceil(local_bin_dt/2))
		#Begin tests
		segment_devs = []
		segment_dev_frac_ind = []
		for i in range(num_segments):
			print("\tCalculating deviations for segment " + segment_names[i])
			segment_spikes = segment_spike_times[i]
			#Generate arrays of start times for calculating the deviation from the mean
			start_segment = segment_times[i]
			end_segment = segment_times[i+1]
			dev_bin_starts = np.arange(start_segment,end_segment,dev_bin_dt)
			#First calculate the firing rates of all small bins
			bin_frs = np.zeros(len(dev_bin_starts)) #Store average firing rate for each bin
			bin_num_neur = np.zeros(len(dev_bin_starts)) #Store number of neurons firing in each bin
			for b_i in tqdm.tqdm(range(len(dev_bin_starts))):
				bin_start_dt = dev_bin_starts[b_i]
				start_db = max(bin_start_dt - half_dev_bin_dt, start_segment)
				end_db = min(bin_start_dt + half_dev_bin_dt, end_segment)
				neur_fc = [len(np.where((np.array(segment_spikes[n_i]) < end_db) & (np.array(segment_spikes[n_i]) > start_db))[0]) for n_i in range(num_neur)]
				bin_fcs = np.array(neur_fc)
				bin_num_neur[b_i] = np.array(len(np.where(np.array(neur_fc) > 0)[0]))
				bin_frs[b_i] = np.sum(bin_fcs,0)/deviation_bin_size
			#Next slide a larger window over the small bins and calculate deviations for each small bin
			bin_devs = np.zeros(len(dev_bin_starts)) #storage array for deviations from mean
			bin_dev_lens = np.zeros(np.shape(bin_devs))
			for b_i in tqdm.tqdm(range(len(dev_bin_starts))): #slide a mean window over all the starts and calculate the small bin firing rate's deviation
				bin_start_dt = dev_bin_starts[b_i]
				#First calculate mean interval bounds
				start_mean_int = max(bin_start_dt - half_local_bin_dt,0)
				start_mean_bin_start_ind = np.where(np.abs(dev_bin_starts - start_mean_int) == np.min(np.abs(dev_bin_starts - start_mean_int)))[0][0]
				end_mean_int = min(start_mean_int + local_bin_dt,end_segment-1)
				end_mean_bin_start_ind = np.where(np.abs(dev_bin_starts - end_mean_int) == np.min(np.abs(dev_bin_starts - end_mean_int)))[0][0]
				#Next calculate mean + std FR for the interval
				local_dev_bin_fr = bin_frs[np.arange(start_mean_bin_start_ind,end_mean_bin_start_ind)]
				local_dev_bin_num_neur = bin_num_neur[np.arange(start_mean_bin_start_ind,end_mean_bin_start_ind)]
				mean_fr = np.mean(local_dev_bin_fr)
				std_fr = np.std(local_dev_bin_fr)
				cutoff = mean_fr + std_cutoff*std_fr
				#Calculate which bins are > mean + 2std
				dev_neur_fr_locations = local_dev_bin_fr > cutoff*np.ones(np.shape(local_dev_bin_fr))
				dev_neur_num_locations = local_dev_bin_num_neur > partic_neur_cutoff*num_neur
				dev_neur_fr_indices = np.where(dev_neur_fr_locations*dev_neur_num_locations == True)[0]
				bin_devs[start_mean_bin_start_ind + dev_neur_fr_indices] += 1
				bin_dev_lens[start_mean_bin_start_ind + np.arange(len(dev_neur_fr_locations))] += 1
			avg_bin_devs = bin_devs/bin_dev_lens
			segment_devs.append([dev_bin_starts,avg_bin_devs])
			dev_inds = np.where(avg_bin_devs > 0)[0]
			segment_dev_frac_ind.append([avg_bin_devs[dev_inds],dev_bin_starts[dev_inds]])
			
		#Save results to .h5
		print("Saving results to .h5")
		#Save to .h5
		hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
		atom = tables.FloatAtom()
		hf5.create_group('/true_calcs', 'segment_devs')
		for s_i in range(num_segments):
			seg_name = ('_').join(segment_names[s_i].split('-'))
			hf5.create_earray('/true_calcs/segment_devs',f'{seg_name}_times',atom,(0,)+np.shape(segment_devs[s_i][0]))
			seg_dev_expand = np.expand_dims(segment_devs[s_i][0],0)
			exec("hf5.root.true_calcs.segment_devs."+f'{seg_name}'+"_times.append(seg_dev_expand)")
			hf5.create_earray('/true_calcs/segment_devs',f'{seg_name}_devs',atom,(0,)+np.shape(segment_devs[s_i][1]))
			seg_dev_expand = np.expand_dims(segment_devs[s_i][1],0)
			exec("hf5.root.true_calcs.segment_devs."+f"{seg_name}"+"_devs.append(seg_dev_expand)")
		hf5.create_group('/true_calcs', 'segment_dev_frac_ind')
		for s_i in range(num_segments):
			seg_name = ('_').join(segment_names[s_i].split('-'))
			hf5.create_earray('/true_calcs/segment_dev_frac_ind',f'{seg_name}_fracs',atom,(0,)+np.shape(segment_dev_frac_ind[s_i][0]))
			seg_dev_expand = np.expand_dims(segment_dev_frac_ind[s_i][0],0)
			exec("hf5.root.true_calcs.segment_dev_frac_ind."+f'{seg_name}'+"_fracs.append(seg_dev_expand)")
			hf5.create_earray('/true_calcs/segment_dev_frac_ind',f'{seg_name}_times',atom,(0,)+np.shape(segment_dev_frac_ind[s_i][1]))
			seg_dev_expand = np.expand_dims(segment_dev_frac_ind[s_i][1],0)
			exec("hf5.root.true_calcs.segment_dev_frac_ind."+f'{seg_name}'+"_times.append(seg_dev_expand)")
		hf5.close()
	
	return segment_devs, segment_dev_frac_ind

def deviation_bout_ibi_calc(hf5_dir,num_segments,segment_names,segment_times,num_neur,dev_save_dir,segment_devs,deviation_bin_size,any_change):
	"""This function calculates segment bouts and their statistics"""
	
	calculate_new = 0
	if any_change == False:
		try:
			#import hf5 data
			hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
			#dev calc imports
			segment_bouts_import = []
			segment_bouts_ibis_vals = []
			segment_bouts_lengths_vals = []
			segment_bouts_times_vals = []
			saved_nodes = hf5.list_nodes('/true_calcs/segment_bouts')
			for s_i in saved_nodes:
				data_name = s_i.name
				data_seg = ('-').join(data_name.split('_')[0:-1])
				data_type = data_name.split('_')[-1]
				if data_type == 'lengths':
					segment_bouts_lengths_vals.append(s_i[0,:])
				elif data_type == 'ibis':
					segment_bouts_ibis_vals.append(s_i[0,:])
				elif data_type == 'times':
					segment_bouts_times_vals.append(s_i[0,:])
				index_match = segment_names.index(data_seg)
				try:
					segment_bouts_import.index(index_match)
				except:
					segment_bouts_import.extend([index_match])
			del saved_nodes, s_i, data_name, data_seg, data_type, index_match
			#set up storage variables
			segment_bouts = []
			segment_bout_lengths = []
			segment_ibis = []
			for ind in range(len(segment_bouts_import)):
				ind_loc = segment_bouts_import.index(ind)
				segment_bouts.append(segment_bouts_times_vals[ind_loc])
				segment_bout_lengths.append(segment_bouts_lengths_vals[ind_loc])
				segment_ibis.append(segment_bouts_ibis_vals[ind_loc])
			del segment_bouts_import, segment_bouts_ibis_vals, segment_bouts_lengths_vals, segment_bouts_times_vals, ind
			#Always close the h5 file!
			hf5.close()	
			
			#Save results to .h5
			print("Imported previously saved deviation calculations.")
		except:
			print("No prior data to import.")
			calculate_new = 1
	else:
		calculate_new = 1
			
	if calculate_new == 1:		
		#Calculate the deviation bout size and frequency
		dev_bin_dt = int(np.ceil(deviation_bin_size*1000)) #converting to ms timescale
		half_dev_bin_dt = int(np.ceil(dev_bin_dt/2))
		segment_bouts = []
		segment_bout_lengths = []
		segment_ibis = []
		for i in tqdm.tqdm(range(num_segments)):
			print("\t Calculating deviation bout sizes and frequencies")
			seg_dev_save_dir = dev_save_dir + ('_').join(segment_names[i].split(' ')) + '/'
			if os.path.isdir(seg_dev_save_dir) == False:
				os.mkdir(seg_dev_save_dir)
			seg_devs = segment_devs[i][1] #Fraction of deviations for each segment bout
			seg_times = segment_devs[i][0] #Original data indices of each segment bout
			dev_inds = np.where(seg_devs > 0)[0] #Indices of deviating segment bouts
			dev_times = seg_times[dev_inds] #Original data deviation data indices
			bout_start_inds = np.concatenate((np.array([0]),np.where(np.diff(dev_inds) > 2)[0] + 1))
			bout_end_inds = np.concatenate((np.where(np.diff(dev_inds) > 2)[0],np.array([-1]))) #>2 because the half-bin leaks into a second bin
			try:
				bout_start_times = dev_times[bout_start_inds] - half_dev_bin_dt
			except:
				bout_start_times = np.empty(0)
			try:
				bout_end_times = dev_times[bout_end_inds] + half_dev_bin_dt
			except:
				bout_end_times = np.empty(0)
			bout_pairs = np.array([bout_start_times,bout_end_times]).T
			for b_i in range(len(bout_pairs)):
				if bout_pairs[b_i][0] == bout_pairs[b_i][1]:
					bout_pairs[b_i][1] += dev_bin_dt
			segment_bouts.append(bout_pairs)
			bout_lengths = bout_end_times - bout_start_times + int(deviation_bin_size*1000) #in ms timescale
			bout_lengths_s = bout_lengths/1000 #in Hz
			segment_bout_lengths.append(bout_lengths_s)
			ibi = (bout_start_times[1:] - bout_end_times[:-1])/1000 #in seconds
			segment_ibis.append(ibi)
		
		#Save results to .h5
		print("Saving results to .h5")
		#Save to .h5
		hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
		atom = tables.FloatAtom()
		hf5.create_group('/true_calcs', 'segment_bouts')
		for s_i in range(num_segments):
			seg_name = ('_').join(segment_names[s_i].split('-'))
			hf5.create_earray('/true_calcs/segment_bouts',f'{seg_name}_times',atom,(0,)+np.shape(segment_bouts[s_i]))
			seg_bout_expand = np.expand_dims(segment_bouts[s_i],0)
			exec("hf5.root.true_calcs.segment_bouts."+f'{seg_name}'+"_times.append(seg_bout_expand)")
			hf5.create_earray('/true_calcs/segment_bouts',f'{seg_name}_lengths',atom,(0,)+np.shape(segment_bout_lengths[s_i]))
			seg_bout_expand = np.expand_dims(segment_bout_lengths[s_i],0)
			exec("hf5.root.true_calcs.segment_bouts."+f"{seg_name}"+"_lengths.append(seg_bout_expand)")
			hf5.create_earray('/true_calcs/segment_bouts',f'{seg_name}_ibis',atom,(0,)+np.shape(segment_ibis[s_i]))
			seg_bout_expand = np.expand_dims(segment_ibis[s_i],0)
			exec("hf5.root.true_calcs.segment_bouts."+f"{seg_name}"+"_ibis.append(seg_bout_expand)")
		hf5.close()
	
	return segment_bouts, segment_bout_lengths, segment_ibis
		
def mean_std_bout_ibi_calc(num_segments,num_neur,segment_names,segment_times,dev_save_dir,segment_bout_lengths,segment_ibis):
	#Calculate mean and standard deviations of bout lengths and ibis
	print("\t Calculating and plotting mean deviation bout length / ibis")
	mean_segment_bout_lengths = []
	std_segment_bout_lengths = []
	mean_segment_ibis = []
	std_segment_ibis = []
	num_dev_per_seg = []
	for i in range(num_segments):
		seg_bout_means = [np.mean(segment_bout_lengths[i])]
		seg_bout_stds = [np.std(segment_bout_lengths[i])]
		seg_ibi_means = [np.mean(segment_ibis[i])]
		seg_ibi_stds = [np.std(segment_ibis[i])]
		mean_segment_bout_lengths.append(seg_bout_means)
		std_segment_bout_lengths.append(seg_bout_stds)
		mean_segment_ibis.append(seg_ibi_means)
		std_segment_ibis.append(seg_ibi_stds)
		num_dev_per_seg.append([len(segment_bout_lengths[i])])
	dev_per_seg_freq = [(num_dev_per_seg[s_i]/(segment_times[s_i+1]-segment_times[s_i]))*1000 for s_i in range(len(segment_names)-1)] #num deviations per second for the segment
	#Convert to np.arrays for easy transposition
	mean_segment_bout_lengths = np.array(mean_segment_bout_lengths)
	std_segment_bout_lengths = np.array(std_segment_bout_lengths)
	mean_segment_ibis = np.array(mean_segment_ibis)
	std_segment_ibis = np.array(std_segment_ibis)
	
	return mean_segment_bout_lengths,std_segment_bout_lengths,mean_segment_ibis,std_segment_ibis,num_dev_per_seg,dev_per_seg_freq
	
def null_dev_calc(hf5_dir,num_segments,num_neur,segment_names,segment_times,num_null_sets,segment_spike_times,deviation_bin_size,local_bin_size,std_cutoff,dev_thresh,partic_neur_cutoff):
	"""This function calculates the number of deviations in time-shuffled data
	to create null distributions for statistical significance
	INPUTS:
		- num_segments:
		- num_neur:
		- segment_names:
		- segment_times:
		- num_null_sets:
		- segment_spike_times:
		- deviation_bin_size:
		- local_bin_size:
		- dev_thresh:
		- sampling_rate:
	OUTPUTS:
		- 
	
	"""
	#Parameters
	dev_bin_dt = int(np.ceil(deviation_bin_size*1000))
	half_dev_bin_dt = int(np.ceil(dev_bin_dt/2))
	local_bin_dt = int(np.ceil(local_bin_size*1000))
	half_local_bin_dt = int(np.ceil(local_bin_dt/2))
	#Storage
	null_segment_dev_counts = []
	null_segment_dev_ibis = []
	null_segment_dev_bout_len = []
	for i in range(num_segments):
		print("\tCalculating null deviations for segment " + segment_names[i])
		segment_spikes = segment_spike_times[i]
		#Generate arrays of start times for calculating the deviation from the mean
		start_segment = segment_times[i]
		end_segment = segment_times[i+1]
		seg_len = int(end_segment - start_segment)
		dev_bin_starts = np.arange(0,seg_len,dev_bin_dt)
		#Calculate values
		results = [shuffle_dev_func(segment_spikes,start_segment,end_segment,dev_bin_starts,half_dev_bin_dt,seg_len,deviation_bin_size,
								 half_local_bin_dt,local_bin_dt,std_cutoff,partic_neur_cutoff,num_neur,dev_thresh) for i in tqdm.tqdm(range(num_null_sets))]
		#Pull apart results
		seg_dev_counts, seg_dev_bout_lens, seg_dev_ibis = zip(*results)
		
		null_segment_dev_counts.append(list(seg_dev_counts))
		null_segment_dev_ibis.append(list(seg_dev_ibis))
		null_segment_dev_bout_len.append(list(seg_dev_bout_lens))
	
	print("Done calculating null results - .h5 file not saved, code needs work.")
	#Save results to .h5
	#print("Saving results to .h5")
	#Save to .h5
	#hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
	#atom = tables.FloatAtom()
	#for s_i in range(num_segments):
	#	seg_name = ('_').join(segment_names[s_i].split('-'))
	#	hf5.create_earray('/null_calcs',f'{seg_name}_counts',atom,(0,)+np.shape(null_segment_dev_counts[s_i][0]))
	#	seg_dev_expand = np.expand_dims(null_segment_dev_counts[s_i][0],0)
	#	exec("hf5.root.null_calcs."+f'{seg_name}_counts'+".append(seg_dev_expand)")
	#	seg_dev = np.array(null_segment_dev_ibis[s_i])
	#	hf5.create_earray('/null_calcs',f'{seg_name}_ibis',atom,(0,)+np.shape(seg_dev))
	#	seg_dev_expand = np.expand_dims(seg_dev,0)
	#	exec("hf5.root.null_calcs."+f'{seg_name}_ibis'+".append(seg_dev_expand)")
	#	seg_dev = np.array(null_segment_dev_bout_len[s_i])
	#	hf5.create_earray('/null_calcs',f'{seg_name}_lengths',atom,(0,)+np.shape(seg_dev))
	#	seg_dev_expand = np.expand_dims(seg_dev,0)
	#	exec("hf5.root.null_calcs."+f'{seg_name}_lengths'+".append(seg_dev_expand)")
	#hf5.close()
	
	return [null_segment_dev_counts, null_segment_dev_ibis, null_segment_dev_bout_len]

def shuffle_dev_func(segment_spikes,segment_start_time,segment_end_time,dev_bin_starts,half_dev_bin_dt,seg_len,deviation_bin_size,
					 half_local_bin_dt,local_bin_dt,std_cutoff,partic_neur_cutoff,num_neur,dev_thresh):
	
	np.seterr(divide='ignore', invalid='ignore')
	#Shuffle spike times
	true_spike_counts = [len(segment_spikes[n_i]) for n_i in range(num_neur)]
	#Create binary matrix
	shuffle_spikes_bin = np.zeros((num_neur,int(segment_end_time - segment_start_time)))
	for n_i in range(num_neur):
		fake_spike_times = random.sample(range(0,int(segment_end_time - segment_start_time)),true_spike_counts[n_i])
		shuffle_spikes_bin[n_i,fake_spike_times] += 1	
	#First calculate the firing rates of all small bins
	bin_frs = np.zeros(len(dev_bin_starts)) #Store average firing rate for each bin
	bin_num_neur = np.zeros(len(dev_bin_starts)) #Store number of neurons firing in each bin
	for b_i in range(len(dev_bin_starts)):
		bin_start_dt = dev_bin_starts[b_i]
		start_db = max(bin_start_dt - half_dev_bin_dt, 0)
		end_db = min(bin_start_dt + half_dev_bin_dt, seg_len)
		neur_fc = np.sum(shuffle_spikes_bin[:,start_db:end_db],1)
		bin_num_neur[b_i] = np.array(len(np.where(np.array(neur_fc) > 0)[0]))
		bin_frs[b_i] = np.sum(neur_fc,0)/deviation_bin_size
	#Next slide a larger window over the small bins and calculate deviations for each small bin
	bin_devs = np.zeros(len(dev_bin_starts)) #storage array for deviations from mean
	bin_dev_lens = np.zeros(np.shape(bin_devs))
	for b_i in range(len(dev_bin_starts)): #slide a mean window over all the starts and calculate the small bin firing rate's deviation
		bin_start_dt = dev_bin_starts[b_i]
		#First calculate mean interval bounds
		start_mean_int = max(bin_start_dt - half_local_bin_dt,0)
		start_mean_bin_start_ind = np.where(np.abs(dev_bin_starts - start_mean_int) == np.min(np.abs(dev_bin_starts - start_mean_int)))[0][0]
		end_mean_int = min(start_mean_int + local_bin_dt,seg_len-1)
		end_mean_bin_start_ind = np.where(np.abs(dev_bin_starts - end_mean_int) == np.min(np.abs(dev_bin_starts - end_mean_int)))[0][0]
		#Next calculate mean + std FR for the interval
		local_dev_bin_fr = bin_frs[np.arange(start_mean_bin_start_ind,end_mean_bin_start_ind)]
		local_dev_bin_num_neur = bin_num_neur[np.arange(start_mean_bin_start_ind,end_mean_bin_start_ind)]
		mean_fr = np.mean(local_dev_bin_fr)
		std_fr = np.std(local_dev_bin_fr)
		cutoff = mean_fr + std_cutoff*std_fr
		#Calculate which bins are > mean + std_cutoff*std
		dev_neur_fr_locations = local_dev_bin_fr > cutoff*np.ones(np.shape(local_dev_bin_fr))
		dev_neur_num_locations = local_dev_bin_num_neur > partic_neur_cutoff*num_neur
		dev_neur_fr_indices = np.where(dev_neur_fr_locations*dev_neur_num_locations == True)[0]
		bin_devs[start_mean_bin_start_ind + dev_neur_fr_indices] += 1
		bin_dev_lens[start_mean_bin_start_ind + np.arange(len(dev_neur_fr_locations))] += 1
	avg_bin_devs = bin_devs/bin_dev_lens
	seg_dev_count_calc = len(np.where(avg_bin_devs > dev_thresh)[0]) #Add brackets if using result_dict below
	#Calculate bout lengths and ibis
	dev_inds = np.where(avg_bin_devs > 0)[0] #Indices of deviating segment bouts
	dev_times = dev_bin_starts[dev_inds] #Original data deviation data indices
	bout_start_inds = np.concatenate((np.array([0]),np.where(np.diff(dev_inds) > 1)[0] + 1))
	bout_end_inds = np.concatenate((np.where(np.diff(dev_inds) > 1)[0],np.array([-1])))
	try:
		bout_start_times = dev_times[bout_start_inds]
	except:
		bout_start_times = np.empty(0)
	try:
		bout_end_times = dev_times[bout_end_inds]
	except:
		bout_end_times = np.empty(0)
	bout_lengths = bout_end_times - bout_start_times + int(deviation_bin_size*1000) #in ms timescale
	seg_dev_bout_lens_calc = list(bout_lengths/1000) #in Hz
	seg_dev_ibis_calc = list((bout_start_times[1:] - bout_end_times[:-1])/1000) #in seconds
	
	#result_dict = {'seg_dev_count_calc':seg_dev_count_calc,'seg_dev_bout_lens_calc':seg_dev_bout_lens_calc,'seg_dev_ibis_calc':seg_dev_ibis_calc}
	result_array = [seg_dev_count_calc,seg_dev_bout_lens_calc,seg_dev_ibis_calc]
	
	return result_array #result_dict


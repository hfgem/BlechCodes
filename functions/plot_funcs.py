#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:54:17 2023

@author: hannahgermaine

This is a collection of functions for plotting data
"""

import os, tqdm, tables
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

def raster_plots(fig_save_dir, dig_in_names, start_dig_in_times, end_dig_in_times, 
				 segment_names, segment_times, spike_times, num_neur, num_tastes, 
				 sampling_rate):
	
	#_____Grab spike times (and rasters) for each segment separately_____
	raster_save_dir = fig_save_dir + 'rasters/'
	if os.path.isdir(raster_save_dir) == False:
		os.mkdir(raster_save_dir)
	segment_spike_times = []
	for s_i in tqdm.tqdm(range(len(segment_names))):
		print("\nGrabbing spike raster for segment " + segment_names[s_i])
		min_time = segment_times[s_i]
		max_time = segment_times[s_i+1]
		max_time_min = (max_time-min_time)*(1/sampling_rate)*(1/60)
		s_name = segment_names[s_i]
		s_t = [list(np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= min_time)*(np.array(spike_times[i]) <= max_time))[0]]) for i in range(num_neur)]
		s_t_time = [list((1/60)*np.array(s_t[i])*(1/sampling_rate)) for i in range(len(s_t))]
		#Plot segment rasters and save
		plt.figure(figsize=(max_time_min,num_neur))
		plt.xlabel('Time (m)')
		plt.eventplot(s_t_time,colors='k')
		plt.title(s_name + " segment")
		plt.tight_layout()
		im_name = ('_').join(s_name.split(' '))
		plt.savefig(raster_save_dir + im_name + '.png')
		plt.savefig(raster_save_dir + im_name + '.svg')
		plt.close()
		segment_spike_times.append(s_t)

	#_____Grab spike times for each taste delivery separately_____
	pre_taste = 0.5 #Seconds before tastant delivery to store
	pre_taste_dt = int(np.ceil(pre_taste*sampling_rate))
	post_taste = 2 #Seconds after tastant delivery to store
	post_taste_dt = int(np.ceil(post_taste*sampling_rate))
	tastant_spike_times = []
	for t_i in tqdm.tqdm(range(num_tastes)):
		print("\nGrabbing spike rasters for tastant " + dig_in_names[t_i] + " deliveries")
		rast_taste_save_dir = raster_save_dir + ('_').join((dig_in_names[t_i]).split(' ')) + '/'
		if os.path.isdir(rast_taste_save_dir) == False:
			os.mkdir(rast_taste_save_dir)
		t_start = start_dig_in_times[t_i]
		t_end = end_dig_in_times[t_i]
		num_deliv = len(t_start)
		t_st = []
		t_fig = plt.figure(figsize=(10,num_deliv))
		for t_d_i in range(len(t_start)):
			start_i = int(max(t_start[t_d_i] - pre_taste_dt,0))
			end_i = int(min(t_end[t_d_i] + post_taste_dt,segment_times[-1]))
			#Grab spike times into one list
			s_t = [list(np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= start_i)*(np.array(spike_times[i]) <= end_i))[0]]) for i in range(num_neur)]
			s_t_time = [list(np.array(s_t[i])*(1/sampling_rate)) for i in range(len(s_t))]
			t_st.append(s_t)
			#Plot the raster
			plt.subplot(num_deliv,1,t_d_i+1)
			plt.eventplot(s_t_time,colors='k')
			plt.xlabel('Time (s)')
			plt.axvline(t_start[t_d_i]/sampling_rate,color='r')
			plt.axvline(t_end[t_d_i]/sampling_rate,color='r')
		#Save the figure
		im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_spike_rasters'
		t_fig.tight_layout()
		t_fig.savefig(rast_taste_save_dir + im_name + '.png')
		t_fig.savefig(rast_taste_save_dir + im_name + '.svg')
		plt.close(t_fig)
		tastant_spike_times.append(t_st)
		
		#Plot rasters for each neuron for each taste separately
		rast_taste_save_dir = raster_save_dir + ('_').join((dig_in_names[t_i]).split(' ')) + '/'
		if os.path.isdir(rast_taste_save_dir) == False:
			os.mkdir(rast_taste_save_dir)
		rast_neur_save_dir = rast_taste_save_dir + 'neurons/'
		if os.path.isdir(rast_neur_save_dir) == False:
			os.mkdir(rast_neur_save_dir)
		for n_i in range(num_neur):
			n_st = [t_st[t_d_i][n_i] - t_start[t_d_i] for t_d_i in range(len(t_start))]
			n_st_time = [list(np.array(n_st[i])*(1/sampling_rate)) for i in range(len(n_st))]
			raster_len_max = (max(t_end) - min(t_start))*(1/sampling_rate)*(1/60)
			t_fig = plt.figure(figsize=(raster_len_max,len(t_start)))
			plt.eventplot(n_st_time,colors='k')
			plt.xlabel('Time (s)')
			plt.ylabel('Trial')
			plt.axvline(pre_taste_dt/sampling_rate,color='r')
			#Save the figure
			im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_unit_' + str(n_i)
			t_fig.tight_layout()
			t_fig.savefig(rast_neur_save_dir + im_name + '.png')
			t_fig.savefig(rast_neur_save_dir + im_name + '.svg')
			plt.close(t_fig)
	
	return segment_spike_times, tastant_spike_times, pre_taste_dt, post_taste_dt

def PSTH_plots(fig_save_dir, sampling_rate, num_tastes, num_neur, dig_in_names, 
			   start_dig_in_times, end_dig_in_times, pre_taste_dt, post_taste_dt,
			   segment_times, spike_times):
	
	PSTH_save_dir = fig_save_dir + 'PSTHs/'
	if os.path.isdir(PSTH_save_dir) == False:
		os.mkdir(PSTH_save_dir)
	bin_width = 0.25 #Gaussian convolution kernel width in seconds
	half_bin_width_dt = int(np.ceil(sampling_rate*bin_width/2))
	bin_step = 25 #Step size in dt to take in PSTH calculation
	PSTH_times = [] #Storage of time bin true times (s) for each tastant
	PSTH_taste_deliv_times = [] #Storage of tastant delivery true times (s) for each tastant [start,end]
	tastant_PSTH = []
	avg_tastant_PSTH = []
	for t_i in tqdm.tqdm(range(num_tastes)):
		print("\nGrabbing PSTHs for tastant " + dig_in_names[t_i] + " deliveries")
		t_start = np.array(start_dig_in_times[t_i])
		t_end = np.array(end_dig_in_times[t_i])
		dt_total = np.max(t_end-t_start) + pre_taste_dt + post_taste_dt
		num_deliv = len(t_start)
		PSTH_start_times = np.arange(0,dt_total,bin_step)
		PSTH_true_times = np.round(PSTH_start_times/sampling_rate,3)
		PSTH_times.append(PSTH_true_times)
		start_deliv_interval = PSTH_true_times[np.where(PSTH_start_times > pre_taste_dt)[0][0]]
		end_deliv_interval = PSTH_true_times[np.where(PSTH_start_times > dt_total - post_taste_dt)[0][0]]
		PSTH_taste_deliv_times.append([start_deliv_interval,end_deliv_interval])
		all_PSTH = np.zeros((num_deliv,num_neur,len(PSTH_start_times)))
		t_fig = plt.figure(figsize=(10,num_deliv))
		for t_d_i in range(len(t_start)):
			start_i = int(max(t_start[t_d_i] - pre_taste_dt,0))
			end_i = int(min(t_end[t_d_i] + post_taste_dt,segment_times[-1]))
			bin_spikes = np.zeros((num_neur,dt_total))
			#Convert spike times into a binary vector
			for i in range(num_neur):
				s_t = np.array((np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= start_i)*(np.array(spike_times[i]) <= end_i))[0]]))
				s_t -= start_i
				bin_spikes[i,s_t] = 1
			del i, s_t
			#Perform Gaussian convolution
			PSTH_spikes = np.zeros((num_neur,len(PSTH_start_times)))
			for i in range(len(PSTH_start_times)):
				PSTH_spikes[:,i] = (np.sum(bin_spikes[:,max(PSTH_start_times[i]-half_bin_width_dt,0):min(PSTH_start_times[i]+half_bin_width_dt,dt_total)],1)/bin_width).T
			del i
			plt.subplot(num_deliv,1,t_d_i+1)
			#Update to have x-axis in time
			for i in range(num_neur):
				plt.plot(PSTH_true_times,PSTH_spikes[i,:])
			del i
			plt.axvline(start_deliv_interval,color='r')
			plt.axvline(end_deliv_interval,color='r')
			all_PSTH[t_d_i,:,:] = PSTH_spikes
		del t_d_i, start_i, end_i, bin_spikes, PSTH_spikes
		tastant_name = dig_in_names[t_i]
		im_name = ('_').join((tastant_name).split(' ')) + '_PSTHs'
		t_fig.tight_layout()
		t_fig.savefig(PSTH_save_dir + im_name + '.png')
		t_fig.savefig(PSTH_save_dir + im_name + '.svg')
		plt.close(t_fig)
		avg_PSTH = np.mean(all_PSTH,axis=0)
		t_fig = plt.figure()
		for i in range(num_neur):
			plt.plot(PSTH_true_times,avg_PSTH[i,:])
		del i
		plt.axvline(start_deliv_interval,color='r',linestyle = 'dashed')
		plt.axvline(end_deliv_interval,color='r',linestyle='solid')
		plt.title('Avg Individual Neuron PSTH for ' + tastant_name + '\nAligned to Taste Delivery Start')
		plt.xlabel('Time (s)')
		plt.ylabel('Firing Rate (Hz)')
		im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_avg_PSTH'
		t_fig.tight_layout()
		t_fig.savefig(PSTH_save_dir + im_name + '.png')
		t_fig.savefig(PSTH_save_dir + im_name + '.svg')
		plt.close(t_fig)
		tastant_PSTH.append(all_PSTH)
		avg_tastant_PSTH.append(avg_PSTH)
	del t_i, t_start, t_end, dt_total, num_deliv, PSTH_start_times, PSTH_true_times, start_deliv_interval, end_deliv_interval, all_PSTH
	del t_fig, tastant_name, im_name, avg_PSTH
		
	#_____Plot avg PSTHs for Individual Neurons_____
	neuron_PSTH_dir = PSTH_save_dir + 'neurons/'
	if os.path.isdir(neuron_PSTH_dir) == False:
		os.mkdir(neuron_PSTH_dir)
	for n_i in range(num_neur):
		plt.figure(figsize=(10,10))
		for t_i in range(num_tastes):
			plt.plot(PSTH_times[t_i],avg_tastant_PSTH[t_i][n_i,:],label=dig_in_names[t_i])
			plt.bar(PSTH_taste_deliv_times[t_i][0],height=-1,width=PSTH_taste_deliv_times[t_i][1] - PSTH_taste_deliv_times[t_i][0],alpha=0.1,label=dig_in_names[t_i] + ' delivery')
		plt.legend(loc='upper right',fontsize=12)
		plt.title('Neuron ' + str(n_i))
		plt.ylabel('Firing Rate (Hz)')
		plt.xlabel('Time (s)')
		im_name = 'neuron_' + str(n_i) + '_PSTHs'
		plt.savefig(neuron_PSTH_dir + im_name + '.png')
		plt.savefig(neuron_PSTH_dir + im_name + '.svg')
		plt.close()
		
	return PSTH_times, PSTH_taste_deliv_times, tastant_PSTH, avg_tastant_PSTH

def FR_deviation_plots(fig_save_dir,sampling_rate,segment_names,segment_times,
					      segment_spike_times,num_neur,num_tastes,local_bin_size,
						  deviation_bin_size,dev_thresh,std_cutoff,fig_buffer_size):
	"""This function calculates firing rate deviations from local means and
	generates plots of different experimental segments with points above 1 + 2 
	deviations highlighted
	INPUTS:
		- fig_save_dir:
		- sampling_rate:
		- segment_names:
		- segment_times:
		- segment_spike_times: 
		- num_neur:
		- num_tastes:
		- local_bin_size:
		- deviation_bin_size:
		- fig_buffer_size: how many seconds to plot out in either direction of a deviation bin
	OUTPUTS:
		- 
	"""
	print("\nBeginning firing rate deviation calculations.")
	
	#Create results save directory
	dev_save_dir = fig_save_dir + 'deviations/'
	if os.path.isdir(dev_save_dir) == False:
		os.mkdir(dev_save_dir)
		
	#Create .h5 save directory
	hdf5_name = 'deviation_results2.h5'
	hf5_dir = dev_save_dir + hdf5_name
	try:
		hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
		sampling_rate_import = hf5.root.settings.sampling_rate[0]
		local_bin_size_import = hf5.root.settings.local_bin_size[0]
		deviation_bin_size_import = hf5.root.settings.deviation_bin_size[0]
		dev_thresh_import = hf5.root.settings.dev_thresh[0]
		std_cutoff_import = hf5.root.settings.std_cutoff[0]
		#ADD IN CODE HERE TO COMPARE WITH THE GIVEN FUNCTION SETTINGS AND ASK FOR USER INPUT
	except:
		hf5 = tables.open_file(hf5_dir, 'w', title = hf5_dir[-1])
		hf5.create_group('/', 'true_calcs')
		hf5.create_group('/', 'null_calcs')
		hf5.create_group('/', 'settings')
		atom = tables.FloatAtom()
		hf5.create_earray('/settings','sampling_rate',atom,(0,))
		exec("hf5.root.settings.sampling_rate.append(np.expand_dims(sampling_rate,0))")
		hf5.create_earray('/settings','local_bin_size',atom,(0,))
		exec("hf5.root.settings.local_bin_size.append(np.expand_dims(local_bin_size,0))")
		hf5.create_earray('/settings','deviation_bin_size',atom,(0,))
		exec("hf5.root.settings.deviation_bin_size.append(np.expand_dims(deviation_bin_size,0))")
		hf5.create_earray('/settings','dev_thresh',atom,(0,))
		exec("hf5.root.settings.dev_thresh.append(np.expand_dims(dev_thresh,0))")
		hf5.create_earray('/settings','std_cutoff',atom,(0,))
		exec("hf5.root.settings.std_cutoff.append(np.expand_dims(std_cutoff,0))")
		hf5.close()
		print('Created nodes in HF5')
		
	#Convert the bin sizes from time to samples
	num_segments = len(segment_names)
	
	#For each segment, a bin for a window size of local_bin_size will slide 
	#through time. At each slide the firing rates of all the small bins within
	#the local window are calculated. Those that are above 2 std. from the mean
	#are marked as being deviating bins
	segment_devs, segment_dev_frac_ind = dev_calcs(hf5_dir,num_neur,num_segments,segment_names,segment_times,
												segment_spike_times,deviation_bin_size,local_bin_size,std_cutoff,sampling_rate)
		
	#Plot deviation points for each neuron in each segment
	deviation_plots(num_segments,num_neur,sampling_rate,segment_names,dev_save_dir,segment_devs,dev_thresh)
	
	#Calculate and plot bouts of deviations and inter-bout-intervals (IBIs)
	segment_bouts, segment_bout_lengths, segment_ibis = deviation_bout_ibi_calc_plot(hf5_dir,num_segments,sampling_rate,
																				  segment_names,segment_times,
																				  num_neur,dev_save_dir,
																				  segment_devs,deviation_bin_size)
	
	#Calculate and plot mean and standard deviations of bout lengths and IBIs
	mean_segment_bout_lengths,std_segment_bout_lengths,mean_segment_ibis,std_segment_ibis,num_dev_per_seg,dev_per_seg_freq = mean_std_bout_ibi_calc_plot(num_segments,num_neur,
																													 segment_names,segment_times,sampling_rate,
																													 dev_save_dir,segment_bout_lengths,segment_ibis)
	
	#Plot deviation bin raster plots
	dev_bin_plots(fig_save_dir,sampling_rate,segment_names,segment_times,
						      segment_spike_times,num_neur,fig_buffer_size,
							  segment_bouts)
	
	#Calculate deviations for time-shuffled segments
	num_null_sets = 100 #Number of null distribution sets to create for testing deviation significance
	null_segment_dev_counts,null_segment_dev_ibis,null_segment_dev_bout_len = null_dev_calc(hf5_dir,num_segments,num_neur,segment_names,
																						 segment_times,num_null_sets,segment_spike_times,
																						 deviation_bin_size,local_bin_size,std_cutoff,
																						 dev_thresh,sampling_rate)
	
	#Plot true deviations against null distribution
	null_v_true_dev_plots(fig_save_dir,segment_names,segment_bouts,segment_bout_lengths,segment_ibis,num_null_sets,null_segment_dev_counts,null_segment_dev_ibis,null_segment_dev_bout_len)
	
	return segment_devs, segment_bouts, segment_bout_lengths, segment_ibis, mean_segment_bout_lengths, std_segment_bout_lengths, mean_segment_ibis, std_segment_ibis
		
def dev_calcs(hf5_dir,num_neur,num_segments,segment_names,segment_times,segment_spike_times,deviation_bin_size,local_bin_size,std_cutoff,sampling_rate):
	"""This function calculates the bins which deviate per segment"""
	#First try importing previously stored data
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
		print("DO SOMETHING")
	
	a = 0
	if a == 1: #USER WANTS TO KEEP IMPORTED RESULTS
		print("DO SOMETHING ^fix above")
	else: #USER WANTS TO RECALCULATE
		#Parameters
		dev_bin_dt = int(np.ceil(deviation_bin_size*sampling_rate))
		half_dev_bin_dt = int(np.ceil(dev_bin_dt/2))
		local_bin_dt = int(np.ceil(local_bin_size*sampling_rate))
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
			for b_i in tqdm.tqdm(range(len(dev_bin_starts))):
				bin_start_dt = dev_bin_starts[b_i]
				start_db = max(bin_start_dt - half_dev_bin_dt, start_segment)
				end_db = min(bin_start_dt + half_dev_bin_dt, end_segment)
				neur_fc = [len(np.where((np.array(segment_spikes[n_i]) < end_db) & (np.array(segment_spikes[n_i]) > start_db))[0]) for n_i in range(num_neur)]
				bin_fcs = np.array(neur_fc)
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
				mean_fr = np.mean(local_dev_bin_fr)
				std_fr = np.std(local_dev_bin_fr)
				cutoff = mean_fr + std_cutoff*std_fr
				#Calculate which bins are > mean + 2std
				dev_neur_fr_locations = local_dev_bin_fr > cutoff*np.ones(np.shape(local_dev_bin_fr))
				dev_neur_fr_indices = np.where(dev_neur_fr_locations == True)[0]
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

def deviation_plots(num_segments,num_neur,sampling_rate,segment_names,dev_save_dir,segment_devs,dev_thresh):
	"""
	INPUTS:
		- num_segments: number of segments in experiment
		- num_neur: number of neurons in data
		- sampling_rate: sampling rate of recording
		- segment_names: names of different segments
		- dev_save_dir: where to save plots
		- segment_devs: deviation fractions and times
		- thresh: threshold for strong deviations in [0,1]
	"""
	#Now plot deviations
	print("Beginning Deviation Plots.")
	for i in tqdm.tqdm(range(num_segments)):
		print("\n\tPlotting deviations for segment " + segment_names[i])
		seg_dev_save_dir = dev_save_dir + ('_').join(segment_names[i].split(' ')) + '/'
		if os.path.isdir(seg_dev_save_dir) == False:
			os.mkdir(seg_dev_save_dir)
		dev_times = segment_devs[i][0]*(1/sampling_rate) #Converted to seconds
		dev_vals = segment_devs[i][1]
		fig_i = plt.figure(figsize=((max(dev_times)-min(dev_times))*(1/60),10))
		plt.plot(dev_times,dev_vals)
		plt.xlabel('Time (s)')
		plt.ylabel('Deviation Fraction')
		im_name = (' ').join(segment_names[i].split('_'))
		plt.title(im_name + ' deviation fractions')
		save_name = ('_').join(segment_names[i].split(' ')) + '_devs'
		fig_i.savefig(seg_dev_save_dir + save_name + '.png')
		fig_i.savefig(seg_dev_save_dir + save_name + '.svg')
		plt.close(fig_i)
		fig_i = plt.figure(figsize=((max(dev_times)-min(dev_times))*(1/60),10))
		plt.plot(dev_times,(dev_vals>dev_thresh).astype('uint8'))
		im_name = (' ').join(segment_names[i].split('_'))
		plt.title(im_name + ' strong deviations (>' + str(dev_thresh) + ')')
		save_name = ('_').join(segment_names[i].split(' ')) + '_high_devs'
		fig_i.savefig(seg_dev_save_dir + save_name + '.png')
		fig_i.savefig(seg_dev_save_dir + save_name + '.svg')
		plt.close(fig_i)

def deviation_bout_ibi_calc_plot(hf5_dir,num_segments,sampling_rate,segment_names,segment_times,num_neur,dev_save_dir,segment_devs,deviation_bin_size):
	#Calculate the deviation bout size and frequency
	dev_bin_dt = int(np.ceil(deviation_bin_size*sampling_rate))
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
		bout_lengths = bout_end_times - bout_start_times + int(deviation_bin_size*sampling_rate) #in samples
		bout_lengths_s = bout_lengths/sampling_rate #in Hz
		segment_bout_lengths.append(bout_lengths_s)
		ibi = (bout_start_times[1:] - bout_end_times[:-1])/sampling_rate
		segment_ibis.append(ibi)
		fig_i = plt.figure(figsize = (10,10))
		plt.subplot(1,2,1)
		plt.hist(bout_lengths_s)
		plt.title('Bout lengths (s) histogram')
		plt.xlabel('Bout length (s)')
		plt.ylabel('Counts')
		plt.subplot(1,2,2)
		plt.hist(ibi)
		plt.title('Inter-bout-intervals (s) histogram')
		plt.xlabel('IBI (s)')
		plt.ylabel('Counts')
		save_name = ('_').join(segment_names[i].split(' ')) + '_dev_hist.png'
		fig_i.savefig(seg_dev_save_dir + save_name)
		plt.close(fig_i)
	
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
		
def mean_std_bout_ibi_calc_plot(num_segments,num_neur,segment_names,segment_times,sampling_rate,dev_save_dir,segment_bout_lengths,segment_ibis):
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
	dev_per_seg_freq = [(num_dev_per_seg[s_i]/(segment_times[s_i+1]-segment_times[s_i]))*sampling_rate for s_i in range(len(segment_names)-1)] #num deviations per second for the segment
	#Convert to np.arrays for easy transposition
	mean_segment_bout_lengths = np.array(mean_segment_bout_lengths)
	std_segment_bout_lengths = np.array(std_segment_bout_lengths)
	mean_segment_ibis = np.array(mean_segment_ibis)
	std_segment_ibis = np.array(std_segment_ibis)
	#Plot mean bouts and ibis
	fig_i = plt.figure(figsize = (10,10))
	#cm_subsection = np.linspace(0,1,num_neur)
	#cmap = [cm.gist_rainbow(x) for x in cm_subsection]
	plt.subplot(1,2,1)
	plt.plot(segment_names,mean_segment_bout_lengths,color='k',label='Mean Across Neurons')
	plt.plot(segment_names,mean_segment_bout_lengths + std_segment_bout_lengths,alpha=0.25,color='k',linestyle='-',label='Mean Across Neurons')
	plt.plot(segment_names,mean_segment_bout_lengths - std_segment_bout_lengths,alpha=0.25,color='k',linestyle='-',label='Mean Across Neurons')
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
	save_name = 'mean_bouts_ibis.png'
	fig_i.savefig(dev_save_dir + save_name)
	plt.close(fig_i)
	
	return mean_segment_bout_lengths,std_segment_bout_lengths,mean_segment_ibis,std_segment_ibis,num_dev_per_seg,dev_per_seg_freq
	
def dev_bin_plots(fig_save_dir,sampling_rate,segment_names,segment_times,
					      segment_spike_times,num_neur,fig_buffer_size,
						  segment_bouts):
	"""This function creates visualizations of bins with high deviation from local mean
	INPUTS:
		- fig_save_dir: directory to save visualizations
		- sampling_rate: sampling rate of data
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
	dev_save_dir = fig_save_dir + 'deviations/'
	if os.path.isdir(dev_save_dir) == False:
		os.mkdir(dev_save_dir)
	#Convert the bin size from time to samples
	num_segments = len(segment_names)
	local_bin_dt = int(np.ceil(fig_buffer_size*sampling_rate))
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
			s_t_time = [list(np.array(s_t[i])*(1/sampling_rate)) for i in range(len(s_t))] #In seconds
			#Plot segment deviation raster
			plt.figure(figsize=(10,num_neur))
			plt.xlabel('Time (s)')
			plt.ylabel('Neuron Index')
			plt.eventplot(s_t_time)
			plt.axvline(segment_dev_start_times[d_i]*(1/sampling_rate),color='r')
			plt.axvline(segment_dev_end_times[d_i]*(1/sampling_rate),color='r')
			plt.title('Deviation ' + str(d_i))
			plt.tight_layout()
			im_name = 'dev_' + str(d_i) + '.png'
			plt.savefig(seg_rast_save_dir + im_name)
			plt.close()
		
def null_dev_calc(hf5_dir,num_segments,num_neur,segment_names,segment_times,num_null_sets,segment_spike_times,deviation_bin_size,local_bin_size,std_cutoff,dev_thresh,sampling_rate):
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
	dev_bin_dt = int(np.ceil(deviation_bin_size*sampling_rate))
	half_dev_bin_dt = int(np.ceil(dev_bin_dt/2))
	local_bin_dt = int(np.ceil(local_bin_size*sampling_rate))
	half_local_bin_dt = int(np.ceil(local_bin_dt/2))
	#Begin tests
	null_segment_dev_counts = []
	null_segment_dev_ibis = []
	null_segment_dev_bout_len = []
	for i in range(num_segments):
		print("\tCalculating null deviations for segment " + segment_names[i])
		segment_spikes = segment_spike_times[i]
		#Generate arrays of start times for calculating the deviation from the mean
		start_segment = segment_times[i]
		end_segment = segment_times[i+1]
		seg_len = end_segment - start_segment
		dev_bin_starts = np.arange(0,seg_len,dev_bin_dt)
		#Create a binary array of spiking to shuffle
		segment_spikes_bin = np.zeros((num_neur,seg_len))
		for n_i in range(num_neur):
			n_i_spikes = np.array(segment_spikes[n_i]) - start_segment
			segment_spikes_bin[n_i,n_i_spikes] = 1
		#Create storage arrays
		seg_dev_counts = []
		seg_dev_ibis = []
		seg_dev_bout_lens = []
		for n_i in tqdm.tqdm(range(num_null_sets)):
			#Create a segment shuffle keeping sampling bin relationships between neurons
			shuffle_spikes_bin = segment_spikes_bin[:, np.random.permutation(segment_spikes_bin.shape[1])]
			#First calculate the firing rates of all small bins
			bin_frs = np.zeros(len(dev_bin_starts)) #Store average firing rate for each bin
			for b_i in range(len(dev_bin_starts)):
				bin_start_dt = dev_bin_starts[b_i]
				start_db = max(bin_start_dt - half_dev_bin_dt, 0)
				end_db = min(bin_start_dt + half_dev_bin_dt, seg_len)
				neur_fc = np.sum(shuffle_spikes_bin[:,start_db:end_db],1)
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
				mean_fr = np.mean(local_dev_bin_fr)
				std_fr = np.std(local_dev_bin_fr)
				cutoff = mean_fr + std_cutoff*std_fr
				#Calculate which bins are > mean + std_cutoff*std
				dev_neur_fr_locations = local_dev_bin_fr > cutoff*np.ones(np.shape(local_dev_bin_fr))
				dev_neur_fr_indices = np.where(dev_neur_fr_locations == True)[0]
				bin_devs[start_mean_bin_start_ind + dev_neur_fr_indices] += 1
				bin_dev_lens[start_mean_bin_start_ind + np.arange(len(dev_neur_fr_locations))] += 1
			avg_bin_devs = bin_devs/bin_dev_lens
			seg_dev_counts.extend([len(np.where(avg_bin_devs > dev_thresh)[0])])
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
			bout_lengths = bout_end_times - bout_start_times + int(deviation_bin_size*sampling_rate) #in samples
			bout_lengths_s = bout_lengths/sampling_rate #in Hz
			seg_dev_bout_lens.extend(list(bout_lengths_s))
			ibi = (bout_start_times[1:] - bout_end_times[:-1])/sampling_rate
			seg_dev_ibis.extend(list(ibi))
		null_segment_dev_counts.append([seg_dev_counts])
		null_segment_dev_ibis.append(seg_dev_ibis)
		null_segment_dev_bout_len.append(seg_dev_bout_lens)
	
	#Save results to .h5
	print("Saving results to .h5")
	#Save to .h5
	hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
	atom = tables.FloatAtom()
	for s_i in range(num_segments):
		seg_name = ('_').join(segment_names[s_i].split('-'))
		hf5.create_earray('/null_calcs',f'{seg_name}_counts',atom,(0,)+np.shape(null_segment_dev_counts[s_i][0]))
		seg_dev_expand = np.expand_dims(null_segment_dev_counts[s_i][0],0)
		exec("hf5.root.null_calcs."+f'{seg_name}_counts'+".append(seg_dev_expand)")
		seg_dev = np.array(null_segment_dev_ibis[s_i])
		hf5.create_earray('/null_calcs',f'{seg_name}_ibis',atom,(0,)+np.shape(seg_dev))
		seg_dev_expand = np.expand_dims(seg_dev,0)
		exec("hf5.root.null_calcs."+f'{seg_name}_ibis'+".append(seg_dev_expand)")
		seg_dev = np.array(null_segment_dev_bout_len[s_i])
		hf5.create_earray('/null_calcs',f'{seg_name}_lengths',atom,(0,)+np.shape(seg_dev))
		seg_dev_expand = np.expand_dims(seg_dev,0)
		exec("hf5.root.null_calcs."+f'{seg_name}_lengths'+".append(seg_dev_expand)")
	hf5.close()
	
	return null_segment_dev_counts, null_segment_dev_ibis, null_segment_dev_bout_len
		
def null_v_true_dev_plots(fig_save_dir,segment_names,segment_bouts,segment_bout_lengths,segment_ibis,num_null_sets,null_segment_dev_counts,null_segment_dev_ibis,null_segment_dev_bout_len):
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
	dev_save_dir = fig_save_dir + 'deviations/'
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
		fig_i = plt.figure()
		seg_true_dev_bout_lens = segment_bout_lengths[s_i]
		seg_null_dev_bout_lens = null_segment_dev_bout_len[s_i]
		plt.hist(seg_null_dev_bout_lens,bins=20,alpha=0.5,label='Null Data')
		plt.hist(seg_true_dev_bout_lens,bins=20,alpha=0.5,label='True Data')
		plt.axvline(np.mean(seg_true_dev_bout_lens),label='Mean of True Data')
		plt.legend()
		plt.xlabel('Deviation Bout Length (s)')
		plt.ylabel('Number of Deviations')
		im_name = (' ').join(segment_names[s_i].split('_'))
		plt.title(im_name + ' deviation lengths x null distribution')
		save_name = ('_').join(segment_names[s_i].split(' ')) + '_null_v_true_lens'
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.png')
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.svg')
		plt.tight_layout()
		plt.close(fig_i)
		#Histogram of IBIs
		fig_i = plt.figure()
		seg_true_dev_ibis = segment_ibis[s_i]
		seg_null_dev_bout_ibis = null_segment_dev_ibis[s_i]
		plt.hist(seg_null_dev_bout_ibis,bins=20,alpha=0.5,label='Null Data')
		plt.hist(seg_true_dev_ibis,bins=20,alpha=0.5,label='True Data')
		plt.axvline(np.mean(seg_true_dev_ibis),label='Mean of True Data')
		plt.legend()
		plt.xlabel('Deviation Inter-Bout-Interals (IBIs) (s)')
		plt.ylabel('Number of Deviations')
		im_name = (' ').join(segment_names[s_i].split('_'))
		plt.title(im_name + ' deviation IBIs x null distribution')
		save_name = ('_').join(segment_names[s_i].split(' ')) + '_null_v_true_ibis'
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.png')
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.svg')
		plt.tight_layout()
		plt.close(fig_i)
		#Histogram of deviation counts
		fig_i = plt.figure()
		seg_true_dev_count = len(segment_bout_lengths[s_i])
		seg_null_dev_counts = null_segment_dev_counts[s_i][0]
		plt.hist(seg_null_dev_counts,bins=20,alpha=0.5,label='Null Data')
		plt.axvline(seg_true_dev_count,label='True Count')
		plt.legend()
		plt.xlabel('Deviation Counts')
		plt.ylabel('Number of Deviations')
		im_name = (' ').join(segment_names[s_i].split('_'))
		plt.title(im_name + ' deviation counts x null distribution')
		save_name = ('_').join(segment_names[s_i].split(' ')) + '_null_v_true_counts'
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.png')
		fig_i.savefig(seg_null_hist_save_dir + save_name + '.svg')
		plt.tight_layout()
		plt.close(fig_i)
		
	#Now compare segment deviations against each other
	print("\tPlotting null distributions for all segments combined")
	cm_subsection = np.linspace(0,1,num_segments)
	cmap = [cm.gist_rainbow(x) for x in cm_subsection] #Color maps for each segment
	
	#Bout lengths
	mean_vals = []
	fig_lens = plt.figure(figsize=(10,10))
	for s_i in range(num_segments):
		segment_name = (' ').join(segment_names[s_i].split('_'))
		seg_true_dev_bout_lens = segment_bout_lengths[s_i]
		mean_true = np.mean(seg_true_dev_bout_lens)
		mean_vals.extend([mean_true])
		seg_null_dev_bout_lens = null_segment_dev_bout_len[s_i]
		plt.hist(seg_null_dev_bout_lens,bins=20,color=cmap[s_i],alpha=0.5,label=segment_name + ' null')
		plt.axvline(mean_true,color=cmap[s_i],label=segment_name + ' mean')
	plt.xlim((0,max(mean_vals) + 0.25))
	plt.legend()
	plt.title('cross-segment deviation lengths x null distribution')
	save_name ='all_seg_null_v_true_lengths'
	fig_lens.savefig(dev_save_dir + save_name + '.png')
	fig_lens.savefig(dev_save_dir + save_name + '.svg')
	plt.tight_layout()
	plt.close(fig_lens)
	
	#Bout ibis
	mean_vals = []
	fig_ibis = plt.figure(figsize=(10,10))
	for s_i in range(num_segments):
		segment_name = (' ').join(segment_names[s_i].split('_'))
		seg_true_dev_ibis = segment_ibis[s_i]
		mean_true = np.mean(seg_true_dev_ibis)
		mean_vals.extend([mean_true])
		seg_null_dev_bout_ibis = null_segment_dev_ibis[s_i]
		plt.hist(seg_null_dev_bout_ibis,bins=20,color=cmap[s_i],alpha=0.5,label=segment_name + ' null')
		plt.axvline(mean_true,color=cmap[s_i],label=segment_name + ' mean')
	#plt.xlim((0,max(mean_vals) + 0.25))
	plt.legend()
	plt.title('cross-segment deviation lengths x null distribution')
	save_name ='all_seg_null_v_true_ibis'
	fig_ibis.savefig(dev_save_dir + save_name + '.png')
	fig_ibis.savefig(dev_save_dir + save_name + '.svg')
	plt.tight_layout()
	plt.close(fig_ibis)
	
	#Bout counts
	true_counts = []
	fig_counts = plt.figure(figsize=(10,10))
	for s_i in range(num_segments):
		segment_name = (' ').join(segment_names[s_i].split('_'))
		seg_true_dev_count = len(segment_bout_lengths[s_i])
		seg_null_dev_counts = null_segment_dev_counts[s_i][0]
		true_counts.extend([seg_true_dev_count])
		plt.hist(seg_null_dev_counts,bins=20,color=cmap[s_i],alpha=0.5,label=segment_name + ' null')
		plt.axvline(seg_true_dev_count,color=cmap[s_i],label=segment_name + ' true')
	plt.legend()
	plt.title('cross-segment deviation counts x null distribution')
	save_name ='all_seg_null_v_true_counts'
	fig_counts.savefig(dev_save_dir + save_name + '.png')
	fig_counts.savefig(dev_save_dir + save_name + '.svg')
	plt.tight_layout()
	plt.close(fig_counts)



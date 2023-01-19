#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:54:17 2023

@author: hannahgermaine

This is a collection of functions for plotting data
"""

import os, tqdm
import numpy as np
import matplotlib.pyplot as plt

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
		s_name = segment_names[s_i]
		s_t = [list(np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= min_time)*(np.array(spike_times[i]) <= max_time))[0]]) for i in range(num_neur)]
		s_t_time = [list((1/60)*np.array(s_t[i])*(1/sampling_rate)) for i in range(len(s_t))]
		#Plot segment rasters and save
		plt.figure(figsize=(30,num_neur))
		plt.xlabel('Time (m)')
		plt.eventplot(s_t_time)
		plt.title(s_name + " segment")
		plt.tight_layout()
		im_name = ('_').join(s_name.split(' ')) + '.png'
		plt.savefig(raster_save_dir + im_name)
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
			plt.eventplot(s_t_time)
			plt.xlabel('Time (s)')
			plt.axvline(t_start[t_d_i]/sampling_rate,color='r')
			plt.axvline(t_end[t_d_i]/sampling_rate,color='r')
		#Save the figure
		im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_spike_rasters.png'
		t_fig.tight_layout()
		t_fig.savefig(raster_save_dir + im_name)
		plt.close(t_fig)
		tastant_spike_times.append(t_st)
	
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
		im_name = ('_').join((tastant_name).split(' ')) + '_PSTHs.png'
		t_fig.tight_layout()
		t_fig.savefig(PSTH_save_dir + im_name)
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
		im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_avg_PSTH.png'
		t_fig.tight_layout()
		t_fig.savefig(PSTH_save_dir + im_name)
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
		plt.savefig(neuron_PSTH_dir + 'neuron_' + str(n_i) + '_PSTHs.png')
		plt.close()
		
	return PSTH_times, PSTH_taste_deliv_times, tastant_PSTH, avg_tastant_PSTH

def FR_deviation_plots(fig_save_dir,sampling_rate,segment_names,segment_times,
					      segment_spike_times,num_neur,num_tastes,local_bin_size,
						  deviation_bin_size):
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
	OUTPUTS:
		- segment_deviations: 
	"""
	print("\nBeginning firing rate deviation calculations.")
	#Create save directory
	dev_save_dir = fig_save_dir + 'deviations/'
	if os.path.isdir(dev_save_dir) == False:
		os.mkdir(dev_save_dir)
	#Convert the bin sizes from time to samples
	num_segments = len(segment_names)
	local_bin_dt = int(np.ceil(local_bin_size*sampling_rate))
	half_local_bin_dt = int(np.ceil(local_bin_dt/2))
	dev_bin_dt = int(np.ceil(deviation_bin_size*sampling_rate))
	half_dev_bin_dt = int(np.ceil(dev_bin_dt/2))
	#For each segment, a bin for a window size of local_bin_size will slide 
	#through time. At each slide the firing rates of all the small bins within
	#the local window are calculated. Those that are above 2 std. from the mean
	#are marked as being deviating bins
	segment_devs = []
	for i in range(num_segments):
		print("\tCalculating deviations for segment " + segment_names[i])
		segment_spikes = segment_spike_times[i]
		#Generate arrays of start times for calculating the deviation from the mean
		start_segment = segment_times[i]
		end_segment = segment_times[i+1]
		dev_bin_starts = np.arange(start_segment,end_segment,dev_bin_dt)
		#First calculate the firing rates of all small bins
		bin_fcs = np.zeros((num_neur,len(dev_bin_starts)))
		bin_frs = np.zeros((num_neur,len(dev_bin_starts)))
		for b_i in tqdm.tqdm(range(len(dev_bin_starts))):
			bin_start_dt = dev_bin_starts[b_i]
			start_db = max(bin_start_dt - half_dev_bin_dt, start_segment)
			end_db = min(bin_start_dt + half_dev_bin_dt, end_segment)
			neur_fc = [len(np.where((np.array(segment_spikes[n_i]) < end_db) & (np.array(segment_spikes[n_i]) > start_db))[0]) for n_i in range(num_neur)]
			bin_fcs[:,b_i] = np.array(neur_fc)
			bin_frs[:,b_i] = bin_fcs[:,b_i]/deviation_bin_size
		#Next slide a larger window over the small bins and calculate deviations for each small bin
		bin_devs = np.zeros((num_neur,len(dev_bin_starts))) #storage array for deviations from mean
		bin_dev_lens = np.zeros(np.shape(bin_devs))
		for b_i in tqdm.tqdm(range(len(dev_bin_starts))): #slide a mean window over all the starts and calculate the small bin firing rate's deviation
			bin_start_dt = dev_bin_starts[b_i]
			#First calculate mean interval bounds
			start_mean_int = max(bin_start_dt - half_local_bin_dt,0)
			start_mean_bin_start_ind = np.where(np.abs(dev_bin_starts - start_mean_int) == np.min(np.abs(dev_bin_starts - start_mean_int)))[0][0]
			end_mean_int = min(start_mean_int + local_bin_dt,end_segment-1)
			end_mean_bin_start_ind = np.where(np.abs(dev_bin_starts - end_mean_int) == np.min(np.abs(dev_bin_starts - end_mean_int)))[0][0]
			#Next calculate mean + std FR for the interval
			local_dev_bin_fr = bin_frs[:,np.arange(start_mean_bin_start_ind,end_mean_bin_start_ind)]
			mean_fr = np.mean(local_dev_bin_fr,axis=1)
			std_fr = np.std(local_dev_bin_fr,axis=1)
			cutoff = np.expand_dims(mean_fr + 2*std_fr,1)
			#Calculate which bins are > mean + 2std
			dev_neur_fr_locations = local_dev_bin_fr > cutoff*np.ones(np.shape(local_dev_bin_fr))
			dev_neur_fr_indices = np.where(dev_neur_fr_locations == True)
			for fr_i in range(len(dev_neur_fr_indices[0])):
				bin_devs[dev_neur_fr_indices[0][fr_i],start_mean_bin_start_ind + dev_neur_fr_indices[1][fr_i]] += 1
			for n_i in range(num_neur):
				bin_dev_lens[n_i,start_mean_bin_start_ind + np.arange(len(dev_neur_fr_locations[n_i,:]))] += 1
		avg_bin_devs = bin_devs/bin_dev_lens
		segment_devs.append([dev_bin_starts,avg_bin_devs])
	#Now plot deviations
	for i in tqdm.tqdm(range(num_segments)):
		print("\tPlotting deviations for segment " + segment_names[i])
		fig_i = plt.figure(figsize=(30,num_neur))
		plt.imshow(segment_devs[i][1],cmap='jet',aspect='auto')
		plt.colorbar()
		im_name = (' ').join(segment_names[i].split('_'))
		plt.title(im_name + ' deviation fractions')
		save_name = ('_').join(segment_names[i].split(' ')) + '_devs.png'
		fig_i.savefig(dev_save_dir + save_name)
		plt.close(fig_i)
		fig_i = plt.figure(figsize=(30,num_neur))
		plt.imshow((segment_devs[i][1]>0.9).astype('uint8'),cmap='jet',aspect='auto')
		plt.colorbar()
		im_name = (' ').join(segment_names[i].split('_'))
		plt.title(im_name + ' strong deviations (>0.9)')
		save_name = ('_').join(segment_names[i].split(' ')) + '_high_devs.png'
		fig_i.savefig(dev_save_dir + save_name)
		plt.close(fig_i)
	#Calculate the deviation bout size and frequency
	segment_bout_lengths = []
	segment_ibis = []
	for i in tqdm.tqdm(range(num_segments)):
		print("\Calculating deviation bout sizes and frequencies")
		seg_devs = segment_devs[i][1]
		seg_times = segment_devs[i][0]
		dev_inds = [np.where(seg_devs[n_i,:] > 0)[0] for n_i in range(num_neur)]
		dev_times = [seg_times[dev_inds[n_i]] for n_i in range(num_neur)]
		bout_starts = [dev_inds[n_i][np.where(np.diff(dev_inds[n_i]) > 1)[0]] for n_i in range(num_neur)]
		bout_start_inds = [list(np.concatenate((np.array([0]),bout_starts[n_i] + 1),axis=0)) for n_i in range(num_neur)]
		bout_ends = [dev_inds[n_i][np.where(np.diff(dev_inds[n_i]) > 1)[0]] - 1 for n_i in range(num_neur)]
		bout_end_inds = [list(np.concatenate((bout_ends[n_i] + 1,np.array([dev_inds[n_i][-1]])),axis=0)) for n_i in range(num_neur)]
		bout_start_times = [dev_times[bout_start_inds[n_i]] for n_i in range(num_neur)]
		bout_end_times = [dev_times[bout_end_inds[n_i]] for n_i in range(num_neur)]
		bout_lengths = [bout_end_times[n_i] - bout_start_times[n_i] for n_i in range(num_neur)] #in samples
		bout_lengths_s = [bout_lengths[n_i]/sampling_rate for n_i in range(num_neur)]#in Hz
		segment_bout_lengths.append(bout_lengths_s)
		ibi = [(bout_start_times[n_i][1:] - bout_end_times[n_i][:-1])/sampling_rate for n_i in range(num_neur)]
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
		fig_i.savefig(dev_save_dir + save_name)
		plt.close(fig_i)
		
	return segment_devs, segment_bout_lengths, segment_ibis
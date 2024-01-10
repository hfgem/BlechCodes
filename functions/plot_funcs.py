#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:54:17 2023

@author: hannahgermaine

This is a collection of miscellaneous functions for plotting data
"""

import os, tqdm, itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.stats as stats
import matplotlib.cm as cm

def raster_plots(fig_save_dir, dig_in_names, start_dig_in_times, end_dig_in_times, 
				 segment_names, segment_times, segment_spike_times, tastant_spike_times, 
				 pre_taste_dt, post_taste_dt, num_neur, num_tastes):
	
	#_____Grab spike times (and rasters) for each segment separately_____
	raster_save_dir = fig_save_dir + 'rasters/'
	if os.path.isdir(raster_save_dir) == False:
		os.mkdir(raster_save_dir)
	for s_i in tqdm.tqdm(range(len(segment_names))):
		print("\nGrabbing spike raster for segment " + segment_names[s_i])
		min_time = segment_times[s_i] #in ms
		max_time = segment_times[s_i+1] #in ms
		max_time_min = (max_time-min_time)*(1/1000)*(1/60)
		s_name = segment_names[s_i]
		s_t = segment_spike_times[s_i]
		s_t_time = [list((1/60)*np.array(s_t[i])*(1/1000)) for i in range(len(s_t))]
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

	#_____Grab spike times for each taste delivery separately_____
	for t_i in tqdm.tqdm(range(num_tastes)):
		print("\nGrabbing spike rasters for tastant " + dig_in_names[t_i] + " deliveries")
		rast_taste_save_dir = raster_save_dir + ('_').join((dig_in_names[t_i]).split(' ')) + '/'
		if os.path.isdir(rast_taste_save_dir) == False:
			os.mkdir(rast_taste_save_dir)
		rast_taste_deliv_save_dir = rast_taste_save_dir + 'deliveries/'
		if os.path.isdir(rast_taste_deliv_save_dir) == False:
			os.mkdir(rast_taste_deliv_save_dir)
		t_start = np.array(start_dig_in_times[t_i])*(1/1000) #Convert to seconds
		t_end = np.array(end_dig_in_times[t_i])*(1/1000) #Convert to seconds
		num_deliv = len(t_start)
		t_st = tastant_spike_times[t_i]
		for t_d_i in range(len(t_start)):
			deliv_fig = plt.figure(figsize=(5,5))
			s_t = t_st[t_d_i]
			s_t_time = [list(np.array(s_t[i])*(1/1000)) for i in range(len(s_t))] #Convert to seconds
			t_st.append(s_t)
			#Plot the raster
			plt.eventplot(s_t_time,colors='k')
			plt.xlabel('Time (s)')
			plt.axvline(t_start[t_d_i],color='r')
			plt.axvline(t_end[t_d_i],color='r')
			im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_raster_' + str(t_d_i)
			deliv_fig.tight_layout()
			deliv_fig.savefig(rast_taste_deliv_save_dir + im_name + '.png')
			deliv_fig.savefig(rast_taste_deliv_save_dir + im_name + '.svg')
			plt.close(deliv_fig)
		t_fig = plt.figure(figsize=(10,num_deliv))
		for t_d_i in range(len(t_start)):
			#Grab spike times into one list
			s_t = t_st[t_d_i]
			s_t_time = [list(np.array(s_t[i])*(1/1000)) for i in range(len(s_t))]
			t_st.append(s_t)
			#Plot the raster
			plt.subplot(num_deliv,1,t_d_i+1)
			plt.eventplot(s_t_time,colors='k')
			plt.xlabel('Time (s)')
			plt.axvline(t_start[t_d_i]/1000,color='r')
			plt.axvline(t_end[t_d_i]/1000,color='r')
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
			#n_st = [t_st[t_d_i][n_i] - t_start[t_d_i] for t_d_i in range(len(t_start))]
			n_st = [t_st[t_d_i][n_i] - t_start[t_d_i]*1000 for t_d_i in range(len(t_start))]
			n_st_time = [list(np.array(n_st[i])*(1/1000)) for i in range(len(n_st))]
			t_fig = plt.figure(figsize=(10,10))
			plt.eventplot(n_st_time,colors='k')
			plt.xlabel('Time (s)')
			plt.ylabel('Trial')
			plt.axvline(pre_taste_dt/1000,color='r')
			#Save the figure
			im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_unit_' + str(n_i)
			t_fig.tight_layout()
			t_fig.savefig(rast_neur_save_dir + im_name + '.png')
			t_fig.savefig(rast_neur_save_dir + im_name + '.svg')
			plt.close(t_fig)
	
	return segment_spike_times, tastant_spike_times, pre_taste_dt, post_taste_dt

def PSTH_plots(fig_save_dir, num_tastes, num_neur, dig_in_names, 
			   start_dig_in_times, end_dig_in_times, pre_taste_dt, post_taste_dt,
			   segment_times, segment_spike_times, bin_width, bin_step):
	
	PSTH_save_dir = fig_save_dir + 'PSTHs/'
	if os.path.isdir(PSTH_save_dir) == False:
		os.mkdir(PSTH_save_dir)
	half_bin_width_dt = int(np.ceil(1000*bin_width/2)) #in ms
	PSTH_times = [] #Storage of time bin true times (s) for each tastant
	PSTH_taste_deliv_times = [] #Storage of tastant delivery true times (s) for each tastant [start,end]
	
	#First determine the minimal start delivery and maximal end delivery
	all_start_dig_in_times = []
	for sdit in range(len(start_dig_in_times)):
		all_start_dig_in_times.extend(start_dig_in_times[sdit])
	min_start_dig_in_time = np.min(np.array(all_start_dig_in_times))
	
	all_end_dig_in_times = []
	for edit in range(len(end_dig_in_times)):
		all_end_dig_in_times.extend(end_dig_in_times[edit])
	
	#Next determine which segment encapsulates these as the taste segment
	#Note: this assumes all taste deliveries fall within one segment (and they should!!)
	closest_seg_start = np.argmin(np.abs(segment_times - min_start_dig_in_time))
	segment_start = segment_times[closest_seg_start]
	raster_start = int(segment_start - pre_taste_dt - 1)
	raster_end = int(segment_times[closest_seg_start + 1] + post_taste_dt + 1) #Expand just in case
	raster_len = int(raster_end - raster_start)
	raster_array = np.zeros((num_neur,raster_len)) #Create binary storage array
	segment_spikes = segment_spike_times[closest_seg_start] #Grab spike times
	for n_i in range(num_neur):
		neur_segment_spikes = np.array(segment_spikes[n_i]).astype('int') - raster_start
		raster_array[n_i,neur_segment_spikes] = 1 #Store to binary array
		
	#Generate PSTH of the binary array for the full taste segment
	print("Grabbing taste interval firing rates")
	firing_rate_array = np.array([(np.sum(raster_array[:,max(t_i-half_bin_width_dt,0):min(t_i+half_bin_width_dt,raster_len)],1)/bin_width).T for t_i in tqdm.tqdm(range(raster_len))]).T
	
	print("Pulling firing rate arrays for each tastant delivery")	
	tastant_PSTH = [] #List of numpy arrays of num_deliv x num_neur x length
	avg_tastant_PSTH = [] #List of numpy arrays of num_neur x length
	for t_i in tqdm.tqdm(range(num_tastes)):
		print("\nGrabbing PSTHs for tastant " + dig_in_names[t_i] + " deliveries")
		t_start = np.array(start_dig_in_times[t_i])
		t_end = np.array(end_dig_in_times[t_i])
		dt_total = int(np.max(t_end-t_start) + pre_taste_dt + post_taste_dt)
		num_deliv = len(t_start)
		PSTH_start_times = np.arange(0,dt_total)
		PSTH_true_times = np.round(PSTH_start_times/1000,3)
		PSTH_times.append(PSTH_true_times)
		start_deliv_interval = PSTH_true_times[np.where(PSTH_start_times > pre_taste_dt)[0][0]]
		end_deliv_interval = PSTH_true_times[np.where(PSTH_start_times > dt_total - post_taste_dt)[0][0]]
		PSTH_taste_deliv_times.append([start_deliv_interval,end_deliv_interval])
		all_PSTH = np.zeros((num_deliv,num_neur,len(PSTH_start_times)))
		t_fig = plt.figure(figsize=(10,num_deliv))
		for t_d_i in range(len(t_start)):
			start_i = max(int(t_start[t_d_i] - pre_taste_dt) - raster_start,0)
			end_i = start_i + dt_total
			#Perform Gaussian convolution
			PSTH_spikes = firing_rate_array[:,start_i:end_i]
			plt.subplot(num_deliv,1,t_d_i+1)
			#Update to have x-axis in time
			for i in range(num_neur):
				plt.plot(PSTH_true_times,PSTH_spikes[i,:])
			del i
			plt.axvline(start_deliv_interval,color='r')
			plt.axvline(end_deliv_interval,color='r')
			all_PSTH[t_d_i,:,:] = PSTH_spikes
		del t_d_i, start_i, end_i, PSTH_spikes
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
		
	#_____Plot avg tastant PSTHs side-by-side
	f_psth, ax = plt.subplots(1,num_tastes,figsize=(10,10),sharex=True,sharey=True)
	for t_i in range(num_tastes):
		tastant_name = dig_in_names[t_i]
		PSTH_true_times = PSTH_times[t_i]
		for i in range(num_neur):
			ax[t_i].plot(PSTH_true_times,avg_tastant_PSTH[t_i][i])
		ax[t_i].axvline(PSTH_taste_deliv_times[t_i][0],color='r',linestyle='dashed')
		ax[t_i].axvline(PSTH_taste_deliv_times[t_i][1],color='r',linestyle='dashed')
		ax[t_i].set_xlabel('Time (s)')
		ax[t_i].set_ylabel('Firing Rate (Hz)')
		ax[t_i].set_title(tastant_name)
	im_name = 'combined_avg_PSTH'
	f_psth.tight_layout()
	f_psth.savefig(PSTH_save_dir + im_name + '.png')
	f_psth.savefig(PSTH_save_dir + im_name + '.svg')
	plt.close(f_psth)
	
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

def LFP_dev_plots(fig_save_dir,segment_names,segment_times,fig_buffer_size,segment_bouts,combined_waveforms,wave_sampling_rate):
	"""This function plots the LFP spectrogram data in intervals surrounding the
	location of deviations
	INPUTS:
		- fig_save_dir: directory to save visualizations
		- segment_names: names of different experiment segments
		- segment_times: time indices of different segment starts/ends
		- fig_buffer_size: how much (in seconds) to plot before and after a deviation event
		- segment_bouts: bouts of time in which deviations occur
		- combined_waveforms: 0-3000 Hz range from recording
		- wave_sampling_rate: sampling rate of waveform data
	OUTPUTS:
		- Figures containing spectrograms and waveforms of LFP data surrounding deviation times
	"""
	
	print("\nBeginning individual deviation segment plots.")
	#Create save directory
	dev_save_dir = fig_save_dir + 'deviations/'
	if os.path.isdir(dev_save_dir) == False:
		os.mkdir(dev_save_dir)
	#Convert the bin size from time to samples
	sampling_rate_ratio = wave_sampling_rate/1000
	num_segments = len(segment_names)
	[num_neur,num_time] = np.shape(combined_waveforms)
	local_bin_dt = int(np.ceil(fig_buffer_size*wave_sampling_rate))
	half_local_bin_dt = int(np.ceil(local_bin_dt/2))
	spect_NFFT = int(wave_sampling_rate*0.05)  # 10ms window
	spect_overlap = 20
	max_recording_time = len(combined_waveforms[0,:])
	#Run through deviation times by segment and plot rasters
	for s_i in range(num_segments):
		print("\nGrabbing waveforms for segment " + segment_names[s_i])
		seg_dev_save_dir = dev_save_dir + ('_').join(segment_names[s_i].split(' ')) + '/'
		if os.path.isdir(seg_dev_save_dir) == False:
			os.mkdir(seg_dev_save_dir)
		seg_wav_save_dir = seg_dev_save_dir + 'dev_waveforms/'
		if os.path.isdir(seg_wav_save_dir) == False:
			os.mkdir(seg_wav_save_dir)
		seg_spect_save_dir = seg_dev_save_dir + 'dev_spectrograms/'
		if os.path.isdir(seg_spect_save_dir) == False:
			os.mkdir(seg_spect_save_dir)
		segment_dev_start_times = segment_bouts[s_i][:,0]*sampling_rate_ratio #Convert all segment times to the waveform sampling rate times
		segment_dev_end_times = segment_bouts[s_i][:,1]*sampling_rate_ratio #Convert all segment times to the waveform sampling rate times
		spect_f = []
		spect_dev_t = []
		for d_i in tqdm.tqdm(range(len(segment_dev_start_times))):
			min_time = int(max(segment_dev_start_times[d_i] - half_local_bin_dt,0))
			max_time = int(min(segment_dev_end_times[d_i] + half_local_bin_dt,max_recording_time))
			dev_start_ind = segment_dev_start_times[d_i] - min_time
			dev_start_time = dev_start_ind/wave_sampling_rate
			len_dev = (segment_dev_end_times[d_i] - segment_dev_start_times[d_i])
			len_dev_time = len_dev/wave_sampling_rate
			dev_waveforms = combined_waveforms[:,int(min_time):int(max_time)]
			avg_waveform = np.mean(dev_waveforms,0)
			#Plot segment deviation raster
			plt.figure(figsize=(10,num_neur))
			plt.xlabel('Time (s)')
			plt.ylabel('Neuron Index')
			for n_i in range(num_neur):
				plt.subplot(num_neur,1,n_i+1)
				plt.plot((1/wave_sampling_rate)*np.arange(min_time,max_time),dev_waveforms[n_i,:])
				plt.axvline((1/wave_sampling_rate)*(min_time + dev_start_ind),color='r')
				plt.axvline((1/wave_sampling_rate)*(min_time + dev_start_ind + len_dev),color='r')
			plt.suptitle('Deviation ' + str(d_i))
			plt.tight_layout()
			im_name = 'dev_' + str(d_i) + '.png'
			plt.savefig(seg_wav_save_dir + im_name)
			plt.close()
			#Plot LFP spectrogram
			plt.figure(figsize=(num_neur,num_neur))
			f, t, Sxx = signal.spectrogram(avg_waveform, wave_sampling_rate, nfft=spect_NFFT, noverlap=spect_overlap)
			max_freqs = [f[np.argmax(Sxx[:,i])] for i in range(len(t))]
			spect_f.append(max_freqs)
			start_dev_int = (1/wave_sampling_rate)*(min_time + dev_start_ind)
			end_dev_int = (1/wave_sampling_rate)*(min_time + dev_start_ind + len_dev)
			ind_plot = np.where(f < 300)[0]
			plt.subplot(1,2,1)
			plt.plot((1/wave_sampling_rate)*np.arange(min_time,max_time),avg_waveform)
			plt.axvline(start_dev_int,color='r')
			plt.axvline(end_dev_int,color='r')
			plt.title('Average Waveform')
			plt.subplot(1,2,2)
			plt.pcolormesh(t, f[ind_plot], Sxx[ind_plot,:], shading='gouraud')
			plt.ylabel('Frequency [Hz]')
			plt.xlabel('Time [sec]')
			start_dev_int_t = np.argmin(np.abs(t - dev_start_time))
			end_dev_int_t = np.argmin(np.abs(t - (dev_start_time + len_dev_time)))
			spect_dev_t.append([start_dev_int_t,end_dev_int_t])
			plt.axvline(dev_start_time,color='r')
			plt.axvline(dev_start_time + len_dev_time,color='r')
			plt.title('Spectrogram')
			plt.tight_layout()
			im_name = 'dev_' + str(d_i) + '.png'
			plt.savefig(seg_spect_save_dir + im_name)
			plt.close()
		#Now look at average spectrogram around the taste delivery interval
		plt.figure(figsize=(10,10))
		plt.subplot(1,2,1)
		for d_i in range(len(segment_dev_start_times)):
			dev_start_t = int(spect_dev_t[d_i][0])
			dev_freq_vals = spect_f[d_i]
			t_vals = np.concatenate((np.arange(-dev_start_t,0),np.arange(len(dev_freq_vals)-dev_start_t)))
			plt.plot(t_vals,dev_freq_vals,alpha=0.05)
		plt.axvline(0)
		plt.title('Dev Start Aligned')
		plt.xlabel('Aligned time (s)')
		plt.ylabel('Spectrogram Max Frequency (Hz)')
		plt.subplot(1,2,2)
		for d_i in range(len(segment_dev_start_times)):
			dev_end_t = int(spect_dev_t[d_i][1])
			dev_freq_vals = spect_f[d_i]
			t_vals = np.concatenate((np.arange(-dev_end_t,0),np.arange(len(dev_freq_vals)-dev_end_t)))
			plt.plot(t_vals,dev_freq_vals,alpha=0.05)
		plt.axvline(0)
		plt.title('Dev End Aligned')
		plt.xlabel('Aligned time (s)')
		plt.ylabel('Spectrogram Max Frequency (Hz)')
		plt.tight_layout()
		im_name = 'aligned_max_frequencies'
		plt.savefig(seg_spect_save_dir + im_name + '.png')
		plt.savefig(seg_spect_save_dir + im_name + '.svg')
		plt.close()
	
def taste_select_success_plot(taste_select_prob_joint, x_vals, x_label, name, save_dir):
	#Calculate the fraction of successful decoding by neuron by taste
	num_neur, num_tastes = np.shape(taste_select_prob_joint)
	chance = 1/num_tastes
	above_chance_all = (taste_select_prob_joint > chance).astype('int')
	num_tastes_above_chance = np.sum(above_chance_all,1)
	#Plot the successful decoding taste count
	plt.figure(figsize=(8,8))
	plt.imshow(num_tastes_above_chance.T)
	plt.xticks(np.arange(len(x_vals)),labels=x_vals,rotation=90)
	plt.yticks(np.arange(num_neur))
	plt.xlabel(x_label)
	plt.ylabel('Neuron Index')
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(save_dir + name + '.png')
	plt.savefig(save_dir + name + '.svg')
	plt.close()
	#Plot the summed successful decoding to show where the most occurs
	plt.figure(figsize=(8,8))
	summed_val = np.sum(num_tastes_above_chance.T,0)
	max_ind = np.argmax(summed_val)
	max_val = x_vals[max_ind]
	plt.plot(x_vals,summed_val,label='summed data')
	plt.axvline(max_val,label=max_val,linestyle='dashed',color='r')
	plt.legend()
	plt.xlabel(x_label)
	plt.ylabel('Summed Decoding')
	plt.tight_layout()
	plt.savefig(save_dir + name + '_summed.png')
	plt.savefig(save_dir + name + '_summed.svg')
	plt.close()
	
	max_neur = np.where(num_tastes_above_chance[max_ind,:] > 1)[0]
	
	return max_val, max_neur

def epoch_taste_select_plot(prob_taste_epoch, dig_in_names, save_dir):
	num_neur, num_tastes, num_deliv, num_cp = np.shape(prob_taste_epoch)
	colors_taste = cm.cool(np.arange(num_tastes)/(num_tastes))
	colors_cp = cm.summer(np.arange(num_cp)/(num_cp))
	#Plot all trial and all taste successful decoding probability
	for t_i in range(num_tastes):
		fig_t, axes = plt.subplots(nrows=1,ncols=num_cp+1,figsize=(15,5),gridspec_kw=dict(width_ratios=[5,5,5,1]))
		for e_i in range(num_cp):
			im = axes[e_i].imshow(prob_taste_epoch[:,t_i,:,e_i],vmin=0, vmax=1)
			axes[e_i].set_title('Taste ' + str(dig_in_names[t_i]) + ' Epoch ' + str(e_i))
		#fig_t.subplots_adjust(left = 0.05, right = 0.9)
		fig_t.colorbar(im, cax = axes[num_cp])
		fig_t.savefig(save_dir + 'taste_'+str(t_i)+'_epoch_taste_selectivity.png')
		fig_t.savefig(save_dir + 'taste_'+str(t_i)+'_epoch_taste_selectivity.svg')
		plt.close(fig_t)
	#Plot distributions of decoding probability for each taste for each neuron
	cross_epoch_dir = save_dir + 'epoch_prob_dist/'
	if os.path.isdir(cross_epoch_dir) == False:
		os.mkdir(cross_epoch_dir)
	for n_i in range(num_neur):
		for t_i in range(num_tastes):
			fig_1 = plt.figure(figsize=(5,5))
			dist_collect = []
			pairs = list(itertools.combinations(np.arange(num_cp),2))
			for c_p in range(num_cp):
				c_p_data = prob_taste_epoch[n_i,t_i,:,c_p]
				dist_collect.append(c_p_data)
				c_p_data = c_p_data[~np.isnan(c_p_data)]
				c_p_data = c_p_data[~np.isinf(c_p_data)]
				if len(c_p_data) > 1:
					plt.hist(c_p_data,bins=num_deliv,density = True,histtype='step',cumulative=True,color=colors_cp[c_p],label='Epoch ' + str(c_p))
			args = [d for d in dist_collect]
			x_ticks = plt.xticks()[0]
			x_tick_diff = np.mean(np.diff(x_ticks))
			y_ticks = plt.yticks()[0]
			y_tick_diff = np.mean(np.diff(y_ticks))
			#cross-group stat sig
			try:
				kw_stat, kw_p_val = stats.kruskal(*args,nan_policy='omit')
			except:
				kw_p_val = 1
			if kw_p_val <= 0.05:
				plt.scatter(np.mean(x_ticks),np.max(y_ticks),marker='*',color='k',s=100)
			#pairwise stat sig
			for pair_i in range(len(pairs)):
				pair = pairs[pair_i]
				ks_pval = stats.ks_2samp(dist_collect[pair[0]],dist_collect[pair[1]])[1]
				if ks_pval < 0.05:
					plt.scatter(np.mean(x_ticks) + (pair_i-(len(pairs)/2))*x_tick_diff/3,np.max(y_ticks) - y_tick_diff/3,marker='*',color=colors_cp[pair[0]],edgecolors=colors_cp[pair[1]],s=100)
			plt.legend()
			plt.xlabel('Decoding Probability')
			plt.ylabel('Cumulative Density of Occurrence')
			fig_1.savefig(cross_epoch_dir + 'neuron_' + str(n_i) + '_taste_' + str(t_i) + '.png')
			fig_1.savefig(cross_epoch_dir + 'neuron_' + str(n_i) + '_taste_' + str(t_i) + '.svg')
			plt.close(fig_1)
		for c_p in range(num_cp):
			fig_2 = plt.figure(figsize=(5,5))
			dist_collect = []
			pairs = list(itertools.combinations(np.arange(num_tastes),2))
			for t_i in range(num_tastes):
				c_p_data = prob_taste_epoch[n_i,t_i,:,c_p]
				dist_collect.append(c_p_data)
				c_p_data = c_p_data[~np.isnan(c_p_data)]
				c_p_data = c_p_data[~np.isinf(c_p_data)]
				if len(c_p_data) > 1:
					plt.hist(c_p_data,bins=num_deliv,density = True,histtype='step',cumulative=True,color=colors_taste[t_i],label=dig_in_names[t_i])
			args = [d for d in dist_collect]
			x_ticks = plt.xticks()[0]
			x_tick_diff = np.mean(np.diff(x_ticks))
			y_ticks = plt.yticks()[0]
			y_tick_diff = np.mean(np.diff(y_ticks))
			#cross-group stat sig
			try:
				kw_stat, kw_p_val = stats.kruskal(*args,nan_policy='omit')
			except:
				kw_p_val = 1
			if kw_p_val <= 0.05:
				plt.scatter(np.mean(x_ticks),np.max(y_ticks),marker='*',color='k',s=100)
			#pairwise stat sig
			for pair_i in range(len(pairs)):
				pair = pairs[pair_i]
				ks_pval = stats.ks_2samp(dist_collect[pair[0]],dist_collect[pair[1]])[1]
				if ks_pval < 0.05:
					plt.scatter(np.mean(x_ticks) + (pair_i-(len(pairs)/2))*x_tick_diff/3,np.max(y_ticks) - y_tick_diff/3,marker='*',color=colors_taste[pair[0]],edgecolors=colors_taste[pair[1]],s=100)
			plt.legend()
			plt.xlabel('Decoding Probability')
			plt.ylabel('Cumulative Density of Occurrence')
			fig_2.savefig(cross_epoch_dir + 'neuron_' + str(n_i) + '_epoch_' + str(c_p) + '.png')
			fig_2.savefig(cross_epoch_dir + 'neuron_' + str(n_i) + '_epoch_' + str(c_p) + '.svg')
			plt.close(fig_2)
			
			
			
		


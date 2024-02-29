#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:54:30 2024

@author: Hannah Germaine

File dedicated to functions related to decoding of tastes from burst events
"""

import numpy as np
import tqdm, os, warnings, itertools, time, random
#file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
#os.chdir(file_path)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate
from multiprocess import Pool
from p_tqdm import p_map
import functions.decode_parallel as dp
from sklearn.mixture import GaussianMixture as gmm
from scipy.stats import pearsonr
from random import sample

def decode_epochs(tastant_fr_dist,segment_spike_times,post_taste_dt,pre_taste_dt,
				   skip_dt,dig_in_names,segment_times,
				   segment_dev_rasters, segment_dev_times,
				   segment_names,start_dig_in_times,taste_num_deliv,
				   taste_select_epoch,use_full,max_decode,max_hz,save_dir,
				   neuron_count_thresh,trial_start_frac=0,
				   epochs_to_analyze=[],segments_to_analyze=[]):		
	"""Decode taste from epoch-specific firing rates"""
	#Variables
	num_tastes = len(start_dig_in_times)
	num_neur = len(segment_spike_times[0])
	max_num_deliv = np.max(taste_num_deliv).astype('int')
	num_cp = len(tastant_fr_dist[0][0])
	num_segments = len(segment_spike_times)
	hist_bins = np.arange(stop=max_hz+1,step=0.25)
	x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
	p_taste = taste_num_deliv/np.sum(taste_num_deliv) #P(taste)
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(num_segments)

	#If trial_start_frac > 0 use only trials after that threshold
	trial_start_ind = np.floor(max_num_deliv*trial_start_frac).astype('int')
	new_max_num_deliv = max_num_deliv - trial_start_ind

	for e_i in epochs_to_analyze: #By epoch conduct decoding
		print('Decoding Epoch ' + str(e_i))
		
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		
		#Fit gmm distributions to fr of each population for each taste
		#P(firing rate | taste) w/ inter-neuron dependencies
		fit_tastant_neur = dict()
		for t_i in range(num_tastes):
			full_data = []
			for d_i in range(max_num_deliv):
				if d_i >= trial_start_ind:
					full_data.extend(tastant_fr_dist[t_i][d_i-trial_start_ind][e_i])
			gm = gmm(n_components=1, n_init=10).fit(full_data)
			fit_tastant_neur[t_i] = gm
					
		#Fit gmm distribution to fr of all tastes
		#P(firing rate) w/ inter-neuron dependencies
		full_data = []
		for t_i in range(num_tastes):
			for d_i in range(max_num_deliv):
				if d_i >= trial_start_ind:
					full_data.extend(tastant_fr_dist[t_i][d_i-trial_start_ind][e_i])
		gm = gmm(n_components=1,  n_init=10).fit(full_data)
		fit_all_neur = gm
		
		#Segment-by-segment use deviation times to zoom in and test decoding
		epoch_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_save_dir):
			os.mkdir(epoch_save_dir)
			
		#Create save arrays for decoding stats
		seg_decode_percents = np.zeros((num_tastes,len(segments_to_analyze)))
		seg_decode_percents_full = np.zeros((num_tastes,len(segments_to_analyze)))
		
		for s_ind, s_i in enumerate(segments_to_analyze):
			seg_save_dir = epoch_save_dir + 'segment_' + str(s_i) + '/'
			if not os.path.isdir(seg_save_dir):
				os.mkdir(seg_save_dir)
			#Get segment variables
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = segment_times[s_i+1] - segment_times[s_i] #in dt = ms
			#Pull segment deviations
			seg_deviations = segment_dev_rasters[s_i]
			seg_deviation_times = segment_dev_times[s_i]
			num_devs = len(seg_deviations)
			#Binerize Spike Times
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in taste_select_neur:
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			#Grab neuron firing rates in sliding bins
			dev_fr_save_dir = seg_save_dir + 'dev_fr/'
			if not os.path.isdir(dev_fr_save_dir):
				os.mkdir(dev_fr_save_dir)
			try:
				burst_decode_array = np.load(epoch_save_dir + 'segment_' + str(s_i) + '_burst_decode.npy')
				print('\tSegment ' + str(s_i) + ' Previously Decoded')
			except:
				print('\tDecoding Segment ' + str(s_i) + ' Bursts')
				#Perform parallel computation for each time bin
				print('\t\tCalculate firing rates for time bins')
				try:
					dev_fr = []
					for dev_i in range(num_devs):
						dev_fr_i = np.load(dev_fr_save_dir + 'dev_' + str(dev_i) + '.npy')
						dev_fr.append(dev_fr_i)
				except:
					dev_fr = []
					for dev_i in range(num_devs):
						dev_start_i = seg_deviation_times[0,dev_i]
						dev_end_i = seg_deviation_times[1,dev_i]
						dev_len = dev_end_i - dev_start_i
						dev_fr_i = np.sum(segment_spike_times_s_i_bin[:,dev_start_i:dev_end_i],1)/(dev_len*(1/1000))
						np.save(dev_fr_save_dir + 'dev_' + str(dev_i) + '.npy',dev_fr_i)
						dev_fr.append(dev_fr_i)
					del dev_i, dev_start_i, dev_end_i, dev_fr_i
				
				#Pass inputs to parallel computation on probabilities
				inputs = zip(dev_fr, itertools.repeat(num_tastes), \
					 itertools.repeat(num_neur),itertools.repeat(x_vals), \
					 itertools.repeat(fit_tastant_neur), itertools.repeat(fit_all_neur), \
					 itertools.repeat(p_taste),itertools.repeat(taste_select_neur))
				tic = time.time()
				pool = Pool(4)
				burst_decode_prob = pool.map(dp.segment_burst_decode_dependent_parallelized, inputs)
				pool.close()
				toc = time.time()
				print('\t\tTime to decode = ' + str(np.round((toc-tic)/60,2)) + ' (min)')
				burst_decode_array = np.squeeze(np.array(burst_decode_prob)).T
				#Save decoding probabilities
				np.save(epoch_save_dir + 'segment_' + str(s_i) + '_burst_decode.npy',burst_decode_array)
			
			#Create array that reflects burst timing
			seg_burst_timing_array = np.zeros((num_tastes,seg_len))
			seg_burst_timing_array[-1,:] = 1
			for dev_i in range(num_devs):
				dev_start_i = seg_deviation_times[0,dev_i]
				dev_end_i = seg_deviation_times[1,dev_i]
				seg_burst_timing_array[:,dev_start_i:dev_end_i] = np.expand_dims(burst_decode_array[:,dev_i],1)
			np.save(epoch_save_dir + 'segment_' + str(s_i) + '_burst_decode_full_time.npy',seg_burst_timing_array)
		
			seg_burst_timing_taste_ind = np.argmax(seg_burst_timing_array,0)
			#Updated decoding based on threshold
			seg_burst_timing_taste_bin = np.zeros(np.shape(seg_burst_timing_array))
			for t_i in range(num_tastes):
				taste_bin = (seg_burst_timing_taste_ind == t_i).astype('int')
				#To ensure starts and ends of bins align
				taste_bin[0] = 0
				taste_bin[-1] = 0
				#Calculate decoding periods
				diff_decoded_taste = np.diff(taste_bin)
				start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
				end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
				num_decoded = len(start_decoded)
				#Calculate number of neurons in each period
				num_neur_decoded = np.zeros(num_decoded)
				for nd_i in range(num_decoded):
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					for n_i in range(num_neur):
						if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
							num_neur_decoded[nd_i] += 1
				#Now cut at threshold and only keep matching decoded intervals
				decode_ind = np.where(num_neur_decoded > neuron_count_thresh)[0]
				for db in decode_ind:
					s_db = start_decoded[db]
					e_db = end_decoded[db]
					seg_burst_timing_taste_bin[t_i,s_db:e_db] = 1
			
			#___Create overall stats plots___
			seg_decode_epoch_prob_nonan = np.zeros(np.shape(burst_decode_array))
			seg_decode_epoch_prob_nonan[:] = burst_decode_array[:]
			seg_decode_epoch_prob_nonan[np.isnan(seg_decode_epoch_prob_nonan)] = 0
			seg_decode_epoch_taste_ind = np.argmax(burst_decode_array,0)
			#Updated decoding based on threshold
			seg_decode_epoch_taste_bin = np.zeros(np.shape(burst_decode_array))
			for t_i in range(num_tastes):
				taste_bin = (seg_decode_epoch_taste_ind == t_i).astype('int')
				#To ensure starts and ends of bins align
				taste_bin[0] = 0
				taste_bin[-1] = 0
				#Calculate decoding periods
				diff_decoded_taste = np.diff(taste_bin)
				start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
				end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
				num_decoded = len(start_decoded)
				#Calculate number of neurons in each period
				num_neur_decoded = np.zeros(num_decoded)
				for nd_i in range(num_decoded):
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					for n_i in range(num_neur):
						if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
							num_neur_decoded[nd_i] += 1
				#Now cut at threshold and only keep matching decoded intervals
				decode_ind = np.where(num_neur_decoded > neuron_count_thresh)[0]
				for db in decode_ind:
					s_db = start_decoded[db]
					e_db = end_decoded[db]
					seg_decode_epoch_taste_bin[t_i,s_db:e_db] = 1
				
			#Line plots of decoded tastes at the deviation start times
			f1 = plt.figure()
			plt.plot(seg_deviation_times[0,:]/1000/60,seg_decode_epoch_prob_nonan.T)
			for t_i in range(num_tastes):
				plt.fill_between(seg_deviation_times[0,:]/1000/60,seg_decode_epoch_taste_bin[t_i,:],alpha=0.2)
			plt.legend(dig_in_names,loc='right')
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f1.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode.png')
			f1.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode.svg')
			plt.close(f1)
			f1 = plt.figure()
			plt.plot(np.arange(seg_len)/1000/60,seg_burst_timing_array.T)
			for t_i in range(num_tastes):
				plt.fill_between(seg_deviation_times[0,:]/1000/60,seg_decode_epoch_taste_bin[t_i,:],alpha=0.2)
			plt.legend(dig_in_names,loc='right')
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f1.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode.png')
			f1.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode.svg')
			plt.close(f1)
			#Imshow
			f2 = plt.figure()
			plt.imshow(seg_decode_epoch_prob_nonan,aspect='auto',interpolation = 'none')
			y_ticks = np.arange(len(dig_in_names))
			plt.yticks(y_ticks,dig_in_names)
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Deviation Event #')
			plt.title('Segment ' + str(s_i))
			f2.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode_im.png')
			f2.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode_im.svg')
			plt.close(f2)
			f2 = plt.figure()
			plt.imshow(seg_burst_timing_array,aspect='auto',interpolation = 'none')
			y_ticks = np.arange(len(dig_in_names))
			plt.yticks(y_ticks,dig_in_names)
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f2.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode_im.png')
			f2.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode_im.svg')
			plt.close(f2)
			#Fraction of occurrences
			f3 = plt.figure()
			dev_decode_pie = np.sum(seg_decode_epoch_taste_bin,1)/np.sum(seg_decode_epoch_taste_bin)
			seg_decode_percents[:,s_ind] = dev_decode_pie.T
			plt.pie(dev_decode_pie,labels=dig_in_names,autopct='%1.1f%%',pctdistance =1.5)
			plt.title('Segment ' + str(s_i))
			f3.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode_pie.png')
			f3.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode_pie.svg')
			plt.close(f3)
			f3 = plt.figure()
			full_time_dev_decode_pie = np.sum(seg_burst_timing_taste_bin,1)/np.sum(seg_burst_timing_taste_bin)
			seg_decode_percents_full[:,s_ind] = full_time_dev_decode_pie.T
			plt.pie(full_time_dev_decode_pie,labels=dig_in_names,autopct='%1.1f%%',pctdistance =1.5)
			plt.title('Segment ' + str(s_i))
			f3.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode_pie.png')
			f3.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode_pie.svg')
			plt.close(f3)
			
			
		f = plt.figure()
		plt.plot(seg_decode_percents.T,label=dig_in_names)
		plt.xticks(np.arange(len(segments_to_analyze)),labels=np.array(segment_names)[segments_to_analyze])
		plt.legend()
		plt.title('Decoded Burst Ratio Trends')
		plt.tight_layout()
		f.savefig(epoch_save_dir + 'burst_decoding_trends.png')
		f.savefig(epoch_save_dir + 'burst_decoding_trends.svg')
		plt.close(f)
		f, ax = plt.subplots(nrows=1,ncols=num_tastes,figsize=(num_tastes*3,3))
		for t_i in range(num_tastes):
			ax[t_i].plot(seg_decode_percents_full[t_i,:])
			ax[t_i].set_xticks(np.arange(len(segments_to_analyze)),labels=np.array(segment_names)[segments_to_analyze],rotation=45)
			ax[t_i].set_title(dig_in_names[t_i])
		plt.suptitle('Full Segment Ratio Trends')
		plt.tight_layout()
		f.savefig(epoch_save_dir + 'full_time_burst_decoding_trends.png')
		f.savefig(epoch_save_dir + 'full_time_burst_decoding_trends.svg')
		plt.close(f)
		
	
def decode_epochs_zscore(tastant_fr_dist_z,segment_spike_times,post_taste_dt,pre_taste_dt,
				   skip_dt,dig_in_names,segment_times,
				   segment_dev_rasters, segment_dev_times,
				   segment_names,start_dig_in_times,taste_num_deliv,
				   taste_select_epoch,use_full,max_decode,max_hz,min_hz,
				   save_dir,neuron_count_thresh,bin_dt,trial_start_frac=0,
				   epochs_to_analyze=[],segments_to_analyze=[]):		
	"""Decode taste from epoch-specific firing rates"""
	#Variables
	num_tastes = len(start_dig_in_times)
	num_neur = len(segment_spike_times[0])
	max_num_deliv = np.max(taste_num_deliv).astype('int')
	num_cp = len(tastant_fr_dist_z[0][0])
	num_segments = len(segment_spike_times)
	hist_bins = np.arange(start=min_hz-1,stop=max_hz+1,step=0.25)
	x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
	p_taste = taste_num_deliv/np.sum(taste_num_deliv) #P(taste)
	half_bin_z_dt = np.floor(bin_dt/2).astype('int')
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(num_segments)

	#If trial_start_frac > 0 use only trials after that threshold
	trial_start_ind = np.floor(max_num_deliv*trial_start_frac).astype('int')
	new_max_num_deliv = max_num_deliv - trial_start_ind

	for e_i in epochs_to_analyze: #By epoch conduct decoding
		print('Decoding Epoch ' + str(e_i))
		
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		
		#Fit gmm distributions to fr of each population for each taste
		#P(firing rate | taste) w/ inter-neuron dependencies
		fit_tastant_neur = dict()
		for t_i in range(num_tastes):
			full_data = []
			for d_i in range(max_num_deliv):
				if d_i >= trial_start_ind:
					full_data.extend(tastant_fr_dist_z[t_i][d_i-trial_start_ind][e_i])
			gm = gmm(n_components=1, n_init=10).fit(full_data)
			fit_tastant_neur[t_i] = gm
					
		#Fit gmm distribution to fr of all tastes
		#P(firing rate) w/ inter-neuron dependencies
		full_data = []
		for t_i in range(num_tastes):
			for d_i in range(max_num_deliv):
				if d_i >= trial_start_ind:
					full_data.extend(tastant_fr_dist_z[t_i][d_i-trial_start_ind][e_i])
		gm = gmm(n_components=1,  n_init=10).fit(full_data)
		fit_all_neur = gm
		
		#Segment-by-segment use deviation times to zoom in and test decoding
		epoch_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_save_dir):
			os.mkdir(epoch_save_dir)
			
		#Create save arrays for decoding stats
		seg_decode_percents = np.zeros((num_tastes,len(segments_to_analyze)))
		seg_decode_percents_full = np.zeros((num_tastes,len(segments_to_analyze)))
		
		for s_ind, s_i in enumerate(segments_to_analyze):
			seg_save_dir = epoch_save_dir + 'segment_' + str(s_i) + '/'
			if not os.path.isdir(seg_save_dir):
				os.mkdir(seg_save_dir)
			#Get segment variables
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = segment_times[s_i+1] - segment_times[s_i] #in dt = ms
			#Pull segment deviations
			seg_deviations = segment_dev_rasters[s_i]
			seg_deviation_times = segment_dev_times[s_i]
			num_devs = len(seg_deviations)
			#Binerize Spike Times
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in taste_select_neur:
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			#Z-score the segment
			time_bin_starts = np.arange(seg_start+half_bin_z_dt,seg_end-half_bin_z_dt,half_bin_z_dt*2)
			tb_fr = np.zeros((num_neur,len(time_bin_starts)))
			for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts)):
				tb_fr[:,tb_i] = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin_z_dt:tb+half_bin_z_dt-seg_start],1)/(2*half_bin_z_dt*(1/1000))
			mean_fr = np.mean(tb_fr,1)
			std_fr = np.std(tb_fr,1)
			
			#Grab neuron firing rates in sliding bins
			dev_fr_save_dir = seg_save_dir + 'dev_fr/'
			if not os.path.isdir(dev_fr_save_dir):
				os.mkdir(dev_fr_save_dir)
			try:
				burst_decode_array = np.load(epoch_save_dir + 'segment_' + str(s_i) + '_burst_decode.npy')
				print('\tSegment ' + str(s_i) + ' Previously Decoded')
			except:
				print('\tDecoding Segment ' + str(s_i) + ' Bursts')
				#Perform parallel computation for each time bin
				print('\t\tCalculate firing rates for time bins')
				try:
					dev_fr_z = []
					for dev_i in range(num_devs):
						dev_fr_i = np.load(dev_fr_save_dir + 'dev_' + str(dev_i) + '.npy')
						dev_fr_z.append(dev_fr_i)
				except:
					#Pull z-scored deviation firing rate vectors
					dev_fr_z = []
					for dev_i in range(num_devs):
						dev_start_i = seg_deviation_times[0,dev_i]
						dev_end_i = seg_deviation_times[1,dev_i]
						dev_len = dev_end_i - dev_start_i
						dev_fr_i = np.sum(segment_spike_times_s_i_bin[:,dev_start_i:dev_end_i],1)/(dev_len*(1/1000))
						dev_fr_i_z = (dev_fr_i - mean_fr)/std_fr
						np.save(dev_fr_save_dir + 'dev_' + str(dev_i) + '.npy',dev_fr_i_z)
						dev_fr_z.append(dev_fr_i_z)
					del dev_i, dev_start_i, dev_end_i, dev_fr_i_z
				
				#Pass inputs to parallel computation on probabilities
				inputs = zip(dev_fr_z, itertools.repeat(num_tastes), \
					 itertools.repeat(num_neur),itertools.repeat(x_vals), \
					 itertools.repeat(fit_tastant_neur), itertools.repeat(fit_all_neur), \
					 itertools.repeat(p_taste),itertools.repeat(taste_select_neur))
				tic = time.time()
				pool = Pool(4)
				burst_decode_prob = pool.map(dp.segment_burst_decode_dependent_parallelized, inputs)
				pool.close()
				toc = time.time()
				print('\t\tTime to decode = ' + str(np.round((toc-tic)/60,2)) + ' (min)')
				burst_decode_array = np.squeeze(np.array(burst_decode_prob)).T
				#Save decoding probabilities
				np.save(epoch_save_dir + 'segment_' + str(s_i) + '_burst_decode.npy',burst_decode_array)
			
			#Create array that reflects burst timing
			seg_burst_timing_array = np.zeros((num_tastes,seg_len))
			seg_burst_timing_array[-1,:] = 1
			for dev_i in range(num_devs):
				dev_start_i = seg_deviation_times[0,dev_i]
				dev_end_i = seg_deviation_times[1,dev_i]
				seg_burst_timing_array[:,dev_start_i:dev_end_i] = np.expand_dims(burst_decode_array[:,dev_i],1)
			np.save(epoch_save_dir + 'segment_' + str(s_i) + '_burst_decode_full_time.npy',seg_burst_timing_array)
		
			seg_burst_timing_taste_ind = np.argmax(seg_burst_timing_array,0)
			#Updated decoding based on threshold
			seg_burst_timing_taste_bin = np.zeros(np.shape(seg_burst_timing_array))
			for t_i in range(num_tastes):
				taste_bin = (seg_burst_timing_taste_ind == t_i).astype('int')
				#To ensure starts and ends of bins align
				taste_bin[0] = 0
				taste_bin[-1] = 0
				#Calculate decoding periods
				diff_decoded_taste = np.diff(taste_bin)
				start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
				end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
				num_decoded = len(start_decoded)
				#Calculate number of neurons in each period
				num_neur_decoded = np.zeros(num_decoded)
				for nd_i in range(num_decoded):
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					for n_i in range(num_neur):
						if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
							num_neur_decoded[nd_i] += 1
				#Now cut at threshold and only keep matching decoded intervals
				decode_ind = np.where(num_neur_decoded > neuron_count_thresh)[0]
				for db in decode_ind:
					s_db = start_decoded[db]
					e_db = end_decoded[db]
					seg_burst_timing_taste_bin[t_i,s_db:e_db] = 1
			
			#___Create overall stats plots___
			seg_decode_epoch_prob_nonan = np.zeros(np.shape(burst_decode_array))
			seg_decode_epoch_prob_nonan[:] = burst_decode_array[:]
			seg_decode_epoch_prob_nonan[np.isnan(seg_decode_epoch_prob_nonan)] = 0
			seg_decode_epoch_taste_ind = np.argmax(burst_decode_array,0)
			#Updated decoding based on threshold
			seg_decode_epoch_taste_bin = np.zeros(np.shape(burst_decode_array))
			for t_i in range(num_tastes):
				taste_bin = (seg_decode_epoch_taste_ind == t_i).astype('int')
				#To ensure starts and ends of bins align
				taste_bin[0] = 0
				taste_bin[-1] = 0
				#Calculate decoding periods
				diff_decoded_taste = np.diff(taste_bin)
				start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
				end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
				num_decoded = len(start_decoded)
				#Calculate number of neurons in each period
				num_neur_decoded = np.zeros(num_decoded)
				for nd_i in range(num_decoded):
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					for n_i in range(num_neur):
						if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
							num_neur_decoded[nd_i] += 1
				#Now cut at threshold and only keep matching decoded intervals
				decode_ind = np.where(num_neur_decoded > neuron_count_thresh)[0]
				for db in decode_ind:
					s_db = start_decoded[db]
					e_db = end_decoded[db]
					seg_decode_epoch_taste_bin[t_i,s_db:e_db] = 1
				
			#Line plots of decoded tastes at the deviation start times
			f1 = plt.figure()
			plt.plot(seg_deviation_times[0,:]/1000/60,seg_decode_epoch_prob_nonan.T)
			for t_i in range(num_tastes):
				plt.fill_between(seg_deviation_times[0,:]/1000/60,seg_decode_epoch_taste_bin[t_i,:],alpha=0.2)
			plt.legend(dig_in_names,loc='right')
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f1.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode.png')
			f1.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode.svg')
			plt.close(f1)
			f1 = plt.figure()
			plt.plot(np.arange(seg_len)/1000/60,seg_burst_timing_array.T)
			for t_i in range(num_tastes):
				plt.fill_between(seg_deviation_times[0,:]/1000/60,seg_decode_epoch_taste_bin[t_i,:],alpha=0.2)
			plt.legend(dig_in_names,loc='right')
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f1.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode.png')
			f1.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode.svg')
			plt.close(f1)
			#Imshow
			f2 = plt.figure()
			plt.imshow(seg_decode_epoch_prob_nonan,aspect='auto',interpolation = 'none')
			y_ticks = np.arange(len(dig_in_names))
			plt.yticks(y_ticks,dig_in_names)
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Deviation Event #')
			plt.title('Segment ' + str(s_i))
			f2.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode_im.png')
			f2.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode_im.svg')
			plt.close(f2)
			f2 = plt.figure()
			plt.imshow(seg_burst_timing_array,aspect='auto',interpolation = 'none')
			y_ticks = np.arange(len(dig_in_names))
			plt.yticks(y_ticks,dig_in_names)
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f2.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode_im.png')
			f2.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode_im.svg')
			plt.close(f2)
			#Fraction of occurrences
			f3 = plt.figure()
			dev_decode_pie = np.sum(seg_decode_epoch_taste_bin,1)/np.sum(seg_decode_epoch_taste_bin)
			seg_decode_percents[:,s_ind] = dev_decode_pie.T
			plt.pie(dev_decode_pie,labels=dig_in_names,autopct='%1.1f%%',pctdistance =1.5)
			plt.title('Segment ' + str(s_i))
			f3.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode_pie.png')
			f3.savefig(seg_save_dir + 'segment_' + str(s_i) + '_dev_decode_pie.svg')
			plt.close(f3)
			f3 = plt.figure()
			full_time_dev_decode_pie = np.sum(seg_burst_timing_taste_bin,1)/np.sum(seg_burst_timing_taste_bin)
			seg_decode_percents_full[:,s_ind] = full_time_dev_decode_pie.T
			plt.pie(full_time_dev_decode_pie,labels=dig_in_names,autopct='%1.1f%%',pctdistance =1.5)
			plt.title('Segment ' + str(s_i))
			f3.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode_pie.png')
			f3.savefig(seg_save_dir + 'segment_' + str(s_i) + '_full_time_dev_decode_pie.svg')
			plt.close(f3)
			
			
		f = plt.figure()
		plt.plot(seg_decode_percents.T,label=dig_in_names)
		plt.xticks(np.arange(len(segments_to_analyze)),labels=np.array(segment_names)[segments_to_analyze])
		plt.legend()
		plt.title('Decoded Burst Ratio Trends')
		plt.tight_layout()
		f.savefig(epoch_save_dir + 'burst_decoding_trends.png')
		f.savefig(epoch_save_dir + 'burst_decoding_trends.svg')
		plt.close(f)
		f, ax = plt.subplots(nrows=1,ncols=num_tastes,figsize=(num_tastes*3,3))
		for t_i in range(num_tastes):
			ax[t_i].plot(seg_decode_percents_full[t_i,:])
			ax[t_i].set_xticks(np.arange(len(segments_to_analyze)),labels=np.array(segment_names)[segments_to_analyze],rotation=45)
			ax[t_i].set_title(dig_in_names[t_i])
		plt.suptitle('Full Segment Ratio Trends')
		plt.tight_layout()
		f.savefig(epoch_save_dir + 'full_time_burst_decoding_trends.png')
		f.savefig(epoch_save_dir + 'full_time_burst_decoding_trends.svg')
		plt.close(f)
		
		
def plot_decoded(fr_dist,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
				 start_dig_in_times,end_dig_in_times,post_taste_dt,pre_taste_dt,
				 pop_taste_cp_raster_inds,bin_dt,dig_in_names,
				 segment_times,segment_names,taste_num_deliv,taste_select_epoch,
				 use_full,save_dir,max_decode,max_hz,seg_stat_bin,
				 neuron_count_thresh,trial_start_frac=0,
				 epochs_to_analyze=[],segments_to_analyze=[],
				 bin_pre_taste=100, decode_prob_cutoff=0.95,min_dev_size=50):
	"""Function to plot the periods when something other than no taste is 
	decoded"""
	num_segments = len(segment_spike_times)
	neur_cut = np.floor(num_neur*neuron_count_thresh).astype('int')
	taste_colors = cm.brg(np.linspace(0,1,num_tastes))
	epoch_seg_taste_percents = np.zeros((num_cp,num_segments,num_tastes))
	half_bin_dev_size = np.ceil(min_dev_size/2).astype('int')
	half_bin_z_dt = np.floor(bin_dt/2).astype('int')
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(num_segments)
		
	#Get taste segment z-score info
	s_i_taste = np.nan*np.ones(1)
	for s_i in range(len(segment_names)):
		if segment_names[s_i].lower() == 'taste':
			s_i_taste[0] = s_i
	if not np.isnan(s_i_taste[0]):
		s_i = int(s_i_taste[0])
		seg_start = segment_times[s_i]
		seg_end = segment_times[s_i+1]
		seg_len = seg_end - seg_start
		time_bin_starts = np.arange(seg_start+half_bin_z_dt,seg_end-half_bin_z_dt,half_bin_z_dt*2)
		segment_spike_times_s_i = segment_spike_times[s_i]
		segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
		for n_i in range(num_neur):
			n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
			segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
		tb_fr = np.zeros((num_neur,len(time_bin_starts)))
		for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts)):
			tb_fr[:,tb_i] = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin_z_dt:tb+half_bin_z_dt-seg_start],1)/(2*half_bin_z_dt*(1/1000))
		mean_fr_taste = np.mean(tb_fr,1)
		std_fr_taste = np.std(tb_fr,1)
		std_fr_taste[std_fr_taste == 0] = 1 #to avoid nan calculations
	else:
		mean_fr_taste = np.zeros(num_neur)
		std_fr_taste = np.ones(num_neur)

	for e_i in epochs_to_analyze:
		print('Plotting Decoding for Epoch ' + str(e_i))
		
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			print("Data not previously decoded, or passed directory incorrect.")
			pass
		
		for s_i in tqdm.tqdm(segments_to_analyze):
			try:
				seg_decode_epoch_prob = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '_burst_decode_full_time.npy')
			except:
				print("Segment " + str(s_i) + " Bursts Never Decoded")
				pass
			
			seg_decode_save_dir = epoch_decode_save_dir + 'segment_' + str(s_i) + '/'
			if not os.path.isdir(seg_decode_save_dir):
				os.mkdir(seg_decode_save_dir)
			
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = seg_end - seg_start #in dt = ms
			
			#Import raster plots for segment
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in taste_select_neur:
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			
			#Z-score the segment
			time_bin_starts = np.arange(seg_start+half_bin_z_dt,seg_end-half_bin_z_dt,half_bin_z_dt*2)
			tb_fr = np.zeros((num_neur,len(time_bin_starts)))
			for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts)):
				tb_fr[:,tb_i] = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin_z_dt:tb+half_bin_z_dt-seg_start],1)/(2*half_bin_z_dt*(1/1000))
			mean_fr = np.mean(tb_fr,1)
			std_fr = np.std(tb_fr,1)
			std_fr[std_fr == 0] = 1
			
			#Calculate maximally decoded taste
			decoded_taste_max = np.argmax(seg_decode_epoch_prob,0)
			#Store binary decoding results
			decoded_taste_bin = np.zeros((num_tastes,len(decoded_taste_max)))
			for t_i in range(num_tastes):
				decoded_taste_bin[t_i,np.where(decoded_taste_max == t_i)[0]] = 1
			#To ensure starts and ends of bins align
			decoded_taste_bin[:,0] = 0
			decoded_taste_bin[:,-1] = 0
			
			#For each taste (except none) calculate firing rates and z-scored firing rates
			all_taste_fr_vecs = []
			all_taste_fr_vecs_z = []
			all_taste_fr_vecs_mean = np.zeros((num_tastes,num_neur))
			all_taste_fr_vecs_mean_z = np.zeros((num_tastes,num_neur))
			all_taste_event_fr_vecs = []
			all_taste_event_fr_vecs_z = []
			#Grab taste firing rate vectors
			for t_i in range(num_tastes):
				#Import taste spike and cp times
				taste_spike_times = tastant_spike_times[t_i]
				taste_deliv_times = start_dig_in_times[t_i]
				max_num_deliv = len(taste_deliv_times)
				pop_taste_cp_times = pop_taste_cp_raster_inds[t_i]
				
				#If trial_start_frac > 0 use only trials after that threshold
				trial_start_ind = np.floor(max_num_deliv*trial_start_frac).astype('int')
				new_max_num_deliv = (max_num_deliv - trial_start_ind).astype('int')
				
				#Store as binary spike arrays
				taste_spike_times_bin = np.zeros((new_max_num_deliv,num_neur,post_taste_dt)) #Taste response spike times
				taste_cp_times = np.zeros((new_max_num_deliv,num_cp+1)).astype('int')
				taste_epoch_fr_vecs = np.zeros((new_max_num_deliv,num_neur)) #original firing rate vecs
				taste_epoch_fr_vecs_z = np.zeros((new_max_num_deliv,num_neur)) #z-scored firing rate vecs
				for d_i in range(len(taste_spike_times)): #store each delivery to binary spike matrix
					if d_i >= trial_start_ind:	
						taste_deliv_i = taste_deliv_times[d_i]
						for n_i in range(num_neur):
							spikes_deliv_i = taste_spike_times[d_i][n_i]
							if t_i == num_tastes-1:
								if len(taste_spike_times[d_i-trial_start_ind][n_i]) > 0:
									d_i_spikes = np.array(spikes_deliv_i - (np.min(spikes_deliv_i)+pre_taste_dt)).astype('int')
								else:
									d_i_spikes = np.empty(0)
							else:
								d_i_spikes = np.array(spikes_deliv_i - taste_deliv_i).astype('int')
							d_i_spikes_posttaste = d_i_spikes[(d_i_spikes<post_taste_dt)*(d_i_spikes>=0)]
							if len(d_i_spikes_posttaste) > 0:
								taste_spike_times_bin[d_i-trial_start_ind,n_i,d_i_spikes_posttaste] = 1
						taste_cp_times[d_i-trial_start_ind,:] = np.concatenate((np.zeros(1),np.cumsum(np.diff(pop_taste_cp_times[d_i,:])))).astype('int')
						#Calculate the FR vectors by epoch for each taste response and the average FR vector
						epoch_len_i = (taste_cp_times[d_i,e_i+1]-taste_cp_times[d_i,e_i])/1000 
						if epoch_len_i == 0:
							taste_epoch_fr_vecs[d_i-trial_start_ind,:] = np.zeros(num_neur)
						else:
							taste_epoch_fr_vecs[d_i-trial_start_ind,:] = np.sum(taste_spike_times_bin[d_i-trial_start_ind,:,taste_cp_times[d_i,e_i]:taste_cp_times[d_i,e_i+1]],1)/epoch_len_i #FR in HZ
						#Calculate z-scored FR vector
						taste_epoch_fr_vecs_z[d_i-trial_start_ind,:] = (taste_epoch_fr_vecs[d_i-trial_start_ind,:].flatten() - mean_fr_taste)/std_fr_taste
						
				all_taste_fr_vecs.append(taste_epoch_fr_vecs)
				all_taste_fr_vecs_z.append(taste_epoch_fr_vecs_z)
				#Calculate average taste fr vec
				taste_fr_vecs_mean = np.nanmean(taste_epoch_fr_vecs,0)
				taste_fr_vecs_z_mean = np.nanmean(taste_epoch_fr_vecs_z,0)
				all_taste_fr_vecs_mean[t_i,:] = taste_fr_vecs_mean
				all_taste_fr_vecs_mean_z[t_i,:] = taste_fr_vecs_z_mean
				#taste_fr_vecs_max_hz = np.max(taste_epoch_fr_vecs)
			
			#Now look at decoded events
			for t_i in range(num_tastes):
				#First calculate neurons decoded in all decoded intervals
				decoded_taste = decoded_taste_bin[t_i,:]
				decoded_taste[0] = 0
				decoded_taste[-1] = 0
				decoded_taste_prob = seg_decode_epoch_prob[t_i,:]
				decoded_taste[decoded_taste_prob<decode_prob_cutoff] = 0
				diff_decoded_taste = np.diff(decoded_taste)
				start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
				end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
				num_decoded = len(start_decoded)
				num_neur_decoded = np.zeros(num_decoded)
				prob_decoded = np.zeros(num_decoded)
				for nd_i in range(num_decoded):
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					d_len = d_end-d_start
					for n_i in range(num_neur):
						if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
							num_neur_decoded[nd_i] += 1
					prob_decoded[nd_i] = np.mean(seg_decode_epoch_prob[t_i,d_start:d_end])
				#Now cut at threshold and only keep matching decoded intervals
				decode_ind = np.where(num_neur_decoded > neur_cut)[0]
				decoded_bin = np.zeros(np.shape(decoded_taste))
				for db in decode_ind:
					s_db = start_decoded[db]
					e_db = end_decoded[db]
					decoded_bin[s_db:e_db] = 1
				#Save the percent taste decoded matching threshold
				epoch_seg_taste_percents[e_i,s_i,t_i] = (np.sum(decoded_bin)/len(decoded_bin))*100	
				#Re-calculate start and end times of the decoded intervals
				start_decoded = start_decoded[decode_ind]
				end_decoded = end_decoded[decode_ind]
				#Re-calculate the decoded statistics
				num_decoded = len(start_decoded)
				num_neur_decoded = num_neur_decoded[decode_ind]
				prob_decoded = prob_decoded[decode_ind]
				len_decoded = end_decoded-start_decoded
				iei_decoded = start_decoded[1:] - end_decoded[:-1]
				
				#Create plots of decoded period statistics
				seg_dist_starts = np.arange(0,seg_len,seg_stat_bin)
				seg_distribution = np.zeros(len(seg_dist_starts)-1)
				prob_distribution = np.zeros(len(seg_dist_starts)-1)
				for sd_i in range(len(seg_dist_starts)-1):
					bin_events = np.where((start_decoded < seg_dist_starts[sd_i+1])*(start_decoded >= seg_dist_starts[sd_i]))[0]
					seg_distribution[sd_i] = len(bin_events)
					prob_distribution[sd_i] = np.mean(prob_decoded[bin_events])
				seg_dist_midbin = seg_dist_starts[:-1] + np.diff(seg_dist_starts)/2
				
				taste_decode_save_dir = seg_decode_save_dir + dig_in_names[t_i] + '_events/'
				if not os.path.isdir(taste_decode_save_dir):
					os.mkdir(taste_decode_save_dir)
				
				#Create statistics plots
				f = plt.figure(figsize=(8,8))
				plt.subplot(3,2,1)
				plt.hist(len_decoded)
				plt.xlabel('Length (ms)')
				plt.ylabel('Number of Occurrences')
				plt.title('Length of Decoded Event')
				plt.subplot(3,2,2)
				plt.hist(iei_decoded)
				plt.xlabel('IEI (ms)')
				plt.ylabel('Number of Occurrences')
				plt.title('Inter-Event-Interval (IEI)')
				plt.subplot(3,2,3)
				plt.hist(num_neur_decoded)
				plt.xlabel('# Neurons')
				plt.ylabel('Number of Occurrences')
				plt.title('Number of Neurons Active')
				plt.subplot(3,2,4)
				plt.bar(seg_dist_midbin,seg_distribution,width=seg_stat_bin)
				plt.xticks(np.linspace(0,seg_len,8),labels=np.round((np.linspace(0,seg_len,8)/1000/60),2),rotation=-45)
				plt.xlabel('Time in Segment (min)')
				plt.ylabel('# Events')
				plt.title('Number of Decoded Events')
				plt.subplot(3,2,5)
				plt.hist(prob_decoded)
				plt.xlabel('Event Avg P(Decoding)')
				plt.ylabel('Number of Occurrences')
				plt.title('Average Decoding Probability')
				plt.subplot(3,2,6)
				plt.bar(seg_dist_midbin,prob_distribution,width=seg_stat_bin)
				plt.xticks(np.linspace(0,seg_len,8),labels=np.round((np.linspace(0,seg_len,8)/1000/60),2),rotation=-45)
				plt.xlabel('Time in Segment (min)')
				plt.ylabel('Avg(Event Avg P(Decoding))')
				plt.title('Average Decoding Probability')
				plt.suptitle('')
				plt.tight_layout()
				f.savefig(taste_decode_save_dir + 'epoch_' + str(e_i) + '_seg_' + str(s_i) + '_event_statistics.png')
				f.savefig(taste_decode_save_dir + 'epoch_' + str(e_i) + '_seg_' + str(s_i) + '_event_statistics.png')
				plt.close(f)
					
				if num_decoded > max_decode: #Reduce number if too many
					decode_prob_avg = np.array([np.mean(seg_decode_epoch_prob[t_i,start_decoded[i]:end_decoded[i]]) for i in range(len(start_decoded))])
					decode_plot_ind = sample(list(np.where(decode_prob_avg >= decode_prob_cutoff)[0]),max_decode)
				else:
					decode_plot_ind = np.arange(num_decoded)
				decode_plot_ind = np.array(decode_plot_ind)
				#Create plots of the decoded periods
				decoded_fr_vecs = [] #Store all decoded events firing rates
				decoded_z_fr_vecs = [] #Store all decoded events z-scored firing rates
				for nd_i in range(num_decoded):
					#Decode variables
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					d_len = d_end-d_start
					d_plot_start = np.max((start_decoded[nd_i]-2*d_len,0))
					d_plot_end = np.min((end_decoded[nd_i]+2*d_len,seg_len))
					d_plot_len = d_plot_end-d_plot_start
					d_plot_x_vals = (np.linspace(d_plot_start-d_start,d_plot_end-d_start,10)).astype('int')
					decode_plot_times = np.arange(d_plot_start,d_plot_end)
					event_spikes = segment_spike_times_s_i_bin[:,d_plot_start:d_plot_end]
					decode_spike_times = []
					for n_i in range(num_neur):
						decode_spike_times.append(list(np.where(event_spikes[n_i,:])[0]))
					event_spikes_expand = segment_spike_times_s_i_bin[:,d_plot_start-half_bin_dev_size:d_plot_end+half_bin_dev_size]
					event_spikes_expand_count = np.sum(event_spikes_expand,0)
					firing_rate_vec = np.zeros(d_plot_len)
					for dpt_i in np.arange(half_bin_dev_size,d_plot_len+half_bin_dev_size):
						firing_rate_vec[dpt_i-half_bin_dev_size] = np.sum(event_spikes_expand_count[dpt_i-half_bin_dev_size:dpt_i+half_bin_dev_size])/(min_dev_size/1000)/num_neur
					d_fr_vec = np.sum(segment_spike_times_s_i_bin[:,d_start:d_end],1)/(d_len/1000)
					decoded_fr_vecs.append(d_fr_vec)
					#Calculate z-scored data
					d_fr_vec_z = (d_fr_vec-mean_fr)/std_fr
					decoded_z_fr_vecs.append(d_fr_vec_z)
					#Find max hz 
					#d_fr_vec_max_hz = np.max(d_fr_vec)
					if len(decode_plot_ind)>0:
						#If it's an event to plot
						if np.sum((decode_plot_ind == nd_i).astype('int')) > 0:
							#Correlation of vector to avg taste vector
							corr_decode_event = [pearsonr(all_taste_fr_vecs_mean[t_i,:],d_fr_vec)[0] for t_i in range(num_tastes)]
							corr_title_norm = [dig_in_names[t_i] + ' corr = ' + str(np.round(corr_decode_event[t_i],2)) for t_i in range(num_tastes)]
							#Correlation of z-scored vector to z-scored avg taste vector
							corr_decode_event_z = [pearsonr(all_taste_fr_vecs_mean_z[t_i,:],d_fr_vec_z)[0] for t_i in range(num_tastes)]
							corr_title_z = [dig_in_names[t_i] + ' z-corr = ' + str(np.round(corr_decode_event_z[t_i],2)) for t_i in range(num_tastes)]
							corr_title = (', ').join(corr_title_norm) + '\n' + (', ').join(corr_title_z)
							#Start Figure
							f, ax = plt.subplots(nrows=5,ncols=2,figsize=(10,10),gridspec_kw=dict(height_ratios=[1,1,2,2,2]))
							gs = ax[0,0].get_gridspec()
							#Decoding probabilities
							ax[0,0].remove()
							ax[1,0].remove()
							axbig = f.add_subplot(gs[0:2,0])
							decode_x_vals = decode_plot_times-d_start
							leg_handles = []
							for t_i_2 in range(num_tastes):
								taste_decode_prob_y = seg_decode_epoch_prob[t_i_2,d_plot_start:d_plot_end]
								p_h, = axbig.plot(decode_x_vals,taste_decode_prob_y,color=taste_colors[t_i_2,:])
								leg_handles.append(p_h)
								taste_decode_prob_y[0] = 0
								taste_decode_prob_y[-1] = 0
								high_prob_binary = np.zeros(len(decode_x_vals))
								high_prob_times = np.where(taste_decode_prob_y >= decode_prob_cutoff)[0]
								high_prob_binary[high_prob_times] = 1
								high_prob_starts = np.where(np.diff(high_prob_binary) == 1)[0] + 1
								high_prob_ends = np.where(np.diff(high_prob_binary) == -1)[0] + 1
								if len(high_prob_starts) > 0:
									for hp_i in range(len(high_prob_starts)):
										axbig.fill_between(decode_x_vals[high_prob_starts[hp_i]:high_prob_ends[hp_i]],taste_decode_prob_y[high_prob_starts[hp_i]:high_prob_ends[hp_i]],alpha=0.2,color=taste_colors[t_i_2,:])
							axbig.axvline(0,color='k',alpha=0.5)
							axbig.axvline(d_len,color='k',alpha=0.5)
							axbig.legend(leg_handles,dig_in_names,loc='right')
							axbig.set_xticks(d_plot_x_vals)
							axbig.set_xlim([decode_x_vals[0],decode_x_vals[-1]])
							axbig.set_ylabel('Decoding Fraction')
							axbig.set_xlabel('Time from Event Start (ms)')
							axbig.set_title('Event ' + str(nd_i) + '\nStart Time = ' + str(round(d_start/1000/60,3)) + ' Minutes' + '\nEvent Length = ' + str(np.round(d_len,2)))
							#Decoded raster
							ax[0,1].eventplot(decode_spike_times)
							ax[0,1].set_xlim([0,d_plot_len])
							x_ticks = np.linspace(0,d_plot_len,10).astype('int')
							ax[0,1].set_xticks(x_ticks,labels=d_plot_x_vals)
							ax[0,1].axvline(d_start-d_plot_start,color='k',alpha=0.5)
							ax[0,1].axvline(d_end-d_plot_start,color='k',alpha=0.5)
							ax[0,1].set_ylabel('Neuron Index')
							ax[0,1].set_title('Event Spike Raster')
							ax[1,0].axis('off')
							#Plot population firing rates w 20ms smoothing
							ax[1,1].plot(decode_x_vals,firing_rate_vec)
							ax[1,1].axvline(0,color='k')
							ax[1,1].axvline(d_len,color='k')
							ax[1,1].set_xlim([decode_x_vals[0],decode_x_vals[-1]])
							ax[1,1].set_xticks(d_plot_x_vals)
							ax[1,1].set_title('Population Avg FR')
							ax[1,1].set_ylabel('FR (Hz)')
							ax[1,1].set_xlabel('Time from Event Start (ms)')
							#Decoded Firing Rates
							img = ax[2,0].imshow(np.expand_dims(d_fr_vec,0),vmin=0,vmax=60)#vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
							ax[2,0].set_xlabel('Neuron Index')
							ax[2,0].set_yticks(ticks=[])
							plt.colorbar(img, location='bottom',orientation='horizontal',label='Firing Rate (Hz)',panchor=(0.9,0.5),ax=ax[2,0])
							ax[2,0].set_title('Event FR')
							#Taste Firing Rates
							img = ax[2,1].imshow(np.expand_dims(taste_fr_vecs_mean,0),vmin=0,vmax=60)#vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
							ax[2,1].set_xlabel('Neuron Index')
							ax[2,1].set_yticks(ticks=[])
							plt.colorbar(img, ax= ax[2,1], location='bottom',orientation='horizontal',label='Firing Rate (Hz)',panchor=(0.9,0.5))
							ax[2,1].set_title('Avg. Taste Resp. FR')
							#Decoded Firing Rates Z-Scored
							img = ax[3,0].imshow(np.expand_dims(d_fr_vec_z,0),vmin=-3,vmax=3,cmap='bwr')
							ax[3,0].set_xlabel('Neuron Index')
							ax[3,0].set_yticks(ticks=[])
							plt.colorbar(img, ax=ax[3,0], location='bottom',orientation='horizontal',label='Z-Scored Firing Rate (Hz)',panchor=(0.9,0.5))
							ax[3,0].set_title('Event FR Z-Scored')
							#Taste Firing Rates Z-Scored
							img = ax[3,1].imshow(np.expand_dims(all_taste_fr_vecs_mean_z[t_i,:],0),vmin=-3,vmax=3,cmap='bwr')
							ax[3,1].set_xlabel('Neuron Index')
							ax[3,1].set_yticks(ticks=[])
							plt.colorbar(img, ax=ax[3,1], location='bottom',orientation='horizontal',label='Z-Scored Firing Rate (Hz)',panchor=(0.9,0.5))
							ax[3,1].set_title('Avg. Taste Resp. FR Z-Scored')
							#Decoded Firing Rates x Average Firing Rates
							max_lim = np.max([np.max(d_fr_vec_z),np.max(taste_fr_vecs_mean)])
							ax[4,0].plot([0,max_lim],[0,max_lim],alpha=0.5,linestyle='dashed')
							ax[4,0].scatter(taste_fr_vecs_mean,d_fr_vec)
							ax[4,0].set_xlabel('Average Taste FR')
							ax[4,0].set_ylabel('Decoded Taste FR')
							ax[4,0].set_title('Firing Rate Similarity')
							#Z-Scored Decoded Firing Rates x Z-Scored Average Firing Rates
							ax[4,1].plot([-3,3],[-3,3],alpha=0.5,linestyle='dashed',color='k')
							ax[4,1].scatter(all_taste_fr_vecs_mean_z[t_i,:],d_fr_vec_z)
							ax[4,1].set_xlabel('Average Taste Neuron FR Std > Mean')
							ax[4,1].set_ylabel('Event Neuron FR Std > Mean')
							ax[4,1].set_title('Z-Scored Firing Rate Similarity')
							plt.suptitle(corr_title,wrap=True)
							plt.tight_layout()
							#Save Figure
							f.savefig(taste_decode_save_dir + 'event_' + str(nd_i) + '.png')
							f.savefig(taste_decode_save_dir + 'event_' + str(nd_i) + '.svg')
							plt.close(f)
				all_taste_event_fr_vecs.append(np.array(decoded_fr_vecs))
				all_taste_event_fr_vecs_z.append(np.array(decoded_z_fr_vecs))
			
			#Taste event scatter plots
			#f, ax = plt.subplots(nrows=1, ncols=num_tastes+1, figsize=(8,8), gridspec_kw=dict(width_ratios=list(np.concatenate((6*np.ones(num_tastes),np.ones(1))))))
			f, ax = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(num_tastes*2,num_tastes*2), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
			max_fr = 0
			max_fr_t_av = 0
			for t_i in range(num_tastes): #Event Taste
				ax[t_i,0].set_ylabel('Decoded ' + dig_in_names[t_i] +' FR')
				taste_event_fr_vecs = all_taste_event_fr_vecs[t_i]
				if len(taste_event_fr_vecs) > 0:
					max_taste_fr = np.max(taste_event_fr_vecs)
					if max_taste_fr > max_fr:
						max_fr = max_taste_fr
					for t_i_c in range(num_tastes): #Average Taste
						average_fr_vec_mat = all_taste_fr_vecs_mean[t_i_c,:]*np.ones(np.shape(taste_event_fr_vecs))
						#Calculate max avg fr
						max_avg_fr = np.max(all_taste_fr_vecs_mean[t_i_c,:])
						if max_avg_fr > max_fr_t_av:
							max_fr_t_av = max_avg_fr
						ax[t_i,t_i_c].set_xlabel('Average ' + dig_in_names[t_i_c] + ' FR')
						ax[t_i,t_i_c].scatter(average_fr_vec_mat,taste_event_fr_vecs,color=taste_colors[t_i,:],alpha = 0.3)
			for t_i in range(num_tastes):
				for t_i_c in range(num_tastes):
					ax[t_i,t_i_c].plot([0,max_fr],[0,max_fr],alpha=0.5,color='k',linestyle='dashed')
					ax[t_i,t_i_c].set_ylim([0,max_fr])
					ax[t_i,t_i_c].set_xlim([0,max_fr_t_av])
					if t_i == t_i_c:
						for child in ax[t_i,t_i_c].get_children():
						    if isinstance(child, matplotlib.spines.Spine):
						        child.set_color('r')
			plt.suptitle('Deviation Events x Average Taste Response')
			plt.tight_layout()
			f.savefig(seg_decode_save_dir + 'event_vs_avg_taste_deliv_scat.png')
			f.savefig(seg_decode_save_dir + 'event_vs_avg_taste_deliv_scat.svg')
			plt.close(f)
			
			#Taste event z-scored scatter plots
			#f, ax = plt.subplots(nrows=1, ncols=num_tastes+1, figsize=(8,8), gridspec_kw=dict(width_ratios=list(np.concatenate((6*np.ones(num_tastes),np.ones(1))))))
			f, ax = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(num_tastes*2,num_tastes*2), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
			max_fr = 0
			max_fr_t_av = 0
			for t_i in range(num_tastes): #Event Taste
				ax[t_i,0].set_ylabel('Decoded ' + dig_in_names[t_i] +' FR')
				taste_event_fr_vecs_z = all_taste_event_fr_vecs_z[t_i]
				if len(taste_event_fr_vecs_z) > 0:
					max_taste_fr = np.max(taste_event_fr_vecs_z)
					if max_taste_fr > max_fr:
						max_fr = max_taste_fr
					for t_i_c in range(num_tastes): #Average Taste
						average_fr_vec_mat = all_taste_fr_vecs_mean_z[t_i_c,:]*np.ones(np.shape(taste_event_fr_vecs_z))
						#Calculate max avg fr
						max_avg_fr = np.max(all_taste_fr_vecs_mean_z[t_i_c,:])
						if max_avg_fr > max_fr_t_av:
							max_fr_t_av = max_avg_fr
						ax[t_i,t_i_c].set_xlabel('Average ' + dig_in_names[t_i_c] + ' FR')
						ax[t_i,t_i_c].scatter(average_fr_vec_mat,taste_event_fr_vecs_z,color=taste_colors[t_i,:],alpha = 0.3)
			for t_i in range(num_tastes):
				for t_i_c in range(num_tastes):
					ax[t_i,t_i_c].plot([0,max_fr],[0,max_fr],alpha=0.5,color='k',linestyle='dashed')
					ax[t_i,t_i_c].set_ylim([0,max_fr])
					ax[t_i,t_i_c].set_xlim([0,max_fr_t_av])
					if t_i == t_i_c:
						for child in ax[t_i,t_i_c].get_children():
						    if isinstance(child, matplotlib.spines.Spine):
						        child.set_color('r')
			plt.suptitle('Z-Scored Deviation Events x Z-Scored Average Taste Response')
			plt.tight_layout()
			f.savefig(seg_decode_save_dir + 'event_vs_avg_taste_deliv_scat_z.png')
			f.savefig(seg_decode_save_dir + 'event_vs_avg_taste_deliv_scat_z.svg')
			plt.close(f)
			
			#Taste event deviation plots
			#f, ax = plt.subplots(nrows=1, ncols=num_tastes+1, figsize=(8,8), gridspec_kw=dict(width_ratios=list(np.concatenate((6*np.ones(num_tastes),np.ones(1))))))
			f, ax = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(num_tastes*4,num_tastes*4), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
			max_y = 0
			min_y = 0
			max_x = 0
			for t_i in range(num_tastes): #Event Taste
				taste_event_fr_vecs = all_taste_event_fr_vecs[t_i]
				max_taste_fr = np.max(taste_event_fr_vecs)
				ax[t_i,0].set_ylabel('Firing Rate Difference')
				for t_i_c in range(num_tastes): #Average Taste
					#Calculate max fr of taste response
					ax[t_i,t_i_c].set_title('Decoded ' + dig_in_names[t_i] +' - Delivery ' + dig_in_names[t_i_c] )
					taste_fr_vecs = all_taste_fr_vecs[t_i_c]
					max_taste_resp_fr = np.max(taste_fr_vecs)
					x_vals = np.arange(num_neur)
					if max_taste_resp_fr > max_x:
						max_x = max_taste_resp_fr
					num_taste_deliv = np.shape(taste_fr_vecs)[0]
					#Calculate mean and std of event away from average
					num_events = np.shape(taste_event_fr_vecs)[0]
					all_diff = np.zeros((num_events*num_taste_deliv,num_neur))
					for td_i in range(num_taste_deliv):
						diff = taste_event_fr_vecs - taste_fr_vecs[td_i,:]
						all_diff[td_i*num_events:(td_i+1)*num_events,:] = diff
					diff_mean = np.mean(all_diff,0)
					diff_std = np.std(all_diff,0)
					if max_y < np.max(diff_mean + 3*diff_std):
						max_y = np.max(diff_mean + 3*diff_std)
					if min_y > np.min(diff_mean - 3*diff_std):
						min_y = np.min(diff_mean - 3*diff_std)
					ax[t_i,t_i_c].plot(x_vals,diff_mean,alpha=0.5,color='b',linestyle='solid')
					#ax[t_i,t_i_c].fill_between(x_vals,diff_mean - diff_std,diff_mean + diff_std,alpha=0.5,color='b')
					ax[t_i,t_i_c].violinplot(all_diff,x_vals)
					#Plot
					ax[t_i,t_i_c].set_xlabel('Neuron Index')
			for t_i in range(num_tastes):
				for t_i_c in range(num_tastes):
					ax[t_i,t_i_c].plot(np.arange(num_neur),np.zeros(num_neur),alpha=0.5,color='k',linestyle='dashed')
					ax[t_i,t_i_c].set_xlim([0,num_neur])
					ax[t_i,t_i_c].set_ylim([min_y-10,max_y+10])
					if t_i == t_i_c:
						for child in ax[t_i,t_i_c].get_children():
						    if isinstance(child, matplotlib.spines.Spine):
						        child.set_color('r')
			plt.suptitle('Deviation Events x Individual Taste Response')
			plt.tight_layout()
			f.savefig(seg_decode_save_dir + 'event_vs_taste_deliv_dev.png')
			f.savefig(seg_decode_save_dir + 'event_vs_taste_deliv_dev.svg')
			plt.close(f)
			
			#Taste event deviation plots z-scored
			#f, ax = plt.subplots(nrows=1, ncols=num_tastes+1, figsize=(8,8), gridspec_kw=dict(width_ratios=list(np.concatenate((6*np.ones(num_tastes),np.ones(1))))))
			f, ax = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(num_tastes*4,num_tastes*4), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
			max_y = 0
			min_y = 0
			max_x = 0
			for t_i in range(num_tastes): #Event Taste
				taste_event_fr_vecs_z = all_taste_event_fr_vecs_z[t_i]
				max_taste_fr = np.max(taste_event_fr_vecs_z)
				ax[t_i,0].set_ylabel('Firing Rate Difference')
				for t_i_c in range(num_tastes): #Average Taste
					#Calculate max fr of taste response
					ax[t_i,t_i_c].set_title('Decoded ' + dig_in_names[t_i] +' - Delivery ' + dig_in_names[t_i_c] )
					taste_fr_vecs_z = all_taste_fr_vecs_z[t_i_c]
					max_taste_resp_fr = np.max(taste_fr_vecs_z)
					x_vals = np.arange(num_neur)
					if max_taste_resp_fr > max_x:
						max_x = max_taste_resp_fr
					num_taste_deliv = np.shape(taste_fr_vecs_z)[0]
					#Calculate mean and std of event away from average
					num_events = np.shape(taste_event_fr_vecs_z)[0]
					all_diff = np.zeros((num_events*num_taste_deliv,num_neur))
					for td_i in range(num_taste_deliv):
						diff = taste_event_fr_vecs_z - taste_fr_vecs_z[td_i,:]
						all_diff[td_i*num_events:(td_i+1)*num_events,:] = diff
					diff_mean = np.mean(all_diff,0)
					diff_std = np.std(all_diff,0)
					if max_y < np.max(diff_mean + 3*diff_std):
						max_y = np.max(diff_mean + 3*diff_std)
					if min_y > np.min(diff_mean - 3*diff_std):
						min_y = np.min(diff_mean - 3*diff_std)
					ax[t_i,t_i_c].plot(x_vals,diff_mean,alpha=0.5,color='b',linestyle='solid')
					#ax[t_i,t_i_c].fill_between(x_vals,diff_mean - diff_std,diff_mean + diff_std,alpha=0.5,color='b')
					ax[t_i,t_i_c].violinplot(all_diff,x_vals)
					#Plot
					ax[t_i,t_i_c].set_xlabel('Neuron Index')
			for t_i in range(num_tastes):
				for t_i_c in range(num_tastes):
					ax[t_i,t_i_c].plot(np.arange(num_neur),np.zeros(num_neur),alpha=0.5,color='k',linestyle='dashed')
					ax[t_i,t_i_c].set_xlim([0,num_neur])
					ax[t_i,t_i_c].set_ylim([min_y-10,max_y+10])
					if t_i == t_i_c:
						for child in ax[t_i,t_i_c].get_children():
						    if isinstance(child, matplotlib.spines.Spine):
						        child.set_color('r')
			plt.suptitle('Deviation Events x Individual Taste Response')
			plt.tight_layout()
			f.savefig(seg_decode_save_dir + 'event_vs_taste_deliv_dev_z.png')
			f.savefig(seg_decode_save_dir + 'event_vs_taste_deliv_dev_z.svg')
			plt.close(f)
			
			
	#Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments		
	f = plt.figure(figsize=(8,8))
	plot_ind = 1
	for e_i in epochs_to_analyze:
		for t_i in range(num_tastes):
			plt.subplot(num_cp,num_tastes,plot_ind)
			plt.plot(segments_to_analyze,(epoch_seg_taste_percents[e_i,segments_to_analyze,t_i]).flatten())
			seg_labels = [segment_names[a] for a in segments_to_analyze]
			plt.xticks(segments_to_analyze,labels=seg_labels,rotation=-45)
			if t_i == 0:
				plt.ylabel('Epoch ' + str(e_i))
			if e_i == 0:
				plt.title('Taste ' + dig_in_names[t_i])
			plot_ind += 1
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents.png')
	f.savefig(save_dir + 'Decoding_Percents.svg')
	plt.close(f)
	

def plot_decoded_func_p(fr_dist,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
				 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
				 e_skip_dt,e_len_dt,dig_in_names,segment_times,
				 segment_names,taste_num_deliv,taste_select_epoch,
				 use_full,save_dir,max_decode,max_hz,seg_stat_bin,
				 epochs_to_analyze=[],segments_to_analyze=[]):	
	"""Function to plot the decoding statistics as a function of average decoding
	probability within the decoded interval."""
	warnings.filterwarnings('ignore')
	num_segments = len(segment_spike_times)
	prob_cutoffs = np.arange(1/num_tastes,1,0.05)
	num_prob = len(prob_cutoffs)
	taste_colors = cm.viridis(np.linspace(0,1,num_tastes))
	epoch_seg_taste_percents = np.zeros((num_prob,num_cp,num_segments,num_tastes))
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	if len(segments_to_analyze == 0):
		segments_to_analyze = np.arange(num_segments)

	for e_i in epochs_to_analyze: #By epoch conduct decoding
		print('Decoding Epoch ' + str(e_i))
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			print("Data not previously decoded, or passed directory incorrect.")
			pass
		
		for s_i in tqdm.tqdm(segments_to_analyze):
			try:
				seg_decode_epoch_prob = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '.npy')
			except:
				print("Segment " + str(s_i) + " Never Decoded")
				pass
			
			seg_decode_save_dir = epoch_decode_save_dir + 'segment_' + str(s_i) + '/'
			if not os.path.isdir(seg_decode_save_dir):
				os.mkdir(seg_decode_save_dir)
			
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = seg_end - seg_start #in dt = ms
			
			#Import raster plots for segment
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in taste_select_neur:
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			
			decoded_taste_max = np.argmax(seg_decode_epoch_prob,0)
			
			#Calculate decoded taste stats by probability cutoff
			num_neur_mean_p = np.zeros((num_tastes,num_prob))
			num_neur_std_p =np.zeros((num_tastes,num_prob))
			iei_mean_p = np.zeros((num_tastes,num_prob))
			iei_std_p = np.zeros((num_tastes,num_prob))
			len_mean_p = np.zeros((num_tastes,num_prob))
			len_std_p = np.zeros((num_tastes,num_prob))
			prob_mean_p = np.zeros((num_tastes,num_prob))
			prob_std_p = np.zeros((num_tastes,num_prob))
			for t_i in range(num_tastes):
				for prob_i, prob_val in enumerate(prob_cutoffs):
					#Find where the decoding matches the probability cutoff for each taste
					decode_prob_bin = seg_decode_epoch_prob[t_i,:] > prob_val
					decode_max_bin =  decoded_taste_max == t_i
					decoded_taste = (decode_prob_bin*decode_max_bin).astype('int')
					decoded_taste[0] = 0
					decoded_taste[-1] = 0
					
					#Store the decoding percents
					epoch_seg_taste_percents[prob_i,e_i,s_i,t_i] = (np.sum(decoded_taste)/len(decoded_taste))*100
				
					#Calculate statistics of num neur, IEI, event length, average decoding prob
					start_decoded = np.where(np.diff(decoded_taste) == 1)[0] + 1
					end_decoded = np.where(np.diff(decoded_taste) == -1)[0] + 1
					num_decoded = len(start_decoded)
					
					#Create plots of decoded period statistics
					#__Length
					len_decoded = end_decoded-start_decoded
					len_mean_p[t_i,prob_i] = np.nanmean(len_decoded)
					len_std_p[t_i,prob_i] = np.nanstd(len_decoded)
					#__IEI
					iei_decoded = start_decoded[1:] - end_decoded[:-1]
					iei_mean_p[t_i,prob_i] = np.nanmean(iei_decoded)
					iei_std_p[t_i,prob_i] = np.nanstd(iei_decoded)
					num_neur_decoded = np.zeros(num_decoded)
					prob_decoded = np.zeros(num_decoded)
					for nd_i in range(num_decoded):
						d_start = start_decoded[nd_i]
						d_end = end_decoded[nd_i]
						d_len = d_end-d_start
						for n_i in range(num_neur):
							if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
								num_neur_decoded[nd_i] += 1
						prob_decoded[nd_i] = np.mean(seg_decode_epoch_prob[t_i,d_start:d_end])
					#__Num Neur
					num_neur_mean_p[t_i,prob_i] = np.nanmean(num_neur_decoded)
					num_neur_std_p[t_i,prob_i] = np.nanstd(num_neur_decoded)
					#__Prob
					prob_mean_p[t_i,prob_i] = np.nanmean(prob_decoded)
					prob_std_p[t_i,prob_i] = np.nanstd(prob_decoded)
				
			#Plot statistics 
			f,ax = plt.subplots(2,3,figsize=(8,8),width_ratios=[10,10,1])
			gs = ax[0,-1].get_gridspec()
			ax[0,0].set_ylim([0,num_neur])
			ax[0,1].set_ylim([0,np.nanmax(len_mean_p) + np.nanmax(len_std_p) + 10])
			ax[1,0].set_ylim([0,np.nanmax(iei_mean_p) + np.nanmax(iei_std_p) + 10])
			ax[1,1].set_ylim([0,1.2])
			for t_i in range(num_tastes):
				#__Num Neur
				ax[0,0].errorbar(prob_cutoffs,num_neur_mean_p[t_i,:],num_neur_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__Length
				ax[0,1].errorbar(prob_cutoffs,len_mean_p[t_i,:],len_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__IEI
				ax[1,0].errorbar(prob_cutoffs,iei_mean_p[t_i,:],iei_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__Prob
				ax[1,1].errorbar(prob_cutoffs,prob_mean_p[t_i,:],prob_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
			ax[0,0].set_title('Number of Neurons')
			ax[0,1].set_title('Length of Event')
			ax[1,0].set_title('IEI (ms)')
			ax[1,1].set_title('Average P(Decoding)')
			for ax_i in ax[:,-1]:
				ax_i.remove() #remove the underlying axes
			axbig = f.add_subplot(gs[:,-1])
			cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),cax=axbig,ticks=np.linspace(0,1,num_tastes),orientation='vertical')
			cbar.ax.set_yticklabels(dig_in_names)
			#Edit and Save
			f.suptitle('Decoding Statistics by Probability Cutoff')
			plt.tight_layout()
			f.savefig(os.path.join(seg_decode_save_dir,'prob_cutoff_stats.png'))
			f.savefig(os.path.join(seg_decode_save_dir,'prob_cutoff_stats.svg'))
			plt.close(f)
				
	#Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments	
	sum_width_ratios = np.concatenate((10*np.ones(len(segments_to_analyze)),np.ones(1)))
	max_decoding_percent = np.max(epoch_seg_taste_percents)
	f, ax = plt.subplots(num_cp,len(segments_to_analyze) + 1,figsize=(10,10),width_ratios=sum_width_ratios)
	gs = ax[0,-1].get_gridspec()
	for e_i in epochs_to_analyze:
		for s_i in segments_to_analyze:
			ax[e_i,s_i].set_ylim([0,max_decoding_percent+10])
			for t_i in range(num_tastes):
				ax[e_i,s_i].plot(prob_cutoffs,(epoch_seg_taste_percents[:,e_i,s_i,t_i]).flatten(),color=taste_colors[t_i,:],alpha=0.8)
			if s_i == 0:
				ax[e_i,s_i].set_ylabel('Epoch ' + str(e_i))
			if e_i == 0:
				ax[e_i,s_i].set_title(segment_names[s_i])
			if e_i == num_cp-1:
				ax[e_i,s_i].set_xlabel('Probability Cutoff')
	for ax_i in ax[:,-1]:
		ax_i.remove() #remove the underlying axes
	axbig = f.add_subplot(gs[:,-1])
	cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),cax=axbig,ticks=np.linspace(0,1,num_tastes),orientation='vertical')
	cbar.ax.set_yticklabels(dig_in_names)
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents_By_Prob_Cutoff.png')
	f.savefig(save_dir + 'Decoding_Percents_By_Prob_Cutoff.svg')
	plt.close(f)
	
	
def plot_decoded_func_n(fr_dist,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
				 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
				 e_skip_dt,e_len_dt,dig_in_names,segment_times,
				 segment_names,taste_num_deliv,taste_select_epoch,
				 use_full,save_dir,max_decode,max_hz,seg_stat_bin,
				 epochs_to_analyze=[],segments_to_analyze=[]):
	"""Function to plot the decoding statistics as a function of number of 
	neurons firing within the decoded interval."""
	warnings.filterwarnings('ignore')
	num_segments = len(segment_spike_times)
	neur_cutoffs = np.arange(1,num_neur)
	num_cuts = len(neur_cutoffs)
	taste_colors = cm.viridis(np.linspace(0,1,num_tastes))
	epoch_seg_taste_percents = np.zeros((num_cuts,num_cp,num_segments,num_tastes))
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	if len(segments_to_analyze == 0):
		segments_to_analyze = np.arange(num_segments)

	for e_i in epochs_to_analyze: #By epoch conduct decoding
		print('Decoding Epoch ' + str(e_i))
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			print("Data not previously decoded, or passed directory incorrect.")
			pass
		
		for s_i in tqdm.tqdm(segments_to_analyze):
			try:
				seg_decode_epoch_prob = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '.npy')
			except:
				print("Segment " + str(s_i) + " Never Decoded")
				pass
			
			seg_decode_save_dir = epoch_decode_save_dir + 'segment_' + str(s_i) + '/'
			if not os.path.isdir(seg_decode_save_dir):
				os.mkdir(seg_decode_save_dir)
			
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = seg_end - seg_start #in dt = ms
			
			#Import raster plots for segment
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in taste_select_neur:
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			
			decoded_taste_max = np.argmax(seg_decode_epoch_prob,0)
			
			#Calculate decoded taste stats by probability cutoff
			num_neur_mean_p = np.zeros((num_tastes,num_cuts))
			num_neur_std_p =np.zeros((num_tastes,num_cuts))
			iei_mean_p = np.zeros((num_tastes,num_cuts))
			iei_std_p = np.zeros((num_tastes,num_cuts))
			len_mean_p = np.zeros((num_tastes,num_cuts))
			len_std_p = np.zeros((num_tastes,num_cuts))
			prob_mean_p = np.zeros((num_tastes,num_cuts))
			prob_std_p = np.zeros((num_tastes,num_cuts))
			for t_i in range(num_tastes):
				#First calculate neurons decoded in all decoded intervals
				decoded_taste = (decoded_taste_max == t_i).astype('int')
				decoded_taste[0] = 0
				decoded_taste[-1] = 0
				diff_decoded_taste = np.diff(decoded_taste)
				start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
				end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
				num_decoded = len(start_decoded)
				num_neur_decoded = np.zeros(num_decoded)
				prob_decoded = np.zeros(num_decoded)
				for nd_i in range(num_decoded):
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					d_len = d_end-d_start
					for n_i in range(num_neur):
						if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
							num_neur_decoded[nd_i] += 1
					prob_decoded[nd_i] = np.mean(seg_decode_epoch_prob[t_i,d_start:d_end])
				#Next look at stats as exclude by #neurons
				for cut_i, cut_val in enumerate(neur_cutoffs):
					#Find where the decoding matches the neuron cutoff
					decode_ind = np.where(num_neur_decoded > cut_val)[0]
					decoded_bin = np.zeros(np.shape(decoded_taste))
					for db in decode_ind:
						s_db = start_decoded[db]
						e_db = end_decoded[db]
						decoded_bin[s_db:e_db] = 1
					#Store the decoding percents
					epoch_seg_taste_percents[cut_i,e_i,s_i,t_i] = (np.sum(decoded_bin)/len(decoded_bin))*100
					#Calculate statistics of num neur, IEI, event length, average decoding prob
					num_decoded_i = num_neur_decoded[decode_ind]
					prob_decoded_i = prob_decoded[decode_ind]
					iei_i = start_decoded[decode_ind[1:]] - end_decoded[decode_ind[:-1]]
					len_i = end_decoded[decode_ind] - start_decoded[decode_ind]
					
					#Create plots of decoded period statistics
					#__Length
					len_decoded = end_decoded-start_decoded
					len_mean_p[t_i,cut_i] = np.nanmean(len_i)
					len_std_p[t_i,cut_i] = np.nanstd(len_i)
					#__IEI
					iei_decoded = start_decoded[1:] - end_decoded[:-1]
					iei_mean_p[t_i,cut_i] = np.nanmean(iei_i)
					iei_std_p[t_i,cut_i] = np.nanstd(iei_i)
					#__Num Neur
					num_neur_mean_p[t_i,cut_i] = np.nanmean(num_decoded_i)
					num_neur_std_p[t_i,cut_i] = np.nanstd(num_decoded_i)
					#__Prob
					prob_mean_p[t_i,cut_i] = np.nanmean(prob_decoded_i)
					prob_std_p[t_i,cut_i] = np.nanstd(prob_decoded_i)
				
			#Plot statistics 
			f,ax = plt.subplots(2,3,figsize=(8,8),width_ratios=[10,10,1])
			gs = ax[0,-1].get_gridspec()
			ax[0,0].set_ylim([0,num_neur])
			ax[0,1].set_ylim([0,np.nanmax(len_mean_p) + np.nanmax(len_std_p) + 10])
			ax[1,0].set_ylim([0,np.nanmax(iei_mean_p) + np.nanmax(iei_std_p) + 10])
			ax[1,1].set_ylim([0,1.2])
			for t_i in range(num_tastes):
				#__Num Neur
				ax[0,0].errorbar(neur_cutoffs,num_neur_mean_p[t_i,:],num_neur_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__Length
				ax[0,1].errorbar(neur_cutoffs,len_mean_p[t_i,:],len_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__IEI
				ax[1,0].errorbar(neur_cutoffs,iei_mean_p[t_i,:],iei_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__Prob
				ax[1,1].errorbar(neur_cutoffs,prob_mean_p[t_i,:],prob_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
			ax[0,0].set_title('Number of Neurons')
			ax[0,1].set_title('Length of Event')
			ax[1,0].set_title('IEI (ms)')
			ax[1,1].set_title('Average P(Decoding)')
			for ax_i in ax[:,-1]:
				ax_i.remove() #remove the underlying axes
			axbig = f.add_subplot(gs[:,-1])
			cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),cax=axbig,ticks=np.linspace(0,1,num_tastes),orientation='vertical')
			cbar.ax.set_yticklabels(dig_in_names)
			#Edit and Save
			f.suptitle('Decoding Statistics by Neuron Cutoff')
			plt.tight_layout()
			f.savefig(os.path.join(seg_decode_save_dir,'neur_cutoff_stats.png'))
			f.savefig(os.path.join(seg_decode_save_dir,'neur_cutoff_stats.svg'))
			plt.close(f)
				
	#Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments	
	sum_width_ratios = np.concatenate((10*np.ones(len(segments_to_analyze)),np.ones(1)))
	max_decoding_percent = np.max(epoch_seg_taste_percents)
	f, ax = plt.subplots(num_cp,num_segments + 1,figsize=(10,10),width_ratios=sum_width_ratios)
	gs = ax[0,-1].get_gridspec()
	for e_i in epochs_to_analyze:
		for s_i in segments_to_analyze:
			ax[e_i,s_i].set_ylim([0,max_decoding_percent+10])
			for t_i in range(num_tastes):
				ax[e_i,s_i].plot(neur_cutoffs,(epoch_seg_taste_percents[:,e_i,s_i,t_i]).flatten(),color=taste_colors[t_i,:],alpha=0.8)
			if s_i == 0:
				ax[e_i,s_i].set_ylabel('Epoch ' + str(e_i))
			if e_i == 0:
				ax[e_i,s_i].set_title(segment_names[s_i])
			if e_i == num_cp-1:
				ax[e_i,s_i].set_xlabel('Neuron Cutoff')
	for ax_i in ax[:,-1]:
		ax_i.remove() #remove the underlying axes
	axbig = f.add_subplot(gs[:,-1])
	cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),cax=axbig,ticks=np.linspace(0,1,num_tastes),orientation='vertical')
	cbar.ax.set_yticklabels(dig_in_names)
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents_By_Neur_Cutoff.png')
	f.savefig(save_dir + 'Decoding_Percents_By_Neur_Cutoff.svg')
	plt.close(f)
	

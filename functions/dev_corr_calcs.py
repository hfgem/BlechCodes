#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:44:18 2023

@author: hannahgermaine

This is a collection of functions for calculating and analyzing deviation
correlations to taste responses.
"""

import tqdm, os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit

def dev_corr(save_dir,segment_spike_times,segment_names,segment_times,dev_times,
			 tastant_spike_times, dig_in_names, start_dig_in_times, end_dig_in_times, 
			 taste_intervals,taste_interval_names):
	"""This is a master function to call all other relevant functions for 
	calculating correlations between taste responses and deviation bins
	INPUTS:
		- save_dir: directory to save correlation results
		- segment_spike_times: true data spike times for each segment for each neuron
		- segment_names: names of each experimental segment
		- segment_times: times of each segment within the recording
		- dev_times: dictionary of deviation times for different number of 
			neurons firing cutoffs within each segment. For each cutoff contains 
			a list with two numpy arrays - start indices and end indices.
		- tastant_spike_times: times when spikes occur for each tastant delivery
		- dig_in_names: the name of each tastant
		- start_dig_in_times: times (in dt = ms) when taste deliveries start
		- end_dig_in_times: times (in dt = ms) when taste deliveries end
		- taste_intervals: times (in dt = ms) when different taste response 
			epochs start/end
		- taste_interval_names: names of different taste response epochs
	"""
	#Define paramters
	num_neur = len(segment_spike_times[0]) #Number of neurons in data
	num_segments = len(segment_names) #Number of segments in data
	conv_bin = 250 #Number of ms for PSTH smoothing (250 ms in usual plots)
	conv_step = 10 #Step size of PSTH smoothing
	indiv_plot_save_dir = save_dir + 'indiv_plots/' #Create a plot save directory
	if os.path.isdir(indiv_plot_save_dir) == False:
		os.mkdir(indiv_plot_save_dir)
	
	#First ask user to clarify which segment is the taste segment
	taste_index = get_taste_segment_input(segment_names)
	
	#Import taste response rasters and segment deviation rasters
	taste_rasters = []
	for t_i in range(len(dig_in_names)):
		taste_rasters.append([spike_templates(end_dig_in_times[t_i][t_t],end_dig_in_times[t_i][t_t]+taste_intervals[-1],segment_spike_times[taste_index],num_neur) for t_t in range(len(start_dig_in_times[t_i]))])
	seg_dev_rasters = dev_spike_templates(segment_names,segment_times,segment_spike_times,dev_times)
	
	#Calculate correlations between individual deviation rasters and taste response rasters
	#Which neurons spiked correlated
	print("\nCalculating Which Neuron Spiked Correlations")
	which_spiked_true_corr = dict() #Storage for true correlation values
	for t_i in range(len(dig_in_names)): #For each tastant
		taste_name = dig_in_names[t_i]
		print("\t" + taste_name)
		taste_corr = dict()
		for s_i in tqdm.tqdm(range(num_segments)): #For each segment
			seg_name = segment_names[s_i]
			#Calculate the true correlation values and store
			taste_corr.update({seg_name:which_spiked_corr(taste_name,seg_name,taste_rasters[t_i],seg_dev_rasters[s_i],taste_intervals,taste_interval_names,indiv_plot_save_dir)})
		which_spiked_true_corr.update({taste_name:taste_corr})
	plot_spiked_corr('Which Neuron Spiked',which_spiked_true_corr,taste_interval_names,save_dir) #Plot results on one set of axes!
	#How many times each neuron spiked correlated
	print("\nCalculating Number of Spikes per Neuron Correlations")
	num_spiked_true_corr = dict() #Storage for true correlation values
	for t_i in range(len(dig_in_names)): #For each tastant
		taste_name = dig_in_names[t_i]
		print("\t" + taste_name)
		taste_corr = dict()
		for s_i in tqdm.tqdm(range(num_segments)): #For each segment
			seg_name = segment_names[s_i]
			#Calculate the true correlation values and store
			taste_corr.update({seg_name:num_spiked_corr(taste_name,seg_name,taste_rasters[t_i],seg_dev_rasters[s_i],taste_intervals,taste_interval_names,indiv_plot_save_dir)})
		num_spiked_true_corr.update({taste_name:taste_corr})
	plot_spiked_corr('Number of Times a Neuron Spiked',num_spiked_true_corr,taste_interval_names,save_dir) #Plot results on one set of axes!
	#When neurons spiked correlated
	print("\nCalculating Binary Spike Time Correlations")
	when_spiked_true_corr = dict() #Storage for true correlation values
	for t_i in range(len(dig_in_names)): #For each tastant
		taste_name = dig_in_names[t_i]
		print("\t" + taste_name)
		taste_corr = dict()
		for s_i in tqdm.tqdm(range(num_segments)): #For each segment
			seg_name = segment_names[s_i]
			#Calculate the true correlation values and store
			taste_corr.update({seg_name:when_spiked_corr(taste_name,seg_name,taste_rasters[t_i],seg_dev_rasters[s_i],num_neur,taste_intervals,taste_interval_names,indiv_plot_save_dir)})
		when_spiked_true_corr.update({taste_name:taste_corr})
	plot_spiked_corr('Binary Spike Matrix',when_spiked_true_corr,taste_interval_names,save_dir) #Plot results on one set of axes!
	#PSTH correlated
	print("\nCalculating PSTH Correlations")
	psth_spiked_true_corr = dict() #Storage for true correlation values
	for t_i in range(len(dig_in_names)): #For each tastant
		taste_name = dig_in_names[t_i]
		print("\t" + taste_name)
		taste_corr = dict()
		for s_i in tqdm.tqdm(range(num_segments)): #For each segment
			seg_name = segment_names[s_i]
			#Calculate the true correlation values and store
			taste_corr.update({seg_name:inst_fr_corr(taste_name,seg_name,taste_rasters[t_i],seg_dev_rasters[s_i],conv_bin,conv_step,num_neur,taste_intervals,taste_interval_names,indiv_plot_save_dir)})
		psth_spiked_true_corr.update({taste_name:taste_corr})
	plot_spiked_corr('PSTH ',when_spiked_true_corr,save_dir) #Plot results on one set of axes!
	
	#Calculate correlations between shuffled individual deviation rasters and taste response rasters
	
	

def get_taste_segment_input(segment_names):
	"""This function prompts the user for the index of the taste segment"""
	prompt_loop = 1
	while prompt_loop == 1:
		print(["Index " + str(i) + " = " + segment_names[i] for i in range(len(segment_names))])
		try:
			taste_index = int(input("INPUT REQUESTED: Please indicate which of the above indices is for the taste delivery interval: "))
			if 0 <= taste_index < len(segment_names):
				print("Taste index selected = " + str(taste_index))
				prompt_loop = 0
			else:
				print("Taste index given does not fall in expected index range. Try again.")
		except:
			print("\nERROR: non-integer entry was provided. Please try again.\n")	
	
	return taste_index
	
def spike_templates(start_ind, end_ind, spike_times, num_neur):
	"""This function uses given start/end times to create binary templates"""
	num_time = end_ind - start_ind
	template = np.zeros((num_neur,num_time))
	for n_i in range(num_neur):
		spike_time_indices = np.where((spike_times[n_i] > start_ind)*(spike_times[n_i] < end_ind))[0]
		ind_spike = [(spike_times[n_i][spike_time_indices[sti]] - start_ind).astype('int') for sti in range(len(spike_time_indices))]
		template[n_i,ind_spike] += 1
	return template

def dev_spike_templates(segment_names,segment_times,segment_spike_times,dev_times):	
	"""This function uses the deviation start/end times to create binary spike 
	templates"""
	#Grab segment rasters
	segment_dev_rasters = []
	for s_i in range(len(segment_names)):
		segment_spikes = segment_spike_times[s_i]
		num_neur = len(segment_spikes)
		start_segment = segment_times[s_i]
		segment_spikes = [segment_spikes[n_i] - start_segment for n_i in range(num_neur)]
		seg_bout_list = dev_times[s_i]
		seg_dev_rasters = [spike_templates(seg_bout_list[0][s_b],seg_bout_list[1][s_b],segment_spikes,num_neur) for s_b in tqdm.tqdm(range(len(seg_bout_list[0])))]
		segment_dev_rasters.append(seg_dev_rasters)
		
	return segment_dev_rasters

@jit(forceobj=True)
def which_spiked_corr(taste_name,seg_name,t_rasters,seg_rasters,
					  taste_intervals,taste_interval_names,save_dir):
	"""This function calculates the correlation of deviation bins with taste
	response intervals based on which neurons are spiking"""
	#Plot colors
	cm_subsection = np.linspace(0,1,len(taste_interval_names))
	cmap = [cm.gist_rainbow(x) for x in cm_subsection] #Color maps for each segment
	#Storage
	t_corr_vals = []
	#Generate a histogram
	plt.figure()
	for t_i in range(len(taste_interval_names)):
		corr_vals = []
		for t_r in range(len(t_rasters)):
			which_t = (np.sum(t_rasters[t_r][:,taste_intervals[t_i]:taste_intervals[t_i+1]],1) > 0).astype('int')
			for s_i in range(len(seg_rasters)):
				which_s = (np.sum(seg_rasters[s_i],1) > 0).astype('int')
				stat_result = stats.pearsonr(which_t,which_s)
				corr_vals.extend([np.abs(stat_result[0])])
		corr_vals = np.array(corr_vals)
		t_corr_vals.append(corr_vals)
		#Plot histogram and mean
		plt.hist(corr_vals,color=cmap[t_i],alpha=0.3,label=taste_interval_names[t_i] + ' corr distribution')
		mean_val = np.round(np.nanmean(corr_vals),2)
		plt.axvline(mean_val,label=str(mean_val),color=cmap[t_i])
	plt.legend()
	plt.title('Which Spiked Corr: ' + taste_name + ' x ' + seg_name)
	plt.tight_layout()
	plt.savefig(save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_which_corr.png')
	plt.savefig(save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_which_corr.svg')
	plt.close()
	t_corr_vals = np.array(t_corr_vals)
			
	return t_corr_vals

@jit(forceobj=True)
def num_spiked_corr(taste_name,seg_name,t_rasters,seg_rasters,
					taste_intervals,taste_interval_names,save_dir):
	"""This function calculates the correlation of deviation bins with taste 
	response intervals based on the number of times each neuron spikes"""
	#Plot colors
	cm_subsection = np.linspace(0,1,len(taste_interval_names))
	cmap = [cm.gist_rainbow(x) for x in cm_subsection] #Color maps for each segment
	#Storage
	t_corr_vals = []
	#Generate a histogram
	plt.figure()
	for t_i in range(len(taste_interval_names)):
		corr_vals = []
		for t_r in range(len(t_rasters)):
			num_t = np.sum(t_rasters[t_r][:,taste_intervals[t_i]:taste_intervals[t_i+1]],1)
			for s_i in range(len(seg_rasters)):
				num_s = np.sum(seg_rasters[s_i],1)
				stat_result = stats.pearsonr(num_t,num_s)
				corr_vals.extend([np.abs(stat_result[0])])
		corr_vals = np.array(corr_vals)
		t_corr_vals.append(corr_vals)
		#Generate a histogram
		plt.hist(corr_vals,color=cmap[t_i],alpha=0.3,label=taste_interval_names[t_i] +' corr distribution')
		mean_val = np.round(np.nanmean(corr_vals),2)
		plt.axvline(mean_val,color=cmap[t_i],label=str(mean_val))
	plt.legend()
	plt.title('Num Spiked Corr: ' + taste_name + ' x ' + seg_name)
	plt.tight_layout()
	plt.savefig(save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_num_corr.png')
	plt.savefig(save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_num_corr.svg')
	plt.close()
	t_corr_vals = np.array(t_corr_vals)
	
	return t_corr_vals

@jit(forceobj=True)
def when_spiked_corr(taste_name,seg_name,t_rasters,seg_rasters,num_neur,
					 taste_intervals,taste_interval_names,save_dir):
	"""This function calculates the correlation of deviation bins with taste 
	response intervals based on the order in which neurons are spiking"""
	#Plot colors
	cm_subsection = np.linspace(0,1,len(taste_interval_names))
	cmap = [cm.gist_rainbow(x) for x in cm_subsection] #Color maps for each segment
	#Storage
	t_corr_vals = []
	#Generate a histogram
	plt.figure()
	for t_i in range(len(taste_interval_names)):
		corr_vals = []
		for t_r in range(len(t_rasters)):
			t_mat = t_rasters[t_r][:,taste_intervals[t_i]:taste_intervals[t_i+1]]
			t_len = len(t_mat[0,:])
			for s_i in range(len(seg_rasters)):
				s_mat = seg_rasters[s_i]
				s_len = len(s_mat[0,:])
				#Make same size
				if s_len < t_len:
					len_rat = t_len/s_len
					s_mat_new = np.zeros((num_neur,t_len))
					for n_i in range(num_neur):
						s_mat_ind = np.where(s_mat[n_i] > 0)[0]
						num_spikes = len(s_mat_ind)
						if num_spikes > 0:
							s_mat_ind_max = [int(np.min((np.ceil(s_mat_ind[smi] + len_rat),t_len))) for smi in range(num_spikes)]
							s_mat_ind_min = [int(np.max((np.ceil(s_mat_ind[smi] - len_rat),0))) for smi in range(num_spikes)]
							for smi in range(num_spikes):
								s_mat_new[n_i,s_mat_ind_min[smi]:s_mat_ind_max[smi]] = 1
					t_mat_new = t_mat
				elif t_len < s_len:
					len_rat = s_len/t_len
					t_mat_new = np.zeros((num_neur,s_len))
					for n_i in range(num_neur):
						t_mat_ind = np.where(t_mat[n_i] > 0)[0]
						num_spikes = len(t_mat_ind)
						if num_spikes > 0:
							t_mat_ind_max = [int(np.min((np.ceil(t_mat_ind[tmi] + len_rat),s_len))) for tmi in range(num_spikes)]
							t_mat_ind_min = [int(np.max((np.ceil(t_mat_ind[tmi] + len_rat),0))) for tmi in range(num_spikes)]
							for tmi in range(num_spikes):
								t_mat_new[n_i,t_mat_ind_min[tmi]:t_mat_ind_max[tmi]] = 1
					s_mat_new = s_mat
				else:
					t_mat_new = t_mat
					s_mat_new = s_mat
				#Calculate correlation by flattening
				stat_result = stats.pearsonr(t_mat_new.flatten(),s_mat_new.flatten())
				corr_vals.extend([np.abs(stat_result[0])])
		corr_vals = np.array(corr_vals)
		t_corr_vals.append(corr_vals)
		#Generate a histogram
		plt.hist(corr_vals,color=cmap[t_i],alpha=0.3,label=taste_interval_names[t_i] +' corr distribution')
		mean_val = np.round(np.nanmean(corr_vals),2)
		plt.axvline(mean_val,color=cmap[t_i],label=str(mean_val))
	plt.legend()
	plt.title('When Spiked Corr: ' + taste_name + ' x ' + seg_name)
	plt.tight_layout()
	plt.savefig(save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_when_corr.png')
	plt.savefig(save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_when_corr.svg')
	plt.close()
	t_corr_vals = np.array(t_corr_vals)
	
	return t_corr_vals
	
@jit(forceobj=True)
def inst_fr_corr(taste_name,seg_name,t_rasters,seg_rasters,conv_bin,conv_step,
				 num_neur,taste_intervals,taste_interval_names,save_dir):
	"""This function calculates the correlation of deviation bins with taste 
	response intervals based on the instantaneous firing rate"""
	#Plot colors
	cm_subsection = np.linspace(0,1,len(taste_interval_names))
	cmap = [cm.gist_rainbow(x) for x in cm_subsection] #Color maps for each segment
	#Storage
	t_corr_vals = []
	#Generate a histogram
	plt.figure()
	for t_i in range(len(taste_interval_names)):
		corr_vals = []
		for t_r in range(len(t_rasters)):
			t_mat = t_rasters[t_r][:,taste_intervals[t_i]:taste_intervals[t_i+1]]
			t_len = len(t_mat[0,:])
			for s_i in range(len(seg_rasters)):
				s_mat = seg_rasters[s_i]
				s_len = len(s_mat[0,:])
				#Make same size
				if s_len < t_len:
					max_len = t_len
					len_rat = t_len/s_len
					s_mat_new = np.zeros((num_neur,t_len))
					for n_i in range(num_neur):
						s_mat_ind = np.where(s_mat[n_i] > 0)[0]
						num_spikes = len(s_mat_ind)
						if num_spikes > 0:
							s_mat_ind_max = [int(np.min((np.ceil(s_mat_ind[smi] + len_rat),t_len))) for smi in range(num_spikes)]
							s_mat_ind_min = [int(np.max((np.ceil(s_mat_ind[smi] - len_rat),0))) for smi in range(num_spikes)]
							for smi in range(num_spikes):
								s_mat_new[n_i,s_mat_ind_min[smi]:s_mat_ind_max[smi]] = 1
					t_mat_new = t_mat
				elif t_len < s_len:
					max_len = s_len
					len_rat = s_len/t_len
					t_mat_new = np.zeros((num_neur,s_len))
					for n_i in range(num_neur):
						t_mat_ind = np.where(t_mat[n_i] > 0)[0]
						num_spikes = len(t_mat_ind)
						if num_spikes > 0:
							t_mat_ind_max = [np.min((np.ceil(t_mat_ind[tmi] + len_rat),s_len)) for tmi in range(num_spikes)]
							t_mat_ind_min = [np.max((np.ceil(t_mat_ind[tmi] + len_rat),0)) for tmi in range(num_spikes)]
							for tmi in range(num_spikes):
								t_mat_new[n_i,t_mat_ind_min[tmi]:t_mat_ind_max[tmi]] = 1
					s_mat_new = s_mat
				else:
					max_len = s_len
					s_mat_new = s_mat
					t_mat_new = t_mat
				#Create PSTH matrices
				t_PSTH_mat = np.zeros((num_neur,max_len))
				s_PSTH_mat = np.zeros((num_neur,max_len))
				PSTH_x_labels = np.arange(0,max_len,conv_step)
				for b_i in range(len(PSTH_x_labels)):
					left_ind = int(max(round(b_i*conv_step - 0.5*conv_bin),0))
					right_ind = int(min(round(b_i*conv_step + 0.5*conv_bin),max_len))
					t_PSTH_mat[:,b_i] = np.sum(t_mat_new[:,left_ind:right_ind],1)
					s_PSTH_mat[:,b_i] = np.sum(s_mat_new[:,left_ind:right_ind],1)
				t_PSTH_avg = (np.mean(t_PSTH_mat,0)/conv_bin)*1000 #Converted to Hz
				s_PSTH_avg = (np.mean(s_PSTH_mat,0)/conv_bin)*1000 #Converted to Hz
				#Calculate correlation by flattening
				stat_result = stats.pearsonr(t_PSTH_avg.flatten(),s_PSTH_avg.flatten())
				corr_vals.extend([np.abs(stat_result[0])])
		corr_vals = np.array(corr_vals)
		t_corr_vals.append(corr_vals)
		#Generate a histogram
		plt.hist(corr_vals,color=cmap[t_i],alpha=0.3,label=taste_interval_names[t_i] +' corr distribution')
		mean_val = np.round(np.nanmean(corr_vals),2)
		plt.axvline(mean_val,color=cmap[t_i],label=str(mean_val))
	plt.legend()
	plt.title('PSTH Corr: ' + taste_name + ' x ' + seg_name)
	plt.tight_layout()
	plt.savefig(save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_PSTH_corr.png')
	plt.savefig(save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_PSTH_corr.svg')
	plt.close()
	t_corr_vals = np.array(t_corr_vals)
	
	return t_corr_vals


@jit(forceobj=True)
def plot_spiked_corr(name_corr,spiked_true_corr,taste_interval_names,save_dir):
	"""Function to plot correlation distributions from any calculator"""
	num_tastes = len(spiked_true_corr)
	for t_i in range(len(taste_interval_names)):
		fig = plt.figure(figsize=(10,5))
		subplot_ind = 1
		for taste_key in spiked_true_corr:
			num_segments = len(spiked_true_corr[taste_key])
			cm_subsection = np.linspace(0,1,num_segments)
			cmap = [cm.gist_rainbow(x) for x in cm_subsection] #Color maps for each segment
			plt.subplot(1,num_tastes,subplot_ind)
			subplot_ind += 1
			s_i = 0
			for seg_key in spiked_true_corr[taste_key]:
				plt.hist(spiked_true_corr[taste_key][seg_key][t_i,:],alpha=0.2,color=cmap[s_i],label=seg_key)
				mean_val = np.nanmean(spiked_true_corr[taste_key][seg_key][t_i,:])
				plt.axvline(mean_val,color=cmap[s_i],label=seg_key + ' mean = ' + str(np.round(mean_val,2)))
				s_i += 1
			plt.legend()
			plt.title(taste_key)
			plt.tight_layout()
		plt.suptitle(taste_interval_names[t_i] + ' |Pearson Correlations| for ' + name_corr)
		plt.tight_layout()
		save_name_corr = taste_interval_names[t_i] + '_' + ('_').join(name_corr.split(' '))
		fig.savefig(save_dir+save_name_corr+'_corr_hist.png')
		fig.savefig(save_dir+save_name_corr+'_corr_hist.svg')
		plt.close()
	




	
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
		- dev_times: list with, for each segment, two numpy arrays - start indices and end indices.
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
	print("\nCreating Spike Templates")
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
	plot_spiked_corr('Which Neuron Spiked',which_spiked_true_corr,taste_interval_names,save_dir) #Plot histogram results on one set of axes!
	plot_violin_corr('Which Neuron Spiked',which_spiked_true_corr,taste_interval_names,save_dir) #Plot violin plot results on one set of axes!
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
	#Normalize by length
	plot_spiked_corr('Number of Times a Neuron Spiked',num_spiked_true_corr,taste_interval_names,save_dir) #Plot results on one set of axes!
	plot_violin_corr('Number of Times a Neuron Spiked',num_spiked_true_corr,taste_interval_names,save_dir) #Plot violin plot results on one set of axes!
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
	plot_violin_corr('Binary Spike Matrix',when_spiked_true_corr,taste_interval_names,save_dir) #Plot violin plot results on one set of axes!
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
	plot_spiked_corr('PSTH ',when_spiked_true_corr,taste_interval_names,save_dir) #Plot results on one set of axes!
	plot_violin_corr('PSTH ',when_spiked_true_corr,taste_interval_names,save_dir) #Plot violin plot results on one set of axes!
	
	#Calculate correlations between shuffled individual deviation rasters and taste response rasters
	print("To Do: add correlation calculations for shuffles.")

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
		#start_segment = segment_times[s_i]
		#segment_spikes = [segment_spikes[n_i] - start_segment for n_i in range(num_neur)]
		seg_bout_list = dev_times[s_i]
		seg_dev_rasters = [spike_templates(seg_bout_list[0][s_b],seg_bout_list[1][s_b],segment_spikes,num_neur) for s_b in tqdm.tqdm(range(len(seg_bout_list[0])))]
		segment_dev_rasters.append(seg_dev_rasters)
		
	return segment_dev_rasters

@jit(forceobj=True)
def which_spiked_corr(taste_name,seg_name,t_rasters,seg_rasters,
					  taste_intervals,taste_interval_names,plot_save_dir):
	"""This function calculates the correlation of deviation bins with taste
	response intervals based on which neurons are spiking"""
	np.seterr(divide='ignore', invalid='ignore')
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
		mean_val = np.round(np.nanmean(corr_vals),2)
		if ~np.isnan(mean_val):
			#Plot histogram and mean
			plt.hist(corr_vals,color=cmap[t_i],alpha=0.3,label=taste_interval_names[t_i] + ' corr distribution')
			plt.axvline(mean_val,label=str(mean_val),color=cmap[t_i])
		else:
			print('\t\tNaN value encountered.')
	plt.legend()
	plt.title('Which Spiked Corr: ' + taste_name + ' x ' + seg_name)
	plt.tight_layout()
	plt.savefig(plot_save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_which_corr.png')
	plt.savefig(plot_save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_which_corr.svg')
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
			t_mat = t_rasters[t_r][:,taste_intervals[t_i]:taste_intervals[t_i+1]]
			num_t = np.sum(t_mat,1)/len(t_mat[0,:])
			for s_i in range(len(seg_rasters)):
				num_s = np.sum(seg_rasters[s_i],1)/len(seg_rasters[s_i][0,:]) #Normalize to length
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
				 num_neur,taste_intervals,taste_interval_names,plot_save_dir):
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
							s_mat_ind_max = [np.min((np.ceil(s_mat_ind[smi] + len_rat),t_len)).astype('int') for smi in range(num_spikes)]
							s_mat_ind_min = [np.max((np.ceil(s_mat_ind[smi] - len_rat),0)).astype('int') for smi in range(num_spikes)]
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
							t_mat_ind_max = [np.min((np.ceil(t_mat_ind[tmi] + len_rat),s_len)).astype('int') for tmi in range(num_spikes)]
							t_mat_ind_min = [np.max((np.ceil(t_mat_ind[tmi] + len_rat),0)).astype('int') for tmi in range(num_spikes)]
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
	plt.savefig(plot_save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_PSTH_corr.png')
	plt.savefig(plot_save_dir + ('_').join(taste_name.split(' ')) + '_' + ('_').join(seg_name.split(' ')) + '_PSTH_corr.svg')
	plt.close()
	t_corr_vals = np.array(t_corr_vals)
	
	return t_corr_vals


@jit(forceobj=True)
def plot_spiked_corr(name_corr,spiked_true_corr,taste_interval_names,plot_save_dir):
	"""Function to plot correlation distributions from any calculator"""
	num_tastes = len(spiked_true_corr)
	for t_i in range(len(taste_interval_names)):
		fig1 = plt.figure(figsize=(10,5))
		fig2 = plt.figure(figsize=(10,5))
		subplot_ind = 1
		for taste_key in spiked_true_corr:
			num_segments = len(spiked_true_corr[taste_key])
			cm_subsection = np.linspace(0,1,num_segments)
			cmap = [cm.gist_rainbow(x) for x in cm_subsection] #Color maps for each segment
			ax1 = fig1.add_subplot(1,num_tastes,subplot_ind)
			ax2 = fig2.add_subplot(1,num_tastes,subplot_ind)
			subplot_ind += 1
			s_i = 0
			for seg_key in spiked_true_corr[taste_key]:
				ax1.hist(spiked_true_corr[taste_key][seg_key][t_i,:],alpha=0.2,color=cmap[s_i],label=seg_key)
				ax2.hist(spiked_true_corr[taste_key][seg_key][t_i,:],100,density=True, histtype='step',
                           cumulative=True,color=cmap[s_i],label=seg_key)
				mean_val = np.nanmean(spiked_true_corr[taste_key][seg_key][t_i,:])
				ax1.axvline(mean_val,color=cmap[s_i],label=seg_key + ' mean = ' + str(np.round(mean_val,2)))
				s_i += 1
			ax1.legend()
			ax1.set_title(taste_key)
			ax1.set_xlabel('|Pearson Correlation|')
			ax1.set_ylabel('Number of Occurrences')
			ax2.legend()
			ax2.set_title(taste_key)
			ax2.set_xlabel('x = |Pearson Correlation|')
			ax2.set_ylabel('P(X <= x)')
		fig1.suptitle(taste_interval_names[t_i] + ' |Pearson Correlations| for ' + name_corr)
		fig2.suptitle(taste_interval_names[t_i] + ' Cumulative Distribution of |Pearson Correlations| for ' + name_corr)
		fig1.tight_layout()
		fig2.tight_layout()
		save_name_corr = taste_interval_names[t_i] + '_' + ('_').join(name_corr.split(' '))
		fig1.savefig(plot_save_dir+save_name_corr+'_corr_hist.png')
		fig1.savefig(plot_save_dir+save_name_corr+'_corr_hist.svg')
		fig2.savefig(plot_save_dir+save_name_corr+'_corr_cum_hist.png')
		fig2.savefig(plot_save_dir+save_name_corr+'_corr_cum_hist.svg')
		plt.close('all')
	

@jit(forceobj=True)
def plot_violin_corr(name_corr,spiked_true_corr,taste_interval_names,plot_save_dir):
	"""Function to create violin plots of correlation values by segment and 
	calculate whether distributions are the same"""
	
	"""Example Violin Plot Code:
		axs[0, 3].violinplot(data, pos, points=60, widths=0.7, showmeans=True,
                     showextrema=True, showmedians=True, bw_method=0.5,
                     quantiles=[[0.1], [], [], [0.175, 0.954], [0.75], [0.25]])
		axs[0, 3].set_title('Custom violinplot 4', fontsize=fs)
	"""
	
	"""Example KS-Test Code:
		stat_vals = stats.ks_2samp(data1, data2, alternative='two-sided', method='auto')
		if stat_vals[1] < 0.05 reject null hypothesis - data come from different distributions.
	"""
	
	num_tastes = len(spiked_true_corr)
	for t_i in range(len(taste_interval_names)):
		fig = plt.figure(figsize=(10,5))
		subplot_ind = 1
		taste_y_vals = []
		taste_x_ticks = []
		taste_subplot_titles = []
		for taste_key in spiked_true_corr:
			taste_x_ticks.extend([taste_key])
			num_segments = len(spiked_true_corr[taste_key])
			cm_subsection = np.linspace(0,1,num_segments)
			cmap = [cm.gist_rainbow(x) for x in cm_subsection] #Color maps for each segment
			#Pull out data for violin plot
			x_tick_labels = []
			x_pos = []
			y_vals = []
			s_i = 0
			for seg_key in spiked_true_corr[taste_key]:
				x_tick_labels.extend([seg_key])
				x_pos.extend([s_i])
				y_vals_with_nan = spiked_true_corr[taste_key][seg_key][t_i,:]
				y_vals.append(y_vals_with_nan[~np.isnan(y_vals_with_nan)])
				s_i += 1
			taste_y_vals.append(y_vals)
			taste_subplot_titles.append(x_tick_labels)
			#Calculate pair-wise KS-Test
			num_keys = len(x_pos)
			sig_array = np.zeros((num_keys,num_keys))
			for n_i_1 in range(num_keys):
				sig_array[n_i_1,n_i_1] = 1
				for n_i_2 in range(num_keys-n_i_1-1):
					stat_vals = stats.ks_2samp(y_vals[n_i_1], y_vals[n_i_1+n_i_2+1], alternative='two-sided')
					sig_array[n_i_1,n_i_1+n_i_2+1] = stat_vals[1]
					sig_array[n_i_1+n_i_2+1,n_i_1] = stat_vals[1]
			sig_bin = sig_array < 0.05
			#Plot results: figure per taste interval, subplot per taste
			ax = plt.subplot(1,num_tastes,subplot_ind)
			plt.violinplot(y_vals,x_pos,points=1000,showmeans=True,showextrema=False,showmedians=True)
			ax.set_xticks(x_pos)
			ax.set_xticklabels(x_tick_labels)
			y_val = 0.9*np.max(ax.get_yticks())
			for n_i_1 in range(num_keys):
				x_1 = n_i_1
				y_1 = y_val
				y_increment = 0.01
				for n_i_2 in range(num_keys - n_i_1 - 1):
					x_2 = n_i_1+n_i_2+1
					if sig_bin[x_1,x_2]:
						plt.plot([x_1, x_1, x_2, x_2], [y_1, y_1+y_increment, y_1+y_increment, y_1], alpha=0.5,lw=1.5, c=cmap[x_1])
						plt.text((x_1+x_2)*.5, y_1+y_increment, "*", ha='center', va='bottom', alpha=0.5, color=cmap[x_1])
						y_1 += 0.01
				y_val += 0.05
			plt.title(taste_key)
			plt.tight_layout()
			subplot_ind += 1
		plt.suptitle(taste_interval_names[t_i] + ' ' + name_corr + ' |Pearson Correlations|')
		plt.tight_layout()
		save_name_corr = taste_interval_names[t_i] + '_' + ('_').join(name_corr.split(' '))
		fig.savefig(plot_save_dir+save_name_corr+'_corr_violin.png')
		fig.savefig(plot_save_dir+save_name_corr+'_corr_violin.svg')
		plt.close()
		#Now create a figure with subplots of different segments and within each the two tastes
		fig = plt.figure(figsize=(10,10))
		subplot_n = len(taste_subplot_titles[0])
		subplot_sqrt = np.ceil(np.sqrt(subplot_n)).astype('int')
		for ax_i in range(subplot_n):
			ax = plt.subplot(subplot_sqrt,subplot_sqrt,ax_i+1)
			combined_data = [taste_y_vals[t_i][ax_i] for t_i in range(len(taste_y_vals))]
			plt.violinplot(combined_data,range(len(combined_data)),points=1000,showmeans=True,showextrema=False,showmedians=True)
			ax.set_xticks(range(len(combined_data)))
			ax.set_xticklabels(taste_x_ticks)
			y_val = 0.9*np.max(ax.get_yticks())
			num_violins = len(combined_data)
			sig_bin = np.zeros((num_violins,num_violins))
			for n_i_1 in range(num_violins):
				x_1 = n_i_1
				y_1 = y_val
				y_increment = 0.01
				for n_i_2 in range(num_violins - n_i_1 -1):
					x_2 = n_i_1+n_i_2+1
					stat_vals = stats.ks_2samp(combined_data[n_i_1], combined_data[x_2], alternative='two-sided')
					if stat_vals[1] < 0.05:
						plt.plot([x_1, x_1, x_2, x_2], [y_1, y_1+y_increment, y_1+y_increment, y_1], alpha=0.5,lw=1.5, c=cmap[x_1])
						plt.text((x_1+x_2)*.5, y_1+y_increment, "*", ha='center', va='bottom', alpha=0.5, color=cmap[x_1])
						y_1 += 0.01
				y_val += 0.05
			plt.title(taste_subplot_titles[0][ax_i])
		plt.suptitle(taste_interval_names[t_i] + ' ' + name_corr + ' |Pearson Correlations|')
		plt.tight_layout()
		save_name_corr = taste_interval_names[t_i] + '_' + ('_').join(name_corr.split(' '))
		fig.savefig(plot_save_dir+save_name_corr+'_corr_violin_by_seg.png')
		fig.savefig(plot_save_dir+save_name_corr+'_corr_violin_by_seg.svg')
		plt.close()
	


	
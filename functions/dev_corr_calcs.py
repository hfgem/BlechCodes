#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:44:18 2023

@author: hannahgermaine

This is a collection of functions for calculating and analyzing deviation
correlations to taste responses.
"""

impot numpy as np

def dev_corr(save_dir,segment_spike_times,segment_names,segment_times,dev_times,
			 tastant_spike_times, dig_in_names, start_dig_in_times, end_dig_in_times, 
			 taste_intervals):
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
		- taste_intervals: times (in dt = ms) when different taste response epochs start/end
		- start_dig_in_times: times (in dt = ms) when taste deliveries start
		- end_dig_in_times: times (in dt = ms) when taste deliveries end
	"""
	
	
	
def spike_templates(start_ind, end_ind, spike_times, num_neur):
	"""This function uses given start/end times to create binary templates"""
	num_time = end_ind - start_ind
	template = np.zeros((num_neur,num_time))
	for n_i in range(num_neur):
		ind_spike = np.where((spike_times[n_i] > start_ind)*(spike_times[n_i] < end_ind))[0] - start_ind
		template[n_i,ind_spike] += 1
	
	
	
	return template
	
def dev_spike_templates():	
	"""This function uses the deviation start/end times to create binary spike 
	templates"""
	
def which_spiked_corr():
	"""This function calculates the correlation of deviation bins with taste
	response intervals based on which neurons are spiking"""



def when_spiked_corr():
	"""This function calculates the correlation of deviation bins with taste 
	response intervals based on the order in which neurons are spiking"""
	
	
	
def inst_fr_corr():
	"""This function calculates the correlation of instantaneous firing rates 
	in deviation bins compared with those in taste response intervals"""



	
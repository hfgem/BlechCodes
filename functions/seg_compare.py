#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:35:22 2023

@author: hannahgermaine

This is a collection of functions for calculating and analyzing general 
cross-segment activity changes.
"""

import os, tqdm, itertools
os.environ["OMP_NUM_THREADS"] = "4"
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def bin_spike_counts(save_dir,segment_spike_times,segment_names,segment_times):
	"""This function calculates the number of spikes across bins in the recording
	and compares the distributions between/across segments to determine if overall
	firing has changed
	INPUTS:
		- save_dir: directory to save results
		- segment_spike_times: [num_segments x num_neur] nested list with indices of spikes in a ms timescale dataset (each index is a ms)
		- segment_names: the name of each segment
		- segment_times: the ms time delineations of each segment
	"""
	#Create save dir for individual distributions
	figure_save_dir = save_dir + 'indiv_distributions/'
	if os.path.isdir(figure_save_dir) == False:
		os.mkdir(figure_save_dir)
	#First calculate individual distributions, plot them, and save
	segment_calculations = dict()
	#Get distributions for different bin sizes
	bin_sizes = np.arange(0.05,2.05,0.05)
	print("\nCalculating distributions for different bin sizes.")
	for s_i in tqdm.tqdm(range(len(segment_names))):
		segment_spikes = segment_spike_times[s_i]
		num_neur = len(segment_spikes)
		segment_start_time = segment_times[s_i]
		segment_end_time = segment_times[s_i+1]
		segment_len = int(segment_end_time-segment_start_time)
		#Convert to a binary spike matrix
		bin_spike = np.zeros((num_neur,segment_len))
		for n_i in range(num_neur):
			spike_indices = (np.array(segment_spikes[n_i]) - segment_start_time).astype('int')
			bin_spike[n_i,spike_indices] = 1
		#Parallel code doesn't work yet - throws an error
		#results = Parallel(n_jobs=-1)(calculate_spike_count_distribution(bin_spike,bin_sizes[i]) for i in tqdm.tqdm(range(len(bin_sizes))))
		results = dict(calculate_spike_count_distribution(bin_spike,bin_sizes[i]) for i in range(len(bin_sizes)))
		#Save results to master dictionary
		segment_calculations.update({segment_names[s_i]:results})
		#Plot distributions
		plot_spike_count_distributions(results,segment_names[s_i],figure_save_dir)
	#Use the KS-Test to calculate if segment distributions are different
	print("\nCalculating KS-Test for pairs of segments.")
	s_i_pairs = list(itertools.combinations(segment_names, 2))
	segment_pair_calculations = dict()
	for pair_i in s_i_pairs:
		seg_1 = pair_i[0]
		seg_2 = pair_i[1]
		seg_1_data = segment_calculations[seg_1]
		seg_2_data = segment_calculations[seg_2]
		pair_results = KS_test_distributions(seg_1_data, seg_2_data)
		segment_pair_calculations.update({seg_1+"_"+seg_2 : pair_results})

def calculate_spike_count_distribution(spike_times,bin_size):
	"""This function calculates the spike count distribution for a given dataset
	and given bin sizes
	INPUTS:
		- spike_times: binary matrix of num_neur x num_time (in ms bins) with 1s where a neuron fires
		- bin_size: width (in seconds) of bins to calculate the number of spikes in
	"""
	bin_dt = int(bin_size*1000)
	bin_borders = np.arange(0,len(spike_times[0,:]),bin_dt)
	bin_counts = np.zeros(len(bin_borders)-1)
	for b_i in range(len(bin_borders)-1):
		bin_counts[b_i] = np.sum(spike_times[:,bin_borders[b_i]:bin_borders[b_i+1]])
	
	return str(bin_size), bin_counts
	

def plot_spike_count_distributions(results,title,save_location):
	"""This function plots given spike count distributions on the same axes for
	easy comparison. Given distributions must be input as a dictionary with name 
	and values to be easily plotted together"""
	fig = plt.figure(figsize=(5,5))
	for key in results:
		plt.hist(results[key],label=str(key),bins=25,alpha=0.5)
	plt.tight_layout()
	plt.legend()
	plt.xlabel('Number of Spikes per Bin')
	plt.ylabel('Occurrences')
	plt.title(title)
	im_name = ('_').join(title.split(' '))
	plt.savefig(save_location + im_name + '.png')
	plt.savefig(save_location + im_name + '.svg')
	plt.close()


def KS_test_distributions(dict_1, dict_2):
	"""This function performs a two-sample KS-test on a given pair of values:
	INPUTS: two dictionaries with matching keys - the matching keys' values 
			will be compared against each other with a KS-test.
	OUTPUTS: a dictionary with KS-test results for each matched key
	"""
	results = dict()
	for key in dict_1:
		values_1 = dict_1[key]
		values_2 = dict_2[key]
		ksresult = stats.ks_2samp(values_1,values_2)
		same = 1
		if ksresult[1] < 0.05:
			same = 0
		results.update({key:same})
	
	return results

def seg_compare():
	"""This function calculates whether given segments of the experiment are different
	from each other, by comparing each segment to a null distrbution which shuffles
	time bins from all given segments randomly"""
	print("Do something")
	

def	null_shuffle():
	"""This function creates a shuffled dataset to analyze"""
	print("Do something")
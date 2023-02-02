#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:01:49 2023

@author: Hannah Germaine

This is a collection of functions for performing changepoint detection on raster matrices
"""
import os
import numpy as np

def run_cd(fig_save_dir, spikes_mat, sampling_rate):
	"""This function runs changepoint detection with different numbers of changepoints
	and determines the best fit for the given data
	INPUTS:
		- fig_save_dir: directory to save changepoint output figures
		- spikes_mat: matrix of num_neur x num_timepoints with 1s where spikes occur
	OUTPUTS:
		- changepoint_times: indices where changepoints are predicted to occur
	"""
	#Create figure storage directory
	changepoint_dir = fig_save_dir + 'changepoints/'
	if os.path.isdir(changepoint_dir) == False:
		os.mkdir(changepoint_dir)
	
	#Calculate changepoints for different numbers
	changepoint_counts = np.arange(5) + 1 #array of changepoint counts to test
	changepoint_predictions = []
	for c_i in changepoint_counts:
		c_times = find_cp(c_i, spikes_mat, changepoint_dir)
		changepoint_predictions.append(c_times)
	
	#Determine which number of changepoints is the best fit
	
	
	return changepoint_times


def find_cp(c_num, spikes_mat, changepoint_dir, sampling_rate):
	"""This function runs changepoint detection given a number of changepoints
	INPUTS:
		- c_num = number of changepoints to calculate
		- spikes_mat = binary matrix of num_neur x num_timepoints with 1s when spikes occur
		- changepoint_dir = filepath to store changepoint figures
		- sampling_rate = number of samples / second (Hz)
	OUTPUTS:
		- changepoint_times = list of indices at which changepoints are predicted to occur
	"""
	min_seg_size = 0.2*sampling_rate #Minimum window size for a stable state (converted from s to "timepoints")
	
	
	return changepoint_times

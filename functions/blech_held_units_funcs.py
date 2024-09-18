#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:08:25 2024

@author: Hannah Germaine

This file contains functions needed for held unit analysis
"""

import numpy as np
from scipy.spatial.distance import cdist

def int_input(prompt):
	#This function asks a user for an integer input
	int_loop = 1	
	while int_loop == 1:
		response = input(prompt)
		try:
			int_val = int(response)
			int_loop = 0
		except:
			print("\tERROR: Incorrect data entry, please input an integer.")
	
	return int_val

def get_unit_info(all_unit_info, 
                  unit_info_labels=['electrode_number', 'fast_spiking', 
                                    'regular_spiking', 'single_unit']):
    return [all_unit_info[i] for i in unit_info_labels]

def calculate_J3(wf_1, wf_2):
    # Send these off to calculate J1 and J2
    J1 = calculate_J1(wf_1, wf_2)
    J2 = calculate_J2(wf_1, wf_2)
    # Get J3 as the ratio of J2 and J1
    J3 = J2/J1
    return J3

def calculate_J2(wf_day1, wf_day2):
    # Get the mean PCA waveforms on days 1 and 2
    day1_mean = np.mean(wf_day1, axis = 0)
    day2_mean = np.mean(wf_day2, axis = 0)
    
    # Get the overall inter-day mean
    overall_mean = np.mean(np.concatenate((wf_day1, wf_day2), axis = 0), axis = 0)

    # Get the distances of the daily means from the inter-day mean
    dist1 = cdist(day1_mean.reshape((-1, 4)), overall_mean.reshape((-1, 4)))
    dist2 = cdist(day2_mean.reshape((-1, 4)), overall_mean.reshape((-1, 4)))

    # Multiply the distances by the number of points on both days and sum to get J2
    J2 = wf_day1.shape[0]*np.sum(dist1) + wf_day2.shape[0]*np.sum(dist2)
    return J2 

def calculate_J1(wf_day1, wf_day2):
    # Get the mean PCA waveforms on days 1 and 2
    day1_mean = np.mean(wf_day1, axis = 0)
    day2_mean = np.mean(wf_day2, axis = 0)

    # Get the Euclidean distances of each day from its daily mean
    day1_dists = cdist(wf_day1, day1_mean.reshape((-1, 4)), metric = 'euclidean')
    day2_dists = cdist(wf_day2, day2_mean.reshape((-1, 4)), metric = 'euclidean')

    # Sum up the distances to get J1
    J1 = np.sum(day1_dists) + np.sum(day2_dists)
    return J1


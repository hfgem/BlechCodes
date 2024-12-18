#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:08:25 2024

@author: Hannah Germaine

This file contains functions needed for held unit analysis
"""

import tqdm
import numpy as np
from scipy.spatial.distance import cdist

def user_held_unit_input():
    # Ask the user for the number of days to be compared
    num_days = int_input("How many days-worth of data are you comparing for held units (integer)? ")
    
    # Ask the user whether or not to use the electrode index in the calculation of unit similarity
    use_electrode = bool_input("Would you like to use the electrode index in the calculation of unit similarity? ")

    # Ask the user for the percentile criterion to use to determine held units
    #percent_criterion = int_input('What percentile of intra-J3 do you want to use to pull out held units (provide an integer)? ')
    #percent_criterion_fr = int_input('What percentile of FR distances do you want to use to pull out held units (provide an integer)? ')

    # Ask the user for the waveform to use to determine held units
    # while_end = 0
    # while while_end == 0:
    #     wf_ind = int_input('Which types of waveforms should be used for held_unit analysis?' + \
    #                          '\n1: raw_CAR_waveform'
    #                          '\n2: norm_waveform' + '\nEnter the index: ')
    #     if wf_ind == 1:
    #         wf_type = 'raw_CAR_waveform'
    #         while_end = 1
    #     elif wf_ind == 2:
    #         wf_type = 'norm_waveform'
    #         while_end = 1
    #     else:
    #         print('Error: Incorrect entry, try again.')
            
    return num_days, use_electrode #, percent_criterion, percent_criterion_fr

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

def bool_input(prompt):
	#This function asks a user for an integer input
	bool_loop = 1	
	while bool_loop == 1:
		print("Respond with Y/y/N/n:")
		response = input(prompt)
		if (response.lower() != 'y')*(response.lower() != 'n'):
			print("\tERROR: Incorrect data entry, only give Y/y/N/n.")
		else:
			bool_val = response.lower()
			bool_loop = 0
	
	return bool_val

def get_unit_info(all_unit_info, 
                  unit_info_labels=['electrode_number', 'fast_spiking', 
                                    'regular_spiking', 'single_unit']):
    return [all_unit_info[i] for i in unit_info_labels]

def calculate_euc_dist(wf_1,wf_2):
    #Calculate the euclidean distance distribution
    num_1,num_components = np.shape(wf_1)
    num_2,_ = np.shape(wf_2)
    euc_dist = cdist(wf_1, wf_2, metric='euclidean')
    return np.nanmean(euc_dist)

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
    
    num_components = len(day1_mean)
    
    # Get the overall inter-day mean
    overall_mean = np.mean(np.concatenate((wf_day1, wf_day2), axis = 0), axis = 0)

    # Get the distances of the daily means from the inter-day mean
    dist1 = cdist(day1_mean.reshape((-1, num_components)), overall_mean.reshape((-1, num_components)))
    dist2 = cdist(day2_mean.reshape((-1, num_components)), overall_mean.reshape((-1, num_components)))

    # Multiply the distances by the number of points on both days and sum to get J2
    J2 = wf_day1.shape[0]*np.sum(dist1) + wf_day2.shape[0]*np.sum(dist2)
    return J2 

def calculate_J1(wf_day1, wf_day2):
    # Get the mean PCA waveforms on days 1 and 2
    day1_mean = np.mean(wf_day1, axis = 0)
    day2_mean = np.mean(wf_day2, axis = 0)
    
    num_components = len(day1_mean)

    # Get the Euclidean distances of each day from its daily mean
    day1_dists = cdist(wf_day1, day1_mean.reshape((-1, num_components)), metric = 'euclidean')
    day2_dists = cdist(wf_day2, day2_mean.reshape((-1, num_components)), metric = 'euclidean')

    # Sum up the distances to get J1
    J1 = np.sum(day1_dists) + np.sum(day2_dists)
    return J1


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:33:41 2022

@author: hannahgermaine
This set of functions pulls spikes without sorting
"""

import tables, os
import numpy as np

def run_spike_sort(data_dir):
	"""This function performs clustering spike sorting to separate out spikes
	from the dataset"""
	
	#Import data
	hf5 = tables.open_file(data_dir, 'r', title = data_dir[-1])
	data = hf5.root.clean_data[0,:,:]
	num_units, num_time = np.shape(data)
	sampling_rate = hf5.root.sampling_rate[0]
	segment_times = hf5.root.segment_times[:]
	segment_names = [hf5.root.segment_names[i].decode('UTF-8') for i in range(len(hf5.root.segment_names))]
	#Need to pull the times of different data segments to improve plotting
	hf5.close()
	del hf5
	downsamp_dir = ('_').join(data_dir.split('_')[:-1])+'_downsampled.h5'
	#Import downsampled dig-in data
	hf5 = tables.open_file(downsamp_dir, 'r', title = downsamp_dir[-1])
	dig_ins = hf5.root.dig_ins.dig_ins[0]
	dig_in_names = [hf5.root.dig_ins.dig_in_names[i].decode('UTF-8') for i in range(len(hf5.root.dig_ins.dig_in_names))]
	hf5.close()
	
	#Create directory for sorted data
	dir_save = ('/').join(data_dir.split('/')[:-1]) + '/nosort_results/'
	if os.path.isdir(dir_save) == False:
		os.mkdir(dir_save)
	#Create .h5 file for storage of results
	sort_hf5_name = dir_save.split('/')[-3].split('.')[0].split('_')[0] + '_nosort.h5'
	sort_hf5_dir = dir_save + sort_hf5_name
	if os.path.isfile(sort_hf5_dir) == False:
		sort_hf5 = tables.open_file(sort_hf5_dir, 'w', title = sort_hf5_dir[-1])
		sort_hf5.create_group('/','sorted_units')
		sort_hf5.close()
		
	
def no_sort():
	"""This function pulls spikes from each channel without true sorting - 
	simply grabbing values above threshold and automatically clustering and 
	removing clusters with too many violations"""
	
	
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:33:13 2022
@author: hannahgermaine
This set of functions deals with splitting electrode data into independent
components using ICA analysis.
"""

#Imports

import os, tqdm, tables, time
import numpy as np
from sklearn.decomposition import FastICA
import functions.data_processing as dp
from joblib import Parallel, delayed

def electrode_data_import(hf5_dir):
	"""This function imports data from the HDF5 file and stores only electrode 
	signals into an array"""
	folder_dir = hf5_dir.split('/')[:-1]
	folder_dir = '/'.join(folder_dir) + '/'
	new_hf5_dir = hf5_dir.split('.h5')[0] + '_downsampled.h5'
	
	#Has the electrode data already been stored in the hdf5 file?
	print("Checking for electrode array.")
	try:
		hf5_new = tables.open_file(new_hf5_dir, 'r', title = hf5_dir[-1])
		print("Data was previously stored in HDF5 file and is being imported. \n \n")
		e_data = hf5_new.root.electrode_array.data[0,:,:]
		dig_ins = hf5_new.root.dig_ins.dig_ins[0,:,:]
		unit_nums = hf5_new.root.electrode_array.unit_nums[:]
		hf5_new.close()
	except:
		print("Data was not previously stored in hf5.")
		#Ask about subsampling data
		sampling_rate = np.fromfile(folder_dir + 'info.rhd', dtype = np.dtype('float32'))
		sampling_rate = int(sampling_rate[2])
		sub_loop = 1
		sub_amount = 0
		while sub_loop == 1:
			print("The current sampling rate is " + str(sampling_rate) + ".")
			sub_q = input("Would you like to downsample? y / n: ")	
			if sub_q != 'n' and sub_q != 'y':
				print('Error, incorrect response, try again')
				sub_loop = 1
			else:
				sub_loop = 0
		if sub_q == 'y':
			sub_loop = 1
			while sub_loop == 1:
				sub_amount = input("Please enter a float describing the amount to downsample (ex. 0.5): ")
				try:
					sub_amount = float(sub_amount)
					sub_loop = 0
				except:
					print("Error. Please enter a valid float.")
					sub_loop = 1
			del sub_loop
		
		#Perform downsampling / regular data import
		e_data, dig_in_data = dp.data_to_list(sub_amount,sampling_rate,hf5_dir)
			
        #Save data to hdf5
		print('Saving Downsampled Electrode Data to New HF5')
		e_data, unit_nums, dig_ins = dp.save_downsampled_data(hf5_dir,e_data,sub_amount,sampling_rate,dig_in_data)
	
		#Perform ICA analysis
	
	return e_data, unit_nums, dig_ins
	

def ICA_analysis(e_data, new_hf5_dir):
	"""This function performs ICA component separation on the electrode recording
	data"""
	
	#Open HF5
	hf5_new = tables.open_file(new_hf5_dir, 'r+', title = new_hf5_dir[-1])
	
	#First pull only the portion of the data desired for ICA using
	#hf5_new.root.experiment_components.segment_times and .segment_names
	
	segment_names = hf5_new.root.experiment_components.segment_names[:]
	segment_times = hf5_new.root.experiment_components.segment_times[:]
	
	hf5_new.close() #Don't forget to close the HF5 file!
	
	#Ask for user input on which segment to use for ICA
	print("Using the entire dataset for ICA is computationally costly.")
	print("As such, this program was designed to run only on a segment.")
	[print(segment_names[i].decode('UTF-8') + " index = " + str(i)) for i in range(len(segment_names))]
	print(segment_names)
	print(np.arange(len(segment_names)))
	print("Above are the segment names and their corresponding indices.")
	seg_loop = 1
	while seg_loop == 1:
		seg_ind = input("Please enter the index of the segment you'll use for ICA: ")
		try:
			seg_ind = int(seg_ind)
			seg_loop = 0
		except:
			print("ERROR: Please enter a valid index.")
			
	data_segment = e_data[:,segment_times[seg_ind]:segment_times[seg_ind+1]]
	
	del e_data
	
	transformer = FastICA(n_components=None,
					   algorithm='parallel',max_iter=200,
					   tol=0.0001, w_init=None, random_state=None)
	X_transformed = transformer.fit_transform(data_segment)
	
	return X_transformed

def component_properties(components):
	"""This function attempts to characterize the pulled out components into spikes
	, noise, and other signals"""
	
	component_names = []
	
	return component_names
	

def performICA(hf5_dir):
	"""This function calls all necessary functions to perform ICA on
	electrode data"""
	
	print("Data Import Phase")
	e_data, unit_nums, dig_ins = electrode_data_import(hf5_dir)
	print("Performing Fast ICA")
	components = ICA_analysis(e_data)
	del e_data, unit_nums
	print("Saving to HF5")
	hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
	atom = tables.IntAtom()
	hf5.create_group('/','ica')
	hf5.create_earray('/ica','components',atom,(0,))
	exec("hf5.root.ica.components.append(components[:])")
	
	
	
	#component_names = component_properties(components)

	return components  #, component_names

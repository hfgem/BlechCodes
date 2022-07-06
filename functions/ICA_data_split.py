#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:33:13 2022
@author: hannahgermaine
This set of functions deals with splitting electrode data into independent
components using ICA analysis.
"""

#Imports

import os, tqdm, csv, tables
import numpy as np
from sklearn.decomposition import FastICA
import functions.data_processing as dp

def electrode_data_import(hf5_dir):
	"""This function imports data from the HDF5 file and stores only electrode 
	signals into an array"""
	#Open file
	hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
	folder_dir = hf5_dir.split('/')[:-2]
	csv_dir = '/'.join(folder_dir) + '/csv_storage/'
	if ~os.path.isdir(csv_dir):
		os.mkdir(csv_dir)
	electrode_save_dir = csv_dir + 'electrodes.csv'
	
	#Grab electrode info
	units = hf5.list_nodes('/raw')
	unit_nums = np.array([str(unit).split('_')[-1] for unit in units])
	unit_nums = np.array([int(unit.split(' (')[0]) for unit in unit_nums])
	
	#Pull one electrode's worth of data at a time into a list
	e_data = []
	print('Pulling electrode data into array')
	for unit in tqdm.tqdm(range(len(unit_nums))):
		e_data.append(np.ndarray.tolist(units[unit][:]))
	
	#Ask about subsampling data
	sampling_rate = np.fromfile('/'.join(folder_dir) + '/' + 'info.rhd', dtype = np.dtype('float32'))
	sampling_rate = int(sampling_rate[2])
	sub_loop = 1
	sub_amount = 0
	while sub_loop == 1:
		print("The current sampling rate is " + str() + ".")
		sub_q = input("Would you like to subsample? y / n: ")	
		if sub_q != 'n' and sub_q != 'y':
			print('Error, incorrect response, try again')
			sub_loop = 1
		else:
			sub_loop = 0
	if sub_q == 'y':
		sub_loop = 1
		while sub_loop == 1:
			sub_amount = input("Please enter a float describing the amount to subsample (ex. 0.5): ")
			try:
				sub_amount = float(sub_amount)
				sub_loop = 0
			except:
				print("Error. Please enter a valid float.")
				sub_loop = 1
	if sub_amount > 0:
		new_rate = round(sampling_rate*sub_amount)
		print("New sampling rate = " + str(new_rate))
		sub_data = dp.subsample(sampling_rate, new_rate, e_data)
		e_data = sub_data
		del sub_data
	
	#Save list to .csv
	with open(electrode_save_dir, 'w') as f:
		# using csv.writer method from CSV package
		write = csv.writer(f)
		write.writerows(e_data)
	
	#Close HDF5 file
	hf5.close()
	
	del units, hf5
	
	return e_data, unit_nums

def ICA_analysis(e_data):
	"""This function performs ICA component separation on the electrode recording
	data"""
	
	transformer = FastICA(n_components=None,
					   algorithm='parallel',max_iter=200,
					   tol=0.0001, w_init=None, random_state=None)
	X_transformed = transformer.fit_transform(e_data)
	
	return X_transformed

def component_properties(components):
	"""This function attempts to characterize the pulled out components into spikes
	, noise, and other signals"""
	
	component_names = []
	
	return component_names
	

def performICA(hf5_dir):
	"""This function calls all necessary functions to perform ICA on
	electrode data"""
	
	e_data, unit_nums = electrode_data_import(hf5_dir)
	components = ICA_analysis(e_data)
	component_names = component_properties(components)

	return components, component_names

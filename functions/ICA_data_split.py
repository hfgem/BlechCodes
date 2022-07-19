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
import functions.hdf5_handling as h5
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
	

def ICA_analysis(e_data, new_hf5_dir):
	"""This function performs ICA component separation on the electrode recording
	data"""
	
	#Open HF5
	hf5_new = tables.open_file(new_hf5_dir, 'r+', title = new_hf5_dir[-1])
	
	#First pull only the portion of the data desired for ICA using
	#hf5_new.root.experiment_components.segment_times and .segment_names
	
	segment_names = hf5_new.root.experiment_components.segment_names[:]
	segment_times = hf5_new.root.experiment_components.segment_times[:]
	sampling_rate = hf5_new.root.sampling_rate[0]
	
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
	
	print("Bandpass Filtering Data")
	low_fq = 300
	high_fq = 3000
	filtered_data = dp.bandpass_filter(data_segment, low_fq, high_fq,
									sampling_rate, order=5)
	
	print("Performing ICA")
	transformer = FastICA(n_components=None,
					   algorithm='parallel',max_iter=200,
					   tol=0.0001, w_init=None, random_state=None)
	ICA_weights = transformer.fit_transform(filtered_data)
	
	print("Plotting ICA Results")
	plot_ICA_results(filtered_data,ICA_weights,sampling_rate,new_hf5_dir)
	
	return ICA_weights

def component_properties(components):
	"""This function attempts to characterize the pulled out components into spikes
	, noise, and other signals"""
	
	component_names = []
	
	return component_names

def plot_ICA_results(filtered_data,ICA_weights,sampling_rate,new_hf5_dir):
	"""This function takes the data segment used for ICA and plots the original 
	segment data as well as the components pulled out by ICA"""
	
	#Create Folder for Image Storage
	split_dir = new_hf5_dir.split('/')
	im_folder_dir = '/'.join(split_dir[:-1]) + '/ICA_images/'
	if os.path.isdir(im_folder_dir) == False:
		os.mkdir(im_folder_dir)
	
	#Grab relevant parameters
	num_neur = len(filtered_data)
	max_ind = 200000
	num_sec = max_ind/sampling_rate
	sec_vec = (1/sampling_rate)*np.arange(max_ind)
	
	#Get Components from ICA
	print("Getting Components for Visualization")
	components = np.matmul(ICA_weights,filtered_data[:,0:max_ind])
	
	#Plot original data - small portion
	plt.figure(figsize=(30,20))
	for i in range(num_neur):
		plt.subplot(num_neur,1,i+1)
		plt.plot(sec_vec,filtered_data[i,0:max_ind])
	plt.suptitle('Original Data',fontsize=40)
	plt.savefig(im_folder_dir + 'orig_data.png', dpi=100)
	plt.show()
	
	#Plot components - small portion
	plt.figure(figsize=(30,20))
	for i in range(num_neur):
		plt.subplot(num_neur,1,i+1)
		plt.plot(sec_vec,components[i,:])
	plt.suptitle('Components',fontsize=40)
	plt.savefig(im_folder_dir + 'components.png', dpi=100)
	plt.show()	
	

def performICA(hf5_dir):
	"""This function calls all necessary functions to perform ICA on
	electrode data"""
	
	print("Downsampled Data Import Phase")
	e_data, unit_nums, dig_ins, new_hf5_dir = h5.downsampled_electrode_data_import(hf5_dir)
	print("Performing Fast ICA")
	ICA_weights = ICA_analysis(e_data, new_hf5_dir)
	del e_data, unit_nums
	print("Saving to HF5")
	hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
	atom = tables.IntAtom()
	hf5.create_group('/','ica')
	hf5.create_earray('/ica','components',atom,(0,))
	exec("hf5.root.ica.components.append(components[:])")
	
	#component_names = component_properties(components)

	return ICA_weights

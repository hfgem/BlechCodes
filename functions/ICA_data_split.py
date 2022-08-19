#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:33:13 2022
@author: hannahgermaine
This set of functions deals with splitting electrode data into independent
components using ICA analysis.
"""

#Imports

import os, tables
import numpy as np
from sklearn.decomposition import FastICA
import functions.hdf5_handling as h5
import functions.data_cleaning as dc
import matplotlib.pyplot as plt
import functions.ICA_hannah_method as ica
	

def data_sample(hf5_dir, clean_data_dir, new_hf5_dir):
	"""This function cleans a dataset using downsampling, bandpass filtering, 
	and signal averaging"""
	
	#Get segments and sampling rate
	##NEED TO GET SEGMENT NAMES AND INDICES 
	hf5_clean = tables.open_file(clean_data_dir, 'r+', title = new_hf5_dir[-1])
	sampling_rate = hf5_clean.root.sampling_rate[0]
	clean_data = hf5_clean.root.clean_data[0]
	hf5_clean.close()
	
	#Cleaned segment
	#Ask for user input on which segment to use for ICA
	print("Using the entire dataset for ICA is computationally costly.")
	print("As such, this program was designed to run only on a segment.")
	[print(segment_names[i].decode('UTF-8') + " index = " + str(i)) for i in range(len(segment_names))]
	print("Above are the segment names and their corresponding indices.")
	seg_loop = 1
	while seg_loop == 1:
		seg_ind = input("Please enter the index of the segment you'll use for ICA: ")
		try:
			seg_ind = int(seg_ind)
			seg_loop = 0
		except:
			print("ERROR: Please enter a valid index.")
	cleaned_data = avg_data[:,segment_times[seg_ind]:segment_times[seg_ind+1]]
		
	print("Saving cleaned dataset")
	if os.path.isdir(ICA_h5_dir) == False:
		os.mkdir(ICA_h5_dir)
	#Create new file for cleaned dataset
	if os.path.isfile(ica_hf5_file_dir) == True:
		os.remove(ica_hf5_file_dir)
	ica_hf5 = tables.open_file(ica_hf5_file_dir, 'w', title = new_hf5_dir[-1])
	atom = tables.IntAtom()
	ica_hf5.create_earray('/','data_segment',atom,(0,))
	ica_hf5.root.data_segment.append([seg_ind])
	atom = tables.Atom.from_dtype(np.dtype('U20')) #tables.StringAtom(itemsize=50)
	segment_names_array = ica_hf5.create_earray('/','segment_names',atom,(0,))
	ica_hf5.root.segment_names.append(segment_names)
	#segment_names = [segment_names[i].decode('UTF-8') for i in range(len(segment_names))] to get back text
	atom = tables.FloatAtom()
	segment_times_array = ica_hf5.create_earray('/','segment_times',atom,(0,) + np.shape(segment_times))
	segment_times_expanded = np.expand_dims(segment_times[:],0)
	ica_hf5.root.segment_times.append(segment_times_expanded)
	cleaned_data_array = ica_hf5.create_earray('/','cleaned_data',atom,(0,) + np.shape(avg_data))
	cleaned_data_expanded = np.expand_dims(avg_data[:],0)
	ica_hf5.root.cleaned_data.append(cleaned_data_expanded)
	cleaned_data_segment_array = ica_hf5.create_earray('/','cleaned_data_segment',atom,(0,) + np.shape(cleaned_data))
	cleaned_data_segment_expanded = np.expand_dims(cleaned_data[:],0)
	ica_hf5.root.cleaned_data.append(cleaned_data_segment_expanded)
	atom = tables.IntAtom()
	ica_hf5.create_group('/','cleaned_data_peaks')
	for i in range(len(peak_ind)):
		array_name = 'peaks_'+str(i)
		data_peak_array = ica_hf5.create_earray('/cleaned_data_peaks',array_name,atom,(0,) + np.shape(peak_ind[i]))
		data_peaks_expanded = np.expand_dims(peak_ind[i],0)
		exec("ica_hf5.root.cleaned_data_peaks."+array_name+".append(data_peaks_expanded)")
	ica_hf5.close()
	
	return cleaned_data, segment_times

def ICA_analysis(clean_data, clean_data_dir, ica_hf5_file_dir):
	"""This function performs ICA component separation on the electrode recording
	data"""
	
	#Get sampling rate from HF5
	hf5_clean = tables.open_file(clean_data_dir, 'r+', title = clean_data_dir[-1])
	
	sampling_rate = hf5_clean.root.sampling_rate[0]
	num_neur = np.shape(hf5_clean.root.clean_data)[1]
	
	hf5_clean.close()
	
	#print("Signal Whitening for ICA")
	#CANNOT PERFORM CURRENTLY - DATA TOO BIG. ADDED TO FIX QUEUE.
	#data_white = dc.signal_whitening(clean_data)
	w_init_naive = np.random.randn(num_neur,num_neur)
	for i in range(num_neur):
		w_init_naive[i,i] = 1
	w_init_naive = np.divide(w_init_naive,np.sum(w_init_naive,0))
		
#	#Python ICA package version
# 	transformer = FastICA(n_components=None,
# 					   algorithm='deflation', whiten='unit-variance', 
# 					   max_iter=200, tol=0.01, w_init=w_init_naive, 
# 					   random_state=None)
# 	ICA_weights = transformer.fit_transform(clean_data)
	
	#Manual ICA implementation by Hannah Germaine
	n_c = num_neur #Number of components to pull
	iter_n = 500 #Number of iterations for ICA
	alpha = 0.05 #Strength of weight change per trial (must be < 1)
	conv_cutoff = 1e-4 #Cutoff for the cost function to estimate convergence
	ICA_weights, kurt, iter_ns = ica.GS_ICA(clean_data, n_c, iter_n, alpha, conv_cutoff)
	num_neur,num_time = np.shape(clean_data)
	comp_elec_matchings = []
	for i in range(num_neur):
		abs_data = abs(ICA_weights[i])
		ind_max = np.where(abs_data == max(abs_data))[0]
		comp_elec_matchings.extend(ind_max)
		
	print("Saving ICA results")
	ica_hf5 = tables.open_file(ica_hf5_file_dir, 'r+', title = new_hf5_dir[-1])
	atom = tables.FloatAtom()
	ICA_array = ica_hf5.create_earray('/','ica_weights',atom,(0,) + np.shape(ICA_weights))
	ICA_weights_expanded = np.expand_dims(ICA_weights[:],0)
	ica_hf5.root.ica_weights.append(ICA_weights_expanded)
	ica_hf5.create_earray('/','ica_kurtosis',atom,(0,))
	ica_hf5.root.ica_kurtosis.append(kurt)
	atom = tables.IntAtom()
	ICA_sampling_rate = ica_hf5.create_earray('/','sampling_rate',atom,(0,))
	ica_hf5.root.sampling_rate.append([sampling_rate])
	ICA_matchings = ica_hf5.create_earray('/','comp_elec_matchings',atom,(0,))
	ica_hf5.root.comp_elec_matchings.append(comp_elec_matchings)
	ica_hf5.close()
		
	print("Plotting ICA Results")
	plot_ICA_results(clean_data,ICA_weights,comp_elec_matchings,sampling_rate,new_hf5_dir)
	
	return ICA_weights, comp_elec_matchings, sampling_rate

def plot_ICA_results(filtered_data,ICA_weights,comp_elec_matchings,sampling_rate,new_hf5_dir):
	"""This function takes the data segment used for ICA and plots the original 
	segment data as well as the components pulled out by ICA"""
	
	#Create Folder for Image Storage
	split_dir = new_hf5_dir.split('/')
	im_folder_dir = '/'.join(split_dir[:-1]) + '/ICA_images/'
	if os.path.isdir(im_folder_dir) == False:
		os.mkdir(im_folder_dir)
	
	#Grab relevant parameters
	num_neur = len(filtered_data)
	max_ind = 10000
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
		plt.title('Electrode # '+str(i))
	plt.suptitle('Original Data',fontsize=40)
	plt.savefig(im_folder_dir + 'orig_data.png', dpi=100)
	plt.show()
	
	#Plot components - small portion
	plt.figure(figsize=(30,20))
	for i in range(num_neur):
		plt.subplot(num_neur,1,i+1)
		plt.plot(sec_vec,components[i,:])
		plt.title('Electrode Match = '+str(comp_elec_matchings[i]))
	plt.suptitle('Components',fontsize=40)
	plt.savefig(im_folder_dir + 'components.png', dpi=100)
	plt.show()	

def performICA(clean_data_dir):
	"""This function calls all necessary functions to perform ICA on
	electrode data"""
	
	print("Checking for existing ICA data")
	exists, ICA_h5_dir = h5.check_ICA_data(clean_data_dir)
	ica_hf5_name = ICA_h5_dir.split('/')[-3].split('.')[0].split('_')[0] + '_ica.h5'
	ica_hf5_file_dir = ICA_h5_dir + ica_hf5_name
	
	re_do = 0 #Flag for re-cleaning
	
	if exists == 1:
		print("Data already exists.")
		re_do = int(input("If you would like to use the existing data, enter 0. \n If you would like to re-do the ICA protocol, enter 1: "))
	
	if re_do == 1 or exists == 0:
		print("Data does not yet exist. Running through ICA protocol.")
		print('\n Grabbing Clean Data')
		clean_hf5 = tables.open_file(clean_data_dir, 'w', title = clean_data_dir[-1])
		clean_data = clean_hf5.root.clean_data[0]
		clean_hf5.close()
		
		print("Performing Fast ICA \n")
		ICA_weights, comp_elec_matchings, sampling_rate = ICA_analysis(clean_data, clean_data_dir, ica_hf5_file_dir)
		#del e_data, unit_nums	

	return ica_hf5_file_dir

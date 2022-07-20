#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:39:05 2022
@author: hannahgermaine
This set of functions deals with the import and storage of electrophysiology 
data in HDF5 files.
"""

#Imports

import os, tables, tqdm
import numpy as np
import tkinter as tk
import tkinter.filedialog as fd
from scipy.signal import butter, lfilter 
from sklearn.cluster import KMeans
	
#Functions

def file_names():
	"""This function pulls .dat file names"""
	print("Select folder with .dat files from recording")
	#Ask for user input of folder where data is stored
	root = tk.Tk()
	root.withdraw() #use to hide tkinter window
	currdir = os.getcwd()
	datadir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')
	# create button to implement destroy()
	tk.Button(root, text="Quit", command=root.destroy).pack()
	#Import .dat files one-by-one and store as array
	file_list = os.listdir(datadir)
	#Pull data files only
	dat_files_list = [name for name in file_list if name.split('.')[1] == 'dat']
	#Pull electrodes only
	electrodes_list = electrodes(dat_files_list)
	#Pull EMG indices if used
	emg_loop = 1
	while emg_loop == 1:
		emg_used = input("Were EMG used? y / n: ")	
		if emg_used != 'n' and emg_used != 'y':
			print('Error, incorrect response, try again')
			emg_loop = 1
		else:
			emg_loop = 0
	if emg_used == 'y':
		emg_ind = getEMG()
	#Pull tastant delivery inputs
	dig_in_list, dig_in_names = dig_ins(dat_files_list)
	
	return datadir, dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names

def dig_ins(dat_files_list):
	"""This function pulls dig-in information and prompts the user to assign names"""
	dig_ins = [name for name in dat_files_list if name.startswith('board-DIN')]
	if len(dig_ins) > 0:
		dig_in_names = list()
		for i in range(len(dig_ins)):
			dig_in_names.append(input("Enter single-word name for dig-in " + str(i) + ": "))
	return dig_ins, dig_in_names
			
def getEMG():
	"""This function asks for EMG indices"""
	emg_in_loop = 1
	emg_ind = list()
	while emg_in_loop == 1:
		try:
			emg_in = int(input("Enter first input index of EMG: "))
			emg_ind.append(emg_in)
			more_emg_loop = 1
			while more_emg_loop == 1:
				more_emg = input("Are there more EMG inputs? y / n: ")
				if more_emg == 'n':
					emg_in_loop = 0
					more_emg_loop = 0
				elif more_emg == 'y':
					more_emg_loop = 0
				elif more_emg != 'n' and more_emg != 'y':
					print('Incorrect entry. Try again.')
		except:
			print("Incorrect entry, please enter integer.")
	return emg_ind
	
def electrodes(dat_files_list):
	"""This fucntion pulls a list of just electrodes"""
	e_list = [name for name in dat_files_list if name.startswith('amp-A-0')]
	return e_list

def data_to_list(sub_amount,sampling_rate,hf5_dir):
	"""This function pulls data from an HDF5 file and  downsamples before
	storing it to a list"""
	
	"""FIX THIS: Passing units does not mean it can access the hdf5 file.
	Need to pass the directory to the file and then perform operations..."""
	
	print("Opening HDF5 file.")
	hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
	
	#Grab electrode info
	print("Grabbing electrode information.")
	units = hf5.list_nodes('/raw')
	unit_nums = np.array([str(unit).split('_')[-1] for unit in units])
	unit_nums = np.array([int(unit.split(' (')[0]) for unit in unit_nums])
	dig_ins = hf5.list_nodes('/digital_in')
	print('Total Number of Units = ' + str(len(unit_nums)))
	
	if sub_amount > 0:
		new_rate = round(sampling_rate*sub_amount)
		print("New sampling rate = " + str(new_rate))
		time_points = len(units[0])
		num_avg = round(sampling_rate/new_rate)
		#Pull one electrode's worth of data at a time into a list
		#mp.set_start_method('spawn')
		print('Pulling electrode data into array')
		arg_instances = [[hf5_dir, unit, num_avg, time_points] for unit in range(len(unit_nums))]
		hf5.close()
		e_data = [downsample_unit(hf5_dir, unit, num_avg, time_points) for hf5_dir, unit, num_avg, time_points in tqdm.tqdm(arg_instances)]
		arg_instances = [[hf5_dir, dig_in, num_avg, time_points] for dig_in in range(len(dig_ins))]
		dig_in_data = [downsample_dig_ins(hf5_dir,dig_in,num_avg,time_points) for hf5_dir, dig_in, num_avg, time_points in tqdm.tqdm(arg_instances)]
		del time_points, num_avg
	else:
		print('Pulling electrode data into array')
		arg_instances = [[hf5_dir, unit] for unit in range(len(unit_nums))]
		hf5.close()
		e_data = [import_units(hf5_dir,unit) for hf5_dir, unit in tqdm.tqdm(arg_instances)]
		arg_instances = [[hf5_dir, dig_in] for dig_in in range(len(dig_ins))]
		dig_in_data = [import_dig_ins(hf5_dir,dig_in) for hf5_dir, dig_in in tqdm.tqdm(arg_instances)]
	return e_data, dig_in_data

def downsample_unit(hf5_dir,unit,num_avg,time_points):
	"""This function downsamples data that is being imported from a .h5 file"""
	hf5 = tables.open_file(hf5_dir, 'r', title = hf5_dir[-1])
	units = hf5.list_nodes('/raw')
	sub_data = np.zeros(int(time_points/num_avg))
	for i in range(round(num_avg)):
		sub_data = np.add(sub_data,units[unit][i::num_avg])
	hf5.close()
	sub_data = np.divide(sub_data,num_avg)
	return sub_data

def downsample_dig_ins(hf5_dir,dig_in,num_avg,time_points):
	"""This function downsamples data that is being imported from a .h5 file"""
	hf5 = tables.open_file(hf5_dir, 'r', title = hf5_dir[-1])
	dig_ins = hf5.list_nodes('/digital_in')
	dig_in_data = np.zeros(int(time_points/num_avg))
	for i in range(round(num_avg)):
		dig_in_data = np.add(dig_in_data,dig_ins[dig_in][i::num_avg])
	hf5.close()
	dig_in_data = np.divide(dig_in_data,num_avg)
	return dig_in_data

def import_units(hf5_dir,unit):
	"""This function imports units from a .h5 file"""
	hf5 = tables.open_file(hf5_dir, 'r', title = hf5_dir[-1])
	units = hf5.list_nodes('/raw')
	unit_data = np.ndarray.tolist(units[unit][:])
	hf5.close()
	return unit_data

def import_dig_ins(hf5_dir,dig_in):
	"""This function imports dig ins from a .h5 file"""
	hf5 = tables.open_file(hf5_dir, 'r', title = hf5_dir[-1])
	dig_ins = hf5.list_nodes('/digital_in')
	dig_in_data = np.ndarray.tolist(dig_ins[dig_in][:])
	hf5.close()
	return dig_in_data
	
def electrode_data_removal(hf5_dir):
	"""WARNING!!! This function removes any downsampled electrode data arrays
	from the HDF5 file directory/name provided!"""
	#Open file
	print("Opening HDF5 file.")
	hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
	print("Removing electrode array from HDF5 file.")
	#hf5.root.electrode_array.data._f_remove()
	#hf5.root.electrode_array.sampling_rate._f_remove()
	#hf5.root.electrode_array.unit_nums._f_remove()
	hf5.root.electrode_array._f_remove()
	hf5.close()
	print("Removal complete")
	
def get_experiment_components(new_hf5_dir):
	"""This function asks for timings of experiment components, and 
	breaks up the data into chunks accordingly"""
	hf5_new = tables.open_file(new_hf5_dir,'r')
	
	sampling_rate = hf5_new.root.sampling_rate[:]
	
	#Create storage for segmentation
	segments_loop = 1
	num_segments = 0
	while segments_loop == 1:
		try:
			print("Please think about your experiment in terms of segments.")
			print("For example, a pre-tastant delivery interval, a taste delivery interval, etc...")
			num_segments = int(input("How many segments comprise your experiment? Enter an integer. "))
			segments_loop = 0
		except:
			print("Incorrect entry, please enter integer.")
	
	segment_names = []
	for i in range(num_segments):
		segments_loop = 1
		while segments_loop == 1:
			try:
				seg_name = input("What is the name of segment number " + str(i+1) + "? ")
				segment_names.append(seg_name)
				segments_loop = 0
			except:
				print("Please enter a valid name.")
	print("Segment Names: " + str(segment_names)	)		
				
	segment_times = np.zeros((num_segments + 1))
	
	#Grab dig in times to separate out taste delivery interval
	print("Pulling taste delivery interval times from digital inputs.")
	dig_ins = hf5_new.root.dig_ins.dig_ins[0,:,:]
	num_dig_ins = len(dig_ins)
	dig_in_ind_range = np.zeros((2))
	for i in tqdm.tqdm(range(num_dig_ins)):
		dig_in_data = dig_ins[i][:]
		dig_in_delivered = np.where(dig_in_data == 1)
		min_dig_in = min(dig_in_delivered[0])
		max_dig_in = max(dig_in_delivered[0])
		if dig_in_ind_range[0] == 0:
			dig_in_ind_range[0] = min_dig_in
		elif dig_in_ind_range[0] > min_dig_in:
			dig_in_ind_range[0] = min_dig_in
		if dig_in_ind_range[1] < max_dig_in:
			dig_in_ind_range[1] = max_dig_in
	taste_loop = 1
	while taste_loop == 1:
		try:
			print(str(segment_names))
			taste_ind = int(input("Which segment number is the taste delivery interval? (Zero-indexed) "))
			segment_times[taste_ind] = dig_in_ind_range[0]
			segment_times[taste_ind + 1] = dig_in_ind_range[1]
			taste_loop = 0
		except:
			print("Please enter a valid integer index.")
	
	#Store end time as last index
	segment_times[num_segments] = len(dig_in_data)
	
	#Grab times of other segments
	remaining_indices = np.setdiff1d(np.arange(num_segments),np.array([0,taste_ind,num_segments-1]))
	for i in remaining_indices:
		int_len_loop = 1
		while int_len_loop == 1:
			try:
				int_len = int(input("How long was the " + segment_names[i] + " interval? (in rounded minutes) "))
				int_len_loop = 0
			except:
				print("Error: Value must be an integer. Try again.")
		int_len_samples = sampling_rate*60*int_len #Converted to seconds and then samples / second
		segment_times[i+1] = segment_times[i] + int_len_samples
		
	hf5_new.close()
		
	return segment_names, segment_times
		
def save_downsampled_data(hf5_dir,e_data,sub_amount,sampling_rate,dig_in_data):
	"""This function creates a new .h5 file to store only downsampled data arrays"""
	print("Saving Electrode Data to New HF5")
	atom = tables.IntAtom()
	new_hf5_dir = hf5_dir.split('.h5')[0] + '_downsampled.h5'
	hf5 = tables.open_file(hf5_dir, 'r', title = hf5_dir[-1])
	hf5_new = tables.open_file(new_hf5_dir, 'w', title = new_hf5_dir[-1])
	units = hf5.list_nodes('/raw')
	unit_nums = np.array([str(unit).split('_')[-1] for unit in units])
	unit_nums = np.array([int(unit.split(' (')[0]) for unit in unit_nums])
	hf5_new.create_group('/','electrode_array')
	np_e_data = np.array(e_data)
	data = hf5_new.create_earray('/electrode_array','data',atom,(0,)+np.shape(np_e_data))
	np_e_data = np.expand_dims(np_e_data,0)
	data.append(np_e_data)
	del np_e_data, e_data
	hf5_new.create_earray('/electrode_array','unit_nums',atom,(0,))
	exec("hf5_new.root.electrode_array.unit_nums.append(unit_nums[:])")
	if sub_amount > 0:
		new_rate = [round(sampling_rate*sub_amount)]
		hf5_new.create_earray('/','sampling_rate',atom,(0,))
		hf5_new.root.sampling_rate.append(new_rate)
	else:
		hf5_new.create_earray('/','sampling_rate',atom,(0,))
		hf5_new.root.sampling_rate.append([sampling_rate])
	print("Saving Dig Ins to New HF5")
	np_dig_ins = np.array(dig_in_data)
	hf5_new.create_group('/','dig_ins')
	data_digs = hf5_new.create_earray('/dig_ins','dig_ins',atom,(0,)+np.shape(np_dig_ins))
	np_dig_ins = np.expand_dims(np_dig_ins,0)
	data_digs.append(np_dig_ins)
	del np_dig_ins, data_digs, dig_in_data
       #Close HDF5 file
	hf5.close()
	hf5_new.close()
	print("Getting Experiment Components")
	segment_names, segment_times = get_experiment_components(new_hf5_dir)
	hf5_new = tables.open_file(new_hf5_dir, 'r+', title = new_hf5_dir[-1])
	hf5_new.create_group('/','experiment_components')
	hf5_new.create_earray('/experiment_components','segment_times',atom,(0,))
	exec("hf5_new.root.experiment_components.segment_times.append(segment_times[:])")
	atom = tables.Atom.from_dtype(np.dtype('U20')) #tables.StringAtom(itemsize=50)
	hf5_new.create_earray('/experiment_components','segment_names',atom,(0,))
	exec("hf5_new.root.experiment_components.segment_names.append(np.array(segment_names))")
	e_data = hf5_new.root.electrode_array.data[0,:,:]
	dig_ins = hf5_new.root.dig_ins.dig_ins[0,:,:]
	hf5_new.close()
	del hf5, units, sub_amount
	
	return e_data, unit_nums, dig_ins		


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data_segment, lowcut, highcut, fs, order=5):
	"""Function to bandpass filter. Calls butter_bandpass function.
	Copied from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
	Scipy documentation for bandpass filter
	data_segment = data to be filtered
	lowcut = low frequency
	highcut = high frequency
	fs = sampling rate of data"""
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data_segment)
	return y

def signal_averaging(data):
	"""Function to increase the signal-to-noise ratio by removing common signals
	across electrodes"""
	
	mean_odd = np.mean(data[np.arange(1,len(data),2),:],0)
	mean_even = np.mean(data[np.arange(0,len(data),2),:],0)
	
	overall_mean = np.mean([[mean_odd],[mean_even]],0)
	
	cleaned_data = data - overall_mean
	
	return cleaned_data

def data_clustering(data):
	"""Function to cluster the cleaned signals into groups for further cleaning"""
	print("Performing K-Means clustering of data.")
	cluster_nums = np.arange(2,round(len(data)/5))
	mean_centroid_dist = np.zeros(len(cluster_nums)) #Store average centroid distances
	cluster_inertia = np.zeros(len(cluster_nums))
	
	#Calculate average centroid distances for different numbers of clusters
	for i in cluster_nums:
		kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
		centers = kmeans.cluster_centers_
		center_dists = np.zeros(i-1)
		for j in range(i-1):
			center_dists[j] = np.sqrt((centers[j+1][0] - centers[j][0])**2 + (centers[j+1][1] - centers[j][1])**2)
		mean_centroid_dist[i-2] = np.mean(center_dists)
		cluster_inertia[i-2] = kmeans.inertia_
	
	#Look for the elbow in the data to determine the maximum number of clusters that's realistic
	
	
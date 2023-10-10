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
	dat_files_list = [] 
	for name in file_list:
		try:
			name_val = name.split('.')[1]
			if name_val == 'dat':
				dat_files_list.append(name)
		except:
			print("Skipped non-.dat file")
	#Pull electrodes only
	electrodes_list = electrodes(dat_files_list)
	#Pull EMG indices if used
	emg_loop = 1
	while emg_loop == 1:
		emg_used = input("\n INPUT REQUESTED: Were EMG used? y / n: ")	
		if emg_used != 'n' and emg_used != 'y':
			print('Error, incorrect response, try again')
			emg_loop = 1
		else:
			emg_loop = 0
	if emg_used == 'y':
		emg_ind = getEMG()
	else:
		emg_ind = []
	#Pull tastant delivery inputs
	dig_in_list, dig_in_names = dig_ins(dat_files_list)
	
	return datadir, dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names

def dig_ins(dat_files_list):
	"""This function pulls dig-in information and prompts the user to assign names"""
	dig_ins = [name for name in dat_files_list if name.startswith('board-DI')]
	if len(dig_ins) > 0:
		dig_in_names = list()
		for i in range(len(dig_ins)):
			d_loop = 1
			while d_loop == 1:
				d_name = input("\n INPUT REQUESTED: Enter single-word name for dig-in " + str(i) + ": ")
				if len(d_name) < 2:
					print("Error, entered name seems too short. Please try again.")
				else:
					d_loop = 0
			dig_in_names.append(d_name)
	else:
		dig_ins = [name for name in dat_files_list if name.startswith('digital')]
		if len(dig_ins) > 0:
			dig_in_names = list()
			dig_in_name_loop = 1
			i = 0
			while dig_in_name_loop == 1:
				dig_in_names.append(input("\n INPUT REQUESTED: Enter single-word name for dig-in " + str(i) + ": "))
				i += 1
				d_cont_loop = 1
				while d_cont_loop == 1:
					d_cont = input("\n Are there more dig-ins [y,n]? ")
					if d_cont == 'n':
						dig_in_name_loop = 0
						d_cont_loop = 0
					elif d_cont == 'y':
						d_cont_loop = 0
					elif d_cont != 'y' or d_cont != 'n':
						print("ERROR: incorrect entry, try again.")
						d_cont_loop = 1
		else:
			dig_ins = []
			dig_in_names = []
	return dig_ins, dig_in_names
			
def getEMG():
	"""This function asks for EMG indices"""
	emg_in_loop = 1
	emg_ind = list()
	while emg_in_loop == 1:
		try:
			emg_in = int(input("\n INPUT REQUESTED: Enter input index of EMG: "))
			emg_ind.append(emg_in)
			more_emg_loop = 1
			while more_emg_loop == 1:
				more_emg = input("\n INPUT REQUESTED: Are there more EMG inputs? y / n: ")
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
	e_list = [name for name in dat_files_list if name.startswith('amp-')]
	if len(e_list) == 0:
		#Single amplifier file
		e_list = ['amplifier.dat']
	return e_list

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

def downsample_emg(hf5_dir,unit,num_avg,time_points):
	"""This function downsamples data that is being imported from a .h5 file"""
	hf5 = tables.open_file(hf5_dir, 'r', title = hf5_dir[-1])
	units = hf5.list_nodes('/raw_emg')
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
	dig_in_data = np.divide(dig_in_data,num_avg) #Needs updating to simply be binary
	dig_in_data = np.ceil(dig_in_data).astype('int64')
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
	
def get_experiment_components(sampling_rate, dig_ins=None, dig_in_ind_range=np.zeros((2)), len_rec=0):
	"""This function asks for timings of experiment components, and 
	breaks up the data into chunks accordingly"""
	
	#Create storage for segmentation
	segments_loop = 1
	num_segments = 0
	while segments_loop == 1:
		try:
			print("\n INPUT REQUESTED: Please think about your experiment in terms of segments.")
			print("For example, a pre-tastant delivery interval, a taste delivery interval, etc...")
			num_segments = int(input("How many segments comprise your experiment? Enter an integer: "))
			segments_loop = 0
		except:
			print("Incorrect entry, please enter integer.")
	
	segment_names = []
	for i in range(num_segments):
		segments_loop = 1
		while segments_loop == 1:
			try:
				seg_name = input("\n INPUT REQUESTED: What is the name of segment number " + str(i+1) + "? ")
				segment_names.append(seg_name)
				segments_loop = 0
			except:
				print("Please enter a valid name.")
	print("Segment Names: " + str(segment_names)	)		
				
	segment_times = np.zeros((num_segments + 1))
	
	#Grab dig in times to separate out taste delivery interval
	if dig_in_ind_range[0] > 0:
		print("Using given taste delivery interval values.")
	else:
		print("Pulling taste delivery interval times from digital inputs.")
		num_dig_ins = len(dig_ins)
		dig_in_ind_range = np.zeros((2))
		for i in tqdm.tqdm(range(num_dig_ins)):
			dig_in_data = np.array(dig_ins[i])
			dig_in_delivered = np.where(dig_in_data == 1)
			if len(dig_in_delivered[0] > 0):
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
			taste_ind = int(input("\n INPUT REQUESTED: Which segment number is the taste delivery interval? (Zero-indexed) "))
			segment_times[taste_ind] = dig_in_ind_range[0]
			segment_times[taste_ind + 1] = dig_in_ind_range[1]
			taste_loop = 0
		except:
			print("Please enter a valid integer index.")
	
	#Store end time as last index
	if len_rec > 0:
		segment_times[-1] = len_rec
	else:
		segment_times[-1] = len(dig_ins[0])
	
	#Grab times of other segments
	remaining_indices = np.setdiff1d(np.arange(num_segments),np.array([0,taste_ind,num_segments-1]))
	for i in remaining_indices:
		int_len_loop = 1
		while int_len_loop == 1:
			try:
				int_len = int(input("\n INPUT REQUESTED: How long was the " + segment_names[i] + " interval? (in rounded minutes) \n"))
				int_len_loop = 0
			except:
				print("Error: Value must be an integer. Try again.")
		int_len_samples = sampling_rate*60*int_len #Converted to seconds and then samples / second
		segment_times[i+1] = segment_times[i] + int_len_samples
		
	return segment_names, segment_times

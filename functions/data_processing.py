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
	#This function pulls .dat file names
	print("Select folder with .dat files from recording")
	#Ask for user input of folder where data is stored
	root = tk.Tk()
	root.withdraw() #use to hide tkinter window
	currdir = os.getcwd()
	datadir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')

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
	#This function pulls dig-in information and prompts the user to assign names
	dig_ins = [name for name in dat_files_list if name.startswith('board-DIN')]
	if len(dig_ins) > 0:
		dig_in_names = list()
		for i in range(len(dig_ins)):
			dig_in_names.append(input("Enter single-word name for dig-in " + str(i) + ": "))
	return dig_ins, dig_in_names
			
def getEMG():
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
	#This fucntion pulls a list of just electrodes
	e_list = [name for name in dat_files_list if name.startswith('amp-A-0')]
	return e_list

def hdf5_exists():
	h_loop = 1
	h_exists = 0
	while h_loop == 1:
		h_exists = input("Does an HDF5 file already exist? y / n: ")	
		if h_exists != 'n' and h_exists != 'y':
			print('Error, incorrect response, try again')
			h_loop = 1
		else:
			h_loop = 0
	if h_exists == 'y':
		h_exists = 1
		
	return h_exists

def subsample(s_rate, new_s_rate, data):
	time_points = len(data[0])
	num_avg = round(s_rate/new_s_rate)
	sub_data = list()
	print("Subsampling data.")
	for i in tqdm.tqdm(range(round(time_points/2))):
		avg_vec = list()
		for j in range(len(data)):
			avg_chunk = np.mean(data[j][i*num_avg:(i+1)*num_avg])
			avg_vec.append(avg_chunk)
		sub_data.append(np.array(avg_vec).T.tolist())
	sub_data = np.array(sub_data).T.tolist()
	
	return sub_data
	
	
	
def file_import(datadir, dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names):
	#Based on read_file.py by abuzarmahmood on GitHub - written by Abuzar Mahmood:
	#https://github.com/abuzarmahmood/blech_clust/blob/master/read_file.py
	#Create HDF5 file with data stored
	
	#filedir where all data is stored
	#dat_files_list contains the names of all .dat files
	#electrodes_list contains the names of all amp- ... -.dat files
	#emg_ind contains the index of the electrode that has EMG data
	#dig_in_list contains the names of the digital input files
	#dig_in_names contains the names of the tastants associated with each digital input file
	
	# Grab directory name to create the hdf5 file
	hdf5_name = str(os.path.dirname(datadir)).split('/')
	hf5_dir = datadir + '/' + hdf5_name[-1]+'.h5'
	hf5 = tables.open_file(hf5_dir, 'w', title = hdf5_name[-1])
	hf5.create_group('/', 'raw')
	hf5.create_group('/', 'raw_emg')
	hf5.create_group('/', 'digital_in')
	hf5.create_group('/', 'digital_out')
	hf5.close()
	print('Created nodes in HF5')
	
	# Sort all lists
	dat_files_list.sort()
	electrodes_list.sort()
	dig_in_list.sort()
	emg_ind.sort()
	#DO NOT SORT dig_in_names - they are already sorted!
	
	#Separate electrodes and emg into their own lists
	all_electrodes = list()
	all_emg = list()
	for i in range(len(electrodes_list)):
		try:
			e_ind = emg_ind.index(i)
			all_emg.append(electrodes_list[i])
		except:
			all_electrodes.append(electrodes_list[i])
	
	# Read the amplifier sampling rate from info.rhd - 
	# look at Intan's website for structure of header files
	sampling_rate = np.fromfile(datadir + '/' + 'info.rhd', dtype = np.dtype('float32'))
	sampling_rate = int(sampling_rate[2])
	
	check_str = f'ports used: {electrodes_list} \n sampling rate: {sampling_rate} Hz'\
            f'\n digital inputs on intan board: {dig_in_list} \n ---------- \n \n'
	print(check_str)
	
	hf5 = tables.open_file(hf5_dir,'r+')
	
	# Create arrays for all components
	atom = tables.IntAtom()
	for i in all_electrodes: #add electrode arrays
		e_name = i.split('.')[0]
		e_name = e_name.split('-')[-1]
		hf5.create_earray('/raw',f'electrode_{e_name}',atom,(0,))
	for i in all_emg: #add emg arrays
		e_name = i.split('.')[0]
		e_name = e_name.split('-')[-1]
		hf5.create_earray('/raw_emg',f'electrode_{e_name}',atom,(0,))	
	for i in range(len(dig_in_list)): #add dig-in arrays
		d_name = dig_in_names[i]
		hf5.create_earray('/digital_in',f'digin_{d_name}',atom,(0,))	
		
	print('Reading electrodes')
	for i in tqdm.tqdm(all_electrodes):
		inputs = np.fromfile(datadir + '/' + i, dtype = np.dtype('uint16'))
		e_name = i.split('.')[0]
		e_name = e_name.split('-')[-1]
		exec("hf5.root.raw.electrode_"+str(e_name)+".append(inputs[:])")
	
	print('Reading emg')
	for i in tqdm.tqdm(all_emg):
		inputs = np.fromfile(datadir + '/' + i, dtype = np.dtype('uint16'))
		e_name = i.split('.')[0]
		e_name = e_name.split('-')[-1]
		exec("hf5.root.raw_emg.electrode_"+str(e_name)+".append(inputs[:])")
	
	print('Reading dig-ins')
	for i in tqdm.tqdm(range(len(dig_in_list))):
		d_name = dig_in_names[i]
		inputs = np.fromfile(datadir + '/' + dig_in_list[i], dtype = np.dtype('uint16'))
		exec("hf5.root.digital_in.digin_"+str(d_name)+".append(inputs[:])")
	
	hf5.close() #Close the file
	
	return hf5_dir
	
	
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:50:15 2022

@author: hannahgermaine
"""

#This file calls functions from data_processing.py to analyze data

import sys, os
sys.path.append('/Users/hannahgermaine/Documents/GitHub/BlechCodes/')
import functions.data_processing as dp
import functions.intan_reader as ir
import functions.ICA_data_split as ica
import tkinter as tk
import tkinter.filedialog as fd


#Ask if data has already been stored as hdf5
h_exists = dp.hdf5_exists()

if h_exists == 0:
	#Pull data names
	print("Pulling data file information.")
	datadir, dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names = dp.file_names(h_exists)

	# Import data by type
	print("Pulling data into HDF5 file format.")
	hf5_dir = dp.file_import(datadir, dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names)

	del h_exists
else:
	#Get the directory of the hdf5 file
	print("Select directory with .h5 file.")
	root = tk.Tk()
	root.withdraw() #use to hide tkinter window
	currdir = os.getcwd()
	datadir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')
	hdf5_name = str(os.path.dirname(datadir)).split('/')
	hf5_dir = datadir + '/' + hdf5_name[-1]+'.h5'
	del hdf5_name, datadir, currdir, root, h_exists
	
#%% Perform ICA on electrode data to separate out spikes and other components
print("Performing ICA")
e_data, unit_nums = ica.electrode_data_import(hf5_dir)
#components, component_names = ica.performICA(hf5_dir)

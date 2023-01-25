#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:50:15 2022

@author: hannahgermaine
"""

#This file calls functions from data_processing.py to analyze data

import os
file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
os.chdir(file_path)
import functions.data_processing as dp
import functions.spike_sort as sort
import functions.data_cleaning as dc
import tkinter as tk
import tkinter.filedialog as fd
import functions.hdf5_handling as h5
import functions.postsort as ps


#Ask if data has already been stored as hdf5
cont_prompt = 'y'
h_exists = h5.hdf5_exists()

if h_exists == 0:
	#Pull data names
	print("Pulling data file information.")
	datadir, dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names = dp.file_names()

	# Import data by type
	print("Pulling data into HDF5 file format.")
	hf5_dir = h5.file_import(datadir, dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names)

	del h_exists
	print("\n NOTICE: Checkpoint 1 Complete: You can quit the program here, if you like, and come back another time.")
	cont_loop = 1
	if cont_loop == 1:
		cont_prompt = input("INPUT REQUESTED: Would you like to continue [y/n]? ")
		if cont_prompt != 'y' and cont_prompt != 'n':
			print("Error. Incorrect input.")
		else:
			cont_loop = 0
else:
	#Get the directory of the hdf5 file
	print("\n INPUT REQUESTED: Select directory with .h5 file.")
	root = tk.Tk()
	currdir = os.getcwd()
	datadir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')
	hdf5_name = str(os.path.dirname(datadir + '/')).split('/')
	hf5_dir = datadir + '/' + hdf5_name[-1]+'.h5'
	# create button to implement destroy()
	#tk.Button(root, text="Quit", command=root.destroy).pack()
	#root.mainloop() #The Tkinter window will now show a button "Quit" to close the window.
	#^ This only works when run from terminal. In Spyder a window will stay open - don't close it, just press Quit!!
	del hdf5_name, currdir, root, h_exists

if cont_prompt == 'y':
	cont_prompt_2 = 'y'
	# Clean the dataset
	data_dir, cont_prompt_2 = dc.data_cleanup(hf5_dir)		
	if cont_prompt_2 == 'y':
		# Spike sort ICA data
		sorted_dir = sort.run_spike_sort(data_dir) #Runs on regular cleaned data, not ICA data
		cont_loop = 1
		while cont_loop == 1:
			cont_prompt_3 = input("\n INPUT REQUESTED: Would you like to continue to post-sorting collision tests / repacking [y/n]? ")
			if cont_prompt_3 != 'n' and cont_prompt_3 != 'y':
				print("Error, incorrect selection. Try again.")
			elif cont_prompt_3 == 'n':
				print("Come back and finish at your leisure!")
				cont_loop = 0
			elif cont_prompt_3 == 'y':
				print("Now beginning post-sorting.")
				cont_loop = 0
				ps.run_postsort(datadir)
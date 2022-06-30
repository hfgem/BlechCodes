#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:39:05 2022

@author: hannahgermaine
"""

#%% Imports
import numpy as np
import os
import tkinter as tk
import tkinter.filedialog as fd
import pandas as pd
import matplotlib.pyplot as plt
	
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
	
	return dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names

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
	
def file_import(file_names):
	#Import all files in the file_names list and store to the same array
	#Based on dataset.py by nubs01 on GitHub - written by Roshan Nanu:
	#https://github.com/nubs01/blechpy/blob/master/blechpy/datastructures/dataset.py
	storage = np.array()
	for f in file_names:
		"do something"
	
	
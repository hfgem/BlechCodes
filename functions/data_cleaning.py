#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:58:29 2022
Data manipulation with bandpass filtering, etc... to pull out cleaner spikes
@author: hannahgermaine
"""
import numpy as np
from scipy.signal import butter, lfilter
import os, tables, tqdm
import functions.hdf5_handling as h5

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
	
	total_time = len(data[0])
	chunk = int(np.ceil(total_time/10000))
	start_times = np.arange(stop = total_time,step = chunk)
	cleaned_data = np.zeros(np.shape(data))
	for t in tqdm.tqdm(range(len(start_times))):
		s_t = start_times[t]
		data_chunk = data[:,s_t:s_t+chunk]
		#Median Subtraction
		#med = np.median(data_chunk,0)
		#cleaned_chunk = data_chunk - med
		#Mean Subtraction
		mean = np.mean(data_chunk,0)
		cleaned_chunk = data_chunk - mean
		cleaned_data[:,s_t:s_t+chunk] = cleaned_chunk
	
	return cleaned_data

def signal_whitening(data):
	"""Function to whiten a dataset for ICA pre-processing"""
	V,D,R = np.linalg.svd(data)
	data_white = np.dot(V, R)
	
	return data_white

def data_to_mv(data):
	"""Data is originally in microVolts, so here we convert to milliVolts"""
	mv_data = data*10**-3
	return mv_data
	
def data_cleanup(hf5_dir):
	"""This function cleans a dataset using downsampling, bandpass filtering, 
	and signal averaging"""

	cont_prompt_2 = 'y' #Continuation prompt initialization
	
	#Save directory
	folder_dir = hf5_dir.split('/')[:-1]
	folder_dir = '/'.join(folder_dir) + '/'
	clean_data_dir = hf5_dir.split('.h5')[0] + '_cleaned.h5'
	
	#First check for cleaned data
	re_clean = 'n'
	clean_exists = 0
	if os.path.isfile(clean_data_dir) == True:
		print("Cleaned Data Already Exists.")
		clean_exists = 1
		re_clean = input("\n INPUT REQUESTED: Would you like to re-clean the dataset [y/n]? ")
		if re_clean == 'y':
			print("Beginning re-cleaning dataset.")
		print("\n")
	#Clean or re-clean as per case
	if clean_exists == 0 or re_clean == 'y':
		
		#First perform downsampling / import downsampled data
		print("Downsampled Data Import Phase")
		e_data, unit_nums, dig_ins, segment_names, segment_times, new_hf5_dir = h5.downsampled_electrode_data_import(hf5_dir)
		#Currently not using dig_ins and unit_nums, so removing
		del unit_nums, dig_ins
		
		#Open HF5
		hf5_new = tables.open_file(new_hf5_dir, 'r+', title = new_hf5_dir[-1])
		
		sampling_rate = hf5_new.root.sampling_rate[0]
		
		hf5_new.close()
		
		#Begin storing data
		#Create new file for cleaned dataset
		if os.path.isfile(clean_data_dir) == True:
			os.remove(clean_data_dir)
		clean_hf5 = tables.open_file(clean_data_dir, 'w', title = clean_data_dir[-1])
		atom = tables.FloatAtom()
		sr_array = clean_hf5.create_earray('/','sampling_rate',atom,(0,))
		sr_array.append([sampling_rate])
		atom = tables.IntAtom()
		st_array = clean_hf5.create_earray('/','segment_times',atom,(0,))
		st_array.append(segment_times)
		atom = tables.Atom.from_dtype(np.dtype('U20')) #tables.StringAtom(itemsize=50)
		sn_array = clean_hf5.create_earray('/','segment_names',atom,(0,))
		sn_array.append(segment_names)
		clean_hf5.close()
		
		print("Converting Data to mV Scale")
		mv_data = data_to_mv(e_data)
		
		del e_data
		
		print("Band Pass Filtering Data for Spike Detection")
		low_fq = 300
		high_fq = 3000
		filtered_data = bandpass_filter(mv_data, low_fq, high_fq,
										sampling_rate, order=5)
		
		print("Low Pass Filtering Data for LFPs")
		low_fq = 0
		high_fq = 300
		LFP_filtered_data = bandpass_filter(mv_data, low_fq, high_fq,
										sampling_rate, order=5)
		
		del mv_data
			
		print("Signal Averaging to Improve Signal-Noise Ratio")
		avg_data = signal_averaging(filtered_data)
		lfp_avg_data = signal_averaging(LFP_filtered_data)
		
		del filtered_data, LFP_filtered_data
		
		print("Saving cleaned data.")
		clean_hf5 = tables.open_file(clean_data_dir, 'r+', title = clean_data_dir[-1])
		atom = tables.FloatAtom()
		clean_hf5.create_earray('/','clean_data',atom,(0,) + np.shape(avg_data))
		avg_data_expanded = np.expand_dims(avg_data[:],0)
		clean_hf5.root.clean_data.append(avg_data_expanded)
		clean_hf5.create_earray('/','lfp_data',atom,(0,) + np.shape(lfp_avg_data))
		lfp_avg_data_expanded = np.expand_dims(lfp_avg_data[:],0)
		clean_hf5.root.clean_data.append(lfp_avg_data_expanded)
		clean_hf5.close()
		
		print("\n NOTICE: Checkpoint 2 Complete: You can quit the program here, if you like, and come back another time.")
		cont_loop = 1
		if cont_loop == 1:
			cont_prompt_2 = input("\n INPUT REQUESTED: Would you like to continue [y/n]? ")
			if cont_prompt_2 != 'y' and cont_prompt_2 != 'n':
				print("Error. Incorrect input.")
			else:
				cont_loop = 0
		
	return clean_data_dir, cont_prompt_2


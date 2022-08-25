#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:58:29 2022
Data manipulation with bandpass filtering, etc... to pull out cleaner spikes
@author: hannahgermaine
"""
import numpy as np
from scipy.signal import butter, lfilter, find_peaks 
from sklearn.cluster import KMeans
import tqdm, os, tables
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
	
	mean_odd = np.mean(data[np.arange(1,len(data),2),:],0)
	mean_even = np.mean(data[np.arange(0,len(data),2),:],0)
	
	overall_mean = np.mean([[mean_odd],[mean_even]],0)
	
	cleaned_data = data - overall_mean
	
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
	
	#Save directory
	folder_dir = hf5_dir.split('/')[:-1]
	folder_dir = '/'.join(folder_dir) + '/'
	clean_data_dir = hf5_dir.split('.h5')[0] + '_cleaned.h5'
	
	#First check for cleaned data
	re_clean = 0
	clean_exists = 0
	if os.path.isfile(clean_data_dir) == True:
		print("Cleaned Data Already Exists.")
		clean_exists = 1
		print("If you would like to re-clean the dataset, enter 1.")
		print("If you would like to keep the already-cleaned dataset, enter 0.")
		re_clean = int(input("Clean [1] or Keep [0]? "))
		if re_clean == 0:
			print("Clean Data Directory Saved.")
	
	#Clean or re-clean as per case
	if clean_exists == 0 or re_clean == 1:
		
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
		
		print("Bandpass Filtering Data")
		low_fq = 300
		high_fq = 3000
		filtered_data = bandpass_filter(mv_data, low_fq, high_fq,
										sampling_rate, order=5)
		
		del mv_data
			
		print("Signal Averaging to Improve Signal-Noise Ratio")
		avg_data = signal_averaging(filtered_data)
		
		del filtered_data
		
		print("Saving cleaned data.")
		clean_hf5 = tables.open_file(clean_data_dir, 'r+', title = clean_data_dir[-1])
		atom = tables.FloatAtom()
		data_array = clean_hf5.create_earray('/','clean_data',atom,(0,) + np.shape(avg_data))
		avg_data_expanded = np.expand_dims(avg_data[:],0)
		clean_hf5.root.clean_data.append(avg_data_expanded)
		clean_hf5.close()
		
	return clean_data_dir


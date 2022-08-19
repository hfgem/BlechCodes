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

def average_filtering(data,sampling_rate):
	"""Function to scrub mean-subtracted data to potential spike times for 
	further cleaning. Peaks outside 2 absolute deviations and 1 ms to the left,
	and 1.5 ms to the right around them are kept, while the rest are scrubbed. 
	Peaks must be at least 2 ms apart."""
	print("This is going to take a while!")
	num_neur, num_time = np.shape(data)
	#min_dist_btwn_peaks = np.round(sampling_rate*(2/1000))
	#NOTE: Testing not separating peaks at this point, but doing so in spike sorting.
	#For this test, lines 76 and 78 below were removed, as was the section 80-84.
	num_pts_left = np.round(sampling_rate*(1/1000))
	num_pts_right = np.round(sampling_rate*(1.5/1000))
	total_pts = num_pts_left + num_pts_right
	#Grab mean and std
	std_dev = np.std(data,1)
	print("Scrubbing data within 2 average deviations")
	mean_avg_data = []
	peak_ind = []
	for i in tqdm.tqdm(range(num_neur)):
		data_copy = data[i,:]
		#Start with positive peaks
		positive_peaks_data = find_peaks(data_copy,height=2*std_dev[i])[0]#,
						  #distance=min_dist_btwn_peaks)[0]
		negative_peaks_data = find_peaks(-1*data_copy,height=2*std_dev[i])[0]#,
						  #distance=min_dist_btwn_peaks)[0]
		#Remove any positive peaks that are too close to negative peaks
# 		all_peaks = np.unique(np.concatenate((positive_peaks_data,negative_peaks_data)))
# 		all_peaks_diff = all_peaks[1:-1] - all_peaks[0:-2]
# 		too_close_peaks = np.where(all_peaks_diff < sampling_rate/1000)[0]
# 		too_close_ind = np.unique(np.concatenate((too_close_peaks,too_close_peaks+1)))
# 		positive_peaks_data = np.setdiff1d(positive_peaks_data,too_close_ind)
		peak_indices = list(np.append(positive_peaks_data,negative_peaks_data))
		keep_ind = []
		for j in peak_indices:
			p_i_l = max(j - num_pts_left,0)
			p_i_r = min(j + num_pts_right,num_time)
			points = list(np.arange(p_i_l,p_i_r))
			if len(points) < total_pts:
				missing_len = int(total_pts - len(points))
				list(points).extend([0 for i in range(0,missing_len)])
				del missing_len
			keep_ind.extend(points)
		keep_ind = np.unique(keep_ind)
		diff_ind = np.setdiff1d(np.arange(0,num_time),keep_ind)
		data_copy[diff_ind] = 0
		mean_avg_data.append(data_copy)
		peak_ind.append(peak_indices)
	mean_avg_data = np.array(mean_avg_data)
	
	return mean_avg_data, peak_ind

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
		
		#Begin storing data in parallel to getting storable data
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
		cleaned_data = signal_averaging(filtered_data)
		
		del filtered_data
		
		#Commented out for data comparison
		avg_data = cleaned_data
# 		print("Performing signal filtering.")
# 		avg_data, peak_ind = average_filtering(cleaned_data,sampling_rate)
		
		#Not in use - fails to run on full dataset!
		#print("Performing median average filtering")
		#avg_data, peak_ind = dc.median_average_filtering(cleaned_data,sampling_rate)
		
		print("Saving cleaned data.")
		clean_hf5 = tables.open_file(clean_data_dir, 'r+', title = clean_data_dir[-1])
		atom = tables.FloatAtom()
		data_array = clean_hf5.create_earray('/','clean_data',atom,(0,) + np.shape(avg_data))
		avg_data_expanded = np.expand_dims(avg_data[:],0)
		clean_hf5.root.clean_data.append(avg_data_expanded)
		clean_hf5.close()
		
	return clean_data_dir


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
import tqdm

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

def median_average_filtering(data,sampling_rate):
	"""Function to perform median average filtering (MAD) on data in order to 
	scrub it to potential spike times for further cleaning. Peaks outside 2 
	absolute deviations and 1 ms to the left, and 1.5 ms to the right around  
	them are kept, while the rest are scrubbed. Peaks must be at least 2 ms 
	apart."""
	print("Buckle up, this takes a while.")
	num_neur, num_time = np.shape(data)
	min_dist_btwn_peaks = np.round(sampling_rate*(2/1000))
	num_pts_left = np.round(sampling_rate*(1/1000))
	num_pts_right = np.round(sampling_rate*(1.5/1000))
	total_pts = num_pts_left + num_pts_right
	median_per_electrode = np.median(data,1)
	median_sub_data = np.subtract(data,np.expand_dims(median_per_electrode,1))
	dev_from_med = np.median(median_sub_data,1)
	print("Scrubbing data within 2 median average deviations")
	med_avg_data = []
	peak_ind = []
	for i in tqdm.tqdm(range(num_neur)):
		data_copy = median_sub_data[i,:]
		#Start with positive peaks
		positive_peaks_data = find_peaks(data_copy,height=2*dev_from_med[i],
						  distance=min_dist_btwn_peaks)
		negative_peaks_data = find_peaks(-1*data_copy,height=2*dev_from_med[i],
						  distance=min_dist_btwn_peaks)
		peak_indices = list(np.append(positive_peaks_data[0],negative_peaks_data[0]))
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
		med_avg_data.append(data_copy)
		peak_ind.append(peak_indices)
	med_avg_data = np.array(med_avg_data)
	
	return med_avg_data, peak_ind

def data_to_mv(data):
	"""Data is originally in microVolts, so here we convert to milliVolts"""
	mv_data = data*10**3
	return mv_data
	
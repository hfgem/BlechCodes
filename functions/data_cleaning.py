#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:58:29 2022
Data manipulation with bandpass filtering, etc... to pull out cleaner spikes
@author: hannahgermaine
"""
import numpy as np
from scipy.signal import butter, lfilter 
from sklearn.cluster import KMeans

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
	
	
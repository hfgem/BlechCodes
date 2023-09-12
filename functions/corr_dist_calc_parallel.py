#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:26:35 2023

@author: Hannah Germaine
Deviation and Correlation calculation functions for parallelization
"""

import numpy as np
from scipy.interpolate import interp1d

def correlation_parallelized(inputs):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	"""
	corr_val = []
	
	return corr_val
	
def distance_parallelized(inputs):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	"""
	#Grab inputs
	n_i = inputs[0]
	deliv_rast = inputs[1]
	cp_vals = inputs[2]
	dev_rast = inputs[3]
	fr_bin = inputs[4]
	
	#Grab rasters
	neur_deliv_cp_rast = deliv_rast[n_i,cp_vals[n_i,0]:cp_vals[n_i,1]]
	neur_dev_rast = dev_rast[n_i,:]
	len_deliv = len(neur_deliv_cp_rast)
	len_dev = len(neur_dev_rast)
	#Reshape the shorter raster
	min_len = min(len_deliv,len_dev)
	if min_len > fr_bin + 2:
		max_len = max(len_deliv,len_dev)
		max_ind = np.argmax((len_deliv,len_dev))
		if max_ind == 0:
			neur_dev_rast_interp = interp1d(np.linspace(0,len_deliv-1,len_dev),neur_dev_rast)
			neur_dev_rast = neur_dev_rast_interp(np.arange(len_deliv))
		else:
			neur_deliv_cp_interp = interp1d(np.linspace(0,len_dev-1,len_deliv),neur_deliv_cp_rast)
			neur_deliv_cp = neur_deliv_cp_interp(np.arange(len_dev))
		#Convert rasters to binned spikes
		neur_deliv_cp_binned = np.array([np.sum(neur_deliv_cp_rast[b_i:b_i+fr_bin]) for b_i in range(max_len - fr_bin)])
		neur_dev_rast_binned = np.array([np.sum(neur_dev_rast[b_i:b_i+fr_bin]) for b_i in range(max_len - fr_bin)])
		#Calculate distance
		dist_val = np.sqrt(np.sum(np.square(np.abs(neur_deliv_cp_binned - neur_dev_rast_binned))))
	else:
		dist_val = np.nan
	
	return dist_val
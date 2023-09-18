#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:26:35 2023

@author: Hannah Germaine
Deviation and Correlation calculation functions for parallelization
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import itertools
from multiprocessing import Pool
from numba import jit
import warnings

def deliv_corr_parallelized(inputs):
	"""Parallelizes the distance calculation for deliveries"""
	warnings.filterwarnings('ignore')
	
	#Grab parameters/data
	deliv_i = inputs[0]
	deliv_st = inputs[1]
	deliv_len = inputs[2] #deliv_rast = np.zeros((total_num_neur,deliv_len))
	neuron_keep_indices = inputs[3]
	taste_cp = inputs[4] 
	deliv_adjustment = inputs[5]
	dev_rast_binned = inputs[6]
	fr_bin = inputs[7]
	total_num_neur = len(neuron_keep_indices)
	num_cp = np.shape(taste_cp)[2]
	#Pull delivery raster
	deliv_rast = np.zeros((total_num_neur,deliv_len))
	for n_i in neuron_keep_indices:
		neur_deliv_st = list(np.array(deliv_st[n_i]).astype('int') - deliv_adjustment)
		deliv_rast[n_i,neur_deliv_st] = 1
	end_ind = np.arange(fr_bin,fr_bin+deliv_len)
	end_ind[end_ind > deliv_len] = deliv_len
	deliv_rast_binned = np.zeros(np.shape(deliv_rast))
	for start_ind in range(deliv_len):
		deliv_rast_binned[:,start_ind] = np.sum(deliv_rast[:,start_ind:end_ind[start_ind]],1)
	
	deliv_corr_storage = np.zeros((total_num_neur,num_cp-1))
	#Calculate correlation with each cp segment
	for c_p in range(num_cp-1):
		cp_vals = (taste_cp[deliv_i,neuron_keep_indices,c_p:c_p+2]).astype('int')
		#Calculate by neuron using the parallelized code
		neur_corrs = np.zeros(total_num_neur)
		for n_i in range(total_num_neur):
			neur_deliv_cp_rast_binned = deliv_rast_binned[n_i,cp_vals[n_i,0]:cp_vals[n_i,1]]
			neur_dev_rast_binned = dev_rast_binned[n_i,:]
			neur_corrs[n_i] = correlation_calcs(n_i, neur_deliv_cp_rast_binned, cp_vals, neur_dev_rast_binned, fr_bin)
		deliv_corr_storage[:,c_p] = neur_corrs
	
	return deliv_corr_storage

def deliv_dist_parallelized(inputs):
	"""Parallelizes the distance calculation for deliveries"""
	warnings.filterwarnings('ignore')
	
	#Grab parameters/data
	deliv_i = inputs[0]
	deliv_st = inputs[1]
	deliv_len = inputs[2] #deliv_rast = np.zeros((total_num_neur,deliv_len))
	neuron_keep_indices = inputs[3]
	taste_cp = inputs[4] 
	deliv_adjustment = inputs[5]
	dev_rast_binned = inputs[6]
	fr_bin = inputs[7]
	total_num_neur = len(neuron_keep_indices)
	num_cp = np.shape(taste_cp)[2]
	#Pull delivery raster
	deliv_rast = np.zeros((total_num_neur,deliv_len))
	for n_i in neuron_keep_indices:
		neur_deliv_st = list(np.array(deliv_st[n_i]).astype('int') - deliv_adjustment)
		deliv_rast[n_i,neur_deliv_st] = 1
	end_ind = np.arange(fr_bin,fr_bin+deliv_len)
	end_ind[end_ind > deliv_len] = deliv_len
	deliv_rast_binned = np.zeros(np.shape(deliv_rast))
	for start_ind in range(deliv_len):
		deliv_rast_binned[:,start_ind] = np.sum(deliv_rast[:,start_ind:end_ind[start_ind]],1)
	
	deliv_distance_storage = np.zeros((total_num_neur,num_cp-1))
	#Calculate correlation with each cp segment
	for c_p in range(num_cp-1):
		cp_vals = (taste_cp[deliv_i,neuron_keep_indices,c_p:c_p+2]).astype('int')
		#Calculate by neuron using the parallelized code
		neur_dists = np.zeros(total_num_neur)
		for n_i in range(total_num_neur):
			neur_deliv_cp_rast_binned = deliv_rast_binned[n_i,cp_vals[n_i,0]:cp_vals[n_i,1]]
			neur_dev_rast_binned = dev_rast_binned[n_i,:]
			neur_dists[n_i] = distance_calcs(n_i, neur_deliv_cp_rast_binned, cp_vals, neur_dev_rast_binned, fr_bin)
		deliv_distance_storage[:,c_p] = neur_dists
	
	return deliv_distance_storage

def correlation_calcs(n_i, neur_deliv_cp_rast_binned, cp_vals, neur_dev_rast_binned, fr_bin):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	"""
	warnings.filterwarnings('ignore')
	#Grab rasters
	len_deliv = len(neur_deliv_cp_rast_binned)
	len_dev = len(neur_dev_rast_binned)
	#Reshape the shorter raster
	min_len = min(len_deliv,len_dev)
	if min_len > fr_bin + 2:
		len_vec = [len_deliv,len_dev]
		max_len = max(len_vec)
		y_interp_vals = (np.linspace(0,max_len-1,min_len)).astype('int')
		if max_len == len_vec[0]:
			interp_mat = np.zeros((len_dev,len_deliv))
			x_interp_vals = np.arange(len_dev)
			for x_interp_val, y_interp_val in zip(x_interp_vals,y_interp_vals):
				interp_mat[x_interp_val,y_interp_val] = 1
			neur_dev_rast_interp = neur_dev_rast_binned@interp_mat
			neur_dev_rast_binned = neur_dev_rast_interp
		else:
			interp_mat = np.zeros((len_deliv,len_dev))
			x_interp_vals = np.arange(len_deliv)
			for x_interp_val, y_interp_val in zip(x_interp_vals,y_interp_vals):
				interp_mat[x_interp_val,y_interp_val] = 1
			neur_deliv_cp_interp = neur_deliv_cp_rast_binned@interp_mat
			neur_deliv_cp_rast_binned = neur_deliv_cp_interp
		#Calculate correlation
		corr_val = pearsonr(neur_deliv_cp_rast_binned,neur_dev_rast_binned)[1]
	else:
		corr_val = 0
	
	return corr_val
	
@jit(nopython=True)
def distance_calcs(n_i, neur_deliv_cp_rast_binned, cp_vals, neur_dev_rast_binned, fr_bin):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	"""
	warnings.filterwarnings('ignore')
	
	#Grab rasters
	len_deliv = len(neur_deliv_cp_rast_binned)
	len_dev = len(neur_dev_rast_binned)
	#Reshape the shorter raster
	min_len = min(len_deliv,len_dev)
	if min_len > fr_bin + 2:
		len_vec = [len_deliv,len_dev]
		max_len = max(len_vec)
		y_interp_vals = (np.linspace(0,max_len-1,min_len)).astype('int')
		if max_len == len_vec[0]:
			interp_mat = np.zeros((len_dev,len_deliv))
			x_interp_vals = np.arange(len_dev)
			for x_interp_val, y_interp_val in zip(x_interp_vals,y_interp_vals):
				interp_mat[x_interp_val,y_interp_val] = 1
			neur_dev_rast_interp = neur_dev_rast_binned@interp_mat
			neur_dev_rast_binned = neur_dev_rast_interp
		else:
			interp_mat = np.zeros((len_deliv,len_dev))
			x_interp_vals = np.arange(len_deliv)
			for x_interp_val, y_interp_val in zip(x_interp_vals,y_interp_vals):
				interp_mat[x_interp_val,y_interp_val] = 1
			neur_deliv_cp_interp = neur_deliv_cp_rast_binned@interp_mat
			neur_deliv_cp_rast_binned = neur_deliv_cp_interp
		#Calculate distance
		dist_val = np.sqrt(np.sum((np.abs(neur_deliv_cp_rast_binned - neur_dev_rast_binned))**2))
	else:
		dist_val = np.nan
	
	return dist_val



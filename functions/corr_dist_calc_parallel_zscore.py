#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:26:35 2023

@author: Hannah Germaine
Deviation and Correlation calculation functions for parallelization on z-scored data
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import itertools
from multiprocessing import Pool
from numba import jit
import warnings
import matplotlib.pyplot as plt

def deliv_corr_parallelized(inputs):
	"""Parallelizes the correlation calculation for deliveries"""
	warnings.filterwarnings('ignore')
	
	#Grab parameters/data
	deliv_i = inputs[0]
	deliv_st = inputs[1]
	deliv_len = inputs[2] #deliv_rast = np.zeros((total_num_neur,deliv_len))
	neuron_keep_indices = inputs[3]
	taste_cp = inputs[4] 
	deliv_adjustment = inputs[5]
	dev_rast_zscored = inputs[6]
	fr_bin = inputs[7]
	total_num_neur = len(neuron_keep_indices)
	num_cp = np.shape(taste_cp)[1]
	#Pull delivery raster
	deliv_rast = np.zeros((total_num_neur,deliv_len))
	for n_i in neuron_keep_indices:
		neur_deliv_st = list(np.array(deliv_st[n_i]).astype('int') - deliv_adjustment)
		deliv_rast[n_i,neur_deliv_st] = 1
	start_ind = (np.arange(-int(fr_bin/2),deliv_len-int(fr_bin/2))).astype('int')
	start_ind[start_ind < 0] = 0
	end_ind = (np.arange(int(fr_bin/2),deliv_len+int(fr_bin/2))).astype('int')
	end_ind[end_ind > deliv_len] = deliv_len
	deliv_rast_binned = np.zeros(np.shape(deliv_rast))
	for si in range(deliv_len):
		deliv_rast_binned[:,si] = np.sum(deliv_rast[:,start_ind[si]:end_ind[si]],1)
	#Z-score the delivery by taking the pre-taste interval bins for the mean and std
	deliv_cp = taste_cp[neuron_keep_indices,:]
	pre_deliv = int(deliv_cp[0,0])
	z_mean = np.expand_dims(np.mean(deliv_rast_binned[:pre_deliv],axis=1),1)
	z_std = np.expand_dims(np.std(deliv_rast_binned[:pre_deliv],axis=1),1)
	z_std[z_std == 0] = 1 #Get rid of NaNs
	deliv_rast_zscored = np.divide(np.subtract(deliv_rast_binned,z_mean),z_std)
	
	deliv_corr_storage = np.zeros((total_num_neur,num_cp-1))
	#Calculate correlation with each cp segment
	for c_p in range(num_cp-1):
		cp_vals = (deliv_cp[:,c_p:c_p+2]).astype('int')
		#Calculate by neuron using the parallelized code
		neur_corrs = np.zeros(total_num_neur)
		for n_i in range(total_num_neur):
			neur_deliv_cp_rast_zscored = deliv_rast_zscored[n_i,cp_vals[n_i,0]:cp_vals[n_i,1]]
			neur_dev_rast_zscored = dev_rast_zscored[n_i,:]
			neur_deliv_cp_rast_zscored,neur_dev_rast_zscored = interp_vecs(neur_deliv_cp_rast_zscored,neur_dev_rast_zscored)
			neur_corrs[n_i] = correlation_calcs(n_i, neur_deliv_cp_rast_zscored, neur_dev_rast_zscored)
		deliv_corr_storage[:,c_p] = neur_corrs
		if np.nanmean(neur_corrs) > 0.9:
			print("Deliv " + str(deliv_i) + " highly correlated.")
			plt.figure()
			plt.subplot(1,2,1)
			plt.imshow(deliv_rast_zscored,aspect='auto')
			plt.title('deliv')
			plt.subplot(1,2,2)
			plt.imshow(dev_rast_zscored,aspect='auto')
			plt.title('dev')
			plt.suptitle('Deliv ' + str(deliv_i))
			plt.tight_layout()
	
	return deliv_corr_storage

def interp_vecs(neur_deliv_cp_rast_binned,neur_dev_rast_binned):
	#Grab rasters
	len_deliv = len(neur_deliv_cp_rast_binned)
	len_dev = len(neur_dev_rast_binned)
	#Reshape the shorter raster
	min_len = min(len_deliv,len_dev)
	if min_len > 2:
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
	else:
		neur_deliv_cp_rast_binned = []
		neur_dev_rast_binned = []
	return neur_deliv_cp_rast_binned, neur_dev_rast_binned


def correlation_calcs(n_i, neur_deliv_cp_rast_binned, neur_dev_rast_binned):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	The binned rasters should be the same size.
	"""
	warnings.filterwarnings('ignore')
	#Grab rasters
	len_deliv = len(neur_deliv_cp_rast_binned)
	len_dev = len(neur_dev_rast_binned)
	#Reshape the shorter raster
	min_len = min(len_deliv,len_dev)
	if min_len >  2:
		#Calculate correlation
		corr_val = pearsonr(neur_deliv_cp_rast_binned,neur_dev_rast_binned)[0]
	else:
		corr_val = 0
	
	return corr_val
	

def correlation_vec_calcs(neur_deliv_cp_vec, neur_dev_vec):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	The binned rasters should be the same size.
	"""
	warnings.filterwarnings('ignore')
	#Grab rasters
	len_deliv = len(neur_deliv_cp_vec)
	len_dev = len(neur_dev_vec)
	#Reshape the shorter raster
	min_len = min(len_deliv,len_dev)
	if min_len >  2:
		#Calculate correlation
		corr_val = pearsonr(neur_deliv_cp_vec,neur_dev_vec)[0]
	else:
		corr_val = 0
	
	return corr_val
	

"""DEPRECATED FUNCTION
def deliv_dist_parallelized(inputs):
	#Parallelizes the distance calculation for deliveries
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
"""

"""DEPRECATED FUNCTION
@jit(nopython=True)
def distance_calcs(n_i, neur_deliv_cp_rast_binned, cp_vals, neur_dev_rast_binned, fr_bin):
	#This set of code calculates binary vectors of where fr deviations occur in 
	#the activity compared to a local mean and standard deviation of fr.
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
"""


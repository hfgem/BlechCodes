#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:50:37 2023

@author: Hannah Germaine
"""

import numpy as np
from scipy.stats import pearsonr
import warnings

def deliv_corr_population_parallelized(inputs):
	"""Parallelizes the correlation calculation for deliveries"""
	warnings.filterwarnings('ignore')
	
	#Grab parameters/data
	deliv_i = inputs[0]
	deliv_st = inputs[1]
	deliv_len = inputs[2] #deliv_rast = np.zeros((total_num_neur,deliv_len))
	neuron_keep_indices = inputs[3] #indices of neurons to keep
	taste_cp = inputs[4] 
	deliv_adjustment = inputs[5]
	dev_rast_binned = inputs[6]
	fr_bin = inputs[7]
	total_num_neur = len(neuron_keep_indices)
	num_cp = np.shape(taste_cp)[1]
	#Pull delivery raster
	deliv_rast = np.zeros((total_num_neur,deliv_len))
	for n_i in range(total_num_neur):
		n_i_val = neuron_keep_indices[n_i]
		n_st = deliv_st[n_i_val]
		if len(n_st) >= 1:
			if len(n_st) > 1:
				neur_deliv_st = list(np.array(n_st).astype('int') - deliv_adjustment)
			else:
				neur_deliv_st = int(n_st[0]) - deliv_adjustment
			deliv_rast[n_i,neur_deliv_st] = 1
	end_ind = np.arange(fr_bin,fr_bin+deliv_len)
	end_ind[end_ind > deliv_len] = deliv_len
	deliv_rast_binned = np.zeros(np.shape(deliv_rast))
	for start_ind in range(deliv_len):
		deliv_rast_binned[:,start_ind] = np.sum(deliv_rast[:,start_ind:end_ind[start_ind]],1)
	
	deliv_corr_storage = np.zeros(num_cp-1)
	#Calculate correlation with each cp segment
	for c_p in range(num_cp-1):
		cp_vals = (taste_cp[deliv_i,c_p:c_p+2]).astype('int')
		#Calculate by neuron using the parallelized code
		max_bin_length = 0
		for n_i in range(total_num_neur):
			neur_deliv_cp_rast_binned = deliv_rast_binned[n_i,cp_vals[0]:cp_vals[1]]
			if len(neur_deliv_cp_rast_binned) > max_bin_length:
				max_bin_length = len(neur_deliv_cp_rast_binned)
			neur_dev_rast_binned = dev_rast_binned[n_i,:]
			if len(neur_dev_rast_binned) > max_bin_length:
				max_bin_length = len(neur_dev_rast_binned)
		
		#Create population vectors scaled together
		neur_deliv_cp_rast_pop = np.zeros((total_num_neur,max_bin_length))
		neur_dev_rast_pop = np.zeros((total_num_neur,max_bin_length))
		for n_i in range(total_num_neur):
			neur_deliv_cp_rast_binned = deliv_rast_binned[n_i,cp_vals[0]:cp_vals[1]]
			neur_dev_rast_binned = dev_rast_binned[n_i,:]
			neur_deliv_cp_rast_binned, neur_dev_rast_binned = interp_vecs_pop(neur_deliv_cp_rast_binned,neur_dev_rast_binned,max_bin_length)
			neur_deliv_cp_rast_pop[n_i,:] = neur_deliv_cp_rast_binned
			neur_dev_rast_pop[n_i,:] = neur_dev_rast_binned
		
		#Calculate population correlation
		pop_corr = correlation_calc_pop(neur_deliv_cp_rast_pop, neur_dev_rast_pop)
		deliv_corr_storage[c_p] = pop_corr
	
	return deliv_corr_storage

def deliv_corr_population_vec_parallelized(inputs):
	"""Parallelizes the correlation calculation for deliveries"""
	warnings.filterwarnings('ignore')
	
	#Grab parameters/data
	deliv_i = inputs[0]
	deliv_st = inputs[1]
	deliv_len = inputs[2] #deliv_rast = np.zeros((total_num_neur,deliv_len))
	neuron_keep_indices = inputs[3] #indices, not binary
	taste_cp = inputs[4] 
	deliv_adjustment = inputs[5]
	dev_vec = inputs[6]
	c_p = inputs[7]
	total_num_neur = len(neuron_keep_indices)
	#Pull delivery raster
	deliv_rast = np.zeros((total_num_neur,deliv_len))
	for n_i in range(total_num_neur):
		n_i_val = neuron_keep_indices[n_i]
		n_st = deliv_st[n_i_val]
		if len(n_st) >= 1:
			if len(n_st) > 1:
				neur_deliv_st = list(np.array(n_st).astype('int') - deliv_adjustment)
			else:
				neur_deliv_st = int(n_st[0]) - deliv_adjustment
			deliv_rast[n_i,neur_deliv_st] = 1
	cp_vals = (taste_cp[deliv_i,c_p:c_p+2]).astype('int')
	epoch_len = cp_vals[1] - cp_vals[0]
	#Pull out the delivery cp fr vector
	deliv_vec = np.sum(deliv_rast[:,cp_vals[0]:cp_vals[1]],1)/(epoch_len/1000) #in Hz
	#Calculate population correlation
	pop_vec_corr = correlation_calc_vec(deliv_vec, dev_vec)
	
	return pop_vec_corr

def interp_vecs_pop(neur_deliv_cp_rast_binned,neur_dev_rast_binned,bin_length):
	#Grab rasters
	len_deliv = len(neur_deliv_cp_rast_binned)
	len_dev = len(neur_dev_rast_binned)
	#Reshape both rasters
	#_____deliv rast_____
	if (len_deliv > 2)*(len_deliv < bin_length):
		y_interp_vals = (np.linspace(0,bin_length-1,len_deliv)).astype('int')
		interp_mat = np.zeros((len_deliv,bin_length))
		x_interp_vals = np.arange(len_deliv)
		for x_interp_val, y_interp_val in zip(x_interp_vals,y_interp_vals):
			interp_mat[x_interp_val,y_interp_val] = 1
		neur_deliv_cp_interp = neur_deliv_cp_rast_binned@interp_mat
		neur_deliv_cp_rast_binned = neur_deliv_cp_interp
	elif len_deliv < bin_length:
		neur_deliv_cp_rast_binned = np.zeros(bin_length)
		
	#_____dev rast_____
	if (len_dev > 2)*(len_dev < bin_length):
		y_interp_vals = (np.linspace(0,bin_length-1,len_dev)).astype('int')
		interp_mat = np.zeros((len_dev,bin_length))
		x_interp_vals = np.arange(len_dev)
		for x_interp_val, y_interp_val in zip(x_interp_vals,y_interp_vals):
			interp_mat[x_interp_val,y_interp_val] = 1
		neur_dev_interp = neur_dev_rast_binned@interp_mat
		neur_dev_rast_binned = neur_dev_interp
	elif len_dev < bin_length:
		neur_dev_rast_binned = np.zeros(bin_length)
	
	return neur_deliv_cp_rast_binned, neur_dev_rast_binned

def correlation_calc_pop(neur_deliv_cp_rast_pop, neur_dev_rast_pop):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	The binned rasters should be the same size.
	"""
	warnings.filterwarnings('ignore')
	
	corr_val = pearsonr(neur_deliv_cp_rast_pop.flatten(),neur_dev_rast_pop.flatten())[0]
	
	return corr_val

def correlation_calc_vec(neur_deliv_cp_vec, neur_dev_vec):
	"""
	This set of code calculates binary vectors of where fr deviations occur in 
	the activity compared to a local mean and standard deviation of fr.
	The binned rasters should be the same size.
	"""
	warnings.filterwarnings('ignore')
	
	corr_val = pearsonr(neur_deliv_cp_vec,neur_dev_vec)[0]
	
	return corr_val

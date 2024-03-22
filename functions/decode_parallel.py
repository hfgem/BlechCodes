#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:42:31 2024

@author: hannahgermaine
Parallelized code for decoding taste from segments - to be used with 
decoding_funcs.py
"""
import numpy as np
import warnings

def segment_taste_decode_parallelized(inputs):
	"""Parallelizes the independent decoding calculation for individual 
	segment samples"""
	warnings.filterwarnings('ignore')
	
	#Grab parameters/data
	tb_fr = inputs[0]
	num_tastes = inputs[1]
	num_neur = inputs[2]
	x_vals = inputs[3]
	fit_tastant_neur = inputs[4]
	joint_fit_neur = inputs[5]
	p_taste = inputs[6]
	taste_select_neur = inputs[7]

	#Calculate decoding probability
	neur_decode_prob = np.nan*np.ones((num_tastes,num_neur))
	for t_i in range(num_tastes):
		for n_i in taste_select_neur:
			neur_spike_fr = tb_fr[n_i]
			closest_x = np.argmin(np.abs(x_vals - neur_spike_fr))
			p_fr_taste = fit_tastant_neur[t_i,n_i,closest_x]
			p_fr = joint_fit_neur[n_i,closest_x]
			if (p_fr > 0)*((p_fr_taste*p_taste[t_i])>0):
				neur_decode_prob[t_i,n_i] = (p_fr_taste)/p_fr
			else:
				neur_decode_prob[t_i,n_i] = np.nan
	joint_decode_prob = [np.prod(neur_decode_prob[t_i,~np.isnan(neur_decode_prob[t_i,:])])*p_taste[t_i] for t_i in range(num_tastes)]
	joint_decode_prob_frac = np.expand_dims(np.array(joint_decode_prob)/np.sum(joint_decode_prob),1) #Return vector of probabilities for each taste
	
	return joint_decode_prob_frac

def segment_taste_decode_dependent_parallelized(inputs):
	"""Parallelizes the dependent decoding calculation for individual 
	segment samples"""
	warnings.filterwarnings('ignore')
	
	#Grab parameters/data
	tb_fr_i = np.expand_dims(inputs[0],0)
	num_tastes = inputs[1]
	num_neur = inputs[2]
	x_vals = inputs[3]
	fit_tastant_neur = inputs[4]
	p_fr_gmm = inputs[5] #fit_all_neur
	p_taste = inputs[6]
	taste_select_neur = inputs[7]

	#Calculate decoding probability
	decode_prob = np.nan*np.ones(num_tastes)
	p_fr_taste_vec = np.nan*np.ones(num_tastes)
	for t_i in range(num_tastes):
		p_fr_taste_gmm = fit_tastant_neur[t_i]
		p_fr_taste = np.exp(p_fr_taste_gmm.score(tb_fr_i))
		p_fr_taste_vec[t_i] = p_fr_taste
		#P(taste|fr) = (P(fr|taste)*P(taste))/P(fr)
		decode_prob[t_i] = p_fr_taste*p_taste[t_i]
	p_fr = np.nansum(p_fr_taste_vec)/num_tastes
	if p_fr > 0:
		decode_prob = decode_prob/p_fr
	else:
		#Do nothing
		pass
	
	return decode_prob

def segment_burst_decode_dependent_parallelized(inputs):
	"""Parallelizes the dependent decoding calculation for individual 
	segment samples"""
	warnings.filterwarnings('ignore')
	
	#Grab parameters/data
	burst_fr_i = np.expand_dims(inputs[0],0) #Matrix of num_neur
	num_tastes = inputs[1]
	num_neur = inputs[2]
	x_vals = inputs[3]
	fit_tastant_neur = inputs[4]
	p_fr_gmm = inputs[5] #fit_all_neur
	p_taste = inputs[6]
	taste_select_neur = inputs[7]

	#Calculate decoding probability
	decode_prob = np.nan*np.ones(num_tastes)
	p_fr_taste_vec = np.nan*np.ones(num_tastes)
	for t_i in range(num_tastes):
		p_fr_taste_gmm = fit_tastant_neur[t_i]
		p_fr_taste = np.exp(p_fr_taste_gmm.score(burst_fr_i))
		p_fr_taste_vec[t_i] = p_fr_taste
		#P(taste|fr) = (P(fr|taste)*P(taste))/P(fr)
		decode_prob[t_i] = p_fr_taste*p_taste[t_i]
	p_fr = np.nansum(p_fr_taste_vec)/num_tastes
	if p_fr > 0:
		decode_prob = decode_prob/p_fr
	else:
		#Do nothing
		pass
	
	return decode_prob
	

def loo_taste_select_decode(inputs):
	"""Parallelizes leave-one-out decoding of taste to determine taste selectivity"""
	
	d_i = inputs[0]
	taste_d_i = inputs[1]
	d_i_o = inputs[2]
	t_i_spike_times = inputs[3]
	t_i_dig_in_times = inputs[4]
	num_neur = inputs[5]
	taste_cp_pop = inputs[6]
	pre_taste_dt = inputs[7]
	post_taste_dt = inputs[8]
	num_cp = inputs[9]
	
	neur_hz = np.nan*np.ones((num_neur,num_cp+1))
	for n_i in range(num_neur):
		total_d_i = taste_d_i + d_i #what is the index out of all deliveries
		if total_d_i != d_i_o:
			raster_times = t_i_spike_times[d_i][n_i]
			start_taste_i = t_i_dig_in_times[d_i]
			deliv_cp_pop = taste_cp_pop[d_i,:] - pre_taste_dt
			#Binerize the firing following taste delivery start
			times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
			bin_post_taste = np.zeros(post_taste_dt)
			bin_post_taste[times_post_taste] += 1
			#Grab FR per epoch for the delivery
			for cp_i in range(num_cp):
				#individual neuron changepoints
				start_epoch = int(deliv_cp_pop[cp_i])
				end_epoch = int(deliv_cp_pop[cp_i+1])
				epoch_len = end_epoch - start_epoch
				#all_hz_bst = []
				#for binsize in np.arange(50,epoch_len):
				#	bin_starts = np.arange(start_epoch,end_epoch-binsize).astype('int') #bin the epoch
				#	if len(bin_starts) != 0:
				#for binsize in epoch_len*np.ones(1):
				#		bst_hz = [np.sum(bin_post_taste[bin_starts[b_i]:bin_starts[b_i]+binsize])/(binsize/1000) for b_i in range(len(bin_starts))]
				#		all_hz_bst.extend(bst_hz)
				neur_hz[n_i,cp_i] = np.sum(bin_post_taste[start_epoch:end_epoch])/(epoch_len/1000)
			#Grab overall FR for the delivery
			first_epoch = int(deliv_cp_pop[0])
			last_epoch = int(deliv_cp_pop[-1])
			deliv_len = last_epoch - first_epoch
	# 						all_hz_bst = []
	# 						for binsize in np.arange(50,deliv_len):
	# 							bin_starts = np.arange(first_epoch,last_epoch-binsize).astype('int') #bin the epoch
	# 							if len(bin_starts) != 0:
	# 								bst_hz = [np.sum(bin_post_taste[bin_starts[b_i]:bin_starts[b_i]+binsize])/(binsize/1000) for b_i in range(len(bin_starts))]
	# 								all_hz_bst.extend(bst_hz)
	# 						all_hz_bst = np.array(all_hz_bst)
			neur_hz[n_i,num_cp] = np.sum(bin_post_taste[first_epoch:last_epoch])/(deliv_len/1000)
	
	return neur_hz
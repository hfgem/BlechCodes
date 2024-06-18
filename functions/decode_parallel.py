#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:42:31 2024

@author: hannahgermaine
Parallelized code for decoding taste from segments - to be used with 
decoding_funcs.py
"""
import warnings
import numpy as np

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
	num_categories = inputs[1]
	fit_tastant_neur = inputs[2] #list of trained gmms
	p_category = inputs[3]
	
	#Calculate decoding probability using Gaussian mixture models	
	decode_prob = np.nan*np.ones(num_categories)
	log_likelihood_fr_cat_vec = np.nan*np.ones(num_categories)
	for c_i in range(num_categories):
		p_fr_taste_gmm = fit_tastant_neur[c_i]
		#Calculate log likelihood of given vector
		log_likelihood_fr_cat_vec[c_i] = p_fr_taste_gmm.score(tb_fr_i)	
	p_fr_cat = np.exp(log_likelihood_fr_cat_vec)	/np.nansum(np.exp(log_likelihood_fr_cat_vec))
	if len(np.where(np.isnan(p_fr_cat))[0]) > 0: #Handle nan exception
		log_likelihood_fr_cat_vec_rescale = log_likelihood_fr_cat_vec/np.nansum(log_likelihood_fr_cat_vec) #Relative Approximation Through Log Likelihood Rescaling
		p_fr_cat = np.exp(log_likelihood_fr_cat_vec_rescale)	/np.nansum(np.exp(log_likelihood_fr_cat_vec_rescale))
	#P(fr|taste)*P(taste)
	p_fr_cat_p_cat = p_fr_cat*p_category
	decode_prob = p_fr_cat_p_cat/np.sum(p_fr_cat_p_cat)
	
	return decode_prob

	
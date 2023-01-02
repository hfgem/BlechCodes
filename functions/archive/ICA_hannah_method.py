#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:10:09 2022

@author: hannahgermaine
ICA method trying to separate sources as much as possible...
"""

import numpy as np
from scipy.stats import kurtosis
import math

def GS_ICA(data, n_c, iter_n, alpha, conv_cutoff):
	"""
	ABOUT: This function performs a manual ICA implementation using gradient
	descent to maximize kurtosis of components. The function will run for
	either the number of iterations or until change in the cost value of <= 
	cost_cutoff is reached, whichever comes first. This implementation
	independently approaches each signal's weights using Gram-Schmidt 
	Orthogonalization.
	
	Inputs:
		data = dataset to be broken into independent components of size 
		       [n_s,n_t], where n_s is the number of signals available and n_t
		       the number of timepoints.
	    n_c = number of independent components to pull out of the data.
		iter_n = number of iterations to run descent
		alpha = weight of how much mixing values can change at each iteration
		conv_cutoff = cutoff value of convergence at which to stop iterating
	Outputs:
		signals = matrix of size [n_c,n_t] with all predicted signals based on 
				  data samples.
	    W = matrix of size [n_c, n_s] which contains the demixing matrix
		kurt = average kurtosis of identified signals
		iter_ns = number of iterations per signal to convergence"""
	
	#Grab dimensions and initialize weights
	n_s = np.shape(data)[0]
	W = np.random.randn(n_c,n_s) #Generate a random starting mixing matrix
	W = np.divide(W,np.sum(W**2,1)) #Have all rows lie on the unit circle
	
	#Create storage matrices
	kurt_vals = np.zeros((n_c,iter_n)); #Store kurtosis values in time for each signal
	cost_vals = np.zeros((n_c,iter_n)); #Store cost values for each signal
	W_last = W;
	cost_last = np.ones((n_c));
	iter_ns = np.ones((n_c)); #Store the number of iterations per component
	
	#1. Calculate starting source signal values
	s_i = W_last@data
	#2. Calculate the kurtosis of the starting source signals
	kurt_i = kurtosis(s_i,1)
	kurt_vals[:,0] = kurt_i
	#3. Calculate the cost - recall we want kurtosis to be high for each 
	#   component, and for the components to be unique from each other
	cost_i = 1 - abs(math.tanh(kurt_i))
	cost_vals[:,0] = cost_i
	
	#4. Loop through kurtosis ICA implementation. Perform for each signal 
	#   independently by estimating only its weights per iteration.
	for w_i in range(n_c):
		index_range = range(iter_n-1) + 1
		for i in index_range:
			W_test = W_last
			#5. Test the new cost
			s_i = W_test[w_i,:]@data
			kurt_i = kurtosis(s_i)
			cost_i = 1 - abs(math.tanh(kurt_i))
			#6. To maximize kurtosis, we set its derivative = 0 to get the
			#   following delta_w calculation for updating.
			delta_w = np.sign(kurt_i)*np.mean(data*s_i**3,1)
			W_test[w_i,:] = W_test[w_i,:] - alpha*delta_w.T
			W_test = W_test/np.sqrt(np.sum(W_test**2,1))
			if w_i > 0:
				#To prevent convergence to the same maxima, we must decorelate
				#the outputs. To do so, we subtract the projections of the 
				#previously estimated vectors.
				for j in range(w_i-1):
					W_test[w_i,:] = W_test[w_i,:] - (W_test[w_i,:]@W_test[j,:].T@W_test[j,:])
					W_test[w_i,:] = 	W_test[w_i,:]/np.sqrt(W_test[w_i,:]@W_test[w_i,:].T)
			delta = abs(np.sqrt((W_test[w_i,:] - W_last[w_i,:])**2))
			#7. Store resulting cost, kurtosis, and new matrix
			cost_vals[w_i,i] = cost_i
			cost_last = cost_i
			kurt_vals[w_i,i] = kurt_i
			W_last = W_test
			if delta <= conv_cutoff:
				break
		cost_vals[w_i+1,1] = cost_last
		iter_ns[w_i,0] = i
	
	W = W_last
	signals_pre = W@data
	
	#Final separation of signals
	for iter_clean in range(2):
		order = np.random.randi((n_c,1,n_c))
		for j in range(n_c):
			row_ind = order[j]
			other_rows = np.setdiff1d(range(n_c),row_ind)
			signals_pre[other_rows,:] = signals_pre[other_rows,:] - (signals_pre[other_rows,:]@(signals_pre[row_ind,:].T@signals_pre[row_ind,:])) #Remove other components
			signals_pre = signals_pre/np.sqrt(np.sum(signals_pre**2,1)) #Renormalize
			
	#Store final results
	W = W_last
	kurt = np.mean(kurt_vals[:,iter_n])
		
	return W, kurt, iter_ns

#TO-DO: Add symmetric orthogonalization function


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:12:51 2023

@author: Hannah Germaine
A collection of decoding functions used across analyses.
"""

import numpy as np
import tqdm, os, warnings, itertools, time, random
#file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
#os.chdir(file_path)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate
from multiprocess import Pool
from p_tqdm import p_map
import functions.decode_parallel as dp

def taste_decoding_cp(tastant_spike_times,pop_taste_cp_raster_inds,num_cp, \
					  start_dig_in_times,end_dig_in_times,dig_in_names, \
					  num_neur,pre_taste_dt,post_taste_dt,save_dir):
	"""Use Bayesian theory to decode tastes from activity and determine which 
	neurons are taste selective. The functions uses a "leave-one-out" method of
	decoding, where the single delivery to be decoded is left out of the fit 
	distributions in the probability calculations.
	
	Note: the last taste in the 
	tastant_spike_times, etc... variables is always 'none'.
	
	This function uses the population changepoint times to bin the spike counts for the 
	distribution."""
	warnings.filterwarnings('ignore')
	
	num_tastes = len(tastant_spike_times)
	
	max_num_deliv = 0 #Find the maximum number of deliveries across tastants
	taste_num_deliv = np.zeros(num_tastes).astype('int')
	deliv_taste_index = []
	for t_i in range(num_tastes):
		num_deliv = len(tastant_spike_times[t_i])
		deliv_taste_index.extend(list((t_i*np.ones(num_deliv)).astype('int')))
		taste_num_deliv[t_i] = num_deliv
		if num_deliv > max_num_deliv:
			max_num_deliv = num_deliv
	total_num_deliv = np.sum(taste_num_deliv)
	del t_i, num_deliv
	
	#Perform "Leave-One-Out" decoding: one delivery is left out of the distributions
	#and then "decoded" probabilistically based on the distribution formed by 
	#the other deliveries
		
	taste_select_success_joint = np.zeros((num_neur, num_tastes, max_num_deliv)) #mark with a 1 if successfully decoded
	taste_select_success_epoch = np.zeros((num_cp, num_neur, num_tastes, max_num_deliv)) #mark with a 1 if successfully decoded
	p_taste_epoch = np.zeros((num_neur, num_tastes, max_num_deliv, num_cp)) #by epoch
	p_taste_joint = np.zeros((num_neur, num_tastes, max_num_deliv)) #joint across epochs
	
	#print("Now performing leave-one-out calculations of decoding.")
	for d_i_o in tqdm.tqdm(range(total_num_deliv)): #d_i_o is the left out delivery
	
		#Determine the spike count distributions for each neuron for each taste for each cp
		#print("\tPulling spike count distributions by taste by neuron")
		tastant_delivery = np.nan*np.ones((num_tastes,num_neur,max_num_deliv)) #FR per post_taste_interval
		tastant_epoch_delivery = np.nan*np.ones((num_tastes,num_neur,max_num_deliv,num_cp)) #FR per epoch
		max_hz_cp = 0
		max_hz = 0
		for t_i in range(num_tastes):
			num_deliv = taste_num_deliv[t_i]
			taste_cp_pop = pop_taste_cp_raster_inds[t_i]
			for n_i in range(num_neur):
				for d_i in range(num_deliv): #index for that taste
					total_d_i = np.sum(taste_num_deliv[:t_i]) + d_i #what is the index out of all deliveries
					if total_d_i != d_i_o:
						raster_times = tastant_spike_times[t_i][d_i][n_i]
						start_taste_i = start_dig_in_times[t_i][d_i]
						deliv_cp_pop = taste_cp_pop[d_i,:] - pre_taste_dt
						#Binerize the firing following taste delivery start
						times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
						bin_post_taste = np.zeros(post_taste_dt)
						bin_post_taste[times_post_taste] += 1
						#Grab FR per epoch for the delivery
						deliv_binned_st = []
						for cp_i in range(num_cp):
							#individual neuron changepoints
							start_epoch = int(deliv_cp_pop[cp_i])
							end_epoch = int(deliv_cp_pop[cp_i+1])
							bst_hz = np.sum(bin_post_taste[start_epoch:end_epoch])/((end_epoch - start_epoch)*(1/1000))
							if bst_hz > max_hz_cp:
								max_hz_cp = bst_hz
							deliv_binned_st.extend([bst_hz])
						tastant_epoch_delivery[t_i,n_i,d_i,:] = deliv_binned_st
						#Grab overall FR for the delivery
						first_epoch = int(deliv_cp_pop[0])
						last_epoch = int(deliv_cp_pop[-1])
						bst_hz = np.sum(bin_post_taste[first_epoch:last_epoch])/((last_epoch - first_epoch)*(1/1000))
						if bst_hz > max_hz:
							max_hz = bst_hz
						tastant_delivery[t_i,n_i,d_i] = bst_hz
						del cp_i, start_epoch, end_epoch, bst_hz
		del t_i, num_deliv, n_i, d_i, total_d_i, raster_times, start_taste_i, times_post_taste, bin_post_taste
		
		#_____Calculate Across Post-Taste Interval_____
		hist_bins = np.arange(max_hz+1)
		x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
		d_i, t_i, p_taste_fr_neur, taste_select_success_neur = loo_full_taste_decode(num_tastes, num_neur, x_vals, hist_bins, tastant_delivery,\
								  max_hz, dig_in_names, d_i_o, save_dir, deliv_taste_index,taste_num_deliv)
		p_taste_joint[:,:,d_i] = p_taste_fr_neur
		taste_select_success_joint[:,t_i,d_i] = taste_select_success_neur
		
		#_____Calculate By Epoch_____
		hist_bins_cp = np.arange(max_hz_cp+1)
		x_vals_cp = hist_bins_cp[:-1] + np.diff(hist_bins_cp)/2
		d_i, t_i, p_taste_fr_cp_neur, taste_success_fr_cp_neur = loo_epoch_decode(num_cp,num_tastes,num_neur,tastant_epoch_delivery,\
							 max_hz_cp,x_vals_cp,hist_bins_cp,dig_in_names,d_i_o,save_dir,deliv_taste_index,taste_num_deliv)
		p_taste_epoch[:,:,d_i,:] = p_taste_fr_cp_neur
		taste_select_success_epoch[:,:,t_i,d_i] = taste_success_fr_cp_neur
			
	#Now calculate the probability of successfully decoding as the fraction of deliveries successful
	taste_select_prob_joint = np.sum(taste_select_success_joint,axis=2)/taste_num_deliv
	taste_select_prob_epoch = np.sum(taste_select_success_epoch,axis=3)/taste_num_deliv
	
	return p_taste_joint, p_taste_epoch, taste_select_prob_joint, taste_select_prob_epoch

def loo_full_taste_decode(num_tastes, num_neur, x_vals, hist_bins, tastant_delivery,
						  max_hz, dig_in_names, d_i_o, save_dir, deliv_taste_index,
						  taste_num_deliv):
	#_____Calculate Across Full Taste Response____
	
	#Fit the spike count distributions for each neuron for each taste (use gamma distribution) and plot
	#print("\tFitting spike count distributions by taste by neuron")
	p_fr_posttaste_taste = np.zeros((num_tastes,num_neur,len(x_vals)+1)) #p(fr_posttaste|taste)
	for t_i in range(num_tastes):
		for n_i in range(num_neur):
			full_data = (tastant_delivery[t_i,n_i,:]).flatten()
			full_data = full_data[~np.isnan(full_data)]
			num_points = len(full_data)
			bin_centers = np.linspace(0,max_hz+1,np.max([8,np.ceil(num_points/5).astype('int')]))
			bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
			bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
			fit_data = np.histogram(full_data,density=True,bins=bins_calc)
			new_fit = interpolate.interp1d(bin_centers,fit_data[0],kind='linear')
			filtered_data = new_fit(hist_bins)
			filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
			p_fr_posttaste_taste[t_i,n_i,:] = filtered_data
		del n_i, full_data, num_points, bin_centers, bins_calc, fit_data, new_fit, filtered_data
	
	#Plot the taste distributions against each other
	fig_t, ax_t = plt.subplots(nrows=num_neur,ncols=1,sharex=True,figsize=(5,num_neur))
	for n_i in range(num_neur):
		if n_i == 0:	
			ax_t[n_i].plot((p_fr_posttaste_taste[:,n_i,:]).T,label=dig_in_names)
			ax_t[n_i].legend()
		else:
			ax_t[n_i].plot((p_fr_posttaste_taste[:,n_i,:]).T)
	ax_t[num_neur-1].set_xlabel('Firing Rate (Hz)')
	fig_t.supylabel('Probability')
	plt.suptitle('LOO Delivery ' + str(d_i_o) + ' all epochs')
	fig_t.tight_layout()
	fig_t.savefig(save_dir + 'loo_' + str(d_i_o) + '_all_epochs.png')
	fig_t.savefig(save_dir + 'loo_' + str(d_i_o) + '_all_epochs.svg')
	plt.close(fig_t)
	
	#Fit the joint distribution across tastes
	#print("\tFitting joint distribution by neuron")
	p_fr_posttaste = np.zeros((num_neur,len(x_vals)+1)) #p(fr_posttaste)
	for n_i in range(num_neur):
		all_taste_fr = (tastant_delivery[:,n_i,:]).flatten()
		all_taste_fr = all_taste_fr[~np.isnan(all_taste_fr)]
		bin_centers = np.linspace(0,max_hz+1)
		bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
		bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
		fit_data = np.histogram(all_taste_fr,density=True,bins=bins_calc)
		new_fit = interpolate.interp1d(bin_centers,fit_data[0])
		filtered_data = new_fit(hist_bins)
		filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
		p_fr_posttaste[n_i,:] = filtered_data
	del n_i, all_taste_fr, bin_centers, bins_calc, fit_data, new_fit, filtered_data
	
	#Calculate which taste and delivery d_i_o is:
	t_i = deliv_taste_index[d_i_o]
	if t_i > 0:
		d_i = d_i_o - np.cumsum(taste_num_deliv)[t_i-1]
	else:
		d_i = d_i_o
	
	#Calculate the taste probabilities by neuron by delivery
	#print("\tCalculating probability of successful decoding")
	#For each neuron, for each taste and its delivery, determine the probability of each of the tastes being that delivery
	#p(taste|fr) = [p(fr|taste)xp(taste)]/p(fr)
	loo_taste_num_deliv = np.zeros(np.shape(taste_num_deliv))
	loo_taste_num_deliv[:] = taste_num_deliv[:]
	loo_taste_num_deliv[t_i] -= 1
	p_taste = loo_taste_num_deliv/np.sum(loo_taste_num_deliv)
	p_taste_fr_neur = np.zeros((num_neur,num_tastes))
	taste_select_success_neur = np.zeros(num_neur)
	for n_i in range(num_neur):
		#Calculate the probability of each taste for the posttaste interval
		p_taste_fr = np.zeros((num_tastes))
		fr_loo = tastant_delivery[t_i,n_i,d_i]
		for t_i_2 in range(num_tastes): #compare each taste against the true taste data
			closest_x = np.argmin(np.abs(x_vals - fr_loo))
			p_fr_taste = p_fr_posttaste_taste[t_i_2,n_i,closest_x]
			p_fr = p_fr_posttaste[n_i,closest_x]
			p_taste_fr[t_i_2] = (p_fr_taste*p_taste[t_i_2])/p_fr
		#Since the probability of the taste for each bin is calculated, 
		#now we want the joint probability across bins in this delivery
		#We're going to treat the bins as independent samples for ease
		p_taste_fr_neur[n_i,:] = p_taste_fr
		if t_i == np.argmax(p_taste_fr):
			taste_select_success_neur[n_i] = 1
	del p_taste, n_i, p_taste_fr, fr_loo, t_i_2, closest_x, p_fr_taste, p_fr
	
	return d_i, t_i, p_taste_fr_neur, taste_select_success_neur

def loo_epoch_decode(num_cp,num_tastes,num_neur,tastant_epoch_delivery,
					 max_hz_cp,x_vals_cp,hist_bins_cp,dig_in_names,d_i_o,
					 save_dir,deliv_taste_index,taste_num_deliv):
	
	p_taste_fr_cp_neur = np.zeros((num_neur,num_tastes,num_cp))
	taste_success_fr_cp_neur = np.zeros((num_cp,num_neur))
	
	#Calculate which taste and delivery d_i_o is:
	t_i = deliv_taste_index[d_i_o]
	if t_i > 0:
		d_i = d_i_o - np.cumsum(taste_num_deliv)[t_i-1]
	else:
		d_i = d_i_o
	
	for cp_i in range(num_cp):
		#Fit the firing rate distributions for each neuron for each taste (use gamma distribution) and plot
		#print("\tFitting firing rate distributions by taste by neuron")
		p_fr_taste = np.zeros((num_tastes,num_neur,len(x_vals_cp)+1))
		for t_i in range(num_tastes):
			for n_i in range(num_neur):
				full_data = (tastant_epoch_delivery[t_i,n_i,:,cp_i]).flatten()
				full_data = full_data[~np.isnan(full_data)]
				num_points = len(full_data)
				bin_centers = np.linspace(0,max_hz_cp+1,np.max([8,np.ceil(num_points/5).astype('int')]))
				bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
				bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
				fit_data = np.histogram(full_data,density=True,bins=bins_calc)
				new_fit = interpolate.interp1d(bin_centers,fit_data[0],kind='linear')
				filtered_data = new_fit(hist_bins_cp)
				filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
				p_fr_taste[t_i,n_i,:] = filtered_data
			del n_i, full_data, num_points, bin_centers, bins_calc, fit_data, new_fit, filtered_data
		
		#Plot the taste distributions against each other
		fig_t, ax_t = plt.subplots(nrows=num_neur,ncols=1,sharex=True,figsize=(5,num_neur))
		for n_i in range(num_neur): 
			if n_i == 0:	
				ax_t[n_i].plot((p_fr_taste[:,n_i,:]).T,label=dig_in_names)
				ax_t[n_i].legend()
			else:
				ax_t[n_i].plot((p_fr_taste[:,n_i,:]).T)
		ax_t[num_neur-1].set_xlabel('Firing Rate (Hz)')
		fig_t.supylabel('Probability')
		plt.suptitle('LOO Delivery ' + str(d_i_o) + ' all epochs')
		fig_t.tight_layout()
		fig_t.savefig(save_dir + 'loo_' + str(d_i_o) + '_epoch_' + str(cp_i) + '.png')
		fig_t.savefig(save_dir + 'loo_' + str(d_i_o) + '_epoch_' + str(cp_i) + '.svg')
		plt.close(fig_t)
		
		#Fit the joint distribution across tastes
		#print("\tFitting joint distribution by neuron")
		p_fr = np.zeros((num_neur,len(x_vals_cp)+1))
		for n_i in range(num_neur):
			all_taste_fr = (tastant_epoch_delivery[:,n_i,:,cp_i]).flatten()
			all_taste_fr = all_taste_fr[~np.isnan(all_taste_fr)]
			bin_centers = np.linspace(0,max_hz_cp+1)
			bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
			bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
			fit_data = np.histogram(all_taste_fr,density=True,bins=bins_calc)
			new_fit = interpolate.interp1d(bin_centers,fit_data[0])
			filtered_data = new_fit(hist_bins_cp)
			filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
			p_fr[n_i,:] = filtered_data
		del n_i, all_taste_fr, bin_centers, bins_calc, fit_data, new_fit, filtered_data
		
		#Calculate the taste probabilities by neuron by delivery
		#print("\tCalculating probability of successful decoding")
		#For each neuron, for each taste and its delivery, determine the probability of each of the tastes being that delivery
		#p(taste|fr) = [p(fr|taste)xp(taste)]/p(fr)
		loo_taste_num_deliv = np.zeros(np.shape(taste_num_deliv))
		loo_taste_num_deliv[:] = taste_num_deliv[:]
		loo_taste_num_deliv[t_i] -= 1
		p_taste = loo_taste_num_deliv/np.sum(loo_taste_num_deliv)
		for n_i in range(num_neur):
			#Calculate the probability of each taste for each epoch
			fr = tastant_epoch_delivery[t_i,n_i,d_i,cp_i]
			for t_i_2 in range(num_tastes): #compare each taste against the true taste data
				closest_x = np.argmin(np.abs(x_vals_cp - fr))
				p_taste_fr_cp_neur[n_i,t_i_2,cp_i] = (p_fr_taste[t_i_2,n_i,closest_x]*p_taste[t_i_2])/p_fr[n_i,closest_x]
			#Since the probability of the taste for each bin is calculated, 
			#now we want the joint probability across bins in this delivery
			#We're going to treat the bins as independent samples for ease
			if t_i == np.argmax(p_taste_fr_cp_neur[n_i,:,cp_i]):
				taste_success_fr_cp_neur[cp_i,n_i] = 1
				
	return d_i, t_i, p_taste_fr_cp_neur, taste_success_fr_cp_neur


def taste_fr_dist(num_neur,num_cp,tastant_spike_times,
				  taste_cp_raster_inds,pop_taste_cp_raster_inds,
				  start_dig_in_times, pre_taste_dt, post_taste_dt):
	"""This function calculates spike count distributions for each neuron for
	each taste delivery for each epoch"""
	
	num_tastes = len(tastant_spike_times)
	
	max_num_deliv = 0 #Find the maximum number of deliveries across tastants
	taste_num_deliv = np.zeros(num_tastes).astype('int')
	deliv_taste_index = []
	for t_i in range(num_tastes):
		num_deliv = len(tastant_spike_times[t_i])
		deliv_taste_index.extend(list((t_i*np.ones(num_deliv)).astype('int')))
		taste_num_deliv[t_i] = num_deliv
		if num_deliv > max_num_deliv:
			max_num_deliv = num_deliv
	del t_i, num_deliv
	
	#Determine the spike fr distributions for each neuron for each taste
	#print("\tPulling spike fr distributions by taste by neuron")
	#____Baby decoder block____
# 	full_taste_fr_dist = np.nan*np.ones((num_tastes,num_neur,max_num_deliv)) #Full taste response firing rate distribution
# 	tastant_fr_dist = np.nan*np.ones((num_tastes,num_neur,max_num_deliv,num_cp)) #Individual neuron firing rate distributions by epoch
# 	tastant_fr_dist_pop = np.nan*np.ones((num_tastes,num_neur,max_num_deliv,num_cp)) #Population firing rate distributions by epoch
	#____
	#Toddler decoder block____
	full_taste_fr_dist = dict() #Full taste response firing rate distribution
	tastant_fr_dist = dict() #Individual neuron firing rate distributions by epoch
	tastant_fr_dist_pop = dict() #Population firing rate distributions by epoch
	for t_i in range(num_tastes):
		full_taste_fr_dist[t_i] = dict()
		tastant_fr_dist[t_i] = dict()
		tastant_fr_dist_pop[t_i] = dict()
		for n_i in range(num_neur):
			full_taste_fr_dist[t_i][n_i] = dict()
			tastant_fr_dist[t_i][n_i] = dict()
			tastant_fr_dist_pop[t_i][n_i] = dict()
			for d_i in range(max_num_deliv):
				full_taste_fr_dist[t_i][n_i][d_i] = dict()
				tastant_fr_dist[t_i][n_i][d_i] = dict()
				tastant_fr_dist_pop[t_i][n_i][d_i] = dict()
				for cp_i in range(num_cp):
					tastant_fr_dist[t_i][n_i][d_i][cp_i] = dict()
					tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = dict()
	#____
	max_hz = 0
	max_hz_pop = 0
	max_hz_full = 0
	for t_i in range(num_tastes):
		num_deliv = taste_num_deliv[t_i]
		taste_cp = taste_cp_raster_inds[t_i]
		taste_cp_pop = pop_taste_cp_raster_inds[t_i]
		for n_i in range(num_neur):
			for d_i in range(num_deliv): #index for that taste
				raster_times = tastant_spike_times[t_i][d_i][n_i]
				start_taste_i = start_dig_in_times[t_i][d_i]
				deliv_cp = taste_cp[d_i,n_i,:] - pre_taste_dt
				deliv_cp_pop = taste_cp_pop[d_i,:] - pre_taste_dt
				#Bin the average firing rates following taste delivery start
				times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
				bin_post_taste = np.zeros(post_taste_dt)
				bin_post_taste[times_post_taste] += 1
				deliv_binned_fr = []
				deliv_binned_fr_pop = []
				for cp_i in range(num_cp):
					#individual neuron changepoints
					start_epoch = int(deliv_cp[cp_i])
					end_epoch = int(deliv_cp[cp_i+1])
					#____Baby decoder block____
# 					bst_hz = np.sum(bin_post_taste[start_epoch:end_epoch])/((end_epoch - start_epoch)*(1/1000))
# 					if bst_hz > max_hz:
# 						max_hz = bst_hz
# 					deliv_binned_fr.extend([bst_hz])
					#____
					#____Toddler decoder block____
					#TODO: add variable to change the bin size
					bin_edges = np.arange(start_epoch,end_epoch,100).astype('int') #bin the epoch
					if len(bin_edges) != 0:
						if (bin_edges[-1] != end_epoch)*(end_epoch - bin_edges[-1] > 10):
							bin_edges = np.concatenate((bin_edges,end_epoch*np.ones(1).astype('int')))
						bst_hz = [np.sum(bin_post_taste[bin_edges[b_i]:bin_edges[b_i+1]])/((bin_edges[b_i+1] - bin_edges[b_i])*(1/1000)) for b_i in range(len(bin_edges)-1)]
						tastant_fr_dist[t_i][n_i][d_i][cp_i] = bst_hz
						if np.max(bst_hz) > max_hz:
							max_hz = np.max(bst_hz)
					#____
					#population changepoints
					start_epoch = int(deliv_cp_pop[cp_i])
					end_epoch = int(deliv_cp_pop[cp_i+1])
					#____Baby decoder block____
# 					bst_hz = np.sum(bin_post_taste[start_epoch:end_epoch])/((end_epoch - start_epoch)*(1/1000))
# 					if bst_hz > max_hz_pop:
# 						max_hz_pop = bst_hz
# 					deliv_binned_fr_pop.extend([bst_hz])
					#____
					#____Toddler decoder block____
					#TODO: add variable to change the bin size
					bin_edges = np.arange(start_epoch,end_epoch,100).astype('int') #bin the epoch
					if len(bin_edges) != 0:
						if (bin_edges[-1] != end_epoch)*(end_epoch - bin_edges[-1] > 10):
							bin_edges = np.concatenate((bin_edges,end_epoch*np.ones(1).astype('int')))
						bst_hz = [np.sum(bin_post_taste[bin_edges[b_i]:bin_edges[b_i+1]])/((bin_edges[b_i+1] - bin_edges[b_i])*(1/1000)) for b_i in range(len(bin_edges)-1)]
						tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = bst_hz
						if np.max(bst_hz) > max_hz_pop:
							max_hz_pop = np.max(bst_hz)
					#____
				del cp_i, start_epoch, end_epoch, bst_hz
				#____Baby decoder block____
# 				full_taste_fr_dist[t_i,n_i,d_i] = np.sum(bin_post_taste)/(post_taste_dt*(1/1000))
# 				tastant_fr_dist[t_i,n_i,d_i,:] = deliv_binned_fr
# 				tastant_fr_dist_pop[t_i,n_i,d_i,:] = deliv_binned_fr_pop
				#____
				#____Toddler decoder block____
				bin_edges = np.arange(0,post_taste_dt,100).astype('int') #bin the epoch
				if len(bin_edges) != 0:
					if (bin_edges[-1] != post_taste_dt)*(post_taste_dt - bin_edges[-1] > 10):
						bin_edges = np.concatenate((bin_edges,post_taste_dt*np.ones(1).astype('int')))
					bst_hz = [np.sum(bin_post_taste[bin_edges[b_i]:bin_edges[b_i+1]])/((bin_edges[b_i+1] - bin_edges[b_i])*(1/1000)) for b_i in range(len(bin_edges)-1)]
					full_taste_fr_dist[t_i][n_i][d_i] = bst_hz
					if np.max(bst_hz) > max_hz_full:
						max_hz_full = np.max(bst_hz)
				#___
	del t_i, num_deliv, taste_cp, n_i, d_i, raster_times, start_taste_i, deliv_cp, times_post_taste, bin_post_taste

	return full_taste_fr_dist, tastant_fr_dist, tastant_fr_dist_pop, taste_num_deliv, max_hz, max_hz_pop, max_hz_full
	
def decode_full(full_taste_fr_dist,segment_spike_times,post_taste_dt,
				   skip_dt,dig_in_names,segment_times,segment_names,
				   start_dig_in_times,taste_num_deliv,taste_select,save_dir):
	"""Decode probability of full taste response from sliding bin spiking 
	across segments"""
	#Pull necessary variables
	num_tastes, num_neur, num_deliv = np.shape(full_taste_fr_dist)
	num_segments = len(segment_spike_times)
	max_hz = np.max(full_taste_fr_dist[~np.isnan(full_taste_fr_dist)])
	hist_bins = np.arange(stop=max_hz+1,step=0.25)
	x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
	p_taste = taste_num_deliv/np.sum(taste_num_deliv)
	half_bin = np.floor(post_taste_dt/2).astype('int')
	
	prev_run = np.zeros(2)
	
	#Get neurons to cycle through based on binary taste_select which is num_neur in length
	taste_select_neur = np.where(taste_select == 1)[0]
	
	#Fit gamma distributions to fr of each neuron for each taste
	try:
		fit_tastant_neur = np.load(save_dir + 'fit_tastant_neur_full_taste.npy')
		prev_run[0] = 1
	except:
		fit_tastant_neur = np.zeros((num_tastes,num_neur,len(x_vals)+1))
		for t_i in range(num_tastes):
			for n_i in taste_select_neur:
				#____Baby decoder block____
# 				full_data = (full_taste_fr_dist[t_i,n_i,:]).flatten()
# 				full_data = full_data[~np.isnan(full_data)]
				#____
				#____Toddler decoder block____
				full_data = np.array(list(full_taste_fr_dist[t_i][n_i].values()))
				#____
				num_points = len(full_data)
				bin_centers = np.linspace(0,max_hz+1,np.max([8,np.ceil(num_points/5).astype('int')]))
				bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
				bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
				fit_data = np.histogram(full_data,density=True,bins=bins_calc)
				new_fit = interpolate.interp1d(bin_centers,fit_data[0],kind='linear')
				filtered_data = new_fit(hist_bins)
				filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
				fit_tastant_neur[t_i,n_i,:] = filtered_data
			del n_i, full_data, num_points, bin_centers, bins_calc, fit_data, new_fit, filtered_data
		np.save(save_dir + 'fit_tastant_neur_full_taste.npy',fit_tastant_neur)
	
	#Fit the joint distribution across tastes
	#print("\tFitting joint distribution by neuron")
	try:
		joint_fit_neur = np.load(save_dir + 'joint_fit_neur_full_taste.npy')
		prev_run[1] = 1
	except:
		joint_fit_neur = np.zeros((num_neur,len(x_vals)+1))
		for n_i in taste_select_neur:
			#____Baby decoder block____
# 			all_taste_fr = (full_taste_fr_dist[:,n_i,:]).flatten()
# 			all_taste_fr = all_taste_fr[~np.isnan(all_taste_fr)]
			#____
			#____Toddler decoder block____
			all_taste_fr = np.array([list(full_taste_fr_dist[t_i][n_i].values()) for t_i in range(num_tastes)]).flatten()
			#____
			bin_centers = np.linspace(0,max_hz+1)
			bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
			bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
			fit_data = np.histogram(all_taste_fr,density=True,bins=bins_calc)
			new_fit = interpolate.interp1d(bin_centers,fit_data[0])
			filtered_data = new_fit(hist_bins)
			filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
			joint_fit_neur[n_i,:] = filtered_data
		del n_i, all_taste_fr, bin_centers, bins_calc, fit_data, new_fit, filtered_data
		np.save(save_dir + 'joint_fit_neur_full_taste.npy',joint_fit_neur)
	
	if np.sum(prev_run) < 2:
		#Plot the taste distributions against each other
		fig_t, ax_t = plt.subplots(nrows=num_neur,ncols=1,sharex=True,figsize=(5,num_neur))
		for n_i in taste_select_neur: 
			if n_i == 0:	
				ax_t[n_i].plot((fit_tastant_neur[:,n_i,:]).T,label=np.array(dig_in_names))
				ax_t[n_i].plot((joint_fit_neur[n_i,:]).T,label='Joint Dist')
				ax_t[n_i].legend()
			else:
				ax_t[n_i].plot((fit_tastant_neur[:,n_i,:]).T)
				ax_t[n_i].plot((joint_fit_neur[n_i,:]).T,label='Joint Dist')
				ax_t[n_i].legend()
		ax_t[num_neur-1].set_xlabel('Firing Rate (Hz)')
		fig_t.supylabel('Probability')
		fig_t.tight_layout()
		fig_t.savefig(save_dir + 'full_taste_fr_dist_fits.png')
		fig_t.savefig(save_dir + 'full_taste_fr_dist_fits.svg')
		plt.close(fig_t)
	
	#Segment-by-segment use sliding bin with a skip
	seg_decode_save_dir = save_dir + 'decode_prob_full_taste/'
	if not os.path.isdir(seg_decode_save_dir):
		os.mkdir(seg_decode_save_dir)
	for s_i in range(num_segments):
		try:
			seg_decode_prob = np.load(seg_decode_save_dir + 'segment_' + str(s_i) + '.npy')
			print("Segment " + str(s_i) + " Previously Decoded")
		except:
			print("Decoding Segment " + str(s_i))
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			time_bin_starts = np.arange(seg_start+post_taste_dt,seg_end-post_taste_dt,skip_dt)
			num_times = len(time_bin_starts)
			seg_decode_prob = np.zeros((num_tastes,num_times))
			for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts)):
				neur_decode_prob = np.nan*np.ones((num_tastes,num_neur))
				for t_i in range(num_tastes):
					for n_i in taste_select_neur:
						neur_spike_fr = len(np.where((np.int64(tb-half_bin) <= segment_spike_times[s_i][n_i])*(np.int64(tb)+half_bin >= segment_spike_times[s_i][n_i]))[0])/(2*half_bin*(1/1000))
						closest_x = np.argmin(np.abs(x_vals - neur_spike_fr))
						p_fr_taste = fit_tastant_neur[t_i,n_i,closest_x]
						p_fr = joint_fit_neur[n_i,closest_x]
						if (p_fr > 0)*((p_fr_taste*p_taste[t_i])>0):
							neur_decode_prob[t_i,n_i] = (p_fr_taste)/p_fr
						else:
							neur_decode_prob[t_i,n_i] = np.nan
				joint_decode_prob = [np.prod(neur_decode_prob[t_i,~np.isnan(neur_decode_prob[t_i,:])])*p_taste[t_i] for t_i in range(num_tastes)]
				seg_decode_prob[:,tb_i] = np.array(joint_decode_prob)
			#Save decoding probabilities
			np.save(seg_decode_save_dir + 'segment_' + str(s_i) + '.npy',seg_decode_prob)
			#Create plots
			seg_decode_pref = seg_decode_prob/np.sum(seg_decode_prob,0)
			seg_decode_taste_ind = np.argmax(seg_decode_pref,0)
			seg_decode_taste_bin = np.zeros(np.shape(seg_decode_pref))
			for t_i in range(num_tastes):
				seg_decode_taste_bin[t_i,np.where(seg_decode_taste_ind == t_i)[0]] = 1
			#Line plot
			f1 = plt.figure()
			plt.plot(time_bin_starts/1000/60,seg_decode_pref.T)
			for t_i in range(num_tastes):
				plt.fill_between(time_bin_starts/1000/60,seg_decode_taste_bin[t_i,:],alpha=0.2)
			plt.legend(dig_in_names)
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f1.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '.png')
			f1.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '.svg')
			plt.close(f1)
			#Imshow
			f2 = plt.figure()
			plt.imshow(seg_decode_pref,aspect='auto',interpolation = 'none')
			x_ticks = np.ceil(np.linspace(0,len(time_bin_starts)-1,10)).astype('int')
			x_tick_labels = np.round(time_bin_starts[x_ticks]/1000/60,2)
			plt.xticks(x_ticks,x_tick_labels)
			y_ticks = np.arange(len(dig_in_names))
			plt.yticks(y_ticks,dig_in_names)
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f2.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_im.png')
			f2.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_im.svg')
			plt.close(f2)
			#Fraction of occurrences
			f3 = plt.figure()
			plt.pie(np.sum(seg_decode_taste_bin,1)/np.sum(seg_decode_taste_bin),labels=['water','saccharin','none'],autopct='%1.1f%%')
			plt.title('Segment ' + str(s_i))
			f3.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_pie.png')
			f3.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_pie.svg')
			plt.close(f3)
			#If it's the taste interval, save separately decoding of each taste delivery
			if segment_names[s_i].lower() == 'taste': #Assumes it's always called just "taste"
				for t_i in range(num_tastes): #Do each taste and find if match
					for st_i,st in enumerate(np.array(start_dig_in_times[t_i])):
						#Plot the decoding to [-post_taste_dt,2*post_taste_dt] around delivery
						f4 = plt.figure()
						start_dec_t = max(st - post_taste_dt,seg_start)
						closest_tbs = np.argmin(np.abs(time_bin_starts - start_dec_t))
						end_dec_t = min(st + 2*post_taste_dt,seg_end)
						closest_tbe = np.argmin(np.abs(time_bin_starts - end_dec_t))
						closest_td = np.argmin(np.abs(time_bin_starts - st))
						decode_tbs = np.arange(closest_tbs,closest_tbe)
						decode_t = time_bin_starts[decode_tbs]
						decode_t_labels = decode_t - st #in ms
						decode_snip = seg_decode_pref[:,decode_tbs]
						plt.plot(decode_t_labels,decode_snip.T)
						for t_i_2 in range(num_tastes):
							plt.fill_between(decode_t_labels,decode_snip[t_i_2,:],alpha=0.2)
						plt.axvline(0)
						plt.legend(dig_in_names)
						plt.ylabel('Decoding Fraction')
						plt.xlabel('Time From Delivery (ms)')
						plt.title(dig_in_names[t_i] + ' delivery #' + str(st_i))
						f4.savefig(seg_decode_save_dir + dig_in_names[t_i] + '_' + str(st_i) + '.png')
						f4.savefig(seg_decode_save_dir + dig_in_names[t_i] + '_' + str(st_i) + '.svg')
						plt.close(f4)
			
								
def decode_epochs(tastant_fr_dist,segment_spike_times,post_taste_dt,
				   skip_dt,e_skip_dt,e_len_dt,dig_in_names,segment_times,
				   segment_names,start_dig_in_times,taste_num_deliv,
				   taste_select_epoch,use_full,max_hz,save_dir,neuron_count_thresh):
	"""Decode probability of each epoch in high-probability replay regions
	found using decode_phase_1 based on full taste profile decoding.
	Use parallelized function to speed up."""
	#Pull necessary variables
	#____Baby decoder block____
# 	num_tastes, num_neur, max_num_deliv, num_cp = np.shape(tastant_fr_dist)
	#____
	#____Toddler decoder block____
	num_tastes = len(start_dig_in_times)
	num_neur = len(segment_spike_times[0])
	max_num_deliv = np.max(taste_num_deliv)
	num_cp = len(tastant_fr_dist[0][0][0])
	#____
	num_segments = len(segment_spike_times)
	hist_bins = np.arange(stop=max_hz+1,step=0.25)
	x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
	p_taste = taste_num_deliv/np.sum(taste_num_deliv)
	half_bin = np.floor(e_len_dt/2).astype('int')
	
	for e_i in range(num_cp): #By epoch conduct decoding
		print('Decoding Epoch ' + str(e_i))
	
		prev_run = np.zeros(2)
		
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		
		#Fit gamma distributions to fr of each neuron for each taste
		try:
			fit_tastant_neur = np.load(save_dir + 'fit_tastant_neur_epoch_' + str(e_i) + '.npy')
			prev_run[0] = 1
		except:	
			fit_tastant_neur = np.zeros((num_tastes,num_neur,len(x_vals)+1))
			for t_i in range(num_tastes):
				for n_i in taste_select_neur:
					#____Baby decoder block____
# 					full_data = (tastant_fr_dist[t_i,n_i,:,e_i]).flatten()
# 					full_data = full_data[~np.isnan(full_data)]
					#____
					#____Toddler decoder block____
					full_data = []
					for d_i in range(max_num_deliv):
						full_data.extend(tastant_fr_dist[t_i][n_i][d_i][e_i])
					#____
					num_points = len(full_data)
					bin_centers = np.linspace(0,max_hz+1,np.max([20,np.ceil(num_points/5).astype('int')]))
					bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
					bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
					fit_data = np.histogram(full_data,density=True,bins=bins_calc)
					new_fit = interpolate.interp1d(bin_centers,fit_data[0],kind='linear')
					filtered_data = new_fit(hist_bins)
					filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
					fit_tastant_neur[t_i,n_i,:] = filtered_data
				del n_i, full_data, num_points, bin_centers, bins_calc, fit_data, new_fit, filtered_data
			np.save(save_dir + 'fit_tastant_neur_epoch_' + str(e_i) + '.npy',fit_tastant_neur)
		
		#Fit the joint distribution across tastes
		#print("\tFitting joint distribution by neuron")
		try:
			joint_fit_neur = np.load(save_dir + 'joint_fit_neur_epoch_' + str(e_i) + '.npy')
			prev_run[1] = 1
		except:
			joint_fit_neur = np.zeros((num_neur,len(x_vals)+1))
			for n_i in taste_select_neur:
				#____Baby decoder block____
# 				all_taste_fr = (tastant_fr_dist[:,n_i,:,e_i]).flatten()
# 				all_taste_fr = all_taste_fr[~np.isnan(all_taste_fr)]
				#____
				#____Toddler decoder block____
				all_taste_fr = []
				for t_i in range(num_tastes):
					for d_i in range(max_num_deliv):
						all_taste_fr.extend(tastant_fr_dist[t_i][n_i][d_i][e_i])
				#____
				bin_centers = np.linspace(0,max_hz+1)
				bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
				bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
				fit_data = np.histogram(all_taste_fr,density=True,bins=bins_calc)
				new_fit = interpolate.interp1d(bin_centers,fit_data[0])
				filtered_data = new_fit(hist_bins)
				filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
				joint_fit_neur[n_i,:] = filtered_data
			del n_i, all_taste_fr, bin_centers, bins_calc, fit_data, new_fit, filtered_data
			np.save(save_dir + 'joint_fit_neur_epoch_' + str(e_i) + '.npy',joint_fit_neur)
	
		#Plot the taste distributions against each other
		if np.sum(prev_run) < 2:
			fig_t, ax_t = plt.subplots(nrows=num_neur,ncols=1,sharex=True,figsize=(5,num_neur))
			for n_i in taste_select_neur: 
				if n_i == 0:	
					ax_t[n_i].plot((fit_tastant_neur[:,n_i,:]).T,label=np.array(dig_in_names))
					ax_t[n_i].plot((joint_fit_neur[n_i,:]).T,label='Joint Dist')
					ax_t[n_i].legend()
				else:
					ax_t[n_i].plot((fit_tastant_neur[:,n_i,:]).T)
					ax_t[n_i].plot((joint_fit_neur[n_i,:]).T,label='Joint Dist')
					ax_t[n_i].legend()
			ax_t[num_neur-1].set_xlabel('Firing Rate (Hz)')
			fig_t.supylabel('Probability')
			fig_t.tight_layout()
			fig_t.savefig(save_dir + 'epoch_' + str(e_i) + '_fr_dist_fits.png')
			fig_t.savefig(save_dir + 'epoch_' + str(e_i) + '_fr_dist_fits.svg')
			plt.close(fig_t)
		
		#Segment-by-segment use full taste decoding times to zoom in and test 
		#	epoch-specific and smaller interval
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			os.mkdir(epoch_decode_save_dir)
		for s_i in range(num_segments):
			#Get segment variables
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = segment_times[s_i+1] - segment_times[s_i] #in dt = ms
			if use_full == 1:
				#Load previously calculated probabilities and pull start/end times
				time_bin_starts = np.arange(seg_start+half_bin,seg_end-post_taste_dt,skip_dt)
				seg_decode_prob = np.load(save_dir + 'decode_prob_full_taste/' + 'segment_' + str(s_i) + '.npy')
				seg_decode_pref = seg_decode_prob/np.sum(seg_decode_prob,0)
				seg_decode_taste_ind = np.argmax(seg_decode_pref,0)
				seg_decode_taste_bin = np.zeros(np.shape(seg_decode_pref)[-1])
				for t_i in range(num_tastes - 1): #Assumes last taste is "none"
 					seg_decode_taste_bin[np.where(seg_decode_taste_ind == t_i)[0]] = 1
				start_decode_bins = np.where(np.diff(seg_decode_taste_bin) == 1)[0] + 1
				end_decode_bins = np.where(np.diff(seg_decode_taste_bin) == -1)[0] + 1
				if len(start_decode_bins) < len(end_decode_bins):
 					start_decode_bins = np.concatenate((np.zeros(1),start_decode_bins)).astype('int')
				if len(end_decode_bins) < len(start_decode_bins):
 					end_decode_bins = np.concatenate((end_decode_bins,(len(seg_decode_taste_bin)-1)*np.ones(1))).astype('int')
				#Get bins of full taste decoding back to original segment times
				start_decode_times = time_bin_starts[start_decode_bins]
				end_decode_times = time_bin_starts[end_decode_bins]
				new_time_bins = []
				for sd_i,sd in enumerate(start_decode_times):
 					decode_starts = np.arange(sd,end_decode_times[sd_i],e_skip_dt)
 					new_time_bins.extend(list(decode_starts))
				new_time_bins = np.array(new_time_bins)
			else:
				new_time_bins = np.arange(seg_start+half_bin,seg_end-half_bin,e_skip_dt)
			#Now pull epoch-specific probabilities
			seg_decode_epoch_prob = np.zeros((num_tastes,seg_len))
			seg_decode_epoch_prob[-1,:] = 1 #Start with assumption of "none" taste at all times
			#Binerize Spike Times
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in taste_select_neur:
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			try:
				seg_decode_epoch_prob = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '.npy')
				tb_fr = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '_tb_fr.npy')
				print('\tSegment ' + str(s_i) + ' Previously Decoded')
			except:
				print('\tDecoding Segment ' + str(s_i))
				#Perform parallel computation for each time bin
				print('\t\tCalculate firing rates for time bins')
				try:
					tb_fr = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '_tb_fr.npy')
				except:
					tb_fr = np.zeros((num_neur,len(new_time_bins)))
					for tb_i,tb in enumerate(tqdm.tqdm(new_time_bins)):
						tb_fr[:,tb_i] = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin:tb+half_bin-seg_start],1)/(2*half_bin*(1/1000))
					np.save(epoch_decode_save_dir + 'segment_' + str(s_i) + '_tb_fr.npy',tb_fr)
					del tb_i, tb
				list_tb_fr = list(tb_fr.T)
				del tb_fr
				#Pass inputs to parallel computation on probabilities
				inputs = zip(list_tb_fr, itertools.repeat(num_tastes), \
					 itertools.repeat(num_neur),itertools.repeat(x_vals), \
					 itertools.repeat(fit_tastant_neur), itertools.repeat(joint_fit_neur), \
					 itertools.repeat(p_taste),itertools.repeat(taste_select_neur))
				tic = time.time()
				pool = Pool(4)
				tb_decode_prob = pool.map(dp.segment_taste_decode_parallelized, inputs)
				pool.close()
				toc = time.time()
				print('\t\tTime to decode = ' + str(np.round((toc-tic)/60,2)) + ' (min)')
				tb_decode_array = np.squeeze(np.array(tb_decode_prob)).T
				#___
				for e_dt in range(e_skip_dt): #The whole skip interval should have the same decode probability
					seg_decode_epoch_prob[:,new_time_bins - seg_start + e_dt] = tb_decode_array
				
				#Save decoding probabilities
				np.save(epoch_decode_save_dir + 'segment_' + str(s_i) + '.npy',seg_decode_epoch_prob)
			#Recreate plots regardless of if data was previously saved
			seg_decode_save_dir = epoch_decode_save_dir + 'segment_' + str(s_i) + '/'
			if not os.path.isdir(seg_decode_save_dir):
				os.mkdir(seg_decode_save_dir)
			
			seg_decode_epoch_prob_nonan = np.zeros(np.shape(seg_decode_epoch_prob))
			seg_decode_epoch_prob_nonan[:] = seg_decode_epoch_prob[:]
			seg_decode_epoch_prob_nonan[np.isnan(seg_decode_epoch_prob_nonan)] = 0
			seg_decode_epoch_taste_ind = np.argmax(seg_decode_epoch_prob,0)
			seg_decode_epoch_taste_bin = np.zeros(np.shape(seg_decode_epoch_prob))
			for t_i in range(num_tastes):
				taste_bin = (seg_decode_epoch_taste_ind == t_i).astype('int')
				#To ensure starts and ends of bins align
				taste_bin[0] = 0
				taste_bin[-1] = 0
				#Calculate decoding periods
				diff_decoded_taste = np.diff(taste_bin)
				start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
				end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
				num_decoded = len(start_decoded)
				#Calculate number of neurons in each period
				num_neur_decoded = np.zeros(num_decoded)
				for nd_i in range(num_decoded):
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					for n_i in range(num_neur):
						if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
							num_neur_decoded[nd_i] += 1
				#Now cut at threshold and only keep matching decoded intervals
				decode_ind = np.where(num_neur_decoded > neuron_count_thresh)[0]
				for db in decode_ind:
					s_db = start_decoded[db]
					e_db = end_decoded[db]
					seg_decode_epoch_taste_bin[t_i,s_db:e_db] = 1
			#Line plot
			f1 = plt.figure()
			plt.plot(np.arange(seg_start,seg_end)/1000/60,seg_decode_epoch_prob_nonan.T)
			for t_i in range(num_tastes):
				plt.fill_between(np.arange(seg_start,seg_end)/1000/60,seg_decode_epoch_taste_bin[t_i,:],alpha=0.2)
			plt.legend(dig_in_names,loc='right')
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f1.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '.png')
			f1.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '.svg')
			plt.close(f1)
			#Imshow
			f2 = plt.figure()
			plt.imshow(seg_decode_epoch_prob_nonan,aspect='auto',interpolation = 'none')
			x_ticks = np.ceil(np.linspace(0,len(new_time_bins)-1,10)).astype('int')
			x_tick_labels = np.round(new_time_bins[x_ticks]/1000/60,2)
			plt.xticks(x_ticks,x_tick_labels)
			y_ticks = np.arange(len(dig_in_names))
			plt.yticks(y_ticks,dig_in_names)
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f2.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_im.png')
			f2.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_im.svg')
			plt.close(f2)
			#Fraction of occurrences
			f3 = plt.figure()
			plt.pie(np.sum(seg_decode_epoch_taste_bin,1)/np.sum(seg_decode_epoch_taste_bin),labels=['water','saccharin','none'],autopct='%1.1f%%',pctdistance =1.5)
			plt.title('Segment ' + str(s_i))
			f3.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_pie.png')
			f3.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_pie.svg')
			plt.close(f3)
			#If it's the taste interval, save separately decoding of each taste delivery
			if segment_names[s_i].lower() == 'taste': #Assumes it's always called just "taste"
				taste_save_dir = seg_decode_save_dir + 'taste_decode/'
				if not os.path.isdir(taste_save_dir):
					os.mkdir(taste_save_dir)
				for t_i in range(num_tastes): #Do each taste and find if match
					for st_i,st in enumerate(np.array(start_dig_in_times[t_i])):
						#Plot the decoding to [-post_taste_dt,2*post_taste_dt] around delivery
						f4 = plt.figure()
						start_dec_t = max(st - post_taste_dt,seg_start)
						closest_tbs = np.argmin(np.abs(new_time_bins - start_dec_t))
						end_dec_t = min(st + 2*post_taste_dt,seg_end)
						closest_tbe = np.argmin(np.abs(new_time_bins - end_dec_t))
						closest_td = np.argmin(np.abs(new_time_bins - st))
						decode_tbs = np.arange(closest_tbs,closest_tbe)
						decode_t = new_time_bins[decode_tbs]
						decode_t_labels = decode_t - st #in ms
						decode_snip = seg_decode_epoch_prob[:,decode_tbs]
						decode_prob_snip = seg_decode_epoch_taste_bin[:,decode_tbs]
						#TODO: Only plot filled background when decoder is above percentage threshold
						plt.plot(decode_t_labels,decode_snip.T)
						for t_i_2 in range(num_tastes):
							plt.fill_between(decode_t_labels,decode_prob_snip[t_i_2,:],alpha=0.2)
						plt.axvline(0)
						plt.legend(dig_in_names)
						plt.ylabel('Decoding Fraction')
						plt.xlabel('Time From Delivery (ms)')
						plt.title(dig_in_names[t_i] + ' delivery #' + str(st_i))
						f4.savefig(taste_save_dir + dig_in_names[t_i] + '_' + str(st_i) + '.png')
						f4.savefig(taste_save_dir + dig_in_names[t_i] + '_' + str(st_i) + '.svg')
						plt.close(f4)


def taste_fr_dist_zscore(num_neur,num_cp,tastant_spike_times,segment_spike_times,
				  segment_names,segment_times,taste_cp_raster_inds,pop_taste_cp_raster_inds,
				  start_dig_in_times,pre_taste_dt,post_taste_dt,bin_dt):
	"""This function calculates spike count distributions for each neuron for
	each taste delivery for each epoch"""
	
	num_tastes = len(tastant_spike_times)
	half_bin = np.floor(bin_dt/2).astype('int')
	
	max_num_deliv = 0 #Find the maximum number of deliveries across tastants
	taste_num_deliv = np.zeros(num_tastes).astype('int')
	deliv_taste_index = []
	for t_i in range(num_tastes):
		num_deliv = len(tastant_spike_times[t_i])
		deliv_taste_index.extend(list((t_i*np.ones(num_deliv)).astype('int')))
		taste_num_deliv[t_i] = num_deliv
		if num_deliv > max_num_deliv:
			max_num_deliv = num_deliv
	del t_i, num_deliv
	
	s_i_taste = np.nan*np.ones(1)
	for s_i in range(len(segment_names)):
		if segment_names[s_i].lower() == 'taste':
			s_i_taste[0] = s_i
	
	if not np.isnan(s_i_taste[0]):
		s_i = int(s_i_taste[0])
		seg_start = segment_times[s_i]
		seg_end = segment_times[s_i+1]
		seg_len = seg_end - seg_start
		time_bin_starts = np.arange(seg_start+half_bin,seg_end-half_bin,bin_dt)
		segment_spike_times_s_i = segment_spike_times[s_i]
		segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
		for n_i in range(num_neur):
			n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
			segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
		tb_fr = np.zeros((num_neur,len(time_bin_starts)))
		for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts)):
			tb_fr[:,tb_i] = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin:tb+half_bin-seg_start],1)/(2*half_bin*(1/1000))
		mean_fr = np.mean(tb_fr,1)
		std_fr = np.std(tb_fr,1)
	else:
		mean_fr = np.zeros(num_neur)
		std_fr = np.zeros(num_neur)
		
	#Determine the spike fr distributions for each neuron for each taste
	#print("\tPulling spike fr distributions by taste by neuron")
	#____Baby decoder block____
# 	full_taste_fr_dist = np.nan*np.ones((num_tastes,num_neur,max_num_deliv)) #Full taste response firing rate distribution
# 	tastant_fr_dist = np.nan*np.ones((num_tastes,num_neur,max_num_deliv,num_cp)) #Individual neuron firing rate distributions by epoch
# 	tastant_fr_dist_pop = np.nan*np.ones((num_tastes,num_neur,max_num_deliv,num_cp)) #Population firing rate distributions by epoch
	#____
	#Toddler decoder block____
	full_taste_fr_dist = dict() #Full taste response firing rate distribution
	tastant_fr_dist = dict() #Individual neuron firing rate distributions by epoch
	tastant_fr_dist_pop = dict() #Population firing rate distributions by epoch
	for t_i in range(num_tastes):
		full_taste_fr_dist[t_i] = dict()
		tastant_fr_dist[t_i] = dict()
		tastant_fr_dist_pop[t_i] = dict()
		for n_i in range(num_neur):
			full_taste_fr_dist[t_i][n_i] = dict()
			tastant_fr_dist[t_i][n_i] = dict()
			tastant_fr_dist_pop[t_i][n_i] = dict()
			for d_i in range(max_num_deliv):
				full_taste_fr_dist[t_i][n_i][d_i] = dict()
				tastant_fr_dist[t_i][n_i][d_i] = dict()
				tastant_fr_dist_pop[t_i][n_i][d_i] = dict()
				for cp_i in range(num_cp):
					tastant_fr_dist[t_i][n_i][d_i][cp_i] = dict()
					tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = dict()
	#____
	max_hz = 0
	min_hz = 0
	max_hz_pop = 0
	min_hz_pop = 0
	max_hz_full = 0
	min_hz_full = 0
	for t_i in range(num_tastes):
		num_deliv = taste_num_deliv[t_i]
		taste_cp = taste_cp_raster_inds[t_i]
		taste_cp_pop = pop_taste_cp_raster_inds[t_i]
		for n_i in range(num_neur):
			for d_i in range(num_deliv): #index for that taste
				raster_times = tastant_spike_times[t_i][d_i][n_i]
				start_taste_i = start_dig_in_times[t_i][d_i]
				deliv_cp = taste_cp[d_i,n_i,:] - pre_taste_dt
				deliv_cp_pop = taste_cp_pop[d_i,:] - pre_taste_dt
				#Bin the average firing rates following taste delivery start
				times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
				bin_post_taste = np.zeros(post_taste_dt)
				bin_post_taste[times_post_taste] += 1
# 				deliv_binned_fr = []
# 				deliv_binned_fr_pop = []
				for cp_i in range(num_cp):
					#individual neuron changepoints
					start_epoch = int(deliv_cp[cp_i])
					end_epoch = int(deliv_cp[cp_i+1])
					#____Baby decoder block____
# 					bst_hz = ((np.sum(bin_post_taste[start_epoch:end_epoch])/((end_epoch - start_epoch)*(1/1000))) - mean_fr[n_i])/std_fr[n_i]
# 					if bst_hz > max_hz:
# 						max_hz = bst_hz
# 					deliv_binned_fr.extend([bst_hz])
					#____
					#____Toddler decoder block____
					#TODO: add variable to change the bin size
					bin_edges = np.arange(start_epoch,end_epoch,100).astype('int') #bin the epoch
					if len(bin_edges) != 0:
						if bin_edges[-1] != end_epoch:
							bin_edges = np.concatenate((bin_edges,end_epoch*np.ones(1).astype('int')))
						bst_hz = [np.sum(bin_post_taste[bin_edges[b_i]:bin_edges[b_i+1]])/((bin_edges[b_i+1] - bin_edges[b_i])*(1/1000)) for b_i in range(len(bin_edges)-1)]
						bst_hz_z = (np.array(bst_hz) - mean_fr[n_i])/std_fr[n_i]
						tastant_fr_dist[t_i][n_i][d_i][cp_i] = bst_hz_z
						if np.max(bst_hz_z) > max_hz:
							max_hz = np.max(bst_hz_z)
						if np.min(bst_hz_z) < min_hz:
							min_hz = np.min(bst_hz_z)
					#____
					#population changepoints
					start_epoch = int(deliv_cp_pop[cp_i])
					end_epoch = int(deliv_cp_pop[cp_i+1])
					#____Baby decoder block____
# 					bst_hz = ((np.sum(bin_post_taste[start_epoch:end_epoch])/((end_epoch - start_epoch)*(1/1000))) - mean_fr[n_i])/std_fr[n_i]
# 					if bst_hz > max_hz_pop:
# 						max_hz_pop = bst_hz
# 					deliv_binned_fr_pop.extend([bst_hz])
					#____
					#____Toddler decoder block____
					#TODO: add variable to change the bin size
					bin_edges = np.arange(start_epoch,end_epoch,100).astype('int') #bin the epoch
					if len(bin_edges) != 0:
						if bin_edges[-1] != end_epoch:
							bin_edges = np.concatenate((bin_edges,end_epoch*np.ones(1).astype('int')))
						bst_hz = [np.sum(bin_post_taste[bin_edges[b_i]:bin_edges[b_i+1]])/((bin_edges[b_i+1] - bin_edges[b_i])*(1/1000)) for b_i in range(len(bin_edges)-1)]
						bst_hz_z = (np.array(bst_hz) - mean_fr[n_i])/std_fr[n_i]
						tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = bst_hz_z
						if np.max(bst_hz_z) > max_hz_pop:
							max_hz_pop = np.max(bst_hz_z)
						if np.min(bst_hz_z) < min_hz_pop:
							min_hz_pop = np.min(bst_hz_z)
					#____
				del cp_i, start_epoch, end_epoch, bst_hz, bst_hz_z
				#____Baby decoder block____
# 				full_taste_fr_dist[t_i,n_i,d_i] = ((np.sum(bin_post_taste)/(post_taste_dt*(1/1000))) - mean_fr[n_i])/std_fr[n_i]
# 				tastant_fr_dist[t_i,n_i,d_i,:] = deliv_binned_fr
# 				tastant_fr_dist_pop[t_i,n_i,d_i,:] = deliv_binned_fr_pop
				#____
				#____Toddler decoder block____
				bin_edges = np.arange(0,post_taste_dt,100).astype('int') #bin the epoch
				if len(bin_edges) != 0:
					if bin_edges[-1] != post_taste_dt:
						bin_edges = np.concatenate((bin_edges,post_taste_dt*np.ones(1).astype('int')))
					bst_hz = [np.sum(bin_post_taste[bin_edges[b_i]:bin_edges[b_i+1]])/((bin_edges[b_i+1] - bin_edges[b_i])*(1/1000)) for b_i in range(len(bin_edges)-1)]
					bst_hz_z = (np.array(bst_hz) - mean_fr[n_i])/std_fr[n_i]
					full_taste_fr_dist[t_i][n_i][d_i] = bst_hz_z
					if np.max(bst_hz_z) > max_hz_full:
						max_hz_full = np.max(bst_hz_z)
					if np.min(bst_hz_z) < min_hz_full:
						min_hz_full = np.max(bst_hz_z)
				#___
	del t_i, num_deliv, taste_cp, n_i, d_i, raster_times, start_taste_i, deliv_cp, times_post_taste, bin_post_taste

	return full_taste_fr_dist, tastant_fr_dist, tastant_fr_dist_pop, \
		taste_num_deliv, max_hz, max_hz_pop, max_hz_full, min_hz, min_hz_pop, min_hz_full


def decode_full_zscore(full_taste_fr_dist_z,segment_spike_times,post_taste_dt,
				   skip_dt,dig_in_names,segment_times,segment_names,bin_dt,
				   start_dig_in_times,taste_num_deliv,taste_select,max_hz,min_hz,save_dir):
	"""Decode probability of full taste response from sliding bin spiking 
	across segments"""
	#Pull necessary variables
	num_tastes, num_neur, num_deliv = np.shape(full_taste_fr_dist_z)
	num_segments = len(segment_spike_times)
	hist_bins = np.arange(min_hz,max_hz+1,0.25)
	x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
	p_taste = taste_num_deliv/np.sum(taste_num_deliv)
	half_bin = np.floor(post_taste_dt/2).astype('int')
	half_bin_z = np.floor(bin_dt/2).astype('int')
	
	prev_run = np.zeros(2)
	
	#Get neurons to cycle through based on binary taste_select which is num_neur in length
	taste_select_neur = np.where(taste_select == 1)[0]
	
	#Fit gamma distributions to fr of each neuron for each taste
	try:
		fit_tastant_neur = np.load(save_dir + 'fit_tastant_neur_full_taste_z.npy')
		prev_run[0] = 1
	except:
		fit_tastant_neur = np.zeros((num_tastes,num_neur,len(x_vals)+1))
		for t_i in range(num_tastes):
			for n_i in taste_select_neur:
				full_data = (full_taste_fr_dist_z[t_i,n_i,:]).flatten()
				full_data = full_data[~np.isnan(full_data)]
				num_points = len(full_data)
				bin_centers = np.linspace(min_hz-1,max_hz+1,np.max([8,np.ceil(num_points/5).astype('int')]))
				bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
				bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
				fit_data = np.histogram(full_data,density=True,bins=bins_calc)
				new_fit = interpolate.interp1d(bin_centers,fit_data[0],kind='linear')
				filtered_data = new_fit(hist_bins)
				filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
				fit_tastant_neur[t_i,n_i,:] = filtered_data
			del n_i, full_data, num_points, bin_centers, bins_calc, fit_data, new_fit, filtered_data
		np.save(save_dir + 'fit_tastant_neur_full_taste_z.npy',fit_tastant_neur)
	
	#Fit the joint distribution across tastes
	#print("\tFitting joint distribution by neuron")
	try:
		joint_fit_neur = np.load(save_dir + 'joint_fit_neur_full_taste_z.npy')
		prev_run[1] = 1
	except:
		joint_fit_neur = np.zeros((num_neur,len(x_vals)+1))
		for n_i in taste_select_neur:
			all_taste_fr = (full_taste_fr_dist_z[:,n_i,:]).flatten()
			all_taste_fr = all_taste_fr[~np.isnan(all_taste_fr)]
			bin_centers = np.linspace(min_hz-1,max_hz+1)
			bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
			bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
			fit_data = np.histogram(all_taste_fr,density=True,bins=bins_calc)
			new_fit = interpolate.interp1d(bin_centers,fit_data[0])
			filtered_data = new_fit(hist_bins)
			filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
			joint_fit_neur[n_i,:] = filtered_data
		del n_i, all_taste_fr, bin_centers, bins_calc, fit_data, new_fit, filtered_data
		np.save(save_dir + 'joint_fit_neur_full_taste_z.npy',joint_fit_neur)
	
	if np.sum(prev_run) < 2:
		#Plot the taste distributions against each other
		fig_t, ax_t = plt.subplots(nrows=num_neur,ncols=1,sharex=True,figsize=(5,num_neur))
		for n_i in taste_select_neur: 
			if n_i == 0:	
				ax_t[n_i].plot((fit_tastant_neur[:,n_i,:]).T,label=np.array(dig_in_names))
				ax_t[n_i].plot((joint_fit_neur[n_i,:]).T,label='Joint Dist')
				ax_t[n_i].legend()
			else:
				ax_t[n_i].plot((fit_tastant_neur[:,n_i,:]).T)
				ax_t[n_i].plot((joint_fit_neur[n_i,:]).T,label='Joint Dist')
				ax_t[n_i].legend()
		ax_t[num_neur-1].set_xlabel('Firing Rate (Hz)')
		fig_t.supylabel('Probability')
		fig_t.tight_layout()
		fig_t.savefig(save_dir + 'full_taste_fr_dist_fits_z.png')
		fig_t.savefig(save_dir + 'full_taste_fr_dist_fits_z.svg')
		plt.close(fig_t)
	
	#Segment-by-segment use sliding bin with a skip
	seg_decode_save_dir = save_dir + 'decode_prob_full_taste/'
	if not os.path.isdir(seg_decode_save_dir):
		os.mkdir(seg_decode_save_dir)
	for s_i in range(num_segments):
		try:
			seg_decode_prob = np.load(seg_decode_save_dir + 'segment_' + str(s_i) + '.npy')
			print("Segment " + str(s_i) + " Previously Decoded")
		except:
			print("Decoding Segment " + str(s_i))
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = seg_end - seg_start
			#Segment mean and std firing rates
			time_bin_starts_z = np.arange(seg_start+half_bin_z,seg_end-half_bin_z,half_bin_z*2)
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in range(num_neur):
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			tb_fr = np.zeros((num_neur,len(time_bin_starts_z)))
			for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts_z)):
				tb_fr[:,tb_i] = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin_z:tb+half_bin_z-seg_start],1)/(2*half_bin_z*(1/1000))
			mean_fr = np.mean(tb_fr,1)
			std_fr = np.std(tb_fr,1)
			#Decode
			time_bin_starts = np.arange(seg_start+half_bin,seg_end-half_bin,skip_dt)
			num_times = len(time_bin_starts)
			seg_decode_prob = np.zeros((num_tastes,seg_len+1))
			try:
				tb_fr = np.load(seg_decode_save_dir + 'segment_' + str(s_i) + '_tb_fr.npy')
			except:
				tb_fr = np.zeros((num_neur,len(time_bin_starts)))
				for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts)):
					tb_fr_orig = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin:tb+half_bin-seg_start],1)/(2*half_bin*(1/1000))
					tb_fr[:,tb_i] = (tb_fr_orig - mean_fr)/std_fr
				del tb_i, tb
				np.save(seg_decode_save_dir + 'segment_' + str(s_i) + '_tb_fr.npy',tb_fr)
			list_tb_fr = list(tb_fr.T)
			del tb_fr
			#Pass inputs to parallel computation on probabilities
			inputs = zip(list_tb_fr, itertools.repeat(num_tastes), \
				 itertools.repeat(num_neur),itertools.repeat(x_vals), \
				 itertools.repeat(fit_tastant_neur), itertools.repeat(joint_fit_neur), \
				 itertools.repeat(p_taste),itertools.repeat(taste_select_neur))
			tic = time.time()
			pool = Pool(4)
			tb_decode_prob = pool.map(dp.segment_taste_decode_parallelized, inputs)
			pool.close()
			toc = time.time()
			print('\t\tTime to decode = ' + str(np.round((toc-tic)/60,2)) + ' (min)')
			tb_decode_array = np.squeeze(np.array(tb_decode_prob)).T
			#___
			for s_dt in range(skip_dt): #The whole skip interval should have the same decode probability
				seg_decode_prob[:,time_bin_starts - seg_start + s_dt] = tb_decode_array
			
			#Save decoding probabilities
			np.save(seg_decode_save_dir + 'segment_' + str(s_i) + '.npy',seg_decode_prob)
			
			#Create plots
			all_time = np.arange(seg_start,seg_end+1)
			seg_decode_taste_ind = np.argmax(seg_decode_prob,0)
			seg_decode_taste_bin = np.zeros(np.shape(seg_decode_prob))
			for t_i in range(num_tastes):
				seg_decode_taste_bin[t_i,np.where(seg_decode_taste_ind == t_i)[0]] = 1
			#Line plot
			f1 = plt.figure()
			plt.plot(all_time/1000/60,seg_decode_prob.T)
			for t_i in range(num_tastes):
				plt.fill_between(all_time/1000/60,seg_decode_taste_bin[t_i,:],alpha=0.2)
			plt.legend(dig_in_names,loc='right')
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f1.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '.png')
			f1.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '.svg')
			plt.close(f1)
			#Imshow
			f2 = plt.figure()
			plt.imshow(seg_decode_prob,aspect='auto',interpolation = 'none')
			x_ticks = np.ceil(np.linspace(0,seg_len,10)).astype('int')
			x_tick_labels = np.round(all_time[x_ticks]/1000/60,2)
			plt.xticks(x_ticks,x_tick_labels)
			y_ticks = np.arange(len(dig_in_names))
			plt.yticks(y_ticks,dig_in_names)
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f2.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_im.png')
			f2.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_im.svg')
			plt.close(f2)
			#Fraction of occurrences
			f3 = plt.figure()
			plt.pie(np.sum(seg_decode_taste_bin,1)/np.sum(seg_decode_taste_bin),labels=['water','saccharin','none'],autopct='%1.1f%%')
			plt.title('Segment ' + str(s_i))
			f3.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_pie.png')
			f3.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_pie.svg')
			plt.close(f3)
			#If it's the taste interval, save separately decoding of each taste delivery
			if segment_names[s_i].lower() == 'taste': #Assumes it's always called just "taste"
				#Create taste decode dir
				seg_taste_decode_save_dir = seg_decode_save_dir + 'taste_decode/'
				if not os.path.isdir(seg_taste_decode_save_dir):
					os.mkdir(seg_taste_decode_save_dir)
				for t_i in range(num_tastes): #Do each taste and find if match
					for st_i,st in enumerate(np.array(start_dig_in_times[t_i])):
						#Plot the decoding to [-post_taste_dt,2*post_taste_dt] around delivery
						f4 = plt.figure()
						start_dec_t = max(st - post_taste_dt,seg_start)
						closest_tbs = np.argmin(np.abs(time_bin_starts - start_dec_t))
						end_dec_t = min(st + 2*post_taste_dt,seg_end)
						closest_tbe = np.argmin(np.abs(time_bin_starts - end_dec_t))
						closest_td = np.argmin(np.abs(time_bin_starts - st))
						decode_tbs = np.arange(closest_tbs,closest_tbe)
						decode_t = time_bin_starts[decode_tbs]
						decode_t_labels = decode_t - st #in ms
						decode_snip = seg_decode_prob[:,decode_tbs]
						plt.plot(decode_t_labels,decode_snip.T)
						for t_i_2 in range(num_tastes):
							plt.fill_between(decode_t_labels,decode_snip[t_i_2,:],alpha=0.2)
						plt.axvline(0)
						plt.legend(dig_in_names)
						plt.ylabel('Decoding Fraction')
						plt.xlabel('Time From Delivery (ms)')
						plt.title(dig_in_names[t_i] + ' delivery #' + str(st_i))
						f4.savefig(seg_taste_decode_save_dir + dig_in_names[t_i] + '_' + str(st_i) + '.png')
						f4.savefig(seg_taste_decode_save_dir + dig_in_names[t_i] + '_' + str(st_i) + '.svg')
						plt.close(f4)




def decode_epochs_zscore(tastant_fr_dist_z,segment_spike_times,post_taste_dt,
				   skip_dt,e_skip_dt,e_len_dt,dig_in_names,segment_times,bin_dt,
				   segment_names,start_dig_in_times,taste_num_deliv,
				   taste_select_epoch,use_full,max_hz,min_hz,save_dir,neuron_count_thresh):
	"""Decode probability of each epoch in high-probability replay regions
	found using decode_phase_1 based on full taste profile decoding.
	Use parallelized function to speed up."""
	#Pull necessary variables
	#____Baby decoder block____
# 	num_tastes, num_neur, max_num_deliv, num_cp = np.shape(tastant_fr_dist_z)
	#____
	#____Toddler decoder block____
	num_tastes = len(start_dig_in_times)
	num_neur = len(segment_spike_times[0])
	max_num_deliv = np.max(taste_num_deliv)
	num_cp = len(tastant_fr_dist_z[0][0][0])	
	#___
	num_segments = len(segment_spike_times)
	hist_bins = np.arange(min_hz-1,max_hz+1,step=0.25)
	x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
	p_taste = taste_num_deliv/np.sum(taste_num_deliv)
	half_bin = np.floor(e_len_dt/2).astype('int')
	half_bin_z = np.floor(bin_dt/2).astype('int')
	
	for e_i in range(num_cp): #By epoch conduct decoding
		print('Decoding Epoch ' + str(e_i))
	
		prev_run = np.zeros(2)
		
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		
		#Fit gamma distributions to fr of each neuron for each taste
		try:
			fit_tastant_neur = np.load(save_dir + 'fit_tastant_neur_epoch_' + str(e_i) + '_z.npy')
			prev_run[0] = 1
		except:	
			fit_tastant_neur = np.zeros((num_tastes,num_neur,len(x_vals)))
			for t_i in range(num_tastes):
				for n_i in taste_select_neur:
					#____Baby decoder block____
# 					full_data = (tastant_fr_dist_z[t_i,n_i,:,e_i]).flatten()
# 					full_data = full_data[~np.isnan(full_data)]
					#____
					#____Toddler decoder block____
					full_data = []
					for d_i in range(max_num_deliv):
						full_data.extend(tastant_fr_dist_z[t_i][n_i][d_i][e_i])
					#____
					num_points = len(full_data)
					bin_centers = np.linspace(min_hz-1,max_hz+1,np.max([20,np.ceil(num_points/5).astype('int')]))
					bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
					bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
					fit_data = np.histogram(full_data,density=True,bins=bins_calc)
					new_fit = interpolate.interp1d(bin_centers,fit_data[0],kind='linear')
					filtered_data = new_fit(x_vals)
					filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
					fit_tastant_neur[t_i,n_i,:] = filtered_data
				del n_i, full_data, num_points, bin_centers, bins_calc, fit_data, new_fit, filtered_data
			np.save(save_dir + 'fit_tastant_neur_epoch_' + str(e_i) + '_z.npy',fit_tastant_neur)
		
		#Fit the joint distribution across tastes
		#print("\tFitting joint distribution by neuron")
		try:
			joint_fit_neur = np.load(save_dir + 'joint_fit_neur_epoch_' + str(e_i) + '_z.npy')
			prev_run[1] = 1
		except:
			joint_fit_neur = np.zeros((num_neur,len(x_vals)))
			for n_i in taste_select_neur:
				#____Baby decoder block____
# 				all_taste_fr = (tastant_fr_dist_z[:,n_i,:,e_i]).flatten()
# 				all_taste_fr = all_taste_fr[~np.isnan(all_taste_fr)]
				#____
				#____Toddler decoder block____
				all_taste_fr = []
				for t_i in range(num_tastes):
					for d_i in range(max_num_deliv):
						all_taste_fr.extend(tastant_fr_dist_z[t_i][n_i][d_i][e_i])
				#____
				num_points = len(all_taste_fr)
				bin_centers = np.linspace(min_hz-1,max_hz+1)
				bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
				bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
				fit_data = np.histogram(all_taste_fr,density=True,bins=bins_calc)
				new_fit = interpolate.interp1d(bin_centers,fit_data[0])
				filtered_data = new_fit(x_vals)
				filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
				joint_fit_neur[n_i,:] = filtered_data
			del n_i, all_taste_fr, bin_centers, bins_calc, fit_data, new_fit, filtered_data
			np.save(save_dir + 'joint_fit_neur_epoch_' + str(e_i) + '_z.npy',joint_fit_neur)
	
		#Plot the taste distributions against each other
		if np.sum(prev_run) < 2:
			fig_t, ax_t = plt.subplots(nrows=num_neur,ncols=1,sharex=True,figsize=(5,num_neur))
			for n_i in taste_select_neur: 
				if n_i == 0:	
					ax_t[n_i].plot(x_vals,(fit_tastant_neur[:,n_i,:]).T,label=np.array(dig_in_names))
					ax_t[n_i].plot(x_vals,(joint_fit_neur[n_i,:]).T,label='Joint Dist')
					ax_t[n_i].legend()
				else:
					ax_t[n_i].plot(x_vals,(fit_tastant_neur[:,n_i,:]).T)
					ax_t[n_i].plot(x_vals,(joint_fit_neur[n_i,:]).T,label='Joint Dist')
					ax_t[n_i].legend()
			ax_t[num_neur-1].set_xlabel('Firing Rate (Hz)')
			fig_t.supylabel('Probability')
			fig_t.tight_layout()
			fig_t.savefig(save_dir + 'epoch_' + str(e_i) + '_fr_dist_fits.png')
			fig_t.savefig(save_dir + 'epoch_' + str(e_i) + '_fr_dist_fits.svg')
			plt.close(fig_t)
		
		#Segment-by-segment use full taste decoding times to zoom in and test 
		#	epoch-specific and smaller interval
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			os.mkdir(epoch_decode_save_dir)
		for s_i in range(num_segments):
			#Get segment variables
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = seg_end - seg_start #in dt = ms
			#Segment mean and std firing rates
			time_bin_starts_z = np.arange(seg_start+half_bin_z,seg_end-half_bin_z,half_bin_z*2)
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in range(num_neur):
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			tb_fr = np.zeros((num_neur,len(time_bin_starts_z)))
			for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts_z)):
				tb_fr[:,tb_i] = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin_z:tb+half_bin_z-seg_start],1)/(2*half_bin_z*(1/1000))
			mean_fr = np.mean(tb_fr,1)
			std_fr = np.std(tb_fr,1)
			#Decode
			if use_full == 1:
				#Load previously calculated probabilities and pull start/end times
				time_bin_starts = np.arange(seg_start+half_bin,seg_end-post_taste_dt,skip_dt)
				seg_decode_prob = np.load(save_dir + 'decode_prob_full_taste/' + 'segment_' + str(s_i) + '.npy')
				seg_decode_pref = seg_decode_prob/np.sum(seg_decode_prob,0)
				seg_decode_taste_ind = np.argmax(seg_decode_pref,0)
				seg_decode_taste_bin = np.zeros(np.shape(seg_decode_pref)[-1])
				for t_i in range(num_tastes - 1): #Assumes last taste is "none"
 					seg_decode_taste_bin[np.where(seg_decode_taste_ind == t_i)[0]] = 1
				start_decode_bins = np.where(np.diff(seg_decode_taste_bin) == 1)[0] + 1
				end_decode_bins = np.where(np.diff(seg_decode_taste_bin) == -1)[0] + 1
				if len(start_decode_bins) < len(end_decode_bins):
 					start_decode_bins = np.concatenate((np.zeros(1),start_decode_bins)).astype('int')
				if len(end_decode_bins) < len(start_decode_bins):
 					end_decode_bins = np.concatenate((end_decode_bins,(len(seg_decode_taste_bin)-1)*np.ones(1))).astype('int')
				#Get bins of full taste decoding back to original segment times
				start_decode_times = time_bin_starts[start_decode_bins]
				end_decode_times = time_bin_starts[end_decode_bins]
				new_time_bins = []
				for sd_i,sd in enumerate(start_decode_times):
 					decode_starts = np.arange(sd,end_decode_times[sd_i],e_skip_dt)
 					new_time_bins.extend(list(decode_starts))
				new_time_bins = np.array(new_time_bins)
			else:
				new_time_bins = np.arange(seg_start+half_bin,seg_end-half_bin,e_skip_dt)
			#Now pull epoch-specific probabilities (only in previously decoded times)
			seg_decode_epoch_prob = np.zeros((num_tastes,seg_len))
			seg_decode_epoch_prob[-1,:] = 1 #Start with assumption of "none" taste at all times
			#Binerize Spike Times
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in taste_select_neur:
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			try:
				seg_decode_epoch_prob = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '.npy')
				tb_fr = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '_tb_fr.npy')
				print('\tSegment ' + str(s_i) + ' Previously Decoded')
			except:
				print('\tDecoding Segment ' + str(s_i))
				#Perform parallel computation for each time bin
				print('\t\tCalculate firing rates for time bins')
				try:
					tb_fr = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '_tb_fr.npy')
				except:
					tb_fr = np.zeros((num_neur,len(new_time_bins)))
					for tb_i,tb in enumerate(tqdm.tqdm(new_time_bins)):
						tb_fr_orig = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin:tb+half_bin-seg_start],1)/(2*half_bin*(1/1000))
						tb_fr_z = (tb_fr_orig - mean_fr)/std_fr
						tb_fr_z[np.isnan(tb_fr_z)] = 0
						tb_fr[:,tb_i] = tb_fr_z
					np.save(epoch_decode_save_dir + 'segment_' + str(s_i) + '_tb_fr.npy',tb_fr)
				list_tb_fr = list(tb_fr.T)
				del tb_fr, tb_i, tb
				#Pass inputs to parallel computation on probabilities
				inputs = zip(list_tb_fr, itertools.repeat(num_tastes), \
					 itertools.repeat(num_neur),itertools.repeat(x_vals), \
					 itertools.repeat(fit_tastant_neur), itertools.repeat(joint_fit_neur), \
					 itertools.repeat(p_taste),itertools.repeat(taste_select_neur))
				tic = time.time()
				pool = Pool(4)
				tb_decode_prob = pool.map(dp.segment_taste_decode_parallelized, inputs)
				pool.close()
				toc = time.time()
				print('\t\tTime to decode = ' + str(np.round((toc-tic)/60,2)) + ' (min)')
				tb_decode_array = np.squeeze(np.array(tb_decode_prob)).T
				#___
				for e_dt in range(e_skip_dt): #The whole skip interval should have the same decode probability
					seg_decode_epoch_prob[:,new_time_bins - seg_start + e_dt] = tb_decode_array
				
				#Save decoding probabilities
				np.save(epoch_decode_save_dir + 'segment_' + str(s_i) + '.npy',seg_decode_epoch_prob)
			#Recreate plots regardless of if data was previously saved
			seg_decode_save_dir = epoch_decode_save_dir + 'segment_' + str(s_i) + '/'
			if not os.path.isdir(seg_decode_save_dir):
				os.mkdir(seg_decode_save_dir)
			
			seg_decode_epoch_prob_nonan = np.zeros(np.shape(seg_decode_epoch_prob))
			seg_decode_epoch_prob_nonan[:] = seg_decode_epoch_prob[:]
			seg_decode_epoch_prob_nonan[np.isnan(seg_decode_epoch_prob_nonan)] = 0
			seg_decode_epoch_taste_ind = np.argmax(seg_decode_epoch_prob,0)
			#Updated decoding based on threshold
			seg_decode_epoch_taste_bin = np.zeros(np.shape(seg_decode_epoch_prob))
			for t_i in range(num_tastes):
				taste_bin = (seg_decode_epoch_taste_ind == t_i).astype('int')
				#To ensure starts and ends of bins align
				taste_bin[0] = 0
				taste_bin[-1] = 0
				#Calculate decoding periods
				diff_decoded_taste = np.diff(taste_bin)
				start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
				end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
				num_decoded = len(start_decoded)
				#Calculate number of neurons in each period
				num_neur_decoded = np.zeros(num_decoded)
				for nd_i in range(num_decoded):
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					for n_i in range(num_neur):
						if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
							num_neur_decoded[nd_i] += 1
				#Now cut at threshold and only keep matching decoded intervals
				decode_ind = np.where(num_neur_decoded > neuron_count_thresh)[0]
				for db in decode_ind:
					s_db = start_decoded[db]
					e_db = end_decoded[db]
					seg_decode_epoch_taste_bin[t_i,s_db:e_db] = 1
			#Line plot
			f1 = plt.figure()
			plt.plot(np.arange(seg_start,seg_end)/1000/60,seg_decode_epoch_prob_nonan.T)
			for t_i in range(num_tastes):
				plt.fill_between(np.arange(seg_start,seg_end)/1000/60,seg_decode_epoch_taste_bin[t_i,:],alpha=0.2)
			plt.legend(dig_in_names,loc='right')
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f1.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '.png')
			f1.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '.svg')
			plt.close(f1)
			#Imshow
			f2 = plt.figure()
			plt.imshow(seg_decode_epoch_prob_nonan,aspect='auto',interpolation = 'none')
			x_ticks = np.ceil(np.linspace(0,len(new_time_bins)-1,10)).astype('int')
			x_tick_labels = np.round(new_time_bins[x_ticks]/1000/60,2)
			plt.xticks(x_ticks,x_tick_labels)
			y_ticks = np.arange(len(dig_in_names))
			plt.yticks(y_ticks,dig_in_names)
			plt.ylabel('Decoding Fraction')
			plt.xlabel('Time (min)')
			plt.title('Segment ' + str(s_i))
			f2.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_im.png')
			f2.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_im.svg')
			plt.close(f2)
			#Fraction of occurrences
			f3 = plt.figure()
			plt.pie(np.sum(seg_decode_epoch_taste_bin,1)/np.sum(seg_decode_epoch_taste_bin),labels=['water','saccharin','none'],autopct='%1.1f%%',pctdistance =1.5)
			plt.title('Segment ' + str(s_i))
			f3.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_pie.png')
			f3.savefig(seg_decode_save_dir + 'segment_' + str(s_i) + '_pie.svg')
			plt.close(f3)
			#If it's the taste interval, save separately decoding of each taste delivery
			if segment_names[s_i].lower() == 'taste': #Assumes it's always called just "taste"
				taste_save_dir = seg_decode_save_dir + 'taste_decode/'
				if not os.path.isdir(taste_save_dir):
					os.mkdir(taste_save_dir)
				for t_i in range(num_tastes): #Do each taste and find if match
					for st_i,st in enumerate(np.array(start_dig_in_times[t_i])):
						#Plot the decoding to [-post_taste_dt,2*post_taste_dt] around delivery
						f4 = plt.figure()
						start_dec_t = max(st - post_taste_dt,seg_start)
						closest_tbs = np.argmin(np.abs(new_time_bins - start_dec_t))
						end_dec_t = min(st + 2*post_taste_dt,seg_end)
						closest_tbe = np.argmin(np.abs(new_time_bins - end_dec_t))
						closest_td = np.argmin(np.abs(new_time_bins - st))
						decode_tbs = np.arange(closest_tbs,closest_tbe)
						decode_t = new_time_bins[decode_tbs]
						decode_t_labels = decode_t - st #in ms
						decode_snip = seg_decode_epoch_prob[:,decode_tbs]
						decode_prob_snip = seg_decode_epoch_taste_bin[:,decode_tbs]
						plt.plot(decode_t_labels,decode_snip.T)
						for t_i_2 in range(num_tastes):
							plt.fill_between(decode_t_labels,decode_prob_snip[t_i_2,:],alpha=0.2)
						plt.axvline(0)
						plt.legend(dig_in_names)
						plt.ylabel('Decoding Fraction')
						plt.xlabel('Time From Delivery (ms)')
						plt.title(dig_in_names[t_i] + ' delivery #' + str(st_i))
						f4.savefig(taste_save_dir + dig_in_names[t_i] + '_' + str(st_i) + '.png')
						f4.savefig(taste_save_dir + dig_in_names[t_i] + '_' + str(st_i) + '.svg')
						plt.close(f4)



		
def plot_decoded(fr_dist,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
				 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
				 e_skip_dt,e_len_dt,dig_in_names,segment_times,
				 segment_names,taste_num_deliv,taste_select_epoch,
				 use_full,save_dir,max_decode,max_hz,seg_stat_bin,neuron_count_thresh):
	"""Function to plot the periods when something other than no taste is 
	decoded"""
	num_segments = len(segment_spike_times)
	neur_cut = np.floor(num_neur*neuron_count_thresh).astype('int')
	taste_colors = cm.viridis(np.linspace(0,1,num_tastes))
	epoch_seg_taste_percents = np.zeros((num_cp,num_segments,num_tastes))
	for e_i in range(num_cp): #By epoch conduct decoding
		print('Plotting Decoding for Epoch ' + str(e_i))
		
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			print("Data not previously decoded, or passed directory incorrect.")
			pass
		
		for s_i in tqdm.tqdm(range(num_segments)):
			try:
				seg_decode_epoch_prob = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '.npy')
			except:
				print("Segment " + str(s_i) + " Never Decoded")
				pass
			
			seg_decode_save_dir = epoch_decode_save_dir + 'segment_' + str(s_i) + '/'
			if not os.path.isdir(seg_decode_save_dir):
				os.mkdir(seg_decode_save_dir)
			
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = seg_end - seg_start #in dt = ms
			
			#Import raster plots for segment
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in taste_select_neur:
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			
			#Calculate maximally decoded taste
			decoded_taste_max = np.argmax(seg_decode_epoch_prob,0)
			#Store binary decoding results
			decoded_taste_bin = np.zeros((num_tastes,len(decoded_taste_max)))
			for t_i in range(num_tastes):
				decoded_taste_bin[t_i,np.where(decoded_taste_max == t_i)[0]] = 1
			#To ensure starts and ends of bins align
			decoded_taste_bin[:,0] = 0
			decoded_taste_bin[:,-1] = 0
			
			#For each taste (except none) calculate start and end times of decoded intervals and plot
			all_taste_fr_vecs = []
			all_taste_fr_vecs_mean = np.zeros((num_tastes,num_neur))
			all_taste_z_fr_vecs_mean = np.zeros((num_tastes,num_neur))
			all_taste_event_fr_vecs = []
			all_taste_event_z_fr_vecs = []
			for t_i in range(num_tastes):
				#First calculate neurons decoded in all decoded intervals
				decoded_taste = decoded_taste_bin[t_i,:]
				decoded_taste[0] = 0
				decoded_taste[-1] = 0
				diff_decoded_taste = np.diff(decoded_taste)
				start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
				end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
				num_decoded = len(start_decoded)
				num_neur_decoded = np.zeros(num_decoded)
				prob_decoded = np.zeros(num_decoded)
				for nd_i in range(num_decoded):
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					d_len = d_end-d_start
					for n_i in range(num_neur):
						if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
							num_neur_decoded[nd_i] += 1
					prob_decoded[nd_i] = np.mean(seg_decode_epoch_prob[t_i,d_start:d_end])
				#Now cut at threshold and only keep matching decoded intervals
				decode_ind = np.where(num_neur_decoded > neur_cut)[0]
				decoded_bin = np.zeros(np.shape(decoded_taste))
				for db in decode_ind:
					s_db = start_decoded[db]
					e_db = end_decoded[db]
					decoded_bin[s_db:e_db] = 1
				#Save the percent taste decoded matching threshold
				epoch_seg_taste_percents[e_i,s_i,t_i] = (np.sum(decoded_bin)/len(decoded_bin))*100	
				#Re-calculate start and end times of the decoded intervals
				start_decoded = start_decoded[decode_ind]
				end_decoded = end_decoded[decode_ind]
				#Re-calculate the decoded statistics
				num_decoded = len(start_decoded)
				num_neur_decoded = num_neur_decoded[decode_ind]
				prob_decoded = prob_decoded[decode_ind]
				len_decoded = end_decoded-start_decoded
				iei_decoded = start_decoded[1:] - end_decoded[:-1]
				
				#Import taste spike and cp times
				taste_spike_times = tastant_spike_times[t_i]
				taste_deliv_times = start_dig_in_times[t_i]
				pop_taste_cp_times = pop_taste_cp_raster_inds[t_i]
				#Store as binary spike arrays
				taste_spike_times_bin = np.zeros((len(taste_spike_times),num_neur,post_taste_dt))
				taste_cp_times = np.zeros((len(taste_spike_times),num_cp+1)).astype('int')
				taste_epoch_fr_vecs = np.zeros((len(taste_spike_times),num_neur))
				for d_i in range(len(taste_spike_times)): #store each delivery to binary spike matrix
					for n_i in range(num_neur):
						if t_i == num_tastes-1:
							if len(taste_spike_times[d_i][n_i]) > 0:
								d_i_spikes = np.array(taste_spike_times[d_i][n_i] - np.min(taste_spike_times[d_i][n_i])).astype('int')
							else:
								d_i_spikes = np.empty(0)
						else:
							d_i_spikes = np.array(taste_spike_times[d_i][n_i] - taste_deliv_times[d_i]).astype('int')
						if len(d_i_spikes) > 0:
							taste_spike_times_bin[d_i,n_i,d_i_spikes[d_i_spikes<post_taste_dt]] = 1
					taste_cp_times[d_i,:] = np.concatenate((np.cumsum(np.diff(pop_taste_cp_times[d_i,:])),post_taste_dt*np.ones(1))).astype('int')
					#Calculate the FR vectors by epoch for each taste response and the average FR vector
					taste_epoch_fr_vecs[d_i,:] = np.sum(taste_spike_times_bin[d_i,:,taste_cp_times[d_i,e_i]:taste_cp_times[d_i,e_i+1]],1)/((taste_cp_times[d_i,e_i+1]-taste_cp_times[d_i,e_i])/1000) #FR in HZ
				all_taste_fr_vecs.append(taste_epoch_fr_vecs)
				#Calculate average taste fr vec
				taste_fr_vecs_mean = np.nanmean(taste_epoch_fr_vecs,0)
				all_taste_fr_vecs_mean[t_i,:] = taste_fr_vecs_mean
				taste_fr_vecs_mean_z = (taste_fr_vecs_mean - np.mean(taste_fr_vecs_mean))/np.std(taste_fr_vecs_mean)
				all_taste_z_fr_vecs_mean[t_i,:] = taste_fr_vecs_mean_z
				taste_fr_vecs_max_hz = np.max(taste_epoch_fr_vecs)
				
				#Create plots of decoded period statistics
				seg_dist_starts = np.arange(0,seg_len,seg_stat_bin)
				seg_distribution = np.zeros(len(seg_dist_starts)-1)
				prob_distribution = np.zeros(len(seg_dist_starts)-1)
				for sd_i in range(len(seg_dist_starts)-1):
					bin_events = np.where((start_decoded < seg_dist_starts[sd_i+1])*(start_decoded >= seg_dist_starts[sd_i]))[0]
					seg_distribution[sd_i] = len(bin_events)
					prob_distribution[sd_i] = np.mean(prob_decoded[bin_events])
				seg_dist_midbin = seg_dist_starts[:-1] + np.diff(seg_dist_starts)/2
				
				taste_decode_save_dir = seg_decode_save_dir + dig_in_names[t_i] + '_events/'
				if not os.path.isdir(taste_decode_save_dir):
					os.mkdir(taste_decode_save_dir)
				
				#Create statistics plots
				f = plt.figure(figsize=(8,8))
				plt.subplot(3,2,1)
				plt.hist(len_decoded)
				plt.xlabel('Length (ms)')
				plt.ylabel('Number of Occurrences')
				plt.title('Length of Decoded Event')
				plt.subplot(3,2,2)
				plt.hist(iei_decoded)
				plt.xlabel('IEI (ms)')
				plt.ylabel('Number of Occurrences')
				plt.title('Inter-Event-Interval (IEI)')
				plt.subplot(3,2,3)
				plt.hist(num_neur_decoded)
				plt.xlabel('# Neurons')
				plt.ylabel('Number of Occurrences')
				plt.title('Number of Neurons Active')
				plt.subplot(3,2,4)
				plt.bar(seg_dist_midbin,seg_distribution,width=seg_stat_bin)
				plt.xticks(np.linspace(0,seg_len,8),labels=np.round((np.linspace(0,seg_len,8)/1000/60),2),rotation=-45)
				plt.xlabel('Time in Segment (min)')
				plt.ylabel('# Events')
				plt.title('Number of Decoded Events')
				plt.subplot(3,2,5)
				plt.hist(prob_decoded)
				plt.xlabel('Event Avg P(Decoding)')
				plt.ylabel('Number of Occurrences')
				plt.title('Average Decoding Probability')
				plt.subplot(3,2,6)
				plt.bar(seg_dist_midbin,prob_distribution,width=seg_stat_bin)
				plt.xticks(np.linspace(0,seg_len,8),labels=np.round((np.linspace(0,seg_len,8)/1000/60),2),rotation=-45)
				plt.xlabel('Time in Segment (min)')
				plt.ylabel('Avg(Event Avg P(Decoding))')
				plt.title('Average Decoding Probability')
				plt.suptitle('')
				plt.tight_layout()
				f.savefig(taste_decode_save_dir + 'epoch_' + str(e_i) + '_seg_' + str(s_i) + '_event_statistics.png')
				f.savefig(taste_decode_save_dir + 'epoch_' + str(e_i) + '_seg_' + str(s_i) + '_event_statistics.png')
				plt.close(f)
					
				#Create plots of the decoded periods
				decoded_fr_vecs = [] #Store all decoded events firing rates
				decoded_z_fr_vecs = [] #Store all decoded events z-scored firing rates
				if num_decoded > max_decode: #Reduce number if too many
					decode_prob_avg = np.array([np.mean(seg_decode_epoch_prob[t_i,start_decoded[i]:end_decoded[i]]) for i in range(len(start_decoded))])
					decode_plot_ind = np.argsort(decode_prob_avg)[-max_decode:]
				for nd_i in range(num_decoded):
					#Decode variables
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					d_len = d_end-d_start
					d_plot_start = np.max((start_decoded[nd_i]-2*d_len,0))
					d_plot_end = np.min((end_decoded[nd_i]+2*d_len,seg_len))
					d_plot_len = d_plot_end-d_plot_start
					decode_plot_times = np.arange(d_plot_start,d_plot_end)
					event_spikes = segment_spike_times_s_i_bin[:,d_plot_start:d_plot_end]
					decode_spike_times = []
					for n_i in range(num_neur):
						decode_spike_times.append(list(np.where(event_spikes[n_i,:])[0]))
					event_spikes_fr = segment_spike_times_s_i_bin[:,d_plot_start-25:d_plot_end+25]
					firing_rate_vec = np.zeros(d_plot_len)
					for t_i in range(d_plot_len):
						min_t_i = max(t_i+25-25,0)
						max_t_i = min(t_i+25+25,d_plot_len+50)
						firing_rate_vec[t_i] = np.mean(np.sum(event_spikes_fr[:,min_t_i:max_t_i],1)/(50/1000))
					d_fr_vec = np.sum(segment_spike_times_s_i_bin[:,d_start:d_end],1)/(d_len/1000)
					decoded_fr_vecs.append(d_fr_vec)
					d_fr_vec_max_hz = np.max(d_fr_vec)
					d_fr_vec_z = (d_fr_vec - np.mean(d_fr_vec))/np.std(d_fr_vec)
					decoded_z_fr_vecs.append(d_fr_vec_z)
					#If it's a event to plot
					if np.sum((decode_plot_ind == nd_i).astype('int')) > 0:
						#Start Figure
						f, ax = plt.subplots(nrows=5,ncols=2,figsize=(10,10),gridspec_kw=dict(height_ratios=[1,1,2,2,2]))
						gs = ax[0,0].get_gridspec()
						#Decoding probabilities
						ax[0,0].remove()
						ax[1,0].remove()
						axbig = f.add_subplot(gs[0:2,0])
						axbig.plot(decode_plot_times-d_start,seg_decode_epoch_prob[:,d_plot_start:d_plot_end].T)
						for t_i_2 in range(num_tastes):
 							axbig.fill_between(decode_plot_times-d_start,decoded_taste_bin[t_i_2,d_plot_start:d_plot_end],alpha=0.2)
						axbig.axvline(0,color='k',alpha=0.5)
						axbig.axvline(d_len,color='k',alpha=0.5)
						axbig.legend(dig_in_names,loc='right')
						axbig.set_ylabel('Decoding Fraction')
						axbig.set_xlabel('Time from Event Start (ms)')
						axbig.set_title('Event ' + str(nd_i) + '\nStart Time = ' + str(round(d_start/1000/60,3)) + ' Minutes')
						#Decoded raster
						ax[0,1].eventplot(decode_spike_times)
						ax[0,1].set_xlim([0,d_plot_len])
						x_ticks = np.linspace(0,d_plot_len,8).astype('int')
						x_tick_labels = np.linspace(d_plot_start,d_plot_end,8).astype('int')-d_start
						ax[0,1].set_xticks(x_ticks,labels=x_tick_labels)
						ax[0,1].axvline(d_start-d_plot_start,color='k',alpha=0.5)
						ax[0,1].axvline(d_end-d_plot_start,color='k',alpha=0.5)
						ax[0,1].set_ylabel('Neuron Index')
						ax[0,1].set_title('Event Spike Raster')
						ax[1,0].axis('off')
						#Plot population firing rates - 50ms smoothing
						ax[1,1].plot(decode_plot_times-d_start,firing_rate_vec)
						ax[1,1].axvline(0,color='k')
						ax[1,1].axvline(d_len,color='k')
						ax[1,1].set_title('Population FR')
						ax[1,1].set_ylabel('FR (Hz)')
						ax[1,1].set_xlabel('Time from Event Start (ms)')
						#Decoded Firing Rates
						img = ax[2,0].imshow(np.expand_dims(d_fr_vec,0),vmin=0,vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
						ax[2,0].set_xlabel('Neuron Index')
						ax[2,0].set_yticks(ticks=[])
						plt.colorbar(img, location='bottom',orientation='horizontal',label='Firing Rate (Hz)',panchor=(0.9,0.5),ax=ax[2,0])
						ax[2,0].set_title('Event FR')
						#Taste Firing Rates
						img = ax[2,1].imshow(np.expand_dims(taste_fr_vecs_mean,0),vmin=0,vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
						ax[2,1].set_xlabel('Neuron Index')
						ax[2,1].set_yticks(ticks=[])
						plt.colorbar(img, ax= ax[2,1], location='bottom',orientation='horizontal',label='Firing Rate (Hz)',panchor=(0.9,0.5))
						ax[2,1].set_title('Avg. Taste Resp. FR')
						#Decoded Firing Rates Z-Scored
						img = ax[3,0].imshow(np.expand_dims(d_fr_vec_z,0),vmin=-3,vmax=3,cmap='bwr')
						ax[3,0].set_xlabel('Neuron Index')
						ax[3,0].set_yticks(ticks=[])
						plt.colorbar(img, ax=ax[3,0], location='bottom',orientation='horizontal',label='Z-Scored Firing Rate (Hz)',panchor=(0.9,0.5))
						ax[3,0].set_title('Event FR Z-Scored')
						#Taste Firing Rates Z-Scored
						img = ax[3,1].imshow(np.expand_dims(taste_fr_vecs_mean_z,0),vmin=-3,vmax=3,cmap='bwr')
						ax[3,1].set_xlabel('Neuron Index')
						ax[3,1].set_yticks(ticks=[])
						plt.colorbar(img, ax=ax[3,1], location='bottom',orientation='horizontal',label='Z-Scored Firing Rate (Hz)',panchor=(0.9,0.5))
						ax[3,1].set_title('Avg. Taste Resp. FR Z-Scored')
						#Decoded Firing Rates x Average Firing Rates
						max_lim = np.max([np.max(d_fr_vec_z),np.max(taste_fr_vecs_mean)])
						ax[4,0].plot([0,max_lim],[0,max_lim],alpha=0.5,linestyle='dashed')
						ax[4,0].scatter(taste_fr_vecs_mean,d_fr_vec)
						ax[4,0].set_xlabel('Average Taste FR')
						ax[4,0].set_ylabel('Decoded Taste FR')
						ax[4,0].set_title('Firing Rate Similarity')
						#Z-Scored Decoded Firing Rates x Z-Scored Average Firing Rates
						ax[4,1].plot([-3,3],[-3,3],alpha=0.5,linestyle='dashed',color='k')
						ax[4,1].scatter(taste_fr_vecs_mean_z,d_fr_vec_z)
						ax[4,1].set_xlabel('Average Taste Neuron FR')
						ax[4,1].set_ylabel('Event Neuron FR')
						ax[4,1].set_title('Z-Scored Firing Rate Similarity')
						plt.tight_layout()
						#Save Figure
						f.savefig(taste_decode_save_dir + 'event_' + str(nd_i) + '.png')
						f.savefig(taste_decode_save_dir + 'event_' + str(nd_i) + '.svg')
						plt.close(f)
				all_taste_event_fr_vecs.append(np.array(decoded_fr_vecs))
				all_taste_event_z_fr_vecs.append(np.array(decoded_z_fr_vecs))
			
			#Taste event scatter plots
			#f, ax = plt.subplots(nrows=1, ncols=num_tastes+1, figsize=(8,8), gridspec_kw=dict(width_ratios=list(np.concatenate((6*np.ones(num_tastes),np.ones(1))))))
			f, ax = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(num_tastes*2,num_tastes*2), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
			max_fr = 0
			max_fr_t_av = 0
			for t_i in range(num_tastes): #Event Taste
				ax[t_i,0].set_ylabel('Decoded ' + dig_in_names[t_i] +' FR')
				taste_event_fr_vecs = all_taste_event_fr_vecs[t_i]
				max_taste_fr = np.max(taste_event_fr_vecs)
				if max_taste_fr > max_fr:
					max_fr = max_taste_fr
				for t_i_c in range(num_tastes): #Average Taste
					average_fr_vec_mat = all_taste_fr_vecs_mean[t_i_c,:]*np.ones(np.shape(taste_event_fr_vecs))
					#Calculate max avg fr
					max_avg_fr = np.max(all_taste_fr_vecs_mean[t_i_c,:])
					if max_avg_fr > max_fr_t_av:
						max_fr_t_av = max_avg_fr
					ax[t_i,t_i_c].set_xlabel('Average ' + dig_in_names[t_i_c] + ' FR')
					ax[t_i,t_i_c].scatter(average_fr_vec_mat,taste_event_fr_vecs,color=taste_colors[t_i,:],alpha = 0.3)
			for t_i in range(num_tastes):
				for t_i_c in range(num_tastes):
					ax[t_i,t_i_c].plot([0,max_fr],[0,max_fr],alpha=0.5,color='k',linestyle='dashed')
					ax[t_i,t_i_c].set_ylim([0,max_fr])
					ax[t_i,t_i_c].set_xlim([0,max_fr_t_av])
					if t_i == t_i_c:
						for child in ax[t_i,t_i_c].get_children():
						    if isinstance(child, matplotlib.spines.Spine):
						        child.set_color('r')
			plt.suptitle('Deviation Events x Average Taste Response')
			plt.tight_layout()
			f.savefig(seg_decode_save_dir + 'event_vs_avg_taste_deliv_scat.png')
			f.savefig(seg_decode_save_dir + 'event_vs_avg_taste_deliv_scat.svg')
			plt.close(f)
			
			#Taste event angled deviation plots
			#f, ax = plt.subplots(nrows=1, ncols=num_tastes+1, figsize=(8,8), gridspec_kw=dict(width_ratios=list(np.concatenate((6*np.ones(num_tastes),np.ones(1))))))
			f, ax = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(num_tastes*3,num_tastes*3), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
			max_y = 0
			min_y = 0
			max_x = 0
			for t_i in range(num_tastes): #Event Taste
				ax[t_i,0].set_ylabel('Decoded ' + dig_in_names[t_i] +' FR')
				taste_event_fr_vecs = all_taste_event_fr_vecs[t_i]
				max_taste_fr = np.max(taste_event_fr_vecs)
				for t_i_c in range(num_tastes): #Average Taste
					#Calculate max fr of taste response
					taste_fr_vecs = all_taste_fr_vecs[t_i_c]
					max_taste_resp_fr = np.max(taste_fr_vecs)
					x_vals = np.linspace(0,max_taste_resp_fr,num_neur)
					if max_taste_resp_fr > max_x:
						max_x = max_taste_resp_fr
					num_taste_deliv = np.shape(taste_fr_vecs)[0]
					#Calculate mean and std of event away from average
					num_events = np.shape(taste_event_fr_vecs)[0]
					all_diff = np.zeros((num_events*num_taste_deliv,num_neur))
					for td_i in range(num_taste_deliv):
						diff = taste_event_fr_vecs - taste_fr_vecs[td_i,:]
						all_diff[td_i*num_events:(td_i+1)*num_events,:] = diff
					diff_mean = np.mean(all_diff,0)
					diff_std = np.std(all_diff,0)
					if max_y < np.max(diff_mean + x_vals + diff_std):
						max_y = np.max(diff_mean + x_vals + diff_std)
					if min_y > np.min(diff_mean + x_vals - diff_std):
						min_y = np.min(diff_mean + x_vals - diff_std)
					ax[t_i,t_i_c].plot(x_vals,diff_mean + x_vals,alpha=0.5,color='b',linestyle='solid')
					ax[t_i,t_i_c].fill_between(x_vals,diff_mean + x_vals - diff_std,diff_mean + x_vals + diff_std,alpha=0.5,color='b')
					#Plot
					ax[t_i,t_i_c].set_xlabel(dig_in_names[t_i_c] + ' delivery FR')
			for t_i in range(num_tastes):
				for t_i_c in range(num_tastes):
					ax[t_i,t_i_c].plot(np.arange(max_x),np.arange(max_x),alpha=0.5,color='k',linestyle='dashed')
					ax[t_i,t_i_c].set_xlim([0,max_x])
					ax[t_i,t_i_c].set_ylim([min_y-10,max_y+10])
					if t_i == t_i_c:
						for child in ax[t_i,t_i_c].get_children():
						    if isinstance(child, matplotlib.spines.Spine):
						        child.set_color('r')
			plt.suptitle('Deviation Events x Individual Taste Response')
			plt.tight_layout()
			f.savefig(seg_decode_save_dir + 'event_vs_taste_deliv_dev.png')
			f.savefig(seg_decode_save_dir + 'event_vs_taste_deliv_dev.svg')
			plt.close(f)
			
			
	#Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments		
	f = plt.figure(figsize=(8,8))
	plot_ind = 1
	for e_i in range(num_cp):
		for t_i in range(num_tastes):
			plt.subplot(num_cp,num_tastes,plot_ind)
			plt.plot(np.arange(num_segments),(epoch_seg_taste_percents[e_i,:,t_i]).flatten())
			plt.xticks(np.arange(num_segments),labels=segment_names,rotation=-45)
			if t_i == 0:
				plt.ylabel('Epoch ' + str(e_i))
			if e_i == 0:
				plt.title('Taste ' + dig_in_names[t_i])
			plot_ind += 1
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents.png')
	f.savefig(save_dir + 'Decoding_Percents.svg')
	plt.close(f)
	

def plot_decoded_func_p(fr_dist,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
				 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
				 e_skip_dt,e_len_dt,dig_in_names,segment_times,
				 segment_names,taste_num_deliv,taste_select_epoch,
				 use_full,save_dir,max_decode,max_hz,seg_stat_bin):	
	"""Function to plot the decoding statistics as a function of average decoding
	probability within the decoded interval."""
	warnings.filterwarnings('ignore')
	num_segments = len(segment_spike_times)
	prob_cutoffs = np.arange(1/num_tastes,1,0.05)
	num_prob = len(prob_cutoffs)
	taste_colors = cm.viridis(np.linspace(0,1,num_tastes))
	epoch_seg_taste_percents = np.zeros((num_prob,num_cp,num_segments,num_tastes))
	for e_i in range(num_cp): #By epoch conduct decoding
		print('Decoding Epoch ' + str(e_i))
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			print("Data not previously decoded, or passed directory incorrect.")
			pass
		
		for s_i in tqdm.tqdm(range(num_segments)):
			try:
				seg_decode_epoch_prob = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '.npy')
			except:
				print("Segment " + str(s_i) + " Never Decoded")
				pass
			
			seg_decode_save_dir = epoch_decode_save_dir + 'segment_' + str(s_i) + '/'
			if not os.path.isdir(seg_decode_save_dir):
				os.mkdir(seg_decode_save_dir)
			
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = seg_end - seg_start #in dt = ms
			
			#Import raster plots for segment
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in taste_select_neur:
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			
			decoded_taste_max = np.argmax(seg_decode_epoch_prob,0)
			
			#Calculate decoded taste stats by probability cutoff
			num_neur_mean_p = np.zeros((num_tastes,num_prob))
			num_neur_std_p =np.zeros((num_tastes,num_prob))
			iei_mean_p = np.zeros((num_tastes,num_prob))
			iei_std_p = np.zeros((num_tastes,num_prob))
			len_mean_p = np.zeros((num_tastes,num_prob))
			len_std_p = np.zeros((num_tastes,num_prob))
			prob_mean_p = np.zeros((num_tastes,num_prob))
			prob_std_p = np.zeros((num_tastes,num_prob))
			for t_i in range(num_tastes):
				for prob_i, prob_val in enumerate(prob_cutoffs):
					#Find where the decoding matches the probability cutoff for each taste
					decode_prob_bin = seg_decode_epoch_prob[t_i,:] > prob_val
					decode_max_bin =  decoded_taste_max == t_i
					decoded_taste = (decode_prob_bin*decode_max_bin).astype('int')
					decoded_taste[0] = 0
					decoded_taste[-1] = 0
					
					#Store the decoding percents
					epoch_seg_taste_percents[prob_i,e_i,s_i,t_i] = (np.sum(decoded_taste)/len(decoded_taste))*100
				
					#Calculate statistics of num neur, IEI, event length, average decoding prob
					start_decoded = np.where(np.diff(decoded_taste) == 1)[0] + 1
					end_decoded = np.where(np.diff(decoded_taste) == -1)[0] + 1
					num_decoded = len(start_decoded)
					
					#Create plots of decoded period statistics
					#__Length
					len_decoded = end_decoded-start_decoded
					len_mean_p[t_i,prob_i] = np.nanmean(len_decoded)
					len_std_p[t_i,prob_i] = np.nanstd(len_decoded)
					#__IEI
					iei_decoded = start_decoded[1:] - end_decoded[:-1]
					iei_mean_p[t_i,prob_i] = np.nanmean(iei_decoded)
					iei_std_p[t_i,prob_i] = np.nanstd(iei_decoded)
					num_neur_decoded = np.zeros(num_decoded)
					prob_decoded = np.zeros(num_decoded)
					for nd_i in range(num_decoded):
						d_start = start_decoded[nd_i]
						d_end = end_decoded[nd_i]
						d_len = d_end-d_start
						for n_i in range(num_neur):
							if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
								num_neur_decoded[nd_i] += 1
						prob_decoded[nd_i] = np.mean(seg_decode_epoch_prob[t_i,d_start:d_end])
					#__Num Neur
					num_neur_mean_p[t_i,prob_i] = np.nanmean(num_neur_decoded)
					num_neur_std_p[t_i,prob_i] = np.nanstd(num_neur_decoded)
					#__Prob
					prob_mean_p[t_i,prob_i] = np.nanmean(prob_decoded)
					prob_std_p[t_i,prob_i] = np.nanstd(prob_decoded)
				
			#Plot statistics 
			f,ax = plt.subplots(2,3,figsize=(8,8),width_ratios=[10,10,1])
			gs = ax[0,-1].get_gridspec()
			ax[0,0].set_ylim([0,num_neur])
			ax[0,1].set_ylim([0,np.nanmax(len_mean_p) + np.nanmax(len_std_p) + 10])
			ax[1,0].set_ylim([0,np.nanmax(iei_mean_p) + np.nanmax(iei_std_p) + 10])
			ax[1,1].set_ylim([0,1.2])
			for t_i in range(num_tastes):
				#__Num Neur
				ax[0,0].errorbar(prob_cutoffs,num_neur_mean_p[t_i,:],num_neur_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__Length
				ax[0,1].errorbar(prob_cutoffs,len_mean_p[t_i,:],len_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__IEI
				ax[1,0].errorbar(prob_cutoffs,iei_mean_p[t_i,:],iei_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__Prob
				ax[1,1].errorbar(prob_cutoffs,prob_mean_p[t_i,:],prob_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
			ax[0,0].set_title('Number of Neurons')
			ax[0,1].set_title('Length of Event')
			ax[1,0].set_title('IEI (ms)')
			ax[1,1].set_title('Average P(Decoding)')
			for ax_i in ax[:,-1]:
				ax_i.remove() #remove the underlying axes
			axbig = f.add_subplot(gs[:,-1])
			cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),cax=axbig,ticks=np.linspace(0,1,num_tastes),orientation='vertical')
			cbar.ax.set_yticklabels(dig_in_names)
			#Edit and Save
			f.suptitle('Decoding Statistics by Probability Cutoff')
			plt.tight_layout()
			f.savefig(os.path.join(seg_decode_save_dir,'prob_cutoff_stats.png'))
			f.savefig(os.path.join(seg_decode_save_dir,'prob_cutoff_stats.svg'))
			plt.close(f)
				
	#Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments	
	sum_width_ratios = np.concatenate((10*np.ones(num_segments),np.ones(1)))
	max_decoding_percent = np.max(epoch_seg_taste_percents)
	f, ax = plt.subplots(num_cp,num_segments + 1,figsize=(10,10),width_ratios=sum_width_ratios)
	gs = ax[0,-1].get_gridspec()
	for e_i in range(num_cp):
		for s_i in range(num_segments):
			ax[e_i,s_i].set_ylim([0,max_decoding_percent+10])
			for t_i in range(num_tastes):
				ax[e_i,s_i].plot(prob_cutoffs,(epoch_seg_taste_percents[:,e_i,s_i,t_i]).flatten(),color=taste_colors[t_i,:],alpha=0.8)
			if s_i == 0:
				ax[e_i,s_i].set_ylabel('Epoch ' + str(e_i))
			if e_i == 0:
				ax[e_i,s_i].set_title(segment_names[s_i])
			if e_i == num_cp-1:
				ax[e_i,s_i].set_xlabel('Probability Cutoff')
	for ax_i in ax[:,-1]:
		ax_i.remove() #remove the underlying axes
	axbig = f.add_subplot(gs[:,-1])
	cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),cax=axbig,ticks=np.linspace(0,1,num_tastes),orientation='vertical')
	cbar.ax.set_yticklabels(dig_in_names)
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents_By_Prob_Cutoff.png')
	f.savefig(save_dir + 'Decoding_Percents_By_Prob_Cutoff.svg')
	plt.close(f)
	
	
def plot_decoded_func_n(fr_dist,num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
				 start_dig_in_times,end_dig_in_times,post_taste_dt,pop_taste_cp_raster_inds,
				 e_skip_dt,e_len_dt,dig_in_names,segment_times,
				 segment_names,taste_num_deliv,taste_select_epoch,
				 use_full,save_dir,max_decode,max_hz,seg_stat_bin):
	"""Function to plot the decoding statistics as a function of number of 
	neurons firing within the decoded interval."""
	warnings.filterwarnings('ignore')
	num_segments = len(segment_spike_times)
	neur_cutoffs = np.arange(1,num_neur)
	num_cuts = len(neur_cutoffs)
	taste_colors = cm.viridis(np.linspace(0,1,num_tastes))
	epoch_seg_taste_percents = np.zeros((num_cuts,num_cp,num_segments,num_tastes))
	for e_i in range(num_cp): #By epoch conduct decoding
		print('Decoding Epoch ' + str(e_i))
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			print("Data not previously decoded, or passed directory incorrect.")
			pass
		
		for s_i in tqdm.tqdm(range(num_segments)):
			try:
				seg_decode_epoch_prob = np.load(epoch_decode_save_dir + 'segment_' + str(s_i) + '.npy')
			except:
				print("Segment " + str(s_i) + " Never Decoded")
				pass
			
			seg_decode_save_dir = epoch_decode_save_dir + 'segment_' + str(s_i) + '/'
			if not os.path.isdir(seg_decode_save_dir):
				os.mkdir(seg_decode_save_dir)
			
			seg_start = segment_times[s_i]
			seg_end = segment_times[s_i+1]
			seg_len = seg_end - seg_start #in dt = ms
			
			#Import raster plots for segment
			segment_spike_times_s_i = segment_spike_times[s_i]
			segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
			for n_i in taste_select_neur:
				n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
				segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
			
			decoded_taste_max = np.argmax(seg_decode_epoch_prob,0)
			
			#Calculate decoded taste stats by probability cutoff
			num_neur_mean_p = np.zeros((num_tastes,num_cuts))
			num_neur_std_p =np.zeros((num_tastes,num_cuts))
			iei_mean_p = np.zeros((num_tastes,num_cuts))
			iei_std_p = np.zeros((num_tastes,num_cuts))
			len_mean_p = np.zeros((num_tastes,num_cuts))
			len_std_p = np.zeros((num_tastes,num_cuts))
			prob_mean_p = np.zeros((num_tastes,num_cuts))
			prob_std_p = np.zeros((num_tastes,num_cuts))
			for t_i in range(num_tastes):
				#First calculate neurons decoded in all decoded intervals
				decoded_taste = (decoded_taste_max == t_i).astype('int')
				decoded_taste[0] = 0
				decoded_taste[-1] = 0
				diff_decoded_taste = np.diff(decoded_taste)
				start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
				end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
				num_decoded = len(start_decoded)
				num_neur_decoded = np.zeros(num_decoded)
				prob_decoded = np.zeros(num_decoded)
				for nd_i in range(num_decoded):
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					d_len = d_end-d_start
					for n_i in range(num_neur):
						if len(np.where(segment_spike_times_s_i_bin[n_i,d_start:d_end])[0]) > 0:
							num_neur_decoded[nd_i] += 1
					prob_decoded[nd_i] = np.mean(seg_decode_epoch_prob[t_i,d_start:d_end])
				#Next look at stats as exclude by #neurons
				for cut_i, cut_val in enumerate(neur_cutoffs):
					#Find where the decoding matches the neuron cutoff
					decode_ind = np.where(num_neur_decoded > cut_val)[0]
					decoded_bin = np.zeros(np.shape(decoded_taste))
					for db in decode_ind:
						s_db = start_decoded[db]
						e_db = end_decoded[db]
						decoded_bin[s_db:e_db] = 1
					#Store the decoding percents
					epoch_seg_taste_percents[cut_i,e_i,s_i,t_i] = (np.sum(decoded_bin)/len(decoded_bin))*100
					#Calculate statistics of num neur, IEI, event length, average decoding prob
					num_decoded_i = num_neur_decoded[decode_ind]
					prob_decoded_i = prob_decoded[decode_ind]
					iei_i = start_decoded[decode_ind[1:]] - end_decoded[decode_ind[:-1]]
					len_i = end_decoded[decode_ind] - start_decoded[decode_ind]
					
					#Create plots of decoded period statistics
					#__Length
					len_decoded = end_decoded-start_decoded
					len_mean_p[t_i,cut_i] = np.nanmean(len_i)
					len_std_p[t_i,cut_i] = np.nanstd(len_i)
					#__IEI
					iei_decoded = start_decoded[1:] - end_decoded[:-1]
					iei_mean_p[t_i,cut_i] = np.nanmean(iei_i)
					iei_std_p[t_i,cut_i] = np.nanstd(iei_i)
					#__Num Neur
					num_neur_mean_p[t_i,cut_i] = np.nanmean(num_decoded_i)
					num_neur_std_p[t_i,cut_i] = np.nanstd(num_decoded_i)
					#__Prob
					prob_mean_p[t_i,cut_i] = np.nanmean(prob_decoded_i)
					prob_std_p[t_i,cut_i] = np.nanstd(prob_decoded_i)
				
			#Plot statistics 
			f,ax = plt.subplots(2,3,figsize=(8,8),width_ratios=[10,10,1])
			gs = ax[0,-1].get_gridspec()
			ax[0,0].set_ylim([0,num_neur])
			ax[0,1].set_ylim([0,np.nanmax(len_mean_p) + np.nanmax(len_std_p) + 10])
			ax[1,0].set_ylim([0,np.nanmax(iei_mean_p) + np.nanmax(iei_std_p) + 10])
			ax[1,1].set_ylim([0,1.2])
			for t_i in range(num_tastes):
				#__Num Neur
				ax[0,0].errorbar(neur_cutoffs,num_neur_mean_p[t_i,:],num_neur_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__Length
				ax[0,1].errorbar(neur_cutoffs,len_mean_p[t_i,:],len_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__IEI
				ax[1,0].errorbar(neur_cutoffs,iei_mean_p[t_i,:],iei_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
				#__Prob
				ax[1,1].errorbar(neur_cutoffs,prob_mean_p[t_i,:],prob_std_p[t_i,:],linestyle='None',marker='o',color=taste_colors[t_i,:],alpha=0.8)
			ax[0,0].set_title('Number of Neurons')
			ax[0,1].set_title('Length of Event')
			ax[1,0].set_title('IEI (ms)')
			ax[1,1].set_title('Average P(Decoding)')
			for ax_i in ax[:,-1]:
				ax_i.remove() #remove the underlying axes
			axbig = f.add_subplot(gs[:,-1])
			cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),cax=axbig,ticks=np.linspace(0,1,num_tastes),orientation='vertical')
			cbar.ax.set_yticklabels(dig_in_names)
			#Edit and Save
			f.suptitle('Decoding Statistics by Neuron Cutoff')
			plt.tight_layout()
			f.savefig(os.path.join(seg_decode_save_dir,'neur_cutoff_stats.png'))
			f.savefig(os.path.join(seg_decode_save_dir,'neur_cutoff_stats.svg'))
			plt.close(f)
				
	#Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments	
	sum_width_ratios = np.concatenate((10*np.ones(num_segments),np.ones(1)))
	max_decoding_percent = np.max(epoch_seg_taste_percents)
	f, ax = plt.subplots(num_cp,num_segments + 1,figsize=(10,10),width_ratios=sum_width_ratios)
	gs = ax[0,-1].get_gridspec()
	for e_i in range(num_cp):
		for s_i in range(num_segments):
			ax[e_i,s_i].set_ylim([0,max_decoding_percent+10])
			for t_i in range(num_tastes):
				ax[e_i,s_i].plot(neur_cutoffs,(epoch_seg_taste_percents[:,e_i,s_i,t_i]).flatten(),color=taste_colors[t_i,:],alpha=0.8)
			if s_i == 0:
				ax[e_i,s_i].set_ylabel('Epoch ' + str(e_i))
			if e_i == 0:
				ax[e_i,s_i].set_title(segment_names[s_i])
			if e_i == num_cp-1:
				ax[e_i,s_i].set_xlabel('Neuron Cutoff')
	for ax_i in ax[:,-1]:
		ax_i.remove() #remove the underlying axes
	axbig = f.add_subplot(gs[:,-1])
	cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),cax=axbig,ticks=np.linspace(0,1,num_tastes),orientation='vertical')
	cbar.ax.set_yticklabels(dig_in_names)
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents_By_Neur_Cutoff.png')
	f.savefig(save_dir + 'Decoding_Percents_By_Neur_Cutoff.svg')
	plt.close(f)


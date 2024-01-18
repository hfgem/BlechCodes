#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:12:51 2023

@author: Hannah Germaine
A collection of decoding functions used across analyses.
"""

import numpy as np
import tqdm, os, warnings, itertools, time
file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
os.chdir(file_path)
import matplotlib.pyplot as plt
from scipy import interpolate
from multiprocess import Pool
from p_tqdm import p_map
import decode_parallel as dp

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
	full_taste_fr_dist = np.nan*np.ones((num_tastes,num_neur,max_num_deliv)) #Full taste response firing rate distribution
	tastant_fr_dist = np.nan*np.ones((num_tastes,num_neur,max_num_deliv,num_cp)) #Individual neuron firing rate distributions by epoch
	tastant_fr_dist_pop = np.nan*np.ones((num_tastes,num_neur,max_num_deliv,num_cp)) #Population firing rate distributions by epoch
	max_hz = 0
	max_hz_pop = 0
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
					bst_hz = np.sum(bin_post_taste[start_epoch:end_epoch])/((end_epoch - start_epoch)*(1/1000))
					if bst_hz > max_hz:
						max_hz = bst_hz
					deliv_binned_fr.extend([bst_hz])
					#population changepoints
					start_epoch = int(deliv_cp_pop[cp_i])
					end_epoch = int(deliv_cp_pop[cp_i+1])
					bst_hz = np.sum(bin_post_taste[start_epoch:end_epoch])/((end_epoch - start_epoch)*(1/1000))
					if bst_hz > max_hz_pop:
						max_hz_pop = bst_hz
					deliv_binned_fr_pop.extend([bst_hz])
				del cp_i, start_epoch, end_epoch, bst_hz
				full_taste_fr_dist[t_i,n_i,d_i] = np.sum(bin_post_taste)/(post_taste_dt*(1/1000))
				tastant_fr_dist[t_i,n_i,d_i,:] = deliv_binned_fr
				tastant_fr_dist_pop[t_i,n_i,d_i,:] = deliv_binned_fr_pop
	del t_i, num_deliv, taste_cp, n_i, d_i, raster_times, start_taste_i, deliv_cp, times_post_taste, bin_post_taste

	return full_taste_fr_dist, tastant_fr_dist, tastant_fr_dist_pop, taste_num_deliv
	
def decode_full(full_taste_fr_dist,segment_spike_times,post_taste_dt,
				   skip_dt,dig_in_names,segment_times,segment_names,
				   start_dig_in_times,taste_num_deliv,taste_select,save_dir):
	"""Decode probability of full taste response from sliding bin spiking 
	across segments"""
	#Pull necessary variables
	num_tastes, num_neur, num_deliv = np.shape(full_taste_fr_dist)
	num_segments = len(segment_spike_times)
	max_hz = np.max(full_taste_fr_dist[~np.isnan(full_taste_fr_dist)])
	hist_bins = np.arange(max_hz+1)
	x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
	p_taste = taste_num_deliv/np.sum(taste_num_deliv)
	
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
				full_data = (full_taste_fr_dist[t_i,n_i,:]).flatten()
				full_data = full_data[~np.isnan(full_data)]
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
			all_taste_fr = (full_taste_fr_dist[:,n_i,:]).flatten()
			all_taste_fr = all_taste_fr[~np.isnan(all_taste_fr)]
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
			time_bin_starts = np.arange(seg_start,seg_end - post_taste_dt,skip_dt)
			num_times = len(time_bin_starts)
			seg_decode_prob = np.zeros((num_tastes,num_times))
			for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts)):
				neur_decode_prob = np.nan*np.ones((num_tastes,num_neur))
				for t_i in range(num_tastes):
					for n_i in taste_select_neur:
						neur_spike_fr = len(np.where((np.int64(tb) <= segment_spike_times[s_i][n_i])*(np.int64(tb)+post_taste_dt >= segment_spike_times[s_i][n_i]))[0])/(post_taste_dt*(1/1000))
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
				   taste_select_epoch,use_full,save_dir):
	"""Decode probability of each epoch in high-probability replay regions
	found using decode_phase_1 based on full taste profile decoding.
	Use parallelized function to speed up."""
	#Pull necessary variables
	num_tastes, num_neur, max_num_deliv, num_cp = np.shape(tastant_fr_dist)
	num_segments = len(segment_spike_times)
	max_hz = np.max(tastant_fr_dist[~np.isnan(tastant_fr_dist)])
	hist_bins = np.arange(max_hz+1)
	x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
	p_taste = taste_num_deliv/np.sum(taste_num_deliv)
	
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
					full_data = (tastant_fr_dist[t_i,n_i,:,e_i]).flatten()
					full_data = full_data[~np.isnan(full_data)]
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
			np.save(save_dir + 'fit_tastant_neur_epoch_' + str(e_i) + '.npy',fit_tastant_neur)
		
		#Fit the joint distribution across tastes
		#print("\tFitting joint distribution by neuron")
		try:
			joint_fit_neur = np.load(save_dir + 'joint_fit_neur_epoch_' + str(e_i) + '.npy')
			prev_run[1] = 1
		except:
			joint_fit_neur = np.zeros((num_neur,len(x_vals)+1))
			for n_i in taste_select_neur:
				all_taste_fr = (tastant_fr_dist[:,n_i,:,e_i]).flatten()
				all_taste_fr = all_taste_fr[~np.isnan(all_taste_fr)]
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
		seg_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(seg_decode_save_dir):
			os.mkdir(seg_decode_save_dir)
		for s_i in range(num_segments):
			try:
				seg_decode_epoch_prob = np.load(seg_decode_save_dir + 'segment_' + str(s_i) + '.npy')
				print('\tSegment ' + str(s_i) + ' Previously Decoded')
			except:
				print('\tDecoding Segment ' + str(s_i))
				#Get segment variables
				seg_start = segment_times[s_i]
				seg_end = segment_times[s_i+1]
				seg_len = segment_times[s_i+1] - segment_times[s_i] #in dt = ms
				if use_full == 1:
					#Load previously calculated probabilities and pull start/end times
					time_bin_starts = np.arange(seg_start,seg_end - post_taste_dt,skip_dt)
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
					new_time_bins = np.arange(seg_start,seg_end)
				#Now pull epoch-specific probabilities (only in previously decoded times)
				seg_decode_epoch_prob = np.zeros((num_tastes,seg_len))
				seg_decode_epoch_prob[-1,:] = 1 #Start with assumption of "none" taste at all times
				#Binerize Spike Times
				segment_spike_times_s_i = segment_spike_times[s_i]
				segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
				for n_i in taste_select_neur:
					n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
					segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
				#Perform parallel computation for each time bin
				print('\t\tCalculate firing rates for time bins')
				tb_fr = np.zeros((num_neur,len(new_time_bins)))
				for tb_i,tb in enumerate(tqdm.tqdm(new_time_bins)):
					tb_fr[:,tb_i] = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start:tb+e_len_dt-seg_start],1)/(e_len_dt*(1/1000))
				list_tb_fr = list(tb_fr.T)
				del tb_fr, tb_i, tb
				#Pass inputs to parallel computation on probabilities
				inputs = zip(list_tb_fr, itertools.repeat(num_tastes), \
					 itertools.repeat(num_neur),itertools.repeat(x_vals), \
					 itertools.repeat(fit_tastant_neur), itertools.repeat(joint_fit_neur), \
					 itertools.repeat(p_taste),itertools.repeat(taste_select_neur))
				#___Needs testing / rewriting to be faster (pass only subsets)
				tic = time.time()
				pool = Pool(4)
				tb_decode_prob = pool.map(dp.segment_taste_decode_parallelized, inputs)
				pool.close()
				toc = time.time()
				print('\t\tTime to decode = ' + str(np.round((toc-tic)/60,2)) + ' (min)')
				tb_decode_array = np.squeeze(np.array(tb_decode_prob)).T
				#___
				seg_decode_epoch_prob[:,new_time_bins - seg_start] = tb_decode_array
				
				#Save decoding probabilities
				np.save(seg_decode_save_dir + 'segment_' + str(s_i) + '.npy',seg_decode_epoch_prob)
				#Create plots
				seg_decode_epoch_prob_nonan = np.zeros(np.shape(seg_decode_epoch_prob))
				seg_decode_epoch_prob_nonan[:] = seg_decode_epoch_prob[:]
				seg_decode_epoch_prob_nonan[np.isnan(seg_decode_epoch_prob_nonan)] = 0
				seg_decode_epoch_taste_ind = np.argmax(seg_decode_epoch_prob,0)
				seg_decode_epoch_taste_bin = np.zeros(np.shape(seg_decode_epoch_prob))
				for t_i in range(num_tastes):
					seg_decode_epoch_taste_bin[t_i,np.where(seg_decode_epoch_taste_ind == t_i)[0]] = 1
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
							plt.plot(decode_t_labels,decode_snip.T)
							for t_i_2 in range(num_tastes):
								plt.fill_between(decode_t_labels,decode_snip[t_i_2,:],alpha=0.2)
							plt.axvline(0)
							plt.legend(dig_in_names)
							plt.ylabel('Decoding Fraction')
							plt.xlabel('Time From Delivery (ms)')
							plt.title(dig_in_names[t_i] + ' delivery #' + str(st_i))
							f4.savefig(taste_save_dir + dig_in_names[t_i] + '_' + str(st_i) + '.png')
							f4.savefig(taste_save_dir + dig_in_names[t_i] + '_' + str(st_i) + '.svg')
							plt.close(f4)

def plot_decoded(num_tastes,num_neur,num_cp,segment_spike_times,tastant_spike_times,
				 start_dig_in_times,end_dig_in_times,post_taste_dt,
				 e_skip_dt,e_len_dt,dig_in_names,segment_times,
				 segment_names,taste_num_deliv,taste_select_epoch,
				 use_full,save_dir):
	"""Function to plot the periods when something other than no taste is 
	decoded"""
	num_segments = len(segment_spike_times)
	
	for e_i in range(num_cp): #By epoch conduct decoding
		print('Plotting Decoding for Epoch ' + str(e_i))
		
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		
		seg_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(seg_decode_save_dir):
			os.mkdir(seg_decode_save_dir)
		for s_i in range(num_segments):
			try:
				seg_decode_epoch_prob = np.load(seg_decode_save_dir + 'segment_' + str(s_i) + '.npy')
			except:
				print("Segment " + str(s_i) + " Never Decoded")
				pass
			
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
			decoded_taste_ind = np.argmax(seg_decode_epoch_prob,0)
			decoded_taste_bin = np.zeros((num_tastes,len(decoded_taste_ind)))
			for t_i in range(num_tastes):
				decoded_taste_bin[t_i,np.where(decoded_taste_ind == t_i)[0]] = 1
			decoded_taste_bin[:,0] = 0
			decoded_taste_bin[:,-1] = 0
			
			#For each taste (except none) calculate start and end times of decoded intervals and plot
			for t_i in range(num_tastes - 1):
				#Import taste raster plots
				taste_spike_times = tastant_spike_times[t_i]
				taste_spike_times_bin = np.zeros((len(taste_spike_times),))
				
				#Binarize where the taste was the maximally decoded taste
				#Calculate start and end times for bins of decoding
				start_decoded = np.where(np.diff(decoded_taste_bin[t_i,:]) == 1)[0] + 1
				end_decoded = np.where(np.diff(decoded_taste_bin[t_i,:]) == -1)[0] + 1
				num_decoded = len(start_decoded)
				
				#Create plots of the decoded periods
				taste_decode_save_dir = seg_decode_save_dir + dig_in_names[t_i] + '_events/'
				if not os.path.isdir(taste_decode_save_dir):
					os.mkdir(taste_decode_save_dir)
				
				for d_i in range(num_decoded):
					d_start = np.max((start_decoded[d_i]-post_taste_dt,0))
					d_end = np.min((end_decoded[d_i]+post_taste_dt,seg_len))
					decode_plot_times = np.arange(d_start,d_end)/1000/60
					
					f = plt.figure()
					#Decoding probabilities (NEEDS WORK)
					plt.subplot(3,2,1)
					plt.plot(decode_plot_times,seg_decode_epoch_prob[:,d_start:d_end].T)
					for t_i in range(num_tastes):
						plt.fill_between(decode_plot_times,decoded_taste_bin[t_i,d_start:d_end],alpha=0.2)
					plt.axvline(d_start/1000/60,color='k',alpha=0.5)
					plt.axvline(d_end/1000/60,color='k',alpha=0.5)
					plt.legend(dig_in_names,loc='right')
					plt.ylabel('Decoding Fraction')
					plt.xlabel('Time (min)')
					plt.title('Event ' + str(d_i))
					#Decoded raster
					plt.subplot(2,1)
					plt.eventplot(segment_times_s_i)
					
					
					f.savefig(taste_decode_save_dir + 'event_' + str(d_i) + '.png')
					f.savefig(taste_decode_save_dir + 'event_' + str(d_i) + '.svg')
					plt.close(f)
			

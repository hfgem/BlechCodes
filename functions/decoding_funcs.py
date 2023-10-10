#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:12:51 2023

@author: Hannah Germaine
A collection of decoding functions used across analyses.
"""

import numpy as np
import tqdm, os, warnings
import matplotlib.pyplot as plt
from scipy import interpolate

def taste_decoding_cp(tastant_spike_times,taste_cp_raster_inds,num_cp,start_dig_in_times, \
				   end_dig_in_times,dig_in_names,num_neur,pre_taste_dt, \
				   post_taste_dt,save_dir):
	"""Use Bayesian theory to decode tastes from activity and determine which 
	neurons are taste selective. The functions uses a "leave-one-out" method of
	decoding, where the single delivery to be decoded is left out of the fit 
	distributions in the probability calculations.
	
	Note: the last taste in the 
	tastant_spike_times, etc... variables is always 'none'.
	
	This function uses the changepoint times to bin the spike counts for the 
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
	p_taste_epoch = np.zeros((num_neur, num_tastes, max_num_deliv, num_cp))
	
	print("Now performing leave-one-out calculations of decoding.")
	for d_i_o in tqdm.tqdm(range(total_num_deliv)): #d_i_o is the left out delivery
		
		#Determine the spike count distributions for each neuron for each taste
		print("\tPulling spike count distributions by taste by neuron")
		tastant_binned_hz = [] #taste x neuron x total bins across all deliveries
		tastant_binned_delivery = np.zeros((num_tastes,num_neur,max_num_deliv,num_cp)) #Individual deliveries binned spike counts for later decoding
		max_hz = 0
		for t_i in range(num_tastes):
			num_deliv = taste_num_deliv[t_i]
			taste_cp = taste_cp_raster_inds[t_i]
			tastant_hz = [] #neuron x len(bin_start_times)
			for n_i in range(num_neur):
				neur_hz = [] #len(bin_start_times)
				for d_i in range(num_deliv): #index for that taste
					total_d_i = np.sum(taste_num_deliv[:t_i]) + d_i #what is the index out of all deliveries
					if total_d_i != d_i_o:
						raster_times = tastant_spike_times[t_i][d_i][n_i]
						start_taste_i = start_dig_in_times[t_i][d_i]
						deliv_cp = taste_cp[d_i,n_i,:] - pre_taste_dt
						#Bin the average firing rates following taste delivery start
						times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
						bin_post_taste = np.zeros(post_taste_dt)
						bin_post_taste[times_post_taste] += 1
						deliv_binned_st = []
						for cp_i in range(num_cp):
							start_epoch = int(deliv_cp[cp_i])
							end_epoch = int(deliv_cp[cp_i+1])
							bst_hz = np.sum(bin_post_taste[start_epoch:end_epoch])/((end_epoch - start_epoch)*(1/1000))
							if bst_hz > max_hz:
								max_hz = bst_hz
							deliv_binned_st.extend([bst_hz])
						#del cp_i, start_epoch, end_epoch, bst_hz
						tastant_binned_delivery[t_i,n_i,d_i,:] = deliv_binned_st
						neur_hz.extend(deliv_binned_st)
				tastant_hz.append(neur_hz)
			tastant_binned_hz.append(tastant_hz)
		del t_i, num_deliv, taste_cp, tastant_hz, n_i, neur_hz, d_i, total_d_i, raster_times, start_taste_i, deliv_cp, times_post_taste, bin_post_taste
	
		#Fit the spike count distributions for each neuron for each taste (use gamma distribution) and plot
		print("\tFitting spike count distributions by taste by neuron")
		hist_bins = np.arange(max_hz+1)
		x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
		fit_tastant_neur = np.zeros((num_tastes,num_neur,len(x_vals)+1))
		for t_i in range(num_tastes):
			fig_t, ax_t = plt.subplots(nrows=num_neur,ncols=1,sharex=True,figsize=(5,num_neur))
			for n_i in range(num_neur):
				full_data = tastant_binned_hz[t_i][n_i]
				num_points = len(full_data)
				bin_centers = np.linspace(0,max_hz+1,np.max([8,np.ceil(num_points/5).astype('int')]))
				bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
				bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
				fit_data = ax_t[n_i].hist(tastant_binned_hz[t_i][n_i],density=True,bins=bins_calc)
				new_fit = interpolate.interp1d(bin_centers,fit_data[0],kind='linear')
				filtered_data = new_fit(hist_bins)
				filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
				ax_t[n_i].plot(hist_bins,filtered_data,color='r')
				#fit_tastant_neur[t_i,n_i,:] = fit_data[0]
				fit_tastant_neur[t_i,n_i,:] = filtered_data
			del n_i, full_data, num_points, bin_centers, bins_calc, fit_data, new_fit, filtered_data
			plt.suptitle(dig_in_names[t_i])
			fig_t.tight_layout()
			fig_t.savefig(save_dir + dig_in_names[t_i] + '_spike_count_dist_epoch_lo_' + str(d_i_o) + '.png')
			fig_t.savefig(save_dir + dig_in_names[t_i] + '_spike_count_dist_epoch_lo_' + str(d_i_o) + '.svg')
			plt.close(fig_t)
		
		#Fit the joint distribution across tastes
		print("\tFitting joint distribution by neuron")
		joint_fit_neur = np.zeros((num_neur,len(x_vals)+1))
		fig_t, ax_t = plt.subplots(nrows=num_neur,ncols=1,sharex=True,figsize=(5,num_neur))
		for n_i in range(num_neur):
			all_taste_fr = []
			for t_i in range(num_tastes):
				all_taste_fr.extend(tastant_binned_hz[t_i][n_i])
			bin_centers = np.linspace(0,max_hz+1)
			bins_calc = [bin_centers[0] - np.mean(np.diff(bin_centers))]
			bins_calc.extend(bin_centers + np.mean(np.diff(bin_centers)))
			fit_data = ax_t[n_i].hist(all_taste_fr,bins=bins_calc,density=True)
			new_fit = interpolate.interp1d(bin_centers,fit_data[0])
			filtered_data = new_fit(hist_bins)
			ax_t[n_i].plot(hist_bins,filtered_data,color='r')
			#joint_fit_neur[n_i,:] = fit_data[0]
			joint_fit_neur[n_i,:] = filtered_data
		plt.suptitle('All Tastes Distributions')
		fig_t.tight_layout()
		fig_t.savefig(save_dir + 'all_taste_spike_count_dist_epoch_lo_' + str(d_i_o) + '.png')
		fig_t.savefig(save_dir + 'all_taste_spike_count_dist_epoch_lo_' + str(d_i_o) + '.svg')
		plt.close(fig_t)
		del n_i, all_taste_fr, bin_centers, bins_calc, fit_data, new_fit, filtered_data, fig_t
		
		#Calculate the taste probabilities by neuron by delivery
		print("\tCalculating probability of successful decoding")
		#For each neuron, for each taste and its delivery, determine the probability of each of the tastes being that delivery
		#p(taste|count) = [p(count|taste)xp(taste)]/p(count)
		#tastant_binned_delivery = num_tastes x num_neur x max_num_deliv x num_bins 'count'
		#fit_tastant_neur = num_tastes x num_neur x len(x_vals) 'p(count|taste)'
		#joint_fit_neur = num_neur x len(x_vals) 'p(count)'
		
		#Calculate which taste and delivery d_i_o is:
		t_i = deliv_taste_index[d_i_o]
		d_i = d_i_o - np.cumsum(taste_num_deliv)[t_i]

		p_taste = taste_num_deliv/np.sum(taste_num_deliv)
		for n_i in range(num_neur):
			#Calculate the probability of each taste for each epoch
			p_taste_count = np.zeros((num_tastes,num_cp))
			for cp_i in range(num_cp):
				count = tastant_binned_delivery[t_i,n_i,d_i,cp_i]
				for t_i_2 in range(num_tastes): #compare each taste against the true taste data
					closest_x = np.argmin(np.abs(x_vals - count))
					p_count_taste = fit_tastant_neur[t_i_2,n_i,closest_x]
					p_count = joint_fit_neur[n_i,closest_x]
					p_taste_count[t_i_2,cp_i] = (p_count_taste*p_taste[t_i_2])/p_count
				#Now let's calculate if on an epoch-specific level it decodes 
				#the taste correctly
				if t_i == np.argmax(p_taste_count[:,cp_i]):
					taste_select_success_epoch[cp_i,n_i,t_i,d_i] = 1
			p_taste_epoch[n_i,t_i,d_i,:] = p_taste_count[t_i,:]
			#Since the probability of the taste for each bin is calculated, 
			#now we want the joint probability across bins in this delivery
			#We're going to treat the bins as independent samples for ease
			p_taste_joint = np.prod(p_taste_count,1)
			if t_i == np.argmax(p_taste_joint):
				taste_select_success_joint[n_i,t_i,d_i] = 1
			#del n_i, p_taste_count, cp_i, count, t_i_2, closest_x, p_count_taste, p_count, p_taste_joint
				
	#Now calculate the probability of successfully decoding as the fraction of deliveries successful
	taste_select_prob_joint = np.sum(taste_select_success_joint,axis=2)/taste_num_deliv
	taste_select_prob_epoch = np.sum(taste_select_success_epoch,axis=3)/taste_num_deliv
	
	return taste_select_prob_joint, taste_select_prob_epoch, p_taste_epoch



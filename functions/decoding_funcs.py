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

def taste_decoding_cp(tastant_spike_times,taste_cp_raster_inds,taste_cp_raster_inds_pop, \
					  num_cp,start_dig_in_times,end_dig_in_times,dig_in_names, \
					  num_neur,pre_taste_dt,post_taste_dt,save_dir):
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
	p_taste_joint = np.zeros((num_neur, num_tastes, max_num_deliv))
	
	#print("Now performing leave-one-out calculations of decoding.")
	for d_i_o in tqdm.tqdm(range(total_num_deliv)): #d_i_o is the left out delivery
	
		#Determine the spike count distributions for each neuron for each taste
		#print("\tPulling spike count distributions by taste by neuron")
		tastant_binned_delivery = np.nan*np.ones((num_tastes,num_neur,max_num_deliv,num_cp)) #Individual deliveries binned spike counts for later decoding
		tastant_binned_delivery_pop = np.nan*np.ones((num_tastes,num_neur,max_num_deliv,num_cp))
		max_hz = 0
		max_hz_pop = 0
		for t_i in range(num_tastes):
			num_deliv = taste_num_deliv[t_i]
			taste_cp = taste_cp_raster_inds[t_i]
			taste_cp_pop = taste_cp_raster_inds_pop[t_i]
			for n_i in range(num_neur):
				for d_i in range(num_deliv): #index for that taste
					total_d_i = np.sum(taste_num_deliv[:t_i]) + d_i #what is the index out of all deliveries
					if total_d_i != d_i_o:
						raster_times = tastant_spike_times[t_i][d_i][n_i]
						start_taste_i = start_dig_in_times[t_i][d_i]
						deliv_cp = taste_cp[d_i,n_i,:] - pre_taste_dt
						deliv_cp_pop = taste_cp_pop[d_i,:] - pre_taste_dt
						#Bin the average firing rates following taste delivery start
						times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
						bin_post_taste = np.zeros(post_taste_dt)
						bin_post_taste[times_post_taste] += 1
						deliv_binned_st = []
						deliv_binned_st_pop = []
						for cp_i in range(num_cp):
							#individual neuron changepoints
							start_epoch = int(deliv_cp[cp_i])
							end_epoch = int(deliv_cp[cp_i+1])
							bst_hz = np.sum(bin_post_taste[start_epoch:end_epoch])/((end_epoch - start_epoch)*(1/1000))
							if bst_hz > max_hz:
								max_hz = bst_hz
							deliv_binned_st.extend([bst_hz])
							#population changepoints
							start_epoch = int(deliv_cp_pop[cp_i])
							end_epoch = int(deliv_cp_pop[cp_i+1])
							bst_hz = np.sum(bin_post_taste[start_epoch:end_epoch])/((end_epoch - start_epoch)*(1/1000))
							if bst_hz > max_hz_pop:
								max_hz_pop = bst_hz
							deliv_binned_st_pop.extend([bst_hz])
						del cp_i, start_epoch, end_epoch, bst_hz
						tastant_binned_delivery[t_i,n_i,d_i,:] = deliv_binned_st
						tastant_binned_delivery_pop[t_i,n_i,d_i,:] = deliv_binned_st_pop
		del t_i, num_deliv, taste_cp, n_i, d_i, total_d_i, raster_times, start_taste_i, deliv_cp, times_post_taste, bin_post_taste
	
		#_____Calculate Across Epochs_____
		
		#Fit the spike count distributions for each neuron for each taste (use gamma distribution) and plot
		#print("\tFitting spike count distributions by taste by neuron")
		hist_bins = np.arange(max_hz+1)
		x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
		fit_tastant_neur = np.zeros((num_tastes,num_neur,len(x_vals)+1))
		for t_i in range(num_tastes):
			for n_i in range(num_neur):
				full_data = (tastant_binned_delivery[t_i,n_i,:,:]).flatten()
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
		
		#Plot the taste distributions against each other
		fig_t, ax_t = plt.subplots(nrows=num_neur,ncols=1,sharex=True,figsize=(5,num_neur))
		for n_i in range(num_neur):
			if n_i == 0:	
				ax_t[n_i].plot((fit_tastant_neur[:,n_i,:]).T,label=dig_in_names)
				ax_t[n_i].legend()
			else:
				ax_t[n_i].plot((fit_tastant_neur[:,n_i,:]).T)
		ax_t[num_neur-1].set_xlabel('Firing Rate (Hz)')
		fig_t.supylabel('Probability')
		plt.suptitle('LOO Delivery ' + str(d_i_o) + ' all epochs')
		fig_t.tight_layout()
		fig_t.savefig(save_dir + 'loo_' + str(d_i_o) + '_all_epochs.png')
		fig_t.savefig(save_dir + 'loo_' + str(d_i_o) + '_all_epochs.svg')
		plt.close(fig_t)
		
		#Fit the joint distribution across tastes
		#print("\tFitting joint distribution by neuron")
		joint_fit_neur = np.zeros((num_neur,len(x_vals)+1))
		for n_i in range(num_neur):
			all_taste_fr = (tastant_binned_delivery[:,n_i,:,:]).flatten()
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
		
		#Calculate which taste and delivery d_i_o is:
		t_i = deliv_taste_index[d_i_o]
		if t_i > 0:
			d_i = d_i_o - np.cumsum(taste_num_deliv)[t_i-1]
		else:
			d_i = d_i_o
		
		#Calculate the taste probabilities by neuron by delivery
		#print("\tCalculating probability of successful decoding")
		#For each neuron, for each taste and its delivery, determine the probability of each of the tastes being that delivery
		#p(taste|count) = [p(count|taste)xp(taste)]/p(count)
		#tastant_binned_delivery = num_tastes x num_neur x max_num_deliv x num_bins 'count'
		#fit_tastant_neur = num_tastes x num_neur x len(x_vals) 'p(count|taste)'
		#joint_fit_neur = num_neur x len(x_vals) 'p(count)'
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
			#Since the probability of the taste for each bin is calculated, 
			#now we want the joint probability across bins in this delivery
			#We're going to treat the bins as independent samples for ease
			p_taste_joint[n_i,:,d_i] = np.prod(p_taste_count,1)
			if t_i == np.argmax(p_taste_joint[n_i,:,d_i]):
				taste_select_success_joint[n_i,t_i,d_i] = 1
			del n_i, p_taste_count, cp_i, count, t_i_2, closest_x, p_count_taste, p_count
		
		#_____Calculate Within Epochs_____
	
		for cp_i in range(num_cp):
			#Fit the spike count distributions for each neuron for each taste (use gamma distribution) and plot
			#print("\tFitting spike count distributions by taste by neuron")
			hist_bins = np.arange(max_hz+1)
			x_vals = hist_bins[:-1] + np.diff(hist_bins)/2
			fit_tastant_neur = np.zeros((num_tastes,num_neur,len(x_vals)+1))
			for t_i in range(num_tastes):
				for n_i in range(num_neur):
					full_data = (tastant_binned_delivery_pop[t_i,n_i,:,cp_i]).flatten()
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
			
			#Plot the taste distributions against each other
			fig_t, ax_t = plt.subplots(nrows=num_neur,ncols=1,sharex=True,figsize=(5,num_neur))
			for n_i in range(num_neur): 
				if n_i == 0:	
					ax_t[n_i].plot((fit_tastant_neur[:,n_i,:]).T,label=dig_in_names)
					ax_t[n_i].legend()
				else:
					ax_t[n_i].plot((fit_tastant_neur[:,n_i,:]).T)
			ax_t[num_neur-1].set_xlabel('Firing Rate (Hz)')
			fig_t.supylabel('Probability')
			plt.suptitle('LOO Delivery ' + str(d_i_o) + ' all epochs')
			fig_t.tight_layout()
			fig_t.savefig(save_dir + 'loo_' + str(d_i_o) + '_epoch_' + str(cp_i) + '.png')
			fig_t.savefig(save_dir + 'loo_' + str(d_i_o) + '_epoch_' + str(cp_i) + '.svg')
			plt.close(fig_t)
			
			#Fit the joint distribution across tastes
			#print("\tFitting joint distribution by neuron")
			joint_fit_neur = np.zeros((num_neur,len(x_vals)+1))
			for n_i in range(num_neur):
				all_taste_fr = (tastant_binned_delivery_pop[:,n_i,:,cp_i]).flatten()
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
			
			#Calculate which taste and delivery d_i_o is:
			t_i = deliv_taste_index[d_i_o]
			if t_i > 0:
				d_i = d_i_o - np.cumsum(taste_num_deliv)[t_i-1]
			else:
				d_i = d_i_o
			
			#Calculate the taste probabilities by neuron by delivery
			#print("\tCalculating probability of successful decoding")
			#For each neuron, for each taste and its delivery, determine the probability of each of the tastes being that delivery
			#p(taste|count) = [p(count|taste)xp(taste)]/p(count)
			#tastant_binned_delivery = num_tastes x num_neur x max_num_deliv x num_bins 'count'
			#fit_tastant_neur = num_tastes x num_neur x len(x_vals) 'p(count|taste)'
			#joint_fit_neur = num_neur x len(x_vals) 'p(count)'
			p_taste = taste_num_deliv/np.sum(taste_num_deliv)
			for n_i in range(num_neur):
				#Calculate the probability of each taste for each epoch
				p_taste_count = np.zeros(num_tastes)
				count = tastant_binned_delivery[t_i,n_i,d_i,cp_i]
				for t_i_2 in range(num_tastes): #compare each taste against the true taste data
					closest_x = np.argmin(np.abs(x_vals - count))
					p_count_taste = fit_tastant_neur[t_i_2,n_i,closest_x]
					p_count = joint_fit_neur[n_i,closest_x]
					p_taste_count[t_i_2] = (p_count_taste*p_taste[t_i_2])/p_count
				#Since the probability of the taste for each bin is calculated, 
				#now we want the joint probability across bins in this delivery
				#We're going to treat the bins as independent samples for ease
				p_taste_epoch[n_i,:,d_i,cp_i] = p_taste_count
				if t_i == np.argmax(p_taste_count):
					taste_select_success_epoch[cp_i,n_i,t_i,d_i] = 1
				del n_i, p_taste_count, count, t_i_2, closest_x, p_count_taste, p_count
			
	#Now calculate the probability of successfully decoding as the fraction of deliveries successful
	taste_select_prob_joint = np.sum(taste_select_success_joint,axis=2)/taste_num_deliv
	taste_select_prob_epoch = np.sum(taste_select_success_epoch,axis=3)/taste_num_deliv
	
	return taste_select_prob_joint, taste_select_prob_epoch, p_taste_epoch



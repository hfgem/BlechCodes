#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:12:51 2023

@author: Hannah Germaine
A collection of decoding functions used across analyses.
"""

import numpy as np
import tqdm, os, warnings, itertools, time, random, json
#file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
#os.chdir(file_path)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate
import multiprocess
import functions.decode_parallel as dp
from scipy.stats import pearsonr
from random import sample

def taste_decoding_cp(tastant_spike_times,pop_taste_cp_raster_inds, \
					 start_dig_in_times,end_dig_in_times,dig_in_names, \
					 num_neur,num_cp,num_tastes,pre_taste_dt,post_taste_dt,save_dir):
	"""Use Bayesian theory to decode tastes from activity and determine which 
	neurons are taste selective. The functions uses a "leave-one-out" method of
	decoding, where the single delivery to be decoded is left out of the fit 
	distributions in the probability calculations.
	
	Note: the last taste in the 
	tastant_spike_times, etc... variables is always 'none'.
	
	This function uses the population changepoint times to bin the spike counts for the 
	distribution."""
	warnings.filterwarnings('ignore')
	
	max_num_deliv = 0 #Find the maximum number of deliveries across tastants
	taste_num_deliv = np.zeros(num_tastes).astype('int')
	deliv_taste_index = []
	for t_i in range(num_tastes): #Only perform on actual tastes
		num_deliv = len(tastant_spike_times[t_i])
		deliv_taste_index.extend(list((t_i*np.ones(num_deliv)).astype('int')))
		taste_num_deliv[t_i] = num_deliv
		if num_deliv > max_num_deliv:
			max_num_deliv = num_deliv
	total_num_deliv = np.sum(taste_num_deliv)
	del t_i, num_deliv
	
	#Determine the firing rate distributions for each neuron for each taste for each cp for each delivery
	#To use in LOO Decoding
	try:
		with open(os.path.join(save_dir,'tastant_epoch_delivery.json'), 'r') as fp:
			tastant_epoch_delivery = json.load(fp)
		print("\tImported previously calculated firing rate distributions")
		max_hz_cp = 0
		for cp_i in range(num_cp):
			for t_i in range(num_tastes):
				for n_i in range(num_neur):
					for d_i in range(max_num_deliv):
						if np.nanmax(tastant_epoch_delivery[str(cp_i)][str(t_i)][str(n_i)][str(d_i)]['full']) > max_hz_cp:
							max_hz_cp = np.nanmax(tastant_epoch_delivery[str(cp_i)][str(t_i)][str(n_i)][str(d_i)]['full'])
	except:
		print("\tNow calculating firing rate distributions")
		tastant_epoch_delivery, max_hz_cp = fr_dist_calculator(num_cp,num_tastes,num_neur,max_num_deliv,taste_num_deliv,
									pop_taste_cp_raster_inds,tastant_spike_times,start_dig_in_times,
									pre_taste_dt,post_taste_dt)
		#Save the dictionary of firing rates so you don't have to calculate it again in the future
		with open(os.path.join(save_dir,'tastant_epoch_delivery.json'), 'w') as fp:
			json.dump(tastant_epoch_delivery, fp)
		#Reload the data
		with open(os.path.join(save_dir,'tastant_epoch_delivery.json'), 'r') as fp:
			tastant_epoch_delivery = json.load(fp)
	
	max_hz_cp = np.ceil(max_hz_cp).astype('int')
	#Perform "Leave-One-Out" decoding: one delivery is left out of the distributions
	#and then "decoded" probabilistically based on the distribution formed by 
	#the other deliveries
	taste_select_success_epoch = np.zeros((num_cp, num_neur, num_tastes, max_num_deliv)) #mark with a 1 if successfully decoded
	p_taste_epoch = np.zeros((num_neur, num_tastes, max_num_deliv, num_cp)) #by epoch
	loo_distribution_save_dir = save_dir + 'LOO_Distributions/'
	if os.path.isdir(loo_distribution_save_dir) == False:
		os.mkdir(loo_distribution_save_dir)
	print("\tNow performing leave-one-out calculations of decoding.")
	for d_i_o in tqdm.tqdm(range(total_num_deliv)): #d_i_o is the left out delivery
			
		#_____Calculate By Epoch_____
		hist_bins_cp = np.arange(0,max_hz_cp+1).astype('int')
		x_vals_cp = hist_bins_cp[:-1] + np.diff(hist_bins_cp)/2
		
		d_i, t_i, p_taste_fr_cp_neur, taste_success_fr_cp_neur = loo_epoch_decode(num_cp,num_tastes,num_neur,max_num_deliv,
							tastant_epoch_delivery,max_hz_cp,x_vals_cp,hist_bins_cp,dig_in_names[:-1],d_i_o,
							loo_distribution_save_dir,deliv_taste_index,taste_num_deliv)
		p_taste_epoch[:,:,d_i,:] = p_taste_fr_cp_neur
		taste_select_success_epoch[:,:,t_i,d_i] = taste_success_fr_cp_neur
		
		del hist_bins_cp, x_vals_cp, d_i, t_i, p_taste_fr_cp_neur, taste_success_fr_cp_neur
	print('\n')
	#Now calculate the probability of successfully decoding as the fraction of deliveries successful
	taste_select_prob_epoch = np.sum(taste_select_success_epoch,axis=3)/taste_num_deliv #num cp x num neur x num tastes
	
	return p_taste_epoch, taste_select_prob_epoch

def fr_dist_calculator(num_cp,num_tastes,num_neur,max_num_deliv,taste_num_deliv,
							pop_taste_cp_raster_inds,tastant_spike_times,start_dig_in_times,
							pre_taste_dt,post_taste_dt):
	"""Calculates firing rate distributions for all conditions to use in LOO decoding"""
	tastant_epoch_delivery = dict() #Create a nested dictionary for storage of firing rates
	for cp_i in range(num_cp):
		tastant_epoch_delivery[cp_i] = dict()
		for t_i in range(num_tastes):
			tastant_epoch_delivery[cp_i][t_i] = dict()
			for n_i in range(num_neur):
				tastant_epoch_delivery[cp_i][t_i][n_i] = dict()
				for d_i in range(max_num_deliv):
					tastant_epoch_delivery[cp_i][t_i][n_i][d_i] = dict()
	max_hz_cp = 0
	for cp_i in range(num_cp):
		for t_i in range(num_tastes):
			print('\t\tChangepoint ' + str(cp_i) + " Taste " + str(t_i))
			#Grab taste-related variables
			#taste_d_i = np.sum(taste_num_deliv[:t_i])
			#num_deliv = taste_num_deliv[t_i]
			taste_cp_pop = pop_taste_cp_raster_inds[t_i]
			t_i_spike_times = tastant_spike_times[t_i]
			t_i_dig_in_times = start_dig_in_times[t_i]
			for n_i in tqdm.tqdm(range(num_neur)):
				for d_i in range(max_num_deliv):
					raster_times = t_i_spike_times[d_i][n_i]
					start_taste_i = t_i_dig_in_times[d_i]
					deliv_cp_pop = taste_cp_pop[d_i,:] - pre_taste_dt
					#Binerize the firing following taste delivery start
					times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
					bin_post_taste = np.zeros(post_taste_dt)
					bin_post_taste[times_post_taste] += 1
					#Grab changepoints
					start_epoch = int(deliv_cp_pop[cp_i])
					end_epoch = int(deliv_cp_pop[cp_i+1])
					epoch_len = end_epoch - start_epoch
					#Set bin sizes to use in breaking up the epoch for firing rates
					bin_sizes = np.arange(50,epoch_len,50) #50 ms increments up to full epoch size
					if bin_sizes[-1] != epoch_len:
						bin_sizes = np.concatenate((bin_sizes,epoch_len*np.ones(1)))
					bin_sizes = bin_sizes.astype('int')
					all_my_fr = []
					#Get firing rates for each bin size and concatenate
					for b_i in bin_sizes:
						all_my_fr.extend([np.sum(bin_post_taste[i:i+b_i])/(b_i/1000) for i in range(end_epoch-b_i+1)])
					tastant_epoch_delivery[cp_i][t_i][n_i][d_i]['full'] = all_my_fr
					#Get singular firing rate for the entire epoch for decoding
					tastant_epoch_delivery[cp_i][t_i][n_i][d_i]['true'] = np.sum(bin_post_taste[start_epoch:end_epoch])/(epoch_len/1000)
					#Calculate and update maximum firing rate
					if np.nanmax(all_my_fr) > max_hz_cp:
						max_hz_cp = np.nanmax(all_my_fr)
	
	return tastant_epoch_delivery, max_hz_cp

def loo_epoch_decode(num_cp,num_tastes,num_neur,max_num_deliv,tastant_epoch_delivery,
					max_hz_cp,x_vals_cp,hist_bins_cp,dig_in_names,d_i_o,
					save_dir,deliv_taste_index,taste_num_deliv):
	
	p_taste_fr_cp_neur = np.zeros((num_neur,num_tastes,num_cp))
	taste_success_fr_cp_neur = np.zeros((num_cp,num_neur))
	
	#Calculate which taste and delivery d_i_o is:
	t_i_true = deliv_taste_index[d_i_o]
	if t_i_true > 0:
		d_i_true = d_i_o - np.cumsum(taste_num_deliv)[t_i_true-1]
	else:
		d_i_true = d_i_o
	
	for cp_i in range(num_cp):
		#Fit the firing rate distributions for each neuron for each taste (use gamma distribution) and plot
		#print("\tFitting firing rate distributions by taste by neuron")
		p_fr_taste = np.zeros((num_tastes,num_neur,len(hist_bins_cp)))
		for t_i in range(num_tastes):
			for n_i in range(num_neur):
				full_data = []
				for d_i in range(max_num_deliv):
					if t_i == t_i_true:
						if d_i != d_i_true:
							d_i_data = np.array(tastant_epoch_delivery[str(cp_i)][str(t_i)][str(n_i)][str(d_i)]['full'])
							d_i_data = d_i_data[~np.isnan(d_i_data)]
							if len(d_i_data) > 0:
								full_data.extend(list(d_i_data))
					else:
						d_i_data = np.array(tastant_epoch_delivery[str(cp_i)][str(t_i)][str(n_i)][str(d_i)]['full'])
						d_i_data = d_i_data[~np.isnan(d_i_data)]
						if len(d_i_data) > 0:
							full_data.extend(list(d_i_data))
				full_data_array = np.array(full_data)
				max_fr_data = np.nanmax(full_data_array)
				bin_centers = np.concatenate((np.linspace(0,np.max(max_fr_data),20),(max_hz_cp+1)*np.ones(1)))
				bin_width = np.diff(bin_centers)[0]
				bin_edges = np.concatenate((bin_centers - bin_width,(bin_centers[-1] + np.diff(bin_centers)[-1])*np.ones(1)))
				fit_data = np.histogram(full_data_array,density=True,bins=bin_edges)
				new_fit = interpolate.interp1d(bin_centers,fit_data[0],kind='linear')
				filtered_data = new_fit(hist_bins_cp)
				filtered_data = filtered_data/np.sum(filtered_data) #return to a probability density
				p_fr_taste[t_i,n_i,:] = filtered_data
			del n_i, full_data, bin_centers, bin_edges, fit_data, new_fit, filtered_data
		
		if d_i_o == 0:
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
			plt.suptitle('LOO Delivery ' + str(d_i_o))
			fig_t.tight_layout()
			fig_t.savefig(save_dir + 'loo_' + str(d_i_o) + '_epoch_' + str(cp_i) + '.png')
			fig_t.savefig(save_dir + 'loo_' + str(d_i_o) + '_epoch_' + str(cp_i) + '.svg')
			plt.close(fig_t)
		
		#Calculate the taste probabilities by neuron by delivery
		#print("\tCalculating probability of successful decoding")
		#For each neuron, for each taste and its delivery, determine the probability of each of the tastes being that delivery
		#p(taste|fr) = [p(fr|taste)xp(taste)]/p(fr)
		loo_taste_num_deliv = np.zeros(np.shape(taste_num_deliv))
		loo_taste_num_deliv[:] = taste_num_deliv[:]
		loo_taste_num_deliv[t_i_true] -= 1
		p_taste = loo_taste_num_deliv/np.sum(loo_taste_num_deliv)
		for n_i in range(num_neur):
			#Calculate the probability of each taste for each epoch
			fr = tastant_epoch_delivery[str(cp_i)][str(t_i_true)][str(n_i)][str(d_i_true)]['true']
			for t_i_2 in range(num_tastes): #compare each taste against the true taste data
				closest_x = np.argmin(np.abs(x_vals_cp - fr))
				p_fr = np.nansum(np.squeeze(p_fr_taste[:,n_i,closest_x]))/num_tastes
				p_taste_fr_cp_neur[n_i,t_i_2,cp_i] = (p_fr_taste[t_i_2,n_i,closest_x]*p_taste[t_i_2])/p_fr
			#Since the probability of each taste is calculated, now we determine
			#	if the highest probability taste aligns with the truly delivered taste
			if t_i_true == np.argmax(p_taste_fr_cp_neur[n_i,:,cp_i]):
				taste_success_fr_cp_neur[cp_i,n_i] = 1
				
	return d_i_true, t_i_true, p_taste_fr_cp_neur, taste_success_fr_cp_neur


def taste_fr_dist(num_neur,num_cp,tastant_spike_times,
				 taste_cp_raster_inds,pop_taste_cp_raster_inds,
				start_dig_in_times, pre_taste_dt, post_taste_dt):
	"""This function calculates fr distributions for each neuron for
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
	tastant_fr_dist_pop = dict() #Population firing rate distributions by epoch
	for t_i in range(num_tastes):
		tastant_fr_dist_pop[t_i] = dict()
		for n_i in range(num_neur):
			tastant_fr_dist_pop[t_i][n_i] = dict()
			for d_i in range(max_num_deliv):
				tastant_fr_dist_pop[t_i][n_i][d_i] = dict()
				for cp_i in range(num_cp):
					tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = dict()
	#____
	max_hz_pop = 0
	for t_i in range(num_tastes):
		num_deliv = taste_num_deliv[t_i]
		taste_cp_pop = pop_taste_cp_raster_inds[t_i]
		for n_i in range(num_neur):
			for d_i in range(num_deliv): #index for that taste
				raster_times = tastant_spike_times[t_i][d_i][n_i]
				start_taste_i = start_dig_in_times[t_i][d_i]
				deliv_cp_pop = taste_cp_pop[d_i,:] - pre_taste_dt
				#Bin the average firing rates following taste delivery start
				times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
				bin_post_taste = np.zeros(post_taste_dt)
				bin_post_taste[times_post_taste] += 1
				for cp_i in range(num_cp):
					#population changepoints
					start_epoch = int(deliv_cp_pop[cp_i])
					end_epoch = int(deliv_cp_pop[cp_i+1])
					epoch_len = end_epoch - start_epoch
					#Set bin sizes to use in breaking up the epoch for firing rates
					bin_sizes = np.arange(50,epoch_len,50) #50 ms increments up to full epoch size
					if bin_sizes[-1] != epoch_len:
						bin_sizes = np.concatenate((bin_sizes,epoch_len*np.ones(1)))
					bin_sizes = bin_sizes.astype('int')
					all_hz_bst = []
					#Get firing rates for each bin size and concatenate
					for b_i in bin_sizes:
						all_hz_bst.extend([np.sum(bin_post_taste[i:i+b_i])/(b_i/1000) for i in range(end_epoch-b_i+1)])
					tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = all_hz_bst
					if np.nanmax(all_hz_bst) > max_hz_pop:
						max_hz_pop = np.nanmax(all_hz_bst)
				del cp_i, start_epoch, end_epoch
				#___
	del t_i, num_deliv, n_i, d_i, raster_times, start_taste_i, times_post_taste, bin_post_taste

	return tastant_fr_dist_pop, max_hz_pop, taste_num_deliv
	

def taste_fr_dist_zscore(num_neur,num_cp,tastant_spike_times,segment_spike_times,
				segment_names,segment_times,taste_cp_raster_inds,pop_taste_cp_raster_inds,
				start_dig_in_times,pre_taste_dt,post_taste_dt,bin_dt):
	"""This function calculates z-scored firing rate distributions for each neuron for
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
	tastant_fr_dist_pop = dict() #Population firing rate distributions by epoch
	for t_i in range(num_tastes):
		tastant_fr_dist_pop[t_i] = dict()
		for n_i in range(num_neur):
			tastant_fr_dist_pop[t_i][n_i] = dict()
			for d_i in range(max_num_deliv):
				tastant_fr_dist_pop[t_i][n_i][d_i] = dict()
				for cp_i in range(num_cp):
					tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = dict()
	#____
	max_hz_pop = 0
	min_hz_pop = 0
	for t_i in range(num_tastes):
		num_deliv = taste_num_deliv[t_i]
		taste_cp_pop = pop_taste_cp_raster_inds[t_i]
		for n_i in range(num_neur):
			for d_i in range(num_deliv): #index for that taste
				raster_times = tastant_spike_times[t_i][d_i][n_i]
				start_taste_i = start_dig_in_times[t_i][d_i]
				deliv_cp_pop = taste_cp_pop[d_i,:] - pre_taste_dt
				#Bin the average firing rates following taste delivery start
				times_post_taste = (np.array(raster_times)[np.where((raster_times >= start_taste_i)*(raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
				bin_post_taste = np.zeros(post_taste_dt)
				bin_post_taste[times_post_taste] += 1
				for cp_i in range(num_cp):
					#population changepoints
					start_epoch = int(deliv_cp_pop[cp_i])
					end_epoch = int(deliv_cp_pop[cp_i+1])
					epoch_len = end_epoch - start_epoch
					#Set bin sizes to use in breaking up the epoch for firing rates
					bin_sizes = np.arange(50,epoch_len,50) #50 ms increments up to full epoch size
					if bin_sizes[-1] != epoch_len:
						bin_sizes = np.concatenate((bin_sizes,epoch_len*np.ones(1)))
					bin_sizes = bin_sizes.astype('int')
					all_my_fr = []
					#Get firing rates for each bin size and concatenate
					for b_i in bin_sizes:
						bst_hz = [np.sum(bin_post_taste[i:i+b_i])/(b_i/1000) for i in range(end_epoch-b_i+1)]
						bst_hz_z = (np.array(bst_hz) - mean_fr[n_i])/std_fr[n_i]
						all_my_fr.extend(list(bst_hz_z))
					tastant_fr_dist_pop[t_i][n_i][d_i][cp_i] = all_my_fr
					if np.nanmax(bst_hz_z) > max_hz_pop:
						max_hz_pop = np.nanmax(bst_hz_z)
					if np.nanmin(bst_hz_z) < min_hz_pop:
						min_hz_pop = np.nanmin(bst_hz_z)					
				del cp_i, start_epoch, end_epoch, bst_hz, bst_hz_z
				
	del t_i, num_deliv, n_i, d_i, raster_times, start_taste_i, times_post_taste, bin_post_taste

	return tastant_fr_dist_pop, max_hz_pop, min_hz_pop, taste_num_deliv


def plot_decoded(fr_dist,num_tastes,num_neur,segment_spike_times,tastant_spike_times,
				start_dig_in_times,end_dig_in_times,post_taste_dt,pre_taste_dt,
				cp_raster_inds,bin_dt,dig_in_names,segment_times,
				segment_names,taste_num_deliv,taste_select_epoch,
				save_dir,max_decode,max_hz,seg_stat_bin,
				neuron_count_thresh,trial_start_frac=0,
				epochs_to_analyze=[],segments_to_analyze=[],
				decode_prob_cutoff=0.95):
	"""Function to plot the periods when something other than no taste is 
	decoded"""
	num_cp = np.shape(cp_raster_inds[0])[-1] - 1
	num_segments = len(segment_spike_times)
	neur_cut = np.floor(num_neur*neuron_count_thresh).astype('int')
	taste_colors = cm.brg(np.linspace(0,1,num_tastes))
	epoch_seg_taste_percents = np.zeros((num_cp,num_segments,num_tastes))
	epoch_seg_taste_percents_neur_cut = np.zeros((num_cp,num_segments,num_tastes))
	epoch_seg_taste_percents_best = np.zeros((num_cp,num_segments,num_tastes))
	half_bin_z_dt = np.floor(bin_dt/2).astype('int')
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(num_segments)
	
	#Get taste segment z-score info
	s_i_taste = np.nan*np.ones(1)
	for s_i in range(len(segment_names)):
		if segment_names[s_i].lower() == 'taste':
			s_i_taste[0] = s_i
	if not np.isnan(s_i_taste[0]):
		s_i = int(s_i_taste[0])
		seg_start = segment_times[s_i]
		seg_end = segment_times[s_i+1]
		seg_len = seg_end - seg_start
		time_bin_starts = np.arange(seg_start+half_bin_z_dt,seg_end-half_bin_z_dt,half_bin_z_dt*2)
		segment_spike_times_s_i = segment_spike_times[s_i]
		segment_spike_times_s_i_bin = np.zeros((num_neur,seg_len+1))
		for n_i in range(num_neur):
			n_i_spike_times = np.array(segment_spike_times_s_i[n_i] - seg_start).astype('int')
			segment_spike_times_s_i_bin[n_i,n_i_spike_times] = 1
		tb_fr = np.zeros((num_neur,len(time_bin_starts)))
		for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts)):
			tb_fr[:,tb_i] = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin_z_dt:tb+half_bin_z_dt-seg_start],1)/(2*half_bin_z_dt*(1/1000))
		mean_fr_taste = np.mean(tb_fr,1)
		std_fr_taste = np.std(tb_fr,1)
		std_fr_taste[std_fr_taste == 0] = 1 #to avoid nan calculations
	else:
		mean_fr_taste = np.zeros(num_neur)
		std_fr_taste = np.ones(num_neur)
	
	#for e_i in range(num_cp): #By epoch conduct decoding
	for e_i in epochs_to_analyze:
		print('Plotting Decoding for Epoch ' + str(e_i))
		
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			print("Data not previously decoded, or passed directory incorrect.")
			pass
		
		for s_i in tqdm.tqdm(segments_to_analyze):
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
			
			#Z-score the segment
			time_bin_starts = np.arange(seg_start+half_bin_z_dt,seg_end-half_bin_z_dt,half_bin_z_dt*2)
			tb_fr = np.zeros((num_neur,len(time_bin_starts)))
			for tb_i,tb in enumerate(tqdm.tqdm(time_bin_starts)):
				tb_fr[:,tb_i] = np.sum(segment_spike_times_s_i_bin[:,tb-seg_start-half_bin_z_dt:tb+half_bin_z_dt-seg_start],1)/(2*half_bin_z_dt*(1/1000))
			mean_fr = np.mean(tb_fr,1)
			std_fr = np.std(tb_fr,1)
			std_fr[std_fr == 0] = 1
			
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
			all_taste_fr_vecs_z = []
			all_taste_fr_vecs_mean = np.zeros((num_tastes,num_neur))
			all_taste_fr_vecs_mean_z = np.zeros((num_tastes,num_neur))
			#Grab taste firing rate vectors
			for t_i in range(num_tastes):
				#Import taste spike and cp times
				taste_spike_times = tastant_spike_times[t_i]
				taste_deliv_times = start_dig_in_times[t_i]
				max_num_deliv = len(taste_deliv_times)
				cp_times = cp_raster_inds[t_i]
				
				#If trial_start_frac > 0 use only trials after that threshold
				trial_start_ind = np.floor(max_num_deliv*trial_start_frac).astype('int')
				new_max_num_deliv = (max_num_deliv - trial_start_ind).astype('int')
				
				#Store as binary spike arrays
				taste_spike_times_bin = np.zeros((new_max_num_deliv,num_neur,post_taste_dt)) #Taste response spike times
				taste_cp_times = np.zeros((new_max_num_deliv,num_cp+1)).astype('int')
				taste_epoch_fr_vecs = np.zeros((new_max_num_deliv,num_neur)) #original firing rate vecs
				taste_epoch_fr_vecs_z = np.zeros((new_max_num_deliv,num_neur)) #z-scored firing rate vecs
				for d_i in range(len(taste_spike_times)): #store each delivery to binary spike matrix
					if d_i >= trial_start_ind:	
						pre_taste_spike_times_bin = np.zeros((num_neur,pre_taste_dt)) #Pre-taste spike times
						taste_deliv_i = taste_deliv_times[d_i]
						for n_i in range(num_neur):
							spikes_deliv_i = taste_spike_times[d_i][n_i]
							if t_i == num_tastes-1:
								if len(taste_spike_times[d_i-trial_start_ind][n_i]) > 0:
									d_i_spikes = np.array(spikes_deliv_i - (np.min(spikes_deliv_i)+pre_taste_dt)).astype('int')
								else:
									d_i_spikes = np.empty(0)
							else:
								d_i_spikes = np.array(spikes_deliv_i - taste_deliv_i).astype('int')
							d_i_spikes_posttaste = d_i_spikes[(d_i_spikes<post_taste_dt)*(d_i_spikes>=0)]
							d_i_spikes_pretaste = d_i_spikes[d_i_spikes<0] + pre_taste_dt
							if len(d_i_spikes_posttaste) > 0:
								taste_spike_times_bin[d_i-trial_start_ind,n_i,d_i_spikes_posttaste] = 1
							if len(d_i_spikes_pretaste) > 0:
								pre_taste_spike_times_bin[n_i,d_i_spikes_pretaste] = 1
						taste_cp_times[d_i-trial_start_ind,:] = np.concatenate((np.zeros(1),np.cumsum(np.diff(cp_times[d_i,:])))).astype('int')
						#Calculate the FR vectors by epoch for each taste response and the average FR vector
						epoch_len_i = (taste_cp_times[d_i,e_i+1]-taste_cp_times[d_i,e_i])/1000 
						if epoch_len_i == 0:
							taste_epoch_fr_vecs[d_i-trial_start_ind,:] = np.zeros(num_neur)
						else:
							taste_epoch_fr_vecs[d_i-trial_start_ind,:] = np.sum(taste_spike_times_bin[d_i-trial_start_ind,:,taste_cp_times[d_i,e_i]:taste_cp_times[d_i,e_i+1]],1)/epoch_len_i #FR in HZ
						#Calculate z-scored FR vector
						taste_epoch_fr_vecs_z[d_i-trial_start_ind,:] = (taste_epoch_fr_vecs[d_i-trial_start_ind,:].flatten() - mean_fr_taste)/std_fr_taste
						
				all_taste_fr_vecs.append(taste_epoch_fr_vecs)
				all_taste_fr_vecs_z.append(taste_epoch_fr_vecs_z)
				#Calculate average taste fr vec
				taste_fr_vecs_mean = np.nanmean(taste_epoch_fr_vecs,0)
				taste_fr_vecs_z_mean = np.nanmean(taste_epoch_fr_vecs_z,0)
				all_taste_fr_vecs_mean[t_i,:] = taste_fr_vecs_mean
				all_taste_fr_vecs_mean_z[t_i,:] = taste_fr_vecs_z_mean
				#taste_fr_vecs_max_hz = np.max(taste_epoch_fr_vecs)
			
			#Now look at decoded events
			all_taste_event_fr_vecs = []
			all_taste_event_fr_vecs_z = []
			all_taste_event_fr_vecs_neur_cut = []
			all_taste_event_fr_vecs_z_neur_cut = []
			all_taste_event_fr_vecs_best = []
			all_taste_event_fr_vecs_z_best = []
			for t_i in range(num_tastes):
				taste_decode_save_dir = seg_decode_save_dir + dig_in_names[t_i] + '_events/'
				if not os.path.isdir(taste_decode_save_dir):
					os.mkdir(taste_decode_save_dir)
				#First calculate neurons decoded in all decoded intervals
				decoded_taste = decoded_taste_bin[t_i,:]
				decoded_taste[0] = 0
				decoded_taste[-1] = 0
				decoded_taste_prob = seg_decode_epoch_prob[t_i,:]
				decoded_taste[decoded_taste_prob<decode_prob_cutoff] = 0
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
				
				#Save the percent taste decoded matching threshold
				epoch_seg_taste_percents[e_i,s_i,t_i] = (np.sum(decoded_taste)/len(decoded_taste))*100	
				
				#____Create plots of decoded period statistics____
				seg_dist_starts = np.arange(0,seg_len,seg_stat_bin)
				seg_dist_midbin = seg_dist_starts[:-1] + np.diff(seg_dist_starts)/2
				
				#________All Decoded________
				num_decoded = len(start_decoded)
				prob_decoded = prob_decoded
				len_decoded = np.array(end_decoded-start_decoded)
				iei_decoded = np.array(start_decoded[1:] - end_decoded[:-1])
				
				seg_distribution = np.zeros(len(seg_dist_starts)-1)
				prob_distribution = np.zeros(len(seg_dist_starts)-1)
				for sd_i in range(len(seg_dist_starts)-1):
					bin_events = np.where((start_decoded < seg_dist_starts[sd_i+1])*(start_decoded >= seg_dist_starts[sd_i]))[0]
					seg_distribution[sd_i] = len(bin_events)
					prob_distribution[sd_i] = np.mean(prob_decoded[bin_events])
				
				save_name = 'all_events'
				plot_overall_decoded_stats(len_decoded,iei_decoded,num_neur_decoded,
											 prob_decoded,prob_distribution,e_i,s_i,
											 seg_dist_midbin,seg_distribution,seg_stat_bin,
											 seg_len,save_name,taste_decode_save_dir)
				
				#________Neuron Cutoff Decoded________
				#Now cut at threshold and only keep matching decoded intervals
				decode_ind = np.where(num_neur_decoded >= neur_cut)[0]
				decoded_bin = np.zeros(np.shape(decoded_taste))
				for db in decode_ind:
					s_db = start_decoded[db]
					e_db = end_decoded[db]
					decoded_bin[s_db:e_db] = 1
				#Grab overall percentages
				epoch_seg_taste_percents_neur_cut[e_i,s_i,t_i] = (np.sum(decoded_bin)/len(decoded_bin))*100
				#Re-calculate start and end times of the decoded intervals
				start_decoded_neur_cut = start_decoded[decode_ind]
				end_decoded_neur_cut = end_decoded[decode_ind]
				#Re-calculate the decoded statistics
				num_decoded_neur_cut = len(start_decoded_neur_cut)
				num_neur_decoded_neur_cut = num_neur_decoded[decode_ind]
				prob_decoded_neur_cut = prob_decoded[decode_ind]
				len_decoded_neur_cut = np.array(end_decoded_neur_cut-start_decoded_neur_cut)
				iei_decoded_neur_cut = np.array(start_decoded_neur_cut[1:] - end_decoded_neur_cut[:-1])
				
				seg_distribution_neur_cut = np.zeros(len(seg_dist_starts)-1)
				prob_distribution_neur_cut = np.zeros(len(seg_dist_starts)-1)
				for sd_i in range(len(seg_dist_starts)-1):
					bin_events = np.where((start_decoded_neur_cut < seg_dist_starts[sd_i+1])*(start_decoded_neur_cut >= seg_dist_starts[sd_i]))[0]
					seg_distribution_neur_cut[sd_i] = len(bin_events)
					prob_distribution_neur_cut[sd_i] = np.mean(prob_decoded_neur_cut[bin_events])
				
				
				#Plot the statistics for those events meeting the minimum neuron cutoff
				save_name = 'neur_cutoff_events'
				plot_overall_decoded_stats(len_decoded_neur_cut,iei_decoded_neur_cut,num_neur_decoded_neur_cut,
											 prob_decoded_neur_cut,prob_distribution_neur_cut,e_i,s_i,
											 seg_dist_midbin,seg_distribution_neur_cut,seg_stat_bin,
											 seg_len,save_name,taste_decode_save_dir)
					
				#________Best Decoded________
				#Calculate correlation data for each decode
				decoded_corr = np.zeros((num_decoded,num_tastes))
				decoded_z_corr = np.zeros((num_decoded,num_tastes))
				decoded_fr_vecs = [] #Store all decoded events firing rates
				decoded_z_fr_vecs = [] #Store all decoded events z-scored firing rates
				for nd_i in range(num_decoded):
					#Grab decoded data
					d_start = start_decoded[nd_i]
					d_end = end_decoded[nd_i]
					d_len = d_end-d_start
					d_fr_vec = np.sum(segment_spike_times_s_i_bin[:,d_start:d_end],1)/(d_len/1000)
					decoded_fr_vecs.append(d_fr_vec)
					#Grab z-scored decoded data
					d_fr_vec_z = (d_fr_vec-mean_fr)/std_fr
					decoded_z_fr_vecs.append(d_fr_vec_z)
					#Grab correlation data
					corr_decode_event = np.array([pearsonr(all_taste_fr_vecs_mean[t_i,:],d_fr_vec)[0] for t_i in range(num_tastes)])
					decoded_corr[nd_i,:] = corr_decode_event
					corr_decode_event_z = np.array([pearsonr(all_taste_fr_vecs_mean_z[t_i,:],d_fr_vec_z)[0] for t_i in range(num_tastes)])
					decoded_z_corr[nd_i] = corr_decode_event_z
				#Find where the correlation data is highest for the given taste
				decoded_corr_match = (np.argmax(decoded_corr,1) == t_i).astype('int')
				decoded_z_corr_match = (np.argmax(decoded_z_corr,1) == t_i).astype('int')
				decode_prob_avg = np.array([np.mean(seg_decode_epoch_prob[t_i,start_decoded[i]:end_decoded[i]]) for i in range(len(start_decoded))])
				#Find where the decoding is higher than a cutoff
				decode_above_cutoff = (decode_prob_avg >= decode_prob_cutoff).astype('int')
				decode_above_neur_cutoff = (num_neur_decoded >= neur_cut).astype('int')
				best_across_metrics = np.where(decode_above_cutoff*decoded_corr_match*decoded_z_corr_match*decode_above_neur_cutoff)[0]
				
				#Store all the firing rate vectors for plotting
				all_taste_event_fr_vecs.append(np.array(decoded_fr_vecs))
				all_taste_event_fr_vecs_z.append(np.array(decoded_z_fr_vecs))
				all_taste_event_fr_vecs_neur_cut.append(np.array(decoded_fr_vecs)[decode_ind])
				all_taste_event_fr_vecs_z_neur_cut.append(np.array(decoded_z_fr_vecs)[decode_ind])
				all_taste_event_fr_vecs_best.append(np.array(decoded_fr_vecs)[best_across_metrics])
				all_taste_event_fr_vecs_z_best.append(np.array(decoded_z_fr_vecs)[best_across_metrics])
				
				#Now only keep matching decoded intervals
				decoded_bin = np.zeros(np.shape(decoded_taste))
				for db in best_across_metrics:
					s_db = start_decoded[db]
					e_db = end_decoded[db]
					decoded_bin[s_db:e_db] = 1
				#Grab overall percentages
				epoch_seg_taste_percents_best[e_i,s_i,t_i] = (np.sum(decoded_bin)/len(decoded_bin))*100
				#Re-calculate start and end times of the decoded intervals
				start_decoded_best = start_decoded[best_across_metrics]
				end_decoded_best = end_decoded[best_across_metrics]
				#Re-calculate the decoded statistics
				num_decoded_best = len(start_decoded_best)
				num_neur_decoded_best = num_neur_decoded[best_across_metrics]
				prob_decoded_best = prob_decoded[best_across_metrics]
				len_decoded_best = np.array(end_decoded_best - start_decoded_best)
				iei_decoded_best = np.array(start_decoded_best[1:] - end_decoded_best[:-1])
				
				seg_distribution_best = np.zeros(len(seg_dist_starts)-1)
				prob_distribution_best = np.zeros(len(seg_dist_starts)-1)
				for sd_i in range(len(seg_dist_starts)-1):
					bin_events = np.where((start_decoded_best < seg_dist_starts[sd_i+1])*(start_decoded_best >= seg_dist_starts[sd_i]))[0]
					seg_distribution_best[sd_i] = len(bin_events)
					prob_distribution_best[sd_i] = np.mean(prob_decoded_best[bin_events])
				
				#Plot the statistics for those decoded events that are best across metrics
				save_name = 'best_events'
				plot_overall_decoded_stats(len_decoded_best,iei_decoded_best,num_neur_decoded_best,
											 prob_decoded_best,prob_distribution_best,e_i,s_i,
											 seg_dist_midbin,seg_distribution_best,seg_stat_bin,
											 seg_len,save_name,taste_decode_save_dir)
				
				if num_decoded > max_decode: #Reduce number if too many
					#TODO: add flag to select which cutoff to use for plotting?
					#Reduce to top decoding probability
					#decode_plot_ind = sample(list(np.where(decode_above_cutoff)[0]),max_decode)
					
					#Reduce to ones with both top decoding probability and highest correlation of both regular and z-scored
					decode_plot_ind = sample(list(best_across_metrics),min(max_decode,len(best_across_metrics)))
				else:
					decode_plot_ind = np.arange(num_decoded)
				decode_plot_ind = np.array(decode_plot_ind)
				#Create plots of the decoded periods
				if len(decode_plot_ind)>0:
					for nd_i in decode_plot_ind:
						#Grab decode variables
						d_start = start_decoded[nd_i]
						d_end = end_decoded[nd_i]
						d_len = d_end-d_start
						d_plot_start = np.max((start_decoded[nd_i]-2*d_len,0))
						d_plot_end = np.min((end_decoded[nd_i]+2*d_len,seg_len))
						d_plot_len = d_plot_end-d_plot_start
						d_plot_x_vals = (np.linspace(d_plot_start-d_start,d_plot_end-d_start,10)).astype('int')
						decode_plot_times = np.arange(d_plot_start,d_plot_end)
						event_spikes = segment_spike_times_s_i_bin[:,d_plot_start:d_plot_end]
						decode_spike_times = []
						for n_i in range(num_neur):
							decode_spike_times.append(list(np.where(event_spikes[n_i,:])[0]))
						event_spikes_expand = segment_spike_times_s_i_bin[:,d_plot_start-10:d_plot_end+10]
						event_spikes_expand_count = np.sum(event_spikes_expand,0)
						firing_rate_vec = np.zeros(d_plot_len)
						for dpt_i in np.arange(10,d_plot_len+10):
							firing_rate_vec[dpt_i-10] = np.sum(event_spikes_expand_count[dpt_i-10:dpt_i+10])/(20/1000)/num_neur
						d_fr_vec = decoded_fr_vecs[nd_i]
						#Grab z-scored data
						d_fr_vec_z = decoded_z_fr_vecs[nd_i]
						#Find max hz 
						#d_fr_vec_max_hz = np.max(d_fr_vec)
						#Correlation of vector to avg taste vector
						corr_decode_event = decoded_corr[nd_i]
						corr_title_norm = [dig_in_names[t_i] + ' corr = ' + str(np.round(corr_decode_event[t_i],2)) for t_i in range(num_tastes)]
						#Correlation of z-scored vector to z-scored avg taste vector
						corr_decode_event_z = decoded_z_corr[nd_i]
						corr_title_z = [dig_in_names[t_i] + ' z-corr = ' + str(np.round(corr_decode_event_z[t_i],2)) for t_i in range(num_tastes)]
						corr_title = (', ').join(corr_title_norm) + '\n' + (', ').join(corr_title_z)
						#Start Figure
						f, ax = plt.subplots(nrows=5,ncols=2,figsize=(10,10),gridspec_kw=dict(height_ratios=[1,1,1,2,2]))
						gs = ax[0,0].get_gridspec()
						#Decoding probabilities
						ax[0,0].remove()
						ax[1,0].remove()
						axbig = f.add_subplot(gs[0:2,0])
						decode_x_vals = decode_plot_times-d_start
						leg_handles = []
						for t_i_2 in range(num_tastes):
							taste_decode_prob_y = seg_decode_epoch_prob[t_i_2,d_plot_start:d_plot_end]
							p_h, = axbig.plot(decode_x_vals,taste_decode_prob_y,color=taste_colors[t_i_2,:])
							leg_handles.append(p_h)
							taste_decode_prob_y[0] = 0
							taste_decode_prob_y[-1] = 0
							high_prob_binary = np.zeros(len(decode_x_vals))
							high_prob_times = np.where(taste_decode_prob_y >= decode_prob_cutoff)[0]
							high_prob_binary[high_prob_times] = 1
							high_prob_starts = np.where(np.diff(high_prob_binary) == 1)[0] + 1
							high_prob_ends = np.where(np.diff(high_prob_binary) == -1)[0] + 1
							if len(high_prob_starts) > 0:
								for hp_i in range(len(high_prob_starts)):
									axbig.fill_between(decode_x_vals[high_prob_starts[hp_i]:high_prob_ends[hp_i]],taste_decode_prob_y[high_prob_starts[hp_i]:high_prob_ends[hp_i]],alpha=0.2,color=taste_colors[t_i_2,:])
						axbig.axvline(0,color='k',alpha=0.5)
						axbig.axvline(d_len,color='k',alpha=0.5)
						axbig.legend(leg_handles,dig_in_names,loc='right')
						axbig.set_xticks(d_plot_x_vals)
						axbig.set_xlim([decode_x_vals[0],decode_x_vals[-1]])
						axbig.set_ylabel('Decoding Fraction')
						axbig.set_xlabel('Time from Event Start (ms)')
						axbig.set_title('Event ' + str(nd_i) + '\nStart Time = ' + str(round(d_start/1000/60,3)) + ' Minutes' + '\nEvent Length = ' + str(np.round(d_len,2)))
						#Decoded raster
						ax[0,1].eventplot(decode_spike_times)
						ax[0,1].set_xlim([0,d_plot_len])
						x_ticks = np.linspace(0,d_plot_len,10).astype('int')
						ax[0,1].set_xticks(x_ticks,labels=d_plot_x_vals)
						ax[0,1].axvline(d_start-d_plot_start,color='k',alpha=0.5)
						ax[0,1].axvline(d_end-d_plot_start,color='k',alpha=0.5)
						ax[0,1].set_ylabel('Neuron Index')
						ax[0,1].set_title('Event Spike Raster')
						ax[1,0].axis('off')
						#Plot population firing rates w 20ms smoothing
						ax[1,1].plot(decode_x_vals,firing_rate_vec)
						ax[1,1].axvline(0,color='k')
						ax[1,1].axvline(d_len,color='k')
						ax[1,1].set_xlim([decode_x_vals[0],decode_x_vals[-1]])
						ax[1,1].set_xticks(d_plot_x_vals)
						ax[1,1].set_title('Population Avg FR')
						ax[1,1].set_ylabel('FR (Hz)')
						ax[1,1].set_xlabel('Time from Event Start (ms)')
						#Decoded Firing Rates
						img = ax[2,0].imshow(np.expand_dims(d_fr_vec,0),vmin=0,vmax=60)#vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
						ax[2,0].set_xlabel('Neuron Index')
						ax[2,0].set_yticks(ticks=[])
						#plt.colorbar(img, location='bottom',orientation='horizontal',label='Firing Rate (Hz)',panchor=(0.9,0.5),ax=ax[2,0])
						ax[2,0].set_title('Event FR')
						#Decoded Firing Rates Z-Scored
						img = ax[2,1].imshow(np.expand_dims(d_fr_vec_z,0),vmin=-3,vmax=3,cmap='bwr')
						ax[2,1].set_xlabel('Neuron Index')
						ax[2,1].set_yticks(ticks=[])
						#plt.colorbar(img, ax=ax[2,1], location='bottom',orientation='horizontal',label='Z-Scored Firing Rate (Hz)',panchor=(0.9,0.5))
						ax[2,1].set_title('Event FR Z-Scored')
						#Taste Firing Rates
						img = ax[3,0].imshow(np.expand_dims(taste_fr_vecs_mean,0),vmin=0,vmax=60)#vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
						ax[3,0].set_xlabel('Neuron Index')
						ax[3,0].set_yticks(ticks=[])
						plt.colorbar(img, ax= ax[3,0], location='bottom',orientation='horizontal',label='Firing Rate (Hz)',panchor=(0.9,0.5))
						ax[3,0].set_title('Avg. Taste Resp. FR')
						#Taste Firing Rates Z-Scored
						img = ax[3,1].imshow(np.expand_dims(all_taste_fr_vecs_mean_z[t_i,:],0),vmin=-3,vmax=3,cmap='bwr')
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
						ax[4,1].scatter(all_taste_fr_vecs_mean_z[t_i,:],d_fr_vec_z)
						ax[4,1].set_xlabel('Average Taste Neuron FR Std > Mean')
						ax[4,1].set_ylabel('Event Neuron FR Std > Mean')
						ax[4,1].set_title('Z-Scored Firing Rate Similarity')
						plt.suptitle(corr_title,wrap=True)
						plt.tight_layout()
						#Save Figure
						f.savefig(taste_decode_save_dir + 'event_' + str(nd_i) + '.png')
						f.savefig(taste_decode_save_dir + 'event_' + str(nd_i) + '.svg')
						plt.close(f)
			
			#Taste event deviation plots
			save_name = 'all_events'
			title='Deviation Events x Individual Taste Response'
			plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs,
									all_taste_fr_vecs,taste_colors,num_neur,save_name,title,
									seg_decode_save_dir)
			
			#Taste event deviation plots z-scored
			save_name = 'all_events_z'
			title='Z-Scored Deviation Events x Individual Z-Scored Taste Response'
			plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs_z,
									all_taste_fr_vecs_z,taste_colors,num_neur,save_name,title,
									seg_decode_save_dir)
			
			#Taste event deviation plots
			save_name = 'neur_cutoff_events'
			title='Deviation Events x Individual Taste Response'
			plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs_neur_cut,
									all_taste_fr_vecs,taste_colors,num_neur,save_name,title,
									seg_decode_save_dir)
			
			#Taste event deviation plots z-scored
			save_name = 'neur_cutoff_events_z'
			title='Z-Scored Deviation Events x Individual Z-Scored Taste Response'
			plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs_z_neur_cut,
									all_taste_fr_vecs_z,taste_colors,num_neur,save_name,title,
									seg_decode_save_dir)
			
			#Taste event deviation plots
			save_name = 'best_events'
			title='Deviation Events x Individual Taste Response'
			plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs_best,
									all_taste_fr_vecs,taste_colors,num_neur,save_name,title,
									seg_decode_save_dir)
			
			#Taste event deviation plots z-scored
			save_name = 'best_events_z'
			title='Z-Scored Deviation Events x Individual Z-Scored Taste Response'
			plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs_z_best,
									all_taste_fr_vecs_z,taste_colors,num_neur,save_name,title,
									seg_decode_save_dir)
			
			
	#Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments		
	f, ax = plt.subplots(nrows = len(epochs_to_analyze), ncols = num_tastes, figsize=(num_tastes*4,len(epochs_to_analyze)*4))
	for e_ind, e_i in enumerate(epochs_to_analyze):
		for t_i in range(num_tastes):
			ax[e_ind,t_i].plot(segments_to_analyze,(epoch_seg_taste_percents[e_i,segments_to_analyze,t_i]).flatten())
			seg_labels = [segment_names[a] for a in segments_to_analyze]
			ax[e_ind,t_i].set_xticks(segments_to_analyze,labels=seg_labels,rotation=-45)
			if t_i == 0:
				ax[e_ind,t_i].set_ylabel('Epoch ' + str(e_i))
			ax[e_ind,t_i].title('Taste ' + dig_in_names[t_i])
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents.png')
	f.savefig(save_dir + 'Decoding_Percents.svg')
	plt.close(f)
	
	#Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments		
	f, ax = plt.subplots(nrows = len(epochs_to_analyze), ncols = num_tastes, figsize=(num_tastes*4,len(epochs_to_analyze)*4))
	for e_ind, e_i in enumerate(epochs_to_analyze):
		for t_i in range(num_tastes):
			ax[e_ind,t_i].plot(segments_to_analyze,(epoch_seg_taste_percents_neur_cut[e_i,segments_to_analyze,t_i]).flatten())
			seg_labels = [segment_names[a] for a in segments_to_analyze]
			ax[e_ind,t_i].set_xticks(segments_to_analyze,labels=seg_labels,rotation=-45)
			if t_i == 0:
				ax[e_ind,t_i].set_ylabel('Epoch ' + str(e_i))
			ax[e_ind,t_i].title('Taste ' + dig_in_names[t_i])
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents_Neuron_Cutoff.png')
	f.savefig(save_dir + 'Decoding_Percents_Neuron_Cutoff.svg')
	plt.close(f)
	
	#Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments		
	f, ax = plt.subplots(nrows = len(epochs_to_analyze), ncols = num_tastes, figsize=(num_tastes*4,len(epochs_to_analyze)*4))
	for e_ind, e_i in enumerate(epochs_to_analyze):
		for t_i in range(num_tastes):
			ax[e_ind,t_i].plot(segments_to_analyze,(epoch_seg_taste_percents_best[e_i,segments_to_analyze,t_i]).flatten())
			seg_labels = [segment_names[a] for a in segments_to_analyze]
			ax[e_ind,t_i].set_xticks(segments_to_analyze,labels=seg_labels,rotation=-45)
			if t_i == 0:
				ax[e_ind,t_i].set_ylabel('Epoch ' + str(e_i))
			ax[e_ind,t_i].title('Taste ' + dig_in_names[t_i])
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents_Best.png')
	f.savefig(save_dir + 'Decoding_Percents_Best.svg')
	plt.close(f)
	

def plot_decoded_func_p(fr_dist,num_tastes,num_neur,segment_spike_times,tastant_spike_times,
				start_dig_in_times,end_dig_in_times,post_taste_dt,cp_raster_inds,
				e_skip_dt,e_len_dt,dig_in_names,segment_times,
				segment_names,taste_num_deliv,taste_select_epoch,
				save_dir,max_decode,max_hz,seg_stat_bin,
				epochs_to_analyze=[],segments_to_analyze=[]):	
	"""Function to plot the decoding statistics as a function of average decoding
	probability within the decoded interval."""
	warnings.filterwarnings('ignore')
	num_cp = np.shape(cp_raster_inds[0])[-1] - 1
	num_segments = len(segment_spike_times)
	prob_cutoffs = np.arange(1/num_tastes,1,0.05)
	num_prob = len(prob_cutoffs)
	taste_colors = cm.viridis(np.linspace(0,1,num_tastes))
	epoch_seg_taste_percents = np.zeros((num_prob,num_cp,num_segments,num_tastes))
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(num_segments)

	for e_i in epochs_to_analyze: #By epoch conduct decoding
		print('Decoding Epoch ' + str(e_i))
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			print("Data not previously decoded, or passed directory incorrect.")
			pass
		
		for s_i in tqdm.tqdm(segments_to_analyze):
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
	sum_width_ratios = np.concatenate((10*np.ones(len(segments_to_analyze)),np.ones(1)))
	max_decoding_percent = np.max(epoch_seg_taste_percents)
	f, ax = plt.subplots(len(epochs_to_analyze),len(segments_to_analyze) + 1,figsize=((len(segments_to_analyze) + 1)*4,len(epochs_to_analyze)*4),width_ratios=sum_width_ratios)
	if len(epochs_to_analyze) > 1:
		gs = ax[0,-1].get_gridspec()
		for e_ind, e_i in enumerate(epochs_to_analyze):
			for s_ind, s_i in enumerate(segments_to_analyze):
				ax[e_ind,s_ind].set_ylim([0,max_decoding_percent+10])
				for t_i in range(num_tastes):
					ax[e_ind,s_ind].plot(prob_cutoffs,(epoch_seg_taste_percents[:,e_i,s_i,t_i]).flatten(),color=taste_colors[t_i,:],alpha=0.8)
				if s_ind == 0:
					ax[e_ind,s_ind].set_ylabel('Epoch ' + str(e_i))
				if e_ind == 0:
					ax[e_ind,s_ind].set_title(segment_names[s_i])
				if e_ind == num_cp-1:
					ax[e_ind,s_ind].set_xlabel('Probability Cutoff')
		for ax_i in ax[:,-1]:
			ax_i.remove() #remove the underlying axes
	else:
		gs = ax[-1].get_gridspec()
		for e_ind, e_i in enumerate(epochs_to_analyze):
			for s_ind, s_i in enumerate(segments_to_analyze):
				ax[s_ind].set_ylim([0,max_decoding_percent+10])
				for t_i in range(num_tastes):
					ax[s_ind].plot(prob_cutoffs,(epoch_seg_taste_percents[:,e_i,s_i,t_i]).flatten(),color=taste_colors[t_i,:],alpha=0.8)
				if s_ind == 0:
					ax[s_ind].set_ylabel('Epoch ' + str(e_i))
				if e_ind == 0:
					ax[s_ind].set_title(segment_names[s_i])
				if e_ind == num_cp-1:
					ax[s_ind].set_xlabel('Probability Cutoff')
		ax[-1].remove() #remove the underlying axes
	axbig = f.add_subplot(gs[:,-1])
	cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),cax=axbig,ticks=np.linspace(0,1,num_tastes),orientation='vertical')
	cbar.ax.set_yticklabels(dig_in_names)
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents_By_Prob_Cutoff.png')
	f.savefig(save_dir + 'Decoding_Percents_By_Prob_Cutoff.svg')
	plt.close(f)
	
	
def plot_decoded_func_n(fr_dist,num_tastes,num_neur,segment_spike_times,tastant_spike_times,
				start_dig_in_times,end_dig_in_times,post_taste_dt,cp_raster_inds,
				e_skip_dt,e_len_dt,dig_in_names,segment_times,
				segment_names,taste_num_deliv,taste_select_epoch,
				save_dir,max_decode,max_hz,seg_stat_bin,
				epochs_to_analyze=[],segments_to_analyze=[]):
	"""Function to plot the decoding statistics as a function of number of 
	neurons firing within the decoded interval."""
	warnings.filterwarnings('ignore')
	num_cp = np.shape(cp_raster_inds[0])[-1] - 1
	num_segments = len(segment_spike_times)
	neur_cutoffs = np.arange(1,num_neur)
	num_cuts = len(neur_cutoffs)
	taste_colors = cm.viridis(np.linspace(0,1,num_tastes))
	epoch_seg_taste_percents = np.zeros((num_cuts,num_cp,num_segments,num_tastes))
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(num_segments)

	for e_i in epochs_to_analyze: #By epoch conduct decoding
		print('Decoding Epoch ' + str(e_i))
		taste_select_neur = np.where(taste_select_epoch[e_i,:] == 1)[0]
		epoch_decode_save_dir = save_dir + 'decode_prob_epoch_' + str(e_i) + '/'
		if not os.path.isdir(epoch_decode_save_dir):
			print("Data not previously decoded, or passed directory incorrect.")
			pass
		
		for s_i in tqdm.tqdm(segments_to_analyze):
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
	sum_width_ratios = np.concatenate((10*np.ones(len(segments_to_analyze)),np.ones(1)))
	max_decoding_percent = np.max(epoch_seg_taste_percents)
	f, ax = plt.subplots(len(epochs_to_analyze),len(segments_to_analyze) + 1,figsize=((len(segments_to_analyze) + 1)*4,len(epochs_to_analyze)*4),width_ratios=sum_width_ratios)
	if len(epochs_to_analyze) > 1:
		gs = ax[0,-1].get_gridspec()
		for e_ind, e_i in enumerate(epochs_to_analyze):
			for s_ind, s_i in enumerate(segments_to_analyze):
				ax[e_ind,s_ind].set_ylim([0,max_decoding_percent+10])
				for t_i in range(num_tastes):
					ax[e_ind,s_ind].plot(neur_cutoffs,(epoch_seg_taste_percents[:,e_i,s_i,t_i]).flatten(),color=taste_colors[t_i,:],alpha=0.8)
				if s_ind == 0:
					ax[e_ind,s_ind].set_ylabel('Epoch ' + str(e_i))
				if e_ind == 0:
					ax[e_ind,s_ind].set_title(segment_names[s_i])
				if e_ind == num_cp-1:
					ax[e_ind,s_ind].set_xlabel('Neuron Cutoff')
		for ax_i in ax[:,-1]:
			ax_i.remove() #remove the underlying axes
	else:
		gs = ax[-1].get_gridspec()
		for e_ind, e_i in enumerate(epochs_to_analyze):
			for s_ind, s_i in enumerate(segments_to_analyze):
				ax[s_ind].set_ylim([0,max_decoding_percent+10])
				for t_i in range(num_tastes):
					ax[s_ind].plot(neur_cutoffs,(epoch_seg_taste_percents[:,e_i,s_i,t_i]).flatten(),color=taste_colors[t_i,:],alpha=0.8)
				if s_ind == 0: 
					ax[s_ind].set_ylabel('Epoch ' + str(e_i))
				if e_ind == 0:
					ax[s_ind].set_title(segment_names[s_i])
				if e_ind == num_cp-1:
					ax[s_ind].set_xlabel('Neuron Cutoff')
		ax[-1].remove() #remove the underlying axes
	axbig = f.add_subplot(gs[:,-1])
	cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),cax=axbig,ticks=np.linspace(0,1,num_tastes),orientation='vertical')
	cbar.ax.set_yticklabels(dig_in_names)
	plt.tight_layout()
	f.savefig(save_dir + 'Decoding_Percents_By_Neur_Cutoff.png')
	f.savefig(save_dir + 'Decoding_Percents_By_Neur_Cutoff.svg')
	plt.close(f)

def plot_overall_decoded_stats(len_decoded,iei_decoded,num_neur_decoded,
							  prob_decoded,prob_distribution,e_i,s_i,
							  seg_dist_midbin,seg_distribution,seg_stat_bin,
							  seg_len,save_name,taste_decode_save_dir):
	"""For use by the plot_decoded function - plots decoded event statistics"""
	
	#Plot the statistics for those decoded events that are best across metrics
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
	f.savefig(taste_decode_save_dir + 'epoch_' + str(e_i) + '_seg_' + str(s_i) + '_' + save_name + '.png')
	f.savefig(taste_decode_save_dir + 'epoch_' + str(e_i) + '_seg_' + str(s_i) + '_'+ save_name + '.svg')
	plt.close(f)
	
def plot_scatter_fr_vecs_taste_mean(num_tastes,dig_in_names,all_taste_event_fr_vecs,
						all_taste_fr_vecs_mean,taste_colors,save_name,title,
						seg_decode_save_dir):
	"""Plot a scatter plot of the decoded event firing rate against the
	average taste response firing rate"""
	f, ax = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(num_tastes*2,num_tastes*2), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
	max_fr = 0
	max_fr_t_av = 0
	for t_i in range(num_tastes): #Event Taste
		ax[t_i,0].set_ylabel('Decoded ' + dig_in_names[t_i] +' FR')
		taste_event_fr_vecs = all_taste_event_fr_vecs[t_i]
		if len(taste_event_fr_vecs) > 0:
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
	plt.suptitle(title)
	plt.tight_layout()
	f.savefig(seg_decode_save_dir + save_name + '.png')
	f.savefig(seg_decode_save_dir + save_name  + '.svg')
	plt.close(f)
	
def plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs,
						all_taste_fr_vecs,taste_colors,num_neur,save_name,title,
						seg_decode_save_dir):
	"""Plot distances in fr space between the decoded event firing rate and the
	taste response firing rate and correlations between decoded event firing
	rates and taste firing rates"""
	f_dist, ax_dist = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(num_tastes*4,num_tastes*4), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
	f_corr, ax_corr = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(num_tastes*4,num_tastes*4), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
	max_x = 0
	all_diff_tastes = []
	all_corr_tastes = []
	max_density = 0
	for t_i in range(num_tastes): #Event Taste
		all_diff_taste = []
		all_corr_taste = []
		taste_event_fr_vecs = all_taste_event_fr_vecs[t_i]
		ax_dist[t_i,0].set_ylabel('Firing Rate Difference')
		ax_corr[t_i,0].set_ylabel(dig_in_names[t_i])
		for t_i_c in range(num_tastes): #Average Taste
			#Calculate max fr of taste response
			ax_dist[t_i,t_i_c].set_title('Decoded ' + dig_in_names[t_i] +' - Delivery ' + dig_in_names[t_i_c] )
			ax_corr[t_i,t_i_c].set_title('Decodes x Deliveries')
			taste_fr_vecs = all_taste_fr_vecs[t_i_c]
			max_taste_resp_fr = np.max(taste_fr_vecs)
			x_vals = np.arange(num_neur)
			if max_taste_resp_fr > max_x:
				max_x = max_taste_resp_fr
			num_taste_deliv = np.shape(taste_fr_vecs)[0]
			num_events = np.shape(taste_event_fr_vecs)[0]
			if num_events > 0:
				all_diff = np.zeros((num_events*num_taste_deliv,num_neur))
				all_corr = []
				for td_i in range(num_taste_deliv):
					diff = np.abs(taste_event_fr_vecs - taste_fr_vecs[td_i,:])
					corr = corr_calculator(taste_fr_vecs[td_i,:], taste_event_fr_vecs)
					all_diff[td_i*num_events:(td_i+1)*num_events,:] = diff
					all_corr.extend(corr)
				all_diff_taste.append(all_diff.flatten())
				all_corr_taste.append(all_corr)
				ax_dist[t_i,t_i_c].violinplot(all_diff,x_vals,showmedians=True,showextrema=False)
				hist_results = ax_corr[t_i,t_i_c].hist(np.array(all_corr),density=True,histtype='step',bins=np.arange(-1,1,0.025))
				if np.max(hist_results[0]) > max_density:
					max_density = np.max(hist_results[0])
				ax_corr[t_i,t_i_c].axvline(np.nanmean(all_corr),linestyle='dashed',color='k',alpha=0.7)
				ax_corr[t_i,t_i_c].text(np.nanmean(all_corr)+0.1,0,str(np.round(np.nanmean(all_corr),2)),rotation=90)
				ax_dist[t_i,t_i_c].set_xlabel('Neuron Index')
				ax_corr[t_i,t_i_c].set_xlabel(dig_in_names[t_i_c])
			else:
				all_diff_taste.append([])
				all_corr_taste.append([])
		all_diff_tastes.append(all_diff_taste)
		all_corr_tastes.append(all_corr_taste)
	for t_i in range(num_tastes):
		for t_i_c in range(num_tastes):
			ax_dist[t_i,t_i_c].axhline(0,alpha=0.5,color='k',linestyle='dashed')
			ax_dist[t_i,t_i_c].set_ylim([-10,100])
			ax_corr[t_i,t_i_c].set_ylim([-0.1,max_density])
			if t_i == t_i_c:
				for child in ax_dist[t_i,t_i_c].get_children():
					if isinstance(child, matplotlib.spines.Spine):
						child.set_color('r')
				for child in ax_corr[t_i,t_i_c].get_children():
					if isinstance(child, matplotlib.spines.Spine):
						child.set_color('r')
	f_dist.suptitle(title)
	f_dist.tight_layout()
	f_dist.savefig(seg_decode_save_dir + save_name + '_distances.png')
	f_dist.savefig(seg_decode_save_dir + save_name + '_distances.svg')
	plt.close(f_dist)
	f_corr.suptitle(title)
	f_corr.tight_layout()
	f_corr.savefig(seg_decode_save_dir + save_name + '_correlations.png')
	f_corr.savefig(seg_decode_save_dir + save_name + '_correlations.svg')
	plt.close(f_corr)
	
	f2_dist, ax2_dist = plt.subplots(nrows=1,ncols=num_tastes,figsize=(num_tastes*4,4))
	f2_corr, ax2_corr = plt.subplots(nrows=1,ncols=num_tastes,figsize=(num_tastes*4,4))
	for t_i in range(num_tastes):
		ax2_dist[t_i].hist(all_diff_tastes[t_i],bins=1000,histtype='step',density=True,cumulative=True,label=dig_in_names)
		ax2_dist[t_i].legend()
		ax2_dist[t_i].set_ylim([-0.1,1.1])
		ax2_dist[t_i].set_xlim([0,100])
		ax2_dist[t_i].set_xlabel('|Distance|')
		ax2_dist[t_i].set_title('Decoded ' + dig_in_names[t_i])
		ax2_corr[t_i].hist(all_corr_tastes[t_i],bins=1000,histtype='step',density=True,cumulative=True,label=dig_in_names)
		ax2_corr[t_i].legend()
		ax2_corr[t_i].set_ylim([-0.1,1.1])
		ax2_corr[t_i].set_xlim([-0.1,1.1])
		ax2_corr[t_i].set_xlabel('Correlation')
		ax2_corr[t_i].set_title('Decoded ' + dig_in_names[t_i])
	ax2_dist[0].set_ylabel('Cumulative Density')
	ax2_corr[0].set_ylabel('Cumulative Density')
	f2_dist.suptitle(title)
	f2_dist.tight_layout()
	f2_dist.savefig(seg_decode_save_dir + save_name + '_population_dist.png')
	f2_dist.savefig(seg_decode_save_dir + save_name + '_population_dist.svg')
	plt.close(f2_dist)
	f2_corr.suptitle(title)
	f2_corr.tight_layout()
	f2_corr.savefig(seg_decode_save_dir + save_name + '_population_corr.png')
	f2_corr.savefig(seg_decode_save_dir + save_name + '_population_corr.svg')
	plt.close(f2_corr)
	
def corr_calculator(deliv_fr_vec, decode_fr_mat):
	"""
	This function calculates correlations between a matrix of firing rate vectors
	and a single firing rate vector. Note, this function assumes the length of
	the vectors is equivalent (i.e. the number of neurons).
	
	INPUTS:
		- deliv_fr_vec: single vector of taste delivery firing rate vector
		- decode_fr_mat: matrix with rows of individual decoded event firing 
			rate vectors
	OUTPUTS:
		- list of correlations of each decoded event to the given delivery event
	"""
	#First convert single vector into a matrix
	deliv_fr_mat = deliv_fr_vec*np.ones(np.shape(decode_fr_mat))
	
	#Calculate the mean-subtracted vectors
	deliv_mean_sub = deliv_fr_mat - np.expand_dims(np.mean(deliv_fr_mat,1),1)*np.ones(np.shape(deliv_fr_mat))
	decode_mean_sub = decode_fr_mat - np.expand_dims(np.mean(decode_fr_mat,1),1)*np.ones(np.shape(decode_fr_mat))
	
	#Calculate the squares of the mean-subtracted vectors
	deliv_mean_sub_squared = np.square(deliv_mean_sub)
	decode_mean_sub_squared = np.square(decode_mean_sub)
	
	#Calculate the numerators and denominators of the pearson's correlation calculation
	pearson_num = np.sum(np.multiply(deliv_mean_sub,decode_mean_sub),1)
	pearson_denom = np.sqrt(np.sum(deliv_mean_sub_squared,1))*np.sqrt(np.sum(decode_mean_sub_squared,1))
	
	#Convert to list
	pearson_correlations = list(np.divide(pearson_num,pearson_denom))
	
	return pearson_correlations
	
	
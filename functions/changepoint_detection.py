#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:01:49 2023

@author: Hannah Germaine

This is a collection of functions for performing changepoint detection on raster matrices
"""
#Quick and dirty changepoint detection algorithm based on percentile of bin fr difference
#find the percentile of each bin difference for each neuron, then sum across 
#neurons and find peaks as the average point of a changepoint
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import kstest
from numba import jit
import tqdm

def calc_cp_taste_PSTH_ks_test(PSTH_times, PSTH_taste_deliv_times, tastant_PSTH,
						  cp_bin, bin_step, dig_in_names,num_cp,before_taste, after_taste,
						  taste_cp_save_dir):
	print("Calculating changepoints around taste delivery")
	num_tastes = len(tastant_PSTH)
	num_steps_cp_bin = int(np.ceil(cp_bin/bin_step)) #How many steps in the PSTH times align with a changepoint bin
	taste_cp_inds = []
	for t_i in range(num_tastes):
		taste_name = dig_in_names[t_i]
		print("Calculating and plotting changepoints for each delivery of tastant " + taste_name)
		[taste_start_deliv,taste_end_deliv] = PSTH_taste_deliv_times[t_i]
		num_deliveries = np.shape(tastant_PSTH[t_i])[0]
		num_neur = np.shape(tastant_PSTH[t_i])[1]
		deliv_cp_inds = np.zeros((num_deliveries,num_cp+1))
		PSTH_taste_times = np.array(PSTH_times[t_i][:])
		taste_start_deliv_ind = np.where(PSTH_taste_times >= taste_start_deliv)[0][0]
		taste_end_deliv_ind = np.where(PSTH_taste_times >= taste_end_deliv)[0][0]
		deliv_cp_inds[:,0] += taste_start_deliv_ind
		for d_i in tqdm.tqdm(range(num_deliveries)):
			deliv_PSTH = tastant_PSTH[t_i][d_i,:] #List of lists with neuron spike times around taste delivery
			#Z-score PSTH
			PSTH_means = np.mean(deliv_PSTH,axis=1)
			PSTH_stds = np.std(deliv_PSTH,axis=1)
			deliv_PSTH_zscore = (deliv_PSTH-PSTH_means[:,None])/PSTH_stds[:,None]
			#Calculate population changepoints using KS-Test
			ks_p_tracker = np.zeros(np.shape(PSTH_taste_times)) #Collect all p-values
			last_good_p = 0
			for nt_i in range(len(PSTH_taste_times)-2):
				#change deliv_PSTH_zscore back to deliv_PSTH if desired
				first_mat = deliv_PSTH_zscore[:,min(last_good_p,nt_i - num_steps_cp_bin):nt_i+1].flatten()
				second_mat = deliv_PSTH_zscore[:,nt_i+1:-1].flatten()
				if (len(first_mat) > 0) and (len(second_mat) > 0):
					[ks_res, ks_pval] = kstest(first_mat,second_mat,alternative='two-sided')
					ks_p_tracker[nt_i] = ks_pval
					if ks_pval < 0.05:
						last_good_p = nt_i
			p_val_peaks = find_peaks(np.ones(np.shape(PSTH_taste_times))-ks_p_tracker,distance=num_steps_cp_bin)[0]		
			p_val_peaks = p_val_peaks[p_val_peaks > taste_end_deliv_ind]
			ordered_peak_ind = np.argsort(ks_p_tracker[p_val_peaks])
			ks_cp_best = p_val_peaks[np.sort(ordered_peak_ind[0:num_cp])]
			deliv_cp_inds[d_i,1:len(ks_cp_best)] = ks_cp_best
			#Plot delivery results
			fig = plt.figure(figsize=(10,10))
			plt.subplot(2,1,1)
			plt.imshow(deliv_PSTH_zscore,aspect='equal')
			for c_p in ks_cp_best:
				plt.axvline(c_p,linestyle='dashed',color='b')
			plt.subplot(2,1,2)
			for n_i in range(num_neur):
				plt.plot(np.array(PSTH_taste_times)*1000 - before_taste,deliv_PSTH[n_i,:])
			plt.axvline(0,color='k',linestyle='dashed')
			for c_p in ks_cp_best:
				plt.axvline(PSTH_taste_times[c_p]*1000 - before_taste,linestyle='dashed',color='b')
			plt.suptitle(taste_name + ' delivery ' + str(d_i))
			plt.tight_layout()
			fig_name = taste_name + '_PSTH-cp_deliv_' + str(d_i)
			fig.savefig(taste_cp_save_dir + fig_name + '.png')
			fig.savefig(taste_cp_save_dir + fig_name + '.svg')
			plt.close(fig)
		taste_cp_inds.append(deliv_cp_inds)
	#Calculate average changepoint times
	taste_avg_cp_inds = np.array([list(np.ceil(np.mean(taste_cp_inds[t_i],0)).astype('int')) for t_i in range(num_tastes)]) #in PSTH_times indices
	taste_avg_cp_times = np.array([list(PSTH_times[t_i][list(taste_avg_cp_inds[t_i])]*1000) for t_i in range(num_tastes)]) #in ms
	#Plot the average changepoint times for tastants
	for t_i in range(num_tastes):
		avg_PSTH = np.mean(tastant_PSTH[t_i],0)
		z_scored_avg_PSTH = (avg_PSTH - np.mean(avg_PSTH,1)[:,None])/np.std(avg_PSTH,1)[:,None]
		fig = plt.figure(figsize=(10,10))
		plt.subplot(2,1,1)
		plt.imshow(avg_PSTH,aspect='equal')
		for c_p in taste_avg_cp_inds[t_i]:
			plt.axvline(c_p,linestyle='dashed',color='b')
		plt.colorbar()
		plt.title('Avg PSTH')
		plt.subplot(2,1,2)
		plt.imshow(z_scored_avg_PSTH,aspect='equal')
		for c_p in taste_avg_cp_inds[t_i]:
			plt.axvline(c_p,linestyle='dashed',color='b')
		plt.colorbar()
		plt.title('Z-Scored Avg PSTH')
		plt.suptitle(dig_in_names[t_i])
		plt.tight_layout()
		fig_name = dig_in_names[t_i] + '_avg_PSTH_cp'
		fig.savefig(taste_cp_save_dir + fig_name + '.png')
		fig.savefig(taste_cp_save_dir + fig_name + '.svg')
		plt.close(fig)
	
	return taste_cp_inds, taste_avg_cp_inds, taste_avg_cp_times

@jit(forceobj=True)
def fit_function(k, lamb):
	# The parameter lamb will be used as the fit parameter
	return poisson.pmf(k, lamb)

def plot_peak_height_dist(peak_heights,cp_save_dir):
	#Plot the distribution of peak heights
	print('Plotting changepoints probability distribution')
	fig = plt.figure(figsize=(10,10))
	plt.hist(peak_heights)
	plt.title('Changepoint Probability Distribution')
	fig_name = 'cp_prob_distribution'
	fig.savefig(cp_save_dir + fig_name + '.png')
	fig.savefig(cp_save_dir + fig_name + '.svg')
	plt.close(fig)
	
def plot_cp_rasters(cp_ind,cp_bin,neur_raster,num_neur,rec_len,indiv_cp_save_dir):
	#Plot rasters of neurons around each predicted changepoint
	print('Plotting rasters about changepoints')
	before_cp = 2*cp_bin #Number of milliseconds to plot before the changepoint
	after_cp = 2*cp_bin #Number of milliseconds to plot before the changepoint
	for cp_i in tqdm.tqdm(cp_ind):
		min_ind = cp_i-before_cp
		if min_ind < 0:
			center = np.abs(min_ind)
			min_ind = 0
		else:
			center = before_cp
		max_ind = min(cp_i+after_cp,rec_len+1)
		rast_chunk = neur_raster[:,min_ind:max_ind]
		spike_ind_chunk = [np.where(rast_chunk[n_i])[0] for n_i in range(num_neur)]
		fig = plt.figure(figsize=(10,10))
		plt.eventplot(spike_ind_chunk,orientation='horizontal',colors='k')
		plt.axvline(center,label='changepoint')
		plt.legend()
		plt.title('CP Time (ms) = ' + str(cp_i))
		fig_name = 'cp_' + str(cp_i)
		fig.savefig(indiv_cp_save_dir + fig_name + '.png')
		fig.savefig(indiv_cp_save_dir + fig_name + '.svg')
		plt.close(fig)
		
def plot_taste_cp(cp_ind,neur_raster,dig_in_names,start_dig_in_times,
				 end_dig_in_times, before_taste,after_taste,rec_len,
				 taste_cp_save_dir):
	print('Plotting Taste PSTHs with local changepoints')
	num_tastes = len(dig_in_names)
	num_neur = np.shape(neur_raster)[0]
	for n_t in tqdm.tqdm(range(num_tastes)):
		taste_name = dig_in_names[n_t]
		taste_start_times = np.array(start_dig_in_times[n_t])
		taste_end_times = np.array(end_dig_in_times[n_t])
		for t_t_i in range(len(taste_start_times)):
			min_ind = taste_start_times[t_t_i] - before_taste
			if min_ind < 0:
				center = np.abs(min_ind)
				min_ind = 0
			else:
				center = before_taste
			max_ind = min(taste_end_times[t_t_i] + after_taste,rec_len+1)
			cp_ind_int = cp_ind[np.where((min_ind < cp_ind)*(cp_ind < max_ind))[0]] - min_ind
			fig = plt.figure(figsize=(10,10))
			rast_chunk = neur_raster[:,min_ind:max_ind]
			spike_ind_chunk = [np.where(rast_chunk[n_i])[0] for n_i in range(num_neur)]
			plt.eventplot(spike_ind_chunk,orientation='horizontal',colors='k')
			for cp_i in cp_ind_int:
				plt.axvline(cp_i,color='b',linestyle='dashed',label='cp')
			plt.axvline(before_taste,color='r',linestyle='dashed',label='taste_start')
			plt.axvline(max_ind-min_ind-after_taste,color='r',linestyle='dashed',label='taste_start')
			plt.legend()
			plt.title(taste_name + ' delivery ' + str(t_t_i) + 'PSTH + changepoints')
			fig_name = taste_name + '_' + str(t_t_i)
			fig.savefig(taste_cp_save_dir + fig_name + '.png')
			fig.savefig(taste_cp_save_dir + fig_name + '.svg')
			plt.close(fig)

def avg_cp_PSTH(dig_in_names, num_cp, taste_cp_inds, avg_tastant_PSTH, 
				PSTH_times, before_taste, after_taste, cp_bin, bin_step, taste_cp_save_dir):
	
	num_tastes = len(dig_in_names)
	half_bin_width_dt = int(np.ceil(cp_bin/2)) #in ms
	dt_total = before_taste+after_taste+1
	PSTH_start_times = np.arange(0,dt_total,bin_step)
	
	#Calculate avg tastant cp values
	cp_approx = []
	for t_i in range(num_tastes):
		binary_cp_vec = np.zeros(dt_total)
		for d_i in range(len(taste_cp_inds[t_i])):
			binary_cp_vec[list(taste_cp_inds[t_i][d_i])] += 1
		PSTH_cp = np.zeros(len(PSTH_start_times))
		for i in range(len(PSTH_start_times)):
			PSTH_cp[i] = np.sum(binary_cp_vec[max(PSTH_start_times[i]-half_bin_width_dt,0):min(PSTH_start_times[i]+half_bin_width_dt,dt_total)])
		plt.plot(binary_cp_vec)
		plt.plot(PSTH_start_times,PSTH_cp)
		PSTH_cp_peak_inds = find_peaks(PSTH_cp,distance=200/bin_step)[0]
		PSTH_cp_peak_vals = PSTH_cp[PSTH_cp_peak_inds]
		PSTH_cp_heights_sorted_inds = PSTH_cp_peak_inds[np.argsort(PSTH_cp_peak_vals)[::-1]]
		cp_approx.append(PSTH_start_times[PSTH_cp_heights_sorted_inds[0:num_cp]])
	
		
	#Plot avg tastant PSTH with avg cp
	fig, ax = plt.subplots(nrows=num_tastes)
	for t_i in range(num_tastes):
		for p_i in range(len(avg_tastant_PSTH[t_i])):
			ax[t_i].plot(np.array(PSTH_times[t_i])*1000 - before_taste,avg_tastant_PSTH[t_i][p_i][:])
		ax[t_i].axvline(0,color='k',linestyle='dashed')
		for c_p in range(num_cp):
			ax[t_i].axvline(cp_approx[t_i][c_p] - before_taste,linestyle='dashed',color='b')
		ax[t_i].set_title(dig_in_names[t_i])
	plt.tight_layout()
	fig_name = 'PSTH_w_avg_cp'
	fig.savefig(taste_cp_save_dir + fig_name + '.png')
	fig.savefig(taste_cp_save_dir + fig_name + '.svg')
	plt.close(fig)
	

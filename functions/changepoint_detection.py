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
from matplotlib import cm
from scipy.stats import poisson
from scipy.stats import kstest
from numba import jit
import tqdm, os

def calc_cp_taste_PSTH_ks_test(PSTH_times, PSTH_taste_deliv_times, tastant_PSTH,
						  cp_bin, bin_step, dig_in_names,num_cp,before_taste, after_taste,
						  taste_cp_save_dir, local_window = 0):
	print("Calculating changepoints around taste delivery")
	num_tastes = len(tastant_PSTH)
	num_steps_cp_bin = int(np.ceil(cp_bin/bin_step)) #How many steps in the PSTH times align with a changepoint bin
	taste_cp_PSTH_inds = []
	taste_cp_true_inds = []
	for t_i in range(num_tastes):
		taste_name = dig_in_names[t_i]
		print("Calculating and plotting changepoints for each delivery of tastant " + taste_name)
		[taste_start_deliv,taste_end_deliv] = PSTH_taste_deliv_times[t_i]
		num_deliveries = np.shape(tastant_PSTH[t_i])[0]
		num_neur = np.shape(tastant_PSTH[t_i])[1]
		deliv_cp_PSTH_inds = np.zeros((num_deliveries,num_neur,num_cp+1))
		deliv_cp_true_inds = np.zeros((num_deliveries,num_neur,num_cp+1))
		PSTH_taste_times = np.array(PSTH_times[t_i][:])
		taste_start_deliv_ind = np.where(PSTH_taste_times >= taste_start_deliv)[0][0]
		taste_end_deliv_ind = np.where(PSTH_taste_times >= taste_end_deliv)[0][0]
		deliv_cp_PSTH_inds[:,:,0] += taste_start_deliv_ind
		deliv_cp_true_inds[:,:,0] += taste_start_deliv*1000
		taste_PSTH = tastant_PSTH[t_i]
		for d_i in tqdm.tqdm(range(num_deliveries)):
			deliv_PSTH = taste_PSTH[d_i,:,:] #List of lists with neuron spike times around taste delivery
			#Z-score PSTH
			PSTH_means = np.mean(deliv_PSTH,axis=1)
			PSTH_stds = np.std(deliv_PSTH,axis=1)
			deliv_PSTH_zscore = (deliv_PSTH-PSTH_means[:,None])/PSTH_stds[:,None]
			#Calculate individual neuron changepoints using KS-Test
			ks_p_tracker = np.zeros((num_neur,len(PSTH_taste_times))) #Collect all p-values
			for n_i in range(num_neur):
				last_good_p = 0
				#change deliv_PSTH_zscore back to deliv_PSTH if desired
				for nt_i in range(len(PSTH_taste_times)-2):
					if local_window == 0:
						first_mat = deliv_PSTH_zscore[n_i,:nt_i+1].flatten()#deliv_PSTH_zscore[n_i,min(last_good_p,nt_i - num_steps_cp_bin):nt_i+1].flatten()
						second_mat = deliv_PSTH_zscore[n_i,nt_i+1:-1].flatten()
					elif local_window > 0:
						first_mat = deliv_PSTH_zscore[n_i,max(nt_i+1-local_window,0):nt_i+1].flatten()#deliv_PSTH_zscore[n_i,min(last_good_p,nt_i - num_steps_cp_bin):nt_i+1].flatten()
						second_mat = deliv_PSTH_zscore[n_i,nt_i+1:min(nt_i+1+local_window,len(deliv_PSTH_zscore[n_i,:]))].flatten()
					else:
						first_mat = deliv_PSTH_zscore[n_i,last_good_p:nt_i+1].flatten()#deliv_PSTH_zscore[n_i,min(last_good_p,nt_i - num_steps_cp_bin):nt_i+1].flatten()
						second_mat = deliv_PSTH_zscore[n_i,nt_i+1:-1].flatten()
					if (len(first_mat) > 0) and (len(second_mat) > 0):
						[ks_res, ks_pval] = kstest(first_mat,second_mat,alternative='two-sided')
						ks_p_tracker[n_i,nt_i] = ks_pval
						if ks_pval < 0.05:
							last_good_p = nt_i
				p_val_peaks = find_peaks(np.ones(np.shape(PSTH_taste_times))-ks_p_tracker[n_i,:],distance=num_steps_cp_bin)[0]		
				p_val_peaks = p_val_peaks[p_val_peaks > taste_end_deliv_ind]
				ordered_peak_ind = np.argsort(ks_p_tracker[n_i,p_val_peaks])
				ks_cp_best = p_val_peaks[np.sort(ordered_peak_ind[0:num_cp])]
				deliv_cp_PSTH_inds[d_i,n_i,1:len(ks_cp_best)+1] = ks_cp_best
				deliv_cp_true_inds[d_i,n_i,1:len(ks_cp_best)+1] = PSTH_taste_times[ks_cp_best]*1000
			#Plot delivery results
			avg_cp_PSTH_deliv(num_cp,d_i,deliv_cp_PSTH_inds[d_i,:,:],PSTH_taste_times,
					 taste_start_deliv_ind,num_steps_cp_bin,deliv_PSTH_zscore,taste_name,taste_cp_save_dir)
		#Plot taste results
		avg_taste_PSTH = np.mean(taste_PSTH,0)
		avg_PSTH_means = np.mean(avg_taste_PSTH,axis=1)
		avg_PSTH_stds = np.std(avg_taste_PSTH,axis=1)
		avg_taste_PSTH_zscore = (avg_taste_PSTH-avg_PSTH_means[:,None])/avg_PSTH_stds[:,None]
		avg_cp_PSTH(num_cp,deliv_cp_PSTH_inds,PSTH_taste_times,taste_start_deliv_ind,
			  num_steps_cp_bin,avg_taste_PSTH_zscore,taste_name,taste_cp_save_dir)
		taste_cp_PSTH_inds.append(deliv_cp_PSTH_inds)
		taste_cp_true_inds.append(deliv_cp_true_inds)
	return taste_cp_true_inds

def calc_cp_bayes(tastant_spike_times,cp_bin,num_cp,start_dig_in_times,
				  end_dig_in_times,before_taste,after_taste,
				  dig_in_names,taste_cp_save_dir):
	neur_save_dir = taste_cp_save_dir + 'neur/'
	if os.path.isdir(neur_save_dir) == False:
		os.mkdir(neur_save_dir)
	deliv_save_dir = taste_cp_save_dir + 'deliv/'
	if os.path.isdir(deliv_save_dir) == False:
		os.mkdir(deliv_save_dir)
	num_tastes = len(tastant_spike_times)
	num_deliv = len(tastant_spike_times[0])
	num_neur = len(tastant_spike_times[0][0])
	taste_cp = []
	for t_i in range(num_tastes):
		print('Calculating changepoints for taste ' + dig_in_names[t_i])
		neur_deliv_cp = np.zeros((num_deliv,num_neur,num_cp+1))
		deliv_st = dict()
		#Calculate changepoints for each neuron for each tastant delivery
		for n_i in tqdm.tqdm(range(num_neur)):
			neur_deliv_st = []
			neur_cp_likelihood = dict() #Collects for 1 neuron the cp likelihood for each delivery index
			bin_length_collect = []
			for d_i in range(num_deliv):
				start_deliv = start_dig_in_times[t_i][d_i]
				end_deliv = end_dig_in_times[t_i][d_i]
				taste_deliv_len = end_deliv-start_deliv
				spike_ind = tastant_spike_times[t_i][d_i][n_i]
				bin_length = before_taste+taste_deliv_len+after_taste+1
				bin_length_collect.extend([bin_length])
				#Create the binary spike matrix
				deliv_bin = np.zeros(bin_length)
				converted_ind = list((np.array(spike_ind) - start_deliv + before_taste).astype('int'))
				try:
					cur_st = deliv_st[d_i]
					cur_st.append(converted_ind)
					deliv_st[d_i] = cur_st
				except:
					deliv_st[d_i] = [converted_ind]
				neur_deliv_st.append(converted_ind)
				deliv_bin[converted_ind] = 1		
				#Run through each timepoint starting at the minimum changepoint bin 
				#size to the length of the segment - the changepoint bin size and
				#calculate the proxy for a changepoint between two Poisson processes
				cp_likelihood_d_i = np.zeros(bin_length)
				for time_i in np.arange(cp_bin,bin_length-cp_bin):
					#N_1 = np.sum(deliv_bin[:time_i]) 
					N_1 = np.sum(deliv_bin[time_i-cp_bin:time_i])
					#N_2 = np.sum(deliv_bin[time_i:])
					N_2 = np.sum(deliv_bin[time_i:time_i+cp_bin])
					cp_likelihood_d_i[time_i] = (((N_1/cp_bin)**N_1)*((N_2/cp_bin)**N_2))#/((N_1+N_2)/(2*cp_bin))**(N_1+N_2)
					#cp_likelihood_d_i[time_i] = (((N_1/time_i)**N_1)*((N_2/(bin_length-time_i))**N_2))#/((N_1+N_2)/(2*bin_length))**(N_1+N_2)
				peak_inds = find_peaks(cp_likelihood_d_i[cp_bin:],distance=cp_bin)[0]
				peak_inds = peak_inds[peak_inds>before_taste] + cp_bin
				ordered_peak_ind = np.argsort(cp_likelihood_d_i[peak_inds])
				best_peak_inds = np.zeros(num_cp+1)
				best_peak_inds[0] = before_taste
				found_inds = peak_inds[np.sort(ordered_peak_ind[0:num_cp])]
				if len(found_inds) < num_cp:
					diff_len_cp = num_cp - len(found_inds)
					best_peak_inds[1:len(found_inds)+1] = found_inds
					best_peak_inds[len(found_inds)+1:] = (bin_length-1)*np.ones(diff_len_cp)
				else:
					best_peak_inds[1:num_cp+1] = found_inds
				neur_cp_likelihood[d_i] = list(best_peak_inds.astype('int'))
			#_____Look at the average cp likelihood for this one neuron across deliveries_____
			neur_cp_likelihood_list = []
			neur_cp_likelihood_bin = np.zeros((num_deliv,np.max(bin_length_collect)))
			for key in neur_cp_likelihood.keys():
				neur_cp_likelihood_list.append(list(neur_cp_likelihood[key]))
				neur_cp_likelihood_bin[int(key),list(neur_cp_likelihood[key])] = 1
			#_____Plot changepoints across deliveries for one neuron_____
			plot_cp_rasters_neur(neur_deliv_st,neur_cp_likelihood_list,before_taste,
				   dig_in_names[t_i],n_i,num_deliv,num_cp,neur_save_dir)
			for key in neur_cp_likelihood.keys():
				neur_deliv_cp[int(key),n_i,:] = neur_cp_likelihood[key]
		#_____Plot changepoints for each delivery across the population_____
		plot_cp_rasters_deliv(deliv_st,neur_deliv_cp,before_taste,dig_in_names[t_i],deliv_save_dir)
		#Store results for tastant
		taste_cp.append(neur_deliv_cp)
		
	return taste_cp

def plot_cp_rasters_neur(neur_deliv_st,neur_cp_likelihood_list,before_taste,dig_in_name,n_i,num_deliv,num_cp,taste_cp_save_dir):
	#Delivery aligned
	fig = plt.figure(figsize=(5,10))
	plt.eventplot(neur_deliv_st,colors='k',alpha=0.5)
	plt.eventplot(neur_cp_likelihood_list,colors='r')
	plt.axvline(before_taste,color='b')
	plt.title(dig_in_name + ' neuron '+ str(n_i) + ' raster aligned by taste deliv')
	plt.xlabel('Time (ms)')
	plt.ylabel('Delivery Index')
	fig_name = dig_in_name + '_neur_' + str(n_i) + '_rast_aligned_taste'
	fig.savefig(taste_cp_save_dir + fig_name + '.png')
	fig.savefig(taste_cp_save_dir + fig_name + '.svg')
	plt.close(fig)
	#CP aligned
	for cp_i in range(num_cp):
		fig = plt.figure(figsize=(5,10))
		realigned_deliv_st = []
		realigned_cp = []
		for d_i in range(num_deliv):
			realign_d_i = np.array(neur_deliv_st[d_i]) - neur_cp_likelihood_list[d_i][cp_i]
			realigned_deliv_st.append(list(realign_d_i))
			realigned_cp.append(list(np.array(neur_cp_likelihood_list[d_i]) - neur_cp_likelihood_list[d_i][cp_i]))
		plt.eventplot(realigned_deliv_st,colors='k',alpha=0.5)
		plt.eventplot(realigned_cp,colors='r')
		plt.title(dig_in_name + ' neuron '+ str(n_i) + ' raster aligned by cp ' + str(cp_i))
		plt.xlabel('Aligned Index')
		plt.ylabel('Delivery Index')
		fig_name = dig_in_name + '_neur_' + str(n_i) + '_rast_aligned_cp_' + str(cp_i)
		fig.savefig(taste_cp_save_dir + fig_name + '.png')
		fig.savefig(taste_cp_save_dir + fig_name + '.svg')
		plt.close(fig)

def plot_cp_rasters_deliv(deliv_st,neur_deliv_cp,before_taste,dig_in_name,taste_cp_save_dir):
	num_deliv, num_neur, num_cp = np.shape(neur_deliv_cp)
	for d_i in range(num_deliv):
		#Delivery aligned
		fig = plt.figure(figsize=(5,5))
		spike_times = deliv_st[d_i]
		plt.eventplot(spike_times,colors='b',alpha=0.5)
		plt.eventplot(neur_deliv_cp[d_i,:,:],colors='r')
		plt.title(dig_in_name + ' delivery '+ str(d_i))
		plt.xlabel('Time (ms)')
		plt.ylabel('Neuron Index')
		fig_name = dig_in_name + '_deliv_' + str(d_i)
		fig.savefig(taste_cp_save_dir + fig_name + '.png')
		fig.savefig(taste_cp_save_dir + fig_name + '.svg')
		plt.close(fig)
		
def avg_cp_PSTH_deliv(num_cp,d_i,deliv_cp_inds_array,PSTH_taste_times,taste_start_deliv_ind,num_steps_cp_bin,deliv_PSTH_zscore,taste_name,taste_cp_save_dir):
	"""Plots the PSTH as an image and the average changepoints across neurons for that
	delivery of a taste as vertical lines"""
	deliv_cp_all_inds = (deliv_cp_inds_array.flatten()).astype('int')
	deliv_cp_density = np.zeros(len(PSTH_taste_times))
	for d_cp_i in deliv_cp_all_inds:
		deliv_cp_density[d_cp_i-int(num_steps_cp_bin/2):d_cp_i+int(num_steps_cp_bin/2)] += 1
	avg_cp_deliv_peaks_ind = find_peaks(deliv_cp_density,distance=num_steps_cp_bin)[0]
	num_cp_given = len(avg_cp_deliv_peaks_ind)
	#Plot PSTH with changepoints
	colors = cm.cool(np.arange(num_cp_given+1)/(num_cp_given+1))
	fig = plt.figure(figsize=(5,5))
	num_neur = np.shape(deliv_PSTH_zscore)[0]
	im_ratio = deliv_PSTH_zscore.shape[0]/deliv_PSTH_zscore.shape[1]
	im = plt.imshow(deliv_PSTH_zscore,aspect='auto',cmap='bone', extent=[min(PSTH_taste_times),max(PSTH_taste_times),num_neur,0])
	plt.axvline(PSTH_taste_times[taste_start_deliv_ind],linestyle='dashed',color = 'k')
	for c_p in range(num_cp_given):
		plt.axvline(PSTH_taste_times[avg_cp_deliv_peaks_ind[c_p]],linestyle='dashed',color=colors[c_p])
	plt.colorbar(im)#,fraction=0.046*im_ratio, pad=0.04)
	plt.title(taste_name + ' delivery ' + str(d_i))
	plt.xlabel('Time (s)')
	plt.tight_layout()
	fig_name = taste_name + '_avg_PSTH_cp_deliv_' + str(d_i)
	fig.savefig(taste_cp_save_dir + fig_name + '.png')
	fig.savefig(taste_cp_save_dir + fig_name + '.svg')
	plt.close(fig)
	
def avg_cp_PSTH(num_cp,deliv_cp_inds,PSTH_taste_times,taste_start_deliv_ind,num_steps_cp_bin,deliv_PSTH_zscore,taste_name,taste_cp_save_dir):
	"""Plots the PSTH as an image and the average changepoints across neurons f
	and across deliveries for that taste as vertical lines"""
	colors = cm.cool(np.arange(num_cp+1)/(num_cp+1))
	deliv_cp_all_inds = (deliv_cp_inds.flatten()).astype('int')
	deliv_cp_density = np.zeros(len(PSTH_taste_times))
	for d_cp_i in deliv_cp_all_inds:
		deliv_cp_density[d_cp_i-int(num_steps_cp_bin/2):d_cp_i+int(num_steps_cp_bin/2)] += 1
	avg_cp_deliv_peaks_ind = find_peaks(deliv_cp_density,distance=num_steps_cp_bin)[0]
	fig = plt.figure(figsize=(5,5))
	num_neur = np.shape(deliv_PSTH_zscore)[0]
	im_ratio = deliv_PSTH_zscore.shape[0]/deliv_PSTH_zscore.shape[1]
	im = plt.imshow(deliv_PSTH_zscore,aspect='auto',cmap='bone', extent=[min(PSTH_taste_times),max(PSTH_taste_times),num_neur,0])
	plt.axvline(PSTH_taste_times[taste_start_deliv_ind],linestyle='dashed',color = 'k')
	for c_p in range(num_cp+1):
		plt.axvline(PSTH_taste_times[avg_cp_deliv_peaks_ind[c_p]],linestyle='dashed',color=colors[c_p])
	plt.colorbar(im)#,fraction=0.046*im_ratio, pad=0.04)
	plt.title(taste_name + ' avg PSTH')
	plt.xlabel('Time (s)')
	fig_name = taste_name + '_avg_PSTH_cp'
	fig.savefig(taste_cp_save_dir + fig_name + '.png')
	fig.savefig(taste_cp_save_dir + fig_name + '.svg')
	plt.close(fig)
	
	
#TODO: Use the below framework to write an HMM approach that outputs the same data structure as other cp calc functions
def calc_cp_HMM(tastant_spike_times,cp_bin,num_cp,start_dig_in_times,
				  end_dig_in_times,before_taste,after_taste,
				  dig_in_names,taste_cp_save_dir):
	neur_save_dir = taste_cp_save_dir + 'neur/'
	if os.path.isdir(neur_save_dir) == False:
		os.mkdir(neur_save_dir)
	deliv_save_dir = taste_cp_save_dir + 'deliv/'
	if os.path.isdir(deliv_save_dir) == False:
		os.mkdir(deliv_save_dir)
	num_tastes = len(tastant_spike_times)
	num_deliv = len(tastant_spike_times[0])
	num_neur = len(tastant_spike_times[0][0])
	taste_cp = []
	for t_i in range(num_tastes):
		print('Calculating changepoints for taste ' + dig_in_names[t_i])
		neur_deliv_cp = np.zeros((num_deliv,num_neur,num_cp+1))
		deliv_st = dict()
		#Calculate changepoints for each neuron for each tastant delivery
		for n_i in tqdm.tqdm(range(num_neur)):
			neur_deliv_st = []
			neur_cp_likelihood = dict() #Collects for 1 neuron the cp likelihood for each delivery index
			bin_length_collect = []
			for d_i in range(num_deliv):
				start_deliv = start_dig_in_times[t_i][d_i]
				end_deliv = end_dig_in_times[t_i][d_i]
				taste_deliv_len = end_deliv-start_deliv
				spike_ind = tastant_spike_times[t_i][d_i][n_i]
				bin_length = before_taste+taste_deliv_len+after_taste+1
				bin_length_collect.extend([bin_length])
				#Create the binary spike matrix
				deliv_bin = np.zeros(bin_length)
				converted_ind = list((np.array(spike_ind) - start_deliv + before_taste).astype('int'))
				try:
					cur_st = deliv_st[d_i]
					cur_st.append(converted_ind)
					deliv_st[d_i] = cur_st
				except:
					deliv_st[d_i] = [converted_ind]
				neur_deliv_st.append(converted_ind)
				deliv_bin[converted_ind] = 1		
				#Run through each timepoint starting at the minimum changepoint bin 
				#size to the length of the segment - the changepoint bin size and
				#calculate the proxy for a changepoint between two Poisson processes
				cp_likelihood_d_i = np.zeros(bin_length)
				for time_i in np.arange(cp_bin,bin_length-cp_bin):
					#N_1 = np.sum(deliv_bin[:time_i]) 
					N_1 = np.sum(deliv_bin[time_i-cp_bin:time_i])
					#N_2 = np.sum(deliv_bin[time_i:])
					N_2 = np.sum(deliv_bin[time_i:time_i+cp_bin])
					cp_likelihood_d_i[time_i] = (((N_1/cp_bin)**N_1)*((N_2/cp_bin)**N_2))#/((N_1+N_2)/(2*cp_bin))**(N_1+N_2)
					#cp_likelihood_d_i[time_i] = (((N_1/time_i)**N_1)*((N_2/(bin_length-time_i))**N_2))#/((N_1+N_2)/(2*bin_length))**(N_1+N_2)
				peak_inds = find_peaks(cp_likelihood_d_i[cp_bin:],distance=cp_bin)[0]
				peak_inds = peak_inds[peak_inds>before_taste] + cp_bin
				ordered_peak_ind = np.argsort(cp_likelihood_d_i[peak_inds])
				best_peak_inds = np.zeros(num_cp+1)
				best_peak_inds[0] = before_taste
				found_inds = peak_inds[np.sort(ordered_peak_ind[0:num_cp])]
				if len(found_inds) < num_cp:
					diff_len_cp = num_cp - len(found_inds)
					best_peak_inds[1:len(found_inds)+1] = found_inds
					best_peak_inds[len(found_inds)+1:] = (bin_length-1)*np.ones(diff_len_cp)
				else:
					best_peak_inds[1:num_cp+1] = found_inds
				neur_cp_likelihood[d_i] = list(best_peak_inds.astype('int'))
			#_____Look at the average cp likelihood for this one neuron across deliveries_____
			neur_cp_likelihood_list = []
			neur_cp_likelihood_bin = np.zeros((num_deliv,np.max(bin_length_collect)))
			for key in neur_cp_likelihood.keys():
				neur_cp_likelihood_list.append(list(neur_cp_likelihood[key]))
				neur_cp_likelihood_bin[int(key),list(neur_cp_likelihood[key])] = 1
			#_____Plot changepoints across deliveries for one neuron_____
			plot_cp_rasters_neur(neur_deliv_st,neur_cp_likelihood_list,before_taste,
				   dig_in_names[t_i],n_i,num_deliv,num_cp,neur_save_dir)
			for key in neur_cp_likelihood.keys():
				neur_deliv_cp[int(key),n_i,:] = neur_cp_likelihood[key]
		#_____Plot changepoints for each delivery across the population_____
		plot_cp_rasters_deliv(deliv_st,neur_deliv_cp,before_taste,dig_in_names[t_i],deliv_save_dir)
		#Store results for tastant
		taste_cp.append(neur_deliv_cp)
		
	return taste_cp
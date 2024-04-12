#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 10:33:10 2023

@author: Hannah Germaine
A collection of functions to calculate statistics and plot comparisons between
null and true datasets
"""

import os, random, json, gzip, tqdm
os.environ["OMP_NUM_THREADS"] = "4"
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide = 'ignore', invalid = 'ignore') 
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

def run_null_create_parallelized(inputs):
	"""
	This set of code creates null distributions for a given dataset where the 
	existing data is passed in as indices of spikes and length of dataset
	"""
	null_ind = inputs[0]
	spikes = inputs[1]
	start_t = inputs[2]
	end_t = inputs[3]
	null_dir = inputs[4]
	fake_spike_times = [random.sample(range(start_t,end_t),len(spikes[n_i])) for n_i in range(len(spikes))]
	json_str = json.dumps(fake_spike_times)
	json_bytes = json_str.encode()
	filepath = null_dir + 'null_' + str(null_ind) + '.json'
	with gzip.GzipFile(filepath, mode="w") as f:
		f.write(json_bytes)
		
def high_bins(inputs):
	"""This function calculates bins of time that have the number of neurons 
	spiking above some given threshold. It's optimized for multiprocessing."""
	warnings.filterwarnings('ignore')
	bin_spikes = inputs[0]
	segment_start_time = inputs[1]
	segment_end_time = inputs[2]
	bin_size = inputs[3]
	num_thresh = inputs[4]
	bin_dt = int(np.ceil(bin_size*1000)) #Convert from seconds to ms = dt
	#Sweep array in bins searching for those with the number of neurons >= num_thresh
	bin_spike_counts = [np.sum(np.sum(bin_spikes[:,b_i:b_i+bin_dt],1)) for b_i in range(segment_end_time-segment_start_time-bin_dt)]
	bin_neur_counts = [np.sum(np.sum(bin_spikes[:,b_i:b_i+bin_dt],1)>0) for b_i in range(segment_end_time-segment_start_time-bin_dt)]
	high_neur_bins = dict()
	for t_i in num_thresh:
		high_neur_bins.update({str(t_i): np.sum((np.array(bin_neur_counts) >= t_i).astype('int'))})
	high_spike_bins = dict()
	num_spikes = np.arange(1,max(np.array(bin_spike_counts)),2)
	for s_i in num_spikes:
		high_spike_bins.update({str(s_i): np.sum((np.array(bin_neur_counts) >= s_i).astype('int'))})
	
	return high_neur_bins, high_spike_bins

def auto_corr(inputs):
	"""This function calculates autocorrelation for a dataset given a binary
	spike matrix and a set of lags to test. It's optimized for multiprocessing."""
	bin_spikes = inputs[0]
	segment_start_time = inputs[1]
	segment_end_time = inputs[2]
	lag_vals = inputs[3]
	"""This function calculates autocorrelations of spiking activity to classify the regularity of spiking"""
	seg_length = segment_end_time - segment_start_time
	#Sweep array in bins performing autocorrelation
	segment_starts = np.sort(np.random.choice(np.arange(seg_length),np.ceil(seg_length/100).astype('int')))
	lag_results = dict()
	for lag_i in range(len(lag_vals)):
		lag = lag_vals[lag_i]
		bin_spike_correlations = [np.corrcoef(bin_spikes[:,b_i],bin_spikes[:,b_i+lag])[0,1] for b_i in segment_starts[segment_starts<seg_length-lag]]
		lag_results[lag_i] = np.nanmean(np.array(bin_spike_correlations))
	
	return lag_results

def plot_indiv_truexnull(true_x_vals,null_x_vals,true_spike_counts,mean_null_spike_counts,
							 std_null_spike_counts,segment_length,norm_val,
							 save_dir,plot_name,seg_name,percentiles=[]):
	"""Plot the results of high_bins() and auto_corr() run on true and null datasets"""
	#True data
	fig = plt.figure(figsize=(5,5))
	plt.plot(true_x_vals,true_spike_counts,label='true bin counts',color='b')
# 	if len(percentiles) > 0:
# 		y_max = max(true_spike_counts[true_spike_counts != np.nan])
# 		y_text_buffer = y_max/100
# 		for p_i in range(len(true_x_vals)):
# 			if not np.isnan(true_x_vals[p_i]):
# 				if not np.isnan(true_spike_counts[p_i]):
# 					plt.text(true_x_vals[p_i],true_spike_counts[p_i]+y_text_buffer,str(percentiles[p_i]))
	plt.fill_between(null_x_vals, 
			   (np.array(mean_null_spike_counts)-std_null_spike_counts), 
			  (mean_null_spike_counts+std_null_spike_counts),
			  alpha=0.4,color='g',label='null bin counts std')
	plt.plot(null_x_vals,mean_null_spike_counts,color='g',label='null bin counts mean')
	plt.legend()
	plt.xlabel(plot_name)
	plt.ylabel('Number of Instances')
	plt.title(plot_name)
	plt.tight_layout()
	im_name = plot_name.replace(' ','_') + '_truexnull'
	fig.savefig(save_dir + im_name + '_' + seg_name + '.png')
	fig.savefig(save_dir + im_name + '_' + seg_name + '.svg')
	plt.close(fig)
	#Log scale
	fig = plt.figure(figsize=(5,5))
	plt.plot(true_x_vals,np.log(true_spike_counts),label='true bin counts',color='b')
# 	if len(percentiles) > 0:
# 		log_spike_counts = np.log(true_spike_counts)
# 		y_max = max(log_spike_counts[log_spike_counts != np.nan])
# 		y_text_buffer = y_max/100
# 		for p_i in range(len(true_x_vals)):
# 			if not np.isnan(true_x_vals[p_i]):
# 				if not np.isnan(np.log(true_spike_counts[p_i])):
# 					plt.text(true_x_vals[p_i],np.log(true_spike_counts[p_i])+y_text_buffer,str(percentiles[p_i]))
	plt.fill_between(null_x_vals, 
			   (np.log(mean_null_spike_counts-std_null_spike_counts)), 
			  (np.log(mean_null_spike_counts+std_null_spike_counts)),
			  alpha=0.4,color='g',label='null bin counts std')
	plt.plot(null_x_vals,np.log(mean_null_spike_counts),color='g',label='null bin counts mean')
	plt.legend()
	plt.xlabel(plot_name)
	plt.ylabel('Log(Number of Instances)')
	plt.title(plot_name)
	plt.tight_layout()
	im_name = plot_name.replace(' ','_') + '_log_truexnull'
	fig.savefig(save_dir + im_name + '_' + seg_name + '.png')
	fig.savefig(save_dir + im_name + '_' + seg_name + '.svg')
	plt.close(fig)
	#Normalized data
	fig = plt.figure(figsize=(5,5))
	plt.plot(true_x_vals,true_spike_counts/norm_val,label='true bin counts',color='b')
# 	if len(percentiles) > 0:
# 		y_max = max(true_spike_counts[true_spike_counts != np.nan]/norm_val)
# 		y_text_buffer = y_max/100
# 		for p_i in range(len(true_x_vals)):
# 			if not np.isnan(true_x_vals[p_i]):
# 				if not np.isnan(true_spike_counts[p_i]):
# 					plt.text(true_x_vals[p_i],true_spike_counts[p_i]+y_text_buffer,str(percentiles[p_i]))
	plt.fill_between(null_x_vals, 
			   (mean_null_spike_counts-std_null_spike_counts)/norm_val, 
			  (mean_null_spike_counts+std_null_spike_counts)/norm_val,
			  alpha=0.4,color='g',label='null bin counts std')
	plt.plot(null_x_vals,mean_null_spike_counts/norm_val,color='g',label='null bin counts mean')
	plt.legend()
	plt.xlabel(plot_name + ' per Second')
	plt.ylabel('Number of Instances')
	plt.title(plot_name + ' Normalized per Second')
	plt.tight_layout()
	im_name = plot_name.replace(' ','_') + '_truexnull_norm'
	fig.savefig(save_dir + im_name + '_' + seg_name + '.png')
	fig.savefig(save_dir + im_name + '_' + seg_name + '.svg')
	plt.close(fig)

def plot_all_truexnull(true_x_vals,null_x_vals,true_spike_counts,mean_null_spike_counts,
							 std_null_spike_counts,norm_val,save_dir,plot_name,segment_names):
	"""Plot the results of high_bins() and autocorr() run on true and null 
	datasets on one set of axes. True and null inputs are expected to be matrices
	with a number of rows representing the different segments."""
	#True data
	colors = ['red','orange','green','blue','cyan','purple','pink']
	fig = plt.figure(figsize=(5,5))
	for s_i in range(len(true_x_vals)):
		plt.plot(true_x_vals[s_i],true_spike_counts[s_i],label=segment_names[s_i] + ' true',linestyle='solid',color=colors[s_i])
		plt.fill_between(null_x_vals[s_i], 
				   (np.array(mean_null_spike_counts[s_i])-std_null_spike_counts[s_i]), 
				  (mean_null_spike_counts[s_i]+std_null_spike_counts[s_i]),
				  alpha=0.4,color=colors[s_i],label=segment_names[s_i] + ' null std')
		plt.plot(null_x_vals[s_i],mean_null_spike_counts[s_i],linestyle='dotted',color=colors[s_i],label=segment_names[s_i] + ' null mean')
	plt.legend()
	plt.xlabel(plot_name)
	plt.ylabel('Number of Instances')
	plt.title(plot_name)
	plt.tight_layout()
	im_name = plot_name.replace(' ','_') + '_truexnull_all'
	fig.savefig(save_dir + im_name + '.png')
	fig.savefig(save_dir + im_name + '.svg')
	plt.close(fig)
	#Log scale
	fig = plt.figure(figsize=(5,5))
	for s_i in range(len(true_x_vals)):
		plt.plot(true_x_vals[s_i],np.log(true_spike_counts[s_i]),label=segment_names[s_i] + ' true',linestyle='solid',color=colors[s_i])
		plt.fill_between(null_x_vals[s_i], 
				   (np.log(np.array(mean_null_spike_counts[s_i])-std_null_spike_counts[s_i])), 
				  np.log((mean_null_spike_counts[s_i]+std_null_spike_counts[s_i])),
				  alpha=0.4,color=colors[s_i],label=segment_names[s_i] + ' null std')
		plt.plot(null_x_vals[s_i],np.log(mean_null_spike_counts[s_i]),linestyle='dotted',color=colors[s_i],label=segment_names[s_i] + ' null mean')
	plt.legend()
	plt.xlabel(plot_name)
	plt.ylabel('Number of Instances')
	plt.title(plot_name)
	plt.tight_layout()
	im_name = plot_name.replace(' ','_') + '_log_truexnull_all'
	fig.savefig(save_dir + im_name + '.png')
	fig.savefig(save_dir + im_name + '.svg')
	plt.close(fig)
	#Normalized data
	fig = plt.figure(figsize=(5,5))
	for s_i in range(len(true_x_vals)):
		plt.plot(true_x_vals[s_i],true_spike_counts[s_i]/norm_val,label=segment_names[s_i] + ' true',linestyle='solid',color=colors[s_i])
		plt.fill_between(null_x_vals[s_i], 
				   (np.array(mean_null_spike_counts[s_i])-std_null_spike_counts[s_i])/norm_val, 
				  (mean_null_spike_counts[s_i]+std_null_spike_counts[s_i])/norm_val,
				  alpha=0.4,color=colors[s_i],label=segment_names[s_i] + ' null std')
		plt.plot(null_x_vals[s_i],mean_null_spike_counts[s_i]/norm_val,linestyle='dotted',color=colors[s_i],label=segment_names[s_i] + ' null mean')
	plt.legend()
	plt.xlabel(plot_name)
	plt.ylabel('Number of Instances')
	plt.title(plot_name)
	plt.tight_layout()
	im_name = plot_name.replace(' ','_') + '_norm_truexnull_all'
	fig.savefig(save_dir + im_name + '.png')
	fig.savefig(save_dir + im_name + '.svg')
	plt.close(fig)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:10:12 2023

@author: Hannah Germaine
Functions to plot deviation stats and the like
"""
import os, tqdm, itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, ks_2samp, ttest_ind, kruskal
import warnings

def plot_dev_rasters(segment_deviations,segment_spike_times,segment_dev_times,
					 segment_times_reshaped,pre_taste,post_taste,segment_names,dev_dir):
	num_segments = len(segment_names)
	dev_buffer = 100 #ms before and after a deviation to plot
	for s_i in range(num_segments):
		print("Plotting deviations in segment " + segment_names[s_i])
		filepath = dev_dir + segment_names[s_i] + '/'
		indiv_dev_filepath = filepath + 'indiv_dev/'
		if os.path.isdir(indiv_dev_filepath) == False:
			os.mkdir(indiv_dev_filepath)
		#Plot when deviations occur
		f = plt.figure(figsize=(5,5))
		plt.plot(segment_deviations[s_i])
		plt.title('Segment ' + segment_names[s_i] + ' deviations')
		x_ticks = plt.xticks()[0]
		x_tick_labels = [np.round(x_ticks[i]/1000/60,2) for i in range(len(x_ticks))]
		plt.xticks(x_ticks,x_tick_labels)
		plt.xlabel('Time (min)')
		plt.yticks([0,1],['No Dev','Dev'])
		plt.tight_layout()
		fig_name = filepath + 'all_deviations'
		f.savefig(fig_name + '.png')
		f.savefig(fig_name + '.svg')
		plt.close(f)
		#Plot individual segments with pre and post time
		segment_rasters = segment_spike_times[s_i]
		segment_times = segment_dev_times[s_i] + segment_times_reshaped[s_i][0]
		num_neur = len(segment_rasters)
		num_deviations = len(segment_times[0,:])
		for dev_i in tqdm.tqdm(range(num_deviations)):
			dev_times = segment_times[:,dev_i]
			dev_start = int(dev_times[0])
			dev_len = dev_times[1] - dev_start
			dev_rast_ind = []
			raster_len = 2*dev_buffer + dev_len
			dev_binary = np.zeros((num_neur,raster_len))
			for n_i in range(num_neur):
				segment_neur_rast = np.array(segment_rasters[n_i])
				seg_dev_ind = np.where((segment_neur_rast > dev_start - dev_buffer)*(segment_neur_rast < dev_times[1] + dev_buffer))[0]
				seg_dev_rast_ind = segment_neur_rast[seg_dev_ind]
				seg_dev_rast_ind_shift = (seg_dev_rast_ind - dev_start + dev_buffer).astype('int')
				dev_binary[n_i,seg_dev_rast_ind_shift] = 1
				dev_rast_ind.append(seg_dev_rast_ind)
			#Create firing rates matrix
			firing_rate_vec = np.zeros(raster_len)
			for t_i in range(raster_len):
				min_t_i = max(t_i-25,0)
				max_t_i = min(t_i+25,raster_len)
				firing_rate_vec[t_i] = np.mean(np.sum(dev_binary[:,min_t_i:max_t_i],1)/(50/1000))
			rate_x_tick_labels = np.arange(-1*dev_buffer,dev_len + dev_buffer)
			#Now plot the rasters with firing rate deviations
			f1, ax1 = plt.subplots(nrows=2,ncols=1,figsize=(5,5),gridspec_kw=dict(height_ratios=[2,1]))
			#Deviation Raster Plot
			adjusted_dev_rast_ind = [list(np.array(dev_rast_ind[n_i]) - dev_start) for n_i in range(num_neur)]
			ax1[0].eventplot(adjusted_dev_rast_ind,colors='b',alpha=0.5)
			ax1[0].axvline(0)
			ax1[0].axvline(dev_len)
			ax1[0].set_ylabel('Neuron Index')
			ax1[0].set_title('Deviation ' + str(dev_i))
			#Deviation population activity plot
			ax1[1].plot(rate_x_tick_labels,firing_rate_vec)
			ax1[1].axvline(0)
			ax1[1].axvline(dev_len)
			ax1[1].set_xlabel('Time (ms)')
			ax1[1].set_ylabel('Population rate (Hz)')
			fig_name = indiv_dev_filepath + 'dev_' + str(dev_i)
			f1.savefig(fig_name + '.png')
			f1.savefig(fig_name + '.svg')
			plt.close(f1)
		

def plot_dev_stats(data, data_name, save_dir, x_label=[], y_label=[]):
	"""General function to plot given statistics"""
	plt.figure(figsize=(5, 5))
	# Plot the trend
	plt.subplot(2, 1, 1)
	plt.plot(data)
	if len(x_label) > 0:
		plt.xlabel(x_label)
	else:
		plt.xlabel('Deviation Index')
	if len(y_label) > 0:
		plt.ylabel(y_label)
	else:
		plt.ylabel(data_name)
	plt.title(data_name + ' trend')
	# Plot the histogram
	plt.subplot(2, 1, 2)
	plt.hist(data)
	if len(y_label) > 0:
		plt.xlabel(y_label)
	else:
		plt.xlabel(data_name)
	plt.ylabel('Number of Occurrences')
	plt.title(data_name + ' histogram')
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '.png')
	plt.savefig(save_dir + im_name + '.svg')
	plt.close()


def plot_dev_stats_dict(dict_data, iteration_names, data_name, save_dir, x_label, y_label):
	labels, data = [*zip(*dict_data.items())]
	#Calculate pairwise significant differences
	x_ticks = np.arange(1,len(labels)+1)
	pairs = list(itertools.combinations(x_ticks,2))
	pair_sig = np.zeros(len(pairs))
	#cross-group stat sig
	args = [d for d in data]
	try:
		kw_stat, kw_p_val = kruskal(*args,nan_policy='omit')
	except:
		kw_p_val = 1
	#pairwise stat sig
	for pair_i in range(len(pairs)):
		pair = pairs[pair_i]
		ks_pval = ks_2samp(data[pair[0]-1],data[pair[1]-1])[1]
		if ks_pval < 0.05:
			pair_sig[pair_i] = 1
	
	#Plot distributions as box and whisker plots comparing across iterations
	bw_fig = plt.figure(figsize=(5,5))
	plt.boxplot(data)
	plt.xticks(x_ticks,labels=iteration_names)
	y_ticks = plt.yticks()[0]
	y_max = np.max(y_ticks)
	y_range = y_max - np.min(y_ticks)
	x_mean = np.mean(x_ticks)
	if kw_p_val <= 0.05:
		plt.plot([x_ticks[0],x_ticks[-1]],[y_max+0.05*y_range,y_max+0.05*y_range],color='k')
		plt.scatter(x_mean,y_max+0.1*y_range,marker='*',color='k')
	jitter_vals = np.linspace(0.9*y_max,y_max,len(pairs))
	step = np.mean(np.diff(jitter_vals))
	for pair_i in range(len(pairs)):
		pair = pairs[pair_i]
		if pair_sig[pair_i] == 1:
			plt.plot([pair[0],pair[1]],[jitter_vals[pair_i],jitter_vals[pair_i]],color='k',linestyle='dashed')
			plt.scatter((pair[0]+pair[1])/2,jitter_vals[pair_i]+step,marker='*',color='k')
	plt.title(data_name)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '_box.png')
	plt.savefig(save_dir + im_name + '_box.svg')
	plt.close(bw_fig)
	#Plot distributions as violin plots
	violin_fig = plt.figure(figsize=(5,5))
	plt.violinplot(data,positions=np.arange(1,len(labels)+1))
	plt.xticks(range(1,len(labels)+1),labels=iteration_names)
	y_ticks = plt.yticks()[0]
	y_max = np.max(y_ticks)
	y_range = y_max - np.min(y_ticks)
	x_mean = np.mean(x_ticks)
	if kw_p_val <= 0.05:
		plt.plot([x_ticks[0],x_ticks[-1]],[y_max+0.05*y_range,y_max+0.05*y_range],color='k')
		plt.scatter(x_mean,y_max+0.1*y_range,marker='*',color='k')
	jitter_vals = np.linspace(0.9*y_max,y_max,len(pairs))
	step = np.mean(np.diff(jitter_vals))
	for pair_i in range(len(pairs)):
		pair = pairs[pair_i]
		if pair_sig[pair_i] == 1:
			plt.plot([pair[0],pair[1]],[jitter_vals[pair_i],jitter_vals[pair_i]],color='k',linestyle='dashed')
			plt.scatter((pair[0]+pair[1])/2,jitter_vals[pair_i]+step,marker='*',color='k')
	plt.title(data_name)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '_violin.png')
	plt.savefig(save_dir + im_name + '_violin.svg')
	plt.close(violin_fig)
	#Plot distributions as PDFs
	pdf_fig = plt.figure(figsize=(3,3))
	for d_i in range(len(data)):
		plt.hist(data[d_i],label=iteration_names[d_i],density=True,histtype='step')
	plt.legend()
	plt.title(data_name)
	plt.xlabel(y_label)
	plt.ylabel('Probability')
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '_pdf.png')
	plt.savefig(save_dir + im_name + '_pdf.svg')
	plt.close(pdf_fig)
	#Plot distributions as CDFs
	cdf_fig = plt.figure(figsize=(3,3))
	for d_i in range(len(data)):
		plt.hist(data[d_i],bins=np.arange(0,np.max(data[d_i])),label=iteration_names[d_i],density=True,cumulative=True,histtype='step')
	plt.legend()
	plt.title(data_name)
	plt.xlabel(y_label)
	plt.ylabel('Probability')
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '_cdf.png')
	plt.savefig(save_dir + im_name + '_cdf.svg')
	plt.close(cdf_fig)


def plot_null_v_true_stats(true_data, null_data, data_name, save_dir, x_label=[]):
	"""General function to plot given null and true statistics
	true_data is given as a numpy array
	null_data is given as a dictionary with keys = null index and values = numpy arrays
	"""
	plt.figure(figsize=(5, 5))
	null_vals = []
	null_x_vals = []
	for key in null_data.keys():
		null_vals.extend(list(null_data[key]))
		null_x_vals.extend([int(key)])
	mean_null_vals = np.mean(null_vals)
	std_null_vals = np.std(null_vals)
	# Plot the histograms
	plt.subplot(3, 1, 1)
	plt.hist(true_data, bins=25, color='b', alpha=0.4, label='True')
	plt.hist(null_vals, bins=25, color='g', alpha=0.4, label='Null')
	plt.legend()
	plt.xlabel(x_label)
	plt.ylabel('Number of Occurrences')
	plt.title(data_name + ' histogram')
	# Plot the probability distribution functions
	plt.subplot(3, 1, 2)
	plt.hist(true_data, bins=25, density=True,
			 histtype='step', color='b', label='True')
	plt.hist(null_vals, bins=25, density=True,
			 histtype='step', color='g', label='Null')
	plt.legend()
	plt.xlabel(x_label)
	plt.ylabel('PDF')
	plt.title(data_name + ' PDF')
	# Plot the cumulative distribution functions
	plt.subplot(3, 1, 3)
	true_sort = np.sort(true_data)
	true_unique = np.unique(true_sort)
	cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
	null_sort = np.sort(null_vals)
	null_unique = np.unique(null_sort)
	cmf_null = np.array([np.sum((null_sort <= u_val).astype('int'))/len(null_sort) for u_val in null_unique])
	plt.plot(true_unique,cmf_true, color='b', label='True')
	plt.plot(null_unique,cmf_null, color='g', label='Null')
	plt.legend()
	plt.xlabel(x_label)
	plt.ylabel('CDF')
	plt.title(data_name + ' CDF')
	plt.tight_layout()
	im_name = ('_').join(data_name.split(' '))
	plt.savefig(save_dir + im_name + '_truexnull.png')
	plt.savefig(save_dir + im_name + '_truexnull.svg')
	plt.close()


def plot_stats(dev_stats, segment_names, dig_in_names, save_dir, dist_name, 
			   neuron_indices):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval. Outputs are saved .npy files with name indicating
	segment and taste containing matrices of shape [num_dev, num_deliv, num_neur, num_cp]
	with the correlations stored.
	
	neuron_indices should be binary and shaped num_neur x num_cp
	"""
	warnings.filterwarnings('ignore')
	#Grab parameters
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	for s_i in range(num_segments):  #Loop through each segment
		print("Beginning plot calcs for segment " + str(s_i))
		seg_stats = dev_stats[s_i]
		for t_i in range(num_tastes):  #Loop through each taste
			print("\tTaste #" + str(t_i + 1))
			taste_stats = seg_stats[t_i]
			#_____Individual Neuron CP Calculations_____
			#Import correlation numpy array
			neuron_data_storage = np.abs(taste_stats['neuron_data_storage'])
			data_shape = np.shape(neuron_data_storage)
			if len(data_shape) == 4:
				num_dev = data_shape[0]
				num_deliv = data_shape[1]
				total_num_neur = data_shape[2]
				num_cp = data_shape[3]
			elif len(data_shape) == 3:
				num_dev = data_shape[0]
				num_deliv = data_shape[1]
				num_cp = data_shape[2]
			if total_num_neur != np.shape(neuron_indices)[0]: #accounts for sub-population calculation case
				neuron_indices = np.ones((total_num_neur,num_cp))
			#Plot the individual neuron distribution for each changepoint index
			f = plt.figure(figsize=(5,5))
			cp_data = []
			plt.subplot(2,1,1)
			for c_p in range(num_cp):
				all_dist_cp = (neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p]).flatten()
				cp_data.append(all_dist_cp)
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_dist_cp = cp_data[c_p]
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i]
			f.savefig(filename + '.png')
			f.savefig(filename + '.svg')
			plt.close(f)
			if dist_name == 'Correlation':
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				min_y = 1
				for c_p in range(num_cp):
					all_dist_cp = (neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p]).flatten()
					cp_data.append(all_dist_cp)
					hist_vals = plt.hist(all_dist_cp,density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
				if min_y == 1:
					min_y = 0.5
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					max_x_val = np.max(np.abs(hist_vals[1]))
					half_max_x = np.floor(max_x_val/2)
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y
					if half_max_x < min_x:
						min_x = half_max_x
					if max_x_val > max_x:
						max_x = max_x_val
				if min_y == 1:
					min_y = 0.5
				plt.xlabel(dist_name)
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				for c_p in range(num_cp):
					all_dist_cp = (neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p]).flatten()
					cp_data.append(all_dist_cp)
					plt.hist(all_dist_cp,density=True,log=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					max_x_val = np.max(np.abs(hist_vals[1]))
					half_max_x = np.floor(max_x_val/2)
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
					if half_max_x < min_x:
						min_x = half_max_x
					if max_x_val > max_x:
						max_x = max_x_val
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_log_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
			#Plot the population average distribution for each changepoint index
			f1 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			avg_cp_data = []
			for c_p in range(num_cp):
				all_avg_dist_cp = (np.nanmean(neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p],2)).flatten()
				avg_cp_data.append(all_avg_dist_cp)
				plt.hist(all_avg_dist_cp[~np.isnan(all_avg_dist_cp)],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel('Average ' + dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_avg_dist_cp = avg_cp_data[c_p]
				plt.hist(all_avg_dist_cp[~np.isnan(all_avg_dist_cp)],bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
				#plt.ecdf(all_avg_dist_cp,label='Epoch ' + str(c_p))
				#true_sort = np.sort(all_avg_dist_cp)
				#true_unique = np.unique(true_sort)
				#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
				#plt.plot(true_unique,cmf_true,label='Epoch ' + str(c_p))
			plt.xlabel('Average ' + dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle(dist_name + ' avg population distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_avg_pop'
			f1.savefig(filename + '.png')
			f1.savefig(filename + '.svg')
			plt.close(f1)
			if dist_name == 'Correlation':
				#Zoomed to high correlation
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				avg_cp_data = []
				for c_p in range(num_cp):
					all_avg_dist_cp = (np.nanmean(neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p],2)).flatten()
					avg_cp_data.append(all_avg_dist_cp)
					plt.hist(all_avg_dist_cp[all_avg_dist_cp >= 0.5],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel('Average ' + dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for c_p in range(num_cp):
					all_avg_dist_cp = avg_cp_data[c_p]
					hist_vals = plt.hist(all_avg_dist_cp,bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					max_x_val = np.max(np.abs(hist_vals[1]))
					half_max_x = np.floor(max_x_val/2)
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y
					if half_max_x < min_x:
						min_x = half_max_x
					if max_x_val > max_x:
						max_x = max_x_val
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.xlabel('Average ' + dist_name)
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' avg population distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_avg_pop_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Now with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				avg_cp_data = []
				for c_p in range(num_cp):
					all_avg_dist_cp = (np.nanmean(neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p],2)).flatten()
					avg_cp_data.append(all_avg_dist_cp)
					plt.hist(all_avg_dist_cp[all_avg_dist_cp >= 0.5],density=True,log=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel('Average ' + dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for c_p in range(num_cp):
					all_avg_dist_cp = avg_cp_data[c_p]
					hist_vals = plt.hist(all_avg_dist_cp,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					max_x_val = np.max(np.abs(hist_vals[1]))
					half_max_x = np.floor(max_x_val/2)
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
					if half_max_x < min_x:
						min_x = half_max_x
					if max_x_val > max_x:
						max_x = max_x_val
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.ylim([min_y,1])
				plt.xlabel('Average ' + dist_name)
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' avg population distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_avg_pop_log_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
			
			#_____Population CP Calculations_____
			#Import correlation numpy array
			pop_data_storage = taste_stats['pop_data_storage']
			data_shape = np.shape(pop_data_storage)
			if len(data_shape) == 3:
				num_dev = data_shape[0]
				num_deliv = data_shape[1]
				num_cp = data_shape[2]
			
			#Plot the individual neuron distribution for each changepoint index
			f = plt.figure(figsize=(5,5))
			cp_data = []
			plt.subplot(2,1,1)
			for c_p in range(num_cp):
				all_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
				cp_data.append(all_dist_cp)
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_dist_cp = cp_data[c_p]
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop'
			f.savefig(filename + '.png')
			f.savefig(filename + '.svg')
			plt.close(f)
			if dist_name == 'Correlation':
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for c_p in range(num_cp):
					all_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
					cp_data.append(all_dist_cp)
					hist_vals = plt.hist(all_dist_cp,density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
					max_x_val = np.max(np.abs(hist_vals[1]))
					half_max_x = np.floor(max_x_val/2)
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
					if half_max_x < min_x:
						min_x = half_max_x
					if max_x_val > max_x:
						max_x = max_x_val
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					max_x_val = np.max(np.abs(hist_vals[1]))
					half_max_x = np.floor(max_x_val/2)
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
					if half_max_x < min_x:
						min_x = half_max_x
					if max_x_val > max_x:
						max_x = max_x_val
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				for c_p in range(num_cp):
					all_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
					cp_data.append(all_dist_cp)
					plt.hist(all_dist_cp,density=True,log=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					max_x_val = np.max(np.abs(hist_vals[1]))
					half_max_x = np.floor(max_x_val/2)
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
					if half_max_x < min_x:
						min_x = half_max_x
					if max_x_val > max_x:
						max_x = max_x_val
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_log_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
				
			#_____Population Vector CP Calculations_____
			#Import correlation numpy array
			pop_vec_data_storage = taste_stats['pop_vec_data_storage']
			data_shape = np.shape(pop_vec_data_storage)
			if len(data_shape) == 3:
				num_dev = data_shape[0]
				num_deliv = data_shape[1]
				num_cp = data_shape[2]
			#Plot the individual neuron distribution for each changepoint index
			f = plt.figure(figsize=(5,5))
			cp_data = []
			plt.subplot(2,1,1)
			for c_p in range(num_cp):
				all_dist_cp = (pop_vec_data_storage[:,:,c_p]).flatten()
				cp_data.append(all_dist_cp)
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for c_p in range(num_cp):
				all_dist_cp = cp_data[c_p]
				plt.hist(all_dist_cp[~np.isnan(all_dist_cp)],bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec'
			f.savefig(filename + '.png')
			f.savefig(filename + '.svg')
			plt.close(f)
			if dist_name == 'Correlation':
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for c_p in range(num_cp):
					all_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
					cp_data.append(all_dist_cp)
					hist_vals = plt.hist(all_dist_cp,density=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
					max_x_val = np.max(np.abs(hist_vals[1]))
					half_max_x = np.floor(max_x_val/2)
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
					if half_max_x < min_x:
						min_x = half_max_x
					if max_x_val > max_x:
						max_x = max_x_val
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					max_x_val = np.max(np.abs(hist_vals[1]))
					half_max_x = np.floor(max_x_val/2)
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
					if half_max_x < min_x:
						min_x = half_max_x
					if max_x_val > max_x:
						max_x = max_x_val
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
				#Plot the individual neuron distribution for each changepoint index
				f = plt.figure(figsize=(5,5))
				cp_data = []
				plt.subplot(2,1,1)
				for c_p in range(num_cp):
					all_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
					cp_data.append(all_dist_cp)
					plt.hist(all_dist_cp,density=True,log=True,cumulative=False,histtype='step',label='Epoch ' + str(c_p))
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for c_p in range(num_cp):
					all_dist_cp = cp_data[c_p]
					hist_vals = plt.hist(all_dist_cp,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Epoch ' + str(c_p))
					max_x_val = np.max(np.abs(hist_vals[1]))
					half_max_x = np.floor(max_x_val/2)
					bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
					bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
					if bin_min_y < min_y:
						min_y = bin_min_y 
					if half_max_x < min_x:
						min_x = half_max_x
					if max_x_val > max_x:
						max_x = max_x_val
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle(dist_name + ' distributions for \nsegment ' + segment_names[s_i] + ', taste ' + dig_in_names[t_i])
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec_log_zoom'
				f.savefig(filename + '.png')
				f.savefig(filename + '.svg')
				plt.close(f)
			

def plot_combined_stats(dev_stats, segment_names, dig_in_names, save_dir, 
						dist_name, neuron_indices):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval. Outputs are saved .npy files with name indicating
	segment and taste containing matrices of shape [num_dev, num_deliv, num_neur, num_cp]
	with the distances stored.
	
	neuron_indices should be binary and shaped num_neur x num_cp
	"""
	warnings.filterwarnings('ignore')
	#Grab parameters
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	
	#Define storage
	segment_data = [] #segments x tastes x cp
	segment_pop_data = [] #segments x tastes x cp from fr population matrix
	segment_data_avg = [] #segments x tastes x cp avged across neurons in the deviation
	segment_pop_vec_data = [] #segments x tastes x cp from fr population vector
	for s_i in range(num_segments):  #Loop through each segment
		seg_stats = dev_stats[s_i]
		print("Beginning combined plot calcs for segment " + str(s_i))
		taste_data = [] #tastes x cp
		taste_pop_data = [] #tastes x cp
		taste_data_avg = []
		taste_pop_vec_data = []
		for t_i in range(num_tastes):  #Loop through each taste
			print("\tTaste #" + str(t_i + 1))
			taste_stats = seg_stats[t_i]
			#Import distance numpy array
			neuron_data_storage = taste_stats['neuron_data_storage']
			pop_data_storage = taste_stats['pop_data_storage']
			pop_vec_data_storage = taste_stats['pop_vec_data_storage']
			num_dev, num_deliv, total_num_neur, num_cp = np.shape(neuron_data_storage)
			if total_num_neur != np.shape(neuron_indices)[0]: #accounts for sub-population calculation case
				neuron_indices = np.ones((total_num_neur,num_cp))
			cp_data = []
			cp_data_pop = []
			cp_data_avg = []
			cp_data_pop_vec = []
			for c_p in range(num_cp):
				all_dist_cp = (neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p]).flatten()
				all_pop_dist_cp = (pop_data_storage[:,:,c_p]).flatten()
				cp_data.append(all_dist_cp)
				cp_data_pop.append(all_pop_dist_cp)
				avg_dist_cp = np.nanmean(neuron_data_storage[:,:,neuron_indices[:,c_p].astype('bool'),c_p],2).flatten()
				cp_data_avg.append(avg_dist_cp)
				all_pop_vec_cp = (pop_vec_data_storage[:,:,c_p]).flatten()
				cp_data_pop_vec.append(all_pop_vec_cp)
			taste_data.append(cp_data)
			taste_pop_data.append(cp_data_pop)
			taste_data_avg.append(cp_data_avg)
			taste_pop_vec_data.append(cp_data_pop_vec)
		segment_data.append(taste_data)
		segment_pop_data.append(taste_pop_data)
		segment_data_avg.append(taste_data_avg)
		segment_pop_vec_data.append(taste_pop_vec_data)
		#Plot taste data against each other
		for c_p in range(num_cp):
			#_____Individual Neurons_____
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
					#true_sort = np.sort(data)
					#true_unique = np.unique(true_sort)
					#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
					#plt.plot(true_unique,cmf_true,label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p)
			f0.savefig(filename + '.png')
			f0.savefig(filename + '.svg')
			plt.close(f0)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				#Plot data across all neurons
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Plot data across all neurons with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_log_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
			#_____Population Data_____
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_pop_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_pop_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_pop'
			f0.savefig(filename + '.png')
			f0.savefig(filename + '.svg')
			plt.close(f0)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				#Plot data across all neurons
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_pop_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_pop_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_zoom_pop'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Plot data across all neurons with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_pop_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_pop_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_log_zoom_pop'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
			#_____Population Vector Data
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_pop_vec_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_pop_vec_data[t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_pop_vec'
			f0.savefig(filename + '.png')
			f0.savefig(filename + '.svg')
			plt.close(f0)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				#Plot data across all neurons
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_pop_vec_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_pop_vec_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_zoom_pop_vec'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Plot data across all neurons with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_pop_vec_data[t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_pop_vec_data[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_log_zoom_pop_vec'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
		#Now plot individual neuron data population averages
		for c_p in range(num_cp):
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_data_avg[t_i][c_p]
					plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_data_avg[t_i][c_p]
					plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Averaged Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_avg_neur'
			f0.savefig(filename + '.png')
			f0.savefig(filename + '.svg')
			plt.close(f0)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				#Plot data across all neurons
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Averaged Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p)  + '_avg_neur_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Plot data across all neurons with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y  
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Averaged Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p)  + '_avg_neur_log_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
			
	for t_i in range(num_tastes): #Loop through each taste
		#Plot segment data against each other by epoch
		#_____Individual Neuron Averages_____
		for c_p in range(num_cp):
			f2 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for s_i in range(num_segments):
				try:
					data = segment_data_avg[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for s_i in range(num_segments):
				try:
					data = segment_data_avg[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_neur_avg'
			f2.savefig(filename + '.png')
			f2.savefig(filename + '.svg')
			plt.close(f2)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y  
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y  
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_neur_avg_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
				#Log
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_neur_avg_log_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
				
		#_____Population Data_____
		for c_p in range(num_cp):
			f2 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for s_i in range(num_segments):
				try:
					data = segment_pop_data[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for s_i in range(num_segments):
				try:
					data = segment_pop_data[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop'
			f2.savefig(filename + '.png')
			f2.savefig(filename + '.svg')
			plt.close(f2)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_pop_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_pop_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
				#Log
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_pop_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_pop_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_log_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
		
		#_____Population Vec Data_____
		for c_p in range(num_cp):
			f2 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for s_i in range(num_segments):
				try:
					data = segment_pop_vec_data[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for s_i in range(num_segments):
				try:
					data = segment_pop_vec_data[s_i][t_i][c_p]
					plt.hist(data[~np.isnan(data)],bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_vec'
			f2.savefig(filename + '.png')
			f2.savefig(filename + '.svg')
			plt.close(f2)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_pop_vec_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_pop_vec_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_vec_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
				#Log
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_pop_vec_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				min_y = 1
				min_x = 1
				max_x = 0
				for s_i in range(num_segments):
					try:
						data = segment_pop_vec_data[s_i][t_i][c_p]
						hist_vals = plt.hist(data,bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						max_x_val = np.max(np.abs(hist_vals[1]))
						half_max_x = np.floor(max_x_val/2)
						bin_05 = np.argmin(np.abs(hist_vals[1] - 0.5))
						bin_min_y = hist_vals[0][np.max(bin_05-1,0)]
						if bin_min_y < min_y:
							min_y = bin_min_y 
						if half_max_x < min_x:
							min_x = half_max_x
						if max_x_val > max_x:
							max_x = max_x_val
					except:
						pass
				if min_y == 1:
					min_y = 0.5
				if max_x_val < 0.5:
					plt.xlim([min_x,max_x])
				else:
					plt.xlim([0.5,1])
				plt.xlabel(dist_name)
				plt.ylim([min_y,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_vec_log_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)
		
	return segment_data, segment_data_avg, segment_pop_data, segment_pop_vec_data
		


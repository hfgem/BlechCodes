#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:38:33 2023

@author: Hannah Germaine
deviation functions for population analyses
"""
import tqdm, itertools
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import functions.corr_dist_calc_parallel_pop as cdcpp


def calculate_correlations_pop(segment_dev_rasters, tastant_spike_times,
						   start_dig_in_times, end_dig_in_times, segment_names, dig_in_names,
						   pre_taste, post_taste, taste_cp_raster_inds, save_dir,
						   neuron_keep_indices=[]):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval"""

	#Grab parameters
	fr_bin = 25 #ms to bin together for number of spikes 'fr'
	num_tastes = len(start_dig_in_times)
	num_segments = len(segment_dev_rasters)
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
	post_taste_dt = np.ceil(post_taste*1000).astype('int')

	for s_i in range(num_segments):  #Loop through each segment
		print("Beginning correlation calcs for segment " + str(s_i))
		#Gather segment data
		seg_rast = segment_dev_rasters[s_i]
		num_dev = len(seg_rast)
		if len(neuron_keep_indices) == 0:
			total_num_neur = np.shape(seg_rast[0])[0]
			neuron_keep_indices = np.arange(total_num_neur)
		else:
			total_num_neur = len(neuron_keep_indices)
			
		for t_i in range(num_tastes):  #Loop through each taste
		
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
			try:
				neuron_corr_storage = np.load(filename)
				print("\tTaste #" + str(t_i + 1) + 'previously calculated')
			except:
				print("\tCalculating Taste #" + str(t_i + 1))
				taste_cp = taste_cp_raster_inds[t_i][:, neuron_keep_indices, :]
				taste_spikes = tastant_spike_times[t_i]
				#Note, num_cp = num_cp+1 with the first value the taste delivery index
				num_deliv, _, num_cp = np.shape(taste_cp)
				taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i][deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
				deliv_adjustment = [start_dig_in_times[t_i][deliv_i] + pre_taste_dt for deliv_i in range(num_deliv)]
				num_deliv, _, num_cp = np.shape(taste_cp)
				#Store the correlation results in a numpy array
				neuron_corr_storage = np.zeros((num_dev, num_deliv, num_cp-1))
				for dev_i in tqdm.tqdm(range(num_dev)): #Loop through all deviations
					dev_rast = seg_rast[dev_i][neuron_keep_indices,:]
					dev_len = np.shape(dev_rast)[1]
					end_ind = np.arange(fr_bin,fr_bin+dev_len)
					end_ind[end_ind > dev_len] = dev_len
					#TODO: test gaussian convolution instead of binning
					dev_rast_binned = np.zeros(np.shape(dev_rast))
					for start_ind in range(dev_len):
						dev_rast_binned[:,start_ind] = np.sum(dev_rast[:,start_ind:end_ind[start_ind]],1)
					
					inputs = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
					 itertools.repeat(neuron_keep_indices), itertools.repeat(taste_cp), \
						 deliv_adjustment, itertools.repeat(dev_rast_binned), itertools.repeat(fr_bin))
					pool = Pool(4)
					deliv_corr_storage = pool.map(cdcpp.deliv_corr_population_parallelized, inputs)
					pool.close()
					neuron_corr_storage[dev_i,:,:] = np.array(deliv_corr_storage)
				
				#Save to a numpy array
				filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
				np.save(filename,neuron_corr_storage)


def pull_corr_dev_stats_pop(segment_names, dig_in_names, save_dir):
	"""For each epoch and each segment pull out the top 10 most correlated deviation 
	bins and plot side-by-side with the epoch they are correlated with"""
	
	#Grab parameters
	dev_stats = dict()
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	for s_i in range(num_segments):  #Loop through each segment
		segment_stats = dict()
		for t_i in range(num_tastes):  #Loop through each taste
			#Import distance numpy array
			filename = save_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
			neuron_data_storage = np.load(filename)
			#Calculate statistics
			data_dict = dict()
			data_dict['segment'] = segment_names[s_i]
			data_dict['taste'] = dig_in_names[t_i]
			num_dev, num_deliv, num_cp = np.shape(neuron_data_storage)
			data_dict['num_dev'] = num_dev
			data_dict['neuron_data_storage'] = neuron_data_storage
			segment_stats[t_i] = data_dict
		dev_stats[s_i] = segment_stats

	return dev_stats

def plot_combined_stats_pop(dev_stats, segment_names, dig_in_names, save_dir, 
						dist_name):
	"""This function takes in deviation rasters, tastant delivery spikes, and
	changepoint indices to calculate correlations of each deviation to each 
	changepoint interval. Outputs are saved .npy files with name indicating
	segment and taste containing matrices of shape [num_dev, num_deliv, num_neur, num_cp]
	with the distances stored.
	
	neuron_indices should be binary and shaped num_neur x num_cp
	"""
	
	#Grab parameters
	num_tastes = len(dig_in_names)
	num_segments = len(segment_names)
	
	#Define storage
	segment_data = [] #segments x tastes x cp
	segment_data_avg = [] #segments x tastes x cp avged across neurons in the deviation
	for s_i in range(num_segments):  #Loop through each segment
		seg_stats = dev_stats[s_i]
		print("Beginning combined plot calcs for segment " + str(s_i))
		taste_data = [] #tastes x cp
		taste_data_avg = []
		for t_i in range(num_tastes):  #Loop through each taste
			print("\tTaste #" + str(t_i + 1))
			taste_stats = seg_stats[t_i]
			#Import distance numpy array
			neuron_data_storage = taste_stats['neuron_data_storage']
			num_dev, num_deliv, num_cp = np.shape(neuron_data_storage)
			cp_data = []
			cp_data_avg = []
			for c_p in range(num_cp):
				all_dist_cp = (neuron_data_storage[:,:,c_p]).flatten()
				cp_data.append(all_dist_cp)
				avg_dist_cp = np.nanmean(neuron_data_storage[:,:,c_p],2).flatten()
				cp_data_avg.append(avg_dist_cp)
			taste_data.append(cp_data)
			taste_data_avg.append(cp_data_avg)
		segment_data.append(taste_data)
		segment_data_avg.append(taste_data_avg)
		#Plot taste data against each other
		for c_p in range(num_cp):
			#Plot data across all neurons
			f0 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for t_i in range(num_tastes):
				try:
					data = taste_data[t_i][c_p]
					plt.hist(data[data != 0],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for t_i in range(num_tastes):
				try:
					data = taste_data[t_i][c_p]
					plt.hist(data[data != 0],bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
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
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						plt.hist(data[data >= 0.5],bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						#true_sort = np.sort(data)
						#true_unique = np.unique(true_sort)
						#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
						#plt.plot(true_unique[cmf_true >= 0.5],cmf_true[cmf_true >= 0.5],label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
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
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for t_i in range(num_tastes):
					try:
						data = taste_data[t_i][c_p]
						plt.hist(data[data >= 0.5],bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						#true_sort = np.sort(data)
						#true_unique = np.unique(true_sort)
						#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
						#plt.plot(true_unique[cmf_true >= 0.5],cmf_true[cmf_true >= 0.5],label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_log_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
		#Now plot population averages
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
			filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p) + '_avg_pop'
			f0.savefig(filename + '.png')
			f0.savefig(filename + '.svg')
			plt.close(f0)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				#Plot data across all neurons
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						plt.hist(data[data >= 0.5],bins=1000,density=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						#true_sort = np.sort(data)
						#true_unique = np.unique(true_sort)
						#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
						#plt.plot(true_unique[cmf_true >= 0.5],cmf_true[cmf_true >= 0.5],label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p)  + '_avg_pop' + '_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
				#Plot data across all neurons with log
				f1 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						plt.hist(data[data >= 0.5],density=True,log=True,cumulative=False,histtype='step',label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for t_i in range(num_tastes):
					try:
						data = taste_data_avg[t_i][c_p]
						plt.hist(data[data >= 0.5],bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Taste ' + dig_in_names[t_i])
						#true_sort = np.sort(data)
						#true_unique = np.unique(true_sort)
						#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
						#plt.plot(true_unique[cmf_true >= 0.5],cmf_true[cmf_true >= 0.5],label='Taste ' + dig_in_names[t_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Neuron Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + segment_names[s_i] + '_epoch' + str(c_p)  + '_avg_pop' + '_log_zoom'
				f1.savefig(filename + '.png')
				f1.savefig(filename + '.svg')
				plt.close(f1)
			
	for t_i in range(num_tastes): #Loop through each taste
		#Plot segment data against each other by epoch
		for c_p in range(num_cp):
			f2 = plt.figure(figsize=(5,5))
			plt.subplot(2,1,1)
			for s_i in range(num_segments):
				try:
					data = segment_data_avg[s_i][t_i][c_p]
					plt.hist(data[data != 0],density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Probability Mass Function - ' + dist_name)
			plt.subplot(2,1,2)
			for s_i in range(num_segments):
				try:
					data = segment_data_avg[s_i][t_i][c_p]
					plt.hist(data[data != 0],bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
					#true_sort = np.sort(data)
					#true_unique = np.unique(true_sort)
					#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
					#plt.plot(true_unique,cmf_true,label='Segment ' + segment_names[s_i])
				except:
					pass
			plt.xlabel(dist_name)
			plt.legend()
			plt.title('Cumulative Mass Function - ' + dist_name)
			plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
			plt.tight_layout()
			filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_avg'
			f2.savefig(filename + '.png')
			f2.savefig(filename + '.svg')
			plt.close(f2)
			#Zoom to correlations > 0.5
			if dist_name == 'Correlation':
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						plt.hist(data[data > 0.5],density=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						plt.hist(data[data > 0.5],bins=1000,density=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						#true_sort = np.sort(data)
						#true_unique = np.unique(true_sort)
						#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
						#plt.plot(true_unique[cmf_true >= 0.5],cmf_true[cmf_true >= 0.5],label='Segment ' + segment_names[s_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_avg_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
				#Log
				f3 = plt.figure(figsize=(5,5))
				plt.subplot(2,1,1)
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						plt.hist(data[data > 0.5],density=True,log=True,cumulative=False,histtype='step',label='Segment ' + segment_names[s_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Probability Mass Function - ' + dist_name)
				plt.subplot(2,1,2)
				for s_i in range(num_segments):
					try:
						data = segment_data_avg[s_i][t_i][c_p]
						plt.hist(data[data > 0.5],bins=1000,density=True,log=True,cumulative=True,histtype='step',label='Segment ' + segment_names[s_i])
						#true_sort = np.sort(data)
						#true_unique = np.unique(true_sort)
						#cmf_true = np.array([np.sum((true_sort <= u_val).astype('int'))/len(true_sort) for u_val in true_unique])
						#plt.plot(true_unique[cmf_true >= 0.5],cmf_true[cmf_true >= 0.5],label='Segment ' + segment_names[s_i])
					except:
						pass
				plt.xlabel(dist_name)
				plt.xlim([0.5,1])
				plt.legend()
				plt.title('Cumulative Mass Function - ' + dist_name)
				plt.suptitle('Population Avg Distributions for \n' + segment_names[s_i] + ' Epoch = ' + str(c_p + 1))
				plt.tight_layout()
				filename = save_dir + dig_in_names[t_i] + '_epoch' + str(c_p) + '_pop_avg_log_zoom'
				f3.savefig(filename + '.png')
				f3.savefig(filename + '.svg')
				plt.close(f3)	
	
		
	return segment_data, segment_data_avg
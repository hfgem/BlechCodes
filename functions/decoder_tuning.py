#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:38:30 2024

@author: hannahgermaine

File dedicated to functions related to testing bayesian decoder parameters for
best decoder outcomes.
"""

import numpy as np
import tqdm, os, itertools, time, csv
import matplotlib.pyplot as plt
from matplotlib import colormaps
from multiprocess import Pool
import functions.decode_parallel as dp
from sklearn.mixture import GaussianMixture as gmm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from random import choices, sample

def test_decoder_params(dig_in_names, start_dig_in_times, num_neur, tastant_spike_times,
						segment_spike_times, cp_raster_inds, pre_taste_dt, post_taste_dt, 
						epochs_to_analyze, segments_to_analyze, taste_select_neur,
						e_skip_dt, e_len_dt, save_dir):
	"""This function tests different decoder types to determine
	the best combination to use in replay decoding
	INPUTS:
		- dig_in_names: name of each taste deliveres
		- start_dig_in_times: start times of tastant deliveries
		
	OUTPUTS:
		- best_components: list of epoch-specific best number of components for gmm
	"""
	
	#Get trial indices for train/test sets
	num_tastes = len(tastant_spike_times)
	all_trial_inds = []
	for t_i in range(num_tastes):
		taste_trials = len(tastant_spike_times[t_i])
		all_trial_inds.append(list(np.arange(taste_trials)))
		
	del t_i, taste_trials
	
	#Collect firing rates for fitting gmms
	tastant_fr_dist, max_hz = taste_fr_dist_lho(num_neur, tastant_spike_times, cp_raster_inds,
					  start_dig_in_times, pre_taste_dt, post_taste_dt, all_trial_inds)
	
	#Run decoder through training and jackknife testing to determine success rates
	gmm_success_rates = run_decoder(num_neur, start_dig_in_times, tastant_fr_dist,
					all_trial_inds, tastant_spike_times, segment_spike_times,
					cp_raster_inds, pre_taste_dt, e_len_dt, e_skip_dt, 
					dig_in_names, max_hz, save_dir, epochs_to_analyze,
					segments_to_analyze, taste_select_neur)
	
	#Run Naive Bayes decoder to test
	nb_success_rates = naive_bayes_decoding(num_neur, tastant_spike_times, cp_raster_inds, 
							 all_trial_inds, dig_in_names, 
							 start_dig_in_times, pre_taste_dt, post_taste_dt, save_dir)
	
	
	

def taste_fr_dist_lho(num_neur, tastant_spike_times, cp_raster_inds,
				  start_dig_in_times, pre_taste_dt, post_taste_dt,
				  all_trial_inds):
	"""This function calculates firing rate distributions for a given set of 
	taste delivery trials to be used in decoding the remaining trials in order
	to determine decoder success
	INPUTS:
		- num_neur: number of neurons in dataset
		- tastant_spike_times: list of spike times for each tastant for each 
			delivery
		- cp_raster_inds: list of changepoint times for each tastant for each
			delivery
		- start_dig_in_times: times that tastes are delivered
		- pre_taste_dt: time before taste delivery that is pulled into
			cp_raster_inds
		- post_taste_dt: time after taste delivery
		- all_trial_inds: trial indices by taste
	OUTPUTS:
		- tastant_fr_dist: collected firing rate distributions for each 
			tastant to use in fitting models
		- max_hz: maximum firing rate collected to use in fitting gmms
	"""	
	
	num_tastes = len(tastant_spike_times)
	num_cp = np.shape(cp_raster_inds[0])[-1] - 1
	
	#Collect the number of deliveries per taste
	taste_num_deliv = np.zeros(num_tastes).astype('int')
	for t_i in range(num_tastes):
		num_deliv = len(all_trial_inds[t_i])
		taste_num_deliv[t_i] = int(num_deliv)
		
	#Set up distribution storage dictionary
	tastant_fr_dist = dict() #Population firing rate distributions by epoch
	for t_i in range(num_tastes):
		tastant_fr_dist[t_i] = dict()
		for d_i in range(taste_num_deliv[t_i]):
			tastant_fr_dist[t_i][d_i] = dict()
			for cp_i in range(num_cp):
				tastant_fr_dist[t_i][d_i][cp_i] = dict()
	
	#Collect firing rates for distribution fits
	max_hz = 0
	for t_i in range(num_tastes):
		num_deliv = int(taste_num_deliv[t_i])
		taste_cp = cp_raster_inds[t_i]
		for d_i, trial_ind in enumerate(all_trial_inds[t_i]):
			#grab spiking information
			raster_times = tastant_spike_times[t_i][trial_ind] #length num_neur list of lists
			start_taste_i = start_dig_in_times[t_i][trial_ind]
			deliv_cp = taste_cp[trial_ind,:] - pre_taste_dt
			#Create a binary matrix of spiking
			times_post_taste = [(np.array(raster_times[n_i])[np.where((raster_times[n_i] >= start_taste_i)*(raster_times[n_i] < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int') for n_i in range(num_neur)]
			bin_post_taste = np.zeros((num_neur,post_taste_dt))
			for n_i in range(num_neur):
				bin_post_taste[n_i,times_post_taste[n_i]] += 1
			#Calculate binned firing rate vectors
			for cp_i in range(num_cp):
				#population changepoints
				start_epoch = int(deliv_cp[cp_i])
				end_epoch = int(deliv_cp[cp_i+1])
				epoch_len = end_epoch - start_epoch
				if epoch_len > 0: 
					all_hz_bst = []
					for binsize in np.arange(100,epoch_len):
						bin_edges = np.arange(start_epoch,end_epoch,binsize).astype('int') #bin the epoch
						if len(bin_edges) != 0:
							if (bin_edges[-1] != end_epoch)*(end_epoch-bin_edges[-1]>10):
								bin_edges = np.concatenate((bin_edges,end_epoch*np.ones(1).astype('int')))
							bst_hz = np.array([np.sum(bin_post_taste[:,bin_edges[b_i]:bin_edges[b_i+1]],1)/((bin_edges[b_i+1] - bin_edges[b_i])*(1/1000)) for b_i in range(len(bin_edges)-1)])
							all_hz_bst.extend(list(bst_hz)) #nxnum_neur transposed
					all_hz_bst = np.array(all_hz_bst) #all_nxnum_neur
					#Store the firing rate vectors
					tastant_fr_dist[t_i][d_i][cp_i] = all_hz_bst.T #num_neurxall_n
					#Store maximum firing rate
					if np.max(all_hz_bst) > max_hz:
						max_hz = np.max(all_hz_bst)
		
	return tastant_fr_dist, max_hz
	
def run_decoder(num_neur, start_dig_in_times, tastant_fr_dist, all_trial_inds,
				tastant_spike_times, segment_spike_times, cp_raster_inds,
				pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names, max_hz, 
				save_dir, epochs_to_analyze = [], segments_to_analyze = []):
	"""This function runs a decoder with a given set of parameters and returns
	the decoding probabilities of taste delivery periods
	INPUTS:
		- num_neur: number of neurons in dataset
		- start_dig_in_times: times of taste deliveries
		- tastant_fr_dist: firing rate distribution to fit over (train set)
		- trial_test_inds: indices of trials used in testing the fit
		- tastant_spike_times: spike times for each tastant delivery
		- cp_raster_inds: changepoint times for all taste deliveries
		- pre_taste_dt: ms before taste delivery in cp_raster_inds
		- e_len_dt: decoding chunk length
		- e_skip_dt: decoding skip length
		- dig_in_names: taste names
		- max_hz: maximum firing rate of train data
		- save_dir: directory where to save results
		- epochs_to_analyze: array of which epochs to analyze
		- segments_to_analyze: array of which segments to analyze
	OUTPUTS:
		- 
	"""
	#TODO: Handle taste selective neurons
	#Variables
	num_tastes = len(start_dig_in_times)
	num_cp = len(tastant_fr_dist[0][0])
	p_taste = np.ones(num_tastes)/num_tastes #P(taste)
	num_segments = len(segment_spike_times)
	cmap = colormaps['jet']
	taste_colors = cmap(np.linspace(0, 1, num_tastes))
	
	#Jackknife decoding total number of trials
	total_trials = np.sum([len(all_trial_inds[t_i]) for t_i in range(num_tastes)])
	total_trial_inds = np.arange(total_trials)
	all_trial_taste_inds = np.array([t_i*np.ones(len(all_trial_inds[t_i])) for t_i in range(num_tastes)]).flatten()
	all_trial_delivery_inds = np.array([all_trial_inds[t_i] for t_i in range(num_tastes)]).flatten()
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	if len(segments_to_analyze) == 0:
		segments_to_analyze = np.arange(num_segments)
		
	#Save dir
	decoder_save_dir = os.path.join(save_dir,'GMM_Decoder_Tests')
	if not os.path.isdir(decoder_save_dir):
		os.mkdir(decoder_save_dir)
		
	#Set up BIC tests
#	param_grid = {"n_components": component_counts, "covariance_type": ["full"], 
#			   "n_init": [5]}
#	grid_search = GridSearchCV(gmm(), param_grid=param_grid, scoring=gmm_bic_score)
#	
	epoch_success_storage = np.zeros(len(epochs_to_analyze))
	
	for e_ind, e_i in enumerate(epochs_to_analyze): #By epoch conduct decoding
		print('\tDecoding Epoch ' + str(e_i))
		
		epoch_decode_save_dir = os.path.join(decoder_save_dir,'decode_prob_epoch_' + str(e_i))
		if not os.path.isdir(epoch_decode_save_dir):
			os.mkdir(epoch_decode_save_dir)
			
		trial_decodes = os.path.join(epoch_decode_save_dir,'Individual_Trials')
		if not os.path.isdir(trial_decodes):
			os.mkdir(trial_decodes)
			
		trial_decode_storage = np.zeros((total_trials,num_tastes)) #Fraction of the trial decoded as each taste for each component count
		trial_success_storage = np.zeros(total_trials) #Binary storage of successful decodes (max fraction of trial = taste delivered)
		
		print('\t\tPerforming LOO Decoding')
		for l_o_ind in tqdm.tqdm(total_trial_inds): #Which trial is being left out for decoding
			l_o_taste_ind = all_trial_taste_inds[l_o_ind].astype('int') #Taste of left out trial
			l_o_delivery_ind = all_trial_delivery_inds[l_o_ind].astype('int') #Delivery index of left out trial
			
			#Run gmm distribution fits to fr of each population for each taste
			train_data = []
			all_train_data = []
			all_train_data_labels = []
			#taste_bic_scores = np.zeros((len(component_counts),num_tastes))
			for t_i in range(num_tastes):
				train_taste_data = []
				taste_num_deliv = len(tastant_fr_dist[t_i])
				for d_i in range(taste_num_deliv):
					if (d_i == l_o_delivery_ind) and (t_i == l_o_taste_ind):
						#This is the Leave-One-Out trial so do nothing
						train_taste_data.extend([])
					else:
						train_taste_data.extend(list(tastant_fr_dist[t_i][d_i][e_i].T))
				train_data.append(np.array(train_taste_data))
				all_train_data.extend(train_taste_data)
				all_train_data_labels.extend(list((t_i*np.ones(len(train_taste_data))).astype('int')))
				#Auto-fit the best number of GMM components
				#grid_search.fit(np.array(test_taste_data))
				#taste_bic_scores[:,t_i] = np.array((grid_search.cv_results_)["mean_test_score"])
			#Calculate the best number of components
			#best_component_count = plot_gmm_bic_scores(taste_bic_scores, component_counts, e_i, \
			#							  dig_in_names, decoder_save_dir)	
			
			#Grab trial firing rate data
			t_spike_times = tastant_spike_times[l_o_taste_ind]
			t_cp_rast = cp_raster_inds[l_o_taste_ind]
			taste_start_dig_in = start_dig_in_times[l_o_taste_ind]
			t_spike_times_td_i = t_spike_times[l_o_delivery_ind]
			deliv_cp = t_cp_rast[l_o_delivery_ind,:] - pre_taste_dt
			sdi = np.ceil(taste_start_dig_in[l_o_delivery_ind] + deliv_cp[e_i]).astype('int')
			edi = np.ceil(taste_start_dig_in[l_o_delivery_ind] + deliv_cp[e_i+1]).astype('int')
			data_len = np.ceil(edi - sdi).astype('int')
			new_time_bins = np.arange(50,data_len-50,25)
			#Binerize spike times
			td_i_bin  = np.zeros((num_neur,data_len+1))
			for n_i in range(num_neur):
				n_i_spike_times = np.array(t_spike_times_td_i[n_i] - sdi).astype('int')
				keep_spike_times = n_i_spike_times[np.where((0 <= n_i_spike_times)*(data_len >= n_i_spike_times))[0]]
				td_i_bin[n_i,keep_spike_times] = 1
			#___Grab neuron firing rates in sliding bins
			tb_fr = np.zeros((num_neur,len(new_time_bins)))
			for tb_i,tb in enumerate(new_time_bins):
				tb_fr[:,tb_i] = np.sum(td_i_bin[:,tb-50:tb+50],1)/(100/1000)
			list_tb_fr = list(tb_fr.T)
			#___Convert to pca space
			#transformed_test_data = pca.transform(tb_fr.T)
			#___Grab single vector of firing rates for epoch to decode
			#fr_vec = [list(np.sum(td_i_bin,1)/(data_len*(1/1000)))]
			
			#Run through different numbers of components for decoding
			f_loo = plt.figure(figsize=(5,5))
			plt.suptitle('Taste ' + dig_in_names[l_o_taste_ind] + ' Delivery ' + str(l_o_delivery_ind))
			#Fit a Gaussian mixture model with the number of dimensions = number of neurons
			all_taste_gmm = dict()
			for t_i in range(num_tastes):
				t_cp_rast = cp_raster_inds[t_i]
				taste_start_dig_in = start_dig_in_times[t_i]
				train_taste_data = train_data[t_i]
				#Since Gaussian distribution fits to both positive and negative, add negative data for better fit
				train_expansion_taste_data = []
				train_expansion_taste_data.extend(list(train_taste_data))
				train_expansion_taste_data.extend(list(-1*train_taste_data))
# 				num_expand = min([len(train_taste_data),100])
# 				num_neur_sample = choices(list(np.arange(1,num_neur)),k=num_expand)
# 				for flip_i in range(num_expand):
# 					flip_neur = sample(list(np.arange(num_neur)),num_neur_sample[flip_i])
# 					n_i_flip = train_taste_data*np.ones(np.shape(train_taste_data))
# 					n_i_flip[:,flip_neur] = -1*n_i_flip[:,flip_neur]
# 					train_expansion_taste_data.extend(list(n_i_flip))
				#___True Data
				gm = gmm(n_components = 1, n_init = 10).fit(np.array(train_expansion_taste_data))
				#gm = gmm(n_components = 1, n_init = 10).fit(np.array(train_taste_data))
				#___PCA Transformed Data
				#transform_test_taste_data = pca.transform(np.array(train_taste_data))
				#Insert here a line of fitting the Gamma-MM
				all_taste_gmm[t_i] = gm
			
			#Calculate decoding probabilities for given jackknifed trial
					
			#Type 1: Bins of firing rates across the epoch of response
			#___Pass inputs to parallel computation on probabilities
			inputs = zip(list_tb_fr, itertools.repeat(num_tastes), \
				 itertools.repeat(all_taste_gmm), itertools.repeat(p_taste))
			pool = Pool(4)
			tb_decode_prob = pool.map(dp.segment_taste_decode_dependent_parallelized, inputs)
			pool.close()
			tb_decode_array = np.squeeze(np.array(tb_decode_prob)).T
			#___Plot decode results
			for t_i_plot in range(num_tastes):
				plt.plot(new_time_bins+deliv_cp[e_i], tb_decode_array[t_i_plot,:],label=dig_in_names[t_i_plot],color=taste_colors[t_i_plot])
				plt.fill_between(new_time_bins+deliv_cp[e_i],tb_decode_array[t_i_plot,:],color=taste_colors[t_i_plot],alpha=0.5,label='_')
			plt.ylabel('P(Taste)')
			plt.ylim([-0.1,1.1])
			plt.xlabel('Time (ms')
			plt.legend(loc='upper right')
			#___Calculate the average fraction of the epoch that was decoded as each taste and store
			taste_max_inds = np.argmax(tb_decode_array,0)
			taste_decode_fracs = [len(np.where(taste_max_inds == t_i_decode)[0])/len(new_time_bins) for t_i_decode in range(num_tastes)]
			trial_decode_storage[l_o_ind,:] = taste_decode_fracs
			#___Calculate the fraction of time in the epoch of each taste being best
			best_taste = np.argmax(taste_decode_fracs)
			if best_taste == t_i:
				trial_success_storage[l_o_ind] = 1
					
			#Type 2: Single vector of firing rates for epoch to decode
			#fr_vec = [list(np.sum(td_i_bin,1)/(data_len*(1/1000)))]
			#inputs = zip(fr_vec,num_tastes,all_taste_gmm,p_taste)
			#tb_decode_prob = dp.segment_taste_decode_dependent_parallelized(inputs)
			#trial_decode_storage[l_o_ind,:] = tb_decode_prob
			#if np.argmax(tb_decode_prob) == l_o_taste_ind:
			#	trial_success_storage[l_o_ind,c_ind] = 1
					
			#Save decoding figure
			plt.tight_layout()
			f_loo.savefig(os.path.join(trial_decodes,'decoding_results_taste_' + str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.png'))
			f_loo.savefig(os.path.join(trial_decodes,'decoding_results_taste_' + str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.svg'))
			plt.close(f_loo)
			
		#Once all trials are decoded, save decoding success results
		np.savetxt(os.path.join(epoch_decode_save_dir,'success_by_trial.csv'), trial_success_storage, delimiter=',')
		np.savetxt(os.path.join(epoch_decode_save_dir,'mean_taste_decode_components.csv'), trial_decode_storage, delimiter=',')
		
		#Calculate overall decoding success by component count
		taste_success_percent = np.round(100*np.mean(trial_success_storage),2)
		epoch_success_storage[e_ind] = taste_success_percent
		
	#Plot the success results for different component counts across epochs
	f_epochs = plt.figure(figsize=(5,5))
	plt.plot(np.arange(len(epochs_to_analyze),epoch_success_storage))
	epoch_labels = ['Epoch ' + str(e_i) for e_i in epochs_to_analyze]
	plt.xticks(np.arange(len(epochs_to_analyze)),labels=epoch_labels)
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Percent')
	plt.title('Decoding Success')
	f_epochs.savefig(os.path.join(decoder_save_dir,'gmm_success.png'))
	f_epochs.savefig(os.path.join(decoder_save_dir,'gmm_success.svg'))
	plt.close(f_epochs)
	
	return epoch_success_storage
	

def naive_bayes_decoding(num_neur, tastant_spike_times, cp_raster_inds, 
						 trial_train_inds, trial_test_inds, all_trial_inds, dig_in_names, 
						 start_dig_in_times, pre_taste_dt, post_taste_dt, save_dir):
	"""This function trains a Gaussian Naive Bayes decoder to decode different 
	taste epochs from activity.
	INPUTS:
		-
	OUTPUTS:
		-
	"""
	
	bayes_storage = os.path.join(save_dir,'Naive_Bayes_Decoder_Tests')
	if not os.path.isdir(bayes_storage):
		os.mkdir(bayes_storage)
	
	num_tastes = len(tastant_spike_times)
	num_cp = np.shape(cp_raster_inds[0])[-1] - 1
		
	#Store train data in two arrays: one with the firing rate vectors, the other
	#with the taste/state labels.
	for cp_i in tqdm.tqdm(range(num_cp)):
		taste_state_inds = [] #matching of index
		taste_state_labels = [] #matching of label
		train_fr_data = [] #firing rate vector storage
		train_fr_labels = [] #firing rate vector labelled indices (from taste_state_inds)
		ts_ind_counter = 0
		for t_i in range(num_tastes):
			t_name = dig_in_names[t_i]
			num_train_deliv = int(len(trial_train_inds[t_i]))
			taste_cp = cp_raster_inds[t_i]
			#Store the current iteration label and index
			taste_state_labels.extend([t_name + '_' + str(cp_i)])
			taste_state_inds.extend([ts_ind_counter])
			#Store firing rate vectors for each train set delivery
			for d_i, trial_ind in enumerate(trial_train_inds[t_i]):
				#grab spiking information
				raster_times = tastant_spike_times[t_i][trial_ind] #length num_neur list of lists
				start_taste_i = start_dig_in_times[t_i][trial_ind]
				deliv_cp = taste_cp[trial_ind,:] - pre_taste_dt
				#Create a binary matrix of spiking
				times_post_taste = [(np.array(raster_times[n_i])[np.where((raster_times[n_i] >= start_taste_i)*(raster_times[n_i] < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int') for n_i in range(num_neur)]
				bin_post_taste = np.zeros((num_neur,post_taste_dt))
				for n_i in range(num_neur):
					bin_post_taste[n_i,times_post_taste[n_i]] += 1
				#population changepoints
				start_epoch = int(deliv_cp[cp_i])
				end_epoch = int(deliv_cp[cp_i+1])
				epoch_len = end_epoch - start_epoch
				if epoch_len > 0: 
					for binsize in np.arange(100,epoch_len): #min bin size of 100 ms
						bin_edges = np.arange(start_epoch,end_epoch,binsize).astype('int') #bin the epoch
						if len(bin_edges) != 0:
							if (bin_edges[-1] != end_epoch)*(end_epoch-bin_edges[-1]>10):
								bin_edges = np.concatenate((bin_edges,end_epoch*np.ones(1).astype('int')))
							#Calculate the firing rate vectors for these bins
							bst_hz = [np.sum(bin_post_taste[:,bin_edges[b_i]:bin_edges[b_i+1]],1)/((bin_edges[b_i+1] - bin_edges[b_i])*(1/1000)) for b_i in range(len(bin_edges)-1)]
							#Create a list of the label index for these vectors
							bst_hz_labels = list(ts_ind_counter*np.ones(len(bst_hz)))
							#Store to dataset
							train_fr_data.extend(list(bst_hz)) #num_neur x n
							train_fr_labels.extend(bst_hz_labels)
			ts_ind_counter += 1
	
		#Now fit the gaussian naive bayes to the dataset and test on the remaining tastant deliveries
		gnb = GaussianNB()
		gnb.fit(np.array(train_fr_data), np.array(train_fr_labels))
		#Predict the state for each test set delivery
		test_predictions = []
		test_prediction_probabilities = []
		test_labels = []
		ts_ind_counter = 0
		for t_i in range(num_tastes):
			t_name = dig_in_names[t_i]
			num_test_deliv = int(len(trial_test_inds[t_i]))
			taste_cp = cp_raster_inds[t_i]
			for d_i, trial_ind in enumerate(trial_test_inds[t_i]):
				#grab spiking information
				raster_times = tastant_spike_times[t_i][trial_ind] #length num_neur list of lists
				start_taste_i = start_dig_in_times[t_i][trial_ind]
				deliv_cp = taste_cp[trial_ind,:] - pre_taste_dt
				#Create a binary matrix of spiking
				times_post_taste = [(np.array(raster_times[n_i])[np.where((raster_times[n_i] >= start_taste_i)*(raster_times[n_i] < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int') for n_i in range(num_neur)]
				bin_post_taste = np.zeros((num_neur,post_taste_dt))
				for n_i in range(num_neur):
					bin_post_taste[n_i,times_post_taste[n_i]] += 1
				#population changepoints
				start_epoch = int(deliv_cp[cp_i])
				end_epoch = int(deliv_cp[cp_i+1])
				epoch_len = end_epoch - start_epoch
				if epoch_len > 0: 
					for binsize in [epoch_len]:#np.arange(100,epoch_len):
						bin_edges = np.arange(start_epoch,end_epoch,binsize).astype('int') #bin the epoch
						if len(bin_edges) != 0:
							if (bin_edges[-1] != end_epoch)*(end_epoch-bin_edges[-1]>10):
								bin_edges = np.concatenate((bin_edges,end_epoch*np.ones(1).astype('int')))
							#Calculate the firing rate vectors for these bins
							bst_hz = [np.sum(bin_post_taste[:,bin_edges[b_i]:bin_edges[b_i+1]],1)/((bin_edges[b_i+1] - bin_edges[b_i])*(1/1000)) for b_i in range(len(bin_edges)-1)]
							bst_hz_labels = list(ts_ind_counter*np.ones(len(bst_hz)))
							#Run predictions through Gaussian Naive Bayes trained algorithm
							deliv_test_predictions = gnb.predict_proba(bst_hz)
							for p_i in range(len(deliv_test_predictions)):
								test_prediction_probabilities.append(list(deliv_test_predictions[p_i]))
							#Store the predictions and true labels for final accuracy reading
							test_predictions.extend([np.argmax(deliv_test_predictions)])
							test_labels.extend(bst_hz_labels)
			ts_ind_counter += 1
		
		#Plot 
		test_prediction_probabilities = np.array(test_prediction_probabilities)
		f_prob, ax_prob = plt.subplots(ncols=num_tastes,figsize=(num_tastes*4,4))
		for t_i in range(num_tastes):
			for t_label_i in range(num_tastes):
				label_match = np.where(np.array(test_labels) == t_label_i)[0]
				ax_prob[t_i].boxplot(test_prediction_probabilities[label_match,t_i],positions=[t_label_i],sym='',meanline=True,medianprops=dict(linestyle='-',color='blue'),showcaps=True,showbox=True,labels='_')
			ax_prob[t_i].scatter(test_labels+np.random.normal(0,0.04,len(test_labels)),test_prediction_probabilities[:,t_i],alpha=0.5,color='g',label='Test Set Deliveries')
			ax_prob[t_i].legend(loc='upper right')
			ax_prob[t_i].set_title('Probability of Taste ' + dig_in_names[t_i])
			ax_prob[t_i].set_xticks(ticks=np.arange(num_tastes),labels=dig_in_names)
			ax_prob[t_i].set_xlabel('Delivered Taste')
			ax_prob[t_i].axhline(1/num_tastes,linestyle='dashed',color='k',alpha=0.2)
			ax_prob[t_i].set_ylim([-0.1,1])
		ax_prob[0].set_ylabel('Decoding Probability')
		plt.tight_layout()
		f_prob.savefig(os.path.join(bayes_storage,'gauss_naive_bayes_success_probabilities_' + str(cp_i) + '.png'))
		f_prob.savefig(os.path.join(bayes_storage,'gauss_naive_bayes_success_probabilities_' + str(cp_i) + '.svg'))
		plt.close(f_prob)
		
		#Calculate overall accuracy and accuracy by group
		success_rates = np.zeros(len(taste_state_inds) + 1)
		success_labels = []
		overall_success_count = 0
		for ts_i in taste_state_inds:
			state_name = taste_state_labels[ts_i]
			success_labels.extend([state_name])
			state_prediction_inds = np.where(np.array(test_labels) == ts_i)[0]
			prediction_vals = np.array(test_predictions)[state_prediction_inds]
			num_total_predictions = len(prediction_vals)
			num_successful_predictions = len(np.where(prediction_vals == ts_i)[0])
			overall_success_count += num_successful_predictions
			success_rates[ts_i] = num_successful_predictions/num_total_predictions
		success_rates[-1] = overall_success_count/len(test_predictions)
		success_labels.extend(['Overall'])
		#Store success to csv
		with open(os.path.join(bayes_storage,'gauss_naive_bayes_success_' + str(cp_i) + '.csv'), 'w', newline='') as csvfile:
			   spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			   spamwriter.writerow(success_labels)
			   spamwriter.writerow(success_rates)
		

def gmm_bic_score(estimator, X):
	"""Callable to pass to GridSearchCV that will calculate the BIC score"""
	return -estimator.bic(X)	
	
def plot_gmm_bic_scores(taste_bic_scores, component_counts, e_i, \
							  dig_in_names, save_dir):
	"""This function plots decoder BIC scores for each number of components
	and returns the number of components that, on average, provides the lowest
	BIC score when fit to half the taste deliveries.
	INPUTS:
		- taste_bic_scores: array of [num_components x num_tastes] with BIC scores
		- component_counts: number of components for each bic score result
		- e_i: index of epoch being tested
		- dig_in_names: names of each taste
		- save_dir: where to save results/figures
	OUTPUTS:
		- best_component_count: the component count that on average provides
			the lowest BIC score across tastes
	"""
	_, num_tastes = np.shape(taste_bic_scores)
	
	#Plot the BIC scores by taste
	f = plt.figure(figsize=(8,8))
	for t_i in range(num_tastes):
		plt.plot(component_counts,taste_bic_scores[:,t_i],label=dig_in_names[t_i],alpha=0.5)
	plt.plot(component_counts,np.mean(taste_bic_scores,1),color='k',linestyle='dashed',alpha=1,label='Mean')
	plt.plot(component_counts,np.mean(taste_bic_scores[:,:-1],1),color='k',linestyle='dotted',alpha=1,label='True Taste Mean')
	plt.legend()
	plt.title('GMM Fit BIC Scores')
	plt.xlabel('# Components')
	plt.ylabel('BIC Score')
	plt.tight_layout()
	f.savefig(os.path.join(save_dir,'gmm_fit_bic_scores.png'))
	f.savefig(os.path.join(save_dir,'gmm_fit_bic_scores.svg'))
	plt.close(f)
	
	#The best number of components is that which has the lowest average BIC for the true tastes
	best_component_count = component_counts[np.argmin(np.mean(taste_bic_scores[:,:-1],1))]
	
	return best_component_count


	
	
	
	

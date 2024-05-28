#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:38:30 2024

@author: hannahgermaine

File dedicated to functions related to testing bayesian decoder parameters for
best decoder outcomes.
"""

import numpy as np
import tqdm, os, itertools#, time, csv
import matplotlib.pyplot as plt
from matplotlib import colormaps
from multiprocess import Pool
import functions.decode_parallel as dp
from sklearn.mixture import GaussianMixture as gmm
#from sklearn.model_selection import GridSearchCV
#from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
#from random import choices, sample

def test_decoder_params(dig_in_names, start_dig_in_times, num_neur, tastant_spike_times,
						tastant_fr_dist, cp_raster_inds, pre_taste_dt, post_taste_dt, 
						taste_select_neur, e_skip_dt, e_len_dt, save_dir):
	"""This function tests different decoder types to determine
	the best combination to use in replay decoding
	INPUTS:
		- dig_in_names: name of each taste deliveres
		- start_dig_in_times: start times of tastant deliveries
		
	OUTPUTS:
		- best_components: list of epoch-specific best number of components for gmm
	"""
	epochs_to_analyze = [0,1,2]
	
	#Get trial indices for train/test sets
	num_tastes = len(tastant_spike_times)
	all_trial_inds = []
	for t_i in range(num_tastes):
		taste_trials = len(tastant_spike_times[t_i])
		all_trial_inds.append(list(np.arange(taste_trials)))
		
	del t_i, taste_trials
	
	#Run decoder through training and jackknife testing to determine success rates
	gmm_success_rates, gmm_success_rates_by_taste = run_decoder(num_neur, start_dig_in_times, 
															 tastant_fr_dist, all_trial_inds, 
															 tastant_spike_times, cp_raster_inds, 
															 pre_taste_dt, e_len_dt, e_skip_dt, 
															 dig_in_names, save_dir, 
															 epochs_to_analyze)
	
	#Run Naive Bayes decoder to test
	nb_success_rates, nb_success_rates_by_taste = naive_bayes_decoding(num_neur, tastant_spike_times, 
																	cp_raster_inds, tastant_fr_dist, 
																	all_trial_inds, dig_in_names, 
																	start_dig_in_times, pre_taste_dt, 
																	post_taste_dt, save_dir, 
																	epochs_to_analyze)
	
	#Both Models Plot
	plot_all_results(tastant_fr_dist, gmm_success_rates, nb_success_rates, num_tastes,
						 save_dir)
		
def run_decoder(num_neur, start_dig_in_times, tastant_fr_dist, all_trial_inds,
				tastant_spike_times, cp_raster_inds,
				pre_taste_dt, e_len_dt, e_skip_dt, dig_in_names, 
				save_dir, epochs_to_analyze = []):
	"""This function runs a decoder with a given set of parameters and returns
	the decoding probabilities of taste delivery periods
	INPUTS:
		- num_neur: number of neurons in dataset
		- start_dig_in_times: times of taste deliveries
		- tastant_fr_dist: firing rate distribution to fit over (train set)
		- all_trial_inds: indices of all trials used in testing the fit
		- tastant_spike_times: spike times for each tastant delivery
		- cp_raster_inds: changepoint times for all taste deliveries
		- pre_taste_dt: ms before taste delivery in cp_raster_inds
		- e_len_dt: decoding chunk length
		- e_skip_dt: decoding skip length
		- dig_in_names: taste names
		- save_dir: directory where to save results
		- epochs_to_analyze: array of which epochs to analyze
	OUTPUTS:
		- Plots of decoder results on individual trials as well as overall success
			metrics.
		- epoch_success_storage: vector of length number of epochs containing success
			percentages overall.
		- epoch_success_by_taste: array of size num_epochs x num_tastes containing
			success percentages by decoded taste by epoch.
	"""
	#TODO: Handle taste selective neurons
	#Variables
	num_tastes = len(start_dig_in_times)
	num_cp = len(tastant_fr_dist[0][0])
	p_taste = np.ones(num_tastes)/num_tastes #P(taste)
	cmap = colormaps['jet']
	taste_colors = cmap(np.linspace(0, 1, num_tastes))
	
	#Jackknife decoding total number of trials
	total_trials = np.sum([len(all_trial_inds[t_i]) for t_i in range(num_tastes)])
	total_trial_inds = np.arange(total_trials)
	all_trial_taste_inds = np.array([t_i*np.ones(len(all_trial_inds[t_i])) for t_i in range(num_tastes)]).flatten()
	all_trial_delivery_inds = np.array([all_trial_inds[t_i] for t_i in range(num_tastes)]).flatten()
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	cmap = colormaps['cividis']
	epoch_colors = cmap(np.linspace(0, 1, len(epochs_to_analyze)))
	
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
	epoch_decode_storage = []
	
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
			t_cp_rast = cp_raster_inds[l_o_taste_ind]
			taste_start_dig_in = start_dig_in_times[l_o_taste_ind]
			deliv_cp = t_cp_rast[l_o_delivery_ind,:] - pre_taste_dt
			sdi = np.ceil(taste_start_dig_in[l_o_delivery_ind] + deliv_cp[e_i]).astype('int')
			edi = np.ceil(taste_start_dig_in[l_o_delivery_ind] + deliv_cp[e_i+1]).astype('int')
			data_len = np.ceil(edi - sdi).astype('int')
			new_time_bins = np.arange(50,data_len-50,25)
			#___Grab neuron firing rates in sliding bins
			tb_fr = tastant_fr_dist[l_o_taste_ind][l_o_delivery_ind][e_i].T
			list_tb_fr = list(tb_fr)
			#___Convert to pca space
			#transformed_test_data = pca.transform(tb_fr.T)
			
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
# 				#___True Data
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
			best_taste = np.where(taste_decode_fracs == np.max(taste_decode_fracs))[0]
			if len(best_taste) == 1:
				if best_taste == l_o_taste_ind:
					trial_success_storage[l_o_ind] = 1
			else:
				if len(np.where(best_taste == l_o_taste_ind)[0]) > 0: #Taste is one of the predicted tastes in a "tie"
					trial_success_storage[l_o_ind] = 1
					
			#Save decoding figure
			plt.tight_layout()
			f_loo.savefig(os.path.join(trial_decodes,'decoding_results_taste_' + str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.png'))
			f_loo.savefig(os.path.join(trial_decodes,'decoding_results_taste_' + str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.svg'))
			plt.close(f_loo)
			
		#Once all trials are decoded, save decoding success results
		np.savetxt(os.path.join(epoch_decode_save_dir,'success_by_trial.csv'), trial_success_storage, delimiter=',')
		np.savetxt(os.path.join(epoch_decode_save_dir,'mean_taste_decode_components.csv'), trial_decode_storage, delimiter=',')
		epoch_decode_storage.append(trial_decode_storage)
		
		#Calculate overall decoding success by component count
		taste_success_percent = np.round(100*np.mean(trial_success_storage),2)
		epoch_success_storage[e_ind] = taste_success_percent
		
	#Plot the overall success results for different component counts across epochs
	f_epochs = plt.figure(figsize=(5,5))
	plt.bar(np.arange(len(epochs_to_analyze)),epoch_success_storage)
	epoch_labels = ['Epoch ' + str(e_i) for e_i in epochs_to_analyze]
	plt.xticks(np.arange(len(epochs_to_analyze)),labels=epoch_labels)
	plt.ylim([0,100])
	plt.axhline(100/num_tastes,linestyle='dashed',color='k',alpha=0.75,label='Chance')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Percent')
	plt.title('Decoding Success')
	f_epochs.savefig(os.path.join(decoder_save_dir,'gmm_success.png'))
	f_epochs.savefig(os.path.join(decoder_save_dir,'gmm_success.svg'))
	plt.close(f_epochs)
	
	#Plot the by-taste success results
	#true taste indices == all_trial_taste_inds
	#decode percents == epoch_decode_storage
	f_percents = plt.figure(figsize=(5,5))
	epoch_success_by_taste = np.zeros((len(epochs_to_analyze),num_tastes))
	for e_ind, e_i in enumerate(epochs_to_analyze):
		epoch_decode_percents = epoch_decode_storage[e_ind]
		success_by_taste = np.zeros(num_tastes)
		for t_i in range(num_tastes):
			taste_trials = np.where(all_trial_taste_inds == t_i)[0]
			taste_trial_results_bin = np.zeros(len(taste_trials))
			for tt_ind, tt_i in enumerate(taste_trials):
				trial_decode_results = epoch_decode_percents[tt_i,:]
				best_taste = np.where(trial_decode_results == np.max(trial_decode_results))[0]
				if len(best_taste) == 1:
					if best_taste == t_i:
						taste_trial_results_bin[tt_ind] = 1
				else:
					if len(np.where(best_taste == t_i)[0]) > 0: #Taste is one of the predicted tastes in a "tie"
						taste_trial_results_bin[tt_ind] = 1
			success_by_taste[t_i] = np.mean(taste_trial_results_bin)
		epoch_success_by_taste[e_ind,:] = success_by_taste
		plt.scatter(np.arange(num_tastes),success_by_taste,label='Epoch ' + str(e_i),color=epoch_colors[e_ind,:])
		plt.plot(np.arange(num_tastes),success_by_taste,label='_',color=epoch_colors[e_ind,:],linestyle='dashed',alpha=0.75)
	np.savetxt(os.path.join(decoder_save_dir,'epoch_success_by_taste.csv'), epoch_success_by_taste, delimiter=',')
	plt.axhline(100/num_tastes,label='Chance',color='k',linestyle='dashed',alpha=0.75)
	plt.legend(loc='lower left')
	plt.xlabel('Taste')
	plt.xticks(np.arange(num_tastes),dig_in_names)
	plt.ylabel('Percent')
	plt.title('Decoding Success by Taste')
	f_percents.savefig(os.path.join(decoder_save_dir,'gmm_success_by_taste.png'))
	f_percents.savefig(os.path.join(decoder_save_dir,'gmm_success_by_taste.svg'))
	plt.close(f_percents)
	
	return epoch_success_storage, epoch_success_by_taste
	

def naive_bayes_decoding(num_neur, tastant_spike_times, cp_raster_inds, 
						 tastant_fr_dist, all_trial_inds, dig_in_names, 
						 start_dig_in_times, pre_taste_dt, post_taste_dt, 
						 save_dir, epochs_to_analyze=[]):
	"""This function trains a Gaussian Naive Bayes decoder to decode different 
	taste epochs from activity.
	INPUTS:
		- num_neur: number of neurons in dataset
		- tastant_spike_times: spike times for each tastant delivery
		- cp_raster_inds: changepoint times for all taste deliveries
		- tastant_fr_dist: firing rate distribution to fit over (train set)
		- all_trial_inds: indices of all trials used in testing the fit
		- dig_in_names: taste names
		- start_dig_in_times: start of each tastant delivery
		- pre_taste_dt: ms before taste delivery in cp_raster_inds
		- post_taste_dt: ms after taste delivery in cp_raster_inds
		- save_dir: directory where to save results
		- epochs_to_analyze: array of which epochs to analyze
	OUTPUTS:
		- Plots of decoder results on individual trials as well as overall success
			metrics.
		- epoch_success_storage: vector of length number of epochs containing success
			percentages overall.
		- epoch_success_by_taste: array of size num_epochs x num_tastes containing
			success percentages by decoded taste by epoch.
	"""
	
	#TODO: Handle taste selective neurons
	
	#Variables
	num_tastes = len(start_dig_in_times)
	num_cp = len(tastant_fr_dist[0][0])
	p_taste = np.ones(num_tastes)/num_tastes #P(taste)
	cmap = colormaps['jet']
	taste_colors = cmap(np.linspace(0, 1, num_tastes))
	
	#Jackknife decoding total number of trials
	total_trials = np.sum([len(all_trial_inds[t_i]) for t_i in range(num_tastes)])
	total_trial_inds = np.arange(total_trials)
	all_trial_taste_inds = np.array([t_i*np.ones(len(all_trial_inds[t_i])) for t_i in range(num_tastes)]).flatten()
	all_trial_delivery_inds = np.array([all_trial_inds[t_i] for t_i in range(num_tastes)]).flatten()
	
	if len(epochs_to_analyze) == 0:
		epochs_to_analyze = np.arange(num_cp)
	cmap = colormaps['cividis']
	epoch_colors = cmap(np.linspace(0, 1, len(epochs_to_analyze)))
	
	bayes_storage = os.path.join(save_dir,'Naive_Bayes_Decoder_Tests')
	if not os.path.isdir(bayes_storage):
		os.mkdir(bayes_storage)
		
	epoch_success_storage = np.zeros(len(epochs_to_analyze))
	epoch_decode_storage = []
	
	for e_ind, e_i in enumerate(epochs_to_analyze): #By epoch conduct decoding
		print('\tDecoding Epoch ' + str(e_i))
		
		epoch_decode_save_dir = os.path.join(bayes_storage,'decode_prob_epoch_' + str(e_i))
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
			
			#Collect trial data for decoder
			taste_state_inds = [] #matching of index
			taste_state_labels = [] #matching of label
			train_fr_data = [] #firing rate vector storage
			train_fr_labels = [] #firing rate vector labelled indices (from taste_state_inds)
			for t_i in range(num_tastes):
				t_name = dig_in_names[t_i]
				#Store the current iteration label and index
				taste_state_labels.extend([t_name + '_' + str(e_i)])
				taste_state_inds.extend([t_i])
				#Store firing rate vectors for each train set delivery
				for d_i, trial_ind in enumerate(all_trial_inds[t_i]):
					if (d_i == l_o_delivery_ind) and (t_i == l_o_taste_ind):
						train_fr_data.extend([]) #Basically do nothing
					else:
						tb_fr = tastant_fr_dist[t_i][d_i][e_i]
						list_tb_fr = list(tb_fr.T)
						train_fr_data.extend(list_tb_fr)
						bst_hz_labels = list(t_i*np.ones(len(list_tb_fr)))
						train_fr_labels.extend(bst_hz_labels)
			
			#Train a Bayesian decoder on all trials but the left out one
			gnb = GaussianNB()
			gnb.fit(np.array(train_fr_data), np.array(train_fr_labels))
			
			#Now perform decoding of the left out trial with the decoder
			taste_cp = cp_raster_inds[l_o_taste_ind]
			raster_times = tastant_spike_times[l_o_taste_ind][l_o_delivery_ind] #length num_neur list of lists
			start_taste_i = start_dig_in_times[l_o_taste_ind][l_o_delivery_ind]
			deliv_cp = taste_cp[l_o_delivery_ind,:] - pre_taste_dt
			times_post_taste = [(np.array(raster_times[n_i])[np.where((raster_times[n_i] >= start_taste_i)*(raster_times[n_i] < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int') for n_i in range(num_neur)]
			start_epoch = int(deliv_cp[e_i])
			end_epoch = int(deliv_cp[e_i+1])
			sdi = start_taste_i + start_epoch
			epoch_len = end_epoch - start_epoch
			if epoch_len > 0: 
				new_time_bins = np.arange(50,epoch_len-50,25)
				f_loo = plt.figure(figsize=(5,5))
				plt.suptitle('Taste ' + dig_in_names[l_o_taste_ind] + ' Delivery ' + str(l_o_delivery_ind))
				
				#___Grab neuron firing rates in sliding bins
				tb_fr = tastant_fr_dist[l_o_taste_ind][l_o_delivery_ind][e_i].T
				list_tb_fr = list(tb_fr.T)
				#Predict the results
				deliv_test_predictions = gnb.predict_proba(list_tb_fr)
				taste_max_inds = np.argmax(deliv_test_predictions,1)
				taste_decode_fracs = [len(np.where(taste_max_inds == t_i_decode)[0])/len(new_time_bins) for t_i_decode in range(num_tastes)]
				trial_decode_storage[l_o_ind,:] = taste_decode_fracs
				#___Plot decode results
				for t_i_plot in range(num_tastes):
					plt.plot(new_time_bins+deliv_cp[e_i], deliv_test_predictions[:,t_i_plot],label=dig_in_names[t_i_plot],color=taste_colors[t_i_plot])
					plt.fill_between(new_time_bins+deliv_cp[e_i],deliv_test_predictions[:,t_i_plot],color=taste_colors[t_i_plot],alpha=0.5,label='_')
				plt.ylabel('P(Taste)')
				plt.ylim([-0.1,1.1])
				plt.xlabel('Time (ms')
				plt.legend(loc='upper right')
				#___Calculate the fraction of time in the epoch of each taste being best
				best_taste = np.where(taste_decode_fracs == np.max(taste_decode_fracs))[0]
				if len(best_taste) == 1:
					if best_taste == l_o_taste_ind:
						trial_success_storage[l_o_ind] = 1
				else:
					if len(np.where(best_taste == l_o_taste_ind)[0]) > 0: #Taste is one of the predicted tastes in a "tie"
						trial_success_storage[l_o_ind] = 1
			
				#Save decoding figure
				plt.tight_layout()
				f_loo.savefig(os.path.join(trial_decodes,'decoding_results_taste_' + str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.png'))
				f_loo.savefig(os.path.join(trial_decodes,'decoding_results_taste_' + str(l_o_taste_ind) + '_delivery_' + str(l_o_delivery_ind) + '.svg'))
				plt.close(f_loo)
			
		#Once all trials are decoded, save decoding success results
		np.savetxt(os.path.join(epoch_decode_save_dir,'success_by_trial.csv'), trial_success_storage, delimiter=',')
		np.savetxt(os.path.join(epoch_decode_save_dir,'mean_taste_decode_components.csv'), trial_decode_storage, delimiter=',')
		epoch_decode_storage.append(trial_decode_storage)
		
		#Calculate overall decoding success by component count
		taste_success_percent = np.round(100*np.mean(trial_success_storage),2)
		epoch_success_storage[e_ind] = taste_success_percent
	
	#Plot the success results for different component counts across epochs
	f_epochs = plt.figure(figsize=(5,5))
	plt.bar(np.arange(len(epochs_to_analyze)),epoch_success_storage)
	epoch_labels = ['Epoch ' + str(e_i) for e_i in epochs_to_analyze]
	plt.xticks(np.arange(len(epochs_to_analyze)),labels=epoch_labels)
	plt.ylim([0,100])
	plt.axhline(100/num_tastes,linestyle='dashed',color='k',alpha=0.75,label='Chance')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Percent')
	plt.title('Decoding Success')
	f_epochs.savefig(os.path.join(bayes_storage,'nb_success.png'))
	f_epochs.savefig(os.path.join(bayes_storage,'nb_success.svg'))
	plt.close(f_epochs)

	#Plot the by-taste success results
	f_percents = plt.figure(figsize=(5,5))
	epoch_success_by_taste = np.zeros((len(epochs_to_analyze),num_tastes))
	for e_ind, e_i in enumerate(epochs_to_analyze):
		epoch_decode_percents = epoch_decode_storage[e_ind]
		success_by_taste = np.zeros(num_tastes)
		for t_i in range(num_tastes):
			taste_trials = np.where(all_trial_taste_inds == t_i)[0]
			taste_trial_results_bin = np.zeros(len(taste_trials))
			for tt_ind, tt_i in enumerate(taste_trials):
				trial_decode_results = epoch_decode_percents[tt_i,:]
				best_taste = np.where(trial_decode_results == np.max(trial_decode_results))[0]
				if len(best_taste) == 1:
					if best_taste == t_i:
						taste_trial_results_bin[tt_ind] = 1
				else:
					if len(np.where(best_taste == t_i)[0]) > 0: #Taste is one of the predicted tastes in a "tie"
						taste_trial_results_bin[tt_ind] = 1
			success_by_taste[t_i] = 100*np.mean(taste_trial_results_bin)
		epoch_success_by_taste[e_ind,:] = success_by_taste
		plt.scatter(np.arange(num_tastes),success_by_taste,label='Epoch ' + str(e_i),color=epoch_colors[e_ind,:])
		plt.plot(np.arange(num_tastes),success_by_taste,label='_',color=epoch_colors[e_ind,:],linestyle='dashed',alpha=0.75)
	np.savetxt(os.path.join(bayes_storage,'epoch_success_by_taste.csv'), epoch_success_by_taste, delimiter=',')
	plt.axhline(100/num_tastes,label='Chance',color='k',linestyle='dashed',alpha=0.75)
	plt.legend(loc='lower left')
	plt.xlabel('Taste')
	plt.xticks(np.arange(num_tastes),dig_in_names)
	plt.ylabel('Percent')
	plt.title('Decoding Success by Taste')
	f_percents.savefig(os.path.join(bayes_storage,'nb_success_by_taste.png'))
	f_percents.savefig(os.path.join(bayes_storage,'nb_success_by_taste.svg'))
	plt.close(f_percents)

	return epoch_success_storage, epoch_success_by_taste

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


def plot_all_results(tastant_fr_dist, gmm_success_rates, nb_success_rates, num_tastes,
					 save_dir):
	"""This function plots the results of both GMM and NB decoder tests on one
	set of axes.
	INPUTS:
		- tastant_fr_dist: firing rate vectors by taste by delivery by epoch
		- gmm_success_rates: vector of success by epoch using gmm
		- nb_success_rates: vector of success by epoch using nb
		- num_tastes: number of tastes
		- save_dir: where to save plots
	OUTPUTS: Figure with model results.
	"""
	model_results_comb = plt.figure(figsize=(8,8))
	num_cp = len(tastant_fr_dist[0][0])
	plt.plot(np.arange(num_cp),gmm_success_rates,label='GMM')
	plt.plot(np.arange(num_cp),nb_success_rates,label='NB')
	plt.axhline(100/num_tastes,label='Chance',linestyle='dashed',color='k',alpha=0.75)
	gmm_avg_success = np.nanmean(gmm_success_rates)
	plt.axhline(gmm_avg_success,label='GMM Mean',linestyle='dashed',alpha=0.75)
	nb_avg_success = np.nanmean(nb_success_rates)
	plt.axhline(nb_avg_success,label='NB Mean',linestyle='dashed',alpha=0.75)
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Percent of Trials Successfully Decoded')
	plt.title('Model Success by Epoch')
	model_results_comb.savefig(os.path.join(save_dir,'Decoder_Success_Results.png'))
	model_results_comb.savefig(os.path.join(save_dir,'Decoder_Success_Results.svg'))
	plt.close(model_results_comb)
	
	
	

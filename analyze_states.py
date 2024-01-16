#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:04:51 2023

@author: hannahgermaine

This code is written to import sorted data and perform state-change analyses
"""
import os, tables, tqdm
file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
os.chdir(file_path)
import numpy as np
import functions.analysis_funcs as af
import functions.hdf5_handling as hf5
import functions.data_processing as dp
import functions.load_intan_rhd_format.load_intan_rhd_format as rhd
import functions.plot_funcs as pf
import functions.seg_compare as sc
import functions.changepoint_detection as cd
import functions.decoding_funcs as df

#_____Get the directory of the hdf5 file_____
sorted_dir, segment_dir, cleaned_dir = hf5.sorted_data_import() #Program will automatically quit if file not found in given folder
data_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'

#_____Import data_____
num_neur, all_waveforms, spike_times, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes = af.import_data(sorted_dir, segment_dir, data_save_dir)

#%%
#_____Calculate spike time datasets_____
#NOTE: make sure the pre_taste value is never more than the post_taste value
pre_taste = 0.5 #Seconds before tastant delivery to store
post_taste = 2 #Seconds after tastant delivery to store

#_____Add "no taste" control segments to the dataset_____
if dig_in_names[-1] != 'none':
	dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes = af.add_no_taste(start_dig_in_times, end_dig_in_times, post_taste, dig_in_names)

#_____Pull out spike times for all tastes (and no taste)_____
segment_spike_times = af.calc_segment_spike_times(segment_times,spike_times,num_neur)
tastant_spike_times = af.calc_tastant_spike_times(segment_times,spike_times,
												  start_dig_in_times,end_dig_in_times,
												  pre_taste,post_taste,num_tastes,num_neur)

#todo: save these values to .npy files or something in the data folder and import if possible / deal with lists of lists in af

#%%

#_____Generate Raster Plots_____
#segment_spike_times is a list nested by segments x num_neur x num_time
#tastant_spike_times is a list with only taste delivery data nested by tastant_delivery x num_neur x num_time
pre_taste_dt = int(np.ceil(pre_taste*(1000/1))) #Convert to ms timescale
post_taste_dt = int(np.ceil(post_taste*(1000/1))) #Convert to ms timescale
bin_width = 0.25 #Gaussian convolution kernel width in seconds
bin_step = 25 #Step size in ms to take in PSTH calculation
data_group_name = 'PSTH_data'
try:
	tastant_PSTH = af.pull_data_from_hdf5(sorted_dir,data_group_name,'tastant_PSTH')
	PSTH_times = af.pull_data_from_hdf5(sorted_dir,data_group_name,'PSTH_times')
	PSTH_taste_deliv_times = af.pull_data_from_hdf5(sorted_dir,data_group_name,'PSTH_taste_deliv_times')
	avg_tastant_PSTH = af.pull_data_from_hdf5(sorted_dir,data_group_name,'avg_tastant_PSTH')
	print("PSTH data imported")
except:
	"Calculating and plotting raster and PSTH data"
	pf.raster_plots(data_save_dir, dig_in_names, start_dig_in_times, end_dig_in_times, 
					segment_names, segment_times, segment_spike_times, tastant_spike_times, 
					pre_taste_dt, post_taste_dt, num_neur, num_tastes)
	PSTH_times, PSTH_taste_deliv_times, tastant_PSTH, avg_tastant_PSTH = pf.PSTH_plots(data_save_dir, num_tastes,
																					   num_neur, dig_in_names, 
																					   start_dig_in_times, end_dig_in_times, 
																					   pre_taste_dt, post_taste_dt, 
																					   segment_times, segment_spike_times,
																					   bin_width, bin_step)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'tastant_PSTH',tastant_PSTH)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'PSTH_times',PSTH_times)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'PSTH_taste_deliv_times',PSTH_taste_deliv_times)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'avg_tastant_PSTH',avg_tastant_PSTH)

#%%
#_____Grab and plot firing rate distributions and comparisons (by segment)_____
#sc_save_dir = data_save_dir + 'Segment_Comparison/'
#if os.path.isdir(sc_save_dir) == False:
#	os.mkdir(sc_save_dir)
	
#All data
#all_sc_save_dir = sc_save_dir + 'All/'
#if os.path.isdir(all_sc_save_dir) == False:
#	os.mkdir(all_sc_save_dir)
#sc.bin_spike_counts(all_sc_save_dir,segment_spike_times,segment_names,segment_times)

  
#%%
#A couple changepoint detection calculations: first using a KS-Test statistic, 
#second using classic Bayes assuming spikes are sampled from a Poisson 
#distribution (taken from Paul's Comp Neuro Textbook)

cp_bin = 250 #minimum state size in ms
num_cp = 3 #number of changepoints to find
before_taste = np.ceil(pre_taste*1000).astype('int') #Milliseconds before taste delivery to plot
after_taste = np.ceil(post_taste*1000).astype('int') #Milliseconds after taste delivery to plot

#Set storage directory
cp_save_dir = data_save_dir + 'Changepoint_Calculations/'
if os.path.isdir(cp_save_dir) == False:
	os.mkdir(cp_save_dir)
	
	
#_____All data_____
taste_cp_save_dir = cp_save_dir + 'All_Taste_CPs/'
if os.path.isdir(taste_cp_save_dir) == False:
	os.mkdir(taste_cp_save_dir)
data_group_name = 'changepoint_data'
#Raster Poisson Bayes Changepoint Calcs Indiv Neurons
try:
	taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_cp_raster_inds')
	pop_taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'pop_taste_cp_raster_inds')
except:	
	taste_cp_raster_save_dir = taste_cp_save_dir + 'neur/'
	if os.path.isdir(taste_cp_raster_save_dir) == False:
		os.mkdir(taste_cp_raster_save_dir)
	taste_cp_raster_inds = cd.calc_cp_iter(tastant_spike_times,cp_bin,num_cp,start_dig_in_times,
				  end_dig_in_times,before_taste,after_taste,
				  dig_in_names,taste_cp_raster_save_dir)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_cp_raster_inds',taste_cp_raster_inds)
	
	taste_cp_raster_pop_save_dir = taste_cp_save_dir + 'pop/'
	if os.path.isdir(taste_cp_raster_pop_save_dir) == False:
		os.mkdir(taste_cp_raster_pop_save_dir)
	pop_taste_cp_raster_inds = cd.calc_cp_iter_pop(tastant_spike_times,cp_bin,num_cp,start_dig_in_times,
				  end_dig_in_times,before_taste,after_taste,
				  dig_in_names,taste_cp_raster_pop_save_dir)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'pop_taste_cp_raster_inds',pop_taste_cp_raster_inds)

#%%	

#____Calculate which neurons are taste selective by bin size_____

data_group_name = 'taste_selectivity'

tastant_binned_frs = af.taste_cp_frs(taste_cp_raster_inds,tastant_spike_times,start_dig_in_times,end_dig_in_times,dig_in_names,num_neur,pre_taste_dt,post_taste_dt)

decoding_save_dir = data_save_dir + 'Taste_Selectivity/'
if os.path.isdir(decoding_save_dir) == False:
	os.mkdir(decoding_save_dir)
	
loo_distribution_save_dir = decoding_save_dir + 'LOO_Distributions/'
if os.path.isdir(loo_distribution_save_dir) == False:
	os.mkdir(loo_distribution_save_dir)
	
#_____Calculate taste decoding probabilities and success probabilities_____
try:
	taste_select_prob_joint = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob_joint')[0]
	taste_select_prob_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob_epoch')[0]
	p_taste_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'p_taste_epoch')[0]
	p_taste_joint = af.pull_data_from_hdf5(sorted_dir,data_group_name,'p_taste_joint')[0]
	taste_select_neur_bin = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_neur_bin')[0]
	taste_select_neur_epoch_bin = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_neur_epoch_bin')[0]
except:
	print("Using population changepoint indices to calculate taste selectivity by epoch.")
	#p_taste_joint = prob of decoding each taste [num_neur x num_tastes x num_deliv]
	#p_taste_epoch = prob of decoding each taste [num_neur x num_tastes x num_deliv x num_cp]
	#taste_select_prob_joint = fraction of correctly decoded deliveries [num_neur, num_tastes]
	#taste_select_prob_epoch = fraction of correctly decoded deliveries [num_cp, num_neur, num_tastes]
	
	p_taste_joint, p_taste_epoch, taste_select_prob_joint, taste_select_prob_epoch = df. taste_decoding_cp(tastant_spike_times,\
												   pop_taste_cp_raster_inds,num_cp,start_dig_in_times,end_dig_in_times,dig_in_names, \
													   num_neur,pre_taste_dt,post_taste_dt,loo_distribution_save_dir)
	#_____Calculate binary matrices of taste selective neurons / taste selective neurons by epoch_____
	#On average, does the neuron decode neurons more often than chance?
	taste_select_neur_bin = np.sum(taste_select_prob_joint,1) > 1/num_tastes
	taste_select_neur_epoch_bin = np.sum(taste_select_prob_epoch,2) > 1/num_tastes
	#Save
	af.add_data_to_hdf5(sorted_dir,data_group_name,'p_taste_joint',p_taste_joint)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_select_prob_joint',taste_select_prob_joint)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'p_taste_epoch',p_taste_epoch)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_select_prob_epoch',taste_select_prob_epoch)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_select_neur_bin',taste_select_neur_bin)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_select_neur_epoch_bin',taste_select_neur_epoch_bin)


#%%
#Plot the decoding - * denotes statitical significance in KW nonparametric multiple-comparisons test
# pf.epoch_taste_select_plot(prob_taste_epoch, dig_in_names, decoding_save_dir)
# epoch_max_decoding, epoch_select_neur = pf.taste_select_success_plot(taste_select_prob_joint, np.arange(num_cp), 'epoch', 'taste_selectivity_by_epoch', decoding_save_dir)

# try:
# 	taste_response_prob = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_response_prob')[0]
# 	taste_select_prob = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob')[0]
# 	taste_response_prob_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_response_prob_epoch')[0]
# 	taste_select_bin_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_bin_epoch')[0]
# except:
# 	#_____All Epochs Included_____
# 	#Select the taste responsive neurons using taste_select_prob_joint (matrix num_neur x num_tastes)
# 	taste_response_prob = np.expand_dims((np.sum(taste_select_prob_joint,1)/3 >= 1/num_tastes).astype('int'),1)*np.ones((num_neur,num_tastes))
# 	#Select the taste selective neurons using taste_select_prob_joint
# 	taste_select_prob = ((taste_select_prob_joint >= 1/num_tastes).astype('int'))*np.ones((num_neur,num_tastes))
# 	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_response_prob',taste_response_prob)
# 	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_select_prob',taste_select_prob)
# 	#_____Using Highest Probability Epoch_____
# 	#Select the taste responsive neurons using taste_select_prob_epoch (epoch,neur,taste)
# 	max_taste_select_prob_epoch = np.max(taste_select_prob_epoch,0) #Select the maximal taste probability epoch's probability as representative
# 	taste_response_prob_epoch = np.expand_dims((np.sum(max_taste_select_prob_epoch,1)/3 >= 1/num_tastes).astype('int'),1)*np.ones((num_neur,num_tastes))
# 	#Select the taste selective neurons using taste_select_prob_epoch
# 	taste_select_bin_epoch = np.zeros((num_cp,num_neur))
# 	for t_i in range(num_tastes-1): #assumes last taste is "none" so if at least 1 taste is highly decodeable for that epoch, we can call taste selective
# 	    taste_select_bin_epoch += np.squeeze(taste_select_prob_epoch[:,:,t_i]) > 1/3
# 	taste_select_bin_epoch = ((taste_select_bin_epoch > 0)*np.ones(np.shape(taste_select_bin_epoch))).T
# 	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_response_prob_epoch',taste_response_prob_epoch)
# 	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_select_bin_epoch',taste_select_bin_epoch)
# 	

# #%% Plot changepoint statistics

# #TODO: move to plot_funcs document

# import matplotlib.pyplot as plt
# from matplotlib import cm

# colors = cm.cool(np.arange(num_cp)/(num_cp))

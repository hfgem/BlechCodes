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
#____Calculate which neurons are taste responsive_____

data_group_name = 'taste_responsivity'
try:
	taste_responsivity_probability = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_responsivity_probability')
	taste_responsivity_binary = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_responsivity_binary')[0]
	taste_responsive_ind = (af.pull_data_from_hdf5(sorted_dir, data_group_name, 'taste_responsive_ind')[0]).astype('int')
except:	
	taste_responsivity_probability, taste_responsivity_binary = af.taste_responsivity_raster(tastant_spike_times,start_dig_in_times,end_dig_in_times,num_neur,pre_taste_dt)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_responsivity_probability',taste_responsivity_probability)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_responsivity_binary',taste_responsivity_binary)
	#Pull out the taste responsive data only
	taste_responsive_ind = np.where(taste_responsivity_binary == 1)[0]
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_responsive_ind',taste_responsive_ind)

taste_responsive_segment_spike_times = []
for s_i in range(len(segment_names)):
	taste_responsive_segment_spike_times.append([segment_spike_times[s_i][tri] for tri in taste_responsive_ind])
taste_responsive_tastant_spike_times = []
taste_responsive_tastant_PSTH = []
for t_i in range(len(dig_in_names)):
	d_collect = []
	for d_i in range(len(start_dig_in_times[t_i])):
		d_collect.append([tastant_spike_times[t_i][d_i][tri] for tri in taste_responsive_ind])
	taste_responsive_tastant_spike_times.append(d_collect)
	taste_responsive_tastant_PSTH.append(tastant_PSTH[t_i][:,taste_responsive_ind,:])
	
#Pull out MOST taste responsive neuron data
most_taste_responsivity_binary = np.array([True for n_i in range(num_neur)])
for t_i in range(len(dig_in_names)):
	top_50_percentile = np.percentile(taste_responsivity_probability[t_i],50)
	most_taste_responsivity_binary *= taste_responsivity_probability[t_i] >= top_50_percentile
most_taste_responsive_ind = np.where(most_taste_responsivity_binary)[0]
af.add_data_to_hdf5(sorted_dir,data_group_name,'most_taste_responsive_ind',most_taste_responsive_ind)

most_taste_responsive_segment_spike_times = []
for s_i in range(len(segment_names)):
	most_taste_responsive_segment_spike_times.append([segment_spike_times[s_i][tri] for tri in most_taste_responsive_ind])
most_taste_responsive_tastant_spike_times = []
most_taste_responsive_tastant_PSTH = []
for t_i in range(len(dig_in_names)):
	d_collect = []
	for d_i in range(len(start_dig_in_times[t_i])):
		d_collect.append([tastant_spike_times[t_i][d_i][tri] for tri in most_taste_responsive_ind])
	most_taste_responsive_tastant_spike_times.append(d_collect)
	most_taste_responsive_tastant_PSTH.append(tastant_PSTH[t_i][:,most_taste_responsive_ind,:])

#%%
#_____Grab and plot firing rate distributions and comparisons (by segment)_____
sc_save_dir = data_save_dir + 'Segment_Comparison/'
if os.path.isdir(sc_save_dir) == False:
	os.mkdir(sc_save_dir)
	
#All data
all_sc_save_dir = sc_save_dir + 'All/'
if os.path.isdir(all_sc_save_dir) == False:
	os.mkdir(all_sc_save_dir)
sc.bin_spike_counts(all_sc_save_dir,segment_spike_times,segment_names,segment_times)
#Taste responsive data
# taste_resp_sc_save_dir = sc_save_dir + 'Taste_Responsive/'
# if os.path.isdir(taste_resp_sc_save_dir) == False:
# 	os.mkdir(taste_resp_sc_save_dir)
# sc.bin_spike_counts(taste_resp_sc_save_dir,taste_responsive_tastant_spike_times,segment_names,segment_times)
# #Most taste responsive data
# most_taste_resp_sc_save_dir = sc_save_dir + 'Most_Taste_Responsive/'
# if os.path.isdir(most_taste_resp_sc_save_dir) == False:
# 	os.mkdir(most_taste_resp_sc_save_dir)
# sc.bin_spike_counts(most_taste_resp_sc_save_dir,most_taste_responsive_tastant_spike_times,segment_names,segment_times)
  
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
	taste_cp_raster_pop_save_dir = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_cp_raster_pop_save_dir')
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
	taste_cp_raster_inds_pop = cd.calc_cp_iter_pop(tastant_spike_times,cp_bin,num_cp,start_dig_in_times,
				  end_dig_in_times,before_taste,after_taste,
				  dig_in_names,taste_cp_raster_pop_save_dir)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_cp_raster_inds_pop',taste_cp_raster_inds_pop)

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
	
#____Calculate which neurons are taste selective by epoch_____
try:
	taste_select_prob_joint = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob_joint')[0]
	taste_select_prob_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob_epoch')[0]
	prob_taste_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'prob_taste_epoch')[0]
except:
	print("Using changepoint indices to calculate taste selectivity by epoch.")
	#taste_select_prob_joint = Fraction of deliveries that are successfully decoded by neuron,taste - multiplying across epochs
	#taste_select_prob_epoch = Fraction of deliveries that are successfully decoded by neuron,taste,epoch
	taste_select_prob_joint, taste_select_prob_epoch, prob_taste_epoch = df.taste_decoding_cp(tastant_spike_times, \
												 taste_cp_raster_inds,num_cp,start_dig_in_times, \
												 end_dig_in_times,dig_in_names,num_neur,pre_taste_dt, \
												 post_taste_dt,loo_distribution_save_dir)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_select_prob_joint',taste_select_prob_joint)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_select_prob_epoch',taste_select_prob_epoch)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'prob_taste_epoch',prob_taste_epoch)

#Plot the decoding - * denotes statitical significance in KW nonparametric multiple-comparisons test
pf.epoch_taste_select_plot(prob_taste_epoch, dig_in_names, decoding_save_dir)
epoch_max_decoding, epoch_select_neur = pf.taste_select_plot(taste_select_prob_epoch, np.arange(num_cp), 'epoch', 'taste_selectivity_by_epoch', decoding_save_dir)

#_____All Epochs Included_____
#Select the taste responsive neurons using taste_select_prob_joint (matrix num_neur x num_tastes)
taste_response_prob = np.expand_dims((np.sum(taste_select_prob_joint,1)/3 >= 1/num_tastes).astype('int'),1)*np.ones((num_neur,num_tastes))
#Select the taste selective neurons using taste_select_prob_joint
taste_select_prob = ((taste_select_prob_joint >= 1/num_tastes).astype('int'))*np.ones((num_neur,num_tastes))

#_____Using Highest Probability Epoch_____



#%% Plot changepoint statistics

#TODO: move to plot_funcs document

import matplotlib.pyplot as plt
from matplotlib import cm

colors = cm.cool(np.arange(num_cp)/(num_cp))

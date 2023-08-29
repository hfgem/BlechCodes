#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:04:51 2023

@author: hannahgermaine

This code is written to import sorted data and perform state-change analyses
"""
import os
file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
os.chdir(file_path)
import numpy as np
import functions.analysis_funcs as af
import functions.hdf5_handling as hf5
import functions.plot_funcs as pf
import functions.dev_calcs as dev_calc
import functions.dev_plots as dev_plot
import functions.seg_compare as sc
import functions.dev_corr_calcs as dcc
import functions.changepoint_detection as cd



#_____Get the directory of the hdf5 file_____
sorted_dir, segment_dir, cleaned_dir = hf5.sorted_data_import() #Program will automatically quit if file not found in given folder
fig_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'
print('\nData Directory:')
print(fig_save_dir)

#%%
#_____Import data_____
#todo: update intan rhd file import code to accept directory input
num_neur, all_waveforms, spike_times, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes = af.import_data(sorted_dir, segment_dir, fig_save_dir)

#%%
#_____Calculate spike time datasets_____
pre_taste = 0.5 #Seconds before tastant delivery to store
post_taste = 2 #Seconds after tastant delivery to store

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
	PSTH_times = af.pull_data_from_hdf5(sorted_dir,data_group_name,'PSTH_times')
	PSTH_taste_deliv_times = af.pull_data_from_hdf5(sorted_dir,data_group_name,'PSTH_taste_deliv_times')
	avg_tastant_PSTH = af.pull_data_from_hdf5(sorted_dir,data_group_name,'avg_tastant_PSTH')
except:
	pf.raster_plots(fig_save_dir, dig_in_names, start_dig_in_times, end_dig_in_times, 
					segment_names, segment_times, segment_spike_times, tastant_spike_times, 
					pre_taste_dt, post_taste_dt, num_neur, num_tastes)
	PSTH_times, PSTH_taste_deliv_times, tastant_PSTH, avg_tastant_PSTH = pf.PSTH_plots(fig_save_dir, num_tastes,
																					   num_neur, dig_in_names, 
																					   start_dig_in_times, end_dig_in_times, 
																					   pre_taste_dt, post_taste_dt, 
																					   segment_times, spike_times,
																					   bin_width, bin_step)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'PSTH_times',PSTH_times)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'PSTH_taste_deliv_times',PSTH_taste_deliv_times)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'avg_tastant_PSTH',avg_tastant_PSTH)

#%%
#____Calculate which neurons are taste responsive_____



#%%
#_____Grab and plot firing rate distributions and comparisons (by segment)_____
sc_save_dir = fig_save_dir + 'Segment_Comparison/'
if os.path.isdir(sc_save_dir) == False:
	os.mkdir(sc_save_dir)
		
sc.bin_spike_counts(sc_save_dir,segment_spike_times,segment_names,segment_times)
  
#%%
#_____Grab and plot bins above a neuron count threshold by different count values_____
num_thresh = np.arange(2,num_neur)
bin_size = 0.05 #size of test bin in seconds

thresh_bin_save_dir = fig_save_dir + 'thresholded_deviations/'
if os.path.isdir(thresh_bin_save_dir) == False:
	os.mkdir(thresh_bin_save_dir)

neur_bout_seg_thresh, neur_bout_seg = sc.bin_neur_spike_counts(thresh_bin_save_dir,segment_spike_times,segment_names,segment_times,num_thresh,bin_size)

#%%
#Quick and dirty changepoint detection algorithm based on percentile of bin fr 
#difference. It finds the percentile of each bin difference for each neuron,  
#then sums across neurons and uses find peaks to find potential changepoints. 
#Peaks above the cutoff probability (fraction of neurons with changepoints) are
#kept as changepoints.

cp_bin = 250 #minimum state size in ms
num_cp = 3 #number of changepoints to find
before_taste = np.ceil(pre_taste*1000).astype('int') #Milliseconds before taste delivery to plot
after_taste = np.ceil(post_taste*1000).astype('int') #Milliseconds after taste delivery to plot

#Set storage directory
cp_save_dir = fig_save_dir + 'Changepoint_Calculations/'
if os.path.isdir(cp_save_dir) == False:
	os.mkdir(cp_save_dir)
taste_cp_save_dir = cp_save_dir + 'Taste_CPs/'
if os.path.isdir(taste_cp_save_dir) == False:
	os.mkdir(taste_cp_save_dir)

data_group_name = 'changepoint_data'

try:
	taste_cp_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_cp_inds')
	taste_avg_cp_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_avg_cp_inds')
	taste_avg_cp_times = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_avg_cp_times')
except:
	taste_cp_inds, taste_avg_cp_inds, taste_avg_cp_times = cd.calc_cp_taste_PSTH_ks_test(PSTH_times, 
								PSTH_taste_deliv_times, tastant_PSTH, cp_bin, bin_step, dig_in_names,
								num_cp, before_taste, after_taste, taste_cp_save_dir)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_cp_inds',taste_cp_inds)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_avg_cp_inds',taste_avg_cp_inds)
	af.add_data_to_hdf5(sorted_dir,data_group_name,'taste_avg_cp_times',taste_avg_cp_times)
	
#%%
#_____Calculate cross-correlation between post-taste-delivery data and threshold deviation bins_____
#uses taste_avg_cp_times from above
#Set up parameters
taste_interval_names = ['Presence','Identity','Palatability']
dc_save_dir = fig_save_dir + 'Thresh_Dev_Correlations/'
if os.path.isdir(dc_save_dir) == False:
	os.mkdir(dc_save_dir)
	
#TEMPORARY WORKAROUND: Select which threshold value to use for this
#NEED TO CHECK: segment_bout_vals should contain original indices, not within-bout indices.
thresh_cutoff = int(np.ceil(0.5*num_neur))
segment_bout_vals = []
for s_i in range(len(segment_names)):
	try:
		ind_cutoff_data = np.where(np.array(neur_bout_seg_thresh[s_i]) == thresh_cutoff)[0][0]
		segment_bout_vals.append(neur_bout_seg[s_i][ind_cutoff_data])
	except:
		print("Cutoff data doesn't exist for segment " + segment_names[s_i])
		segment_bout_vals.append(np.empty(0))
	
dcc.dev_corr(dc_save_dir,segment_spike_times,segment_names,segment_times,segment_bout_vals,
			 tastant_spike_times, dig_in_names, start_dig_in_times, end_dig_in_times, 
			 taste_avg_cp_times, taste_interval_names)

#%%





#%%
#For future changes: add user input to asign parameters

#_____Grab and plot firing rate deviations from local mean (by segment)_____
local_bin_size = 20 #bin size for local interval to compute mean firing rate (in seconds)
deviation_bin_size = 0.05 #bin size for which to compute deviation value (in seconds)
fig_buffer_size = 1; #How many seconds in either direction to plot for a deviation event raster
dev_thresh = 0.95 #Cutoff for high deviation bins to keep
std_cutoff = 4 #Cutoff of number of standard deviations above mean a deviation must be to be considered a potential replay bin
partic_neur_cutoff = 1/3 #Cutoff for minimum fraction of neurons present in a deviation
num_null_sets = 100 #Number of null datasets to create

#Calculator functions
dev_save_dir,segment_devs,segment_dev_frac_ind,segment_bouts,segment_bout_lengths,\
segment_ibis,mean_segment_bout_lengths,std_segment_bout_lengths,mean_segment_ibis,\
std_segment_ibis,num_dev_per_seg,dev_per_seg_freq,null_segment_dev_counts,\
null_segment_dev_ibis,null_segment_dev_bout_len = dev_calc.FR_dev_calcs(fig_save_dir,segment_names,segment_times,\
											   segment_spike_times,num_neur,num_tastes,local_bin_size,\
												   deviation_bin_size,dev_thresh,std_cutoff,\
													   fig_buffer_size,partic_neur_cutoff,num_null_sets)
				
#Plot functions
dev_plot.plot_deviations(dev_save_dir, num_neur, segment_names, segment_times, 
					dev_thresh, segment_devs, segment_bouts, 
					segment_bout_lengths, segment_ibis, segment_spike_times, 
					num_null_sets, null_segment_dev_counts, null_segment_dev_ibis,
					null_segment_dev_bout_len, fig_buffer_size)

#%%
#_____Calculate cross-correlation between post-taste-delivery data and activity deviation bins_____
#Set up parameters
taste_intervals = [0,200,750,1500] #Must be in milliseconds = sampling rate of data
taste_interval_names = ['Presence','Identity','Palatability']
dc_save_dir = fig_save_dir + 'dev_correlations/'
if os.path.isdir(dc_save_dir) == False:
	os.mkdir(dc_save_dir)

segment_bout_vals = [] #Resave in format of each segment has 2 numpy arrays - 0 index with start indices, 1 index with end indices
for s_i in range(len(segment_names)):
	seg_bout_starts = segment_bouts[s_i][:,0].astype('int')
	seg_bout_ends = segment_bouts[s_i][:,1].astype('int')
	segment_bout_vals.append([seg_bout_starts.flatten(),seg_bout_ends.flatten()])

segment_names_short = segment_names[0:3] #Temporary: to only plot first 3 segments
segment_times_short = segment_times[0:4] #Temporary: to only plot first 3 segments
dcc.dev_corr(dc_save_dir,segment_spike_times,segment_names_short,segment_times_short,segment_bout_vals,
			 tastant_spike_times, dig_in_names, start_dig_in_times, end_dig_in_times, 
			 taste_intervals, taste_interval_names)


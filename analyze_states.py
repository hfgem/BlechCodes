#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:04:51 2023

@author: hannahgermaine

This code is written to import sorted data and perform state-change analyses
"""
import os, tables, tqdm, time, random
import tkinter as tk
import tkinter.filedialog as fd
file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
os.chdir(file_path)
import numpy as np
import functions.hdf5_handling as hf5
import functions.data_processing as dp
import functions.load_intan_rhd_format.load_intan_rhd_format as rhd
import functions.plot_funcs as pf
import functions.dev_calcs as dc
import functions.seg_compare as sc
import functions.dev_corr_calcs as dcc
import matplotlib.pyplot as plt

def import_data(sorted_dir, segment_dir, fig_save_dir):
	"""Import data from .h5 file and grab any missing data through user inputs.
	Note, all spike times, digital input times, etc... are converted to the ms
	timescale and returned as such.
	"""
	print("Beginning Data Import")
	tic = time.time()
	#_____Import spike times and waveforms_____
	#Grab data from hdf5 file
	blech_clust_h5 = tables.open_file(sorted_dir, 'r+', title = sorted_dir[-1])

	#Grab sampling rate
	print("\tGrabbing sampling rate")
	try:
		sampling_rate = blech_clust_h5.root.sampling_rate[0]
	except:
		#The old method doesn't currently store sampling_rate, so this picks it up
		rhd_dict = rhd.import_data()
		sampling_rate = int(rhd_dict["frequency_parameters"]["amplifier_sample_rate"])
		atom = tables.IntAtom()
		blech_clust_h5.create_earray('/','sampling_rate',atom,(0,))
		blech_clust_h5.root.sampling_rate.append([sampling_rate])
	#Calculate the conversion from samples to ms
	ms_conversion = (1/sampling_rate)*(1000/1) #ms/samples units
	
	sorted_units_node = blech_clust_h5.get_node('/sorted_units')
	num_neur = len([s_n for s_n in sorted_units_node])
	#Grab waveforms
	print("\tGrabbing waveforms")
	all_waveforms = [sn.waveforms[0] for sn in sorted_units_node]
	#Grab times
	print("\tGrabbing spike times")
	spike_times = []
	i = 0
	for s_n in sorted_units_node:
		spike_times.append(list(s_n.times))
		i+= 1
	#Converting spike times to ms timescale
	spike_times = [np.ceil(np.array(spike_times[i])*ms_conversion) for i in range(len(spike_times))]
	
	#Grab digital inputs
	print("\tGrabbing digital input times")
	dig_in_node = blech_clust_h5.list_nodes('/digital_in')
	dig_in_names = np.array([d_i.name.split('_')[-1] for d_i in dig_in_node])
	dig_in_ind = []
	i = 0
	for d_i in dig_in_names:
		try:
			int(d_i)
			dig_in_ind.extend([i])
		except:
			"not an input - do nothing"
		i += 1
	dig_in_data = [list(dig_in_node[d_i]) for d_i in dig_in_ind]
	num_dig_in = len(dig_in_data)
	del dig_in_node
	
	#Grab dig in names
	print("\tGrabbing digital input names")
	try:
		dig_in_names = [blech_clust_h5.root.digital_in.dig_in_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.digital_in.dig_in_names))]
	except:
		#The old method doesn't currently store tastant names, so this probes the user
		dig_in_names = list()
		for i in range(num_dig_in):
			d_loop = 1
			while d_loop == 1:
				d_name = input("\n INPUT REQUESTED: Enter single-word name for dig-in " + str(i) + ": ")
				if len(d_name) < 2:
					print("Error, entered name seems too short. Please try again.")
				else:
					d_loop = 0
			dig_in_names.append(d_name)
		#Save data for future use
		atom = tables.Atom.from_dtype(np.dtype('U20')) #tables.StringAtom(itemsize=50)
		dig_names = blech_clust_h5.create_earray('/digital_in','dig_in_names',atom,(0,))
		dig_names.append(np.array(dig_in_names))

	#_____Import segment times - otherwise ask for user segment time input_____
	print("\tGrabbing segment times and names")
	try:
		segment_times = blech_clust_h5.root.experiment_components.segment_times[:]
		#Convert segment_times to ms timescale
		segment_times = np.ceil(segment_times*ms_conversion)
		segment_names = [blech_clust_h5.root.experiment_components.segment_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.experiment_components.segment_names))]
	except:
		segment_names, segment_times = dp.get_experiment_components(sampling_rate, dig_in_data)
		#Save data for future use
		blech_clust_h5 = tables.open_file(sorted_dir, 'r+', title = sorted_dir[-1])
		try:
			blech_clust_h5.create_group('/','experiment_components')
		except:
			print("\n\tExperiment components group already exists in .h5 file")
		atom = tables.IntAtom()
		try:
			blech_clust_h5.create_earray('/experiment_components','segment_times',atom,(0,))
			exec("blech_clust_h5.root.experiment_components.segment_times.append(segment_times[:])")
		except:
			print("\t\tSegment times previously stored")
		atom = tables.Atom.from_dtype(np.dtype('U20'))
		try:
			blech_clust_h5.create_earray('/experiment_components','segment_names',atom,(0,))
			exec("blech_clust_h5.root.experiment_components.segment_names.append(np.array(segment_names))")
		except:
			print("\t\tSegment names previously stored")
		#Convert segment times to ms timescale
		segment_times = np.ceil(segment_times*ms_conversion)
		
	blech_clust_h5.close() #Always close the file

	#_____Convert dig_in_data to indices of dig_in start and end times_____
	print("\tConverting digital inputs to free memory")
	#Again, all are converted to ms timescale
	start_dig_in_times = [list(np.ceil((np.where(np.diff(np.array(dig_in_data[i])) == 1)[0] + 1)*ms_conversion).astype('int')) for i in range(len(dig_in_data))]	
	end_dig_in_times = [list(np.ceil((np.where(np.diff(np.array(dig_in_data[i])) == -1)[0] + 1)*ms_conversion).astype('int')) for i in range(len(dig_in_data))]
	num_tastes = len(start_dig_in_times)
	del dig_in_data
	toc = time.time()
	print("Time to import data = " + str(round((toc - tic)/60)) + " minutes \n")	
	return num_neur, all_waveforms, spike_times, num_dig_in, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes

#%%_____Get the directory of the hdf5 file_____
sorted_dir, segment_dir, cleaned_dir = hf5.sorted_data_import() #Program will automatically quit if file not found in given folder
fig_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'
print('\nData Directory:')
print(fig_save_dir)

#%%
#_____Import data_____
num_neur, all_waveforms, spike_times, num_dig_in, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes = import_data(sorted_dir, segment_dir, fig_save_dir)

#%%

#_____Grab Spike Times + Generate Raster Plots_____
#segment_spike_times is a list nested by segments x num_neur x num_time
#tastant_spike_times is a list with only taste delivery data nested by tastant_delivery x num_neur x num_time
segment_spike_times, tastant_spike_times, pre_taste_dt, post_taste_dt = pf.raster_plots(fig_save_dir, 
														   dig_in_names, start_dig_in_times, 
														   end_dig_in_times, segment_names, 
														   segment_times, spike_times, num_neur, 
														   num_tastes)

#_____Grab and plot PSTHs for each taste separately_____

PSTH_times, PSTH_taste_deliv_times, tastant_PSTH, avg_tastant_PSTH = pf.PSTH_plots(fig_save_dir, num_tastes,
																				   num_neur, dig_in_names, 
																				   start_dig_in_times, end_dig_in_times, 
																				   pre_taste_dt, post_taste_dt, 
																				   segment_times, spike_times)

#%% IN PROGRESS
#_____Grab and plot firing rate deviations from local mean (by segment)_____
local_bin_size = 30 #bin size for local interval to compute mean firing rate (in seconds)
deviation_bin_size = 0.05 #bin size for which to compute deviation value (in seconds)
fig_buffer_size = 1; #How many seconds in either direction to plot for a deviation event raster
dev_thresh = 0.95 #Cutoff for high deviation bins to keep
std_cutoff = 4 #Cutoff of number of standard deviations above mean a deviation must be to be considered a potential replay bin
partic_neur_cutoff = 0.1 #Cutoff for minimum fraction of neurons present in a deviation

#Calculator functions: THESE STILL THROW AN ERROR: USE THE BELOW CODE BLOCK IN THE MEANTIME
segment_devs,segment_dev_frac_ind,segment_bouts,segment_bout_lengths,segment_ibis,\
	mean_segment_bout_lengths,std_segment_bout_lengths,mean_segment_ibis,std_segment_ibis,\
		num_dev_per_seg,dev_per_seg_freq,null_segment_dev_counts,null_segment_dev_ibis,\
			null_segment_dev_bout_len = dc.FR_dev_calcs(fig_save_dir,segment_names,segment_times,\
											   segment_spike_times,num_neur,num_tastes,local_bin_size,\
												   deviation_bin_size,dev_thresh,std_cutoff,\
													   fig_buffer_size,partic_neur_cutoff)
				
#Plot functions [INSERT BELOW ONCE WRITTEN]

#%%	
#_____Grab and plot firing rate deviations from local mean (by segment)_____
local_bin_size = 30 #bin size for local interval to compute mean firing rate (in seconds)
deviation_bin_size = 0.05 #bin size for which to compute deviation value (in seconds)
fig_buffer_size = 1; #How many seconds in either direction to plot for a deviation event raster
dev_thresh = 0.95 #Cutoff for high deviation bins to keep
std_cutoff = 4 #Cutoff of number of standard deviations above mean a deviation must be to be considered a potential replay bin
partic_neur_cutoff = 0.1 #Cutoff for minimum fraction of neurons present in a deviation
#Create results save directory
dev_save_dir = fig_save_dir + 'deviations/'
if os.path.isdir(dev_save_dir) == False:
	os.mkdir(dev_save_dir)

segment_devs, segment_bouts, segment_bout_lengths, segment_ibis, mean_segment_bout_lengths,\
	 std_segment_bout_lengths, mean_segment_ibis, std_segment_ibis, null_segment_dev_counts,\
		 null_segment_dev_ibis, null_segment_dev_bout_len, null_segment_dev_rasters = pf.FR_deviation_plots(dev_save_dir,\
																		   segment_names,segment_times,\
																			   segment_spike_times,num_neur,\
																				   num_tastes,local_bin_size,\
																					   deviation_bin_size,dev_thresh,\
																						   std_cutoff,fig_buffer_size,\
																							   partic_neur_cutoff)

#%%
#_____Grab and plot bins above a neuron count threshold by different count values_____
num_thresh = np.arange(2,num_neur)
bin_size = 0.05 #size of test bin in seconds

thresh_bin_save_dir = fig_save_dir + 'thresholded_deviations/'
if os.path.isdir(thresh_bin_save_dir) == False:
	os.mkdir(thresh_bin_save_dir)

sc.bin_neur_spike_counts(thresh_bin_save_dir,segment_spike_times,segment_names,segment_times,num_thresh,bin_size)


#%%
#_____Grab and plot firing rate distributions and comparisons (by segment)_____
sc_save_dir = fig_save_dir + 'Segment_Comparison/'
if os.path.isdir(sc_save_dir) == False:
	os.mkdir(sc_save_dir)
		
sc.bin_spike_counts(sc_save_dir,segment_spike_times,segment_names,segment_times)
  


#%%
#_____Calculate cross-correlation between post-taste-delivery data and deviation bins_____
#Set up parameters
taste_intervals = [0,200,700,1500]
dc_save_dir = fig_save_dir + 'Dev_Correlations/'
if os.path.isdir(dc_save_dir) == False:
	os.mkdir(dc_save_dir)

#Grab segment rasters
segment_dev_rasters = []
for s_i in range(len(segment_names)):
	#First create a binary segment array
	segment_spikes = segment_spike_times[s_i]
	start_segment = segment_times[s_i]
	end_segment = segment_times[s_i+1]
	seg_len = int(end_segment - start_segment)
	segment_bin = np.zeros((num_neur,seg_len))
	for n_i in range(num_neur):
		segment_bin[n_i,segment_spikes[n_i]] += 1
	#Then pull out individual bouts
	seg_bout_list = segment_bouts[s_i]
	segment_dev_rasts = []
	for s_b in range(len(seg_bout_list)):
		segment_dev_rasts.append(segment_bin[:,seg_bout_list[s_b,0]:seg_bout_list[s_b,1]])
	segment_dev_rasters.append(segment_dev_rasts)
	
dcc.dev_corr(dc_save_dir,segment_dev_rasters,null_segment_dev_rasters,taste_intervals,tastant_spike_times)


#%%
#_____Plot what the original recording looks like around times of deviation_____
if len(cleaned_dir) < 2: #First check if clean data actually exists for this analysis
	print("No cleaned LFP data found in directory given.")
	clean_loop = 0
	while clean_loop == 0:
		clean_exists = input("Does cleaned data exist elsewhere (y/n)? ")
		if clean_exists != 'y' and clean_exists != 'n':
			print("Incorrect response given. Try again.")
		else:
			clean_loop = 1
	if clean_exists == 'y':
		print("Please select the directory where the cleaned data exists.")
		clean_loop = 0
		while clean_loop == 0:
			root = tk.Tk()
			currdir = os.getcwd()
			cleaned_folder_dir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')
			files_in_dir = os.listdir(cleaned_folder_dir)
			for i in range(len(files_in_dir)): #Checks for repacked and downsampled .h5 in the directory
				filename = files_in_dir[i]
				if filename.split('_')[-1] == 'cleaned.h5':
					cleaned_filename = filename
					print(filename)
					cleaned_dir = cleaned_folder_dir + '/' + cleaned_filename
					clean_loop = 1
			if clean_loop == 0:
				print("Clean data not found in folder provided. LFP analysis will not proceed.")
				clean_loop = 1
else:
	clean_exists = 'y'
#Now perform the analysis
if clean_exists == 'y':
	#Import LFP data
	clean_h5 = tables.open_file(cleaned_dir, 'r+', title = cleaned_dir[-1])
	LFP_data = clean_h5.root.lfp_data[0]
	wave_sampling_rate = clean_h5.root.sampling_rate[0]
	#clean_data = clean_h5.root.clean_data[0]
	clean_h5.close()
	#time_bouts = np.arange(len(clean_data[0,:]),step=sampling_rate)
	#combined_waveforms = np.zeros(np.shape(clean_data))
	#print("Combining LFP and spike waveforms for full range of frequencies.")
	#for t_i in tqdm.tqdm(range(len(time_bouts)-1)):
	#	combined_waveforms[:,time_bouts[t_i]:time_bouts[t_i+1]] = LFP_data[:,time_bouts[t_i]:time_bouts[t_i+1]] + clean_data[:,time_bouts[t_i]:time_bouts[t_i+1]]
	#del clean_data, LFP_data
	pf.LFP_dev_plots(fig_save_dir,segment_names,segment_times,fig_buffer_size,segment_bouts,LFP_data,wave_sampling_rate)


#NOTES TO SELF:
#Normalize counts by length of segment as well
#Add more bins to histograms

#%%	
#_____Perform changepoint detection on individual trial raster plots_____
# for t_i in range(num_tastes): #By tastant
# 	t_st = tastant_spike_times[t_i]
# 	num_trials = len(t_st)
# 	trial_times = []
# 	for n_t in range(num_trials): #By delivery trial
# 		n_st = t_st[n_t]
# 		#Convert spike times to binary raster
# 		max_index = 0
# 		for i in range(num_neur):
# 			try:
# 				m_i = np.max(n_st[i])
# 			except:
# 				m_i = 0
# 			if m_i > max_index:
# 				max_index = m_i
# 		binary_raster = np.zeros((num_neur,max_index + 1))
# 		for i in range(num_neur):
# 			binary_raster[i,n_st[i]] = 1
# 		#Send binary raster to changepoint detection algorithm
		

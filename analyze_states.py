#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:04:51 2023

@author: hannahgermaine

This code is written to import sorted data and perform state-change analyses
"""
import os, tables
file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
os.chdir(file_path)
import numpy as np
import functions.hdf5_handling as hf5
import functions.data_processing as dp
import functions.load_intan_rhd_format.load_intan_rhd_format as rhd
import functions.plot_funcs as pf

def import_data(sorted_dir, segment_dir, fig_save_dir):
	"""Import data from .h5 file and grab any missing data through user inputs"""
	print("Beginning Data Import")
	#_____Import spike times and waveforms_____
	#Grab data from hdf5 file
	blech_clust_h5 = tables.open_file(sorted_dir, 'r', title = sorted_dir[-1])
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
			a = "not an input - do nothing"
		i += 1
	dig_in_data = [list(dig_in_node[d_i]) for d_i in dig_in_ind]
	num_dig_in = len(dig_in_data)
	del dig_in_node
	#Grab sampling rate
	print("\tGrabbing sampling rate")
	try:
		sampling_rate = blech_clust_h5.root.sampling_rate[0]
	except:
		#The old method doesn't currently store sampling_rate, so this picks it up
		blech_clust_h5.close() #Close file
		rhd_dict = rhd.import_data()
		sampling_rate = int(rhd_dict["frequency_parameters"]["amplifier_sample_rate"])
		blech_clust_h5 = tables.open_file(sorted_dir, 'r+', title = sorted_dir[-1])
		atom = tables.IntAtom()
		blech_clust_h5.create_earray('/','sampling_rate',atom,(0,))
		blech_clust_h5.root.sampling_rate.append([sampling_rate])
	#Grab dig in names
	print("\tGrabbing digital input names")
	try:
		dig_in_names = [blech_clust_h5.root.digital_in.dig_in_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.digital_in.dig_in_names))]
	except:
		blech_clust_h5.close() #Close file
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
		blech_clust_h5 = tables.open_file(sorted_dir, 'r+', title = sorted_dir[-1])
		atom = tables.Atom.from_dtype(np.dtype('U20')) #tables.StringAtom(itemsize=50)
		dig_names = blech_clust_h5.create_earray('/digital_in','dig_in_names',atom,(0,))
		dig_names.append(np.array(dig_in_names))

	#_____Import segment times - otherwise ask for user segment time input_____
	print("\tGrabbing segment times and names")
	try:
		segment_times = blech_clust_h5.root.experiment_components.segment_times[:]
		segment_names = [blech_clust_h5.root.experiment_components.segment_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.experiment_components.segment_names))]
	except:
		segment_names, segment_times = dp.get_experiment_components(sampling_rate, dig_in_data)
		blech_clust_h5.close() #Close file
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
		
	blech_clust_h5.close() #Always close the file

	#_____Convert dig_in_data to indices of dig_in start and end times_____
	print("\tConverting digital inputs to free memory")
	start_dig_in_times = [list(np.where(np.diff(np.array(dig_in_data[i])) == 1)[0] + 1) for i in range(len(dig_in_data))]	
	end_dig_in_times = [list(np.where(np.diff(np.array(dig_in_data[i])) == -1)[0] + 1) for i in range(len(dig_in_data))]
	num_tastes = len(start_dig_in_times)
	del dig_in_data
	
	return num_neur, all_waveforms, spike_times, num_dig_in, sampling_rate, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes

#_____Get the directory of the hdf5 file_____
sorted_dir, segment_dir = hf5.sorted_data_import() #Program will automatically quit if file not found in given folder
fig_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'
print('\nData Directory:')
print(fig_save_dir)

#_____Import data_____
num_neur, all_waveforms, spike_times, num_dig_in, sampling_rate, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes = import_data(sorted_dir, segment_dir, fig_save_dir)

#%%

#_____Grab Spike Times + Generate Raster Plots_____
#segment_spike_times is a list nested by segments x num_neur x num_time
#tastant_spike_times is a list with only taste delivery data nested by tastant_delivery x num_neur x num_time
segment_spike_times, tastant_spike_times, pre_taste_dt, post_taste_dt = pf.raster_plots(fig_save_dir, 
														   dig_in_names, start_dig_in_times, 
														   end_dig_in_times, segment_names, 
														   segment_times, spike_times, num_neur, 
														   num_tastes, sampling_rate)

#_____Grab and plot PSTHs for each taste separately_____

PSTH_times, PSTH_taste_deliv_times, tastant_PSTH, avg_tastant_PSTH = pf.PSTH_plots(fig_save_dir, sampling_rate,
																				   num_tastes, num_neur, dig_in_names, 
																				   start_dig_in_times, end_dig_in_times, 
																				   pre_taste_dt, post_taste_dt, 
																				   segment_times, spike_times)

#%%
#_____Grab and plot firing rate deviations from local mean (by segment)_____
local_bin_size = 30 #bin size for local interval to compute mean firing rate (in seconds)
deviation_bin_size = 0.05 #bin size for which to compute deviation value (in seconds)
fig_buffer_size = 1; #How many seconds in either direction to plot for a deviation event raster
segment_devs, segment_bouts, segment_bout_lengths, segment_ibis, mean_segment_bout_lengths, std_segment_bout_lengths, mean_segment_ibis, std_segment_ibis = pf.FR_deviation_plots(fig_save_dir,sampling_rate,
																																																			 segment_names,segment_times,
																																																			 segment_spike_times,num_neur,
																																																			 num_tastes,local_bin_size,
																																																			 deviation_bin_size,fig_buffer_size)



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
		

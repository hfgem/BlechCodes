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
import matplotlib.pyplot as plt
import functions.hdf5_handling as hf5
import functions.data_processing as dp
import functions.load_intan_rhd_format.load_intan_rhd_format as rhd


#Get the directory of the hdf5 file
sorted_dir, segment_dir = hf5.sorted_data_import() #Program will automatically quit if file not found in given folder

#_____Import spike times and waveforms_____
#Transform data from blech_clust hdf5 file into correct format
blech_clust_h5 = tables.open_file(sorted_dir, 'r', title = sorted_dir[-1])
sorted_units_node = blech_clust_h5.get_node('/sorted_units')
num_neur = len([s_n for s_n in sorted_units_node])
#Grab waveforms
all_waveforms = [sn.waveforms[:] for sn in sorted_units_node]
#Grab times
spike_times = []
i = 0
for s_n in sorted_units_node:
	spike_times.append(list(s_n.times[:]))
	i+= 1
del s_n, sorted_units_node

#Grab digital inputs
dig_in_node = blech_clust_h5.get_node('/digital_in')
dig_in_data = [list(d_i) for d_i in dig_in_node]
num_dig_in = len(dig_in_data)
del dig_in_node
#Grab sampling rate
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

blech_clust_h5.close() #Always close the file

#_____Import segment times - otherwise ask for user segment time input_____
if len(segment_dir) <= 1: #segment data doesn't exist, ask for user input
	
	segment_names, segment_times = dp.get_experiment_components(sampling_rate, dig_in_data)
else: #segment data exists
	try:
		segment_h5 = tables.open_file(segment_dir, 'r', title = segment_dir[-1])
		segment_times = segment_h5.root.experiment_components.segment_times[:]
		segment_names = [segment_h5.root.experiment_components.segment_names[i].decode('UTF-8') for i in range(len(segment_h5.root.segment_names))]
		segment_h5.close()
	except:
		segment_names, segment_times = dp.get_experiment_components(sampling_rate, dig_in_data)

#_____Convert dig_in_data to indices of dig_in start and end times_____
start_dig_in_times = [list(np.where(np.diff(np.array(dig_in_data[i])) == 1)[0] + 1) for i in range(len(dig_in_data))]	
end_dig_in_times = [list(np.where(np.diff(np.array(dig_in_data[i])) == -1)[0] + 1) for i in range(len(dig_in_data))]
del dig_in_data

#_____Grab spike times for each segment separately_____
segment_spike_times = []
for s_i in tqdm.tqdm(range(len(segment_names))):
	print("\nGrabbing spike raster for segment " + segment_names[s_i])
	min_time = segment_times[s_i]
	max_time = segment_times[s_i+1]
	s_t = [list(np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= min_time)*(np.array(spike_times[i]) <= max_time))[0]]) for i in range(num_neur)]
	#Plot segment rasters and save
	plt.figure()
	plt.eventplot(s_t)
	plt.title(segment_names[s_i] + " segment")
	plt.tight_layout()
	im_name = ('_').join(segment_names[s_i].split(' ')) + '.png'
	plt.savefig(('/').join(sorted_dir.split('/')[0:-1]) + '/' + im_name)
	segment_spike_times.append(s_t)

#_____



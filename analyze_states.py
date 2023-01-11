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
fig_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'

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
del s_n, sorted_units_node, i

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
#Grab dig in names
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
try:
	segment_times = blech_clust_h5.root.experiment_components.segment_times[:]
	segment_names = [blech_clust_h5.root.experiment_components.segment_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.segment_names))]
except:
	segment_names, segment_times = dp.get_experiment_components(sampling_rate, dig_in_data)
	blech_clust_h5.close() #Close file
	#Save data for future use
	blech_clust_h5 = tables.open_file(sorted_dir, 'r+', title = sorted_dir[-1])
	try:
		blech_clust_h5.create_group('/','experiment_components')
	except:
		print("Experiment components group already exists in .h5 file.")
	atom = tables.IntAtom()
	blech_clust_h5.create_earray('/experiment_components','segment_times',atom,(0,))
	exec("blech_clust_h5.root.experiment_components.segment_times.append(segment_times[:])")
	atom = tables.Atom.from_dtype(np.dtype('U20'))
	blech_clust_h5.create_earray('/experiment_components','segment_names',atom,(0,))
	exec("blech_clust_h5.root.experiment_components.segment_names.append(np.array(segment_names))")

blech_clust_h5.close() #Always close the file

#_____Convert dig_in_data to indices of dig_in start and end times_____
start_dig_in_times = [list(np.where(np.diff(np.array(dig_in_data[i])) == 1)[0] + 1) for i in range(len(dig_in_data))]	
end_dig_in_times = [list(np.where(np.diff(np.array(dig_in_data[i])) == -1)[0] + 1) for i in range(len(dig_in_data))]
num_tastes = len(start_dig_in_times)
del dig_in_data

#_____Grab spike times (and rasters) for each segment separately_____
raster_save_dir = fig_save_dir + 'rasters/'
if os.path.isdir(raster_save_dir) == False:
	os.mkdir(raster_save_dir)
segment_spike_times = []
for s_i in tqdm.tqdm(range(len(segment_names))):
	print("\nGrabbing spike raster for segment " + segment_names[s_i])
	min_time = segment_times[s_i]
	max_time = segment_times[s_i+1]
	s_t = [list(np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= min_time)*(np.array(spike_times[i]) <= max_time))[0]]) for i in range(num_neur)]
	#Plot segment rasters and save
	plt.figure(figsize=(30,num_neur))
	plt.eventplot(s_t)
	plt.title(segment_names[s_i] + " segment")
	plt.tight_layout()
	im_name = ('_').join(segment_names[s_i].split(' ')) + '.png'
	plt.savefig(raster_save_dir + im_name)
	plt.close()
	segment_spike_times.append(s_t)

#_____Grab spike times for each taste delivery separately_____
pre_taste = 0.5 #Seconds before tastant delivery to store
pre_taste_dt = int(np.ceil(pre_taste*sampling_rate))
post_taste = 2 #Seconds after tastant delivery to store
post_taste_dt = int(np.ceil(post_taste*sampling_rate))
tastant_spike_times = []
for t_i in tqdm.tqdm(range(num_tastes)):
	print("\nGrabbing spike rasters for tastant " + dig_in_names[t_i] + " deliveries")
	t_start = start_dig_in_times[t_i]
	t_end = end_dig_in_times[t_i]
	num_deliv = len(t_start)
	t_st = []
	t_fig = plt.figure(figsize=(10,num_deliv))
	for t_d_i in range(len(t_start)):
		start_i = int(max(t_start[t_d_i] - pre_taste_dt,0))
		end_i = int(min(t_end[t_d_i] + post_taste_dt,segment_times[-1]))
		#Grab spike times into one list
		s_t = [list(np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= start_i)*(np.array(spike_times[i]) <= end_i))[0]]) for i in range(num_neur)]
		t_st.append(s_t)
		#Plot the raster
		plt.subplot(num_deliv,1,t_d_i+1)
		plt.eventplot(s_t)
		plt.axvline(t_start[t_d_i],color='r')
		plt.axvline(t_end[t_d_i],color='r')
	#Save the figure
	im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_spike_rasters.png'
	t_fig.tight_layout()
	t_fig.savefig(raster_save_dir + im_name)
	t_fig.close()
	tastant_spike_times.append(t_st)

#_____Grab and plot PSTHs for each taste separately_____
PSTH_save_dir = fig_save_dir + 'PSTHs/'
if os.path.isdir(PSTH_save_dir) == False:
	os.mkdir(PSTH_save_dir)
bin_width = 0.25 #Gaussian convolution kernel width in seconds
half_bin_width_dt = int(np.ceil(sampling_rate*bin_width/2))
bin_step = 25 #Step size in dt to take in PSTH calculation
PSTH_times = [] #Storage of time bin true times (s) for each tastant
PSTH_taste_deliv_times = [] #Storage of tastant delivery true times (s) for each tastant [start,end]
tastant_PSTH = []
avg_tastant_PSTH = []
for t_i in tqdm.tqdm(range(num_tastes)):
	print("\nGrabbing PSTHs for tastant " + dig_in_names[t_i] + " deliveries")
	t_start = np.array(start_dig_in_times[t_i])
	t_end = np.array(end_dig_in_times[t_i])
	dt_total = np.max(t_end-t_start) + pre_taste_dt + post_taste_dt
	num_deliv = len(t_start)
	PSTH_start_times = np.arange(0,dt_total,bin_step)
	PSTH_true_times = np.round(PSTH_start_times/sampling_rate,3)
	PSTH_times.append(PSTH_true_times)
	start_deliv_interval = PSTH_true_times[np.where(PSTH_start_times > pre_taste_dt)[0][0]]
	end_deliv_interval = PSTH_true_times[np.where(PSTH_start_times > dt_total - post_taste_dt)[0][0]]
	PSTH_taste_deliv_times.append([start_deliv_interval,end_deliv_interval])
	all_PSTH = np.zeros((num_deliv,num_neur,len(PSTH_start_times)))
	t_fig = plt.figure(figsize=(10,num_deliv))
	for t_d_i in range(len(t_start)):
		start_i = int(max(t_start[t_d_i] - pre_taste_dt,0))
		end_i = int(min(t_end[t_d_i] + post_taste_dt,segment_times[-1]))
		bin_spikes = np.zeros((num_neur,dt_total))
		#Convert spike times into a binary vector
		for i in range(num_neur):
			s_t = np.array((np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= start_i)*(np.array(spike_times[i]) <= end_i))[0]]))
			s_t -= start_i
			bin_spikes[i,s_t] = 1
		del i, s_t
		#Perform Gaussian convolution
		PSTH_spikes = np.zeros((num_neur,len(PSTH_start_times)))
		for i in range(len(PSTH_start_times)):
			PSTH_spikes[:,i] = (np.sum(bin_spikes[:,max(PSTH_start_times[i]-half_bin_width_dt,0):min(PSTH_start_times[i]+half_bin_width_dt,dt_total)],1)/bin_width).T
		del i
		plt.subplot(num_deliv,1,t_d_i+1)
		#Update to have x-axis in time
		for i in range(num_neur):
			plt.plot(PSTH_true_times,PSTH_spikes[i,:])
		del i
		plt.axvline(start_deliv_interval,color='r')
		plt.axvline(end_deliv_interval,color='r')
		all_PSTH[t_d_i,:,:] = PSTH_spikes
	del t_d_i, start_i, end_i, bin_spikes, PSTH_spikes
	tastant_name = dig_in_names[t_i]
	im_name = ('_').join((tastant_name).split(' ')) + '_PSTHs.png'
	t_fig.tight_layout()
	t_fig.savefig(PSTH_save_dir + im_name)
	t_fig.close()
	avg_PSTH = np.mean(all_PSTH,axis=0)
	t_fig = plt.figure()
	for i in range(num_neur):
		plt.plot(PSTH_true_times,avg_PSTH[i,:])
	del i
	plt.axvline(start_deliv_interval,color='r',linestyle = 'dashed')
	plt.axvline(end_deliv_interval,color='r',linestyle='solid')
	plt.title('Avg Individual Neuron PSTH for ' + tastant_name + '\nAligned to Taste Delivery Start')
	plt.xlabel('Time (s)')
	plt.ylabel('Firing Rate (Hz)')
	im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_avg_PSTH.png'
	t_fig.tight_layout()
	t_fig.savefig(PSTH_save_dir + im_name)
	t_fig.close()
	tastant_PSTH.append(all_PSTH)
	avg_tastant_PSTH.append(avg_PSTH)
del t_i, t_start, t_end, dt_total, num_deliv, PSTH_start_times, PSTH_true_times, start_deliv_interval, end_deliv_interval, all_PSTH
del t_fig, tastant_name, im_name, avg_PSTH
	
#_____Plot avg PSTHs for Individual Neurons_____
neuron_PSTH_dir = PSTH_save_dir + 'neurons/'
if os.path.isdir(neuron_PSTH_dir) == False:
	os.mkdir(neuron_PSTH_dir)
for n_i in range(num_neur):
	plt.figure(figsize=(10,10))
	for t_i in range(num_tastes):
		plt.plot(PSTH_times[t_i],avg_tastant_PSTH[t_i][n_i,:],label=dig_in_names[t_i])
		plt.bar(PSTH_taste_deliv_times[t_i][0],height=-1,width=PSTH_taste_deliv_times[t_i][1] - PSTH_taste_deliv_times[t_i][0],alpha=0.1,label=dig_in_names[t_i] + ' delivery')
	plt.legend(loc='upper right',fontsize=12)
	plt.title('Neuron ' + str(n_i))
	plt.ylabel('Firing Rate (Hz)')
	plt.xlabel('Time (s)')
	plt.savefig(neuron_PSTH_dir + 'neuron_' + str(n_i) + '_PSTHs.png')
	plt.close()
	
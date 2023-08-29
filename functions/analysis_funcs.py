#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:46:03 2023

@author: Hannah Germaine
Set of miscellaneous functions to support analyses in analyze_states.py.
"""

import time,tables,tqdm,os,csv
import numpy as np
import functions.load_intan_rhd_format.load_intan_rhd_format as rhd
import functions.data_processing as dp

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
	max_times = []
	i = 0
	for s_n in sorted_units_node:
		try:
			spike_times.append(list(s_n.times[0]))
			max_times.append(max(s_n.times[0]))
		except:
			spike_times.append(list(s_n.times))
			max_times.append(max(s_n.times))
		i+= 1
	max_time = np.ceil(max(max_times)*ms_conversion)
	#Converting spike times to ms timescale
	spike_times = [np.ceil(np.array(spike_times[i])*ms_conversion) for i in range(len(spike_times))]
	
	#Grab digital inputs
	print("\tGrabbing digital input times")
	start_dig_in_times_csv = fig_save_dir + 'start_dig_in_times.csv'
	end_dig_in_times_csv = fig_save_dir + 'end_dig_in_times.csv'
	if os.path.isfile(start_dig_in_times_csv):
		print("\t\tImporting previously saved digital input times")
		start_dig_in_times = []
		with open(start_dig_in_times_csv, 'r') as file:
		    csvreader = csv.reader(file)
		    for row in csvreader:
		        start_dig_in_times.append(list(np.array(row).astype('int')))
		end_dig_in_times = []
		with open(end_dig_in_times_csv, 'r') as file:
		    csvreader = csv.reader(file)
		    for row in csvreader:
		        end_dig_in_times.append(list(np.array(row).astype('int')))
		num_tastes = len(start_dig_in_times)
	else:
		dig_in_node = blech_clust_h5.list_nodes('/digital_in')
		dig_in_indices = np.array([d_i.name.split('_')[-1] for d_i in dig_in_node])
		dig_in_ind = []
		i = 0
		for d_i in dig_in_indices:
			try:
				int(d_i)
				dig_in_ind.extend([i])
			except:
				"not an input - do nothing"
			i += 1
		del dig_in_indices
		try:
			if len(dig_in_node[0][0]):
				dig_in_data = [list(dig_in_node[d_i][0][:]) for d_i in dig_in_ind]
		except:
			dig_in_data = [list(dig_in_node[d_i][:]) for d_i in dig_in_ind]
		num_tastes = len(dig_in_data)
		del dig_in_node
		#_____Convert dig_in_data to indices of dig_in start and end times_____
		print("\tConverting digital inputs to free memory")
		#Again, all are converted to ms timescale
		start_dig_in_times = [list(np.ceil((np.where(np.diff(np.array(dig_in_data[i])) == 1)[0] + 1)*ms_conversion).astype('int')) for i in range(num_tastes)]	
		end_dig_in_times = [list(np.ceil((np.where(np.diff(np.array(dig_in_data[i])) == -1)[0] + 1)*ms_conversion).astype('int')) for i in range(num_tastes)]
		#Store these into csv for import in future instead of full dig_in_data load which takes forever!
		with open(start_dig_in_times_csv, 'w') as f:
			write = csv.writer(f,delimiter=',')
			write.writerows(start_dig_in_times)
		with open(end_dig_in_times_csv, 'w') as f:
			write = csv.writer(f,delimiter=',')
			write.writerows(end_dig_in_times)
		#del dig_in_data, d_i, i, dig_in_ind
	
	#Grab dig in names
	print("\tGrabbing digital input names")
	try:
		dig_in_names = [blech_clust_h5.root.digital_in.dig_in_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.digital_in.dig_in_names))]
	except:
		#The old method doesn't currently store tastant names, so this probes the user
		dig_in_names = list()
		for i in range(num_tastes):
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
		segment_names = [blech_clust_h5.root.experiment_components.segment_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.experiment_components.segment_names))]
	except:
		min_dig_in_time = min([min(start_dig_in_times[i]) for i in range(num_tastes)])
		max_dig_in_time = max([max(end_dig_in_times[i]) for i in range(num_tastes)])
		dig_in_ind_range=np.array([min_dig_in_time/ms_conversion,max_dig_in_time/ms_conversion]) #unconvert because dp looks at sampling rate values
		segment_names, segment_times = dp.get_experiment_components(sampling_rate, dig_in_ind_range=dig_in_ind_range, len_rec=max_time/ms_conversion)  #unconvert because dp looks at sampling rate values
		#Convert segment times to ms timescale for saving
		segment_times = np.ceil(segment_times*ms_conversion)
	
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
		blech_clust_h5.remove_node('/experiment_components','segment_times')
		blech_clust_h5.create_earray('/experiment_components','segment_times',atom,(0,))
		exec("blech_clust_h5.root.experiment_components.segment_times.append(segment_times[:])")
	atom = tables.Atom.from_dtype(np.dtype('U20'))
	try:
		blech_clust_h5.create_earray('/experiment_components','segment_names',atom,(0,))
		exec("blech_clust_h5.root.experiment_components.segment_names.append(np.array(segment_names))")
	except:
		blech_clust_h5.remove_node('/experiment_components','segment_names')
		blech_clust_h5.create_earray('/experiment_components','segment_names',atom,(0,))
		exec("blech_clust_h5.root.experiment_components.segment_names.append(np.array(segment_names))")
		
	blech_clust_h5.close() #Always close the file
	
	toc = time.time()
	print("Time to import data = " + str(round((toc - tic)/60)) + " minutes \n")	
	return num_neur, all_waveforms, spike_times, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes


def calc_segment_spike_times(segment_times,spike_times,num_neur):
	segment_spike_times = []
	for s_i in tqdm.tqdm(range(len(segment_times)-1)):
		min_time = segment_times[s_i] #in ms
		max_time = segment_times[s_i+1] #in ms
		s_t = [list(np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= min_time)*(np.array(spike_times[i]) <= max_time))[0]]) for i in range(num_neur)]
		segment_spike_times.append(s_t)

	return segment_spike_times

def calc_tastant_spike_times(segment_times,spike_times,start_dig_in_times,end_dig_in_times,pre_taste,post_taste,num_tastes,num_neur):
	tastant_spike_times = []
	pre_taste_dt = int(np.ceil(pre_taste*(1000/1))) #Convert to ms timescale
	post_taste_dt = int(np.ceil(post_taste*(1000/1))) #Convert to ms timescale
	for t_i in tqdm.tqdm(range(num_tastes)):
		t_start = start_dig_in_times[t_i]
		t_end = end_dig_in_times[t_i]
		t_st = []
		for t_d_i in range(len(t_start)):
			start_i = int(max(t_start[t_d_i] - pre_taste_dt,0))
			end_i = int(min(t_end[t_d_i] + post_taste_dt,segment_times[-1]*1000))
			#Grab spike times into one list
			s_t = [list(np.array(spike_times[i])[np.where((np.array(spike_times[i]) >= start_i)*(np.array(spike_times[i]) <= end_i))[0]]) for i in range(num_neur)]
			t_st.append(s_t)
		tastant_spike_times.append(t_st)
		
	return tastant_spike_times

def add_data_to_hdf5(sorted_dir,data_group_name,data_name,data_array):
	"Note, this assumes the data is a float"
	blech_clust_h5 = tables.open_file(sorted_dir, 'r+', title = sorted_dir[-1])
	try:
		blech_clust_h5.create_group('/',data_group_name)
	except:
		print("\n\t" + data_group_name + " group already exists in .h5 file")
	atom = tables.FloatAtom()
	if type(data_array) == list:
		num_vals = len(data_array)
		for l_i in range(num_vals):
			try:
				blech_clust_h5.create_earray('/'+data_group_name,data_name+'_'+str(l_i),atom,(0,)+np.shape(np.array(data_array[l_i][:])))
				exec("blech_clust_h5.root."+data_group_name+"."+data_name+"_"+str(l_i)+".append(np.expand_dims(np.array(data_array[l_i][:]),0))")
			except:
				blech_clust_h5.remove_node('/'+data_group_name,data_name+"_"+str(l_i))
				blech_clust_h5.create_earray('/'+data_group_name,data_name+'_'+str(l_i),atom,(0,)+np.shape(np.array(data_array[l_i][:])))
				exec("blech_clust_h5.root."+data_group_name+"."+data_name+"_"+str(l_i)+".append(np.expand_dims(np.array(data_array[l_i][:]),0))")
	elif type(data_array) == np.ndarray:
		try:
			blech_clust_h5.create_earray('/'+data_group_name,data_name,atom,(0,)+np.shape(data_array[:]))
			exec("blech_clust_h5.root."+data_group_name+"."+data_name+".append(np.expand_dims(data_array[:],0))")
		except:
			blech_clust_h5.remove_node('/'+data_group_name,data_name)
			blech_clust_h5.create_earray('/'+data_group_name,data_name,atom,(0,)+np.shape(data_array[:]))
			exec("blech_clust_h5.root."+data_group_name+"."+data_name+".append(np.expand_dims(data_array[:],0))")
	
	blech_clust_h5.close() #Always close the file
	
	#todo: add handling of lists of lists with different sizes nested
	
def pull_data_from_hdf5(sorted_dir,data_group_name,data_name):
	"Note, this assumes the data is a float"
	blech_clust_h5 = tables.open_file(sorted_dir, 'r+', title = sorted_dir[-1])
	data_names = blech_clust_h5.list_nodes("/"+data_group_name)
	data_list = []
	for datum in data_names:
		if datum.name[0:len(data_name)] == data_name:
			data_list.append(datum[0][:])
	blech_clust_h5.close()
	if len(data_list) == 1:
		data = np.array(data_list)
	else:
		data = data_list
	
	return data
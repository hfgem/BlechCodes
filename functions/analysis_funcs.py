#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:46:03 2023

@author: Hannah Germaine
Set of miscellaneous functions to support analyses in analyze_states.py.
"""

import time,tables,tqdm,os,csv,random
import numpy as np
import functions.load_intan_rhd_format.load_intan_rhd_format as rhd
import functions.data_processing as dp
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm

# def import_data(sorted_dir, segment_dir, data_save_dir):
# 	"""Import data from .h5 file and grab any missing data through user inputs.
# 	Note, all spike times, digital input times, etc... are converted to the ms
# 	timescale and returned as such.
# 	"""
# 	print("Beginning Data Import")
# 	tic = time.time()
# 	#_____Import spike times and waveforms_____
# 	#Grab data from hdf5 file
# 	blech_clust_h5 = tables.open_file(sorted_dir, 'r+', title = sorted_dir[-1])

# 	#Grab sampling rate
# 	print("\tGrabbing sampling rate")
# 	try:
# 		sampling_rate = blech_clust_h5.root.sampling_rate[0]
# 	except:
# 		#The old method doesn't currently store sampling_rate, so this picks it up
# 		rhd_dict = rhd.import_data(data_save_dir)
# 		sampling_rate = int(rhd_dict["frequency_parameters"]["amplifier_sample_rate"])
# 		atom = tables.IntAtom()
# 		blech_clust_h5.create_earray('/','sampling_rate',atom,(0,))
# 		blech_clust_h5.root.sampling_rate.append([sampling_rate])
# 	#Calculate the conversion from samples to ms
# 	ms_conversion = (1/sampling_rate)*(1000/1) #ms/samples units
# 	
# 	sorted_units_node = blech_clust_h5.get_node('/sorted_units')
# 	num_neur = len([s_n for s_n in sorted_units_node])
# 	#Grab waveforms
# 	print("\tGrabbing waveforms")
# 	all_waveforms = [sn.waveforms[0] for sn in sorted_units_node]
# 	#Grab times
# 	print("\tGrabbing spike times")
# 	spike_times = []
# 	max_times = []
# 	i = 0
# 	for s_n in sorted_units_node:
# 		try:
# 			spike_times.append(list(s_n.times[0]))
# 			max_times.append(max(s_n.times[0]))
# 		except:
# 			spike_times.append(list(s_n.times))
# 			max_times.append(max(s_n.times))
# 		i+= 1
# 	max_time = np.ceil(max(max_times)*ms_conversion)
# 	#Converting spike times to ms timescale
# 	spike_times = [np.ceil(np.array(spike_times[i])*ms_conversion) for i in range(len(spike_times))]
# 	
# 	#Grab digital inputs
# 	print("\tGrabbing digital input times")
# 	start_dig_in_times_csv = data_save_dir + 'start_dig_in_times.csv'
# 	end_dig_in_times_csv = data_save_dir + 'end_dig_in_times.csv'
# 	if os.path.isfile(start_dig_in_times_csv):
# 		print("\t\tImporting previously saved digital input times")
# 		start_dig_in_times = []
# 		with open(start_dig_in_times_csv, 'r') as file:
# 		    csvreader = csv.reader(file)
# 		    for row in csvreader:
# 		        start_dig_in_times.append(list(np.array(row).astype('int')))
# 		end_dig_in_times = []
# 		with open(end_dig_in_times_csv, 'r') as file:
# 		    csvreader = csv.reader(file)
# 		    for row in csvreader:
# 		        end_dig_in_times.append(list(np.array(row).astype('int')))
# 		num_tastes = len(start_dig_in_times)
# 	else:
# 		dig_in_node = blech_clust_h5.list_nodes('/digital_in')
# 		dig_in_indices = np.array([d_i.name.split('_')[-1] for d_i in dig_in_node])
# 		dig_in_ind = []
# 		i = 0
# 		for d_i in dig_in_indices:
# 			try:
# 				int(d_i)
# 				dig_in_ind.extend([i])
# 			except:
# 				"not an input - do nothing"
# 			i += 1
# 		del dig_in_indices
# 		try:
# 			if len(dig_in_node[0][0]):
# 				dig_in_data = [list(dig_in_node[d_i][0][:]) for d_i in dig_in_ind]
# 		except:
# 			dig_in_data = [list(dig_in_node[d_i][:]) for d_i in dig_in_ind]
# 		num_tastes = len(dig_in_data)
# 		del dig_in_node
# 		#_____Convert dig_in_data to indices of dig_in start and end times_____
# 		print("\tConverting digital inputs to free memory")
# 		#Again, all are converted to ms timescale
# 		start_dig_in_times = [list(np.ceil((np.where(np.diff(np.array(dig_in_data[i])) == 1)[0] + 1)*ms_conversion).astype('int')) for i in range(num_tastes)]	
# 		end_dig_in_times = [list(np.ceil((np.where(np.diff(np.array(dig_in_data[i])) == -1)[0] + 1)*ms_conversion).astype('int')) for i in range(num_tastes)]
# 		#Store these into csv for import in future instead of full dig_in_data load which takes forever!
# 		with open(start_dig_in_times_csv, 'w') as f:
# 			write = csv.writer(f,delimiter=',')
# 			write.writerows(start_dig_in_times)
# 		with open(end_dig_in_times_csv, 'w') as f:
# 			write = csv.writer(f,delimiter=',')
# 			write.writerows(end_dig_in_times)
# 		#del dig_in_data, d_i, i, dig_in_ind
# 	
# 	#Grab dig in names
# 	print("\tGrabbing digital input names")
# 	try:
# 		dig_in_names = [blech_clust_h5.root.digital_in.dig_in_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.digital_in.dig_in_names))]
# 	except:
# 		#The old method doesn't currently store tastant names, so this probes the user
# 		dig_in_names = list()
# 		for i in range(num_tastes):
# 			d_loop = 1
# 			while d_loop == 1:
# 				d_name = input("\n INPUT REQUESTED: Enter single-word name for dig-in " + str(i) + ": ")
# 				if len(d_name) < 2:
# 					print("Error, entered name seems too short. Please try again.")
# 				else:
# 					d_loop = 0
# 			dig_in_names.append(d_name)
# 		#Save data for future use
# 		atom = tables.Atom.from_dtype(np.dtype('U20')) #tables.StringAtom(itemsize=50)
# 		dig_names = blech_clust_h5.create_earray('/digital_in','dig_in_names',atom,(0,))
# 		dig_names.append(np.array(dig_in_names))

# 	#_____Import segment times - otherwise ask for user segment time input_____
# 	print("\tGrabbing segment times and names")
# 	try:
# 		segment_times = blech_clust_h5.root.experiment_components.segment_times[:]
# 		segment_names = [blech_clust_h5.root.experiment_components.segment_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.experiment_components.segment_names))]
# 	except:
# 		min_dig_in_time = min([min(start_dig_in_times[i]) for i in range(num_tastes)])
# 		max_dig_in_time = max([max(end_dig_in_times[i]) for i in range(num_tastes)])
# 		dig_in_ind_range=np.array([min_dig_in_time/ms_conversion,max_dig_in_time/ms_conversion]) #unconvert because dp looks at sampling rate values
# 		segment_names, segment_times = dp.get_experiment_components(sampling_rate, dig_in_ind_range=dig_in_ind_range, len_rec=max_time/ms_conversion)  #unconvert because dp looks at sampling rate values
# 		#Convert segment times to ms timescale for saving
# 		segment_times = np.ceil(segment_times*ms_conversion)
# 	
# 	try:
# 		blech_clust_h5.create_group('/','experiment_components')
# 	except:
# 		print("\n\tExperiment components group already exists in .h5 file")
# 	atom = tables.IntAtom()
# 	try:
# 		blech_clust_h5.create_earray('/experiment_components','segment_times',atom,(0,))
# 		exec("blech_clust_h5.root.experiment_components.segment_times.append(segment_times[:])")
# 	except:
# 		blech_clust_h5.remove_node('/experiment_components','segment_times')
# 		blech_clust_h5.create_earray('/experiment_components','segment_times',atom,(0,))
# 		exec("blech_clust_h5.root.experiment_components.segment_times.append(segment_times[:])")
# 	atom = tables.Atom.from_dtype(np.dtype('U20'))
# 	try:
# 		blech_clust_h5.create_earray('/experiment_components','segment_names',atom,(0,))
# 		exec("blech_clust_h5.root.experiment_components.segment_names.append(np.array(segment_names))")
# 	except:
# 		blech_clust_h5.remove_node('/experiment_components','segment_names')
# 		blech_clust_h5.create_earray('/experiment_components','segment_names',atom,(0,))
# 		exec("blech_clust_h5.root.experiment_components.segment_names.append(np.array(segment_names))")
# 		
# 	blech_clust_h5.close() #Always close the file
# 	
# 	toc = time.time()
# 	print("Time to import data = " + str(round((toc - tic)/60)) + " minutes \n")	
# 	return num_neur, all_waveforms, spike_times, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes

def add_no_taste(start_dig_in_times, end_dig_in_times, dig_in_names):
	
	#Add to the tastants / dig_ins a "no taste delivered" control
	all_start_times = []
	all_end_times = []
	num_deliv = []
	for t_i in range(len(start_dig_in_times)): #for each taste
		all_start_times.extend(start_dig_in_times[t_i])
		num_deliv.extend([len(start_dig_in_times[t_i])])
		all_end_times.extend(end_dig_in_times[t_i])
		
	all_start_times = np.array(all_start_times)
	all_end_times = np.array(all_end_times)
	num_none = np.ceil(np.mean(np.array(num_deliv))).astype('int')
	time_before_vec = np.random.random_integers(10000,20000,size=(num_none)) #Number of seconds before taste delivery to grab
	dig_in_len_vec = all_end_times - all_start_times
	none_len_vec = np.array(random.sample(list(dig_in_len_vec),num_none))
	none_start_times = np.array(random.sample(list(all_start_times),num_none)) - time_before_vec
	none_end_times = none_start_times + none_len_vec
	
	dig_in_names.extend(['none'])
	start_dig_in_times.append(list(none_start_times))
	end_dig_in_times.append(list(none_end_times))
	
	num_tastes = len(start_dig_in_times)
	
	
	return dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes

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
#TODO: add handling of lists of lists with different sizes nested
	
def pull_data_from_hdf5(sorted_dir,data_group_name,data_name):
	"Note, this assumes the data is a float"
	blech_clust_h5 = tables.open_file(sorted_dir, 'r+', title = sorted_dir[-1])
	data_names = blech_clust_h5.list_nodes("/"+data_group_name)
	data_list = []
	for datum in data_names:
		if datum.name[0:len(data_name)] == data_name:
			data_list.append(datum[0][:])
	blech_clust_h5.close()
	if len(data_list) >= 1:
		if len(data_list) == 1:
			data = np.array(data_list)
		else:
			data = data_list
	else:
		raise Exception(data_name + " does not exist in group " + data_group_name)
	
	return data

def taste_responsivity_PSTH(PSTH_times,PSTH_taste_deliv_times,tastant_PSTH):
	"""A test of whether or not each neuron is taste responsive by looking at
	PSTH activity before and after taste delivery and calculating if there is a 
	significant change for each delivery - probability of taste responsivity 
	is then the fraction of deliveries where there was a significant response"""
	
	num_tastes = len(PSTH_taste_deliv_times)
	taste_responsivity_probability = []
	for t_i in range(num_tastes):
		[num_deliv, num_neur, len_PSTH] = np.shape(tastant_PSTH[t_i])
		taste_responsive_neur = np.zeros((num_neur,num_deliv))
		for n_i in range(num_neur):
			start_taste_i = np.where(PSTH_times[t_i] == PSTH_taste_deliv_times[t_i][0])[0][0]
			end_taste_i = np.where(PSTH_times[t_i] == PSTH_taste_deliv_times[t_i][1])[0][0]
			for d_i in range(num_deliv):
				pre_taste_PSTH_vals = tastant_PSTH[t_i][d_i,n_i,:start_taste_i]
				post_taste_PSTH_vals = tastant_PSTH[t_i][d_i,n_i,end_taste_i:end_taste_i+start_taste_i]
				p_val, _ = stats.ks_2samp(pre_taste_PSTH_vals,post_taste_PSTH_vals)
				if p_val < 0.05:
					taste_responsive_neur[n_i,d_i] = 1
		taste_responsivity_probability.append(np.sum(taste_responsive_neur,1)/num_deliv)
	#for t_i in range(num_tastes):
	#    plt.plot(taste_responsivity_probability[t_i])
	
	return taste_responsivity_probability
	
def taste_responsivity_raster(tastant_spike_times,start_dig_in_times,end_dig_in_times,num_neur,pre_taste_dt):
	"""A test of whether or not each neuron is taste responsive by looking at
	spike activity before and after taste delivery and calculating if there is 
	a significant change for each delivery - probability of taste responsivity 
	is then the fraction of deliveries where there was a significant response"""
	bin_sum_dt = 100 #in ms = dt
	num_tastes = len(tastant_spike_times)
	colors = cm.cool(np.arange(num_tastes-1)/(num_tastes-1))
	taste_responsivity_probability = []
	taste_responsivity_binary = np.array([True for n_i in range(num_neur)])
	for t_i in range(num_tastes-1): #Last taste is always "none" and shouldn't be counted
		num_deliv = len(tastant_spike_times[t_i])
		taste_responsive_neur = np.zeros((num_neur,num_deliv))
		for n_i in tqdm.tqdm(range(num_neur)):
			for d_i in range(num_deliv):
				raster_times = tastant_spike_times[t_i][d_i][n_i]
				start_taste_i = start_dig_in_times[t_i][d_i]
				end_taste_i = end_dig_in_times[t_i][d_i]
				times_pre_taste = (np.array(raster_times)[np.where(raster_times < start_taste_i)[0]] - (start_taste_i - pre_taste_dt)).astype('int')
				bin_pre_taste = np.zeros(pre_taste_dt)
				bin_pre_taste[times_pre_taste] += 1
				times_post_taste = (np.array(raster_times)[np.where((raster_times > end_taste_i)*(raster_times < end_taste_i + pre_taste_dt))[0]] - end_taste_i).astype('int')
				bin_post_taste = np.zeros(pre_taste_dt)
				bin_post_taste[times_post_taste] += 1
				pre_taste_spike_nums = [sum(bin_pre_taste[b_i:b_i+bin_sum_dt]) for b_i in range(pre_taste_dt - bin_sum_dt)]
				post_taste_spike_nums = [sum(bin_post_taste[b_i:b_i+bin_sum_dt]) for b_i in range(pre_taste_dt - bin_sum_dt)]
				#Since these are stochastic samples, assuming a mean fr pre and post, we can use the Mann-Whitney-U Test
				try:
					_, p_val = stats.mannwhitneyu(pre_taste_spike_nums,post_taste_spike_nums)
					if p_val < 0.05:
						taste_responsive_neur[n_i,d_i] = 1
				except:
					pass
		taste_responsivity_probability.append(np.sum(taste_responsive_neur,1)/num_deliv)
		taste_responsivity_binary *= (np.sum(taste_responsive_neur,1)/num_deliv > 1/2)
	for t_i in range(num_tastes-1):
		plt.plot(taste_responsivity_probability[t_i],color=colors[t_i],label=str(t_i))
	plt.legend()
	
	return taste_responsivity_probability, taste_responsivity_binary

# def taste_cp_frs(taste_cp_raster_inds,tastant_spike_times,start_dig_in_times,end_dig_in_times,dig_in_names,num_neur,pre_taste_dt,post_taste_dt):
# 	"""binned firing rates within epochs between given changepoints"""
# 	
# 	num_tastes = len(tastant_spike_times)
# 	
# 	num_cp = np.shape(taste_cp_raster_inds[0])[-1]
# 	for t_i in range(num_tastes-1):
# 		if np.shape(taste_cp_raster_inds[t_i+1])[-1] < num_cp:
# 			num_cp = np.shape(taste_cp_raster_inds[t_i+1])[-1] - 1
# 	
# 	tastant_binned_frs = [] #taste x neuron x delivery
# 	for t_i in range(num_tastes):
# 		num_deliv = len(tastant_spike_times[t_i])
# 		t_cp_rast = taste_cp_raster_inds[t_i] #num_deliv x num_neur x num_cp+1
# 		num_deliv, _, _ = np.shape(t_cp_rast)
# 		t_cp_rast_edit = np.concatenate((t_cp_rast,(post_taste_dt+pre_taste_dt)*np.ones((num_deliv,num_neur,1))),axis=-1)
# 		deliv_binned_spikes = np.zeros((num_neur,num_deliv,num_cp))
# 		for n_i in range(num_neur):
# 			for d_i in range(num_deliv):
# 				raster_times = tastant_spike_times[t_i][d_i][n_i]
# 				start_taste_i = start_dig_in_times[t_i][d_i]
# 				#times_pre_taste = (np.array(raster_times)[np.where(raster_times < start_taste_i)[0]] - (start_taste_i - pre_taste_dt)).astype('int')
# 				#fr_avg_pre = (len(times_pre_taste)/pre_taste_dt)*(1000/1) #in hz
# 				times_post_taste = (np.array(raster_times)[np.where((raster_times > start_taste_i)*(raster_times < start_taste_i + post_taste_dt))[0]] - start_taste_i).astype('int')
# 				bin_post_taste = np.zeros(post_taste_dt)
# 				bin_post_taste[times_post_taste] += 1
# 				#Epoch-binned average firing rate
# 				cp_bin_inds = t_cp_rast_edit[d_i,n_i,:].astype('int')
# 				post_taste_bins = np.zeros(num_cp)
# 				for cp_i in range(num_cp):
# 					if cp_bin_inds[cp_i+1]-cp_bin_inds[cp_i] != 0:
# 						post_taste_bins[cp_i] = np.sum(bin_post_taste[cp_bin_inds[cp_i]:cp_bin_inds[cp_i+1]])/((cp_bin_inds[cp_i+1]-cp_bin_inds[cp_i])/1000)
# 					else:
# 						post_taste_bins[cp_i] = 0
# 				deliv_binned_spikes[n_i,d_i,:] = post_taste_bins
# 		tastant_binned_frs.append(deliv_binned_spikes)
# 	
# 	return tastant_binned_frs
	
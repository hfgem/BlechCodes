#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:58:10 2023

@author: Hannah Germaine
Calculate the correlation results for the full population of neurons.
Must be run following find_deviations.py
"""
		
if __name__ == '__main__':

	import os,json,gzip,itertools,tqdm
	import numpy as np
	import functions.analysis_funcs as af
	import functions.dev_funcs as df
	import functions.hdf5_handling as hf5
	from multiprocessing import Pool
	import functions.corr_dist_calc_parallel as cdcp
	import functions.corr_dist_calc_parallel_pop as cdcpp
	
	#_____Get the directory of the hdf5 file_____
	sorted_dir, segment_dir, cleaned_dir = hf5.sorted_data_import() #Program will automatically quit if file not found in given folder
	fig_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'
	print('\nData Directory:')
	print(fig_save_dir)

	#_____Import data_____
	#todo: update intan rhd file import code to accept directory input
	num_neur, all_waveforms, spike_times, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes = af.import_data(sorted_dir, segment_dir, fig_save_dir)
	
	#_____Calculate spike time datasets_____
	pre_taste = 0.5 #Seconds before tastant delivery to store
	post_taste = 2 #Seconds after tastant delivery to store
	
	#_____Add "no taste" control segments to the dataset_____
	if dig_in_names[-1] != 'none':
		dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes = af.add_no_taste(start_dig_in_times, end_dig_in_times, post_taste, dig_in_names)

	segment_spike_times = af.calc_segment_spike_times(segment_times,spike_times,num_neur)
	tastant_spike_times = af.calc_tastant_spike_times(segment_times,spike_times,
													  start_dig_in_times,end_dig_in_times,
													  pre_taste,post_taste,num_tastes,num_neur)
	num_segments = len(segment_spike_times)
	segment_times_reshaped = [[segment_times[i],segment_times[i+1]] for i in range(num_segments)]
	
	#Deviation storage directory
	dev_dir = fig_save_dir + 'Deviations/'
	
	print("Now importing calculated deviations")
	segment_deviations = []
	for s_i in tqdm.tqdm(range(num_segments)):
		filepath = dev_dir + segment_names[s_i] + '/deviations.json'
		with gzip.GzipFile(filepath, mode="r") as f:
			json_bytes = f.read()
			json_str = json_bytes.decode('utf-8')            
			data = json.loads(json_str) 
			segment_deviations.append(data)
	
	#_____Pull rasters of deviations and plot_____
	#Calculate segment deviation spikes
	print("Now pulling true deviation rasters")
	segment_dev_rasters, segment_dev_times = df.create_dev_rasters(num_segments, segment_spike_times, 
						   np.array(segment_times_reshaped), segment_deviations)
	
	#Import changepoint data
	data_group_name = 'changepoint_data'
	taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_cp_raster_inds')
	pop_taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'pop_taste_cp_raster_inds')
	
	#Generate comparable binary matrix for regular data
	num_cp = np.shape(taste_cp_raster_inds[0])[-1] - 1
	for t_i in range(num_tastes-1):
		if np.shape(taste_cp_raster_inds[t_i+1])[-1] - 1 < num_cp:
			num_cp = np.shape(taste_cp_raster_inds[t_i+1])[-1] - 1
	neuron_keep_indices = np.ones((num_neur,num_cp))

	#_____Calculate correlation between taste and deviation rasters for individual neurons_____
	#Create directory to store analysis results
	comp_dir = fig_save_dir + 'dev_x_taste/'
	if os.path.isdir(comp_dir) == False:
		os.mkdir(comp_dir)
		
	#Create folder to store correlation results
	corr_dir = comp_dir + 'corr/'
	if os.path.isdir(corr_dir) == False:
		os.mkdir(corr_dir)
	
	#__________For all neurons__________
	all_neur_corr_dir = corr_dir + 'all_neur/'
	if os.path.isdir(all_neur_corr_dir) == False:
		os.mkdir(all_neur_corr_dir)
		
	#Grab parameters
	fr_bin = 25 #ms to bin together for number of spikes 'fr'
	num_tastes = len(start_dig_in_times)
	num_segments = len(segment_dev_rasters)
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
	post_taste_dt = np.ceil(post_taste*1000).astype('int')
	
	#Run calculations using continuous firing rates
	for s_i in range(num_segments):  #Loop through each segment
		print("Beginning timeseries correlation calcs for segment " + str(s_i))
		#Gather segment data
		seg_rast = segment_dev_rasters[s_i]
		num_dev = len(seg_rast)
			
		for t_i in range(num_tastes):  #Loop through each taste
			#Find the number of neurons
			if np.shape(neuron_keep_indices)[0] == 0:
				total_num_neur = np.shape(seg_rast[0])[0]
				taste_keep_ind = np.arange(total_num_neur)
			else:
				total_num_neur = np.sum(neuron_keep_indices[:,t_i]).astype('int')
				taste_keep_ind = ((neuron_keep_indices[:,t_i]).astype('int')).flatten()
			#Try to import previously stored data if exists
			filename = all_neur_corr_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '.npy'
			filename_loaded = 0
			filename_pop = all_neur_corr_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop.npy'
			filename_pop_loaded = 0
			try:
				neuron_corr_storage = np.load(filename)
				filename_loaded = 1
			except:
				print("Individual Neuron Timeseries Correlations Need to Be Calculated")
			try:
				neuron_vec_corr_storage = np.load(filename_pop)
				filename_pop_loaded = 1
			except:
				print("Population Timeseries Correlations Need to Be Calculated")
			if filename_loaded*filename_pop_loaded == 0:
				print("\tCalculating Taste #" + str(t_i + 1))
				taste_cp = taste_cp_raster_inds[t_i][:, taste_keep_ind, :]
				taste_cp_pop = pop_taste_cp_raster_inds[t_i]
				taste_spikes = tastant_spike_times[t_i]
				#Note, num_cp = num_cp+1 with the first value the taste delivery index
				num_deliv, _, num_cp = np.shape(taste_cp)
				taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i][deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
				deliv_adjustment = [start_dig_in_times[t_i][deliv_i] + pre_taste_dt for deliv_i in range(num_deliv)]
				num_deliv, _, num_cp = np.shape(taste_cp)
				#Store the correlation results in a numpy array
				neuron_corr_storage = np.nan*np.ones((num_dev, num_deliv, total_num_neur, num_cp-1))
				neuron_pop_corr_storage = np.nan*np.ones((num_dev, num_deliv, num_cp-1))
				for dev_i in tqdm.tqdm(range(num_dev)): #Loop through all deviations
					dev_rast = seg_rast[dev_i][taste_keep_ind,:]
					dev_len = np.shape(dev_rast)[1]
					end_ind = np.arange(fr_bin,fr_bin+dev_len)
					end_ind[end_ind > dev_len] = dev_len
					#TODO: test gaussian convolution instead of binning
					dev_rast_binned = np.zeros(np.shape(dev_rast)) #timeseries information kept
					for start_ind in range(dev_len):
						dev_rast_binned[:,start_ind] = np.sum(dev_rast[:,start_ind:end_ind[start_ind]],1)
					#Individual neuron changepoints
					if filename_loaded == 0:
						inputs = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
						 itertools.repeat(np.arange(0,len(taste_keep_ind))), itertools.repeat(taste_cp), \
							 deliv_adjustment, itertools.repeat(dev_rast_binned), itertools.repeat(fr_bin))
						pool = Pool(4)
						deliv_corr_storage = pool.map(cdcp.deliv_corr_parallelized, inputs)
						pool.close()
						neuron_corr_storage[dev_i,:,:,:] = np.array(deliv_corr_storage)
					#Population changepoints
					if filename_pop_loaded == 0:
						pool = Pool(4)
						inputs_pop = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
							itertools.repeat(np.arange(0,len(taste_keep_ind))), itertools.repeat(taste_cp_pop), \
							deliv_adjustment, itertools.repeat(dev_rast_binned), itertools.repeat(fr_bin))
						deliv_vec_corr_storage = pool.map(cdcpp.deliv_corr_population_parallelized, inputs_pop)
						pool.close()
						neuron_pop_corr_storage[dev_i,:,:] = np.array(deliv_vec_corr_storage)
				#Save to a numpy array
				if filename_loaded == 0:
					np.save(filename,neuron_corr_storage)
				if filename_pop_loaded == 0:
					np.save(filename_pop,neuron_pop_corr_storage)
	
	#Run calculations using population vector firing rates
	num_tastes = len(start_dig_in_times)
	num_segments = len(segment_dev_rasters)
	pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
	post_taste_dt = np.ceil(post_taste*1000).astype('int')

	for s_i in range(num_segments):  #Loop through each segment
		print("Beginning population vector correlation calcs for segment " + str(s_i))
		#Gather segment data
		seg_rast = segment_dev_rasters[s_i]
		num_dev = len(seg_rast)
			
		for t_i in range(num_tastes):  #Loop through each taste
			#Find the number of neurons
			if np.shape(neuron_keep_indices)[0] == 0:
				total_num_neur = np.shape(seg_rast[0])[0]
				taste_keep_ind = np.arange(total_num_neur)
			else:
				total_num_neur = np.sum(neuron_keep_indices[:,t_i]).astype('int')
				taste_keep_ind = ((neuron_keep_indices[:,t_i]).astype('int')).flatten()
		
			#Set storage directory and check if data previously stored
			filename_pop_vec = all_neur_corr_dir + segment_names[s_i] + '_' + dig_in_names[t_i] + '_pop_vec.npy'
			filename_pop_vec_loaded = 0
			try:
				neuron_pop_vec_corr_storage = np.load(filename_pop_vec_loaded)
				filename_pop_vec_loaded = 1
			except:
				pass
			if filename_pop_vec_loaded == 0:
				print("\tCalculating Taste #" + str(t_i + 1))
				taste_cp_pop = pop_taste_cp_raster_inds[t_i]
				taste_spikes = tastant_spike_times[t_i]
				#Note, num_cp = num_cp+1 with the first value the taste delivery index
				num_deliv, num_cp = np.shape(taste_cp_pop)
				taste_deliv_len = [(end_dig_in_times[t_i][deliv_i] - start_dig_in_times[t_i][deliv_i] + pre_taste_dt + post_taste_dt + 1) for deliv_i in range(num_deliv)]
				deliv_adjustment = [start_dig_in_times[t_i][deliv_i] + pre_taste_dt for deliv_i in range(num_deliv)]
				#Store the correlation results in a numpy array
				neuron_pop_vec_corr_storage = np.nan*np.ones((num_dev, num_deliv, num_cp-1))
				for dev_i in tqdm.tqdm(range(num_dev)): #Loop through all deviations
					dev_rast = seg_rast[dev_i][taste_keep_ind,:]
					dev_len = np.shape(dev_rast)[1]
					dev_vec = np.sum(dev_rast,1)/(dev_len/1000) #in Hz
					#Population fr vector changepoints
					pool = Pool(4)
					inputs_pop = zip(range(num_deliv), taste_spikes, taste_deliv_len, \
						itertools.repeat(taste_keep_ind), itertools.repeat(taste_cp_pop), \
						deliv_adjustment, itertools.repeat(dev_vec))
					deliv_vec_corr_storage = pool.map(cdcpp.deliv_corr_population_vec_parallelized, inputs_pop)
					pool.close()
					neuron_pop_vec_corr_storage[dev_i,:,:] = np.array(deliv_vec_corr_storage)
				np.save(filename_pop_vec,neuron_pop_vec_corr_storage)
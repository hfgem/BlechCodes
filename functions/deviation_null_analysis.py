#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:28:37 2024

@author: Hannah Germaine

This is the fourth step of the analysis pipeline: deviation events are compared to null shuffled distributions
"""

import os, sys, tqdm, gzip, itertools
from multiprocessing import Pool
import numpy as np
import scipy.stats as stats
current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)
import functions.dev_funcs as df
import functions.null_distributions as nd

class run_deviation_null_analysis():
	
	def __init__(self,args):
		self.metadata = args[0]
		self.data_dict = args[1]
		self.gather_variables()
		self.import_deviations()
		self.gen_null_distributions()
		self.convert_to_rasters()
		
	def gather_variables(self,):
		self.pre_taste = self.metadata['params_dict']['pre_taste']
		self.post_taste = self.metadata['params_dict']['post_taste']
		self.segments_to_analyze = self.metadata['params_dict']['segments_to_analyze']
		self.segment_spike_times = self.data_dict['segment_spike_times']
		self.segment_names = self.data_dict['segment_names']
		self.num_segments = len(self.segment_names)
		self.segment_times = self.data_dict['segment_times']
		self.segment_times_reshaped = [[self.segment_times[i],self.segment_times[i+1]] for i in range(self.num_segments)]
		self.local_size = self.metadata['params_dict']['local_size']
		self.min_dev_size = self.metadata['params_dict']['min_dev_size']
		self.num_null = self.metadata['params_dict']['num_null']
		self.max_plot = self.metadata['params_dict']['max_plot']
		self.count_cutoff = np.arange(1,self.num_neur)
		self.bin_size = self.metadata['params_dict']['compare_null_params']['bin_size']
		lag_min = self.metadata['params_dict']['compare_null_params']['lag_min']
		lag_max = self.metadata['params_dict']['compare_null_params']['lag_max']
		self.lag_vals = np.arange(lag_min,lag_max).astype('int')
		self.dev_dir = self.metadata['dir_name'] + 'Deviations/'
		self.null_dir = self.metadata['dir_name'] + 'null_data/'
		if os.path.isdir(self.null_dir) == False:
			os.mkdir(self.null_dir)
		self.bin_dir = self.metadata['dir_name'] + 'thresholded_deviations/'
		if os.path.isdir(self.bin_dir) == False:
			os.mkdir(self.bin_dir)	
	
	def import_deviations(self,):
		try: #test if the data exists by trying to import the last 
			filepath = self.dev_dir + self.segment_names[-1] + '/deviations.json'
			with gzip.GzipFile(filepath, mode="r") as f:
				json_bytes = f.read()
				json_str = json_bytes.decode('utf-8')
				data = json.loads(json_str)
				
			print("Now importing calculated deviations")
			segment_deviations = []
			for s_i in tqdm.tqdm(range(num_segments)):
				filepath = dev_dir + segment_names[s_i] + '/deviations.json'
				with gzip.GzipFile(filepath, mode="r") as f:
					json_bytes = f.read()
					json_str = json_bytes.decode('utf-8')			
					data = json.loads(json_str) 
					segment_deviations.append(data)
			self.segment_deviations = segment_deviations
		except:
			print("Deviations were not calculated previously as expected.")
			print("Something went wrong in the analysis pipeline.")
			print("Please try reverting your analysis_state_tracker.csv to 1 to rerun.")
			print("If issues persist, contact Hannah.")
			sys.exit()
	
	def gen_null_distributions(self,):
		#_____Import or calculate null deviations for all segments_____
		try: #test if the data exists by trying to import the last 
			filepath = self.dev_dir + 'null_data/' + self.segment_names[-1] + '/null_0_deviations.json'
			with gzip.GzipFile(filepath, mode="r") as f:
				json_bytes = f.read()
				json_str = json_bytes.decode('utf-8')
				data = json.loads(json_str)
		except:
			print("\nNow calculating deviations")
			for null_i in range(self.num_null):
				seg_dirs = []
				for s_i in range(self.num_segments):
					#create storage directory
					seg_dir = self.dev_dir + 'null_data/' + self.segment_names[s_i] + '/'
					if os.path.isdir(seg_dir) == False:
						os.mkdir(seg_dir)
					seg_dir = self.dev_dir + 'null_data/' + self.segment_names[s_i] + '/null_' + str(null_i) + '_'
					seg_dirs.append(seg_dir)
				with Pool(processes=4) as pool:  # start 4 worker processes
					pool.map(df.run_dev_pull_parallelized, zip(self.segment_spike_times,
													 itertools.repeat(self.local_size),
													 itertools.repeat(self.min_dev_size),
													 self.segment_times_reshaped,
													 self.seg_dirs))
				pool.close()
		
		print("Now importing calculated null deviations")
		all_null_deviations = []
		for null_i in tqdm.tqdm(range(self.num_null)):
			null_segment_deviations = []
			for s_i in range(self.num_segments):
				filepath = self.dev_dir + 'null_data/' + self.segment_names[s_i] + '/null_' + str(null_i) + '_deviations.json'
				with gzip.GzipFile(filepath, mode="r") as f:
					json_bytes = f.read()
					json_str = json_bytes.decode('utf-8')			
					data = json.loads(json_str) 
					null_segment_deviations.append(data)
			all_null_deviations.append(null_segment_deviations)
		del null_i, null_segment_deviations, s_i, filepath, json_bytes, json_str, data
		
		self.all_null_deviations = all_null_deviations
		
	def convert_to_rasters(self,):
		#Calculate segment deviation spikes
		print("Now pulling true deviation rasters")
		segment_dev_rasters, segment_dev_times, segment_dev_rasters_zscore = df.create_dev_rasters(self.num_segments, self.segment_spike_times, 
							   np.array(self.segment_times_reshaped), self.segment_deviations, self.pre_taste)
		self.segment_dev_rasters = segment_dev_rasters
		self.segment_dev_times = segment_dev_times
		self.segment_dev_rasters_zscore = segment_dev_rasters_zscore
		
		#Calculate segment deviation spikes
		print("Now pulling null deviation rasters")
		null_dev_rasters = []
		null_dev_times = []
		null_segment_dev_rasters_zscore = []
		for null_i in tqdm.tqdm(range(self.num_null)):
			null_segment_deviations = self.all_null_deviations[null_i]
			null_segment_dev_rasters_i, null_segment_dev_times_i, null_segment_dev_rasters_zscore_i = df.create_dev_rasters(self.num_segments, self.segment_spike_times, 
								   np.array(self.segment_times_reshaped), self.null_segment_deviations, self.pre_taste)
			null_dev_rasters.append(null_segment_dev_rasters_i)
			null_dev_times.append(null_segment_dev_times_i)
			null_segment_dev_rasters_zscore.append(null_segment_dev_rasters_zscore_i)
			
		self.__dict__.pop('all_null_deviations',None)
		self.null_dev_rasters = null_dev_rasters
		self.null_dev_times = null_dev_times
		self.null_segment_dev_rasters_zscore = null_segment_dev_rasters_zscore
		
	def calc_statistics(self,):
		try: #Import calculated dictionaries if they exist
			filepath = self.bin_dir + 'neur_count_dict.npy'
			neur_count_dict = np.load(filepath, allow_pickle=True).item()
			filepath = self.bin_dir + 'neur_spike_dict.npy'
			neur_spike_dict = np.load(filepath, allow_pickle=True).item()
			filepath = self.bin_dir + 'neur_len_dict.npy'
			neur_spike_dict = np.load(filepath, allow_pickle=True).item()
			print('\tThresholded datasets previously calculated.')
		except: #Calculate dictionaries
			neur_count_dict = dict()
			neur_spike_dict = dict()
			neur_len_dict = dict()
			for s_i in tqdm.tqdm(self.segments_to_analyze):
				#Gather data / parameters
				seg_name = self.segment_names[s_i]
				segment_spikes = self.segment_spike_times[s_i]
				segment_start_time = self.segment_times[s_i]
				segment_end_time = self.segment_times[s_i+1]
				segment_length = segment_end_time - segment_start_time
				#_____Gather null data deviation event stats_____
				null_dev_lengths = []
				null_dev_neuron_counts = []
				null_dev_spike_counts = []
				for null_i in range(self.num_null):
					all_rast = self.null_dev_rasters[null_i][s_i]
					null_i_num_neur, null_i_num_spikes, all_len = df.calculate_dev_null_stats(all_rast, self.null_dev_times[null_i][s_i])
					null_dev_neuron_counts.append(null_i_num_neur)
					null_dev_spike_counts.append(null_i_num_spikes)
					null_dev_lengths.append(all_len)
				#_____Gather true data deviation event stats_____
				true_dev_neuron_counts = []
				true_dev_spike_counts = []
				all_rast = self.segment_dev_rasters[s_i]
				true_dev_neuron_counts, true_dev_spike_counts, true_dev_lengths = df.calculate_dev_null_stats(all_rast, self.segment_dev_times[s_i])
				#_____Gather data as dictionary of number of events as a function of cutoff
				#Neuron count data
				null_max_neur_count = np.max([np.max(null_dev_neuron_counts[null_i]) for null_i in range(self.num_null)])
				max_neur_count = int(np.max([np.max(null_max_neur_count),np.max(true_dev_neuron_counts)]))
				neur_x_vals = np.arange(10,max_neur_count)
				true_neur_x_val_counts = np.zeros(np.shape(neur_x_vals))
				null_neur_x_val_counts_all = []
				null_neur_x_val_counts_mean = np.zeros(np.shape(neur_x_vals))
				null_neur_x_val_counts_std = np.zeros(np.shape(neur_x_vals))
				for n_cut_i, n_cut in enumerate(neur_x_vals):
					true_neur_x_val_counts[n_cut_i] = np.sum((np.array(true_dev_neuron_counts) > n_cut).astype('int'))
					null_neur_x_val_counts = []
					for null_i in range(self.num_null):
						null_neur_x_val_counts.append(np.sum((np.array(null_dev_neuron_counts[null_i]) > n_cut).astype('int')))
					null_neur_x_val_counts_all.append(null_neur_x_val_counts)
					null_neur_x_val_counts_mean[n_cut_i] = np.mean(null_neur_x_val_counts)
					null_neur_x_val_counts_std[n_cut_i] = np.std(null_neur_x_val_counts)
				neur_count_dict[seg_name + '_true'] =  [list(neur_x_vals),
											   list(true_neur_x_val_counts)]
				neur_count_dict[seg_name + '_null'] =  [list(neur_x_vals),
											   list(null_neur_x_val_counts_mean),
											   list(null_neur_x_val_counts_std)]
				percentiles = [] #Calculate percentile of true data point in null data distribution
				for n_cut in neur_x_vals:
					try:
						percentiles.extend([round(stats.percentileofscore(null_neur_x_val_counts_all[n_cut-1],true_neur_x_val_counts[n_cut-1]),2)])
					except:
						percentiles.extend([100])
				neur_count_dict[seg_name + '_percentile'] =  [list(neur_x_vals),percentiles]
				
				#Spike count data
				null_max_neur_spikes = np.max([np.max(null_dev_spike_counts[null_i]) for null_i in range(self.num_null)])
				max_spike_count = int(np.max([np.max(null_max_neur_spikes),np.max(true_dev_spike_counts)]))
				spike_x_vals = np.arange(1,max_spike_count)
				true_neur_x_val_spikes = np.zeros(np.shape(spike_x_vals))
				null_neur_x_val_spikes_all = []
				null_neur_x_val_spikes_mean = np.zeros(np.shape(spike_x_vals))
				null_neur_x_val_spikes_std = np.zeros(np.shape(spike_x_vals))
				for s_cut_i, s_cut in enumerate(spike_x_vals):
					true_neur_x_val_spikes[s_cut_i] = np.sum((np.array(true_dev_spike_counts) > s_cut).astype('int'))
					null_neur_x_val_spikes = []
					for null_i in range(self.num_null):
						null_neur_x_val_spikes.append(np.sum((np.array(null_dev_spike_counts[null_i]) > s_cut).astype('int')))
					null_neur_x_val_spikes_all.append(null_neur_x_val_spikes)
					null_neur_x_val_spikes_mean[s_cut_i] = np.mean(null_neur_x_val_spikes)
					null_neur_x_val_spikes_std[s_cut_i] = np.std(null_neur_x_val_spikes)
				neur_spike_dict[seg_name + '_true'] =  [list(spike_x_vals),
											   list(true_neur_x_val_spikes)]
				neur_spike_dict[seg_name + '_null'] =  [list(spike_x_vals),
											   list(null_neur_x_val_spikes_mean),
											   list(null_neur_x_val_spikes_std)]
				percentiles = [] #Calculate percentile of true data point in null data distribution
				for s_cut in spike_x_vals:
					try:
						percentiles.extend([round(stats.percentileofscore(null_neur_x_val_spikes_all[s_cut-1],true_neur_x_val_spikes[s_cut-1]),2)])
					except:
						percentiles.extend([100])
				neur_spike_dict[seg_name + '_percentile'] =  [list(spike_x_vals),percentiles]
				
				
				#Burst length data
				null_max_neur_len = np.max([np.max(null_dev_lengths[null_i]) for null_i in range(self.num_null)])
				max_len = int(np.max([np.max(null_max_neur_len),np.max(true_dev_lengths)]))
				len_x_vals = np.arange(1,max_len)
				true_neur_x_val_lengths = np.zeros(np.shape(len_x_vals))
				null_neur_x_val_lengths_all = []
				null_neur_x_val_lengths_mean = np.zeros(np.shape(len_x_vals))
				null_neur_x_val_lengths_std = np.zeros(np.shape(len_x_vals))
				for l_cut_i, l_cut in enumerate(len_x_vals):
					true_neur_x_val_lengths[l_cut_i] = np.sum((np.array(true_dev_lengths) > l_cut).astype('int'))
					null_neur_x_val_lengths = []
					for null_i in range(self.num_null):
						null_neur_x_val_lengths.append(np.sum((np.array(null_dev_lengths[null_i]) > l_cut).astype('int')))
					null_neur_x_val_lengths_all.append(null_neur_x_val_lengths)
					null_neur_x_val_lengths_mean[l_cut_i] = np.mean(null_neur_x_val_lengths)
					null_neur_x_val_lengths_std[l_cut_i] = np.std(null_neur_x_val_lengths)
				neur_len_dict[seg_name + '_true'] =  [list(len_x_vals),
											   list(true_neur_x_val_lengths)]
				neur_len_dict[seg_name + '_null'] =  [list(len_x_vals),
											   list(null_neur_x_val_lengths_mean),
											   list(null_neur_x_val_lengths_std)]
				percentiles = [] #Calculate percentile of true data point in null data distribution
				for l_cut in len_x_vals:
					try:
						percentiles.extend([round(stats.percentileofscore(null_neur_x_val_lengths_all[l_cut-1],true_neur_x_val_lengths[l_cut-1]),2)])
					except:
						percentiles.extend([100])
				neur_len_dict[seg_name + '_percentile'] =  [list(len_x_vals),percentiles]
			
			#Save the dictionaries
			filepath = self.bin_dir + 'neur_count_dict.npy'
			np.save(filepath, neur_count_dict)
			filepath = self.bin_dir + 'neur_spike_dict.npy'
			np.save(filepath, neur_spike_dict) 
			filepath = self.bin_dir + 'neur_len_dict.npy'
			np.save(filepath, neur_len_dict)

	def plot_statistics(self,):
		print('\tPlotting thresholded datasets')
		filepath = self.bin_dir + 'neur_count_dict.npy'
		neur_count_dict =  np.load(filepath, allow_pickle=True).item()
		filepath = self.bin_dir + 'neur_spike_dict.npy'
		neur_spike_dict = np.load(filepath, allow_pickle=True).item()
		filepath = self.bin_dir + 'neur_len_dict.npy'
		neur_spike_dict = np.load(filepath, allow_pickle=True).item()
		
		#_____Plotting_____
		neur_true_count_x = []
		neur_true_count_vals = []
		neur_null_count_x = []
		neur_null_count_mean = []
		neur_null_count_std = []
		neur_true_spike_x = []
		neur_true_spike_vals = []
		neur_null_spike_x = []
		neur_null_spike_mean = []
		neur_null_spike_std = []
		neur_true_len_x = []
		neur_true_len_vals = []
		neur_null_len_x = []
		neur_null_len_mean = []
		neur_null_len_std = []
		for s_i in tqdm.tqdm(self.segments_to_analyze):
			seg_name = self.segment_names[s_i]
			segment_start_time = self.segment_times[s_i]
			segment_end_time = self.segment_times[s_i+1]
			segment_length = segment_end_time - segment_start_time
			neur_true_count_data = neur_count_dict[seg_name + '_true']
			neur_null_count_data = neur_count_dict[seg_name + '_null']
			percentile_count_data = neur_count_dict[seg_name + '_percentile']
			neur_true_spike_data = neur_spike_dict[seg_name + '_true']
			neur_null_spike_data = neur_spike_dict[seg_name + '_null']
			percentile_spike_data = neur_spike_dict[seg_name + '_percentile']
			neur_true_len_data = neur_len_dict[seg_name + '_true']
			neur_null_len_data = neur_len_dict[seg_name + '_null']
			percentile_len_data = neur_len_dict[seg_name + '_percentile']
			#Plot the neuron count data
			norm_val = segment_length/1000 #Normalizing the number of bins to number of bins / second
			nd.plot_indiv_truexnull(np.array(neur_true_count_data[0]),np.array(neur_null_count_data[0]),np.array(neur_true_count_data[1]),np.array(neur_null_count_data[1]),
							   np.array(neur_null_count_data[2]),segment_length,norm_val,self.bin_dir,'Neuron Counts',seg_name,np.array(percentile_count_data[1]))	
			neur_true_count_x.append(np.array(neur_true_count_data[0]))
			neur_null_count_x.append(np.array(neur_null_count_data[0]))
			neur_true_count_vals.append(np.array(neur_true_count_data[1]))
			neur_null_count_mean.append(np.array(neur_null_count_data[1]))
			neur_null_count_std.append(np.array(neur_null_count_data[2]))
			#Plot the spike count data
			nd.plot_indiv_truexnull(np.array(neur_true_spike_data[0]),np.array(neur_null_spike_data[0]),np.array(neur_true_spike_data[1]),np.array(neur_null_spike_data[1]),
							   np.array(neur_null_spike_data[2]),np.array(segment_length),norm_val,self.bin_dir,'Spike Counts',seg_name,np.array(percentile_spike_data[1]))
			neur_true_spike_x.append(np.array(neur_true_spike_data[0]))
			neur_null_spike_x.append(np.array(neur_null_spike_data[0]))
			neur_true_spike_vals.append(np.array(neur_true_spike_data[1]))
			neur_null_spike_mean.append(np.array(neur_null_spike_data[1]))
			neur_null_spike_std.append(np.array(neur_null_spike_data[2]))
			#Plot the length data
			nd.plot_indiv_truexnull(np.array(neur_true_len_data[0]),np.array(neur_null_len_data[0]),np.array(neur_true_len_data[1]),np.array(neur_null_len_data[1]),
							   np.array(neur_null_len_data[2]),np.array(segment_length),norm_val,self.bin_dir,'Spike Counts',seg_name,np.array(percentile_len_data[1]))
			neur_true_len_x.append(np.array(neur_true_len_data[0]))
			neur_null_len_x.append(np.array(neur_null_len_data[0]))
			neur_true_len_vals.append(np.array(neur_true_len_data[1]))
			neur_null_len_mean.append(np.array(neur_null_len_data[1]))
			neur_null_len_std.append(np.array(neur_null_len_data[2]))
		#Plot all neuron count data
		nd.plot_all_truexnull(neur_true_count_x,neur_null_count_x,neur_true_count_vals,neur_null_count_mean,
										 neur_null_count_std,norm_val,self.bin_dir,'Neuron Counts',list(np.array(self.segment_names)[self.segments_to_analyze]))
		#Plot all spike count data
		nd.plot_all_truexnull(neur_true_spike_x,neur_null_spike_x,neur_true_spike_vals,neur_null_spike_mean,
										 neur_null_spike_std,norm_val,self.bin_dir,'Spike Counts',list(np.array(self.segment_names)[self.segments_to_analyze]))
		
		#Plot all length data
		nd.plot_all_truexnull(neur_true_len_x,neur_null_len_x,neur_true_len_vals,neur_null_len_mean,
										 neur_null_len_std,norm_val,self.bin_dir,'Lengths',list(np.array(self.segment_names)[self.segments_to_analyze]))
		
		
		


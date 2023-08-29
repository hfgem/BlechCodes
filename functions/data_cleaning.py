#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:58:29 2022
Data manipulation with bandpass filtering, etc... to pull out cleaner spikes
@author: hannahgermaine
"""
import numpy as np
from scipy.signal import butter, filtfilt
import os, tables, tqdm
import functions.hdf5_handling as h5
import functions.data_processing as dp


def butter_lowpass(cutoff, fs, order=5):
	"""Function per https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units"""
	return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(clean_data_dir, num_electrodes, cutoff, fs, order=5):
	"""Function per https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units"""
	y = []
	b, a = butter_lowpass(cutoff, fs, order=order)
	clean_hf5 = tables.open_file(clean_data_dir, 'r+', title = clean_data_dir[-1])
	data_list = clean_hf5.list_nodes('/common_avg_reference')
	for e_i in tqdm.tqdm(range(num_electrodes)):
		y.append(filtfilt(b, a, data_list[e_i][0]))
	clean_hf5.close()
	y = np.array(y)
	return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(clean_data_dir, num_electrodes, lowcut, highcut, fs, order=5):
	"""Function to bandpass filter. Calls butter_bandpass function.
	Copied from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
	Scipy documentation for bandpass filter
	data_segment = data to be filtered
	lowcut = low frequency
	highcut = high frequency
	fs = sampling rate of data"""
	y = []
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	clean_hf5 = tables.open_file(clean_data_dir, 'r+', title = clean_data_dir[-1])
	data_list = clean_hf5.list_nodes('/common_avg_reference')
	for e_i in tqdm.tqdm(range(num_electrodes)):
		y.append(filtfilt(b, a, data_list[e_i][0]))
	clean_hf5.close()
	y = np.array(y)
	return y

def signal_averaging(hf5_dir,clean_data_dir):
	"""Function to increase the signal-to-noise ratio by removing common signals
	across electrodes"""
	hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
	raw_electrodes = hf5.list_nodes('/raw')
	sort_order = np.argsort([x.__str__() for x in raw_electrodes])
	raw_electrodes = [raw_electrodes[i] for i in sort_order]
	total_time = raw_electrodes[0][:].shape[0]
	num_electrodes = len(raw_electrodes)
	electrode_data = np.zeros((num_electrodes,total_time))
	print('Convert data to mV')
	for e_i in tqdm.tqdm(range(num_electrodes)):
		#Convert data to mv scale
		electrode_data[e_i,:] = data_to_mv(raw_electrodes[e_i][:])
	hf5.close()
	common_average_reference = np.zeros(total_time)
	chunk = int(np.ceil(total_time/10000))
	chunk_starts = np.ceil(np.linspace(0,total_time,chunk))
	print('Calculating common average')
	for c_i in tqdm.tqdm(chunk_starts):
		   c_start = int(max(c_i,0))
		   c_end = int(min(c_i+chunk,total_time))
		   data_chunk = electrode_data[:,c_start:c_end]
		   mean_vec = np.mean(data_chunk,0)
		   std_vec = np.std(data_chunk,0)
		   pos_dev_mat = np.ones(np.shape(data_chunk))*(mean_vec + 3*std_vec)
		   neg_dev_mat =  np.ones(np.shape(data_chunk))*(mean_vec - 3*std_vec)
		   match_vals = np.sum((neg_dev_mat <= data_chunk)*(data_chunk <= pos_dev_mat),1)
		   cutoff = np.percentile(match_vals,90) #Keep only 90th percentile datasets for common average reference
		   keep_vals = match_vals <= cutoff
		   common_average_reference[c_start:c_end] = np.mean(data_chunk[keep_vals,:])
	print('Cleaning data')
	clean_hf5 = tables.open_file(clean_data_dir, 'r+', title = clean_data_dir[-1])
	atom = tables.FloatAtom()
	clean_hf5.create_group('/','common_avg_reference')
	for e_i in tqdm.tqdm(range(num_electrodes)):
		if e_i != 0:
			e_cleaned = electrode_data[e_i,:] - common_average_reference
			clean_hf5.create_earray('/common_avg_reference','electrode_' + str(e_i),atom,(0,) + np.shape(e_cleaned))
			exec('clean_hf5.root.common_avg_reference.electrode_'+str(e_i)+'.append(np.expand_dims(e_cleaned,0))')
	clean_hf5.close()

	return num_electrodes

def data_to_mv(data):
	"""Data is originally in microVolts, so here we convert to milliVolts"""
	mv_data = data*0.195*10**-3 #Multiply by 0.195 to convert to microVolts
	return mv_data
	
def data_cleanup(hf5_dir):
	"""This function cleans a dataset using downsampling, bandpass filtering, 
	and signal averaging"""

	cont_prompt_2 = 'y' #Continuation prompt initialization
	
	#Save directory
	folder_dir = hf5_dir.split('/')[:-1]
	folder_dir = '/'.join(folder_dir) + '/'
	clean_data_dir = hf5_dir.split('.h5')[0] + '_cleaned.h5'
	
	#First check for cleaned data
	re_clean = 'n'
	clean_exists = 0
	if os.path.isfile(clean_data_dir) == True:
		print("Cleaned Data Already Exists.")
		clean_exists = 1
		re_clean = input("\n INPUT REQUESTED: Would you like to re-clean the dataset [y/n]? ")
		if re_clean == 'y':
			print("Beginning re-cleaning dataset.")
		print("\n")
	#Clean or re-clean as per case
	if clean_exists == 0 or re_clean == 'y':
		
		#First perform downsampling / import downsampled data
		print("Downsampled Data Import/Handling Phase")
		h5.downsampled_electrode_data_handling(hf5_dir)
		
		#Open HF5
		hf5 = tables.open_file(hf5_dir, 'r+', title = hf5_dir[-1])
		
		sampling_rate = hf5.root.sampling_rate[0]
		dig_ins = hf5.list_nodes('/digital_in')
		dig_in_data = [dig_ins[d_i][:] for d_i in range(len(dig_ins))]
		del dig_ins
		
		hf5.close()
		
		
		#Get experiment components
		segment_names, segment_times = dp.get_experiment_components(sampling_rate, dig_in_data)
		
		#Begin storing data
		#Create new file for cleaned dataset
		if os.path.isfile(clean_data_dir) == True:
			os.remove(clean_data_dir)
		clean_hf5 = tables.open_file(clean_data_dir, 'w', title = clean_data_dir[-1])
		atom = tables.FloatAtom()
		sr_array = clean_hf5.create_earray('/','sampling_rate',atom,(0,))
		sr_array.append([sampling_rate])
		atom = tables.IntAtom()
		st_array = clean_hf5.create_earray('/','segment_times',atom,(0,))
		st_array.append(segment_times)
		atom = tables.Atom.from_dtype(np.dtype('U20')) #tables.StringAtom(itemsize=50)
		sn_array = clean_hf5.create_earray('/','segment_names',atom,(0,))
		sn_array.append(segment_names)
		clean_hf5.close()
		
		print("Common Average Filtering Data to Improve Signal-Noise Ratio")
		num_electrodes = signal_averaging(hf5_dir,clean_data_dir)
		
		print("Band Pass Filtering Data for Spike Detection")
		low_fq = 300
		high_fq = 3000
		filtered_data = bandpass_filter(clean_data_dir, num_electrodes, low_fq, high_fq,
										sampling_rate, order=5)
		
		print("Low Pass Filtering Data for LFPs")
		cutoff = 300  # desired cutoff frequency of the filter, Hz
		order = 6
		LFP_filtered_data = butter_lowpass_filter(clean_data_dir, num_electrodes, cutoff, sampling_rate, order)
		
		del mv_data, low_fq, high_fq, cutoff, order
		
		print("Saving LFP Data")
		clean_hf5 = tables.open_file(clean_data_dir, 'r+', title = clean_data_dir[-1])
		atom = tables.FloatAtom()
		clean_hf5.create_earray('/','lfp_data',atom,(0,) + np.shape(LFP_filtered_data))
		lfp_filtered_data_expanded = np.expand_dims(LFP_filtered_data,0)
		clean_hf5.root.lfp_data.append(lfp_filtered_data_expanded)
		clean_hf5.close()
		
		del LFP_filtered_data
		
		print("Saving cleaned data.")
		clean_hf5 = tables.open_file(clean_data_dir, 'r+', title = clean_data_dir[-1])
		atom = tables.FloatAtom()
# 		clean_hf5.create_earray('/','clean_data',atom,(0,) + np.shape(avg_data))
# 		avg_data_expanded = np.expand_dims(avg_data[:],0)
		clean_hf5.create_earray('/','clean_data',atom,(0,) + np.shape(filtered_data))
		avg_data_expanded = np.expand_dims(filtered_data[:],0)
		clean_hf5.root.clean_data.append(avg_data_expanded)
		clean_hf5.close()
		
		print("\n NOTICE: Checkpoint 2 Complete: You can quit the program here, if you like, and come back another time.")
		cont_loop = 1
		if cont_loop == 1:
			cont_prompt_2 = input("\n INPUT REQUESTED: Would you like to continue [y/n]? ")
			if cont_prompt_2 != 'y' and cont_prompt_2 != 'n':
				print("Error. Incorrect input.")
			else:
				cont_loop = 0
		
	return clean_data_dir, cont_prompt_2


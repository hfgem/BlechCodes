#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 19:12:11 2022

@author: hannahgermaine
A collection of functions to handle HDF5 data storage
"""
import tables, os, tqdm
import numpy as np
import functions.data_processing as dp

def hdf5_exists():
	"""This function asks for user input on whether a .h5 file exists"""
	h_loop = 1
	h_exists = 0
	while h_loop == 1:
		h_q = input("\n INPUT REQUESTED: Does an HDF5 file already exist? y / n: ")	
		if h_q != 'n' and h_q != 'y':
			print('Error, incorrect response, try again')
			h_loop = 1
		else:
			h_loop = 0
	if h_q == 'y':
		h_exists = 1
		
	return h_exists

def file_import(datadir, dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names):
	#Based on read_file.py by abuzarmahmood on GitHub - written by Abuzar Mahmood:
	#https://github.com/abuzarmahmood/blech_clust/blob/master/read_file.py
	#Create HDF5 file with data stored
	
	#filedir where all data is stored
	#dat_files_list contains the names of all .dat files
	#electrodes_list contains the names of all amp- ... -.dat files
	#emg_ind contains the index of the electrode that has EMG data
	#dig_in_list contains the names of the digital input files
	#dig_in_names contains the names of the tastants associated with each digital input file
	
	# Grab directory name to create the hdf5 file
	hdf5_name = str(os.path.dirname(datadir + '/')).split('/')
	hf5_dir = datadir + '/' + hdf5_name[-1]+'.h5'
	hf5 = tables.open_file(hf5_dir, 'w', title = hdf5_name[-1])
	hf5.create_group('/', 'raw')
	hf5.create_group('/', 'raw_emg')
	hf5.create_group('/', 'digital_in')
	hf5.create_group('/', 'digital_out')
	hf5.close()
	print('Created nodes in HF5')
	
	# Sort all lists
	dat_files_list.sort()
	electrodes_list.sort()
	dig_in_list.sort()
	emg_ind.sort()
	#DO NOT SORT dig_in_names - they are already sorted!
	
	#Separate electrodes and emg into their own lists
	all_electrodes = list()
	all_emg = list()
	for i in range(len(electrodes_list)):
		try:
			e_ind = emg_ind.index(i)
			all_emg.append(electrodes_list[i])
		except:
			all_electrodes.append(electrodes_list[i])
	
	# Read the amplifier sampling rate from info.rhd - 
	# look at Intan's website for structure of header files
	sampling_rate = np.fromfile(datadir + '/' + 'info.rhd', dtype = np.dtype('float32'))
	sampling_rate = int(sampling_rate[2])
	
	check_str = f'ports used: {electrodes_list} \n sampling rate: {sampling_rate} Hz'\
            f'\n digital inputs on intan board: {dig_in_list} \n ---------- \n \n'
	print(check_str)
	
	hf5 = tables.open_file(hf5_dir,'r+')
	
	# Create arrays for all components
	atom = tables.IntAtom()
	for i in all_electrodes: #add electrode arrays
		e_name = i.split('.')[0]
		e_name = e_name.split('-')[-1]
		hf5.create_earray('/raw',f'electrode_{e_name}',atom,(0,))
	for i in all_emg: #add emg arrays
		e_name = i.split('.')[0]
		e_name = e_name.split('-')[-1]
		hf5.create_earray('/raw_emg',f'electrode_{e_name}',atom,(0,))	
	for i in range(len(dig_in_list)): #add dig-in arrays
		d_name = dig_in_names[i]
		hf5.create_earray('/digital_in',f'digin_{d_name}',atom,(0,))	
		
	print('Reading electrodes')
	for i in tqdm.tqdm(all_electrodes):
		inputs = np.fromfile(datadir + '/' + i, dtype = np.dtype('uint16'))
		e_name = i.split('.')[0]
		e_name = e_name.split('-')[-1]
		exec("hf5.root.raw.electrode_"+str(e_name)+".append(inputs[:])")
	
	print('Reading emg')
	for i in tqdm.tqdm(all_emg):
		inputs = np.fromfile(datadir + '/' + i, dtype = np.dtype('uint16'))
		e_name = i.split('.')[0]
		e_name = e_name.split('-')[-1]
		exec("hf5.root.raw_emg.electrode_"+str(e_name)+".append(inputs[:])")
	
	print('Reading dig-ins')
	for i in tqdm.tqdm(range(len(dig_in_list))):
		d_name = dig_in_names[i]
		inputs = np.fromfile(datadir + '/' + dig_in_list[i], dtype = np.dtype('uint16'))
		exec("hf5.root.digital_in.digin_"+str(d_name)+".append(inputs[:])")
	
	hf5.close() #Close the file
	
	return hf5_dir

def downsampled_electrode_data_import(hf5_dir):
	"""This function imports data from the HDF5 file and stores only electrode 
	signals into an array"""
	folder_dir = hf5_dir.split('/')[:-1]
	folder_dir = '/'.join(folder_dir) + '/'
	new_hf5_dir = hf5_dir.split('.h5')[0] + '_downsampled.h5'
	
	#Has the electrode data already been stored in the hdf5 file?
	print("Checking for downsampled data.")
	if os.path.exists(new_hf5_dir):
		hf5_new = tables.open_file(new_hf5_dir, 'r', title = hf5_dir[-1])
		print("Data was previously stored in HDF5 file and is being imported. \n \n")
		e_data = hf5_new.root.electrode_array.data[0,:,:]
		dig_ins = hf5_new.root.dig_ins.dig_ins[0,:,:]
		unit_nums = hf5_new.root.electrode_array.unit_nums[:]
		segment_names = hf5_new.root.experiment_components.segment_names[:]
		segment_times = hf5_new.root.experiment_components.segment_times[:]
		hf5_new.close()
	else:
		print("Data was not previously stored in hf5. \n")
		#Ask about subsampling data
		sampling_rate = np.fromfile(folder_dir + 'info.rhd', dtype = np.dtype('float32'))
		sampling_rate = int(sampling_rate[2])
		sub_loop = 1
		sub_amount = 0
		while sub_loop == 1:
			print("The current sampling rate is " + str(sampling_rate) + ".")
			sub_q = input("\n INPUT REQUESTED: Would you like to downsample? y / n: ")	
			if sub_q != 'n' and sub_q != 'y':
				print('Error, incorrect response, try again')
				sub_loop = 1
			else:
				sub_loop = 0
		if sub_q == 'y':
			sub_loop = 1
			while sub_loop == 1:
				sub_amount = input("\n INPUT REQUESTED: Please enter a float describing the amount to downsample (ex. 0.5): ")
				try:
					sub_amount = float(sub_amount)
					sub_loop = 0
				except:
					print("Error. Please enter a valid float.")
					sub_loop = 1
			del sub_loop
		
		#Perform downsampling / regular data import
		e_data, dig_in_data, dig_in_names = dp.data_to_list(sub_amount,sampling_rate,hf5_dir)
			
        #Save data to hdf5
		e_data, unit_nums, dig_ins, segment_names, segment_times = save_downsampled_data(hf5_dir,e_data,sub_amount,sampling_rate,dig_in_data,dig_in_names)
	
	return e_data, unit_nums, dig_ins, segment_names, segment_times, new_hf5_dir

def save_downsampled_data(hf5_dir,e_data,sub_amount,sampling_rate,dig_in_data,dig_in_names):
	"""This function creates a new .h5 file to store only downsampled data arrays"""
	print("Saving Electrode Data to New HF5")
	atom = tables.FloatAtom()
	new_hf5_dir = hf5_dir.split('.h5')[0] + '_downsampled.h5'
	hf5 = tables.open_file(hf5_dir, 'r', title = hf5_dir[-1])
	hf5_new = tables.open_file(new_hf5_dir, 'w', title = new_hf5_dir[-1])
	units = hf5.list_nodes('/raw')
	unit_nums = np.array([str(unit).split('_')[-1] for unit in units])
	unit_nums = np.array([int(unit.split(' (')[0]) for unit in unit_nums])
	hf5_new.create_group('/','electrode_array')
	np_e_data = np.array(e_data)
	data = hf5_new.create_earray('/electrode_array','data',atom,(0,)+np.shape(np_e_data))
	np_e_data = np.expand_dims(np_e_data,0)
	data.append(np_e_data)
	del np_e_data, e_data
	atom = tables.IntAtom()
	hf5_new.create_earray('/electrode_array','unit_nums',atom,(0,))
	exec("hf5_new.root.electrode_array.unit_nums.append(unit_nums[:])")
	if sub_amount > 0:
		new_rate = [round(sampling_rate*sub_amount)]
		hf5_new.create_earray('/','sampling_rate',atom,(0,))
		hf5_new.root.sampling_rate.append(new_rate)
	else:
		hf5_new.create_earray('/','sampling_rate',atom,(0,))
		hf5_new.root.sampling_rate.append([sampling_rate])
	print("Saving Dig Ins to New HF5")
	np_dig_ins = np.array(dig_in_data)
	atom = tables.FloatAtom()
	hf5_new.create_group('/','dig_ins')
	data_digs = hf5_new.create_earray('/dig_ins','dig_ins',atom,(0,)+np.shape(np_dig_ins))
	np_dig_ins = np.expand_dims(np_dig_ins,0)
	data_digs.append(np_dig_ins)
	atom = tables.Atom.from_dtype(np.dtype('U20')) #tables.StringAtom(itemsize=50)
	dig_names = hf5_new.create_earray('/dig_ins','dig_in_names',atom,(0,))
	dig_names.append(np.array(dig_in_names))
	del np_dig_ins, data_digs, dig_in_data, dig_names
       #Close HDF5 file
	hf5.close()
	hf5_new.close()
	print("Getting Experiment Components")
	segment_names, segment_times = dp.get_experiment_components(new_hf5_dir)
	hf5_new = tables.open_file(new_hf5_dir, 'r+', title = new_hf5_dir[-1])
	hf5_new.create_group('/','experiment_components')
	atom = tables.IntAtom()
	hf5_new.create_earray('/experiment_components','segment_times',atom,(0,))
	exec("hf5_new.root.experiment_components.segment_times.append(segment_times[:])")
	atom = tables.Atom.from_dtype(np.dtype('U20')) #tables.StringAtom(itemsize=50)
	hf5_new.create_earray('/experiment_components','segment_names',atom,(0,))
	exec("hf5_new.root.experiment_components.segment_names.append(np.array(segment_names))")
	e_data = hf5_new.root.electrode_array.data[0,:,:]
	dig_ins = hf5_new.root.dig_ins.dig_ins[0,:,:]
	hf5_new.close()
	del hf5, units, sub_amount
	
	return e_data, unit_nums, dig_ins, segment_names, segment_times

def check_ICA_data(hf5_dir):
	ica_data_dir = ('/').join(hf5_dir.split('/')[:-1]) + '/ica_results/'
	ica_hf5_name = hf5_dir.split('/')[-1].split('.')[0] + '_ica.h5'
	ica_hf5_dir = ica_data_dir + ica_hf5_name
	if os.path.exists(ica_hf5_dir):
		exists = 1
	else:
		exists = 0
	return exists, ica_data_dir
	
def save_sorted_spikes(final_h5_dir,spike_raster,sort_stats):
	"""This function saves the sorted results to an HDF5 file"""
	final_hf5 = tables.open_file(final_h5_dir, 'w', title = final_h5_dir[-1])
	atom = tables.IntAtom()
	final_hf5.create_earray('/','spike_raster',atom,(0,)+np.shape(spike_raster))
	spike_expand = np.expand_dims(spike_raster,0)
	final_hf5.root.spike_raster.append(spike_expand)
	atom = tables.FloatAtom()
	final_hf5.create_earray('/','sort_stats',atom,(0,)+np.shape(sort_stats))
	sort_stats_expand = np.expand_dims(sort_stats,0)
	final_hf5.root.sort_stats.append(sort_stats_expand)
	final_hf5.close()
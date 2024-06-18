#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:39:29 2024

@author: Hannah Germaine

Data Import Utils
"""

import os
import tables
import csv
import numpy as np
import functions.load_intan_rhd_format.load_intan_rhd_format as rhd
import functions.data_processing as dp
import functions.analysis_funcs as af

class import_data():
	
	def __init__(self,args):
		self.data_path = args[0]
		self.hdf5_path = args[1]
		dig_in_names = args[2]
		self.get_sampling_rate()
		self.get_spikes()
		self.get_dig_ins(dig_in_names)
		self.get_segments()
		self.add_no_taste()
	
	def get_sampling_rate(self,):
		#Grab data from hdf5 file
		blech_clust_h5 = tables.open_file(self.hdf5_path, 'r+', title = 'hdf5_file')

		#Grab sampling rate
		try:
			sampling_rate = blech_clust_h5.root.sampling_rate[0]
		except:
			#The old method doesn't currently store sampling_rate, so this picks it up
			rhd_dict = rhd.import_data(self.data_path)
			sampling_rate = int(rhd_dict["frequency_parameters"]["amplifier_sample_rate"])
			atom = tables.IntAtom()
			blech_clust_h5.create_earray('/','sampling_rate',atom,(0,))
			blech_clust_h5.root.sampling_rate.append([sampling_rate])
		ms_conversion = (1/sampling_rate)*(1000/1) #ms/samples units
		self.sampling_rate = sampling_rate
		self.ms_conversion = ms_conversion
		
		blech_clust_h5.close()
		
	def get_spikes(self,):
		
		#Grab data from hdf5 file
		blech_clust_h5 = tables.open_file(self.hdf5_path, 'r+', title = 'hdf5_file')
		
		sorted_units_node = blech_clust_h5.get_node('/sorted_units')
		num_neur = len([s_n for s_n in sorted_units_node])
		self.num_neur = num_neur
		#Grab waveforms
		print("\tGrabbing waveforms")
		all_waveforms = [sn.waveforms[0] for sn in sorted_units_node]
		self.all_waveforms = all_waveforms
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
		self.max_time = np.ceil(max(max_times)*self.ms_conversion)
		#Converting spike times to ms timescale
		spike_times = [np.ceil(np.array(spike_times[i])*self.ms_conversion) for i in range(len(spike_times))]
		self.spike_times = spike_times		

		blech_clust_h5.close()
	
	def get_dig_ins(self, dig_in_names):
		blech_clust_h5 = tables.open_file(self.hdf5_path, 'r+', title = 'hdf5_file')
		
		#Dig In Times
		start_dig_in_times_csv = self.data_path + 'start_dig_in_times.csv'
		end_dig_in_times_csv = self.data_path + 'end_dig_in_times.csv'
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
			start_dig_in_times = [list(np.ceil((np.where(np.diff(np.array(dig_in_data[i])) == 1)[0] + 1)*self.ms_conversion).astype('int')) for i in range(num_tastes)]	
			end_dig_in_times = [list(np.ceil((np.where(np.diff(np.array(dig_in_data[i])) == -1)[0] + 1)*self.ms_conversion).astype('int')) for i in range(num_tastes)]
			#Store these into csv for import in future instead of full dig_in_data load which takes forever!
			with open(start_dig_in_times_csv, 'w') as f:
				write = csv.writer(f,delimiter=',')
				write.writerows(start_dig_in_times)
			with open(end_dig_in_times_csv, 'w') as f:
				write = csv.writer(f,delimiter=',')
				write.writerows(end_dig_in_times)
		self.start_dig_in_times = start_dig_in_times
		self.end_dig_in_times = end_dig_in_times
		self.num_tastes = num_tastes
		
		#Dig In Names
		if len(dig_in_names) > 0:
			atom = tables.Atom.from_dtype(np.dtype('U20')) #tables.StringAtom(itemsize=50)
			try:
				blech_clust_h5.remove_node('/digital_in','dig_in_names')
				dig_names = blech_clust_h5.create_earray('/digital_in','dig_in_names',atom,(0,))
			except:
				dig_names = blech_clust_h5.create_earray('/digital_in','dig_in_names',atom,(0,))
			dig_names.append(np.array(dig_in_names))
		else:
			try: #maybe they were saved before!
				dig_in_names = [blech_clust_h5.root.digital_in.dig_in_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.digital_in.dig_in_names))]
			except:
				#Probe the user
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
		self.dig_in_names = dig_in_names
		blech_clust_h5.close()
		
	def get_segments(self,):
		blech_clust_h5 = tables.open_file(self.hdf5_path, 'r+', title = 'hdf5_file')
		try:
			segment_times = blech_clust_h5.root.experiment_components.segment_times[:]
			segment_names = [blech_clust_h5.root.experiment_components.segment_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.experiment_components.segment_names))]
			self.segment_times = segment_times
			self.segment_names = segment_names
		except:
			min_dig_in_time = min([min(self.start_dig_in_times[i]) for i in range(self.num_tastes)])
			max_dig_in_time = max([max(self.end_dig_in_times[i]) for i in range(self.num_tastes)])
			dig_in_ind_range=np.array([min_dig_in_time/self.ms_conversion,max_dig_in_time/self.ms_conversion]) #unconvert because dp looks at sampling rate values
			segment_names, segment_times = dp.get_experiment_components(self.sampling_rate, dig_in_ind_range=dig_in_ind_range, len_rec=self.max_time/self.ms_conversion)  #unconvert because dp looks at sampling rate values
			#Convert segment times to ms timescale for saving
			segment_times = np.ceil(segment_times*self.ms_conversion)
			self.segment_times = segment_times
			self.segment_names = segment_names
			
		try:
			blech_clust_h5.create_group('/','experiment_components')
		except:
			print("\n\tExperiment components group already exists in .h5 file")
		atom = tables.IntAtom()
		try:
			blech_clust_h5.create_earray('/experiment_components','segment_times',atom,(0,))
			exec("blech_clust_h5.root.experiment_components.segment_times.append(self.segment_times[:])")
		except:
			blech_clust_h5.remove_node('/experiment_components','segment_times')
			blech_clust_h5.create_earray('/experiment_components','segment_times',atom,(0,))
			exec("blech_clust_h5.root.experiment_components.segment_times.append(self.segment_times[:])")
		atom = tables.Atom.from_dtype(np.dtype('U20'))
		try:
			blech_clust_h5.create_earray('/experiment_components','segment_names',atom,(0,))
			exec("blech_clust_h5.root.experiment_components.segment_names.append(np.array(self.segment_names))")
		except:
			blech_clust_h5.remove_node('/experiment_components','segment_names')
			blech_clust_h5.create_earray('/experiment_components','segment_names',atom,(0,))
			exec("blech_clust_h5.root.experiment_components.segment_names.append(np.array(self.segment_names))")
		
		blech_clust_h5.close()

	def add_no_taste(self,):
		if self.dig_in_names[-1] != 'none':
			dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes = af.add_no_taste(self.start_dig_in_times, self.end_dig_in_times, self.dig_in_names)
			self.dig_in_names = dig_in_names
			self.start_dig_in_times = start_dig_in_times
			self.end_dig_in_times = end_dig_in_times
			self.num_tastes = num_tastes
		
		
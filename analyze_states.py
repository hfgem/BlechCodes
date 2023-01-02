#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:04:51 2023

@author: hannahgermaine

This code is written to import sorted data and perform state-change analyses
"""

import functions.hdf5_handling as hf5
import tables

#Get the directory of the hdf5 file
sorted_dir = hf5.sorted_data_import() #Program will automatically close if file not found in given folder

#Import spike times and waveforms
#Transform data from blech_clust hdf5 file into correct format
blech_clust_h5 = tables.open_file(sorted_dir, 'r', title = sorted_dir[-1])
sorted_units_node = blech_clust_h5.get_node('/sorted_units')
num_old_neur = len([s_n for s_n in sorted_units_node])
#Grab waveforms
all_waveforms = [sn.waveforms[:] for sn in sorted_units_node]
#Grab times
spike_times = []
i = 0
for s_n in sorted_units_node:
	spike_times.append(list(s_n.times[:]))
	i+= 1
del s_n
#If segment data exists, grab, otherwise ask for user input


blech_clust_h5.close()


	

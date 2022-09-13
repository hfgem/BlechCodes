#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:13:55 2022

@author: hannahgermaine

This code is written to import sorted data from BlechClust and compare it 
against this package's sort results.
"""

import os, tables
import tkinter as tk
import tkinter.filedialog as fd

#Get the directory of the hdf5 files
print("\n INPUT REQUESTED: Select directory with blech_clust .h5 file.")
root = tk.Tk()
currdir = os.getcwd()
blech_clust_datadir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')
blech_clust_hdf5_name = str(os.path.dirname(blech_clust_datadir + '/')).split('/')
blech_clust_hf5_dir = blech_clust_datadir + '/' + blech_clust_hdf5_name[-1]+'.h5'

print("\n INPUT REQUESTED: Select directory with new sort method .h5 file.")
root = tk.Tk()
currdir = os.getcwd()
new_datadir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')
new_hdf5_name = str(os.path.dirname(new_datadir + '/')).split('/')
new_hf5_dir = new_datadir + '/' + new_hdf5_name[-1]+'.h5'

#Import relevant data
new_hf5 = tables.open_file(new_hf5_dir,'r',title = new_hf5_dir[-1])
spikes_bin = new_hf5.root.spike_raste[0]
sort_stats = new_hf5.root.sort_stats[0]
sampling_rate = new_hf5.root.sampling_rate[0]
new_hf5.close()

#Transform data from blech_clust hdf5 file into correct format
blech_clust_h5 = tables.open_file(blech_clust_hf5_dir, 'r', title = blech_clust_hf5_dir[-1])
sorted_units_node = blech_clust_h5.get_node('/sorted_units')
assumed_sampling_rate = 30000
downsample_div = round(assumed_sampling_rate/sampling_rate)

for i in range(len(sorted_units_node)):
	node_name = sorted_units_node.name
	node_ind = node_name.split('unit')[-1]
	node_times = sorted_units_node.times[:]
	node_times_downsampled = round(node_times/2)

blech_clust_h5.close()



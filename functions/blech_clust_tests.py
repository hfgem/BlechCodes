#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:13:55 2022

@author: hannahgermaine

This code is written to import sorted data from BlechClust and compare it 
against this package's sort results.
"""

import os, tables, tqdm, csv
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import matplotlib.pyplot as plt

#Get the directory of the hdf5 files
print("\n INPUT REQUESTED: Select directory with blech_clust .h5 file.")
root = tk.Tk()
currdir = os.getcwd()
blech_clust_datadir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')
blech_clust_hdf5_name = str(os.path.dirname(blech_clust_datadir + '/')).split('/')
blech_clust_hf5_dir = blech_clust_datadir + '/' + blech_clust_hdf5_name[-1]+'.h5'
print("\n INPUT REQUESTED: Select directory with new sort method .h5 file.")
new_datadir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')
new_hdf5_name = str(os.path.dirname(new_datadir + '/')).split('/')
new_hf5_dir = new_datadir + '/' + new_hdf5_name[-1] + '_sorted_results.h5'

#Create a directory to store overlap data
overlap_folder = new_datadir + '/overlaps/'
if os.path.isdir(overlap_folder) == False:
	os.mkdir(overlap_folder)

#Import relevant data
new_hf5 = tables.open_file(new_hf5_dir,'r',title = new_hf5_dir[-1])
new_spikes_bin = new_hf5.root.spike_raster[0]
sort_stats = new_hf5.root.sort_stats[0]
sampling_rate = new_hf5.root.sampling_rate[0]
new_hf5.close()

num_new_neur, num_new_time = np.shape(new_spikes_bin)

#Transform data from blech_clust hdf5 file into correct format
blech_clust_h5 = tables.open_file(blech_clust_hf5_dir, 'r', title = blech_clust_hf5_dir[-1])
sorted_units_node = blech_clust_h5.get_node('/sorted_units')
assumed_sampling_rate = 30000
downsample_div = round(assumed_sampling_rate/sampling_rate)
num_old_neur = len(sorted_units_node)
old_spikes_bin = np.zeros((num_old_neur,num_new_time))
for i in range(num_old_neur):
	node_name = sorted_units_node.name
	node_ind = node_name.split('unit')[-1]
	node_times = sorted_units_node.times[:]
	node_times_downsampled = round(node_times/2)
	bin_unit_vector = np.zeros((1,num_new_time))
	bin_unit_vector[0,node_times_downsampled] = 1
	old_spikes_bin[i,:] = bin_unit_vector
blech_clust_h5.close()

#Perform pairwise comparison - with 3 bin smudge as in collision tests
blur_ind = 3
collision_cutoff = 50 #Percent cutoff for collisions
colorCodes = np.array([[0,1,0],[0,0,1]]) #Colors for plotting collision rasters
new_percents = np.zeros((num_new_neur,num_old_neur))
old_percents = np.zeros((num_new_neur,num_old_neur))
for new_i in tqdm.tqdm(range(num_new_neur)):
	#Grab new unit spikes and blur
	new_unit = new_spikes_bin[new_i,:]
	new_u_blur = np.zeros(np.shape(new_unit))
	new_u_blur += new_unit
	num_spikes_new = np.sum(new_unit)
	for b_i in range(blur_ind):
		new_u_blur[0:-1*(b_i+1)] += new_unit[b_i+1:]
		new_u_blur[b_i+1:] += new_unit[0:-1*(b_i+1)]
	for old_i in range(num_old_neur):
		#Grab old unit spikes and blur
		old_unit = old_spikes_bin[old_i,:]
		old_u_blur = np.zeros(np.shape(old_unit))
		for b_i in range(blur_ind):
			old_u_blur[0:-1*(b_i+1)] += old_unit[b_i+1:]
			old_u_blur[b_i+1:] += old_unit[0:-1*(b_i+1)]
		num_spikes_old = np.sum(old_unit)
		#Find the number of overlaps and store percents
		overlaps_pair = np.multiply(new_u_blur,old_u_blur)
		overlaps_count = len(np.where(np.diff(overlaps_pair)>1)[0] + 1)
		new_percents[new_i,old_i] = round((overlaps_count/num_spikes_new)*100,2)
		old_percents[new_i,old_i] = round((overlaps_count/num_spikes_old)*100,2)
#Create a figure of the percents
fig = plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(new_percents)
plt.colorbar()
plt.title("Percents of new sorts")
plt.subplot(1,2,2)
plt.imshow(old_percents)
plt.colorbar()
plt.title("Percents of old sorts")
plt.savefig(overlap_folder+'percents.png',dpi=100)
plt.close(fig)

#For each new sorted unit, find the index of the old sorted unit that best matches
old_matches = np.zeros((num_new_neur,2))
for new_i in tqdm.tqdm(range(num_new_neur)):
	old_matches[new_i,0] = np.argmax(old_percents[new_i,:])
	old_matches[new_i,1] = np.amax(old_percents[new_i,:])
overlaps_csv = overlap_folder + 'overlap_vals.csv'
with open(overlaps_csv, 'w') as f:
	write = csv.writer(f)
	write.writerows(old_matches)

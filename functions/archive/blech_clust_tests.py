#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:13:55 2022

@author: hannahgermaine

This code is written to import sorted data from BlechClust and compare it 
against this package's sort results.
"""

import os, tables, tqdm, csv, sys
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import matplotlib.pyplot as plt
try:
	from functions.postsort import collision_func
except:
	from postsort import collision_func

#Get the directory of the hdf5 files
print("\n INPUT REQUESTED: Select directory with blech_clust .h5 file.")
root = tk.Tk()
currdir = os.getcwd()
blech_clust_datadir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')
files_in_dir = os.listdir(blech_clust_datadir)
for i in range(len(files_in_dir)): #Assumes the correct file is the only .h5 in the directory
	filename = files_in_dir[i]
	if filename.split('.')[-1] == 'h5':
		blech_clust_hdf5_name = filename
try:
	blech_clust_hf5_dir = blech_clust_datadir + '/' + blech_clust_hdf5_name
except:
	print("Old .h5 file not found. Quitting program.")
	sys.exit()
	
print("\n INPUT REQUESTED: Select directory with new sort method .h5 file.")
new_datadir = fd.askdirectory(parent=root, initialdir=currdir, title='Please select the folder where data is stored.')
info_file = np.fromfile(new_datadir + '/' + 'info.rhd', dtype = np.dtype('float32'))
original_sampling_rate = int(info_file[2])
new_hdf5_name = str(os.path.dirname(new_datadir + '/')).split('/')
new_hf5_info_dir = new_datadir + '/' + new_hdf5_name[-1] + '_downsampled.h5'
new_hf5_info = tables.open_file(new_hf5_info_dir,'r',title=new_hf5_info_dir[-1])
num_new_time = np.shape(new_hf5_info.root.electrode_array.data)[-1]
new_sampling_rate = new_hf5_info.root.sampling_rate[0]
new_hf5_info.close()
new_hf5_dir = new_datadir + '/' + new_datadir.split('/')[-1] + '_repacked.h5'
#Create a directory to store overlap data
overlap_folder = new_datadir + '/compare_with_blech_clust/'
if os.path.isdir(overlap_folder) == False:
	os.mkdir(overlap_folder)

#Import relevant data
new_hf5 = tables.open_file(new_hf5_dir,'r',title = new_hf5_dir[-1])
sorted_units_node = new_hf5.get_node('/sorted_units')
num_new_neur = len([s_n for s_n in sorted_units_node])
downsample_div = round(original_sampling_rate/new_sampling_rate)
new_spike_times = []
new_spike_times_combined = []
i = 0
for s_n in sorted_units_node:
	unit_times = list(eval('s_n.times[0]').round().astype(int))
	new_spike_times.append(unit_times)
	new_spike_times_combined.extend(unit_times)
del s_n, unit_times
new_spike_times_combined = np.sort(new_spike_times_combined)
new_hf5.close()

#Transform data from blech_clust hdf5 file into correct format
blech_clust_h5 = tables.open_file(blech_clust_hf5_dir, 'r', title = blech_clust_hf5_dir[-1])
sorted_units_node = blech_clust_h5.get_node('/sorted_units')
num_old_neur = len([s_n for s_n in sorted_units_node])
all_old_waveforms = [sn.waveforms[:] for sn in sorted_units_node]
old_spikes_times = []
old_spike_times_combined = []
i = 0
for s_n in sorted_units_node:
	node_times = s_n.times[:]
	node_times_downsampled = (node_times/downsample_div).round().astype(int)
	old_spikes_times.append(list(node_times_downsampled))
	old_spike_times_combined.extend(list(node_times_downsampled))
	i+= 1
del s_n, node_times, node_times_downsampled
old_spike_times_combined = np.sort(old_spike_times_combined)
blech_clust_h5.close()

num_old_neur = len(old_spike_times_combined)
num_new_neur = len(new_spike_times_combined)

print("Both datasets imported.")
print("Total old spike times = " + str(num_old_neur))
print("Total new spike times = " + str(num_new_neur))
print("Ratio of number of new:old spikes = " + str(int(np.ceil(num_new_neur/num_old_neur))) + ":1")

#%% Compare spike times running through each old spike and seeing if it has a collision in the new spike times
print("Checking old indices represented in new dataset.")
blur_ind = round((0.5/1000)*new_sampling_rate) #amount of blurring allowed
num_old_neur = len(old_spike_times_combined)
old_rep = np.zeros(num_old_neur)
for o_i in tqdm.tqdm(range(len(old_spike_times_combined))):
	old_ind = old_spike_times_combined[o_i]
	close_ind = np.where((new_spike_times_combined <= old_ind + blur_ind)&(new_spike_times_combined >= old_ind - blur_ind))[0]
	old_rep[o_i] = len(close_ind)
#Number of old neurons represented
num_old_overlap = len(np.where(old_rep > 0)[0])
num_over_overlap = len(np.where(old_rep > 1)[0])
total_overlap = np.sum(old_rep)
print("Percent of old neurons represented = " + str(np.ceil(100*num_old_overlap/num_old_neur)))
print("Percent of over-represented represented-neurons = " + str(np.ceil(100*num_over_overlap/num_old_overlap)))

#%% Compare overall spike counts


#%% Perform bulk comparison - with all new spikes collapsed and compared against old
blur_ind = round((0.5/1000)*new_sampling_rate) #amount of blurring allowed
combined_percents = np.zeros(num_old_neur)
for old_i in tqdm.tqdm(range(num_old_neur)):
    #Grab old unit spikes and blur
    old_unit_list = old_spikes_times[old_i]
    num_spikes_old = len(old_unit_list)
    old_overlaps, new_overlaps = collision_func(old_unit_list,new_spike_times_combined,blur_ind)
    #Find the number of overlaps and store percents
    col_perc_old = np.round(100*old_overlaps/num_spikes_old,2)
    combined_percents[old_i] = col_perc_old

print(combined_percents)
print(np.mean(combined_percents))
print(np.std(combined_percents))

#%% Perform pairwise comparison - with 3 bin smudge as in collision tests

blur_ind = round((0.5/1000)*new_sampling_rate) #amount of blurring allowed
collision_cutoff = 50 #Percent cutoff for collisions
colorCodes = np.array([[0,1,0],[0,0,1]]) #Colors for plotting collision rasters
new_percents = np.zeros((num_new_neur,num_old_neur))
old_percents = np.zeros((num_new_neur,num_old_neur))
for new_i in tqdm.tqdm(range(num_new_neur)):
 	#Grab new unit spikes and blur
 	new_unit = new_spike_times[new_i]
 	num_spikes_new = len(new_unit)
 	for old_i in range(num_old_neur):
		 #Grab old unit spikes and blur
		 old_unit = old_spikes_times[old_i]
		 num_spikes_old = len(old_unit)
		 #Find the number of overlaps and store percents
		 old_overlaps, new_overlaps = collision_func(old_unit,new_unit,blur_ind)
		 new_percents[new_i,old_i] = round((new_overlaps/num_spikes_new)*100,2)
		 old_percents[new_i,old_i] = round((old_overlaps/num_spikes_old)*100,2)
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

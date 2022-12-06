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
from scipy.fftpack import rfft, fftfreq
from scipy.signal import find_peaks
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

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
new_hdf5_name = str(os.path.dirname(new_datadir + '/')).split('/')
new_hf5_info_dir = new_datadir + '/' + new_hdf5_name[-1] + '_downsampled.h5'
new_hf5_info = tables.open_file(new_hf5_info_dir,'r',title=new_hf5_info_dir[-1])
num_new_time = np.shape(new_hf5_info.root.electrode_array.data)[-1]
new_hf5_info.close()
new_hf5_dir = new_datadir + '/sort_results/' + new_hdf5_name[-1].split('_')[0] + '_sort.h5'

#Create a directory to store overlap data
overlap_folder = new_datadir + '/overlaps/'
if os.path.isdir(overlap_folder) == False:
	os.mkdir(overlap_folder)

#Import relevant data
new_hf5 = tables.open_file(new_hf5_dir,'r',title = new_hf5_dir[-1])
sorted_units_node = new_hf5.get_node('/sorted_units')
sampling_rate = 30000
assumed_sampling_rate = 30000
downsample_div = round(assumed_sampling_rate/assumed_sampling_rate)
num_new_neur = sum([len([w for w in s_n.waveforms]) for s_n in sorted_units_node])
new_spikes_bin = np.zeros((num_new_neur,num_new_time))
i = 0
for s_n in sorted_units_node:
	num_units = len([w_n for w_n in s_n.times])
	for n_u in range(num_units):
		unit_times = eval('s_n.times.neuron_' + str(n_u) + '[0]').round().astype(int)
		new_spikes_bin[i,unit_times] = 1
		i += 1
new_hf5.close()

#Transform data from blech_clust hdf5 file into correct format
blech_clust_h5 = tables.open_file(blech_clust_hf5_dir, 'r', title = blech_clust_hf5_dir[-1])
sorted_units_node = blech_clust_h5.get_node('/sorted_units')
num_old_neur = len([s_n for s_n in sorted_units_node])
all_old_waveforms = [sn.waveforms[:] for sn in sorted_units_node]
old_spikes_bin = np.zeros((num_old_neur,num_new_time))
i = 0
for s_n in sorted_units_node:
	node_times = s_n.times[:]
	node_times_downsampled = (node_times/2).round().astype(int)
	bin_unit_vector = np.zeros((1,num_new_time))
	bin_unit_vector[0,node_times_downsampled] = 1
	old_spikes_bin[i,:] = bin_unit_vector
	i+= 1
blech_clust_h5.close()

#%% Perform bulk comparison - with all new spikes collapsed and compared against old
blur_ind = round((0.5/1000)*sampling_rate)
combined_percents = np.zeros(num_old_neur)
combined_spikes = np.sum(new_spikes_bin,0)
combined_spikes_blur = np.zeros(np.shape(combined_spikes))
combined_spikes_blur += combined_spikes
for b_i in range(blur_ind):
    combined_spikes_blur[0:-1*(b_i+1)] += combined_spikes[b_i+1:]
    combined_spikes_blur[b_i+1:] += combined_spikes[0:-1*(b_i+1)]
for old_i in tqdm.tqdm(range(num_old_neur)):
    #Grab old unit spikes and blur
    old_unit = old_spikes_bin[old_i,:]
    num_spikes_old = np.sum(old_unit)
    old_u_blur = np.zeros(np.shape(old_unit))
    old_u_blur += old_unit
    for b_i in range(blur_ind):
        old_u_blur[0:-1*(b_i+1)] += old_unit[b_i+1:]
        old_u_blur[b_i+1:] += old_unit[0:-1*(b_i+1)]
    #Find the number of overlaps and store percents
    overlaps_pair = np.multiply(combined_spikes_blur,old_u_blur)
    overlaps_count = len(np.where(np.diff(overlaps_pair)>1)[0])+1
    combined_percents[old_i] = round((overlaps_count/num_spikes_old)*100,2)

print(combined_percents)
print(np.mean(combined_percents))
print(np.std(combined_percents))

#%% Perform pairwise comparison - with 3 bin smudge as in collision tests

###THIS CODE HAS ERRORS IN IT - NEED TO FIX
blur_ind = 3
collision_cutoff = 50 #Percent cutoff for collisions
colorCodes = np.array([[0,1,0],[0,0,1]]) #Colors for plotting collision rasters
new_percents = np.zeros((num_new_neur,num_old_neur))
old_percents = np.zeros((num_new_neur,num_old_neur))
for new_i in tqdm.tqdm(range(num_new_neur)):
 	#Grab new unit spikes and blur
 	new_unit = new_spikes_bin[new_i,:]
 	num_spikes_new = np.sum(new_unit)
 	new_u_blur = np.zeros(np.shape(new_unit))
 	new_u_blur += new_unit
 	for b_i in range(blur_ind):
		 new_u_blur[0:-1*(b_i+1)] += new_unit[b_i+1:]
		 new_u_blur[b_i+1:] += new_unit[0:-1*(b_i+1)]
 	for old_i in range(num_old_neur):
		 #Grab old unit spikes and blur
		 old_unit = old_spikes_bin[old_i,:]
		 num_spikes_old = np.sum(old_unit)
		 old_u_blur = np.zeros(np.shape(old_unit))
		 old_u_blur += old_unit
		 for b_i in range(blur_ind):
 			old_u_blur[0:-1*(b_i+1)] += old_unit[b_i+1:]
 			old_u_blur[b_i+1:] += old_unit[0:-1*(b_i+1)]
		 #Find the number of overlaps and store percents
		 overlaps_pair = np.multiply(new_u_blur,old_u_blur)
		 overlaps_count = len(np.where(np.diff(overlaps_pair)>1)[0])+1
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

#%% Perform Fourier Transform on Old Spike Data

for u_i in range(num_old_neur):
	unit_waveforms = all_old_waveforms[u_i]
	unit_fourier = np.array([list(rfft(unit_waveforms[s_i])) for s_i in range(len(unit_waveforms))])
	freqs = fftfreq(len(unit_waveforms[0]),d=1/sampling_rate)
	fourier_peaks = np.array([list(find_peaks(unit_fourier[s_i])[0][0:20]) for s_i in range(len(unit_waveforms))])
	
	freq_ind = (freqs<1000)*(freqs>0)
	im_vals = unit_fourier[:,freq_ind]
	norm_fourier = normalize(im_vals,axis=1,norm='max')
	im_freqs = freqs[freq_ind]
	indices = np.unique(np.round(np.linspace(0,len(norm_fourier[0]),10)).astype(int))
	indices[-1] -= 1
	fig = plt.figure(figsize=(20,20))
	plt.imshow(norm_fourier,aspect='auto')
	plt.xticks(ticks=indices,labels=im_freqs[indices])
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Waveform #')
	plt.title('Normalized Fourier Transform of Unit ' + str(u_i))
	fig.savefig(blech_clust_datadir + '/' + 'fourier_unit_' + str(u_i) + '.png', dpi=100)
	plt.close(fig)

#fourier = rfft(mean_bit)
#freqs = fftfreq(len(mean_bit), d=1/sampling_rate)
#fourier_peaks = find_peaks(fourier)[0]
#peak_freqs = freqs[fourier_peaks]
#peak_freqs = peak_freqs[peak_freqs>0]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 08:22:25 2022
@author: Hannah Germaine
Collection of functions related to spike clustering.

"""

import os, csv, tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, fftfreq
from scipy.signal import find_peaks
from sklearn.metrics import silhouette_score, silhouette_samples

def cluster(spikes, peak_indices, i, sort_data_dir, axis_labels, type_spike, 
			segment_times, segment_names, dig_in_times, dig_in_names, 
			sampling_rate, re_sort):
	"""This function tests different numbers of clusters for spike clustering
	using the silhouette score method. It outputs the best clustering results
	and returns indices of good waveforms.
	"""
	
	viol_1 = sampling_rate*(1/1000)
	viol_2 = sampling_rate*(2/1000)
	
	#First test different numbers of clusters
	clust_num_vec = np.arange(3,16)
	silhouette_scores = np.zeros(np.shape(clust_num_vec))
	for i in tqdm.tqdm(range(len(clust_num_vec))):
		clust_num = clust_num_vec[i]
		silhouette_scores[i] = clust_num_test(spikes,clust_num,sort_data_dir)
	
	#Next pick the best number of clusters
	clust_num = clust_num_vec[np.where(silhouette_scores == np.max(silhouette_scores)[0])[0]]
	
	#Finally cluster by the best number of clusters
	neuron_spike_ind, neuron_waveform_ind = spike_clust(spikes, peak_indices, 
													 clust_num, i, sort_data_dir, 
													 axis_labels, viol_1, viol_2, 
													 type_spike, segment_times, 
													 segment_names, dig_in_times,
													 dig_in_names,sampling_rate,
													 re_sort='y')
	
	return neuron_spike_ind, neuron_waveform_ind

def spike_clust(spikes, peak_indices, clust_num, i, sort_data_dir, axis_labels, 
				viol_1, viol_2, type_spike, segment_times, segment_names, 
				dig_in_times,dig_in_names,sampling_rate,re_sort='y'):
	"""This function performs clustering on spikes pulled from each component.
	Inputs:
		spikes = list of spike samples num_spikes x length_spike
		peak_indices = indices of each spike
		clust_num = maximum number of clusters to test for clustering
		i = index of component being clustered
		sort_data_dir = directory to store images in
		axis_labels = x-labels for plotting spike samples
		viol_1 = number of indices btwn spikes for 1 ms violation
		viol_2 = number of indices btwn spikes for 2 ms violation
		type_spike = type of peak (pos or neg)
		segment_times = time of different experiment segments
		segment_names = names of different segments
		dig_in_times = array times of tastant delivery - each tastant separately
		dig_in_names = names of different tastants
		sampling_rate = number of samples per second
		re_sort = whether to re-sort if data has been previously sorted.
	Outputs:
		neuron_spike_ind = indices of spikes selected as true - the indices
						reflect the parsed data indices.
		waveform_ind = indices of waveforms, aka index in the list 'spikes'"""
	
	#Create storage folder
	sort_neur_dir = sort_data_dir + 'unit_' + str(i) + '/'
	if os.path.isdir(sort_neur_dir) == False:
		os.mkdir(sort_neur_dir)
	try:
		type_spike.split('/')[1]
		sort_neur_type_dir = sort_neur_dir + type_spike.split('/')[0] + '/'
		if os.path.isdir(sort_neur_type_dir) == False:
			os.mkdir(sort_neur_type_dir)
	except:
		dir_exists = 1
	sort_neur_type_dir = sort_neur_dir + type_spike + '/'
	if os.path.isdir(sort_neur_type_dir) == False:
		os.mkdir(sort_neur_type_dir)
	sort_neur_stats_csv = sort_neur_type_dir + 'sort_stats.csv'
	sort_neur_ind_csv = sort_neur_type_dir + 'neuron_spike_ind.csv'
	sort_neur_wav_csv = sort_neur_type_dir + 'neuron_spike_wav.csv'
	if os.path.isfile(sort_neur_ind_csv) == False:
		re_sort = 'y'
	else:
		if re_sort != 'n':
			#MODIFY TO SEARCH FOR A CSV OF SPIKE INDICES
			print('\t INPUT REQUESTED: The ' + type_spike + ' sorting has been previously performed.')
			re_sort = input('\t Would you like to re-sort [y/n]? ')
	
	if re_sort == 'y':
		#Set parameters
		viol_2_cutoff = 2 #Maximum allowed violation percentage for 2 ms
		viol_1_cutoff = 1 #Maximum allowed violation percentage for 1 ms
		num_vis = 500 #Number of waveforms to visualize for example plot
		all_dig_in_times = np.unique(np.array(dig_in_times).flatten())
		PSTH_left_ms = 500
		PSTH_right_ms = 2000
		center = np.where(axis_labels == 0)[0]
		
 		#Project data to lower dimensions
		print("\t Projecting data to lower dimensions")
		pca = PCA(n_components = 3)
		spikes_pca = pca.fit_transform(spikes)
		#Grab spike amplitude and energy of spike to add to spikes_pca
		amp_vals = np.array([np.abs(spikes[i][center]) for i in range(len(spikes))])
		energy_vals = np.expand_dims(np.sum(np.square(spikes),1),1)
		
		#Perform kmeans clustering on downsampled data
		print("\t Performing clustering of data.")
		rand_ind = np.random.randint(len(spikes),size=(100000,))
		rand_spikes = list(np.array(spikes)[rand_ind])
		print('\t Performing fitting.')
		kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(rand_spikes)
		print('\t Performing label prediction.')
		labels = kmeans.predict(spikes)
		centers = kmeans.cluster_centers_
		center_pca = pca.fit_transform(centers)
		print('\t Now testing/plotting clusters.')
		violations = []
		any_good = 0
		possible_colors = ['b','g','r','c','m','k','y','brown','pink','olive','gray'] #Colors for plotting different tastant deliveries
		clust_stats = np.zeros((clust_num,5))
		#Create cluster projection plot
		clust_fig = plt.figure(figsize=(15,15))
		ax = clust_fig.add_subplot(111, projection='3d')
		ax2 = clust_fig.add_subplot(333)
		ax3 = clust_fig.add_subplot(331)
		ax4 = clust_fig.add_subplot(339, projection='3d')
		for li in range(clust_num):
			ind_labelled = np.where(labels == li)[0]
			pca_labelled = spikes_pca[ind_labelled]
			pca_labelled_2 = spikes_pca[np.where(labels == clust_num - li)[0]]
			#3D plot
			ax.scatter(pca_labelled[:,0],pca_labelled[:,1],pca_labelled[:,2],
				  c=possible_colors[li],label='cluster '+str(li))
			#2D plot top
			ax2.scatter(pca_labelled[:,0],pca_labelled[:,1],
			   c=possible_colors[li],label='cluster '+str(li),alpha=0.2)
			#2D plot bottom
			ax3.scatter(pca_labelled_2[:,0],pca_labelled_2[:,1],
			   c=possible_colors[clust_num - li],label='cluster '+str(clust_num - li),alpha=0.2)
			#Centroid 3D plot
			ax4.scatter(center_pca[li,0],center_pca[li,1],center_pca[li,2],
			   c = possible_colors[li])
		ax.legend(loc='lower left')
		clust_fig.savefig(sort_neur_type_dir + 'cluster_projections.png', dpi=100)
		plt.close(clust_fig)
		#Create waveform/histogram/PSTH plots
		for li in range(clust_num):
			ind_labelled = np.where(labels == li)[0]
			spikes_labelled = np.array(spikes)[ind_labelled]
			#Check for violations first
			peak_ind = np.unique(np.array(peak_indices)[ind_labelled])
			peak_diff = np.subtract(peak_ind[1:-1],peak_ind[0:-2])
			viol_1_times = len(np.where(peak_diff <= viol_1)[0])
			viol_2_times = len(np.where(peak_diff <= viol_2)[0])
			viol_1_percent = round(viol_1_times/len(peak_diff)*100,2)
			viol_2_percent = round(viol_2_times/len(peak_diff)*100,2)
			violations.append([viol_1_percent,viol_2_percent])
			avg_fr = round(len(peak_ind)/(segment_times[-1])*sampling_rate,2) #in Hz
			if type_spike == 'noise_removal': #Noise removal phase needs higher cutoffs
				viol_1_cutoff = 100
				viol_2_cutoff = 100
			if viol_2_percent < viol_2_cutoff:
				if viol_1_percent < viol_1_cutoff:
					any_good += 1
					#Adding in pass test in meantime
					pass_val = (viol_2_percent < viol_2_cutoff) and (viol_1_percent < viol_1_cutoff)
					if pass_val == True:
						print("\t \t Cluster " + str(li) + " passed violation cutoffs. Now plotting.")
					else:
						print("\t \t Cluster " + str(li) + " did not pass violation cutoffs. Now plotting.")
					#Select sub-population of spikes to plot as an example
					plot_num_vis = min(num_vis,len(spikes_labelled)-1)
					plot_ind = np.random.randint(0,len(peak_ind),size=(plot_num_vis,)) #Pick 500 random waveforms to plot
					
					#Pull PSTH data by tastant
					PSTH_left = -1*round((1/1000)*sampling_rate*PSTH_left_ms)
					PSTH_right = round((1/1000)*sampling_rate*PSTH_right_ms)
					PSTH_width = PSTH_right - PSTH_left
					PSTH_avg_taste = []
					for t_i in range(len(dig_in_times)): #By tastant
						spike_raster = np.zeros((len(dig_in_times[t_i]),PSTH_width))
						for d_i in range(len(dig_in_times[t_i])): #By delivery
							d_time = dig_in_times[t_i][d_i] #Time of delivery
							spike_inds_left = np.where(peak_ind - d_time > PSTH_left)[0]
							spike_inds_right = np.where(peak_ind - d_time < PSTH_right)[0]
							overlap_inds = np.intersect1d(spike_inds_left,spike_inds_right)
							spike_inds_overlap = peak_ind[overlap_inds] - d_time + PSTH_left
							spike_raster[d_i][spike_inds_overlap] = 1
						#The following bin size and step size come from Sadacca et al. 2016 
						bin_ms = 250 #Number of ms for binning the spike counts
						bin_size = round((bin_ms/1000)*sampling_rate)
						bin_step_size_ms = 10 #Number of ms for sliding the bin over
						bin_step_size = round((bin_step_size_ms/1000)*sampling_rate)
						PSTH_x_labels = np.arange(-PSTH_left_ms,PSTH_right_ms,bin_step_size_ms)
						PSTH_mat = np.zeros((len(dig_in_times[t_i]),len(PSTH_x_labels)))
						for b_i in range(len(PSTH_x_labels)):
							PSTH_mat[:,b_i] = np.sum(spike_raster[:,b_i*bin_step_size:(b_i*bin_step_size)+bin_size],1)
						PSTH_avg = (np.mean(PSTH_mat,0)/bin_ms)*1000 #Converted to Hz
						PSTH_avg_taste.append(PSTH_avg)
					#CREATE FIGURE
					fig = plt.figure(figsize=(30,20))
					#Plot average waveform
					plt.subplot(3,3,1)
					spike_bit = spikes_labelled[np.random.randint(0,len(spikes_labelled),size=(10000,))]
					mean_bit = np.mean(spike_bit,axis=0)
					std_bit = np.std(spike_bit,axis=0)
					plt.errorbar(axis_labels,mean_bit,yerr=std_bit,xerr=None)
					plt.title('Cluster ' + str(li) + ' Average Waveform + Std Range')
					#Plot spike overlay
					plt.subplot(3,3,2)
					for si in plot_ind:
						try:
							plt.plot(axis_labels,spikes_labelled[si],'-b',alpha=0.1)
						except:
							print("\t \t Error: Skipped plotting a waveform.")
					plt.ylabel('mV')
					plt.title('Cluster ' + str(li) + ' x' + str(plot_num_vis) + ' Waveforms')
					#Find ISI distribution and plot
					plt.subplot(3,3,3)
					plt.hist(peak_diff[np.where(peak_diff < sampling_rate)[0]],bins=min(100,round(len(peak_diff)/10)))
					plt.title('Cluster ' + str(li) + ' ISI Distribution')
					#Histogram of time of spike occurrence
					plt.subplot(3,3,4)
					plt.hist(peak_ind,bins=min(100,round(len(peak_ind)/10)))
					[plt.axvline(segment_times[i],label=segment_names[i],alpha=0.2,c=possible_colors[i]) for i in range(len(segment_names))]
					plt.legend()
					plt.title('Cluster ' + str(li) + ' Spike Time Histogram')
					#Histogram of time of spike occurrence zoomed to taste delivery
					plt.subplot(3,3,5)
					plt.hist(peak_ind,bins=2*len(all_dig_in_times))
					for d_i in range(len(dig_in_names)):
						[plt.axvline(dig_in_times[d_i][i],c=possible_colors[d_i],alpha=0.2, label=dig_in_names[d_i]) for i in range(len(dig_in_times[d_i]))]
					plt.xlim((min(all_dig_in_times)- sampling_rate,max(all_dig_in_times) + sampling_rate))
					plt.title('Cluster ' + str(li) + ' Spike Time Histogram - Taste Interval')
					#Fourier Transform of Average Waveform For Cluster
					plt.subplot(3,3,6)
					fourier = rfft(mean_bit)
					freqs = fftfreq(len(mean_bit), d=1/sampling_rate)
					fourier_peaks = find_peaks(fourier)[0]
					peak_freqs = freqs[fourier_peaks]
					peak_freqs = peak_freqs[peak_freqs>0]
					plt.plot(freqs,fourier)
					for p_f in range(len(peak_freqs)):
						plt.axvline(peak_freqs[p_f],color=possible_colors[p_f],label=str(round(peak_freqs[p_f],2)))
					plt.xlim((0,max(freqs)))
					plt.legend()
					plt.xlabel('Frequency (Hz)')
					plt.title('Fourier Transform of Mean Waveform')
					#PSTH figure
					plt.subplot(3,3,7)
# 					for p_i in range(len(dig_in_times)): #Plot individual instances
# 						plt.plot(PSTH_x_labels,PSTH_mat[p_i,:],alpha=0.1)
					for d_i in range(len(dig_in_times)):
						plt.plot(PSTH_x_labels,PSTH_avg_taste[d_i],c=possible_colors[d_i],label=dig_in_names[d_i])
					plt.legend()
					plt.axvline(0,c='k')
					plt.xlabel('Milliseconds from delivery')
					plt.ylabel('Average firing rate (Hz)')
					plt.title('Cluster ' + str(li) + ' Average PSTH')
					#Title and save figure
					line_1 = 'Number of Waveforms = ' + str(len(spikes_labelled))
					line_2 = '1 ms violation percent = ' + str(viol_1_percent)
					line_3 = '2 ms violation percent = ' + str(viol_2_percent)
					line_4 = 'Average firing rate = ' + str(avg_fr)
					plt.suptitle(line_1 + '\n' + line_2 + ' ; ' + line_3 + '\n' + line_4,fontsize=24)
					fig.savefig(sort_neur_type_dir + 'waveforms_' + str(li) + '.png', dpi=100)
					plt.close(fig)
					clust_stats[li,0:4] = np.array([len(spikes_labelled),viol_1_percent,viol_2_percent,avg_fr])
		neuron_spike_ind = []
		neuron_waveform_ind = []
		if (any_good > 0) & (type_spike != 'noise_removal'): #Automatically skip over bad sorts
			print("\n \t INPUT REQUESTED: Please navigate to the directory " + sort_neur_type_dir)
			print("\t Inspect the output visuals of spike clusters, and decide which you'd like to keep.")
			keep_loop = 1
			while keep_loop == 1:
				keep_any = input("\t Would you like to keep any of the clusters as spikes (y/n)? ")
				if keep_any != 'y' and keep_any != 'n':
					print("\t Error, please enter a valid value.")
				else:
					keep_loop = 0
			if keep_any == 'y':	
				print("\n \t INPUT REQUESTED: Please enter a comma-separated list of indices you'd like to keep (ex. 0,4,6)")
				ind_good = input("\t Keep-indices: ").split(',')
			try:
				ind_good = [int(ind_good[i]) for i in range(len(ind_good))]
				clust_stats[np.array(ind_good),4] = 1
				combine_spikes = 'n'
				comb_loop = 1
				while comb_loop == 1:
					if len(ind_good) > 1:
						combine_spikes = input("\t Do any of these spikes come from the same neuron (y/n)? ")
						if combine_spikes != 'y' and combine_spikes != 'n':
							print("\t Error, please enter a valid value.")
						else:
							#Find if there are any that need to be combined into 1
							comb_loop = 0
							which_comb_loop = 1
							which_comb = []
							if combine_spikes == 'y':
								 while which_comb_loop == 1:
									 which_together = input("\t Which indices belong together [comma separated list]? ").split(',')
									 try:
										  together_ind = [int(which_together[i]) for i in range(len(which_together))]
										  which_comb.append(together_ind)
										  if len(together_ind) < len(ind_good):
											  cont_loop_2 = 1
											  while cont_loop_2 == 1:
												  continue_statement = input("\t Are there more indices which belong together (y/n)? ")
												  if continue_statement != 'y' and continue_statement != 'n':
													  print("\t Error, try again.")
												  elif continue_statement == 'y':
													  cont_loop_2 = 0
												  else:
													  cont_loop_2 = 0
													  which_comb_loop = 0
										  else:
											  which_comb_loop = 0
									 except:
										 print("Error, try again.")
								 all_to_combine = []
								 for c_i in range(len(which_comb)):
									 all_to_combine.extend(which_comb[c_i])
								 all_to_combine = np.array(all_to_combine)
								 not_to_combine = np.setdiff1d(np.array(ind_good),all_to_combine)
								 ind_good = []
								 if len(not_to_combine) > 0:
									  ind_good.append(list(not_to_combine))
								 for w_i in range(len(which_comb)):
									 ind_good.append(which_comb[w_i])
					else:
 						combine_spikes = 'y'
 						comb_loop = 0
				for ig in ind_good:
					if np.size(ig) > 1:
						peak_ind = []
						wav_ind = []
						for ind_g in ig:
							wav_ind.extend(list(np.where(labels == ind_g)[0]))
							peak_ind.extend(list(np.array(peak_indices)[np.where(labels == ind_g)[0]]))
						neuron_spike_ind.append(peak_ind)
						neuron_waveform_ind.append(wav_ind)
					else:
						wav_ind = []
						wav_ind.extend(list(np.where(labels == ig)[0]))
						neuron_waveform_ind.append(wav_ind)
						peak_ind = []
						peak_ind.extend(list(np.array(peak_indices)[np.where(labels == ig)[0]]))
						neuron_spike_ind.append(peak_ind)
			except:
				print("\t No spikes selected.")
		elif type_spike == 'noise_removal':
			ind_good = np.arange(clust_num)
			for ig in ind_good:
				neuron_spike_ind.append(list(np.array(peak_indices)[np.where(labels == ig)[0]]))
				neuron_waveform_ind.append(list(np.where(labels == ig)[0]))
		else:
			print("\t No good clusters.")
		#Save to CSV spike indices
		with open(sort_neur_ind_csv, 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows(neuron_spike_ind)
		#Save to CSV waveform indices (index within group)
		with open(sort_neur_wav_csv, 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows(neuron_waveform_ind)
		#Save stats to CSV
		with open(sort_neur_stats_csv, 'w') as f:
			write = csv.writer(f,delimiter=',')
			write.writerows([['Number of Spikes','1 ms Violations','2 ms Violations','Average Firing Rate','Good']])
			write.writerows(clust_stats)
	else:
		print("\t Importing previously sorted data.")
		with open(sort_neur_ind_csv, newline='') as f:
			reader = csv.reader(f)
			neuron_spike_ind_csv = list(reader)
		neuron_spike_ind = []
		for i_c in range(len(neuron_spike_ind_csv)):
			str_list = neuron_spike_ind_csv[i_c]
			int_list = [int(str_list[i]) for i in range(len(str_list))]
			neuron_spike_ind.append(int_list)
		with open(sort_neur_wav_csv, newline='') as f:
			reader = csv.reader(f)
			neuron_spike_wav_csv = list(reader)
		neuron_waveform_ind = []
		for i_c in range(len(neuron_spike_wav_csv)):
			str_list = neuron_spike_wav_csv[i_c]
			int_list = [int(str_list[i]) for i in range(len(str_list))]
			neuron_waveform_ind.append(int_list)
			
	return neuron_spike_ind, neuron_waveform_ind


def clust_num_test(spikes,clust_num,save_dir):
	"""This function performs silhouette score tests on different numbers of
	clusters used on spike clustering to determine the best cluster number.
	Inputs:
		- spikes: matrix with all spikes being clustered (num_spikes x num_features)
		- clust_num: vector with the number of clusters to test
		- save_dir: directory to save silhouette test figures and cluster images
	Outputs:
		- slh_avg: silhouette score for this cluster count
	"""
	
	#Set up save folder
	silh_dir = save_dir + 'silhouette_analysis/'
	if os.path.isdir(silh_dir) == False:
		os.mkdir(silh_dir)
	
	#Cluster data
	rand_ind = np.random.randint(len(spikes),size=(100000,))
	rand_spikes = list(np.array(spikes)[rand_ind])
	kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(rand_spikes)
	labels = kmeans.predict(spikes)
	centers = kmeans.cluster_centers_
	
	#Calculate silhouette scores and plot histograms
	plot_dim = int(np.round(np.sqrt(clust_num)))
	slh_avg = silhouette_score(spikes,labels)
	slh_vals = silhouette_samples(spikes,labels)
	slh_fig = plt.figure(figsize=(15,15))
	for i in range(clust_num):
		plt.subplot(plot_dim,plot_dim,i)
		slh_vals_clust_i = slh_vals[np.where(labels == i)]
		hist_slh_i = plt.hist(slh_vals_clust_i)
		plt.axvline(np.mean(slh_vals_clust_i),label='Mean')
		plt.legend()
		plt.title('Cluster #' + str(i))
		plt.xlabel('Silhouette Score Value')
		plt.ylabel('Number of Cluster Members')
	plt.suptitle('Silhouette Score Distributions by Cluster')
	slh_fig.savefig(silh_dir + 'slh_scores_clust_count_' + str(clust_num) + '.png', dpi=100)
	plt.close(slh_fig)
	
	#Plot clusters
	pca = PCA(n_components = 3)
	spikes_pca = pca.fit_transform(spikes)
	center_pca = pca.fit_transform(centers)
	possible_colors = ['b','g','r','c','m','k','y','brown','pink','olive','gray'] #Colors for plotting different tastant deliveries
	clust_fig = plt.figure(figsize=(15,15))
	ax = clust_fig.add_subplot(111, projection='3d')
	ax2 = clust_fig.add_subplot(333)
	ax3 = clust_fig.add_subplot(331)
	ax4 = clust_fig.add_subplot(339, projection='3d')
	for li in range(clust_num):
		ind_labelled = np.where(labels == li)[0]
		pca_labelled = spikes_pca[ind_labelled]
		pca_labelled_2 = spikes_pca[np.where(labels == clust_num - li)[0]]
		#3D plot
		ax.scatter(pca_labelled[:,0],pca_labelled[:,1],pca_labelled[:,2],
			  c=possible_colors[li],label='cluster '+str(li))
		#2D plot top
		ax2.scatter(pca_labelled[:,0],pca_labelled[:,1],
		   c=possible_colors[li],label='cluster '+str(li),alpha=0.2)
		#2D plot bottom
		ax3.scatter(pca_labelled_2[:,0],pca_labelled_2[:,1],
		   c=possible_colors[clust_num - li],label='cluster '+str(clust_num - li),alpha=0.2)
		#Centroid 3D plot
		ax4.scatter(center_pca[li,0],center_pca[li,1],center_pca[li,2],
		   c = possible_colors[li])
	ax.legend(loc='lower left')
	clust_fig.savefig(silh_dir + 'projections_clust_count_' + str(clust_num) + '.png', dpi=100)
	plt.close(clust_fig)
	
	return slh_avg
	
	
	
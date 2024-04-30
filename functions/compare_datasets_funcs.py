#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:33:13 2024

@author: Hannah Germaine

A collection of functions for comparing different datasets against each other 
in their correlation trends.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ks_2samp, ttest_ind
from scipy.signal import savgol_filter
import warnings

def cross_segment_diffs(corr_data,save_dir,unique_given_names,unique_corr_names,\
			  unique_segment_names,unique_taste_names):
	"""This function collects statistics across different correlation types and
	plots them together
	INPUTS:
		- corr_data: dictionary containing correlation data across conditions.
			length = number of datasets
			corr_data[name] = dictionary of dataset data
			corr_data[name]['corr_data'] = dict of length #correlation types
			corr_data[name]['corr_data'][corr_name] = dict of length #segments
			corr_data[name]['corr_data'][corr_name][seg_name] = dict of length #tastes
			corr_data[name]['corr_data'][corr_name][seg_name][taste_name]['data'] = numpy 
				array of population average vector correlations [num_dev x num_trials x num_epochs]
		- results_dir: directory to save the resulting plots
		- unique_corr_names: unique names of correlation analyses to compare
	OUTPUTS: plots and statistical significance tests
	"""
	#Set parameters
	warnings.filterwarnings('ignore')
	bin_edges = np.linspace(0,1,1001)
	bin_x_vals = np.arange(0,1,1/1000)
	bin_width = bin_edges[1] - bin_edges[0]
	
	#Create further save dirs
	true_save = os.path.join(save_dir,'True')
	if not os.path.isdir(true_save):
		os.mkdir(true_save)
	norm_save = os.path.join(save_dir,'Normalized')
	if not os.path.isdir(norm_save):
		os.mkdir(norm_save)
	mean_save = os.path.join(save_dir,'Mean')
	if not os.path.isdir(mean_save):
		os.mkdir(mean_save)
	mean_norm_save = os.path.join(save_dir,'Mean_Normalized')
	if not os.path.isdir(mean_norm_save):
		os.mkdir(mean_norm_save)
	
	class cross_segment_attributes:
		def __init__(self,combo,names,i_1,i_2,i_3,unique_corr_names,\
				 unique_taste_names,unique_epochs):
			setattr(self, names[0], eval(combo[0])[i_1])
			setattr(self, names[1], eval(combo[1])[i_2])
			setattr(self, names[2], eval(combo[2])[i_3])
	
	#_____Reorganize data by unique correlation type_____
	unique_data_dict = dict()
	for ucn in unique_corr_names:
		unique_data_dict[ucn] = dict()
		for udn in unique_given_names:
			unique_data_dict[ucn][udn] = dict()
			for usn in unique_segment_names:
				unique_data_dict[ucn][udn][usn] = dict()
				for utn in unique_taste_names:
					unique_data_dict[ucn][udn][usn][utn] = dict()
	max_epochs = 0
	for d_i in corr_data: #Each dataset
		dataset = corr_data[d_i]
		given_name = d_i
		dataset_corr_data = dataset['corr_data']
		corr_names = list(dataset_corr_data.keys())
		for cn_i in corr_names:
			corr_name = cn_i
			corr_dev_stats = dataset_corr_data[cn_i]
			seg_names = list(corr_dev_stats.keys())
			for seg_name in seg_names:
				taste_names = list(corr_dev_stats[seg_name].keys())
				for taste_name in taste_names:
					data = corr_dev_stats[seg_name][taste_name]['data']
					num_epochs = data.shape[-1]
					if num_epochs > max_epochs:
						max_epochs = num_epochs
					unique_data_dict[corr_name][given_name][seg_name][taste_name]['data'] = data.reshape(-1,data.shape[-1])
	
	#Plot all combinations
	unique_epochs = np.arange(max_epochs)
	characteristic_list = ['unique_corr_names','unique_taste_names','unique_epochs']
	characteristic_dict = dict()
	for cl in characteristic_list:
		characteristic_dict[cl] = eval(cl)
	name_list = ['corr_name','taste_name','e_i']
	#Get attribute pairs for plotting views
	all_combinations = list(combinations(characteristic_list,2))
	all_combinations_full = []
	all_names_full = []
	for ac in all_combinations:
		ac_list = list(ac)
		missing = np.setdiff1d(characteristic_list,ac_list)
		full_combo = ac_list
		full_combo.extend(missing)
		all_combinations_full.append(full_combo)
		names_combo = [name_list[characteristic_list.index(c)] for c in full_combo]
		all_names_full.append(names_combo)
	#Get segment pairs for comparison
	segment_combinations = list(combinations(unique_segment_names,2))
	
	for c_i in range(len(all_combinations_full)):
		combo = all_combinations_full[c_i]
		combo_lengths = [len(characteristic_dict[combo[i]]) for i in range(len(combo))]
		names = all_names_full[c_i]
		for i_1 in range(combo_lengths[0]):
			combo_1 = eval(combo[0])[i_1]
			if type(combo_1) == np.int64:
				combo_1 = "epoch_" + str(combo_1)
			for i_2 in range(combo_lengths[1]):
				combo_2 = eval(combo[1])[i_2]
				if type(combo_2) == np.int64:
					combo_2 = "epoch_" + str(combo_2)
				f, ax = plt.subplots(nrows = combo_lengths[2], ncols = len(segment_combinations), figsize=(len(segment_combinations)*5,combo_lengths[2]*5))
				f_norm, ax_norm = plt.subplots(nrows = combo_lengths[2], ncols = len(segment_combinations), figsize=(len(segment_combinations)*5,combo_lengths[2]*5))
				f_mean, ax_mean = plt.subplots(nrows = combo_lengths[2], ncols = len(segment_combinations), figsize=(len(segment_combinations)*5,combo_lengths[2]*5))
				f_mean_norm, ax_mean_norm = plt.subplots(nrows = combo_lengths[2], ncols = len(segment_combinations), figsize=(len(segment_combinations)*5,combo_lengths[2]*5))
				max_diff = 0
				min_diff = 0
				for i_3 in range(combo_lengths[2]):
					xlabel = eval(combo[2])[i_3]
					if type(xlabel) == np.int64:
						xlabel = "epoch_" + str(xlabel)
					#Begin pulling data
					att = cross_segment_attributes(combo,names,i_1,i_2,i_3,unique_corr_names,\
							 unique_taste_names, unique_epochs)
					corr_name = att.corr_name
					taste_name = att.taste_name
					e_i = att.e_i
					#Pit segment pairs against each other
					for sp_i, sp in enumerate(segment_combinations):
						seg_1 = sp[0]
						seg_2 = sp[1]
						cum_dist_collection = []
						cum_dist_labels = []
						counter = 0
						for g_n in unique_given_names:
							try:
								data_1 = unique_data_dict[corr_name][g_n][seg_1][taste_name]['data'][:,e_i]
								counts_1, _ = np.histogram(data_1,bin_edges)
								counts_1_smooth = savgol_filter(counts_1,np.ceil(len(data_1)/100).astype('int'),polyorder=1)
								counts_1_smooth_density = counts_1_smooth/np.sum(counts_1_smooth)
								data_2 = unique_data_dict[corr_name][g_n][seg_2][taste_name]['data'][:,e_i]
								counts_2, _ = np.histogram(data_2,bin_edges)
								counts_2_smooth = savgol_filter(counts_2,np.ceil(len(data_2)/100).astype('int'),polyorder=1)
								counts_2_smooth_density = counts_2_smooth/np.sum(counts_2_smooth)
								cum_dist_collection.append(counts_2_smooth_density-counts_1_smooth_density)
								cum_dist_labels.append([g_n + '(' + str(counter) + ')'])
								counter += 1
							except:
								print("\tSkipping invalid dataset.")
						try:
							mean_cum_dist = np.nanmean(np.array(cum_dist_collection),0)
							std_cum_dist = np.nanstd(np.array(cum_dist_collection),0)
							half_means = np.concatenate((np.mean(mean_cum_dist[:500])*np.ones(500),np.mean(mean_cum_dist[500:])*np.ones(500)))
							max_cum_dist = np.max(np.array(cum_dist_collection))
							if max_cum_dist > max_diff:
								max_diff = max_cum_dist
							min_cum_dist = np.min(np.array(cum_dist_collection))
							if min_cum_dist < min_diff:
								min_diff = min_cum_dist
						except:
							max_cum_dist = 0
						title = seg_2 + ' - ' + seg_1
						#Plot distribution differences as is
						ax[i_3,sp_i].axhline(0,label='_',alpha=0.2,color='k')
						ax[i_3,sp_i].axvline(0.5,label='_',alpha=0.2,color='k')
						for c_i in range(len(cum_dist_collection)):
							ax[i_3,sp_i].plot(bin_x_vals,cum_dist_collection[c_i],label=cum_dist_labels[c_i],alpha=0.75)
						try:
							#Create std shading
							std_y_low = mean_cum_dist-std_cum_dist
							std_y_high = mean_cum_dist+std_cum_dist
							ax[i_3,sp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							#Plot mean of all curves
							ax[i_3,sp_i].plot(bin_x_vals,mean_cum_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax[i_3,sp_i].plot(bin_x_vals,half_means,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
							
							ax_mean[i_3,sp_i].axhline(0,label='_',alpha=0.2,color='k')
							ax_mean[i_3,sp_i].axvline(0.5,label='_',alpha=0.2,color='k')
							ax_mean[i_3,sp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							ax_mean[i_3,sp_i].plot(bin_x_vals,mean_cum_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax_mean[i_3,sp_i].plot(bin_x_vals,half_means,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
						except:
							print("\tNo Mean Plotted.")
						ax[i_3,sp_i].legend(fontsize='8', loc ='lower left')
						ax[i_3,sp_i].set_xlim([0,1.1])
						ax[i_3,sp_i].set_xlabel(xlabel)
						ax[i_3,sp_i].set_title(title)
						#Plot normalized distribution differences: |max diff| = 1
						ax_norm[i_3,sp_i].axhline(0,label='_',alpha=0.2,color='k')
						ax_norm[i_3,sp_i].axvline(0.5,label='_',alpha=0.2,color='k')
						try:
							norm_dist = np.array(cum_dist_collection)
							norm_dist = norm_dist/(np.expand_dims(np.nanmax(np.abs(cum_dist_collection),1),1)*np.ones(np.shape(norm_dist)))
							for c_i in range(len(cum_dist_collection)):
								ax_norm[i_3,sp_i].plot(bin_x_vals,norm_dist[c_i,:],label=cum_dist_labels[c_i],alpha=0.75)
							try:
								mean_norm_dist = np.nanmean(norm_dist,0)
								std_norm_dist = np.nanstd(norm_dist,0)
								half_means_norm = np.concatenate((np.nanmean(mean_norm_dist[:500])*np.ones(500),np.nanmean(mean_norm_dist[500:])*np.ones(500)))
								#Create std shading
								std_y_low = mean_norm_dist-std_norm_dist
								std_y_high = mean_norm_dist+std_norm_dist
								ax_norm[i_3,sp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
								#Plot mean of all curves
								ax_norm[i_3,sp_i].plot(bin_x_vals,mean_norm_dist,linewidth=3, linestyle='--', color='k',label='Mean')
								ax_norm[i_3,sp_i].plot(bin_x_vals,half_means_norm,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
								
								ax_mean_norm[i_3,sp_i].axhline(0,label='_',alpha=0.2,color='k')
								ax_mean_norm[i_3,sp_i].axvline(0.5,label='_',alpha=0.2,color='k')
								ax_mean_norm[i_3,sp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
								ax_mean_norm[i_3,sp_i].plot(bin_x_vals,mean_norm_dist,linewidth=3, linestyle='--', color='k',label='Mean')
								ax_mean_norm[i_3,sp_i].plot(bin_x_vals,half_means_norm,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
							except:
								print("\tNo Mean Plotted.")
						except:
							print("\tNo Norm Calculated.")
						ax_norm[i_3,sp_i].legend(fontsize='8', loc ='lower left')
						ax_norm[i_3,sp_i].set_xlim([0,1.1])
						ax_norm[i_3,sp_i].set_xlabel(xlabel)
						ax_norm[i_3,sp_i].set_title(title)
						#Mean plots cleanups
						ax_mean[i_3,sp_i].legend(fontsize='8', loc ='lower left')
						ax_mean[i_3,sp_i].set_xlim([0,1.1])
						ax_mean[i_3,sp_i].set_xlabel(xlabel)
						ax_mean[i_3,sp_i].set_title(title)
						ax_mean_norm[i_3,sp_i].legend(fontsize='8', loc ='lower left')
						ax_mean_norm[i_3,sp_i].set_xlim([0,1.1])
						ax_mean_norm[i_3,sp_i].set_xlabel(xlabel)
						ax_mean_norm[i_3,sp_i].set_title(title)
					ax[i_3,0].set_ylabel('Density Difference')
					ax_norm[i_3,0].set_ylabel('Normalized Density Difference')
					ax_mean[i_3,0].set_ylabel('Density Difference')
					ax_mean_norm[i_3,0].set_ylabel('Normalized Density Difference')
				for i_3 in range(combo_lengths[2]):
					for sp_i in range(len(segment_combinations)):
						ax[i_3,sp_i].set_ylim([min_diff,max_diff])
						ax_norm[i_3,sp_i].set_ylim([-1.01,1.01])
						ax_mean[i_3,sp_i].set_ylim([min_diff,max_diff])
						ax_mean_norm[i_3,sp_i].set_ylim([-1.01,1.01])
				#Finish plots with titles and save
				f.suptitle('Cumulative Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_')
				f.savefig(os.path.join(true_save,f_pop_vec_plot_name) + '.png')
				f.savefig(os.path.join(true_save,f_pop_vec_plot_name) + '.svg')
				plt.close(f) 
				
				f_norm.suptitle('Normalized Cumulative Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_norm.tight_layout()
				f_norm.savefig(os.path.join(norm_save,f_pop_vec_plot_name) + '_norm.png')
				f_norm.savefig(os.path.join(norm_save,f_pop_vec_plot_name) + '_norm.svg')
				plt.close(f_norm) 
				
				f_mean.suptitle('Mean Cumulative Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean.tight_layout()
				f_mean.savefig(os.path.join(mean_save,f_pop_vec_plot_name) + '_mean.png')
				f_mean.savefig(os.path.join(mean_save,f_pop_vec_plot_name) + '_mean.svg')
				plt.close(f_mean) 
				
				f_mean_norm.suptitle('Normalized Mean Cumulative Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean_norm.tight_layout()
				f_mean_norm.savefig(os.path.join(mean_norm_save,f_pop_vec_plot_name) + '_mean_norm.png')
				f_mean_norm.savefig(os.path.join(mean_norm_save,f_pop_vec_plot_name) + '_mean_norm.svg')
				plt.close(f_mean_norm)


def cross_taste_diffs(corr_data,save_dir,unique_given_names,unique_corr_names,\
			  unique_segment_names,unique_taste_names):
	"""This function collects statistics across different correlation types and
	plots them together
	INPUTS:
		- corr_data: dictionary containing correlation data across conditions.
			length = number of datasets
			corr_data[name] = dictionary of dataset data
			corr_data[name]['corr_data'] = dict of length #correlation types
			corr_data[name]['corr_data'][corr_name] = dict of length #segments
			corr_data[name]['corr_data'][corr_name][seg_name] = dict of length #tastes
			corr_data[name]['corr_data'][corr_name][seg_name][taste_name]['data'] = numpy 
				array of population average vector correlations [num_dev x num_trials x num_epochs]
		- results_dir: directory to save the resulting plots
		- unique_corr_names: unique names of correlation analyses to compare
	OUTPUTS: plots and statistical significance tests
	"""
	#Set parameters
	warnings.filterwarnings('ignore')
	bin_edges = np.linspace(0,1,1001)
	bin_x_vals = np.arange(0,1,1/1000)
	bin_width = bin_edges[1] - bin_edges[0]
	
	#Create further save dirs
	true_save = os.path.join(save_dir,'True')
	if not os.path.isdir(true_save):
		os.mkdir(true_save)
	norm_save = os.path.join(save_dir,'Normalized')
	if not os.path.isdir(norm_save):
		os.mkdir(norm_save)
	mean_save = os.path.join(save_dir,'Mean')
	if not os.path.isdir(mean_save):
		os.mkdir(mean_save)
	mean_norm_save = os.path.join(save_dir,'Mean_Normalized')
	if not os.path.isdir(mean_norm_save):
		os.mkdir(mean_norm_save)
	
	class cross_taste_attributes:
		def __init__(self,combo,names,i_1,i_2,i_3,unique_corr_names,\
				 unique_segment_names,unique_epochs):
			setattr(self, names[0], eval(combo[0])[i_1])
			setattr(self, names[1], eval(combo[1])[i_2])
			setattr(self, names[2], eval(combo[2])[i_3])
	
	#_____Reorganize data by unique correlation type_____
	unique_data_dict = dict()
	for ucn in unique_corr_names:
		unique_data_dict[ucn] = dict()
		for udn in unique_given_names:
			unique_data_dict[ucn][udn] = dict()
			for usn in unique_segment_names:
				unique_data_dict[ucn][udn][usn] = dict()
				for utn in unique_taste_names:
					unique_data_dict[ucn][udn][usn][utn] = dict()
	max_epochs = 0
	for d_i in corr_data: #Each dataset
		dataset = corr_data[d_i]
		given_name = d_i
		dataset_corr_data = dataset['corr_data']
		corr_names = list(dataset_corr_data.keys())
		for cn_i in corr_names:
			corr_name = cn_i
			corr_dev_stats = dataset_corr_data[cn_i]
			seg_names = list(corr_dev_stats.keys())
			for seg_name in seg_names:
				taste_names = list(corr_dev_stats[seg_name].keys())
				for taste_name in taste_names:
					data = corr_dev_stats[seg_name][taste_name]['data']
					num_epochs = data.shape[-1]
					if num_epochs > max_epochs:
						max_epochs = num_epochs
					unique_data_dict[corr_name][given_name][seg_name][taste_name]['data'] = data.reshape(-1,data.shape[-1])
	
	#Plot all combinations
	unique_epochs = np.arange(max_epochs)
	characteristic_list = ['unique_corr_names','unique_segment_names','unique_epochs']
	characteristic_dict = dict()
	for cl in characteristic_list:
		characteristic_dict[cl] = eval(cl)
	name_list = ['corr_name','seg_name','e_i']
	#Get attribute pairs for plotting views
	all_combinations = list(combinations(characteristic_list,2))
	all_combinations_full = []
	all_names_full = []
	for ac in all_combinations:
		ac_list = list(ac)
		missing = np.setdiff1d(characteristic_list,ac_list)
		full_combo = ac_list
		full_combo.extend(missing)
		all_combinations_full.append(full_combo)
		names_combo = [name_list[characteristic_list.index(c)] for c in full_combo]
		all_names_full.append(names_combo)
	#Get segment pairs for comparison
	taste_combinations = list(combinations(unique_taste_names,2))
	
	for c_i in range(len(all_combinations_full)):
		combo = all_combinations_full[c_i]
		combo_lengths = [len(characteristic_dict[combo[i]]) for i in range(len(combo))]
		names = all_names_full[c_i]
		for i_1 in range(combo_lengths[0]):
			combo_1 = eval(combo[0])[i_1]
			if type(combo_1) == np.int64:
				combo_1 = "epoch_" + str(combo_1)
			for i_2 in range(combo_lengths[1]):
				combo_2 = eval(combo[1])[i_2]
				if type(combo_2) == np.int64:
					combo_2 = "epoch_" + str(combo_2)
				f, ax = plt.subplots(nrows = combo_lengths[2], ncols = len(taste_combinations), figsize=(len(taste_combinations)*5,combo_lengths[2]*5))
				f_norm, ax_norm = plt.subplots(nrows = combo_lengths[2], ncols = len(taste_combinations), figsize=(len(taste_combinations)*5,combo_lengths[2]*5))
				f_mean, ax_mean = plt.subplots(nrows = combo_lengths[2], ncols = len(taste_combinations), figsize=(len(taste_combinations)*5,combo_lengths[2]*5))
				f_mean_norm, ax_mean_norm = plt.subplots(nrows = combo_lengths[2], ncols = len(taste_combinations), figsize=(len(taste_combinations)*5,combo_lengths[2]*5))
				max_diff = 0
				min_diff = 0
				for i_3 in range(combo_lengths[2]):
					xlabel = eval(combo[2])[i_3]
					if type(xlabel) == np.int64:
						xlabel = "epoch_" + str(xlabel)
					#Begin pulling data
					att = cross_taste_attributes(combo,names,i_1,i_2,i_3,unique_corr_names,\
							 unique_segment_names, unique_epochs)
					corr_name = att.corr_name
					seg_name = att.seg_name
					e_i = att.e_i
					#Pit segment pairs against each other
					for tp_i, tp in enumerate(taste_combinations):
						taste_1 = tp[0]
						taste_2 = tp[1]
						cum_dist_collection = []
						cum_dist_labels = []
						counter = 0
						for g_n in unique_given_names:
							try:
								data_1 = unique_data_dict[corr_name][g_n][seg_name][taste_1]['data'][:,e_i]
								counts_1, _ = np.histogram(data_1,bin_edges)
								counts_1_smooth = savgol_filter(counts_1,np.ceil(len(data_1)/100).astype('int'),polyorder=1)
								counts_1_smooth_density = counts_1_smooth/np.sum(counts_1_smooth)
								data_2 = unique_data_dict[corr_name][g_n][seg_name][taste_2]['data'][:,e_i]
								counts_2, _ = np.histogram(data_2,bin_edges)
								counts_2_smooth = savgol_filter(counts_2,np.ceil(len(data_2)/100).astype('int'),polyorder=1)
								counts_2_smooth_density = counts_2_smooth/np.sum(counts_2_smooth)
								cum_dist_collection.append(counts_2_smooth_density-counts_1_smooth_density)
								cum_dist_labels.append([g_n + '(' + str(counter) + ')'])
								counter += 1
							except:
								print("\tSkipping invalid dataset.")
						try:
							mean_cum_dist = np.nanmean(np.array(cum_dist_collection),0)
							std_cum_dist = np.nanstd(np.array(cum_dist_collection),0)
							half_means = np.concatenate((np.mean(mean_cum_dist[:500])*np.ones(500),np.mean(mean_cum_dist[500:])*np.ones(500)))
							max_cum_dist = np.max(np.array(cum_dist_collection))
							if max_cum_dist > max_diff:
								max_diff = max_cum_dist
							min_cum_dist = np.min(np.array(cum_dist_collection))
							if min_cum_dist < min_diff:
								min_diff = min_cum_dist
						except:
							max_cum_dist = 0
						title = taste_2 + ' - ' + taste_1
						#Plot distribution differences as is
						ax[i_3,tp_i].axhline(0,label='_',alpha=0.2,color='k')
						ax[i_3,tp_i].axvline(0.5,label='_',alpha=0.2,color='k')
						for c_i in range(len(cum_dist_collection)):
							ax[i_3,tp_i].plot(bin_x_vals,cum_dist_collection[c_i],label=cum_dist_labels[c_i],alpha=0.75)
						try:
							#Create std shading
							std_y_low = mean_cum_dist-std_cum_dist
							std_y_high = mean_cum_dist+std_cum_dist
							ax[i_3,tp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							#Plot mean of all curves
							ax[i_3,tp_i].plot(bin_x_vals,mean_cum_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax[i_3,tp_i].plot(bin_x_vals,half_means,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
							
							ax_mean[i_3,tp_i].axhline(0,label='_',alpha=0.2,color='k')
							ax_mean[i_3,tp_i].axvline(0.5,label='_',alpha=0.2,color='k')
							ax_mean[i_3,tp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							ax_mean[i_3,tp_i].plot(bin_x_vals,mean_cum_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax_mean[i_3,tp_i].plot(bin_x_vals,half_means,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
						except:
							print("\tNo Mean Plotted.")
						ax[i_3,tp_i].legend(fontsize='8', loc ='lower left')
						ax[i_3,tp_i].set_xlim([0,1.1])
						ax[i_3,tp_i].set_xlabel(xlabel)
						ax[i_3,tp_i].set_title(title)
						#Plot normalized distribution differences: |max diff| = 1
						ax_norm[i_3,tp_i].axhline(0,label='_',alpha=0.2,color='k')
						ax_norm[i_3,tp_i].axvline(0.5,label='_',alpha=0.2,color='k')
						try:
							norm_dist = np.array(cum_dist_collection)
							norm_dist = norm_dist/(np.expand_dims(np.nanmax(np.abs(cum_dist_collection),1),1)*np.ones(np.shape(norm_dist)))
							for c_i in range(len(cum_dist_collection)):
								ax_norm[i_3,tp_i].plot(bin_x_vals,norm_dist[c_i,:],label=cum_dist_labels[c_i],alpha=0.75)
							try:
								mean_norm_dist = np.nanmean(norm_dist,0)
								std_norm_dist = np.nanstd(norm_dist,0)
								half_means_norm = np.concatenate((np.nanmean(mean_norm_dist[:500])*np.ones(500),np.nanmean(mean_norm_dist[500:])*np.ones(500)))
								#Create std shading
								std_y_low = mean_norm_dist-std_norm_dist
								std_y_high = mean_norm_dist+std_norm_dist
								ax_norm[i_3,tp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
								#Plot mean of all curves
								ax_norm[i_3,tp_i].plot(bin_x_vals,mean_norm_dist,linewidth=3, linestyle='--', color='k',label='Mean')
								ax_norm[i_3,tp_i].plot(bin_x_vals,half_means_norm,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
								
								ax_mean_norm[i_3,tp_i].axhline(0,label='_',alpha=0.2,color='k')
								ax_mean_norm[i_3,tp_i].axvline(0.5,label='_',alpha=0.2,color='k')
								ax_mean_norm[i_3,tp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
								ax_mean_norm[i_3,tp_i].plot(bin_x_vals,mean_norm_dist,linewidth=3, linestyle='--', color='k',label='Mean')
								ax_mean_norm[i_3,tp_i].plot(bin_x_vals,half_means_norm,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
							except:
								print("\tNo Mean Plotted.")
						except:
							print("\tNo Norm Calculated.")
						ax_norm[i_3,tp_i].legend(fontsize='8', loc ='lower left')
						ax_norm[i_3,tp_i].set_xlim([0,1.1])
						ax_norm[i_3,tp_i].set_xlabel(xlabel)
						ax_norm[i_3,tp_i].set_title(title)
						#Mean plots cleanups
						ax_mean[i_3,tp_i].legend(fontsize='8', loc ='lower left')
						ax_mean[i_3,tp_i].set_xlim([0,1.1])
						ax_mean[i_3,tp_i].set_xlabel(xlabel)
						ax_mean[i_3,tp_i].set_title(title)
						ax_mean_norm[i_3,tp_i].legend(fontsize='8', loc ='lower left')
						ax_mean_norm[i_3,tp_i].set_xlim([0,1.1])
						ax_mean_norm[i_3,tp_i].set_xlabel(xlabel)
						ax_mean_norm[i_3,tp_i].set_title(title)
					ax[i_3,0].set_ylabel('Density Difference')
					ax_norm[i_3,0].set_ylabel('Normalized Density Difference')
					ax_mean[i_3,0].set_ylabel('Density Difference')
					ax_mean_norm[i_3,0].set_ylabel('Normalized Density Difference')
				for i_3 in range(combo_lengths[2]):
					for tp_i in range(len(taste_combinations)):
						ax[i_3,tp_i].set_ylim([min_diff,max_diff])
						ax_norm[i_3,tp_i].set_ylim([-1.01,1.01])
						ax_mean[i_3,tp_i].set_ylim([min_diff,max_diff])
						ax_mean_norm[i_3,tp_i].set_ylim([-1.01,1.01])
				#Finish plots with titles and save
				f.suptitle('Cumulative Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_')
				f.savefig(os.path.join(true_save,f_pop_vec_plot_name) + '.png')
				f.savefig(os.path.join(true_save,f_pop_vec_plot_name) + '.svg')
				plt.close(f) 
				
				f_norm.suptitle('Normalized Cumulative Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_norm.tight_layout()
				f_norm.savefig(os.path.join(norm_save,f_pop_vec_plot_name) + '_norm.png')
				f_norm.savefig(os.path.join(norm_save,f_pop_vec_plot_name) + '_norm.svg')
				plt.close(f_norm) 
				
				f_mean.suptitle('Mean Cumulative Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean.tight_layout()
				f_mean.savefig(os.path.join(mean_save,f_pop_vec_plot_name) + '_mean.png')
				f_mean.savefig(os.path.join(mean_save,f_pop_vec_plot_name) + '_mean.svg')
				plt.close(f_mean) 
				
				f_mean_norm.suptitle('Normalized Mean Cumulative Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean_norm.tight_layout()
				f_mean_norm.savefig(os.path.join(mean_norm_save,f_pop_vec_plot_name) + '_mean_norm.png')
				f_mean_norm.savefig(os.path.join(mean_norm_save,f_pop_vec_plot_name) + '_mean_norm.svg')
				plt.close(f_mean_norm)





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
	mean_diff_save = os.path.join(save_dir,'Corr_Mean_Diffs')
	if not os.path.isdir(mean_diff_save):
		os.mkdir(mean_diff_save)
	
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
				f_mean_diff, ax_mean_diff = plt.subplots(ncols = combo_lengths[2], nrows = 2, figsize=(combo_lengths[2]*5,2*5))
				max_diff = 0
				min_diff = 0
				max_mean_diff = 0
				min_mean_diff = 0
				significance_storage = dict()
				for i_3 in range(combo_lengths[2]):
					significance_storage[i_3] = dict()
					max_mean_diff_i = 0
					min_mean_diff_i = 0
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
					mean_diff_collection = dict()
					mean_diff_labels = []
					for sp_i, sp in enumerate(segment_combinations):
						seg_1 = sp[0]
						seg_2 = sp[1]
						title = seg_2 + ' - ' + seg_1
						mean_diffs = []
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
								mean_diffs.extend([np.nanmean(data_2) - np.nanmean(data_1)])
								counter += 1
							except:
								print("\tSkipping invalid dataset.")
						cum_dist_collection_array = np.array(cum_dist_collection)
						num_cum_dist = len(cum_dist_labels)
						#Plot distribution differences as is
						ax[i_3,sp_i].axhline(0,label='_',alpha=0.2,color='k')
						ax[i_3,sp_i].axvline(0.5,label='_',alpha=0.2,color='k')
						ax_mean[i_3,sp_i].axhline(0,label='_',alpha=0.2,color='k')
						ax_mean[i_3,sp_i].axvline(0.5,label='_',alpha=0.2,color='k')
						for c_i in range(num_cum_dist):
							ax[i_3,sp_i].plot(bin_x_vals,cum_dist_collection[c_i],label=cum_dist_labels[c_i],alpha=0.75)
						#Collect mean distribution differences
						mean_diff_collection[sp_i] = dict()
						mean_diff_collection[sp_i]['data'] = mean_diffs
						mean_diff_collection[sp_i]['labels'] = cum_dist_labels
						mean_diff_labels.append(seg_2 + ' - ' + seg_1)
						if num_cum_dist >= 1:
							mean_cum_dist = np.nanmean(cum_dist_collection_array,0)
							std_cum_dist = np.nanstd(cum_dist_collection_array,0)
							half_means = np.concatenate((np.mean(mean_cum_dist[:500])*np.ones(500),np.mean(mean_cum_dist[500:])*np.ones(500)))
							max_cum_dist = np.nanmax(cum_dist_collection_array)
							if max_cum_dist > max_diff:
								max_diff = max_cum_dist
							min_cum_dist = np.nanmin(cum_dist_collection_array)
							if min_cum_dist < min_diff:
								min_diff = min_cum_dist
							std_y_low = mean_cum_dist-std_cum_dist
							std_y_high = mean_cum_dist+std_cum_dist
							#Create std shading
							ax[i_3,sp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							#Plot mean of all curves
							ax[i_3,sp_i].plot(bin_x_vals,mean_cum_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax[i_3,sp_i].plot(bin_x_vals,half_means,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
							#Plot mean on separate axes
							ax_mean[i_3,sp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							ax_mean[i_3,sp_i].plot(bin_x_vals,mean_cum_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax_mean[i_3,sp_i].plot(bin_x_vals,half_means,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
						ax[i_3,sp_i].legend(fontsize='8', loc ='lower left')
						ax[i_3,sp_i].set_xlim([0,1.1])
						ax[i_3,sp_i].set_xlabel(xlabel)
						ax[i_3,sp_i].set_title(title)
						#Plot normalized distribution differences: |max diff| = 1
						ax_norm[i_3,sp_i].axhline(0,label='_',alpha=0.2,color='k')
						ax_norm[i_3,sp_i].axvline(0.5,label='_',alpha=0.2,color='k')
						if num_cum_dist > 0:
							try:
								norm_dist = cum_dist_collection_array/(np.expand_dims(np.nanmax(np.abs(cum_dist_collection),1),1)*np.ones(np.shape(cum_dist_collection_array)))
							except:
								norm_dist = cum_dist_collection_array/(np.nanmax(np.abs(cum_dist_collection))*np.ones(np.shape(cum_dist_collection_array)))
						for c_i in range(num_cum_dist):
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
							#Plot mean to separate axes
							ax_mean_norm[i_3,sp_i].axhline(0,label='_',alpha=0.2,color='k')
							ax_mean_norm[i_3,sp_i].axvline(0.5,label='_',alpha=0.2,color='k')
							ax_mean_norm[i_3,sp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							ax_mean_norm[i_3,sp_i].plot(bin_x_vals,mean_norm_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax_mean_norm[i_3,sp_i].plot(bin_x_vals,half_means_norm,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
						except:
							print("\tNo Mean Plotted.")
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
						#Calculate distribution differences significance above 0 using percentile
						true_sig_bins = np.zeros(len(bin_x_vals)+2)
						norm_sig_bins = np.zeros(len(bin_x_vals)+2)
						if num_cum_dist > 1: #Hopefully that means that there are far more than 1, or else this is silly
							for b_i, b_val in enumerate(bin_x_vals):
								#True data
								bin_dist = cum_dist_collection_array[:,b_i]
								cutoff_percentile = np.percentile(bin_dist,5)
								if 0 < cutoff_percentile:
									true_sig_bins[b_i+1] = 1
								#Normalized data
								bin_dist = norm_dist[:,b_i]
								cutoff_percentile = np.percentile(bin_dist,5)
								if 0 < cutoff_percentile:
									norm_sig_bins[b_i+1] = 1
						#Plot significance intervals as yellow shaded vertical bars
						true_bin_starts = np.where(np.diff(true_sig_bins) == 1)[0]
						true_bin_ends = np.where(np.diff(true_sig_bins) == -1)[0] - 1
						norm_bin_starts = np.where(np.diff(norm_sig_bins) == 1)[0]
						norm_bin_ends = np.where(np.diff(norm_sig_bins) == -1)[0] - 1
						if len(true_bin_starts) > 0:
							for tb_i in range(len(true_bin_starts)):
								start_i = bin_x_vals[true_bin_starts[tb_i]]
								end_i = bin_x_vals[true_bin_ends[tb_i]]
								ax[i_3,sp_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
								ax_mean[i_3,sp_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
						if len(norm_bin_starts) > 0:
							for nb_i in range(len(norm_bin_starts)):
								start_i = bin_x_vals[norm_bin_starts[nb_i]]
								end_i = bin_x_vals[norm_bin_ends[nb_i]]
								ax_norm[i_3,sp_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
								ax_mean_norm[i_3,sp_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
					#Plot box plots and trends of mean correlation differences with significance
					ax_mean_diff[0,i_3].axhline(0,label='_',alpha=0.2,color='k',linestyle='dashed')
					points_boxplot = []
					for m_i in range(len(mean_diff_collection)):
						points = mean_diff_collection[m_i]['data']
						if len(points)>0:
							if np.max(points) > max_mean_diff_i:
								max_mean_diff_i = np.max(points)
							if np.min(points) < min_mean_diff_i:
								min_mean_diff_i = np.min(points)
							ax_mean_diff[0,i_3].scatter(np.random.normal(m_i+1,0.04,size=len(points)),points,color='g',alpha=0.2)
						points_boxplot.append(list(points))
					ax_mean_diff[0,i_3].boxplot(points_boxplot,sym='',medianprops=dict(linestyle='-',color='blue'))
					if len(points_boxplot) > 0:
						if max_mean_diff < max_mean_diff_i:
							max_mean_diff = max_mean_diff_i
						if min_mean_diff > min_mean_diff_i:
							min_mean_diff = min_mean_diff_i
					#Now plot the points by animal as lines
					ax_mean_diff[1,i_3].axhline(0,label='_',alpha=0.2,color='k',linestyle='dashed')
					all_data_labels = []
					for m_i in range(len(mean_diff_collection)):
						all_data_labels.extend(mean_diff_collection[m_i]['labels'])
					unique_data_labels = np.array(list(np.unique(all_data_labels)))
					reshape_collection = np.nan*np.ones((len(unique_data_labels),len(mean_diff_labels)))
					for label_i, label_name in enumerate(mean_diff_labels):
						data = mean_diff_collection[label_i]['data']
						data_labels = mean_diff_collection[label_i]['labels']
						for l_i, l_val in enumerate(data_labels):
							for u_index, ulabel in enumerate(unique_data_labels):
								if l_val[0] == ulabel:
									reshape_collection[u_index,label_i] = data[l_i]
					for u_i in range(len(unique_data_labels)):
						ax_mean_diff[1,i_3].plot(reshape_collection[u_i,:],alpha=0.5,label=unique_data_labels[u_i])
					ax_mean_diff[1,i_3].plot(np.arange(len(mean_diff_labels)),np.nanmean(reshape_collection,0),color='k',alpha=1,linewidth=3)
					ax_mean_diff[1,i_3].set_xticks(np.arange(len(mean_diff_labels)),mean_diff_labels,rotation=45)
					ax_mean_diff[1,i_3].set_ylabel('Mean Correlation Difference')
					#Calculate significances
					pair_diffs = list(combinations(np.arange(len(mean_diff_collection)),2))
					step = max_mean_diff_i/10
					for pair_i, pair in enumerate(pair_diffs):
						data_1 = mean_diff_collection[pair[0]]['data']
						data_2 = mean_diff_collection[pair[1]]['data']
						if (len(data_1) > 0)*(len(data_2) > 0):
							result = ks_2samp(data_1,data_2)
							if result[1] <= 0.05:
								marker='*'
								ind_1 = pair[0] + 1
								ind_2 = pair[1] + 1
								significance_storage[i_3][pair_i] = dict()
								significance_storage[i_3][pair_i]['ind_1'] = ind_1
								significance_storage[i_3][pair_i]['ind_2'] = ind_2
								significance_storage[i_3][pair_i]['marker'] = marker
					ax_mean_diff[0,i_3].set_xticks(np.arange(1,len(mean_diff_collection)+1),mean_diff_labels,rotation=45)
					ax_mean_diff[0,i_3].set_title(xlabel)
					ax_mean_diff[0,i_3].set_ylabel('Mean Correlation Difference')
					#Update axis labels of different plots
					ax[i_3,0].set_ylabel('Density Difference')
					ax_norm[i_3,0].set_ylabel('Normalized Density Difference')
					ax_mean[i_3,0].set_ylabel('Density Difference')
					ax_mean_norm[i_3,0].set_ylabel('Normalized Density Difference')
				#Update all plots with remaining values
				for i_3 in range(combo_lengths[2]):
					#Add significance data
					sig_pair_data = significance_storage[i_3]
					step = max_mean_diff/10
					sig_height = max_mean_diff + step
					for sp_i in list(sig_pair_data.keys()):
						ind_1 = sig_pair_data[sp_i]['ind_1']
						ind_2 = sig_pair_data[sp_i]['ind_2']
						marker = sig_pair_data[sp_i]['marker']
						ax_mean_diff[0,i_3].plot([ind_1,ind_2],[sig_height,sig_height],color='k',linestyle='solid')
						ax_mean_diff[0,i_3].plot([ind_1,ind_1],[sig_height-step/2,sig_height+step/2],color='k',linestyle='solid')
						ax_mean_diff[0,i_3].plot([ind_2,ind_2],[sig_height-step/2,sig_height+step/2],color='k',linestyle='solid')
						ax_mean_diff[0,i_3].text(ind_1 + (ind_2-ind_1)/2, sig_height + step/2,marker,horizontalalignment='center',verticalalignment='center')
						sig_height += step
					#Adjust y-limits
					ax_mean_diff[0,i_3].set_ylim([min_mean_diff - np.abs(min_mean_diff)/5,sig_height + sig_height/5])
					ax_mean_diff[1,i_3].set_ylim([min_mean_diff - np.abs(min_mean_diff)/5,sig_height + sig_height/5])
					ax_mean_diff[1,i_3].legend()
					for sp_i in range(len(segment_combinations)):
						ax[i_3,sp_i].set_ylim([min_diff,max_diff])
						ax_norm[i_3,sp_i].set_ylim([-1.01,1.01])
						ax_mean[i_3,sp_i].set_ylim([min_diff,max_diff])
						ax_mean_norm[i_3,sp_i].set_ylim([-1.01,1.01])
				#Finish plots with titles and save
				f.suptitle('Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_')
				f.savefig(os.path.join(true_save,f_pop_vec_plot_name) + '.png')
				f.savefig(os.path.join(true_save,f_pop_vec_plot_name) + '.svg')
				plt.close(f) 
				
				f_norm.suptitle('Normalized Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_norm.tight_layout()
				f_norm.savefig(os.path.join(norm_save,f_pop_vec_plot_name) + '_norm.png')
				f_norm.savefig(os.path.join(norm_save,f_pop_vec_plot_name) + '_norm.svg')
				plt.close(f_norm) 
				
				f_mean.suptitle('Mean Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean.tight_layout()
				f_mean.savefig(os.path.join(mean_save,f_pop_vec_plot_name) + '_mean.png')
				f_mean.savefig(os.path.join(mean_save,f_pop_vec_plot_name) + '_mean.svg')
				plt.close(f_mean) 
				
				f_mean_norm.suptitle('Normalized Mean Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean_norm.tight_layout()
				f_mean_norm.savefig(os.path.join(mean_norm_save,f_pop_vec_plot_name) + '_mean_norm.png')
				f_mean_norm.savefig(os.path.join(mean_norm_save,f_pop_vec_plot_name) + '_mean_norm.svg')
				plt.close(f_mean_norm)
				
				f_mean_diff.suptitle('Mean Correlation Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean_diff.tight_layout()
				f_mean_diff.savefig(os.path.join(mean_diff_save,f_pop_vec_plot_name) + '_mean_diff.png')
				f_mean_diff.savefig(os.path.join(mean_diff_save,f_pop_vec_plot_name) + '_mean_diff.svg')
				plt.close(f_mean_diff)


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
	mean_diff_save = os.path.join(save_dir,'Corr_Mean_Diffs')
	if not os.path.isdir(mean_diff_save):
		os.mkdir(mean_diff_save)
	
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
				f_mean_diff, ax_mean_diff = plt.subplots(ncols = combo_lengths[2], figsize=(combo_lengths[2]*5,5))
				max_diff = 0
				min_diff = 0
				max_mean_diff = 0
				min_mean_diff = 0
				significance_storage = dict()
				for i_3 in range(combo_lengths[2]):
					significance_storage[i_3] = dict()
					max_mean_diff_i = 0
					min_mean_diff_i = 0
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
					mean_diff_collection = dict()
					mean_diff_labels = []
					for tp_i, tp in enumerate(taste_combinations):
						taste_1 = tp[0]
						taste_2 = tp[1]
						title = taste_2 + ' - ' + taste_1
						mean_diffs = []
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
								mean_diffs.extend([np.nanmean(data_2) - np.nanmean(data_1)])
								counter += 1
							except:
								print("\tSkipping invalid dataset.")
						cum_dist_collection_array = np.array(cum_dist_collection)
						num_cum_dist = len(cum_dist_labels)
						#Collect mean distribution differences
						mean_diff_collection[tp_i] = dict()
						mean_diff_collection[tp_i]['data'] = mean_diffs
						mean_diff_collection[tp_i]['labels'] = cum_dist_labels
						mean_diff_labels.append(taste_2 + ' - ' + taste_1)
						if num_cum_dist >= 1:
							mean_cum_dist = np.nanmean(np.array(cum_dist_collection),0)
							std_cum_dist = np.nanstd(np.array(cum_dist_collection),0)
							half_means = np.concatenate((np.mean(mean_cum_dist[:500])*np.ones(500),np.mean(mean_cum_dist[500:])*np.ones(500)))
							max_cum_dist = np.max(np.array(cum_dist_collection))
							if max_cum_dist > max_diff:
								max_diff = max_cum_dist
							min_cum_dist = np.min(np.array(cum_dist_collection))
							if min_cum_dist < min_diff:
								min_diff = min_cum_dist
							std_y_low = mean_cum_dist-std_cum_dist
							std_y_high = mean_cum_dist+std_cum_dist
						#Plot distribution differences as is
						ax[i_3,tp_i].axhline(0,label='_',alpha=0.2,color='k')
						ax[i_3,tp_i].axvline(0.5,label='_',alpha=0.2,color='k')
						for c_i in range(len(cum_dist_collection)):
							ax[i_3,tp_i].plot(bin_x_vals,cum_dist_collection[c_i],label=cum_dist_labels[c_i],alpha=0.75)
						try:
							#Create std shading
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
						ax_mean_norm[i_3,tp_i].axhline(0,label='_',alpha=0.2,color='k')
						ax_mean_norm[i_3,tp_i].axvline(0.5,label='_',alpha=0.2,color='k')
						if num_cum_dist > 0:
							try:
								norm_dist = cum_dist_collection_array/(np.expand_dims(np.nanmax(np.abs(cum_dist_collection),1),1)*np.ones(np.shape(cum_dist_collection_array)))
							except:
								norm_dist = cum_dist_collection_array/(np.nanmax(np.abs(cum_dist_collection))*np.ones(np.shape(cum_dist_collection_array)))
						for c_i in range(len(cum_dist_collection)):
							ax_norm[i_3,tp_i].plot(bin_x_vals,norm_dist[c_i,:],label=cum_dist_labels[c_i],alpha=0.75)
						if num_cum_dist > 0:
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
							#Plot mean on separate axes
							ax_mean_norm[i_3,tp_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							ax_mean_norm[i_3,tp_i].plot(bin_x_vals,mean_norm_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax_mean_norm[i_3,tp_i].plot(bin_x_vals,half_means_norm,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
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
						#Calculate distribution differences significance above 0 using percentile
						true_sig_bins = np.zeros(len(bin_x_vals)+2)
						norm_sig_bins = np.zeros(len(bin_x_vals)+2)
						if num_cum_dist > 1: #Hopefully that means that there are far more than 1, or else this is silly
							for b_i, b_val in enumerate(bin_x_vals):
								#True data
								bin_dist = cum_dist_collection_array[:,b_i]
								cutoff_percentile = np.percentile(bin_dist,5)
								if 0 < cutoff_percentile:
									true_sig_bins[b_i+1] = 1
								#Normalized data
								bin_dist = norm_dist[:,b_i]
								cutoff_percentile = np.percentile(bin_dist,5)
								if 0 < cutoff_percentile:
									norm_sig_bins[b_i+1] = 1
						#Plot significance intervals as yellow shaded vertical bars
						true_bin_starts = np.where(np.diff(true_sig_bins) == 1)[0]
						true_bin_ends = np.where(np.diff(true_sig_bins) == -1)[0] - 1
						norm_bin_starts = np.where(np.diff(norm_sig_bins) == 1)[0]
						norm_bin_ends = np.where(np.diff(norm_sig_bins) == -1)[0] - 1
						if len(true_bin_starts) > 0:
							for tb_i in range(len(true_bin_starts)):
								start_i = bin_x_vals[true_bin_starts[tb_i]]
								end_i = bin_x_vals[true_bin_ends[tb_i]]
								ax[i_3,tp_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
								ax_mean[i_3,tp_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
						if len(norm_bin_starts) > 0:
							for nb_i in range(len(norm_bin_starts)):
								start_i = bin_x_vals[norm_bin_starts[nb_i]]
								end_i = bin_x_vals[norm_bin_ends[nb_i]]
								ax_norm[i_3,tp_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
								ax_mean_norm[i_3,tp_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
					#Plot box plots of mean correlation differences
					ax_mean_diff[i_3].axhline(0,label='_',alpha=0.2,color='k',linestyle='dashed')
					points_boxplot = []
					for m_i in range(len(mean_diff_collection)):
						points = mean_diff_collection[m_i]['data']
						if len(points)>0:
							if np.max(points) > max_mean_diff_i:
								max_mean_diff_i = np.max(points)
							if np.min(points) < min_mean_diff_i:
								min_mean_diff_i = np.min(points)
							ax_mean_diff[i_3].scatter(np.random.normal(m_i+1,0.04,size=len(points)),points,color='g',alpha=0.2)
						points_boxplot.append(list(points))
					ax_mean_diff[i_3].boxplot(points_boxplot,sym='',medianprops=dict(linestyle='-',color='blue'))
					if len(points_boxplot) > 0:
						if max_mean_diff < max_mean_diff_i:
							max_mean_diff = max_mean_diff_i
						if min_mean_diff > min_mean_diff_i:
							min_mean_diff = min_mean_diff_i
					#Calculate significances
					pair_diffs = list(combinations(np.arange(len(mean_diff_collection)),2))
					step = max_mean_diff/10
					for pair_i, pair in enumerate(pair_diffs):
						data_1 = mean_diff_collection[pair[0]]['data']
						data_2 = mean_diff_collection[pair[1]]['data']
						if (len(data_1) > 0)*(len(data_2) > 0):
							result = ks_2samp(data_1,data_2)
							if result[1] <= 0.05:
								marker='*'
								ind_1 = pair[0] + 1
								ind_2 = pair[1] + 1
								significance_storage[i_3][pair_i] = dict()
								significance_storage[i_3][pair_i]['ind_1'] = ind_1
								significance_storage[i_3][pair_i]['ind_2'] = ind_2
								significance_storage[i_3][pair_i]['marker'] = marker
					ax_mean_diff[i_3].set_xticks(np.arange(1,len(mean_diff_collection)+1),mean_diff_labels,rotation=45)
					ax_mean_diff[i_3].set_title(xlabel)
					ax_mean_diff[i_3].set_ylabel('Mean Correlation Difference')
					#Update axis labels of different plots
					ax[i_3,0].set_ylabel('Density Difference')
					ax_norm[i_3,0].set_ylabel('Normalized Density Difference')
					ax_mean[i_3,0].set_ylabel('Density Difference')
					ax_mean_norm[i_3,0].set_ylabel('Normalized Density Difference')
				#Update all plots with remaining values
				for i_3 in range(combo_lengths[2]):
					#Add significance data
					sig_pair_data = significance_storage[i_3]
					step = max_mean_diff/10
					sig_height = max_mean_diff + step
					for sp_i in list(sig_pair_data.keys()):
						ind_1 = sig_pair_data[sp_i]['ind_1']
						ind_2 = sig_pair_data[sp_i]['ind_2']
						marker = sig_pair_data[sp_i]['marker']
						ax_mean_diff[i_3].plot([ind_1,ind_2],[sig_height,sig_height],color='k',linestyle='solid')
						ax_mean_diff[i_3].plot([ind_1,ind_1],[sig_height-step/2,sig_height+step/2],color='k',linestyle='solid')
						ax_mean_diff[i_3].plot([ind_2,ind_2],[sig_height-step/2,sig_height+step/2],color='k',linestyle='solid')
						ax_mean_diff[i_3].text(ind_1 + (ind_2-ind_1)/2, sig_height + step/2,marker,horizontalalignment='center',verticalalignment='center')
						sig_height += step
					#Adjust y-limits
					ax_mean_diff[i_3].set_ylim([min_mean_diff - np.abs(min_mean_diff)/5,sig_height + sig_height/5])
					for tp_i in range(len(taste_combinations)):
						ax[i_3,tp_i].set_ylim([min_diff,max_diff])
						ax_norm[i_3,tp_i].set_ylim([-1.01,1.01])
						ax_mean[i_3,tp_i].set_ylim([min_diff,max_diff])
						ax_mean_norm[i_3,tp_i].set_ylim([-1.01,1.01])
				#Finish plots with titles and save
				f.suptitle('Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_')
				f.savefig(os.path.join(true_save,f_pop_vec_plot_name) + '.png')
				f.savefig(os.path.join(true_save,f_pop_vec_plot_name) + '.svg')
				plt.close(f) 
				
				f_norm.suptitle('Normalized Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_norm.tight_layout()
				f_norm.savefig(os.path.join(norm_save,f_pop_vec_plot_name) + '_norm.png')
				f_norm.savefig(os.path.join(norm_save,f_pop_vec_plot_name) + '_norm.svg')
				plt.close(f_norm) 
				
				f_mean.suptitle('Mean Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean.tight_layout()
				f_mean.savefig(os.path.join(mean_save,f_pop_vec_plot_name) + '_mean.png')
				f_mean.savefig(os.path.join(mean_save,f_pop_vec_plot_name) + '_mean.svg')
				plt.close(f_mean) 
				
				f_mean_norm.suptitle('Normalized Mean Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean_norm.tight_layout()
				f_mean_norm.savefig(os.path.join(mean_norm_save,f_pop_vec_plot_name) + '_mean_norm.png')
				f_mean_norm.savefig(os.path.join(mean_norm_save,f_pop_vec_plot_name) + '_mean_norm.svg')
				plt.close(f_mean_norm)
				
				f_mean_diff.suptitle('Mean Correlation Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean_diff.tight_layout()
				f_mean_diff.savefig(os.path.join(mean_diff_save,f_pop_vec_plot_name) + '_mean_diff.png')
				f_mean_diff.savefig(os.path.join(mean_diff_save,f_pop_vec_plot_name) + '_mean_diff.svg')
				plt.close(f_mean_diff)


def cross_epoch_diffs(corr_data,save_dir,unique_given_names,unique_corr_names,\
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
	mean_diff_save = os.path.join(save_dir,'Corr_Mean_Diffs')
	if not os.path.isdir(mean_diff_save):
		os.mkdir(mean_diff_save)
	
	class cross_epoch_attributes:
		def __init__(self,combo,names,i_1,i_2,i_3,unique_corr_names,\
				 unique_segment_names,unique_taste_names):
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
	characteristic_list = ['unique_corr_names','unique_segment_names','unique_taste_names']
	characteristic_dict = dict()
	for cl in characteristic_list:
		characteristic_dict[cl] = eval(cl)
	name_list = ['corr_name','seg_name','taste_name']
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
	epoch_combinations = list(combinations(unique_epochs,2))
	
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
				f, ax = plt.subplots(nrows = combo_lengths[2], ncols = len(epoch_combinations), figsize=(len(epoch_combinations)*5,combo_lengths[2]*5))
				f_norm, ax_norm = plt.subplots(nrows = combo_lengths[2], ncols = len(epoch_combinations), figsize=(len(epoch_combinations)*5,combo_lengths[2]*5))
				f_mean, ax_mean = plt.subplots(nrows = combo_lengths[2], ncols = len(epoch_combinations), figsize=(len(epoch_combinations)*5,combo_lengths[2]*5))
				f_mean_norm, ax_mean_norm = plt.subplots(nrows = combo_lengths[2], ncols = len(epoch_combinations), figsize=(len(epoch_combinations)*5,combo_lengths[2]*5))
				f_mean_diff, ax_mean_diff = plt.subplots(ncols = combo_lengths[2], figsize=(combo_lengths[2]*5,5))
				max_diff = 0
				min_diff = 0
				max_mean_diff = 0
				min_mean_diff = 0
				significance_storage = dict()
				for i_3 in range(combo_lengths[2]):
					significance_storage[i_3] = dict()
					max_mean_diff_i = 0
					min_mean_diff_i = 0
					xlabel = eval(combo[2])[i_3]
					if type(xlabel) == np.int64:
						xlabel = "epoch_" + str(xlabel)
					#Begin pulling data
					att = cross_epoch_attributes(combo,names,i_1,i_2,i_3,unique_corr_names,\
							 unique_segment_names, unique_taste_names)
					corr_name = att.corr_name
					seg_name = att.seg_name
					taste_name = att.taste_name
					#Pit segment pairs against each other
					mean_diff_collection = dict()
					mean_diff_labels = []
					for ep_i, ep in enumerate(epoch_combinations):
						epoch_1 = ep[0]
						epoch_2 = ep[1]
						title = 'Epoch ' + str(epoch_2) + ' - Epoch ' + str(epoch_1)
						mean_diffs = []
						cum_dist_collection = []
						cum_dist_labels = []
						counter = 0
						for g_n in unique_given_names:
							try:
								data_1 = unique_data_dict[corr_name][g_n][seg_name][taste_name]['data'][:,epoch_1]
								counts_1, _ = np.histogram(data_1,bin_edges)
								counts_1_smooth = savgol_filter(counts_1,np.ceil(len(data_1)/100).astype('int'),polyorder=1)
								counts_1_smooth_density = counts_1_smooth/np.sum(counts_1_smooth)
								data_2 = unique_data_dict[corr_name][g_n][seg_name][taste_name]['data'][:,epoch_2]
								counts_2, _ = np.histogram(data_2,bin_edges)
								counts_2_smooth = savgol_filter(counts_2,np.ceil(len(data_2)/100).astype('int'),polyorder=1)
								counts_2_smooth_density = counts_2_smooth/np.sum(counts_2_smooth)
								cum_dist_collection.append(counts_2_smooth_density-counts_1_smooth_density)
								cum_dist_labels.append([g_n + '(' + str(counter) + ')'])
								mean_diffs.extend([np.nanmean(data_2) - np.nanmean(data_1)])
								counter += 1
							except:
								print("\tSkipping invalid dataset.")
						cum_dist_collection_array = np.array(cum_dist_collection)
						num_cum_dist = len(cum_dist_labels)
						#Collect mean distribution differences
						mean_diff_collection[ep_i] = dict()
						mean_diff_collection[ep_i]['data'] = mean_diffs
						mean_diff_collection[ep_i]['labels'] = cum_dist_labels
						mean_diff_labels.append(title)
						if num_cum_dist >= 1:
							mean_cum_dist = np.nanmean(np.array(cum_dist_collection),0)
							std_cum_dist = np.nanstd(np.array(cum_dist_collection),0)
							half_means = np.concatenate((np.mean(mean_cum_dist[:500])*np.ones(500),np.mean(mean_cum_dist[500:])*np.ones(500)))
							max_cum_dist = np.max(np.array(cum_dist_collection))
							if max_cum_dist > max_diff:
								max_diff = max_cum_dist
							min_cum_dist = np.min(np.array(cum_dist_collection))
							if min_cum_dist < min_diff:
								min_diff = min_cum_dist
							std_y_low = mean_cum_dist-std_cum_dist
							std_y_high = mean_cum_dist+std_cum_dist
						#Plot distribution differences as is
						ax[i_3,ep_i].axhline(0,label='_',alpha=0.2,color='k')
						ax[i_3,ep_i].axvline(0.5,label='_',alpha=0.2,color='k')
						for c_i in range(len(cum_dist_collection)):
							ax[i_3,ep_i].plot(bin_x_vals,cum_dist_collection[c_i],label=cum_dist_labels[c_i],alpha=0.75)
						try:
							#Create std shading
							ax[i_3,ep_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							#Plot mean of all curves
							ax[i_3,ep_i].plot(bin_x_vals,mean_cum_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax[i_3,ep_i].plot(bin_x_vals,half_means,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
							
							ax_mean[i_3,ep_i].axhline(0,label='_',alpha=0.2,color='k')
							ax_mean[i_3,ep_i].axvline(0.5,label='_',alpha=0.2,color='k')
							ax_mean[i_3,ep_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							ax_mean[i_3,ep_i].plot(bin_x_vals,mean_cum_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax_mean[i_3,ep_i].plot(bin_x_vals,half_means,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
						except:
							print("\tNo Mean Plotted.")
						ax[i_3,ep_i].legend(fontsize='8', loc ='lower left')
						ax[i_3,ep_i].set_xlim([0,1.1])
						ax[i_3,ep_i].set_xlabel(xlabel)
						ax[i_3,ep_i].set_title(title)
						#Plot normalized distribution differences: |max diff| = 1
						ax_norm[i_3,ep_i].axhline(0,label='_',alpha=0.2,color='k')
						ax_norm[i_3,ep_i].axvline(0.5,label='_',alpha=0.2,color='k')
						ax_mean_norm[i_3,ep_i].axhline(0,label='_',alpha=0.2,color='k')
						ax_mean_norm[i_3,ep_i].axvline(0.5,label='_',alpha=0.2,color='k')
						if num_cum_dist > 0:
							try:
								norm_dist = cum_dist_collection_array/(np.expand_dims(np.nanmax(np.abs(cum_dist_collection),1),1)*np.ones(np.shape(cum_dist_collection_array)))
							except:
								norm_dist = cum_dist_collection_array/(np.nanmax(np.abs(cum_dist_collection))*np.ones(np.shape(cum_dist_collection_array)))
						for c_i in range(len(cum_dist_collection)):
							ax_norm[i_3,ep_i].plot(bin_x_vals,norm_dist[c_i,:],label=cum_dist_labels[c_i],alpha=0.75)
						if num_cum_dist > 0:
							mean_norm_dist = np.nanmean(norm_dist,0)
							std_norm_dist = np.nanstd(norm_dist,0)
							half_means_norm = np.concatenate((np.nanmean(mean_norm_dist[:500])*np.ones(500),np.nanmean(mean_norm_dist[500:])*np.ones(500)))
							#Create std shading
							std_y_low = mean_norm_dist-std_norm_dist
							std_y_high = mean_norm_dist+std_norm_dist
							ax_norm[i_3,ep_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							#Plot mean of all curves
							ax_norm[i_3,ep_i].plot(bin_x_vals,mean_norm_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax_norm[i_3,ep_i].plot(bin_x_vals,half_means_norm,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
							#Plot mean on separate axes
							ax_mean_norm[i_3,ep_i].fill_between(bin_x_vals,std_y_low,std_y_high,alpha=0.2,color='k',label='Std')
							ax_mean_norm[i_3,ep_i].plot(bin_x_vals,mean_norm_dist,linewidth=3, linestyle='--', color='k',label='Mean')
							ax_mean_norm[i_3,ep_i].plot(bin_x_vals,half_means_norm,linewidth=3, linestyle='dotted', color='k',label='Mean-Binned')
						ax_norm[i_3,ep_i].legend(fontsize='8', loc ='lower left')
						ax_norm[i_3,ep_i].set_xlim([0,1.1])
						ax_norm[i_3,ep_i].set_xlabel(xlabel)
						ax_norm[i_3,ep_i].set_title(title)
						#Mean plots cleanups
						ax_mean[i_3,ep_i].legend(fontsize='8', loc ='lower left')
						ax_mean[i_3,ep_i].set_xlim([0,1.1])
						ax_mean[i_3,ep_i].set_xlabel(xlabel)
						ax_mean[i_3,ep_i].set_title(title)
						ax_mean_norm[i_3,ep_i].legend(fontsize='8', loc ='lower left')
						ax_mean_norm[i_3,ep_i].set_xlim([0,1.1])
						ax_mean_norm[i_3,ep_i].set_xlabel(xlabel)
						ax_mean_norm[i_3,ep_i].set_title(title)
						#Calculate distribution differences significance above 0 using percentile
						true_sig_bins = np.zeros(len(bin_x_vals)+2)
						norm_sig_bins = np.zeros(len(bin_x_vals)+2)
						if num_cum_dist > 1: #Hopefully that means that there are far more than 1, or else this is silly
							for b_i, b_val in enumerate(bin_x_vals):
								#True data
								bin_dist = cum_dist_collection_array[:,b_i]
								cutoff_percentile = np.percentile(bin_dist,5)
								if 0 < cutoff_percentile:
									true_sig_bins[b_i+1] = 1
								#Normalized data
								bin_dist = norm_dist[:,b_i]
								cutoff_percentile = np.percentile(bin_dist,5)
								if 0 < cutoff_percentile:
									norm_sig_bins[b_i+1] = 1
						#Plot significance intervals as yellow shaded vertical bars
						true_bin_starts = np.where(np.diff(true_sig_bins) == 1)[0]
						true_bin_ends = np.where(np.diff(true_sig_bins) == -1)[0] - 1
						norm_bin_starts = np.where(np.diff(norm_sig_bins) == 1)[0]
						norm_bin_ends = np.where(np.diff(norm_sig_bins) == -1)[0] - 1
						if len(true_bin_starts) > 0:
							for tb_i in range(len(true_bin_starts)):
								start_i = bin_x_vals[true_bin_starts[tb_i]]
								end_i = bin_x_vals[true_bin_ends[tb_i]]
								ax[i_3,ep_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
								ax_mean[i_3,ep_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
						if len(norm_bin_starts) > 0:
							for nb_i in range(len(norm_bin_starts)):
								start_i = bin_x_vals[norm_bin_starts[nb_i]]
								end_i = bin_x_vals[norm_bin_ends[nb_i]]
								ax_norm[i_3,ep_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
								ax_mean_norm[i_3,ep_i].fill_betweenx(np.array([-1,1]),start_i,end_i,alpha=0.2,color='yellow')
					#Plot box plots of mean correlation differences
					ax_mean_diff[i_3].axhline(0,label='_',alpha=0.2,color='k',linestyle='dashed')
					points_boxplot = []
					for m_i in range(len(mean_diff_collection)):
						points = mean_diff_collection[m_i]['data']
						if len(points)>0:
							if np.max(points) > max_mean_diff_i:
								max_mean_diff_i = np.max(points)
							if np.min(points) < min_mean_diff_i:
								min_mean_diff_i = np.min(points)
							ax_mean_diff[i_3].scatter(np.random.normal(m_i+1,0.04,size=len(points)),points,color='g',alpha=0.2)
						points_boxplot.append(list(points))
					ax_mean_diff[i_3].boxplot(points_boxplot,sym='',medianprops=dict(linestyle='-',color='blue'))
					if len(points_boxplot) > 0:
						if max_mean_diff < max_mean_diff_i:
							max_mean_diff = max_mean_diff_i
						if min_mean_diff > min_mean_diff_i:
							min_mean_diff = min_mean_diff_i
					#Calculate significances
					pair_diffs = list(combinations(np.arange(len(mean_diff_collection)),2))
					step = max_mean_diff/10
					for pair_i, pair in enumerate(pair_diffs):
						data_1 = mean_diff_collection[pair[0]]['data']
						data_2 = mean_diff_collection[pair[1]]['data']
						if (len(data_1) > 0)*(len(data_2) > 0):
							result = ks_2samp(data_1,data_2)
							if result[1] <= 0.05:
								marker='*'
								ind_1 = pair[0] + 1
								ind_2 = pair[1] + 1
								significance_storage[i_3][pair_i] = dict()
								significance_storage[i_3][pair_i]['ind_1'] = ind_1
								significance_storage[i_3][pair_i]['ind_2'] = ind_2
								significance_storage[i_3][pair_i]['marker'] = marker
					ax_mean_diff[i_3].set_xticks(np.arange(1,len(mean_diff_collection)+1),mean_diff_labels,rotation=45)
					ax_mean_diff[i_3].set_title(xlabel)
					ax_mean_diff[i_3].set_ylabel('Mean Correlation Difference')
					#Update axis labels of different plots
					ax[i_3,0].set_ylabel('Density Difference')
					ax_norm[i_3,0].set_ylabel('Normalized Density Difference')
					ax_mean[i_3,0].set_ylabel('Density Difference')
					ax_mean_norm[i_3,0].set_ylabel('Normalized Density Difference')
				#Update all plots with remaining values
				for i_3 in range(combo_lengths[2]):
					#Add significance data
					sig_pair_data = significance_storage[i_3]
					step = max_mean_diff/10
					sig_height = max_mean_diff + step
					for sp_i in list(sig_pair_data.keys()):
						ind_1 = sig_pair_data[sp_i]['ind_1']
						ind_2 = sig_pair_data[sp_i]['ind_2']
						marker = sig_pair_data[sp_i]['marker']
						ax_mean_diff[i_3].plot([ind_1,ind_2],[sig_height,sig_height],color='k',linestyle='solid')
						ax_mean_diff[i_3].plot([ind_1,ind_1],[sig_height-step/2,sig_height+step/2],color='k',linestyle='solid')
						ax_mean_diff[i_3].plot([ind_2,ind_2],[sig_height-step/2,sig_height+step/2],color='k',linestyle='solid')
						ax_mean_diff[i_3].text(ind_1 + (ind_2-ind_1)/2, sig_height + step/2,marker,horizontalalignment='center',verticalalignment='center')
						sig_height += step
					#Adjust y-limits
					ax_mean_diff[i_3].set_ylim([min_mean_diff - np.abs(min_mean_diff)/5,sig_height + sig_height/5])
					for ep_i in range(len(epoch_combinations)):
						ax[i_3,ep_i].set_ylim([min_diff,max_diff])
						ax_norm[i_3,ep_i].set_ylim([-1.01,1.01])
						ax_mean[i_3,ep_i].set_ylim([min_diff,max_diff])
						ax_mean_norm[i_3,ep_i].set_ylim([-1.01,1.01])
				#Finish plots with titles and save
				f.suptitle('Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_')
				f.savefig(os.path.join(true_save,f_pop_vec_plot_name) + '.png')
				f.savefig(os.path.join(true_save,f_pop_vec_plot_name) + '.svg')
				plt.close(f) 
				
				f_norm.suptitle('Normalized Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_norm.tight_layout()
				f_norm.savefig(os.path.join(norm_save,f_pop_vec_plot_name) + '_norm.png')
				f_norm.savefig(os.path.join(norm_save,f_pop_vec_plot_name) + '_norm.svg')
				plt.close(f_norm) 
				
				f_mean.suptitle('Mean Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean.tight_layout()
				f_mean.savefig(os.path.join(mean_save,f_pop_vec_plot_name) + '_mean.png')
				f_mean.savefig(os.path.join(mean_save,f_pop_vec_plot_name) + '_mean.svg')
				plt.close(f_mean) 
				
				f_mean_norm.suptitle('Normalized Mean Probability Density Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean_norm.tight_layout()
				f_mean_norm.savefig(os.path.join(mean_norm_save,f_pop_vec_plot_name) + '_mean_norm.png')
				f_mean_norm.savefig(os.path.join(mean_norm_save,f_pop_vec_plot_name) + '_mean_norm.svg')
				plt.close(f_mean_norm)
				
				f_mean_diff.suptitle('Mean Correlation Difference: ' + combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				f_mean_diff.tight_layout()
				f_mean_diff.savefig(os.path.join(mean_diff_save,f_pop_vec_plot_name) + '_mean_diff.png')
				f_mean_diff.savefig(os.path.join(mean_diff_save,f_pop_vec_plot_name) + '_mean_diff.svg')
				plt.close(f_mean_diff)

	

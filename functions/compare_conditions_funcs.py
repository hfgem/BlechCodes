#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:22:39 2023

@author: Hannah Germaine
Plot and analysis functions to support compare_conditions.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ks_2samp, ttest_ind
import warnings
		

def cross_corr_name(data_dict,results_dir,unique_given_names,unique_corr_names,\
			  unique_segment_names,unique_taste_names):
	"""This function collects statistics across different correlation types and
	plots them together
	INPUTS:
		- data_dict: dictionary containing correlation data across conditions.
			length = number of datasets
			data_dict[i] = dictionary of dataset data
			data_dict[i]['corr_dev_stats'] = dict of length #segments
			data_dict[i]['corr_dev_stats'][s_i] = dict of length #tastes
			data_dict[i]['corr_dev_stats'][s_i][t_i] = dict containing the 3 correlation calculation types
				1. neuron_data_storage = individual neuron timeseries correlations [num_dev x num_trials x num_neur x num_epochs]
				2. pop_data_storage  = population timeseries correlations [num_dev x num_trials x num_epochs]
				3. pop_vec_data_storage = population average vector correlations [num_dev x num_trials x num_epochs]
		- results_dir: directory to save the resulting plots
		- unique_corr_names: unique names of correlation analyses to compare
	OUTPUTS: plots and statistical significance tests
	"""
	warnings.filterwarnings('ignore')
	#TODO: add significance tests
	
	class cross_corr_name_attributes:
		def __init__(self,combo,names,i_1,i_2,i_3,unique_segment_names,\
				 unique_taste_names,unique_epochs):
			setattr(self, names[0], eval(combo[0])[i_1])
			setattr(self, names[1], eval(combo[1])[i_2])
			setattr(self, names[2], eval(combo[2])[i_3])
	
	#_____Reorganize data by unique correlation type_____
	unique_corr_dict = dict()
	for ucn in unique_corr_names:
		unique_corr_dict[ucn] = dict()
		for udn in unique_given_names:
			unique_corr_dict[ucn][udn] = dict()
			for usn in unique_segment_names:
				unique_corr_dict[ucn][udn][usn] = dict()
				for utn in unique_taste_names:
					unique_corr_dict[ucn][udn][usn][utn] = dict()
	max_epochs = 0
	for d_i in data_dict:
		dataset = data_dict[d_i]
		given_name = dataset['given_name']
		corr_name = dataset['corr_name']
		corr_dev_stats = dataset['corr_dev_stats']
		num_seg = len(corr_dev_stats)
		for s_i in range(num_seg):
			num_tastes = len(corr_dev_stats[s_i])
			for t_i in range(num_tastes):
				seg_name = corr_dev_stats[s_i][t_i]['segment']
				taste_name = corr_dev_stats[s_i][t_i]['taste']
				avg_taste_neur_data = np.nanmean(corr_dev_stats[s_i][t_i]['neuron_data_storage'],2)
				unique_corr_dict[corr_name][given_name][seg_name][taste_name]['avg_neuron_data_storage'] = avg_taste_neur_data.reshape(-1,avg_taste_neur_data.shape[-1])
				pop_taste_data = corr_dev_stats[s_i][t_i]['pop_data_storage']
				num_epochs = pop_taste_data.shape[-1]
				if num_epochs > max_epochs:
					max_epochs = num_epochs
				unique_corr_dict[corr_name][given_name][seg_name][taste_name]['pop_data_storage'] = pop_taste_data.reshape(-1,pop_taste_data.shape[-1])
				pop_vec_taste_data = corr_dev_stats[s_i][t_i]['pop_vec_data_storage']
				unique_corr_dict[corr_name][given_name][seg_name][taste_name]['pop_vec_data_storage'] = pop_vec_taste_data.reshape(-1,pop_taste_data.shape[-1])
	
	#Plot all combinations
	unique_epochs = np.arange(max_epochs)
	characteristic_list = ['unique_segment_names','unique_epochs','unique_taste_names']
	name_list = ['seg_name','e_i','taste_name']
	all_combinations_full = []
	all_names_full = []
	for ac in characteristic_list:
		missing = np.setdiff1d(characteristic_list,ac)
		full_combo = [ac]
		full_combo.extend(missing)
		all_combinations_full.append(full_combo)
		names_combo = [name_list[characteristic_list.index(c)] for c in full_combo]
		all_names_full.append(names_combo)
	
	for c_i in range(len(all_combinations_full)):
		combo = all_combinations_full[c_i]
		names = all_names_full[c_i]
		for i_1 in range(len(eval(combo[0]))):
			combo_1 = eval(combo[0])[i_1]
			if type(combo_1) == np.int64:
				combo_1 = "epoch_" + str(combo_1)
			f_avg_neur, ax_avg_neur = plt.subplots(nrows = len(eval(combo[1])), ncols = len(eval(combo[2])), figsize=(len(eval(combo[2]))*4,len(eval(combo[1]))*4))
			f_pop, ax_pop = plt.subplots(nrows = len(eval(combo[1])), ncols = len(eval(combo[2])), figsize=(len(eval(combo[2]))*4,len(eval(combo[1]))*4))
			f_pop_vec, ax_pop_vec = plt.subplots(nrows = len(eval(combo[1])), ncols = len(eval(combo[2])), figsize=(len(eval(combo[2]))*4,len(eval(combo[1]))*4))
			for i_2 in range(len(eval(combo[1]))):
				ylabel = eval(combo[1])[i_2]
				if type(ylabel) == np.int64:
					ylabel = "epoch_" + str(ylabel)
				for i_3 in range(len(eval(combo[2]))):
					xlabel = eval(combo[2])[i_3]
					if type(xlabel) == np.int64:
						xlabel = "epoch_" + str(xlabel)
					att = cross_corr_name_attributes(combo,names,i_1,i_2,i_3,unique_segment_names,\
							 unique_taste_names,unique_epochs)
					seg_name = att.seg_name
					taste_name = att.taste_name
					e_i = att.e_i
					avg_neuron_data_storage_collection = []
					pop_data_storage_collection = []
					pop_vec_data_storage_collection = []
					data_names = []
					for corr_name in unique_corr_names:
						for given_name in unique_given_names:
							avg_neuron_data_storage_collection.append(unique_corr_dict[corr_name][given_name][seg_name][taste_name]['avg_neuron_data_storage'][:,e_i])
							pop_data_storage_collection.append(unique_corr_dict[corr_name][given_name][seg_name][taste_name]['pop_data_storage'][:,e_i])
							pop_vec_data_storage_collection.append(unique_corr_dict[corr_name][given_name][seg_name][taste_name]['pop_vec_data_storage'][:,e_i])
							data_names.extend([given_name + '_' + corr_name])
					ax_avg_neur[i_2,i_3].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
					ax_avg_neur[i_2,i_3].legend(fontsize='12', loc ='lower right')
					ax_avg_neur[i_2,i_3].set_xlim([0,1.1])
					ax_avg_neur[i_2,i_3].set_ylim([0,1.1])
					ax_avg_neur[i_2,i_3].set_ylabel(ylabel)
					ax_avg_neur[i_2,i_3].set_xlabel(xlabel)
					ax_pop[i_2,i_3].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
					ax_pop[i_2,i_3].legend(fontsize='12', loc ='lower right')
					ax_pop[i_2,i_3].set_xlim([0,1.1])
					ax_pop[i_2,i_3].set_ylim([0,1.1])
					ax_pop[i_2,i_3].set_ylabel(ylabel)
					ax_pop[i_2,i_3].set_xlabel(xlabel)
					ax_pop_vec[i_2,i_3].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
					ax_pop_vec[i_2,i_3].legend(fontsize='12', loc ='lower right')
					ax_pop_vec[i_2,i_3].set_xlim([0,1.1])
					ax_pop_vec[i_2,i_3].set_ylim([0,1.1])
					ax_pop_vec[i_2,i_3].set_ylabel(ylabel)
					ax_pop_vec[i_2,i_3].set_xlabel(xlabel)
			plt.figure(1)
			plt.suptitle(combo_1.replace(' ','_'))
			plt.tight_layout()
			f_avg_neur_plot_name = combo_1.replace(' ','_') + '_avg_neuron'
			f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.png')
			f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.svg')
			plt.close(f_avg_neur)
			plt.figure(2)
			plt.suptitle(combo_1.replace(' ','_'))
			plt.tight_layout()
			f_pop_plot_name = combo_1.replace(' ','_') + '_pop'
			f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.png')
			f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.svg')
			plt.close(f_pop)
			plt.figure(3)
			plt.suptitle(combo_1.replace(' ','_'))
			plt.tight_layout()
			f_pop_vec_plot_name = combo_1.replace(' ','_') + '_pop_vec'
			f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.png')
			f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.svg')
			plt.close(f_pop_vec) 
				
		
def cross_data(data_dict,results_dir,unique_given_names,unique_corr_names,\
			  unique_segment_names,unique_taste_names):
	"""This function collects statistics across different datasets and plots 
	them together
	INPUTS:
		- data_dict: dictionary containing correlation data across conditions.
			length = number of datasets
			data_dict[i] = dictionary of dataset data
			data_dict[i]['corr_dev_stats'] = dict of length #segments
			data_dict[i]['corr_dev_stats'][s_i] = dict of length #tastes
			data_dict[i]['corr_dev_stats'][s_i][t_i] = dict containing the 3 correlation calculation types
				1. neuron_data_storage = individual neuron timeseries correlations [num_dev x num_trials x num_neur x num_epochs]
				2. pop_data_storage  = population timeseries correlations [num_dev x num_trials x num_epochs]
				3. pop_vec_data_storage = population average vector correlations [num_dev x num_trials x num_epochs]
		- results_dir: directory to save the resulting plots
		- unique_corr_names: unique names of correlation analyses to compare
	OUTPUTS: plots and statistical significance tests
	"""
	warnings.filterwarnings('ignore')
	
	class cross_data_attributes:
		def __init__(self,combo,names,i_1,i_2,i_3,i_4,unique_corr_names,\
				 unique_segment_names,unique_taste_names,\
					 unique_epochs):
			setattr(self, names[0], eval(combo[0])[i_1])
			setattr(self, names[1], eval(combo[1])[i_2])
			setattr(self, names[2], eval(combo[2])[i_3])
			setattr(self, names[3], eval(combo[3])[i_4])
	
	#_____Reorganize data by unique correlation type_____
	unique_data_dict = dict()
	for ucn in unique_corr_names:
		unique_data_dict[ucn] = dict()
		for ugn in unique_given_names:
			unique_data_dict[ucn][ugn] = dict()
			for usn in unique_segment_names:
				unique_data_dict[ucn][ugn][usn] = dict()
				for utn in unique_taste_names:
					unique_data_dict[ucn][ugn][usn][utn] = dict()
	
	max_epochs = 0
	for d_i in data_dict:
		dataset = data_dict[d_i]
		given_name = dataset['given_name']
		corr_name = dataset['corr_name']
		corr_dev_stats = dataset['corr_dev_stats']
		num_seg = len(corr_dev_stats)
		for s_i in range(num_seg):
			num_tastes = len(corr_dev_stats[s_i])
			for t_i in range(num_tastes):
				seg_name = corr_dev_stats[s_i][t_i]['segment']
				taste_name = corr_dev_stats[s_i][t_i]['taste']
				avg_taste_neur_data = np.nanmean(corr_dev_stats[s_i][t_i]['neuron_data_storage'],2)
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['avg_neuron_data_storage'] = avg_taste_neur_data.reshape(-1,avg_taste_neur_data.shape[-1])
				pop_taste_data = corr_dev_stats[s_i][t_i]['pop_data_storage']
				num_epochs = pop_taste_data.shape[-1]
				if num_epochs > max_epochs:
					max_epochs = num_epochs
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_data_storage'] = pop_taste_data.reshape(-1,pop_taste_data.shape[-1])
				pop_vec_taste_data = corr_dev_stats[s_i][t_i]['pop_vec_data_storage']
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_vec_data_storage'] = pop_vec_taste_data.reshape(-1,pop_taste_data.shape[-1])
	
	
	#Plot all combinations
	unique_epochs = np.arange(max_epochs)
	characteristic_list = ['unique_corr_names','unique_segment_names','unique_epochs','unique_taste_names']
	characteristic_dict = dict()
	for cl in characteristic_list:
		characteristic_dict[cl] = eval(cl)
	name_list = ['corr_name','seg_name','e_i','taste_name']
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
				f_avg_neur, ax_avg_neur = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				f_pop, ax_pop = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				f_pop_vec, ax_pop_vec = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				for i_3 in range(combo_lengths[2]):
					ylabel = eval(combo[2])[i_3]
					if type(ylabel) == np.int64:
						ylabel = "epoch_" + str(ylabel)
					for i_4 in range(combo_lengths[3]):
						xlabel = eval(combo[3])[i_4]
						if type(xlabel) == np.int64:
							xlabel = "epoch_" + str(xlabel)
						att = cross_data_attributes(combo,names,i_1,i_2,i_3,i_4,unique_corr_names,\
								 unique_segment_names,unique_taste_names,\
									 unique_epochs)
						avg_neuron_data_storage_collection = []
						pop_data_storage_collection = []
						pop_vec_data_storage_collection = []
						corr_name = att.corr_name
						seg_name = att.seg_name
						taste_name = att.taste_name
						e_i = att.e_i
						for given_name in unique_given_names:
							 avg_neuron_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['avg_neuron_data_storage'][:,e_i])
							 pop_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_data_storage'][:,e_i])
							 pop_vec_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_vec_data_storage'][:,e_i])
						#Pairwise significance tests
						sig_ind_pairs = list(combinations(np.arange(len(unique_given_names)),2))
						sig_titles = 'i1,  i2,  KS*,  T*, 1v2'
						avg_neuron_sig_pair_results = sig_titles
						pop_data_sig_pair_results = sig_titles
						pop_vec_sig_pair_results = sig_titles
						for sip_i in range(len(sig_ind_pairs)):
							sip = sig_ind_pairs[sip_i]
							#avg_neuron
							avg_neuron_sig_text = '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = avg_neuron_data_storage_collection[sip[0]]
							data_2 = avg_neuron_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '*'
							else:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '<'
								elif result == 0: #>
									avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '>'
							else:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							avg_neuron_sig_pair_results = avg_neuron_sig_pair_results + avg_neuron_sig_text
							#pop_data
							pop_data_sig_text =  '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = pop_data_storage_collection[sip[0]]
							data_2 = pop_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								pop_data_sig_text = pop_data_sig_text + ',  ' + '*'
							else:
								pop_data_sig_text = pop_data_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								pop_data_sig_text = pop_data_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									pop_data_sig_text = pop_data_sig_text + ',  ' + '<'
								elif result == 0: #>
									pop_data_sig_text = pop_data_sig_text + ',  ' + '>'
							else:
								pop_data_sig_text = pop_data_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							pop_data_sig_pair_results = pop_data_sig_pair_results + pop_data_sig_text
							#pop_vec
							pop_vec_sig_text =  '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = pop_vec_data_storage_collection[sip[0]]
							data_2 = pop_vec_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + '*'
							else:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									pop_vec_sig_text = pop_vec_sig_text + ',  ' + '<'
								elif result == 0: #>
									pop_vec_sig_text = pop_vec_sig_text + ',  ' + '>'
							else:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							pop_vec_sig_pair_results = pop_vec_sig_pair_results + pop_vec_sig_text
						#Plot results
						txt_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
						labels = [unique_given_names[i] + ' (' + str(i) + ')' for i in range(len(unique_given_names))]
						if (combo_lengths[2] == 1)*(combo_lengths[3] > 1):
							#avg_neur
							ax_avg_neur[i_4].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_avg_neur[i_4].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_4].set_xlim([0,1.1])
							ax_avg_neur[i_4].set_ylim([0,1.1])
							ax_avg_neur[i_4].set_ylabel(ylabel)
							ax_avg_neur[i_4].set_xlabel(xlabel)
							ax_avg_neur[i_4].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_4].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop[i_4].set_xlim([0,1.1])
							ax_pop[i_4].set_ylim([0,1.1])
							ax_pop[i_4].set_ylabel(ylabel)
							ax_pop[i_4].set_xlabel(xlabel)
							ax_pop[i_4].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_4].set_xlim([0,1.1])
							ax_pop_vec[i_4].set_ylim([0,1.1])
							ax_pop_vec[i_4].set_ylabel(ylabel)
							ax_pop_vec[i_4].set_xlabel(xlabel)
							ax_pop_vec[i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						elif (combo_lengths[2] > 1)*(combo_lengths[3] == 1):
							#avg_neur
							ax_avg_neur[i_3].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_avg_neur[i_3].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_3].set_xlim([0,1.1])
							ax_avg_neur[i_3].set_ylim([0,1.1])
							ax_avg_neur[i_3].set_ylabel(ylabel)
							ax_avg_neur[i_3].set_xlabel(xlabel)
							ax_avg_neur[i_3].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_3].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop[i_3].set_xlim([0,1.1])
							ax_pop[i_3].set_ylim([0,1.1])
							ax_pop[i_3].set_ylabel(ylabel)
							ax_pop[i_3].set_xlabel(xlabel)
							ax_pop[i_3].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_3].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3].set_xlim([0,1.1])
							ax_pop_vec[i_3].set_ylim([0,1.1])
							ax_pop_vec[i_3].set_ylabel(ylabel)
							ax_pop_vec[i_3].set_xlabel(xlabel)
							ax_pop_vec[i_3].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						else:
							#avg_neur
							ax_avg_neur[i_3,i_4].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_avg_neur[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_3,i_4].set_xlim([0,1.1])
							ax_avg_neur[i_3,i_4].set_ylim([0,1.1])
							ax_avg_neur[i_3,i_4].set_ylabel(ylabel)
							ax_avg_neur[i_3,i_4].set_xlabel(xlabel)
							ax_avg_neur[i_3,i_4].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_3,i_4].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop[i_3,i_4].set_xlim([0,1.1])
							ax_pop[i_3,i_4].set_ylim([0,1.1])
							ax_pop[i_3,i_4].set_ylabel(ylabel)
							ax_pop[i_3,i_4].set_xlabel(xlabel)
							ax_pop[i_3,i_4].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_3,i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3,i_4].set_xlim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylabel(ylabel)
							ax_pop_vec[i_3,i_4].set_xlabel(xlabel)
							ax_pop_vec[i_3,i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
				plt.figure(1)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_avg_neur_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_avg_neuron'
				f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.png')
				f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.svg')
				plt.close(f_avg_neur)
				plt.figure(2)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop'
				f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.png')
				f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.svg')
				plt.close(f_pop)
				plt.figure(3)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop_vec'
				f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.png')
				f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.svg')
				plt.close(f_pop_vec) 
	

def cross_segment(data_dict,results_dir,unique_given_names,unique_corr_names,\
			  unique_segment_names,unique_taste_names):
	"""This function collects statistics across different segments and plots 
	them together
	INPUTS:
		- data_dict: dictionary containing correlation data across conditions.
			length = number of datasets
			data_dict[i] = dictionary of dataset data
			data_dict[i]['corr_dev_stats'] = dict of length #segments
			data_dict[i]['corr_dev_stats'][s_i] = dict of length #tastes
			data_dict[i]['corr_dev_stats'][s_i][t_i] = dict containing the 3 correlation calculation types
				1. neuron_data_storage = individual neuron timeseries correlations [num_dev x num_trials x num_neur x num_epochs]
				2. pop_data_storage  = population timeseries correlations [num_dev x num_trials x num_epochs]
				3. pop_vec_data_storage = population average vector correlations [num_dev x num_trials x num_epochs]
		- results_dir: directory to save the resulting plots
		- unique_corr_names: unique names of correlation analyses to compare
	OUTPUTS: plots and statistical significance tests
	"""
	warnings.filterwarnings('ignore')
	
	class cross_segment_attributes:
		def __init__(self,combo,names,i_1,i_2,i_3,i_4,unique_corr_names,\
				 unique_given_names,unique_taste_names,\
					 unique_epochs):
			setattr(self, names[0], eval(combo[0])[i_1])
			setattr(self, names[1], eval(combo[1])[i_2])
			setattr(self, names[2], eval(combo[2])[i_3])
			setattr(self, names[3], eval(combo[3])[i_4])
	
	#_____Reorganize data by unique correlation type_____
	unique_data_dict = dict()
	for ucn in unique_corr_names:
		unique_data_dict[ucn] = dict()
		for ugn in unique_given_names:
			unique_data_dict[ucn][ugn] = dict()
			for usn in unique_segment_names:
				unique_data_dict[ucn][ugn][usn] = dict()
				for utn in unique_taste_names:
					unique_data_dict[ucn][ugn][usn][utn] = dict()
	
	max_epochs = 0
	for d_i in data_dict:
		dataset = data_dict[d_i]
		given_name = dataset['given_name']
		corr_name = dataset['corr_name']
		corr_dev_stats = dataset['corr_dev_stats']
		num_seg = len(corr_dev_stats)
		for s_i in range(num_seg):
			num_tastes = len(corr_dev_stats[s_i])
			for t_i in range(num_tastes):
				seg_name = corr_dev_stats[s_i][t_i]['segment']
				taste_name = corr_dev_stats[s_i][t_i]['taste']
				avg_taste_neur_data = np.nanmean(corr_dev_stats[s_i][t_i]['neuron_data_storage'],2)
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['avg_neuron_data_storage'] = avg_taste_neur_data.reshape(-1,avg_taste_neur_data.shape[-1])
				pop_taste_data = corr_dev_stats[s_i][t_i]['pop_data_storage']
				num_epochs = pop_taste_data.shape[-1]
				if num_epochs > max_epochs:
					max_epochs = num_epochs
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_data_storage'] = pop_taste_data.reshape(-1,pop_taste_data.shape[-1])
				pop_vec_taste_data = corr_dev_stats[s_i][t_i]['pop_vec_data_storage']
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_vec_data_storage'] = pop_vec_taste_data.reshape(-1,pop_taste_data.shape[-1])
	
	
	#Plot all combinations
	unique_epochs = np.arange(max_epochs)
	characteristic_list = ['unique_corr_names','unique_given_names','unique_taste_names','unique_epochs']
	characteristic_dict = dict()
	for cl in characteristic_list:
		characteristic_dict[cl] = eval(cl)
	name_list = ['corr_name','given_name','taste_name','e_i']
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
				f_avg_neur, ax_avg_neur = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				f_pop, ax_pop = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				f_pop_vec, ax_pop_vec = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				for i_3 in range(combo_lengths[2]):
					ylabel = eval(combo[2])[i_3]
					if type(ylabel) == np.int64:
						ylabel = "epoch_" + str(ylabel)
					for i_4 in range(combo_lengths[3]):
						xlabel = eval(combo[3])[i_4]
						if type(xlabel) == np.int64:
							xlabel = "epoch_" + str(xlabel)
						att = cross_segment_attributes(combo,names,i_1,i_2,i_3,i_4,unique_corr_names,\
								 unique_given_names,unique_taste_names,\
									 unique_epochs)
						avg_neuron_data_storage_collection = []
						pop_data_storage_collection = []
						pop_vec_data_storage_collection = []
						corr_name = att.corr_name
						given_name = att.given_name
						taste_name = att.taste_name
						e_i = att.e_i
						for seg_name in unique_segment_names:
							 avg_neuron_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['avg_neuron_data_storage'][:,e_i])
							 pop_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_data_storage'][:,e_i])
							 pop_vec_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_vec_data_storage'][:,e_i])
						#Pairwise significance tests
						sig_ind_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
						sig_titles = 'i1,  i2,  KS*,  T*, 1v2'
						avg_neuron_sig_pair_results = sig_titles
						pop_data_sig_pair_results = sig_titles
						pop_vec_sig_pair_results = sig_titles
						for sip_i in range(len(sig_ind_pairs)):
							sip = sig_ind_pairs[sip_i]
							#avg_neuron
							avg_neuron_sig_text = '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = avg_neuron_data_storage_collection[sip[0]]
							data_2 = avg_neuron_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '*'
							else:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '<'
								elif result == 0: #>
									avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '>'
							else:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							avg_neuron_sig_pair_results = avg_neuron_sig_pair_results + avg_neuron_sig_text
							#pop_data
							pop_data_sig_text =  '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = pop_data_storage_collection[sip[0]]
							data_2 = pop_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								pop_data_sig_text = pop_data_sig_text + ',  ' + '*'
							else:
								pop_data_sig_text = pop_data_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								pop_data_sig_text = pop_data_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									pop_data_sig_text = pop_data_sig_text + ',  ' + '<'
								elif result == 0: #>
									pop_data_sig_text = pop_data_sig_text + ',  ' + '>'
							else:
								pop_data_sig_text = pop_data_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							pop_data_sig_pair_results = pop_data_sig_pair_results + pop_data_sig_text
							#pop_vec
							pop_vec_sig_text =  '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = pop_vec_data_storage_collection[sip[0]]
							data_2 = pop_vec_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + '*'
							else:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									pop_vec_sig_text = pop_vec_sig_text + ',  ' + '<'
								elif result == 0: #>
									pop_vec_sig_text = pop_vec_sig_text + ',  ' + '>'
							else:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							pop_vec_sig_pair_results = pop_vec_sig_pair_results + pop_vec_sig_text
						#Plot results
						txt_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
						labels = [unique_segment_names[i] + ' (' + str(i) + ')' for i in range(len(unique_segment_names))]
						if (combo_lengths[2] == 1)*(combo_lengths[3] > 1):
							#avg_neur
							ax_avg_neur[i_4].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_avg_neur[i_4].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_4].set_xlim([0,1.1])
							ax_avg_neur[i_4].set_ylim([0,1.1])
							ax_avg_neur[i_4].set_ylabel(ylabel)
							ax_avg_neur[i_4].set_xlabel(xlabel)
							ax_avg_neur[i_4].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_4].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop[i_4].set_xlim([0,1.1])
							ax_pop[i_4].set_ylim([0,1.1])
							ax_pop[i_4].set_ylabel(ylabel)
							ax_pop[i_4].set_xlabel(xlabel)
							ax_pop[i_4].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_4].set_xlim([0,1.1])
							ax_pop_vec[i_4].set_ylim([0,1.1])
							ax_pop_vec[i_4].set_ylabel(ylabel)
							ax_pop_vec[i_4].set_xlabel(xlabel)
							ax_pop_vec[i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						elif (combo_lengths[2] > 1)*(combo_lengths[3] == 1):
							#avg_neur
							ax_avg_neur[i_3].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_avg_neur[i_3].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_3].set_xlim([0,1.1])
							ax_avg_neur[i_3].set_ylim([0,1.1])
							ax_avg_neur[i_3].set_ylabel(ylabel)
							ax_avg_neur[i_3].set_xlabel(xlabel)
							ax_avg_neur[i_3].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_3].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop[i_3].set_xlim([0,1.1])
							ax_pop[i_3].set_ylim([0,1.1])
							ax_pop[i_3].set_ylabel(ylabel)
							ax_pop[i_3].set_xlabel(xlabel)
							ax_pop[i_3].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_3].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3].set_xlim([0,1.1])
							ax_pop_vec[i_3].set_ylim([0,1.1])
							ax_pop_vec[i_3].set_ylabel(ylabel)
							ax_pop_vec[i_3].set_xlabel(xlabel)
							ax_pop_vec[i_3].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						else:
							#avg_neur
							ax_avg_neur[i_3,i_4].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_avg_neur[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_3,i_4].set_xlim([0,1.1])
							ax_avg_neur[i_3,i_4].set_ylim([0,1.1])
							ax_avg_neur[i_3,i_4].set_ylabel(ylabel)
							ax_avg_neur[i_3,i_4].set_xlabel(xlabel)
							ax_avg_neur[i_3,i_4].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_3,i_4].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop[i_3,i_4].set_xlim([0,1.1])
							ax_pop[i_3,i_4].set_ylim([0,1.1])
							ax_pop[i_3,i_4].set_ylabel(ylabel)
							ax_pop[i_3,i_4].set_xlabel(xlabel)
							ax_pop[i_3,i_4].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_3,i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3,i_4].set_xlim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylabel(ylabel)
							ax_pop_vec[i_3,i_4].set_xlabel(xlabel)
							ax_pop_vec[i_3,i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
				plt.figure(1)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_avg_neur_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_avg_neuron'
				f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.png')
				f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.svg')
				plt.close(f_avg_neur)
				plt.figure(2)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop'
				f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.png')
				f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.svg')
				plt.close(f_pop)
				plt.figure(3)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop_vec'
				f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.png')
				f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.svg')
				plt.close(f_pop_vec) 
	

def cross_taste(data_dict,results_dir,unique_given_names,unique_corr_names,\
			  unique_segment_names,unique_taste_names):
	"""This function collects statistics across different tastes and plots 
	them together
	INPUTS:
		- data_dict: dictionary containing correlation data across conditions.
			length = number of datasets
			data_dict[i] = dictionary of dataset data
			data_dict[i]['corr_dev_stats'] = dict of length #segments
			data_dict[i]['corr_dev_stats'][s_i] = dict of length #tastes
			data_dict[i]['corr_dev_stats'][s_i][t_i] = dict containing the 3 correlation calculation types
				1. neuron_data_storage = individual neuron timeseries correlations [num_dev x num_trials x num_neur x num_epochs]
				2. pop_data_storage  = population timeseries correlations [num_dev x num_trials x num_epochs]
				3. pop_vec_data_storage = population average vector correlations [num_dev x num_trials x num_epochs]
		- results_dir: directory to save the resulting plots
		- unique_corr_names: unique names of correlation analyses to compare
	OUTPUTS: plots and statistical significance tests
	"""
	warnings.filterwarnings('ignore')
	
	class cross_taste_attributes:
		def __init__(self,combo,names,i_1,i_2,i_3,i_4,unique_corr_names,\
				 unique_given_names,unique_segment_names,\
					 unique_epochs):
			setattr(self, names[0], eval(combo[0])[i_1])
			setattr(self, names[1], eval(combo[1])[i_2])
			setattr(self, names[2], eval(combo[2])[i_3])
			setattr(self, names[3], eval(combo[3])[i_4])
	
	#_____Reorganize data by unique correlation type_____
	unique_data_dict = dict()
	for ucn in unique_corr_names:
		unique_data_dict[ucn] = dict()
		for ugn in unique_given_names:
			unique_data_dict[ucn][ugn] = dict()
			for usn in unique_segment_names:
				unique_data_dict[ucn][ugn][usn] = dict()
				for utn in unique_taste_names:
					unique_data_dict[ucn][ugn][usn][utn] = dict()
	
	max_epochs = 0
	for d_i in data_dict:
		dataset = data_dict[d_i]
		given_name = dataset['given_name']
		corr_name = dataset['corr_name']
		corr_dev_stats = dataset['corr_dev_stats']
		num_seg = len(corr_dev_stats)
		for s_i in range(num_seg):
			num_tastes = len(corr_dev_stats[s_i])
			for t_i in range(num_tastes):
				seg_name = corr_dev_stats[s_i][t_i]['segment']
				taste_name = corr_dev_stats[s_i][t_i]['taste']
				avg_taste_neur_data = np.nanmean(corr_dev_stats[s_i][t_i]['neuron_data_storage'],2)
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['avg_neuron_data_storage'] = avg_taste_neur_data.reshape(-1,avg_taste_neur_data.shape[-1])
				pop_taste_data = corr_dev_stats[s_i][t_i]['pop_data_storage']
				num_epochs = pop_taste_data.shape[-1]
				if num_epochs > max_epochs:
					max_epochs = num_epochs
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_data_storage'] = pop_taste_data.reshape(-1,pop_taste_data.shape[-1])
				pop_vec_taste_data = corr_dev_stats[s_i][t_i]['pop_vec_data_storage']
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_vec_data_storage'] = pop_vec_taste_data.reshape(-1,pop_taste_data.shape[-1])
	
	
	#Plot all combinations
	unique_epochs = np.arange(max_epochs)
	characteristic_list = ['unique_corr_names','unique_given_names','unique_segment_names','unique_epochs']
	characteristic_dict = dict()
	for cl in characteristic_list:
		characteristic_dict[cl] = eval(cl)
	name_list = ['corr_name','given_name','seg_name','e_i']
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
				f_avg_neur, ax_avg_neur = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				f_pop, ax_pop = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				f_pop_vec, ax_pop_vec = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				for i_3 in range(combo_lengths[2]):
					ylabel = eval(combo[2])[i_3]
					if type(ylabel) == np.int64:
						ylabel = "epoch_" + str(ylabel)
					for i_4 in range(combo_lengths[3]):
						xlabel = eval(combo[3])[i_4]
						if type(xlabel) == np.int64:
							xlabel = "epoch_" + str(xlabel)
						att = cross_taste_attributes(combo,names,i_1,i_2,i_3,i_4,unique_corr_names,\
								 unique_given_names,unique_segment_names,\
									 unique_epochs)
						avg_neuron_data_storage_collection = []
						pop_data_storage_collection = []
						pop_vec_data_storage_collection = []
						corr_name = att.corr_name
						seg_name = att.seg_name
						given_name = att.given_name
						e_i = att.e_i
						for taste_name in unique_taste_names:
							 avg_neuron_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['avg_neuron_data_storage'][:,e_i])
							 pop_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_data_storage'][:,e_i])
							 pop_vec_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_vec_data_storage'][:,e_i])
						#Pairwise significance tests
						sig_ind_pairs = list(combinations(np.arange(len(unique_taste_names)),2))
						sig_titles = 'i1,  i2,  KS*,  T*, 1v2'
						avg_neuron_sig_pair_results = sig_titles
						pop_data_sig_pair_results = sig_titles
						pop_vec_sig_pair_results = sig_titles
						for sip_i in range(len(sig_ind_pairs)):
							sip = sig_ind_pairs[sip_i]
							#avg_neuron
							avg_neuron_sig_text = '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = avg_neuron_data_storage_collection[sip[0]]
							data_2 = avg_neuron_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '*'
							else:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '<'
								elif result == 0: #>
									avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '>'
							else:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							avg_neuron_sig_pair_results = avg_neuron_sig_pair_results + avg_neuron_sig_text
							#pop_data
							pop_data_sig_text =  '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = pop_data_storage_collection[sip[0]]
							data_2 = pop_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								pop_data_sig_text = pop_data_sig_text + ',  ' + '*'
							else:
								pop_data_sig_text = pop_data_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								pop_data_sig_text = pop_data_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									pop_data_sig_text = pop_data_sig_text + ',  ' + '<'
								elif result == 0: #>
									pop_data_sig_text = pop_data_sig_text + ',  ' + '>'
							else:
								pop_data_sig_text = pop_data_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							pop_data_sig_pair_results = pop_data_sig_pair_results + pop_data_sig_text
							#pop_vec
							pop_vec_sig_text =  '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = pop_vec_data_storage_collection[sip[0]]
							data_2 = pop_vec_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + '*'
							else:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									pop_vec_sig_text = pop_vec_sig_text + ',  ' + '<'
								elif result == 0: #>
									pop_vec_sig_text = pop_vec_sig_text + ',  ' + '>'
							else:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							pop_vec_sig_pair_results = pop_vec_sig_pair_results + pop_vec_sig_text
						#Plot results
						txt_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
						labels = [unique_taste_names[i] + ' (' + str(i) + ')' for i in range(len(unique_taste_names))]
						if (combo_lengths[2] == 1)*(combo_lengths[3] > 1):
							#avg_neur
							ax_avg_neur[i_4].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_avg_neur[i_4].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_4].set_xlim([0,1.1])
							ax_avg_neur[i_4].set_ylim([0,1.1])
							ax_avg_neur[i_4].set_ylabel(ylabel)
							ax_avg_neur[i_4].set_xlabel(xlabel)
							ax_avg_neur[i_4].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_4].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop[i_4].set_xlim([0,1.1])
							ax_pop[i_4].set_ylim([0,1.1])
							ax_pop[i_4].set_ylabel(ylabel)
							ax_pop[i_4].set_xlabel(xlabel)
							ax_pop[i_4].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_4].set_xlim([0,1.1])
							ax_pop_vec[i_4].set_ylim([0,1.1])
							ax_pop_vec[i_4].set_ylabel(ylabel)
							ax_pop_vec[i_4].set_xlabel(xlabel)
							ax_pop_vec[i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						elif (combo_lengths[2] > 1)*(combo_lengths[3] == 1):
							#avg_neur
							ax_avg_neur[i_3].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_avg_neur[i_3].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_3].set_xlim([0,1.1])
							ax_avg_neur[i_3].set_ylim([0,1.1])
							ax_avg_neur[i_3].set_ylabel(ylabel)
							ax_avg_neur[i_3].set_xlabel(xlabel)
							ax_avg_neur[i_3].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_3].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop[i_3].set_xlim([0,1.1])
							ax_pop[i_3].set_ylim([0,1.1])
							ax_pop[i_3].set_ylabel(ylabel)
							ax_pop[i_3].set_xlabel(xlabel)
							ax_pop[i_3].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_3].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3].set_xlim([0,1.1])
							ax_pop_vec[i_3].set_ylim([0,1.1])
							ax_pop_vec[i_3].set_ylabel(ylabel)
							ax_pop_vec[i_3].set_xlabel(xlabel)
							ax_pop_vec[i_3].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						else:
							#avg_neur
							ax_avg_neur[i_3,i_4].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_avg_neur[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_3,i_4].set_xlim([0,1.1])
							ax_avg_neur[i_3,i_4].set_ylim([0,1.1])
							ax_avg_neur[i_3,i_4].set_ylabel(ylabel)
							ax_avg_neur[i_3,i_4].set_xlabel(xlabel)
							ax_avg_neur[i_3,i_4].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_3,i_4].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop[i_3,i_4].set_xlim([0,1.1])
							ax_pop[i_3,i_4].set_ylim([0,1.1])
							ax_pop[i_3,i_4].set_ylabel(ylabel)
							ax_pop[i_3,i_4].set_xlabel(xlabel)
							ax_pop[i_3,i_4].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_3,i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3,i_4].set_xlim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylabel(ylabel)
							ax_pop_vec[i_3,i_4].set_xlabel(xlabel)
							ax_pop_vec[i_3,i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
				plt.figure(1)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_avg_neur_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_avg_neuron'
				f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.png')
				f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.svg')
				plt.close(f_avg_neur)
				plt.figure(2)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop'
				f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.png')
				f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.svg')
				plt.close(f_pop)
				plt.figure(3)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop_vec'
				f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.png')
				f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.svg')
				plt.close(f_pop_vec) 
	

def cross_epoch(data_dict,results_dir,unique_given_names,unique_corr_names,\
			  unique_segment_names,unique_taste_names):
	"""This function collects statistics across different epochs and plots 
	them together
	INPUTS:
		- data_dict: dictionary containing correlation data across conditions.
			length = number of datasets
			data_dict[i] = dictionary of dataset data
			data_dict[i]['corr_dev_stats'] = dict of length #segments
			data_dict[i]['corr_dev_stats'][s_i] = dict of length #tastes
			data_dict[i]['corr_dev_stats'][s_i][t_i] = dict containing the 3 correlation calculation types
				1. neuron_data_storage = individual neuron timeseries correlations [num_dev x num_trials x num_neur x num_epochs]
				2. pop_data_storage  = population timeseries correlations [num_dev x num_trials x num_epochs]
				3. pop_vec_data_storage = population average vector correlations [num_dev x num_trials x num_epochs]
		- results_dir: directory to save the resulting plots
		- unique_corr_names: unique names of correlation analyses to compare
	OUTPUTS: plots and statistical significance tests
	"""
	warnings.filterwarnings('ignore')
	
	class cross_epoch_attributes:
		def __init__(self,combo,names,i_1,i_2,i_3,i_4,unique_corr_names,\
				 unique_given_names,unique_segment_names,\
					 unique_taste_names):
			setattr(self, names[0], eval(combo[0])[i_1])
			setattr(self, names[1], eval(combo[1])[i_2])
			setattr(self, names[2], eval(combo[2])[i_3])
			setattr(self, names[3], eval(combo[3])[i_4])
	
	#_____Reorganize data by unique correlation type_____
	unique_data_dict = dict()
	for ucn in unique_corr_names:
		unique_data_dict[ucn] = dict()
		for ugn in unique_given_names:
			unique_data_dict[ucn][ugn] = dict()
			for usn in unique_segment_names:
				unique_data_dict[ucn][ugn][usn] = dict()
				for utn in unique_taste_names:
					unique_data_dict[ucn][ugn][usn][utn] = dict()
	
	max_epochs = 0
	for d_i in data_dict:
		dataset = data_dict[d_i]
		given_name = dataset['given_name']
		corr_name = dataset['corr_name']
		corr_dev_stats = dataset['corr_dev_stats']
		num_seg = len(corr_dev_stats)
		for s_i in range(num_seg):
			num_tastes = len(corr_dev_stats[s_i])
			for t_i in range(num_tastes):
				seg_name = corr_dev_stats[s_i][t_i]['segment']
				taste_name = corr_dev_stats[s_i][t_i]['taste']
				avg_taste_neur_data = np.nanmean(corr_dev_stats[s_i][t_i]['neuron_data_storage'],2)
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['avg_neuron_data_storage'] = avg_taste_neur_data.reshape(-1,avg_taste_neur_data.shape[-1])
				pop_taste_data = corr_dev_stats[s_i][t_i]['pop_data_storage']
				num_epochs = pop_taste_data.shape[-1]
				if num_epochs > max_epochs:
					max_epochs = num_epochs
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_data_storage'] = pop_taste_data.reshape(-1,pop_taste_data.shape[-1])
				pop_vec_taste_data = corr_dev_stats[s_i][t_i]['pop_vec_data_storage']
				unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_vec_data_storage'] = pop_vec_taste_data.reshape(-1,pop_taste_data.shape[-1])
	
	
	#Plot all combinations
	unique_epochs = np.arange(max_epochs)
	characteristic_list = ['unique_corr_names','unique_given_names','unique_segment_names','unique_taste_names']
	characteristic_dict = dict()
	for cl in characteristic_list:
		characteristic_dict[cl] = eval(cl)
	name_list = ['corr_name','given_name','seg_name','taste_name']
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
				f_avg_neur, ax_avg_neur = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				f_pop, ax_pop = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				f_pop_vec, ax_pop_vec = plt.subplots(nrows = combo_lengths[2], ncols = combo_lengths[3], figsize=(combo_lengths[3]*4,combo_lengths[2]*4))
				for i_3 in range(combo_lengths[2]):
					ylabel = eval(combo[2])[i_3]
					if type(ylabel) == np.int64:
						ylabel = "epoch_" + str(ylabel)
					for i_4 in range(combo_lengths[3]):
						xlabel = eval(combo[3])[i_4]
						if type(xlabel) == np.int64:
							xlabel = "epoch_" + str(xlabel)
						att = cross_epoch_attributes(combo,names,i_1,i_2,i_3,i_4,unique_corr_names,\
								 unique_given_names,unique_segment_names,\
									 unique_taste_names)
						avg_neuron_data_storage_collection = []
						pop_data_storage_collection = []
						pop_vec_data_storage_collection = []
						corr_name = att.corr_name
						seg_name = att.seg_name
						taste_name = att.taste_name
						given_name = att.given_name
						for e_i in unique_epochs:
							 avg_neuron_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['avg_neuron_data_storage'][:,e_i])
							 pop_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_data_storage'][:,e_i])
							 pop_vec_data_storage_collection.append(unique_data_dict[corr_name][given_name][seg_name][taste_name]['pop_vec_data_storage'][:,e_i])
						#Pairwise significance tests
						sig_ind_pairs = list(combinations(np.arange(len(unique_epochs)),2))
						sig_titles = 'i1,  i2,  KS*,  T*, 1v2'
						avg_neuron_sig_pair_results = sig_titles
						pop_data_sig_pair_results = sig_titles
						pop_vec_sig_pair_results = sig_titles
						for sip_i in range(len(sig_ind_pairs)):
							sip = sig_ind_pairs[sip_i]
							#avg_neuron
							avg_neuron_sig_text = '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = avg_neuron_data_storage_collection[sip[0]]
							data_2 = avg_neuron_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '*'
							else:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '<'
								elif result == 0: #>
									avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + '>'
							else:
								avg_neuron_sig_text = avg_neuron_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							avg_neuron_sig_pair_results = avg_neuron_sig_pair_results + avg_neuron_sig_text
							#pop_data
							pop_data_sig_text =  '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = pop_data_storage_collection[sip[0]]
							data_2 = pop_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								pop_data_sig_text = pop_data_sig_text + ',  ' + '*'
							else:
								pop_data_sig_text = pop_data_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								pop_data_sig_text = pop_data_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									pop_data_sig_text = pop_data_sig_text + ',  ' + '<'
								elif result == 0: #>
									pop_data_sig_text = pop_data_sig_text + ',  ' + '>'
							else:
								pop_data_sig_text = pop_data_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							pop_data_sig_pair_results = pop_data_sig_pair_results + pop_data_sig_text
							#pop_vec
							pop_vec_sig_text =  '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = pop_vec_data_storage_collection[sip[0]]
							data_2 = pop_vec_data_storage_collection[sip[1]]
							result = ks_2samp(data_1[~np.isnan(data_1)],data_2[~np.isnan(data_2)])
							if result[1] < 0.05:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + '*'
							else:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + 'n.s.'
							result = ttest_ind(data_1,data_2,nan_policy='omit',alternative='two-sided')
							if result[1] < 0.05:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + '*'
								result = (np.nanmean(data_1) < np.nanmean(data_2)).astype('int') #ttest_ind(data_1,data_2,nan_policy='omit',alternative='less')
								if result == 1: #<
									pop_vec_sig_text = pop_vec_sig_text + ',  ' + '<'
								elif result == 0: #>
									pop_vec_sig_text = pop_vec_sig_text + ',  ' + '>'
							else:
								pop_vec_sig_text = pop_vec_sig_text + ',  ' + 'n.s.' + ',  ' + 'n.a.'
							pop_vec_sig_pair_results = pop_vec_sig_pair_results + pop_vec_sig_text
						#Plot results
						txt_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
						if (combo_lengths[2] == 1)*(combo_lengths[3] > 1):
							#avg_neur
							ax_avg_neur[i_4].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=unique_epochs)
							ax_avg_neur[i_4].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_4].set_xlim([0,1.1])
							ax_avg_neur[i_4].set_ylim([0,1.1])
							ax_avg_neur[i_4].set_ylabel(ylabel)
							ax_avg_neur[i_4].set_xlabel(xlabel)
							ax_avg_neur[i_4].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_4].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=unique_epochs)
							ax_pop[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop[i_4].set_xlim([0,1.1])
							ax_pop[i_4].set_ylim([0,1.1])
							ax_pop[i_4].set_ylabel(ylabel)
							ax_pop[i_4].set_xlabel(xlabel)
							ax_pop[i_4].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=unique_epochs)
							ax_pop_vec[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_4].set_xlim([0,1.1])
							ax_pop_vec[i_4].set_ylim([0,1.1])
							ax_pop_vec[i_4].set_ylabel(ylabel)
							ax_pop_vec[i_4].set_xlabel(xlabel)
							ax_pop_vec[i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						elif (combo_lengths[2] > 1)*(combo_lengths[3] == 1):
							#avg_neur
							ax_avg_neur[i_3].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=unique_epochs)
							ax_avg_neur[i_3].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_3].set_xlim([0,1.1])
							ax_avg_neur[i_3].set_ylim([0,1.1])
							ax_avg_neur[i_3].set_ylabel(ylabel)
							ax_avg_neur[i_3].set_xlabel(xlabel)
							ax_avg_neur[i_3].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_3].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=unique_epochs)
							ax_pop[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop[i_3].set_xlim([0,1.1])
							ax_pop[i_3].set_ylim([0,1.1])
							ax_pop[i_3].set_ylabel(ylabel)
							ax_pop[i_3].set_xlabel(xlabel)
							ax_pop[i_3].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_3].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=unique_epochs)
							ax_pop_vec[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3].set_xlim([0,1.1])
							ax_pop_vec[i_3].set_ylim([0,1.1])
							ax_pop_vec[i_3].set_ylabel(ylabel)
							ax_pop_vec[i_3].set_xlabel(xlabel)
							ax_pop_vec[i_3].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						else:
							#avg_neur
							ax_avg_neur[i_3,i_4].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=unique_epochs)
							ax_avg_neur[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_avg_neur[i_3,i_4].set_xlim([0,1.1])
							ax_avg_neur[i_3,i_4].set_ylim([0,1.1])
							ax_avg_neur[i_3,i_4].set_ylabel(ylabel)
							ax_avg_neur[i_3,i_4].set_xlabel(xlabel)
							ax_avg_neur[i_3,i_4].text(0.05, 1, avg_neuron_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop
							ax_pop[i_3,i_4].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=unique_epochs)
							ax_pop[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop[i_3,i_4].set_xlim([0,1.1])
							ax_pop[i_3,i_4].set_ylim([0,1.1])
							ax_pop[i_3,i_4].set_ylabel(ylabel)
							ax_pop[i_3,i_4].set_xlabel(xlabel)
							ax_pop[i_3,i_4].text(0.05, 1, pop_data_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
							#pop_vec
							ax_pop_vec[i_3,i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=unique_epochs)
							ax_pop_vec[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3,i_4].set_xlim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylabel(ylabel)
							ax_pop_vec[i_3,i_4].set_xlabel(xlabel)
							ax_pop_vec[i_3,i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
				plt.figure(1)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_avg_neur_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_avg_neuron'
				f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.png')
				f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.svg')
				plt.close(f_avg_neur)
				plt.figure(2)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop'
				f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.png')
				f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.svg')
				plt.close(f_pop)
				plt.figure(3)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop_vec'
				f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.png')
				f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.svg')
				plt.close(f_pop_vec) 
				
				
				
				

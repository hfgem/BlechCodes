#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:22:39 2023

@author: Hannah Germaine
Plot and analysis functions to support compare_conditions.py
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ks_2samp, ttest_ind
		

def cross_corr_name(corr_data,save_dir,unique_given_names,unique_corr_names,\
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
	warnings.filterwarnings('ignore')
	#TODO: add significance tests
	
	class cross_corr_name_attributes:
		def __init__(self,combo,names,i_1,i_2,i_3,unique_segment_names,\
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
					data_storage = []
					data_names = []
					for corr_name in unique_corr_names:
						for given_name in unique_given_names:
							dataset = unique_data_dict[corr_name][given_name][seg_name][taste_name]['data'][:,e_i]
							if len(dataset[~np.isnan(dataset)]) > 0:
								data_storage.append(dataset)
								data_names.extend([given_name + '_' + corr_name])
					ax_pop_vec[i_2,i_3].hist(data_storage,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
					ax_pop_vec[i_2,i_3].legend(fontsize='12', loc ='lower right')
					ax_pop_vec[i_2,i_3].set_xlim([0,1.1])
					ax_pop_vec[i_2,i_3].set_ylim([0,1.1])
					ax_pop_vec[i_2,i_3].set_ylabel(ylabel)
					ax_pop_vec[i_2,i_3].set_xlabel(xlabel)
			plt.figure(3)
			plt.suptitle(combo_1.replace(' ','_'))
			plt.tight_layout()
			f_pop_vec_plot_name = combo_1.replace(' ','_') + '_pop_vec'
			f_pop_vec.savefig(os.path.join(save_dir,f_pop_vec_plot_name) + '.png')
			f_pop_vec.savefig(os.path.join(save_dir,f_pop_vec_plot_name) + '.svg')
			plt.close(f_pop_vec) 
				
def cross_segment(corr_data,save_dir,unique_given_names,unique_corr_names,\
			  unique_segment_names,unique_taste_names):
	"""This function collects statistics across different segments and plots 
	them together
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
						pop_vec_data_storage_collection = []
						seg_used = []
						corr_name = att.corr_name
						given_name = att.given_name
						taste_name = att.taste_name
						e_i = att.e_i
						#Only keep datasets that contain actual data!
						for seg_name in unique_segment_names:
							seg_data = unique_data_dict[corr_name][given_name][seg_name][taste_name]['data'][:,e_i]
							if len(seg_data[~np.isnan(seg_data)]):
								pop_vec_data_storage_collection.append(seg_data)
								seg_used.append(seg_name)
						labels = [seg_used[i] + ' (' + str(i) + ')' for i in range(len(seg_used))]
						#Pairwise significance tests
						sig_ind_pairs = list(combinations(np.arange(len(seg_used)),2))
						sig_titles = 'i1,  i2,  KS*,  T*, 1v2'
						pop_vec_sig_pair_results = sig_titles
						for sip_i in range(len(sig_ind_pairs)):
							sip = sig_ind_pairs[sip_i]
							#pop_vec
							pop_vec_sig_text =  '\n' + str(sip[0]) + ',  ' + str(sip[1])
							data_1 = pop_vec_data_storage_collection[sip[0]]
							data_2 = pop_vec_data_storage_collection[sip[1]]
							if (len(data_1[~np.isnan(data_1)]) > 0)*(len(data_2[~np.isnan(data_2)]) > 0):
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
							#pop_vec
							ax_pop_vec[i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_4].set_xlim([0,1.1])
							ax_pop_vec[i_4].set_ylim([0,1.1])
							ax_pop_vec[i_4].set_ylabel(ylabel)
							ax_pop_vec[i_4].set_xlabel(xlabel)
							ax_pop_vec[i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						elif (combo_lengths[2] > 1)*(combo_lengths[3] == 1):
							#pop_vec
							ax_pop_vec[i_3].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3].set_xlim([0,1.1])
							ax_pop_vec[i_3].set_ylim([0,1.1])
							ax_pop_vec[i_3].set_ylabel(ylabel)
							ax_pop_vec[i_3].set_xlabel(xlabel)
							ax_pop_vec[i_3].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						else:
							#pop_vec
							ax_pop_vec[i_3,i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3,i_4].set_xlim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylabel(ylabel)
							ax_pop_vec[i_3,i_4].set_xlabel(xlabel)
							ax_pop_vec[i_3,i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
				plt.figure(3)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop_vec'
				f_pop_vec.savefig(os.path.join(save_dir,f_pop_vec_plot_name) + '.png')
				f_pop_vec.savefig(os.path.join(save_dir,f_pop_vec_plot_name) + '.svg')
				plt.close(f_pop_vec) 
	

def cross_taste(corr_data,save_dir,unique_given_names,unique_corr_names,\
			  unique_segment_names,unique_taste_names):
	"""This function collects statistics across different tastes and plots 
	them together
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
						pop_vec_data_storage_collection = []
						taste_used = []
						corr_name = att.corr_name
						seg_name = att.seg_name
						given_name = att.given_name
						e_i = att.e_i
						for taste_name in unique_taste_names:
							dataset = unique_data_dict[corr_name][given_name][seg_name][taste_name]['data'][:,e_i]
							if len(dataset[~np.isnan(dataset)])>0:
								pop_vec_data_storage_collection.append(dataset)
								taste_used.append(taste_name)
						#Pairwise significance tests
						sig_ind_pairs = list(combinations(np.arange(len(taste_used)),2))
						sig_titles = 'i1,  i2,  KS*,  T*, 1v2'
						pop_vec_sig_pair_results = sig_titles
						for sip_i in range(len(sig_ind_pairs)):
							sip = sig_ind_pairs[sip_i]
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
						labels = [taste_used[i] + ' (' + str(i) + ')' for i in range(len(taste_used))]
						if (combo_lengths[2] == 1)*(combo_lengths[3] > 1):
							#pop_vec
							ax_pop_vec[i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_4].set_xlim([0,1.1])
							ax_pop_vec[i_4].set_ylim([0,1.1])
							ax_pop_vec[i_4].set_ylabel(ylabel)
							ax_pop_vec[i_4].set_xlabel(xlabel)
							ax_pop_vec[i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						elif (combo_lengths[2] > 1)*(combo_lengths[3] == 1):
							#pop_vec
							ax_pop_vec[i_3].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3].set_xlim([0,1.1])
							ax_pop_vec[i_3].set_ylim([0,1.1])
							ax_pop_vec[i_3].set_ylabel(ylabel)
							ax_pop_vec[i_3].set_xlabel(xlabel)
							ax_pop_vec[i_3].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						else:
							#pop_vec
							ax_pop_vec[i_3,i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=labels)
							ax_pop_vec[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3,i_4].set_xlim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylabel(ylabel)
							ax_pop_vec[i_3,i_4].set_xlabel(xlabel)
							ax_pop_vec[i_3,i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
				plt.figure(3)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop_vec'
				f_pop_vec.savefig(os.path.join(save_dir,f_pop_vec_plot_name) + '.png')
				f_pop_vec.savefig(os.path.join(save_dir,f_pop_vec_plot_name) + '.svg')
				plt.close(f_pop_vec) 
	

def cross_epoch(corr_data,save_dir,unique_given_names,unique_corr_names,\
			  unique_segment_names,unique_taste_names):
	"""This function collects statistics across different epochs and plots 
	them together
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
						pop_vec_data_storage_collection = []
						epoch_used = []
						corr_name = att.corr_name
						seg_name = att.seg_name
						taste_name = att.taste_name
						given_name = att.given_name
						for e_i in unique_epochs:
							dataset = unique_data_dict[corr_name][given_name][seg_name][taste_name]['data'][:,e_i]
							if len(dataset[~np.isnan(dataset)]) > 0:
								pop_vec_data_storage_collection.append(dataset)
								epoch_used.append('epoch ' + str(e_i))
						#Pairwise significance tests
						sig_ind_pairs = list(combinations(np.arange(len(epoch_used)),2))
						sig_titles = 'i1,  i2,  KS*,  T*, 1v2'
						pop_vec_sig_pair_results = sig_titles
						for sip_i in range(len(sig_ind_pairs)):
							sip = sig_ind_pairs[sip_i]
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
							#pop_vec
							ax_pop_vec[i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=epoch_used)
							ax_pop_vec[i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_4].set_xlim([0,1.1])
							ax_pop_vec[i_4].set_ylim([0,1.1])
							ax_pop_vec[i_4].set_ylabel(ylabel)
							ax_pop_vec[i_4].set_xlabel(xlabel)
							ax_pop_vec[i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						elif (combo_lengths[2] > 1)*(combo_lengths[3] == 1):
							#pop_vec
							ax_pop_vec[i_3].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=epoch_used)
							ax_pop_vec[i_3].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3].set_xlim([0,1.1])
							ax_pop_vec[i_3].set_ylim([0,1.1])
							ax_pop_vec[i_3].set_ylabel(ylabel)
							ax_pop_vec[i_3].set_xlabel(xlabel)
							ax_pop_vec[i_3].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
						else:
							#pop_vec
							ax_pop_vec[i_3,i_4].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=epoch_used)
							ax_pop_vec[i_3,i_4].legend(fontsize='12', loc ='lower right')
							ax_pop_vec[i_3,i_4].set_xlim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylim([0,1.1])
							ax_pop_vec[i_3,i_4].set_ylabel(ylabel)
							ax_pop_vec[i_3,i_4].set_xlabel(xlabel)
							ax_pop_vec[i_3,i_4].text(0.05, 1, pop_vec_sig_pair_results, fontsize=12, verticalalignment='top', bbox=txt_props)
				plt.figure(3)
				plt.suptitle(combo_1.replace(' ','_') + ' x ' + combo_2.replace(' ','_'))
				plt.tight_layout()
				f_pop_vec_plot_name = combo_1.replace(' ','_') + '_' + combo_2.replace(' ','_') + '_pop_vec'
				f_pop_vec.savefig(os.path.join(save_dir,f_pop_vec_plot_name) + '.png')
				f_pop_vec.savefig(os.path.join(save_dir,f_pop_vec_plot_name) + '.svg')
				plt.close(f_pop_vec) 
				
	
def int_input(prompt):
	"""
	This function asks a user for an integer input
	INPUTS:
		prompt = string containing boolean input prompt
	RETURNS:
		int_val = integer value
	NOTE:
		This function will continue to prompt the user for an answer until the 
		answer given is an integer.
	"""
	int_loop = 1	
	while int_loop == 1:
		response = input(prompt)
		try:
			int_val = int(response)
			int_loop = 0
		except:
			print("\tERROR: Incorrect data entry, please input an integer.")
	
	return int_val
	
def bool_input(prompt):
	"""
	This function asks a user for a boolean input of y/n.
	INPUTS:
		prompt = string containing boolean input prompt
	RETURNS:
		response = y / n
	NOTE:
		This function will continue to prompt the user for an answer until the 
		answer given is y or n.
	"""
	bool_loop = 1	
	while bool_loop == 1:
		response = input(prompt).lower()
		if (response == 'y') or (response == 'n'):
			bool_loop = 0
		else:
			print("\tERROR: Incorrect data entry, please try again with Y/y/N/n.")
	
	return response			
				
def int_list_input(prompt):
	"""
	This function asks a user for a list of integer inputs.
	INPUTS:
		prompt = string containing boolean input prompt
	RETURNS:
		int_list = list of integer values
	NOTE:
		This function will continue to prompt the user for an answer until the 
		answer given is a list of integers.
	"""
	int_loop = 1	
	while int_loop == 1:
		response = input(prompt)
		response_list = response.split(',')
		try:
			int_list = []
			for item in response_list:
				int_list.append(int(item))
			int_loop = 0
		except:
			print("\tERROR: Incorrect data entry, please input a list of integers.")
	
	return int_list
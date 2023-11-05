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

def cross_corr_name(data_dict,results_dir,unique_corr_names):
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
	 
	 #_____Reorganize data by unique correlation type_____
	 unique_corr_dict = dict()
	 max_segments = 0
	 segment_names = []
	 max_tastes = 0
	 taste_names = []
	 max_epochs = 0
	 for d_i in data_dict:
		 dataset = data_dict[d_i]
		 given_name = dataset['given_name']
		 corr_name = dataset['corr_name']
		 corr_dev_stats = dataset['corr_dev_stats']
		 num_seg = len(corr_dev_stats)
		 if num_seg > max_segments:
			 max_segments = num_seg
		 for s_i in range(num_seg):
			 num_tastes = len(corr_dev_stats[s_i])
			 if num_tastes > max_tastes:
				 max_tastes = num_tastes
			 avg_neur_data_storage_taste = [] #flattened arrays by epoch
			 pop_data_storage_taste = [] #flattened arrays by epoch
			 pop_vec_data_storage_taste = [] #flattened arrays by epoch
			 for t_i in range(num_tastes):
				 seg_name = corr_dev_stats[s_i][t_i]['segment'].replace('-','_').replace(' ','_')
				 if len(np.where(segment_names == seg_name)[0]) == 0:
					 segment_names.extend([seg_name])
				 try:
					 a = unique_corr_dict[seg_name]
				 except:
					 unique_corr_dict[seg_name] = dict()
				 taste_name = corr_dev_stats[s_i][t_i]['taste'].replace('-','_').replace(' ','_')
				 if len(np.where(taste_names == taste_name)[0]) == 0:
					 taste_names.extend([taste_name])
				 try:
					 a = unique_corr_dict[seg_name][taste_name]
				 except:
					 unique_corr_dict[seg_name][taste_name] = dict()
				 try:
					 a = unique_corr_dict[seg_name][taste_name][corr_name]
				 except:
					 unique_corr_dict[seg_name][taste_name][corr_name] = dict()
				 try:
					 a = unique_corr_dict[seg_name][taste_name][corr_name][given_name]
				 except:
					 unique_corr_dict[seg_name][taste_name][corr_name][given_name] = dict()
				 avg_taste_neur_data = np.nanmean(corr_dev_stats[s_i][t_i]['neuron_data_storage'],2)
				 unique_corr_dict[seg_name][taste_name][corr_name][given_name]['avg_neuron_data_storage'] = avg_taste_neur_data.reshape(-1,avg_taste_neur_data.shape[-1])
				 pop_taste_data = corr_dev_stats[s_i][t_i]['pop_data_storage']
				 num_epochs = pop_taste_data.shape[-1]
				 if num_epochs > max_epochs:
					 max_epochs = num_epochs
				 unique_corr_dict[seg_name][taste_name][corr_name][given_name]['pop_data_storage'] = pop_taste_data.reshape(-1,pop_taste_data.shape[-1])
				 pop_vec_taste_data = corr_dev_stats[s_i][t_i]['pop_vec_data_storage']
				 unique_corr_dict[seg_name][taste_name][corr_name][given_name]['pop_vec_data_storage'] = pop_vec_taste_data.reshape(-1,pop_taste_data.shape[-1])
	 
	 #Cross-data cross-corr plotting still needs to be by segment, by taste, by epoch:
		 # subplots of segment x taste for each epoch with cross-data on same axes
		 # subplots of segment x epoch for each taste with cross-data on same axes
		 # subplots of taste x epoch for each segment with cross-data on same axes
	 #So cross-data datasets should be stored in a nested list structure of segment x taste x epoch
	 #These datasets further need to be broken up by avg(neuron_data_storage),pop_data_storage, and pop_vec_data_storage
	 
	 
	 #Plot 1: Figure by epoch with cross-data on same axes and subplots of segment x taste
	 for e_i in range(max_epochs): #For each epoch
		 f_avg_neur, ax_avg_neur = plt.subplots(nrows = max_segments, ncols = max_tastes, figsize=(max_tastes*4,max_segments*4))
		 f_pop, ax_pop = plt.subplots(nrows = max_segments, ncols = max_tastes, figsize=(max_tastes*4,max_segments*4))
		 f_pop_vec, ax_pop_vec = plt.subplots(nrows = max_segments, ncols = max_tastes, figsize=(max_tastes*4,max_segments*4))
		 seg_names = list(unique_corr_dict.keys())
		 for s_i in range(len(seg_names)):
			 seg_name = seg_names[s_i]
			 taste_names = list(unique_corr_dict[seg_name].keys())
			 for t_i in range(len(taste_names)):
				 taste_name = taste_names[t_i]
				 corr_names = list(unique_corr_dict[seg_name][taste_name].keys())
				 avg_neuron_data_storage_collection = []
				 pop_data_storage_collection = []
				 pop_vec_data_storage_collection = []
				 data_names = []
				 for c_i in range(len(corr_names)):
					 corr_name = corr_names[c_i]
					 given_names = list(unique_corr_dict[seg_name][taste_name][corr_name].keys())
					 for g_i in range(len(given_names)):
						 given_name = given_names[g_i]
						 data_names.extend([given_name + '_' + corr_name])
						 avg_neuron_data_storage_collection.append(unique_corr_dict[seg_name][taste_name][corr_name][given_name]['avg_neuron_data_storage'][:,e_i])
						 pop_data_storage_collection.append(unique_corr_dict[seg_name][taste_name][corr_name][given_name]['pop_data_storage'][:,e_i])
						 pop_vec_data_storage_collection.append(unique_corr_dict[seg_name][taste_name][corr_name][given_name]['pop_vec_data_storage'][:,e_i])
				 ax_avg_neur[s_i,t_i].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
				 ax_avg_neur[s_i,t_i].legend(fontsize='12', loc ='lower right')
				 ax_avg_neur[s_i,t_i].set_xlim([0,1.1])
				 ax_avg_neur[s_i,t_i].set_ylim([0,1.1])
				 ax_avg_neur[s_i,t_i].set_ylabel(seg_name)
				 ax_avg_neur[s_i,t_i].set_xlabel(taste_name)
				 ax_pop[s_i,t_i].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
				 ax_pop[s_i,t_i].legend(fontsize='12', loc ='lower right')
				 ax_pop[s_i,t_i].set_xlim([0,1.1])
				 ax_pop[s_i,t_i].set_ylim([0,1.1])
				 ax_pop[s_i,t_i].set_ylabel(seg_name)
				 ax_pop[s_i,t_i].set_xlabel(taste_name)
				 ax_pop_vec[s_i,t_i].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
				 ax_pop_vec[s_i,t_i].legend(fontsize='12', loc ='lower right')
				 ax_pop_vec[s_i,t_i].set_xlim([0,1.1])
				 ax_pop_vec[s_i,t_i].set_ylim([0,1.1])
				 ax_pop_vec[s_i,t_i].set_ylabel(seg_name)
				 ax_pop_vec[s_i,t_i].set_xlabel(taste_name)
		 plt.figure(1)
		 plt.tight_layout()
		 plt.suptitle('Epoch ' + str(e_i))
		 f_avg_neur_plot_name = 'epoch_' + str(e_i) + '_avg_neuron'
		 f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.png')
		 f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.svg')
		 plt.close(f_avg_neur)
		 plt.figure(2)
		 plt.tight_layout()
		 plt.suptitle('Epoch ' + str(e_i))
		 f_pop_plot_name = 'epoch_' + str(e_i) + '_pop'
		 f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.png')
		 f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.svg')
		 plt.close(f_pop)
		 plt.figure(3)
		 plt.tight_layout()
		 plt.suptitle('Epoch ' + str(e_i))
		 f_pop_vec_plot_name = 'epoch_' + str(e_i) + '_pop_vec'
		 f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.png')
		 f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.svg')
		 plt.close(f_pop_vec)
		 
	 #Plot 2: Figure by taste with cross-data on same axes and subplots of segment x epoch
	 for taste_name in taste_names: #For each taste
		 f_avg_neur, ax_avg_neur = plt.subplots(nrows = max_segments, ncols = max_epochs, figsize=(max_epochs*4,max_segments*4))
		 f_pop, ax_pop = plt.subplots(nrows = max_segments, ncols = max_epochs, figsize=(max_epochs*4,max_segments*4))
		 f_pop_vec, ax_pop_vec = plt.subplots(nrows = max_segments, ncols = max_epochs, figsize=(max_epochs*4,max_segments*4))
		 seg_names = list(unique_corr_dict.keys())
		 for s_i in range(len(seg_names)):
			 seg_name = seg_names[s_i]
			 corr_names = list(unique_corr_dict[seg_name][taste_name].keys())
			 for e_i in range(max_epochs):
				 avg_neuron_data_storage_collection = []
				 pop_data_storage_collection = []
				 pop_vec_data_storage_collection = []
				 data_names = []
				 for c_i in range(len(corr_names)):
					 corr_name = corr_names[c_i]
					 given_names = list(unique_corr_dict[seg_name][taste_name][corr_name].keys())
					 for g_i in range(len(given_names)):
						 given_name = given_names[g_i]
						 data_names.extend([given_name + '_' + corr_name])
						 avg_neuron_data_storage_collection.append(unique_corr_dict[seg_name][taste_name][corr_name][given_name]['avg_neuron_data_storage'][:,e_i])
						 pop_data_storage_collection.append(unique_corr_dict[seg_name][taste_name][corr_name][given_name]['pop_data_storage'][:,e_i])
						 pop_vec_data_storage_collection.append(unique_corr_dict[seg_name][taste_name][corr_name][given_name]['pop_vec_data_storage'][:,e_i])
				 ax_avg_neur[s_i,e_i].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
				 ax_avg_neur[s_i,e_i].legend(fontsize='12', loc ='lower right')
				 ax_avg_neur[s_i,e_i].set_xlim([0,1.1])
				 ax_avg_neur[s_i,e_i].set_ylim([0,1.1])
				 ax_avg_neur[s_i,e_i].set_ylabel(seg_name)
				 ax_avg_neur[s_i,e_i].set_xlabel('Epoch ' + str(e_i))
				 ax_pop[s_i,e_i].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
				 ax_pop[s_i,e_i].legend(fontsize='12', loc ='lower right')
				 ax_pop[s_i,e_i].set_xlim([0,1.1])
				 ax_pop[s_i,e_i].set_ylim([0,1.1])
				 ax_pop[s_i,e_i].set_ylabel(seg_name)
				 ax_pop[s_i,e_i].set_xlabel('Epoch ' + str(e_i))
				 ax_pop_vec[s_i,e_i].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
				 ax_pop_vec[s_i,e_i].legend(fontsize='12', loc ='lower right')
				 ax_pop_vec[s_i,e_i].set_xlim([0,1.1])
				 ax_pop_vec[s_i,e_i].set_ylim([0,1.1])
				 ax_pop_vec[s_i,e_i].set_ylabel(seg_name)
				 ax_pop_vec[s_i,e_i].set_xlabel('Epoch ' + str(e_i))
		 plt.figure(1)
		 plt.tight_layout()
		 plt.suptitle('Taste ' + taste_name)
		 f_avg_neur_plot_name = 'taste_' + taste_name.replace(' ','_') + '_avg_neuron'
		 f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.png')
		 f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.svg')
		 plt.close(f_avg_neur)
		 plt.figure(2)
		 plt.tight_layout()
		 plt.suptitle('Taste ' + taste_name)
		 f_pop_plot_name = 'taste_' + taste_name.replace(' ','_') + '_pop'
		 f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.png')
		 f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.svg')
		 plt.close(f_pop)
		 plt.figure(3)
		 plt.tight_layout()
		 plt.suptitle('Taste ' + taste_name)
		 f_pop_vec_plot_name = 'taste_' + taste_name.replace(' ','_') + '_pop_vec'
		 f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.png')
		 f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.svg')
		 plt.close(f_pop_vec)
		 
	 #Plot 3: Figure by segment with cross-data on same axes and subplots of taste x epoch
	 for seg_name in segment_names: #For each taste
		 f_avg_neur, ax_avg_neur = plt.subplots(nrows = max_tastes, ncols = max_epochs, figsize=(max_epochs*4,max_tastes*4))
		 f_pop, ax_pop = plt.subplots(nrows = max_tastes, ncols = max_epochs, figsize=(max_epochs*4,max_tastes*4))
		 f_pop_vec, ax_pop_vec = plt.subplots(nrows = max_tastes, ncols = max_epochs, figsize=(max_epochs*4,max_tastes*4))
		 taste_names = list(unique_corr_dict[seg_name].keys())
		 for t_i in range(len(taste_names)):
			 taste_name = taste_names[t_i]
			 corr_names = list(unique_corr_dict[seg_name][taste_name].keys())
			 for e_i in range(max_epochs):
				 avg_neuron_data_storage_collection = []
				 pop_data_storage_collection = []
				 pop_vec_data_storage_collection = []
				 data_names = []
				 for c_i in range(len(corr_names)):
					 corr_name = corr_names[c_i]
					 given_names = list(unique_corr_dict[seg_name][taste_name][corr_name].keys())
					 for g_i in range(len(given_names)):
						 given_name = given_names[g_i]
						 data_names.extend([given_name + '_' + corr_name])
						 avg_neuron_data_storage_collection.append(unique_corr_dict[seg_name][taste_name][corr_name][given_name]['avg_neuron_data_storage'][:,e_i])
						 pop_data_storage_collection.append(unique_corr_dict[seg_name][taste_name][corr_name][given_name]['pop_data_storage'][:,e_i])
						 pop_vec_data_storage_collection.append(unique_corr_dict[seg_name][taste_name][corr_name][given_name]['pop_vec_data_storage'][:,e_i])
				 ax_avg_neur[t_i,e_i].hist(avg_neuron_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
				 ax_avg_neur[t_i,e_i].legend(fontsize='12', loc ='lower right')
				 ax_avg_neur[t_i,e_i].set_xlim([0,1.1])
				 ax_avg_neur[t_i,e_i].set_ylim([0,1.1])
				 ax_avg_neur[t_i,e_i].set_ylabel(taste_name)
				 ax_avg_neur[t_i,e_i].set_xlabel('Epoch ' + str(e_i))
				 ax_pop[t_i,e_i].hist(pop_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
				 ax_pop[t_i,e_i].legend(fontsize='12', loc ='lower right')
				 ax_pop[t_i,e_i].set_xlim([0,1.1])
				 ax_pop[t_i,e_i].set_ylim([0,1.1])
				 ax_pop[t_i,e_i].set_ylabel(taste_name)
				 ax_pop[t_i,e_i].set_xlabel('Epoch ' + str(e_i))
				 ax_pop_vec[t_i,e_i].hist(pop_vec_data_storage_collection,bins=1000,histtype='step',density=True,cumulative=True,label=data_names)
				 ax_pop_vec[t_i,e_i].legend(fontsize='12', loc ='lower right')
				 ax_pop_vec[t_i,e_i].set_xlim([0,1.1])
				 ax_pop_vec[t_i,e_i].set_ylim([0,1.1])
				 ax_pop_vec[t_i,e_i].set_ylabel(taste_name)
				 ax_pop_vec[t_i,e_i].set_xlabel('Epoch ' + str(e_i))
		 plt.figure(1)
		 plt.tight_layout()
		 plt.suptitle('Segment ' + seg_name)
		 f_avg_neur_plot_name = 'segment_' + seg_name.replace(' ','_') + '_avg_neuron'
		 f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.png')
		 f_avg_neur.savefig(os.path.join(results_dir,f_avg_neur_plot_name) + '.svg')
		 plt.close(f_avg_neur)
		 plt.figure(2)
		 plt.tight_layout()
		 plt.suptitle('Segment ' + seg_name)
		 f_pop_plot_name = 'segment_' + seg_name.replace(' ','_') + '_pop'
		 f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.png')
		 f_pop.savefig(os.path.join(results_dir,f_pop_plot_name) + '.svg')
		 plt.close(f_pop)
		 plt.figure(3)
		 plt.tight_layout()
		 plt.suptitle('Segment ' + seg_name)
		 f_pop_vec_plot_name = 'segment_' + seg_name.replace(' ','_') + '_pop_vec'
		 f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.png')
		 f_pop_vec.savefig(os.path.join(results_dir,f_pop_vec_plot_name) + '.svg')
		 plt.close(f_pop_vec)
		 
						 






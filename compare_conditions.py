#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 10:48:05 2023

@author: Hannah Germaine
Compare results across conditions in plots and with stats
"""

def int_input(prompt):
	#This function asks a user for an integer input
	int_loop = 1	
	while int_loop == 1:
		response = input(prompt)
		try:
			int_val = int(response)
			int_loop = 0
		except:
			print("\tERROR: Incorrect data entry, please input an integer.")
	
	return int_val

if __name__ == '__main__':

	import os, easygui
	import numpy as np
	import functions.analysis_funcs as af
	import functions.dev_funcs as df
	import functions.hdf5_handling as hf5
	import functions.compare_conditions_funcs as ccf
	
	#_____Prompt user for the number of datasets needed in the analysis_____
	num_cond = int_input("How many animals-worth of correlation data do you wish to import for this comparative analysis (integer value)? ")
	if num_cond >= 1:
		print("Multiple file import selected.")
	else:
		print("Single file import selected.")

	#_____Pull all data into a dictionary_____
	data_dict = dict()
	data_ind = 0
	for nc in range(num_cond):
		cond_dict = dict()
		#_____Get the directory of the hdf5 file_____
		print("Please select the folder where the data # " + str(nc+1) + " is stored.")
		sorted_dir, segment_dir, cleaned_dir = hf5.sorted_data_import() #Program will automatically quit if file not found in given folder
		fig_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'
		data_name = fig_save_dir.split('/')[-2]
		print("Give a more colloquial name to the dataset.")
		given_name = input("How would you rename " + data_name + "? ")
		
		#_____Import relevant data_____
		num_neur, _, _, dig_in_names, segment_times, segment_names, _, _, _ = af.import_data(sorted_dir, segment_dir, fig_save_dir)
		data_group_name = 'taste_selectivity'
		taste_select_prob_epoch = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_select_prob_epoch')[0]
		
		#_____Establish storage directories and import correlation results_____
		#All directories must already exist from prior calculations
		comp_dir = fig_save_dir + 'dev_x_taste/'
		corr_dir = comp_dir + 'corr/'
		#Find the number of correlation calculation directories are within this directory
		num_corr_types = os.listdir(corr_dir)
		for nct_i in range(len(num_corr_types)):
			nct = num_corr_types[nct_i]
			result_dir = corr_dir + nct + '/'
			try:
				corr_dev_stats = df.pull_corr_dev_stats(segment_names, dig_in_names, result_dir)
				corr_dict = dict()
				corr_dict['fig_save_dir'] = fig_save_dir
				corr_dict['data_name'] = data_name
				corr_dict['given_name'] = given_name
				corr_dict['taste_select_prob_epoch'] = taste_select_prob_epoch
				corr_dict['corr_name'] = nct
				corr_dict['corr_dev_stats'] = corr_dev_stats
				data_dict[data_ind] = corr_dict
				data_ind += 1
			except:
				print("No data in directory " + result_dir)
			
	#_____Analysis Storage Directory_____
	print('Please select a directory to save all results from this set of analyses.')
	#DO NOT use the same directory as where the correlation analysis results are stored
	results_dir = easygui.diropenbox(title='Please select the storage folder.')
		
	#_____Grab Unique Groupings_____
	num_datasets = len(data_dict)
	unique_given_names = [data_dict[i]['given_name'] for i in range(num_datasets)] #How many datasets
	unique_given_indices = np.sort(np.unique(unique_given_names, return_index=True)[1])
	unique_given_names = [unique_given_names[i] for i in unique_given_indices]
	unique_corr_names = [data_dict[i]['corr_name'] for i in range(num_datasets)] #How many types of correlation analyses
	unique_corr_indices = np.sort(np.unique(unique_corr_names, return_index=True)[1])
	unique_corr_names = [unique_corr_names[i] for i in unique_corr_indices]
	unique_segment_names = []
	unique_taste_names = []
	for d_i in range(num_datasets):
		corr_dev_stats = data_dict[d_i]['corr_dev_stats']
		num_seg = len(corr_dev_stats)
		for ns in range(num_seg):
			unique_segment_names.append(corr_dev_stats[ns][0]['segment'])
			num_tastes = len(corr_dev_stats[ns])
			for nt in range(num_tastes):
				unique_taste_names.append(corr_dev_stats[ns][nt]['taste'])
	unique_segment_indices = np.sort(np.unique(unique_segment_names, return_index=True)[1])
	unique_segment_names = [unique_segment_names[i] for i in unique_segment_indices]
	unique_taste_indices = np.sort(np.unique(unique_taste_names, return_index=True)[1])
	unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
	
	#_____Plot Results Across Conditions_____
	print("Beginning Plots.")
	if num_cond > 1:
		#Cross-Dataset: different given names on the same axes
		cross_data_dir = os.path.join(results_dir,'cross_data_plots')
		if os.path.isdir(cross_data_dir) == False:
			os.mkdir(cross_data_dir)
		print("\tCross Dataset Plots.")
		ccf.cross_data(data_dict,cross_data_dir,unique_given_names,unique_corr_names,unique_segment_names,unique_taste_names)
		
	else:
		#Cross-Corr: all neur, taste selective, all neuron z-score, and taste selective z-score on same axes
		cross_corr_dir = os.path.join(results_dir,'cross_corr_plots')
		if os.path.isdir(cross_corr_dir) == False:
			os.mkdir(cross_corr_dir)
		print("\tCross Condition Plots.")
		ccf.cross_corr_name(data_dict,cross_corr_dir,unique_given_names,unique_corr_names,\
					  unique_segment_names,unique_taste_names)
		
		#Cross-Segment: different segments on the same axes
		cross_segment_dir = os.path.join(results_dir,'cross_segment_plots')
		if os.path.isdir(cross_segment_dir) == False:
			os.mkdir(cross_segment_dir)
		print("\tCross Segment Plots.")
		ccf.cross_segment(data_dict,cross_segment_dir,unique_given_names,unique_corr_names,unique_segment_names,unique_taste_names)
		
		#Cross-Taste: different tastes on the same axes
		cross_taste_dir = os.path.join(results_dir,'cross_taste_plots')
		if os.path.isdir(cross_taste_dir) == False:
			os.mkdir(cross_taste_dir)
		print("\tCross Taste Plots.")
		ccf.cross_taste(data_dict,cross_taste_dir,unique_given_names,unique_corr_names,unique_segment_names,unique_taste_names)
		
		#Cross-Epoch: different epochs on the same axes
		cross_epoch_dir = os.path.join(results_dir,'cross_epoch_plots')
		if os.path.isdir(cross_epoch_dir) == False:
			os.mkdir(cross_epoch_dir)
		print("\tCross Epoch Plots.")
		ccf.cross_epoch(data_dict,cross_epoch_dir,unique_given_names,unique_corr_names,unique_segment_names,unique_taste_names)
	
	print("Done.")
		
	
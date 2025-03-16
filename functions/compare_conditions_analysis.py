#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:18:06 2024

@author: Hannah Germaine

Functions to support compare_conditions.py in running cross-dataset analyses.
"""

import os
import warnings
import pickle
import numpy as np
import functions.hdf5_handling as hf5
from tkinter.filedialog import askdirectory

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.compare_datasets_funcs as cdf
import functions.compare_conditions_funcs as ccf

warnings.filterwarnings("ignore")


class run_compare_conditions_analysis():

    def __init__(self, args):
        self.all_data_dict = args[0]
        self.save_dir = args[1]
        #Import/Load data
        if len(self.save_dir) > 0:
            try:
                self.import_corr()
            except:
                self.gather_corr_data()
            try:
                self.import_seg_data()
            except:
                self.gather_seg_data()
            try:
                self.import_cp_data()
            except:
                self.gather_cp_data()
            try:
                self.import_rate_corr_data()
            except:
                self.gather_rate_corr_data()
            try:
                self.import_dev_stats()
            except:
                self.gather_dev_stats_data()
            try:
                self.import_dev_null_stats()
            except:
                self.gather_dev_null_data()
            try:
                self.import_dev_split_corr_data()
            except:
                self.gather_dev_split_data()
            try:
                self.import_dev_decode_data()
            except:
                self.gather_dev_decode_data()
            try:
                self.import_dev_split_decode_data()
            except:
                self.gather_dev_split_decode_data()
        else:
            print("Please select a storage folder for results.")
            print('Please select the storage folder.')
            self.save_dir = askdirectory()
            np.save(os.path.join(self.save_dir,'all_data_dict.npy'),\
                    self.all_data_dict,allow_pickle=True)
            self.gather_corr_data()
            self.gather_seg_data()
            self.gather_cp_data()
            self.gather_rate_corr_data()
            self.gather_dev_stats_data()
            self.gather_dev_null_data()
            self.gather_dev_split_data()
            self.gather_dev_decode_data()
            self.gather_dev_split_decode_data()
        #Correlation comparisons
        self.find_corr_groupings()
        self.plot_corr_results()
        #Segment comparisons
        self.find_seg_groupings()
        self.plot_seg_results()
        #Changepoint comparisons
        self.find_cp_groupings()
        self.plot_cp_results()
        #Pop Rate x Taste Corr comparisons
        self.find_rate_corr_groupings()
        self.plot_rate_corr_results()
        #Deviation Statistic comparisons
        self.find_dev_stats_groupings()
        self.plot_dev_stat_results()
        #Deviation True x Null comparisons
        self.find_dev_null_groupings()
        self.plot_dev_null_results()
        #Sliding bin decoding results
        self.find_dev_split_corr_groupings()
        self.plot_dev_split_results()
        #Dev decode results
        self.find_dev_decode_groupings()
        self.plot_dev_decode_results()
        #Dev split decode results
        self.find_dev_split_decode_groupings()
        self.plot_dev_split_decode_results()

    def import_corr(self,):
        """Import previously saved correlation data"""
        dict_save_dir = os.path.join(self.save_dir, 'corr_data.npy')
        corr_data = np.load(dict_save_dir,allow_pickle=True).item()
        self.corr_data = corr_data
        if not os.path.isdir(os.path.join(self.save_dir,'Correlations')):
            os.mkdir(os.path.join(self.save_dir,'Correlations'))
        self.corr_results_dir = os.path.join(self.save_dir,'Correlations')

    def gather_corr_data(self,):
        """Import the relevant data from each dataset to be analyzed. This 
        includes the number of neurons, segments to analyze, segment names, 
        segment start and end times, taste dig in names, and the correlation
        data for all neurons and taste-selective neurons"""

        num_datasets = len(self.all_data_dict)
        dataset_names = list(self.all_data_dict.keys())
        corr_data = dict()
        for n_i in range(num_datasets):
            data_name = dataset_names[n_i]
            data_dict = self.all_data_dict[data_name]['data']
            metadata = self.all_data_dict[data_name]['metadata']
            data_save_dir = data_dict['data_path']
            dev_corr_save_dir = os.path.join(
                data_save_dir, 'dev_x_taste', 'corr')
            num_corr_types = os.listdir(dev_corr_save_dir)
            corr_data[data_name] = dict()
            corr_data[data_name]['num_neur'] = data_dict['num_neur']
            segments_to_analyze = metadata['params_dict']['segments_to_analyze']
            corr_data[data_name]['segments_to_analyze'] = segments_to_analyze
            corr_data[data_name]['segment_names'] = data_dict['segment_names']
            segment_times = data_dict['segment_times']
            num_segments = len(corr_data[data_name]['segment_names'])
            corr_data[data_name]['segment_times_reshaped'] = [
                [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
            dig_in_names = data_dict['dig_in_names']
            corr_data[data_name]['dig_in_names'] = dig_in_names
            corr_data[data_name]['corr_data'] = dict()
            for nct_i in range(len(num_corr_types)):
                nct = num_corr_types[nct_i]
                if nct[0] != '.':
                    result_dir = os.path.join(dev_corr_save_dir, nct)
                    corr_data[data_name]['corr_data'][nct] = dict()
                    for s_i in segments_to_analyze:
                        seg_name = corr_data[data_name]['segment_names'][s_i]
                        corr_data[data_name]['corr_data'][nct][seg_name] = dict()
                        filename_best_corr = os.path.join(result_dir,seg_name + '_best_taste_epoch_array.npy')
                        best_data = np.load(filename_best_corr)
                        corr_data[data_name]['corr_data'][nct][seg_name]['best'] = best_data
                        for t_i in range(len(dig_in_names)):
                            taste_name = dig_in_names[t_i]
                            corr_data[data_name]['corr_data'][nct][seg_name][taste_name] = dict(
                            )
                            try:
                                filename_corr_pop_vec = os.path.join(
                                    result_dir, seg_name + '_' + taste_name + '_pop_vec.npy')
                                data = np.load(filename_corr_pop_vec)
                                corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['data'] = data
                                num_dev, num_deliv, num_cp = np.shape(data)
                                corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['num_dev'] = num_dev
                                corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['num_deliv'] = num_deliv
                                corr_data[data_name]['corr_data'][nct][seg_name][taste_name]['num_cp'] = num_cp
                            except:
                                print("No data in directory " + result_dir)
        self.corr_data = corr_data
        dict_save_dir = os.path.join(self.save_dir, 'corr_data.npy')
        np.save(dict_save_dir,corr_data,allow_pickle=True)
        # Save the combined dataset somewhere...
        # _____Analysis Storage Directory_____
        if not os.path.isdir(os.path.join(self.save_dir,'Correlations')):
            os.mkdir(os.path.join(self.save_dir,'Correlations'))
        self.corr_results_dir = os.path.join(self.save_dir,'Correlations')

    def find_corr_groupings(self,):
        """Across the different datasets, get the unique data names/indices,
        correlation combinations and names/indices, unique segment names/indices,
        and unique taste names/indices to align datasets to each other in these
        different groups."""

        corr_data = self.corr_data
        unique_given_names = list(corr_data.keys())
        unique_given_indices = np.sort(
            np.unique(unique_given_names, return_index=True)[1])
        unique_given_names = [unique_given_names[i]
                              for i in unique_given_indices]
        unique_corr_names = []
        for name in unique_given_names:
            unique_corr_names.extend(list(corr_data[name]['corr_data'].keys()))
        unique_corr_names = np.array(unique_corr_names)
        unique_corr_indices = np.sort(
            np.unique(unique_corr_names, return_index=True)[1])
        unique_corr_names = [unique_corr_names[i] for i in unique_corr_indices]
        unique_segment_names = []
        unique_taste_names = []
        for name in unique_given_names:
            for corr_name in unique_corr_names:
                try:
                    seg_names = list(
                        corr_data[name]['corr_data'][corr_name].keys())
                    unique_segment_names.extend(seg_names)
                    for seg_name in seg_names:
                        taste_names = list(np.setdiff1d(list(
                            corr_data[name]['corr_data'][corr_name][seg_name].keys()),['best']))
                        unique_taste_names.extend(taste_names)
                except:
                    print(name + " does not have correlation data for " + corr_name)
        unique_segment_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i]
                                for i in unique_segment_indices]
        unique_taste_indices = np.sort(
            np.unique(unique_taste_names, return_index=True)[1])
        unique_taste_names = [unique_taste_names[i]
                              for i in unique_taste_indices]

        self.unique_given_names = unique_given_names
        self.unique_corr_names = unique_corr_names
        self.unique_segment_names = unique_segment_names
        self.unique_taste_names = unique_taste_names

    def plot_corr_results(self,):
        num_cond = len(self.corr_data)
        results_dir = self.corr_results_dir

        print("Beginning Plots.")
        if num_cond > 1:
            # ____Deviation Event Frequencies____
            dev_freq_dir = os.path.join(results_dir, 'dev_frequency_plots')
            if os.path.isdir(dev_freq_dir) == False:
                os.mkdir(dev_freq_dir)
            print("\tCalculating Cross-Segment Deviation Frequencies")
            cdf.cross_dataset_dev_freq(self.corr_data, self.unique_given_names,
                                             self.unique_corr_names, self.unique_segment_names,
                                             self.unique_taste_names, dev_freq_dir)
            # ____Correlation Distributions____
            cross_segment_dir = os.path.join(
                results_dir, 'cross_segment_plots')
            if os.path.isdir(cross_segment_dir) == False:
                os.mkdir(cross_segment_dir)
            print("\tComparing Segments")
            cdf.cross_segment_diffs(self.corr_data, cross_segment_dir, self.unique_given_names,
                                    self.unique_corr_names, self.unique_segment_names, self.unique_taste_names)
            cdf.combined_corr_by_segment_dist(self.corr_data, cross_segment_dir, self.unique_given_names, 
                                              self.unique_corr_names,self.unique_segment_names, self.unique_taste_names)
            cross_taste_dir = os.path.join(results_dir, 'cross_taste_plots')
            if os.path.isdir(cross_taste_dir) == False:
                os.mkdir(cross_taste_dir)
            print("\tComparing Tastes")
            cdf.cross_taste_diffs(self.corr_data, cross_taste_dir, self.unique_given_names,
                                  self.unique_corr_names, self.unique_segment_names, self.unique_taste_names)
            cdf.combined_corr_by_taste_dist(self.corr_data, cross_taste_dir, self.unique_given_names, 
                                              self.unique_corr_names,self.unique_segment_names, self.unique_taste_names)
            cross_epoch_dir = os.path.join(results_dir, 'cross_epoch_plots')
            if os.path.isdir(cross_epoch_dir) == False:
                os.mkdir(cross_epoch_dir)
            print("\tComparing Epochs")
            cdf.cross_epoch_diffs(self.corr_data, cross_epoch_dir, self.unique_given_names,
                                  self.unique_corr_names, self.unique_segment_names, self.unique_taste_names)
        else:
            # Cross-Corr: all neur, taste selective, all neuron z-score, and taste selective z-score on same axes
            cross_corr_dir = os.path.join(results_dir, 'cross_corr_plots')
            if os.path.isdir(cross_corr_dir) == False:
                os.mkdir(cross_corr_dir)
            print("\tCross Condition Plots.")
            ccf.cross_corr_name(self.corr_data, cross_corr_dir, self.unique_given_names,
                                self.unique_corr_names, self.unique_segment_names, self.unique_taste_names)

            # Cross-Segment: different segments on the same axes
            cross_segment_dir = os.path.join(
                results_dir, 'cross_segment_plots')
            if os.path.isdir(cross_segment_dir) == False:
                os.mkdir(cross_segment_dir)
            print("\tCross Segment Plots.")
            ccf.cross_segment(self.corr_data, cross_segment_dir, self.unique_given_names,
                              self.unique_corr_names, self.unique_segment_names, self.unique_taste_names)

            # Cross-Taste: different tastes on the same axes
            cross_taste_dir = os.path.join(results_dir, 'cross_taste_plots')
            if os.path.isdir(cross_taste_dir) == False:
                os.mkdir(cross_taste_dir)
            print("\tCross Taste Plots.")
            ccf.cross_taste(self.corr_data, cross_taste_dir, self.unique_given_names,
                            self.unique_corr_names, self.unique_segment_names, self.unique_taste_names)

            # Cross-Epoch: different epochs on the same axes
            cross_epoch_dir = os.path.join(results_dir, 'cross_epoch_plots')
            if os.path.isdir(cross_epoch_dir) == False:
                os.mkdir(cross_epoch_dir)
            print("\tCross Epoch Plots.")
            ccf.cross_epoch(self.corr_data, cross_epoch_dir, self.unique_given_names,
                            self.unique_corr_names, self.unique_segment_names, self.unique_taste_names)

    def import_seg_data(self,):
        """Import previously saved segment data"""
        dict_save_dir = os.path.join(self.save_dir, 'seg_data.npy')
        seg_data = np.load(dict_save_dir,allow_pickle=True).item()
        self.seg_data = seg_data
        if not os.path.isdir(os.path.join(self.save_dir,'Segment_Comparison')):
            os.mkdir(os.path.join(self.save_dir,'Segment_Comparison'))
        self.seg_results_dir = os.path.join(self.save_dir,'Segment_Comparison')
        
    def gather_seg_data(self,):
        """Import the relevant data from each dataset to be analyzed. This 
        includes the number of neurons, segments to analyze, segment names, 
        segment start and end times, taste dig in names, and the segment
        statistics data"""

        num_datasets = len(self.all_data_dict)
        dataset_names = list(self.all_data_dict.keys())
        seg_data = dict()
        for n_i in range(num_datasets):
            data_name = dataset_names[n_i]
            data_dict = self.all_data_dict[data_name]['data']
            metadata = self.all_data_dict[data_name]['metadata']
            data_save_dir = data_dict['data_path']
            seg_save_dir = os.path.join(data_save_dir,'Segment_Comparison','indiv_distributions')
            num_dicts = os.listdir(seg_save_dir)
            seg_data[data_name] = dict()
            seg_data[data_name]['num_neur'] = data_dict['num_neur']
            segments_to_analyze = metadata['params_dict']['segments_to_analyze']
            seg_data[data_name]['segments_to_analyze'] = segments_to_analyze
            seg_data[data_name]['segment_names'] = data_dict['segment_names']
            segment_times = data_dict['segment_times']
            num_segments = len(seg_data[data_name]['segment_names'])
            seg_data[data_name]['segment_times_reshaped'] = [
                [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
            dig_in_names = data_dict['dig_in_names']
            seg_data[data_name]['dig_in_names'] = dig_in_names
            seg_data[data_name]['seg_data'] = dict()
            for nd_i in range(len(num_dicts)):
                dict_name = num_dicts[nd_i]
                dict_dir = os.path.join(seg_save_dir,dict_name)
                seg_data[data_name]['seg_data'][dict_name.split('.npy')[0]] = \
                    np.load(dict_dir,allow_pickle=True).item()
                #This data is organized by [seg_name][bin_size] gives the result array
        self.seg_data = seg_data
        np.save(os.path.join(self.save_dir, 'seg_data.npy'),seg_data,allow_pickle=True)
        # Save the combined dataset somewhere...
        # _____Analysis Storage Directory_____
        if not os.path.isdir(os.path.join(self.save_dir,'Segment_Comparison')):
            os.mkdir(os.path.join(self.save_dir,'Segment_Comparison'))
        self.seg_results_dir = os.path.join(self.save_dir,'Segment_Comparison')
        
    def find_seg_groupings(self,):
        """Across the different datasets, get the unique data names/indices,
        correlation combinations and names/indices, unique segment names/indices,
        and unique taste names/indices to align datasets to each other in these
        different groups."""

        seg_data = self.seg_data
        unique_given_names = list(seg_data.keys())
        unique_given_indices = np.sort(
            np.unique(unique_given_names, return_index=True)[1])
        unique_given_names = [unique_given_names[i]
                              for i in unique_given_indices]
        unique_analysis_names = np.array([list(seg_data[name]['seg_data'].keys(
        )) for name in unique_given_names]).flatten()  # How many types of segment analyses
        unique_analysis_indices = np.sort(
            np.unique(unique_analysis_names, return_index=True)[1])
        unique_analysis_names = [unique_analysis_names[i] for i in unique_analysis_indices]
        unique_segment_names = []
        unique_bin_sizes = []
        for name in unique_given_names:
            for analysis_name in unique_analysis_names:
                try:
                    seg_to_analyze = seg_data[name]['segments_to_analyze']
                    segment_names = np.array(seg_data[name]['segment_names'])
                    seg_names = list(
                        segment_names[seg_to_analyze])
                    unique_segment_names.extend(seg_names)
                    for seg_name in seg_names:
                        bin_sizes = list(
                            seg_data[name]['seg_data'][analysis_name][seg_name].keys())
                        bin_sizes_float = [float(bs) for bs in bin_sizes]
                        unique_bin_sizes.extend(bin_sizes_float)
                except:
                    print(name + " does not have data for " + analysis_name)
        unique_segment_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i]
                                for i in unique_segment_indices]
        unique_bin_indices = np.sort(
            np.unique(unique_bin_sizes, return_index=True)[1])
        unique_bin_sizes = [unique_bin_sizes[i]
                              for i in unique_bin_indices]

        self.unique_given_names = unique_given_names
        self.unique_analysis_names = unique_analysis_names
        self.unique_segment_names = unique_segment_names
        self.unique_bin_sizes = unique_bin_sizes
        
    def plot_seg_results(self,):
        num_cond = len(self.seg_data)
        results_dir = self.seg_results_dir

        print("Beginning Plots.")
        if num_cond > 1:
            cdf.cross_dataset_seg_compare_means(self.seg_data,self.unique_given_names,
                                          self.unique_analysis_names,
                                          self.unique_segment_names,
                                          self.unique_bin_sizes,
                                          results_dir)
            cdf.cross_dataset_seg_compare_mean_diffs(self.seg_data,self.unique_given_names,
                                          self.unique_analysis_names,
                                          self.unique_segment_names,
                                          self.unique_bin_sizes,
                                          results_dir)
        else:
           print("Not enough animals for segment comparison.")
           
    def import_cp_data(self,):
        """Import previously saved segment data"""
        dict_save_dir = os.path.join(self.save_dir, 'cp_data.npy')
        cp_data = np.load(dict_save_dir,allow_pickle=True).item()
        self.cp_data = cp_data
        if not os.path.isdir(os.path.join(self.save_dir,'Changepoint_Statistics')):
            os.mkdir(os.path.join(self.save_dir,'Changepoint_Statistics'))
        self.cp_results_dir = os.path.join(self.save_dir,'Changepoint_Statistics')
        
    def gather_cp_data(self,):
        """Import the relevant data from each dataset to be analyzed. This 
        includes the number of neurons, segments to analyze, segment names, 
        segment start and end times, taste dig in names, and the segment
        statistics data"""

        num_datasets = len(self.all_data_dict)
        dataset_names = list(self.all_data_dict.keys())
        cp_data = dict()
        for n_i in range(num_datasets):
            data_name = dataset_names[n_i]
            data_dict = self.all_data_dict[data_name]['data']
            metadata = self.all_data_dict[data_name]['metadata']
            hdf5_dir = metadata['hdf5_dir']
            cp_data[data_name] = dict()
            cp_data[data_name]['num_neur'] = data_dict['num_neur']
            segments_to_analyze = metadata['params_dict']['segments_to_analyze']
            cp_data[data_name]['segments_to_analyze'] = segments_to_analyze
            cp_data[data_name]['segment_names'] = data_dict['segment_names']
            segment_times = data_dict['segment_times']
            num_segments = len(cp_data[data_name]['segment_names'])
            cp_data[data_name]['segment_times_reshaped'] = [
                [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
            dig_in_names = data_dict['dig_in_names']
            cp_data[data_name]['dig_in_names'] = dig_in_names
            cp_data[data_name]['cp_data'] = dict()
            data_group_name = 'changepoint_data'
            pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
                hdf5_dir, data_group_name, 'pop_taste_cp_raster_inds')
            for t_i in range(len(dig_in_names)):
                taste_cp_data = pop_taste_cp_raster_inds[t_i] #num deliv x num_cp + 2
                cp_data[data_name]['cp_data'][dig_in_names[t_i]] = taste_cp_data
        self.cp_data = cp_data
        np.save(os.path.join(self.save_dir, 'cp_data.npy'),cp_data,allow_pickle=True)
        # Save the combined dataset somewhere...
        # _____Analysis Storage Directory_____
        if not os.path.isdir(os.path.join(self.save_dir,'Changepoint_Statistics')):
            os.mkdir(os.path.join(self.save_dir,'Changepoint_Statistics'))
        self.cp_results_dir = os.path.join(self.save_dir,'Changepoint_Statistics')
    
    def find_cp_groupings(self,):
        """Across the different datasets, get the unique data names/indices,
        correlation combinations and names/indices, unique segment names/indices,
        and unique taste names/indices to align datasets to each other in these
        different groups."""

        cp_data = self.cp_data
        unique_given_names = list(cp_data.keys())
        unique_given_indices = np.sort(
            np.unique(unique_given_names, return_index=True)[1])
        unique_given_names = [unique_given_names[i]
                              for i in unique_given_indices]
        unique_taste_names = np.array([list(cp_data[name]['cp_data'].keys(
        )) for name in unique_given_names]).flatten()  # How many types of segment analyses
        unique_taste_indices = np.sort(
            np.unique(unique_taste_names, return_index=True)[1])
        unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
        max_cp_counts = 0
        for name in unique_given_names:
            for taste_name in unique_taste_names:
                try:
                    taste_cp_data = cp_data[name]['cp_data'][taste_name]
                    num_cp = np.shape(taste_cp_data)[1] - 2
                    if num_cp > max_cp_counts:
                        max_cp_counts = num_cp
                except:
                    print(name + " does not have data for " + taste_name)
        
        self.unique_given_names = unique_given_names
        self.unique_taste_names = unique_taste_names
        self.max_cp_counts = max_cp_counts
        
    def plot_cp_results(self,):
        num_cond = len(self.cp_data)
        results_dir = self.cp_results_dir

        print("Beginning Plots.")
        if num_cond > 1:
            cdf.cross_dataset_cp_plots(self.cp_data, self.unique_given_names, 
                                       self.unique_taste_names, self.max_cp_counts,
                                       results_dir)
        else:
            print("Not enough animals for segment comparison.")
        
    def import_rate_corr_data(self,):
        """Import previously saved pop rate x taste corr data"""
        dict_save_dir = os.path.join(self.save_dir, 'rate_corr_data.npy')
        rate_corr_data = np.load(dict_save_dir,allow_pickle=True).item()
        self.rate_corr_data = rate_corr_data
        if not os.path.isdir(os.path.join(self.save_dir,'Sliding_Correlation_Comparison')):
            os.mkdir(os.path.join(self.save_dir,'Sliding_Correlation_Comparison'))
        self.rate_corr_results_dir = os.path.join(self.save_dir,'Sliding_Correlation_Comparison')
    
    def gather_rate_corr_data(self,):
        """Import the relevant data from each dataset to be analyzed. This 
        includes the number of neurons, segments to analyze, segment names, 
        segment start and end times, taste dig in names, and the population rate 
        x taste correlation data for all sliding bins"""

        num_datasets = len(self.all_data_dict)
        dataset_names = list(self.all_data_dict.keys())
        rate_corr_data = dict()
        for n_i in range(num_datasets):
            data_name = dataset_names[n_i]
            data_dict = self.all_data_dict[data_name]['data']
            metadata = self.all_data_dict[data_name]['metadata']
            data_save_dir = data_dict['data_path']
            rate_corr_save_dir = os.path.join(data_save_dir,'Sliding_Correlations')
            num_corr_types = os.listdir(rate_corr_save_dir)
            rate_corr_data[data_name] = dict()
            rate_corr_data[data_name]['num_neur'] = data_dict['num_neur']
            segments_to_analyze = metadata['params_dict']['segments_to_analyze']
            rate_corr_data[data_name]['segments_to_analyze'] = segments_to_analyze
            rate_corr_data[data_name]['segment_names'] = data_dict['segment_names']
            segment_times = data_dict['segment_times']
            num_segments = len(rate_corr_data[data_name]['segment_names'])
            rate_corr_data[data_name]['segment_times_reshaped'] = [
                [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
            dig_in_names = data_dict['dig_in_names']
            rate_corr_data[data_name]['dig_in_names'] = dig_in_names
            seg_names_to_analyze = np.array(rate_corr_data[data_name]['segment_names'])[segments_to_analyze]
            rate_corr_data[data_name]['rate_corr_data'] = dict()
            for nct in range(len(num_corr_types)):
                corr_type = num_corr_types[nct]
                if corr_type[0] != '.': #Ignore '.DS_Store'
                    rate_corr_data[data_name]['rate_corr_data'][corr_type] = dict()
                    corr_dir = os.path.join(rate_corr_save_dir,corr_type)
                    try:
                        rate_corr_data[data_name]['rate_corr_data'][corr_type] = np.load(os.path.join(corr_dir,'popfr_corr_storage.npy'), allow_pickle=True).item()
                    except:
                        print("No population fr x taste correlation dictionary found for " + data_name + " corr " + corr_type)
                    #This data is organized by [seg_name][bin_size] gives the result array
        self.rate_corr_data = rate_corr_data
        np.save(os.path.join(self.save_dir, 'rate_corr_data.npy'),rate_corr_data,allow_pickle=True)
        # Save the combined dataset somewhere...
        # _____Analysis Storage Directory_____
        if not os.path.isdir(os.path.join(self.save_dir,'Sliding_Correlation_Comparison')):
            os.mkdir(os.path.join(self.save_dir,'Sliding_Correlation_Comparison'))
        self.rate_corr_results_dir = os.path.join(self.save_dir,'Sliding_Correlation_Comparison')
        
    def find_rate_corr_groupings(self,):
        """Across the different datasets, get the unique data names/indices,
        correlation combinations and names/indices, unique segment names/indices,
        and unique taste names/indices to align datasets to each other in these
        different groups."""

        rate_corr_data = self.rate_corr_data
        unique_given_names = list(rate_corr_data.keys())
        unique_given_indices = np.sort(
            np.unique(unique_given_names, return_index=True)[1])
        unique_given_names = [unique_given_names[i]
                              for i in unique_given_indices]
        unique_corr_types = []
        for name in unique_given_names:
            unique_corr_types.extend(list(rate_corr_data[name]['rate_corr_data'].keys()))
        unique_corr_types = np.array(unique_corr_types)
        unique_corr_indices = np.sort(
            np.unique(unique_corr_types, return_index=True)[1])
        unique_corr_types = [unique_corr_types[i] for i in unique_corr_indices]
        unique_segment_names = []
        unique_taste_names = []
        for name in unique_given_names:
            for corr_name in unique_corr_types:
                try:
                    segment_names = list(rate_corr_data[name]['rate_corr_data'][corr_name].keys())
                    unique_segment_names.extend(segment_names)
                    for seg_name in segment_names:
                        taste_names = list(rate_corr_data[name]['rate_corr_data'][corr_name][seg_name].keys())
                        unique_taste_names.extend(taste_names)
                except:
                    print(name + " does not have data for " + corr_name)
        unique_segment_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i]
                                for i in unique_segment_indices]
        unique_taste_indices = np.sort(
            np.unique(unique_taste_names, return_index=True)[1])
        unique_taste_names = [unique_taste_names[i]
                              for i in unique_taste_indices]

        self.unique_given_names = unique_given_names
        self.unique_corr_types = unique_corr_types
        self.unique_segment_names = unique_segment_names
        self.unique_taste_names = unique_taste_names
    
    def plot_rate_corr_results(self,):
        num_cond = len(self.rate_corr_data)
        results_dir = self.rate_corr_results_dir

        print("Beginning Plots.")
        if num_cond > 1:
            cdf.cross_dataset_pop_rate_taste_corr_plots(self.rate_corr_data, self.unique_given_names, 
                                                        self.unique_corr_types, self.unique_segment_names, 
                                                        self.unique_taste_names, results_dir)
        else:
           print("Not enough animals for segment comparison.")
           
    def import_dev_stats(self,):
        """Import previously saved correlation data"""
        dict_save_dir = os.path.join(self.save_dir, 'dev_stats_data.npy')
        dev_stats_data = np.load(dict_save_dir,allow_pickle=True).item()
        self.dev_stats_data = dev_stats_data
        if not os.path.isdir(os.path.join(self.save_dir,'Dev_Stats')):
            os.mkdir(os.path.join(self.save_dir,'Dev_Stats'))
        self.dev_stats_results_dir = os.path.join(self.save_dir,'Dev_Stats')

    def gather_dev_stats_data(self,):
        """Import the relevant data from each dataset to be analyzed. This 
        includes the number of neurons, segments to analyze, segment names, 
        segment start and end times, taste dig in names, and the correlation
        data for all neurons and taste-selective neurons"""

        num_datasets = len(self.all_data_dict)
        dataset_names = list(self.all_data_dict.keys())
        dev_stats_data = dict()
        for n_i in range(num_datasets):
            data_name = dataset_names[n_i]
            data_dict = self.all_data_dict[data_name]['data']
            metadata = self.all_data_dict[data_name]['metadata']
            data_save_dir = data_dict['data_path']
            
            dev_stats_save_dir = os.path.join(
                data_save_dir, 'Deviations')
            dev_dir_files = os.listdir(dev_stats_save_dir)
            dev_dict_dirs = []
            for dev_f in dev_dir_files:
                if dev_f[-4:] == '.npy':
                    dev_dict_dirs.append(dev_f)
            dev_stats_data[data_name] = dict()
            dev_stats_data[data_name]['num_neur'] = data_dict['num_neur']
            segments_to_analyze = metadata['params_dict']['segments_to_analyze']
            dev_stats_data[data_name]['segments_to_analyze'] = segments_to_analyze
            dev_stats_data[data_name]['segment_names'] = data_dict['segment_names']
            segment_names_to_analyze = np.array(data_dict['segment_names'])[segments_to_analyze]
            segment_times = data_dict['segment_times']
            num_segments = len(dev_stats_data[data_name]['segment_names'])
            dev_stats_data[data_name]['segment_times_reshaped'] = [
                [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
            dig_in_names = data_dict['dig_in_names']
            dev_stats_data[data_name]['dig_in_names'] = dig_in_names
            dev_stats_data[data_name]['dev_stats'] = dict()
            for stat_i in range(len(dev_dict_dirs)):
                stat_dir_name = dev_dict_dirs[stat_i]
                stat_name = stat_dir_name.split('.')[0]
                result_dir = os.path.join(dev_stats_save_dir, stat_dir_name)
                result_dict = np.load(result_dir,allow_pickle=True).item()
                dev_stats_data[data_name]['dev_stats'][stat_name] = dict()
                for s_i, s_name in enumerate(segment_names_to_analyze):
                    dev_stats_data[data_name]['dev_stats'][stat_name][s_name] = result_dict[s_i]
                    
        self.dev_stats_data = dev_stats_data
        dict_save_dir = os.path.join(self.save_dir, 'dev_stats_data.npy')
        np.save(dict_save_dir,dev_stats_data,allow_pickle=True)
        # _____Analysis Storage Directory_____
        if not os.path.isdir(os.path.join(self.save_dir,'Dev_Stats')):
            os.mkdir(os.path.join(self.save_dir,'Dev_Stats'))
        self.dev_stats_results_dir = os.path.join(self.save_dir,'Dev_Stats')

    def find_dev_stats_groupings(self,):
        """Across the different datasets, get the unique data names/indices,
        deviation statistic combinations and names/indices, and unique segment 
        names/indices to align datasets to each other in these
        different groups."""

        dev_stats_data = self.dev_stats_data
        unique_given_names = list(dev_stats_data.keys())
        unique_given_indices = np.sort(
            np.unique(unique_given_names, return_index=True)[1])
        unique_given_names = [unique_given_names[i]
                              for i in unique_given_indices]
        unique_dev_stats_names = []
        for name in unique_given_names:
            unique_dev_stats_names.extend(list(dev_stats_data[name]['dev_stats'].keys()))
        unique_dev_stats_names = np.array(unique_dev_stats_names)
        unique_dev_stats_indices = np.sort(
            np.unique(unique_dev_stats_names, return_index=True)[1])
        unique_dev_stats_names = [unique_dev_stats_names[i] for i in unique_dev_stats_indices]
        unique_segment_names = []
        for name in unique_given_names:
            for dev_stat_name in unique_dev_stats_names:
                try:
                    seg_names = list(
                        dev_stats_data[name]['dev_stats'][dev_stat_name].keys())
                    unique_segment_names.extend(seg_names)
                except:
                    print(name + " does not have correlation data for " + dev_stat_name)
        unique_segment_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i]
                                for i in unique_segment_indices]
        
        self.unique_given_names = unique_given_names
        self.unique_dev_stats_names = unique_dev_stats_names
        self.unique_segment_names = unique_segment_names

    def plot_dev_stat_results(self,):
        num_cond = len(self.dev_stats_data)
        results_dir = self.dev_stats_results_dir

        print("Beginning Plots.")
        if num_cond > 1:
            cdf.cross_dataset_dev_stats_plots(self.dev_stats_data, self.unique_given_names, 
                                              self.unique_dev_stats_names, 
                                              self.unique_segment_names, 
                                              results_dir)
        else:
            print("Not enough animals for cross-animal dev stat plots.")

    def import_dev_null_stats(self,):
        """Import previously saved deviation true x null data"""
        dict_save_dir = os.path.join(self.save_dir, 'dev_null_data.npy')
        dev_null_data = np.load(dict_save_dir,allow_pickle=True).item()
        self.dev_null_data = dev_null_data
        if not os.path.isdir(os.path.join(self.save_dir,'Dev_Null')):
            os.mkdir(os.path.join(self.save_dir,'Dev_Null'))
        self.dev_null_results_dir = os.path.join(self.save_dir,'Dev_Null')

    def gather_dev_null_data(self,):
        """Import the relevant data from each dataset to be analyzed. This 
        includes the number of neurons, segments to analyze, segment names, 
        segment start and end times, taste dig in names, and the correlation
        data for all neurons and taste-selective neurons"""

        num_datasets = len(self.all_data_dict)
        dataset_names = list(self.all_data_dict.keys())
        dev_null_data = dict()
        for n_i in range(num_datasets):
            data_name = dataset_names[n_i]
            data_dict = self.all_data_dict[data_name]['data']
            metadata = self.all_data_dict[data_name]['metadata']
            data_save_dir = data_dict['data_path']
            
            dev_null_save_dir = os.path.join(
                data_save_dir, 'Deviations','null_x_true_deviations')
            dev_dir_files = os.listdir(dev_null_save_dir)
            dev_dict_dirs = []
            for dev_f in dev_dir_files:
                if dev_f[-4:] == '.npy':
                    dev_dict_dirs.append(dev_f)
            dev_null_data[data_name] = dict()
            dev_null_data[data_name]['num_neur'] = data_dict['num_neur']
            segments_to_analyze = metadata['params_dict']['segments_to_analyze']
            dev_null_data[data_name]['segments_to_analyze'] = segments_to_analyze
            dev_null_data[data_name]['segment_names'] = data_dict['segment_names']
            segment_names_to_analyze = np.array(data_dict['segment_names'])[segments_to_analyze]
            segment_times = data_dict['segment_times']
            num_segments = len(dev_null_data[data_name]['segment_names'])
            dev_null_data[data_name]['segment_times_reshaped'] = [
                [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
            dig_in_names = data_dict['dig_in_names']
            dev_null_data[data_name]['dig_in_names'] = dig_in_names
            dev_null_data[data_name]['dev_null'] = dict()
            for stat_i in range(len(dev_dict_dirs)):
                stat_dir_name = dev_dict_dirs[stat_i]
                null_name = stat_dir_name.split('.')[0]
                result_dir = os.path.join(dev_null_save_dir, stat_dir_name)
                result_dict = np.load(result_dir,allow_pickle=True).item()
                result_keys = list(result_dict.keys())
                dev_null_data[data_name]['dev_null'][null_name] = dict()
                for s_i, s_name in enumerate(segment_names_to_analyze):
                    dev_null_data[data_name]['dev_null'][null_name][s_name] = dict()
                    for rk_i, rk in enumerate(result_keys):
                        if rk[:len(s_name)] == s_name:
                            rk_type = rk.split('_')[1]
                            dev_null_data[data_name]['dev_null'][null_name][s_name][rk_type] = \
                                result_dict[rk]
                    
                    
        self.dev_null_data = dev_null_data
        dict_save_dir = os.path.join(self.save_dir, 'dev_null_data.npy')
        np.save(dict_save_dir,dev_null_data,allow_pickle=True)
        # _____Analysis Storage Directory_____
        if not os.path.isdir(os.path.join(self.save_dir,'Dev_Null')):
            os.mkdir(os.path.join(self.save_dir,'Dev_Null'))
        self.dev_null_results_dir = os.path.join(self.save_dir,'Dev_Null')

    def find_dev_null_groupings(self,):
        """Across the different datasets, get the unique data names/indices,
        deviation null statistic combinations and names/indices, and unique  
        segment names/indices to align datasets to each other in these
        different groups."""

        dev_null_data = self.dev_null_data
        unique_given_names = list(dev_null_data.keys())
        unique_given_indices = np.sort(
            np.unique(unique_given_names, return_index=True)[1])
        unique_given_names = [unique_given_names[i]
                              for i in unique_given_indices]
        unique_dev_null_names = []
        for name in unique_given_names:
            unique_dev_null_names.extend(list(dev_null_data[name]['dev_null'].keys()))
        unique_dev_null_names = np.array(unique_dev_null_names)
        unique_dev_null_indices = np.sort(
            np.unique(unique_dev_null_names, return_index=True)[1])
        unique_dev_null_names = [unique_dev_null_names[i] for i in unique_dev_null_indices]
        unique_segment_names = []
        for name in unique_given_names:
            for dev_null_name in unique_dev_null_names:
                try:
                    seg_names = list(
                        dev_null_data[name]['dev_null'][dev_null_name].keys())
                    unique_segment_names.extend(seg_names)
                except:
                    print(name + " does not have correlation data for " + dev_null_name)
        unique_segment_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i]
                                for i in unique_segment_indices]
        
        self.unique_given_names = unique_given_names
        self.unique_dev_null_names = unique_dev_null_names
        self.unique_segment_names = unique_segment_names

    def plot_dev_null_results(self,):
        num_cond = len(self.dev_null_data)
        results_dir = self.dev_null_results_dir

        print("Beginning Plots.")
        if num_cond > 1:
            cdf.cross_dataset_dev_null_plots(self.dev_null_data, self.unique_given_names, 
                                             self.unique_dev_null_names, self.unique_segment_names, 
                                             results_dir)
        else:
            print("Not enough animals for cross-animal dev stat plots.")

           
    def import_dev_split_corr_data(self,):
        """Import previously saved deviation true x null data"""
        dict_save_dir = os.path.join(self.save_dir, 'dev_split_corr_data.npy')
        dev_split_corr_data = np.load(dict_save_dir,allow_pickle=True).item()
        self.dev_split_corr_data = dev_split_corr_data
        if not os.path.isdir(os.path.join(self.save_dir,'Dev_Split_Corr')):
            os.mkdir(os.path.join(self.save_dir,'Dev_Split_Corr'))
        self.dev_split_corr_results_dir = os.path.join(self.save_dir,'Dev_Split_Corr')

    def gather_dev_split_data(self,):
        """Import the relevant data from each dataset to be analyzed. This 
        includes the number of neurons, segments to analyze, segment names, 
        segment start and end times, taste dig in names, and the correlation
        data for all neurons and taste-selective neurons"""

        num_datasets = len(self.all_data_dict)
        dataset_names = list(self.all_data_dict.keys())
        dev_split_corr_data = dict()
        for n_i in range(num_datasets):
            data_name = dataset_names[n_i]
            data_dict = self.all_data_dict[data_name]['data']
            metadata = self.all_data_dict[data_name]['metadata']
            dev_split_corr_data[data_name] = dict()
            dev_split_corr_data[data_name]['num_neur'] = data_dict['num_neur']
            segments_to_analyze = metadata['params_dict']['segments_to_analyze']
            dev_split_corr_data[data_name]['segments_to_analyze'] = segments_to_analyze
            dev_split_corr_data[data_name]['segment_names'] = data_dict['segment_names']
            segment_names_to_analyze = np.array(data_dict['segment_names'])[segments_to_analyze]
            segment_times = data_dict['segment_times']
            num_segments = len(dev_split_corr_data[data_name]['segment_names'])
            dev_split_corr_data[data_name]['segment_times_reshaped'] = [
                [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
            dig_in_names = data_dict['dig_in_names']
            dev_split_corr_data[data_name]['dig_in_names'] = dig_in_names
            data_save_dir = data_dict['data_path']
            dev_split_save_dir = os.path.join(
                data_save_dir, 'Deviation_Sequence_Analysis')
            #Subfolders we care about: corr_tests and decode_splits
            #First load correlation data
            dev_split_corr_dir = os.path.join(dev_split_save_dir,'corr_tests','zscore_firing_rates')
            dev_split_corr_files = os.listdir(dev_split_corr_dir)
            dev_split_corr_dict_files = []
            for dev_corr_f in dev_split_corr_files:
                if dev_corr_f[-4:] == '.npy':
                    dev_split_corr_dict_files.append(dev_corr_f)
            dev_split_corr_data[data_name]['corr_data'] = dict()
            for stat_i in range(len(dev_split_corr_dict_files)):
                stat_filename = dev_split_corr_dict_files[stat_i]
                stat_segment = (stat_filename.split('.')[0]).split('_')[0]
                dev_split_corr_data[data_name]['corr_data'][stat_segment] = dict()
                dict_data = np.load(os.path.join(dev_split_corr_dir,stat_filename),allow_pickle=True).item()
                epoch_pairs = list(dict_data.keys())
                dev_split_corr_data[data_name]['corr_data'][stat_segment]['epoch_pairs'] = epoch_pairs
                num_tastes = len(dig_in_names)
                for t_i, t_name in enumerate(dig_in_names):
                    taste_corr_data = []
                    for ep_i, ep in enumerate(epoch_pairs):
                        taste_corr_data.append(dict_data[ep]['taste_corrs'][t_i])
                    taste_corr_data = np.array(taste_corr_data)
                    _, num_dev = np.shape(taste_corr_data)
                    dev_split_corr_data[data_name]['corr_data'][stat_segment][t_name] = taste_corr_data
                    
        self.dev_split_corr_data = dev_split_corr_data
        dict_save_dir = os.path.join(self.save_dir, 'dev_split_corr_data.npy')
        np.save(dict_save_dir,dev_split_corr_data,allow_pickle=True)
        # _____Analysis Storage Directory_____
        if not os.path.isdir(os.path.join(self.save_dir,'Dev_Split_Corr')):
            os.mkdir(os.path.join(self.save_dir,'Dev_Split_Corr'))
        self.dev_split_corr_results_dir = os.path.join(self.save_dir,'Dev_Split_Corr')

    def find_dev_split_corr_groupings(self,):
        """Across the different datasets, get the unique data names/indices,
        deviation null statistic combinations and names/indices, and unique  
        segment names/indices to align datasets to each other in these
        different groups."""
        dev_split_corr_data = self.dev_split_corr_data

        unique_given_names = list(dev_split_corr_data.keys())
        unique_given_indices = np.sort(
            np.unique(unique_given_names, return_index=True)[1])
        unique_given_names = [unique_given_names[i]
                              for i in unique_given_indices]
        unique_epoch_pairs = []
        unique_segment_names = []
        unique_taste_names = []
        for name in unique_given_names:
            segment_names = list(dev_split_corr_data[name]['corr_data'].keys())
            unique_segment_names.extend(segment_names)
            for seg_name in segment_names:
                epoch_pairs = dev_split_corr_data[name]['corr_data'][seg_name]['epoch_pairs']
                unique_epoch_pairs.extend(epoch_pairs)
                taste_names = list(dev_split_corr_data[name]['corr_data'][seg_name].keys())
                epoch_pairs_ind = np.where(np.array(taste_names) == 'epoch_pairs')
                if len(np.shape(np.where(np.array(taste_names) == 'epoch_pairs'))) == 2:
                    taste_names.pop(epoch_pairs_ind[0][0])
                else:
                    taste_names.pop(epoch_pairs_ind[0])
                unique_taste_names.extend(taste_names)
       
        unique_epoch_indices = np.sort(
            np.unique(unique_epoch_pairs, return_index=True)[1])
        unique_epoch_pairs = [unique_epoch_pairs[i] for i in unique_epoch_indices]
        unique_segment_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i] for i in unique_segment_indices]
        unique_taste_indices = np.sort(
            np.unique(unique_taste_names, return_index=True)[1])
        unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
        
        self.unique_given_names = unique_given_names
        self.unique_epoch_pairs = unique_epoch_pairs
        self.unique_segment_names = unique_segment_names
        self.unique_taste_names = unique_taste_names

    def plot_dev_split_results(self,):
        num_cond = len(self.dev_split_corr_data)
        results_dir = self.dev_split_corr_results_dir

        print("Beginning Plots.")
        if num_cond > 1:
            cdf.cross_dataset_dev_split_corr_plots(self.dev_split_corr_data, self.unique_given_names, 
                                             self.unique_epoch_pairs, self.unique_segment_names, 
                                             self.unique_taste_names, results_dir)
        else:
            print("Not enough animals for cross-animal dev stat plots.")
            
    def import_sliding_decode_data(self,):
        """Import previously saved sliding bin decode data"""
        dict_save_dir = os.path.join(self.save_dir, 'sliding_decode_data.npy')
        sliding_decode_data = np.load(dict_save_dir,allow_pickle=True).item()
        self.sliding_decode_data = sliding_decode_data
        if not os.path.isdir(os.path.join(self.save_dir,'Sliding_Decode')):
            os.mkdir(os.path.join(self.save_dir,'Sliding_Decode'))
        self.sliding_decode_results_dir = os.path.join(self.save_dir,'Sliding_Decode')

    def gather_sliding_decode_data(self,):
        """Import the relevant data from each dataset to be analyzed. This 
        includes the number of neurons, segments to analyze, segment names, 
        segment start and end times, taste dig in names, and the sliding
        decode data for all neurons"""

        num_datasets = len(self.all_data_dict)
        dataset_names = list(self.all_data_dict.keys())
        sliding_decode_data = dict()
        for n_i in range(num_datasets):
            data_name = dataset_names[n_i]
            data_dict = self.all_data_dict[data_name]['data']
            metadata = self.all_data_dict[data_name]['metadata']
            sliding_decode_data[data_name] = dict()
            sliding_decode_data[data_name]['num_neur'] = data_dict['num_neur']
            segments_to_analyze = metadata['params_dict']['segments_to_analyze']
            sliding_decode_data[data_name]['segments_to_analyze'] = segments_to_analyze
            sliding_decode_data[data_name]['segment_names'] = data_dict['segment_names']
            segment_names_to_analyze = np.array(data_dict['segment_names'])[segments_to_analyze]
            segment_times = data_dict['segment_times']
            num_segments = len(sliding_decode_data[data_name]['segment_names'])
            sliding_decode_data[data_name]['segment_times_reshaped'] = [
                [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
            epochs_to_analyze = metadata['params_dict']['epochs_to_analyze']
            sliding_decode_data[data_name]['epochs_to_analyze'] = epochs_to_analyze
            dig_in_names = data_dict['dig_in_names']
            sliding_decode_data[data_name]['dig_in_names'] = dig_in_names
            data_save_dir = data_dict['data_path']
            sliding_decode_save_dir = os.path.join(
                data_save_dir, 'Sliding_Decoding')
            #Subfolders we care about: corr_tests and decode_splits
            #First load correlation data
            sliding_decode_zscore_dir = os.path.join(sliding_decode_save_dir,'All_Neurons_Z_Scored')
            sliding_decode_zscore_files = os.listdir(sliding_decode_zscore_dir)
            sliding_decode_zscore_dict_files = []
            for sd_z_f in sliding_decode_zscore_files:
                if sd_z_f[-4:] == '.npy':
                    sliding_decode_zscore_dict_files.append(sd_z_f)
            sliding_decode_data[data_name]['pop_corr_data'] = dict()
            sliding_decode_data[data_name]['frac_decode_data'] = dict()
            for stat_i in range(len(sliding_decode_zscore_dict_files)):
                stat_full_filename = sliding_decode_zscore_dict_files[stat_i]
                stat_filename = stat_full_filename.split('.')[0]
                stat_type = stat_filename[-4:]
                if stat_type == 'corr': #population rate correlation
                    corr_type = ('_').join(stat_filename.split('_')[1:3])
                    sliding_decode_data[data_name]['pop_corr_data'][corr_type] = np.load(os.path.join(sliding_decode_zscore_dir,stat_full_filename),
                                                                                         allow_pickle=True)
                elif stat_type == 'frac': #fraction of decodes
                    frac_type = ('_').join(stat_filename.split('_')[1:3])
                    sliding_decode_data[data_name]['frac_decode_data'][frac_type] = np.load(os.path.join(sliding_decode_zscore_dir,stat_full_filename),
                                                                                         allow_pickle=True)
                
        self.sliding_decode_data = sliding_decode_data
        dict_save_dir = os.path.join(self.save_dir, 'sliding_decode_data.npy')
        np.save(dict_save_dir,sliding_decode_data,allow_pickle=True)
        # _____Analysis Storage Directory_____
        if not os.path.isdir(os.path.join(self.save_dir,'Sliding_Decode')):
            os.mkdir(os.path.join(self.save_dir,'Sliding_Decode'))
        self.sliding_decode_results_dir = os.path.join(self.save_dir,'Sliding_Decode')

    def find_sliding_decode_groupings(self,):
        """Across the different datasets, get the unique data names/indices,
        deviation null statistic combinations and names/indices, and unique  
        segment names/indices to align datasets to each other in these
        different groups."""
        sliding_decode_data = self.sliding_decode_data

        unique_given_names = list(sliding_decode_data.keys())
        unique_given_indices = np.sort(
            np.unique(unique_given_names, return_index=True)[1])
        unique_given_names = [unique_given_names[i]
                              for i in unique_given_indices]
        unique_segment_names = []
        unique_epochs = []
        unique_taste_names = []
        unique_decode_types = []
        for name in unique_given_names:
            epoch_inds = list(sliding_decode_data[name]['epochs_to_analyze'])
            unique_epochs.extend(epoch_inds)
            segment_names = sliding_decode_data[name]['segment_names']
            used_segment_names = np.array(segment_names)[sliding_decode_data[name]['segments_to_analyze']]
            unique_segment_names.extend(used_segment_names)
            decode_types = list(sliding_decode_data[name]['frac_decode_data'].keys())
            unique_decode_types.extend(decode_types)
            unique_taste_names.extend(sliding_decode_data[name]['dig_in_names'][:-1])
               
        unique_segment_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i] for i in unique_segment_indices]
        unique_epoch_indices = np.sort(
            np.unique(unique_epochs, return_index=True)[1])
        unique_epochs = [unique_epochs[i] for i in unique_epoch_indices]
        unique_taste_indices = np.sort(
            np.unique(unique_taste_names, return_index=True)[1])
        unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
        unique_decode_indices = np.sort(
            np.unique(unique_decode_types, return_index=True)[1])
        unique_decode_types = [unique_decode_types[i] for i in unique_decode_indices]
        
        
        self.unique_given_names = unique_given_names
        self.unique_segment_names = unique_segment_names
        self.unique_epochs = unique_epochs
        self.unique_taste_names = unique_taste_names
        self.unique_decode_types = unique_decode_types

    def plot_sliding_decode_results(self,):
        num_cond = len(self.sliding_decode_data)
        results_dir = self.sliding_decode_results_dir

        print("Beginning Plots.")
        if num_cond > 1:
            cdf.cross_dataset_sliding_decode_frac_plots(self.sliding_decode_data,
                                            self.unique_given_names,self.unique_segment_names,
                                            self.unique_epochs,self.unique_taste_names,
                                            self.unique_decode_types,self.results_dir)
            cdf.cross_dataset_sliding_decode_corr_plots(self.sliding_decode_data,
                                            self.unique_given_names,self.unique_segment_names,
                                            self.unique_epochs,self.unique_taste_names,
                                            self.unique_decode_types,self.results_dir)
        else:
            print("Not enough animals for cross-animal dev stat plots.")
           
    def import_dev_decode_data(self,):
        """Import previously saved sliding bin decode data"""
        dict_save_dir = os.path.join(self.save_dir, 'dev_decode_data.npy')
        dev_decode_data = np.load(dict_save_dir,allow_pickle=True).item()
        self.dev_decode_data = dev_decode_data
        if not os.path.isdir(os.path.join(self.save_dir,'Dev_Decode')):
            os.mkdir(os.path.join(self.save_dir,'Dev_Decode'))
        self.dev_decode_results_dir = os.path.join(self.save_dir,'Dev_Decode')

    def gather_dev_decode_data(self,):
        """Import the relevant data from each dataset to be analyzed. This 
        includes the number of neurons, segments to analyze, segment names, 
        segment start and end times, taste dig in names, and the sliding
        decode data for all neurons"""

        num_datasets = len(self.all_data_dict)
        dataset_names = list(self.all_data_dict.keys())
        dev_decode_data = dict()
        for n_i in range(num_datasets):
            data_name = dataset_names[n_i]
            data_dict = self.all_data_dict[data_name]['data']
            metadata = self.all_data_dict[data_name]['metadata']
            dev_decode_data[data_name] = dict()
            dev_decode_data[data_name]['num_neur'] = data_dict['num_neur']
            segments_to_analyze = metadata['params_dict']['segments_to_analyze']
            dev_decode_data[data_name]['segments_to_analyze'] = segments_to_analyze
            dev_decode_data[data_name]['segment_names'] = data_dict['segment_names']
            segment_names_to_analyze = np.array(data_dict['segment_names'])[segments_to_analyze]
            segment_times = data_dict['segment_times']
            num_segments = len(dev_decode_data[data_name]['segment_names'])
            dev_decode_data[data_name]['segment_times_reshaped'] = [
                [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
            epochs_to_analyze = metadata['params_dict']['epochs_to_analyze']
            dev_decode_data[data_name]['epochs_to_analyze'] = epochs_to_analyze
            dig_in_names = data_dict['dig_in_names']
            dev_decode_data[data_name]['dig_in_names'] = dig_in_names
            data_save_dir = data_dict['data_path']
            dev_decode_dir = os.path.join(
                data_save_dir, 'Deviation_Dependent_Decoding')
            #Subfolders we care about: corr_tests and decode_splits
            #First load correlation data
            dev_decode_zscore_dir = os.path.join(dev_decode_dir,'All_Neurons_Z_Scored',\
                                                 'GMM_Decoding','Is_Taste_Which_Taste')
            dev_decode_zscore_files = os.listdir(dev_decode_zscore_dir)
            dev_decode_zscore_dict_files = []
            for dd_z_f in dev_decode_zscore_files:
                if dd_z_f[-4:] == '.npy':
                    dev_decode_zscore_dict_files.append(dd_z_f)
            dev_decode_data[data_name]['is_taste'] = dict()
            dev_decode_data[data_name]['which_taste'] = dict()
            dev_decode_data[data_name]['which_epoch'] = dict()
            for stat_i, stat_full_filename in enumerate(dev_decode_zscore_dict_files):
                stat_filename = stat_full_filename.split('.')[0]
                stat_filename_split = stat_filename.split('_')
                if len(stat_filename_split) == 5: #Deviation data
                    seg_ind = int(stat_filename_split[1])
                    seg_name = data_dict['segment_names'][seg_ind]
                    stat_type = ('_').join(stat_filename_split[-2:])
                    dev_decode_data[data_name][stat_type][seg_name] = np.load(os.path.join(\
                                    dev_decode_zscore_dir,stat_full_filename),
                                                        allow_pickle=True)
                
        self.dev_decode_data = dev_decode_data
        dict_save_dir = os.path.join(self.save_dir, 'dev_decode_data.npy')
        np.save(dict_save_dir,dev_decode_data,allow_pickle=True)
        # _____Analysis Storage Directory_____
        if not os.path.isdir(os.path.join(self.save_dir,'Dev_Decode')):
            os.mkdir(os.path.join(self.save_dir,'Dev_Decode'))
        self.dev_decode_results_dir = os.path.join(self.save_dir,'Dev_Decode')

    def find_dev_decode_groupings(self,):
        """Across the different datasets, get the unique data names/indices,
        deviation null statistic combinations and names/indices, and unique  
        segment names/indices to align datasets to each other in these
        different groups."""
        dev_decode_data = self.dev_decode_data

        unique_given_names = list(dev_decode_data.keys())
        unique_given_indices = np.sort(
            np.unique(unique_given_names, return_index=True)[1])
        unique_given_names = [unique_given_names[i]
                              for i in unique_given_indices]
        unique_segment_names = []
        unique_epochs = []
        unique_taste_names = []
        unique_decode_types = ['is_taste','which_epoch','which_taste']
        for name in unique_given_names:
            epoch_inds = list(dev_decode_data[name]['epochs_to_analyze'])
            unique_epochs.extend(epoch_inds)
            segment_names = dev_decode_data[name]['segment_names']
            used_segment_names = np.array(segment_names)[dev_decode_data[name]['segments_to_analyze']]
            unique_segment_names.extend(used_segment_names)
            unique_taste_names.extend(dev_decode_data[name]['dig_in_names'][:-1])
               
        unique_segment_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i] for i in unique_segment_indices]
        unique_epoch_indices = np.sort(
            np.unique(unique_epochs, return_index=True)[1])
        unique_epochs = [unique_epochs[i] for i in unique_epoch_indices]
        unique_taste_indices = np.sort(
            np.unique(unique_taste_names, return_index=True)[1])
        unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
        
        self.unique_given_names = unique_given_names
        self.unique_segment_names = unique_segment_names
        self.unique_epochs = unique_epochs
        self.unique_taste_names = unique_taste_names
        self.unique_decode_types = unique_decode_types

    def plot_dev_decode_results(self,):
        num_cond = len(self.dev_decode_data)
        results_dir = self.dev_decode_results_dir

        print("Beginning Plots.")
        if num_cond > 1:
            cdf.cross_dataset_dev_decode_frac_plots(self.dev_decode_data, self.unique_given_names,
                                                    self.unique_segment_names, self.unique_epochs,
                                                    self.unique_taste_names, self.unique_decode_types,
                                                    results_dir)
        else:
            print("Not enough animals for cross-animal dev stat plots.")
            
    def import_dev_split_decode_data(self,):
        """Import previously saved deviation true x null data"""
        dict_save_dir = os.path.join(self.save_dir, 'dev_split_decode_data.npy')
        dev_split_decode_data = np.load(dict_save_dir,allow_pickle=True).item()
        self.dev_split_decode_data = dev_split_decode_data
        if not os.path.isdir(os.path.join(self.save_dir,'Dev_Split_Decode')):
            os.mkdir(os.path.join(self.save_dir,'Dev_Split_Decode'))
        self.dev_split_decode_results_dir = os.path.join(self.save_dir,'Dev_Split_Decode')

    def gather_dev_split_decode_data(self,):
        """Import the relevant data from each dataset to be analyzed. This 
        includes the number of neurons, segments to analyze, segment names, 
        segment start and end times, taste dig in names, and the decoding
        data for all neurons"""
        
        decode_types = ['is_taste','which_taste','which_epoch']

        num_datasets = len(self.all_data_dict)
        dataset_names = list(self.all_data_dict.keys())
        dev_split_decode_data = dict()
        for n_i in range(num_datasets):
            data_name = dataset_names[n_i]
            data_dict = self.all_data_dict[data_name]['data']
            metadata = self.all_data_dict[data_name]['metadata']
            dev_split_decode_data[data_name] = dict()
            dev_split_decode_data[data_name]['num_neur'] = data_dict['num_neur']
            segments_to_analyze = metadata['params_dict']['segments_to_analyze']
            dev_split_decode_data[data_name]['segments_to_analyze'] = segments_to_analyze
            dev_split_decode_data[data_name]['segment_names'] = data_dict['segment_names']
            segment_names_to_analyze = np.array(data_dict['segment_names'])[segments_to_analyze]
            segment_times = data_dict['segment_times']
            num_segments = len(dev_split_decode_data[data_name]['segment_names'])
            dev_split_decode_data[data_name]['segment_times_reshaped'] = [
                [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
            dig_in_names = data_dict['dig_in_names']
            dev_split_decode_data[data_name]['dig_in_names'] = dig_in_names
            data_save_dir = data_dict['data_path']
            dev_split_save_dir = os.path.join(
                data_save_dir, 'Deviation_Sequence_Analysis')
            #Subfolders we care about: corr_tests and decode_splits
            #First load correlation data
            dev_split_decode_dir = os.path.join(dev_split_save_dir,'decode_splits','zscore_firing_rates')
            dev_split_decode_files = os.listdir(dev_split_decode_dir)
            dev_split_decode_dict_files = []
            for dev_dec_f in dev_split_decode_files:
                if dev_dec_f[-4:] == '.npy':
                    dev_split_decode_dict_files.append(dev_dec_f)
            dev_split_decode_data[data_name]['decode_data'] = dict()
            for sna in segment_names_to_analyze:
                dev_split_decode_data[data_name]['decode_data'][sna] = dict()
            for stat_i, stat_filename in enumerate(dev_split_decode_dict_files):
                stat_filename_split = (stat_filename.split('.')[0]).split('_')
                if stat_filename_split[-1] != 'argmax':
                    stat_seg_name = stat_filename_split[0]
                    file_decode_type = ('_').join(stat_filename_split[-2:])
                    file_decode_type_ind = [i for i in range(len(decode_types)) if file_decode_type == decode_types[i]]
                    if len(file_decode_type_ind) > 0:
                        dev_split_decode_data[data_name]['decode_data'][stat_seg_name][file_decode_type] = \
                            np.load(os.path.join(dev_split_decode_dir,stat_filename),allow_pickle=True)
                    
        self.dev_split_decode_data = dev_split_decode_data
        dict_save_dir = os.path.join(self.save_dir, 'dev_split_decode_data.npy')
        np.save(dict_save_dir,dev_split_decode_data,allow_pickle=True)
        # _____Analysis Storage Directory_____
        if not os.path.isdir(os.path.join(self.save_dir,'Dev_Split_Decode')):
            os.mkdir(os.path.join(self.save_dir,'Dev_Split_Decode'))
        self.dev_split_decode_results_dir = os.path.join(self.save_dir,'Dev_Split_Decode')

    def find_dev_split_decode_groupings(self,):
        """Across the different datasets, get the unique data names/indices,
        deviation null statistic combinations and names/indices, and unique  
        segment names/indices to align datasets to each other in these
        different groups."""
        dev_split_decode_data = self.dev_split_decode_data

        unique_given_names = list(dev_split_decode_data.keys())
        unique_given_indices = np.sort(
            np.unique(unique_given_names, return_index=True)[1])
        unique_given_names = [unique_given_names[i]
                              for i in unique_given_indices]
        unique_segment_names = []
        unique_taste_names = []
        for name in unique_given_names:
            segment_names = list(dev_split_decode_data[name]['decode_data'].keys())
            unique_segment_names.extend(segment_names)
            for seg_name in segment_names:
                taste_names = list(dev_split_decode_data[name]['dig_in_names'][:-1])
                unique_taste_names.extend(taste_names)
       
        unique_segment_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i] for i in unique_segment_indices]
        unique_taste_indices = np.sort(
            np.unique(unique_taste_names, return_index=True)[1])
        unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
        
        self.unique_given_names = unique_given_names
        self.unique_segment_names = unique_segment_names
        self.unique_taste_names = unique_taste_names

    def plot_dev_split_decode_results(self,):
        num_cond = len(self.dev_split_decode_data)
        results_dir = self.dev_split_decode_results_dir

        print("Beginning Plots.")
        if num_cond > 1:
            cdf.cross_dataset_dev_split_decode_frac_plots(self.dev_split_decode_data, self.unique_given_names,
                                                        self.unique_segment_names, self.unique_taste_names, 
                                                        self.decode_types, results_dir)
        else:
            print("Not enough animals for cross-animal dev stat plots.")
            
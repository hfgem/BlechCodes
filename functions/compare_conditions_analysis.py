#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:18:06 2024

@author: Hannah Germaine

Functions to support compare_conditions.py in running cross-dataset analyses.
"""

import os
import warnings
import easygui
import pickle
import numpy as np

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
        else:
            print("Please select a storage folder for results.")
            self.save_dir = easygui.diropenbox(
                title='Please select the storage folder.')
            np.save(os.path.join(self.save_dir,'all_data_dict.npy'),\
                    self.all_data_dict,allow_pickle=True)
            self.gather_corr_data()
            self.gather_seg_data()
        #Correlation comparisons
        self.find_corr_groupings()
        self.plot_corr_results()
        #Segment comparisons
        self.find_seg_groupings()
        self.plot_seg_results()

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
                        for t_i in range(len(dig_in_names)):
                            taste_name = dig_in_names[t_i]
                            corr_data[data_name]['corr_data'][nct][seg_name][taste_name] = dict(
                            )
                            try:
                                filename_pop_vec = os.path.join(
                                    result_dir, seg_name + '_' + taste_name + '_pop_vec.npy')
                                data = np.load(filename_pop_vec)
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
                        taste_names = list(
                            corr_data[name]['corr_data'][corr_name][seg_name].keys())
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
            # Cross-Dataset: different given names on the same axes
            # ____Deviation Event Frequencies____
            dev_freq_dir = os.path.join(results_dir, 'dev_frequency_plots')
            if os.path.isdir(dev_freq_dir) == False:
                os.mkdir(dev_freq_dir)
            print("\tCalculating Cross-Taste Deviation Frequencies")
            taste_dev_freq_dir = os.path.join(dev_freq_dir, 'cross_tastes')
            if os.path.isdir(taste_dev_freq_dir) == False:
                os.mkdir(taste_dev_freq_dir)
            cdf.cross_dataset_dev_freq_taste(self.corr_data, self.unique_given_names,
                                             self.unique_corr_names, self.unique_segment_names,
                                             self.unique_taste_names, taste_dev_freq_dir)
            print("\tCalculating Cross-Segment Deviation Frequencies")
            seg_dev_freq_dir = os.path.join(dev_freq_dir, 'cross_segments')
            if os.path.isdir(seg_dev_freq_dir) == False:
                os.mkdir(seg_dev_freq_dir)
            cdf.cross_dataset_dev_freq_seg(self.corr_data, self.unique_given_names,
                                           self.unique_corr_names, self.unique_segment_names,
                                           self.unique_taste_names, seg_dev_freq_dir)
            # ____Correlation Distributions____
            cross_segment_dir = os.path.join(
                results_dir, 'cross_segment_plots')
            if os.path.isdir(cross_segment_dir) == False:
                os.mkdir(cross_segment_dir)
            print("\tComparing Segments")
            cdf.cross_segment_diffs(self.corr_data, cross_segment_dir, self.unique_given_names,
                                    self.unique_corr_names, self.unique_segment_names, self.unique_taste_names)
            cross_taste_dir = os.path.join(results_dir, 'cross_taste_plots')
            if os.path.isdir(cross_taste_dir) == False:
                os.mkdir(cross_taste_dir)
            print("\tComparing Tastes")
            cdf.cross_taste_diffs(self.corr_data, cross_taste_dir, self.unique_given_names,
                                  self.unique_corr_names, self.unique_segment_names, self.unique_taste_names)
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

        print("Done.")

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
        segment start and end times, taste dig in names, and the correlation
        data for all neurons and taste-selective neurons"""

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
        
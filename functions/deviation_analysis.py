#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:24:35 2024

@author: Hannah Germaine

In this step of the analysis pipeline, deviation events are calculated and 
analyzed for basic statistics.
"""
import os
import json
import gzip
import itertools
import tqdm
from multiprocessing import Pool

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.dev_plot_funcs as dpf
import functions.dev_funcs as df

class run_find_deviations():

    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.calc_deviations()
        self.import_deviations()
        self.pull_devs()
        self.plot_devs()
        self.calc_dev_stats()

    def calc_deviations(self,):
        segment_names = self.data_dict['segment_names']
        num_segments = len(segment_names)
        segment_spike_times = self.data_dict['segment_spike_times']
        local_size = self.metadata['params_dict']['local_size']
        min_dev_size = self.metadata['params_dict']['min_dev_size']
        segment_times = self.data_dict['segment_times']
        segment_times_reshaped = [
            [segment_times[i], segment_times[i+1]] for i in range(num_segments)]
        self.segment_times_reshaped = segment_times_reshaped
        self.segments_to_analyze = self.metadata['params_dict']['segments_to_analyze']
        # Create deviation storage directory
        dev_dir = self.metadata['dir_name'] + 'Deviations/'
        if os.path.isdir(dev_dir) == False:
            os.mkdir(dev_dir)
        self.dev_dir = dev_dir

        # _____Import or calculate deviations for all segments_____
        """Deviations are calculated by (1) finding the prominence of min_dev_size 
		bin firing rates compared to firing rates from a local window of size local_size,
		(2) calculating the 90th percentile of positive prominence values, and (3) 
		pulling out those bins of time where the activity is above the 90th percentile
		prominence"""
        try:  # test if the data exists by trying to import the last
            filepath = dev_dir + segment_names[-1] + '/deviations.json'
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)
        except:
            seg_dirs = []
            seg_spike_times = []
            seg_times_reshaped = []
            for s_i in self.segments_to_analyze:
                # create storage directory
                seg_dir = dev_dir + segment_names[s_i] + '/'
                if os.path.isdir(seg_dir) == False:
                    os.mkdir(seg_dir)
                seg_dirs.append(seg_dir)
                seg_spike_times.append(segment_spike_times[s_i])
                seg_times_reshaped.append(segment_times_reshaped[s_i])
            print("\n\tNow calculating deviations")
            with Pool(processes=3) as pool:  # start 4 worker processes
                pool.map(df.run_dev_pull_parallelized, zip(seg_spike_times,
                                                           itertools.repeat(
                                                               local_size),
                                                           itertools.repeat(
                                                               min_dev_size),
                                                           seg_times_reshaped,
                                                           seg_dirs))
            pool.close()

    def import_deviations(self,):
        segment_names = self.data_dict['segment_names']
        dev_dir = self.dev_dir

        print("\tNow importing calculated deviations")
        segment_deviations = []
        for s_i in tqdm.tqdm(self.segments_to_analyze):
            filepath = dev_dir + segment_names[s_i] + '/deviations.json'
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)
                segment_deviations.append(data)

        self.segment_deviations = segment_deviations

    def pull_devs(self,):
        num_segments = len(self.segments_to_analyze)
        segment_spike_times = [self.data_dict['segment_spike_times'][i]
                               for i in self.segments_to_analyze]
        segment_times_reshaped = [self.segment_times_reshaped[i]
                                  for i in self.segments_to_analyze]
        segment_deviations = self.segment_deviations
        z_bin = self.metadata['params_dict']['z_bin']
        # _____Pull rasters of deviations and plot_____
        # Calculate segment deviation spikes
        print("\tNow pulling true deviation rasters")
        segment_dev_rasters, segment_dev_times, segment_dev_vec, \
            segment_dev_vec_zscore, _, _ = df.create_dev_rasters(num_segments,
                                            segment_spike_times, segment_times_reshaped,
                                            segment_deviations, z_bin)
        self.segment_dev_times = segment_dev_times
        self.segment_dev_rasters = segment_dev_rasters

    def plot_devs(self,):
        segment_deviations = self.segment_deviations
        segment_spike_times = [self.data_dict['segment_spike_times'][i]
                               for i in self.segments_to_analyze]
        segment_times_reshaped = [self.segment_times_reshaped[i]
                                  for i in self.segments_to_analyze]
        pre_taste = self.metadata['params_dict']['pre_taste']
        post_taste = self.metadata['params_dict']['post_taste']
        min_dev_size = self.metadata['params_dict']['min_dev_size']
        segment_names = [self.data_dict['segment_names'][i]
                         for i in self.segments_to_analyze]
        dig_in_names = self.data_dict['dig_in_names']
        dev_dir = self.dev_dir
        max_plot = self.metadata['params_dict']['max_plot']

        # Plot deviations
        print("\tNow plotting deviations")
        dpf.plot_dev_rasters(segment_deviations, segment_spike_times, self.segment_dev_times,
                             segment_times_reshaped, pre_taste, post_taste, min_dev_size,
                             segment_names, dev_dir, max_plot)

    def calc_dev_stats(self,):
        segment_names = [self.data_dict['segment_names'][i]
                         for i in self.segments_to_analyze]
        dev_dir = self.dev_dir

        # _____Calculate segment deviation statistics - length,IDI_____
        print("\tNow calculating and plotting true deviation statistics")
        segment_length_dict, segment_IDI_dict, segment_num_spike_dict, segment_num_neur_dict = df.calculate_dev_stats(self.segment_dev_rasters,
                                                                                                                      self.segment_dev_times, segment_names, dev_dir)

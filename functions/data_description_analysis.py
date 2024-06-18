#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:44:41 2024

@author: Hannah Germaine

This is the first step of the analysis pipeline: PSTHs, Raster Plots, etc... are analyzed here
"""

import os
import numpy as np

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.seg_compare as sc
import functions.plot_funcs as pf
import functions.analysis_funcs as af
import functions.hdf5_handling as hf5

class run_data_description_analysis():

    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.get_spike_time_datasets()
        self.get_psth_raster()
        self.seg_compare()

    def get_spike_time_datasets(self,):
        print('\tPulling Spike Times')

        # _____Pull out spike times for all tastes (and no taste)_____
        segment_spike_times = af.calc_segment_spike_times(
            self.data_dict['segment_times'], self.data_dict['spike_times'], self.data_dict['num_neur'])
        tastant_spike_times = af.calc_tastant_spike_times(self.data_dict['segment_times'], self.data_dict['spike_times'],
                                                          self.data_dict['start_dig_in_times'], self.data_dict['end_dig_in_times'],
                                                          self.metadata['params_dict']['pre_taste'], self.metadata['params_dict']['post_taste'], self.data_dict['num_tastes'], self.data_dict['num_neur'])

        self.data_dict['segment_spike_times'] = segment_spike_times
        self.data_dict['tastant_spike_times'] = tastant_spike_times

    def get_psth_raster(self,):
        print('\tMoving On to PSTH/Raster Plots')
        hdf5_dir = self.metadata['hdf5_dir']
        # Convert to ms timescale
        pre_taste_dt = int(
            np.ceil(self.metadata['params_dict']['pre_taste']*(1000/1)))
        # Convert to ms timescale
        post_taste_dt = int(
            np.ceil(self.metadata['params_dict']['post_taste']*(1000/1)))
        bin_width = 0.25  # Gaussian convolution kernel width in seconds
        bin_step = 25  # Step size in ms to take in PSTH calculation
        data_group_name = 'PSTH_data'
        try:
            tastant_PSTH = hf5.pull_data_from_hdf5(
                hdf5_dir, data_group_name, 'tastant_PSTH')
            PSTH_times = hf5.pull_data_from_hdf5(
                hdf5_dir, data_group_name, 'PSTH_times')
            PSTH_taste_deliv_times = hf5.pull_data_from_hdf5(
                hdf5_dir, data_group_name, 'PSTH_taste_deliv_times')
            avg_tastant_PSTH = hf5.pull_data_from_hdf5(
                hdf5_dir, data_group_name, 'avg_tastant_PSTH')
            print("\t\tPreviously Completed")
        except:
            data_save_dir = self.metadata['dir_name']
            start_dig_in_times = self.data_dict['start_dig_in_times']
            end_dig_in_times = self.data_dict['end_dig_in_times']
            segment_names = self.data_dict['segment_names']
            segment_times = self.data_dict['segment_times']
            segment_spike_times = self.data_dict['segment_spike_times']
            tastant_spike_times = self.data_dict['tastant_spike_times']
            num_neur = self.data_dict['num_neur']
            num_tastes = self.data_dict['num_tastes']
            dig_in_names = self.data_dict['dig_in_names']
            pf.raster_plots(data_save_dir, self.data_dict['dig_in_names'], start_dig_in_times, end_dig_in_times,
                            segment_names, segment_times, segment_spike_times,
                            tastant_spike_times, pre_taste_dt, post_taste_dt,
                            num_neur, num_tastes)
            PSTH_times, PSTH_taste_deliv_times, tastant_PSTH, avg_tastant_PSTH = pf.PSTH_plots(data_save_dir, num_tastes,
                                                                                               num_neur, dig_in_names,
                                                                                               start_dig_in_times, end_dig_in_times,
                                                                                               pre_taste_dt, post_taste_dt,
                                                                                               segment_times, segment_spike_times,
                                                                                               bin_width, bin_step)
            hf5.add_data_to_hdf5(hdf5_dir, data_group_name,
                                'tastant_PSTH', tastant_PSTH)
            hf5.add_data_to_hdf5(hdf5_dir, data_group_name,
                                'PSTH_times', PSTH_times)
            hf5.add_data_to_hdf5(hdf5_dir, data_group_name,
                                'PSTH_taste_deliv_times', PSTH_taste_deliv_times)
            hf5.add_data_to_hdf5(hdf5_dir, data_group_name,
                                'avg_tastant_PSTH', avg_tastant_PSTH)
            print("\t\tPlots Completed")

    def seg_compare(self,):
        print('\tComparing Segments')
        data_save_dir = self.metadata['dir_name']

        # _____Grab and plot firing rate distributions and comparisons (by segment)_____
        sc_save_dir = data_save_dir + 'Segment_Comparison/'
        if os.path.isdir(sc_save_dir) == False:
            os.mkdir(sc_save_dir)

        # All data
        all_sc_save_dir = sc_save_dir + 'All/'
        if os.path.isdir(all_sc_save_dir) == False:
            os.mkdir(all_sc_save_dir)
            sc.bin_spike_counts(all_sc_save_dir, self.data_dict['segment_spike_times'],
                                self.data_dict['segment_names'], self.data_dict['segment_times'])

    def taste_anova(self,):
        print('\tComparing Taste Deliveries')
        data_save_dir = self.metadata['dir_name']
        num_neur = self.data_dict['num_neur']
        num_tastes = self.data_dict['num_tastes']
        tastant_spike_times = self.data_dict['tastant_spike_times']
        start_dig_in_times = self.data_dict['start_dig_in_times']
        bin_size = 250
        max_time = 1500

        # _____Grab and plot firing rate distributions and comparisons (by segment)_____
        tds_save_dir = data_save_dir + 'Taste_Delivery_Similarity/'
        if os.path.isdir(tds_save_dir) == False:
            os.mkdir(tds_save_dir)

        # Full taste response
        af.full_taste_interval_2way_anova(num_tastes, num_neur, tastant_spike_times,
                                          start_dig_in_times, bin_size, max_time,
                                          tds_save_dir)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:08:51 2025

@author: Hannah Germaine

In this step of the analysis pipeline, held unit calculations (from 
detect_held_units.py) are imported and used to run deviation decoding with
the next (test) day's taste responses as well as the train day's responses.
"""

import os
import tqdm
import gzip
import json
import numpy as np

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

from functions.blech_held_units_funcs import *
from tkinter.filedialog import askdirectory

class run_multiday_analysis():
    
    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.num_days = len(self.data_dict)
        self.create_save_dir()
        self.gather_variables()
        
    def create_save_dir(self,):
        # Using the directories of the different days find a common root folder and create save dir there
        joint_name = self.metadata[0]['info_dict']['name']
        day_dirs = []
        for n_i in range(self.num_days):
            day_name = self.metadata[n_i]['info_dict']['exp_type']
            joint_name = joint_name + '_' + day_name
            day_dir = self.data_dict[n_i]['data_path']
            day_dirs.append(os.path.split(day_dir))
        day_dir_array = np.array(day_dirs)
        stop_ind = -1
        while stop_ind == -1:
            for i in range(np.shape(day_dir_array)[1]):
                if len(np.unique(day_dir_array[:,i])) > 1:
                    stop_ind = i
            if stop_ind == -1:
                day_dirs = []
                for n_i in range(self.num_days):
                    day_dir_list = day_dirs[n_i]
                    new_day_dir_list = os.path.split(day_dir_list[0])
                    new_day_dir_list.extend(day_dir_list[1:])
                    day_dirs.append(new_day_dir_list)
        #Now create the new folder in the shared root path
        root_path = os.path.join(*list(day_dir_array[0,:stop_ind]))
        self.save_dir = os.path.join(root_path,joint_name)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        
    def gather_variables(self,):
        
        day_vars = dict()
        for n_i in range(self.num_days):
            day_vars[n_i] = dict()
            # Directories
            day_vars[n_i]['hdf5_dir'] = self.metadata[0]['hdf5_dir']
            # General Params/Variables
            num_neur = self.data_dict['num_neur']
            keep_neur = self.metadata['held_units'][:,n_i]
            day_vars[n_i]['pre_taste'] = self.metadata[n_i]['params_dict']['pre_taste']
            day_vars[n_i]['post_taste'] = self.metadata[n_i]['params_dict']['post_taste']
            day_vars[n_i]['pre_taste_dt'] = np.ceil(day_vars[n_i]['pre_taste']*1000).astype('int')
            day_vars[n_i]['post_taste_dt'] = np.ceil(day_vars[n_i]['pre_taste']*1000).astype('int')
            day_vars[n_i]['segments_to_analyze'] = self.metadata['params_dict']['segments_to_analyze']
            day_vars[n_i]['epochs_to_analyze'] = self.metadata['params_dict']['epochs_to_analyze']
            self.segment_names = self.data_dict['segment_names']
            self.num_segments = len(self.segment_names)
            self.segment_spike_times = self.data_dict['segment_spike_times']
            self.segment_times = self.data_dict['segment_times']
            self.segment_times_reshaped = [
                [self.segment_times[i], self.segment_times[i+1]] for i in range(self.num_segments)]
            # Remember this imported value is 1 less than the number of epochs
            self.num_cp = self.metadata['params_dict']['num_cp'] + 1
            self.tastant_spike_times = self.data_dict['tastant_spike_times']
            self.start_dig_in_times = self.data_dict['start_dig_in_times']
            self.end_dig_in_times = self.data_dict['end_dig_in_times']
            self.dig_in_names = self.data_dict['dig_in_names']
            self.num_tastes = len(self.dig_in_names)
            self.fr_bins = self.metadata['params_dict']['fr_bins']
            #Bayes Params/Variables
            self.skip_time = self.metadata['params_dict']['bayes_params']['skip_time']
            self.skip_dt = np.ceil(self.skip_time*1000).astype('int')
            self.e_skip_time = self.metadata['params_dict']['bayes_params']['e_skip_time']
            self.e_skip_dt = np.ceil(self.e_skip_time*1000).astype('int')
            self.taste_e_len_time = self.metadata['params_dict']['bayes_params']['taste_e_len_time']
            self.taste_e_len_dt = np.ceil(self.taste_e_len_time*1000).astype('int') 
            self.seg_e_len_time = self.metadata['params_dict']['bayes_params']['seg_e_len_time']
            self.seg_e_len_dt = np.ceil(self.seg_e_len_time*1000).astype('int') 
            self.bayes_fr_bins = self.metadata['params_dict']['bayes_params']['fr_bins']
            self.neuron_count_thresh = self.metadata['params_dict']['bayes_params']['neuron_count_thresh']
            self.max_decode = self.metadata['params_dict']['bayes_params']['max_decode']
            self.seg_stat_bin = self.metadata['params_dict']['bayes_params']['seg_stat_bin']
            self.trial_start_frac = self.metadata['params_dict']['bayes_params']['trial_start_frac']
            self.decode_prob_cutoff = self.metadata['params_dict']['bayes_params']['decode_prob_cutoff']
            self.bin_time = self.metadata['params_dict']['bayes_params']['z_score_bin_time']
            self.bin_dt = np.ceil(self.bin_time*1000).astype('int')
            self.num_null = 100 #self.metadata['params_dict']['num_null']
            # Import changepoint data
            self.pop_taste_cp_raster_inds = hf5.pull_data_from_hdf5(
                self.hdf5_dir, 'changepoint_data', 'pop_taste_cp_raster_inds')
            self.num_pt_cp = self.num_cp + 2
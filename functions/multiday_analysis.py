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

from utils.input_funcs import *
from functions.blech_held_units_funcs import *
import functions.analysis_funcs as af
import functions.dev_funcs as dev_f
import functions.hdf5_handling as hf5
import functions.dependent_decoding_funcs as ddf
import functions.multiday_dev_functions as mdf
import functions.multiday_nn_funcs as mnf
import functions.dev_funcs as df


class run_multiday_analysis():
    
    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.num_days = len(self.data_dict)
        self.create_save_dir()
        self.gather_variables()
        self.decode_groups()
        self.import_deviations()
        self.get_training_data()
        self.multiday_dev_corr()
        
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
        #Now create the new folders in the shared root path
        root_path = os.path.join(*list(day_dir_array[0,:stop_ind]))
        self.save_dir = os.path.join(root_path,joint_name)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        self.corr_dir = os.path.join(self.save_dir,'Correlations')
        if os.path.isdir(self.corr_dir) == False:
            os.mkdir(self.corr_dir)
        self.bayes_dir = os.path.join(self.save_dir,\
                                      'Naive_Bayes_GMM')
        if os.path.isdir(self.bayes_dir) == False:
            os.mkdir(self.bayes_dir)
        
    def gather_variables(self,):
        #For each day store the relevant variables/parameters
        day_vars = dict()
        for n_i in range(self.num_days):
            day_vars[n_i] = dict()
            # Directories
            day_vars[n_i]['hdf5_dir'] = os.path.join(self.metadata[n_i]['dir_name'], self.metadata[n_i]['hdf5_dir'])
            day_vars[n_i]['dev_dir'] = os.path.join(self.metadata[n_i]['dir_name'],'Deviations')
            day_vars[n_i]['null_dir'] = os.path.join(self.metadata[n_i]['dir_name'],'null_data')
            # General Params/Variables
            num_neur = self.data_dict[n_i]['num_neur']
            keep_neur = self.metadata['held_units'][:,n_i]
            day_vars[n_i]['keep_neur'] = keep_neur
            day_vars[n_i]['pre_taste'] = self.metadata[n_i]['params_dict']['pre_taste']
            day_vars[n_i]['post_taste'] = self.metadata[n_i]['params_dict']['post_taste']
            day_vars[n_i]['pre_taste_dt'] = np.ceil(day_vars[n_i]['pre_taste']*1000).astype('int')
            day_vars[n_i]['post_taste_dt'] = np.ceil(day_vars[n_i]['pre_taste']*1000).astype('int')
            day_vars[n_i]['segments_to_analyze'] = self.metadata[n_i]['params_dict']['segments_to_analyze']
            day_vars[n_i]['epochs_to_analyze'] = self.metadata[n_i]['params_dict']['epochs_to_analyze']
            day_vars[n_i]['segment_names'] = self.data_dict[n_i]['segment_names']
            day_vars[n_i]['num_segments'] = len(day_vars[n_i]['segment_names'])
            day_vars[n_i]['segment_times'] = self.data_dict[n_i]['segment_times']
            day_vars[n_i]['segment_times_reshaped'] = [
                [day_vars[n_i]['segment_times'][i], day_vars[n_i]['segment_times'][i+1]] for i in range(day_vars[n_i]['num_segments'])]
            # Remember this imported value is 1 less than the number of epochs
            day_vars[n_i]['num_cp'] = self.metadata[n_i]['params_dict']['num_cp'] + 1
            day_vars[n_i]['start_dig_in_times'] = self.data_dict[n_i]['start_dig_in_times']
            day_vars[n_i]['end_dig_in_times'] = self.data_dict[n_i]['end_dig_in_times']
            day_vars[n_i]['dig_in_names'] = self.data_dict[n_i]['dig_in_names']
            day_vars[n_i]['num_tastes'] = len(day_vars[n_i]['dig_in_names'])
            day_vars[n_i]['fr_bins'] = self.metadata[n_i]['params_dict']['fr_bins']
            day_vars[n_i]['z_bin'] = self.metadata[n_i]['params_dict']['z_bin']
            segment_spike_times, tastant_spike_times = self.get_spike_time_datasets(
                [day_vars[n_i]['segment_times'],self.data_dict[n_i]['spike_times'],
                 num_neur, keep_neur, day_vars[n_i]['start_dig_in_times'],
                 day_vars[n_i]['end_dig_in_times'], day_vars[n_i]['pre_taste'],
                 day_vars[n_i]['post_taste'], day_vars[n_i]['num_tastes']])
            day_vars[n_i]['segment_spike_times'] = segment_spike_times
            day_vars[n_i]['tastant_spike_times'] = tastant_spike_times
            #Bayes Params/Variables
            day_vars[n_i]['skip_time'] = self.metadata[n_i]['params_dict']['bayes_params']['skip_time']
            day_vars[n_i]['skip_dt'] = np.ceil(day_vars[n_i]['skip_time']*1000).astype('int')
            day_vars[n_i]['e_skip_time'] = self.metadata[n_i]['params_dict']['bayes_params']['e_skip_time']
            day_vars[n_i]['e_skip_dt'] = np.ceil(day_vars[n_i]['e_skip_time']*1000).astype('int')
            day_vars[n_i]['taste_e_len_time'] = self.metadata[n_i]['params_dict']['bayes_params']['taste_e_len_time']
            day_vars[n_i]['taste_e_len_dt'] = np.ceil(day_vars[n_i]['taste_e_len_time']*1000).astype('int') 
            day_vars[n_i]['seg_e_len_time'] = self.metadata[n_i]['params_dict']['bayes_params']['seg_e_len_time']
            day_vars[n_i]['seg_e_len_dt'] = np.ceil(day_vars[n_i]['seg_e_len_time']*1000).astype('int') 
            day_vars[n_i]['bayes_fr_bins'] = self.metadata[n_i]['params_dict']['bayes_params']['fr_bins']
            day_vars[n_i]['neuron_count_thresh'] = self.metadata[n_i]['params_dict']['bayes_params']['neuron_count_thresh']
            day_vars[n_i]['max_decode'] = self.metadata[n_i]['params_dict']['bayes_params']['max_decode']
            day_vars[n_i]['seg_stat_bin'] = self.metadata[n_i]['params_dict']['bayes_params']['seg_stat_bin']
            day_vars[n_i]['trial_start_frac'] = self.metadata[n_i]['params_dict']['bayes_params']['trial_start_frac']
            day_vars[n_i]['decode_prob_cutoff'] = self.metadata[n_i]['params_dict']['bayes_params']['decode_prob_cutoff']
            day_vars[n_i]['bin_time'] = self.metadata[n_i]['params_dict']['bayes_params']['z_score_bin_time']
            day_vars[n_i]['bin_dt'] = np.ceil(day_vars[n_i]['bin_time']*1000).astype('int')
            day_vars[n_i]['num_null'] = 100 #self.metadata['params_dict']['num_null']
            # Import changepoint data
            day_vars[n_i]['pop_taste_cp_raster_inds'] = hf5.pull_data_from_hdf5(
                day_vars[n_i]['hdf5_dir'], 'changepoint_data', 'pop_taste_cp_raster_inds')
            day_vars[n_i]['num_pt_cp'] = day_vars[n_i]['num_cp'] + 2
            
        self.day_vars = day_vars
        
    def get_spike_time_datasets(self,args):
        segment_times, spike_times, num_neur, keep_neur, start_dig_in_times, \
            end_dig_in_times, pre_taste, post_taste, num_tastes = args
        
        # _____Pull out spike times for all tastes (and no taste)_____
        segment_spike_times = af.calc_segment_spike_times(
            segment_times, spike_times, num_neur)
        tastant_spike_times = af.calc_tastant_spike_times(segment_times, spike_times,
                                                          start_dig_in_times, end_dig_in_times,
                                                          pre_taste, post_taste, 
                                                          num_tastes, num_neur)
        
        # _____Update spike times for only held units_____
        keep_segment_spike_times = []
        for s_i in range(len(segment_spike_times)):
            seg_neur_keep = []
            for n_i in keep_neur:
                seg_neur_keep.append(segment_spike_times[s_i][n_i])
            keep_segment_spike_times.append(seg_neur_keep)
            
        keep_tastant_spike_times = []
        for t_i in range(len(tastant_spike_times)):
            taste_deliv_list = []
            for d_i in range(len(tastant_spike_times[t_i])):
                deliv_neur_keep = []
                for n_i in keep_neur:
                    deliv_neur_keep.append(tastant_spike_times[t_i][d_i][n_i])
                taste_deliv_list.append(deliv_neur_keep)
            keep_tastant_spike_times.append(taste_deliv_list)
            
        return keep_segment_spike_times, keep_tastant_spike_times
    
    def decode_groups(self,):
        print("Determine decoding groups")
        all_dig_in_names = []
        for n_i in range(self.num_days):
            #Collect tastant names
            day_names = self.day_vars[n_i]['dig_in_names']
            new_day_names = [dn + '_' + str(n_i) for dn in day_names]
            all_dig_in_names.extend(new_day_names)
        self.all_dig_in_names = all_dig_in_names
        
        #Create fr vector grouping instructions: list of epoch,taste pairs
        non_none_tastes = [taste for taste in self.all_dig_in_names if taste[:4] != 'none']
        self.non_none_tastes = non_none_tastes
        group_list, group_names = ddf.decode_groupings(self.day_vars[0]['epochs_to_analyze'],
                                                       self.all_dig_in_names,
                                                       self.non_none_tastes)
        #Save the group information for cross-animal use 
        group_dict = dict()
        for gn_i, gn in enumerate(group_names):
            group_dict[gn] = group_list[gn_i]
        np.save(os.path.join(self.bayes_dir,'group_dict.npy'),group_dict,allow_pickle=True)

        self.group_list = group_list
        self.group_names = group_names
        
    def import_deviations(self,):
        print("\tNow importing calculated deviations for first day")
        
        num_seg_to_analyze = len(self.day_vars[0]['segments_to_analyze'])
        segment_names_to_analyze = [self.day_vars[0]['segment_names'][i] for i in self.day_vars[0]['segments_to_analyze']]
        segment_times_to_analyze_reshaped = [
            [self.day_vars[0]['segment_times'][i], self.day_vars[0]['segment_times'][i+1]] for i in self.day_vars[0]['segments_to_analyze']]
        segment_spike_times_to_analyze = [self.day_vars[0]['segment_spike_times'][i] for i in self.day_vars[0]['segments_to_analyze']]
        self.segment_names_to_analyze = segment_names_to_analyze
        
        segment_deviations = []
        for s_i in tqdm.tqdm(range(num_seg_to_analyze)):
            filepath = os.path.join(self.day_vars[0]['dev_dir'],segment_names_to_analyze[s_i],'deviations.json')
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)
                segment_deviations.append(data)
        self.segment_deviations = segment_deviations
        
        print("\tNow pulling true deviation rasters")
        #Note, these will already reflect the held units
        segment_dev_rasters, segment_dev_times, segment_dev_fr_vecs, \
            segment_dev_fr_vecs_zscore, _, _ = dev_f.create_dev_rasters(num_seg_to_analyze, 
                                                                segment_spike_times_to_analyze,
                                                                np.array(segment_times_to_analyze_reshaped),
                                                                segment_deviations, self.day_vars[0]['pre_taste'])
        self.segment_dev_rasters = segment_dev_rasters
        self.segment_dev_times = segment_dev_times
        self.segment_dev_fr_vecs = segment_dev_fr_vecs
        self.segment_dev_fr_vecs_zscore = segment_dev_fr_vecs_zscore
        
    def get_training_data(self,):
        group_train_data = ddf.pull_group_training_data(group_list, group_names, \
                                                        all_dig_in_names, day_vars)
        self.group_train_data = group_train_data
        np.save(os.path.join(self.bayes_dir,'group_train_data.npy'),group_train_data,allow_pickle=True)
        
        control_data = ddf.create_control_dict(group_train_data,day_vars,\
                                               segment_deviations,segment_dev_rasters)
        self.control_data = control_data
        np.save(os.path.join(self.bayes_dir,'control_data.npy'),control_data,allow_pickle=True)
        
    def multiday_dev_corr(self,):
        """
        Runs correlation between deviation events and taste responses as well
        as probabilistic decoding of deviation events using taste responses. 
        """
        
        mdf.multiday_dev_analysis(self.save_dir,self.all_dig_in_names,self.tastant_fr_dist_pop,
                                  self.taste_num_deliv,self.max_hz_pop,self.tastant_fr_dist_z_pop,
                                  self.max_hz_z_pop,self.min_hz_z_pop,self.max_num_cp,
                                  self.segment_dev_rasters,self.segment_dev_times,
                                  self.segment_dev_fr_vecs,self.segment_dev_fr_vecs_zscore,
                                  self.day_vars[0]['segments_to_analyze'],
                                  self.day_vars[0]['segment_times'], 
                                  self.day_vars[0]['segment_spike_times'],
                                  self.day_vars[0]['bin_dt'],self.segment_names_to_analyze)
    
    def decode_zscored(self,):
        """
        Runs decoding of zscored data
        """
        self.decode_dir = os.path.join(self.bayes_dir,'All_Neurons_Z_Scored')
        if os.path.isdir(self.decode_dir) == False:
            os.mkdir(self.decode_dir)
        self.z_score = True
        self.tastant_fr_dist = self.tastant_fr_dist_z_pop
        self.dev_vecs = self.segment_dev_fr_vecs_zscore
        self.segment_times = self.day_vars[0]['segment_times']
        self.segment_names = self.day_vars[0]['segment_names']
        self.start_dig_in_times = self.day_vars[0]['start_dig_in_times']
        self.bin_dt = self.day_vars[0]['bin_dt']
        self.epochs_to_analyze = self.day_vars[0]['epochs_to_analyze']
        self.segments_to_analyze = self.day_vars[0]['segments_to_analyze']
        self.decode_dev()
        
    def decode_nonzscored(self,):
        print("\tRun non-z-scored data decoder pipeline")
        self.decode_dir = os.path.join(self.bayes_dir,'All_Neurons')
        if os.path.isdir(self.decode_dir) == False:
            os.mkdir(self.decode_dir)
        self.z_score = False
        self.tastant_fr_dist = self.tastant_fr_dist_pop
        self.dev_vecs = self.segment_dev_fr_vecs
        self.segment_times = self.day_vars[0]['segment_times']
        self.segment_names = self.day_vars[0]['segment_names']
        self.start_dig_in_times = self.day_vars[0]['start_dig_in_times']
        self.bin_dt = self.day_vars[0]['bin_dt']
        self.epochs_to_analyze = self.day_vars[0]['epochs_to_analyze']
        self.segments_to_analyze = self.day_vars[0]['segments_to_analyze']
        self.decode_dev()
    
    def decode_dev(self,):
        print("\t\tDecoding deviation events.")
        ddf.decoder_accuracy_tests(self.tastant_fr_dist, self.segment_spike_times, 
                        self.all_dig_in_names, self.segment_times, self.segment_names, 
                        self.start_dig_in_times, self.taste_num_deliv,
                        self.group_list, self.group_names, self.non_none_tastes, 
                        self.decode_dir, self.bin_dt, self.z_score, 
                        self.epochs_to_analyze, self.segments_to_analyze)
        
        ddf.decode_sliding_bins(self.tastant_fr_dist, self.segment_spike_times, 
                                self.all_dig_in_names, self.segment_times, 
                                self.segment_names, self.start_dig_in_times, 
                                self.taste_num_deliv, self.bin_dt, self.group_list, 
                                self.group_names, self.non_none_tastes, self.decode_dir, 
                                self.z_score, self.segments_to_analyze)

        ddf.decode_deviations(self.tastant_fr_dist, self.tastant_spike_times,
                              self.segment_spike_times, self.all_dig_in_names, 
                              self.segment_times, self.segment_names, 
                              self.start_dig_in_times, self.taste_num_deliv, 
                              self.segment_dev_times, self.dev_vecs, 
                              self.bin_dt, self.group_list, self.group_names, 
                              self.non_none_tastes, self.decode_dir, self.z_score, 
                              self.epochs_to_analyze, self.segments_to_analyze)
        
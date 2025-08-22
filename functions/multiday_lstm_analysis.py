#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:37:54 2025

@author: hannahgermaine
"""

import os

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import gzip
import json
import tqdm
import numpy as np
import functions.analysis_funcs as af
import functions.hdf5_handling as hf5
import functions.lstm_decoding_funcs as lstm


class run_multiday_lstm_analysis():
    
    def __init__(self,args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.num_days = len(self.data_dict)
        self.num_bins = 4
        self.z_bin_dt = 100
        self.create_save_dir()
        self.gather_variables()
        self.get_deviation_matrices()
        self.run_size_tests()
        
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
            
        self.lstm_dir = os.path.join(self.save_dir,'LSTM_Decoding')
        if os.path.isdir(self.lstm_dir) == False:
            os.mkdir(self.lstm_dir)
    
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
            day_vars[n_i]['min_dev_size'] = self.metadata[n_i]['params_dict']['min_dev_size']
            day_vars[n_i]['local_size'] = self.metadata[n_i]['params_dict']['local_size']
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
    
    def get_taste_response_matrices(self,null_taste):
        day_1_tastes = self.day_vars[0]['dig_in_names']
        all_dig_in_names = []
        all_dig_in_names.extend([d1 + '_0' for d1 in day_1_tastes])
        all_segment_times = []
        all_tastant_spike_times = []
        for n_i in np.arange(1,self.num_days):
            new_day_tastes =self.day_vars[n_i]['dig_in_names']
            all_dig_in_names.extend([ndt + '_' + str(n_i) for ndt in new_day_tastes if 
                                     len(np.intersect1d(np.array([ndt.split('_')]),np.array(day_1_tastes))) == 0])
            
        taste_unique_categories, training_matrices, training_labels = lstm.create_taste_matrices(\
                               self.day_vars, null_taste, all_dig_in_names, \
                                   self.num_bins, self.z_bin_dt, start_bins=0)
        
        self.taste_unique_categories = taste_unique_categories
        self.training_matrices = training_matrices
        self.training_labels = training_labels
        
    def get_deviation_matrices(self,):
        
        num_seg_to_analyze = len(self.day_vars[0]['segments_to_analyze'])
        segment_names_to_analyze = [self.day_vars[0]['segment_names'][i] for i in self.day_vars[0]['segments_to_analyze']]
        segment_times_to_analyze_reshaped = [
            [self.day_vars[0]['segment_times'][i],self.day_vars[0]['segment_times'][i+1]] for i in self.day_vars[0]['segments_to_analyze']]
        segment_spike_times_to_analyze = [self.day_vars[0]['segment_spike_times'][i] for i in self.day_vars[0]['segments_to_analyze']]
        
        segment_deviations = []
        for s_i in tqdm.tqdm(range(num_seg_to_analyze)):
            filepath = os.path.join(self.day_vars[0]['dev_dir'],segment_names_to_analyze[s_i],'deviations.json')
            with gzip.GzipFile(filepath, mode="r") as f:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)
                segment_deviations.append(data)

        # get deviation matrices        
        try:
            dev_matrices = np.load(os.path.join(self.lstm_dir,'dev_fr_vecs_zscore.npy'),allow_pickle=True).item()
            null_dev_matrices = np.load(os.path.join(self.lstm_dir,'null_dev_fr_vecs_zscore.npy'),allow_pickle=True).item()
        except:
            dev_matrices, null_dev_matrices = lstm.create_dev_matrices(self.day_vars, segment_deviations, self.z_bin_dt, self.num_bins)
            np.save(os.path.join(self.lstm_dir,'dev_fr_vecs_zscore.npy'),dev_matrices,allow_pickle=True)
            np.save(os.path.join(self.lstm_dir,'null_dev_fr_vecs_zscore.npy'),null_dev_matrices,allow_pickle=True)
        
        self.dev_matrices = dev_matrices
        
    def run_cross_validation(self,):
        
        print("\n--- Running cross-validation ---")
            
        cv_save_dir = os.path.join(self.lstm_dir,'cross_validation')
        if not os.path.isdir(cv_save_dir):
            os.mkdir(cv_save_dir)
        
        #Cross-validation
        try:
            fold_dict = np.load(os.path.join(cv_save_dir,'fold_dict.npy'),allow_pickle=True).item()
        except:
            taste_unique_categories, training_matrices, training_labels = self.get_taste_response_matrices()
            
            lstm.lstm_cross_validation(training_matrices,\
                                    training_labels,taste_unique_categories,\
                                        cv_save_dir)
                
            fold_dict = np.load(os.path.join(cv_save_dir,'fold_dict.npy'),allow_pickle=True).item()
            
        #Best size calculation
        best_dim, score_curve, tested_latent_dim = lstm.get_best_size(fold_dict,cv_save_dir)
        
        # plot the score curves and best dim
        lstm.cross_start_scores(self.start_bin_array, score_curves, latent_dims, \
                                   best_dims, self.lstm_dir)
            
        self.best_dim = best_dim
        self.score_curve = score_curve
        self.tested_latent_dim = tested_latent_dim
       
    def run_deviation_decoding(self,):
        # across start times compare the decoding of each event - using a
        # democratic approach to assigning the decoded taste
        try:
            predictions = np.load(os.path.join(self.lstm_dir,'predictions.npy'),allow_pickle=True).item()
        except:
            taste_unique_categories = np.load(os.path.join(self.lstm_dir,'taste_unique_categories.npy'))
            training_matrices = list(np.load(os.path.join(self.lstm_dir,'training_matrices.npy')))
            training_labels = list(np.load(os.path.join(self.lstm_dir,'training_labels.npy')))
            
            #Run decoding
            
            predictions = lstm.lstm_dev_decoding(self.dev_matrices, training_matrices, training_labels,\
                                  self.best_dim, taste_unique_categories, self.lstm_dir)
            
            np.save(os.path.join(sb_save_dir,'predictions.npy'),predictions,allow_pickle=True)
            
        
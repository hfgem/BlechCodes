#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 10:48:23 2025

@author: hannahgermaine
"""

import os

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.dependent_decoding_funcs as ddf
import functions.analysis_funcs as af
import functions.dev_funcs as dev_f
import functions.lstm_funcs as lstm
import numpy as np


class run_multiday_LSTM_analysis():
    
    def __init__(self,args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.num_days = len(self.data_dict)
        self.create_save_dir()
        self.gather_variables()
        self.get_spike_time_datasets()
        self.pull_taste_fr_trajectories()
        self.import_deviations()
        self.run_lstm()
        
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
        self.lstm_dir = os.path.join(self.save_dir,\
                                      'LSTM_Decoding')
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
    
    def pull_taste_fr_trajectories(self,):
        num_days = len(day_vars)
        fr_bin_size = 100 #ms/samples per bin to use in taste trajectory
        resp_len = 2000 #length of taste response interval
        bin_starts = np.arange(0,resp_len,fr_bin_size)

        all_dig_in_names = []
        fr_dict = dict()
        fr_z_dict = dict()
        for n_i in range(num_days):
            fr_dict[n_i] = dict()
            fr_z_dict[n_i] = dict()
            
            #Gather variables
            tastant_spike_times = day_vars[n_i]['tastant_spike_times']
            segment_names = day_vars[n_i]['segment_names']
            seg_taste_ind = [i for i in range(len(segment_names)) if segment_names[i] == 'taste'][0]
            segment_times = day_vars[n_i]['segment_times']
            taste_segment_times = day_vars[n_i]['segment_spike_times'][seg_taste_ind]
            start_dig_in_times = day_vars[n_i]['start_dig_in_times']
            dig_in_names = day_vars[n_i]['dig_in_names']
            new_day_names = [dn + '_' + str(n_i) for dn in dig_in_names]
            all_dig_in_names.extend(new_day_names)
            num_neur = len(day_vars[n_i]['keep_neur'])
            
            #Calculate z-scoring info
            neur_mean, neur_std = self.calc_seg_z_score(segment_times,seg_taste_ind,num_neur,seg_len,\
                                 taste_segment_times,fr_bin_size)
            
            #Calculate taste response vectors
            for t_i, t_name in enumerate(dig_in_names):
                fr_dict[n_i][t_name] = []
                fr_z_dict[n_i][t_name] = []
                t_starts = start_dig_in_times[t_i]
                t_times = tastant_spike_times[t_i]
                for d_i in range(len(t_starts)): #Go through each delivery
                    ts_i = t_starts[d_i]
                    tt_i = t_times[d_i]
                    #Get spike raster
                    bin_spikes = np.zeros((num_neur,resp_len))
                    for nn_i in range(num_neur):
                        s_times = (np.array(tt_i[nn_i]) - ts_i).astype('int')
                        bin_spikes[nn_i,s_times[s_times < resp_len]] = 1
                    #Convert to binned fr vectors
                    fr_vecs = np.zeros((num_neur,len(bin_starts)))
                    for bs_ind, bs_i in enumerate(bin_starts):
                        fr_i = np.sum(bin_spikes[:,bs_i:bs_i+fr_bin_size],1)/(fr_bin_size/1000)
                        fr_vecs[:,bs_ind] = fr_i
                    fr_z_vecs = (fr_vecs - np.expand_dims(neur_mean,1))/np.expand_dims(neur_std,1)
                    fr_dict[n_i][t_name].append(fr_vecs)
                    fr_z_dict[n_i][t_name].append(fr_z_vecs)
                    
        self.fr_dict = fr_dict
        self.fr_z_dict = fr_z_dict
        self.all_dig_in_names = all_dig_in_names
        
    def calc_seg_z_score(self,segment_times,seg_taste_ind,num_neur,seg_len,\
                         taste_segment_times,fr_bin_size):
        """Calculate the mean and standard deviations of neuron firing rates
        for the taste segment"""
        #Calculate z-scoring info
        seg_start = segment_times[seg_taste_ind]
        seg_end = segment_times[seg_taste_ind+1]
        seg_len = seg_end - seg_start
        seg_bin = np.zeros((num_neur,seg_len+1))
        for nn_i in range(num_neur):
            spike_inds = (np.array(taste_segment_times[nn_i]) - seg_start).astype('int')
            seg_bin[nn_i,spike_inds] = 1
        seg_bin_starts = np.arange(0,seg_len,fr_bin_size)
        seg_fr_array = np.zeros((num_neur,len(seg_bin_starts)))
        for sbs_ind, sbs_i in enumerate(seg_bin_starts):
            seg_fr_array[:,sbs_ind] = np.sum(seg_bin[:,sbs_i:sbs_i+fr_bin_size],1)/(fr_bin_size/1000)
        neur_mean = np.mean(seg_fr_array,1)
        neur_std = np.std(seg_fr_array,1)
        
        return neur_mean, neur_std
    
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

        print("\tNow pulling true deviation rasters")
        #Note, these will already reflect the held units
        dev_rasters, dev_times, _, dev_fr_vecs_zscore, _, _ = lstm.create_dev_rasters(num_seg_to_analyze, 
                                                            segment_spike_times_to_analyze,
                                                            np.array(segment_times_to_analyze_reshaped),
                                                            segment_deviations, day_vars[0]['pre_taste'])
        
        self.dev_rasters = dev_rasters
        self.dev_times = dev_times
        self.dev_fr_vecs_zscore = dev_fr_vecs_zscore
        
    def run_lstm_training(self,):
        """Run LSTM training tests on taste dataset"""
        
        day_1_tastes = day_vars[0]['dig_in_names']
        #Format training data
        self.data_array, self.data_labels, self.data_inds = lstm.prep_lstm_data(self.fr_z_dict,\
                                                                        day_1_tastes,\
                                                                        self.all_dig_in_names)
        #Run LSTM size tests
        self.best_lstm_accuracies, self.best_lstm_size = lstm.run_model_tests(self.data_array,\
                                                                        self.data_inds)
        
    def run_lstm_on_dev(self,):
        """Run LSTM on deviations"""
        
        #Run LSTM on deviations
        
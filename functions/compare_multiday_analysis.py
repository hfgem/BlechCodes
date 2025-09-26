#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:03:51 2025

@author: Hannah Germaine

Functions to support comparing multiday analysis results for multiple animals.
"""

import os
import warnings
import tqdm
import random
import numpy as np
import functions.compare_multiday_funcs as cmf
import functions.cross_animal_multiday_lstm_funcs as camlf
from tkinter.filedialog import askdirectory
from utils.input_funcs import select_analysis_groups

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

warnings.filterwarnings("ignore")

class run_compare_multiday_analysis():
    
    def __init__(self, args):
        self.multiday_data_dict = args[0]
        self.save_dir = args[1]
        self.verbose = args[2]
        self.gather_corr_data()
        self.find_corr_groupings()
        self.gather_null_corr_data()
        self.run_corr_analysis()
        self.gather_decode_data()
        self.find_decode_groupings()
        self.run_decode_analysis()
        
    def gather_corr_data(self,):
        print("Collecting correlation data")
        corr_dict_path = os.path.join(self.save_dir,'corr_data_dict.npy')
        self.corr_dict_path = corr_dict_path
        try:
            corr_dict = np.load(corr_dict_path,allow_pickle=True).item()
        except:
            corr_dict = dict()
            data_names = list(self.multiday_data_dict.keys())
            for nc_i, dn in enumerate(data_names):
                corr_dict[dn] = dict()
                data_dir = self.multiday_data_dict[dn]['data_dir']
                corr_dir = os.path.join(data_dir,'Correlations')
                corr_files = os.listdir(corr_dir)
                npy_files = [cf for cf in corr_files if cf[-3:] == 'npy']
                for npy_f in npy_files:
                    seg_name = npy_f.split('_')[0]
                    #Regular data
                    corr_dict[dn][seg_name] = np.load(os.path.join(corr_dir,npy_f),allow_pickle=True).item()
                    #Fix CM46 naming of salt
                    try:
                        salt_data = corr_dict[dn][seg_name]['Salt_1 Identity']
                        del corr_dict[dn][seg_name]['Salt_1 Identity']
                        corr_dict[dn][seg_name]['Nacl_1 Identity'] = salt_data
                    except:
                        if self.verbose == True:
                            print("No Salt_1 data to rename for " + dn)
            
            np.save(corr_dict_path,corr_dict,allow_pickle=True)   
        self.corr_dict = corr_dict
        
    def find_corr_groupings(self,):
        print("Finding unique correlation groups")
        num_datasets = len(self.corr_dict)
        unique_given_names = list(self.corr_dict.keys())
        #Pull unique correlation analysis names
        unique_segment_names = []
        unique_group_names = []
        for name in unique_given_names:
            unique_gn_seg_names = list(self.corr_dict[name].keys())
            unique_segment_names.extend(unique_gn_seg_names)
            unique_group_names.extend(list(self.corr_dict[name][unique_gn_seg_names[0]].keys()))
        unique_seg_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i] for i in unique_seg_indices]
        unique_group_indices = np.sort(
            np.unique(unique_group_names, return_index=True)[1])
        unique_group_names = [unique_group_names[i] for i in unique_group_indices]
            
        self.unique_given_names = unique_given_names
        self.unique_group_names = unique_group_names
        self.unique_segment_names = unique_segment_names
    
    def run_corr_analysis(self,):
        print("Running correlation analysis")
        cmf.compare_corr_data(self.corr_dict, self.multiday_data_dict, 
                              self.unique_given_names, self.unique_segment_names, 
                              self.unique_group_names, self.save_dir)
        
    def gather_decode_data(self,):
        print("Collecting decoding data")
        decode_dict_path = os.path.join(self.save_dir,'decode_data_dict.npy')
        self.decode_dict_path = decode_dict_path
        try:
            decode_dict = np.load(decode_dict_path,allow_pickle=True).item()
        except:
            decode_dict = dict()
            data_names = list(self.multiday_data_dict.keys())
            for nc_i, dn in enumerate(data_names):
                decode_dict[dn] = dict()
                data_dir = self.multiday_data_dict[dn]['data_dir']
                seg_names = self.multiday_data_dict[dn]['segment_names']
                seg_inds_to_use = []
                for sn in self.unique_segment_names:
                    s_ind = [s_i for s_i in range(len(seg_names)) if seg_names[s_i] == sn][0]
                    seg_inds_to_use.append(s_ind)
                seg_inds_to_use = np.sort(seg_inds_to_use)
                decode_dir = os.path.join(data_dir,'Deviation_Dependent_Decoding')
                group_dict = np.load(os.path.join(decode_dir,'group_dict.npy'),allow_pickle=True).item()
                decode_dict[dn]['group_dict'] = group_dict
                decode_types = os.listdir(decode_dir) #All_Neurons_Z_Scored
                for dt in decode_types:
                    if not len(dt.split('.')) > 1:
                        decode_dict[dn][dt] = dict()
                        decode_type_files = os.listdir(os.path.join(decode_dir,dt))
                        for f in decode_type_files:
                            #Accuracy data
                            if f == 'Decoder_Accuracy':
                                decode_dict[dn][dt][f] = dict()
                                try:
                                    nb_decode_predictions = np.load(os.path.join(decode_dir,dt,\
                                                                                 f,'nb_decode_predictions.npy'),\
                                                                                 allow_pickle=True).item()
                                    decode_dict[dn][dt][f]['nb_decode_predictions'] = nb_decode_predictions
                                    nb_decoder_accuracy_dict = np.load(os.path.join(decode_dir,dt,\
                                                                                 f,'nb_decoder_accuracy_dict.npy'),\
                                                                                 allow_pickle=True).item()
                                    decode_dict[dn][dt][f]['nb_decoder_accuracy_dict'] = nb_decoder_accuracy_dict
                                except:
                                    if self.verbose == True:
                                        print("Missing decoder accuracy data.")
                            #Deviation decoding data
                            elif f == 'NB_Decoding':
                                decode_dict[dn][dt][f] = dict()
                                try:
                                    for s_ind in seg_inds_to_use:
                                        seg_decodes = np.load(os.path.join(decode_dir,dt,\
                                                                           f,'segment_' + str(s_ind),\
                                                                            'segment_' + str(s_ind) + '_deviation_decodes.npy'),\
                                                                                     allow_pickle=True)
                                        decode_dict[dn][dt][f]['segment_' + str(s_ind)] = seg_decodes
                                except:
                                    if self.verbose == True:
                                        print("Missing deviation decode data.")
                            #Sliding decoding data
                            elif f == 'Sliding_Decoding':
                                decode_dict[dn][dt][f] = dict()
                                try:
                                    seg_group_frac = np.load(os.path.join(decode_dir,dt,\
                                                                       f,'seg_group_frac.npy'),\
                                                                                 allow_pickle=True)
                                    decode_dict[dn][dt][f]['seg_group_frac'] = seg_group_frac
                                    seg_group_rate_corr = np.load(os.path.join(decode_dir,dt,\
                                                                       f,'seg_group_rate_corr.npy'),\
                                                                                 allow_pickle=True)
                                    decode_dict[dn][dt][f]['seg_group_rate_corr'] = seg_group_rate_corr
                                except:
                                    if self.verbose == True:
                                        print("Missing deviation decode data.")
                    #Add storage of taste order info
                    corr_type_keys = self.corr_dict[dn].keys()
                    for ctk in corr_type_keys:
                        try:
                            decode_dict[dn][dt]['tastes'] = self.corr_dict[dn][ctk]['tastes']
                        except:
                            if self.verbose == True:
                                error = 'No taste info for animal ' + dn + ' corr key ' + str(ctk)
            np.save(decode_dict_path,decode_dict,allow_pickle=True)
        self.decode_dict = decode_dict   
        
    def find_decode_groupings(self,):
        print("Finding decoding groupings")
        num_datasets = len(self.decode_dict)
        unique_given_names = list(self.decode_dict.keys())
        #Pull unique decode analysis names
        unique_decode_names = []
        for name in unique_given_names:
            unique_decode_names.extend(list(self.decode_dict[name].keys()))
        unique_decode_indices = np.sort(
            np.unique(unique_decode_names, return_index=True)[1])
        unique_decode_names = [unique_decode_names[i] for i in unique_decode_indices]
        #Select which decode type to use in the analysis
        unique_decode_names = select_analysis_groups(unique_decode_names)
        
        self.unique_given_names = unique_given_names
        self.unique_decode_names = unique_decode_names
        
    def run_decode_analysis(self,):
        print("Running decoding analysis")
        cmf.compare_decode_data(self.decode_dict, self.multiday_data_dict, self.unique_given_names,
                              self.unique_decode_names, self.unique_segment_names, 
                              self.unique_taste_names, self.max_cp, self.save_dir,
                              self.verbose)
        
    def gather_lstm_data(self,):
        print("Collecting LSTM decoding data")
        lstm_dict_path = os.path.join(self.save_dir,'lstm_data_dict.npy')
        self.lstm_dict_path = lstm_dict_path
        #Start with real corr data
        try:
            self.lstm_dict = np.load(lstm_dict_path,allow_pickle=True).item()
        except:
            lstm_dict = dict()
            data_names = list(self.multiday_data_dict.keys())
            for nc_i, dn in enumerate(data_names):
                lstm_dict[dn] = dict()
                data_dir = self.multiday_data_dict[dn]['data_dir']
                lstm_dir = os.path.join(data_dir,'LSTM_Decoding')
                bin_counts = os.listdir(lstm_dir)
                for bc in bin_counts:
                    bc_dir = os.path.join(lstm_dir,bc)
                    try: #Regular data
                        bc_i = int(bc.split('_')[0])
                        lstm_dict[dn][bc] = dict()
                        lstm_dict[dn][bc]['num_bins'] = bc_i
                        #Get the training data
                        lstm_dict[dn][bc]['training_data'] = dict()
                        lstm_dict[dn][bc]['training_data']['training_matrices'] = np.load(os.path.join(bc_dir,'training_matrices.npy'))
                        lstm_dict[dn][bc]['training_data']['training_labels'] = np.load(os.path.join(bc_dir,'training_labels.npy'))
                        #Get the deviation data
                        lstm_dict[dn][bc]['vectors'] = dict()
                        lstm_dict[dn][bc]['vectors']['dev_fr_vecs'] = np.load(os.path.join(bc_dir,'dev_fr_vecs.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['vectors']['null_dev_fr_vecs'] = np.load(os.path.join(bc_dir,'null_dev_fr_vecs.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['vectors']['scaled_dev_fr_vecs'] = np.load(os.path.join(bc_dir,'scaled_dev_fr_vecs.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['vectors']['scaled_null_dev_fr_vecs'] = np.load(os.path.join(bc_dir,'scaled_null_dev_matrices.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['vectors']['shuffled_dev_fr_vecs'] = np.load(os.path.join(bc_dir,'shuffled_dev_fr_vecs.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['vectors']['shuffled_scaled_dev_fr_vecs'] = np.load(os.path.join(bc_dir,'shuffled_scaled_dev_fr_vecs.npy'),allow_pickle=True).item()
                        #Get the unscaled predictions
                        lstm_dict[dn][bc]['predictions'] = dict()
                        lstm_dict[dn][bc]['predictions']['unscaled'] = dict()
                        lstm_dict[dn][bc]['predictions']['unscaled']['true'] = np.load(os.path.join(bc_dir,'predictions.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['predictions']['unscaled']['null'] = np.load(os.path.join(bc_dir,'null_predictions.npy'),allow_pickle=True).item()
                        #Get the unscaled thresholded predictions
                        lstm_dict[dn][bc]['predictions']['unscaled_thresholded'] = dict()
                        lstm_dict[dn][bc]['predictions']['unscaled_thresholded']['true'] = np.load(os.path.join(bc_dir,'thresholded_predictions.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['predictions']['unscaled_thresholded']['null'] = np.load(os.path.join(bc_dir,'null_thresholded_predictions.npy'),allow_pickle=True).item()
                        #Get the scaled predictions
                        lstm_dict[dn][bc]['predictions']['scaled'] = dict()
                        lstm_dict[dn][bc]['predictions']['scaled']['true'] = np.load(os.path.join(bc_dir,'scaled_predictions.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['predictions']['scaled']['null'] = np.load(os.path.join(bc_dir,'scaled_null_predictions.npy'),allow_pickle=True).item()
                        #Get the scaled thresholded predictions
                        lstm_dict[dn][bc]['predictions']['scaled_thresholded'] = dict()
                        lstm_dict[dn][bc]['predictions']['scaled_thresholded']['true'] = np.load(os.path.join(bc_dir,'scaled_thresholded_predictions.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['predictions']['scaled_thresholded']['null'] = np.load(os.path.join(bc_dir,'scaled_null_thresholded_predictions.npy'),allow_pickle=True).item()
                        #Get unscaled time shuffled predictions
                        lstm_dict[dn][bc]['predictions']['unscaled_time_controlled'] = dict()
                        lstm_dict[dn][bc]['predictions']['unscaled_time_controlled']['true'] = np.load(os.path.join(bc_dir,'predictions.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['predictions']['unscaled_time_controlled']['time_shuffled'] = np.load(os.path.join(bc_dir,'shuffled_predictions.npy'),allow_pickle=True).item()
                        #Get unscaled time shuffled thresholded predictions
                        lstm_dict[dn][bc]['predictions']['unscaled_thresholded_time_controlled'] = dict()
                        lstm_dict[dn][bc]['predictions']['unscaled_thresholded_time_controlled']['true'] = np.load(os.path.join(bc_dir,'thresholded_predictions.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['predictions']['unscaled_thresholded_time_controlled']['time_shuffled'] = np.load(os.path.join(bc_dir,'shuffled_thresholded_predictions.npy'),allow_pickle=True).item()
                        #Get scaled time shuffled predictions
                        lstm_dict[dn][bc]['predictions']['scaled_time_controlled'] = dict()
                        lstm_dict[dn][bc]['predictions']['scaled_time_controlled']['true'] = np.load(os.path.join(bc_dir,'scaled_predictions.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['predictions']['scaled_time_controlled']['time_shuffled'] = np.load(os.path.join(bc_dir,'shuffled_scaled_predictions.npy'),allow_pickle=True).item()
                        #Get scaled time shuffled thresholded predictions
                        lstm_dict[dn][bc]['predictions']['scaled_thresholded_time_controlled'] = dict()
                        lstm_dict[dn][bc]['predictions']['scaled_thresholded_time_controlled']['true'] = np.load(os.path.join(bc_dir,'scaled_thresholded_predictions.npy'),allow_pickle=True).item()
                        lstm_dict[dn][bc]['predictions']['scaled_thresholded_time_controlled']['time_shuffled'] = np.load(os.path.join(bc_dir,'shuffled_scaled_thresholded_predictions.npy'),allow_pickle=True).item()
                    except: #Some other folder
                        if self.verbose == True:
                            print("Not a binned decoding folder: " + os.path.join(lstm_dir,bc))
            self.lstm_dict = lstm_dict
            np.save(lstm_dict_path,lstm_dict,allow_pickle=True)
        
    def find_lstm_groupings(self,):
        print("Finding LSTM decoding groupings")
        num_datasets = len(self.lstm_dict)
        unique_given_names = list(self.lstm_dict.keys())
        #Pull unique decode analysis names
        unique_bin_counts = []
        unique_decode_pairs = []
        for gn in unique_given_names:
            bc_names = list(self.lstm_dict[gn].keys())
            for bc in bc_names:
                bc_i = self.lstm_dict[gn][bc]['num_bins']
                unique_bin_counts.append(bc_i)
                unique_decode_pairs.extend(list(self.lstm_dict[gn][bc]['predictions'].keys()))
        unique_bin_count_inds = np.sort(
            np.unique(unique_bin_counts, return_index=True)[1])
        unique_bin_counts = [unique_bin_counts[i] for i in unique_bin_count_inds]
        unique_decode_pair_inds = np.sort(
            np.unique(unique_decode_pairs, return_index=True)[1])
        unique_decode_pairs = [unique_decode_pairs[i] for i in unique_decode_pair_inds]
        
        self.unique_given_names = unique_given_names
        self.unique_bin_counts = unique_bin_counts
        self.unique_decode_pairs = unique_decode_pairs
        
    def run_lstm_analysis(self,):
        print("Running LSTM analysis")
        
        camlf.run_analysis_plots_by_decode_pair(self.lstm_dict, self.unique_given_names,
                                self.unique_training_categories, self.unique_bin_counts, 
                                self.unique_decode_pairs, self.lstm_save_dir)
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
                corr_types = os.listdir(corr_dir)
                for ct in corr_types:
                    if ct == 'Null':
                        skip = 1
                    else: #Regular data
                        corr_dict[dn][ct] = dict()
                        corr_type_files = os.listdir(os.path.join(corr_dir,ct))
                        for f in corr_type_files:
                            if f.split('.')[-1] == 'npy':
                                f_name = f.split('.')[0]
                                if f_name == 'all_taste_names':
                                    taste_names = np.load(os.path.join(corr_dir,ct,f),allow_pickle=True)
                                    taste_name_list = []
                                    for tn in taste_names:
                                        if tn == 'NaCl_1':
                                            taste_name_list.append('salt_1')
                                        else:
                                            taste_name_list.append(tn)
                                    
                                    corr_dict[dn][ct]['tastes'] = taste_name_list
                                else:
                                    seg_name = f_name.split('_')[0]
                                    corr_dict_keys = list(corr_dict[dn][ct].keys())
                                    if len(np.where(np.array(corr_dict_keys) == seg_name)[0]) == 0: #Segment not stored yet
                                        corr_dict[dn][ct][seg_name] = dict()
                                        
                                    if f_name.split('_')[-1] == 'dict': #Dictionary of all correlation values
                                        f_data = np.load(os.path.join(corr_dir,ct,f),allow_pickle=True).item()
                                        corr_dict[dn][ct][seg_name]['all'] = dict()
                                        num_tastes = len(f_data)
                                        for nt_i in range(num_tastes):
                                            taste_name = f_data[nt_i]['name']
                                            if taste_name == 'NaCl_1':
                                                taste_name = 'salt_1'
                                            num_cp = len(f_data[nt_i]['data'])
                                            num_points = len(f_data[nt_i]['data'][0])
                                            data_concat = np.zeros((num_cp,num_points))
                                            for cp_i in range(num_cp):
                                                data_concat[cp_i,:] = np.array(f_data[nt_i]['data'][cp_i])
                                            corr_dict[dn][ct][seg_name]['all'][taste_name] = dict()
                                            corr_dict[dn][ct][seg_name]['all'][taste_name]['data'] = data_concat
                                            try:
                                                corr_dict[dn][ct][seg_name]['all'][taste_name]['num_dev'] = f_data[nt_i]['num_dev']
                                                corr_dict[dn][ct][seg_name]['all'][taste_name]['taste_num_deliv'] = f_data[nt_i]['taste_num_deliv']
                                            except:
                                                skip_val = 1 #Place holder skip
                                    else: #best correlations file
                                        f_data = np.load(os.path.join(corr_dir,ct,f),allow_pickle=True)
                                        corr_dict[dn][ct][seg_name]['best'] = f_data
            np.save(corr_dict_path,corr_dict,allow_pickle=True)     
        self.corr_dict = corr_dict
        
    def find_corr_groupings(self,):
        print("Finding unique correlation groups")
        num_datasets = len(self.corr_dict)
        unique_given_names = list(self.corr_dict.keys())
        #Pull unique correlation analysis names
        unique_corr_names = []
        for name in unique_given_names:
            unique_corr_names.extend(list(self.corr_dict[name].keys()))
        unique_corr_indices = np.sort(
            np.unique(unique_corr_names, return_index=True)[1])
        unique_corr_names = [unique_corr_names[i] for i in unique_corr_indices]
        #Select which corr types to use in the analysis
        print("\nSelect which corr types to use in this analysis.")
        unique_corr_names = select_analysis_groups(unique_corr_names)
        
        #Pull unique segment and taste names and max cp
        unique_segment_names = []
        unique_taste_names = []
        max_cp = 0
        for name in unique_given_names:
            for corr_name in unique_corr_names:
                seg_names = list(self.corr_dict[name][corr_name].keys())
                taste_names = self.corr_dict[name][corr_name]['tastes']
                unique_taste_names.extend(list(taste_names))
                for s_n in seg_names:
                    if type(self.corr_dict[name][corr_name][s_n]) == dict:
                        unique_segment_names.extend([s_n])
                        try:
                            num_cp, _ = np.shape(self.corr_dict[name][corr_name][s_n]['all'][taste_names[0]]['data'])
                            if num_cp > max_cp:
                                max_cp = num_cp
                        except:
                            print("Unable to grab changepoint count.")
        unique_seg_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i] for i in unique_seg_indices]
        #Select which segments to use in the analysis
        print("\nSelect which segments to use in this analysis.")
        unique_segment_names = select_analysis_groups(unique_segment_names)
        
        unique_taste_indices = np.sort(
            np.unique(unique_taste_names, return_index=True)[1])
        unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
        #Select which tastes to use in the analysis
        print("\nSelect which tastes to use in this analysis.")
        unique_taste_names = select_analysis_groups(unique_taste_names)
            
        self.unique_given_names = unique_given_names
        self.unique_corr_names = unique_corr_names
        self.unique_segment_names = unique_segment_names
        self.unique_taste_names = unique_taste_names
        self.max_cp = max_cp        
     
    def gather_null_corr_data(self,):
        print("Collecting null correlation data")
        null_corr_dict_path = os.path.join(self.save_dir,'null_corr_data_dict.npy')
        try:
            self.null_corr_dict = np.load(null_corr_dict_path,allow_pickle=True).item()
        except:
            null_corr_dict = dict()
            data_names = list(self.multiday_data_dict.keys())
            for dn in data_names:
                print("\tImporting null data for " + dn)
                null_corr_dict[dn] = dict()
                data_dir = self.multiday_data_dict[dn]['data_dir']
                null_corr_dir = os.path.join(data_dir,'Correlations','Null')
                null_folders = os.listdir(null_corr_dir)
                num_null = len(null_folders)
                null_corr_dict[dn]['num_null'] = num_null
                #First set up folder structure
                for sn in self.unique_segment_names:
                    null_corr_dict[dn][sn] = dict()
                    for tn in self.unique_taste_names:
                        null_corr_dict[dn][sn][tn] = dict()
                        for cn in self.unique_corr_names:
                            null_corr_dict[dn][sn][tn][cn] = dict() 
                            for cp_i in range(self.max_cp):
                                null_corr_dict[dn][sn][tn][cn][cp_i] = [] #Compiled from samples from all null datasets
                #Now collect correlations across null datasets into this folder structure
                for cn in self.unique_corr_names:
                    print('\t\tNow collecting null data for ' + cn)
                    for nf_i, nf in tqdm.tqdm(enumerate(null_folders)):
                        null_data_folder = os.path.join(null_corr_dir,'null_' + str(nf_i),cn)
                        if os.path.isdir(null_data_folder):
                            null_datasets = os.listdir(null_data_folder)
                            for sn in self.unique_segment_names:
                                for nd_name in null_datasets:
                                    if nd_name.split('_')[0] == sn:
                                        if nd_name.split('_')[-1] == 'dict.npy': #Only save complete correlation datasets
                                            null_dict = np.load(os.path.join(null_data_folder,\
                                                                     nd_name),allow_pickle=True).item()
                                            for ndk_i in null_dict.keys():
                                                tn = null_dict[ndk_i]['name']
                                                if tn == 'NaCl_1':
                                                    tn_true = 'salt_1'
                                                else:
                                                    tn_true = tn
                                                all_cp_data = null_dict[ndk_i]['data']
                                                num_null_dev = null_dict[ndk_i]['num_dev']
                                                num_taste_deliv, _ = np.shape(null_dict[ndk_i]['taste_num'])
                                                for cp_i in range(self.max_cp):
                                                    try:
                                                        all_cp_data_reshaped = np.reshape(all_cp_data[cp_i],(num_taste_deliv,num_null_dev))
                                                        avg_null_corr = np.nanmean(all_cp_data_reshaped,0) #Average correlation across deliveries
                                                        null_corr_dict[dn][sn][tn_true][cn][cp_i].extend(avg_null_corr)
                                                    except:
                                                        skip_taste = 1
            self.null_corr_dict = null_corr_dict
            np.save(null_corr_dict_path,null_corr_dict,allow_pickle=True)   
    
    def run_corr_analysis(self,):
        print("Running correlation analysis")
        cmf.compare_corr_data(self.corr_dict, self.null_corr_dict, self.multiday_data_dict, 
                              self.unique_given_names, self.unique_corr_names, 
                              self.unique_segment_names, self.unique_taste_names, 
                              self.max_cp, self.save_dir)
        
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
        #ADD CODE HERE
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
                    corr_dict[dn][ct] = dict()
                    corr_type_files = os.listdir(os.path.join(corr_dir,ct))
                    for f in corr_type_files:
                        if f.split('.')[-1] == 'npy':
                            f_name = f.split('.')[0]
                            if f_name == 'all_taste_names':
                                taste_names = np.load(os.path.join(corr_dir,ct,f),allow_pickle=True)
                                taste_name_list = [str(tn) for tn in taste_names]
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
        num_datasets = len(self.corr_dict)
        unique_given_names = list(self.corr_dict.keys())
        #Pull unique correlation analysis names
        unique_corr_names = []
        for name in unique_given_names:
            unique_corr_names.extend(list(self.corr_dict[name].keys()))
        unique_corr_indices = np.sort(
            np.unique(unique_corr_names, return_index=True)[1])
        unique_corr_names = [unique_corr_names[i] for i in unique_corr_indices]
        #Pull unique segment and taste names and max cp
        unique_segment_names = []
        unique_taste_names = []
        max_cp = 0
        for name in unique_given_names:
            for corr_name in unique_corr_names:
                seg_names = list(self.corr_dict[name][corr_name].keys())
                taste_names = self.corr_dict[name][corr_name]['tastes']
                nacl_ind = [i for i in range(len(taste_names)) if taste_names[i] == 'NaCl_1']
                if len(nacl_ind) > 0: #Stupid on my end - rename so they're all salt_1
                    taste_names[nacl_ind[0]] = 'salt_1'
                    self.corr_dict[name][corr_name]['tastes'] = list(taste_names)
                    np.save(self.corr_dict_path,self.corr_dict,allow_pickle=True)
                unique_taste_names.extend(list(taste_names))
                for s_n in seg_names:
                    if type(self.corr_dict[name][corr_name][s_n]) == dict:
                        unique_segment_names.extend([s_n])
                    try:
                        num_cp, _ = np.shape(self.corr_dict[name][corr_name][s_n]['all'][taste_names[0]])
                        if num_cp > max_cp:
                            max_cp = num_cp
                    except:
                        print("Unable to grab changepoint count.")
        unique_seg_indices = np.sort(
            np.unique(unique_segment_names, return_index=True)[1])
        unique_segment_names = [unique_segment_names[i] for i in unique_seg_indices]
        unique_taste_indices = np.sort(
            np.unique(unique_taste_names, return_index=True)[1])
        unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
            
        self.unique_given_names = unique_given_names
        self.unique_corr_names = unique_corr_names
        self.unique_segment_names = unique_segment_names
        self.unique_taste_names = unique_taste_names
        self.max_cp = max_cp        
     
    def gather_null_corr_data(self,):
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
        cmf.compare_corr_data(self.corr_dict, self.multiday_data_dict, self.unique_given_names,
                              self.unique_corr_names, self.unique_segment_names, 
                              self.unique_taste_names, self.max_cp, self.save_dir)
        
    def gather_decode_data(self,):
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
                decode_dir = os.path.join(data_dir,'Decodes')
                decode_types = os.listdir(decode_dir)
                for dt in decode_types:
                    decode_dict[dn][dt] = dict()
                    decode_type_files = os.listdir(os.path.join(decode_dir,dt))
                    for f in decode_type_files:
                        if f.split('.')[-1] == 'npy':
                            f_name = f.split('.')[0]
                            name_components = f_name.split('_')
                            #Check if dict has a segment storage started yet and make if not
                            seg_name = name_components[0]
                            decode_dict_keys = list(decode_dict[dn][dt].keys())
                            if len(np.where(np.array(decode_dict_keys) == seg_name)[0]) == 0: #Segment not stored yet
                                decode_dict[dn][dt][seg_name] = dict()
                            #Create storage for the type of decode
                            decode_type = ('_').join(name_components[-2:])
                            decode_dict[dn][dt][seg_name][decode_type] = \
                                np.load(os.path.join(decode_dir,dt,f),allow_pickle=True)
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
        num_datasets = len(self.decode_dict)
        unique_given_names = list(self.decode_dict.keys())
        #Pull unique decode analysis names
        unique_decode_names = []
        for name in unique_given_names:
            unique_decode_names.extend(list(self.decode_dict[name].keys()))
        unique_decode_indices = np.sort(
            np.unique(unique_decode_names, return_index=True)[1])
        unique_decode_names = [unique_decode_names[i] for i in unique_decode_indices]
        
        self.unique_given_names = unique_given_names
        self.unique_decode_names = unique_decode_names
        
    def run_decode_analysis(self,):
        cmf.compare_decode_data(self.decode_dict, self.multiday_data_dict, self.unique_given_names,
                              self.unique_decode_names, self.unique_segment_names, 
                              self.unique_taste_names, self.max_cp, self.save_dir,
                              self.verbose)
        

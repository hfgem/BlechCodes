#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:03:51 2025

@author: Hannah Germaine

Functions to support comparing multiday analysis results for multiple animals.
"""

import os
import warnings
import pickle
import numpy as np
from tkinter.filedialog import askdirectory

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

warnings.filterwarnings("ignore")

class run_compare_multiday_analysis():
    
    def __init__(self, args):
        self.multiday_data_dict = args[0]
        self.save_dir = args[1]
        self.gather_corr_data()
        self.find_corr_groupings()
        self.run_corr_analysis()
        self.gather_decode_data()
        self.run_decode_analysis()
        
    def gather_corr_data(self,):
        corr_dict_path = os.path.join(self.save_dir,'corr_data_dict.npy')
        try:
            corr_dict = np.load(corr_dict_path,allow_pickle=True).item()
        except:
            corr_dict = dict()
            data_names = list(multiday_data_dict.keys())
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
                                        corr_dict[dn][ct][seg_name]['all'][taste_name] = data_concat
                                
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
            unique_corr_names.extend(list(corr_dict[name].keys()))
        unique_corr_indices = np.sort(
            np.unique(unique_corr_names, return_index=True)[1])
        unique_corr_names = [unique_corr_names[i] for i in unique_corr_indices]
        #Pull unique segment and taste names and max cp
        unique_segment_names = []
        unique_taste_names = []
        max_cp = 0
        for name in unique_given_names:
            for corr_name in unique_corr_names:
                seg_names = list(corr_dict[name][corr_name].keys())
                for s_n in seg_names:
                    if type(corr_dict[name][corr_name][s_n]) == dict:
                        unique_segment_names.extend([s_n])
                taste_names = corr_dict[name][corr_name]['tastes']
                unique_taste_names.extend(taste_names)
                try:
                    num_cp, _ = np.shape(corr_dict[dn][ct][seg_name]['all'][taste_names[0]])
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
        
    # def run_corr_analysis(self,):
        
        
    # def gather_decode_data(self,):
        
    
    # def run_decode_analysis(self,):
        
        

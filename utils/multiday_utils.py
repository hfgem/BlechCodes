#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:05:35 2025

@author: Hannah Germaine

This file contains utility functions to support multiday analysis
"""

import os
import csv
import json
import glob

class import_metadata():
    
    def __init__(self,args): #args assumed to include held data dict
        self.held_data_dict = args[0]
        self.num_days = len(self.held_data_dict)
        self.metadata_dict = self.handle_days()
        
    def handle_days(self,):
        #Collect metadata for all days
        metadata_dict = dict()
        for n_i in range(self.num_days):
            day_dict = self.load_day([n_i])
            metadata_dict[n_i] = day_dict
        
        return metadata_dict
        
    def load_day(self,args):
        #Work on one day at a time
        this_day_dict = self.held_data_dict[args[0]]
        day_dict = dict()
        day_dict['dir_name'] = this_day_dict['dir_name']
        day_dict['hdf5_dir'] = this_day_dict['hdf5_name']
        day_dict['params_file_path'] = os.path.join(this_day_dict['dir_name'],'analysis_params.json')
        with open(day_dict['params_file_path'], 'r') as params_file:
            day_dict['params_dict'] = json.load(params_file)
        self.get_info_path([day_dict['dir_name']])
        day_dict['info_file_path'] = self.info_file_path
        with open(day_dict['info_file_path'], 'r') as info_file:
            day_dict['info_dict'] = json.load(info_file)
        
        return day_dict
    
    def get_info_path(self,args):
        file_list = glob.glob(os.path.join(args[0],'**.info'))
        if len(file_list) > 0:
            if len(file_list) > 1:
                file_found = 0
                for f_i in range(len(file_list)):
                    is_file = self.bool_input('Is ' + file_list[f_i] + ' the correct info file? ')
                    if is_file == 'y':
                        self.info_file_path = file_list[f_i]
                        file_found = 1
            else:
                self.info_file_path = file_list[0]
                file_found = 1
            if file_found == 0:
                print('No info file found. Please ensure you ran clustering / are in the correct folder!')
                quit()    
                
    def bool_input(self,prompt):
        #This function asks a user for an integer input
        bool_loop = 1    
        while bool_loop == 1:
            print("Respond with Y/y/N/n:")
            response = input(prompt)
            if (response.lower() != 'y')*(response.lower() != 'n'):
                print("\tERROR: Incorrect data entry, only give Y/y/N/n.")
            else:
                bool_val = response.lower()
                bool_loop = 0
        
        return bool_val
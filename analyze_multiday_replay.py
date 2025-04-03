#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:18:05 2025

@author: Hannah Germaine

Pipeline master for analyzing replay across multiple days.
"""

#Import necessary packages and functions
import os
import csv
import numpy as np
from tkinter.filedialog import askdirectory
from functions.blech_held_units_funcs import int_input
from utils.multiday_utils import import_metadata
from utils.data_utils import import_multiday_data
from functions.multiday_analysis import run_multiday_analysis

if __name__ == '__main__':
    
    import multiprocess
    multiprocess.set_start_method('spawn')
    
    # Grab current directory and data directory / metadata
    script_path = os.path.realpath(__file__)
    blechcodes_dir = os.path.dirname(script_path)
    
    print('Where did you save the held units pickle file?')
    held_save_dir = askdirectory()
    held_data_dict = np.load(os.path.join(held_save_dir,'data_dict.npy'),allow_pickle=True).item()
    held_unit_csv = os.path.join(held_save_dir,'held_units.csv')
    held_units = []
    with open(held_unit_csv, 'r') as heldunitcsv:
        heldreader = csv.reader(heldunitcsv, delimiter=' ', quotechar='|')
        for row in heldreader:
            row_vals = row[0].split(',')
            try:
                is_int = int(row_vals[0])
                held_units.append([int(row_vals[i]) for i in range(len(row_vals))])
            except:
                is_header = row_vals
                
    num_days = len(held_units[0])
    
    metadata_handler = import_metadata([held_data_dict])
    
    metadata = dict()
    metadata['held_units'] = np.array(held_units)
    data_dict = dict()
    
    #Now go day by day and import data
    for n_i in range(num_days):
        print("Day " + str(n_i+1))
        day_metadata = metadata_handler.metadata_dict[n_i]
        metadata[n_i] = day_metadata
        try:
            dig_in_names = day_metadata.info_dict['taste_params']['tastes']
        except:
            dig_in_names = []
        
        data_handler = import_multiday_data([day_metadata, dig_in_names])
        day_data = dict()
        for var in vars(data_handler):
            day_data[var] = getattr(data_handler,var)
        del data_handler
        data_dict[n_i] = day_data
        
    #Run the multiday analysis
    run_multiday_analysis([metadata,data_dict])
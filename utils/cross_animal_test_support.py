#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:03:51 2024

@author: Hannah Germaine

Cross-Animal Analysis Test Support: Use to test updates to functions and debug
"""

#%% Compare Conditions Support

import os
import easygui
import numpy as np
from utils.replay_utils import import_metadata
from utils.data_utils import import_data
from functions.compare_conditions_analysis import run_compare_conditions_analysis
from functions.compare_conditions_funcs import int_input, bool_input
import functions.hdf5_handling as hf5

# Grab current directory and data directory / metadata
script_path = os.path.realpath(__file__)
blechcodes_dir = os.path.dirname(script_path)

all_data_dict = dict()
save_dir = ''

# _____Prompt user if they'd like to use previously stored correlation data_____
print("If you previously started an analysis, you may have a all_data_dict.npy file in the analysis folder.")
bool_val = bool_input(
    "Do you have a file stored you'd like to continue analyzing [y/n]? ")
if bool_val == 'y':
    save_dir = easygui.diropenbox(
        title='Please select the storage folder.')
    try:
        all_data_dict = np.load(os.path.join(save_dir,'all_data_dict.npy'),allow_pickle=True).item()
    except:
        print("All data dict not found in given save folder. Aborting.")
        quit()
else:
    # _____Prompt user for the number of datasets needed in the analysis_____
    print("Conditions include across days and across animals (the number of separate spike sorted datasets).")
    num_cond = int_input(
        "How many conditions-worth of correlation data do you wish to import for this comparative analysis (integer value)? ")
    if num_cond >= 1:
        print("Multiple file import selected.")
    else:
        print("Single file import selected.")

    # _____Pull all data into a dictionary_____
    all_data_dict = dict()
    for nc in range(num_cond):
        # _____Get the directory of the hdf5 file_____
        print("Please select the folder where the data # " +
              str(nc+1) + " is stored.")

        # _____Import relevant data_____
        metadata_handler = import_metadata([blechcodes_dir])
        try:
            dig_in_names = metadata_handler.info_dict['taste_params']['tastes']
        except:
            dig_in_names = []

        # import data from hdf5
        data_handler = import_data(
            [metadata_handler.dir_name, metadata_handler.hdf5_dir, dig_in_names])

        # repackage data from all handlers
        metadata = dict()
        for var in vars(metadata_handler):
            metadata[var] = getattr(metadata_handler, var)
        del metadata_handler

        data_dict = dict()
        for var in vars(data_handler):
            data_dict[var] = getattr(data_handler, var)
        del data_handler

        # Grab colloquial name
        print("Give a more colloquial name to the dataset.")
        data_name = data_dict['data_path'].split('/')[-2]
        given_name = input("How would you rename " + data_name + "? ")

        all_data_dict[given_name] = dict()
        all_data_dict[given_name]['data'] = data_dict
        all_data_dict[given_name]['metadata'] = metadata

        del data_dict, data_name, given_name, metadata, dig_in_names
    del nc
    
min_best_cutoff = 0.5


import os
import warnings
import easygui
import pickle
import numpy as np
import functions.hdf5_handling as hf5
from tkinter.filedialog import askdirectory

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import functions.compare_datasets_funcs as cdf
import functions.compare_conditions_funcs as ccf

if len(save_dir) == 0:
    print("Please select a storage folder for results.")
    print('Please select the storage folder.')
    save_dir = askdirectory()
    np.save(os.path.join(save_dir,'all_data_dict.npy'),\
            all_data_dict,allow_pickle=True)

#%% import data

import os

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

import warnings
import numpy as np
import functions.hdf5_handling as hf5
from tkinter.filedialog import askdirectory
from itertools import combinations
from functions.data_description_analysis import run_data_description_analysis
import functions.compare_datasets_funcs as cdf
import functions.compare_conditions_funcs as ccf
import functions.cross_animal_seg_stats as cass
import functions.cross_animal_taste_stats as cats
import functions.cross_animal_dev_stats as cads
import functions.cross_animal_dev_null_plots as cadnp
import functions.dependent_decoding_funcs as ddf

verbose = False

segments_to_analyze = [0,2,4]

seg_stat_save_dir = os.path.join(save_dir,'seg_stats')
if not os.path.isdir(seg_stat_save_dir):
    os.mkdir(seg_stat_save_dir)

try:
    neur_rates = np.load(os.path.join(seg_stat_save_dir,'neur_rates.npy'),allow_pickle=True).item()
    pop_rates = np.load(os.path.join(seg_stat_save_dir,'pop_rates.npy'),allow_pickle=True).item()
    isis = np.load(os.path.join(seg_stat_save_dir,'isis.npy'),allow_pickle=True).item()
except:
    unique_given_names, unique_segment_names, neur_rates, pop_rates, isis, \
        cvs = cass.seg_stat_collection(all_data_dict)

    np.save(os.path.join(seg_stat_save_dir,'neur_rates.npy'),neur_rates,allow_pickle=True)
    np.save(os.path.join(seg_stat_save_dir,'pop_rates.npy'),pop_rates,allow_pickle=True)
    np.save(os.path.join(seg_stat_save_dir,'isis.npy'),isis,allow_pickle=True)

cass.seg_stat_analysis(unique_given_names, unique_segment_names, neur_rates, \
                      pop_rates, isis, cvs, segments_to_analyze, seg_stat_save_dir)

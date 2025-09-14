#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:15:54 2025

@author: Hannah Germaine

Cross-Animal Analysis Test Support: Use to test updates to functions and debug
"""

#%% Compare Multiday Support

import os
import tables
import numpy as np
from tkinter.filedialog import askdirectory
from functions.compare_multiday_analysis import run_compare_multiday_analysis
from functions.compare_conditions_funcs import int_input, bool_input

# Grab current directory and data directory / metadata
script_path = os.path.realpath(__file__)
blechcodes_dir = os.path.dirname(script_path)

multiday_data_dict = dict()
save_dir = ''

# _____Prompt user if they'd like to use previously stored correlation data_____
print("If you previously started an analysis, you may have a multiday_data_dict.npy file in the analysis folder.")
bool_val = bool_input(
    "Do you have a file stored you'd like to continue analyzing [y/n]? ")
if bool_val == 'y':
    print('Please select the storage folder.')
    save_dir = askdirectory()
    try:
        multiday_data_dict = np.load(os.path.join(save_dir,'multiday_data_dict.npy'),allow_pickle=True).item()
    except:
        print("Multiday data dict not found in given save folder. Aborting.")
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
    multiday_data_dict = dict()
    for nc in range(num_cond):
        # _____Get the directory of the hdf5 file_____
        print("Please select the folder where the data # " +
              str(nc+1) + " is stored.")
        data_dir = askdirectory()
        subfolders = os.listdir(data_dir)
        corr_exists = 0
        for sf in subfolders:
            if sf == 'Correlations':
                corr_exists = 1
             
        for corr_attempts in range(3):
            while corr_exists == 0:
                print("Try again - directory selected does not contain expected subfolders.")
                data_dir = askdirectory()
                subfolders = os.listdir(data_dir)
                corr_exists = 0
                for sf in subfolders:
                    if sf == 'Correlations':
                        corr_exists = 1
        
        if corr_exists == 0:
            print("Number of folder selection attempts exceeded. Program quitting.")
            quit()
            
        # Grab colloquial name
        print("Give a more colloquial name to the dataset.")
        folder_name = os.path.split(data_dir)[-1]
        data_name = input("Give a name to the dataset in folder /" + folder_name + "/: ")
        multiday_data_dict[data_name] = dict()
        multiday_data_dict[data_name]['data_dir'] = data_dir
        
        # Get day 1 folder
        folder_containing_data_dir = os.path.split(data_dir)[0]
        possible_day_1_folders = list(np.setdiff1d(os.listdir(folder_containing_data_dir),[folder_name, '.DS_Store']))
        folder_prompt = ''
        for fn_i, fn in enumerate(possible_day_1_folders):
            folder_prompt += str(fn_i) + ': ' + fn + '\n'
        folder_prompt += 'Please provide the above index of the day 1 data folder: '
        day_1_folder_ind = int_input(folder_prompt)
        day_1_folder = possible_day_1_folders[day_1_folder_ind]
        
        # Grab segment lengths from day 1 data
        day_1_folder_contents = os.listdir(os.path.join(folder_containing_data_dir,day_1_folder))
        for d1fc in day_1_folder_contents:
            if d1fc.split('.')[-1] == 'h5':
                hdf5_path = os.path.join(folder_containing_data_dir,day_1_folder,d1fc)
        blech_clust_h5 = tables.open_file(hdf5_path, 'r+', title = 'hdf5_file')
        segment_times = blech_clust_h5.root.experiment_components.segment_times[:]
        segment_names = [blech_clust_h5.root.experiment_components.segment_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.experiment_components.segment_names))]
        blech_clust_h5.close()
        multiday_data_dict[data_name]['segment_times'] = segment_times
        multiday_data_dict[data_name]['segment_names'] = segment_names
        
        del data_dir, data_name
    del nc
    
    print('Please select the storage folder for comparative multiday analyses.')
    save_dir = askdirectory()
    np.save(os.path.join(save_dir,'multiday_data_dict.npy'),multiday_data_dict,allow_pickle=True)

import os
import warnings
import tqdm
import random
import numpy as np
from tkinter.filedialog import askdirectory
from functions.compare_multiday_funcs import select_analysis_groups

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

warnings.filterwarnings("ignore")

#%% gather_lstm_data()
verbose = False

lstm_dict_path = os.path.join(save_dir,'lstm_data_dict.npy')
#Start with real corr data
try:
    lstm_dict = np.load(lstm_dict_path,allow_pickle=True).item()
except:
    lstm_dict = dict()
    data_names = list(multiday_data_dict.keys())
    for nc_i, dn in enumerate(data_names):
        lstm_dict[dn] = dict()
        data_dir = multiday_data_dict[dn]['data_dir']
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
                lstm_dict[dn][bc]['predictions'] = dict()
                #Get the unscaled predictions
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
                if verbose == True:
                    print("Not a binned decoding folder: " + os.path.join(lstm_dir,bc))
                
    np.save(lstm_dict_path,lstm_dict,allow_pickle=True)   
    
#%% find_corr_groupings()
num_datasets = len(corr_dict)
unique_given_names = list(corr_dict.keys())
#Pull unique correlation analysis names
unique_corr_names = []
for name in unique_given_names:
    unique_corr_names.extend(list(corr_dict[name].keys()))
unique_corr_indices = np.sort(
    np.unique(unique_corr_names, return_index=True)[1])
unique_corr_names = [unique_corr_names[i] for i in unique_corr_indices]
#Select which corr types to use in the analysis
unique_corr_names = select_analysis_groups(unique_corr_names)

#Pull unique segment and taste names and max cp
unique_segment_names = []
unique_taste_names = []
max_cp = 0
for name in unique_given_names:
    for corr_name in unique_corr_names:
        seg_names = list(corr_dict[name][corr_name].keys())
        taste_names = corr_dict[name][corr_name]['tastes']
        unique_taste_names.extend(list(taste_names))
        for s_n in seg_names:
            if type(corr_dict[name][corr_name][s_n]) == dict:
                unique_segment_names.extend([s_n])
                try:
                    num_cp, _ = np.shape(corr_dict[name][corr_name][s_n]['all'][taste_names[0]]['data'])
                    if num_cp > max_cp:
                        max_cp = num_cp
                except:
                    error = "Unable to grab changepoint count."
unique_seg_indices = np.sort(
    np.unique(unique_segment_names, return_index=True)[1])
unique_segment_names = [unique_segment_names[i] for i in unique_seg_indices]
#Select which segments to use in the analysis
unique_segment_names = select_analysis_groups(unique_segment_names)

unique_taste_indices = np.sort(
    np.unique(unique_taste_names, return_index=True)[1])
unique_taste_names = [unique_taste_names[i] for i in unique_taste_indices]
#Select which tastes to use in the analysis
unique_taste_names = select_analysis_groups(unique_taste_names)

#%% gather_null_corr_data()
print("Collecting null correlation data")
null_corr_dict_path = os.path.join(save_dir,'null_corr_data_dict.npy')
try:
    null_corr_dict = np.load(null_corr_dict_path,allow_pickle=True).item()
except:
    null_corr_dict = dict()
    data_names = list(multiday_data_dict.keys())
    for dn in data_names:
        print("\tImporting null data for " + dn)
        null_corr_dict[dn] = dict()
        data_dir = multiday_data_dict[dn]['data_dir']
        null_corr_dir = os.path.join(data_dir,'Correlations','Null')
        null_folders = os.listdir(null_corr_dir)
        num_null = len(null_folders)
        null_corr_dict[dn]['num_null'] = num_null
        #First set up folder structure
        for sn in unique_segment_names:
            null_corr_dict[dn][sn] = dict()
            for tn in unique_taste_names:
                null_corr_dict[dn][sn][tn] = dict()
                for cn in unique_corr_names:
                    null_corr_dict[dn][sn][tn][cn] = dict() 
                    for cp_i in range(max_cp):
                        null_corr_dict[dn][sn][tn][cn][cp_i] = [] #Compiled from samples from all null datasets
        #Now collect correlations across null datasets into this folder structure
        for cn in unique_corr_names:
            print('\t\tNow collecting null data for ' + cn)
            for nf_i, nf in tqdm.tqdm(enumerate(null_folders)):
                null_data_folder = os.path.join(null_corr_dir,'null_' + str(nf_i),cn)
                if os.path.isdir(null_data_folder):
                    null_datasets = os.listdir(null_data_folder)
                    for sn in unique_segment_names:
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
                                        for cp_i in range(max_cp):
                                            try:
                                                all_cp_data_reshaped = np.reshape(all_cp_data[cp_i],(num_taste_deliv,num_null_dev))
                                                avg_null_corr = np.nanmean(all_cp_data_reshaped,0) #Average correlation across deliveries
                                                null_corr_dict[dn][sn][tn_true][cn][cp_i].append(avg_null_corr)
                                            except:
                                                skip_taste = 1
    np.save(null_corr_dict_path,null_corr_dict,allow_pickle=True) 

#%% run_corr_analysis


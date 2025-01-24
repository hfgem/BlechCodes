#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:05:21 2025

@author: Hannah Germaine
Compare multiday results across animals in plots with stats
"""

if __name__ == '__main__':
    
    import os
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
            data_name = data_dir.split('/')[-2]

            multiday_data_dict[data_name] = dict()
            multiday_data_dict[data_name]['data_dir'] = data_dir
            
            del data_dir, data_name
        del nc
        
        print('Please select the storage folder for comparative multiday analyses.')
        save_dir = askdirectory()
        np.save(os.path.join(save_dir,'multiday_data_dict.npy'),multiday_data_dict,allow_pickle=True)

    # _____Pass Data to Analysis_____
    run_compare_multiday_analysis([multiday_data_dict, save_dir])


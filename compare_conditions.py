#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 10:48:05 2023

@author: Hannah Germaine
Compare results across conditions in plots and with stats
"""

if __name__ == '__main__':

    import os
    import easygui
    from utils.replay_utils import import_metadata
    from utils.data_utils import import_data
    from functions.compare_conditions_analysis import run_compare_conditions_analysis
    from functions.compare_conditions_funcs import int_input, bool_input

    # Grab current directory and data directory / metadata
    script_path = os.path.realpath(__file__)
    blechcodes_dir = os.path.dirname(script_path)

    all_data_dict = dict()
    save_dir = ''

    # _____Prompt user if they'd like to use previously stored correlation data_____
    print("If you previously started an analysis, you may have a corr_data.pkl file in the analysis folder.")
    bool_val = bool_input(
        "Do you have a correlation pickle file stored you'd like to continue analyzing [y/n]? ")
    if bool_val == 'y':
        save_dir = easygui.diropenbox(
            title='Please select the storage folder.')
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

    # _____Pass Data to Analysis_____
    run_compare_conditions_analysis([all_data_dict, save_dir])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 19:12:11 2022

@author: hannahgermaine
A collection of functions to handle HDF5 data storage and imports
"""
import os
import tqdm
import tables
import easygui
# If this file is not being run from the directory of .../BlechCodes/, uncomment the next two lines
#file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
# os.chdir(file_path)
import numpy as np


def hdf5_exists():
    """This function asks for user input on whether a .h5 file exists"""
    h_loop = 1
    h_exists = 0
    while h_loop == 1:
        h_q = input(
            "\n INPUT REQUESTED: Does an HDF5 file already exist? y / n: ")
        if h_q != 'n' and h_q != 'y':
            print('Error, incorrect response, try again')
            h_loop = 1
        else:
            h_loop = 0
    if h_q == 'y':
        h_exists = 1

    return h_exists


def file_import(datadir, dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names):
    # Based on read_file.py by abuzarmahmood on GitHub - written by Abuzar Mahmood:
    # https://github.com/abuzarmahmood/blech_clust/blob/master/read_file.py
    # Create HDF5 file with data stored

    # filedir where all data is stored
    # dat_files_list contains the names of all .dat files
    # electrodes_list contains the names of all amp- ... -.dat files
    # emg_ind contains the index of the electrode that has EMG data
    # dig_in_list contains the names of the digital input files
    # dig_in_names contains the names of the tastants associated with each digital input file

    # Grab directory name to create the hdf5 file
    hdf5_name = str(os.path.dirname(datadir + '/')).split('/')
    hf5_dir = datadir + '/' + hdf5_name[-1]+'.h5'
    hf5 = tables.open_file(hf5_dir, 'w', title=hdf5_name[-1])
    hf5.create_group('/', 'raw')
    hf5.create_group('/', 'raw_emg')
    hf5.create_group('/', 'digital_in')
    hf5.create_group('/', 'digital_out')
    hf5.close()
    print('Created nodes in HF5')

    # Read the amplifier sampling rate from info.rhd -
    # look at Intan's website for structure of header files
    info_file = np.fromfile(datadir + '/' + 'info.rhd',
                            dtype=np.dtype('float32'))
    sampling_rate = int(info_file[2])
    hf5 = tables.open_file(hf5_dir, 'r+', title=hdf5_name[-1])
    atom = tables.FloatAtom()
    hf5.create_earray('/', 'sampling_rate', atom, (0,))
    hf5.root.sampling_rate.append([sampling_rate])
    hf5.close()

    # Read the time.dat file
    num_recorded_samples = len(np.fromfile(
        datadir + '/' + 'time.dat', dtype=np.dtype('float32')))
    total_recording_time = num_recorded_samples/sampling_rate  # In seconds

    check_str = f'Amplifier files: {electrodes_list} \nSampling rate: {sampling_rate} Hz'\
        f'\nDigital input files: {dig_in_list} \n ---------- \n \n'
    print(check_str)

    # Sort all lists
    dat_files_list.sort()
    electrodes_list.sort()
    dig_in_list.sort()
    emg_ind.sort()
    # DO NOT SORT dig_in_names - they are already sorted!

    # Pull data into arrays first
    print("Separating Neuron Electrodes and EMG Electrodes")
    if len(electrodes_list) == 1:  # Single amplifier file
        print("\tSingle Amplifier File Detected")
        amplifier_data = np.fromfile(
            datadir + '/' + electrodes_list[0], dtype=np.dtype('uint16'))
        num_electrodes = int(len(amplifier_data)/num_recorded_samples)
        amp_reshape = np.reshape(amplifier_data, (int(
            len(amplifier_data)/num_electrodes), num_electrodes)).T
        all_electrodes = list()
        all_emg = list()
        for i in range(num_electrodes):
            ind_data = amp_reshape[i, :]
            try:
                e_ind = emg_ind.index(i)
                all_emg.append(ind_data)
            except:
                all_electrodes.append(ind_data)
        num_emg = len(all_emg)
        num_neur = len(all_electrodes)
    else:
        # Separate electrodes and emg into their own lists
        all_electrodes = list()
        all_emg = list()
        for i in range(len(electrodes_list)):
            ind_data = np.fromfile(
                datadir + '/' + electrodes_list[i], dtype=np.dtype('uint16'))
            try:
                e_ind = emg_ind.index(i)
                all_emg.append(ind_data)
            except:
                all_electrodes.append(ind_data)
        num_emg = len(all_emg)
        num_neur = len(all_electrodes)
    print("\tNum Neur Electrodes = " + str(num_neur))
    print("\tNum EMG Electrodes = " + str(num_emg))

    print("Grabbing Digital Inputs")
    if len(dig_in_list) == 1:  # Single dig-in file
        print("\tSingle Amplifier File Detected")
        num_dig_ins = len(dig_in_names)
        d_inputs = np.fromfile(
            datadir + '/' + dig_in_list[0], dtype=np.dtype('uint16'))
        d_inputs_str = d_inputs.astype('str')
        d_in_str_int = d_inputs_str.astype('int64')
        d_diff = np.diff(d_in_str_int)
        dig_inputs = np.zeros((num_dig_ins, len(d_inputs)))
        for n_i in range(num_dig_ins):
            start_ind = np.where(d_diff == n_i + 1)[0]
            end_ind = np.where(d_diff == -1*(n_i + 1))[0]
            for s_i in range(len(start_ind)):
                dig_inputs[n_i, start_ind[s_i]:end_ind[s_i]] = 1
    else:
        dig_inputs = np.array([np.fromfile(datadir + '/' + dig_in_list[i],
                              dtype=np.dtype('uint16')) for i in range(len(dig_in_list))])
        num_dig_ins = np.shape(dig_inputs)[0]
    print("\tNum Dig Ins = " + str(num_dig_ins))

    print("Saving data to hdf5 file.")
    hf5 = tables.open_file(hf5_dir, 'r+', title=hdf5_name[-1])
    atom = tables.FloatAtom()
    print("\tSaving Neuron Electrode Data")
    for i in tqdm.tqdm(range(num_neur)):  # add electrode arrays
        e_name = str(i)
        hf5.create_earray('/raw', f'electrode_{e_name}', atom, (0,))
        exec("hf5.root.raw.electrode_"+str(e_name) +
             ".append(all_electrodes[i][:])")
    print("\tSaving EMG Electrode Data")
    for i in tqdm.tqdm(range(num_emg)):  # add emg arrays
        e_name = str(i)
        hf5.create_earray('/raw_emg', f'emg_{e_name}', atom, (0,))
        exec("hf5.root.raw_emg.emg_"+str(e_name)+".append(all_emg[i][:])")
    print("\tSaving Tastant Delivery Data")
    for i in tqdm.tqdm(range(num_dig_ins)):  # add dig-in arrays
        d_name = dig_in_names[i]
        hf5.create_earray('/digital_in', f'digin_{d_name}', atom, (0,))
        exec("hf5.root.digital_in.digin_" +
             str(d_name)+".append(dig_inputs[i,:])")

    hf5.close()  # Close the file

    return hf5_dir


def sorted_data_import():
    """This function asks for user input to retrieve the directory of sorted data
    it also returns the directory of cleaned data where the segment times/names are
    stored (if it exists)"""

    print("\n INPUT REQUESTED: Select directory with the sorted .h5 file (name = '...._repacked.h5').")
    blech_clust_datadir = easygui.diropenbox(
        title='Please select the folder where data is stored.')
    files_in_dir = os.listdir(blech_clust_datadir)
    # Checks for repacked and downsampled .h5 in the directory
    for i in range(len(files_in_dir)):
        filename = files_in_dir[i]
        if filename.split('_')[-1] == 'repacked.h5':
            blech_clust_hdf5_name = filename
        elif filename.split('_')[-1] == 'downsampled.h5':
            downsampled_hf5_name = filename
        elif filename.split('_')[-1] == 'cleaned.h5':
            cleaned_hf5_name = filename

    try:
        blech_clust_hf5_dir = blech_clust_datadir + '/' + blech_clust_hdf5_name
    except:
        print("Old .h5 file not found. Quitting program.")
        quit()
    try:
        downsampled_hf5_dir = blech_clust_datadir + '/' + downsampled_hf5_name
    except:
        downsampled_hf5_dir = ''

    try:
        cleaned_hf5_dir = blech_clust_datadir + '/' + cleaned_hf5_name
    except:
        cleaned_hf5_dir = ''

    return blech_clust_hf5_dir, downsampled_hf5_dir, cleaned_hf5_dir

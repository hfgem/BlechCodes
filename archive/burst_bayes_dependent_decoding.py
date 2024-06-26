#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:55:18 2024

@author: Hannah Germaine

Script to run decoding of tastes during rest interval burst events while
maintaining dependencies between neurons.

Assumes analyze_states.py and find_deviations.py were run first.
"""

if __name__ == '__main__':

    import os
    import tqdm
    import json
    import gzip
    file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])
    os.chdir(file_path)
    import numpy as np
    import functions.hdf5_handling as hf5
    import functions.analysis_funcs as af
    import functions.dependent_decoding_funcs as ddf
    import functions.burst_decoding_funcs as bdf
    import functions.decoding_funcs as decode_f
    import functions.dev_funcs as dev_f

    # _____Get the directory of the hdf5 file_____
    # Program will automatically quit if file not found in given folder
    sorted_dir, segment_dir, cleaned_dir = hf5.sorted_data_import()
    fig_save_dir = ('/').join(sorted_dir.split('/')[0:-1]) + '/'
    print('\nData Directory:')
    print(fig_save_dir)

    # _____Import data_____
    # todo: update intan rhd file import code to accept directory input
    num_neur, all_waveforms, spike_times, dig_in_names, segment_times, segment_names, start_dig_in_times, end_dig_in_times, num_tastes = af.import_data(
        sorted_dir, segment_dir, fig_save_dir)

    # _____Calculate spike time datasets_____
    pre_taste = 2  # Seconds before tastant delivery to store
    post_taste = 2  # Seconds after tastant delivery to store
    # Milliseconds before taste delivery to plot
    pre_taste_dt = np.ceil(pre_taste*1000).astype('int')
    # Milliseconds after taste delivery to plot
    post_taste_dt = np.ceil(post_taste*1000).astype('int')

    # _____Add "no taste" control segments to the dataset_____
    if dig_in_names[-1] != 'none':
        dig_in_names, start_dig_in_times, end_dig_in_times, num_tastes = af.add_no_taste(
            start_dig_in_times, end_dig_in_times, post_taste, dig_in_names)

    segment_spike_times = af.calc_segment_spike_times(
        segment_times, spike_times, num_neur)
    tastant_spike_times = af.calc_tastant_spike_times(segment_times, spike_times,
                                                      start_dig_in_times, end_dig_in_times,
                                                      pre_taste, post_taste, num_tastes, num_neur)
    num_segments = len(segment_spike_times)
    segment_times_reshaped = [
        [segment_times[i], segment_times[i+1]] for i in range(num_segments)]

    # Raster Poisson Bayes Changepoint Calcs Indiv Neurons
    data_group_name = 'changepoint_data'
    #taste_cp_raster_inds = af.pull_data_from_hdf5(sorted_dir,data_group_name,'taste_cp_raster_inds')
    pop_taste_cp_raster_inds = af.pull_data_from_hdf5(
        sorted_dir, data_group_name, 'pop_taste_cp_raster_inds')
    num_cp = np.shape(pop_taste_cp_raster_inds[0])[-1] - 1

    bayes_dir = fig_save_dir + 'Burst_Dependent_Decoding/'
    if os.path.isdir(bayes_dir) == False:
        os.mkdir(bayes_dir)

    # Deviation storage directory
    dev_dir = fig_save_dir + 'Deviations/'

    print("Now importing calculated deviations")
    segment_deviations = []
    for s_i in tqdm.tqdm(range(num_segments)):
        filepath = dev_dir + segment_names[s_i] + '/deviations.json'
        with gzip.GzipFile(filepath, mode="r") as f:
            json_bytes = f.read()
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)
            segment_deviations.append(data)

    # _____Pull rasters of deviations and plot_____
    # Calculate segment deviation spikes
    print("Now pulling true deviation rasters")
    segment_dev_rasters, segment_dev_times, _ = dev_f.create_dev_rasters(num_segments, segment_spike_times,
                                                                         np.array(segment_times_reshaped), segment_deviations, pre_taste)

    # %% Set decoding variables

    print("Setting Decoding Variables")

    # Specify which parts to decode
    epochs_to_analyze = np.array([1])
    segments_to_analyze = np.array([0, 2, 4])

    # If first decoding full taste response
    use_full = 0  # Decode the full taste response first or not
    skip_time = 0.05  # Seconds to skip forward in sliding bin
    skip_dt = np.ceil(skip_time*1000).astype('int')

    # Decoding settings
    # Fraction of total population that must be active to consider a decoding event
    neuron_count_thresh = 0  # 1/3
    max_decode = 50  # number of example decodes to plot
    seg_stat_bin = 60*1000  # ms to bin segment for decoding counts in bins
    trial_start_frac = 0  # Fractional start of trials to use in decoding
    decode_prob_cutoff = 0.9  # 0.95
    # minimum bin size for a deviation (in number of ms = dt)
    min_dev_size = 20

    # Z-scoring settings
    bin_time = 0.1  # Seconds to skip forward in calculating firing rates
    bin_dt = np.ceil(bin_time*1000).astype('int')
    # Timesteps (ms) before taste delivery to bin for z-scoring FR during taste response
    bin_pre_taste = 100

    # %% Grab fr distributions

    print("Pulling FR Distributions")

    tastant_fr_dist_pop, taste_num_deliv, max_hz_pop = ddf.taste_fr_dist(num_neur,
                                                                         num_cp, tastant_spike_times,
                                                                         pop_taste_cp_raster_inds,
                                                                         start_dig_in_times, pre_taste_dt,
                                                                         post_taste_dt, trial_start_frac)

    tastant_fr_dist_z_pop, taste_num_deliv, max_hz_z_pop, min_hz_z_pop = ddf.taste_fr_dist_zscore(num_neur,
                                                                                                  num_cp, tastant_spike_times,
                                                                                                  segment_spike_times, segment_names,
                                                                                                  segment_times, pop_taste_cp_raster_inds,
                                                                                                  start_dig_in_times, pre_taste_dt,
                                                                                                  post_taste_dt, bin_dt, trial_start_frac)

    # %% Decode all neurons
    print("\n===Decoding using all neurons.===\n")

    bayes_dir_all = bayes_dir + 'All_Neurons/'
    if os.path.isdir(bayes_dir_all) == False:
        os.mkdir(bayes_dir_all)

    # stand in to use full population
    taste_select_epoch = np.ones((num_cp, num_neur))

    # Run decoding!
    bdf.decode_epochs(tastant_fr_dist_pop, segment_spike_times, post_taste_dt, pre_taste_dt,
                      skip_dt, dig_in_names, segment_times,
                      segment_dev_rasters, segment_dev_times,
                      segment_names, start_dig_in_times, taste_num_deliv,
                      taste_select_epoch, use_full, max_decode, max_hz_pop, bayes_dir_all,
                      neuron_count_thresh, trial_start_frac,
                      epochs_to_analyze, segments_to_analyze)

    # Plot decodings!
    bdf.plot_decoded(tastant_fr_dist_pop, num_tastes, num_neur, num_cp, segment_spike_times, tastant_spike_times,
                     start_dig_in_times, end_dig_in_times, post_taste_dt, pre_taste_dt,
                     pop_taste_cp_raster_inds, bin_dt, dig_in_names,
                     segment_times, segment_names, taste_num_deliv, taste_select_epoch,
                     use_full, bayes_dir_all, max_decode, max_hz_pop, seg_stat_bin,
                     neuron_count_thresh, trial_start_frac,
                     epochs_to_analyze, segments_to_analyze,
                     bin_pre_taste, decode_prob_cutoff, min_dev_size)

    # %% Decode taste selective neurons
    print("\n===Decoding using only taste selective neurons.===\n")

    bayes_dir_select = bayes_dir + 'Taste_Selective/'
    if os.path.isdir(bayes_dir_select) == False:
        os.mkdir(bayes_dir_select)

    data_group_name = 'taste_selectivity'
    try:
        taste_select_neur_bin = af.pull_data_from_hdf5(
            sorted_dir, data_group_name, 'taste_select_neur_bin')[0]
        taste_select_neur_epoch_bin = af.pull_data_from_hdf5(
            sorted_dir, data_group_name, 'taste_select_neur_epoch_bin')[0]
    except:
        print("ERROR: No taste selective data.")
        quit()

    # Run decoding!
    bdf.decode_epochs(tastant_fr_dist_pop, segment_spike_times, post_taste_dt, pre_taste_dt,
                      skip_dt, dig_in_names, segment_times,
                      segment_dev_rasters, segment_dev_times,
                      segment_names, start_dig_in_times, taste_num_deliv,
                      taste_select_neur_epoch_bin, use_full, max_decode, max_hz_pop, bayes_dir_select,
                      neuron_count_thresh, trial_start_frac,
                      epochs_to_analyze, segments_to_analyze)

    # Plot decodings!
    bdf.plot_decoded(tastant_fr_dist_pop, num_tastes, num_neur, num_cp, segment_spike_times, tastant_spike_times,
                     start_dig_in_times, end_dig_in_times, post_taste_dt, pre_taste_dt,
                     pop_taste_cp_raster_inds, bin_dt, dig_in_names,
                     segment_times, segment_names, taste_num_deliv, taste_select_neur_epoch_bin,
                     use_full, bayes_dir_select, max_decode, max_hz_pop, seg_stat_bin,
                     neuron_count_thresh, trial_start_frac,
                     epochs_to_analyze, segments_to_analyze,
                     bin_pre_taste, decode_prob_cutoff, min_dev_size)

    # %% Decode all neurons z-scored
    print("\n===Decoding using all neurons z-scored.===\n")

    bayes_dir_all_z = bayes_dir + 'All_Neurons_ZScore/'
    if os.path.isdir(bayes_dir_all_z) == False:
        os.mkdir(bayes_dir_all_z)

    taste_select = np.ones(num_neur)  # stand in to use full population
    # stand in to use full population
    taste_select_epoch = np.ones((num_cp, num_neur))

    # Run decoding!
    bdf.decode_epochs_zscore(tastant_fr_dist_z_pop, segment_spike_times, post_taste_dt, pre_taste_dt,
                             skip_dt, dig_in_names, segment_times,
                             segment_dev_rasters, segment_dev_times,
                             segment_names, start_dig_in_times, taste_num_deliv,
                             taste_select_epoch, use_full, max_decode, max_hz_z_pop, min_hz_z_pop,
                             bayes_dir_all_z, neuron_count_thresh, bin_dt, trial_start_frac,
                             epochs_to_analyze, segments_to_analyze)

    # Plot decodings!
    bdf.plot_decoded(tastant_fr_dist_z_pop, num_tastes, num_neur, num_cp, segment_spike_times, tastant_spike_times,
                     start_dig_in_times, end_dig_in_times, post_taste_dt, pre_taste_dt,
                     pop_taste_cp_raster_inds, bin_dt, dig_in_names,
                     segment_times, segment_names, taste_num_deliv, taste_select_epoch,
                     use_full, bayes_dir_all_z, max_decode, max_hz_z_pop, seg_stat_bin,
                     neuron_count_thresh, trial_start_frac,
                     epochs_to_analyze, segments_to_analyze,
                     bin_pre_taste, decode_prob_cutoff, min_dev_size)

    # %% Decode taste selective neurons
    print("\n===Decoding using only taste selective neurons z-scored.===\n")

    bayes_dir_select_z = bayes_dir + 'Taste_Selective_ZScore/'
    if os.path.isdir(bayes_dir_select_z) == False:
        os.mkdir(bayes_dir_select_z)

    data_group_name = 'taste_selectivity'
    try:
        taste_select_neur_bin = af.pull_data_from_hdf5(
            sorted_dir, data_group_name, 'taste_select_neur_bin')[0]
        taste_select_neur_epoch_bin = af.pull_data_from_hdf5(
            sorted_dir, data_group_name, 'taste_select_neur_epoch_bin')[0]
    except:
        print("ERROR: No taste selective data.")
        quit()

    # Run decoding!
    bdf.decode_epochs_zscore(tastant_fr_dist_z_pop, segment_spike_times, post_taste_dt, pre_taste_dt,
                             skip_dt, dig_in_names, segment_times,
                             segment_dev_rasters, segment_dev_times,
                             segment_names, start_dig_in_times, taste_num_deliv,
                             taste_select_epoch, use_full, max_decode, max_hz_z_pop, min_hz_z_pop,
                             bayes_dir_select_z, neuron_count_thresh, bin_dt, trial_start_frac,
                             epochs_to_analyze, segments_to_analyze)

    # Plot decodings!
    bdf.plot_decoded(tastant_fr_dist_z_pop, num_tastes, num_neur, num_cp, segment_spike_times, tastant_spike_times,
                     start_dig_in_times, end_dig_in_times, post_taste_dt, pre_taste_dt,
                     pop_taste_cp_raster_inds, bin_dt, dig_in_names,
                     segment_times, segment_names, taste_num_deliv, taste_select_neur_epoch_bin,
                     use_full, bayes_dir_select_z, max_decode, max_hz_z_pop, seg_stat_bin,
                     neuron_count_thresh, trial_start_frac,
                     epochs_to_analyze, segments_to_analyze,
                     bin_pre_taste, decode_prob_cutoff, min_dev_size)

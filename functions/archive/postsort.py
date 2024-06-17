#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:11:45 2022

@author: hannahgermaine

This code is written to perform post-sorting functions such as collisions 
testing and re-combination of oversplit neurons.
"""
import tables
import tqdm
import os
import csv
import itertools
# If this file is not being run from the directory of .../BlechCodes/, uncomment the next two lines
#file_path = ('/').join(os.path.abspath(__file__).split('/')[0:-2])
# os.chdir(file_path)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from numba import jit
from scipy import stats
from functions.spike_clust import calculate_spike_stats


def run_postsort(datadir):
    """This function serves to run through the post-sorting flow of importing
    individual neuron waveforms, performing collision tests, and recombining 
    those neurons that are oversplit.
    INPUTS:
            - datadir = directory of all of the data (level containing sort_results folder)

    """
    # Get directories / names
    orig_dir = datadir + '/' + datadir.split('/')[-1] + '.h5'
    downsampled_dir = datadir + '/' + \
        datadir.split('/')[-1] + '_downsampled.h5'
    sort_data_dir = datadir + '/sort_results/'
    sort_hf5_name = datadir.split(
        '/')[-1].split('.')[0].split('_')[0] + '_sort.h5'
    sort_hf5_dir = sort_data_dir + sort_hf5_name
    save_sort_dir = datadir + '/' + datadir.split('/')[-1] + '_repacked.h5'
    sort_data_settings = sort_data_dir + 'sort_settings.csv'

    # Check for sort settings
    if os.path.isfile(sort_data_settings) == True:
        with open(sort_data_settings, newline='') as f:
            reader = csv.reader(f)
            sort_settings_list = list(reader)
        clust_type = sort_settings_list[0][0]
    else:
        clust_type = 'no_sort_file'

    # Import data info
    orig_dir_info = tables.open_file(orig_dir, 'r', title=orig_dir[-1])
    raw_emg = []
    for i in orig_dir_info.root.raw_emg:
        raw_emg.append(i[:])
    orig_dir_info.close()
    downsampled_dir_info = tables.open_file(
        downsampled_dir, 'r', title=downsampled_dir[-1])
    num_new_time = np.shape(downsampled_dir_info.root.electrode_array.data)[-1]
    sampling_rate = downsampled_dir_info.root.sampling_rate[0]
    segment_names = [downsampled_dir_info.root.experiment_components.segment_names[i].decode(
        'UTF-8') for i in range(len(downsampled_dir_info.root.experiment_components.segment_names))]
    segment_times = downsampled_dir_info.root.experiment_components.segment_times[:]
    dig_in_names = [downsampled_dir_info.root.dig_ins.dig_in_names[i].decode(
        'UTF-8') for i in range(len(downsampled_dir_info.root.dig_ins.dig_in_names))]
    dig_ins = downsampled_dir_info.root.dig_ins.dig_ins[0]
    downsampled_dir_info.close()

    # Import sorted data
    separated_spikes_ind, separated_spikes_wav, separated_spikes_stats = import_sorted(datadir,
                                                                                       sort_data_dir,
                                                                                       sort_hf5_dir,
                                                                                       sampling_rate,
                                                                                       num_new_time)

    if clust_type == 'nosort':
        separated_spikes_ind, separated_spikes_wav = reorg_unsorted(
            separated_spikes_ind, separated_spikes_wav)
        separated_spikes_ind, separated_spikes_wav, separated_spikes_stats, collision_mat = compare_unsorted(
            separated_spikes_ind, separated_spikes_wav, separated_spikes_stats, sampling_rate, num_new_time)

    else:
        # Ask for the user to select waveforms to keep and remove
        separated_spikes_ind, separated_spikes_wav, separated_spikes_stats = user_removal(
            separated_spikes_ind, separated_spikes_wav, separated_spikes_stats, datadir)

        # Plot and combine waveforms
        separated_spikes_ind, separated_spikes_wav, separated_spikes_stats = compare_waveforms(separated_spikes_wav, separated_spikes_ind,
                                                                                               separated_spikes_stats, sampling_rate, num_new_time)

        # Perform collision tests of sorted data
        collision_results_dir = sort_data_dir + 'collision_results/'
        if os.path.isdir(collision_results_dir) == False:
            os.mkdir(collision_results_dir)
        collision_mat, remove_ind = test_collisions(
            sampling_rate, separated_spikes_ind, separated_spikes_wav, collision_results_dir)
        if len(remove_ind) > 1:
            remove_ind.sort(reverse=True)
        # Remove data selected above
        for r_i in remove_ind:
            del separated_spikes_ind[r_i]
            del separated_spikes_wav[r_i]
            del separated_spikes_stats[r_i+1]
            np.delete(collision_mat, r_i, axis=0)
            np.delete(collision_mat, r_i, axis=1)
        del remove_ind

    # Save new sort data in new .h5 file with ending "_repacked.h5"
    save_sort_hdf5(sampling_rate, separated_spikes_ind, separated_spikes_wav,
                   separated_spikes_stats, save_sort_dir, segment_names,
                   segment_times, dig_in_names, dig_ins, raw_emg, collision_mat)


def import_sorted(dir_save, sort_data_dir, sort_hf5_dir, sampling_rate, num_new_time):
    """This function imports the already sorted data into arrays + the sorting
    statistics
    INPUTS:
            - dir_save: the directory of all data
            - sort_data_dir: the directory of the sort results ('.../sort_results')
            - sort_hf5_dir: the directory of the dataset results (_sort.h5)
            - sampling_rate: sampling rate of data
            - num_new_time: max recording time
    OUTPUTS:
            - separated_spikes_ind: a list of lists containing the spike times for each unit
            - sort_stats: an array of averaged sorting statistics for each unit
    """

    print("Now importing sorted data.")
    separated_spikes_ind = []
    separated_spikes_wav = []
    separated_spikes_stats = [['Electrode Index', 'Number Waveforms',
                               '1 ms Violations', '2 ms Violations', 'Average Firing Rate']]
    sort_hf5 = tables.open_file(sort_hf5_dir, 'r', title=sort_hf5_dir[-1])
    sorted_units_node = sort_hf5.get_node('/sorted_units')
    i = -1  # Counter for total number of electrodes
    for s_n in sorted_units_node:
        i += 1
        # Number neurons in electrode
        num_sub_u = len([w_n for w_n in s_n.times])
        # For each electrode pull out unit spike times
        unit_spike_times = []
        for n_u in range(num_sub_u):
            unit_times = eval('s_n.times.neuron_' +
                              str(n_u) + '[0]').round().astype(int)
            separated_spikes_ind.append([unit_times])
            unit_spike_times.append([unit_times])
            unit_wavs = eval('s_n.waveforms.neuron_' +
                             str(n_u) + '[0]').round().astype(float)
            separated_spikes_wav.append([unit_wavs])
        del n_u, unit_times, unit_wavs
        # For each electrode pull out the sort statistics
        sort_neur_dir = sort_data_dir + 'unit_' + str(num_sub_u) + '/'
        save_folder = sort_neur_dir + 'final_results/'
        # Import grouped stats from CSV
        new_sort_neur_stats_csv = save_folder + 'sort_stats.csv'
        if os.path.isdir(new_sort_neur_stats_csv) == 'True':
            with open(new_sort_neur_stats_csv, newline='') as f:
                reader = csv.reader(f)
                sort_stats_list = list(reader)
            for i_s in range(len(sort_stats_list) - 1):
                stat_row = sort_stats_list[i_s + 1]
                stat_row_float = [i, i_s + 1]
                stat_row_float.extend([float(stat_row[i])
                                      for i in range(len(stat_row) - 1)])
                if float(stat_row[-1]) == 1:
                    separated_spikes_stats.append(stat_row_float)
            del reader, sort_stats_list, i_s, stat_row, stat_row_float
        else:
            # If separated_spikes_stats is not populated, populate it
            viol_1 = sampling_rate*(1/1000)  # 1 ms in samples
            viol_2 = sampling_rate*(2/1000)  # 2 ms in samples
            for n_u in range(num_sub_u):
                peak_ind = unit_spike_times[n_u][0]
                num_spikes, viol_1_percent, viol_2_percent, avg_fr = calculate_spike_stats(peak_ind,
                                                                                           sampling_rate, num_new_time,
                                                                                           viol_1, viol_2)
                separated_spikes_stats.append(
                    [i, num_spikes, viol_1_percent, viol_2_percent, avg_fr])
        del num_sub_u, sort_neur_dir, save_folder, new_sort_neur_stats_csv
    sort_hf5.close()

    return separated_spikes_ind, separated_spikes_wav, separated_spikes_stats


def user_removal(separated_spikes_ind, separated_spikes_wav, separated_spikes_stats, dir_save):
    """This function allows the user to run through each spike waveform and 
    remove any they feel are not true spikes before collisions testing and storage"""

    print("Now beginning manual cluster removal process.")
    # First check for previously selected keep and remove indices
    remove_csv = dir_save + '/sort_results/remove_indices.csv'
    keep_csv = dir_save + '/sort_results/keep_indices.csv'
    if os.path.isfile(keep_csv) & os.path.isfile(remove_csv):
        print(
            "Keep/Remove indices already exist. Would you like to import them or reselect?")
        keep_loop = 1
        while keep_loop == 1:
            keep_val = input("Import [y/n]? ")
            if keep_val != 'y' and keep_val != 'n':
                print("Incorrect selection. Try again.")
            else:
                keep_loop = 0
    else:
        keep_val = 'n'

    if keep_val == 'y':
        with open(keep_csv, newline='') as f:
            reader = csv.reader(f)
            keep_list = list(reader)
        keep_ind = []
        for i_s in range(len(keep_list[0])):
            keep_ind.append(int(keep_list[0][i_s]))
    else:
        print("\nINSTRUCTIONS: Please look at each figure and decide whether to keep it or not.")
        print("Close the figure once decided, and select 'y' or 'n'.")
        # Run through each "neuron" and plot it and ask for the user to "keep" or "reject"
        num_neur = len(separated_spikes_ind)
        stat_names = separated_spikes_stats[0]
        keep_ind = []
        remove_ind = []
        for i in range(num_neur):
            # Pull data
            ind_spike = separated_spikes_ind[i][0]
            ind_wav = separated_spikes_wav[i][0]
            ind_stats = separated_spikes_stats[i+1]
            mean_wav = np.mean(ind_wav, axis=0)
            std_wav = np.std(ind_wav, axis=0)
            num_wavs = len(ind_spike)
            # Grab rand ind to visualize
            plot_num_vis = min(1500, num_wavs-1)
            # Pick 500 random waveforms to plot
            plot_ind = np.random.randint(0, num_wavs, size=(plot_num_vis,))
            # Visualize waveform
            fig = plt.figure(figsize=(8, 8))
            plt.subplot(2, 2, 1)
            plt.plot(mean_wav, c='b', alpha=1)
            plt.plot(mean_wav + std_wav, c='b', alpha=0.5)
            plt.plot(mean_wav - std_wav, c='b', alpha=0.5)
            plt.title('Average Waveform')
            plt.subplot(2, 2, 2)
            for p_i in plot_ind:
                plt.plot(ind_wav[p_i], c='b', alpha=0.01)
            plt.title('Waveform Overlays')
            plt.subplot(2, 2, 3)
            plt.hist(ind_spike, min(100, num_wavs/10))
            plt.title('Spike Time Histogram')
            title_name = ''
            for p_i in range(len(stat_names)):
                title_name += (stat_names[p_i] + ' ' +
                               str(ind_stats[p_i]) + '\n')
            plt.suptitle(title_name)
            plt.tight_layout()
            plt.show()
            keep_loop = 1
            while keep_loop == 1:
                keep_neuron = input(
                    "Would you like to keep neuron " + str(i) + " [y/n]? ")
                if keep_neuron != 'n' and keep_neuron != 'y':
                    print("Incorrect entry, try again.")
                elif keep_neuron == 'y':
                    keep_ind.extend([i])
                    keep_loop = 0
                    plt.close()
                elif keep_neuron == 'n':
                    remove_ind.extend([i])
                    keep_loop = 0
                    plt.close()
        # Print and save indices to be removed, for user knowledge
        print("Indices to be removed:\n")
        print(remove_ind)
        with open(remove_csv, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows([remove_ind])
        # Save indices to be kept, for user knoweldge
        with open(keep_csv, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows([keep_ind])

    # Now clean the dataset and return only the keep indices
    new_separated_spikes_ind = []
    new_separated_spikes_wav = []
    new_separated_spikes_stats = [separated_spikes_stats[0]]
    for k_i in keep_ind:
        new_separated_spikes_ind.append(separated_spikes_ind[k_i])
        new_separated_spikes_wav.append(separated_spikes_wav[k_i])
        new_separated_spikes_stats.append(separated_spikes_stats[k_i + 1])

    return new_separated_spikes_ind, new_separated_spikes_wav, new_separated_spikes_stats


def test_collisions(sampling_rate, spike_times, spike_wavs, dir_save):
    """This function tests the final selected neurons for collisions across 
    all units. It performs pairwise tests and looks for spike times within 3 
    time bins, totalling the number of overlaps / average number of spikes 
    between the two neurons. If the percentage is over 50, the pair is flagged 
    and the user can determine which to remove based on the statistics.
    INPUTS:
            - sampling_rate = sampling rate of data
            - spike_times = list of lists containing indices of spike times
            - spike_wavs = list of lists containing waveforms
            - spike_stats = list of lists containing neuron stats
            - dir_save = directory to store collision results (.../collision_results/)
    OUTPUTS:
            - remove_ind = list of indices the user has selected to remove from the dataset
            - collision_percents = np array of collision percents between units
    """
    num_neur_plot = 500
    num_neur = len(spike_times)
    all_pairs = list(itertools.combinations(np.arange(0, num_neur), 2))
    blur_ind = round((1/1000)*sampling_rate)  # 1 ms collision window
    collision_cutoff = 50  # Percent cutoff for collisions
    # Colors for plotting collision rasters
    colorCodes = np.array([[0, 1, 0], [0, 0, 1]])
    colllision_csv = dir_save + 'collision_percents.csv'

    # First check for data to import
    if os.path.isfile(colllision_csv):
        print("Collisions previously calculated. Would you like to import them or recalculate?")
        recalc_loop = 1
        while recalc_loop == 1:
            import_val = input("Import [y/n]? ")
            if import_val != 'y' and import_val != 'n':
                print("Incorrect entry. Enter only n or y.")
            else:
                recalc_loop = 0
    else:
        import_val = 'n'
    # Now import or calculate
    if import_val == 'y':
        with open(colllision_csv, newline='') as f:
            reader = csv.reader(f)
            collision_csv_import = list(reader)
        collision_percents = []
        for i_c in range(len(collision_csv_import)):
            str_list = collision_csv_import[i_c]
            float_list = [float(str_list[i]) for i in range(len(str_list))]
            collision_percents.append(float_list)
        collision_percents = np.array(collision_percents)
        collision_percents_flat = collision_percents[np.triu_indices(
            np.shape(collision_percents)[1])]
        # Calculate number of collisions
        collisions_indices = np.where(collision_percents > collision_cutoff)
        if len(collisions_indices[0]) > 0:
            collisions_detected = len(collisions_indices[0])
            for i in range(len(collisions_indices)):
                print("Collisions detected between unit " +
                      str(collisions_indices[i][0]) + " and unit " + str(collisions_indices[i][1]))
        else:
            collisions_detected = 0
    else:
        print("\t Testing all units pairwise.")
        collisions_detected = 0
        # row, column value = % of row index that collided with column index
        collision_percents = np.zeros((num_neur, num_neur))
        for i in tqdm.tqdm(range(len(all_pairs))):
            ind_1 = all_pairs[i][0]
            ind_2 = all_pairs[i][1]
            spike_1_array = spike_times[ind_1]
            spike_2_array = spike_times[ind_2]
            spike_1_list = list(spike_1_array)
            spike_2_list = list(spike_2_array)

            # Calculate overlaps
            spike_1_overlaps, spike_2_overlaps = collision_func(
                spike_1_list, spike_2_list, blur_ind)

            col_perc_1 = np.round(100*spike_1_overlaps/len(spike_1_list), 2)
            collision_percents[ind_1, ind_2] = col_perc_1
            col_perc_2 = np.round(100*spike_2_overlaps/len(spike_2_list), 2)
            collision_percents[ind_2, ind_1] = col_perc_2
            if (col_perc_1 >= collision_cutoff) or (col_perc_2 >= collision_cutoff):
                collisions_detected += 1
                spikes_combined = [spike_1_list, spike_2_list]
                spikes_1_count = len(spike_1_list)
                spikes_2_count = len(spike_2_list)
                # Create a figure of the spike rasters together and save
                fig = plt.figure(figsize=(20, 20))
                plt.subplot(2, 2, 1)
                spike_1_wavs = spike_wavs[ind_1]
                mean_bit = np.mean(spike_1_wavs, axis=0)
                std_bit = np.std(spike_1_wavs, axis=0)
                plt.plot(mean_bit, '-b', alpha=1)
                plt.plot(mean_bit + std_bit, '-b', alpha=0.5)
                plt.plot(mean_bit - std_bit, '-b', alpha=0.5)
                plt.xlabel('Time (samples)')
                plt.ylabel('mV')
                plt.title('Unit ' + str(ind_1))
                plt.subplot(2, 2, 2)
                spike_2_wavs = spike_wavs[ind_2]
                mean_bit = np.mean(spike_2_wavs, axis=0)
                std_bit = np.std(spike_2_wavs, axis=0)
                plt.plot(mean_bit, '-b', alpha=1)
                plt.plot(mean_bit + std_bit, '-b', alpha=0.5)
                plt.plot(mean_bit - std_bit, '-b', alpha=0.5)
                plt.xlabel('Time (samples)')
                plt.ylabel('mV')
                plt.title('Unit ' + str(ind_2))
                plt.subplot(2, 2, 3)
                plt.eventplot(spikes_combined, colors=colorCodes)
                plt.xlim([0, 500000])
                plt.ylabel('Neuron')
                plt.xlabel('Spike')
                plt.title('Zoomed Raster Plot')
                # Final title stuff
                line_1 = 'Unit ' + str(ind_1) + ' vs. Unit ' + str(ind_2)
                line_2 = 'Collision Percents = ' + \
                    str(col_perc_1) + ' and ' + str(col_perc_2)
                line_3 = 'U' + str(ind_1) + ' counts = ' + str(spikes_1_count)
                line_4 = 'U' + str(ind_2) + ' counts = ' + str(spikes_2_count)
                plt.suptitle(line_1 + '\n' + line_2 + '\n' +
                             line_3 + '\n' + line_4, fontsize=20)
                plt.savefig(dir_save + 'unit_' + str(ind_1) +
                            '_v_unit_' + str(ind_2) + '.png', dpi=100)
                plt.close(fig)

        # Create figure of collision percents and store
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(collision_percents)
        plt.colorbar()
        plt.title('Collision Percents for All Pairs')
        plt.savefig(dir_save + 'collision_percents.png', dpi=100)
        plt.close(fig)

        # Save .csv of collision percents and store
        # Import the existing settings
        with open(colllision_csv, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(list(collision_percents))

        print("Collisions detected: " + str(collisions_detected))

    if collisions_detected > 0:
        print(
            '\n INPUT REQUESTED: Collision plots have been made and stored in ' + dir_save)
        remove_ind = input(
            'Please provide the indices you would like to remove (comma-separated ex. 0,1,2): ').split(',')
        remove_ind = [int(remove_ind[i]) for i in range(len(remove_ind))]
    else:
        remove_ind = []

    return collision_percents, remove_ind


@jit(forceobj=True)
def collision_func(spikes_1, spikes_2, blur_ind):
    """Numba compiled function to compute overlaps for 2 neurons
    INPUTS:
            -spikes_1: spike indices for neuron 1
            -spikes_2: spike indices for neuron 2
            -blur_ind: number of indices within which to test for collisions
    OUTPUTS:
            -spike_1_overlaps: number of spikes fired by neuron 1 overlapping with neuron 2 spikes
            -spike_2_overlaps: number of spikes fired by neuron 2 overlapping with neuron 1 spikes
    """
    spike_1_overlaps = 0
    spike_2_overlaps = 0
    for s_1 in spikes_1:
        for s_2 in spikes_2:
            if abs(s_1 - s_2) <= blur_ind:
                spike_1_overlaps += 1
                spike_2_overlaps += 1

    return spike_1_overlaps, spike_2_overlaps


def compare_waveforms(separated_spikes_wav, separated_spikes_ind,
                      separated_spikes_stats, sampling_rate, num_new_time):
    """This function compares remaining neurons by their waveforms and combines
    them based on waveform similarity"""
    print("Now beginning waveform overlap tests.")
    num_neur = len(separated_spikes_wav)
    mean_dist = np.zeros((num_neur, num_neur))
    for i in range(num_neur):
        mean_i = np.mean(separated_spikes_wav[i][0], 0)
        for j in range(num_neur - 1):
            mean_j = np.mean(separated_spikes_wav[j+1][0], 0)
            dist = np.round(np.sqrt(np.sum((mean_i - mean_j)**2)), 2)
            mean_dist[i, j+1] = dist
            mean_dist[j+1, i] = dist
    mean_dist_flat = mean_dist[np.triu_indices(num_neur)]
    # Calculate percentiles of distance - the higher the value, the greater the distance
    percentiles_dist = np.zeros((num_neur, num_neur))
    candidate_comb = np.zeros((num_neur, num_neur))
    for i in range(num_neur):
        for j in range(num_neur - 1):
            percent = stats.percentileofscore(
                mean_dist_flat, mean_dist[i, j+1])
            percentiles_dist[i, j+1] = percent
            percentiles_dist[j+1, i] = percent
            if percent < 50:
                candidate_comb[i, j+1] = 1
                candidate_comb[j+1, i] = 1

    # To store pairs of neurons that you'd like to combine
    combine_pairs = np.zeros((num_neur, num_neur))
    already_combined = []
    for i in range(num_neur):
        wav_i = separated_spikes_wav[i][0]
        mean_i = np.mean(wav_i, 0)
        std_i = np.std(wav_i, 0)
        ind_i = separated_spikes_ind[i][0]
        stats_i = separated_spikes_stats[i+1]
        ind_overlap = np.setdiff1d(
            np.where(percentiles_dist[i, :] <= 20)[0], already_combined)
        if len(ind_overlap) > 1:
            dim_plot = int(np.ceil(np.sqrt(len(ind_overlap) + 1)))
            plt.figure(figsize=(20, 20))
            ax_last = plt.subplot(dim_plot, dim_plot, len(ind_overlap) + 1)
            for i_o in range(len(ind_overlap)):
                j = ind_overlap[i_o]
                wav_j = separated_spikes_wav[j][0]
                mean_j = np.mean(wav_j, 0)
                std_j = np.std(wav_j, 0)
                ind_j = separated_spikes_ind[j][0]
                stats_j = separated_spikes_stats[j+1]
                # Plot the matching waveform
                plt.subplot(dim_plot, dim_plot, i_o+1)
                plt.plot(mean_j, c='r', alpha=1)
                plt.plot(mean_j + std_j, c='r', alpha=0.5)
                plt.plot(mean_j - std_j, c='r', alpha=0.5)
                plt.title("Unit " + str(j))
                ax_last.plot(mean_j, alpha=1)
                ax_last.plot(mean_j + std_j, alpha=0.5)
                ax_last.plot(mean_j - std_j, alpha=0.5)
            plt.suptitle('Average and Overlaid Waveforms')
            plt.tight_layout()
            plt.show()
            # Ask for user input on whether to combine
            combine_loop = 1
            while combine_loop == 1:
                combine_neuron = input(
                    "Would you like to combine these neurons [y/n]? ")
                if combine_neuron != 'n' and combine_neuron != 'y':
                    print("Incorrect entry, try again.")
                elif combine_neuron == 'y':
                    for i_o in ind_overlap:
                        combine_pairs[i, i_o] = 1
                        combine_pairs[i_o, i] = 1
                        already_combined.extend([i_o])
                    combine_loop = 0
                    plt.close()
                elif combine_neuron == 'n':
                    combine_loop = 0
                    plt.close()

    remaining_ind = np.arange(num_neur)
    neur_remaining = num_neur
    already_combined = []
    new_separated_spikes_wav = []
    new_separated_spikes_ind = []
    new_separated_spikes_stats = [separated_spikes_stats[0]]
    while neur_remaining > 0:
        n_c = remaining_ind[0]
        overlaps = np.where(combine_pairs[n_c, :] == 1)[0]
        unmatched_neur = np.setdiff1d(overlaps, np.array(already_combined))
        used_neur = list(unmatched_neur)
        if len(overlaps) > 0:
            if len(unmatched_neur) > 0:
                combined_wav = list(separated_spikes_wav[n_c][0])
                combined_ind = list(separated_spikes_ind[n_c][0])
                for u_n in unmatched_neur:
                    combined_wav.extend(list(separated_spikes_wav[u_n][0]))
                    combined_ind.extend(list(separated_spikes_ind[u_n][0]))
                # Calculate updated stats
                # If separated_spikes_stats is not populated, populate it
                viol_1 = sampling_rate*(1/1000)  # 1 ms in samples
                viol_2 = sampling_rate*(2/1000)  # 2 ms in samples
                num_spikes, viol_1_percent, viol_2_percent, avg_fr = calculate_spike_stats(combined_ind,
                                                                                           sampling_rate, num_new_time,
                                                                                           viol_1, viol_2)
                combined_stats = [-1, num_spikes,
                                  viol_1_percent, viol_2_percent, avg_fr]
                print([separated_spikes_stats[0][i] + ': ' + str(combined_stats[i])
                      for i in range(len(combined_stats))])
                print(
                    '\nINPUT REQUESTED: Please review the above combined neuron statistics.')
                keep_loop = 1
                while keep_loop == 1:
                    keep_clust = input(
                        'Would you like to keep this combined neuron [y/n]? ')
                    if keep_clust != 'y' and keep_clust != 'n':
                        print("Incorrect entry. Please try again.")
                    else:
                        keep_loop = 0
                if keep_clust == 'y':
                    new_separated_spikes_wav.append(np.array(combined_wav))
                    new_separated_spikes_ind.append(np.array(combined_ind))
                    new_separated_spikes_stats.append(combined_stats)
                used_neur.extend([n_c])
                # Update while loop parameters
                remaining_ind = np.setdiff1d(remaining_ind, used_neur)
                neur_remaining = len(remaining_ind)
            else:
                new_separated_spikes_wav.append(
                    np.array(separated_spikes_wav[n_c][0]))
                new_separated_spikes_ind.append(
                    np.array(separated_spikes_ind[n_c][0]))
                new_separated_spikes_stats.append(
                    separated_spikes_stats[n_c+1])
                used_neur.extend([n_c])
                remaining_ind = np.setdiff1d(remaining_ind, used_neur)
                neur_remaining = len(remaining_ind)
        else:
            new_separated_spikes_wav.append(
                np.array(separated_spikes_wav[n_c][0]))
            new_separated_spikes_ind.append(
                np.array(separated_spikes_ind[n_c][0]))
            new_separated_spikes_stats.append(separated_spikes_stats[n_c+1])
            used_neur.extend([n_c])
            remaining_ind = np.setdiff1d(remaining_ind, used_neur)
            neur_remaining = len(remaining_ind)

    return new_separated_spikes_ind, new_separated_spikes_wav, new_separated_spikes_stats


def save_sort_hdf5(sampling_rate, separated_spikes_ind, separated_spikes_wav,
                   separated_spikes_stats, save_sort_dir, segment_names,
                   segment_times, dig_in_names, dig_ins, raw_emg, collision_mat):
    """This function takes the final sort results after collision tests and 
    creates a new .h5 file with each neuron separately stored and re-numbered.
    Any additional necessary data is also stored in this file for use in analyses.
    Formatting:
            - sorted_units: folder contains unit_xxx (ex. unit_000, unit_001, etc...)
                    sub_folders with times and waveforms arrays
            - raw_emg: folder with raw emg data stored in matrix
            - digital_in: folder contains arrays dig_in_x (ex. dig_in_0, dig_in_1, ...)
                    of binary vectors containing times of dig in delivery as well as array
                    dig_in_names containing the names of each dig_in
            - sampling_rate: array with single value of sampling rate
            - sort_settings: array with settings used in sorting the data
    INPUTS:
            - sampling_rate: 
            - separated_spikes_ind:
            - separated_spikes_wav:
            - separated_spikes_stats:
            - save_sort_dir:
            - segment_names:
            - segment_times:
            - dig_in_names: names of each digital input
            - dig_ins: binary arrays of digital inputs across entire session
            - raw_emg: list of numpy arrays containing raw emg data
            - collision_mat: numpy array of collision percents between units
    """

    print("Now saving repacked .h5 file.")
    num_neur = len(separated_spikes_ind)

    # Create HDF5 file and set up folders
    hf5 = tables.open_file(save_sort_dir, 'w', title=save_sort_dir[-1])
    hf5.create_group('/', 'digital_in')
    hf5.create_group('/', 'digital_out')
    hf5.create_group('/', 'raw_emg')
    hf5.create_group('/', 'sorted_units')
    hf5.create_group('/', 'segments')
    print('Created nodes in HF5')

    # Save dig in data
    atom = tables.FloatAtom()
    for i in range(len(dig_ins)):
        np_dig_ins = dig_ins[i]
        data_digs = hf5.create_earray(
            '/digital_in', 'dig_in_' + str(i), atom, (0,)+np.shape(np_dig_ins))
        np_dig_ins = np.expand_dims(np_dig_ins, 0)
        data_digs.append(np_dig_ins)
    # tables.StringAtom(itemsize=50)
    atom = tables.Atom.from_dtype(np.dtype('U20'))
    dig_names = hf5.create_earray('/digital_in', 'dig_in_names', atom, (0,))
    dig_names.append(np.array(dig_in_names))

    # Save raw emg data
    if len(raw_emg) > 0:
        atom = tables.FloatAtom()
        for i in range(len(raw_emg)):
            data_emg = hf5.create_earray(
                '/raw_emg', 'emg_' + str(i), atom, (0,)+np.shape(raw_emg[i]))
            np_data_emg = np.expand_dims(raw_emg[i], 0)
            data_emg.append(np_data_emg)

    # Save sorted units
    for i in range(num_neur):
        unit_name = 'unit' + str(i).zfill(3)
        atom = tables.IntAtom()
        hf5.create_group('/sorted_units', unit_name)
        spike_times = separated_spikes_ind[i]
        times = hf5.create_earray(
            '/sorted_units/' + unit_name, 'times', atom, (0,) + np.shape(spike_times))
        np_times = np.expand_dims(spike_times, 0)
        times.append(np_times[:])
        atom = tables.FloatAtom()
        spike_wavs = separated_spikes_wav[i]
        waves = hf5.create_earray(
            '/sorted_units/' + unit_name, 'waveforms', atom, (0,) + np.shape(spike_wavs))
        np_waves = np.expand_dims(spike_wavs, 0)
        waves.append(np_waves[:])
        spike_stats = separated_spikes_stats[i+1]
        for s_i in range(len(spike_stats)):
            stats = hf5.create_earray(
                '/sorted_units/' + unit_name, separated_spikes_stats[0][s_i], atom, (0,))
            stats.append(np.expand_dims(spike_stats[s_i], 0))

    # Save collision_mat as "unit_distances" matrix
    atom = tables.FloatAtom()
    col_array = hf5.create_earray(
        '/', 'unit_distances', atom, (0,) + np.shape(collision_mat))
    np_collisions = np.expand_dims(collision_mat, 0)
    col_array.append(np_collisions)

    # Save sampling rate as "sampling_rate" matrix
    atom = tables.IntAtom()
    samp_array = hf5.create_earray(
        '/', 'sampling_rate', atom, (0,)+np.shape(sampling_rate))
    samp_array.append(np.expand_dims(sampling_rate, 0))

    # Save segment times/names
    atom = tables.FloatAtom()
    np_seg_times = segment_times
    data_segs = hf5.create_earray(
        '/segments', 'segment_times', atom, (0,)+np.shape(np_seg_times))
    np_seg_times = np.expand_dims(np_seg_times, 0)
    data_segs.append(np_seg_times)
    # tables.StringAtom(itemsize=50)
    atom = tables.Atom.from_dtype(np.dtype('U20'))
    seg_names = hf5.create_earray('/segments', 'segment_names', atom, (0,))
    exec('hf5.root.segments.segment_names.append(np.array(segment_names))')

    # NEVER FORGET TO CLOSE THE HDF5 FILE
    hf5.close()


def reorg_unsorted(separated_spikes_ind, separated_spikes_wav):
    """This function reorganizes the imported data for easier analysis/
    manipulation moving forward"""
    num_channels = len(separated_spikes_ind)
    separated_spikes_ind_reshaped = []
    separated_spikes_wav_reshaped = []
    for n_i in range(num_channels):
        separated_spikes_ind_reshaped.extend(separated_spikes_ind[n_i])
        separated_spikes_wav_reshaped.extend(separated_spikes_wav[n_i])

    return separated_spikes_ind_reshaped, separated_spikes_wav_reshaped


def compare_unsorted(separated_spikes_ind, separated_spikes_wav, separated_spikes_stats, sampling_rate, num_new_time):
    """This function is dedicated to post-sort protocols on unsorted data - aka
    data that has only been thresholded
    INPUTS:
            - separated_spikes_ind: indices of spike times for each channel
            - separated_spikes_wav: waveforms of spikes for each channel
            - separated_spikes_stats: spike statistics for each channel
    OUTPUTS:
            - separated_spikes_ind updated
            - separated_spikes_wav updated
            - separated_spikes_stats updated
    """
    # Set up parameters
    num_channels = len(separated_spikes_ind)
    collision_window = int((1/1000)*sampling_rate)
    overlap_cutoff = int(np.ceil(0.5*num_channels))  # number of channels

    # Binary rewrite of spike times
    print("Pulling binary spike data")
    bin_spikes = np.zeros((num_channels, num_new_time))
    for c_i in tqdm.tqdm(range(num_channels)):
        bin_spikes[c_i, separated_spikes_ind[c_i]] = 1
    bin_summed = np.sum(bin_spikes, 0)
    time_check = np.where(bin_summed > 0)[0]

    # Calculate binary vector of which indices have collisions
    print("Calculating collisions")
    collision_mat = np.zeros((num_channels, num_channels))
    wav_to_remove = [[] for n_i in range(num_channels)]
    for t_c_i in tqdm.tqdm(range(len(time_check))):
        t_i = time_check[t_c_i]
        min_t_i = int(np.max([0, t_i - collision_window]))
        max_t_i = int(np.min([num_new_time, t_i + collision_window]))
        chunk = bin_spikes[:, min_t_i:max_t_i]
        total_overlaps = np.sum(chunk)
        if total_overlaps > overlap_cutoff:
            # Remove collision indices across channels
            bin_spikes[:, min_t_i:max_t_i] = 0
            neur_collide = np.unique(np.where(chunk > 0)[0])
            if len(neur_collide) > 0:
                for n_c in neur_collide:
                    collision_mat[n_c, neur_collide] += 1
            # Mark which waveforms to remove
            [wav_to_remove[n_i].extend(list(np.where((separated_spikes_wav[n_i] >= min_t_i) & (
                separated_spikes_wav[n_i] <= max_t_i))[0])) for n_i in range(num_channels)]

    # Delete waveforms
    print("Removing waveforms")
    for n_i in range(num_channels):
        wav_array = separated_spikes_wav[n_i]
        wav_remove_i = np.sort(wav_to_remove[n_i])
        wav_remove_resort = wav_remove_i[::-1]
        for w_i in wav_remove_resort:
            np.delete(wav_array, w_i)
        separated_spikes_wav[n_i] = wav_array

    # Next re-calculate the spike stats for each channel
    viol_1 = sampling_rate*(1/1000)  # 1 ms in samples
    viol_2 = sampling_rate*(2/1000)  # 2 ms in samples
    for c_i in tqdm.tqdm(range(num_channels)):
        spike_inds = separated_spikes_ind[c_i]  # Grab its spike indices
        num_spikes, viol_1_percent, viol_2_percent, avg_fr = calculate_spike_stats(spike_inds,
                                                                                   sampling_rate, num_new_time,
                                                                                   viol_1, viol_2)
        separated_spikes_stats[c_i+1] = [c_i, num_spikes,
                                         viol_1_percent, viol_2_percent, avg_fr]

    return separated_spikes_ind, separated_spikes_wav, separated_spikes_stats, collision_mat

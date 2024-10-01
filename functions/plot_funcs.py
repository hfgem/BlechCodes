#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:54:17 2023

@author: hannahgermaine

This is a collection of miscellaneous functions for plotting data
"""

import os
import tqdm
import itertools
import numpy as np
from scipy import signal
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def raster_plots(fig_save_dir, dig_in_names, start_dig_in_times, end_dig_in_times,
                 segment_names, segment_times, segment_spike_times, tastant_spike_times,
                 pre_taste_dt, post_taste_dt, num_neur, num_tastes):

    # _____Grab spike times (and rasters) for each segment separately_____
    raster_save_dir = fig_save_dir + 'rasters/'
    if os.path.isdir(raster_save_dir) == False:
        os.mkdir(raster_save_dir)
    for s_i in tqdm.tqdm(range(len(segment_names))):
        print("\nGrabbing spike raster for segment " + segment_names[s_i])
        min_time = segment_times[s_i]  # in ms
        max_time = segment_times[s_i+1]  # in ms
        max_time_min = (max_time-min_time)*(1/1000)*(1/60)
        s_name = segment_names[s_i]
        s_t = segment_spike_times[s_i]
        s_t_time = [list((1/60)*np.array(s_t[i])*(1/1000))
                    for i in range(len(s_t))]
        del s_t
        # Plot segment rasters and save
        plt.figure(figsize=(max_time_min, num_neur))
        plt.xlabel('Time (m)')
        plt.eventplot(s_t_time, colors='k')
        plt.title(s_name + " segment")
        plt.tight_layout()
        im_name = ('_').join(s_name.split(' '))
        plt.savefig(raster_save_dir + im_name + '.png')
        plt.savefig(raster_save_dir + im_name + '.svg')
        plt.close()
        del min_time, max_time, max_time_min, s_name, s_t_time, im_name

    # _____Grab spike times for each taste delivery separately_____
    for t_i in tqdm.tqdm(range(num_tastes)):
        print("\nGrabbing spike rasters for tastant " +
              dig_in_names[t_i] + " deliveries")
        rast_taste_save_dir = raster_save_dir + \
            ('_').join((dig_in_names[t_i]).split(' ')) + '/'
        if os.path.isdir(rast_taste_save_dir) == False:
            os.mkdir(rast_taste_save_dir)
        rast_taste_deliv_save_dir = rast_taste_save_dir + 'deliveries/'
        if os.path.isdir(rast_taste_deliv_save_dir) == False:
            os.mkdir(rast_taste_deliv_save_dir)
        # Convert to seconds
        t_start = np.array(start_dig_in_times[t_i])*(1/1000)
        t_end = np.array(end_dig_in_times[t_i])*(1/1000)  # Convert to seconds
        num_deliv = len(t_start)
        t_st = tastant_spike_times[t_i]
        for t_d_i in range(len(t_start)):
            deliv_fig = plt.figure(figsize=(5, 5))
            s_t = t_st[t_d_i]
            s_t_time = [list(np.array(s_t[i])*(1/1000))
                        for i in range(len(s_t))]  # Convert to seconds
            # Plot the raster
            plt.eventplot(s_t_time, colors='k')
            plt.xlabel('Time (s)')
            plt.axvline(t_start[t_d_i], color='r')
            plt.axvline(t_end[t_d_i], color='r')
            im_name = ('_').join((dig_in_names[t_i]).split(
                ' ')) + '_raster_' + str(t_d_i)
            deliv_fig.tight_layout()
            deliv_fig.savefig(rast_taste_deliv_save_dir + im_name + '.png')
            deliv_fig.savefig(rast_taste_deliv_save_dir + im_name + '.svg')
            plt.close(deliv_fig)
        t_fig = plt.figure(figsize=(10, num_deliv))
        for t_d_i in range(len(t_start)):
            # Grab spike times into one list
            s_t = t_st[t_d_i]
            s_t_time = [list(np.array(s_t[i])*(1/1000))
                        for i in range(len(s_t))]
            # t_st.append(s_t)
            # Plot the raster
            plt.subplot(num_deliv, 1, t_d_i+1)
            plt.eventplot(s_t_time, colors='k')
            plt.xlabel('Time (s)')
            plt.axvline(t_start[t_d_i], color='r')
            plt.axvline(t_end[t_d_i], color='r')
        # Save the figure
        im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_spike_rasters'
        t_fig.tight_layout()
        t_fig.savefig(rast_taste_save_dir + im_name + '.png')
        t_fig.savefig(rast_taste_save_dir + im_name + '.svg')
        plt.close(t_fig)

        # Plot rasters for each neuron for each taste separately
        rast_taste_save_dir = raster_save_dir + \
            ('_').join((dig_in_names[t_i]).split(' ')) + '/'
        if os.path.isdir(rast_taste_save_dir) == False:
            os.mkdir(rast_taste_save_dir)
        rast_neur_save_dir = rast_taste_save_dir + 'neurons/'
        if os.path.isdir(rast_neur_save_dir) == False:
            os.mkdir(rast_neur_save_dir)
        for n_i in range(num_neur):
            #n_st = [t_st[t_d_i][n_i] - t_start[t_d_i] for t_d_i in range(len(t_start))]
            n_st = [t_st[t_d_i][n_i] - t_start[t_d_i]
                    * 1000 for t_d_i in range(len(t_start))]
            n_st_time = [list(np.array(n_st[i])*(1/1000))
                         for i in range(len(n_st))]
            t_fig = plt.figure(figsize=(10, 10))
            plt.eventplot(n_st_time, colors='k')
            plt.xlabel('Time (s)')
            plt.ylabel('Trial')
            plt.axvline(pre_taste_dt/1000, color='r')
            # Save the figure
            im_name = ('_').join(
                (dig_in_names[t_i]).split(' ')) + '_unit_' + str(n_i)
            t_fig.tight_layout()
            t_fig.savefig(rast_neur_save_dir + im_name + '.png')
            t_fig.savefig(rast_neur_save_dir + im_name + '.svg')
            plt.close(t_fig)


def PSTH_plots(fig_save_dir, num_tastes, num_neur, dig_in_names,
               start_dig_in_times, end_dig_in_times, pre_taste_dt, post_taste_dt,
               segment_times, segment_spike_times, bin_width, bin_step):

    PSTH_save_dir = fig_save_dir + 'PSTHs/'
    if os.path.isdir(PSTH_save_dir) == False:
        os.mkdir(PSTH_save_dir)
    half_bin_width_dt = int(np.ceil(1000*bin_width/2))  # in ms
    PSTH_times = []  # Storage of time bin true times (s) for each tastant
    # Storage of tastant delivery true times (s) for each tastant [start,end]
    PSTH_taste_deliv_times = []

    # First determine the minimal start delivery and maximal end delivery
    all_start_dig_in_times = []
    for sdit in range(len(start_dig_in_times)):
        all_start_dig_in_times.extend(start_dig_in_times[sdit])
    min_start_dig_in_time = np.min(np.array(all_start_dig_in_times))

    all_end_dig_in_times = []
    for edit in range(len(end_dig_in_times)):
        all_end_dig_in_times.extend(end_dig_in_times[edit])

    # Next determine which segment encapsulates these as the taste segment
    # Note: this assumes all taste deliveries fall within one segment (and they should!!)
    closest_seg_start = np.argmin(
        np.abs(segment_times - min_start_dig_in_time))
    segment_start = segment_times[closest_seg_start]
    raster_start = int(segment_start - pre_taste_dt - 1)
    # Expand just in case
    raster_end = int(segment_times[closest_seg_start + 1] + post_taste_dt + 1)
    raster_len = int(raster_end - raster_start)
    # Create binary storage array
    raster_array = np.zeros((num_neur, raster_len))
    segment_spikes = segment_spike_times[closest_seg_start]  # Grab spike times
    for n_i in range(num_neur):
        neur_segment_spikes = np.array(
            segment_spikes[n_i]).astype('int') - raster_start
        raster_array[n_i, neur_segment_spikes] = 1  # Store to binary array

    # Generate PSTH of the binary array for the full taste segment
    print("Grabbing taste interval firing rates")
    firing_rate_array = np.array([(np.sum(raster_array[:, max(t_i-half_bin_width_dt, 0):min(
        t_i+half_bin_width_dt, raster_len)], 1)/bin_width).T for t_i in tqdm.tqdm(range(raster_len))]).T

    print("Pulling firing rate arrays for each tastant delivery")
    tastant_PSTH = []  # List of numpy arrays of num_deliv x num_neur x length
    avg_tastant_PSTH = []  # List of numpy arrays of num_neur x length
    for t_i in tqdm.tqdm(range(num_tastes)):
        print("\nGrabbing PSTHs for tastant " +
              dig_in_names[t_i] + " deliveries")
        t_start = np.array(start_dig_in_times[t_i])
        t_end = np.array(end_dig_in_times[t_i])
        dt_total = int(np.max(t_end-t_start) + pre_taste_dt + post_taste_dt)
        num_deliv = len(t_start)
        PSTH_start_times = np.arange(0, dt_total)
        PSTH_true_times = np.round(PSTH_start_times/1000, 3)
        PSTH_times.append(PSTH_true_times)
        start_deliv_interval = PSTH_true_times[np.where(
            PSTH_start_times > pre_taste_dt)[0][0]]
        end_deliv_interval = PSTH_true_times[np.where(
            PSTH_start_times > dt_total - post_taste_dt)[0][0]]
        PSTH_taste_deliv_times.append(
            [start_deliv_interval, end_deliv_interval])
        all_PSTH = np.zeros((num_deliv, num_neur, len(PSTH_start_times)))
        t_fig = plt.figure(figsize=(10, num_deliv))
        for t_d_i in range(len(t_start)):
            start_i = max(int(t_start[t_d_i] - pre_taste_dt) - raster_start, 0)
            end_i = start_i + dt_total
            # Perform Gaussian convolution
            PSTH_spikes = firing_rate_array[:, start_i:end_i]
            len_PSTH_spikes = np.shape(PSTH_spikes)[1]
            plt.subplot(num_deliv, 1, t_d_i+1)
            # Update to have x-axis in time
            for i in range(num_neur):
                plt.plot(PSTH_true_times[:len_PSTH_spikes], PSTH_spikes[i, :])
            del i
            plt.axvline(start_deliv_interval, color='r')
            plt.axvline(end_deliv_interval, color='r')
            all_PSTH[t_d_i, :, :len_PSTH_spikes] = PSTH_spikes
        del t_d_i, start_i, end_i, PSTH_spikes
        tastant_name = dig_in_names[t_i]
        im_name = ('_').join((tastant_name).split(' ')) + '_PSTHs'
        t_fig.tight_layout()
        t_fig.savefig(PSTH_save_dir + im_name + '.png')
        t_fig.savefig(PSTH_save_dir + im_name + '.svg')
        plt.close(t_fig)
        avg_PSTH = np.mean(all_PSTH, axis=0)
        t_fig = plt.figure()
        for i in range(num_neur):
            plt.plot(PSTH_true_times, avg_PSTH[i, :])
        del i
        plt.axvline(start_deliv_interval, color='r', linestyle='dashed')
        plt.axvline(end_deliv_interval, color='r', linestyle='solid')
        plt.title('Avg Individual Neuron PSTH for ' +
                  tastant_name + '\nAligned to Taste Delivery Start')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing Rate (Hz)')
        im_name = ('_').join((dig_in_names[t_i]).split(' ')) + '_avg_PSTH'
        t_fig.tight_layout()
        t_fig.savefig(PSTH_save_dir + im_name + '.png')
        t_fig.savefig(PSTH_save_dir + im_name + '.svg')
        plt.close(t_fig)
        tastant_PSTH.append(all_PSTH)
        avg_tastant_PSTH.append(avg_PSTH)
    del t_i, t_start, t_end, dt_total, num_deliv, PSTH_start_times, PSTH_true_times, start_deliv_interval, end_deliv_interval, all_PSTH
    del t_fig, tastant_name, im_name, avg_PSTH

    # _____Plot avg tastant PSTHs side-by-side
    f_psth, ax = plt.subplots(1, num_tastes, figsize=(
        10, 10), sharex=True, sharey=True)
    for t_i in range(num_tastes):
        tastant_name = dig_in_names[t_i]
        PSTH_true_times = PSTH_times[t_i]
        for i in range(num_neur):
            ax[t_i].plot(PSTH_true_times, avg_tastant_PSTH[t_i][i])
        ax[t_i].axvline(PSTH_taste_deliv_times[t_i][0],
                        color='r', linestyle='dashed')
        ax[t_i].axvline(PSTH_taste_deliv_times[t_i][1],
                        color='r', linestyle='dashed')
        ax[t_i].set_xlabel('Time (s)')
        ax[t_i].set_ylabel('Firing Rate (Hz)')
        ax[t_i].set_title(tastant_name)
    im_name = 'combined_avg_PSTH'
    f_psth.tight_layout()
    f_psth.savefig(PSTH_save_dir + im_name + '.png')
    f_psth.savefig(PSTH_save_dir + im_name + '.svg')
    plt.close(f_psth)

    # _____Plot avg PSTHs for Individual Neurons_____
    neuron_PSTH_dir = PSTH_save_dir + 'neurons/'
    if os.path.isdir(neuron_PSTH_dir) == False:
        os.mkdir(neuron_PSTH_dir)
    for n_i in range(num_neur):
        plt.figure(figsize=(10, 10))
        for t_i in range(num_tastes):
            plt.plot(PSTH_times[t_i], avg_tastant_PSTH[t_i]
                     [n_i, :], label=dig_in_names[t_i])
            plt.bar(PSTH_taste_deliv_times[t_i][0], height=-1, width=PSTH_taste_deliv_times[t_i]
                    [1] - PSTH_taste_deliv_times[t_i][0], alpha=0.1, label=dig_in_names[t_i] + ' delivery')
        plt.legend(loc='upper right', fontsize=12)
        plt.title('Neuron ' + str(n_i))
        plt.ylabel('Firing Rate (Hz)')
        plt.xlabel('Time (s)')
        im_name = 'neuron_' + str(n_i) + '_PSTHs'
        plt.savefig(neuron_PSTH_dir + im_name + '.png')
        plt.savefig(neuron_PSTH_dir + im_name + '.svg')
        plt.close()

    return PSTH_times, PSTH_taste_deliv_times, tastant_PSTH, avg_tastant_PSTH


def LFP_dev_plots(fig_save_dir, segment_names, segment_times, fig_buffer_size, segment_bouts, combined_waveforms, wave_sampling_rate):
    """This function plots the LFP spectrogram data in intervals surrounding the
    location of deviations
    INPUTS:
            - fig_save_dir: directory to save visualizations
            - segment_names: names of different experiment segments
            - segment_times: time indices of different segment starts/ends
            - fig_buffer_size: how much (in seconds) to plot before and after a deviation event
            - segment_bouts: bouts of time in which deviations occur
            - combined_waveforms: 0-3000 Hz range from recording
            - wave_sampling_rate: sampling rate of waveform data
    OUTPUTS:
            - Figures containing spectrograms and waveforms of LFP data surrounding deviation times
    """

    print("\nBeginning individual deviation segment plots.")
    # Create save directory
    dev_save_dir = fig_save_dir + 'deviations/'
    if os.path.isdir(dev_save_dir) == False:
        os.mkdir(dev_save_dir)
    # Convert the bin size from time to samples
    sampling_rate_ratio = wave_sampling_rate/1000
    num_segments = len(segment_names)
    [num_neur, num_time] = np.shape(combined_waveforms)
    local_bin_dt = int(np.ceil(fig_buffer_size*wave_sampling_rate))
    half_local_bin_dt = int(np.ceil(local_bin_dt/2))
    spect_NFFT = int(wave_sampling_rate*0.05)  # 10ms window
    spect_overlap = 20
    max_recording_time = len(combined_waveforms[0, :])
    # Run through deviation times by segment and plot rasters
    for s_i in range(num_segments):
        print("\nGrabbing waveforms for segment " + segment_names[s_i])
        seg_dev_save_dir = dev_save_dir + \
            ('_').join(segment_names[s_i].split(' ')) + '/'
        if os.path.isdir(seg_dev_save_dir) == False:
            os.mkdir(seg_dev_save_dir)
        seg_wav_save_dir = seg_dev_save_dir + 'dev_waveforms/'
        if os.path.isdir(seg_wav_save_dir) == False:
            os.mkdir(seg_wav_save_dir)
        seg_spect_save_dir = seg_dev_save_dir + 'dev_spectrograms/'
        if os.path.isdir(seg_spect_save_dir) == False:
            os.mkdir(seg_spect_save_dir)
        # Convert all segment times to the waveform sampling rate times
        segment_dev_start_times = segment_bouts[s_i][:, 0]*sampling_rate_ratio
        # Convert all segment times to the waveform sampling rate times
        segment_dev_end_times = segment_bouts[s_i][:, 1]*sampling_rate_ratio
        spect_f = []
        spect_dev_t = []
        for d_i in tqdm.tqdm(range(len(segment_dev_start_times))):
            min_time = int(
                max(segment_dev_start_times[d_i] - half_local_bin_dt, 0))
            max_time = int(
                min(segment_dev_end_times[d_i] + half_local_bin_dt, max_recording_time))
            dev_start_ind = segment_dev_start_times[d_i] - min_time
            dev_start_time = dev_start_ind/wave_sampling_rate
            len_dev = (segment_dev_end_times[d_i] -
                       segment_dev_start_times[d_i])
            len_dev_time = len_dev/wave_sampling_rate
            dev_waveforms = combined_waveforms[:, int(min_time):int(max_time)]
            avg_waveform = np.mean(dev_waveforms, 0)
            # Plot segment deviation raster
            plt.figure(figsize=(10, num_neur))
            plt.xlabel('Time (s)')
            plt.ylabel('Neuron Index')
            for n_i in range(num_neur):
                plt.subplot(num_neur, 1, n_i+1)
                plt.plot((1/wave_sampling_rate) *
                         np.arange(min_time, max_time), dev_waveforms[n_i, :])
                plt.axvline((1/wave_sampling_rate) *
                            (min_time + dev_start_ind), color='r')
                plt.axvline((1/wave_sampling_rate)*(min_time +
                            dev_start_ind + len_dev), color='r')
            plt.suptitle('Deviation ' + str(d_i))
            plt.tight_layout()
            im_name = 'dev_' + str(d_i) + '.png'
            plt.savefig(seg_wav_save_dir + im_name)
            plt.close()
            # Plot LFP spectrogram
            plt.figure(figsize=(num_neur, num_neur))
            f, t, Sxx = signal.spectrogram(
                avg_waveform, wave_sampling_rate, nfft=spect_NFFT, noverlap=spect_overlap)
            max_freqs = [f[np.argmax(Sxx[:, i])] for i in range(len(t))]
            spect_f.append(max_freqs)
            start_dev_int = (1/wave_sampling_rate)*(min_time + dev_start_ind)
            end_dev_int = (1/wave_sampling_rate) * \
                (min_time + dev_start_ind + len_dev)
            ind_plot = np.where(f < 300)[0]
            plt.subplot(1, 2, 1)
            plt.plot((1/wave_sampling_rate) *
                     np.arange(min_time, max_time), avg_waveform)
            plt.axvline(start_dev_int, color='r')
            plt.axvline(end_dev_int, color='r')
            plt.title('Average Waveform')
            plt.subplot(1, 2, 2)
            plt.pcolormesh(t, f[ind_plot], Sxx[ind_plot, :], shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            start_dev_int_t = np.argmin(np.abs(t - dev_start_time))
            end_dev_int_t = np.argmin(
                np.abs(t - (dev_start_time + len_dev_time)))
            spect_dev_t.append([start_dev_int_t, end_dev_int_t])
            plt.axvline(dev_start_time, color='r')
            plt.axvline(dev_start_time + len_dev_time, color='r')
            plt.title('Spectrogram')
            plt.tight_layout()
            im_name = 'dev_' + str(d_i) + '.png'
            plt.savefig(seg_spect_save_dir + im_name)
            plt.close()
        # Now look at average spectrogram around the taste delivery interval
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        for d_i in range(len(segment_dev_start_times)):
            dev_start_t = int(spect_dev_t[d_i][0])
            dev_freq_vals = spect_f[d_i]
            t_vals = np.concatenate(
                (np.arange(-dev_start_t, 0), np.arange(len(dev_freq_vals)-dev_start_t)))
            plt.plot(t_vals, dev_freq_vals, alpha=0.05)
        plt.axvline(0)
        plt.title('Dev Start Aligned')
        plt.xlabel('Aligned time (s)')
        plt.ylabel('Spectrogram Max Frequency (Hz)')
        plt.subplot(1, 2, 2)
        for d_i in range(len(segment_dev_start_times)):
            dev_end_t = int(spect_dev_t[d_i][1])
            dev_freq_vals = spect_f[d_i]
            t_vals = np.concatenate(
                (np.arange(-dev_end_t, 0), np.arange(len(dev_freq_vals)-dev_end_t)))
            plt.plot(t_vals, dev_freq_vals, alpha=0.05)
        plt.axvline(0)
        plt.title('Dev End Aligned')
        plt.xlabel('Aligned time (s)')
        plt.ylabel('Spectrogram Max Frequency (Hz)')
        plt.tight_layout()
        im_name = 'aligned_max_frequencies'
        plt.savefig(seg_spect_save_dir + im_name + '.png')
        plt.savefig(seg_spect_save_dir + im_name + '.svg')
        plt.close()


def taste_select_success_plot(taste_select_prob_joint, x_vals, x_label, name, save_dir):
    # Calculate the fraction of successful decoding by neuron by taste
    num_neur, num_tastes = np.shape(taste_select_prob_joint)
    chance = 1/num_tastes
    above_chance_all = (taste_select_prob_joint > chance).astype('int')
    num_tastes_above_chance = np.sum(above_chance_all, 1)
    # Plot the successful decoding taste count
    plt.figure(figsize=(8, 8))
    plt.imshow(num_tastes_above_chance.T)
    plt.xticks(np.arange(len(x_vals)), labels=x_vals, rotation=90)
    plt.yticks(np.arange(num_neur))
    plt.xlabel(x_label)
    plt.ylabel('Neuron Index')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_dir + name + '.png')
    plt.savefig(save_dir + name + '.svg')
    plt.close()
    # Plot the summed successful decoding to show where the most occurs
    plt.figure(figsize=(8, 8))
    summed_val = np.sum(num_tastes_above_chance.T, 0)
    max_ind = np.argmax(summed_val)
    max_val = x_vals[max_ind]
    plt.plot(x_vals, summed_val, label='summed data')
    plt.axvline(max_val, label=max_val, linestyle='dashed', color='r')
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('Summed Decoding')
    plt.tight_layout()
    plt.savefig(save_dir + name + '_summed.png')
    plt.savefig(save_dir + name + '_summed.svg')
    plt.close()

    max_neur = np.where(num_tastes_above_chance[max_ind, :] > 1)[0]

    return max_val, max_neur


def epoch_taste_select_plot(prob_taste_epoch, dig_in_names, save_dir):
    num_neur, num_tastes, num_deliv, num_cp = np.shape(prob_taste_epoch)
    colors_taste = cm.cool(np.arange(num_tastes)/(num_tastes))
    colors_cp = cm.summer(np.arange(num_cp)/(num_cp))
    # Plot all trial and all taste successful decoding probability
    for t_i in range(num_tastes):
        fig_t, axes = plt.subplots(
            nrows=1, ncols=num_cp+1, figsize=(15, 5), gridspec_kw=dict(width_ratios=[5, 5, 5, 1]))
        for e_i in range(num_cp):
            im = axes[e_i].imshow(
                prob_taste_epoch[:, t_i, :, e_i], vmin=0, vmax=1)
            axes[e_i].set_title(
                'Taste ' + str(dig_in_names[t_i]) + ' Epoch ' + str(e_i))
        #fig_t.subplots_adjust(left = 0.05, right = 0.9)
        fig_t.colorbar(im, cax=axes[num_cp])
        fig_t.savefig(save_dir + 'taste_'+str(t_i) +
                      '_epoch_taste_selectivity.png')
        fig_t.savefig(save_dir + 'taste_'+str(t_i) +
                      '_epoch_taste_selectivity.svg')
        plt.close(fig_t)
    # Plot distributions of decoding probability for each taste for each neuron
    cross_epoch_dir = save_dir + 'epoch_prob_dist/'
    if os.path.isdir(cross_epoch_dir) == False:
        os.mkdir(cross_epoch_dir)
    for n_i in range(num_neur):
        for t_i in range(num_tastes):
            fig_1 = plt.figure(figsize=(5, 5))
            dist_collect = []
            pairs = list(itertools.combinations(np.arange(num_cp), 2))
            for c_p in range(num_cp):
                c_p_data = prob_taste_epoch[n_i, t_i, :, c_p]
                dist_collect.append(c_p_data)
                c_p_data = c_p_data[~np.isnan(c_p_data)]
                c_p_data = c_p_data[~np.isinf(c_p_data)]
                if len(c_p_data) > 1:
                    plt.hist(c_p_data, bins=num_deliv, density=True, histtype='step',
                             cumulative=True, color=colors_cp[c_p], label='Epoch ' + str(c_p))
            args = [d for d in dist_collect]
            x_ticks = plt.xticks()[0]
            x_tick_diff = np.mean(np.diff(x_ticks))
            y_ticks = plt.yticks()[0]
            y_tick_diff = np.mean(np.diff(y_ticks))
            # cross-group stat sig
            try:
                kw_stat, kw_p_val = stats.kruskal(*args, nan_policy='omit')
            except:
                kw_p_val = 1
            if kw_p_val <= 0.05:
                plt.scatter(np.mean(x_ticks), np.max(y_ticks),
                            marker='*', color='k', s=100)
            # pairwise stat sig
            for pair_i in range(len(pairs)):
                pair = pairs[pair_i]
                ks_pval = stats.ks_2samp(
                    dist_collect[pair[0]], dist_collect[pair[1]])[1]
                if ks_pval < 0.05:
                    plt.scatter(np.mean(x_ticks) + (pair_i-(len(pairs)/2))*x_tick_diff/3, np.max(y_ticks) -
                                y_tick_diff/3, marker='*', color=colors_cp[pair[0]], edgecolors=colors_cp[pair[1]], s=100)
            plt.legend()
            plt.xlabel('Decoding Probability')
            plt.ylabel('Cumulative Density of Occurrence')
            fig_1.savefig(cross_epoch_dir + 'neuron_' +
                          str(n_i) + '_taste_' + str(t_i) + '.png')
            fig_1.savefig(cross_epoch_dir + 'neuron_' +
                          str(n_i) + '_taste_' + str(t_i) + '.svg')
            plt.close(fig_1)
        for c_p in range(num_cp):
            fig_2 = plt.figure(figsize=(5, 5))
            dist_collect = []
            pairs = list(itertools.combinations(np.arange(num_tastes), 2))
            for t_i in range(num_tastes):
                c_p_data = prob_taste_epoch[n_i, t_i, :, c_p]
                dist_collect.append(c_p_data)
                c_p_data = c_p_data[~np.isnan(c_p_data)]
                c_p_data = c_p_data[~np.isinf(c_p_data)]
                if len(c_p_data) > 1:
                    plt.hist(c_p_data, bins=num_deliv, density=True, histtype='step',
                             cumulative=True, color=colors_taste[t_i], label=dig_in_names[t_i])
            args = [d for d in dist_collect]
            x_ticks = plt.xticks()[0]
            x_tick_diff = np.mean(np.diff(x_ticks))
            y_ticks = plt.yticks()[0]
            y_tick_diff = np.mean(np.diff(y_ticks))
            # cross-group stat sig
            try:
                kw_stat, kw_p_val = stats.kruskal(*args, nan_policy='omit')
            except:
                kw_p_val = 1
            if kw_p_val <= 0.05:
                plt.scatter(np.mean(x_ticks), np.max(y_ticks),
                            marker='*', color='k', s=100)
            # pairwise stat sig
            for pair_i in range(len(pairs)):
                pair = pairs[pair_i]
                ks_pval = stats.ks_2samp(
                    dist_collect[pair[0]], dist_collect[pair[1]])[1]
                if ks_pval < 0.05:
                    plt.scatter(np.mean(x_ticks) + (pair_i-(len(pairs)/2))*x_tick_diff/3, np.max(y_ticks) -
                                y_tick_diff/3, marker='*', color=colors_taste[pair[0]], edgecolors=colors_taste[pair[1]], s=100)
            plt.legend()
            plt.xlabel('Decoding Probability')
            plt.ylabel('Cumulative Density of Occurrence')
            fig_2.savefig(cross_epoch_dir + 'neuron_' +
                          str(n_i) + '_epoch_' + str(c_p) + '.png')
            fig_2.savefig(cross_epoch_dir + 'neuron_' +
                          str(n_i) + '_epoch_' + str(c_p) + '.svg')
            plt.close(fig_2)


def taste_response_similarity_plots(num_tastes, num_cp, num_neur, num_segments,
                                    tastant_spike_times, start_dig_in_times,
                                    end_dig_in_times, pop_taste_cp_raster_inds,
                                    post_taste_dt, dig_in_names, save_dir):
    """Plot the individual taste responses against each other to show how similar
    different tastes are + calculate correlations"""

    epoch_trial_out_of_bounds = []
    for e_i in tqdm.tqdm(range(num_cp+1)):
        # Save directory for epoch
        epoch_save_dir = save_dir + 'epoch_' + str(e_i) + '/'
        if not os.path.isdir(epoch_save_dir):
            os.mkdir(epoch_save_dir)

        # Grab tastant epoch firing rate vectors
        all_taste_fr_vecs = []
        all_taste_fr_vecs_mean = np.zeros((num_tastes, num_neur))
        all_taste_trials_out_of_bounds = []
        max_num_trials_epoch = 0
        for t_i in range(num_tastes):
            # Import taste spike and cp times
            taste_spike_times = tastant_spike_times[t_i]
            num_deliv = len(taste_spike_times)
            if num_deliv > max_num_trials_epoch:
                max_num_trials_epoch = num_deliv
            taste_deliv_times = start_dig_in_times[t_i]
            pop_taste_cp_times = pop_taste_cp_raster_inds[t_i]
            # Store as binary spike arrays
            taste_spike_times_bin = np.zeros(
                (len(taste_spike_times), num_neur, post_taste_dt))
            taste_cp_times = np.zeros(
                (len(taste_spike_times), num_cp+2)).astype('int')
            taste_epoch_fr_vecs = np.zeros((len(taste_spike_times), num_neur))
            for d_i in range(num_deliv):  # store each delivery to binary spike matrix
                for n_i in range(num_neur):
                    if t_i == num_tastes-1:
                        if len(taste_spike_times[d_i][n_i]) > 0:
                            d_i_spikes = np.array(
                                taste_spike_times[d_i][n_i] - np.min(taste_spike_times[d_i][n_i])).astype('int')
                        else:
                            d_i_spikes = np.empty(0)
                    else:
                        d_i_spikes = np.array(
                            taste_spike_times[d_i][n_i] - taste_deliv_times[d_i]).astype('int')
                    if len(d_i_spikes) > 0:
                        taste_spike_times_bin[d_i, n_i,
                                              d_i_spikes[d_i_spikes < post_taste_dt]] = 1
                taste_cp_times[d_i, :] = np.cumsum(np.concatenate(
                    (np.zeros(1), np.diff(pop_taste_cp_times[d_i, :])))).astype('int')
                # Calculate the FR vectors by epoch for each taste response and the average FR vector
                taste_epoch_fr_vecs[d_i, :] = np.sum(taste_spike_times_bin[d_i, :, taste_cp_times[d_i, e_i]:taste_cp_times[d_i, e_i+1]], 1)/(
                    (taste_cp_times[d_i, e_i+1]-taste_cp_times[d_i, e_i])/1000)  # FR in HZ
            all_taste_fr_vecs.append(taste_epoch_fr_vecs)
            # Calculate average response
            taste_fr_vecs_mean = np.nanmean(taste_epoch_fr_vecs, 0)
            all_taste_fr_vecs_mean[t_i, :] = taste_fr_vecs_mean
            # ___Correlations
            # Correlation matrix
            corr_mat = np.zeros((num_deliv, num_deliv))
            for d_1 in range(num_deliv):  # First delivery
                t_vec_1 = taste_epoch_fr_vecs[d_1, :]
                for d_2 in np.arange(d_1, num_deliv):  # Second delivery
                    t_vec_2 = taste_epoch_fr_vecs[d_2, :]
                    corr_mat[d_1, d_2] = np.abs(stats.pearsonr(t_vec_1, t_vec_2)[0])
            # Correlation timeseries average
                # pre trial to current trial
            corr_mean_vec_pre = np.nansum(
                corr_mat, 0)/np.arange(1, num_deliv+1)
            corr_std_vec_pre = np.std(corr_mean_vec_pre)
            try:
                height_change = np.abs(
                    max(corr_mean_vec_pre) - min(corr_mean_vec_pre))
                p_start = [height_change, 1, 0.5]
                cmvprevar, cmvprecov = curve_fit(exp_decay, np.arange(num_deliv), corr_mean_vec_pre, nan_policy='omit',
                                                 sigma=corr_std_vec_pre*np.ones(num_deliv), p0=p_start, bounds=([-3*height_change, 0, 0], [3*height_change, 1, 1]))
                corr_mean_vec_pre_fit = exp_decay(
                    np.arange(num_deliv), *cmvprevar)
            except:
                corr_mean_vec_pre_fit = np.zeros(num_deliv)
                for ndf_i in np.arange(num_deliv):
                    corr_mean_vec_pre_fit[ndf_i] = np.mean(
                        corr_mean_vec_pre[max(ndf_i-2, 0):min(ndf_i+2, num_deliv)])
            trials_in_pre = (corr_mean_vec_pre <= corr_mean_vec_pre_fit + 2*corr_std_vec_pre)*(
                corr_mean_vec_pre >= corr_mean_vec_pre_fit - 2*corr_std_vec_pre)
            # post trial to current trial
            corr_mean_vec_post = np.nansum(
                corr_mat, 1).T/np.flip(np.arange(1, num_deliv+1))
            corr_std_vec_post = np.std(corr_mean_vec_post)
            try:
                height_change = np.abs(
                    max(corr_mean_vec_post) - min(corr_mean_vec_post))
                p_start = [height_change, 1, 0.5]
                cmvpostvar, cmvpostcov = curve_fit(exp_decay, np.arange(num_deliv), corr_mean_vec_post, nan_policy='omit',
                                                   sigma=corr_std_vec_post*np.ones(num_deliv), p0=p_start, bounds=([-3*height_change, 0, 0], [3*height_change, 1, 1]))
                corr_mean_vec_post_fit = exp_decay(
                    np.arange(num_deliv), *cmvpostvar)
            except:
                corr_mean_vec_post_fit = np.zeros(num_deliv)
                for ndf_i in np.arange(num_deliv):
                    corr_mean_vec_pre_fit[ndf_i] = np.mean(
                        corr_mean_vec_post[max(ndf_i-2, 0):min(ndf_i+2, num_deliv)])
            trials_in_post = (corr_mean_vec_post <= corr_mean_vec_post_fit + 2*corr_std_vec_post)*(
                corr_mean_vec_post >= corr_mean_vec_post_fit - 2*corr_std_vec_post)
            trials_out_of_bounds = np.where(
                (trials_in_pre*trials_in_post).astype('int') == 0)[0]
            all_taste_trials_out_of_bounds.append(trials_out_of_bounds)
            # Rolling correlation average for block of 5
            corr_block_mean_pre = np.nan*np.ones(num_deliv)
            corr_block_mean_post = np.nan*np.ones(num_deliv)
            for b_i in range(num_deliv):
                try:
                    corr_block_mean_pre[b_i] = np.nansum(
                        corr_mat[b_i-5:b_i, b_i])/5
                except:
                    "do nothing"
                try:
                    corr_block_mean_post[b_i] = np.nansum(
                        corr_mat[b_i, b_i:b_i+5])/5
                except:
                    "do nothing"
            corr_block_std_pre = np.std(corr_block_mean_pre[5:])
            corr_block_std_post = np.std(corr_block_mean_post[:-5])
            try:
                height_change = max(
                    corr_block_mean_pre[5:]) - min(corr_block_mean_pre[5:])
                p_start = [height_change, 1, 0.5]
                cbvprevar, cbvprecov = curve_fit(exp_decay, np.arange(5, num_deliv), corr_block_mean_pre[5:], nan_policy='omit',
                                                 sigma=corr_block_std_pre*np.ones(num_deliv-5), p0=p_start, bounds=([-3*height_change, 0, 0], [3*height_change, 1, 1]))
                corr_block_mean_pre_fit = exp_decay(
                    np.arange(5, num_deliv), *cbvprevar)
            except:
                corr_block_mean_pre_fit = np.zeros(num_deliv)
                for ndf_i in np.arange(5, num_deliv):
                    corr_block_mean_pre_fit[ndf_i] = np.mean(
                        corr_block_mean_pre[max(ndf_i-2, 0):min(ndf_i+2, num_deliv)])
                corr_block_mean_pre_fit = corr_block_mean_pre_fit[5:]
            try:
                height_change = max(
                    corr_block_mean_post[:-5]) - min(corr_block_mean_post[:-5])
                p_start = [height_change, 1, 0.5]
                cbvpostvar, cbvpostcov = curve_fit(exp_decay, np.arange(num_deliv-5), corr_block_mean_post[:-5], nan_policy='omit',
                                                   sigma=corr_block_std_post*np.ones(num_deliv-5), p0=p_start, bounds=([-3*height_change, 0, 0], [3*height_change, 1, 1]))
                corr_block_mean_post_fit = exp_decay(
                    np.arange(num_deliv-5), *cbvpostvar)
            except:
                corr_block_mean_post_fit = np.zeros(num_deliv)
                for ndf_i in np.arange(num_deliv-5):
                    corr_block_mean_post_fit[ndf_i] = np.mean(
                        corr_block_mean_post[max(ndf_i-2, 0):min(ndf_i+2, num_deliv)])
                corr_block_mean_post_fit = corr_block_mean_post_fit[:-5]
            # Indiv taste deliv similarity/correlation plots
            f, ax = plt.subplots(nrows=5, ncols=1, figsize=(
                10, 15), gridspec_kw=dict(height_ratios=[5, 1, 1, 1, 1]))
            # Corr mat
            img = ax[0].imshow(corr_mat, cmap='binary')
            plt.colorbar(img, ax=ax[0], location='right')
            ax[0].set_title('epoch ' + str(e_i) + ' ' +
                            dig_in_names[t_i] + ' delivery correlation')
            ax[0].set_xlabel('Delivery Index')
            ax[0].set_ylabel('Delivery Index')
            # Pre corr mean
            ax[1].plot(np.arange(num_deliv), corr_mean_vec_pre,
                       color='b', linestyle='solid', alpha=0.5, label='true')
            ax[1].plot(np.arange(num_deliv), corr_mean_vec_pre_fit,
                       color='r', linestyle='dashed', alpha=1.0, label='fit')
            ax[1].plot(np.arange(num_deliv), corr_mean_vec_pre_fit + 2*corr_std_vec_pre,
                       color='r', linestyle='dashed', alpha=0.3, label='fit+2std')
            ax[1].plot(np.arange(num_deliv), corr_mean_vec_pre_fit - 2*corr_std_vec_pre,
                       color='r', linestyle='dashed', alpha=0.3, label='fit-2std')
            ax[1].legend()
            ax[1].scatter(trials_out_of_bounds, np.ones(
                len(trials_out_of_bounds)))
            ax[1].set_ylim([-0.1, 1.1])
            ax[1].set_title('Average corr of pre to given index')
            ax[1].set_xlabel('Delivery Index')
            ax[1].set_ylabel('Correlation')
            # Post corr mean
            ax[2].plot(np.arange(num_deliv), corr_mean_vec_post,
                       color='b', linestyle='solid', alpha=0.5, label='true')
            ax[2].plot(np.arange(num_deliv), corr_mean_vec_post_fit,
                       color='r', linestyle='dashed', alpha=1.0, label='fit')
            ax[2].plot(np.arange(num_deliv), corr_mean_vec_post_fit + 2*corr_std_vec_post,
                       color='r', linestyle='dashed', alpha=0.3, label='fit+2std')
            ax[2].plot(np.arange(num_deliv), corr_mean_vec_post_fit - 2*corr_std_vec_post,
                       color='r', linestyle='dashed', alpha=0.3, label='fit-2std')
            ax[2].legend()
            ax[2].scatter(trials_out_of_bounds, np.ones(
                len(trials_out_of_bounds)))
            ax[2].set_ylim([-0.1, 1.1])
            ax[2].set_title('Average corr of post to given index')
            ax[2].set_xlabel('Delivery Index')
            ax[2].set_ylabel('Correlation')
            # Pre corr 5block
            ax[3].plot(np.arange(5, num_deliv), corr_block_mean_pre[5:],
                       color='b', linestyle='solid', alpha=0.5, label='true')
            ax[3].plot(np.arange(5, num_deliv), corr_block_mean_pre_fit,
                       color='r', linestyle='dashed', alpha=1.0, label='fit')
            ax[3].plot(np.arange(5, num_deliv), corr_block_mean_pre_fit + 2 *
                       corr_block_std_pre, color='r', linestyle='dashed', alpha=0.3, label='fit+2std')
            ax[3].plot(np.arange(5, num_deliv), corr_block_mean_pre_fit - 2 *
                       corr_block_std_pre, color='r', linestyle='dashed', alpha=0.3, label='fit-2std')
            ax[3].legend()
            ax[3].scatter(trials_out_of_bounds, np.ones(
                len(trials_out_of_bounds)))
            ax[3].set_ylim([-0.1, 1.1])
            ax[3].set_title('Average 5-trial corr of pre to given index')
            ax[3].set_xlabel('Delivery Index')
            ax[3].set_ylabel('Correlation')
            # Post corr 5block
            ax[4].plot(np.arange(num_deliv-5), corr_block_mean_post[:-5],
                       color='b', linestyle='solid', alpha=0.5, label='true')
            ax[4].plot(np.arange(num_deliv-5), corr_block_mean_post_fit,
                       color='r', linestyle='dashed', alpha=1.0, label='fit')
            ax[4].plot(np.arange(num_deliv-5), corr_block_mean_post_fit + 2 *
                       corr_block_std_post, color='r', linestyle='dashed', alpha=0.3, label='fit+2std')
            ax[4].plot(np.arange(num_deliv-5), corr_block_mean_post_fit - 2 *
                       corr_block_std_post, color='r', linestyle='dashed', alpha=0.3, label='fit-2std')
            ax[4].legend()
            ax[4].scatter(trials_out_of_bounds, np.ones(
                len(trials_out_of_bounds)))
            ax[4].set_ylim([-0.1, 1.1])
            ax[4].set_title('Average 5-trial corr of post to given index')
            ax[4].set_xlabel('Delivery Index')
            ax[4].set_ylabel('Correlation')
            plt.tight_layout()
            f.savefig(epoch_save_dir +
                      dig_in_names[t_i] + '_deliv_correlation_mat.png')
            f.savefig(epoch_save_dir +
                      dig_in_names[t_i] + '_deliv_correlation_mat.svg')
            plt.close(f)
            # ___Distances
            # Distance matrix
            dist_mat = np.zeros((num_deliv, num_deliv))
            for d_1 in range(num_deliv):  # First delivery
                t_vec_1 = taste_epoch_fr_vecs[d_1, :]
                for d_2 in np.arange(d_1, num_deliv):  # Second delivery
                    t_vec_2 = taste_epoch_fr_vecs[d_2, :]
                    dist_mat[d_1, d_2] = np.sqrt(np.sum((t_vec_1-t_vec_2)**2))
            # Distance timeseries average
            dist_mean_vec_pre = np.nansum(
                dist_mat, 0)/np.arange(1, num_deliv+1)
            dist_mean_vec_post = np.nansum(
                dist_mat, 1).T/np.flip(np.arange(1, num_deliv+1))
            # Rolling correlation average for block of 5
            dist_block_mean_pre = np.nan*np.ones(num_deliv)
            dist_block_mean_post = np.nan*np.ones(num_deliv)
            for b_i in range(num_deliv):
                try:
                    dist_block_mean_pre[b_i] = np.nansum(
                        dist_mat[b_i-5:b_i, b_i])/5
                except:
                    "do nothing"
                try:
                    dist_block_mean_post[b_i] = np.nansum(
                        dist_mat[b_i, b_i:b_i+5])/5
                except:
                    "do nothing"
            # Indiv taste deliv distance plots
            f, ax = plt.subplots(nrows=5, ncols=1, figsize=(
                10, 10), gridspec_kw=dict(height_ratios=[10, 1, 1, 1, 1]))
            img = ax[0].imshow(dist_mat, cmap='binary')
            plt.colorbar(img, ax=ax[0], location='right')
            ax[0].set_title('epoch ' + str(e_i) + ' ' +
                            dig_in_names[t_i] + ' delivery distance')
            ax[0].set_xlabel('Delivery Index')
            ax[0].set_ylabel('Delivery Index')
            ax[1].plot(np.arange(num_deliv), dist_mean_vec_pre)
            # ax[1].set_ylim([0,1])
            ax[1].set_title('Average distance of pre to given index')
            ax[1].set_xlabel('Delivery Index')
            ax[1].set_ylabel('Distance')
            ax[2].plot(np.arange(num_deliv), dist_mean_vec_post)
            # ax[2].set_ylim([0,1])
            ax[2].set_title('Average distance of post to given index')
            ax[2].set_xlabel('Delivery Index')
            ax[2].set_ylabel('Distance')
            ax[3].plot(np.arange(5, num_deliv), dist_block_mean_pre[5:])
            # ax[3].set_ylim([0,1])
            ax[3].set_title('Average 5-trial distance of pre to given index')
            ax[3].set_xlabel('Delivery Index')
            ax[3].set_ylabel('Distance')
            ax[4].plot(np.arange(num_deliv-5), dist_block_mean_post[:-5])
            # ax[4].set_ylim([0,1])
            ax[4].set_title('Average 5-trial distance of post to given index')
            ax[4].set_xlabel('Delivery Index')
            ax[4].set_ylabel('Distance')
            plt.tight_layout()
            f.savefig(epoch_save_dir +
                      dig_in_names[t_i] + '_deliv_distance_mat.png')
            f.savefig(epoch_save_dir +
                      dig_in_names[t_i] + '_deliv_distance_mat.svg')
            plt.close(f)
        all_taste_trials_out_binarized = np.zeros(
            (num_tastes, max_num_trials_epoch))
        for t_i in range(num_tastes):
            all_taste_trials_out_binarized[t_i,
                                           all_taste_trials_out_of_bounds[t_i]] = 1
        epoch_trial_out_of_bounds.append(all_taste_trials_out_binarized)

        # ___Taste avg plots
        num_pairs = len(list(itertools.combinations(np.arange(num_tastes), 2)))
        f, ax = plt.subplots(nrows=1, ncols=num_pairs)
        pair_i = 0
        for t_1 in np.arange(num_tastes-1):
            x_taste_vec = all_taste_fr_vecs_mean[t_1, :]
            for t_2 in np.arange(t_1+1, num_tastes):
                y_taste_vec = all_taste_fr_vecs_mean[t_2, :]
                corr_result = np.abs(stats.pearsonr(x_taste_vec, y_taste_vec)[0])
                dist_result = np.sqrt(np.sum((x_taste_vec-y_taste_vec)**2))
                max_vec_fr = np.max([np.max(x_taste_vec), np.max(y_taste_vec)])
                ax[pair_i].plot([0, max_vec_fr], [0, max_vec_fr],
                                alpha=0.5, linestyle='dashed', color='b')
                ax[pair_i].scatter(x_taste_vec, y_taste_vec,
                                   alpha=0.8, color='k')
                ax[pair_i].set_title(
                    'Corr = ' + str(np.round(corr_result, 2)) + '\nDist = ' + str(np.round(dist_result, 2)))
                ax[pair_i].set_xlabel(dig_in_names[t_1] + ' Avg. FR')
                ax[pair_i].set_ylabel(dig_in_names[t_2] + ' Avg. FR')
                pair_i += 1
        plt.suptitle('Average Taste Response Similarity')
        plt.tight_layout()
        f.savefig(epoch_save_dir + 'avg_tastes_compare.png')
        f.savefig(epoch_save_dir + 'avg_tastes_compare.svg')
        plt.close(f)

        # ___Indiv taste deliv plots
        num_pairs = len(list(itertools.combinations(np.arange(num_tastes), 2)))
        f, ax = plt.subplots(nrows=3, ncols=num_pairs, figsize=(10, 10))
        pair_i = 0
        min_corr = 1
        max_corr = 0
        max_num_corr = 0
        min_dist = 100000
        max_dist = 0
        max_num_dist = 0
        for t_1 in np.arange(num_tastes-1):
            x_taste_mat = all_taste_fr_vecs[t_1]
            num_taste_1 = np.shape(x_taste_mat)[0]
            for t_2 in np.arange(t_1+1, num_tastes):
                y_taste_mat = all_taste_fr_vecs[t_2]
                num_taste_2 = np.shape(y_taste_mat)[0]
                taste_pairs = list(itertools.product(
                    np.arange(num_taste_1), np.arange(num_taste_2)))
                pair_corrs = np.zeros(len(taste_pairs))
                for tp_i, tp in enumerate(taste_pairs):
                    pair_corrs[tp_i] = np.abs(
                        stats.pearsonr(x_taste_mat[tp[0], :], y_taste_mat[tp[1], :])[0])
                if np.min(pair_corrs) < min_corr:
                    min_corr = np.min(pair_corrs)
                if np.max(pair_corrs) > max_corr:
                    max_corr = np.max(pair_corrs)
                pair_dists = np.zeros(len(taste_pairs))
                for tp_i, tp in enumerate(taste_pairs):
                    pair_dists[tp_i] = np.sqrt(
                        np.sum((x_taste_mat[tp[0], :]-y_taste_mat[tp[1], :])**2))
                if np.min(pair_dists) < min_dist:
                    min_dist = np.min(pair_dists)
                if np.max(pair_dists) > max_dist:
                    max_dist = np.max(pair_dists)
                max_vec_fr = np.max([np.max(x_taste_mat), np.max(y_taste_mat)])
                ax[0, pair_i].plot([0, max_vec_fr], [
                                   0, max_vec_fr], alpha=0.5, linestyle='dashed', color='b')
                if len(x_taste_mat) == len(y_taste_mat):
                    ax[0, pair_i].scatter(
                        x_taste_mat, y_taste_mat, alpha=0.1, color='k')
                else:
                    for xt_i in range(num_taste_1):
                        for yt_i in range(num_taste_2):
                            ax[0, pair_i].scatter(
                                x_taste_mat[xt_i,:], y_taste_mat[yt_i,:], alpha=0.1, color='k')
                ax[0, pair_i].set_title('Avg Corr = ' + str(np.round(np.nanmean(
                    pair_corrs), 2)) + '\nAvg Dist = ' + str(np.round(np.nanmean(pair_dists), 2)))
                ax[0, pair_i].set_xlabel(dig_in_names[t_1] + ' FR')
                ax[0, pair_i].set_ylabel(dig_in_names[t_2] + ' FR')
                h1 = ax[1, pair_i].hist(pair_corrs)
                if np.max(h1[0]) > max_num_corr:
                    max_num_corr = np.max(h1[0])
                ax[1, pair_i].set_title('Histogram of Correlations')
                ax[1, pair_i].set_xlabel('|Correlation|')
                h2 = ax[2, pair_i].hist(pair_dists)
                if np.max(h2[0]) > max_num_dist:
                    max_num_dist = np.max(h2[0])
                ax[2, pair_i].set_title('Histogram of Distances')
                ax[2, pair_i].set_xlabel('|Distance|')
                pair_i += 1
        for p_i in range(pair_i-1):
            ax[1, p_i].set_xlim([min_corr, max_corr])
            ax[1, p_i].set_ylim([0, max_num_corr])
            ax[2, p_i].set_xlim([min_dist, max_dist])
            ax[2, p_i].set_ylim([0, max_num_dist])
        plt.suptitle('Individual Taste Response Similarity')
        plt.tight_layout()
        f.savefig(epoch_save_dir + 'all_delivered_tastes_compare.png')
        f.savefig(epoch_save_dir + 'all_delivered_tastes_compare.svg')
        plt.close(f)

    epoch_trial_out_of_bounds = np.array(epoch_trial_out_of_bounds)
    return epoch_trial_out_of_bounds


def exp_decay(x, a, b, c):
    # Function for exponential decay fit
    return a*np.exp(-b*x)+c

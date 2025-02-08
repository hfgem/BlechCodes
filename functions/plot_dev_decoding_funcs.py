#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:28:46 2024

@author: Hannah Germaine

This file is dedicated to plotting functions for deviation decoded replay events. 
"""

import os
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm
from scipy.stats import pearsonr

def plot_decoded(num_tastes, num_neur, segment_spike_times, tastant_spike_times,
                 start_dig_in_times, post_taste_dt, pre_taste_dt,
                 cp_raster_inds, z_bin_dt, dig_in_names, segment_times,
                 segment_names, save_dir, max_hz,
                 segment_dev_times, segment_dev_fr_vecs, segment_dev_fr_vecs_zscore,
                 neuron_count_thresh, e_len_dt, trial_start_frac=0,
                 epochs_to_analyze=[], segments_to_analyze=[],
                 decode_prob_cutoff=0.95):
    """Function to plot the deviation events with a buffer on either side with
    the decoding results"""
    
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    num_segments = len(segment_spike_times)
    neur_cut = np.floor(num_neur*neuron_count_thresh).astype('int')
    taste_colors = cm.brg(np.linspace(0, 1, num_tastes))
    epoch_colors = cm.jet(np.linspace(0, 1, num_cp))
    epoch_seg_taste_times = np.zeros((num_cp, num_segments, num_tastes))
    epoch_seg_taste_times_neur_cut = np.zeros(
        (num_cp, num_segments, num_tastes))
    epoch_seg_taste_times_best = np.zeros((num_cp, num_segments, num_tastes))
    epoch_seg_lengths = np.zeros((num_cp, num_segments, num_tastes))
    half_bin_z_dt = np.floor(z_bin_dt/2).astype('int')
    half_bin_decode_dt = np.floor(e_len_dt/2).astype('int')
    dev_buffer = 50 #ms
    num_to_plot = 0 #100
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)

    # Get taste segment z-score info
    s_i_taste = np.nan*np.ones(1)
    for s_i in range(len(segment_names)):
        if segment_names[s_i].lower() == 'taste':
            s_i_taste[0] = s_i
    if not np.isnan(s_i_taste[0]):
        s_i = int(s_i_taste[0])
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = seg_end - seg_start
        time_bin_starts = np.arange(
            seg_start+half_bin_z_dt, seg_end-half_bin_z_dt, half_bin_z_dt*2)
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = np.array(
                segment_spike_times_s_i[n_i] - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        tb_fr = np.zeros((num_neur, len(time_bin_starts)))
        for tb_i, tb in enumerate(tqdm.tqdm(time_bin_starts)):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_bin_z_dt:tb+half_bin_z_dt-seg_start], 1)/(2*half_bin_z_dt*(1/1000))
        mean_fr_taste = np.mean(tb_fr, 1)
        std_fr_taste = np.std(tb_fr, 1)
        std_fr_taste[std_fr_taste == 0] = 1  # to avoid nan calculations
    else:
        mean_fr_taste = np.zeros(num_neur)
        std_fr_taste = np.ones(num_neur)
        
    # Get taste response firing rate vectors
    all_taste_fr_vecs = []
    all_taste_fr_vecs_z = []
    all_taste_fr_vecs_mean = []
    all_taste_fr_vecs_mean_z = []
    
    # Grab taste firing rate vectors
    for t_i in range(num_tastes):
        # Import taste spike and cp times
        taste_spike_times = tastant_spike_times[t_i]
        taste_deliv_times = start_dig_in_times[t_i]
        max_num_deliv = len(taste_deliv_times)
        cp_times = cp_raster_inds[t_i]

        # If trial_start_frac > 0 use only trials after that threshold
        trial_start_ind = np.floor(
            max_num_deliv*trial_start_frac).astype('int')
        new_max_num_deliv = (
            max_num_deliv - trial_start_ind).astype('int')

        # Store as binary spike arrays
        by_epoch_all_taste_fr_vecs = []
        by_epoch_all_taste_fr_vecs_z = []
        by_epoch_taste_fr_vecs_mean = np.zeros((num_cp, num_neur))
        by_epoch_taste_fr_vecs_mean_z = np.zeros((num_cp, num_neur))
        for e_i in epochs_to_analyze:
            taste_spike_times_bin = np.zeros(
                (new_max_num_deliv, num_neur, post_taste_dt))  # Taste response spike times
            taste_cp_times = np.zeros(
                (new_max_num_deliv, num_cp+1)).astype('int')
            taste_epoch_fr_vecs = np.zeros(
                (new_max_num_deliv, num_neur))  # original firing rate vecs
            taste_epoch_fr_vecs_z = np.zeros(
                (new_max_num_deliv, num_neur))  # z-scored firing rate vecs
            # store each delivery to binary spike matrix
            for d_i in range(len(taste_spike_times)):
                if d_i >= trial_start_ind:
                    pre_taste_spike_times_bin = np.zeros(
                        (num_neur, pre_taste_dt))  # Pre-taste spike times
                    taste_deliv_i = taste_deliv_times[d_i]
                    for n_i in range(num_neur):
                        spikes_deliv_i = taste_spike_times[d_i][n_i]
                        if t_i == num_tastes-1:
                            if len(taste_spike_times[d_i-trial_start_ind][n_i]) > 0:
                                d_i_spikes = np.array(
                                    spikes_deliv_i - (np.min(spikes_deliv_i)+pre_taste_dt)).astype('int')
                            else:
                                d_i_spikes = np.empty(0)
                        else:
                            d_i_spikes = np.array(
                                spikes_deliv_i - taste_deliv_i).astype('int')
                        d_i_spikes_posttaste = d_i_spikes[(
                            d_i_spikes < post_taste_dt)*(d_i_spikes >= 0)]
                        d_i_spikes_pretaste = d_i_spikes[d_i_spikes <
                                                         0] + pre_taste_dt
                        if len(d_i_spikes_posttaste) > 0:
                            taste_spike_times_bin[d_i-trial_start_ind,
                                                  n_i, d_i_spikes_posttaste] = 1
                        if len(d_i_spikes_pretaste) > 0:
                            pre_taste_spike_times_bin[n_i,
                                                      d_i_spikes_pretaste] = 1
                    taste_cp_times[d_i-trial_start_ind, :] = np.concatenate(
                        (np.zeros(1), np.cumsum(np.diff(cp_times[d_i, :])))).astype('int')
                    # Calculate the FR vectors by epoch for each taste response and the average FR vector
                    
                    epoch_len_i = (taste_cp_times[d_i, e_i+1]-taste_cp_times[d_i, e_i])/1000
                    if epoch_len_i == 0:
                        taste_epoch_fr_vecs[d_i-trial_start_ind,
                                            :] = np.zeros(num_neur)
                    else:
                        taste_epoch_fr_vecs[d_i-trial_start_ind, :] = np.sum(
                            taste_spike_times_bin[d_i-trial_start_ind, :, taste_cp_times[d_i, e_i]:taste_cp_times[d_i, e_i+1]], 1)/epoch_len_i  # FR in HZ
                    # Calculate z-scored FR vector
                    taste_epoch_fr_vecs_z[d_i-trial_start_ind, :] = (
                        taste_epoch_fr_vecs[d_i-trial_start_ind, :].flatten() - mean_fr_taste)/std_fr_taste

            by_epoch_all_taste_fr_vecs.append(taste_epoch_fr_vecs)
            by_epoch_all_taste_fr_vecs_z.append(taste_epoch_fr_vecs_z)
            # Calculate average taste fr vec
            taste_fr_vecs_mean = np.nanmean(taste_epoch_fr_vecs, 0)
            taste_fr_vecs_z_mean = np.nanmean(taste_epoch_fr_vecs_z, 0)
            by_epoch_taste_fr_vecs_mean[e_i, :] = taste_fr_vecs_mean
            by_epoch_taste_fr_vecs_mean_z[e_i, :] = taste_fr_vecs_z_mean
            #taste_fr_vecs_max_hz = np.max(taste_epoch_fr_vecs)
        all_taste_fr_vecs.append(by_epoch_all_taste_fr_vecs)
        all_taste_fr_vecs_z.append(by_epoch_all_taste_fr_vecs_z)
        all_taste_fr_vecs_mean.append(by_epoch_taste_fr_vecs_mean)
        all_taste_fr_vecs_mean_z.append(by_epoch_taste_fr_vecs_mean_z)
        
    dev_decode_stats = dict()
    dev_decode_save_dir = os.path.join(
        save_dir, 'cross_epoch_dev_decodes')
    if not os.path.isdir(dev_decode_save_dir):
        os.mkdir(dev_decode_save_dir)
    for e_i in epochs_to_analyze:
        print('\t\t\tPlotting Decoding for Epoch ' + str(e_i))

        taste_select_neur = np.where(taste_select_epoch[e_i, :] == 1)[0]

        epoch_decode_save_dir = os.path.join(
            save_dir, 'decode_prob_epoch_' + str(e_i))
        if not os.path.isdir(epoch_decode_save_dir):
            print("\t\t\t\tData not previously decoded, or passed directory incorrect.")
            pass
        print('\t\t\t\tProgressing through all ' +
              str(len(segments_to_analyze)) + ' segments.')
        
        dev_decode_stats[e_i] = dict()
        for seg_ind, s_i in tqdm.tqdm(enumerate(segments_to_analyze)):
            try:
                dev_decode_array = np.load(
                    os.path.join(epoch_decode_save_dir,'segment_' + str(s_i) + \
                                 '_deviations.npy'))
                pre_dev_decode_array = np.load(
                    os.path.join(epoch_decode_save_dir,'segment_' + str(s_i) + \
                                 '_pre_deviations.npy'))
                post_dev_decode_array = np.load(
                    os.path.join(epoch_decode_save_dir,'segment_' + str(s_i) + \
                                 '_post_deviations.npy'))
            except:
                print("\t\t\t\tSegment " + str(s_i) + " Never Decoded")
                pass
            
            seg_decode_save = os.path.join(
                epoch_decode_save_dir,'segment_' + str(s_i))
            if not os.path.isdir(seg_decode_save):
                os.mkdir(seg_decode_save)
            
            dev_decode_stats[e_i][s_i] = dict()
            
            _, num_dev = np.shape(dev_decode_array)
            dev_decode_stats[e_i][s_i]['num_dev'] = num_dev
            
            plot_inds = np.array(random.sample(list(np.arange(num_dev)),num_to_plot))
            
            seg_start = segment_times[s_i]
            seg_end = segment_times[s_i+1]
            seg_len = seg_end - seg_start  # in dt = ms

            # Pull binary spikes for segment
            segment_spike_times_s_i = segment_spike_times[s_i]
            segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
            for n_i in taste_select_neur:
                n_i_spike_times = np.array(
                    segment_spike_times_s_i[n_i] - seg_start).astype('int')
                segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1

            # Z-score the segment
            time_bin_starts = np.arange(
                seg_start+half_bin_z_dt, seg_end-half_bin_z_dt, half_bin_z_dt*2)
            tb_fr = np.zeros((num_neur, len(time_bin_starts)))
            for tb_i, tb in enumerate(time_bin_starts):
                tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                        half_bin_z_dt:tb+half_bin_z_dt-seg_start], 1)/(2*half_bin_z_dt*(1/1000))
            mean_fr = np.mean(tb_fr, 1)
            std_fr = np.std(tb_fr, 1)
            std_fr[std_fr == 0] = 1
            
            #Calculate the number of deviation events decoded for each taste
            dev_decode_stats[e_i][s_i]['decode_probabilities'] = dev_decode_array
            taste_decoded = np.argmax(dev_decode_array,0)
            taste_decoded_bin = np.zeros((num_tastes,num_dev))
            taste_decoded_start_times = []
            taste_decoded_end_times = []
            for t_i in range(num_tastes):
                taste_decoded_ind = np.where(taste_decoded == t_i)[0]
                taste_decoded_bin[t_i,taste_decoded_ind] = 1
                taste_decoded_start_times.append(list(segment_dev_times[seg_ind][0,taste_decoded_ind]))
                taste_decoded_end_times.append(list(segment_dev_times[seg_ind][1,taste_decoded_ind]))
            dev_decode_stats[e_i][s_i]['num_dev_per_taste'] = np.nansum(taste_decoded_bin,1)
            
            #For each deviation event create plots of decoded results
            dev_taste_corrs = np.zeros((num_tastes,num_dev))
            dev_taste_z_corrs = np.zeros((num_tastes,num_dev))
            for dev_i in range(num_dev):
                #Collect deviation decoding probabilities
                dev_prob_decoded = dev_decode_array[:,dev_i]
                pre_dev_prob_decoded = pre_dev_decode_array[:,dev_i]
                post_dev_prob_decoded = post_dev_decode_array[:,dev_i]
                decoded_taste_ind = np.argmax(dev_prob_decoded)
                
                #Collect deviation times
                dev_times = segment_dev_times[seg_ind][:,dev_i]
                dev_start_time = dev_times[0]
                dev_end_time = dev_times[1]
                dev_len = dev_end_time - dev_start_time
                pre_dev_time = np.max([dev_start_time - dev_buffer,0])
                post_dev_time = np.min([dev_end_time + dev_buffer,seg_len])
                decode_x_vals = np.arange(pre_dev_time,post_dev_time) - dev_start_time
                d_plot_len = len(decode_x_vals)
                decode_x_ticks = np.linspace(decode_x_vals[0], decode_x_vals[-1]-1, 10).astype('int')
                raster_x_ticks = np.linspace(0,d_plot_len-1,10).astype('int')
                raster_x_tick_labels = decode_x_vals[raster_x_ticks]
                
                #Collect spike times for raster plot and firing rate plotå
                plot_spike_bin = segment_spike_times_s_i_bin[:,pre_dev_time:post_dev_time]
                plot_spike_times = [np.where(plot_spike_bin[n_i,:] == 1)[0] for n_i in range(num_neur)]
                firing_rate_vec = np.zeros(d_plot_len)
                fr_vec_x = np.arange(pre_dev_time, post_dev_time)
                for dpt_ind, dpt_i in enumerate(fr_vec_x):
                    min_t_i = np.max([dpt_i-10,0])
                    max_t_i = np.min([dpt_i + 10,seg_len])
                    spikes_i = segment_spike_times_s_i_bin[:,min_t_i:max_t_i]
                    firing_rate_vec[dpt_ind] = np.sum(spikes_i)/((max_t_i-min_t_i)/1000)/num_neur
                
                #Collect deviation FR vals
                dev_fr_vec = segment_dev_fr_vecs[seg_ind][dev_i]
                dev_fr_vec_zscore = segment_dev_fr_vecs_zscore[seg_ind][dev_i]
                
                #Reshape decoding probabilities for plotting
                decoding_array = np.zeros((num_tastes,d_plot_len))
                decoding_array[:,0:dev_buffer] = np.expand_dims(pre_dev_prob_decoded,1)*np.ones((num_tastes,dev_buffer))
                decoding_array[:,dev_buffer:dev_buffer + dev_len] = np.expand_dims(dev_prob_decoded,1)*np.ones((num_tastes,dev_len))
                decoding_array[:,-dev_buffer:] = np.expand_dims(post_dev_prob_decoded,1)*np.ones((num_tastes,dev_buffer))
                
                #Save to decoding counters
                epoch_seg_taste_times[e_i, s_i, decoded_taste_ind] += 1
                epoch_seg_lengths[e_i, s_i, decoded_taste_ind] += dev_len
                
                #Calculate correlation to mean taste responses
                corr_dev_event = np.array([pearsonr(all_taste_fr_vecs_mean[t_i][e_i, :], dev_fr_vec)[
                                             0] for t_i in range(num_tastes)])
                dev_taste_corrs[:,dev_i] = corr_dev_event
                corr_dev_event_zscore = np.array([pearsonr(all_taste_fr_vecs_mean_z[t_i][e_i, :], dev_fr_vec_zscore)[
                                             0] for t_i in range(num_tastes)])
                dev_taste_z_corrs[:,dev_i] = corr_dev_event_zscore
                
                if len(np.where(plot_inds == dev_i)[0])>0: 
                    corr_title_norm = [dig_in_names[t_i] + ' corr = ' + str(
                        np.round(corr_dev_event[t_i], 2)) for t_i in range(num_tastes)]
                    corr_title_z = [dig_in_names[t_i] + ' z-corr = ' + str(
                        np.round(corr_dev_event_zscore[t_i], 2)) for t_i in range(num_tastes)]
                    corr_title = (', ').join(
                        corr_title_norm) + '\n' + (', ').join(corr_title_z)
                    
                    #Start Figure
                    f, ax = plt.subplots(nrows=5, ncols=2, figsize=(
                        10, 10), gridspec_kw=dict(height_ratios=[1, 1, 1, 2, 2]))
                    gs = ax[0, 0].get_gridspec()
                    # Decoding probabilities
                    ax[0, 0].remove()
                    ax[1, 0].remove()
                    axbig = f.add_subplot(gs[0:2, 0])
                    leg_handles = []
                    for t_i_2 in range(num_tastes):
                        taste_decode_prob_y = decoding_array[t_i_2,:]
                        p_h, = axbig.plot(
                            decode_x_vals, decoding_array[t_i_2,:], color=taste_colors[t_i_2, :])
                        leg_handles.append(p_h)
                        taste_decode_prob_y[0] = 0
                        taste_decode_prob_y[-1] = 0
                        high_prob_binary = np.zeros(len(decode_x_vals))
                        high_prob_times = np.where(
                            taste_decode_prob_y >= decode_prob_cutoff)[0]
                        high_prob_binary[high_prob_times] = 1
                        high_prob_starts = np.where(
                            np.diff(high_prob_binary) == 1)[0] + 1
                        high_prob_ends = np.where(
                            np.diff(high_prob_binary) == -1)[0] + 1
                        if len(high_prob_starts) > 0:
                            for hp_i in range(len(high_prob_starts)):
                                axbig.fill_between(decode_x_vals[high_prob_starts[hp_i]:high_prob_ends[hp_i]], taste_decode_prob_y[
                                                   high_prob_starts[hp_i]:high_prob_ends[hp_i]], alpha=0.2, color=taste_colors[t_i_2, :])
                    axbig.axvline(0, color='k', alpha=0.5)
                    axbig.axvline(dev_len, color='k', alpha=0.5)
                    axbig.legend(leg_handles, dig_in_names, loc='right')
                    axbig.set_xticks(decode_x_ticks)
                    axbig.set_xlim([decode_x_vals[0], decode_x_vals[-1]])
                    axbig.set_ylabel('Decoding Fraction')
                    axbig.set_xlabel('Time from Deviation (ms)')
                    axbig.set_title('Event ' + str(dev_i) + '\nStart Time = ' + str(round(
                        dev_start_time/1000/60, 3)) + ' Minutes' + '\nEvent Length = ' + \
                            str(np.round(dev_len, 2)))
                    # Decoded raster
                    ax[0, 1].eventplot(plot_spike_times)
                    ax[0, 1].set_xlim([0, d_plot_len])
                    ax[0, 1].set_xticks(raster_x_ticks,labels=raster_x_tick_labels)
                    ax[0, 1].axvline(dev_buffer, color='k', alpha=0.5)
                    ax[0, 1].axvline(dev_buffer + dev_len, color='k', alpha=0.5)
                    ax[0, 1].set_ylabel('Neuron Index')
                    ax[0, 1].set_title('Event Spike Raster')
                    ax[1, 0].axis('off')
                    # Plot population firing rates w 20ms smoothing
                    ax[1, 1].plot(decode_x_vals, firing_rate_vec)
                    ax[1, 1].set_xlim([decode_x_vals[0], decode_x_vals[-1]])
                    ax[1, 1].set_xticks(decode_x_ticks)
                    ax[1, 1].axvline(0, color='k', alpha=0.5)
                    ax[1, 1].axvline(dev_len, color='k', alpha=0.5)
                    ax[1, 1].set_title('Population Avg FR (20 ms smooth)')
                    ax[1, 1].set_ylabel('FR (Hz)')
                    ax[1, 1].set_xlabel('Time from Deviation (ms)')
                    # Decoded Firing Rates
                    img = ax[2, 0].imshow(np.expand_dims(dev_fr_vec, 0), vmin=0, vmax=60)
                    ax[2, 0].set_xlabel('Neuron Index')
                    ax[2, 0].set_yticks(ticks=[])
                    #plt.colorbar(img, location='bottom',orientation='horizontal',label='Firing Rate (Hz)',panchor=(0.9,0.5),ax=ax[2,0])
                    ax[2, 0].set_title('Event FR')
                    # Decoded Firing Rates Z-Scored
                    img = ax[2, 1].imshow(np.expand_dims(
                        dev_fr_vec_zscore, 0), vmin=-3, vmax=3, cmap='bwr')
                    ax[2, 1].set_xlabel('Neuron Index')
                    ax[2, 1].set_yticks(ticks=[])
                    #plt.colorbar(img, ax=ax[2,1], location='bottom',orientation='horizontal',label='Z-Scored Firing Rate (Hz)',panchor=(0.9,0.5))
                    ax[2, 1].set_title('Event FR Z-Scored')
                    # Taste Firing Rates
                    # vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
                    img = ax[3, 0].imshow(np.expand_dims(
                        all_taste_fr_vecs_mean[decoded_taste_ind][e_i, :], 0), vmin=0, vmax=60)
                    ax[3, 0].set_xlabel('Neuron Index')
                    ax[3, 0].set_yticks(ticks=[])
                    plt.colorbar(
                        img, ax=ax[3, 0], location='bottom', orientation='horizontal', label='Firing Rate (Hz)', panchor=(0.9, 0.5))
                    ax[3, 0].set_title('Avg. Taste Resp. FR')
                    # Taste Firing Rates Z-Scored
                    img = ax[3, 1].imshow(np.expand_dims(
                        all_taste_fr_vecs_mean_z[decoded_taste_ind][e_i, :], 0), vmin=-3, vmax=3, cmap='bwr')
                    ax[3, 1].set_xlabel('Neuron Index')
                    ax[3, 1].set_yticks(ticks=[])
                    plt.colorbar(img, ax=ax[3, 1], location='bottom', orientation='horizontal',
                                 label='Z-Scored Firing Rate (Hz)', panchor=(0.9, 0.5))
                    ax[3, 1].set_title('Avg. Taste Resp. FR Z-Scored')
                    # Decoded Firing Rates x Average Firing Rates
                    max_lim = np.max([np.max(dev_fr_vec), np.max(all_taste_fr_vecs_mean[decoded_taste_ind][e_i, :])])
                    ax[4, 0].plot([0, max_lim], [0, max_lim],
                                  alpha=0.5, linestyle='dashed')
                    ax[4, 0].scatter(all_taste_fr_vecs_mean[decoded_taste_ind][e_i, :], dev_fr_vec)
                    ax[4, 0].set_xlabel('Average Taste FR')
                    ax[4, 0].set_ylabel('Decoded Taste FR')
                    ax[4, 0].set_title('Firing Rate Similarity')
                    # Z-Scored Decoded Firing Rates x Z-Scored Average Firing Rates
                    ax[4, 1].plot([-5, 5], [-5, 5], alpha=0.5, linestyle='dashed', color='k')
                    ax[4, 1].scatter(
                        all_taste_fr_vecs_mean_z[decoded_taste_ind][e_i, :], dev_fr_vec_zscore)
                    ax[4, 1].set_xlabel(
                        'Average Taste Neuron FR Std > Mean')
                    ax[4, 1].set_ylabel('Event Neuron FR Std > Mean')
                    ax[4, 1].set_title(
                        'Z-Scored Firing Rate Similarity')
                    plt.suptitle(corr_title, wrap=True)
                    plt.tight_layout()
                    # Save Figure
                    f.savefig(os.path.join(
                        seg_decode_save, 'dev_' + str(dev_i) + '.png'))
                    f.savefig(os.path.join(
                        seg_decode_save, 'dev_' + str(dev_i) + '.svg'))
                    plt.close(f)
            dev_decode_stats[e_i][s_i]['dev_taste_corrs'] = dev_taste_corrs
            dev_decode_stats[e_i][s_i]['dev_taste_z_corrs'] = dev_taste_z_corrs
    np.save(os.path.join(save_dir,'dev_decode_stats.npy'),dev_decode_stats)
        
    #Now go by segment and for plot the different decode statistics by 
    #epoch to compare - histograms of decoding probability, fractions of
    #decoded taste
    for seg_ind, s_i in tqdm.tqdm(enumerate(segments_to_analyze)):
        decode_prob_by_epoch = [] #num epochs x (num tastes x num dev)
        decode_taste_counts_by_epoch = [] #num epochs x num tastes
        dev_taste_corrs_by_epoch = [] #num epochs x (num tastes x num dev)
        dev_taste_z_corrs_by_epoch = [] #num epochs x (num tastes x num dev)
        for e_i in epochs_to_analyze:
            num_dev = dev_decode_stats[e_i][s_i]['num_dev']
            decode_prob_by_epoch.append(dev_decode_stats[e_i][s_i]['decode_probabilities'])
            decode_taste_counts_by_epoch.append(dev_decode_stats[e_i][s_i]['num_dev_per_taste'])
            dev_taste_corrs_by_epoch.append(dev_decode_stats[e_i][s_i]['dev_taste_corrs'])
            dev_taste_z_corrs_by_epoch.append(dev_decode_stats[e_i][s_i]['dev_taste_z_corrs'])
        
        #Create histograms of decoding probability of the best decoded taste by epoch for each taste
        f_decode_hist, ax_decode_hist = plt.subplots(ncols=len(epochs_to_analyze),\
                                                     nrows=2,figsize=(8,5))
        min_decode_prob = 1
        for e_ind, e_i in enumerate(epochs_to_analyze):
            max_taste = np.argmax(decode_prob_by_epoch[e_ind],0)
            for t_i in range(num_tastes):
                decode_prob_data = decode_prob_by_epoch[e_ind][t_i,np.where(max_taste == t_i)[0]]
                if len(decode_prob_data) > 0:
                    if np.nanmin(decode_prob_data) < min_decode_prob:
                        min_decode_prob = np.nanmin(decode_prob_data)
                
                ax_decode_hist[0,e_ind].hist(decode_prob_data, \
                                           histtype='step',cumulative = True, \
                                           density=True,color=taste_colors[t_i,:], \
                                           label=dig_in_names[t_i])
                ax_decode_hist[1,t_i].hist(decode_prob_data, \
                                           histtype='step',cumulative = True, \
                                           density=True,color=epoch_colors[e_i,:], \
                                           label='Epoch ' + str(e_i))
            ax_decode_hist[0,e_ind].set_xlabel('Decode Probability')
            ax_decode_hist[0,e_ind].set_ylabel('Cumulative Density')
            ax_decode_hist[0,e_ind].set_title('Epoch ' + str(e_i))
            if e_ind == 0:
                ax_decode_hist[0,e_ind].legend()
        #for e_ind, e_i in enumerate(epochs_to_analyze):
            #ax_decode_hist[0,e_ind].set_xlim([min_decode_prob - 0.1,1.1])
        for t_i in range(num_tastes):
            ax_decode_hist[1,t_i].set_xlabel('Decode Probability')
            ax_decode_hist[1,t_i].set_ylabel('Cumulative Density')
            ax_decode_hist[1,t_i].set_title(dig_in_names[t_i])
            #ax_decode_hist[1,t_i].set_xlim([min_decode_prob - 0.1,1.1])
            if t_i == 0:
                ax_decode_hist[1,t_i].legend()
        plt.suptitle('Best Decode Probabilities in ' + segment_names[s_i])
        plt.tight_layout()
        f_decode_hist.savefig(os.path.join(dev_decode_save_dir,segment_names[s_i] + '_best_decode_prob_hist.png'))
        f_decode_hist.savefig(os.path.join(dev_decode_save_dir,segment_names[s_i] + '_best_decode_prob_hist.svg'))
        plt.close(f_decode_hist)
        
        #Create histograms of all decoding probabilities by epoch for each taste
        f_decode_all_hist, ax_decode_all_hist = plt.subplots(ncols=len(epochs_to_analyze),\
                                                     nrows=2,figsize=(8,5))
        min_decode_prob = 1
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for t_i in range(num_tastes):
                decode_prob_data = decode_prob_by_epoch[e_ind][t_i,:]
                if np.nanmin(decode_prob_data) < min_decode_prob:
                    min_decode_prob = np.nanmin(decode_prob_data)
                
                ax_decode_all_hist[0,e_ind].hist(decode_prob_data, \
                                           histtype='step',cumulative = True, \
                                           density=True,color=taste_colors[t_i,:], \
                                           label=dig_in_names[t_i])
                ax_decode_all_hist[1,t_i].hist(decode_prob_data, \
                                           histtype='step',cumulative = True, \
                                           density=True,color=epoch_colors[e_i,:], \
                                           label='Epoch ' + str(e_i))
            ax_decode_all_hist[0,e_ind].set_xlabel('Decode Probability')
            ax_decode_all_hist[0,e_ind].set_ylabel('Cumulative Density')
            ax_decode_all_hist[0,e_ind].set_title('Epoch ' + str(e_i))
            if e_ind == 0:
                ax_decode_all_hist[0,e_ind].legend()
        #for e_ind, e_i in enumerate(epochs_to_analyze):
            #ax_decode_all_hist[0,e_ind].set_xlim([min_decode_prob - 0.1,1.1])
        for t_i in range(num_tastes):
            ax_decode_all_hist[1,t_i].set_xlabel('Decode Probability')
            ax_decode_all_hist[1,t_i].set_ylabel('Cumulative Density')
            ax_decode_all_hist[1,t_i].set_title(dig_in_names[t_i])
            #ax_decode_all_hist[1,t_i].set_xlim([min_decode_prob - 0.1,1.1])
            if t_i == 0:
                ax_decode_all_hist[1,t_i].legend()
        plt.suptitle('All Decode Probabilities in ' + segment_names[s_i])
        plt.tight_layout()
        f_decode_all_hist.savefig(os.path.join(dev_decode_save_dir,segment_names[s_i] + '_all_decode_prob_hist.png'))
        f_decode_all_hist.savefig(os.path.join(dev_decode_save_dir,segment_names[s_i] + '_all_decode_prob_hist.svg'))
        plt.close(f_decode_all_hist)
        
        #Plot the fractions of decoded taste by epoch
        f_decode_frac = plt.figure(figsize=(5,5))
        x_ticks = np.arange(len(epochs_to_analyze))
        decode_taste_counts_by_epoch_array = np.array(decode_taste_counts_by_epoch)
        decode_taste_fracs_by_epoch_array = decode_taste_counts_by_epoch_array/np.sum(decode_taste_counts_by_epoch_array,1)
        for t_i in range(num_tastes):
            plt.plot(x_ticks,decode_taste_fracs_by_epoch_array[:,t_i].T, \
                     label=dig_in_names[t_i], color=taste_colors[t_i,:])
            for e_i in range(len(epochs_to_analyze)):
                decode_val = decode_taste_fracs_by_epoch_array[e_i,t_i]
                plt.text(e_i,decode_val, str(np.round(decode_val,2)))
        plt.legend()
        plt.xticks(x_ticks,epochs_to_analyze)
        plt.xlabel('Epoch')
        plt.ylabel('Fraction of Deviations')
        plt.title('Dev Decode Fractions in ' + segment_names[s_i])
        f_decode_frac.savefig(os.path.join(dev_decode_save_dir,segment_names[s_i] + '_decode_fracs.png'))
        f_decode_frac.savefig(os.path.join(dev_decode_save_dir,segment_names[s_i] + '_decode_fracs.svg'))
        plt.close(f_decode_frac)
            

def plot_is_taste_which_taste_decoded(num_tastes, num_neur, segment_spike_times, 
                tastant_spike_times, start_dig_in_times, post_taste_dt, pre_taste_dt,
                cp_raster_inds, z_bin_dt, dig_in_names, segment_times,
                segment_names, save_dir, segment_dev_times, segment_dev_fr_vecs, 
                segment_dev_fr_vecs_zscore, neuron_count_thresh, e_len_dt,
                epochs_to_analyze=[], segments_to_analyze=[],
                decode_prob_cutoff=0.95):
    """Function to plot the deviation events with a buffer on either side with
    the decoding results"""
    
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    num_segments = len(segment_spike_times)
    cmap = colormaps['gist_rainbow']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    cmap = colormaps['seismic']
    is_taste_colors = cmap(np.linspace(0, 1, 2))
    
    epoch_seg_taste_times = np.zeros((num_segments, num_tastes))
    epoch_seg_lengths = np.zeros((num_segments, num_tastes))
    half_bin_z_dt = np.floor(z_bin_dt/2).astype('int')
    half_bin_decode_dt = np.floor(e_len_dt/2).astype('int')
    dev_buffer = 50 #ms
    num_to_plot = 50 #100
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
    
    cmap = colormaps['plasma']
    segment_colors = cmap(np.linspace(0, 1, len(segments_to_analyze)))

    # Get taste segment z-score info
    s_i_taste = np.nan*np.ones(1)
    for s_i in range(len(segment_names)):
        if segment_names[s_i].lower() == 'taste':
            s_i_taste[0] = s_i
    if not np.isnan(s_i_taste[0]):
        s_i = int(s_i_taste[0])
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = seg_end - seg_start
        time_bin_starts = np.arange(
            seg_start+half_bin_z_dt, seg_end-half_bin_z_dt, half_bin_z_dt*2)
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = np.array(
                segment_spike_times_s_i[n_i] - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        tb_fr = np.zeros((num_neur, len(time_bin_starts)))
        for tb_i, tb in enumerate(tqdm.tqdm(time_bin_starts)):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_bin_z_dt:tb+half_bin_z_dt-seg_start], 1)/(2*half_bin_z_dt*(1/1000))
        mean_fr_taste = np.mean(tb_fr, 1)
        std_fr_taste = np.std(tb_fr, 1)
        std_fr_taste[std_fr_taste == 0] = 1  # to avoid nan calculations
    else:
        mean_fr_taste = np.zeros(num_neur)
        std_fr_taste = np.ones(num_neur)
        
    # Get taste response firing rate vectors
    all_taste_fr_vecs = []
    all_taste_fr_vecs_z = []
    all_taste_fr_vecs_mean = []
    all_taste_fr_vecs_mean_z = []
    
    # Grab taste firing rate vectors
    for t_i in range(num_tastes):
        # Import taste spike and cp times
        taste_spike_times = tastant_spike_times[t_i]
        taste_deliv_times = start_dig_in_times[t_i]
        max_num_deliv = len(taste_deliv_times)
        cp_times = cp_raster_inds[t_i]

        # Store as binary spike arrays
        by_epoch_all_taste_fr_vecs = []
        by_epoch_all_taste_fr_vecs_z = []
        by_epoch_taste_fr_vecs_mean = np.zeros((num_cp, num_neur))
        by_epoch_taste_fr_vecs_mean_z = np.zeros((num_cp, num_neur))
        for e_i in epochs_to_analyze:
            taste_spike_times_bin = np.zeros(
                (max_num_deliv, num_neur, post_taste_dt))  # Taste response spike times
            taste_cp_times = np.zeros(
                (max_num_deliv, num_cp+1)).astype('int')
            taste_epoch_fr_vecs = np.zeros(
                (max_num_deliv, num_neur))  # original firing rate vecs
            taste_epoch_fr_vecs_z = np.zeros(
                (max_num_deliv, num_neur))  # z-scored firing rate vecs
            # store each delivery to binary spike matrix
            for d_i in range(len(taste_spike_times)):
                pre_taste_spike_times_bin = np.zeros(
                    (num_neur, pre_taste_dt))  # Pre-taste spike times
                taste_deliv_i = taste_deliv_times[d_i]
                for n_i in range(num_neur):
                    spikes_deliv_i = taste_spike_times[d_i][n_i]
                    if t_i == num_tastes-1:
                        if len(taste_spike_times[d_i][n_i]) > 0:
                            d_i_spikes = np.array(
                                spikes_deliv_i - (np.min(spikes_deliv_i)+pre_taste_dt)).astype('int')
                        else:
                            d_i_spikes = np.empty(0)
                    else:
                        d_i_spikes = np.array(
                            spikes_deliv_i - taste_deliv_i).astype('int')
                    d_i_spikes_posttaste = d_i_spikes[(
                        d_i_spikes < post_taste_dt)*(d_i_spikes >= 0)]
                    d_i_spikes_pretaste = d_i_spikes[d_i_spikes <
                                                     0] + pre_taste_dt
                    if len(d_i_spikes_posttaste) > 0:
                        taste_spike_times_bin[d_i,
                                              n_i, d_i_spikes_posttaste] = 1
                    if len(d_i_spikes_pretaste) > 0:
                        pre_taste_spike_times_bin[n_i,
                                                  d_i_spikes_pretaste] = 1
                taste_cp_times[d_i, :] = np.concatenate(
                    (np.zeros(1), np.cumsum(np.diff(cp_times[d_i, :])))).astype('int')
                # Calculate the FR vectors by epoch for each taste response and the average FR vector
                
                epoch_len_i = (taste_cp_times[d_i, e_i+1]-taste_cp_times[d_i, e_i])/1000
                if epoch_len_i == 0:
                    taste_epoch_fr_vecs[d_i,
                                        :] = np.zeros(num_neur)
                else:
                    taste_epoch_fr_vecs[d_i, :] = np.sum(
                        taste_spike_times_bin[d_i, :, taste_cp_times[d_i, e_i]:taste_cp_times[d_i, e_i+1]], 1)/epoch_len_i  # FR in HZ
                # Calculate z-scored FR vector
                taste_epoch_fr_vecs_z[d_i, :] = (
                    taste_epoch_fr_vecs[d_i, :].flatten() - mean_fr_taste)/std_fr_taste

            by_epoch_all_taste_fr_vecs.append(taste_epoch_fr_vecs)
            by_epoch_all_taste_fr_vecs_z.append(taste_epoch_fr_vecs_z)
            # Calculate average taste fr vec
            taste_fr_vecs_mean = np.nanmean(taste_epoch_fr_vecs, 0)
            taste_fr_vecs_z_mean = np.nanmean(taste_epoch_fr_vecs_z, 0)
            by_epoch_taste_fr_vecs_mean[e_i, :] = taste_fr_vecs_mean
            by_epoch_taste_fr_vecs_mean_z[e_i, :] = taste_fr_vecs_z_mean
            #taste_fr_vecs_max_hz = np.max(taste_epoch_fr_vecs)
        all_taste_fr_vecs.append(by_epoch_all_taste_fr_vecs)
        all_taste_fr_vecs_z.append(by_epoch_all_taste_fr_vecs_z)
        all_taste_fr_vecs_mean.append(by_epoch_taste_fr_vecs_mean)
        all_taste_fr_vecs_mean_z.append(by_epoch_taste_fr_vecs_mean_z)
        
    dev_decode_stats = dict()
    decode_save_dir = os.path.join(save_dir,'Is_Taste_Which_Taste')
    
    print('\t\t\t\tProgressing through all ' +
          str(len(segments_to_analyze)) + ' segments.')
    
    for seg_ind, s_i in tqdm.tqdm(enumerate(segments_to_analyze)):
        try:
            decode_is_taste_prob_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_deviations_is_taste.npy'))
            pre_dev_decode_is_taste_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_pre_deviations_is_taste.npy'))
            post_dev_decode_is_taste_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_post_deviations_is_taste.npy'))
            decode_which_taste_prob_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_deviations_which_taste.npy'))
            pre_dev_decode_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_pre_deviations_which_taste.npy'))
            post_dev_decode_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_post_deviations_which_taste.npy')) 
            decode_epoch_prob_array = np.load(
                os.path.join(decode_save_dir,'segment_' + str(s_i) + \
                             '_deviations_which_epoch.npy')) 
        except:
            print("\t\t\t\tSegment " + str(s_i) + " Never Decoded")
            pass
        
        seg_decode_save_dir = os.path.join(decode_save_dir,
            'segment_' + str(s_i) + '/')
        
        num_dev, _ = np.shape(decode_is_taste_prob_array)
        
        #Dev decodes
        dev_decode_is_taste_ind = np.argmax(decode_is_taste_prob_array, 1)
        dev_decode_taste_ind = np.argmax(decode_which_taste_prob_array, 1)
        dev_decode_is_taste_bin = np.zeros((2,num_dev)) #First row is taste, second row not taste
        for t_i in range(2):
            dev_decode_is_taste_bin[t_i,np.where(dev_decode_is_taste_ind == t_i)[0]] = 1
        is_taste_inds = np.where(dev_decode_is_taste_ind == 0)[0]
        dev_decode_bin = np.zeros((num_tastes-1,num_dev))
        for t_i in range(num_tastes-1):
            taste_inds = np.where(dev_decode_taste_ind == t_i)[0]
            true_taste_inds = np.intersect1d(is_taste_inds, taste_inds)
            dev_decode_bin[t_i,true_taste_inds] = 1
        dev_decode_array_prob_fixed = decode_which_taste_prob_array.T*dev_decode_bin
        #Pre-Dev decodes
        pre_dev_decode_is_taste_ind = np.argmax(pre_dev_decode_is_taste_array, 1)
        pre_dev_decode_taste_ind = np.argmax(pre_dev_decode_array, 1)
        pre_dev_decode_is_taste_bin = np.zeros((2,num_dev)) #First row is taste, second row not taste
        is_taste_inds = np.where(pre_dev_decode_is_taste_ind == 0)[0]
        pre_dev_decode_is_taste_bin[:,is_taste_inds] = 1
        pre_dev_decode_bin = np.zeros((num_tastes-1,num_dev))
        for t_i in range(num_tastes-1):
            taste_inds = np.where(pre_dev_decode_taste_ind == t_i)[0]
            true_taste_inds = np.intersect1d(is_taste_inds, taste_inds)
            pre_dev_decode_bin[t_i,true_taste_inds] = 1
        pre_dev_decode_array_prob_fixed = pre_dev_decode_array.T*pre_dev_decode_bin
        #Post-Dev decodes
        post_dev_decode_is_taste_ind = np.argmax(post_dev_decode_is_taste_array, 1)
        post_dev_decode_taste_ind = np.argmax(post_dev_decode_array, 1)
        post_dev_decode_is_taste_bin = np.zeros((2,num_dev)) #First row is taste, second row not taste
        is_taste_inds = np.where(post_dev_decode_is_taste_ind == 0)[0]
        post_dev_decode_is_taste_bin[:,is_taste_inds] = 1
        post_dev_decode_bin = np.zeros((num_tastes-1,num_dev))
        for t_i in range(num_tastes-1):
            is_taste_inds = np.where(post_dev_decode_is_taste_ind == 0)[0]
            taste_inds = np.where(post_dev_decode_taste_ind == t_i)[0]
            true_taste_inds = np.intersect1d(is_taste_inds, taste_inds)
            post_dev_decode_bin[t_i,true_taste_inds] = 1
        post_dev_decode_array_prob_fixed = post_dev_decode_array.T*post_dev_decode_bin
        
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = seg_end - seg_start  # in dt = ms

        # Pull binary spikes for segment
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = np.array(
                segment_spike_times_s_i[n_i] - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1

        # Z-score the segment
        time_bin_starts = np.arange(
            seg_start+half_bin_z_dt, seg_end-half_bin_z_dt, half_bin_z_dt*2)
        tb_fr = np.zeros((num_neur, len(time_bin_starts)))
        for tb_i, tb in enumerate(time_bin_starts):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_bin_z_dt:tb+half_bin_z_dt-seg_start], 1)/(2*half_bin_z_dt*(1/1000))
        mean_fr = np.mean(tb_fr, 1)
        std_fr = np.std(tb_fr, 1)
        std_fr[std_fr == 0] = 1
        
        #For each deviation event create plots of decoded results
        dev_to_plot = np.where(dev_decode_is_taste_ind == 0)[0]
        plot_inds = np.sort(np.array(random.sample(list(dev_to_plot),np.min([num_to_plot,len(dev_to_plot)]))))
        
        for dev_i in plot_inds:
            #Collect deviation decoding probabilities
            # dev_prob_is_taste_decoded = dev_decode_is_taste_array[:,dev_i]
            # pre_dev_prob_is_taste_decoded = pre_dev_decode_is_taste_array[:,dev_i]
            # post_dev_prob_is_taste_decoded = post_dev_decode_is_taste_array[:,dev_i]
            dev_prob_decoded = dev_decode_array_prob_fixed[:,dev_i]
            pre_dev_prob_decoded = pre_dev_decode_array_prob_fixed[:,dev_i]
            post_dev_prob_decoded = post_dev_decode_array_prob_fixed[:,dev_i]
            decoded_taste_ind = np.argmax(dev_prob_decoded)
            
            #Collect deviation times
            dev_times = segment_dev_times[seg_ind][:,dev_i]
            dev_start_time = dev_times[0]
            dev_end_time = dev_times[1]
            dev_len = dev_end_time - dev_start_time
            pre_dev_time = np.max([dev_start_time - dev_buffer,0])
            post_dev_time = np.min([dev_end_time + dev_buffer,seg_len])
            decode_x_vals = np.arange(pre_dev_time,post_dev_time) - dev_start_time
            d_plot_len = len(decode_x_vals)
            decode_x_ticks = np.linspace(decode_x_vals[0], decode_x_vals[-1]-1, 10).astype('int')
            raster_x_ticks = np.linspace(0,d_plot_len-1,10).astype('int')
            raster_x_tick_labels = decode_x_ticks #decode_x_vals[raster_x_ticks]
            
            #Collect spike times for raster plot and firing rate plots
            plot_spike_bin = segment_spike_times_s_i_bin[:,pre_dev_time:post_dev_time]
            plot_spike_times = [np.where(plot_spike_bin[n_i,:] == 1)[0] for n_i in range(num_neur)]
            firing_rate_vec = np.zeros(d_plot_len)
            fr_vec_x = np.arange(pre_dev_time, post_dev_time)
            for dpt_ind, dpt_i in enumerate(fr_vec_x):
                min_t_i = np.max([dpt_i-10,0])
                max_t_i = np.min([dpt_i + 10,seg_len])
                spikes_i = segment_spike_times_s_i_bin[:,min_t_i:max_t_i]
                firing_rate_vec[dpt_ind] = np.sum(spikes_i)/((max_t_i-min_t_i)/1000)/num_neur
            
            #Collect deviation FR vals
            dev_fr_vec = segment_dev_fr_vecs[seg_ind][dev_i]
            dev_fr_vec_zscore = segment_dev_fr_vecs_zscore[seg_ind][dev_i]
            
            #Reshape decoding probabilities for plotting
            decoding_array = np.zeros((num_tastes-1,d_plot_len))
            decoding_array[:,0:dev_buffer] = np.expand_dims(pre_dev_prob_decoded,1)*np.ones((num_tastes-1,dev_buffer))
            decoding_array[:,dev_buffer:dev_buffer + dev_len] = np.expand_dims(dev_prob_decoded,1)*np.ones((num_tastes-1,dev_len))
            decoding_array[:,-dev_buffer:] = np.expand_dims(post_dev_prob_decoded,1)*np.ones((num_tastes-1,dev_buffer))
            
            #Save to decoding counters
            epoch_seg_taste_times[s_i, decoded_taste_ind] += 1
            epoch_seg_lengths[s_i, decoded_taste_ind] += dev_len
            
            #Calculate correlation to mean taste responses
            corr_dev_event = np.array([pearsonr(all_taste_fr_vecs_mean[t_i][e_i, :], dev_fr_vec)[
                                         0] for t_i in range(num_tastes)])
            corr_dev_event_zscore = np.array([pearsonr(all_taste_fr_vecs_mean_z[t_i][e_i, :], dev_fr_vec_zscore)[
                                         0] for t_i in range(num_tastes)])
            
            if len(np.where(plot_inds == dev_i)[0])>0: 
                corr_title_norm = [dig_in_names[t_i] + ' corr = ' + str(
                    np.round(corr_dev_event[t_i], 2)) for t_i in range(num_tastes)]
                corr_title_z = [dig_in_names[t_i] + ' z-corr = ' + str(
                    np.round(corr_dev_event_zscore[t_i], 2)) for t_i in range(num_tastes)]
                corr_title = (', ').join(
                    corr_title_norm) + '\n' + (', ').join(corr_title_z)
                
                #Start Figure
                f, ax = plt.subplots(nrows=5, ncols=2, figsize=(
                    10, 10), gridspec_kw=dict(height_ratios=[1, 1, 1, 2, 2]))
                gs = ax[0, 0].get_gridspec()
                # Decoding probabilities
                ax[0, 0].remove()
                ax[1, 0].remove()
                axbig = f.add_subplot(gs[0:2, 0])
                leg_handles = []
                for t_i_2 in range(num_tastes-1):
                    taste_decode_prob_y = decoding_array[t_i_2,:]
                    p_h, = axbig.plot(
                        decode_x_vals, decoding_array[t_i_2,:], color=taste_colors[t_i_2, :])
                    leg_handles.append(p_h)
                    taste_decode_prob_y[0] = 0
                    taste_decode_prob_y[-1] = 0
                    high_prob_binary = np.zeros(len(decode_x_vals))
                    high_prob_times = np.where(
                        taste_decode_prob_y >= decode_prob_cutoff)[0]
                    high_prob_binary[high_prob_times] = 1
                    high_prob_starts = np.where(
                        np.diff(high_prob_binary) == 1)[0] + 1
                    high_prob_ends = np.where(
                        np.diff(high_prob_binary) == -1)[0] + 1
                    if len(high_prob_starts) > 0:
                        for hp_i in range(len(high_prob_starts)):
                            axbig.fill_between(decode_x_vals[high_prob_starts[hp_i]:high_prob_ends[hp_i]], taste_decode_prob_y[
                                               high_prob_starts[hp_i]:high_prob_ends[hp_i]], alpha=0.2, color=taste_colors[t_i_2, :])
                axbig.axvline(0, color='k', alpha=0.5)
                axbig.axvline(dev_len, color='k', alpha=0.5)
                axbig.legend(leg_handles, dig_in_names, loc='right')
                axbig.set_xticks(decode_x_ticks)
                axbig.set_xlim([decode_x_vals[0], decode_x_vals[-1]])
                axbig.set_ylabel('Decoding Fraction')
                axbig.set_xlabel('Time from Deviation (ms)')
                axbig.set_title('Event ' + str(dev_i) + '\nStart Time = ' + str(round(
                    dev_start_time/1000/60, 3)) + ' Minutes' + '\nEvent Length = ' + \
                        str(np.round(dev_len, 2)))
                # Decoded raster
                ax[0, 1].eventplot(plot_spike_times)
                ax[0, 1].set_xlim([0, d_plot_len])
                ax[0, 1].set_xticks(raster_x_ticks,labels=raster_x_tick_labels)
                ax[0, 1].axvline(dev_buffer, color='k', alpha=0.5)
                ax[0, 1].axvline(dev_buffer + dev_len, color='k', alpha=0.5)
                ax[0, 1].set_ylabel('Neuron Index')
                ax[0, 1].set_title('Event Spike Raster')
                ax[1, 0].axis('off')
                # Plot population firing rates w 20ms smoothing
                ax[1, 1].plot(decode_x_vals, firing_rate_vec)
                ax[1, 1].set_xlim([decode_x_vals[0], decode_x_vals[-1]])
                ax[1, 1].set_xticks(decode_x_ticks)
                ax[1, 1].axvline(0, color='k', alpha=0.5)
                ax[1, 1].axvline(dev_len, color='k', alpha=0.5)
                ax[1, 1].set_title('Population Avg FR (20 ms smooth)')
                ax[1, 1].set_ylabel('FR (Hz)')
                ax[1, 1].set_xlabel('Time from Deviation (ms)')
                # Decoded Firing Rates
                img = ax[2, 0].imshow(np.expand_dims(dev_fr_vec, 0), vmin=0, vmax=60)
                ax[2, 0].set_xlabel('Neuron Index')
                ax[2, 0].set_yticks(ticks=[])
                #plt.colorbar(img, location='bottom',orientation='horizontal',label='Firing Rate (Hz)',panchor=(0.9,0.5),ax=ax[2,0])
                ax[2, 0].set_title('Event FR')
                # Decoded Firing Rates Z-Scored
                img = ax[2, 1].imshow(np.expand_dims(
                    dev_fr_vec_zscore, 0), vmin=-3, vmax=3, cmap='bwr')
                ax[2, 1].set_xlabel('Neuron Index')
                ax[2, 1].set_yticks(ticks=[])
                #plt.colorbar(img, ax=ax[2,1], location='bottom',orientation='horizontal',label='Z-Scored Firing Rate (Hz)',panchor=(0.9,0.5))
                ax[2, 1].set_title('Event FR Z-Scored')
                # Taste Firing Rates
                # vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
                img = ax[3, 0].imshow(np.expand_dims(
                    all_taste_fr_vecs_mean[decoded_taste_ind][e_i, :], 0), vmin=0, vmax=60)
                ax[3, 0].set_xlabel('Neuron Index')
                ax[3, 0].set_yticks(ticks=[])
                plt.colorbar(
                    img, ax=ax[3, 0], location='bottom', orientation='horizontal', label='Firing Rate (Hz)', panchor=(0.9, 0.5))
                ax[3, 0].set_title('Avg. Taste Resp. FR')
                # Taste Firing Rates Z-Scored
                img = ax[3, 1].imshow(np.expand_dims(
                    all_taste_fr_vecs_mean_z[decoded_taste_ind][e_i, :], 0), vmin=-3, vmax=3, cmap='bwr')
                ax[3, 1].set_xlabel('Neuron Index')
                ax[3, 1].set_yticks(ticks=[])
                plt.colorbar(img, ax=ax[3, 1], location='bottom', orientation='horizontal',
                             label='Z-Scored Firing Rate (Hz)', panchor=(0.9, 0.5))
                ax[3, 1].set_title('Avg. Taste Resp. FR Z-Scored')
                # Decoded Firing Rates x Average Firing Rates
                max_lim = np.max([np.max(dev_fr_vec), np.max(all_taste_fr_vecs_mean[decoded_taste_ind][e_i, :])])
                ax[4, 0].plot([0, max_lim], [0, max_lim],
                              alpha=0.5, linestyle='dashed')
                ax[4, 0].scatter(all_taste_fr_vecs_mean[decoded_taste_ind][e_i, :], dev_fr_vec)
                ax[4, 0].set_xlabel('Average Taste FR')
                ax[4, 0].set_ylabel('Decoded Taste FR')
                ax[4, 0].set_title('Firing Rate Similarity')
                # Z-Scored Decoded Firing Rates x Z-Scored Average Firing Rates
                ax[4, 1].plot([-5, 5], [-5, 5], alpha=0.5, linestyle='dashed', color='k')
                ax[4, 1].scatter(
                    all_taste_fr_vecs_mean_z[decoded_taste_ind][e_i, :], dev_fr_vec_zscore)
                ax[4, 1].set_xlabel(
                    'Average Taste Neuron FR Std > Mean')
                ax[4, 1].set_ylabel('Event Neuron FR Std > Mean')
                ax[4, 1].set_title(
                    'Z-Scored Firing Rate Similarity')
                plt.suptitle(corr_title, wrap=True)
                plt.tight_layout()
                # Save Figure
                f.savefig(os.path.join(
                    seg_decode_save_dir, 'dev_' + str(dev_i) + '.png'))
                f.savefig(os.path.join(
                    seg_decode_save_dir, 'dev_' + str(dev_i) + '.svg'))
                plt.close(f)
    
    
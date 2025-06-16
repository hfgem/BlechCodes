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

def plot_decode_trends(num_groups, grouped_train_names, segments_to_analyze,
                       seg_decode_frac, seg_decode_counts, seg_names, decode_dir):
    """
    This function generates plots of decoding rates

    Parameters
    ----------
    num_groups : int
        Number of decode groups in dataset.
    grouped_train_names : list
        List of strings of the group names.
    segments_to_analyze : list
        List of segment indices to analyze.
    seg_decode_frac : numpy array
        Array of size len(segments_to_analyze) x num_groups containing fraction
        of decodes of each group within the segment.
    seg_decode_counts : numpy array
        Array of size len(segments_to_analyze) x num_groups containing fraction
        of decodes of each group within the segment.
    seg_names : list
        List of segment names.
    decode_dir : string
        Directory to save results.

    Returns
    -------
    Plots stored to directory 'decode_dir'.

    """
    #Plot decode fraction trends
    f_trends, ax_trends = plt.subplots(nrows = 1, ncols = num_groups, \
                                       sharex = True, figsize = (num_groups*4,4))
    for g_ind, g_name in enumerate(grouped_train_names):
        ax_trends[g_ind].plot(np.arange(len(segments_to_analyze)),seg_decode_frac[:,g_ind])
        ax_trends[g_ind].set_xticks(np.arange(len(segments_to_analyze)),seg_names)
        ax_trends[g_ind].set_xlabel('Segment')
        ax_trends[g_ind].set_ylabel('Decode Fraction')
        ax_trends[g_ind].set_title(g_name)
    plt.suptitle('Decoding Fraction Across Segments')
    plt.tight_layout()
    f_trends.savefig(os.path.join(decode_dir,'seg_group_frac_trends.png'))
    f_trends.savefig(os.path.join(decode_dir,'seg_group_frac_trends.svg'))
    plt.close(f_trends)
    
    f_trends_combined = plt.figure(figsize=(5,5))
    for g_ind, g_name in enumerate(grouped_train_names):
        plt.plot(np.arange(len(segments_to_analyze)),seg_decode_frac[:,g_ind],\
                 label=g_name)
    plt.legend(loc='upper left')
    plt.xticks(np.arange(len(segments_to_analyze)),seg_names)
    plt.xlabel('Segment')
    plt.ylabel('Decode Fraction')
    plt.suptitle('Decoding Fraction Across Segments')
    plt.tight_layout()
    f_trends_combined.savefig(os.path.join(decode_dir,'seg_group_frac_trends_combined.png'))
    f_trends_combined.savefig(os.path.join(decode_dir,'seg_group_frac_trends_combined.svg'))
    plt.close(f_trends_combined)
    
    f_trends_pie, ax_trends_pie = plt.subplots(nrows = 1, ncols = len(segments_to_analyze), \
                                       figsize = (len(segments_to_analyze)*4,4))
    for s_ind, s_i in enumerate(segments_to_analyze):
        pie_labels = [g_name + '\n' + str(np.round(100*seg_decode_frac[s_ind,g_i],2)) + \
                      '%' for g_i, g_name in enumerate(grouped_train_names)]
        ax_trends_pie[s_ind].pie(seg_decode_frac[s_ind,:],labels=pie_labels)
        ax_trends_pie[s_ind].set_title(seg_names[s_ind])
    plt.suptitle('Decode Fractions by Segment')
    plt.tight_layout()
    f_trends_pie.savefig(os.path.join(decode_dir,'seg_group_frac_trends_pie.png'))
    f_trends_pie.savefig(os.path.join(decode_dir,'seg_group_frac_trends_pie.svg'))
    plt.close(f_trends_pie)
    
    #Plot true taste decode fraction trends
    seg_true_decode_fracs = seg_decode_counts[:,:-2]/np.expand_dims(np.nansum(seg_decode_counts[:,:-2],1),1)
    f_trends, ax_trends = plt.subplots(nrows = 1, ncols = num_groups-2, \
                                       sharex = True, figsize = (num_groups*4,4))
    for g_ind, g_name in enumerate(grouped_train_names[:-2]):
        ax_trends[g_ind].plot(np.arange(len(segments_to_analyze)),seg_true_decode_fracs[:,g_ind])
        ax_trends[g_ind].set_xticks(np.arange(len(segments_to_analyze)),seg_names)
        ax_trends[g_ind].set_xlabel('Segment')
        ax_trends[g_ind].set_ylabel('Decode Fraction')
        ax_trends[g_ind].set_title(g_name)
    plt.suptitle('Decoding Fraction Across Segments')
    plt.tight_layout()
    f_trends.savefig(os.path.join(decode_dir,'seg_group_frac_trends_taste_decoded.png'))
    f_trends.savefig(os.path.join(decode_dir,'seg_group_frac_trends_taste_decoded.svg'))
    plt.close(f_trends)
    
    f_trends_combined = plt.figure(figsize=(5,5))
    for g_ind, g_name in enumerate(grouped_train_names[:-2]):
        plt.plot(np.arange(len(segments_to_analyze)),seg_true_decode_fracs[:,g_ind],\
                 label=g_name)
    plt.legend(loc='upper left')
    plt.xticks(np.arange(len(segments_to_analyze)),seg_names)
    plt.xlabel('Segment')
    plt.ylabel('Decode Fraction')
    plt.suptitle('Decoding Fraction Across Segments')
    plt.tight_layout()
    f_trends_combined.savefig(os.path.join(decode_dir,'seg_group_frac_trends_combined_taste_decoded.png'))
    f_trends_combined.savefig(os.path.join(decode_dir,'seg_group_frac_trends_combined_taste_decoded.svg'))
    plt.close(f_trends_combined)
    
    f_trends_pie, ax_trends_pie = plt.subplots(nrows = 1, ncols = len(segments_to_analyze), \
                                       figsize = (len(segments_to_analyze)*4,4))
    for s_ind, s_i in enumerate(segments_to_analyze):
        pie_labels = [g_name + '\n' + str(np.round(100*seg_true_decode_fracs[s_ind,g_i],2)) + \
                      '%' for g_i, g_name in enumerate(grouped_train_names[:-2])]
        if np.sum(seg_true_decode_fracs[s_ind,:]) >  0:
            ax_trends_pie[s_ind].pie(seg_true_decode_fracs[s_ind,:],labels=pie_labels)
            ax_trends_pie[s_ind].set_title(seg_names[s_ind])
    plt.title('Decode Fractions by Segment')
    plt.tight_layout()
    f_trends_pie.savefig(os.path.join(decode_dir,'seg_group_frac_trends_pie_taste_decoded.png'))
    f_trends_pie.savefig(os.path.join(decode_dir,'seg_group_frac_trends_pie_taste_decoded.svg'))
    plt.close(f_trends_pie)
    
    

def plot_decoded(tastant_fr_dist, tastant_spike_times, segment_spike_times, 
                 dig_in_names, segment_times, segment_names, start_dig_in_times, 
                 taste_num_deliv, segment_dev_times, segment_dev_fr_vecs, bin_dt, 
                 num_groups, grouped_train_names, grouped_train_data, 
                 non_none_tastes, decode_save_dir, 
                 z_score = False, epochs_to_analyze=[], segments_to_analyze=[]):
    """Function to plot the deviation events with a buffer on either side with
    the decoding results"""
    
    # Variables
    num_tastes = len(start_dig_in_times)
    num_neur = len(segment_spike_times[0])
    num_cp = len(tastant_fr_dist[0][0])
    num_segments = len(segment_spike_times)
    #p_taste = taste_num_deliv/np.sum(taste_num_deliv)  # P(taste)
    cmap = colormaps['gist_rainbow']
    group_colors = cmap(np.linspace(0, 1, num_groups))
    dev_buffer = 50
    half_bin_dt = np.floor(bin_dt/2).astype('int')
    
    # if len(epochs_to_analyze) == 0:
    #     epochs_to_analyze = np.arange(num_cp)
    epochs_to_analyze = np.array([0,1,2])
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)
        
    seg_names = list(np.array(segment_names)[segments_to_analyze])
    
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
            seg_start+half_bin_dt, seg_end-half_bin_dt, half_bin_dt*2)
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = np.array(
                segment_spike_times_s_i[n_i] - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        tb_fr = np.zeros((num_neur, len(time_bin_starts)))
        for tb_i, tb in enumerate(tqdm.tqdm(time_bin_starts)):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_bin_dt:tb+half_bin_dt-seg_start], 1)/(2*half_bin_dt*(1/1000))
        mean_fr_taste = np.mean(tb_fr, 1)
        std_fr_taste = np.std(tb_fr, 1)
        std_fr_taste[std_fr_taste == 0] = 1  # to avoid nan calculations
    else:
        mean_fr_taste = np.zeros(num_neur)
        std_fr_taste = np.ones(num_neur)
        
    # Get taste response firing rate vectors  
    all_group_fr_vecs_mean = []
    for g_i in range(num_groups):
        all_group_fr_vecs_mean.append(np.nanmean(np.array(grouped_train_data[g_i]),0))    
    
    dev_decode_stats = dict()
    for seg_ind, s_i in tqdm.tqdm(enumerate(segments_to_analyze)):
        dev_decode_stats[seg_ind] = dict()
        
        print('\t\t\tPlotting Decoding for Segment ' + str(s_i))
        seg_decode_save_dir = os.path.join(decode_save_dir,
            'segment_' + str(s_i) + '/')
        dev_decode_prob_array = np.load( os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                      '_deviation_decodes.npy'))
        pre_dev_decode_prob_array = np.load(
            os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                         '_pre_deviations_decodes.npy'))
        post_dev_decode_prob_array = np.load(
            os.path.join(seg_decode_save_dir,'segment_' + str(s_i) + \
                         '_post_deviations_decodes.npy'))

        #Create plot save dir
        seg_plot_save = os.path.join(seg_decode_save_dir,'indiv_events')
        if not os.path.isdir(seg_plot_save):
            os.mkdir(seg_plot_save)
        
        num_dev, _ = np.shape(dev_decode_prob_array)
        plot_inds = np.sort(random.sample(list(np.arange(num_dev)),50))
        
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = seg_end - seg_start  # in dt = ms

        # Pull binary spikes for segment
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in np.arange(num_neur):
            n_i_spike_times = np.array(
                segment_spike_times_s_i[n_i] - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1

        # Z-score the segment
        time_bin_starts = np.arange(
            seg_start+half_bin_dt, seg_end-half_bin_dt, half_bin_dt*2)
        tb_fr = np.zeros((num_neur, len(time_bin_starts)))
        for tb_i, tb in enumerate(time_bin_starts):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_bin_dt:tb+half_bin_dt-seg_start], 1)/(2*half_bin_dt*(1/1000))
        mean_fr = np.mean(tb_fr, 1)
        std_fr = np.std(tb_fr, 1)
        std_fr[std_fr == 0] = 1
        
        #Calculate the number of deviation events decoded for each group
        dev_decode_stats[seg_ind]['decode_probabilities'] = dev_decode_prob_array
        group_decoded = np.argmax(dev_decode_prob_array,1)
        group_decoded_bin = np.zeros((num_groups,num_dev))
        group_decoded_start_times = []
        group_decoded_end_times = []
        for g_i in range(num_groups):
            group_decoded_ind = np.where(group_decoded == g_i)[0]
            group_decoded_bin[g_i,group_decoded_ind] = 1
            group_decoded_start_times.append(list(segment_dev_times[seg_ind][0,group_decoded_ind]))
            group_decoded_end_times.append(list(segment_dev_times[seg_ind][1,group_decoded_ind]))
        dev_decode_stats[seg_ind]['num_dev_per_group'] = np.nansum(group_decoded_bin,1)
        
        #For each deviation event create plots of decoded results
        dev_group_corrs = np.zeros((num_groups,num_dev))
        for dev_i in range(num_dev):
            #Collect deviation decoding probabilities
            dev_prob_decoded = dev_decode_prob_array[dev_i,:]
            pre_dev_prob_decoded = pre_dev_decode_prob_array[dev_i,:]
            post_dev_prob_decoded = post_dev_decode_prob_array[dev_i,:]
            decoded_group_ind = group_decoded[dev_i]
            
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
            
            #Reshape decoding probabilities for plotting
            decoding_array = np.zeros((num_groups,d_plot_len))
            decoding_array[:,0:dev_buffer] = np.expand_dims(pre_dev_prob_decoded,1)*np.ones((num_groups,dev_buffer))
            decoding_array[:,dev_buffer:dev_buffer + dev_len] = np.expand_dims(dev_prob_decoded,1)*np.ones((num_groups,dev_len))
            decoding_array[:,-dev_buffer:] = np.expand_dims(post_dev_prob_decoded,1)*np.ones((num_groups,dev_buffer))
            
            #Calculate correlation to mean taste responses
            corr_dev_event = np.array([pearsonr(all_group_fr_vecs_mean[g_i], dev_fr_vec)[
                                         0] for g_i in range(num_groups)])
            dev_group_corrs[:,dev_i] = corr_dev_event
            
            if len(np.where(plot_inds == dev_i)[0])>0: 
                corr_title = [grouped_train_names[g_i] + ' corr = ' + str(
                    np.round(corr_dev_event[g_i], 2)) for g_i in range(num_groups)]
                
                #Start Figure
                f, ax = plt.subplots(nrows=5, ncols=2, figsize=(
                    10, 10), gridspec_kw=dict(height_ratios=[1, 1, 1, 2, 2]))
                gs = ax[0, 0].get_gridspec()
                # Decoding probabilities
                ax[0, 0].remove()
                ax[1, 0].remove()
                axbig = f.add_subplot(gs[0:2, 0])
                leg_handles = []
                for g_i in range(num_groups):
                    taste_decode_prob_y = decoding_array[g_i,:]
                    p_h, = axbig.plot(
                        decode_x_vals, decoding_array[g_i,:], color=group_colors[g_i, :])
                    leg_handles.append(p_h)
                    taste_decode_prob_y[0] = 0
                    taste_decode_prob_y[-1] = 0
                    
                axbig.axvline(0, color='k', alpha=0.5)
                axbig.axvline(dev_len, color='k', alpha=0.5)
                axbig.legend(leg_handles, grouped_train_names, loc='right')
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
                ax[2, 0].remove()
                ax[2, 1].remove()
                axfr = f.add_subplot(gs[2, 0:2])
                img = axfr.imshow(np.expand_dims(dev_fr_vec, 0))
                axfr.set_xlabel('Neuron Index')
                axfr.set_yticks(ticks=[])
                #plt.colorbar(img, location='bottom',orientation='horizontal',label='Firing Rate (Hz)',panchor=(0.9,0.5),ax=ax[2,0])
                axfr.set_title('Event FR')
                # Taste Firing Rates
                # vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
                ax[3, 0].remove()
                ax[3, 1].remove()
                axtfr = f.add_subplot(gs[3, 0:2])
                img = axtfr.imshow(np.expand_dims(
                    all_group_fr_vecs_mean[decoded_group_ind], 0))
                axtfr.set_xlabel('Neuron Index')
                axtfr.set_yticks(ticks=[])
                plt.colorbar(
                    img, ax=axtfr, location='bottom', orientation='horizontal', label='Firing Rate (Hz)', panchor=(0.9, 0.5))
                axtfr.set_title('Avg. Taste Resp. FR')
                # Decoded Firing Rates x Average Firing Rates
                max_lim = np.max([np.max(dev_fr_vec), np.max(all_group_fr_vecs_mean[decoded_group_ind])])
                ax[4, 0].plot([0, max_lim], [0, max_lim],
                              alpha=0.5, linestyle='dashed')
                ax[4, 0].scatter(all_group_fr_vecs_mean[decoded_group_ind], dev_fr_vec)
                ax[4, 0].set_xlabel('Average Group FR')
                ax[4, 0].set_ylabel('Dev FR')
                ax[4, 0].set_title('Firing Rate Similarity')
                ax[4, 1].remove()
                # Final Cleanup
                plt.suptitle(corr_title, wrap=True)
                plt.tight_layout()
                # Save Figure
                f.savefig(os.path.join(
                    seg_plot_save, 'dev_' + str(dev_i) + '.png'))
                f.savefig(os.path.join(
                    seg_plot_save, 'dev_' + str(dev_i) + '.svg'))
                plt.close(f)
        dev_decode_stats[seg_ind]['dev_group_corrs'] = dev_group_corrs
    np.save(os.path.join(decode_save_dir,'dev_decode_stats.npy'),dev_decode_stats)
        

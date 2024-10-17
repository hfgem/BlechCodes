#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:28:46 2024

@author: Hannah Germaine

This file is dedicated to plotting functions for deviation decoded replay events. 
"""

def plot_decoded(fr_dist, num_tastes, num_neur, segment_spike_times, tastant_spike_times,
                 start_dig_in_times, end_dig_in_times, post_taste_dt, pre_taste_dt,
                 cp_raster_inds, z_bin_dt, dig_in_names, segment_times,
                 segment_names, taste_num_deliv, taste_select_epoch,
                 save_dir, max_decode, max_hz, seg_stat_bin,
                 neuron_count_thresh, e_len_dt, trial_start_frac=0,
                 epochs_to_analyze=[], segments_to_analyze=[],
                 decode_prob_cutoff=0.95):
    """Function to plot the deviation events with a buffer on either side with
    the decoding results"""
    
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    num_segments = len(segment_spike_times)
    neur_cut = np.floor(num_neur*neuron_count_thresh).astype('int')
    taste_colors = cm.brg(np.linspace(0, 1, num_tastes))
    epoch_seg_taste_times = np.zeros((num_cp, num_segments, num_tastes))
    epoch_seg_taste_times_neur_cut = np.zeros(
        (num_cp, num_segments, num_tastes))
    epoch_seg_taste_times_best = np.zeros((num_cp, num_segments, num_tastes))
    epoch_seg_lengths = np.zeros((num_cp, num_segments, num_tastes))
    half_bin_z_dt = np.floor(z_bin_dt/2).astype('int')
    half_bin_decode_dt = np.floor(e_len_dt/2).astype('int')
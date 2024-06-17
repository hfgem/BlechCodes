#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:16:18 2022

@author: hannahgermaine

This code is written to perform template-matching
"""

import os
import numpy as np
from scipy.signal import find_peaks
import scipy.stats as ss
import matplotlib.pyplot as plt
from numba import jit

def spike_template_sort(all_spikes, sampling_rate, num_pts_left, num_pts_right,
                        cut_percentile, unit_dir, clust_ind):
    """This function performs template-matching to pull out potential spikes.
    INPUTS:
            - all_spikes
            - sampling_rate
            - num_pts_left
            - num_pts_right
            - cut_percentile
            - unit_dir - directory of unit's storage data
            - clust_ind
    OUTPUTS:
            - potential_spikes
            - good_ind
    """
    template_dir = unit_dir + 'template_matching/'
    if os.path.isdir(template_dir) == False:
        os.mkdir(template_dir)

    # Setting
    avg_peak = 0  # If = 1, will use the average between the 2 max peaks for a cutoff, if 0 will just use the percentile cutoff

    # Grab templates
    print("\t Preparing Data for Template Matching")
    num_spikes = len(all_spikes)
    peak_val = np.abs(all_spikes[:, num_pts_left])
    peak_val = np.expand_dims(peak_val, 1)
    norm_spikes = np.divide(all_spikes, peak_val)  # Normalize the data
    num_peaks = np.array([len(find_peaks(norm_spikes[s], 0.5)[
                         0]) + len(find_peaks(-1*norm_spikes[s], 0.5)[0]) for s in range(num_spikes)])
    remaining_ind = list(np.arange(num_spikes))
    # Grab templates of spikes
    spike_templates = generate_templates(
        sampling_rate, num_pts_left, num_pts_right)
    new_templates = np.zeros(np.shape(spike_templates))
    num_types = np.shape(spike_templates)[0]
    good_ind = []
    print("\t Performing Template Comparison.")
    for i in range(num_types):
        # Template distance scores
        spike_mat = np.multiply(
            np.ones(np.shape(norm_spikes[remaining_ind, :])), spike_templates[i, :])
        dist = np.sqrt(
            np.sum(np.square(np.subtract(norm_spikes[remaining_ind, :], spike_mat)), 1))
        num_peaks_i = num_peaks[remaining_ind]
        score = dist*num_peaks_i
        percentile = np.percentile(score, cut_percentile)
        # Calculate the first peak location and generate a new mean template
        hist_counts = np.histogram(score, 100)
        hist_peaks = find_peaks(hist_counts[0])
        first_peak_value = hist_counts[1][hist_peaks[0][0]]
        try:
            second_peak_value = hist_counts[1][hist_peaks[0][1]]
        except:
            second_peak_value = percentile
        halfway_value = (first_peak_value + second_peak_value)/2
        new_template_waveform_ind = list(
            np.array(remaining_ind)[list(np.where(score < halfway_value)[0])])
        new_templates[i, :] = np.mean(
            norm_spikes[new_template_waveform_ind, :], axis=0)
    # Plot a histogram of the scores and save to the template_matching dir
    fig = plt.figure(figsize=(20, 20))
    for i in range(num_types):
        # Calculate new template distance scores
        new_template = new_templates[i, :]
        num_peaks_i = num_peaks[remaining_ind]
        spike_mat_2 = np.multiply(
            np.ones(np.shape(norm_spikes[remaining_ind, :])), new_template)
        dist_2 = np.sqrt(
            np.sum(np.square(np.subtract(norm_spikes[remaining_ind, :], spike_mat_2)), 1))
        score_2 = dist_2*num_peaks_i
        percentile = np.percentile(score_2, cut_percentile)
        # Create subplot to plot histogram and percentile cutoff
        plt.subplot(2, num_types, i + 1)
        hist_counts = plt.hist(
            score_2, 150, label='Mean Template Similarity Scores')
        hist_peaks = find_peaks(hist_counts[0])
        hist_peak_vals = hist_counts[0][list(hist_peaks[0])]
        if avg_peak == 1:
            max_peak = hist_counts[1][hist_peaks[0][list(
                np.where(hist_peak_vals == np.sort(hist_peak_vals)[-1])[0])]]
            max_peak_2 = hist_counts[1][hist_peaks[0][list(
                np.where(hist_peak_vals == np.sort(hist_peak_vals)[-2])[0])]]
            halfway_value = (max_peak + max_peak_2)/2
            if len(halfway_value) > 1:
                halfway_value = halfway_value[0]
            if halfway_value < percentile:
                cut_val = halfway_value
            else:
                cut_val = percentile
        else:
            #cut_val = hist_counts[1][hist_peaks[0][0] + 1]
            cut_val = percentile
        plt.axvline(cut_val, color='r', linestyle='--',
                    label='Cutoff Threshold')
        plt.legend()
        plt.xlabel('Score = distance*peak_count')
        plt.ylabel('Number of occurrences')
        plt.title('Scores in comparison to template #' + str(i))
        plt.subplot(2, num_types, i + 1 + num_types)
        plt.plot(new_template)
        plt.title('Template #' + str(i))
        good_i = list(np.array(remaining_ind)[
                      list(np.where(score_2 < cut_val)[0])])
        good_ind.append(good_i)
        remaining_ind = list(np.setdiff1d(remaining_ind, good_i))
    fig.savefig(template_dir + 'template_matching_results_cluster' +
                str(clust_ind) + '.png', dpi=100)
    plt.close(fig)
    potential_spikes = [all_spikes[g_i] for g_i in good_ind]

#	#Plots for checking
#  	axis_labels = np.arange(-num_pts_left,num_pts_right)
#  	bad_ind = np.setdiff1d(np.arange(len(all_spikes)),good_ind)
#  	num_vis = 10
#  	samp_bad = [random.randint(0,len(bad_ind)) for i in range(num_vis)]
#  	samp_good = [random.randint(0,len(good_ind)) for i in range(num_vis)]
#  	fig = plt.figure(figsize=(10,10))
#  	for i in range(num_vis):
# 		 plt.subplot(num_vis,2,(2*i)+1)
# 		 plt.plot(axis_labels,norm_spikes[samp_good[i]])
# 		 plt.title('Good Example')
# 		 plt.subplot(num_vis,2,(2*i)+2)
# 		 plt.plot(axis_labels,norm_spikes[samp_bad[i]])
# 		 plt.title('Bad Example')
#  	plt.tight_layout()

    return potential_spikes, good_ind


@jit(forceobj=True)
def generate_templates(sampling_rate, num_pts_left, num_pts_right):
    """This function generates 3 template vectors of neurons with a peak 
    centered between num_pts_left and num_pts_right."""

    x_points = np.arange(-num_pts_left, num_pts_right)
    #templates = np.zeros((3,len(x_points)))
    templates = np.zeros((2, len(x_points)))

    fast_spike_width = sampling_rate*(1/1000)
    sd = fast_spike_width/20

    pos_spike = ss.norm.pdf(x_points, 0, sd)
    max_pos_spike = max(abs(pos_spike))
    pos_spike = pos_spike/max_pos_spike
    #fast_spike = -1*pos_spike
    reg_spike_bit = ss.gamma.pdf(np.arange(fast_spike_width-1), 5)
    peak_reg = find_peaks(reg_spike_bit)[0][0]
    reg_spike = np.concatenate(
        (np.zeros(num_pts_left-peak_reg), -1*reg_spike_bit), axis=0)
    reg_spike = np.concatenate(
        (reg_spike, np.zeros(len(pos_spike) - len(reg_spike))), axis=0)
    max_reg_spike = max(abs(reg_spike))
    reg_spike = reg_spike/max_reg_spike

    templates[0, :] = reg_spike
    templates[1, :] = pos_spike
    #templates[2,:] = fast_spike

    # fig = plt.figure()
    # plt.subplot(3,1,1)
    # plt.plot(x_points,pos_spike)
    # plt.title('Positive Spike')
    # plt.subplot(3,1,2)
    # plt.plot(x_points,fast_spike)
    # plt.title('Fast Spike')
    # plt.subplot(3,1,3)
    # plt.plot(x_points,reg_spike)
    # plt.title('Regular Spike')
    # plt.tight_layout()

    return templates

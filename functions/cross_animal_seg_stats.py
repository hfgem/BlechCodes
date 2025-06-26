#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:23:58 2025

@author: hannahgermaine
"""

import os
import csv
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
from itertools import combinations

def seg_stat_collection(all_data_dict):
    """

    Parameters
    ----------
    all_data_dict : dict
        Dictionary containing collected data by animal for cross-animal analysis.

    Returns
    -------
    unique_given_names : list
        List of unique animal names.
    unique_segment_names : list
        List of unique segment names across animals.
    pop_rates : dict
        Dictionary containing population rate calculations for each 
        animal and segment.
    isis : dict
        Dictionary containing ISI calculations for each animal, segment, 
        and neuron.
    cvs : dict
        Dictionary containing ISI calculations for each animal, segment, 
        and neuron.

    """
    unique_given_names = list(all_data_dict.keys())
    all_segment_names = []
    bin_size = 100 #ms
    skip_size = 25 #ms
    pop_rates = dict()
    isis = dict()
    cvs = dict()
    for gn in tqdm.tqdm(unique_given_names):
        pop_rates[gn] = dict()
        isis[gn] = dict()
        cvs[gn] = dict()
        data = all_data_dict[gn]['data']
        segment_names = data['segment_names']
        all_segment_names.extend(segment_names)
        segment_times = data['segment_times']
        spike_times = data['spike_times']
        num_neur = len(spike_times)
        num_seg = len(segment_times)
        for s_i, s_name in enumerate(segment_names):
            isis[gn][s_name] = dict()
            cvs[gn][s_name] = np.zeros(num_neur)
            #Segment times in ms
            s_start = segment_times[s_i]
            s_end = segment_times[s_i+1]
            s_len = s_end-s_start
            #Create spike raster
            spike_rast = np.zeros((num_neur,s_len))
            for n_i in range(num_neur):
                n_spikes = spike_times[n_i] #neuron spike times in ms
                n_seg_spike_times = (n_spikes[np.where((n_spikes < s_end)*(s_start <= n_spikes))[0]]).astype('int') - int(s_start)
                spike_rast[n_i,n_seg_spike_times] = 1
                #Calculate neuron firing stats
                n_isis = np.diff(n_seg_spike_times)
                isis[gn][s_name][n_i] = n_isis
                cvs[gn][s_name][n_i] = np.nanstd(n_isis)/np.nanmean(n_isis)
            #Calculate population rate
            spike_sum = np.nansum(spike_rast,0)
            pop_rate = np.array([np.sum(spike_sum[i:i+bin_size])/(bin_size/1000) for i in np.arange(0,s_len-bin_size,skip_size)])
            pop_rates[gn][s_name] = pop_rate

    all_segment_names = np.array(all_segment_names)
    unique_segment_indices = np.sort(
        np.unique(all_segment_names, return_index=True)[1])
    unique_segment_names = [all_segment_names[i]
                          for i in unique_segment_indices]
    
    return unique_given_names, unique_segment_names, pop_rates, isis, cvs

def seg_stat_analysis(unique_given_names, unique_segment_names, pop_rates, \
                      isis, cvs, seg_stat_save_dir):
    
    
    pop_rate_analysis_plots(unique_given_names,unique_segment_names,\
                                pop_rates,seg_stat_save_dir)
    
    
def pop_rate_analysis_plots(unique_given_names,unique_segment_names,\
                            pop_rates,seg_stat_save_dir):
    """
    Conduct significance tests and plot population firing rate coefficients
    of variation across animals and segments.

    Parameters
    ----------
    unique_given_names : list
        List of unique animal names.
    unique_segment_names : list
        List of unique segment names across animals.
    pop_rates : dict
        Dictionary containing population rate calculations for each 
        animal and segment.
    seg_stat_save_dir : str
        Save path.

    Returns
    -------
    None.
    
    Outputs
    -------
    Within animal population rate ANOVA results across segments to .csv file.
    Boxplot of across animal population rate CVs by segment.

    """
    num_anim = len(unique_given_names)
    cmap = colormaps['gist_rainbow']
    anim_colors = cmap(np.linspace(0, 1, num_anim))
    num_seg = len(unique_segment_names)
    
    #Collect pop rate significance and CV
    sig_table = [['Animal Name','ANOVA p', 'Mean Order']]
    seg_cv_table = np.zeros((num_anim,num_seg))
    for gn_i, gn in enumerate(unique_given_names):
        #Collect segment data for animal
        all_seg_data = []
        all_seg_means = []
        for s_i, seg_name in enumerate(unique_segment_names):
            data = np.array(pop_rates[gn][seg_name])
            nonnan_data = data[~np.isnan(data)]
            all_seg_data.append(nonnan_data)
            all_seg_means.append(np.mean(nonnan_data))
            seg_cv_table[gn_i,s_i] = np.std(nonnan_data)/np.mean(nonnan_data)
        #Test significant differences
        anim_sig = [gn]
        result = f_oneway(*(all_seg_data))
        anim_sig.append(result.pvalue)
        mean_ascend = np.argsort(all_seg_means)
        mean_text = unique_segment_names[mean_ascend[0]]
        for s_i in mean_ascend[1:]:
            mean_text += ' < ' + unique_segment_names[s_i]
        anim_sig.append(mean_text)
        sig_table.append(anim_sig)
    #Save significance results
    with open(os.path.join(seg_stat_save_dir,'seg_pop_rate_sig.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(sig_table)
        
    #Plot CV results
    f_cv = plt.figure(figsize=(5,5))
    plt.boxplot(seg_cv_table,labels=unique_segment_names)
    for s_i in range(num_seg):
        x_vals = s_i + 1 + np.random.randn(num_anim)/10
        plt.scatter(x_vals,seg_cv_table[:,s_i],color='g',alpha=0.3)
    seg_pairs = list(combinations(np.arange(num_seg),2))
    pairwise_ttest = np.zeros(len(seg_pairs))
    for s_i, (s_1, s_2) in enumerate(seg_pairs):
        result = mannwhitneyu(seg_cv_table[:,s_1],seg_cv_table[:,s_2])
        if result.pvalue <= 0.05:
            rand_jitter = np.random.randn(1)/10
            plt.plot([s_1+1,s_2+1],[1+rand_jitter,1+rand_jitter],\
                     color='k')
            plt.text(1+s_1+(s_2-s_1)/2,1+rand_jitter+0.05,'*')
    plt.title('Population Firing Rate CV')
    plt.tight_layout()
    f_cv.savefig(os.path.join(seg_stat_save_dir,'pop_rate_cv_boxplot.png'))
    f_cv.savefig(os.path.join(seg_stat_save_dir,'pop_rate_cv_boxplot.svg'))
    
    
def isi_analysis_plots(unique_given_names,unique_segment_names,\
                            isis,seg_stat_save_dir):
    """
    Conduct significance tests and plot population firing rate coefficients
    of variation across animals and segments.

    Parameters
    ----------
    unique_given_names : list
        List of unique animal names.
    unique_segment_names : list
        List of unique segment names across animals.
    isis : dict
        Dictionary containing inter-spike-interval calculations for each 
        animal, segment, and neuron.
    seg_stat_save_dir : str
        Save path.

    Returns
    -------
    None.
    
    Outputs
    -------
    Within animal population rate ANOVA results across segments to .csv file.
    Boxplot of across animal population rate CVs by segment.

    """
    num_anim = len(unique_given_names)
    cmap = colormaps['gist_rainbow']
    anim_colors = cmap(np.linspace(0, 1, num_anim))
    num_seg = len(unique_segment_names)
    
    #Collect isi significance and CV
    x_vals = [0,1,2,5,10,1000]
    num_x = len(x_vals)
    bar_w = 1/num_anim
    half_bar = bar_w/2
    labels = [str(x_vals[i]) + '-' + str(x_vals[i+1]) for i in range(num_x-2)]
    labels.extend([str(x_vals[-2]) + '+'])
    
    anim_cvs = []
    for gn_i, gn in enumerate(unique_given_names):
        num_neur = len(isis[gn][unique_segment_names[0]])
        neur_cv = np.zeros((num_neur,num_seg))
        for s_i, seg_name in enumerate(unique_segment_names):
            data = isis[gn][seg_name]
            for n_i in range(num_neur):
                n_data = data[n_i]/1000 #convert to seconds
                neur_cv[n_i,s_i] = np.nanstd(n_data)/np.nanmean(n_data)
        anim_cvs.append(neur_cv)
    
    f_cv, ax_cv = plt.subplots(ncols=num_seg,figsize=(15,5))
    for s_i, seg_name in enumerate(unique_segment_names):
        anim_seg_vals = []
        for gn_i, gn in enumerate(unique_given_names):
            anim_cv = anim_cvs[gn_i][:,s_i]
            anim_seg_vals.append(anim_cv)
            y_vals = [len(np.where((anim_cv < x_vals[i+1])*(x_vals[i]<=anim_cv))[0])/len(flat_cv) for i in range(num_x-1)]
            ax_cv[s_i].bar(np.arange(num_x-1)+gn_i*bar_w+half_bar,y_vals,\
                    color=anim_colors[gn_i,:],alpha=0.3,label=gn,\
                        width=bar_w)
        anova_result = f_oneway(*anim_seg_vals)
        if anova_result.pvalue <= 0.05:
            ax_cv[s_i].set_title(seg_name + ' CV (*ANOVA)')
        else:
            ax_cv[s_i].set_title(seg_name + ' CV (n.s. ANOVA)')
        ax_cv[s_i].set_xticks(np.arange(num_x)+0.5,labels=labels,rotation=45)
        for x_i in np.arange(num_x):
            ax_cv[s_i].axvline(x_i,color='k',linestyle='dashed',alpha=0.2)
    ax_cv[s_i].legend(loc='upper right')
    plt.suptitle('By-Segment Neuron CV Distributions')
    plt.tight_layout()
    f_cv.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_cv_boxplot.png'))
    f_cv.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_cv_boxplot.svg'))
    
    
    #Plot ISI results
    f_cv = plt.figure(figsize=(5,5))
    plt.boxplot(seg_cv_table,labels=unique_segment_names)
    for s_i in range(num_seg):
        x_vals = s_i + 1 + np.random.randn(num_anim)/10
        plt.scatter(x_vals,seg_cv_table[:,s_i],color='g',alpha=0.3)
    seg_pairs = list(combinations(np.arange(num_seg),2))
    pairwise_ttest = np.zeros(len(seg_pairs))
    for s_i, (s_1, s_2) in enumerate(seg_pairs):
        result = mannwhitneyu(seg_cv_table[:,s_1],seg_cv_table[:,s_2])
        if result.pvalue <= 0.05:
            rand_jitter = np.random.randn(1)/10
            plt.plot([s_1+1,s_2+1],[1+rand_jitter,1+rand_jitter],\
                     color='k')
            plt.text(1+s_1+(s_2-s_1)/2,1+rand_jitter+0.05,'*')
    plt.title('Population Firing Rate CV')
    plt.tight_layout()
    f_cv.savefig(os.path.join(seg_stat_save_dir,'pop_rate_cv_boxplot.png'))
    f_cv.savefig(os.path.join(seg_stat_save_dir,'pop_rate_cv_boxplot.svg'))
    
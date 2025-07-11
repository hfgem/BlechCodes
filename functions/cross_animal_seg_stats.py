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
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, ks_2samp
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
    neur_rates = dict()
    pop_rates = dict()
    isis = dict()
    cvs = dict()
    for gn in tqdm.tqdm(unique_given_names):
        neur_rates[gn] = dict()
        pop_rates[gn] = dict()
        isis[gn] = dict()
        cvs[gn] = dict()
        data = all_data_dict[gn]['data']
        segment_names = data['segment_names']
        all_segment_names.extend(segment_names)
        segment_times = data['segment_times']
        spike_times = data['spike_times']
        num_neur = len(spike_times)
        for s_i, s_name in enumerate(segment_names):
            neur_rates[gn][s_name] = dict()
            isis[gn][s_name] = dict()
            cvs[gn][s_name] = np.zeros(num_neur)
            #Segment times in ms
            s_start = segment_times[s_i]
            s_end = segment_times[s_i+1]
            s_len = s_end-s_start
            #Create spike raster
            spike_rast = np.zeros((num_neur,s_len))
            for n_i in range(num_neur):
                n_spikes = np.sort(spike_times[n_i]) #neuron spike times in ms
                n_seg_spike_times = (n_spikes[np.where((n_spikes < s_end)*(s_start <= n_spikes))[0]]).astype('int') - int(s_start)
                spike_rast[n_i,n_seg_spike_times] = 1
                #Calculate neuron binned firing rates
                neur_rates[gn][s_name][n_i] = np.array([np.sum(spike_rast[n_i,i:i+bin_size])/(bin_size/1000) for i in np.arange(0,s_len-bin_size,skip_size)])
                #Calculate neuron firing stats
                n_isis = np.diff(n_seg_spike_times)
                isi_zscore = (n_isis - np.nanmean(n_isis))/np.nanstd(n_isis)
                super_outliers = np.where(isi_zscore > 10)[0]
                # if len(super_outliers) > 0: #Plot the extreme outliers - neuron signal loss?
                #     plt.figure()    
                #     plt.plot(n_isis)
                #     plt.title(gn + ' ' + s_name + ' ' + str(n_i))
                isis[gn][s_name][n_i] = n_isis
                cvs[gn][s_name][n_i] = np.nanstd(n_isis)/np.nanmean(n_isis)
            #Calculate population rate
            spike_sum = np.nansum(spike_rast,0)
            pop_rate = np.array([np.sum(spike_sum[i:i+bin_size])/(bin_size/1000)/num_neur for i in np.arange(0,s_len-bin_size,skip_size)])
            pop_rates[gn][s_name] = pop_rate

    all_segment_names = np.array(all_segment_names)
    unique_segment_indices = np.sort(
        np.unique(all_segment_names, return_index=True)[1])
    unique_segment_names = [all_segment_names[i]
                          for i in unique_segment_indices]
    
    return unique_given_names, unique_segment_names, neur_rates, pop_rates, isis, cvs

def seg_stat_analysis(unique_given_names, unique_segment_names, neur_rates, \
                      pop_rates, isis, cvs, segments_to_analyze, seg_stat_save_dir):
    
    neur_rate_analysis_plots(unique_given_names,unique_segment_names,\
                                neur_rates,segments_to_analyze,seg_stat_save_dir)
    
    pop_rate_analysis_plots(unique_given_names,unique_segment_names,\
                                pop_rates,segments_to_analyze,seg_stat_save_dir)
        
    isi_cv_analysis_plots(unique_given_names,unique_segment_names,\
                                isis,cvs,segments_to_analyze,seg_stat_save_dir)
        
def neur_rate_analysis_plots(unique_given_names,unique_segment_names,\
                            neur_rates,segments_to_analyze,seg_stat_save_dir):
    """
    Conduct significance tests and plot neuron firing rate distributions
    across animals and segments.

    Parameters
    ----------
    unique_given_names : list
        List of unique animal names.
    unique_segment_names : list
        List of unique segment names across animals.
    neur_rates : dict
        Dictionary containing neuron rate calculations for each 
        animal and segment and neuron.
    seg_stat_save_dir : str
        Save path.

    Returns
    -------
    None.
    
    Outputs
    -------
    Within animal neuron rate ANOVA results across segments to .csv file.
    Violin plot of across animal neuron rates and stds of rates by segment.

    """
    num_anim = len(unique_given_names)
    cmap = colormaps['gist_rainbow']
    anim_colors = cmap(np.linspace(0, 1, num_anim))
    num_seg = len(segments_to_analyze)
    seg_pairs = list(combinations(np.arange(num_seg), 2))

    #Collect neuron rate significance and CV
    mean_sig_table = [['Animal Name', 'ANOVA p', 'Mean Order']]
    seg_cv_table = np.zeros((num_anim, num_seg))
    pairwise_sig_data = dict()
    for sp_i in range(len(seg_pairs)):
        pairwise_sig_data[seg_pairs[sp_i]] = np.zeros(num_anim)
    all_anim_means = []
    all_anim_stds = []
    for gn_i, gn in enumerate(unique_given_names):
        #Collect segment data for animal
        combined_seg_means = []
        all_seg_means = []
        all_seg_stds = []
        for s_i in segments_to_analyze:
            seg_name = unique_segment_names[s_i]
            data = neur_rates[gn][seg_name]
            num_neur = len(data)
            neur_means = np.zeros(num_neur)
            neur_stds = np.zeros(num_neur)
            for n_i in range(num_neur):
                neur_means[n_i] = np.nanmean(data[n_i])
                neur_stds[n_i] = np.nanstd(data[n_i])
            all_seg_means.append(list(neur_means))
            combined_seg_means.append(np.nanmean(neur_means))
            all_seg_stds.append(list(neur_stds))
        all_anim_means.append(all_seg_means)
        all_anim_stds.append(all_seg_stds)
        #Test significant differences
        anim_sig = [gn]
        result = f_oneway(*(all_seg_means))
        anim_sig.append(result.pvalue)
        mean_ascend = np.argsort(combined_seg_means)
        mean_text = unique_segment_names[segments_to_analyze[0]]
        for s_ind in mean_ascend[1:]:
            mean_text += ' < ' + \
                unique_segment_names[segments_to_analyze[s_ind]]
        anim_sig.append(mean_text)
        mean_sig_table.append(anim_sig)
        #Test pairwise ttest
        for sp_i in range(len(seg_pairs)):
            sp_1, sp_2 = seg_pairs[sp_i]
            result = ttest_ind(all_seg_means[sp_1], all_seg_means[sp_2])
            pairwise_sig_data[seg_pairs[sp_i]][gn_i] = result.pvalue
    #Save significance results
    with open(os.path.join(seg_stat_save_dir, 'seg_neur_mean_rate_dist_sig.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(mean_sig_table)
    np.save(os.path.join(seg_stat_save_dir, 'seg_neur_mean_rate_ttest_pair.npy'),
            pairwise_sig_data, allow_pickle=True)

    #Plot neuron rates
    f_neur_fr_std = plt.figure(figsize=(5, 5))
    joint_seg_stds = []
    for s_ind, s_i in enumerate(segments_to_analyze):
        seg_name = unique_segment_names[s_i]
        seg_stds = []
        for gn_i, gn in enumerate(unique_given_names):
            seg_stds.extend(list(all_anim_stds[gn_i][s_ind]))
        joint_seg_stds.append(seg_stds)
    plt.violinplot(joint_seg_stds, showmedians=True)
    result = f_oneway(*(joint_seg_stds))
    if result.pvalue < 0.05:
        plt.title('Neuron std(Firing Rate) Distributions\nANOVA p<0.05')
    else:
        plt.title('Neuron std(Firing Rate) Distributions\nANOVA n.s.')
    plt.xticks(np.arange(len(segments_to_analyze))+1,
               np.array(unique_segment_names)[segments_to_analyze])
    plt.xlabel('Segment')
    plt.ylabel('100 ms Bin Neuron std(Firing Rate) (Hz)')
    plt.tight_layout()
    f_neur_fr_std.savefig(os.path.join(
        seg_stat_save_dir, 'neur_std_rate_violinplots.png'))
    f_neur_fr_std.savefig(os.path.join(
        seg_stat_save_dir, 'neur_std_rate_violinplots.svg'))
    plt.close(f_neur_fr_std)

    #Plot population rate pairwise sig results
    f_neur_fr_ttest = plt.figure(figsize=(5, 5))
    for sp_i in range(len(seg_pairs)):
        sp_1, sp_2 = seg_pairs[sp_i]
        seg_1 = np.array(unique_segment_names)[segments_to_analyze[sp_1]]
        seg_2 = np.array(unique_segment_names)[segments_to_analyze[sp_2]]
        plt.boxplot(pairwise_sig_data[seg_pairs[sp_i]], positions=[
                    sp_i], labels=[seg_1 + ' v ' + seg_2])
        x_vals = sp_i + np.random.randn(num_anim)/10
        plt.scatter(x_vals, pairwise_sig_data[seg_pairs[sp_i]], color='g', alpha=0.3)
    plt.axhline(0.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    f_neur_fr_ttest.savefig(os.path.join(
        seg_stat_save_dir, 'neur_mean_rate_ttest_box.png'))
    f_neur_fr_ttest.savefig(os.path.join(
        seg_stat_save_dir, 'neur_mean_rate_ttest_box.svg'))
    plt.close(f_neur_fr_ttest)

    #Plot neuron std(rates)
    f_neur_fr = plt.figure(figsize=(5, 5))
    joint_seg_means = []
    for s_ind, s_i in enumerate(segments_to_analyze):
        seg_name = unique_segment_names[s_i]
        seg_means = []
        for gn_i, gn in enumerate(unique_given_names):
            seg_means.extend(list(all_anim_means[gn_i][s_ind]))
        joint_seg_means.append(seg_means)
    plt.violinplot(joint_seg_means, showmedians=True)
    result = f_oneway(*(joint_seg_means))
    if result.pvalue < 0.05:
        plt.title('Neuron Firing Rate Distributions\nANOVA p<0.05')
    else:
        plt.title('Neuron Firing Rate Distributions\nANOVA n.s.')
    plt.xticks(np.arange(len(segments_to_analyze))+1,
               np.array(unique_segment_names)[segments_to_analyze])
    plt.xlabel('Segment')
    plt.ylabel('100 ms Bin Neuron Firing Rate (Hz)')
    plt.tight_layout()
    f_neur_fr.savefig(os.path.join(
        seg_stat_save_dir, 'neur_rate_violinplots.png'))
    f_neur_fr.savefig(os.path.join(
        seg_stat_save_dir, 'neur_rate_violinplots.svg'))
    plt.close(f_neur_fr)


def pop_rate_analysis_plots(unique_given_names, unique_segment_names,
                            pop_rates, segments_to_analyze, seg_stat_save_dir):
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
    num_seg = len(segments_to_analyze)
    seg_pairs = list(combinations(np.arange(num_seg), 2))

    #Collect pop rate significance and CV
    sig_table = [['Animal Name', 'ANOVA p', 'Mean Order']]
    seg_cv_table = np.zeros((num_anim, len(segments_to_analyze)))
    seg_mean_fr = np.zeros((num_anim,len(segments_to_analyze)))
    all_anim_data = []
    pairwise_sig_data = dict()
    mean_pairwise_sig_data = np.zeros(len(seg_pairs))
    for sp_i in range(len(seg_pairs)):
        pairwise_sig_data[seg_pairs[sp_i]] = np.zeros(num_anim)
    for gn_i, gn in enumerate(unique_given_names):
        #Collect segment data for animal
        all_seg_data = []
        all_seg_means = []
        for s_ind, s_i in enumerate(segments_to_analyze):
            seg_name = unique_segment_names[s_i]
            data = np.array(pop_rates[gn][seg_name])
            nonnan_data = data[~np.isnan(data)]
            all_seg_data.append(nonnan_data)
            all_seg_means.append(np.mean(nonnan_data))
            seg_cv_table[gn_i, s_ind] = np.std(
                nonnan_data)/np.mean(nonnan_data)
        all_anim_data.append(all_seg_data)
        seg_mean_fr[gn_i,:] = all_seg_means
        #Test significant differences
        #    One-way ANOVA
        anim_sig = [gn]
        result = f_oneway(*(all_seg_data))
        anim_sig.append(result.pvalue)
        mean_ascend = np.argsort(all_seg_means)
        mean_text = unique_segment_names[segments_to_analyze[0]]
        for s_ind in mean_ascend[1:]:
            mean_text += ' < ' + \
                unique_segment_names[segments_to_analyze[s_ind]]
        anim_sig.append(mean_text)
        sig_table.append(anim_sig)
        #    pairwise ttest
        for sp_i in range(len(seg_pairs)):
            sp_1, sp_2 = seg_pairs[sp_i]
            data_1 = all_seg_data[sp_1]
            data_2 = all_seg_data[sp_2]
            result = ttest_ind(data_1, data_2)
            pairwise_sig_data[seg_pairs[sp_i]][gn_i] = result.pvalue
    for sp_i in range(len(seg_pairs)):
        sp_1, sp_2 = seg_pairs[sp_i]
        result = ttest_ind(seg_mean_fr[:,sp_1], seg_mean_fr[:,sp_2])
        mean_pairwise_sig_data[sp_i] = result.pvalue
    #Save significance results
    with open(os.path.join(seg_stat_save_dir, 'seg_pop_rate_sig.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(sig_table)
    np.save(os.path.join(seg_stat_save_dir, 'pairwise_pop_rate_ttest_sig_data.npy'),
            pairwise_sig_data, allow_pickle=True)
    np.save(os.path.join(seg_stat_save_dir, 'mean_pairwise_pop_rate_ttest_sig_data.npy'),
            mean_pairwise_sig_data, allow_pickle=True)

    #Plot population rates
    f_pop_fr = plt.figure(figsize=(5, 5))
    joint_seg_data = []
    for s_ind, s_i in enumerate(segments_to_analyze):
        seg_name = unique_segment_names[s_i]
        seg_data = []
        for gn_i, gn in enumerate(unique_given_names):
            seg_data.extend(list(all_anim_data[gn_i][s_ind]))
        joint_seg_data.append(seg_data)
    result = f_oneway(*joint_seg_data)
    plt.violinplot(joint_seg_data, showmedians=True)
    plt.xticks(np.arange(len(segments_to_analyze))+1,
               np.array(unique_segment_names)[segments_to_analyze])
    plt.xlabel('Segment')
    plt.ylabel('100 ms Bin Population Firing Rate (Hz)')
    if result.pvalue < 0.05:
        plt.title('Population Firing Rate Distributions\nANOVA p<0.05')
    else:
        plt.title('Population Firing Rate Distributions\nANOVA n.s.')
    plt.tight_layout()
    f_pop_fr.savefig(os.path.join(
        seg_stat_save_dir, 'pop_rate_violinplots.png'))
    f_pop_fr.savefig(os.path.join(
        seg_stat_save_dir, 'pop_rate_violinplots.svg'))
    plt.close(f_pop_fr)

    #Plot population rate pairwise sig results
    f_pop_fr_ttest = plt.figure(figsize=(5, 5))
    for sp_i in range(len(seg_pairs)):
        sp_1, sp_2 = seg_pairs[sp_i]
        seg_1 = np.array(unique_segment_names)[segments_to_analyze[sp_1]]
        seg_2 = np.array(unique_segment_names)[segments_to_analyze[sp_2]]
        plt.boxplot(pairwise_sig_data[seg_pairs[sp_i]], positions=[
                    sp_i], labels=[seg_1 + ' v ' + seg_2])
    plt.axhline(0.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    f_pop_fr_ttest.savefig(os.path.join(
        seg_stat_save_dir, 'pop_rate_ttest_box.png'))
    f_pop_fr_ttest.savefig(os.path.join(
        seg_stat_save_dir, 'pop_rate_ttest_box.svg'))
    plt.close(f_pop_fr_ttest)

    #Plot population rate pairwise sig results
    f_pop_fr_ttest = plt.figure(figsize=(5, 5))
    for sp_i in range(len(seg_pairs)):
        sp_1, sp_2 = seg_pairs[sp_i]
        seg_1 = np.array(unique_segment_names)[segments_to_analyze[sp_1]]
        seg_2 = np.array(unique_segment_names)[segments_to_analyze[sp_2]]
        plt.boxplot(nonzero_pairwise_sig_data[seg_pairs[sp_i]], positions=[
                    sp_i], labels=[seg_1 + ' v ' + seg_2])
    plt.axhline(0.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    f_pop_fr_ttest.savefig(os.path.join(
        seg_stat_save_dir, 'pop_rate_nonzero_ttest_box.png'))
    f_pop_fr_ttest.savefig(os.path.join(
        seg_stat_save_dir, 'pop_rate_nonzero_ttest_box.svg'))
    plt.close(f_pop_fr_ttest)

    #Plot CV results
    f_cv = plt.figure(figsize=(5, 5))
    plt.boxplot(seg_cv_table,
                labels=np.array(unique_segment_names)[segments_to_analyze])
    result = f_oneway(*list(seg_cv_table.T))
    #    pairwise ttest
    pairwise_ttest = np.zeros(len(seg_pairs))
    for sp_i in range(len(seg_pairs)):
        sp_1, sp_2 = seg_pairs[sp_i]
        data_1 = seg_cv_table[:, sp_1]
        data_2 = seg_cv_table[:, sp_2]
        result = ttest_ind(data_1, data_2)
        pairwise_ttest[sp_i] = result.pvalue
    for s_ind, s_i in enumerate(segments_to_analyze):
        seg_name = unique_segment_names[s_i]
        x_vals = s_ind + 1 + np.random.randn(num_anim)/10
        plt.scatter(x_vals, seg_cv_table[:, s_ind], color='g', alpha=0.3)
    seg_pairs = list(combinations(np.arange(len(segments_to_analyze)), 2))
    ttext = 'Population Firing Rate CV'
    if result.pvalue < 0.05:
        ttext += '\nANOVA p<0.05'
    else:
        ttext += '\nANOVA n.s.'
    pair_tsig = np.where(pairwise_ttest < 0.05)[0]
    ttext += '\n'
    if len(pair_tsig) > 0:
        for p_i in pair_tsig:
            s_1, s_2 = seg_pairs[p_i]
            ttext += unique_segment_names[segments_to_analyze[s_1]] + \
                ' v ' + \
                unique_segment_names[segments_to_analyze[s_2]] + ' *TT\n'
    else:
        ttext += 'All pairwise ttest n.s.'
    plt.title(ttext)
    plt.xlabel('Segment')
    plt.tight_layout()
    f_cv.savefig(os.path.join(seg_stat_save_dir, 'pop_rate_cv_boxplot.png'))
    f_cv.savefig(os.path.join(seg_stat_save_dir, 'pop_rate_cv_boxplot.svg'))
    plt.close(f_cv)

def isi_cv_analysis_plots(unique_given_names,unique_segment_names,\
                            isis,cvs,segments_to_analyze,seg_stat_save_dir,\
                                verbose=False):
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
    cvs : dict
        Dictionary containing coefficient-of-variation calculations for each
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
    segment_names_to_analyze = list(np.array(unique_segment_names)[segments_to_analyze])
    num_seg = len(segments_to_analyze)
    seg_pairs = list(combinations(np.arange(num_seg),2))
    
    anim_cvs = []
    anim_isi_dict = dict()
    pairwise_sig_data = dict()
    for sp_i in range(len(seg_pairs)):
        pairwise_sig_data[seg_pairs[sp_i]] = np.zeros(num_anim)
    for gn_i, gn in enumerate(unique_given_names):
        anim_isi_dict[gn] = dict()
        anim_isi_dict[gn]['burst'] = []
        anim_isi_dict[gn]['regular'] = []
        num_neur = len(cvs[gn][unique_segment_names[0]])
        neur_cv = np.nan*np.ones((num_neur,num_seg))
        neur_isis = dict()
        for s_ind, s_i in enumerate(segments_to_analyze):
            seg_name = unique_segment_names[s_i]
            isi_data = isis[gn][seg_name]
            cv_data = cvs[gn][seg_name]
            neur_cv[np.where(cv_data > 0)[0],s_ind] = cv_data[np.where(cv_data > 0)[0]]
            for n_i in range(num_neur):
                if len(np.intersect1d(list(neur_isis.keys()),[n_i])) == 0:
                    neur_isis[n_i] = []
                try:
                    n_data = isi_data[n_i] #in ms
                    neur_isis[n_i].extend(list(n_data[n_data > 0]))
                except:
                    if verbose == True:
                        print(gn + ' ' + seg_name + ' neuron ' + str(n_i) + ' is an excluded outlier.')
        anim_cvs.append(neur_cv)
        for n_i in range(num_neur):
            neur_cv_avg = np.nanmean(neur_cv[n_i,:])
            if neur_cv_avg > 1: #bursty
                anim_isi_dict[gn]['burst'].extend(list(neur_isis[n_i]))
            else:
                anim_isi_dict[gn]['regular'].extend(list(neur_isis[n_i]))
        #    pairwise KS
        for sp_i in range(len(seg_pairs)):
            sp_1, sp_2 = seg_pairs[sp_i]
            data_1 = neur_cv[:,sp_1]
            data_2 = neur_cv[:,sp_2]
            result = ttest_ind(data_1,data_2)
            pairwise_sig_data[seg_pairs[sp_i]][gn_i] = result.pvalue
    
    #Plot neuron cv pairwise sig results
    f_neur_cv_ttest = plt.figure(figsize=(5,5))
    for sp_i in range(len(seg_pairs)):
        sp_1, sp_2 = seg_pairs[sp_i]
        seg_1 = np.array(unique_segment_names)[segments_to_analyze[sp_1]]
        seg_2 = np.array(unique_segment_names)[segments_to_analyze[sp_2]]
        plt.boxplot(pairwise_sig_data[seg_pairs[sp_i]],positions=[sp_i],labels=[seg_1 + ' v ' + seg_2])
        x_vals = sp_i + np.random.randn(num_anim)/10
        plt.scatter(x_vals, pairwise_sig_data[seg_pairs[sp_i]], color='g', alpha=0.3)
    plt.axhline(0.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    f_neur_cv_ttest.savefig(os.path.join(seg_stat_save_dir,'neur_cv_ttest_box.png'))
    f_neur_cv_ttest.savefig(os.path.join(seg_stat_save_dir,'neur_cv_ttest_box.svg'))
    plt.close(f_neur_cv_ttest)
    
    #Individual Animals CV Separate
    x_vals = [0,1,1.1,1.2,1.5,2,10,1000] #[0,1,2,5,10,1000]
    num_x = len(x_vals)
    bar_w = 1/num_anim
    half_bar = bar_w/2
    labels = [str(x_vals[i]) + '-' + str(x_vals[i+1]) for i in range(num_x-2)]
    labels.extend([str(x_vals[-2]) + '+'])
    f_cv, ax_cv = plt.subplots(ncols=num_seg,figsize=(15,5),\
                               sharex = True,sharey = True)
    for s_i, seg_name in enumerate(segment_names_to_analyze):
        anim_seg_vals = []
        for gn_i, gn in enumerate(unique_given_names):
            anim_cv = anim_cvs[gn_i][:,s_i]
            anim_seg_vals.append(anim_cv)
            y_vals = [len(np.where((anim_cv < x_vals[i+1])*(x_vals[i]<=anim_cv))[0])/len(anim_cv) for i in range(num_x-1)]
            ax_cv[s_i].bar(np.arange(num_x-1)+gn_i*bar_w+half_bar,y_vals,\
                    color=anim_colors[gn_i,:],alpha=0.3,label=gn,\
                        width=bar_w)
        anova_result = f_oneway(*anim_seg_vals)
        if anova_result.pvalue <= 0.05:
            ax_cv[s_i].set_title(seg_name + ' CV (*ANOVA)')
        else:
            ax_cv[s_i].set_title(seg_name + ' CV (n.s. ANOVA)')
        ax_cv[s_i].set_xticks(np.arange(num_x-1)+0.5,labels=labels,rotation=45)
        ax_cv[s_i].set_xlabel('CV Range')
        for x_i in np.arange(num_x):
            ax_cv[s_i].axvline(x_i,color='k',linestyle='dashed',alpha=0.2)
    ax_cv[s_i].legend(loc='upper right')
    ax_cv[0].set_ylabel('Fraction of Neurons')
    plt.suptitle('By-Segment Neuron CV Distributions')
    plt.tight_layout()
    f_cv.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_cv_boxplot.png'))
    f_cv.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_cv_boxplot.svg'))
    plt.close(f_cv)
    
    #Across Animals CV
    seg_vals = []
    for s_i, seg_name in enumerate(segment_names_to_analyze):
        anim_seg_vals = []
        for gn_i, gn in enumerate(unique_given_names):
            anim_cv = anim_cvs[gn_i][:,s_i]
            anim_seg_vals.extend(list(np.log(anim_cv)))
        seg_vals.append(anim_seg_vals)
    f_cv_violin = plt.figure(figsize=(5,5))
    plt.violinplot(seg_vals)
    result = f_oneway(*(seg_vals))
    plt.axhline(np.log(1),linestyle='dashed',color='k',alpha=0.2)
    plt.xticks(np.arange(num_seg)+1,segment_names_to_analyze)
    plt.ylabel('ln(CV)')
    plt.xlabel('Segment')
    if result.pvalue < 0.05:
        plt.title('Neuron ln(CV) Distributions\nANOVA p<0.05')
    else:
        plt.title('Neuron ln(CV) Distributions\nANOVA n.s.')
    plt.tight_layout()
    f_cv_violin.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_ln_cv_violin.png'))
    f_cv_violin.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_ln_cv_violin.svg'))
    plt.close(f_cv_violin)
    f_cv_box = plt.figure(figsize=(5,5))
    plt.boxplot(seg_vals)
    plt.axhline(np.log(1),linestyle='dashed',color='k',alpha=0.2)
    plt.xticks(np.arange(num_seg)+1,segment_names_to_analyze)
    plt.ylabel('ln(CV)')
    plt.xlabel('Segment')
    if result.pvalue < 0.05:
        plt.title('Neuron ln(CV) Distributions\nANOVA p<0.05')
    else:
        plt.title('Neuron ln(CV) Distributions\nANOVA n.s.')
    plt.tight_layout()
    f_cv_box.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_ln_cv_box.png'))
    f_cv_box.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_ln_cv_box.svg'))
    plt.close(f_cv_box)
    
    seg_vals = []
    for s_i, seg_name in enumerate(segment_names_to_analyze):
        anim_seg_vals = []
        for gn_i, gn in enumerate(unique_given_names):
            anim_cv = anim_cvs[gn_i][:,s_i]
            anim_seg_vals.extend(list(anim_cv))
        seg_vals.append(anim_seg_vals)
    f_cv_box = plt.figure(figsize=(5,5))
    plt.boxplot(seg_vals)
    result = f_oneway(*(seg_vals))
    plt.axhline(1,linestyle='dashed',color='k',alpha=0.2)
    plt.xticks(np.arange(num_seg)+1,segment_names_to_analyze)
    plt.ylabel('CV')
    plt.xlabel('Segment')
    if result.pvalue < 0.05:
        plt.title('Neuron CV Distributions\nANOVA p<0.05')
    else:
        plt.title('Neuron CV Distributions\nANOVA n.s.')
    plt.tight_layout()
    f_cv_box.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_cv_box.png'))
    f_cv_box.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_cv_box.svg'))
    plt.close(f_cv_box)
    f_cv_violin = plt.figure(figsize=(5,5))
    plt.violinplot(seg_vals)
    plt.axhline(1,linestyle='dashed',color='k',alpha=0.2)
    plt.xticks(np.arange(num_seg)+1,segment_names_to_analyze)
    plt.ylabel('CV')
    plt.xlabel('Segment')
    if result.pvalue < 0.05:
        plt.title('Neuron CV Distributions\nANOVA p<0.05')
    else:
        plt.title('Neuron CV Distributions\nANOVA n.s.')
    plt.tight_layout()
    f_cv_violin.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_cv_violin.png'))
    f_cv_violin.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_cv_violin.svg'))
    plt.close(f_cv_violin)
    
    
    #Burst vs Regular ISI Distributions
    burst_combined = []
    regular_combined = []
    for gn_i, gn in enumerate(unique_given_names):
        burst_combined.extend(anim_isi_dict[gn]['burst'])
        regular_combined.extend(anim_isi_dict[gn]['regular'])
    log_burst_combined = np.log(np.array(burst_combined))
    log_regular_combined = np.log(np.array(regular_combined))
    log_bins = np.log(x_vals)
    f_isi_hist = plt.figure(figsize=(5,5))
    plt.violinplot([log_burst_combined,log_regular_combined])
    plt.xticks([1,2],['Bursting','Regular'])
    # plt.hist(log_burst_combined,bins=log_bins,histtype='step',density=True,\
    #          cumulative=False,alpha=0.5,label='Bursting')
    # plt.hist(log_regular_combined,bins=log_bins,histtype='step',density=True,\
    #          cumulative=False,alpha=0.5,label='Regular')
    # plt.legend(loc='upper right')
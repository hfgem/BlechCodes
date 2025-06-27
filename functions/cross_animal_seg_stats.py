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
                n_spikes = np.sort(spike_times[n_i]) #neuron spike times in ms
                n_seg_spike_times = (n_spikes[np.where((n_spikes < s_end)*(s_start <= n_spikes))[0]]).astype('int') - int(s_start)
                spike_rast[n_i,n_seg_spike_times] = 1
                #Calculate neuron firing stats
                n_isis = np.diff(n_seg_spike_times)
                isi_zscore = (n_isis - np.nanmean(n_isis))/np.nanstd(n_isis)
                super_outliers = np.where(isi_zscore > 10)[0]
                if len(super_outliers) > 0: #Plot/remove the extreme outliers - neuron signal loss?
                    plt.figure()    
                    plt.plot(n_isis)
                    plt.title(gn + ' ' + s_name + ' ' + str(n_i))
                isis_cleaned = n_isis[n_isis < 60*1000]
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
        
    isi_cv_analysis_plots(unique_given_names,unique_segment_names,\
                                isis,cvs,seg_stat_save_dir)
        
    
    
    
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
    all_anim_data = []
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
        all_anim_data.append(all_seg_data)
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
        
    #Plot population rates
    f_pop_fr = plt.figure(figsize=(5,5))
    joint_seg_data = []
    for s_i, seg_name in enumerate(unique_segment_names):
        seg_data = []
        for gn_i, gn in enumerate(unique_given_names):
            seg_data.extend(list(all_anim_data[gn_i][s_i]))
        joint_seg_data.append(seg_data)
    plt.violinplot(joint_seg_data,showmedians=True)
    plt.xticks(np.arange(num_seg)+1,unique_segment_names)
    plt.xlabel('Segment')
    plt.ylabel('100 ms Bin Population Firing Rate (Hz)')
    plt.title('Population Firing Rate Distributions')
    plt.tight_layout()
    f_pop_fr.savefig(os.path.join(seg_stat_save_dir,'pop_rate_violinplots.png'))
    f_pop_fr.savefig(os.path.join(seg_stat_save_dir,'pop_rate_violinplots.svg'))
    plt.close(f_pop_fr)
        
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
    plt.close(f_cv)
    
def isi_cv_analysis_plots(unique_given_names,unique_segment_names,\
                            isis,cvs,seg_stat_save_dir,verbose=False):
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
    num_seg = len(unique_segment_names)
    
    #Collect isi significance and CV
    x_vals = [0,1,1.1,1.2,1.5,2,10,1000] #[0,1,2,5,10,1000]
    num_x = len(x_vals)
    bar_w = 1/num_anim
    half_bar = bar_w/2
    labels = [str(x_vals[i]) + '-' + str(x_vals[i+1]) for i in range(num_x-2)]
    labels.extend([str(x_vals[-2]) + '+'])
    
    anim_cvs = []
    anim_isi_dict = dict()
    for gn_i, gn in enumerate(unique_given_names):
        anim_isi_dict[gn] = dict()
        anim_isi_dict[gn]['burst'] = []
        anim_isi_dict[gn]['regular'] = []
        num_neur = len(cvs[gn][unique_segment_names[0]])
        neur_cv = np.nan*np.ones((num_neur,num_seg))
        neur_isis = dict()
        for s_i, seg_name in enumerate(unique_segment_names):
            isi_data = isis[gn][seg_name]
            cv_data = cvs[gn][seg_name]
            neur_cv[np.where(cv_data > 0)[0],s_i] = cv_data[np.where(cv_data > 0)[0]]
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
                
    
    #Individual Animals CV Separate
    f_cv, ax_cv = plt.subplots(ncols=num_seg,figsize=(15,5),\
                               sharex = True,sharey = True)
    for s_i, seg_name in enumerate(unique_segment_names):
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
    for s_i, seg_name in enumerate(unique_segment_names):
        anim_seg_vals = []
        for gn_i, gn in enumerate(unique_given_names):
            anim_cv = anim_cvs[gn_i][:,s_i]
            anim_seg_vals.extend(list(np.log(anim_cv)))
        seg_vals.append(anim_seg_vals)
    f_cv_violin = plt.figure(figsize=(5,5))
    plt.violinplot(seg_vals)
    plt.axhline(np.log(1),linestyle='dashed',color='k',alpha=0.2)
    plt.xticks(np.arange(num_seg)+1,unique_segment_names)
    plt.ylabel('ln(CV)')
    plt.title('Neuron ln(CV) Distributions')
    plt.tight_layout()
    f_cv_violin.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_ln_cv_violin.png'))
    f_cv_violin.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_ln_cv_violin.svg'))
    plt.close(f_cv_violin)
    f_cv_box = plt.figure(figsize=(5,5))
    plt.boxplot(seg_vals)
    plt.axhline(np.log(1),linestyle='dashed',color='k',alpha=0.2)
    plt.xticks(np.arange(num_seg)+1,unique_segment_names)
    plt.ylabel('CV')
    plt.title('Neuron CV Distributions')
    plt.tight_layout()
    f_cv_box.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_ln_cv_box.png'))
    f_cv_box.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_ln_cv_box.svg'))
    plt.close(f_cv_box)
    
    seg_vals = []
    for s_i, seg_name in enumerate(unique_segment_names):
        anim_seg_vals = []
        for gn_i, gn in enumerate(unique_given_names):
            anim_cv = anim_cvs[gn_i][:,s_i]
            anim_seg_vals.extend(list(anim_cv))
        seg_vals.append(anim_seg_vals)
    f_cv_box = plt.figure(figsize=(5,5))
    plt.boxplot(seg_vals)
    plt.axhline(1,linestyle='dashed',color='k',alpha=0.2)
    plt.xticks(np.arange(num_seg)+1,unique_segment_names)
    plt.ylabel('CV')
    plt.title('Neuron CV Distributions')
    plt.tight_layout()
    f_cv_box.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_cv_box.png'))
    f_cv_box.savefig(os.path.join(seg_stat_save_dir,'segment_neuron_cv_box.svg'))
    plt.close(f_cv_box)
    f_cv_violin = plt.figure(figsize=(5,5))
    plt.violinplot(seg_vals)
    plt.axhline(1,linestyle='dashed',color='k',alpha=0.2)
    plt.xticks(np.arange(num_seg)+1,unique_segment_names)
    plt.ylabel('CV')
    plt.title('Neuron CV Distributions')
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
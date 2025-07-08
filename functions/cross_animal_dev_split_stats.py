#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 10:32:16 2025

@author: hannahgermaine
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib import colormaps

def run_cross_animal_dev_split_corr_analyses(dev_split_corr_data, unique_given_names,
                                        unique_segment_names, unique_taste_names,
                                        unique_epoch_pairs, results_dir):
    """Calculate statistics and plot results for dev split decodes"""
    dev_split_hotellings_plots(dev_split_corr_data, unique_given_names,
                                   unique_segment_names, results_dir)
    
    cross_dataset_dev_split_corr_plots(dev_split_corr_data, unique_given_names, 
                                     unique_epoch_pairs, unique_segment_names, 
                                     unique_taste_names, results_dir)
    
    
def dev_split_hotellings_plots(dev_split_corr_data, unique_given_names,
                               unique_segment_names, results_dir):
    """Plot the results of hotellings test on split deviation events"""
    num_segments = len(unique_segment_names)
    num_anim = len(unique_given_names)
    
    f_hotellings = plt.figure(figsize = (5,5))
    seg_p_vals = []
    for sn_i, sn in enumerate(unique_segment_names):
        anim_p_vals = []
        for gn in unique_given_names:
            an_h_data = np.array(dev_split_corr_data[gn]['hotellings'])
            s_inds = np.where(an_h_data[:,0] == sn)[0]
            p_inds = np.where(an_h_data[:,1] == 'p-val')[0]
            anim_p_vals.append((an_h_data[np.intersect1d(s_inds,p_inds)[0],-1]).astype('float'))
        seg_p_vals.append(anim_p_vals)
        x_scat = 1 + sn_i + 0.1*np.random.randn(num_anim)
        plt.scatter(x_scat,anim_p_vals,alpha=0.25,color='g')
    plt.axhline(0.05,linestyle='dashed',alpha=0.5,color='k')
    plt.boxplot(seg_p_vals)
    plt.xticks(1 + np.arange(num_segments),unique_segment_names)
    plt.title('Split Dev Hotellings P-Values')
    plt.tight_layout()
    f_hotellings.savefig(os.path.join(results_dir,'split_dev_hotellings.png'))
    f_hotellings.savefig(os.path.join(results_dir,'split_dev_hotellings.svg'))
    
def cross_dataset_dev_split_corr_plots(dev_split_corr_data, unique_given_names, 
                                 unique_epoch_pairs, unique_segment_names, 
                                 unique_taste_names, results_dir):
    """
    This function plots the results of split deviation event correlation data
    across multiple animals.
    """
    #Variables
    num_epoch_pairs = len(unique_epoch_pairs)
    num_segments = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    
    taste_pair_inds = list(combinations(np.arange(len(unique_taste_names)),2))
    segment_pair_inds = list(combinations(np.arange(len(unique_segment_names)), 2))
    
    #Plot epoch pair x segment the taste correlations against each other
    f_e_seg, ax_e_seg = plt.subplots(nrows = num_epoch_pairs, ncols = num_segments,
                                     sharex = True, sharey = True, figsize=(8,num_epoch_pairs*2))
    for ep_i, ep_str in enumerate(unique_epoch_pairs):
        for s_i, seg_name in enumerate(unique_segment_names):
            #Collect cmf data
            t_data = []
            for t_i, t_name in enumerate(unique_taste_names):
                t_data_combined = []
                for g_i, g_name in enumerate(unique_given_names):
                    try: #If data exists for this animal
                        corr_array = dev_split_corr_data[g_name]['corr_data'][seg_name][t_name]
                        ep_list = dev_split_corr_data[g_name]['corr_data'][seg_name]['epoch_pairs']
                        ep_ind = [i for i in range(len(ep_list)) if str(ep_list[i]) == ep_str]
                        t_data_combined.extend(list(corr_array[ep_ind[0],:]))
                    except:
                        t_data_combined.extend([])
                t_data.append(t_data_combined)
            #Plot cmf
            for td_i, td in enumerate(t_data):
                nonnan_data = td[~np.isnan(td)]
                min_data = np.nanmin(nonnan_data)
                max_data = np.nanmax(nonnan_data)
                cdf_x_vals = np.linspace(min_data,max_data,1000)
                cdf_y_vals = [len(np.where(nonnan_data <= i)[0])/len(nonnan_data) for i in cdf_x_vals]
                ax_e_seg[ep_i,s_i].plot(cdf_x_vals,cdf_y_vals,label=unique_taste_names[td_i])
            if (ep_i == 0)*(s_i == 0):
                ax_e_seg[ep_i,s_i].legend(loc='upper left')
            if ep_i == 0:
                ax_e_seg[ep_i,s_i].set_title(seg_name)
            if ep_i == num_epoch_pairs-1:
                ax_e_seg[ep_i,s_i].set_xlabel('Pearson Correlation')
            if s_i == 0:
                ax_e_seg[ep_i,s_i].set_ylabel(ep_str)
            #Calculate pairwise significances
            t_sig_text = "Sig:\n"
            for tp_i, tp in enumerate(taste_pair_inds):
                t_1 = tp[0]
                t_2 = tp[1]
                try:
                    k_res = ks_2samp(t_data[t_1],t_data[t_2])
                    if k_res.pvalue < 0.05:
                        t_sig_text = t_sig_text + unique_taste_names[t_1] + \
                            "x" + unique_taste_names[t_2] + "\n"
                except: #Not enough data for KS test
                    t_sig_text = t_sig_text
            ax_e_seg[ep_i,s_i].text(-0.5,0.05,t_sig_text)
            
    plt.suptitle('CMF Split Dev Epoch Pair Corr')
    plt.tight_layout()
    f_e_seg.savefig(os.path.join(results_dir,'epoch_x_seg_taste_cmfs.png'))
    f_e_seg.savefig(os.path.join(results_dir,'epoch_x_seg_taste_cmfs.svg'))
    plt.close(f_e_seg)
    
    
    #Plot epoch pair x taste the segment correlations against each other
    f_e_taste, ax_e_taste = plt.subplots(nrows = num_epoch_pairs, ncols = num_tastes,
                                     sharex = True, sharey = True, figsize=(8,num_epoch_pairs*2))
    for ep_i, ep_str in enumerate(unique_epoch_pairs):
        for t_i, t_name in enumerate(unique_taste_names):
            #Collect cmf data
            s_data = []
            for s_i, seg_name in enumerate(unique_segment_names):
                s_data_combined = []
                for g_i, g_name in enumerate(unique_given_names):
                    try: #If data exists for this animal
                        corr_array = dev_split_corr_data[g_name]['corr_data'][seg_name][t_name]
                        ep_list = dev_split_corr_data[g_name]['corr_data'][seg_name]['epoch_pairs']
                        ep_ind = [i for i in range(len(ep_list)) if str(ep_list[i]) == ep_str]
                        s_data_combined.extend(list(corr_array[ep_ind[0],:]))
                    except:
                        s_data_combined.extend([])
                s_data.append(s_data_combined)
            #Plot cmf
            for sd_i, sd in enumerate(s_data):
                nonnan_data = sd[~np.isnan(sd)]
                min_x = np.nanmin(nonnan_data)
                max_x = np.nanmax(nonnan_data)
                cdf_x = np.linspace(min_x,max_x,1000)
                cdf_y = np.array([len(np.where(nonnan_data <= i)[0])/len(nonnan_data) for i in cdf_x])
                ax_e_taste[ep_i,t_i].plot(cdf_x,cdf_y,label=unique_segment_names[sd_i])
                # ax_e_taste[ep_i,t_i].hist(sd,density=True,cumulative=True,bins=1000,
                #                         histtype='step',label=unique_segment_names[sd_i])
            if (ep_i == 0)*(t_i == 0):
                ax_e_taste[ep_i,t_i].legend(loc='upper left')
            if ep_i == 0:
                ax_e_taste[ep_i,t_i].set_title(t_name)
            if ep_i == num_epoch_pairs-1:
                ax_e_taste[ep_i,t_i].set_xlabel('Pearson Correlation')
            if t_i == 0:
                ax_e_taste[ep_i,t_i].set_ylabel(ep_str)
            #Calculate pairwise significances
            s_sig_text = "Sig:\n"
            for sp_i, sp in enumerate(segment_pair_inds):
                s_1 = sp[0]
                s_2 = sp[1]
                try:
                    k_res = ks_2samp(s_data[s_1],s_data[s_2])
                    if k_res.pvalue < 0.05:
                        s_sig_text = s_sig_text + unique_segment_names[s_1] + \
                            "x" + unique_segment_names[s_2] + "\n"
                except: #Not enough data for KS test
                    s_sig_text = s_sig_text
            ax_e_taste[ep_i,t_i].text(-0.5,0.05,s_sig_text)
            
    plt.suptitle('CMF Split Dev Epoch Pair Corr')
    plt.tight_layout()
    f_e_taste.savefig(os.path.join(results_dir,'epoch_x_taste_seg_cmfs.png'))
    f_e_taste.savefig(os.path.join(results_dir,'epoch_x_taste_seg_cmfs.svg'))
    plt.close(f_e_taste)
    
    #Plot segment x taste grid of epoch pairs against each other
    f_seg_taste, ax_seg_taste = plt.subplots(nrows = num_segments, ncols = num_tastes,
                                     sharex = True, sharey = True, figsize=(8,8))
    for s_i, seg_name in enumerate(unique_segment_names):
        for t_i, t_name in enumerate(unique_taste_names):
            #Collect cmf data
            e_data = []
            for ep_i, ep in enumerate(unique_epoch_pairs):
                e_data_combined = []
                for g_i, g_name in enumerate(unique_given_names):
                    try: #If data exists for this animal
                        corr_array = dev_split_corr_data[g_name]['corr_data'][seg_name][t_name]
                        ep_list = dev_split_corr_data[g_name]['corr_data'][seg_name]['epoch_pairs']
                        ep_ind = [i for i in range(len(ep_list)) if str(ep_list[i]) == ep_str]
                        e_data_combined.extend(list(corr_array[ep_ind[0],:]))
                    except:
                        e_data_combined.extend([])
                e_data.append(e_data_combined)
            #Plot cmf
            for ed_i, ed in enumerate(e_data):
                nonnan_data = ed[~np.isnan(ed)]
                min_x = np.nanmin(nonnan_data)
                max_x = np.nanmax(nonnan_data)
                cdf_x = np.linspace(min_x,max_x,1000)
                cdf_y = np.array([len(np.where(nonnan_data <= i)[0])/len(nonnan_data) for i in cdf_x])
                ax_seg_taste[s_i,t_i].plot(cdf_x,cdf_y,label=unique_epoch_pairs[ed_i])
            if (s_i == 0)*(t_i == 0):
                ax_seg_taste[s_i,t_i].legend(loc='upper left')
            if s_i == 0:
                ax_seg_taste[s_i,t_i].set_title(t_name)
            if s_i == num_epoch_pairs-1:
                ax_seg_taste[s_i,t_i].set_xlabel('Pearson Correlation')
            if t_i == 0:
                ax_seg_taste[s_i,t_i].set_ylabel(seg_name)
            #Calculate pairwise significances
            e_sig_text = "Sig:\n"
            for ep_i, ep in enumerate(epoch_pair_inds):
                e_1 = ep[0]
                e_2 = ep[1]
                try:
                    k_res = ks_2samp(e_data[e_1],e_data[e_2])
                    if k_res.pvalue < 0.05:
                        e_sig_text = e_sig_text + unique_epoch_pairs[e_1] + \
                            "x" + unique_epoch_pairs[e_2] + "\n"
                except: #Not enough data for KS test
                    e_sig_text = e_sig_text
            ax_seg_taste[s_i,t_i].text(-0.5,0.05,e_sig_text)
            
    plt.suptitle('CMF Split Dev Epoch Pair Corr')
    plt.tight_layout()
    f_seg_taste.savefig(os.path.join(results_dir,'seg_x_taste_epoch_cmfs.png'))
    f_seg_taste.savefig(os.path.join(results_dir,'seg_x_taste_epoch_cmfs.svg'))
    plt.close(f_seg_taste)

def run_cross_animal_dev_split_decode_analyses(dev_split_decode_data, group_pair_dict,
                                               unique_given_names,unique_segment_names, 
                                               unique_taste_names,unique_group_names,
                                               unique_group_pair_names,results_dir):
    """Calculate statistics and plot results for dev split decodes"""
    
    #Plot decode rates
    cross_dataset_dev_split_decode_rate_plots(dev_split_decode_data, group_pair_dict,
                                                   unique_given_names,unique_segment_names, 
                                                   unique_taste_names,unique_group_names,
                                                   unique_group_pair_names,results_dir,
                                                   verbose = False)
    
    
def cross_dataset_dev_split_decode_prob_plots(dev_split_decode_data, group_pair_dict,
                                               unique_given_names,unique_segment_names, 
                                               unique_taste_names,unique_group_names,
                                               unique_group_pair_names,results_dir,
                                               verbose = False):
    """Plot the dev split correlation results"""
    print("Do something")
    
def cross_dataset_dev_split_decode_rate_plots(dev_split_decode_data, group_pair_dict,
                                               unique_given_names,unique_segment_names, 
                                               unique_taste_names,unique_group_names,
                                               unique_group_pair_names,results_dir,
                                               verbose = False):
    """Plot the dev decode fractions by decode type"""
    #Variables
    num_anim = len(unique_given_names)
    num_segments = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    cmap = colormaps['gist_rainbow']
    anim_colors = cmap(np.linspace(0, 1, num_anim))
    group_pair_dict_keys = list(group_pair_dict.keys())
    num_group_pairs = len(group_pair_dict_keys)
    
    #Collect decode rates by group pair
    for sn_i, sn in enumerate(unique_segment_names):
        group_pair_rates = []
        group_pair_totals = []
        for gp_i, gp in enumerate(group_pair_dict_keys):
            key_options = group_pair_dict[gp]
            gp_rates = []
            gp_counts = []
            for gn_i, gn in enumerate(unique_given_names):
                anim_gp_counts = []
                num_dev, _ = np.shape(dev_split_decode_data[gn]['decode_data'][sn]['argmax'])
                for ko in key_options:
                    try:
                        anim_gp_counts.append(dev_split_decode_data[gn]['decode_data'][sn]['group_dict'][ko]['count'])
                    except:
                        anim_gp_counts.extend([])
                gp_rates.append(np.nansum(anim_gp_counts)/num_dev)
                gp_counts.append(np.nansum(anim_gp_counts))
            group_pair_rates.append(gp_rates)
            group_pair_totals.append(np.nansum(np.array(gp_counts)))
        group_pair_rate_medians = [np.nanmedian(gpr) for gpr in group_pair_rates]
        rate_order = np.argsort(group_pair_rate_medians)[::-1]
        reordered_rates = [group_pair_rates[i] for i in rate_order]
        reordered_names = [group_pair_dict_keys[i] for i in rate_order]
        #Box plots
        f_box = plt.figure(figsize=(8,5))
        for rr_i, rr in enumerate(reordered_rates):
            x_scat = 1 + rr_i + 0.1*np.random.randn(len(rr))
            plt.scatter(x_scat,rr,alpha=0.25,\
                                        color='g')
        plt.boxplot(reordered_rates)
        plt.xticks(np.arange(num_group_pairs) + 1,\
                                       reordered_names,ha = 'right',\
                                           rotation=45)
        plt.title(sn + ' decode rates')
        plt.tight_layout()
        f_box.savefig(os.path.join(results_dir,sn + '_decode_rates_box.png'))
        f_box.savefig(os.path.join(results_dir,sn + '_decode_rates_box.svg'))
        #Pie plots
        summed_rates = np.array(group_pair_totals)
        f_pie = plt.figure(figsize=(8,8))
        plt.pie(summed_rates,autopct='%1.1f%%')
        plt.title(sn + ' total rates')
        plt.legend(labels=group_pair_dict_keys,loc='best')
        plt.tight_layout()
        f_pie.savefig(os.path.join(results_dir,sn + '_decode_rates_pie.png'))
        f_pie.savefig(os.path.join(results_dir,sn + '_decode_rates_pie.svg'))
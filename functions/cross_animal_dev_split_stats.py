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
                                        unique_group_names, results_dir):
    """Calculate statistics and plot results for dev split decodes"""
    print("Do something")

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
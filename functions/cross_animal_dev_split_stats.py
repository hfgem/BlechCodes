#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 10:32:16 2025

@author: hannahgermaine
"""

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
            gp_rates = []
            gp_counts = []
            for gn_i, gn in enumerate(unique_given_names):
                num_dev, _ = np.shape(dev_split_decode_data[gn]['decode_data'][sn]['argmax'])
                try:
                    gp_counts.append(dev_split_decode_data[gn]['decode_data'][sn]['group_dict'][gp]['count'])
                    gp_rates.append(dev_split_decode_data[gn]['decode_data'][sn]['group_dict'][gp]['count']/num_dev)
                except:
                    gp_counts.extend([])
                    gp_rates.extend([])
                    if verbose == True:
                        print('Missing data for ' + gn + ' ' + sn + ' ' + gp)
            group_pair_rates.append(gp_rates)
            group_pair_totals.append(np.nansum(np.array(gp_counts)))
            x_scat = 1 + gp_i + 0.1*np.random.randn(len(gp_rates))
            ax_pair_rates[sn_i].scatter(x_scat,np.array(gp_rates),alpha=0.25,\
                                        color='g')
        #Box plots
        f_box = plt.figure(figsize=(8,8))
        plt.boxplot(group_pair_rates)
        plt.xticks(np.arange(num_group_pairs) + 1,\
                                       group_pair_dict_keys,ha = 'right',\
                                           rotation=45)
        plt.title(sn + ' decode rates')
        plt.tight_layout()
        # f_box.savefig(os.path.join(results_dir,sn + '_decode_rates_box.png'))
        # f_box.savefig(os.path.join(results_dir,sn + '_decode_rates_box.svg'))
        #Pie plots
        summed_rates = np.array(group_pair_totals)
        f_pie = plt.figure(figsize=(8,8))
        plt.pie(summed_rates,autopct='%1.1f%%')
        plt.title(sn + ' total rates')
        plt.legend(labels=group_pair_dict_keys,loc='best')
        plt.tight_layout()
        # f_pie.savefig(os.path.join(results_dir,sn + '_decode_rates_pie.png'))
        # f_pie.savefig(os.path.join(results_dir,sn + '_decode_rates_pie.svg'))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:32:10 2025

@author: Hannah Germaine

A collection of functions for comparing different datasets against each other 
in their multi-day correlation and decoding data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def compare_corr_data(corr_dict, multiday_data_dict, unique_given_names,
                      unique_corr_names, unique_segment_names, unique_taste_names, 
                      max_cp, save_dir):
    
    corr_results_save_dir = os.path.join(save_dir,'Correlations')
    if not os.path.isdir(corr_results_save_dir):
        os.mkdir(corr_results_save_dir)
        
    #Create corr plots by segment
    plot_corr_by_segment(corr_dict, unique_given_names, unique_corr_names,
                             unique_segment_names, unique_taste_names, max_cp,
                             corr_results_save_dir)
        
    
def plot_corr_by_segment(corr_dict, unique_given_names, unique_corr_names,
                         unique_segment_names, unique_taste_names, max_cp,
                         plot_save_dir):
    #Plot joint best taste correlation distributions against each other
    #in a grid of epoch x segment
    for corr_name in unique_corr_names:
        f_taste_cdf, ax_taste_cdf = plt.subplots(max_cp,len(unique_segment_names),
                                                 sharex = True, sharey = True, figsize = (8,8))
        for s_i, seg_name in enumerate(unique_segment_names):
            for ep_i in range(max_cp):
                taste_corrs = []
                for taste_name in unique_taste_names:
                    this_taste_corrs = []
                    for data_name in unique_given_names:
                        best_data = corr_dict[data_name][corr_name][seg_name]['best']
                        taste_names_data = corr_dict[data_name][corr_name]['tastes']
                        taste_name_ind = np.where(np.array(taste_names_data) == taste_name)[0]
                        if len(taste_name_ind) > 0: #Taste exists in data
                            best_dev_taste_inds = np.where(best_data[:,0] == taste_name_ind)[0]
                            best_dev_epoch_inds = np.where(best_data[:,1] == ep_i)[0]
                            joint_inds = np.intersect1d(best_dev_taste_inds,best_dev_epoch_inds)
                            this_taste_corrs.extend(list(best_data[joint_inds,2]))
                    taste_corrs.append(this_taste_corrs)
                    ax_taste_cdf[ep_i,s_i].hist(this_taste_corrs,bins=1000,
                                                density=True,cumulative=True,
                                                histtype='step',label=taste_name)
                if s_i == 0:
                    ax_taste_cdf[ep_i,s_i].set_ylabel('Epoch ' + str(ep_i) + '\nCumulative Density')
                if ep_i == max_cp-1:
                    ax_taste_cdf[ep_i,s_i].set_xlabel('Pearson Correlation')
                if ep_i == 0:
                    ax_taste_cdf[ep_i,s_i].set_title(seg_name)
        ax_taste_cdf[0,0].legend(loc='upper left')
        plt.suptitle(corr_name + ' Best Corr Distributions')
        plt.tight_layout()
        f_taste_cdf.savefig(os.path.join(plot_save_dir,corr_name+'_best_corr_joint_cdf.png'))
        f_taste_cdf.savefig(os.path.join(plot_save_dir,corr_name+'_best_corr_joint_cdf.svg'))
        
                
    

    
    
    
    
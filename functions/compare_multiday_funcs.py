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
        f_best_taste_cdf, ax_best_taste_cdf = plt.subplots(max_cp,len(unique_segment_names),
                                                 sharex = True, sharey = True, figsize = (8,8))
        f_best_taste_counts, ax_best_taste_counts = plt.subplots(max_cp,len(unique_segment_names),
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
                    ax_best_taste_cdf[ep_i,s_i].hist(this_taste_corrs,bins=1000,
                                                density=True,cumulative=True,
                                                histtype='step',label=taste_name)
                best_taste_corr_counts = np.array([len(tc) for tc in taste_corrs])
                best_taste_corr_percents = np.round(100*best_taste_corr_counts/np.sum(best_taste_corr_counts),2)
                if (ep_i == 0) and (s_i == 0):
                    best_taste_corr_labels = [unique_taste_names[t_i] + '\n' + str(best_taste_corr_percents[t_i]) for t_i in range(len(unique_taste_names))]
                    ax_best_taste_counts[ep_i,s_i].pie(best_taste_corr_counts,labels=best_taste_corr_labels)
                else:
                    best_taste_corr_labels = [str(best_taste_corr_percents[t_i]) for t_i in range(len(unique_taste_names))]
                    ax_best_taste_counts[ep_i,s_i].pie(best_taste_corr_counts,labels=best_taste_corr_labels)
                if s_i == 0:
                    ax_best_taste_cdf[ep_i,s_i].set_ylabel('Epoch ' + str(ep_i) + '\nCumulative Density')
                    ax_best_taste_counts[ep_i,s_i].set_ylabel('Epoch ' + str(ep_i))
                if ep_i == max_cp-1:
                    ax_best_taste_cdf[ep_i,s_i].set_xlabel('Pearson Correlation')
                if ep_i == 0:
                    ax_best_taste_cdf[ep_i,s_i].set_title(seg_name)
                    ax_best_taste_counts[ep_i,s_i].set_title(seg_name)
        ax_best_taste_cdf[0,0].legend(loc='upper left')
        plt.figure(f_best_taste_cdf.number)
        plt.suptitle(corr_name + ' Best Corr Distributions')
        plt.tight_layout()
        f_best_taste_cdf.savefig(os.path.join(plot_save_dir,corr_name+'_best_corr_joint_cdf.png'))
        f_best_taste_cdf.savefig(os.path.join(plot_save_dir,corr_name+'_best_corr_joint_cdf.svg'))
        plt.close(f_best_taste_cdf)
        plt.figure(f_best_taste_counts.number)
        plt.suptitle(corr_name + ' Best Corr Distribution Counts')
        plt.tight_layout()
        f_best_taste_counts.savefig(os.path.join(plot_save_dir,corr_name+'_best_corr_pie.png'))
        f_best_taste_counts.savefig(os.path.join(plot_save_dir,corr_name+'_best_corr_pie.svg'))
        plt.close(f_best_taste_counts)
        
    #Create the same plots as above but without epoch separation - combine for tastes
    for corr_name in unique_corr_names:
        f_best_taste_cdf, ax_best_taste_cdf = plt.subplots(nrows=1,ncols=len(unique_segment_names),
                                                 sharex = True, sharey = True, figsize = (8,8))
        f_best_taste_counts, ax_best_taste_counts = plt.subplots(nrows=1,ncols=len(unique_segment_names),
                                                 sharex = True, sharey = True, figsize = (8,8))
        for s_i, seg_name in enumerate(unique_segment_names):
            taste_corrs = []
            for taste_name in unique_taste_names:
                this_taste_corrs = []
                for data_name in unique_given_names:
                    best_data = corr_dict[data_name][corr_name][seg_name]['best']
                    taste_names_data = corr_dict[data_name][corr_name]['tastes']
                    taste_name_ind = np.where(np.array(taste_names_data) == taste_name)[0]
                    if len(taste_name_ind) > 0: #Taste exists in data
                        best_dev_taste_inds = np.where(best_data[:,0] == taste_name_ind)[0]
                        for ep_i in range(max_cp):
                            best_dev_epoch_inds = np.where(best_data[:,1] == ep_i)[0]
                            joint_inds = np.intersect1d(best_dev_taste_inds,best_dev_epoch_inds)
                            this_taste_corrs.extend(list(best_data[joint_inds,2]))
                taste_corrs.append(this_taste_corrs)
                ax_best_taste_cdf[s_i].hist(this_taste_corrs,bins=1000,
                                                density=True,cumulative=True,
                                                histtype='step',label=taste_name)
            best_taste_corr_counts = np.array([len(tc) for tc in taste_corrs])
            best_taste_corr_percents = np.round(100*best_taste_corr_counts/np.sum(best_taste_corr_counts),2)
            if s_i == 0:
                best_taste_corr_labels = [unique_taste_names[t_i] + '\n' + str(best_taste_corr_percents[t_i]) for t_i in range(len(unique_taste_names))]
                ax_best_taste_counts[s_i].pie(best_taste_corr_counts,labels=best_taste_corr_labels)
            else:
                best_taste_corr_labels = [str(best_taste_corr_percents[t_i]) for t_i in range(len(unique_taste_names))]
                ax_best_taste_counts[s_i].pie(best_taste_corr_counts,labels=best_taste_corr_labels)
            ax_best_taste_cdf[s_i].set_xlabel('Pearson Correlation')
        ax_best_taste_cdf[0].legend(loc='upper left')
        plt.figure(f_best_taste_cdf.number)
        plt.suptitle(corr_name + ' Best Corr Distributions')
        plt.tight_layout()
        f_best_taste_cdf.savefig(os.path.join(plot_save_dir,corr_name+'_best_corr_joint_cdf_all_epochs.png'))
        f_best_taste_cdf.savefig(os.path.join(plot_save_dir,corr_name+'_best_corr_joint_cdf_all_epochs.svg'))
        plt.close(f_best_taste_cdf)
        plt.figure(f_best_taste_counts.number)
        plt.suptitle(corr_name + ' Best Corr Distribution Counts')
        plt.tight_layout()
        f_best_taste_counts.savefig(os.path.join(plot_save_dir,corr_name+'_best_corr_pie_all_epochs.png'))
        f_best_taste_counts.savefig(os.path.join(plot_save_dir,corr_name+'_best_corr_pie_all_epochs.svg'))
        plt.close(f_best_taste_counts)
        
                
    

    
    
    
    
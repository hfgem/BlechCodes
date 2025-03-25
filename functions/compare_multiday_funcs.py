#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:32:10 2025

@author: Hannah Germaine

A collection of functions for comparing different datasets against each other 
in their multi-day correlation and decoding data.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

def compare_corr_data(corr_dict, multiday_data_dict, unique_given_names,
                      unique_corr_names, unique_segment_names, unique_taste_names, 
                      max_cp, save_dir):
    
    corr_results_save_dir = os.path.join(save_dir,'Correlations')
    if not os.path.isdir(corr_results_save_dir):
        os.mkdir(corr_results_save_dir)
        
    #Create corr plots by segment
    # plot_corr_by_segment(corr_dict, unique_given_names, unique_corr_names,
    #                          unique_segment_names, unique_taste_names, max_cp,
    #                          corr_results_save_dir)
    
    plot_corr_cutoff_by_segment(corr_dict, unique_given_names, unique_corr_names,
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
        
def plot_corr_cutoff_by_segment(corr_dict, unique_given_names, unique_corr_names,
                         unique_segment_names, unique_taste_names, max_cp,
                         plot_save_dir):
    
    """Plot number of events above correlation cutoff for each taste"""
    #corr_dict[given_name][corr_name][seg_name]['all'][taste_name] #num_cp x (num_dev x num_deliv)
    
    # Set parameters
    warnings.filterwarnings('ignore')
    colors = ['red','orange','yellow','green','royalblue','purple', \
              'magenta','brown', 'cyan']
    corr_cutoffs = np.round(np.arange(0,1.01,0.01),2)
    
    for corr_name in unique_corr_names:
        #Collect data
        corr_cutoff_dict = dict()
        for seg_name in unique_segment_names:
            corr_cutoff_dict[seg_name] = dict()
            for t_i, taste in enumerate(unique_taste_names):
                corr_cutoff_dict[seg_name][taste] = dict()
                for cp_i in range(max_cp):
                    corr_cutoff_dict[seg_name][taste][cp_i] = dict()
                    corr_cutoff_dict[seg_name][taste][cp_i]['counts'] = []
                    corr_cutoff_dict[seg_name][taste][cp_i]['num_dev'] = []
                #By animal
                for g_n in unique_given_names:
                    try:
                        data = corr_dict[g_n][corr_name][seg_name]['all'][taste]['data']
                        data_cp, num_pts = np.shape(data)
                        num_dev = corr_dict[g_n][corr_name][seg_name]['all'][taste]['num_dev']
                        num_deliv = int(num_pts/num_dev)
                        data_reshape = np.reshape(data,(data_cp,num_deliv,num_dev))
                        deliv_means = np.squeeze(np.nanmean(data_reshape,1))
                        for cp_i in range(max_cp):
                            corr_cutoff_counts = [len(np.where(deliv_means[cp_i,:] >= cc)[0]) for cc in corr_cutoffs]
                            corr_cutoff_dict[seg_name][taste][cp_i]['counts'].append(corr_cutoff_counts)
                            corr_cutoff_dict[seg_name][taste][cp_i]['num_dev'].append(num_dev)
                    except:
                        no_data = 1 #holder for missing data for given taste
                #Total
                for cp_i in range(max_cp):
                    data_nansum = np.nansum(np.array(corr_cutoff_dict[seg_name][taste][cp_i]['counts']),0)
                    data_frac = data_nansum/np.nansum(np.array(corr_cutoff_dict[seg_name][taste][cp_i]['num_dev']))
                    corr_cutoff_dict[seg_name][taste][cp_i]['total_frac'] = data_frac
        
        #Plot results comparing tastes
        f_cutoff_tastes, ax_cutoff_tastes = plt.subplots(nrows = len(unique_segment_names),\
                                                         ncols = max_cp, figsize=(8,8),\
                                                             sharey = True, sharex = True)
        f_cutoff_tastes_notext, ax_cutoff_tastes_notext = plt.subplots(nrows = len(unique_segment_names),\
                                                         ncols = max_cp, figsize=(8,8),\
                                                             sharey = True, sharex = True)
        f_cutoff_tastes_zoom, ax_cutoff_tastes_zoom = plt.subplots(nrows = len(unique_segment_names),\
                                                         ncols = max_cp, figsize=(8,8),\
                                                             sharey = True, sharex = True)
        above_0_5 = np.where(corr_cutoffs >= 0.5)[0][0]
        for s_i, seg_name in enumerate(unique_segment_names):
            for cp_i in range(max_cp):
                for t_i, taste in enumerate(unique_taste_names):
                    taste_frac = corr_cutoff_dict[seg_name][taste][cp_i]['total_frac']
                    ax_cutoff_tastes[s_i,cp_i].plot(corr_cutoffs,taste_frac,\
                                                    color=colors[t_i],label=taste)
                    ax_cutoff_tastes_notext[s_i,cp_i].plot(corr_cutoffs,taste_frac,\
                                                    color=colors[t_i],label=taste)
                    ax_cutoff_tastes_zoom[s_i,cp_i].plot(corr_cutoffs[above_0_5:],\
                                                    taste_frac[above_0_5:],\
                                                    color=colors[t_i],label=taste)
                    frac_0 = corr_cutoffs[np.where(taste_frac == 0)[0]]
                    if len(frac_0) > 0:
                        ax_cutoff_tastes[s_i,cp_i].axvline(frac_0[0],color=colors[t_i],\
                                                           linestyle='dashed',alpha=0.3,\
                                                               label='_')
                        ax_cutoff_tastes_notext[s_i,cp_i].axvline(frac_0[0],color=colors[t_i],\
                                                           linestyle='dashed',alpha=0.3,\
                                                               label='_')
                        ax_cutoff_tastes_zoom[s_i,cp_i].axvline(frac_0[0],color=colors[t_i],\
                                                           linestyle='dashed',alpha=0.3,\
                                                               label='_')
                        ax_cutoff_tastes[s_i,cp_i].text(frac_0[0],0.5+np.random.rand(1)/10,\
                                                        str(np.round(frac_0[0],2)),rotation=90)
                if s_i == 0:
                    ax_cutoff_tastes[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                    ax_cutoff_tastes_notext[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                    ax_cutoff_tastes_zoom[s_i,cp_i].set_title('Epoch ' + str(cp_i))
                if cp_i == 0:
                    ax_cutoff_tastes[s_i,cp_i].set_ylabel(seg_name + '\nFraction of Events')
                    ax_cutoff_tastes_notext[s_i,cp_i].set_ylabel(seg_name + '\nFraction of Events')
                    ax_cutoff_tastes_zoom[s_i,cp_i].set_ylabel(seg_name + '\nFraction of Events')
                if cp_i == max_cp-1:
                    ax_cutoff_tastes[s_i,cp_i].set_xlabel('Correlation Cutoff')
                    ax_cutoff_tastes_notext[s_i,cp_i].set_xlabel('Correlation Cutoff')
                    ax_cutoff_tastes_zoom[s_i,cp_i].set_xlabel('Correlation Cutoff')
        ax_cutoff_tastes[0,0].legend(loc='upper left')
        plt.suptitle('Multiday Frac Events by Cutoff')
        plt.tight_layout()
        f_cutoff_tastes.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes.png'))
        f_cutoff_tastes.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes.svg'))
        plt.close(f_cutoff_tastes)
        ax_cutoff_tastes_notext[0,0].legend(loc='upper left')
        plt.suptitle('Multiday Frac Events by Cutoff')
        plt.tight_layout()
        f_cutoff_tastes_notext.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_notext.png'))
        f_cutoff_tastes_notext.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_notext.svg'))
        plt.close(f_cutoff_tastes_notext)
        ax_cutoff_tastes_zoom[0,0].legend(loc='upper left')
        ax_cutoff_tastes_zoom[0,0].set_xlim([0.49,1.01])
        plt.suptitle('Multiday Frac Events by Cutoff')
        plt.tight_layout()
        f_cutoff_tastes_zoom.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_zoom.png'))
        f_cutoff_tastes_zoom.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_zoom.svg'))
        plt.close(f_cutoff_tastes_zoom)
        
        Plot results comparing epochs
        f_cutoff_epochs, ax_cutoff_epochs = plt.subplots(nrows = len(unique_segment_names),\
                                                          ncols = len(unique_taste_names), figsize=(8,8),\
                                                              sharey = True, sharex = True)
        for s_i, seg_name in enumerate(unique_segment_names):
            for t_i, taste in enumerate(unique_taste_names):
                for cp_i in range(max_cp):
                    taste_frac = corr_cutoff_dict[seg_name][taste][cp_i]['total_frac']
                    ax_cutoff_epochs[s_i,t_i].plot(corr_cutoffs,taste_frac,\
                                                    color=colors[cp_i],\
                                                        label='Epoch ' + str(cp_i))
                    frac_0 = corr_cutoffs[np.where(taste_frac == 0)[0]]
                    if len(frac_0) > 0:
                        ax_cutoff_epochs[s_i,t_i].axvline(frac_0[0],color=colors[cp_i],\
                                                            linestyle='dashed',alpha=0.3,\
                                                                label='_')
                if s_i == 0:
                    ax_cutoff_epochs[s_i,t_i].set_title(taste)
                if cp_i == 0:
                    ax_cutoff_epochs[s_i,t_i].set_ylabel(seg_name + '\nFraction of Events')
                if cp_i == max_cp-1:
                    ax_cutoff_epochs[s_i,t_i].set_xlabel('Correlation Cutoff')
        ax_cutoff_epochs[0,0].legend(loc='upper left')
        plt.suptitle('Multiday Frac Events by Cutoff')
        plt.tight_layout()
        f_cutoff_epochs.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_epochs.png'))
        f_cutoff_epochs.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_epochs.svg'))
        plt.close(f_cutoff_epochs)
    
    
    
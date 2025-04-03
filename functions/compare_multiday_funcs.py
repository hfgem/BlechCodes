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
    
    # plot_corr_cutoff_by_segment(corr_dict, unique_given_names, unique_corr_names,
    #                          unique_segment_names, unique_taste_names, max_cp,
    #                          corr_results_save_dir)
    
    plot_corr_cutoff_composition_tastes(corr_dict, unique_given_names, unique_corr_names,
                              unique_segment_names, unique_taste_names, max_cp,
                              corr_results_save_dir)
    
    # plot_corr_cutoff_composition_epochs(corr_dict, unique_given_names, unique_corr_names,
    #                          unique_segment_names, unique_taste_names, max_cp,
    #                          corr_results_save_dir)
        
    
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
        #Collect data by correlation cutoff
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
                        if taste == 'none_0':
                            cp_means = np.nanmean(deliv_means,0)
                            corr_cutoff_counts = [len(np.where(cp_means >= cc)[0]) for cc in corr_cutoffs]
                            for cp_i in range(max_cp):
                                corr_cutoff_dict[seg_name][taste][cp_i]['counts'].append(corr_cutoff_counts)
                                corr_cutoff_dict[seg_name][taste][cp_i]['num_dev'].append(num_dev)
                        else:
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
        f_cutoff_tastes_day1, ax_cutoff_tastes_day1 = plt.subplots(nrows = len(unique_segment_names),\
                                                         ncols = max_cp, figsize=(8,8),\
                                                             sharey = True, sharex = True)
        f_cutoff_tastes_day2, ax_cutoff_tastes_day2 = plt.subplots(nrows = len(unique_segment_names),\
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
                    frac_0 = corr_cutoffs[np.where(taste_frac == 0)[0]]
                    if s_i == len(unique_segment_names)-1:
                        is_last_row = 1
                    else:
                        is_last_row = 0
                    #Regular plot
                    plot_corr_cutoff_updates(f_cutoff_tastes, ax_cutoff_tastes, \
                                             s_i, cp_i, corr_cutoffs, taste_frac, \
                                             frac_0[0], colors[t_i], taste, seg_name, \
                                            'Epoch ' + str(cp_i), is_last_row, \
                                                'Correlation Cutoff', 1)
                    #No text plot
                    plot_corr_cutoff_updates(f_cutoff_tastes_notext, ax_cutoff_tastes_notext, \
                                             s_i, cp_i, corr_cutoffs, taste_frac, \
                                             frac_0[0], colors[t_i], taste, seg_name, \
                                            'Epoch ' + str(cp_i), is_last_row, \
                                                'Correlation Cutoff', 0)
                    #No text zoom plot
                    plot_corr_cutoff_updates(f_cutoff_tastes_zoom, ax_cutoff_tastes_zoom, \
                                             s_i, cp_i, corr_cutoffs[above_0_5:], taste_frac[above_0_5:], \
                                             frac_0[0], colors[t_i], taste, seg_name, \
                                            'Epoch ' + str(cp_i), is_last_row, \
                                                'Correlation Cutoff', 0)
                    
                    if int(taste.split('_')[1]) == 0: #Day 1 only
                        plot_corr_cutoff_updates(f_cutoff_tastes_day1, ax_cutoff_tastes_day1, \
                                                 s_i, cp_i, corr_cutoffs, taste_frac, \
                                                 frac_0[0], colors[t_i], taste, seg_name, \
                                                'Epoch ' + str(cp_i), is_last_row, \
                                                    'Correlation Cutoff', 0)
                    else: #Day 2 only
                        plot_corr_cutoff_updates(f_cutoff_tastes_day2, ax_cutoff_tastes_day2, \
                                                 s_i, cp_i, corr_cutoffs, taste_frac, \
                                                 frac_0[0], colors[t_i], taste, seg_name, \
                                                'Epoch ' + str(cp_i), is_last_row, \
                                                    'Correlation Cutoff', 0)
        ax_cutoff_tastes[0,0].legend(loc='upper left')
        plt.suptitle('Multiday Frac Events by Cutoff')
        plt.tight_layout()
        f_cutoff_tastes.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes.png'))
        f_cutoff_tastes.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes.svg'))
        plt.close(f_cutoff_tastes)
        ax_cutoff_tastes_day1[0,0].legend(loc='upper left')
        plt.suptitle('Multiday Frac Events by Cutoff')
        plt.tight_layout()
        f_cutoff_tastes_day1.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_day1.png'))
        f_cutoff_tastes_day1.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_day1.svg'))
        plt.close(f_cutoff_tastes_day1)
        ax_cutoff_tastes_day2[0,0].legend(loc='upper left')
        plt.suptitle('Multiday Frac Events by Cutoff')
        plt.tight_layout()
        f_cutoff_tastes_day2.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_day2.png'))
        f_cutoff_tastes_day2.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_day2.svg'))
        plt.close(f_cutoff_tastes_day2)
        ax_cutoff_tastes_notext[0,0].legend(loc='upper left')
        plt.suptitle('Multiday Frac Events by Cutoff')
        plt.tight_layout()
        f_cutoff_tastes_notext.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_notext.png'))
        f_cutoff_tastes_notext.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_notext.svg'))
        plt.close(f_cutoff_tastes_notext)
        ax_cutoff_tastes_zoom[0,0].legend(loc='upper left')
        ax_cutoff_tastes_zoom[0,0].set_xlim([0.5,1])
        plt.suptitle('Multiday Frac Events by Cutoff')
        plt.tight_layout()
        f_cutoff_tastes_zoom.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_zoom.png'))
        f_cutoff_tastes_zoom.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_tastes_zoom.svg'))
        plt.close(f_cutoff_tastes_zoom)
        
        #Plot results comparing epochs
        f_cutoff_epochs, ax_cutoff_epochs = plt.subplots(nrows = len(unique_segment_names),\
                                                          ncols = len(unique_taste_names), figsize=(8,8),\
                                                              sharey = True, sharex = True)
        for s_i, seg_name in enumerate(unique_segment_names):
            for t_i, taste in enumerate(unique_taste_names):
                for cp_i in range(max_cp):
                    taste_frac = corr_cutoff_dict[seg_name][taste][cp_i]['total_frac']
                    frac_0 = corr_cutoffs[np.where(taste_frac == 0)[0]]
                    if s_i == len(unique_segment_names)-1:
                        is_last_row = 1
                    else:
                        is_last_row = 0
                    plot_corr_cutoff_updates(f_cutoff_epochs, ax_cutoff_epochs, \
                                             s_i, t_i, corr_cutoffs, taste_frac, \
                                             frac_0[0], colors[cp_i], 'Epoch ' + str(cp_i), \
                                            seg_name, taste, is_last_row, 'Correlation Cutoff', 0)
        ax_cutoff_epochs[0,0].legend(loc='upper left')
        plt.suptitle('Multiday Frac Events by Cutoff')
        plt.tight_layout()
        f_cutoff_epochs.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_epochs.png'))
        f_cutoff_epochs.savefig(os.path.join(plot_save_dir,corr_name+'_corr_cutoff_epochs.svg'))
        plt.close(f_cutoff_epochs)
    
    
def plot_corr_cutoff_updates(fig_name, ax_name, ax_r, ax_c, x_vals, y_vals, 
                             vert_ind, c_name, l_text, row_name, col_name, 
                             is_last_row, x_name, add_text):
    plt.figure(fig_name)
    ax_name[ax_r,ax_c].plot(x_vals,y_vals,color=c_name,label=l_text)
    ax_name[ax_r,ax_c].axvline(vert_ind,color=c_name,linestyle='dashed',\
                               alpha=0.3,label='_')
    if add_text == 1:
        ax_name[ax_r,ax_c].text(vert_ind,0.5+np.random.rand(1)/10,\
                                str(np.round(vert_ind,2)),rotation=90)
    if ax_r == 0:
        ax_name[ax_r,ax_c].set_title(col_name)
    if ax_c == 0:
        ax_name[ax_r,ax_c].set_ylabel(row_name + '\nFraction of Events')
    if is_last_row == 1:
        ax_name[ax_r,ax_c].set_xlabel(x_name)
        
def plot_corr_cutoff_composition_tastes(corr_dict, unique_given_names, unique_corr_names,
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
        #Collect cutoff indices            
        dev_corr_ind_dict = dict()
        for seg_name in unique_segment_names:
            dev_corr_ind_dict[seg_name] = dict()
            for t_i, taste in enumerate(unique_taste_names):
                dev_corr_ind_dict[seg_name][taste] = dict()
                for cp_i in range(max_cp):        
                    dev_corr_ind_dict[seg_name][taste][cp_i] = []
        for seg_name in unique_segment_names:
            for t_i, taste in enumerate(unique_taste_names):
                try:
                    for cp_i in range(max_cp):
                        animal_inds = dict()
                        for g_n in unique_given_names:
                            animal_inds[g_n] = []
                            data = corr_dict[g_n][corr_name][seg_name]['all'][taste]['data']
                            data_cp, num_pts = np.shape(data)
                            num_dev = corr_dict[g_n][corr_name][seg_name]['all'][taste]['num_dev']
                            num_deliv = int(num_pts/num_dev)
                            data_reshape = np.reshape(data,(data_cp,num_deliv,num_dev))
                            deliv_means = np.squeeze(np.nanmean(data_reshape,1)) #num_cp x num_dev
                            if taste == 'none_0':
                                cp_means = np.nanmean(deliv_means,0)
                                corr_cut_inds = [np.where(cp_means >= cc)[0] for cc in corr_cutoffs]
                                animal_inds[g_n] = corr_cut_inds
                            else:
                                corr_cut_inds = [np.where(deliv_means[cp_i,:] >= cc)[0] for cc in corr_cutoffs]
                                animal_inds[g_n] = corr_cut_inds
                        dev_corr_ind_dict[seg_name][taste][cp_i] = animal_inds
                except:
                    print("No data.")
                    
        #Both days
        f_frac, ax_frac = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        f_count, ax_count = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        unique_taste_inds = dict() #seg_name x cp_i x cc x gn x taste
        for s_i, seg_name in enumerate(unique_segment_names):
            cp_taste_counts = dict()
            seg_unique_taste_inds = dict()
            for cp_i in range(max_cp):
                taste_inds = dict()
                for t_i, taste in enumerate(unique_taste_names):
                    taste_inds[taste] = dev_corr_ind_dict[seg_name][taste][cp_i]
                taste_fracs_dict = dict()
                taste_counts_dict = dict()
                epoch_unique_taste_inds = dict()
                for taste in unique_taste_names:
                    taste_fracs_dict[taste] = []
                    taste_counts_dict[taste] = []
                for cc_i, cc in enumerate(corr_cutoffs):
                    cc_taste_counts_dict = dict()
                    cc_taste_inds_dict = dict()
                    for t_i, taste in enumerate(unique_taste_names):
                        cc_taste_counts_dict[taste] = 0
                    cc_taste_counts_dict['all'] = 0
                    for gn_i, gn in enumerate(unique_given_names):
                        animal_unique_inds = dict()
                        for t_i, taste in enumerate(unique_taste_names):
                            try:
                                animal_inds[taste] = taste_inds[taste][gn][cc_i]
                            except:
                                animal_inds[taste] = []
                        for t_i, taste in enumerate(unique_taste_names):
                            other_tastes = np.setdiff1d(unique_taste_names,taste)
                            other_taste_inds = []
                            for ot in other_tastes:
                                other_taste_inds.extend(animal_inds[ot])
                            unique_inds = np.unique(np.setdiff1d(animal_inds[taste],other_taste_inds))
                            animal_unique_inds[taste] = unique_inds
                            cc_taste_counts_dict[taste] += len(unique_inds)
                        cc_taste_counts_dict['all'] += np.sum([len(animal_inds[ai]) for ai in animal_inds])
                        cc_taste_inds_dict[gn] = animal_unique_inds
                    epoch_unique_taste_inds[cc] = cc_taste_inds_dict
                    for taste in unique_taste_names:
                        taste_fracs_dict[taste].extend([cc_taste_counts_dict[taste]/cc_taste_counts_dict['all']])
                        taste_counts_dict[taste].append(cc_taste_counts_dict[taste])
                seg_unique_taste_inds[cp_i] = epoch_unique_taste_inds
                #Plot unique counts and fractions
                for t_i, taste in enumerate(unique_taste_names):
                    ax_frac[s_i, cp_i].plot(corr_cutoffs,taste_fracs_dict[taste],label=taste + ' only',
                                            color=colors[t_i])
                    ax_count[s_i, cp_i].plot(corr_cutoffs,taste_counts_dict[taste],label=taste + ' only',
                                            color=colors[t_i])
                if s_i == 0:
                    ax_frac[s_i, cp_i].set_title('Epoch ' + str(cp_i))
                    ax_count[s_i, cp_i].set_title('Epoch ' + str(cp_i))
                if s_i == len(unique_segment_names)-1:
                    ax_frac[s_i, cp_i].set_xlabel('Correlation Cutoff')
                    ax_count[s_i, cp_i].set_xlabel('Correlation Cutoff')
                if cp_i == 0:
                    ax_frac[s_i, cp_i].set_ylabel(seg_name + '\nFraction of Events Above Cutoff')
                    ax_count[s_i, cp_i].set_ylabel(seg_name + '\nNumber of Events Above Cutoff')
            unique_taste_inds[seg_name] = seg_unique_taste_inds
        plt.figure(f_frac)
        ax_frac[0,0].legend(loc='upper left')
        ax_frac[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Fraction of Events Above Cutoff')
        plt.tight_layout()
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff.svg'))
        ax_frac[0,0].set_xlim([0.5,1])
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff_zoom.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff_zoom.svg'))
        plt.close(f_frac)
        plt.figure(f_count)
        ax_count[0,0].legend(loc='upper left')
        ax_count[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Number of Events Above Cutoff')
        plt.tight_layout()
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff.svg'))
        ax_count[0,0].set_xlim([0.5,0.8])
        ax_count[0,0].set_ylim([0,150])
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff_zoom.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff_zoom.svg'))
        plt.close(f_count) 
        
        #Plot the unique epochs for unique tastes
        #unique_taste_inds seg_name x cp_i x cc x gn x taste
        f_count_epoch, ax_count_epoch = plt.subplots(nrows = len(unique_segment_names),\
                                                     ncols = len(unique_taste_names), \
                                                    figsize = (len(unique_taste_names)*4,len(unique_segment_names)*4), \
                                                        sharex = True, sharey = True)
        f_frac_epoch, ax_frac_epoch = plt.subplots(nrows = len(unique_segment_names),\
                                                     ncols = len(unique_taste_names), \
                                                    figsize = (len(unique_taste_names)*4,len(unique_segment_names)*4), \
                                                        sharex = True, sharey = True)
        for s_i, seg_name in enumerate(unique_segment_names):
            seg_unique_inds = unique_taste_inds[seg_name]
            for t_i, taste in enumerate(unique_taste_names):
                corr_cutoff_epoch_counts = dict()
                for cc_i, cc in enumerate(corr_cutoffs):
                    animal_epoch_unique_counts = dict()
                    for cp_i in range(max_cp):
                        animal_epoch_unique_counts[cp_i] = 0
                    for gn_i, gn in enumerate(unique_given_names):
                        all_epoch_inds = []
                        epoch_inds = []
                        for cp_i in range(max_cp):
                            taste_unique_inds = list(seg_unique_inds[cp_i][cc][gn][taste])
                            all_epoch_inds.extend(taste_unique_inds)
                            epoch_inds.append(taste_unique_inds)
                        all_epoch_inds = np.unique(all_epoch_inds)
                        for cp_i in range(max_cp):
                            other_epochs= np.setdiff1d(np.arange(max_cp),cp_i*np.ones(1))
                            other_inds = []
                            for oe_i in other_epochs:
                                other_inds.extend(epoch_inds[oe_i])
                            other_inds = np.unique(other_inds)
                            animal_epoch_unique_counts[cp_i] += len(np.unique(np.setdiff1d(epoch_inds[cp_i],other_inds)))
                    corr_cutoff_epoch_counts[cc] = animal_epoch_unique_counts
                for cp_i in range(max_cp):
                    plot_y_vals = []
                    for cc_i, cc in enumerate(corr_cutoffs):
                        plot_y_vals.append(corr_cutoff_epoch_counts[cc][cp_i])
                    ax_count_epoch[s_i,t_i].plot(corr_cutoffs,plot_y_vals,\
                                                 label='Epoch ' + str(cp_i),\
                                                     color=colors[cp_i])
                if t_i == 0:
                    ax_count_epoch[s_i,t_i].set_ylabel(seg_name + '\nNumber of Events')
                if s_i == 0:
                    ax_count_epoch[s_i,t_i].set_title(taste)
                if s_i == len(unique_segment_names)-1:
                    ax_count_epoch[s_i,t_i].set_xlabel('Correlation Cutoff')
        ax_count_epoch[0,0].legend(loc='upper left')
        plt.suptitle('Unique Epoch Counts for Unique Taste Events')
        plt.tight_layout()
        f_count_epoch.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff_unique_epochs.png'))
        f_count_epoch.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff_unique_epochs.svg'))
        plt.close(f_count_epoch)
        
        #Day 1 Only
        f_frac, ax_frac = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        f_count, ax_count = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True)
        for s_i, seg_name in enumerate(unique_segment_names):
            for cp_i in range(max_cp):
                taste_inds = dict()
                for t_i, taste in enumerate(unique_taste_names):
                    if int(taste.split('_')[1]) == 0:
                        taste_inds[taste] = dev_corr_ind_dict[seg_name][taste][cp_i]
                taste_fracs_dict = dict()
                taste_counts_dict = dict()
                for taste in unique_taste_names:
                    if int(taste.split('_')[1]) == 0:
                        taste_fracs_dict[taste] = []
                        taste_counts_dict[taste] = []
                for cc_i, cc in enumerate(corr_cutoffs):
                    cc_taste_counts_dict = dict()
                    for t_i, taste in enumerate(unique_taste_names):
                        if int(taste.split('_')[1]) == 0:
                            cc_taste_counts_dict[taste] = 0
                    cc_taste_counts_dict['all'] = 0
                    for gn_i, gn in enumerate(unique_given_names):
                        animal_inds = dict()
                        for t_i, taste in enumerate(unique_taste_names):
                            if int(taste.split('_')[1]) == 0:
                                try:
                                    animal_inds[taste] = taste_inds[taste][gn][cc_i]
                                except:
                                    animal_inds[taste] = []
                        for t_i, taste in enumerate(unique_taste_names):
                            if int(taste.split('_')[1]) == 0:
                                other_tastes = np.setdiff1d(unique_taste_names,taste)
                                other_taste_inds = []
                                for ot in other_tastes:
                                    try:
                                        other_taste_inds.extend(animal_inds[ot])
                                    except:
                                        do_nothing = 1
                                cc_taste_counts_dict[taste] += len(np.setdiff1d(animal_inds[taste],other_taste_inds))
                        cc_taste_counts_dict['all'] += np.sum([len(animal_inds[ai]) for ai in animal_inds])
                    for taste in unique_taste_names:
                        if int(taste.split('_')[1]) == 0:
                            taste_fracs_dict[taste].extend([cc_taste_counts_dict[taste]/cc_taste_counts_dict['all']])
                            taste_counts_dict[taste].append(cc_taste_counts_dict[taste])
                #Plot unique counts and fractions
                for t_i, taste in enumerate(unique_taste_names):
                    if int(taste.split('_')[1]) == 0:
                        ax_frac[s_i, cp_i].plot(corr_cutoffs,taste_fracs_dict[taste],label=taste + ' only',
                                                color=colors[t_i])
                        ax_count[s_i, cp_i].plot(corr_cutoffs,taste_counts_dict[taste],label=taste + ' only',
                                                color=colors[t_i])
                if s_i == 0:
                    ax_frac[s_i, cp_i].set_title('Epoch ' + str(cp_i))
                    ax_count[s_i, cp_i].set_title('Epoch ' + str(cp_i))
                if s_i == len(unique_segment_names)-1:
                    ax_frac[s_i, cp_i].set_xlabel('Correlation Cutoff')
                    ax_count[s_i, cp_i].set_xlabel('Correlation Cutoff')
                if cp_i == 0:
                    ax_frac[s_i, cp_i].set_ylabel(seg_name + '\nFraction of Events Above Cutoff')
                    ax_count[s_i, cp_i].set_ylabel(seg_name + '\nNumber of Events Above Cutoff')
        plt.figure(f_frac)
        ax_frac[0,0].legend(loc='upper left')
        ax_frac[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Fraction of Events Above Cutoff')
        plt.tight_layout()
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff_day1.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff_day1.svg'))
        plt.close(f_frac)
        plt.figure(f_count)
        ax_count[0,0].legend(loc='upper left')
        ax_count[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Number of Events Above Cutoff')
        plt.tight_layout()
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff_day1.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff_day1.svg'))
        plt.close(f_count)      

        #Day 2 Only
        f_frac, ax_frac = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True, sharey = True)
        f_count, ax_count = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = max_cp, figsize=(8,8),\
                                    sharex = True)
        for s_i, seg_name in enumerate(unique_segment_names):
            for cp_i in range(max_cp):
                taste_inds = dict()
                for t_i, taste in enumerate(unique_taste_names):
                    if int(taste.split('_')[1]) == 1:
                        taste_inds[taste] = dev_corr_ind_dict[seg_name][taste][cp_i]
                taste_fracs_dict = dict()
                taste_counts_dict = dict()
                for taste in unique_taste_names:
                    if int(taste.split('_')[1]) == 1:
                        taste_fracs_dict[taste] = []
                        taste_counts_dict[taste] = []
                for cc_i, cc in enumerate(corr_cutoffs):
                    cc_taste_counts_dict = dict()
                    for t_i, taste in enumerate(unique_taste_names):
                        if int(taste.split('_')[1]) == 1:
                            cc_taste_counts_dict[taste] = 0
                    cc_taste_counts_dict['all'] = 0
                    for gn_i, gn in enumerate(unique_given_names):
                        animal_inds = dict()
                        for t_i, taste in enumerate(unique_taste_names):
                            if int(taste.split('_')[1]) == 1:
                                try:
                                    animal_inds[taste] = taste_inds[taste][gn][cc_i]
                                except:
                                    animal_inds[taste] = []
                        for t_i, taste in enumerate(unique_taste_names):
                            if int(taste.split('_')[1]) == 1:
                                other_tastes = np.setdiff1d(unique_taste_names,taste)
                                other_taste_inds = []
                                for ot in other_tastes:
                                    try:
                                        other_taste_inds.extend(animal_inds[ot])
                                    except:
                                        do_nothing = 1
                                cc_taste_counts_dict[taste] += len(np.setdiff1d(animal_inds[taste],other_taste_inds))
                        cc_taste_counts_dict['all'] += np.sum([len(animal_inds[ai]) for ai in animal_inds])
                    for taste in unique_taste_names:
                        if int(taste.split('_')[1]) == 1:
                            taste_fracs_dict[taste].extend([cc_taste_counts_dict[taste]/cc_taste_counts_dict['all']])
                            taste_counts_dict[taste].append(cc_taste_counts_dict[taste])
                #Plot unique counts and fractions
                for t_i, taste in enumerate(unique_taste_names):
                    if int(taste.split('_')[1]) == 1:
                        ax_frac[s_i, cp_i].plot(corr_cutoffs,taste_fracs_dict[taste],label=taste + ' only',
                                                color=colors[t_i])
                        ax_count[s_i, cp_i].plot(corr_cutoffs,taste_counts_dict[taste],label=taste + ' only',
                                                color=colors[t_i])
                if s_i == 0:
                    ax_frac[s_i, cp_i].set_title('Epoch ' + str(cp_i))
                    ax_count[s_i, cp_i].set_title('Epoch ' + str(cp_i))
                if s_i == len(unique_segment_names)-1:
                    ax_frac[s_i, cp_i].set_xlabel('Correlation Cutoff')
                    ax_count[s_i, cp_i].set_xlabel('Correlation Cutoff')
                if cp_i == 0:
                    ax_frac[s_i, cp_i].set_ylabel(seg_name + '\nFraction of Events Above Cutoff')
                    ax_count[s_i, cp_i].set_ylabel(seg_name + '\nNumber of Events Above Cutoff')
        plt.figure(f_frac)
        ax_frac[0,0].legend(loc='upper left')
        ax_frac[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Fraction of Events Above Cutoff')
        plt.tight_layout()
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff_day2.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_taste_by_cutoff_day2.svg'))
        plt.close(f_frac)
        plt.figure(f_count)
        ax_count[0,0].legend(loc='upper left')
        ax_count[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Number of Events Above Cutoff')
        plt.tight_layout()
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff_day2.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_taste_by_cutoff_day2.svg'))
        plt.close(f_count)                
        
def plot_corr_cutoff_composition_epochs(corr_dict, unique_given_names, unique_corr_names,
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
        #Collect cutoff indices            
        dev_corr_ind_dict = dict()
        for seg_name in unique_segment_names:
            dev_corr_ind_dict[seg_name] = dict()
            for t_i, taste in enumerate(unique_taste_names):
                dev_corr_ind_dict[seg_name][taste] = dict()
                for cp_i in range(max_cp):        
                    dev_corr_ind_dict[seg_name][taste][cp_i] = []
        for seg_name in unique_segment_names:
            for t_i, taste in enumerate(unique_taste_names):
                try:
                    for cp_i in range(max_cp):
                        animal_inds = dict()
                        for g_n in unique_given_names:
                            animal_inds[g_n] = []
                            data = corr_dict[g_n][corr_name][seg_name]['all'][taste]['data']
                            data_cp, num_pts = np.shape(data)
                            num_dev = corr_dict[g_n][corr_name][seg_name]['all'][taste]['num_dev']
                            num_deliv = int(num_pts/num_dev)
                            data_reshape = np.reshape(data,(data_cp,num_deliv,num_dev))
                            deliv_means = np.squeeze(np.nanmean(data_reshape,1)) #num_cp x num_dev
                            if taste == 'none_0':
                                cp_means = np.nanmean(deliv_means,0)
                                corr_cut_inds = [np.where(cp_means >= cc)[0] for cc in corr_cutoffs]
                                animal_inds[g_n] = corr_cut_inds
                            else:
                                corr_cut_inds = [np.where(deliv_means[cp_i,:] >= cc)[0] for cc in corr_cutoffs]
                                animal_inds[g_n] = corr_cut_inds
                        dev_corr_ind_dict[seg_name][taste][cp_i] = animal_inds
                except:
                    print("No data.")
                    
        #Both days
        f_frac, ax_frac = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = len(unique_taste_names), figsize=(4*len(unique_taste_names),4*len(unique_segment_names)),\
                                    sharex = True, sharey = True)
        f_count, ax_count = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = len(unique_taste_names), figsize=(4*len(unique_taste_names),4*len(unique_segment_names)),\
                                    sharex = True)
        for s_i, seg_name in enumerate(unique_segment_names):
            for t_i, taste in enumerate(unique_taste_names):
                epoch_inds = dict()
                for cp_i in range(max_cp):
                    epoch_inds[cp_i] = dev_corr_ind_dict[seg_name][taste][cp_i]
                epoch_fracs_dict = dict()
                epoch_counts_dict = dict()
                for cp_i in range(max_cp):
                    epoch_fracs_dict[cp_i] = []
                    epoch_counts_dict[cp_i] = []
                for cc_i, cc in enumerate(corr_cutoffs):
                    cc_epoch_counts_dict = dict()
                    for cp_i in range(max_cp):
                        cc_epoch_counts_dict[cp_i] = 0
                    cc_epoch_counts_dict['all'] = 0
                    for gn_i, gn in enumerate(unique_given_names):
                        animal_inds = dict()
                        for cp_i in range(max_cp):
                            try:
                                animal_inds[cp_i] = epoch_inds[cp_i][gn][cc_i]
                            except:
                                animal_inds[cp_i] = []
                        for cp_i in range(max_cp):
                            other_epochs = np.setdiff1d(np.arange(max_cp),cp_i)
                            other_epoch_inds = []
                            for ocp_i in other_epochs:
                                other_epoch_inds.extend(animal_inds[ocp_i])
                            cc_epoch_counts_dict[cp_i] += len(np.setdiff1d(animal_inds[cp_i],other_epoch_inds))
                        cc_epoch_counts_dict['all'] += np.sum([len(animal_inds[ai]) for ai in animal_inds])
                    for cp_i in range(max_cp):
                        epoch_fracs_dict[cp_i].extend([cc_epoch_counts_dict[cp_i]/cc_epoch_counts_dict['all']])
                        epoch_counts_dict[cp_i].append(cc_epoch_counts_dict[cp_i])
                #Plot unique counts and fractions
                for cp_i in range(max_cp):
                    ax_frac[s_i, t_i].plot(corr_cutoffs,epoch_fracs_dict[cp_i],label='Epoch ' + str(cp_i) + ' only',
                                            color=colors[cp_i])
                    ax_count[s_i, t_i].plot(corr_cutoffs,epoch_counts_dict[cp_i],label='Epoch ' + str(cp_i) + ' only',
                                            color=colors[cp_i])
                if s_i == 0:
                    ax_frac[s_i, t_i].set_title(taste)
                    ax_count[s_i, t_i].set_title(taste)
                if s_i == len(unique_segment_names)-1:
                    ax_frac[s_i, t_i].set_xlabel('Correlation Cutoff')
                    ax_count[s_i, t_i].set_xlabel('Correlation Cutoff')
                if cp_i == 0:
                    ax_frac[s_i, t_i].set_ylabel(seg_name + '\nFraction of Events Above Cutoff')
                    ax_count[s_i, t_i].set_ylabel(seg_name + '\nNumber of Events Above Cutoff')
        plt.figure(f_frac)
        ax_frac[0,0].legend(loc='upper left')
        ax_frac[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Fraction of Events Above Cutoff')
        plt.tight_layout()
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff.svg'))
        ax_frac[0,0].set_xlim([0.5,1])
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff_zoom.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff_zoom.svg'))
        plt.close(f_frac)
        plt.figure(f_count)
        ax_count[0,0].legend(loc='upper left')
        ax_count[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Number of Events Above Cutoff')
        plt.tight_layout()
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff.svg'))
        ax_count[0,0].set_xlim([0.5,1])
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff_zoom.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff_zoom.svg'))
        plt.close(f_count) 

        #Day 1 Only
        day_1_tastes = [tn for tn in unique_taste_names if int(tn.split('_')[1]) == 0]
        f_frac, ax_frac = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = len(day_1_tastes), \
                                           figsize=(4*len(day_1_tastes),4*len(unique_segment_names)),\
                                    sharex = True, sharey = True)
        f_count, ax_count = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = len(day_1_tastes), \
                                           figsize=(4*len(day_1_tastes),4*len(unique_segment_names)),\
                                    sharex = True)
        for s_i, seg_name in enumerate(unique_segment_names):
            for t_i, taste in enumerate(day_1_tastes):
                epoch_inds = dict()
                for cp_i in range(max_cp):
                    epoch_inds[cp_i] = dev_corr_ind_dict[seg_name][taste][cp_i]
                epoch_fracs_dict = dict()
                epoch_counts_dict = dict()
                for cp_i in range(max_cp):
                    epoch_fracs_dict[cp_i] = []
                    epoch_counts_dict[cp_i] = []
                for cc_i, cc in enumerate(corr_cutoffs):
                    cc_epoch_counts_dict = dict()
                    for cp_i in range(max_cp):
                        cc_epoch_counts_dict[cp_i] = 0
                    cc_epoch_counts_dict['all'] = 0
                    for gn_i, gn in enumerate(unique_given_names):
                        animal_inds = dict()
                        for cp_i in range(max_cp):
                            try:
                                animal_inds[cp_i] = epoch_inds[cp_i][gn][cc_i]
                            except:
                                animal_inds[cp_i] = []
                        for cp_i in range(max_cp):
                            other_epochs = np.setdiff1d(np.arange(max_cp),cp_i)
                            other_epoch_inds = []
                            for ocp_i in other_epochs:
                                other_epoch_inds.extend(animal_inds[ocp_i])
                            cc_epoch_counts_dict[cp_i] += len(np.setdiff1d(animal_inds[cp_i],other_epoch_inds))
                        cc_epoch_counts_dict['all'] += np.sum([len(animal_inds[ai]) for ai in animal_inds])
                    for cp_i in range(max_cp):
                        epoch_fracs_dict[cp_i].extend([cc_epoch_counts_dict[cp_i]/cc_epoch_counts_dict['all']])
                        epoch_counts_dict[cp_i].append(cc_epoch_counts_dict[cp_i])
                #Plot unique counts and fractions
                for cp_i in range(max_cp):
                    ax_frac[s_i, t_i].plot(corr_cutoffs,epoch_fracs_dict[cp_i],label='Epoch ' + str(cp_i) + ' only',
                                            color=colors[cp_i])
                    ax_count[s_i, t_i].plot(corr_cutoffs,epoch_counts_dict[cp_i],label='Epoch ' + str(cp_i) + ' only',
                                            color=colors[cp_i])
                if s_i == 0:
                    ax_frac[s_i, t_i].set_title(taste)
                    ax_count[s_i, t_i].set_title(taste)
                if s_i == len(unique_segment_names)-1:
                    ax_frac[s_i, t_i].set_xlabel('Correlation Cutoff')
                    ax_count[s_i, t_i].set_xlabel('Correlation Cutoff')
                if cp_i == 0:
                    ax_frac[s_i, t_i].set_ylabel(seg_name + '\nFraction of Events Above Cutoff')
                    ax_count[s_i, t_i].set_ylabel(seg_name + '\nNumber of Events Above Cutoff')
        plt.figure(f_frac)
        ax_frac[0,0].legend(loc='upper left')
        ax_frac[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Fraction of Events Above Cutoff')
        plt.tight_layout()
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff_day1.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff_day1.svg'))
        plt.close(f_frac)
        plt.figure(f_count)
        ax_count[0,0].legend(loc='upper left')
        ax_count[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Number of Events Above Cutoff')
        plt.tight_layout()
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff_day1.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff_day1.svg'))
        plt.close(f_count)     

        #Day 2 Only
        day_2_tastes = [tn for tn in unique_taste_names if int(tn.split('_')[1]) == 1]
        f_frac, ax_frac = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = len(day_2_tastes), \
                                           figsize=(4*len(day_2_tastes),4*len(unique_segment_names)),\
                                    sharex = True, sharey = True)
        f_count, ax_count = plt.subplots(nrows = len(unique_segment_names),\
                                       ncols = len(day_2_tastes), \
                                       figsize=(4*len(day_2_tastes),4*len(unique_segment_names)),\
                                    sharex = True)
        for s_i, seg_name in enumerate(unique_segment_names):
            for t_i, taste in enumerate(day_2_tastes):
                epoch_inds = dict()
                for cp_i in range(max_cp):
                    epoch_inds[cp_i] = dev_corr_ind_dict[seg_name][taste][cp_i]
                epoch_fracs_dict = dict()
                epoch_counts_dict = dict()
                for cp_i in range(max_cp):
                    epoch_fracs_dict[cp_i] = []
                    epoch_counts_dict[cp_i] = []
                for cc_i, cc in enumerate(corr_cutoffs):
                    cc_epoch_counts_dict = dict()
                    for cp_i in range(max_cp):
                        cc_epoch_counts_dict[cp_i] = 0
                    cc_epoch_counts_dict['all'] = 0
                    for gn_i, gn in enumerate(unique_given_names):
                        animal_inds = dict()
                        for cp_i in range(max_cp):
                            try:
                                animal_inds[cp_i] = epoch_inds[cp_i][gn][cc_i]
                            except:
                                animal_inds[cp_i] = []
                        for cp_i in range(max_cp):
                            other_epochs = np.setdiff1d(np.arange(max_cp),cp_i)
                            other_epoch_inds = []
                            for ocp_i in other_epochs:
                                other_epoch_inds.extend(animal_inds[ocp_i])
                            cc_epoch_counts_dict[cp_i] += len(np.setdiff1d(animal_inds[cp_i],other_epoch_inds))
                        cc_epoch_counts_dict['all'] += np.sum([len(animal_inds[ai]) for ai in animal_inds])
                    for cp_i in range(max_cp):
                        epoch_fracs_dict[cp_i].extend([cc_epoch_counts_dict[cp_i]/cc_epoch_counts_dict['all']])
                        epoch_counts_dict[cp_i].append(cc_epoch_counts_dict[cp_i])
                #Plot unique counts and fractions
                for cp_i in range(max_cp):
                    ax_frac[s_i, t_i].plot(corr_cutoffs,epoch_fracs_dict[cp_i],label='Epoch ' + str(cp_i) + ' only',
                                            color=colors[cp_i])
                    ax_count[s_i, t_i].plot(corr_cutoffs,epoch_counts_dict[cp_i],label='Epoch ' + str(cp_i) + ' only',
                                            color=colors[cp_i])
                if s_i == 0:
                    ax_frac[s_i, t_i].set_title(taste)
                    ax_count[s_i, t_i].set_title(taste)
                if s_i == len(unique_segment_names)-1:
                    ax_frac[s_i, t_i].set_xlabel('Correlation Cutoff')
                    ax_count[s_i, t_i].set_xlabel('Correlation Cutoff')
                if cp_i == 0:
                    ax_frac[s_i, t_i].set_ylabel(seg_name + '\nFraction of Events Above Cutoff')
                    ax_count[s_i, t_i].set_ylabel(seg_name + '\nNumber of Events Above Cutoff')
        plt.figure(f_frac)
        ax_frac[0,0].legend(loc='upper left')
        ax_frac[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Fraction of Events Above Cutoff')
        plt.tight_layout()
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff_day2.png'))
        f_frac.savefig(os.path.join(plot_save_dir,corr_name+'_frac_unique_epoch_by_cutoff_day2.svg'))
        plt.close(f_frac)
        plt.figure(f_count)
        ax_count[0,0].legend(loc='upper left')
        ax_count[0,0].set_xticks(np.arange(0,1.25,0.25))
        plt.suptitle('Number of Events Above Cutoff')
        plt.tight_layout()
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff_day2.png'))
        f_count.savefig(os.path.join(plot_save_dir,corr_name+'_num_unique_epoch_by_cutoff_day2.svg'))
        plt.close(f_count) 
        
def compare_decode_data(decode_dict, multiday_data_dict, unique_given_names,
                       unique_decode_names, unique_segment_names, 
                       unique_taste_names, max_cp, save_dir, verbose):
    
    decode_results_save_dir = os.path.join(save_dir,'Decodes')
    if not os.path.isdir(decode_results_save_dir):
        os.mkdir(decode_results_save_dir)
        
    #Plot cross-animal rates of decodes
    decode_rates_plots(decode_dict,unique_given_names,unique_decode_names,
                           unique_segment_names,unique_taste_names,
                           max_cp,decode_results_save_dir,verbose)
    
    
def decode_rates_plots(decode_dict,unique_given_names,unique_decode_names,
                       unique_segment_names,unique_taste_names,
                       max_cp,decode_results_save_dir,verbose=False):
    
    colors = ['red','orange','yellow','green','royalblue','purple', \
              'magenta','brown', 'cyan']
    num_anim = len(unique_given_names)
    num_seg = len(unique_segment_names)
    num_tastes = len(unique_taste_names)
    unique_segment_names = ['pre-taste','post-taste','sickness'] #manual order override
    
    for dt in unique_decode_names:
        
        #Is-Taste Decode Results
        is_taste_rates = [] #num seg x num anim
        f_box = plt.figure(figsize = (5,5))
        for s_i, sn in enumerate(unique_segment_names):
            seg_is_taste_rates = []
            for gn in unique_given_names:
                try:
                    is_taste_data = decode_dict[gn][dt][sn]['is_taste'] #num_dev x 2
                    num_dev, _ = np.shape(is_taste_data)
                    is_taste_max = np.argmax(is_taste_data,1)
                    num_is_taste = len(np.where(np.array(is_taste_max) == 0)[0])
                    seg_is_taste_rates.append(num_is_taste/num_dev)
                except:
                    seg_is_taste_rates.append(np.nan)
                    errormsg = "Is-Taste data does not exist for " + gn + " " + sn
                    if verbose == True:
                        print(errormsg)
            is_taste_rates.append(seg_is_taste_rates)
            plt.boxplot(seg_is_taste_rates,positions = [s_i])
            x_scat = s_i + np.random.rand(num_anim)/10
            plt.scatter(x_scat,seg_is_taste_rates,color='g',alpha=0.3)
        plt.xticks(np.arange(num_seg),unique_segment_names)
        plt.xlabel('Segment')
        plt.ylim([0,1])
        plt.ylabel('Fraction of Deviation Events')
        plt.title('Fraction of Events Decoded as Taste')
        plt.tight_layout()
        f_box.savefig(os.path.join(decode_results_save_dir,dt + '_is_taste_rates_box.png'))
        f_box.savefig(os.path.join(decode_results_save_dir,dt + '_is_taste_rates_box.svg'))
        plt.close(f_box)
        
        f_line = plt.figure(figsize = (5,5))
        mean_fraction = np.nanmean(np.array(is_taste_rates),1)
        std_fraction = np.nanstd(np.array(is_taste_rates),1)
        plt.fill_between(np.arange(num_seg),mean_fraction-std_fraction,\
                         mean_fraction+std_fraction,color='k',alpha=0.2)
        plt.plot(mean_fraction,color='k')
        plt.xticks(np.arange(num_seg),unique_segment_names)
        plt.xlabel('Segment')
        plt.ylim([0,1])
        plt.ylabel('Fraction of Deviation Events')
        plt.title('Fraction of Events Decoded as Taste')
        plt.tight_layout()
        f_line.savefig(os.path.join(decode_results_save_dir,dt + '_is_taste_rates_line.png'))
        f_line.savefig(os.path.join(decode_results_save_dir,dt + '_is_taste_rates_line.svg'))
        plt.close(f_line)
        
        #Which-Taste Decode Results
        sqrt_taste = np.ceil(np.sqrt(num_tastes)).astype('int')
        sqr_taste = sqrt_taste**2
        not_plot_box = np.setdiff1d(np.arange(sqr_taste),np.arange(num_tastes))
        taste_ind_ref = np.reshape(np.arange(sqrt_taste**2),(sqrt_taste,sqrt_taste))
        epoch_x_labels = np.reshape(np.array([np.arange(max_cp) for s_i in range(num_seg)]),(max_cp*num_seg))
        f_box, ax_box = plt.subplots(nrows = sqrt_taste, ncols = sqrt_taste,\
                                     figsize = (8,8), sharex = True, sharey = True)
        f_line, ax_line = plt.subplots(nrows = sqrt_taste, ncols = sqrt_taste,\
                                     figsize = (8,8), sharex = True, sharey = True)
        f_box_epoch, ax_box_epoch = plt.subplots(nrows = sqrt_taste, ncols = sqrt_taste,\
                                     figsize = (8,8), sharex = True)
        which_taste_rates = dict()
        for t_i, tn in enumerate(unique_taste_names):
            t_plot_ind = np.where(taste_ind_ref == t_i)
            ax_r = t_plot_ind[0][0]
            ax_c = t_plot_ind[1][0]
            which_taste_rates[tn] = dict()
            seg_data = []
            for s_i, sn in enumerate(unique_segment_names):
                which_taste_rates[tn][sn] = dict()
                which_taste_rates[tn][sn]['which_taste'] = []
                for cp_i in range(max_cp):
                    which_taste_rates[tn][sn][cp_i] = []
                for gn in unique_given_names:
                    try:
                        gn_tastes = decode_dict[gn][dt]['tastes']
                        gn_t_i = [i for i in range(len(gn_tastes)) if gn_tastes[i] == tn]
                        if len(gn_t_i) > 0:
                            #Which Taste Data
                            try:
                                which_taste_data = decode_dict[gn][dt][sn]['which_taste'][:,gn_t_i] #num_dev
                                num_dev = len(which_taste_data)
                                gn_dev_inds = np.where(np.array(which_taste_data) == 1)[0]
                                num_which_taste = len(gn_dev_inds)
                                which_taste_rates[tn][sn]['which_taste'].append(num_which_taste/num_dev)
                                try:
                                    #Which Epoch Data
                                    which_epoch_data = decode_dict[gn][dt][sn]['which_epoch'][gn_dev_inds,:] #num_gn_dev x num_cp
                                    which_epoch_counts = np.nansum(np.array(which_epoch_data),0)
                                    for cp_i in range(len(which_epoch_counts)):
                                        which_taste_rates[tn][sn][cp_i].append(which_epoch_counts[cp_i]/num_dev)
                                except:
                                    for cp_i in range(max_cp):
                                        which_taste_rates[tn][sn][cp_i].append(np.nan)
                                    errormsg = "Which-Epoch data does not exist for " + gn + " " + sn + " " + tn
                                    if verbose == True:
                                        print(errormsg)
                            except:
                                which_taste_rates[tn][sn]['which_taste'].append(np.nan)
                                errormsg = "Which-Taste data does not exist for " + gn + " " + sn + " " + tn
                                if verbose == True:
                                    print(errormsg)
                    except:
                        which_taste_rates[tn][sn]['which_taste'].append(np.nan)
                        for cp_i in range(max_cp):
                            which_taste_rates[tn][sn][cp_i].append(np.nan)
                        errormsg = "Missing data for "+ gn + " " + dt
                        if verbose == True:
                            print(errormsg)
                seg_data.append(which_taste_rates[tn][sn]['which_taste'])
                #Plot Which-Taste Data
                t_data = which_taste_rates[tn][sn]['which_taste']
                non_nan_data = np.array(t_data)
                non_nan_data = non_nan_data[~np.isnan(non_nan_data)]
                ax_box[ax_r,ax_c].boxplot(non_nan_data,positions=[s_i])
                x_scat = s_i + np.random.rand(len(t_data))/10
                ax_box[ax_r,ax_c].scatter(x_scat,t_data,color='g',alpha=0.3)
                #Plot Which-Epoch Data
                max_y = 0
                for cp_i in range(max_cp):
                    cp_data = which_taste_rates[tn][sn][cp_i]
                    non_nan_data = np.array(cp_data)
                    non_nan_data = non_nan_data[~np.isnan(non_nan_data)]
                    if len(non_nan_data) > 0:
                        if max(non_nan_data) > max_y:
                            max_y = max(non_nan_data)
                        data_x = s_i*max_cp + cp_i
                        ax_box_epoch[ax_r,ax_c].boxplot(non_nan_data,positions=[data_x])
                        x_scat = data_x + np.random.rand(len(cp_data))/10
                        ax_box_epoch[ax_r,ax_c].scatter(x_scat,cp_data,\
                                                  color='g',alpha=0.3)
                ax_box_epoch[ax_r,ax_c].plot(np.arange(s_i*max_cp,s_i*max_cp+max_cp),\
                                             (max_y+0.1*max_y)*np.ones(max_cp),label=sn,
                                             color = colors[s_i])
            ax_box[ax_r,ax_c].set_xticks(np.arange(num_seg),unique_segment_names,
                                         rotation=45)
            ax_box[ax_r,ax_c].set_title(tn)
            ax_box_epoch[ax_r,ax_c].set_xticks(np.arange(num_seg*max_cp),epoch_x_labels)
            ax_box_epoch[ax_r,ax_c].set_title(tn)
            mean_data = np.nanmean(np.array(seg_data),1)
            std_data = np.nanstd(np.array(seg_data),1)
            ax_line[ax_r,ax_c].plot(np.arange(num_seg),mean_data,color='k')
            ax_line[ax_r,ax_c].fill_between(np.arange(num_seg),mean_data+std_data,\
                                            mean_data-std_data,color='k',alpha=0.2)
            ax_line[ax_r,ax_c].set_xticks(np.arange(num_seg),unique_segment_names,
                                          rotation=45)
            ax_line[ax_r,ax_c].set_title(tn)
        ax_box_epoch[0,0].legend()
        for ax_i in range(sqrt_taste):
            ax_box[sqrt_taste-1,ax_i].set_xlabel('Segment')
            ax_line[sqrt_taste-1,ax_i].set_xlabel('Segment')
            ax_box_epoch[sqrt_taste-1,ax_i].set_xlabel('Segment')
            ax_box[ax_i,0].set_ylabel('Fraction of Events')
            ax_line[ax_i,0].set_ylabel('Fraction of Events')
            ax_box_epoch[ax_i,0].set_ylabel('Fraction of Events')
        for ax_i in not_plot_box:
            ax_r, ax_c = np.where(taste_ind_ref == ax_i)
            ax_box[ax_r[0],ax_c[0]].axis('off')
            ax_box_epoch[ax_r[0],ax_c[0]].axis('off')
            ax_line[ax_r[0],ax_c[0]].axis('off')
        plt.figure(f_box)
        plt.suptitle(dt + ' Taste Decode Rates')
        f_box.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_rates_box.png'))
        f_box.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_rates_box.svg'))
        plt.close(f_box)
        plt.figure(f_line)
        plt.suptitle(dt + ' Taste Decode Rates')
        f_line.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_rates_line.png'))
        f_line.savefig(os.path.join(decode_results_save_dir,dt + '_which_taste_rates_line.svg'))
        plt.close(f_line)
        plt.figure(f_box_epoch)
        plt.suptitle(dt + ' Epoch Decode Rates')
        f_box_epoch.savefig(os.path.join(decode_results_save_dir,dt + '_which_epoch_rates_box.png'))
        f_box_epoch.savefig(os.path.join(decode_results_save_dir,dt + '_which_epoch_rates_box.svg'))
        plt.close(f_box_epoch)
        
    
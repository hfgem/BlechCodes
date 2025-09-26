#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:59:01 2025

@author: Hannah Germaine

A collection of functions dedicated to testing deviations' correlations against multiple days of taste responses.
"""

import os
import csv
import time
import tqdm
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import functions.decode_parallel as dp
from matplotlib import colormaps, cm
from scipy.stats import pearsonr, ks_2samp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as gmm
from multiprocess import Pool


def multiday_dev_analysis(corr_dir,all_dig_in_names,group_train_data,control_data,
                          segment_dev_rasters,segment_dev_times,segment_dev_fr_vecs,
                          segment_names_to_analyze):
    """
    This function serves as the main function which calls all others for 
    analyses of deviation events from the train day in comparison to taste
    responses across days.
    """
        
    #Variables
    num_seg = len(segment_dev_fr_vecs)
    num_neur = len(segment_dev_fr_vecs[0][0])
    num_tastes = len(all_dig_in_names)
    
    #Now go through segments and their deviation events and compare
    for s_i in range(num_seg):
        seg_name = segment_names_to_analyze[s_i]
        dev_rast = segment_dev_rasters[s_i]
        dev_times = segment_dev_times[s_i]
        dev_fr_vecs = segment_dev_fr_vecs[s_i]
        
        #Run correlation analyses
        correlate_dev_to_taste(num_neur,all_dig_in_names,group_train_data,control_data,\
                                      dev_rast,dev_times,dev_fr_vecs,seg_name,corr_dir)
        
        
def correlate_dev_to_taste(num_neur,all_dig_in_names,group_train_data,control_data,\
                              dev_rast,dev_times,dev_fr_vecs,seg_name,corr_dir,\
                                  plot_flag=True):
    
    seg_dir = os.path.join(corr_dir,seg_name)
    if not os.path.isdir(seg_dir):
        os.mkdir(seg_dir)
    
    fr_dir = os.path.join(seg_dir,'true_corrs')
    if not os.path.isdir(fr_dir):
        os.mkdir(fr_dir)
        
    dev_vec_mat = np.array(dev_fr_vecs)
    num_dev, _ = np.shape(dev_vec_mat)
    dev_num = dev_vec_mat - np.expand_dims(np.nanmean(dev_vec_mat,1),1)
    dev_denom = np.sum(dev_num**2,1)
    group_names = list(group_train_data.keys())
    num_groups = len(group_names)
    
    #Group correlations
    try:
        corr_dict = np.load(os.path.join(corr_dir,seg_name+'_corr_dict.npy'),allow_pickle=True).item()
        avg_corr_by_group = np.nan*np.ones((num_dev,num_groups))
        for g_i, g_name in enumerate(group_names):
            corr_vals_by_response = corr_dict[g_name]['corr_vals_by_response']
            avg_corr = np.nanmean(corr_vals_by_response,1)
            avg_corr_by_group[:,g_i] = avg_corr
    except:
        corr_dict = dict()
        avg_corr_by_group = np.nan*np.ones((num_dev,num_groups))
        for g_i, g_name in enumerate(group_names):
            corr_dict[g_name] = dict()
            g_fr_vecs = np.array(group_train_data[g_name]['fr_vecs']) #num_vec x num_neur
            num_vec, _ = np.shape(g_fr_vecs)
            
            #Run all pairwise correlations
            taste_num = g_fr_vecs - np.expand_dims(np.nanmean(
                g_fr_vecs,1),1)
            taste_denom = np.sum(taste_num**2,1)
            
            all_corr_vals = []
            corr_vals_by_response = np.nan*np.ones((num_dev,num_vec))
            for v_i in range(num_vec):
                p_num = np.sum(dev_num*(taste_num[v_i,:]*np.ones(np.shape(dev_num))),1)
                p_denom = np.sqrt(dev_denom*taste_denom[v_i])
                corr_vec = p_num/p_denom
                corr_vals_by_response[:,v_i] = corr_vec
                all_corr_vals.extend(list(corr_vec))
            avg_corr = np.nanmean(corr_vals_by_response,1)
            avg_corr_by_group[:,g_i] = avg_corr
            
            corr_dict[g_name]['all_corr_vals'] = all_corr_vals
            corr_dict[g_name]['corr_vals_by_response'] = corr_vals_by_response
            corr_dict[g_name]['num_dev'] = num_dev
            corr_dict[g_name]['num_vec'] = num_vec
                       
        #Correlate to non-dev control
        g_name = 'Rescaled Non-Dev Control'
        corr_dict[g_name] = dict()
        g_fr_vecs = np.array(control_data[g_name]['fr_vecs']) #num_vec x num_neur
        num_vec, _ = np.shape(g_fr_vecs)
        
        #Run all pairwise correlations
        taste_num = g_fr_vecs - np.expand_dims(np.nanmean(
            g_fr_vecs,1),1)
        taste_denom = np.sum(taste_num**2,1)
        
        all_corr_vals = []
        corr_vals_by_response = np.nan*np.ones((num_dev,num_vec))
        for v_i in range(num_vec):
            p_num = np.sum(dev_num*(taste_num[v_i,:]*np.ones(np.shape(dev_num))),1)
            p_denom = np.sqrt(dev_denom*taste_denom[v_i])
            corr_vec = p_num/p_denom
            corr_vals_by_response[:,v_i] = corr_vec
            all_corr_vals.extend(list(corr_vec))
        
        corr_dict[g_name]['all_corr_vals'] = all_corr_vals
        corr_dict[g_name]['corr_vals_by_response'] = corr_vals_by_response
        corr_dict[g_name]['num_dev'] = num_dev
        corr_dict[g_name]['num_vec'] = num_vec
        
        np.save(os.path.join(corr_dir,seg_name+'_corr_dict.npy'),corr_dict,allow_pickle=True)
    
    rescaled_nondev_control_corr = np.nanmean(corr_vals_by_response,1)
    #Now plot
    if plot_flag == True:
        plot_corr_dist(avg_corr_by_group,rescaled_nondev_control_corr,\
                       group_names,seg_name,fr_dir)    
    
def correlate_null_dev_to_taste(num_neur,all_dig_in_names,tastant_fr_dist_z_pop,
                                  taste_num_deliv,max_hz_z_pop,min_hz_z_pop,
                                  max_num_cp,dev_rast,dev_times,dev_fr_vecs_z,
                                  seg_name,corr_dir):
    """
    This function correlates z-scored deviation events pulled out of null 
    datasets to z-scored taste responses.

    Parameters
    ----------
    num_neur : integer
        Number of neurons in dataset.
    all_dig_in_names : list of strings
        Names of all digital inputs in dataset.
    tastant_fr_dist_z_pop : dictionary
        Contains organized z-scored taste response firing rate vectors.
    taste_num_deliv : list if integers
        Number of deliveries of each tastant.
    max_hz_z_pop : boolean
        Maximum population firing rate.
    min_hz_z_pop : boolean
        Minimum population firing rate.
    max_num_cp : integer
        Maximum number of epochs across datasets.
    dev_rast :
        TODO: remove this variable.
    dev_times : 
        TODO: remove this variable.
    dev_fr_vecs_z : list
        List of deviation firing rate vectors.
    seg_name : string
        Name of analyzed segment, for labelling purposes.
    corr_dir : string
        Directory to store results.

    Returns
    -------
    None.
    
    Outputs
    -------
    Dictionary of correlation values + plot call.

    """
    
    fr_dir = os.path.join(corr_dir,'null_corrs')
    if not os.path.isdir(fr_dir):
        os.mkdir(fr_dir)
        
    dev_vec_mat = np.array(dev_fr_vecs_z)
    num_dev, _ = np.shape(dev_vec_mat)
    dev_num = dev_vec_mat - np.expand_dims(np.nanmean(dev_vec_mat,1),1)
    dev_denom = np.sum(dev_num**2,1)
    
    #Z-Scored correlations
    corr_z_dict = dict()
    avg_corr_z_array = np.nan*np.ones((num_dev,len(all_dig_in_names),max_num_cp))
    for t_i, t_name in enumerate(all_dig_in_names):
        corr_z_dict[t_i] = dict()
        corr_z_dict[t_i]['name'] = t_name
        corr_z_dict[t_i]['data'] = dict()
        for e_i in range(max_num_cp):
            #Gather taste fr vecs
            all_epoch_taste_vecs = []
            for d_i in range(int(taste_num_deliv[t_i])):
                all_epoch_taste_vecs.append(np.squeeze(
                    tastant_fr_dist_z_pop[t_i][d_i][e_i]))
            all_epoch_taste_vecs_array = np.array(all_epoch_taste_vecs) #num_deliv x num_neur
            
            #Run all pairwise correlations
            taste_num = all_epoch_taste_vecs_array - np.expand_dims(np.nanmean(
                all_epoch_taste_vecs_array,1),1)
            taste_denom = np.sum(taste_num**2,1)
            
            all_corr_vals = []
            corr_vals_by_deliv = np.zeros((num_dev,int(taste_num_deliv[t_i])))
            for d_i in range(int(taste_num_deliv[t_i])):
                p_num = np.sum(dev_num*(taste_num[d_i,:]*np.ones(np.shape(dev_num))),1)
                p_denom = np.sqrt(dev_denom*taste_denom[d_i])
                corr_vec = p_num/p_denom
                corr_vals_by_deliv[:,d_i] = corr_vec
                all_corr_vals.extend(list(corr_vec))
            avg_taste_epoch_corr = np.nanmean(corr_vals_by_deliv,1)
            avg_corr_z_array[:,t_i,e_i] = avg_taste_epoch_corr
              
            corr_z_dict[t_i]['data'][e_i] = all_corr_vals
            corr_z_dict[t_i]['num_dev'] = num_dev
            corr_z_dict[t_i]['taste_num'] = taste_num
            
    np.save(os.path.join(fr_z_dir,seg_name+'_corr_z_dict.npy'),corr_z_dict,allow_pickle=True)
    
    #Calculate best taste,epoch for each deviation
    best_dev_inds = np.zeros((num_dev,3))
    for dev_i in range(num_dev):
        max_corr = np.nanmax(np.squeeze(avg_corr_z_array[dev_i,:,:]))
        ind_1, ind_2 = np.where(np.squeeze(avg_corr_z_array[dev_i,:,:]) == np.nanmax(np.squeeze(avg_corr_z_array[dev_i,:,:])))
        best_dev_inds[dev_i,0] = ind_1[0]
        best_dev_inds[dev_i,1] = ind_2[0]
        best_dev_inds[dev_i,2] = max_corr
    np.save(os.path.join(fr_z_dir,seg_name+'_best_corr.npy'),best_dev_inds,allow_pickle=True)
    if not os.path.isfile(os.path.join(fr_z_dir,'all_taste_names.npy')):
        np.save(os.path.join(fr_z_dir,'all_taste_names.npy'),all_dig_in_names,allow_pickle=True)
            
    #Now plot
    plot_corr_dist(avg_corr_by_group,corr_dict,group_names,seg_name,fr_dir)
    
def plot_corr_dist(avg_corr_by_group,rescaled_nondev_control_corr,group_names,seg_name,fr_dir):
    
    num_groups = len(group_names)
    group_pairs = list(itertools.combinations(np.arange(num_groups),2))
    
    #Plot all groups against each other
    f_all = plt.figure(figsize=(5,5))
    plt.ylim([0,1])
    plt.xlim([-1,1])
    for g_i, g_name in enumerate(group_names):
        plt.hist(avg_corr_by_group[:,g_i],bins=1000,density=True,cumulative=True,\
                 histtype='step',label=g_name)
    plt.hist(rescaled_nondev_control_corr,bins=1000,density=True,cumulative=True,\
             histtype='step',label='Non-Dev Control')
    plt.title('Correlation CDFs - All Groups')
    plt.legend(loc='upper left')
    plt.ylabel('Fraction of DE')
    plt.xlabel('Avg Pearson Correlation Coefficient')
    plt.tight_layout()
    f_all.savefig(os.path.join(fr_dir,seg_name + '_all_corr.png'))
    f_all.savefig(os.path.join(fr_dir,seg_name + '_all_corr.svg'))
    
    #Pairwise plots + significance
    for g_1, g_2 in group_pairs:
        g_1_name = group_names[g_1]
        g_2_name = group_names[g_2]
        f_pair = plt.figure(figsize=(5,5))
        plt.ylim([0,1])
        plt.xlim([-1,1])
        plt.hist(avg_corr_by_group[:,g_1],bins=1000,density=True,cumulative=True,\
                 histtype='step',label=g_1_name)
        plt.hist(avg_corr_by_group[:,g_2],bins=1000,density=True,cumulative=True,\
                 histtype='step',label=g_2_name)
        #Significance
        p_val = ks_2samp(avg_corr_by_group[:,g_1],avg_corr_by_group[:,g_2]).pvalue
        if p_val <= 0.05:
            plt.title(g_1_name + ' x ' + g_2_name + '\nK.S. Sig p = ' + str(np.round(p_val,4)))
        else:
            plt.title(g_1_name + ' x ' + g_2_name + '\nK.S. N.S.')
        plt.legend(loc='upper left')
        plt.tight_layout()
        f_pair.savefig(os.path.join(fr_dir,seg_name + '_' + g_1_name \
                                    + '_x_' + g_2_name + '_corr.png'))
        f_pair.savefig(os.path.join(fr_dir,seg_name + '_' + g_1_name \
                                    + '_x_' + g_2_name + '_corr.svg'))
        plt.close(f_pair)
    
    #Now plot true groups against control
    for g_i, g_name in enumerate(group_names):
        f_control = plt.figure(figsize=(5,5))
        plt.ylim([0,1])
        plt.xlim([-1,1])
        plt.hist(avg_corr_by_group[:,g_i],bins=1000,density=True,cumulative=True,\
                 histtype='step',label=g_name)
        plt.hist(rescaled_nondev_control_corr,bins=1000,density=True,cumulative=True,\
                 histtype='step',label='Non-Dev Control')
        #Significance
        p_val = ks_2samp(avg_corr_by_group[:,g_i],rescaled_nondev_control_corr).pvalue
        if p_val <= 0.05:
            plt.title(g_name + ' x non-dev control\nK.S. Sig p = ' + str(np.round(p_val,4)))
        else:
            plt.title(g_name + ' x non-dev control\nK.S. N.S.')
        plt.legend(loc='upper left')
        plt.tight_layout()
        f_control.savefig(os.path.join(fr_dir,seg_name + '_' + g_name \
                                    + '_x_nondev_control_corr.png'))
        f_control.savefig(os.path.join(fr_dir,seg_name + '_' + g_name \
                                    + '_x_nondev_control_corr.svg'))
        plt.close(f_control)
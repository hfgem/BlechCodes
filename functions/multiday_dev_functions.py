#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:59:01 2025

@author: Hannah Germaine

A collection of functions dedicated to testing deviations' correlations and
decodes against multiple days of taste responses.
"""

import os
import csv
import time
from scipy.stats import pearsonr


def multiday_dev_analysis(save_dir,all_dig_in_names,tastant_fr_dist_pop,
                          taste_num_deliv,max_hz_pop,tastant_fr_dist_z_pop,
                          max_hz_z_pop,min_hz_z_pop,max_num_cp,segment_dev_rasters,
                          segment_dev_times,segment_dev_fr_vecs,
                          segment_dev_fr_vecs_zscore,segment_names_to_analyze):
    """
    This function serves as the main function which calls all others for 
    analyses of deviation events from the train day in comparison to taste
    responses across days.
    """
    
    corr_dir = os.path.join(save_dir,'Correlations')
    if not os.path.isdir(corr_dir):
        os.mkdir(corr_dir)
    decode_dir = os.path.join(save_dir,'Decodes')
    if not os.path.isdir(decode_dir):
        os.mkdir(decode_dir)
        
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
        dev_fr_vecs_z = segment_dev_fr_vecs_zscore[s_i]
        
        #Run correlation analyses
        # correlate_dev_to_taste(num_neur,all_dig_in_names,tastant_fr_dist_pop,
        #                            taste_num_deliv,max_hz_pop,max_num_cp,dev_rast,
        #                            dev_times,dev_fr_vecs,seg_name,corr_dir)
        correlate_dev_to_taste_zscore(num_neur,all_dig_in_names,tastant_fr_dist_z_pop,
                                          taste_num_deliv,max_hz_z_pop,min_hz_z_pop,
                                          max_num_cp,dev_rast,dev_times,dev_fr_vecs_z,
                                          seg_name,corr_dir)
        
        #Run decode analyses
        
        
def correlate_dev_to_taste(num_neur,all_dig_in_names,tastant_fr_dist_pop,
                           taste_num_deliv,max_hz_pop,max_num_cp,dev_rast,
                           dev_times,dev_fr_vecs,seg_name,corr_dir):
    
    fr_dir = os.path.join(corr_dir,'fr_corrs')
    if not os.path.isdir(fr_dir):
        os.mkdir(fr_dir)
    
    #Regular correlations
    corr_dict = dict()
    for t_i, t_name in enumerate(all_dig_in_names):
        corr_dict[t_i] = dict()
        corr_dict[t_i]['name'] = t_name
        
        
def correlate_dev_to_taste_zscore(num_neur,all_dig_in_names,tastant_fr_dist_z_pop,
                                  taste_num_deliv,max_hz_z_pop,min_hz_z_pop,
                                  max_num_cp,dev_rast,dev_times,dev_fr_vecs_z,
                                  seg_name,corr_dir):
    
    fr_z_dir = os.path.join(corr_dir,'zscore_fr_corrs')
    if not os.path.isdir(fr_z_dir):
        os.mkdir(fr_z_dir)
        
    dev_vec_mat = np.array(dev_fr_vecs_z) #num_dev x num_neur
    dev_num = dev_vec_mat - np.expand_dims(np.nanmean(dev_vec_mat,1),1)
    dev_denom = np.sum(dev_num**2,1)
    
    #Z-Scored correlations
    corr_z_dict = dict()
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
            for d_i in range(int(taste_num_deliv[t_i])):
                p_num = np.sum(dev_num*(taste_num[d_i,:]*np.ones(np.shape(dev_num))),1)
                p_denom = np.sqrt(dev_denom*taste_denom[d_i])
                all_corr_vals.extend(list(p_num/p_denom))
              
            corr_z_dict[t_i]['data'][e_i] = all_corr_vals
    
    np.save(os.path.join(fr_z_dir,seg_name+'_corr_z_dict.npy'),corr_z_dict,allow_pickle=True)
            
    
def plot_corr_dist(corr_save_dir,corr_z_dict,all_dig_in_names,max_num_cp):
    
    num_tastes = len(all_dig_in_names)
    #Plot epochs against each other for each taste
    f_taste, ax_taste = plt.subplots(nrows = 1, ncols = num_tastes, figsize=(8,8))
    
    
    #Plot tastes against each other for each epoch
    f_epoch, ax_epoch = plt.subplots(nrows = 1, ncols = max_num_cp, figsize=(8,8))
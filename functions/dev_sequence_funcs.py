#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:14:16 2024

@author: Hannah Germaine

A collection of functions dedicated to testing deviations for sequential
activity.
"""

import os
import itertools
from scipy import stats
import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt

def split_euc_diff(num_neur, segment_dev_rasters,segment_zscore_means,segment_zscore_stds,
                   tastant_fr_dist_pop,tastant_fr_dist_z_pop,dig_in_names,segment_names,
                   save_dir,segments_to_analyze, epochs_to_analyze = []):
    """
    This function is dedicated to an analysis of whether, when a deviation event
    is split in half down the middle, the two sides display different patterns
    of firing rates across the population. To do so, this function calculates the
    firing rate vectors across the population for each event when split down 
    the middle, and then calculates euclidean distances between the two halves. 
    The function outputs the distributions of these distances. The same analysis 
    is done for pairs of epochs in taste responses and the distributions are 
    plotted alongside those of the deviation events for comparison. This is done 
    for both z-scored and normal firing rate vectors.
    """
    
    # Variables
    num_tastes = len(dig_in_names)
    num_segments = len(segment_dev_rasters)
    num_taste_deliv = [len(tastant_fr_dist_pop[t_i]) for t_i in range(num_tastes)]
    max_num_cp = 0
    for t_i in range(num_tastes):
        for d_i in range(num_taste_deliv[t_i]):
            if len(tastant_fr_dist_pop[t_i][d_i]) > max_num_cp:
                max_num_cp = len(tastant_fr_dist_pop[t_i][d_i])
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes + 1)) #+1 for the deviation distribution color
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, len(epochs_to_analyze) + 1))
    
    epoch_pairs = list(itertools.combinations(epochs_to_analyze, 2))
    
    #Collect the firing rate vectors for the taste deliveries by epoch
    all_taste_fr_vecs = [] #num tastes x num epochs x (num neur x num deliv)
    all_taste_fr_vecs_zscore = []
    all_taste_euc_dist = [] #num tastes x num epoch pairs x num deliv
    all_taste_euc_dist_90_percentile = [] #num tastes x num epoch pairs
    all_taste_euc_dist_zscore = [] #num tastes x num epoch pairs x num deliv
    all_taste_euc_dist_z_90_percentile = [] #num tastes x num epoch pairs
    for t_i in range(num_tastes):
        taste_epoch_fr_vecs = []
        taste_epoch_fr_vecs_zscore = []
        for e_i in range(max_num_cp):
            epoch_vec_collection = []
            epoch_zscore_vec_collection = []
            for d_i in range(num_taste_deliv[t_i]):
                try:
                    epoch_vec_collection.append(tastant_fr_dist_pop[t_i][d_i][e_i])
                    epoch_zscore_vec_collection.append(tastant_fr_dist_z_pop[t_i][d_i][e_i])
                except:
                    epoch_vec_collection.extend([])
            epoch_vec_collection = np.squeeze(np.array(epoch_vec_collection)).T #num neur x num trials
            epoch_zscore_vec_collection = np.squeeze(np.array(epoch_zscore_vec_collection)).T
            taste_epoch_fr_vecs.append(epoch_vec_collection)
            taste_epoch_fr_vecs_zscore.append(epoch_zscore_vec_collection)
        all_taste_fr_vecs.append(taste_epoch_fr_vecs)
        all_taste_fr_vecs_zscore.append(taste_epoch_fr_vecs_zscore)
        
        #Calculate euclidean distances for all epoch pairs
        e_pair_dist = []
        e_pair_dist_z = []
        e_pair_90_percentile = []
        e_pair_z_90_percentile = []
        for e_pair in epoch_pairs:
            e_0 = e_pair[0]
            e_1 = e_pair[1]
            e_0_vecs = taste_epoch_fr_vecs[e_0]
            e_1_vecs = taste_epoch_fr_vecs[e_1]
            euc_val = np.sqrt(np.sum(np.square(e_0_vecs-e_1_vecs),0))
            e_pair_dist.append(euc_val)
            e_pair_90_percentile.extend([np.percentile(euc_val,90)])
            e_0_vecs_z = taste_epoch_fr_vecs_zscore[e_0]
            e_1_vecs_z = taste_epoch_fr_vecs_zscore[e_1]
            euc_val_z = np.sqrt(np.sum(np.square(e_0_vecs_z-e_1_vecs_z),0))
            e_pair_dist_z.append(euc_val_z)
            e_pair_z_90_percentile.extend([np.percentile(euc_val_z,90)])
        all_taste_euc_dist.append(e_pair_dist)
        all_taste_euc_dist_zscore.append(e_pair_dist_z)
        all_taste_euc_dist_90_percentile.append(e_pair_90_percentile)
        all_taste_euc_dist_z_90_percentile.append(e_pair_z_90_percentile)
    
    #Now go through each segment's deviation events and calculate the distance
    #between the halves and compare that to the distance between epoch pairs
    for seg_ind, s_i in enumerate(segments_to_analyze):
        seg_dev_rast = segment_dev_rasters[seg_ind]
        seg_z_mean = segment_zscore_means[seg_ind]
        seg_z_std = segment_zscore_stds[seg_ind]
        num_dev = len(seg_dev_rast)
        
        dev_dist = np.zeros(num_dev)
        dev_dist_z = np.zeros(num_dev)
        for dev_i in range(num_dev):
            #Pull raster firing rate vectors
            dev_rast = seg_dev_rast[dev_i]
            _, num_dt = np.shape(dev_rast)
            half_dt = np.ceil(num_dt/2).astype('int')
            first_half_rast = dev_rast[:,:half_dt]
            second_half_rast = dev_rast[:,-half_dt:]
            #Create fr vecs
            first_half_fr_vec = np.sum(first_half_rast,1)/(half_dt/1000) #In Hz
            second_half_fr_vec = np.sum(second_half_rast,1)/(half_dt/1000) #In Hz
            #Create z-scored fr vecs
            first_half_fr_vec_z = (first_half_fr_vec - seg_z_mean)/seg_z_std
            second_half_fr_vec_z = (second_half_fr_vec - seg_z_mean)/seg_z_std
            #Calculate euclidean distances
            euc_val = np.sqrt(np.sum(np.square(first_half_fr_vec-second_half_fr_vec)))
            euc_val_z = np.sqrt(np.sum(np.square(first_half_fr_vec_z-second_half_fr_vec_z)))
            dev_dist[dev_i] = euc_val
            dev_dist_z[dev_i] = euc_val_z
        
        #Now we plot the deviation distance distribution against the taste epoch
        #pair distributions
        f_hist, ax_hist = plt.subplots(nrows = len(epoch_pairs), ncols = 2, \
                                       figsize = (8,8))
        for e_pair_ind, e_pair in enumerate(epoch_pairs):
            #Normal Distances
            ax_hist[e_pair_ind,0].hist(dev_dist, bins = 100, label='Deviation Distances', \
                                       color= taste_colors[-1,:], alpha = 0.4, density=True)
            for t_i in range(num_tastes):
                ax_hist[e_pair_ind,0].hist(all_taste_euc_dist[t_i][e_pair_ind], bins = 20, label=dig_in_names[t_i], \
                                           color= taste_colors[t_i,:], alpha = 0.4, density=True)
            ax_hist[e_pair_ind,0].set_title('Epoch ' + str(e_pair[0]) + ', Epoch ' + str(e_pair[1]))
            if e_pair_ind == len(epoch_pairs) - 1:
                ax_hist[e_pair_ind,0].set_xlabel('Euclidean Distance')
            ax_hist[e_pair_ind,0].set_ylabel('Distribution Density')
            if e_pair_ind == 0:
                ax_hist[e_pair_ind,0].legend()
                
            #Z-Scored Distances
            ax_hist[e_pair_ind,1].hist(dev_dist_z, bins = 100, label='Deviation Distances', \
                                       color= taste_colors[-1,:], alpha = 0.4, density=True)
            for t_i in range(num_tastes):
                ax_hist[e_pair_ind,1].hist(all_taste_euc_dist_zscore[t_i][e_pair_ind], bins = 20, label=dig_in_names[t_i], \
                                           color= taste_colors[t_i,:], alpha = 0.4, density=True)
            ax_hist[e_pair_ind,1].set_title('Epoch ' + str(e_pair[0]) + ', Epoch ' + str(e_pair[1]))
            if e_pair_ind == len(epoch_pairs) - 1:
                ax_hist[e_pair_ind,1].set_xlabel('Z-Scored Euclidean Distance')
            if e_pair_ind == 0:
                ax_hist[e_pair_ind,1].legend()
        
        plt.tight_layout()
        plt.suptitle(segment_names[s_i])
        f_hist.savefig(os.path.join(save_dir,segment_names[s_i] + '_split_distances.png'))
        f_hist.savefig(os.path.join(save_dir,segment_names[s_i] + '_split_distances.svg'))
        plt.close(f_hist)
    
def split_match_calc(num_neur, segment_dev_rasters,segment_zscore_means,segment_zscore_stds,
                   tastant_fr_dist_pop,tastant_fr_dist_z_pop,dig_in_names,segment_names,
                   save_dir,segments_to_analyze, epochs_to_analyze = []):
    """
    This function is dedicated to an analysis of whether, when a deviation event
    is split in half down the middle, the pair of sides looks similar to adjacent
    pairs of epochs for different tastes. To do so, this function calculates the
    firing rate vectors across the population for each event when split down 
    the middle, and then calculates the correlation between the resulting matrix
    and the taste epoch pair matrices. The function outputs the distributions 
    of these correlations into a plot.
    """
    
    # Variables
    num_tastes = len(dig_in_names)
    num_segments = len(segment_dev_rasters)
    num_taste_deliv = [len(tastant_fr_dist_pop[t_i]) for t_i in range(num_tastes)]
    max_num_cp = 0
    for t_i in range(num_tastes):
        for d_i in range(num_taste_deliv[t_i]):
            if len(tastant_fr_dist_pop[t_i][d_i]) > max_num_cp:
                max_num_cp = len(tastant_fr_dist_pop[t_i][d_i])
    cmap = colormaps['jet']
    taste_colors = cmap(np.linspace(0, 1, num_tastes)) #+1 for the deviation distribution color
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    cmap = colormaps['cividis']
    
    epoch_pairs = [[epochs_to_analyze[e_i],epochs_to_analyze[e_i+1]] for e_i in range(len(epochs_to_analyze)-1)]
    epoch_colors = cmap(np.linspace(0, 1, len(epoch_pairs)))
    
    #Collect the firing rate pair matrices for the taste deliveries
    all_taste_fr_mats = [] #num tastes x num epoch pairs x num deliv x (num neur x 2)
    all_taste_fr_mats_zscore = [] #num tastes x num epoch pairs x num deliv x (num neur x 2)
    for t_i in range(num_tastes):
        t_pair_mats = []
        t_pair_z_mats = []
        for e_p_ind, e_pair in enumerate(epoch_pairs):
            e_pair_mats = []
            e_pair_z_mats = []
            for d_i in range(num_taste_deliv[t_i]):
                try:
                    e_1_vec = tastant_fr_dist_pop[t_i][d_i][e_pair[0]]
                    e_2_vec = tastant_fr_dist_pop[t_i][d_i][e_pair[1]]
                    e_pair_mats.append(np.concatenate((e_1_vec,e_2_vec),1))
                    e_1_vec_z = tastant_fr_dist_z_pop[t_i][d_i][e_pair[0]].T
                    e_2_vec_z = tastant_fr_dist_z_pop[t_i][d_i][e_pair[1]].T
                    e_pair_z_mats.append(np.concatenate((e_1_vec_z,e_2_vec_z),1))
                except:
                    e_pair_mats.extend([])
            t_pair_mats.append(e_pair_mats)
            t_pair_z_mats.append(e_pair_z_mats)
        all_taste_fr_mats.append(t_pair_mats)
        all_taste_fr_mats_zscore.append(t_pair_z_mats)
        
    #Now go through segments and their deviation events and compare
    for seg_ind, s_i in enumerate(segments_to_analyze):
        seg_dev_rast = segment_dev_rasters[seg_ind]
        seg_z_mean = segment_zscore_means[seg_ind]
        seg_z_std = segment_zscore_stds[seg_ind]
        num_dev = len(seg_dev_rast)
        
        dev_mats = []
        dev_mats_z = []
        for dev_i in range(num_dev):
            #Pull raster firing rate vectors
            dev_rast = seg_dev_rast[dev_i]
            _, num_dt = np.shape(dev_rast)
            half_dt = np.ceil(num_dt/2).astype('int')
            first_half_rast = dev_rast[:,:half_dt]
            second_half_rast = dev_rast[:,-half_dt:]
            
            #Create fr vecs
            first_half_fr_vec = np.expand_dims(np.sum(first_half_rast,1)/(half_dt/1000),1) #In Hz
            second_half_fr_vec = np.expand_dims(np.sum(second_half_rast,1)/(half_dt/1000),1) #In Hz
            dev_mat = np.concatenate((first_half_fr_vec,second_half_fr_vec),1)
            dev_mats.append(dev_mat)
            #Create z-scored fr vecs
            first_half_fr_vec_z = (first_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
            second_half_fr_vec_z = (second_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
            dev_mat_z = np.concatenate((first_half_fr_vec_z,second_half_fr_vec_z),1)
            dev_mats_z.append(dev_mat_z)
        dev_mats_array = np.array(dev_mats) #num dev x num neur x 2
        dev_mats_z_array = np.array(dev_mats_z) #num dev x num neur x 2
            
        #For each taste and epoch pair, calculate the distribution of correlations
        #for the deviation event matrices to the taste response matrices
        f_dist_tastes, ax_dist_tastes = plt.subplots(nrows = len(epoch_pairs), ncols = 2, \
                                       figsize = (8,8))
        f_dist_epochs, ax_dist_epochs = plt.subplots(nrows = num_tastes, ncols = 2, \
                                       figsize = (8,8))
        f_dist_tastes_cdf, ax_dist_tastes_cdf = plt.subplots(nrows = len(epoch_pairs), ncols = 2, \
                                       figsize = (8,8))
        f_dist_epochs_cdf, ax_dist_epochs_cdf = plt.subplots(nrows = num_tastes, ncols = 2, \
                                       figsize = (8,8))
        all_taste_dist_vals = [] #num tastes x num epoch pairs x num dev x num deliv
        all_taste_dist_vals_z = [] #num tastes x num epoch pairs x num dev x num deliv
        all_taste_avg_dist_vals = [] #num tastes x num epoch pairs x num dev
        all_taste_avg_dist_vals_z = [] #num tastes x num epoch pairs x num dev
        for t_i in range(num_tastes):
            taste_dist_vals = [] #num epoch pairs x num dev x num deliv
            taste_dist_vals_z = [] #num epoch pairs x num dev x num deliv
            taste_avg_dist_vals = [] #num epoch pairs x num dev
            taste_avg_dist_vals_z = [] #num epoch pairs x num dev
            for e_p_ind, e_pair in enumerate(epoch_pairs):
                e_pair_name = 'Epoch ' + str(e_pair[0]) + ', Epoch ' + str(e_pair[1])
                
                taste_deliv_mat_array = np.array(all_taste_fr_mats[t_i][e_p_ind]) #num deliv x num neur x 2
                taste_deliv_mat_z_array = np.array(all_taste_fr_mats_zscore[t_i][e_p_ind]) #num deliv x num neur x 2
                
                all_dist_vals = [] #size num dev x num deliv
                all_dist_vals_z = [] #size num dev x num deliv
                avg_dist_vals = [] #size num dev
                avg_dist_vals_z = [] #size num dev
                for dev_i in range(num_dev):
                    dev_i_dists = []
                    dev_i_z_dists = []
                    for deliv_i in range(num_taste_deliv[t_i]):
                        #Calculate euclidean distances
                        dev_i_dists.extend([np.sqrt(np.sum(np.square(dev_mats_array[dev_i,:,:].squeeze()-taste_deliv_mat_array[deliv_i,:,:].squeeze())))])
                        dev_i_z_dists.extend([np.sqrt(np.sum(np.square(dev_mats_z_array[dev_i,:,:].squeeze()-taste_deliv_mat_z_array[deliv_i,:,:].squeeze())))])
                        
                    all_dist_vals.append(dev_i_dists)
                    all_dist_vals_z.append(dev_i_z_dists)
                    #Now the average correlation to that taste's template
                    dev_i_avg_dist = np.nanmean(np.array(dev_i_dists))
                    dev_i_avg_z_dist = np.nanmean(np.array(dev_i_z_dists))
                    avg_dist_vals.extend([dev_i_avg_dist])
                    avg_dist_vals_z.extend([dev_i_avg_z_dist])
                
                taste_dist_vals.append(all_dist_vals)
                taste_dist_vals_z.append(all_dist_vals_z)
                taste_avg_dist_vals.append(avg_dist_vals)
                taste_avg_dist_vals_z.append(avg_dist_vals_z)
                    
                #Plot average distance values
                ax_dist_tastes[e_p_ind,0].hist(avg_dist_vals,bins=100,density=True,\
                                               label=dig_in_names[t_i],alpha=0.4,\
                                                   color=taste_colors[t_i])
                ax_dist_tastes[e_p_ind,0].set_ylabel('Density')
                ax_dist_tastes[e_p_ind,0].set_title(e_pair_name)
                ax_dist_tastes[e_p_ind,1].hist(avg_dist_vals_z,bins=100,density=True,\
                                               label=dig_in_names[t_i],alpha=0.4,\
                                                   color=taste_colors[t_i])
                ax_dist_tastes[e_p_ind,1].set_title(e_pair_name)
                
                ax_dist_tastes_cdf[e_p_ind,0].hist(avg_dist_vals,bins=100,density=True,\
                                               label=dig_in_names[t_i],alpha=0.4,\
                                                cumulative=True,histtype='step',\
                                                    color=taste_colors[t_i])
                ax_dist_tastes_cdf[e_p_ind,0].set_ylabel('Cumulative Density')
                ax_dist_tastes_cdf[e_p_ind,0].set_title(e_pair_name)
                ax_dist_tastes_cdf[e_p_ind,1].hist(avg_dist_vals_z,bins=100,density=True,\
                                               label=dig_in_names[t_i],alpha=0.4,\
                                                cumulative=True,histtype='step',\
                                                   color=taste_colors[t_i])
                ax_dist_tastes_cdf[e_p_ind,1].set_title(e_pair_name)
                    
                ax_dist_epochs[t_i,0].hist(avg_dist_vals,bins=100,density=True,\
                                            alpha=0.4,color=epoch_colors[e_p_ind],\
                                            label=e_pair_name)
                ax_dist_epochs[t_i,0].set_ylabel('Density')
                ax_dist_epochs[t_i,0].set_title(dig_in_names[t_i])
                ax_dist_epochs[t_i,1].hist(avg_dist_vals_z,bins=100,density=True,\
                                            alpha=0.4,color=epoch_colors[e_p_ind],\
                                            label=e_pair_name)
                ax_dist_epochs[t_i,1].set_title(dig_in_names[t_i])
                
                ax_dist_epochs_cdf[t_i,0].hist(avg_dist_vals,bins=100,density=True,\
                                            alpha=0.4,color=epoch_colors[e_p_ind],\
                                            cumulative=True,histtype='step',\
                                            label=e_pair_name)
                ax_dist_epochs_cdf[t_i,0].set_ylabel('Cumulative Density')
                ax_dist_epochs_cdf[t_i,0].set_title(dig_in_names[t_i])
                ax_dist_epochs_cdf[t_i,1].hist(avg_dist_vals_z,bins=100,density=True,\
                                            alpha=0.4,color=epoch_colors[e_p_ind],\
                                            cumulative=True,histtype='step',\
                                            label=e_pair_name)
                ax_dist_epochs_cdf[t_i,1].set_title(dig_in_names[t_i])
            all_taste_dist_vals.append(taste_dist_vals)
            all_taste_dist_vals_z.append(taste_dist_vals_z)
            all_taste_avg_dist_vals.append(taste_avg_dist_vals)
            all_taste_avg_dist_vals_z.append(taste_avg_dist_vals_z)
            
        ax_dist_tastes[0,0].legend()
        ax_dist_tastes[-1,0].set_xlabel('Matrix Distance')
        ax_dist_tastes[-1,1].set_xlabel('Z-Scored Matrix Distance')
        plt.tight_layout()
        f_dist_tastes.savefig(os.path.join(save_dir,segment_names[s_i] + '_dev_taste_dist_by_epoch_pair.png'))
        f_dist_tastes.savefig(os.path.join(save_dir,segment_names[s_i] + '_dev_taste_dist_by_epoch_pair.svg'))
        plt.close(f_dist_tastes)
        ax_dist_epochs[0,0].legend()
        ax_dist_epochs[-1,0].set_xlabel('Matrix Distance')
        ax_dist_epochs[-1,1].set_xlabel('Z-Scored Matrix Distance')
        plt.tight_layout()
        f_dist_epochs.savefig(os.path.join(save_dir,segment_names[s_i] + '_dev_taste_dist_by_taste.png'))
        f_dist_epochs.savefig(os.path.join(save_dir,segment_names[s_i] + '_dev_taste_dist_by_taste.svg'))
        plt.close(f_dist_epochs)
        ax_dist_tastes_cdf[0,0].legend()
        ax_dist_tastes_cdf[-1,0].set_xlabel('Matrix Distance')
        ax_dist_tastes_cdf[-1,1].set_xlabel('Z-Scored Matrix Distance')
        plt.tight_layout()
        f_dist_tastes_cdf.savefig(os.path.join(save_dir,segment_names[s_i] + '_dev_taste_dist_by_epoch_pair_cdf.png'))
        f_dist_tastes_cdf.savefig(os.path.join(save_dir,segment_names[s_i] + '_dev_taste_dist_by_epoch_pair_cdf.svg'))
        plt.close(f_dist_tastes_cdf)
        ax_dist_epochs_cdf[0,0].legend()
        ax_dist_epochs_cdf[-1,0].set_xlabel('Matrix Distance')
        ax_dist_epochs_cdf[-1,1].set_xlabel('Z-Scored Matrix Distance')
        plt.tight_layout()
        f_dist_epochs_cdf.savefig(os.path.join(save_dir,segment_names[s_i] + '_dev_taste_dist_by_taste_cdf.png'))
        f_dist_epochs_cdf.savefig(os.path.join(save_dir,segment_names[s_i] + '_dev_taste_dist_by_taste_cdf.svg'))
        plt.close(f_dist_epochs_cdf)
        
        #Now compare true tastes to no taste by calculating the bottom 5th
        #percentile of the no-taste distributions and using it as a cutoff
        #to determine how many deviation events actually have a significantly 
        #closer distance to a taste than control
        
        f_cutoff_dist, ax_cutoff_dist = plt.subplots(nrows = len(epoch_pairs), ncols= 2, \
                                                     figsize = (8,8))
        for e_p_ind, e_pair in enumerate(epoch_pairs):
            e_pair_name = 'Epoch ' + str(e_pair[0]) + ', Epoch ' + str(e_pair[1])
            #No-taste 5th percentile cutoffs
            # cutoff_val = np.percentile(np.array(all_taste_dist_vals[-1][e_p_ind]),5)
            # cutoff_z_val = np.percentile(np.array(all_taste_dist_vals_z[-1][e_p_ind]),5)
            cutoff_avg_val = np.percentile(all_taste_avg_dist_vals[-1][e_p_ind],1)
            cutoff_avg_z_val = np.percentile(all_taste_avg_dist_vals_z[-1][e_p_ind],1)
            
            # taste_cutoff_dist_vals = []
            # taste_cutoff_z_dist_vals = []
            taste_cutoff_avg_dist_vals = []
            taste_cutoff_avg_z_dist_vals = []
            taste_cutoff_avg_dist_inds = []
            taste_cutoff_avg_z_dist_inds = []
            for t_i in range(num_tastes-1):
                t_i_cutoff_dist_ind = np.where(all_taste_avg_dist_vals[t_i][e_p_ind] < cutoff_avg_val)[0]
                taste_cutoff_avg_dist_inds.append(t_i_cutoff_dist_ind)
                taste_cutoff_avg_dist_vals.append(np.array(all_taste_avg_dist_vals[t_i][e_p_ind])[t_i_cutoff_dist_ind])
                t_i_cutoff_z_dist_ind = np.where(all_taste_avg_dist_vals_z[t_i][e_p_ind] < cutoff_avg_z_val)[0]
                taste_cutoff_avg_z_dist_inds.append(t_i_cutoff_z_dist_ind)
                taste_cutoff_avg_z_dist_vals.append(np.array(all_taste_avg_dist_vals_z[t_i][e_p_ind])[t_i_cutoff_z_dist_ind])
            
            #Plot these distributions side-by-side
            for t_i in range(num_tastes-1):
                ax_cutoff_dist[e_p_ind,0].hist(taste_cutoff_avg_dist_vals[t_i],density=True,
                                               cumulative=True,label=dig_in_names[t_i],
                                               histtype='step',color=taste_colors[t_i])
                ax_cutoff_dist[e_p_ind,0].set_title(e_pair_name + '\n#Events ' + \
                                                    str(len(taste_cutoff_avg_dist_vals[t_i])))
                ax_cutoff_dist[e_p_ind,0].set_ylabel('Density')
                ax_cutoff_dist[e_p_ind,1].hist(taste_cutoff_avg_z_dist_vals[t_i],density=True,
                                               cumulative=True,label=dig_in_names[t_i],
                                               histtype='step',color=taste_colors[t_i])
                ax_cutoff_dist[e_p_ind,1].set_title(e_pair_name + '\n#Events ' + \
                                                    str(len(taste_cutoff_avg_z_dist_vals[t_i])))
                ax_cutoff_dist[e_p_ind,1].set_ylabel('Cumulative Density')
            
        ax_cutoff_dist[0,0].legend()
        ax_cutoff_dist[-1,0].set_xlabel('Distance')
        ax_cutoff_dist[-1,1].set_xlabel('Z-Score Distance')
        plt.suptitle(segment_names[s_i] + ' Significantly Close Deviation Distances')
        plt.tight_layout()
        f_cutoff_dist.savefig(os.path.join(save_dir,segment_names[s_i] + '_dev_taste_sig_dist_by_epoch_pair_cdf.png'))
        f_cutoff_dist.savefig(os.path.join(save_dir,segment_names[s_i] + '_dev_taste_sig_dist_by_epoch_pair_cdf.svg'))
        plt.close(f_cutoff_dist)
    
    
    


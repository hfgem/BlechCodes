#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:55:59 2025

@author: hannahgermaine

File dedicated to functions related to LSTM decoding of tastes where responses
are timeseries of firing rates.
"""

import os
import tqdm
import itertools
import umap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import curve_fit
from matplotlib import colormaps
os.environ["OMP_NUM_THREADS"] = "4"

def create_taste_matrices(day_vars, segment_deviations, all_dig_in_names, num_bins, z_bin_dt):
    """Function to take spike times following taste delivery and create 
    matrices of timeseries firing trajectories"""
    
    print("\n--- Creating Taste Matrices ---")
    half_z_bin = np.floor(z_bin_dt/2).astype('int')
    bin_starts = np.ceil(np.linspace(0,2000,num_bins+1)).astype('int')
    num_neur = len(day_vars[0]['keep_neur'])
    
    #Get whole-experiment mean rate info
    exp_len = 0
    tastant_spike_times = day_vars[0]['tastant_spike_times']
    num_neur = len(tastant_spike_times[0][0])
    neur_spike_counts = np.zeros(num_neur)
    for s_i in range(len(day_vars[0]['segment_names'])):
        seg_start = int(day_vars[0]['segment_times'][s_i])
        seg_end = int(day_vars[0]['segment_times'][s_i+1])
        seg_len = seg_end - seg_start
        exp_len += seg_len
        time_bin_starts = np.arange(
            seg_start+half_z_bin, seg_end-half_z_bin, z_bin_dt)
        segment_spike_times_s_i = day_vars[0]['segment_spike_times'][s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = (np.array(
                segment_spike_times_s_i[n_i]) - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        neur_spike_counts += np.sum(segment_spike_times_s_i_bin,1)
    
    #Get taste segment z-score info
    day_zscore = dict()
    for day in range(len(day_vars)):
        segment_names = day_vars[day]['segment_names']
        segment_times = day_vars[day]['segment_times']
        segment_spike_times = day_vars[day]['segment_spike_times']
        s_i_taste = np.nan*np.ones(1)
        for s_i in range(len(segment_names)):
            if segment_names[s_i].lower() == 'taste':
                s_i_taste[0] = s_i
        if not np.isnan(s_i_taste[0]):
            s_i = int(s_i_taste[0])
            seg_start = int(segment_times[s_i])
            seg_end = int(segment_times[s_i+1])
            seg_len = seg_end - seg_start
            time_bin_starts = np.arange(
                seg_start+half_z_bin, seg_end-half_z_bin, z_bin_dt)
            segment_spike_times_s_i = segment_spike_times[s_i]
            segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
            for n_i in range(num_neur):
                n_i_spike_times = (np.array(
                    segment_spike_times_s_i[n_i]) - seg_start).astype('int')
                segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
            tb_fr = np.zeros((num_neur, len(time_bin_starts)))
            for tb_i, tb in enumerate(time_bin_starts):
                tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                        half_z_bin:tb+half_z_bin-seg_start], 1)/(2*half_z_bin*(1/1000))
            mean_fr = np.mean(tb_fr, 1)
            std_fr = np.std(tb_fr, 1)
        else:
            mean_fr = np.nan*np.ones(num_neur)
            std_fr = np.nan*np.ones(num_neur)
        day_zscore[day] = dict()
        day_zscore[day]['mean_fr'] = mean_fr
        day_zscore[day]['std_fr'] = std_fr
    
    #Create storage matrices and outputs
    taste_unique_categories = list(all_dig_in_names)
    training_matrices = []
    training_labels = []
    deliv_counts = []
    taste_pop_fr = []
    #Get individual taste responses
    for t_i, t_name in tqdm.tqdm(enumerate(all_dig_in_names)):
        taste, day = t_name.split('_')
        day = int(day)
        t_i_day = np.where(np.array(day_vars[day]['dig_in_names']) == taste)[0][0]
        mean_fr = day_zscore[day]['mean_fr']
        std_fr = day_zscore[day]['std_fr']
        
        segment_times = day_vars[day]['segment_times']
        segment_spike_times = day_vars[day]['segment_spike_times']
        tastant_spike_times = day_vars[day]['tastant_spike_times']
        start_dig_in_times = day_vars[day]['start_dig_in_times']
        num_deliv = len(tastant_spike_times[t_i_day])
        deliv_counts.append(num_deliv)
        
        #Generate response matrices
        for d_i in range(num_deliv):  # index for that taste
            raster_times = tastant_spike_times[t_i_day][d_i]
            start_taste_i = start_dig_in_times[t_i_day][d_i]
            # Binerize the activity following taste delivery start
            times_post_taste = [(np.array(raster_times[n_i])[np.where((raster_times[n_i] >= start_taste_i)*(
                raster_times[n_i] < start_taste_i + 2000))[0]] - start_taste_i).astype('int') for n_i in range(num_neur)]
            bin_post_taste = np.zeros((num_neur, 2000))
            for n_i in range(num_neur):
                bin_post_taste[n_i, times_post_taste[n_i]] += 1
            taste_pop_fr.append(np.sum(bin_post_taste)/(2000/1000))
            
            #Calculate binned firing rate matrix
            fr_mat = np.zeros((num_neur,num_bins))
            for bin_i in range(num_bins):
                bs_i = bin_starts[bin_i]
                be_i = bin_starts[bin_i+1]
                b_len = (be_i - bs_i)/1000
                fr_mat[:,bin_i] = np.sum(bin_post_taste[:,bs_i:be_i],1)/b_len
            training_matrices.append(fr_mat)
            #Convert to z-scored matrix
            # fr_z_mat = (fr_mat - np.expand_dims(mean_fr,1))/np.expand_dims(std_fr,1)
            # training_matrices.append(fr_z_mat)
            training_labels.append(t_i)
            
    # Generate random responses
    null_taste = get_null_controls(day_vars,segment_deviations) #Get binary matrices
    mean_pop_fr = np.nanmean(taste_pop_fr)
    std_pop_fr = np.nanstd(taste_pop_fr)
    mean_fr = day_zscore[0]['mean_fr']
    std_fr = day_zscore[0]['std_fr']
    for null_i in range(len(null_taste)):
        #Calculate scaling factor to bring null to taste range
        null_pop_fr = np.sum(null_taste[null_i])/(2000/1000)
        scale = (std_pop_fr*np.random.randn() + mean_pop_fr)/null_pop_fr
        #Create binned null taste response
        fr_mat = np.zeros((num_neur,num_bins))
        for bin_i in range(num_bins):
            bs_i = bin_starts[bin_i]
            be_i = bin_starts[bin_i+1]
            b_len = (be_i - bs_i)/1000
            fr_mat[:,bin_i] = np.sum(null_taste[null_i][:,bs_i:be_i],1)/b_len
        rescaled_fr_mat = fr_mat*scale
        #rescaled_fr_mat = (fr_mat*scale - np.expand_dims(mean_fr,1))/np.expand_dims(std_fr,1)
        training_matrices.append(rescaled_fr_mat)
        training_labels.append(len(all_dig_in_names))
    taste_unique_categories.append('null')
    
    return taste_unique_categories, training_matrices, training_labels, \
        mean_pop_fr, std_pop_fr

def get_null_controls(day_vars,segment_deviations):
    """Function to find periods of the pre-taste interval where deviation events
    don't occur to get control tastes and deviation events"""
    segment_spike_times = day_vars[0]['segment_spike_times']
    segments_to_analyze = day_vars[0]['segments_to_analyze']
    segment_names = day_vars[0]['segment_names']
    segment_times = day_vars[0]['segment_times']
    segment_spike_times = day_vars[0]['segment_spike_times']
    tastant_spike_times = day_vars[0]['tastant_spike_times']
    num_null_taste = max([len(tastant_spike_times[i]) for i in range(len(tastant_spike_times))])
    num_neur = len(day_vars[0]['keep_neur'])
    min_dev_size = day_vars[0]['min_dev_size']
    local_size = day_vars[0]['local_size']
    half_min_dev_size = int(np.ceil(min_dev_size/2))
    half_local_size = int(np.ceil(local_size/2))
    
    #Get index of pre-taste interval
    pre_ind = np.nan*np.ones(1)
    for s_i in range(len(segment_names)):
        if segment_names[s_i].lower()[:3] == 'pre':
            pre_ind[0] = s_i
    
    #Create taste and deviation storage
    null_taste = dict()
    #Calculate non-dev periods
    if not np.isnan(pre_ind[0]):
        pre_ind = int(pre_ind[0])
        pre_deviations = segment_deviations[pre_ind]
        pre_deviations[0] = 0
        pre_deviations[-1] = 0
        dev_starts = np.where(np.diff(pre_deviations) == 1)[0] + 1
        pre_non_dev = np.ones(len(pre_deviations)) - pre_deviations
        pre_non_dev[0] = 0
        pre_non_dev[-1] = 0
        num_null_dev = len(dev_starts)
        
        seg_start = segment_times[pre_ind]
        seg_end = segment_times[pre_ind+1]
        seg_len = seg_end - seg_start
        segment_spike_times_s_i = segment_spike_times[pre_ind]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = (np.array(
                segment_spike_times_s_i[n_i]) - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        spike_sum = np.nansum(segment_spike_times_s_i_bin,0)
        
        change_inds = np.diff(pre_non_dev)
        start_nondev_bouts = np.where(change_inds == 1)[0] + 1
        end_nondev_bouts = np.where(change_inds == -1)[0]
        
        num_choice = min([len(start_nondev_bouts),num_null_taste])
        rand_inds = np.random.choice(np.arange(len(start_nondev_bouts)),len(start_nondev_bouts),replace=False)
        taste_done = 0
        ri = 0
        while taste_done < num_choice:
            s_i = start_nondev_bouts[ri]
            e_i = end_nondev_bouts[ri]
            len_i = e_i - s_i
            if len_i > 2000:
                null_taste[taste_done] = segment_spike_times_s_i_bin[:,s_i:s_i+2000]
                taste_done += 1
            ri += 1
            
    else:
        print("ERROR: no pre-taste interval found in segment names.")
        
    return null_taste
    
def get_taste_distributions_and_plots(taste_unique_categories,training_matrices,\
                                      training_labels,savedir):
    """Calculate the variance of each category for LSTM training to get a sense
    of the optimization landscape. Plot some reduced form of the responses to 
    visualize similarity/difference"""
    
    num_cat = len(taste_unique_categories)
    num_train, num_neur, num_bins = np.shape(np.array(training_matrices))
    reshape_training_data = []
    for n_i in range(num_train):
        reshape_training_data.extend(list(training_matrices[n_i].T))
    reshape_training_data = np.array(reshape_training_data)
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(reshape_training_data)
    
    reshape_embedding = np.reshape(embedding,(num_train,num_bins,2))
    
    cat_points = dict()
    for c_i in range(num_cat):
        cat_points[c_i] = []
    
    f_umap_cross, ax_umap_cross = plt.subplots(nrows = 1, ncols = num_bins,\
                                   figsize = (num_bins*5,5),\
                                       sharex = True, sharey = True)
    f_umap_ellipse, ax_umap_ellipse = plt.subplots(nrows = 1, ncols = num_bins,\
                                   figsize = (num_bins*5,5),\
                                       sharex = True, sharey = True)
    #Plot individual points
    for nb_i in range(num_bins):
        for nt_i in range(num_train):
            c_i = training_labels[nt_i]
            if nb_i == 0:
                cat_points[c_i].append(np.squeeze(reshape_embedding[nt_i,:,:]))
            ax_umap_cross[nb_i].scatter(reshape_embedding[nt_i,nb_i,0],reshape_embedding[nt_i,nb_i,1],\
                                  alpha=0.1,c = colormaps['tab10'](c_i),label='_')
            ax_umap_ellipse[nb_i].scatter(reshape_embedding[nt_i,nb_i,0],reshape_embedding[nt_i,nb_i,1],\
                                  alpha=0.1,c = colormaps['tab10'](c_i),label='_')
    mean_points = np.zeros((num_cat,num_bins,2))
    std_points = np.zeros((num_cat,num_bins,2))
    #Plot averages
    for c_i in range(num_cat):
        cat_array = np.array(cat_points[c_i])
        for nb_i in range(num_bins):
            mean_point = np.nanmean(np.squeeze(cat_array[:,nb_i,:]),0)
            mean_points[c_i,nb_i,:] = mean_point
            std_point = np.nanstd(np.squeeze(cat_array[:,nb_i,:]),0)
            std_points[c_i,nb_i,:] = std_point
            #Cross
            ax_umap_cross[nb_i].plot([mean_point[0]-std_point[0],mean_point[0]+std_point[0]],\
                               [mean_point[1],mean_point[1]],c = colormaps['tab10'](c_i),\
                                   label='_')
            ax_umap_cross[nb_i].plot([mean_point[0],mean_point[0]],\
                               [mean_point[1]-std_point[1],mean_point[1]+std_point[1]],\
                                   c = colormaps['tab10'](c_i),\
                                   label='_')
            #Ellipse
            width = 2*std_point[0]
            height = 2*std_point[1]
            angle = 0
            ellipse = Ellipse(mean_point, width, height, angle, alpha=0.1, \
                              color=colormaps['tab10'](c_i),label='_')
            ax_umap_ellipse[nb_i].add_patch(ellipse)
            #Points
            ax_umap_cross[nb_i].scatter(mean_point[0],mean_point[1],\
                                  c = colormaps['tab10'](c_i),\
                                      label=taste_unique_categories[c_i])
            ax_umap_ellipse[nb_i].scatter(mean_point[0],mean_point[1],\
                                  c = colormaps['tab10'](c_i),\
                                      label=taste_unique_categories[c_i])
    ax_umap_cross[0].legend(loc='upper left')
    plt.suptitle('UMAP Category Bins')
    plt.tight_layout()
    f_umap_cross.savefig(os.path.join(savedir,'UMAP_bin_scatter_cross.png'))
    f_umap_cross.savefig(os.path.join(savedir,'UMAP_bin_scatter_cross.svg'))  
    ax_umap_ellipse[0].legend(loc='upper left')
    plt.suptitle('UMAP Category Bins')
    plt.tight_layout()
    f_umap_ellipse.savefig(os.path.join(savedir,'UMAP_bin_scatter_ellipse.png'))
    f_umap_ellipse.savefig(os.path.join(savedir,'UMAP_bin_scatter_ellipse.svg'))  
    
    f_worm_ellipse, ax_worm_ellipse = plt.subplots()
    f_worm_cross, ax_worm_cross = plt.subplots()
    #Plot average trajectories
    for c_i in range(num_cat):
        #Add ellipse at each point for std
        for b_i in range(num_bins):
            center = np.squeeze(mean_points[c_i,b_i,:])
            width = 2*np.squeeze(std_points[c_i,b_i,0])
            height = 2*np.squeeze(std_points[c_i,b_i,1])
            angle = 0
            ellipse = Ellipse(center, width, height, angle, alpha=0.1, \
                              color=colormaps['tab10'](c_i),label='_')
            ax_worm_ellipse.add_patch(ellipse)
        #Add cross at each point for std
        #X-std
        for b_i in range(num_bins):
            x_vals = [np.squeeze(mean_points[c_i,b_i,0]) - np.squeeze(std_points[c_i,b_i,0]),\
                      np.squeeze(mean_points[c_i,b_i,0]) + np.squeeze(std_points[c_i,b_i,0])]
            y_vals = [np.squeeze(mean_points[c_i,b_i,1]),np.squeeze(mean_points[c_i,b_i,1])]
            plt.plot(x_vals,y_vals,c = colormaps['tab10'](c_i),label='_',alpha=0.5)
        #Y-std
        for b_i in range(num_bins):
            x_vals = [np.squeeze(mean_points[c_i,b_i,0]),np.squeeze(mean_points[c_i,b_i,0])]
            y_vals = [np.squeeze(mean_points[c_i,b_i,1]) - np.squeeze(std_points[c_i,b_i,1]),\
                      np.squeeze(mean_points[c_i,b_i,1]) + np.squeeze(std_points[c_i,b_i,1])]
            plt.plot(x_vals,y_vals,c = colormaps['tab10'](c_i),label='_',alpha=0.5)
        #Plot average trajectory
        ax_worm_ellipse.plot(np.squeeze(mean_points[c_i,:,0]),np.squeeze(mean_points[c_i,:,1]),\
                 c = colormaps['tab10'](c_i),label=taste_unique_categories[c_i])
        ax_worm_cross.plot(np.squeeze(mean_points[c_i,:,0]),np.squeeze(mean_points[c_i,:,1]),\
                 c = colormaps['tab10'](c_i),label=taste_unique_categories[c_i])
    
    plt.figure(f_worm_ellipse)
    plt.legend(loc='lower right')
    plt.title('Average UMAP Trajectories')
    plt.tight_layout()
    f_worm_ellipse.savefig(os.path.join(savedir,'UMAP_avg_worm_ellipse.png'))
    f_worm_ellipse.savefig(os.path.join(savedir,'UMAP_avg_worm_ellipse.svg'))
    plt.figure(f_worm_cross)
    plt.legend(loc='lower right')
    plt.title('Average UMAP Trajectories')
    plt.tight_layout()
    f_worm_cross.savefig(os.path.join(savedir,'UMAP_avg_worm_cross.png'))
    f_worm_cross.savefig(os.path.join(savedir,'UMAP_avg_worm_cross.svg'))
    
    f_worm_cat, ax_worm_cat = plt.subplots(ncols = num_cat, figsize = (5*num_cat, 5),\
                                           sharex = True, sharey = True)
    for c_i in range(num_cat):
        cat_array = np.array(cat_points[c_i])
        num_point, _, _ = np.shape(cat_array)
        for np_i in range(num_point):
            ax_worm_cat[c_i].plot(np.squeeze(cat_array[np_i,:,0]),np.squeeze(cat_array[np_i,:,1]),\
                     c = colormaps['tab10'](c_i),alpha=0.05)
        ax_worm_cat[c_i].plot(np.squeeze(mean_points[c_i,:,0]),np.squeeze(mean_points[c_i,:,1]),\
                 c = colormaps['tab10'](c_i))
        ax_worm_cat[c_i].set_title(taste_unique_categories[c_i])
    plt.tight_layout()
    plt.suptitle('Taste UMAP Trajectories')
    f_worm_cat.savefig(os.path.join(savedir,'UMAP_all_worm.png'))
    f_worm_cat.savefig(os.path.join(savedir,'UMAP_all_worm.svg'))
    
def create_dev_matrices(day_vars, deviations, z_bin_dt, num_bins, mean_taste_pop_fr):
    """Function to take spike times during deviation events and create 
    matrices of timeseries firing trajectories the same size as taste trajectories"""
    
    print("\n--- Creating Deviation Matrices ---")
    segment_spike_times = day_vars[0]['segment_spike_times']
    segments_to_analyze = day_vars[0]['segments_to_analyze']
    start_end_times = np.array(day_vars[0]['segment_times_reshaped'])[segments_to_analyze]
    
    half_z_bin = np.floor(z_bin_dt/2).astype('int')
    dev_matrices = dict()
    scaled_dev_matrices = dict()
    null_dev_matrices = dict()
    
    for s_ind, s_i in tqdm.tqdm(enumerate(segments_to_analyze)):
        seg_spikes = segment_spike_times[s_i]
        seg_start = int(start_end_times[s_ind][0])
        seg_end = int(start_end_times[s_ind][1])
        seg_len = seg_end - seg_start
        num_neur = len(seg_spikes)
        spikes_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            neur_spikes = np.array(seg_spikes[n_i]).astype(
                'int') - seg_start
            spikes_bin[n_i, neur_spikes] = 1
        # Calculate z-score mean and std
        seg_fr = np.zeros(np.shape(spikes_bin))
        for tb_i in range(seg_len - z_bin_dt):
            seg_fr[:, tb_i] = np.sum(
                spikes_bin[:, tb_i:tb_i+z_bin_dt], 1)/(z_bin_dt/1000)
        mean_fr = np.nanmean(seg_fr, 1)
        std_fr = np.nanstd(seg_fr, 1)
        
        #Now pull deviation matrices
        seg_dev_matrices = []
        seg_dev = deviations[s_ind]
        seg_dev[0] = 0
        seg_dev[-1] = 0
        change_inds = np.diff(seg_dev)
        start_dev_bouts = np.where(change_inds == 1)[0] + 1
        end_dev_bouts = np.where(change_inds == -1)[0]
        seg_dev_pop_fr = []
        for b_i in range(len(start_dev_bouts)):
            dev_s_i = start_dev_bouts[b_i]
            dev_e_i = end_dev_bouts[b_i]
            dev_len = dev_e_i - dev_s_i
            
            dev_rast_i = spikes_bin[:, dev_s_i:dev_e_i]
            
            #Calculate population rate for rescaling to taste levels
            dev_pop_fr = np.sum(dev_rast_i)/(dev_len/1000)
            seg_dev_pop_fr.append(dev_pop_fr)
            
            bin_starts = np.ceil(np.linspace(0,dev_len,num_bins+2)).astype('int')
            
            dev_fr_mat = np.zeros((num_neur,num_bins))
            for nb_i in range(num_bins):
                bs_i = bin_starts[nb_i]
                be_i = bin_starts[nb_i+2]
                dev_fr_mat[:,nb_i] = np.sum(dev_rast_i[:,bs_i:be_i],1)/((be_i-bs_i)/1000)
            seg_dev_matrices.append(dev_fr_mat)
            # z_dev_fr_mat = (dev_fr_mat - np.expand_dims(mean_fr,1))/np.expand_dims(std_fr,1)
            # seg_dev_matrices.append(z_dev_fr_mat)
        
        #Calculate scaling to bring to taste levels
        seg_dev_mean_pop_fr = np.nanmean(np.array(seg_dev_pop_fr))
        scale = mean_taste_pop_fr/seg_dev_mean_pop_fr
        
        dev_matrices[s_ind] = np.array(seg_dev_matrices)
        scaled_dev_matrices[s_ind] = scale*np.array(seg_dev_matrices)
        
        #Create null deviation matrices
        seg_non_dev_matrices = []
        mean_len = np.nanmean(end_dev_bouts - start_dev_bouts)
        std_len = np.nanstd(end_dev_bouts - start_dev_bouts)
        non_dev = np.ones(len(seg_dev)) - seg_dev
        non_dev[0] = 0
        non_dev[-1] = 0
        change_inds = np.diff(non_dev)
        start_non_dev_bouts = np.where(change_inds == 1)[0] + 1
        end_non_dev_bouts = np.where(change_inds == -1)[0]
        null_dev_made = 0
        null_pop_fr = []
        while null_dev_made < len(start_dev_bouts):
            nondev_s_i = start_non_dev_bouts[b_i]
            nondev_e_i = end_non_dev_bouts[b_i]
            nondev_len = (np.ceil(std_len*np.random.randn() + mean_len)).astype('int')
            
            nondev_rast_i = spikes_bin[:, nondev_s_i:nondev_e_i]
            nondev_pop_fr = np.sum(dev_rast_i)/(dev_len/1000)
            null_pop_fr.append(nondev_pop_fr)
            
            bin_starts = np.ceil(np.linspace(0,nondev_len,num_bins+2)).astype('int')
            
            nondev_fr_mat = np.zeros((num_neur,num_bins))
            for nb_i in range(num_bins):
                bs_i = bin_starts[nb_i]
                be_i = bin_starts[nb_i+2]
                nondev_fr_mat[:,nb_i] = np.sum(nondev_rast_i[:,bs_i:be_i],1)/((be_i-bs_i)/1000)
            seg_non_dev_matrices.append(nondev_fr_mat)
            # z_nondev_fr_mat = (nondev_fr_mat - np.expand_dims(mean_fr,1))/np.expand_dims(std_fr,1)
            # seg_non_dev_matrices.append(z_nondev_fr_mat)
            null_dev_made += 1
        
        null_dev_matrices[s_ind] = np.array(seg_non_dev_matrices)
        
    return dev_matrices, scaled_dev_matrices, null_dev_matrices

def rescale_taste_to_dev(dev_matrices,training_matrices):
    """Rescale taste responses to deviation rates to test LSTM imperviance to
    rate scales"""
    
    num_seg = len(dev_matrices)
    all_dev_matrices = []
    for s_i in range(num_seg):
        all_dev_matrices.extend(list(dev_matrices[s_i]))
    all_dev_matrices_array = np.array(all_dev_matrices)
    mean_dev = np.nanmean(all_dev_matrices_array)
    
    training_array = np.array(training_matrices)
    mean_train = np.nanmean(training_array)
    
    scale = mean_dev/mean_train
    
    rescaled_training_matrices = list(scale*training_array)
    
    return rescaled_training_matrices

def time_shuffled_dev_controls(day_vars, deviations, z_bin_dt, num_bins, mean_taste_pop_fr):
    """Function to take spike times during deviation events and create 
    time-shuffled matrices of timeseries firing trajectories the same size as 
    taste trajectories"""
    
    print("\n--- Creating Deviation Matrices ---")
    segment_spike_times = day_vars[0]['segment_spike_times']
    segments_to_analyze = day_vars[0]['segments_to_analyze']
    start_end_times = np.array(day_vars[0]['segment_times_reshaped'])[segments_to_analyze]
    
    half_z_bin = np.floor(z_bin_dt/2).astype('int')
    shuffled_dev_matrices = dict()
    shuffled_scaled_dev_matrices = dict()
    null_dev_matrices = dict()
    
    for s_ind, s_i in tqdm.tqdm(enumerate(segments_to_analyze)):
        seg_spikes = segment_spike_times[s_i]
        seg_start = int(start_end_times[s_ind][0])
        seg_end = int(start_end_times[s_ind][1])
        seg_len = seg_end - seg_start
        num_neur = len(seg_spikes)
        spikes_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            neur_spikes = np.array(seg_spikes[n_i]).astype(
                'int') - seg_start
            spikes_bin[n_i, neur_spikes] = 1
        # Calculate z-score mean and std
        seg_fr = np.zeros(np.shape(spikes_bin))
        for tb_i in range(seg_len - z_bin_dt):
            seg_fr[:, tb_i] = np.sum(
                spikes_bin[:, tb_i:tb_i+z_bin_dt], 1)/(z_bin_dt/1000)
        mean_fr = np.nanmean(seg_fr, 1)
        std_fr = np.nanstd(seg_fr, 1)
        
        #Now pull deviation matrices
        seg_dev_matrices = []
        seg_dev = deviations[s_ind]
        seg_dev[0] = 0
        seg_dev[-1] = 0
        change_inds = np.diff(seg_dev)
        start_dev_bouts = np.where(change_inds == 1)[0] + 1
        end_dev_bouts = np.where(change_inds == -1)[0]
        seg_dev_pop_fr = []
        for b_i in range(len(start_dev_bouts)):
            dev_s_i = start_dev_bouts[b_i]
            dev_e_i = end_dev_bouts[b_i]
            dev_len = dev_e_i - dev_s_i
            
            dev_rast_i = spikes_bin[:, dev_s_i:dev_e_i]
            
            dev_shuffle_rast_i = np.zeros(np.shape(dev_rast_i))
            #Shuffle spike times in raster
            for n_i in range(num_neur):
                n_dev_t = len(np.where(dev_rast_i[n_i,:] == 1)[0])
                dev_shuffle_rast_i[n_i,np.random.choice(np.arange(dev_len),n_dev_t,replace=False)] = 1
            
            #Calculate population rate for rescaling to taste levels
            dev_pop_fr = np.sum(dev_rast_i)/(dev_len/1000)
            seg_dev_pop_fr.append(dev_pop_fr)
            
            bin_starts = np.ceil(np.linspace(0,dev_len,num_bins+2)).astype('int')
            
            dev_fr_mat = np.zeros((num_neur,num_bins))
            for nb_i in range(num_bins):
                bs_i = bin_starts[nb_i]
                be_i = bin_starts[nb_i+2]
                dev_fr_mat[:,nb_i] = np.sum(dev_shuffle_rast_i[:,bs_i:be_i],1)/((be_i-bs_i)/1000)
            seg_dev_matrices.append(dev_fr_mat)
            # z_dev_fr_mat = (dev_fr_mat - np.expand_dims(mean_fr,1))/np.expand_dims(std_fr,1)
            # seg_dev_matrices.append(z_dev_fr_mat)
        
        #Calculate scaling to bring to taste levels
        seg_dev_mean_pop_fr = np.nanmean(np.array(seg_dev_pop_fr))
        scale = mean_taste_pop_fr/seg_dev_mean_pop_fr
        
        shuffled_dev_matrices[s_ind] = np.array(seg_dev_matrices)
        shuffled_scaled_dev_matrices[s_ind] = scale*np.array(seg_dev_matrices)
        
    return shuffled_dev_matrices, shuffled_scaled_dev_matrices
    
def time_shuffled_taste_controls(training_matrices):
    """Function to take spike times during deviation events and create 
    time-shuffled matrices of timeseries firing trajectories the same size as 
    taste trajectories"""
    
    print("\n--- Creating Time-Shuffled Taste Matrices ---")
    num_neur, num_bins = np.shape(training_matrices[0])
    shuffled_training_matrices = []
    for t_i in range(len(training_matrices)):
        t_i_mat = training_matrices[t_i]
        shuffle_order = np.random.choice(np.arange(num_bins),num_bins,replace=False)
        while np.product(shuffle_order == np.arange(num_bins)) == 1:
            shuffle_order = np.random.choice(np.arange(num_bins),num_bins,replace=False)
        shuffled_training_matrices.append(t_i_mat[:,shuffle_order])
    
    return shuffled_training_matrices

def lstm_cross_validation(training_matrices,training_labels,\
                          taste_unique_categories,savedir):
    """Function to perform training and cross-validation of a LSTM model using
    taste response firing trajectories to determine best model size"""
    
    latent_dim_sizes = np.arange(20,150,10)
    num_classes = len(np.unique(training_labels))
    ex_per_class = [len(np.where(np.array(training_labels) == i)[0]) for i in range(num_classes)]
    min_count = np.min(ex_per_class)
    
    X = np.array(training_matrices)
    Y = np.array(tf.one_hot(training_labels, num_classes))
    
    num_samples, timesteps, features = X.shape
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_dict = dict() #For each size return fold matrix
    print("\n--- Beginning Cross-Validation ---")
    for l_i, latent_dim in enumerate(latent_dim_sizes):
        print("\t Dim = " + str(latent_dim))
        
        fold_dict[l_i] = dict()
        fold_dict[l_i]["latent_dim"] = latent_dim
        fold_dict[l_i]["taste_unique_categories"] = taste_unique_categories
        
        histories = []                # To store training history (loss, accuracy) per fold
        val_accuracy_per_fold = []     # To store final validation loss and accuracy per fold
        val_loss_per_fold = []
        prediction_probabilities = np.nan*np.ones((num_samples,num_classes))
        state_cs = np.nan*np.ones((num_samples,latent_dim))
        val_inds = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X,training_labels)):
            # Split data into training and validation sets for the current fold
            #Ensure shuffling of categories
            train_index_rand = train_index[np.random.choice(np.arange(len(train_index)),len(train_index))]
            val_index_rand = val_index[np.random.choice(np.arange(len(val_index)),len(val_index))]
            
            train_data, test_data = X[train_index_rand,:,:], X[val_index_rand,:,:]
            train_cat, test_cat = Y[train_index_rand,:], Y[val_index_rand,:]
            
            history, val_loss, val_accuracy, predictions, state_c = fit_model(train_data,\
                                                    train_cat,test_data,\
                                                    test_cat,num_classes,\
                                                    latent_dim,fold,savedir)
            histories.append(history.history)
            val_accuracy_per_fold.append(val_accuracy)
            val_loss_per_fold.append(val_loss)
            prediction_probabilities[val_index_rand,:] = predictions
            state_cs[val_index_rand,:] = state_c
            val_inds.extend(list(val_index_rand))
            
        val_inds = np.sort(np.array(val_inds))   
        fold_dict[l_i]["histories"] = histories
        fold_dict[l_i]["val_accuracy_per_fold"] = val_accuracy_per_fold
        fold_dict[l_i]["val_loss_per_fold"] = val_loss_per_fold
        fold_dict[l_i]["predictions"] = prediction_probabilities[val_inds,:]
        fold_dict[l_i]["true_labels"] = Y[val_inds,:]
        
        argmax_predict = np.argmax(prediction_probabilities,1)
        predict_onehot = np.array(tf.one_hot(argmax_predict, np.shape(prediction_probabilities)[1]))
        
        #Plot predictions
        f, ax = plt.subplots(ncols=3)
        ax[0].imshow(Y[val_inds,:],aspect='auto')
        ax[0].set_title('Categories')
        ax[0].set_xticks(np.arange(num_classes),taste_unique_categories,
                         rotation=45)
        ax[1].imshow(prediction_probabilities[val_inds,:],aspect='auto')
        ax[1].set_title('Predictions')
        ax[1].set_xticks(np.arange(num_classes),taste_unique_categories,
                         rotation=45)
        ax[2].imshow(predict_onehot[val_inds,:],aspect='auto')
        ax[2].set_title('One-hot predictions')
        ax[2].set_xticks(np.arange(num_classes),taste_unique_categories,
                         rotation=45)
        plt.tight_layout()
        f.savefig(os.path.join(savedir,'latent_' + str(latent_dim) + '_predictions.png'))
        f.savefig(os.path.join(savedir,'latent_' + str(latent_dim) + '_predictions.svg'))
        plt.close(f)
        
        #Plot hidden states
        avg_state = []
        for class_i in range(num_classes):
            class_inds = np.where(np.array(training_labels) == class_i)[0]
            avg_state.append(np.nanmean(state_cs[class_inds,:],0))
            
        f_state, ax_state = plt.subplots(ncols=2)
        ax_state[0].imshow(state_cs[val_inds,:],aspect='auto')
        ax_state[0].set_title('Test LSTM Hidden State')
        for class_i, class_n in enumerate(taste_unique_categories):
            ax_state[1].plot(np.arange(latent_dim),avg_state[class_i],\
                             label=class_n)
        ax_state[1].legend(loc='upper left')
        ax_state[1].set_title('Avg LSTM State C')
        plt.tight_layout()
        f_state.savefig(os.path.join(savedir,'latent_' + str(latent_dim) + '_state_c.png'))
        f_state.savefig(os.path.join(savedir,'latent_' + str(latent_dim) + '_state_c.svg'))
        plt.close(f_state)
        
    np.save(os.path.join(savedir,'fold_dict.npy'),fold_dict,allow_pickle=True)
    
def fit_model(train_data,train_cat,test_data,test_cat,num_classes,latent_dim,fold,savedir):
    """Function to fit model"""
    
    model = _get_lstm_model(np.shape(train_data[0]),latent_dim,num_classes)
    #Print model summary
    #model.summary()
    
    history = model.fit(train_data, train_cat, epochs = 20, batch_size = 40,\
                        validation_data = (test_data,test_cat),\
                            verbose=0)
    
    val_loss, val_accuracy = model.evaluate(test_data, test_cat, verbose=0)
    
    lstm_output_extractor_model = Model(inputs=model.input,
                                            outputs=model.get_layer('lstm_layer').output)
    lstm_outputs, state_h, state_c = lstm_output_extractor_model.predict(test_data)
    
    predictions = model.predict(test_data)
    
    return history, val_loss, val_accuracy, predictions, state_c
    
def _get_lstm_model(input_shape, latent_dim, num_classes):
    """Function to define and return an LSTM model for training/prediction."""
    
    inputs = layers.Input(shape=input_shape)
    lstm_outputs, state_h, state_c = layers.LSTM(units = int(latent_dim), 
                                                 dropout=0.1,return_state=True,\
                                                name='lstm_layer')(inputs)
    predictions = layers.Dense(units = int(num_classes), activation='softmax',\
                               name='dense_layer')(lstm_outputs)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

def get_best_size(fold_dict, savedir):
    """Calculate best model size"""
    
    num_tested = len(list(fold_dict.keys()))
    class_names = fold_dict[0]["taste_unique_categories"]
    num_classes = len(class_names)
    true_taste_inds = [i for i in range(num_classes) if (class_names[i].split('_')[0] != 'none') and (class_names[i].split('_')[0] != 'null')]
    tested_latent_dim = np.array([fold_dict[l_i]["latent_dim"] for l_i in range(num_tested)])
    
    accuracy = np.nan*np.ones((num_tested,num_classes))
    precision = np.nan*np.ones((num_tested,num_classes))
    strong_accuracy = np.nan*np.ones((num_tested,num_classes))
    confusion_matrices = np.nan*np.ones((num_tested,num_classes,num_classes))
    
    for l_i, latent_dim in enumerate(tested_latent_dim):
        predictions = fold_dict[l_i]["predictions"]
        true_labels = fold_dict[l_i]["true_labels"]
        true_inds = np.where(true_labels == 1)[1]
        argmax_predictions = np.argmax(predictions,1)
        matching_predictions = np.where(true_inds == argmax_predictions)[0]
        for c_i in range(num_classes):
            #Calculate accuracy of prediction
            class_inds = np.where(true_inds == c_i)[0]
            predicted_inds = np.intersect1d(class_inds,matching_predictions)
            accuracy[l_i,c_i] = len(predicted_inds)/len(class_inds)
            precision[l_i,c_i] = len(predicted_inds)/len(np.where(argmax_predictions == c_i)[0])
            #Calculate strong accuracy of prediction - at least double that of next category
            strong_inds = []
            for pi in predicted_inds:
                val_order = np.argsort(predictions[pi,:])
                if predictions[pi,val_order[-1]] >= 2*predictions[pi,val_order[-2]]:
                    strong_inds.append(pi)
            strong_accuracy[l_i,c_i] = len(strong_inds)/len(class_inds)
            #Confusion matrix population
            predicted_classes = argmax_predictions[class_inds]
            confusion_matrices[l_i,c_i,:] = np.array([len(np.where(predicted_classes == c_i2)[0])/len(class_inds) for c_i2 in range(num_classes)])
        
        #Plot precision/accuracy
        f_prec_acc = plt.figure()
        plt.plot(np.arange(num_classes),accuracy[l_i,:],label='Accuracy')
        plt.plot(np.arange(num_classes),precision[l_i,:],label='Precision')
        plt.ylim([0,1])
        plt.legend(loc='upper right')
        plt.xticks(np.arange(num_classes),class_names)
        plt.tight_layout()
        f_prec_acc.savefig(os.path.join(savedir,'latent_' + str(latent_dim) + '_prec_acc.png'))
        f_prec_acc.savefig(os.path.join(savedir,'latent_' + str(latent_dim) + '_prec_acc.svg'))
        plt.close(f_prec_acc)
    
    average_accuracy = np.nanmean(accuracy,1)
    average_strong_accuracy = np.nanmean(strong_accuracy,1)
    average_true_taste_accuracy = np.nanmean(accuracy[:,true_taste_inds],1)
    average_strong_true_taste_accuracy = np.nanmean(strong_accuracy[:,true_taste_inds],1)
    
    #Plot results
    f_accuracy, ax_accuracy = plt.subplots(nrows = 2, ncols = 2, figsize = (7.5,7.5))
    #Plot accuracy
    img = ax_accuracy[0,0].imshow(accuracy,aspect='auto',cmap='viridis')
    ax_accuracy[0,0].set_xticks(np.arange(num_classes),class_names,rotation=45)
    ax_accuracy[0,0].set_yticks(np.arange(num_tested),tested_latent_dim)
    ax_accuracy[0,0].set_ylabel('Latent Dim')
    img.set_clim(0, 1)
    ax_accuracy[0,0].set_title('Accurate Predictions')
    plt.colorbar(mappable=img,ax=ax_accuracy[0,0])
    #Plot strong accuracy
    img = ax_accuracy[0,1].imshow(strong_accuracy,aspect='auto',cmap='viridis')
    ax_accuracy[0,1].set_xticks(np.arange(num_classes),class_names,rotation=45)
    ax_accuracy[0,1].set_yticks(np.arange(num_tested),tested_latent_dim)
    img.set_clim(0, 1)
    ax_accuracy[0,1].set_title('Accurate Predictions w Probability >= 2x next decoded')
    plt.colorbar(mappable=img,ax=ax_accuracy[0,1])
    #Plot average accuracy by size
    img = ax_accuracy[1,0].imshow(accuracy - strong_accuracy,aspect='auto',cmap='viridis')
    ax_accuracy[1,0].set_xticks(np.arange(num_classes),class_names,rotation=45)
    ax_accuracy[1,0].set_yticks(np.arange(num_tested),tested_latent_dim)
    img.set_clim(0, 1)
    ax_accuracy[1,0].set_title('All - Strong')
    plt.colorbar(mappable=img,ax=ax_accuracy[1,0])
    #Plot average strong accuracy by size
    ax_accuracy[1,1].plot(tested_latent_dim,average_accuracy,label='Average Accuracy')
    ax_accuracy[1,1].plot(tested_latent_dim,average_strong_accuracy,label='Average Strong Accuracy')
    ax_accuracy[1,1].plot(tested_latent_dim,average_true_taste_accuracy,label='Average True Accuracy')
    ax_accuracy[1,1].plot(tested_latent_dim,average_strong_true_taste_accuracy,label='Average Strong True Accuracy')
    ax_accuracy[1,1].set_ylim([0,1])
    ax_accuracy[1,1].set_ylabel('Average Accuracy')
    ax_accuracy[1,1].set_xlabel('Latent Dim')
    ax_accuracy[1,1].legend(loc='upper right')
    #Finish and save
    plt.tight_layout()
    f_accuracy.savefig(os.path.join(savedir,'accuracy_plots.png'))
    f_accuracy.savefig(os.path.join(savedir,'accuracy_plots.svg'))
    plt.close(f_accuracy)
    
    #Fit choose best size based on accuracy - std accuracy scoring
    f_log = plt.figure(figsize=(5,5))
    all_y = []
    all_x = []
    for t_i in true_taste_inds:
        all_y.extend(list(strong_accuracy[:,t_i]))
        all_x.extend(list(tested_latent_dim))
    plt.scatter(all_x,all_y,color='g',alpha=0.5,label='True Taste Accuracies')
    score = average_strong_true_taste_accuracy - np.nanstd(strong_accuracy[:,true_taste_inds],1)
    plt.plot(tested_latent_dim,score,linestyle='dashed',color='b',label='Score Curve')
    # try:
    #     params, covariance = curve_fit(shifted_log_func, all_x, all_y)
    #     a_fit, b_fit, c_fit = params
    #     log_y = shifted_log_func(tested_latent_dim, a_fit, b_fit, c_fit)
    #     plt.plot(tested_latent_dim,log_y,linestyle='dashed',color='k',label='Log Fit')
    #     #Calculate elbow
    #     deriv_1 = np.diff(log_y)/np.diff(tested_latent_dim)
    #     deriv_2 = np.diff(deriv_1)
    #     m = (deriv_2[-1] - deriv_2[0])/(len(deriv_2)-1)
    #     line = m*np.arange(len(deriv_2)) + deriv_2[0]
    #     best_ind = np.argmax(deriv_2 - line) + 1
    #     best_latent_dim = tested_latent_dim[best_ind]
    # except: #A log can't be fit
    #     #Calculate best score by taking the mean accuracy and subtracting the std
    #     best_ind = np.argmax(score)
    #     best_latent_dim = tested_latent_dim[best_ind]
    best_ind = np.argmax(score)
    best_latent_dim = tested_latent_dim[best_ind]
    plt.axvline(best_latent_dim,label='Best Size = ' + str(best_latent_dim),\
                color='r',linestyle='dashed')#Finish plot
    plt.ylabel('Strong Accuracy')
    plt.xlabel('Latent Dim')
    plt.legend(loc='upper left')
    plt.title('Calculated Best Latent Dim')
    plt.tight_layout()
    f_log.savefig(os.path.join(savedir,'best_latent_dim.png'))
    f_log.savefig(os.path.join(savedir,'best_latent_dim.svg'))
    plt.close(f_log)
    
    return best_latent_dim, score, tested_latent_dim
    
# def shifted_log_func(x, a, b, c):
#         return a * np.log(x + c) + b

def lstm_control_decoding(test_training_matrices, training_matrices, training_labels,\
                      latent_dim, taste_unique_categories, type_pred, savedir):
    """Function to run the best model on classifying the deviation events"""
    
    print("\n ---Running " + (' ').join(type_pred.split('_')) + " decoding.---")
    
    training_labels = np.array(training_labels)
    num_classes = len(np.unique(training_labels))
    class_inds = [np.where(training_labels == i)[0] for i in range(num_classes)]     
    ex_per_class = [len(ci) for ci in class_inds]
    min_count = np.min(ex_per_class)
    # none_ind = np.where(np.array(taste_unique_categories) == 'none_0')[0][0]
    # null_ind = np.where(np.array(taste_unique_categories) == 'null')[0][0]
    
    #Equalize the data to be the same number of samples per class
    keep_class_inds = []
    for i in range(num_classes):
        keep_class_inds.extend(list(np.random.choice(class_inds[i],min_count,replace=False)))
    #Shuffle
    keep_class_inds = list(np.random.choice(keep_class_inds,len(keep_class_inds),replace=False))
    
    X = np.array(training_matrices)[keep_class_inds]
    train_labels_balanced = training_labels[keep_class_inds]
    Y = np.array(tf.one_hot(train_labels_balanced, num_classes))
    
    num_samples, timesteps, features = X.shape
    
    model = _get_lstm_model(np.shape(X[0]),latent_dim,num_classes)
    
    history = model.fit(X, Y, epochs = 20, batch_size = 40,\
                            verbose=0)
    
    lstm_output_extractor_model = Model(inputs=model.input,
                                            outputs=model.get_layer('lstm_layer').output)
    
    predictions = model.predict(np.array(test_training_matrices))
    np.save(os.path.join(savedir,type_pred + '_predictions.npy'),predictions)
    
    
    argmax_predict = np.argmax(predictions,1)
    predict_onehot = np.array(tf.one_hot(argmax_predict, np.shape(predictions)[1]))
    true_onehot = tf.one_hot(training_labels, num_classes)
    np.save(os.path.join(savedir,'true_labels.npy'),true_onehot)
    predict_ind = np.argmax(predict_onehot,1)
    accuracy = len(np.where(predict_ind == training_labels)[0])/len(training_labels)
    by_taste_accuracy = np.zeros(num_classes)
    by_taste_precision = np.zeros(num_classes)
    for c_i in range(num_classes):
        class_ind = np.where(training_labels == c_i)[0]
        by_taste_accuracy[c_i] = len(np.where(predict_ind[class_ind] == training_labels[class_ind])[0])/len(class_ind)
        by_taste_precision[c_i] = len(np.where(predict_ind[class_ind] == training_labels[class_ind])[0])/len(np.where(predict_ind == c_i)[0])
    
    #Plot predictions
    f, ax = plt.subplots(ncols=3)
    ax[0].imshow(true_onehot,aspect='auto')
    ax[0].set_title('Categories')
    ax[0].set_xticks(np.arange(num_classes),taste_unique_categories,
                     rotation=45)
    ax[1].imshow(predictions,aspect='auto')
    ax[1].set_title('Predictions')
    ax[1].set_xticks(np.arange(num_classes),taste_unique_categories,
                     rotation=45)
    ax[2].imshow(predict_onehot,aspect='auto')
    ax[2].set_title('One-hot predictions')
    ax[2].set_xticks(np.arange(num_classes),taste_unique_categories,
                     rotation=45)
    plt.tight_layout()
    f.savefig(os.path.join(savedir,type_pred + '_predictions.png'))
    f.savefig(os.path.join(savedir,type_pred + '_predictions.svg'))
    plt.close(f)
    
    #Plot accuracy and precision
    f_accuracy = plt.figure()
    plt.plot(np.arange(num_classes),by_taste_accuracy,label='Accuracy')
    plt.plot(np.arange(num_classes),by_taste_precision,label='Precision')
    plt.ylim([0,1])
    plt.legend(loc='upper right')
    plt.xticks(np.arange(num_classes),taste_unique_categories)
    plt.tight_layout()
    f_accuracy.savefig(os.path.join(savedir,type_pred + '_prediction_accuracy_precision.png'))
    f_accuracy.savefig(os.path.join(savedir,type_pred + '_prediction_accuracy_precision.svg'))
    plt.close(f_accuracy)
    
    return predictions


def lstm_dev_decoding(dev_matrices, training_matrices, training_labels,\
                      latent_dim, taste_unique_categories, savedir):
    """Function to run the best model on classifying the deviation events"""
    
    training_labels = np.array(training_labels)
    num_classes = len(np.unique(training_labels))
    class_inds = [np.where(training_labels == i)[0] for i in range(num_classes)]     
    ex_per_class = [len(ci) for ci in class_inds]
    min_count = np.min(ex_per_class)
    
    #Equalize the data to be the same number of samples per class
    keep_class_inds = []
    for i in range(num_classes):
        keep_class_inds.extend(list(np.random.choice(class_inds[i],min_count,replace=False)))
    #Shuffle
    keep_class_inds = list(np.random.choice(keep_class_inds,len(keep_class_inds),replace=False))
    
    X = np.array(training_matrices)[keep_class_inds]
    train_labels_balanced = training_labels[keep_class_inds]
    Y = np.array(tf.one_hot(train_labels_balanced, num_classes))
    
    num_samples, timesteps, features = X.shape
    
    model = _get_lstm_model(np.shape(X[0]),latent_dim,num_classes)
    
    history = model.fit(X, Y, epochs = 20, batch_size = 40,\
                            verbose=0)
    
    lstm_output_extractor_model = Model(inputs=model.input,
                                            outputs=model.get_layer('lstm_layer').output)
    
    seg_predictions = dict()
    for seg_i in range(len(dev_matrices)):
        # lstm_outputs, state_h, state_c = lstm_output_extractor_model.predict(np.array(dev_matrices[seg_i]))
        
        predictions = model.predict(np.array(dev_matrices[seg_i]))
        seg_predictions[seg_i] = predictions
    
    return seg_predictions

def prediction_plots(seg_predictions,segment_names,savedir,savename):
    """Take predictions from all models and create a democratic prediction"""
    
    taste_unique_categories = seg_predictions["taste_unique_categories"]
    num_seg = len(seg_predictions) - 1
    num_classes = len(taste_unique_categories)
    # strong_cutoff = 1.1*(1/(num_classes - 1))
    true_taste_inds = [c for c in range(num_classes) if taste_unique_categories[c].split('_')[0] != 'none']
    true_taste_categories = [taste_unique_categories[c] for c in true_taste_inds]
    true_taste_pairs = list(itertools.combinations(true_taste_inds,2))
    
    thresholded_predictions = dict()
    thresholded_predictions["variables"] = dict()
    thresholded_predictions["variables"]["categories"] = taste_unique_categories
    # seg_predictions["variables"]["strong_cutoff"] = strong_cutoff
    thresholded_predictions["variables"]["strong_cutoff"]  = '2x'
    thresholded_predictions["variables"]["true_taste_inds"] = true_taste_inds
    thresholded_predictions["variables"]["true_taste_categories"] = true_taste_categories
    thresholded_predictions["variables"]["true_taste_pairs"] = true_taste_pairs
    
    for seg_i in range(num_seg):
        predictions = seg_predictions[seg_i]
        num_dev, _ = np.shape(predictions)
        
        #Average prediction across start bins
        cat_predictions = np.nan*np.ones(num_dev)
        for dev_i in range(num_dev):
            sort_ind = np.argsort(predictions[dev_i,:])
            if predictions[dev_i,sort_ind[-1]] > 2*predictions[dev_i,sort_ind[-2]]:
                cat_predictions[dev_i] = sort_ind[-1]
            
            # cat_argmax = np.argmax(mean_predictions[dev_i,:])
            
            # if mean_predictions[dev_i,cat_argmax] > strong_cutoff:
            #     cat_predictions[dev_i] = cat_argmax
        
        thresholded_predictions[seg_i] = cat_predictions
        
    np.save(os.path.join(savedir,savename + '_thresholded_predictions.npy'),thresholded_predictions,allow_pickle=True)
    
    plot_lstm_predictions(thresholded_predictions,segment_names,savedir,savename)
    
    return thresholded_predictions
        
def plot_lstm_predictions(thresholded_predictions,segment_names,savedir,savename):
    
    num_seg = len(thresholded_predictions) - 1
    categories = thresholded_predictions["variables"]["categories"]
    num_classes = len(categories)
    strong_cutoff = thresholded_predictions["variables"]["strong_cutoff"] 
    true_taste_inds = thresholded_predictions["variables"]["true_taste_inds"]
    true_taste_pairs = thresholded_predictions["variables"]["true_taste_pairs"]
    num_pairs = len(true_taste_pairs)
    
    #Plot regular histogram of decoding counts
    seg_pred_hist, ax_seg_pred = plt.subplots(ncols = num_seg, sharey = True,\
                                              figsize=(num_seg*5,5))
    for seg_i in range(num_seg):
        cat_predictions = thresholded_predictions[seg_i]
        num_dev = len(cat_predictions)
        try:
            hist_vals = ax_seg_pred[seg_i].hist(cat_predictions[cat_predictions != np.nan],\
                                                bins = np.arange(num_classes+1))
            for c_i in range(num_classes):
                ax_seg_pred[seg_i].text(c_i,hist_vals[0][c_i] + 5,str(hist_vals[0][c_i]))
            ax_seg_pred[seg_i].set_xticks(np.arange(num_classes)+0.5,categories)
            ax_seg_pred[seg_i].set_title(segment_names[seg_i])
        except:
            ax_seg_pred[seg_i].set_title(segment_names[seg_i] + '\nAll NaN.')
        ax_seg_pred[seg_i].set_ylim([0,num_dev])
    plt.suptitle('Democratic decoding histograms')
    plt.tight_layout()
    seg_pred_hist.savefig(os.path.join(savedir,savename + '_democratic_decoding_histograms.png'))
    seg_pred_hist.savefig(os.path.join(savedir,savename + '_democratic_decoding_histograms.svg'))
    plt.close(seg_pred_hist)
    
    #Plot ratios of decoding across segments
    f_pred_ratios, ax_pred_ratios = plt.subplots(ncols = num_pairs, \
                                                 figsize = (5*num_pairs,5))
    for tp_i, (t_i1,t_i2) in enumerate(true_taste_pairs):
        ratios = []
        for seg_i in range(num_seg):
            cat_predictions = thresholded_predictions[seg_i]
            tc_1 = len(np.where(cat_predictions == t_i1)[0])
            tc_2 = len(np.where(cat_predictions == t_i2)[0])
            if tc_2 > 0:
                ratios.append(tc_1/tc_2)
            else:
                ratios.append(1)
        ax_pred_ratios[tp_i].plot(np.arange(num_seg),ratios)
        ax_pred_ratios[tp_i].set_xticks(np.arange(num_seg),segment_names)
        ax_pred_ratios[tp_i].set_title(categories[t_i1] + ' / ' + \
                                       categories[t_i2])
    plt.suptitle('True Taste Decoding Ratios')
    plt.tight_layout()
    f_pred_ratios.savefig(os.path.join(savedir,savename + '_democratic_decoding_ratios.png'))
    f_pred_ratios.savefig(os.path.join(savedir,savename + '_democratic_decoding_ratios.svg'))
    plt.close(f_pred_ratios)
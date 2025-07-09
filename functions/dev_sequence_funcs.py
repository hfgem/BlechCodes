#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:14:16 2024

@author: Hannah Germaine

A collection of functions dedicated to testing deviations for sequential
activity.
"""

import os
import csv
import time
import itertools
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import functions.decode_parallel as dp
from scipy import stats
from random import sample
from scipy.stats import f
from scipy.signal import savgol_filter
from matplotlib import colormaps, cm
from multiprocess import Pool
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture as gmm
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import functions.dependent_decoding_funcs as ddf

def split_match_calc(num_neur,segment_dev_rasters,segment_zscore_means,segment_zscore_stds,
                   taste_num_deliv,tastant_raster_dict,tastant_fr_dist_z_pop,
                   dig_in_names,segment_names,segment_times,segment_spike_times,
                   z_bin_dt,num_null,group_list,group_names,non_none_tastes,
                   save_dir,segments_to_analyze=[],epochs_to_analyze = []):
    """
    This function is dedicated to an analysis of whether, when a deviation event
    is split in half down the middle, the pair of sides looks similar to adjacent
    pairs of epochs for different tastes. To do so, this function calculates the
    firing rate vectors across the population for each event when split down 
    the middle, and then calculates the correlation between the resulting matrix
    and the taste epoch pair matrices. The function outputs the distributions 
    of these correlations into a plot.
    """
    
    #Split Dirs
    base_split_dir = os.path.join(save_dir,'base_splits')
    if not os.path.isdir(base_split_dir):
        os.mkdir(base_split_dir)
    corr_dir = os.path.join(save_dir, 'corr_tests')
    if not os.path.isdir(corr_dir):
        os.mkdir(corr_dir)
    z_split_corr_dir = os.path.join(corr_dir,'zscore_firing_rates')
    if not os.path.isdir(z_split_corr_dir):
        os.mkdir(z_split_corr_dir)
    decode_split_dir = os.path.join(save_dir,'decode_splits')
    if not os.path.isdir(decode_split_dir):
        os.mkdir(decode_split_dir)
    z_decode_dir = os.path.join(decode_split_dir,'zscore_firing_rates')
    if not os.path.isdir(z_decode_dir):
        os.mkdir(z_decode_dir)
    null_z_decode_dir = os.path.join(z_decode_dir,'null_decodes_win_neur')
    if not os.path.isdir(null_z_decode_dir):
        os.mkdir(null_z_decode_dir)
    null_z_decode_dir_2 = os.path.join(z_decode_dir,'null_decodes_across_neur')
    if not os.path.isdir(null_z_decode_dir_2):
        os.mkdir(null_z_decode_dir_2)
        
    # Sequence Dirs
    sequence_dir = os.path.join(save_dir,'sequence_tests')
    if not os.path.isdir(sequence_dir):
        os.mkdir(sequence_dir)
    null_sequence_dir = os.path.join(sequence_dir,'null_sequences_win_neur')
    if not os.path.isdir(null_sequence_dir):
        os.mkdir(null_sequence_dir)
    null_sequence_dir_2 = os.path.join(sequence_dir,'null_sequences_across_neur')
    if not os.path.isdir(null_sequence_dir_2):
        os.mkdir(null_sequence_dir_2)        
    
    # Variables
    num_tastes = len(dig_in_names)
    num_taste_deliv = [len(tastant_fr_dist_z_pop[t_i]) for t_i in range(num_tastes)]
    max_num_cp = 0
    for t_i in range(num_tastes):
        for d_i in range(num_taste_deliv[t_i]):
            if len(tastant_fr_dist_z_pop[t_i][d_i]) > max_num_cp:
                max_num_cp = len(tastant_fr_dist_z_pop[t_i][d_i])
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(max_num_cp)
    
    taste_pairs = list(itertools.combinations(np.arange(num_tastes),2))
    taste_pair_names = []
    for tp_i, tp in enumerate(taste_pairs):
        taste_pair_names.append(dig_in_names[tp[0]] + ' v. ' + dig_in_names[tp[1]])
        
    #Now go through segments and their deviation events and compare
    for seg_ind, s_i in enumerate(segments_to_analyze):
        seg_dev_rast = segment_dev_rasters[seg_ind]
        seg_z_mean = segment_zscore_means[seg_ind]
        seg_z_std = segment_zscore_stds[seg_ind]
        num_dev = len(seg_dev_rast)
        
        #Split in half calcs
        create_splits_run_calcs(num_null, num_dev, seg_dev_rast, seg_z_mean, seg_z_std, 
                                taste_num_deliv, num_neur, dig_in_names, segments_to_analyze, 
                                segment_times, segment_spike_times, z_bin_dt, segment_names, 
                                s_i, epochs_to_analyze, group_list, group_names, 
                                non_none_tastes, tastant_fr_dist_z_pop, 
                                z_decode_dir, null_z_decode_dir, null_z_decode_dir_2,
                                z_split_corr_dir, base_split_dir)
        
        
def create_splits_run_calcs(num_null, num_dev, seg_dev_rast, seg_z_mean, seg_z_std, 
                            taste_num_deliv, num_neur, dig_in_names, 
                            segments_to_analyze, segment_times, 
                            segment_spike_times, bin_dt, segment_names, s_i, 
                            epochs_to_analyze, group_list, group_names, non_none_tastes,
                            tastant_fr_dist_z_pop, z_decode_dir, 
                            null_z_decode_dir, null_z_decode_dir_2,
                            z_split_corr_dir, base_split_dir):
    dev_mats_z = []
    null_dev_z_dict = dict()
    null_dev_z_dict_2 = dict()
    for null_i in range(num_null):
        null_dev_z_dict[null_i] = []
        null_dev_z_dict_2[null_i] = []
    for dev_i in range(num_dev):
        #Pull raster for firing rate vectors
        dev_rast = seg_dev_rast[dev_i]
        num_spikes_per_neur = np.sum(dev_rast,1).astype('int')
        _, num_dt = np.shape(dev_rast)
        half_dt = np.ceil(num_dt/2).astype('int')
        first_half_rast = dev_rast[:,:half_dt]
        second_half_rast = dev_rast[:,-half_dt:]
        #Create fr vecs
        first_half_fr_vec = np.expand_dims(np.sum(first_half_rast,1)/(half_dt/1000),1) #In Hz
        second_half_fr_vec = np.expand_dims(np.sum(second_half_rast,1)/(half_dt/1000),1) #In Hz
        #Create z-scored fr vecs
        first_half_fr_vec_z = (first_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
        second_half_fr_vec_z = (second_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
        dev_mat_z = np.concatenate((first_half_fr_vec_z,second_half_fr_vec_z),1)
        dev_mats_z.append(dev_mat_z)
        #Create null versions of the event
        for null_i in range(num_null):
            #Shuffle within-neuron spike times
            shuffle_rast = np.zeros(np.shape(dev_rast))
            for neur_i in range(num_neur):
                new_spike_ind = sample(list(np.arange(num_dt)),num_spikes_per_neur[neur_i])
                shuffle_rast[neur_i,new_spike_ind] = 1
            first_half_shuffle_rast = shuffle_rast[:,:half_dt]
            second_half_shuffle_rast = shuffle_rast[:,-half_dt:]
            #Create fr vecs
            first_half_fr_vec = np.expand_dims(np.sum(first_half_shuffle_rast,1)/(half_dt/1000),1) #In Hz
            second_half_fr_vec = np.expand_dims(np.sum(second_half_shuffle_rast,1)/(half_dt/1000),1) #In Hz
            #Create z-scored fr vecs
            first_half_fr_vec_z = (first_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
            second_half_fr_vec_z = (second_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
            shuffle_dev_mat_z = np.concatenate((first_half_fr_vec_z,second_half_fr_vec_z),1)
            null_dev_z_dict[null_i].append(shuffle_dev_mat_z)
            #Shuffle across-neuron spike times
            shuffle_rast_2 = np.zeros(np.shape(dev_rast))
            new_neuron_order = sample(list(np.arange(num_neur)),num_neur)
            for nn_ind, nn in enumerate(new_neuron_order):
                shuffle_rast_2[nn_ind,:] = shuffle_rast[nn,:]
            first_half_shuffle_rast_2 = shuffle_rast_2[:,:half_dt]
            second_half_shuffle_rast_2 = shuffle_rast_2[:,-half_dt:]
            #Create fr vecs
            first_half_fr_vec_2 = np.expand_dims(np.sum(first_half_shuffle_rast_2,1)/(half_dt/1000),1) #In Hz
            second_half_fr_vec_2 = np.expand_dims(np.sum(second_half_shuffle_rast_2,1)/(half_dt/1000),1) #In Hz
            #Create z-scored fr vecs
            first_half_fr_vec_z_2 = (first_half_fr_vec_2 - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
            second_half_fr_vec_z_2 = (second_half_fr_vec_2 - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
            shuffle_dev_mat_z_2 = np.concatenate((first_half_fr_vec_z_2,second_half_fr_vec_z_2),1)
            null_dev_z_dict_2[null_i].append(shuffle_dev_mat_z_2)      
        
    dev_mats_z_array = np.array(dev_mats_z) #num dev x num neur x 2
    for null_i in range(num_null):
        null_dev_z_dict[null_i] = np.array(null_dev_z_dict[null_i]) #num dev x num neur x 2
        null_dev_z_dict_2[null_i] = np.array(null_dev_z_dict_2[null_i]) #num dev x num neur x 2
       
    #Hotellings test the dev splits
    test_dev_split_hotellings(dev_mats_z_array, segment_names, s_i, base_split_dir)
    
    #Plot the dev splits in reduced form against taste epochs
    dev_split_halves_PCA_plot(dev_mats_z_array,tastant_fr_dist_z_pop,dig_in_names,
                              segment_names,s_i,base_split_dir)
    
    #Correlate deviation splits with epoch orders
    correlate_splits_epoch_pairs(tastant_fr_dist_z_pop, 
                    dig_in_names, dev_mats_z_array, segment_names, s_i,
                    z_split_corr_dir, epochs_to_analyze)
        
    #Decode each deviation event split
    decode_deviation_splits(tastant_fr_dist_z_pop, taste_num_deliv,
                    dig_in_names, dev_mats_z_array, segment_names, s_i, 
                    segments_to_analyze, segment_times, segment_spike_times, 
                    group_list, group_names, non_none_tastes, bin_dt, 
                    z_decode_dir, True, epochs_to_analyze)
     
    
def test_dev_split_hotellings(dev_mats_array, segment_names, s_i, base_split_dir):
    #Grab parameters/variables
    num_dev, num_neur, _ = np.shape(dev_mats_array)
    split_1 = np.squeeze(dev_mats_array[:,:,0])
    split_2 = np.squeeze(dev_mats_array[:,:,1])
    
    sig_csv_file = os.path.join(base_split_dir,'dev_split_sig_hotellings.csv')
    if not os.path.isfile(sig_csv_file):
        with open(sig_csv_file, 'w') as f_sig:
            write = csv.writer(f_sig, delimiter=',')
            title_list = ['Taste Name', 'Stat Name', 'Value']
            write.writerow(title_list)
            
    try:
        T2, F_hot, p_value = hotelling_t2(split_1, split_2)
        sig_vec = [T2, F_hot, p_value]
    except:
        sig_vec = [np.nan, np.nan, np.nan]
    
    with open(sig_csv_file, 'a') as f_sig:
        write = csv.writer(f_sig, delimiter=',')
        hot_t_list = [segment_names[s_i], 'Hotellings T', sig_vec[0]]
        write.writerow(hot_t_list)
        f_stat_list = [segment_names[s_i], 'F-Stat', sig_vec[1]]
        write.writerow(f_stat_list)
        p_val_list = [segment_names[s_i], 'p-val', sig_vec[2]]
        write.writerow(p_val_list)

def correlate_splits_epoch_pairs(tastant_fr_dist, 
                dig_in_names, dev_mats, segment_names, s_i,
                split_corr_dir, epochs_to_analyze=[]):
    """Correlate split deviation event firing rate vectors to pairs of epoch
    firing rate vectors to determine if there's a pair of epochs most
    represented"""
    print('\t\tRunning split corr to epoch pairs')
    
    #Variables
    num_tastes = len(dig_in_names)
    num_dev, num_neur, num_splits = np.shape(dev_mats)
    num_cp = len(tastant_fr_dist[0][0])
    cmap = colormaps['cividis']
    epoch_colors = cmap(np.linspace(0, 1, num_cp))
    cmap = colormaps['gist_rainbow']
    taste_colors = cmap(np.linspace(0, 1, num_tastes))
    cmap = colormaps['seismic']
    is_taste_colors = cmap(np.linspace(0, 1, 3))
    #Epoch pair options
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    epoch_splits = list(itertools.combinations(epochs_to_analyze, 2))
    epoch_splits.extend(list(itertools.combinations(np.fliplr(np.expand_dims(epochs_to_analyze,0)).squeeze(), 2)))
    epoch_splits.extend([(e_i,e_i) for e_i in epochs_to_analyze])
    epoch_split_inds = np.arange(len(epoch_splits))
    epoch_split_names = [str(ep) for ep in epoch_splits]
    epoch_root = np.ceil(np.sqrt(len(epoch_splits))).astype('int')
    epoch_ind_reference = np.reshape(np.arange(epoch_root**2),(epoch_root,epoch_root))
    #Taste pair options
    taste_pairs = list(itertools.combinations(np.arange(num_tastes),2))
    taste_pair_names = []
    for tp_i, tp in enumerate(taste_pairs):
        taste_pair_names.append(dig_in_names[tp[0]] + ' v. ' + dig_in_names[tp[1]])
    
    #Collect deviation event firing rate matrices as concatenated vectors
    dev_fr_vecs = []
    for dev_i in range(num_dev):
        #Converting to list for parallel processing
        dev_fr_mat = np.squeeze(dev_mats[dev_i,:,:]) #Shape num_neur x 2
        dev_fr_vecs.append(np.squeeze(np.concatenate((dev_fr_mat[:,0],dev_fr_mat[:,1]),0)))
    
    avg_taste_fr_vecs_by_es = []
    for es_ind, es in enumerate(epoch_splits):
        epoch_1 = es[0]
        epoch_2 = es[1]
        all_avg_taste_fr_vecs = []
        for t_i in range(num_tastes):
            taste_fr_vecs = []
            num_deliveries = len(tastant_fr_dist[t_i])
            for d_i in range(num_deliveries):
                vec1 = np.squeeze(tastant_fr_dist[t_i][d_i][epoch_1])
                vec2 = np.squeeze(tastant_fr_dist[t_i][d_i][epoch_2])
                try:
                    if np.shape(vec1)[0] == num_neur:
                        taste_fr_vecs.append(np.concatenate((vec1,vec2),0))
                    else:
                        taste_fr_vecs.append(np.concatenate((vec1.T,vec2.T),0))
                except: #No taste response stored
                    except_hold = len(taste_fr_vecs)
            taste_fr_vecs = np.array(taste_fr_vecs)
            if np.shape(taste_fr_vecs)[0] == 2*num_neur:
                avg_taste_fr_vec = np.squeeze(np.nanmean(taste_fr_vecs,1))
            else:
                avg_taste_fr_vec = np.squeeze(np.nanmean(taste_fr_vecs,0))
            all_avg_taste_fr_vecs.append(avg_taste_fr_vec)
        avg_taste_fr_vecs_by_es.append(all_avg_taste_fr_vecs)
    
    #Begin correlation analysis
    try:
        #Add an import statement here
        corr_dict = np.load(os.path.join(split_corr_dir,segment_names[s_i] + '_corr_dict.npy'),allow_pickle=True).item()
        print('\t\t\t\t' + segment_names[s_i] + ' Previously Pair-Correlated')
    except:
        print('\t\t\t\tCorrelating ' + segment_names[s_i] + ' Epoch Pairs')
        
        tic = time.time()
        
        corr_dict = dict()
        f_taste_cdf, ax_taste_cdf = plt.subplots(nrows = epoch_root, ncols = epoch_root, 
                                                 sharex = True, sharey = True,
                                                 figsize=(10,10))
        for es_ind, es in enumerate(epoch_splits):
            es_name = epoch_split_names[es_ind]
            corr_dict[es] = dict()
            corr_dict[es]['name'] = es_name
            corr_dict[es]['pair'] = es
            epoch_1 = es[0]
            epoch_2 = es[1]
            epoch_plot_ind = np.where(epoch_ind_reference == es_ind)
            
            #Collect average taste response firing rate correlations
            all_taste_corrs = []
            for t_i in range(num_tastes):
                avg_taste_fr_vec = avg_taste_fr_vecs_by_es[es_ind][t_i]
                dev_fr_vec_corrs = []
                for dev_i in range(num_dev):
                    pearson_corr = stats.pearsonr(dev_fr_vecs[dev_i],avg_taste_fr_vec)
                    dev_fr_vec_corrs.extend([pearson_corr[0]])
                all_taste_corrs.append(dev_fr_vec_corrs)
            corr_dict[es]['taste_corrs'] = all_taste_corrs
            
            #Create a CDF plot with KS-Test Stats
            plot_title = 'Epoch pair ' + str(es)
            for t_i in range(num_tastes):
                ax_taste_cdf[epoch_plot_ind[0][0],epoch_plot_ind[1][0]].hist(all_taste_corrs[t_i],bins=1000,density=True,\
                         cumulative=True,histtype='step',label=dig_in_names[t_i])
            #Calculate pairwise significances
            ks_sig = np.zeros(len(taste_pairs))
            ks_dir = np.zeros(len(taste_pairs))
            for p_i, p_vals in enumerate(taste_pairs):
                ks_res = stats.ks_2samp(all_taste_corrs[p_vals[0]], all_taste_corrs[p_vals[1]], \
                               alternative='two-sided')
                if ks_res.pvalue <= 0.05:
                    ks_sig[p_i] = 1
                    ks_dir[p_i] = ks_res.statistic_sign
            sig_text = 'Significant Pairs:'
            if np.sum(ks_sig) > 0:
                sig_inds = np.where(ks_sig > 0)[0]
                for sig_i in sig_inds:
                    if ks_dir[sig_i] == 1:
                        sig_text += '\n' + taste_pair_names[sig_i] + ' < '
                    if ks_dir[sig_i] == -1:
                        sig_text += '\n' + taste_pair_names[sig_i] + ' > '
            ax_taste_cdf[epoch_plot_ind[0][0],epoch_plot_ind[1][0]].text(0,0.2,sig_text)
            ax_taste_cdf[epoch_plot_ind[0][0],epoch_plot_ind[1][0]].legend(loc='upper left')
            ax_taste_cdf[epoch_plot_ind[0][0],epoch_plot_ind[1][0]].set_title(plot_title)
        for er_i in range(epoch_root):
            ax_taste_cdf[-1,er_i].set_xlabel('Pearson Correlation')
            ax_taste_cdf[er_i,0].set_ylabel('Cumulative Probability')
        plt.suptitle(segment_names[s_i] + ' epoch pair CDFs')
        plt.tight_layout()
        f_taste_cdf.savefig(os.path.join(split_corr_dir,segment_names[s_i] + '_epoch_pair_cdfs_taste_compare.png'))
        f_taste_cdf.savefig(os.path.join(split_corr_dir,segment_names[s_i] + '_epoch_pair_cdfs_taste_compare.svg'))
        plt.close(f_taste_cdf)
        
        #Now look by taste at epoch pair CDF
        f_epoch_cdf, ax_epoch_cdf = plt.subplots(nrows = 1, ncols = num_tastes, 
                                                 sharex = True, sharey = True,
                                                 figsize=(15,5))
        epoch_taste_means = np.zeros((len(epoch_splits),num_tastes))
        for t_i in range(num_tastes):
            #Plot epoch pair correlation distributions for this taste
            epoch_data_means = []
            epoch_names_compiled = []
            for es_ind, es in enumerate(epoch_splits):
                es_name = corr_dict[es]['name']
                epoch_names_compiled.append(es_name)
                data = corr_dict[es]['taste_corrs'][t_i]
                data_mean = np.nanmean(data)
                epoch_data_means.append(data_mean)
                epoch_taste_means[es_ind,t_i] = data_mean
                ax_epoch_cdf[t_i].hist(data, bins=1000,density=True,\
                         cumulative=True,histtype='step',label=es_name)
            ax_epoch_cdf[t_i].legend(loc='upper left')
            #Rank distributions based on mean
            rank_inds = np.argsort(epoch_data_means)
            sort_vals = str(epoch_data_means[rank_inds[0]])
            sort_text = str(epoch_splits[rank_inds[0]])
            sort_text_newline_flag = 0
            for r_i in range(len(rank_inds)-1):
                last_mean = epoch_data_means[rank_inds[r_i]]
                this_mean = epoch_data_means[rank_inds[r_i+1]]
                if (len(sort_text) > 35) and (sort_text_newline_flag == 0):
                    sort_text = sort_text + '\n'
                    sort_text_newline_flag = 1
                if this_mean > last_mean:
                    sort_text = sort_text + ' < ' + str(epoch_splits[rank_inds[r_i+1]])
                elif this_mean == last_mean:
                    sort_text = sort_text + ' = ' + str(epoch_splits[rank_inds[r_i+1]])
                else:
                    sort_text = sort_text + ' > ' + str(epoch_splits[rank_inds[r_i+1]])
            ax_epoch_cdf[t_i].set_title(dig_in_names[t_i] + '\n' + sort_text)
            ax_epoch_cdf[t_i].set_xlabel('Pearson Correlation')
        ax_epoch_cdf[0].set_ylabel('Cumulative Probability')
        max_mean = np.where(epoch_taste_means == np.max(epoch_taste_means))
        plt.suptitle(segment_names[s_i] + ' overall max = ' + dig_in_names[max_mean[1][0]] + \
                     ' ' + str(epoch_splits[max_mean[0][0]]))
        plt.tight_layout()
        f_epoch_cdf.savefig(os.path.join(split_corr_dir,segment_names[s_i] + '_taste_cdfs_epoch_pair_compare.png'))
        f_epoch_cdf.savefig(os.path.join(split_corr_dir,segment_names[s_i] + '_taste_cdfs_taste_compare.svg'))
        plt.close(f_epoch_cdf)
        
        toc = time.time()    
        print('\t\t\t\t\tTime to correlate ' + segment_names[s_i] + \
              ' deviation splits = ' + str(np.round((toc-tic)/60, 2)) + ' (min)')
            
        np.save(os.path.join(split_corr_dir,segment_names[s_i] + '_corr_dict.npy'),corr_dict,allow_pickle=True)    
            
def dev_split_halves_PCA_plot(dev_mats_array,tastant_fr_dist,dig_in_names,
                              segment_names,s_i,base_split_dir):
    """This function is dedicated to plotting in reduced dimensions the two 
    halves of a deviation event in reduced space to visualize their difference"""
    
    print("\t\tPlotting Dev Splits for Segment " + segment_names[s_i])
    
    #Collect variables
    colors = ['forestgreen','maroon','royalblue','orange','teal','palevioletred',
              'darkslateblue','red','green','blue']
    num_dev, num_neur, _ = np.shape(dev_mats_array)
    num_tastes = len(dig_in_names)
    split_1 = np.squeeze(dev_mats_array[:,:,0])
    split_2 = np.squeeze(dev_mats_array[:,:,1])
    X = np.concatenate((split_1,split_2),0)
    X_norm = (X - np.expand_dims(np.nanmean(X,1),1))/np.expand_dims(np.nanstd(X,1),1)
    num_cp = len(tastant_fr_dist[0][0])
    epochs_to_analyze = np.arange(num_cp)
    epoch_splits = list(itertools.combinations(epochs_to_analyze, 2))
    epoch_splits.extend(list(itertools.combinations(np.fliplr(np.expand_dims(epochs_to_analyze,0)).squeeze(), 2)))
    epoch_splits.extend([(e_i,e_i) for e_i in epochs_to_analyze])
    
    #Collect taste responses
    avg_taste_fr_vecs_by_e = []
    e_ind_collection = []
    t_ind_collection = []
    for e_i in range(num_cp):
        for t_i in range(num_tastes):
            taste_fr_vecs = []
            num_deliveries = len(tastant_fr_dist[t_i])
            for d_i in range(num_deliveries):
                vec = np.squeeze(tastant_fr_dist[t_i][d_i][e_i])
                try:
                    if len(vec) == num_neur:
                        taste_fr_vecs.append(vec)
                except: #No taste response stored
                    except_hold = len(taste_fr_vecs)
            taste_fr_vecs = np.array(taste_fr_vecs)
            avg_taste_fr_vecs = np.nanmean(taste_fr_vecs,0)
            avg_taste_fr_vecs_by_e.append(avg_taste_fr_vecs)
            e_ind_collection.append(e_i)
            t_ind_collection.append(t_i)
    avg_taste_fr_vecs_by_e = np.array(avg_taste_fr_vecs_by_e)
    e_ind_collection = np.array(e_ind_collection)
    t_ind_collection = np.array(t_ind_collection)
    avg_tastes_norm = (avg_taste_fr_vecs_by_e - np.expand_dims(np.nanmean(avg_taste_fr_vecs_by_e,
                                                1),1))/np.expand_dims(np.nanstd(
                                                avg_taste_fr_vecs_by_e,1),1)
    
    #Fit SVM to dev splits
    #Rescale values using z-scoring
    y_svm = np.concatenate((np.ones(num_dev),2*np.ones(num_dev)))
    svm_class = svm.SVC(kernel='linear')
    svm_class.fit(X_norm,y_svm)
    w = svm_class.coef_[0] #weights of classifier normal vector
    w_norm = np.linalg.norm(w)
    X_projected = w@X_norm.T/w_norm**2
    split_1_projected = X_projected[:num_dev]
    split_2_projected = X_projected[num_dev:]
    avg_tastes_projected = w@avg_tastes_norm.T/w_norm**2
    
    #Calculate orthogonal vectors with significantly different distributions for 2D plot
    sig_u = [] #significant vector storage
    u_p = [] #p-vals of significance
    for i in range(100):
        inds_to_use = sample(list(np.arange(num_neur)),2)
        u = np.zeros(num_neur)
        u[inds_to_use[0]] = -1*w[inds_to_use[1]]
        u[inds_to_use[1]] = w[inds_to_use[0]]
        u_norm = np.linalg.norm(u)
        u_proj = u@X_norm.T/u_norm**2
        sp_1_u_proj = u_proj[:num_dev]
        sp_2_u_proj = u_proj[num_dev:]
        ks_stats = stats.ks_2samp(sp_1_u_proj,sp_2_u_proj)
        if ks_stats.pvalue <= 0.05:
            sig_u.append(u)
            u_p.append(ks_stats.pvalue)
    if len(u_p) > 0:
        min_p = np.argmin(u_p)
        u = sig_u[min_p]
        u_norm = np.linalg.norm(u)
        X_orth_projected = u@X_norm.T/u_norm**2
        split_1_orth_projected = X_orth_projected[:num_dev]
        split_2_orth_projected = X_orth_projected[num_dev:]
        avg_taste_orth_projected = u@avg_tastes_norm.T/u_norm**2
        
        split_diff = split_1_projected-split_2_projected
        split_orth_diff = split_1_orth_projected-split_2_orth_projected
        max_split_diff = np.ceil(np.nanmax(np.concatenate((split_diff,split_orth_diff))))
        
        #Plot the projected dev split values as both scatters and hists with taste epochs
        for t_i, t_name in enumerate(dig_in_names):
            taste_data_ind = np.where(t_ind_collection == t_i)[0]
            f_split, ax_split = plt.subplots(ncols=3,figsize=(8,4))
            ax_split[0].scatter(split_1_projected,split_1_orth_projected,alpha=0.1,
                                color = colors[0], marker='o',label='Split 1')
            ax_split[0].scatter(split_2_projected,split_2_orth_projected,alpha=0.1,
                                color = colors[1], marker='o',label='Split 2')
            ax_split[0].set_xlabel('Normal Projection')
            ax_split[0].set_ylabel('In-Plane Projection')
            ax_split[0].set_title('Split Projections')
            ax_split[1].hist(split_1_projected,bins=20,histtype='step',alpha=0.5,
                             color=colors[0],density=True,cumulative=False,label='Split 1')
            ax_split[1].hist(split_2_projected,bins=20,histtype='step',alpha=0.5,
                             color=colors[1],density=True,cumulative=False,label='Split 2')
            for e_i in range(num_cp):
                epoch_data_ind = np.where(e_ind_collection == e_i)[0]
                et_ind = np.intersect1d(epoch_data_ind,taste_data_ind)
                ax_split[0].scatter(avg_tastes_projected[et_ind],avg_taste_orth_projected[et_ind],
                                    color=colors[e_i+2],alpha=0.9,marker='^',s=70,
                                    label='Epoch ' + str(e_i))
                ax_split[1].axvline(avg_tastes_projected[et_ind],alpha=0.9,
                                    color=colors[e_i+2],linestyle='dashed',
                                    label='Epoch ' + str(e_i))
            
            ax_split[1].set_xlabel('Normalized Orthogonal Projection')
            ax_split[1].set_ylabel('Distribution Density')
            ax_split[1].set_title('Split Othogonal Projections')
            ax_split[0].legend(loc='upper left')
            # ax_split[2].plot([0,max_split_diff],[0,max_split_diff],alpha=0.1,
            #                  color='k',linestyle='dashed',label='_')
            ax_split[2].scatter(split_diff,split_orth_diff,alpha=0.1,marker='o',
                                color=colors[0],label='Split 1 - Split 2')
            ax_split[2].set_xlabel('Diff Orth Proj')
            ax_split[2].set_ylabel('Diff Plane Proj')
            plt.suptitle(segment_names[s_i] + ' ' + dig_in_names[t_i] + ' Dev Split Projection')
            plt.tight_layout()
            f_split.savefig(os.path.join(base_split_dir,segment_names[s_i] +
                                         '_' + dig_in_names[t_i] + '_svm_orth_proj_splits.png'))
            f_split.savefig(os.path.join(base_split_dir,segment_names[s_i] +
                                         '_' + dig_in_names[t_i] + '_svm_orth_proj_splits.svg'))
            plt.close(f_split)
                                                
    #Plot reduced deviation splits                                                         
    split_pca = PCA(2)
    transformed_data = split_pca.fit_transform(X_norm)
    f_split_dev, ax_split_dev = plt.subplots(ncols=2, figsize=(8,4))
    ax_split_dev[0].scatter(transformed_data[:num_dev,0],transformed_data[:num_dev,1],
                alpha=0.2,label='Half 1')
    ax_split_dev[0].scatter(transformed_data[num_dev:,0],transformed_data[num_dev:,1],
                alpha=0.2,label='Half 2')
    ax_split_dev[0].set_title('PCA')
    ax_split_dev[0].legend(loc='upper left')
    mds_split = MDS(n_components=2, normalized_stress='auto')
    transformed_data = mds_split.fit_transform(X_norm)
    ax_split_dev[1].scatter(transformed_data[:num_dev,0],transformed_data[:num_dev,1],
                alpha=0.2,label='Half 1')
    ax_split_dev[1].scatter(transformed_data[num_dev:,0],transformed_data[num_dev:,1],
                alpha=0.2,label='Half 2')
    ax_split_dev[1].set_title('MDS')
    plt.suptitle(segment_names[s_i] + ' Split Dev Projection')
    plt.tight_layout()
    f_split_dev.savefig(os.path.join(base_split_dir,segment_names[s_i] + 
                                     '_split_dev_2D.png'))
    f_split_dev.savefig(os.path.join(base_split_dir,segment_names[s_i] + 
                                     '_split_dev_2D.svg'))
    plt.close(f_split_dev)        
            
    
def dev_split_vs_taste_PCA(plot_dev_fr_vecs, avg_taste_vecs, 
                           dig_in_names, epoch_pair, segment_names,
                           s_i, save_dir, name_modifier = ''):
    """This function is dedicated to plotting in reduced dimensions the splits of
    deviation events against each other as well as with taste response epochs"""
    
    #Grab parameters/variables
    save_name = segment_names[s_i] + '_epoch_' + str(epoch_pair[0]) + '_' \
        + str(epoch_pair[1]) + '_' + name_modifier
    title = segment_names[s_i] + ' Epoch pair (' + str(epoch_pair[0]) + ',' \
        + str(epoch_pair[1]) + ') ' + name_modifier
    num_tastes = len(dig_in_names)
    colors = ['forestgreen','maroon','royalblue','orange','teal','palevioletred',
              'darkslateblue','red','green','blue']
    num_dev, split_dim = np.shape(plot_dev_fr_vecs)
    num_neur = int(split_dim/2)
    concat_vecs = np.concatenate((avg_taste_vecs,plot_dev_fr_vecs),0)
    concat_vecs_self_norm = (concat_vecs - np.expand_dims(np.nanmean(concat_vecs,1),1))/np.expand_dims(np.nanstd(concat_vecs,1),1)
    concat_half1 = concat_vecs_self_norm[:,:num_neur]
    concat_half2 = concat_vecs_self_norm[:,num_neur:]
    dev_split1 = concat_half1[num_tastes:,:]
    dev_split2 = concat_half2[num_tastes:,:]
    taste_split1 = concat_half1[:num_tastes,:]
    taste_split2 = concat_half2[:num_tastes,:]
    
    #Perform SVM split + projection
    y_svm = np.concatenate((np.ones(num_dev),2*np.ones(num_dev)))
    X_norm = np.concatenate((dev_split1,dev_split2),0)
    svm_class = svm.SVC(kernel='linear')
    svm_class.fit(X_norm,y_svm)
    w = svm_class.coef_[0] #weights of classifier normal vector
    w_norm = np.linalg.norm(w)
    X_projected = w@X_norm.T/w_norm**2
    split_1_projected = X_projected[:num_dev]
    split_2_projected = X_projected[num_dev:]
    taste_1_projected = w@taste_split1.T/w_norm**2
    taste_2_projected = w@taste_split2.T/w_norm**2
    
    #Calculate orthogonal vectors with significantly different distributions for 2D plot
    sig_u = [] #significant vector storage
    u_p = [] #p-vals of significance
    for i in range(100):
        inds_to_use = sample(list(np.arange(num_neur)),2)
        u = np.zeros(num_neur)
        u[inds_to_use[0]] = -1*w[inds_to_use[1]]
        u[inds_to_use[1]] = w[inds_to_use[0]]
        u_norm = np.linalg.norm(u)
        u_proj = u@X_norm.T/u_norm**2
        sp_1_u_proj = u_proj[:num_dev]
        sp_2_u_proj = u_proj[num_dev:]
        ks_stats = stats.ks_2samp(sp_1_u_proj,sp_2_u_proj)
        if ks_stats.pvalue <= 0.05:
            sig_u.append(u)
            u_p.append(ks_stats.pvalue)
    if len(u_p) > 0:
        min_p = np.argmin(u_p)
        u = sig_u[min_p]
        u_norm = np.linalg.norm(u)
        X_orth_projected = u@X_norm.T/u_norm**2
        split_1_orth_projected = X_orth_projected[:num_dev]
        split_2_orth_projected = X_orth_projected[num_dev:]
        taste_1_orth_projected = u@taste_split1.T/u_norm**2
        taste_2_orth_projected = u@taste_split2.T/u_norm**2
    
        f_scaling = plt.figure(figsize=(5,5))
        plt.scatter(split_1_projected,split_1_orth_projected,alpha=0.2,
                            edgecolor = colors[0], facecolor = 'none',
                            s = 20, marker='o',label='Split 1')
        plt.scatter(split_2_projected,split_2_orth_projected,alpha=0.2,
                            edgecolor = colors[1], facecolor = 'none',
                            s = 20, marker='o',label='Split 2')
        for t_i, t_name in enumerate(dig_in_names):
            plt.scatter(taste_1_projected[t_i],taste_1_orth_projected[t_i],alpha=0.9,
                                color = colors[t_i+2], s = 50, marker='^',
                                label=t_name + ' ' + str(epoch_pair[0]))
            plt.scatter(taste_2_projected[t_i],taste_2_orth_projected[t_i],alpha=0.9,
                                color = colors[t_i+2], s = 50, marker='x',
                                label=t_name + ' ' + str(epoch_pair[1]))
        plt.legend(loc='upper left')
        plt.suptitle(title)
        plt.tight_layout()
        f_scaling.savefig(os.path.join(save_dir,save_name+'.png'))
        f_scaling.savefig(os.path.join(save_dir,save_name+'.svg'))
        plt.close(f_scaling)
        

def decode_deviation_splits(tastant_fr_dist, taste_num_deliv,
                dig_in_names, dev_mats_array, segment_names, s_i_test, 
                segments_to_analyze, segment_times, segment_spike_times, 
                group_list, group_names, non_none_tastes, bin_dt, decode_dir, 
                z_score = False, epochs_to_analyze=[]):
    """Decode taste from epoch-specific firing rates"""
    print('\t\tRunning NB Decoder on Segment ' + segment_names[s_i_test])
    
    # Variables
    num_tastes = len(dig_in_names)
    num_dev, num_neur, num_splits = np.shape(dev_mats_array)
    num_cp = len(tastant_fr_dist[0][0])
    if not group_names[-1] == 'Null Data':
        num_groups = len(group_names) + 1
        group_names.append('Null Data')
    else:
        num_groups = len(group_names)
    cmap = colormaps['jet']
    group_colors = cmap(np.linspace(0, 1, num_groups))
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    epoch_splits = list(itertools.combinations(epochs_to_analyze, 2))
    epoch_splits.extend(list(itertools.combinations(np.fliplr(np.expand_dims(epochs_to_analyze,0)).squeeze(), 2)))
    epoch_splits.extend([(e_i,e_i) for e_i in epochs_to_analyze])
    epoch_split_inds = np.arange(len(epoch_splits))
    epoch_split_names = [str(ep) for ep in epoch_splits]
        
    #Grab all taste-epoch pairs in the training groups for testing
    taste_epoch_pairs = []
    for gl_i, gl in enumerate(group_list):
        for gp_i, gp in enumerate(gl):
            taste_epoch_pairs.append([gp[1],gp[0]])
    
    #Create null dataset from shuffled rest spikes
    shuffled_fr_vecs, segment_spike_times_bin, \
        seg_means, seg_stds = ddf.create_null_decode_dataset(segments_to_analyze, \
                                    segment_times, segment_spike_times, \
                                    num_neur, bin_dt, z_score)
    
    #Create training groups of firing rate vectors
    grouped_train_data = [] #Using group_list above, create training groups
    grouped_train_counts = [] #Number of values in the group
    grouped_train_names = []
    for g_i, g_list in enumerate(group_list):
        group_data_collection = []
        for (e_i,t_i) in g_list:
            for d_i in range(int(taste_num_deliv[t_i])):
                try:
                    if np.shape(tastant_fr_dist[t_i][d_i][e_i])[0] == num_neur:
                        group_data_collection.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i].T))
                    else:
                        group_data_collection.extend(
                            list(tastant_fr_dist[t_i][d_i][e_i]))
                except:
                    group_data_collection.extend([])
        if len(group_data_collection) > 0:
            grouped_train_data.append(group_data_collection)
            grouped_train_counts.append(len(group_data_collection))
            grouped_train_names.append(group_names[g_i])
    #Now add the generated null data
    avg_count = int(np.ceil(np.nanmean(grouped_train_counts)))
    null_inds_to_use = sample(list(np.arange(len(shuffled_fr_vecs))),avg_count)
    grouped_train_data.append(np.array(shuffled_fr_vecs)[null_inds_to_use,:])
    grouped_train_counts.append(avg_count)
    grouped_train_names.append('Null')
    num_groups = len(grouped_train_names)
    
    #Create categorical NB dataset
    categorical_train_data = []
    categorical_train_y = []
    for g_i, g_data in enumerate(grouped_train_data):
        categorical_train_data.extend(g_data)
        categorical_train_y.extend(g_i*np.ones(len(g_data)).astype('int'))
    categorical_train_data = np.array(categorical_train_data)
    categorical_train_y = np.array(categorical_train_y)
    
    #Run PCA transform only on non-z-scored data
    if z_score == True:
        train_data = categorical_train_data
    else:
        pca_reduce_taste = ddf.train_taste_PCA(num_neur,taste_epoch_pairs,\
                                           non_none_tastes,dig_in_names,
                                           taste_num_deliv,tastant_fr_dist)
        train_data = pca_reduce_taste.transform(categorical_train_data)
   
    #Fit NB
    nb = GaussianNB()
    nb.fit(train_data, categorical_train_y) 
       
    
    # Segment-by-segment use deviation rasters and times to zoom in and test
    #	epoch-specific decoding of tastes. Add decoding of 50 ms on either
    #	side of the deviation event as well for context decoding.
    
    # Grab neuron firing rates in sliding bins
    try:
        dev_split_decode_array = np.load(
            os.path.join(decode_dir,segment_names[s_i_test] + \
                         '_dev_split_decode_array.npy'))
        dev_split_decode_prob_array = np.load(
            os.path.join(decode_dir,segment_names[s_i_test] + \
                         '_dev_split_decode_prob_array.npy'))
         
        print('\t\t\t\t' + segment_names[s_i_test] + ' Previously Decoded')
    except:
        print('\t\t\t\tDecoding ' + segment_names[s_i_test] + ' Deviation Splits')
        
        #Run through each deviation event to decode 
        tic = time.time()
        
        dev_fr_list = []
        for dev_i in range(num_dev):
            #Converting to list for parallel processing
            dev_fr_mat = np.squeeze(dev_mats_array[dev_i,:,:]) #Shape num_neur x 2
            if z_score == False:    
                dev_fr_pca = pca_reduce_taste.transform(dev_fr_mat.T)
                list_dev_fr = list(dev_fr_pca)
            else:
                list_dev_fr = list(dev_fr_mat.T)
            dev_fr_list.extend(list_dev_fr)
           
        split_dev_decode_prob_array = nb.predict_proba(dev_fr_list) 
        dev_split_decode_prob_array = np.reshape(split_dev_decode_prob_array,(num_dev,num_splits,num_groups))
        split_dev_decode_argmax = np.argmax(split_dev_decode_prob_array,1)
        dev_split_decode_array  = np.reshape(split_dev_decode_argmax,(num_dev,num_splits))
         
        # Save decoding probabilities        
        np.save(os.path.join(decode_dir,segment_names[s_i_test] + \
                         '_dev_split_decode_prob_array.npy'), dev_split_decode_prob_array)
        np.save(os.path.join(decode_dir,segment_names[s_i_test] + \
                         '_dev_split_decode_array.npy'), dev_split_decode_array)
        
        toc = time.time()
        print('\t\t\t\t\tTime to decode ' + segment_names[s_i_test] + \
              ' deviation splits = ' + str(np.round((toc-tic)/60, 2)) + ' (min)')
            
    # Plot outcomes
    print('\t\t\t\t\tPlotting outcomes now.')
    
    split_dev_names = [grouped_train_names[dev_split_decode_array[sd_i,0]] +\
                       ', ' + grouped_train_names[dev_split_decode_array[sd_i,1]] \
                           for sd_i in range(num_dev)]
    unique_split_inds = np.sort(
        np.unique(split_dev_names, return_index=True)[1])
    unique_split_dev_names = [split_dev_names[i] for i in unique_split_inds]
    split_counts = []
    split_dict = dict()
    for usdn in unique_split_dev_names:
        sd_match = [i for i, sdn in enumerate(split_dev_names) if sdn == usdn]
        split_counts.append(len(sd_match))
        split_dict[usdn] = dict()
        split_dict[usdn]['indices'] = sd_match
        split_dict[usdn]['count'] = len(sd_match)
    np.save(os.path.join(decode_dir,segment_names[s_i_test] + \
                     '_split_dict.npy'), split_dict, allow_pickle=True)
    split_counts = np.array(split_counts)
    f_split_counts = plt.figure(figsize=(5,5))
    plt.bar(np.arange(len(unique_split_dev_names)),split_counts/np.sum(split_counts))
    plt.xticks(np.arange(len(unique_split_dev_names)),unique_split_dev_names,\
               rotation=45,ha='right')
    plt.title('Pair Distribution')
    plt.ylabel('Fraction of Events')
    plt.tight_layout()
    f_split_counts.savefig(os.path.join(decode_dir,segment_names[s_i_test] + \
                        '_split_rates.png'))
    f_split_counts.savefig(os.path.join(decode_dir,segment_names[s_i_test] + \
                        '_split_rates.svg'))
    plt.close(f_split_counts)
    
def hotelling_t2(X, Y):
    """
    Perform Hotelling's T-squared test on two samples.
    """
    n1, p = X.shape
    n2, _ = Y.shape

    X_bar = np.mean(X, axis=0)
    Y_bar = np.mean(Y, axis=0)

    S_pooled = ((n1 - 1) * np.cov(X.T) + (n2 - 1) * np.cov(Y.T)) / (n1 + n2 - 2)

    T2 = ((n1 * n2) / (n1 + n2)) * (X_bar - Y_bar) @ np.linalg.inv(S_pooled) @ (X_bar - Y_bar).T

    F = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * T2

    p_value = 1 - f.cdf(F, p, n1 + n2 - p - 1)

    return T2, F, p_value

    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 09:36:26 2025

@author: hannahgermaine
"""

import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

def run_analysis_plots_by_decode_pair(unique_given_names,unique_training_categories,\
                                      unique_bin_counts, unique_decode_pairs, lstm_save_dir):
    
    num_anim = len(unique_given_names)
    num_tastes = len(unique_training_categories)
    segment_keep_inds = [0,2,4]
    num_seg = len(segment_keep_inds)
    segment_names = ['Pre-Taste', 'Post-Taste', 'Sickness']
    
    for b_i in unique_bin_counts:
        #Save dir
        bin_name = str(b_i) + '_bins'
        bin_save_dir = os.path.join(lstm_save_dir,bin_name)
        if not os.path.isdir(bin_save_dir):
            os.mkdir(bin_save_dir)
        #Run difference by pair type
        #Run difference analysis by pair type   
        for udp_i, udp in enumerate(unique_decode_pairs):
            anim_true_data = np.nan*np.ones((num_anim,num_tastes,num_seg)) #Fraction of all events
            anim_diff_data = np.nan*np.ones((num_anim,num_tastes,num_seg)) #Fraction of all events diff (or all nonnan)
            anim_overlap_data = np.nan*np.ones((num_anim,num_tastes,num_seg)) #Fraction of true decoded
            anim_corr_data = np.nan*np.ones((num_anim,num_tastes,num_seg)) #Binary vectors of length all events
            #Collect data
            for gn_i, gn in enumerate(unique_given_names):
                gn_categories = lstm_dict[gn][bin_name]['training_categories']
                gn_data = lstm_dict[gn][bin_name]['predictions'][udp]
                #gn_data contains null and true data
                for s_i in range(num_seg):
                    #Pull max indices
                    if len(np.shape(gn_data['true'][s_i])) > 1: #Not yet argmax
                        true_argmax = np.argmax(gn_data['true'][s_i],1)
                        null_argmax = np.argmax(gn_data['control'][s_i],1)
                    else: #Already argmax and contains NaNs
                        true_data = gn_data['true'][s_i]
                        true_nonnan_inds = np.where(~np.isnan(true_data))[0]
                        null_data = gn_data['control'][s_i]
                        null_nonnan_inds = np.where(~np.isnan(null_data))[0]
                        nonnan_inds = np.intersect1d(true_nonnan_inds,null_nonnan_inds)
                        true_argmax = true_data[nonnan_inds]
                        null_argmax = null_data[nonnan_inds]
                    #Collect data by taste
                    for utc_i, utc in enumerate(unique_training_categories):
                        t_i = np.where(np.array(gn_categories) == utc)[0]
                        #For each taste, collect fraction of predictions data
                        if len(true_argmax) > 0:
                            true_inds = np.where(true_argmax == t_i)[0]
                            null_inds = np.where(null_argmax == t_i)[0]
                            true_frac = len(true_inds)/len(true_argmax)
                            anim_true_data[gn_i,utc_i,s_i] = true_frac
                            null_frac = len(null_inds)/len(true_argmax)
                            anim_diff_data[gn_i,utc_i,s_i] = true_frac - null_frac
                            if len(true_inds) > 0:
                                #For each taste collect fraction of overlapping to true predictions
                                intersect_inds = np.intersect1d(true_inds,null_inds)
                                anim_overlap_data[gn_i,utc_i,s_i] = len(intersect_inds)/len(true_inds)
                                if len(null_inds) > 0:
                                    #For each taste get correlation of predictions
                                    true_bin = np.zeros(len(true_argmax))
                                    true_bin[true_inds] = 1
                                    null_bin = np.zeros(len(true_argmax))
                                    null_bin[null_inds] = 1
                                    pearson_stat = pearsonr(true_bin,null_bin)
                                    anim_corr_data[gn_i,utc_i,s_i] = pearson_stat.statistic
                                else:
                                    anim_corr_data[gn_i,utc_i,s_i] = 0
                            else:
                                anim_overlap_data[gn_i,utc_i,s_i] = 0
                                anim_corr_data[gn_i,utc_i,s_i] = 0
                        else:
                            anim_true_data[gn_i,utc_i,s_i] = 0
                            anim_diff_data[gn_i,utc_i,s_i] = 0
                            anim_overlap_data[gn_i,utc_i,s_i] = 0
                            anim_corr_data[gn_i,utc_i,s_i] = 0
            #Plot data
            plot_diff_data(anim_true_data, anim_diff_data, anim_overlap_data, \
                           anim_corr_data, segment_names, unique_training_categories, \
                           udp, bin_save_dir)

def plot_diff_data(anim_true_data, anim_diff_data, anim_overlap_data, \
               anim_corr_data, segment_names, unique_training_categories, \
               udp, bin_save_dir):
    
    #Plot results
    # True Fractions
    plot_by_seg(unique_given_names,unique_training_categories,\
                      segment_names,anim_true_data,bin_save_dir,udp,\
                    'Fraction True','frac','Decode Fraction by Segment')
    
    plot_by_taste(unique_given_names,unique_training_categories,\
                     segment_names,anim_true_data,bin_save_dir,udp,\
                    'Fraction True','frac','Decode Fraction by Taste')
    
    #    Difference
    plot_by_seg(unique_given_names,unique_training_categories,\
                      segment_names,anim_diff_data,bin_save_dir,udp,\
                    'Fraction True - Fraction Control','diff','Decode Fraction by Segment')
    
    plot_by_taste(unique_given_names,unique_training_categories,\
                     segment_names,anim_diff_data,bin_save_dir,udp,\
                    'Fraction True - Fraction Control','diff','Decode Fraction by Taste')
    
    #    Overlap Fraction
    plot_by_seg(unique_given_names,unique_training_categories,\
                      segment_names,anim_overlap_data,bin_save_dir,udp,\
                    'Num Overlap / Num True','overlap','Overlap Fraction by Segment')
        
    plot_by_taste(unique_given_names,unique_training_categories,\
                     segment_names,anim_overlap_data,bin_save_dir,udp,\
                    'Num Overlap / Num True','overlap','Overlap Fraction by Taste')
    
    #    Correlation
    plot_by_seg(unique_given_names,unique_training_categories,\
                      segment_names,anim_corr_data,bin_save_dir,udp,\
                    'Pearson R Correlation','corr','Correlation by Segment')
    
    plot_by_taste(unique_given_names,unique_training_categories,\
                     segment_names,anim_corr_data,bin_save_dir,udp,\
                    'Pearson R Correlation','corr','Correlation by Taste')
    
def plot_by_seg(unique_given_names,unique_training_categories,\
                  segment_names,data,bin_save_dir,udp,\
                      y_label_text,name_type,title):
    
    num_anim = len(unique_given_names)
    num_tastes = len(unique_training_categories)
    num_seg = len(segment_names)
    
    f_seg, ax_seg = plt.subplots(ncols = num_seg, figsize = (num_seg*5,5))
    for s_i in range(num_seg):
        seg_data = np.squeeze(data[:,:,s_i])
        ax_seg[s_i].axhline(0,linestyle='dashed',color='k',alpha=0.5)
        for gn_i in range(num_anim):
            ax_seg[s_i].scatter(np.arange(num_tastes)+1,seg_data[gn_i,:],color='g',\
                        alpha=0.5)
        ax_seg[s_i].boxplot(seg_data)
        ax_seg[s_i].set_xticks(np.arange(num_tastes)+1,unique_training_categories)
        ax_seg[s_i].set_ylabel(y_label_text)
        ax_seg[s_i].set_title(segment_names[s_i])
    plt.suptitle(title)
    plt.tight_layout()
    f_seg.savefig(os.path.join(bin_save_dir,udp + '_' + name_type + '_boxplots_by_seg.png'))
    f_seg.savefig(os.path.join(bin_save_dir,udp + '_' + name_type + '_boxplots_by_seg.svg'))
    plt.close(f_seg)
    
def plot_by_taste(unique_given_names,unique_training_categories,\
                  segment_names,data,bin_save_dir,udp,\
                      y_label_text,name_type,title):
    
    num_anim = len(unique_given_names)
    num_tastes = len(unique_training_categories)
    num_seg = len(segment_names)
    
    f_taste, ax_taste = plt.subplots(ncols = num_tastes, figsize = (num_seg*5,5))
    for t_i in range(num_tastes):
        taste_data = np.squeeze(data[:,t_i,:])
        ax_taste[t_i].axhline(0,linestyle='dashed',color='k',alpha=0.5)
        for gn_i in range(num_anim):
            ax_taste[t_i].scatter(np.arange(num_seg)+1,taste_data[gn_i,:],color='g',\
                        alpha=0.5)
        ax_taste[t_i].boxplot(taste_data)
        ax_taste[t_i].set_xticks(np.arange(num_seg)+1,segment_names)
        ax_taste[t_i].set_ylabel(y_label_text)
        ax_taste[t_i].set_title(unique_training_categories[t_i])
    plt.suptitle(title)
    plt.tight_layout()
    f_taste.savefig(os.path.join(bin_save_dir,udp + '_' + name_type + '_boxplots_by_taste.png'))
    f_taste.savefig(os.path.join(bin_save_dir,udp + '_' + name_type + '_boxplots_by_taste.svg'))
    plt.close(f_taste)
    
def plot_accuracy_data(lstm_dict,unique_given_names,unique_training_categories,\
                       unique_bin_counts,lstm_save_dir):
    
    num_anim = len(unique_given_names)
    num_tastes = len(unique_training_categories)
    true_tastes = np.array([unique_training_categories[i] for i in range(len(unique_training_categories)) if \
                       (unique_training_categories[i][:4] != 'none')*(unique_training_categories[i][:4] != 'null')])
    num_bins = len(unique_bin_counts)
    
    #Build out cross-validation analysis
    cross_val_save_dir = os.path.join(lstm_save_dir,'cross_validation')
    if not os.path.isdir(cross_val_save_dir):
        os.mkdir(cross_val_save_dir)
    cross_val_max_size_true = np.zeros((num_anim,num_bins))
    cross_val_max_accuracy_true = np.zeros((num_anim,num_bins))
    cross_val_max_size_all = np.zeros((num_anim,num_bins))
    cross_val_max_accuracy_all = np.zeros((num_anim,num_bins))
    cross_validation_data = dict()
    for gn_i, gn in enumerate(unique_given_names):
        cross_validation_data[gn] = dict()
        for bc_ind, bc_i in enumerate(unique_bin_counts):
            bc_name = str(bc_i) + '_bins'
            cross_validation_data[gn][bc_name] = dict()
            training_categories = lstm_dict[gn][bc_name]['training_categories']
            true_cat = np.array([i for i in range(len(training_categories)) if \
                               (training_categories[i][:4] != 'none')*(training_categories[i][:4] != 'null')])
            control_cat = np.setdiff1d(np.arange(len(training_categories)),true_cat)
            cross_val_dict = lstm_dict[gn][bc_name]['cross_validation']
            num_layer_sizes = len(cross_val_dict)
            layer_sizes = np.zeros(num_layer_sizes) #Store latent dim sizes
            true_accuracy = np.zeros(num_layer_sizes)
            total_accuracy = np.zeros(num_layer_sizes)
            for l_i in range(num_layer_sizes):
                layer_sizes[l_i] = cross_val_dict[l_i]['latent_dim']
                predictions = cross_val_dict[l_i]['predictions']
                argmax_predictions = np.argmax(predictions,1)
                true_labels = cross_val_dict[l_i]['true_labels']
                true_label_inds = np.argmax(true_labels,1)
                #Calculate true taste accuracy
                total_true_correct = 0
                total_true = 0
                for t_i in true_cat:
                    true_i = np.where(true_label_inds == t_i)[0]
                    total_true += len(true_i)
                    total_true_correct += len(np.where(argmax_predictions[true_i] == t_i)[0])
                #Calculate control accuracy
                control_inds = []
                for c_i in control_cat:
                    control_inds.extend(list(np.where(true_label_inds == c_i)[0]))
                control_inds = np.array(control_inds)
                total_control = len(control_inds)
                total_true_control = 0
                for pc in argmax_predictions[control_inds]:
                    if len(np.where(control_cat == pc)[0]) > 0:
                        total_true_control += 1
                #Store results        
                true_accuracy[l_i] = total_true_correct/total_true
                total_accuracy[l_i] = (total_true_correct+total_true_control)/(total_true+total_control)
            cross_validation_data[gn][bc_name]['layer_sizes'] = layer_sizes
            cross_validation_data[gn][bc_name]['true_accuracy'] = true_accuracy
            cross_validation_data[gn][bc_name]['total_accuracy'] = total_accuracy
            #Store maximal locations and values
            true_max_ind = np.argmax(true_accuracy)
            cross_val_max_size_true[gn_i,bc_ind] = layer_sizes[true_max_ind]
            cross_val_max_accuracy_true[gn_i,bc_ind] = true_accuracy[true_max_ind]
            all_max_ind = np.argmax(total_accuracy)
            cross_val_max_size_all[gn_i,bc_ind] = layer_sizes[all_max_ind]
            cross_val_max_accuracy_all[gn_i,bc_ind] = total_accuracy[all_max_ind]
            
    np.save(os.path.join(cross_val_save_dir,'cross_validation_data.npy'),cross_validation_data,allow_pickle=True)
    np.save(os.path.join(cross_val_save_dir,'cross_val_max_size_true.npy'),cross_val_max_size_true,allow_pickle=True)
    np.save(os.path.join(cross_val_save_dir,'cross_val_max_accuracy_true.npy'),cross_val_max_accuracy_true,allow_pickle=True)
    np.save(os.path.join(cross_val_save_dir,'cross_val_max_size_all.npy'),cross_val_max_size_all,allow_pickle=True)
    np.save(os.path.join(cross_val_save_dir,'cross_val_max_accuracy_all.npy'),cross_val_max_accuracy_all,allow_pickle=True)

    #Plot
    f_accuracy, ax_accuracy = plt.subplots(nrows = 2,ncols = 2,figsize=(5,5))
    #Plot true accuracy
    ax_accuracy[0,0].axhline(1/len(true_tastes),linestyle='dashed',color='k',alpha=0.3)
    for gn_i in range(num_anim):
        ax_accuracy[0,0].scatter(np.arange(num_bins)+1,cross_val_max_accuracy_true[gn_i,:],\
                                 color='g',alpha=0.5,)
    ax_accuracy[0,0].boxplot(cross_val_max_accuracy_true)
    ax_accuracy[0,0].set_title('True Taste Accuracy')
    ax_accuracy[0,0].set_xticks(np.arange(num_bins)+1,unique_bin_counts)
    ax_accuracy[0,0].set_ylabel('Accuracy')
    ax_accuracy[0,0].set_xlabel('Num Bins')
    #Plot true accuracy best latent dim sizes
    for gn_i in range(num_anim):
        ax_accuracy[0,1].scatter(np.arange(num_bins)+1,cross_val_max_size_true[gn_i,:],\
                                 color='g',alpha=0.5,)
    ax_accuracy[0,1].boxplot(cross_val_max_size_true)
    ax_accuracy[0,1].set_title('Best Latent Dim Size')
    ax_accuracy[0,1].set_xticks(np.arange(num_bins)+1,unique_bin_counts)
    ax_accuracy[0,1].set_ylabel('True Taste Best\nLatent Dim Size')
    ax_accuracy[0,1].set_xlabel('Num Bins')
    #Plot true accuracy
    ax_accuracy[1,0].axhline(1/len(true_tastes),linestyle='dashed',color='k',alpha=0.3)
    for gn_i in range(num_anim):
        ax_accuracy[1,0].scatter(np.arange(num_bins)+1,cross_val_max_accuracy_all[gn_i,:],\
                                 color='g',alpha=0.5,)
    ax_accuracy[1,0].boxplot(cross_val_max_accuracy_all)
    ax_accuracy[1,0].set_title('All Taste Accuracy')
    ax_accuracy[1,0].set_xticks(np.arange(num_bins)+1,unique_bin_counts)
    ax_accuracy[1,0].set_ylabel('Accuracy')
    ax_accuracy[1,0].set_xlabel('Num Bins')
    #Plot true accuracy best latent dim sizes
    for gn_i in range(num_anim):
        ax_accuracy[1,1].scatter(np.arange(num_bins)+1,cross_val_max_size_all[gn_i,:],\
                                 color='g',alpha=0.5,)
    ax_accuracy[1,1].boxplot(cross_val_max_size_all)
    ax_accuracy[1,1].set_title('Best Latent Dim Size')
    ax_accuracy[1,1].set_xticks(np.arange(num_bins)+1,unique_bin_counts)
    ax_accuracy[1,1].set_ylabel('All Taste Best\nLatent Dim Size')
    ax_accuracy[1,1].set_xlabel('Num Bins')
    #Finish plot
    plt.suptitle('LSTM Decoding Accuracies')
    plt.tight_layout()
    f_accuracy.savefig(os.path.join(cross_val_save_dir,'accuracy_boxplots.png'))
    f_accuracy.savefig(os.path.join(cross_val_save_dir,'accuracy_boxplots.svg'))
    plt.close(f_accuracy)
    
    
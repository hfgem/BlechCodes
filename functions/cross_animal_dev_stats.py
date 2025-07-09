#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:33:04 2025

@author: hannahgermaine
"""

import os
import csv
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.stats import f_oneway, ttest_ind, ks_2samp
from itertools import combinations

def cross_dataset_dev_stats_plots(dev_stats_data, unique_given_names, 
                                            unique_dev_stats_names, unique_segment_names, 
                                            results_dir):   
    """This function is dedicated to plotting deviation statistics across animals.
    INPUTS:
        - dev_stats_data: dictionary containing data regarding deviation statistics
            organized as follows:
                - dev_stats_data.keys() are the unique_given_names
                - dev_stats_data[name]['dev_stats'] = dict containing deviation stats
                - dev_stats_data[name]['dev_stats'].keys() are the unique_dev_stats_names
                - dev_stats_data[name]['dev_stats'][stats_name] = dict containing specific statistic results
                - dev_stats_data[name]['dev_stats'][stats_name].keys() are the unique_segment_names
                - dev_stats_data[name]['dev_stats'][stats_name][seg_name] = array of deviation statistics
        - unique_given_names: names of imported datasets in dict
        - unique_dev_stats_names: names of types of statistics
        - unique_segment_names: names of segments analyzed
        - results_dir: storage directory
    OUTPUTS:
        - Plots with statistical results
    """
    colors = ['green','royalblue','blueviolet','teal','deeppink', \
              'springgreen','turquoise', 'midnightblue', 'lightskyblue', \
              'palevioletred', 'darkslateblue']
    
    plot_side = np.ceil(np.sqrt(len(unique_dev_stats_names))).astype('int')
    plot_inds = np.reshape(np.arange(plot_side**2),(plot_side,plot_side))
    seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
    f_stats, ax_stats = plt.subplots(nrows=plot_side,ncols=plot_side,
                                     figsize=(8,8),sharex=True)
    for ds_i, ds in enumerate(unique_dev_stats_names):
        ds_name = (' ').join(ds.split('_')[:-1])
        ax_r, ax_c = np.where(plot_inds == ds_i)
        all_animal_stats = np.zeros((len(unique_given_names),len(unique_segment_names)))
        for gn_i, gn in enumerate(unique_given_names):
            seg_means = np.zeros(len(unique_segment_names))
            for s_i, sn in enumerate(unique_segment_names):
                all_animal_stats[gn_i,s_i] = np.nanmean(dev_stats_data[gn]['dev_stats'][ds][sn])
        ax_stats[ax_r[0],ax_c[0]].boxplot(all_animal_stats)
        for s_i in range(len(unique_segment_names)):
            scat_x = s_i+1+np.random.randn(len(unique_given_names))/10
            ax_stats[ax_r[0],ax_c[0]].scatter(scat_x,all_animal_stats[:,s_i],\
                                              color='g',alpha=0.3)
        ax_stats[ax_r[0],ax_c[0]].set_xticks(np.arange(len(unique_segment_names))+1,unique_segment_names,\
                                 rotation=45)
        ax_stats[ax_r[0],ax_c[0]].set_ylabel(ds_name)
        max_y = np.nanmax(all_animal_stats)
        #ANOVA test
        result = f_oneway(*list(all_animal_stats.T))
        if result.pvalue <= 0.05:
            ax_stats[ax_r[0],ax_c[0]].set_title(ds_name + ' *ANOVA')
        else:
            ax_stats[ax_r[0],ax_c[0]].set_title(ds_name)
        #Pairwise TTest
        for sp_0, sp_1 in seg_pairs:
            result = ttest_ind(all_animal_stats[:,sp_0],all_animal_stats[:,sp_1])
            if result.pvalue <= 0.05:
                max_y += max_y*0.1
                plt.plot([sp_0+1,sp_1+1],[max_y,max_y])
                max_y += max_y*0.1
                plt.text(1+sp_0+(sp_1-sp_0)/2,max_y,'*TT')
            result = ks_2samp(all_animal_stats[:,sp_0],all_animal_stats[:,sp_1])
            if result.pvalue <= 0.05:
                max_y += max_y*0.1
                plt.plot([sp_0+1,sp_1+1],[max_y,max_y])
                max_y += max_y*0.1
                plt.text(1+sp_0+(sp_1-sp_0)/2,max_y,'*KS')
    plt.tight_layout()
    f_stats.savefig(os.path.join(results_dir,'overview_boxplots.png'))
    f_stats.savefig(os.path.join(results_dir,'overview_boxplots.svg'))
  
    #By-stat distributions
    for dev_stat in unique_dev_stats_names:
        max_val = 0
        max_mean = 0
        cross_animal_means = []
        combined_animal_results = []
        for seg_name in unique_segment_names:
            combined_seg_results = []
            seg_animal_means = []
            for name in unique_given_names:
                dataset = dev_stats_data[name]['dev_stats'][dev_stat][seg_name]
                try:
                    num_vals = len(dataset)
                    combined_seg_results.extend(list(dataset))
                    seg_animal_means.extend([np.nanmean(dataset)])
                except:
                    if (dataset.dtype == float) or (dataset.dtype == int):
                        combined_seg_results.extend([dataset])
                        seg_animal_means.extend([dataset])
            if max(np.array(combined_seg_results)) > max_val:
                max_val = max(np.array(combined_seg_results))
            if max(np.array(seg_animal_means)) > max_mean:
                max_mean = max(np.array(seg_animal_means))
            cross_animal_means.append(seg_animal_means)
            combined_animal_results.append(combined_seg_results)
        max_val = np.ceil(max_val).astype('int')
        max_mean = np.ceil(max_mean).astype('int')
        #Create figure of combined data
        f, ax = plt.subplots(nrows=1, ncols=2, figsize = (4*2,4))
        #plt.boxplot(combined_animal_results,labels=unique_segment_names)
        for s_i, seg_name in enumerate(unique_segment_names):
            cdf_data = np.array(combined_animal_results[s_i])
            nonnan_data = cdf_data[~np.isnan(cdf_data)]
            min_x = np.nanmin(nonnan_data)
            max_x = np.nanmax(nonnan_data)
            cdf_x = np.linspace(min_x,max_x,1000)
            cdf_vals = np.array([len(np.where(nonnan_data <= i)[0])/len(nonnan_data) for i in cdf_x])
            ax[0].plot(cdf_x,cdf_vals,label=seg_name,color=colors[s_i])
            ax[1].hist(nonnan_data,bins=max_val,histtype='step',\
                     density=True,label='seg_name',color=colors[s_i])
            data_mean = np.nanmean(combined_animal_results[s_i])
            ax[1].axvline(data_mean,color=colors[s_i],\
                          label=seg_name + 'mean = ' + str(np.round(data_mean,2)))
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='upper right')
        ax[0].set_xlabel(dev_stat)
        ax[1].set_xlabel(dev_stat)
        ax[0].set_ylabel('Cumulative Fraction')
        ax[1].set_xlabel('Density')
        ax[0].set_title('Cumulative Distribution')
        ax[1].set_title('Density Distribution')
        seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
        pair_statistics = ('Segment 1').ljust(10,' ') + ' | ' + ('Segment 2').ljust(10,' ') + ' | ' + ('TTest').ljust(5,' ')
        for sp in seg_pairs:
            seg_1 = sp[0]
            seg_1_data = np.array(combined_animal_results[seg_1])
            seg_2 = sp[1]
            seg_2_data = np.array(combined_animal_results[seg_2])
            ttest_result = ttest_ind(seg_1_data[~np.isnan(seg_1_data)],seg_2_data[~np.isnan(seg_2_data)])
            if ttest_result[1] <= 0.05:
                pair_statistics = pair_statistics + '\n' + \
                    unique_segment_names[seg_1].ljust(10, ' ') + '   ' + \
                        unique_segment_names[seg_2].ljust(10, ' ') + '   ' + ('*').ljust(5,' ')
            else:
                pair_statistics = pair_statistics + '\n' + \
                    unique_segment_names[seg_1].ljust(10, ' ') + '   ' + \
                        unique_segment_names[seg_2].ljust(10, ' ') + '   ' + ('n.s.').ljust(5,' ')
        ax[0].text(max_val,0,pair_statistics,horizontalalignment='right',\
                 verticalalignment='bottom',fontsize=10,color='k',
                 bbox=dict(boxstyle="round",color="grey", alpha=0.5))
        plt.tight_layout()
        f.savefig(os.path.join(results_dir,dev_stat + '_distributions.png'))
        f.savefig(os.path.join(results_dir,dev_stat + '_distributions.svg'))
        plt.close(f)
        
        #Create figure of animal means
        f_means = plt.figure(figsize=(5,5))
        for n_i, name in enumerate(unique_given_names):
            animal_means = []
            for s_i, seg_name in enumerate(unique_segment_names):
                animal_means.extend([cross_animal_means[s_i][n_i]])
            plt.plot(np.arange(len(unique_segment_names)),animal_means,\
                     color=colors[n_i],alpha=0.5,label=name + ' Mean')
        all_animal_means = []
        all_animal_stds = []
        for s_i, seg_name in enumerate(unique_segment_names):
            all_animal_means.extend([np.nanmean(combined_animal_results[s_i])])
            all_animal_stds.extend([np.nanstd(combined_animal_results[s_i])])
        plt.plot(np.arange(len(unique_segment_names)),all_animal_means,\
                 color='k',linestyle='dashed',label='Mean')
        plt.fill_between(np.arange(len(unique_segment_names)),\
                         np.array(all_animal_means) - np.array(all_animal_stds),\
                        np.array(all_animal_means) + np.array(all_animal_stds),\
                        color='k',alpha=0.2,label='Std')
        plt.xticks(np.arange(len(unique_segment_names)),unique_segment_names)
        plt.legend()
        plt.ylim([0,max_mean + np.nanmax(all_animal_stds)])
        plt.title(dev_stat)
        f_means.savefig(os.path.join(results_dir,dev_stat + '_means.png'))
        f_means.savefig(os.path.join(results_dir,dev_stat + '_means.svg'))
        plt.close(f_means)
    
def basic_dev_stats_plots(dev_stats_data,unique_given_names,unique_dev_stats_names,\
                          unique_segment_names,dev_stats_results_dir):
    
    plot_side = np.ceil(np.sqrt(len(unique_dev_stats_names))).astype('int')
    plot_inds = np.reshape(np.arange(plot_side**2),(plot_side,plot_side))
    seg_pairs = list(combinations(np.arange(len(unique_segment_names)),2))
    f_stats, ax_stats = plt.subplots(nrows=plot_side,ncols=plot_side,
                                     figsize=(8,8),sharex=True)
    for ds_i, ds in enumerate(unique_dev_stats_names):
        ds_name = (' ').join(ds.split('_')[:-1])
        ax_r, ax_c = np.where(plot_inds == ds_i)
        all_animal_stats = np.zeros((len(unique_given_names),len(unique_segment_names)))
        for gn_i, gn in enumerate(unique_given_names):
            seg_means = np.zeros(len(unique_segment_names))
            for s_i, sn in enumerate(unique_segment_names):
                all_animal_stats[gn_i,s_i] = np.nanmean(dev_stats_data[gn]['dev_stats'][ds][sn])
        ax_stats[ax_r[0],ax_c[0]].boxplot(all_animal_stats)
        for s_i in range(len(unique_segment_names)):
            scat_x = s_i+1+np.random.randn(len(unique_given_names))/10
            ax_stats[ax_r[0],ax_c[0]].scatter(scat_x,all_animal_stats[:,s_i],\
                                              color='g',alpha=0.3)
        ax_stats[ax_r[0],ax_c[0]].set_xticks(np.arange(len(unique_segment_names))+1,unique_segment_names,\
                                 rotation=45)
        ax_stats[ax_r[0],ax_c[0]].set_ylabel(ds_name)
        max_y = np.nanmax(all_animal_stats)
        #ANOVA test
        result = f_oneway(*list(all_animal_stats.T))
        if result.pvalue <= 0.05:
            ax_stats[ax_r[0],ax_c[0]].set_title(ds_name + ' *ANOVA')
        else:
            ax_stats[ax_r[0],ax_c[0]].set_title(ds_name)
        #Pairwise TTest
        for sp_0, sp_1 in seg_pairs:
            result = ttest_ind(all_animal_stats[:,sp_0],all_animal_stats[:,sp_1])
            if result.pvalue <= 0.05:
                max_y += max_y*0.1
                plt.plot([sp_0+1,sp_1+1],[max_y,max_y])
                max_y += max_y*0.1
                plt.text(1+sp_0+(sp_1-sp_0)/2,max_y,'*TT')
            result = ks_2samp(all_animal_stats[:,sp_0],all_animal_stats[:,sp_1])
            if result.pvalue <= 0.05:
                max_y += max_y*0.1
                plt.plot([sp_0+1,sp_1+1],[max_y,max_y])
                max_y += max_y*0.1
                plt.text(1+sp_0+(sp_1-sp_0)/2,max_y,'*KS')
    plt.tight_layout()
    f_stats.savefig(os.path.join(dev_stats_results_dir,'overview_boxplots.png'))
    f_stats.savefig(os.path.join(dev_stats_results_dir,'overview_boxplots.svg'))
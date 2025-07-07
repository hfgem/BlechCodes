#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 11:31:49 2025

@author: hannahgermaine
"""

import os
import csv
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import f

def taste_stat_collection(taste_data,taste_results_dir,verbose=False):
    
    
    unique_given_names = list(taste_data.keys())
    all_taste_names = []
    taste_stats = dict()
    taste_stats['state_correlations'] = dict()
    taste_stats['taste_correlations'] = dict()
    for gn_i, gn in enumerate(unique_given_names):
        taste_names = taste_data[gn]['dig_in_names']
        num_tastes = len(taste_names)
        all_taste_names.extend(list(taste_names))
        taste_stat_keys = list(taste_stats['state_correlations'].keys())
        taste_pairs = list(combinations(np.arange(len(taste_names)),2))
        taste_pair_names = [taste_names[t_0] + ' v. ' + taste_names[t_1] for t_0,t_1 in taste_pairs]
        taste_responses = taste_data[gn]['tastant_fr_dist_z_pop']
        taste_num_deliv = taste_data[gn]['taste_num_deliv']
        max_cp = 0
        
        #Test each taste's state responses against each other w/ Hotellings
        for t_i, tname in enumerate(taste_names):
            if len(np.intersect1d(taste_stat_keys,[tname])) == 0:
                taste_stats['state_correlations'][tname] = dict()
            cp_pair_keys = list(taste_stats['state_correlations'][tname].keys())
            num_deliv = int(taste_num_deliv[t_i])
            num_cp = len(taste_responses[t_i][0])
            if num_cp > max_cp:
                max_cp = num_cp
            cp_pairs = list(combinations(np.arange(num_cp),2))
            for cp_1, cp_2 in cp_pairs:
                cp_pair_name = str(cp_1) + ' v. ' + str(cp_2)
                if len(np.intersect1d(cp_pair_keys,[cp_pair_name])) == 0:
                    taste_stats['state_correlations'][tname][cp_pair_name] = []
                cp_1_mat = []
                cp_2_mat = []
                for d_i in range(num_deliv):
                    cp_1_mat.append(list(np.squeeze(taste_responses[t_i][d_i][cp_1])))
                    cp_2_mat.append(list(np.squeeze(taste_responses[t_i][d_i][cp_2])))
                cp_1_mat = np.array(cp_1_mat)
                cp_2_mat = np.array(cp_2_mat)
                _,_,p = hotelling_t2(cp_1_mat,cp_2_mat)
                taste_stats['state_correlations'][tname][cp_pair_name].append(p)
                
        #Test pairs of tastes' state responses against each other w/ Hotellings
        taste_stat_keys = list(taste_stats['taste_correlations'].keys())
        for t_i, tp in enumerate(taste_pairs):
            tp_name = taste_pair_names[t_i]
            if len(np.intersect1d(taste_stat_keys,[tp_name])) == 0:
                taste_stats['taste_correlations'][tp_name] = dict()
            tp_keys = list(taste_stats['taste_correlations'][tp_name].keys())
            t_i1 = tp[0]
            t_i2 = tp[1]
            for cp_i in range(max_cp):
                if len(np.intersect1d(tp_keys,[str(cp_i)])) == 0:
                    taste_stats['taste_correlations'][tp_name][str(cp_i)] = []
                try:
                    t_1_mat = []
                    for d_i in range(int(taste_num_deliv[t_i1])):
                        t_1_mat.append(list(np.squeeze(taste_responses[t_i1][d_i][cp_i])))
                    t_2_mat = []
                    for d_i in range(int(taste_num_deliv[t_i2])):
                        t_2_mat.append(list(np.squeeze(taste_responses[t_i2][d_i][cp_i])))
                    t_1_mat = np.array(t_1_mat)
                    t_2_mat = np.array(t_2_mat)
                    _,_,p = hotelling_t2(t_1_mat,t_2_mat)
                    taste_stats['taste_correlations'][tp_name][str(cp_i)].append(p)
                except:
                    if verbose == True:
                        print(tp_name + ' state ' + str(cp_i) + ' pair not available.')
    
    return taste_stats
        
def plot_corr_outputs(taste_stats,taste_results_dir):
    state_correlations = taste_stats['state_correlations']
    unique_taste_names = list(state_correlations.keys())
    unique_state_pairs = []
    for tn in unique_taste_names:
        unique_state_pairs.extend(list(state_correlations[tn].keys()))
    unique_state_pair_indices = np.sort(
        np.unique(unique_state_pairs, return_index=True)[1])
    unique_state_pairs = [unique_state_pairs[i]
                            for i in unique_state_pair_indices]
    taste_correlations = taste_stats['taste_correlations']
    unique_taste_pair_names = list(taste_correlations.keys())
    unique_cp_names = []
    for tpn in unique_taste_pair_names:
        unique_cp_names.extend(list(taste_correlations[tpn].keys()))
    unique_cp_indices = np.sort(
        np.unique(unique_cp_names, return_index=True)[1])
    unique_cp_names = [unique_cp_names[i]
                            for i in unique_cp_indices]
    max_pairs = np.max([len(unique_state_pairs),len(unique_cp_names)])
    f_pvals, ax_pvals = plt.subplots(nrows = 2, ncols = max_pairs,\
                                           sharex = False, sharey = True, \
                                               figsize=(8,5))
    for usp_i, usp in enumerate(unique_state_pairs):
        for t_i, tname in enumerate(unique_taste_names):
            ax_pvals[0,usp_i].boxplot(state_correlations[tname][usp],\
                                       positions=[t_i])
        ax_pvals[0,usp_i].axhline(0.05,color='k',linestyle='dashed',alpha=0.25)
        ax_pvals[0,usp_i].set_xticks(np.arange(len(unique_taste_names)),\
                                  unique_taste_names,rotation=45)
        ax_pvals[0,usp_i].set_title('State pair ' + usp + ' p-values')
    for ucn_i, ucn in enumerate(unique_cp_names):
        for tp_i, tpname in enumerate(unique_taste_pair_names):
            ax_pvals[1,ucn_i].boxplot(taste_correlations[tpname][ucn],\
                                       positions=[tp_i])
        ax_pvals[1,ucn_i].axhline(0.05,color='k',linestyle='dashed',alpha=0.25)
        ax_pvals[1,ucn_i].set_xticks(np.arange(len(unique_taste_pair_names)),\
                                  unique_taste_pair_names,rotation=45)
        ax_pvals[1,ucn_i].set_title('State ' + ucn + ' p-values')
    plt.tight_layout()
    f_pvals.savefig(os.path.join(taste_results_dir,'pairwise_corr_boxplots.png'))
    f_pvals.savefig(os.path.join(taste_results_dir,'pairwise_corr_boxplots.svg'))
    plt.close(f_pvals)
        
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

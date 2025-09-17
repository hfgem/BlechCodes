#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:15:54 2025

@author: Hannah Germaine

Cross-Animal Analysis Test Support: Use to test updates to functions and debug
"""

#%% Compare Multiday Support

import os
import tables
import numpy as np
from tkinter.filedialog import askdirectory
from functions.compare_multiday_analysis import run_compare_multiday_analysis
from functions.compare_conditions_funcs import int_input, bool_input

# Grab current directory and data directory / metadata
script_path = os.path.realpath(__file__)
blechcodes_dir = os.path.dirname(script_path)

multiday_data_dict = dict()
save_dir = ''

# _____Prompt user if they'd like to use previously stored correlation data_____
print("If you previously started an analysis, you may have a multiday_data_dict.npy file in the analysis folder.")
bool_val = bool_input(
    "Do you have a file stored you'd like to continue analyzing [y/n]? ")
if bool_val == 'y':
    print('Please select the storage folder.')
    save_dir = askdirectory()
    try:
        multiday_data_dict = np.load(os.path.join(save_dir,'multiday_data_dict.npy'),allow_pickle=True).item()
    except:
        print("Multiday data dict not found in given save folder. Aborting.")
        quit()
else:
    # _____Prompt user for the number of datasets needed in the analysis_____
    print("Conditions include across days and across animals (the number of separate spike sorted datasets).")
    num_cond = int_input(
        "How many conditions-worth of correlation data do you wish to import for this comparative analysis (integer value)? ")
    if num_cond >= 1:
        print("Multiple file import selected.")
    else:
        print("Single file import selected.")

    # _____Pull all data into a dictionary_____
    multiday_data_dict = dict()
    for nc in range(num_cond):
        # _____Get the directory of the hdf5 file_____
        print("Please select the folder where the data # " +
              str(nc+1) + " is stored.")
        data_dir = askdirectory()
        subfolders = os.listdir(data_dir)
        corr_exists = 0
        for sf in subfolders:
            if sf == 'Correlations':
                corr_exists = 1
             
        for corr_attempts in range(3):
            while corr_exists == 0:
                print("Try again - directory selected does not contain expected subfolders.")
                data_dir = askdirectory()
                subfolders = os.listdir(data_dir)
                corr_exists = 0
                for sf in subfolders:
                    if sf == 'Correlations':
                        corr_exists = 1
        
        if corr_exists == 0:
            print("Number of folder selection attempts exceeded. Program quitting.")
            quit()
            
        # Grab colloquial name
        print("Give a more colloquial name to the dataset.")
        folder_name = os.path.split(data_dir)[-1]
        data_name = input("Give a name to the dataset in folder /" + folder_name + "/: ")
        multiday_data_dict[data_name] = dict()
        multiday_data_dict[data_name]['data_dir'] = data_dir
        
        # Get day 1 folder
        folder_containing_data_dir = os.path.split(data_dir)[0]
        possible_day_1_folders = list(np.setdiff1d(os.listdir(folder_containing_data_dir),[folder_name, '.DS_Store']))
        folder_prompt = ''
        for fn_i, fn in enumerate(possible_day_1_folders):
            folder_prompt += str(fn_i) + ': ' + fn + '\n'
        folder_prompt += 'Please provide the above index of the day 1 data folder: '
        day_1_folder_ind = int_input(folder_prompt)
        day_1_folder = possible_day_1_folders[day_1_folder_ind]
        
        # Grab segment lengths from day 1 data
        day_1_folder_contents = os.listdir(os.path.join(folder_containing_data_dir,day_1_folder))
        for d1fc in day_1_folder_contents:
            if d1fc.split('.')[-1] == 'h5':
                hdf5_path = os.path.join(folder_containing_data_dir,day_1_folder,d1fc)
        blech_clust_h5 = tables.open_file(hdf5_path, 'r+', title = 'hdf5_file')
        segment_times = blech_clust_h5.root.experiment_components.segment_times[:]
        segment_names = [blech_clust_h5.root.experiment_components.segment_names[i].decode('UTF-8') for i in range(len(blech_clust_h5.root.experiment_components.segment_names))]
        blech_clust_h5.close()
        multiday_data_dict[data_name]['segment_times'] = segment_times
        multiday_data_dict[data_name]['segment_names'] = segment_names
        
        del data_dir, data_name
    del nc
    
    print('Please select the storage folder for comparative multiday analyses.')
    save_dir = askdirectory()
    np.save(os.path.join(save_dir,'multiday_data_dict.npy'),multiday_data_dict,allow_pickle=True)

import os
import warnings
import tqdm
import random
import numpy as np
from tkinter.filedialog import askdirectory
from functions.compare_multiday_funcs import select_analysis_groups

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)

warnings.filterwarnings("ignore")

#%% gather_lstm_data()
verbose = False

lstm_dict_path = os.path.join(save_dir,'lstm_data_dict.npy')
#Start with real corr data
try:
    lstm_dict = np.load(lstm_dict_path,allow_pickle=True).item()
except:
    lstm_dict = dict()
    data_names = list(multiday_data_dict.keys())
    for nc_i, dn in enumerate(data_names):
        lstm_dict[dn] = dict()
        data_dir = multiday_data_dict[dn]['data_dir']
        lstm_dir = os.path.join(data_dir,'LSTM_Decoding')
        bin_counts = os.listdir(lstm_dir)
        for bc in bin_counts:
            bc_dir = os.path.join(lstm_dir,bc)
            try: #Regular data
                bc_i = int(bc.split('_')[0])
                lstm_dict[dn][bc] = dict()
                lstm_dict[dn][bc]['num_bins'] = bc_i
                training_categories = np.load(os.path.join(bc_dir,'taste_unique_categories.npy'))
                if len(np.where(training_categories == 'salt_1')[0]) > 0:
                    training_categories[np.where(training_categories == 'salt_1')[0][0]] = 'NaCl_1'
                lstm_dict[dn][bc]['training_categories'] = training_categories
                #Get the training data
                lstm_dict[dn][bc]['training_data'] = dict()
                lstm_dict[dn][bc]['training_data']['training_matrices'] = np.load(os.path.join(bc_dir,'training_matrices.npy'))
                lstm_dict[dn][bc]['training_data']['training_labels'] = np.load(os.path.join(bc_dir,'training_labels.npy'))
                #Get the deviation data
                lstm_dict[dn][bc]['vectors'] = dict()
                lstm_dict[dn][bc]['vectors']['dev_fr_vecs'] = np.load(os.path.join(bc_dir,'dev_fr_vecs.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['vectors']['null_dev_fr_vecs'] = np.load(os.path.join(bc_dir,'null_dev_fr_vecs.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['vectors']['scaled_dev_fr_vecs'] = np.load(os.path.join(bc_dir,'scaled_dev_fr_vecs.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['vectors']['scaled_null_dev_fr_vecs'] = np.load(os.path.join(bc_dir,'scaled_null_dev_matrices.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['vectors']['shuffled_dev_fr_vecs'] = np.load(os.path.join(bc_dir,'shuffled_dev_fr_vecs.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['vectors']['shuffled_scaled_dev_fr_vecs'] = np.load(os.path.join(bc_dir,'shuffled_scaled_dev_fr_vecs.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['predictions'] = dict()
                #Get the unscaled predictions
                lstm_dict[dn][bc]['predictions']['unscaled'] = dict()
                lstm_dict[dn][bc]['predictions']['unscaled']['true'] = np.load(os.path.join(bc_dir,'predictions.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['predictions']['unscaled']['control'] = np.load(os.path.join(bc_dir,'null_predictions.npy'),allow_pickle=True).item()
                #Get the unscaled thresholded predictions
                lstm_dict[dn][bc]['predictions']['unscaled_thresholded'] = dict()
                lstm_dict[dn][bc]['predictions']['unscaled_thresholded']['true'] = np.load(os.path.join(bc_dir,'thresholded_predictions.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['predictions']['unscaled_thresholded']['control'] = np.load(os.path.join(bc_dir,'null_thresholded_predictions.npy'),allow_pickle=True).item()
                #Get the scaled predictions
                lstm_dict[dn][bc]['predictions']['scaled'] = dict()
                lstm_dict[dn][bc]['predictions']['scaled']['true'] = np.load(os.path.join(bc_dir,'scaled_predictions.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['predictions']['scaled']['control'] = np.load(os.path.join(bc_dir,'scaled_null_predictions.npy'),allow_pickle=True).item()
                #Get the scaled thresholded predictions
                lstm_dict[dn][bc]['predictions']['scaled_thresholded'] = dict()
                lstm_dict[dn][bc]['predictions']['scaled_thresholded']['true'] = np.load(os.path.join(bc_dir,'scaled_thresholded_predictions.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['predictions']['scaled_thresholded']['control'] = np.load(os.path.join(bc_dir,'scaled_null_thresholded_predictions.npy'),allow_pickle=True).item()
                #Get unscaled time shuffled predictions
                lstm_dict[dn][bc]['predictions']['unscaled_time_controlled'] = dict()
                lstm_dict[dn][bc]['predictions']['unscaled_time_controlled']['true'] = np.load(os.path.join(bc_dir,'predictions.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['predictions']['unscaled_time_controlled']['control'] = np.load(os.path.join(bc_dir,'shuffled_predictions.npy'),allow_pickle=True).item()
                #Get unscaled time shuffled thresholded predictions
                lstm_dict[dn][bc]['predictions']['unscaled_thresholded_time_controlled'] = dict()
                lstm_dict[dn][bc]['predictions']['unscaled_thresholded_time_controlled']['true'] = np.load(os.path.join(bc_dir,'thresholded_predictions.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['predictions']['unscaled_thresholded_time_controlled']['control'] = np.load(os.path.join(bc_dir,'shuffled_thresholded_predictions.npy'),allow_pickle=True).item()
                #Get scaled time shuffled predictions
                lstm_dict[dn][bc]['predictions']['scaled_time_controlled'] = dict()
                lstm_dict[dn][bc]['predictions']['scaled_time_controlled']['true'] = np.load(os.path.join(bc_dir,'scaled_predictions.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['predictions']['scaled_time_controlled']['control'] = np.load(os.path.join(bc_dir,'shuffled_scaled_predictions.npy'),allow_pickle=True).item()
                #Get scaled time shuffled thresholded predictions
                lstm_dict[dn][bc]['predictions']['scaled_thresholded_time_controlled'] = dict()
                lstm_dict[dn][bc]['predictions']['scaled_thresholded_time_controlled']['true'] = np.load(os.path.join(bc_dir,'scaled_thresholded_predictions.npy'),allow_pickle=True).item()
                lstm_dict[dn][bc]['predictions']['scaled_thresholded_time_controlled']['control'] = np.load(os.path.join(bc_dir,'shuffled_scaled_thresholded_predictions.npy'),allow_pickle=True).item()
            except: #Some other folder
                if verbose == True:
                    print("Not a binned decoding folder: " + os.path.join(lstm_dir,bc))
                
    np.save(lstm_dict_path,lstm_dict,allow_pickle=True)   
    
#%% find_lstm_groupings()

print("Finding LSTM decoding groupings")
num_datasets = len(lstm_dict)
unique_given_names = list(lstm_dict.keys())
#Pull unique decode analysis names
unique_bin_counts = []
unique_decode_pairs = []
unique_training_categories = []
for gn in unique_given_names:
    bc_names = list(lstm_dict[gn].keys())
    for bc in bc_names:
        bc_i = lstm_dict[gn][bc]['num_bins']
        unique_bin_counts.append(bc_i)
        unique_decode_pairs.extend(list(lstm_dict[gn][bc]['predictions'].keys()))
        unique_training_categories.extend(list(lstm_dict[gn][bc]['training_categories']))
unique_bin_count_inds = np.sort(
    np.unique(unique_bin_counts, return_index=True)[1])
unique_bin_counts = np.sort([unique_bin_counts[i] for i in unique_bin_count_inds])
unique_decode_pair_inds = np.sort(
    np.unique(unique_decode_pairs, return_index=True)[1])
unique_decode_pairs = [unique_decode_pairs[i] for i in unique_decode_pair_inds]
unique_training_cat_inds = np.sort(
    np.unique(unique_training_categories, return_index=True)[1])
unique_training_categories = [unique_training_categories[i] for i in unique_training_cat_inds]

#%% run_lstm_analysis

#Create plots of decoding count differences between true and control

#Create plots of decoding difference based on true data decoding - of those 
#decoded as saccharin, how many are still decoded as saccharin in control? 
#Include correlation between decode matrices.

#Group none and null as interchangeable categories

from scipy.stats import pearsonr
import matplotlib.pyplot as plt

num_anim = len(unique_given_names)
num_tastes = len(unique_training_categories)
segment_keep_inds = [0,2,4]
num_seg = len(segment_keep_inds)
segment_names = ['Pre-Taste', 'Post-Taste', 'Sickness']

lstm_save_dir = os.path.join(save_dir,'LSTM')
if not os.path.isdir(lstm_save_dir):
    os.mkdir(lstm_save_dir)

for b_i in unique_bin_counts:
    #Save dir
    bin_name = str(b_i) + '_bins'
    bin_save_dir = os.path.join(lstm_save_dir,bin_name)
    if not os.path.isdir(bin_save_dir):
        os.mkdir(bin_save_dir)
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
        #Plot results
        # True Fractions
        f_true_seg, ax_true_seg = plt.subplots(ncols = num_seg, figsize = (num_seg*5,5))
        for s_i in range(num_seg):
            seg_data = np.squeeze(anim_true_data[:,:,s_i])
            ax_true_seg[s_i].axhline(0,linestyle='dashed',color='k',alpha=0.5)
            for gn_i in range(num_anim):
                ax_true_seg[s_i].scatter(np.arange(num_tastes)+1,seg_data[gn_i,:],color='g',\
                            alpha=0.5)
            ax_true_seg[s_i].boxplot(seg_data)
            ax_true_seg[s_i].set_xticks(np.arange(num_tastes)+1,unique_training_categories)
            ax_true_seg[s_i].set_ylabel('Fraction True')
            ax_true_seg[s_i].set_title(segment_names[s_i])
        plt.suptitle('Decode Fraction by Segment')
        plt.tight_layout()
        f_true_seg.savefig(os.path.join(bin_save_dir,udp + '_frac_boxplots_by_seg.png'))
        f_true_seg.savefig(os.path.join(bin_save_dir,udp + '_frac_boxplots_by_seg.svg'))
        plt.close(f_true_seg)
        
        f_true_taste, ax_true_taste = plt.subplots(ncols = num_tastes, figsize = (num_seg*5,5))
        for t_i in range(num_tastes):
            taste_data = np.squeeze(anim_true_data[:,t_i,:])
            ax_true_taste[t_i].axhline(0,linestyle='dashed',color='k',alpha=0.5)
            for gn_i in range(num_anim):
                ax_true_taste[t_i].scatter(np.arange(num_seg)+1,taste_data[gn_i,:],color='g',\
                            alpha=0.5)
            ax_true_taste[t_i].boxplot(taste_data)
            ax_true_taste[t_i].set_xticks(np.arange(num_seg)+1,segment_names)
            ax_true_taste[t_i].set_ylabel('Fraction True')
            ax_true_taste[t_i].set_title(unique_training_categories[t_i])
        plt.suptitle('Decode Fraction by Taste')
        plt.tight_layout()
        f_true_taste.savefig(os.path.join(bin_save_dir,udp + '_frac_boxplots_by_taste.png'))
        f_true_taste.savefig(os.path.join(bin_save_dir,udp + '_frac_boxplots_by_taste.svg'))
        plt.close(f_true_taste)
        
        #    Difference
        f_diff_seg, ax_diff_seg = plt.subplots(ncols = num_seg, figsize = (num_seg*5,5))
        for s_i in range(num_seg):
            seg_data = np.squeeze(anim_diff_data[:,:,s_i])
            ax_diff_seg[s_i].axhline(0,linestyle='dashed',color='k',alpha=0.5)
            for gn_i in range(num_anim):
                ax_diff_seg[s_i].scatter(np.arange(num_tastes)+1,seg_data[gn_i,:],color='g',\
                            alpha=0.5)
            ax_diff_seg[s_i].boxplot(seg_data)
            ax_diff_seg[s_i].set_xticks(np.arange(num_tastes)+1,unique_training_categories)
            ax_diff_seg[s_i].set_ylabel('Fraction True')
            ax_diff_seg[s_i].set_title(segment_names[s_i])
        plt.suptitle('Decode Fraction by Segment')
        plt.tight_layout()
        f_diff_seg.savefig(os.path.join(bin_save_dir,udp + '_diff_boxplots_by_seg.png'))
        f_diff_seg.savefig(os.path.join(bin_save_dir,udp + '_diff_boxplots_by_seg.svg'))
        plt.close(f_diff_seg)
        
        f_diff_taste, ax_diff_taste = plt.subplots(ncols = num_tastes, figsize = (num_seg*5,5))
        for t_i in range(num_tastes):
            taste_data = np.squeeze(anim_diff_data[:,t_i,:])
            ax_diff_taste[t_i].axhline(0,linestyle='dashed',color='k',alpha=0.5)
            for gn_i in range(num_anim):
                ax_diff_taste[t_i].scatter(np.arange(num_seg)+1,taste_data[gn_i,:],color='g',\
                            alpha=0.5)
            ax_diff_taste[t_i].boxplot(taste_data)
            ax_diff_taste[t_i].set_xticks(np.arange(num_seg)+1,segment_names)
            ax_diff_taste[t_i].set_ylabel('Fraction True - Fraction Control')
            ax_diff_taste[t_i].set_title(unique_training_categories[t_i])
        plt.suptitle('Decode Fraction by Taste')
        plt.tight_layout()
        f_diff_taste.savefig(os.path.join(bin_save_dir,udp + '_diff_boxplots_by_taste.png'))
        f_diff_taste.savefig(os.path.join(bin_save_dir,udp + '_diff_boxplots_by_taste.svg'))
        plt.close(f_diff_taste)
        
        #    Overlap Fraction
        f_overlap_seg, ax_overlap_seg = plt.subplots(ncols = num_seg, figsize = (num_seg*5,5))
        for s_i in range(num_seg):
            seg_data = np.squeeze(anim_overlap_data[:,:,s_i])
            ax_overlap_seg[s_i].set_ylim([0,1])
            for gn_i in range(num_anim):
                ax_overlap_seg[s_i].scatter(np.arange(num_tastes)+1,seg_data[gn_i,:],color='g',\
                            alpha=0.5)
            ax_overlap_seg[s_i].boxplot(seg_data)
            ax_overlap_seg[s_i].set_xticks(np.arange(num_tastes)+1,unique_training_categories)
            ax_overlap_seg[s_i].set_ylabel('Num Overlap / Num True')
            ax_overlap_seg[s_i].set_title(segment_names[s_i])
        plt.suptitle('Overlap Fraction by Segment')
        plt.tight_layout()
        f_overlap_seg.savefig(os.path.join(bin_save_dir,udp + '_overlap_boxplots_by_seg.png'))
        f_overlap_seg.savefig(os.path.join(bin_save_dir,udp + '_overlap_boxplots_by_seg.svg'))
        plt.close(f_overlap_seg)
        
        f_overlap_taste, ax_overlap_taste = plt.subplots(ncols = num_tastes, figsize = (num_seg*5,5))
        for t_i in range(num_tastes):
            taste_data = np.squeeze(anim_overlap_data[:,t_i,:])
            for gn_i in range(num_anim):
                ax_overlap_taste[t_i].scatter(np.arange(num_seg)+1,taste_data[gn_i,:],color='g',\
                            alpha=0.5)
            ax_overlap_taste[t_i].boxplot(taste_data)
            ax_overlap_taste[t_i].set_xticks(np.arange(num_seg)+1,segment_names)
            ax_overlap_taste[t_i].set_ylabel('Num Overlap / Num True')
            ax_overlap_taste[t_i].set_title(unique_training_categories[t_i])
        plt.suptitle('Overlap Fraction by Taste')
        plt.tight_layout()
        f_overlap_taste.savefig(os.path.join(bin_save_dir,udp + '_overlap_boxplots_by_taste.png'))
        f_overlap_taste.savefig(os.path.join(bin_save_dir,udp + '_overlap_boxplots_by_taste.svg'))
        plt.close(f_overlap_taste)    
        
        #    Correlation
        f_corr_seg, ax_corr_seg = plt.subplots(ncols = num_seg, figsize = (num_seg*5,5))
        for s_i in range(num_seg):
            seg_data = np.squeeze(anim_corr_data[:,:,s_i])
            ax_corr_seg[s_i].axhline(0,linestyle='dashed',color='k',alpha=0.5)
            for gn_i in range(num_anim):
                ax_corr_seg[s_i].scatter(np.arange(num_tastes)+1,seg_data[gn_i,:],color='g',\
                            alpha=0.5)
            ax_corr_seg[s_i].boxplot(seg_data)
            ax_corr_seg[s_i].set_xticks(np.arange(num_tastes)+1,unique_training_categories)
            ax_corr_seg[s_i].set_ylabel('Pearson R Correlation')
            ax_corr_seg[s_i].set_title(segment_names[s_i])
        plt.suptitle('Correlation by Segment')
        plt.tight_layout()
        f_corr_seg.savefig(os.path.join(bin_save_dir,udp + '_corr_boxplots_by_seg.png'))
        f_corr_seg.savefig(os.path.join(bin_save_dir,udp + '_corr_boxplots_by_seg.svg'))
        plt.close(f_corr_seg)
        
        f_corr_taste, ax_corr_taste = plt.subplots(ncols = num_tastes, figsize = (num_seg*5,5))
        for t_i in range(num_tastes):
            taste_data = np.squeeze(anim_corr_data[:,t_i,:])
            ax_corr_taste[t_i].axhline(0,linestyle='dashed',color='k',alpha=0.5)
            for gn_i in range(num_anim):
                ax_corr_taste[t_i].scatter(np.arange(num_seg)+1,taste_data[gn_i,:],color='g',\
                            alpha=0.5)
            ax_corr_taste[t_i].boxplot(taste_data)
            ax_corr_taste[t_i].set_xticks(np.arange(num_seg)+1,segment_names)
            ax_corr_taste[t_i].set_ylabel('Pearson R Correlation')
            ax_corr_taste[t_i].set_title(unique_training_categories[t_i])
        plt.suptitle('Correlation by Taste')
        plt.tight_layout()
        f_corr_taste.savefig(os.path.join(bin_save_dir,udp + '_corr_boxplots_by_taste.png'))
        f_corr_taste.savefig(os.path.join(bin_save_dir,udp + '_corr_boxplots_by_taste.svg'))
        plt.close(f_corr_taste)    
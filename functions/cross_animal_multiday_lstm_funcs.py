#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 09:36:26 2025

@author: hannahgermaine
"""

def run_analysis_plots_by_decode_pair(unique_given_names,unique_training_categories,\
                                      unique_bin_counts, lstm_save_dir):
    
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
                           bin_save_dir)

def plot_diff_data(anim_true_data, anim_diff_data, anim_overlap_data, \
               anim_corr_data, segment_names, unique_training_categories, \
               bin_save_dir):
    
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
    
    
    


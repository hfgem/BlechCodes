#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:29:50 2024

@author: Hannah Germaine

A collection of functions dedicated to testing deviation decodes against null
distributions for significance in decoding / which tastes are being decoded.
"""

def dev_x_null_calc(num_neur,segment_dev_rasters,segment_zscore_means,segment_zscore_stds,
                   tastant_fr_dist_pop,tastant_fr_dist_z_pop,dig_in_names,segment_names,
                   num_null, save_dir, segments_to_analyze, epochs_to_analyze = []):
    """
    This function is dedicated to an analysis of what each deviation event is
    decoded as looking like (taste, which taste, and which epoch) and comparing
    the results to two types of null distributions for significance testing. The
    function outputs the distributions of these correlations into plots.
    """
    
    decode_dir = os.path.join(save_dir,'decode_devs')
    if not os.path.isdir(decode_dir):
        os.mkdir(decode_dir)
    non_z_decode_dir = os.path.join(decode_dir,'firing_rates')
    if not os.path.isdir(non_z_decode_dir):
        os.mkdir(non_z_decode_dir)
    z_decode_dir = os.path.join(decode_dir,'zscore_firing_rates')
    if not os.path.isdir(z_decode_dir):
        os.mkdir(z_decode_dir)
    null_decode_dir_2 = os.path.join(non_z_decode_dir,'null_decodes_across_neur')
    if not os.path.isdir(null_decode_dir_2):
        os.mkdir(null_decode_dir_2)
    null_z_decode_dir_2 = os.path.join(z_decode_dir,'null_decodes_across_neur')
    if not os.path.isdir(null_z_decode_dir_2):
        os.mkdir(null_z_decode_dir_2)
        
    # Variables
    num_tastes = len(dig_in_names)
    num_taste_deliv = [len(tastant_fr_dist_pop[t_i]) for t_i in range(num_tastes)]
    max_num_cp = 0
    for t_i in range(num_tastes):
        for d_i in range(num_taste_deliv[t_i]):
            if len(tastant_fr_dist_pop[t_i][d_i]) > max_num_cp:
                max_num_cp = len(tastant_fr_dist_pop[t_i][d_i])
    
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
        
        dev_vecs = []
        dev_vecs_z = []
        null_dev_dict = dict()
        null_dev_z_dict = dict()
        null_dev_dict_2 = dict()
        null_dev_z_dict_2 = dict()
        for null_i in range(num_null):
            null_dev_dict[null_i] = []
            null_dev_z_dict[null_i] = []
            null_dev_dict_2[null_i] = []
            null_dev_z_dict_2[null_i] = []
            
        for dev_i in range(num_dev):
            #Pull dev firing rate vectors
            dev_rast = seg_dev_rast[dev_i]
            num_spikes_per_neur = np.sum(dev_rast,1).astype('int')
            _, num_dt = np.shape(dev_rast)
            dev_vec = num_spikes_per_neur/(num_dt/1000)
            dev_vecs.append(dev_vec)
            dev_vec_z = (dev_vec - seg_z_mean)/seg_z_std
            dev_vecs_z.append(dev_vec_z)
            #Create null versions of the event
            for null_i in range(num_null):
                #Shuffle across-neuron spike times
                shuffle_rast = np.zeros(np.shape(dev_rast))
                for neur_i in range(num_neur):
                    new_spike_ind = random.sample(list(np.arange(num_dt)),num_spikes_per_neur[neur_i])
                    shuffle_rast[neur_i,new_spike_ind] = 1
                #Create fr vec
                null_dev_vec = np.sum(/(num_dt/1000)
                dev_vecs.append(dev_vec)
                #Create z-scored fr vecs
                first_half_fr_vec_z = (first_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
                second_half_fr_vec_z = (second_half_fr_vec - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
                shuffle_dev_mat_z = np.concatenate((first_half_fr_vec_z,second_half_fr_vec_z),1)
                null_dev_dict[null_i].append(shuffle_dev_mat)
                null_dev_z_dict[null_i].append(shuffle_dev_mat_z)
                #Shuffle across-neuron spike times
                shuffle_rast_2 = np.zeros(np.shape(dev_rast))
                new_neuron_order = random.sample(list(np.arange(num_neur)),num_neur)
                for nn_ind, nn in enumerate(new_neuron_order):
                    shuffle_rast_2[nn_ind,:] = shuffle_rast[nn,:]
                first_half_shuffle_rast_2 = shuffle_rast_2[:,:half_dt]
                second_half_shuffle_rast_2 = shuffle_rast_2[:,-half_dt:]
                #Create fr vecs
                first_half_fr_vec_2 = np.expand_dims(np.sum(first_half_shuffle_rast_2,1)/(half_dt/1000),1) #In Hz
                second_half_fr_vec_2 = np.expand_dims(np.sum(second_half_shuffle_rast_2,1)/(half_dt/1000),1) #In Hz
                shuffle_dev_mat_2 = np.concatenate((first_half_fr_vec_2,second_half_fr_vec_2),1)
                #Create z-scored fr vecs
                first_half_fr_vec_z_2 = (first_half_fr_vec_2 - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
                second_half_fr_vec_z_2 = (second_half_fr_vec_2 - np.expand_dims(seg_z_mean,1))/np.expand_dims(seg_z_std,1)
                shuffle_dev_mat_z_2 = np.concatenate((first_half_fr_vec_z_2,second_half_fr_vec_z_2),1)
                null_dev_dict_2[null_i].append(shuffle_dev_mat_2)
                null_dev_z_dict_2[null_i].append(shuffle_dev_mat_z_2)    


def decode_deviations_is_taste_which_taste_which_epoch(tastant_fr_dist, 
                dig_in_names, dev_mats_array, segment_names, s_i,
                decode_dir, epochs_to_analyze=[]):
    """Decode each deviation event"""
    print('\t\tRunning Is-Taste-Which-Taste GMM Decoder')
    
    
def decode_null_deviations_is_taste_which_taste_which_epoch(tastant_fr_dist, 
                dig_in_names, null_dev_dict, segment_names, s_i,
                null_decode_dir, true_decode_dir, epochs_to_analyze=[]):
    """Decode each null deviation event"""
    print('\t\tRunning Null Is-Taste-Which-Taste GMM Decoder')
    

    
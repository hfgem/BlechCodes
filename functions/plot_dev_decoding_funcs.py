#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:28:46 2024

@author: Hannah Germaine

This file is dedicated to plotting functions for deviation decoded replay events. 
"""

def plot_decoded(fr_dist, num_tastes, num_neur, segment_spike_times, tastant_spike_times,
                 start_dig_in_times, end_dig_in_times, post_taste_dt, pre_taste_dt,
                 cp_raster_inds, z_bin_dt, dig_in_names, segment_times,
                 segment_names, taste_num_deliv, taste_select_epoch,
                 save_dir, max_decode, max_hz, seg_stat_bin,
                 segment_dev_times, segment_dev_fr_vecs, segment_dev_fr_vecs_zscore,
                 neuron_count_thresh, e_len_dt, trial_start_frac=0,
                 epochs_to_analyze=[], segments_to_analyze=[],
                 decode_prob_cutoff=0.95):
    """Function to plot the deviation events with a buffer on either side with
    the decoding results"""
    
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    num_segments = len(segment_spike_times)
    neur_cut = np.floor(num_neur*neuron_count_thresh).astype('int')
    taste_colors = cm.brg(np.linspace(0, 1, num_tastes))
    epoch_seg_taste_times = np.zeros((num_cp, num_segments, num_tastes))
    epoch_seg_taste_times_neur_cut = np.zeros(
        (num_cp, num_segments, num_tastes))
    epoch_seg_taste_times_best = np.zeros((num_cp, num_segments, num_tastes))
    epoch_seg_lengths = np.zeros((num_cp, num_segments, num_tastes))
    half_bin_z_dt = np.floor(z_bin_dt/2).astype('int')
    half_bin_decode_dt = np.floor(e_len_dt/2).astype('int')
    dev_buffer = 50
    
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)

    # Get taste segment z-score info
    s_i_taste = np.nan*np.ones(1)
    for s_i in range(len(segment_names)):
        if segment_names[s_i].lower() == 'taste':
            s_i_taste[0] = s_i
    if not np.isnan(s_i_taste[0]):
        s_i = int(s_i_taste[0])
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = seg_end - seg_start
        time_bin_starts = np.arange(
            seg_start+half_bin_z_dt, seg_end-half_bin_z_dt, half_bin_z_dt*2)
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
            n_i_spike_times = np.array(
                segment_spike_times_s_i[n_i] - seg_start).astype('int')
            segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1
        tb_fr = np.zeros((num_neur, len(time_bin_starts)))
        for tb_i, tb in enumerate(tqdm.tqdm(time_bin_starts)):
            tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                    half_bin_z_dt:tb+half_bin_z_dt-seg_start], 1)/(2*half_bin_z_dt*(1/1000))
        mean_fr_taste = np.mean(tb_fr, 1)
        std_fr_taste = np.std(tb_fr, 1)
        std_fr_taste[std_fr_taste == 0] = 1  # to avoid nan calculations
    else:
        mean_fr_taste = np.zeros(num_neur)
        std_fr_taste = np.ones(num_neur)
        
    dev_decode_stats = dict()
        
    for e_i in epochs_to_analyze:
        print('\t\t\tPlotting Decoding for Epoch ' + str(e_i))

        taste_select_neur = np.where(taste_select_epoch[e_i, :] == 1)[0]

        epoch_decode_save_dir = os.path.join(
            save_dir, 'decode_prob_epoch_' + str(e_i))
        if not os.path.isdir(epoch_decode_save_dir):
            print("\t\t\t\tData not previously decoded, or passed directory incorrect.")
            pass
        print('\t\t\t\tProgressing through all ' +
              str(len(segments_to_analyze)) + ' segments.')
        
        dev_decode_stats[e_i] = dict()
        for seg_ind, s_i in tqdm.tqdm(enumerate(segments_to_analyze)):
            try:
                dev_decode_array = np.load(
                    os.path.join(epoch_decode_save_dir,'segment_' + str(s_i) + \
                                 '_deviations.npy'))
                pre_dev_decode_array = np.load(
                    os.path.join(epoch_decode_save_dir,'segment_' + str(s_i) + \
                                 '_pre_deviations.npy'))
                post_dev_decode_array = np.load(
                    os.path.join(epoch_decode_save_dir,'segment_' + str(s_i) + \
                                 '_post_deviations.npy'))
            except:
                print("\t\t\t\tSegment " + str(s_i) + " Never Decoded")
                pass
            
            dev_decode_stats[e_i][s_i] = dict()
            
            _, num_dev = np.shape(dev_decode_array)
            dev_decode_stats[e_i][s_i]['num_dev'] = num_dev
            
            seg_start = segment_times[s_i]
            seg_end = segment_times[s_i+1]
            seg_len = seg_end - seg_start  # in dt = ms

            # Pull binary spikes for segment
            segment_spike_times_s_i = segment_spike_times[s_i]
            segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
            for n_i in taste_select_neur:
                n_i_spike_times = np.array(
                    segment_spike_times_s_i[n_i] - seg_start).astype('int')
                segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1

            # Z-score the segment
            time_bin_starts = np.arange(
                seg_start+half_bin_z_dt, seg_end-half_bin_z_dt, half_bin_z_dt*2)
            tb_fr = np.zeros((num_neur, len(time_bin_starts)))
            for tb_i, tb in enumerate(time_bin_starts):
                tb_fr[:, tb_i] = np.sum(segment_spike_times_s_i_bin[:, tb-seg_start -
                                        half_bin_z_dt:tb+half_bin_z_dt-seg_start], 1)/(2*half_bin_z_dt*(1/1000))
            mean_fr = np.mean(tb_fr, 1)
            std_fr = np.std(tb_fr, 1)
            std_fr[std_fr == 0] = 1
            
            #Calculate the number of deviation events decoded for each taste
            taste_decoded = np.argmax(dev_decode_array,0)
            taste_decoded_bin = np.zeros((num_tastes,num_dev))
            taste_decoded_start_times = []
            taste_decoded_end_times = []
            for t_i in range(num_tastes):
                taste_decoded_ind = np.where(taste_decoded == t_i)[0]
                taste_decoded_bin[t_i,taste_decoded_ind] = 1
                taste_decoded_start_times.append(list(segment_dev_times[seg_ind][0,taste_decoded_ind]))
                taste_decoded_end_times.append(list(segment_dev_times[seg_ind][1,taste_decoded_ind]))
            dev_decode_stats[e_i][s_i]['num_dev_per_taste'] = np.nansum(taste_decoded_bin,1)
            
            
            #For each deviation event create plots of decoded results
            for dev_i in range(num_dev):
                #Collect deviation decoding probabilities
                dev_prob_decoded = dev_decode_array[:,dev_i]
                pre_dev_prob_decoded = pre_dev_decode_array[:,dev_i]
                post_dev_prob_decoded = post_dev_decode_array[:,dev_i]
                decoded_taste_ind = np.argmax(dev_prob_decoded)
                
                #Collect deviation times
                dev_times = segment_dev_times[seg_ind][:,dev_i]
                dev_start_time = dev_times[0]
                dev_end_time = dev_times[1]
                dev_len = dev_end_time - dev_start_time
                pre_dev_time = np.max(dev_start_time - dev_buffer,0)
                post_dev_time = np.min(dev_end_time + dev_buffer,seg_len)
                
                #Collect deviation FR vals
                dev_fr_vec = segment_dev_fr_vecs[seg_ind][dev_i]
                dev_fr_vec_zscore = segment_dev_fr_vecs_zscore[seg_ind][dev_i]
                
                #Reshape decoding probabilities for plotting
                decoding_array = np.zeros((num_tastes,post_dev_time - pre_dev_time))
                decoding_array[:,0:dev_buffer] = np.expand_dims(pre_dev_prob_decoded,1)*np.ones((num_neur,dev_buffer))
                decoding_array[:,dev_buffer:dev_buffer + dev_len] = np.expand_dims(dev_prob_decoded,1)*np.ones((num_neur,dev_len))
                decoding_array[:,-dev_buffer:] = np.expand_dims(post_dev_prob_decoded,1)*np.ones((num_neur,dev_buffer))
                
                #Save to decoding counters
                epoch_seg_taste_times[e_i, s_i, decoded_taste_ind] += 1
                epoch_seg_lengths[e_i, s_i, decoded_taste_ind] += dev_len
                
                #Calculate correlation to mean taste responses
                corr_dev_event = np.array([pearsonr(all_taste_fr_vecs_mean[t_i, :], dev_fr_vec)[
                                             0] for t_i in range(num_tastes)])
                corr_dev_event_zscore = np.array([pearsonr(all_taste_fr_vecs_mean_z[t_i, :], dev_fr_vec_zscore)[
                                             0] for t_i in range(num_tastes)])
                
                

    
    
    
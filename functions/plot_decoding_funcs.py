#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:25:40 2024

@author: Hannah Germaine

This file is dedicated to plotting functions for sliding bin decoded replay events.
"""

def plot_decoded(fr_dist, num_tastes, num_neur, segment_spike_times, tastant_spike_times,
                 start_dig_in_times, end_dig_in_times, post_taste_dt, pre_taste_dt,
                 cp_raster_inds, z_bin_dt, dig_in_names, segment_times,
                 segment_names, taste_num_deliv, taste_select_epoch,
                 save_dir, max_decode, max_hz, seg_stat_bin,
                 neuron_count_thresh, e_len_dt, trial_start_frac=0,
                 epochs_to_analyze=[], segments_to_analyze=[],
                 decode_prob_cutoff=0.95):
    """Function to plot the periods when something other than no taste is 
    decoded"""
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

    # for e_i in range(num_cp): #By epoch conduct decoding
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
        for s_i in tqdm.tqdm(segments_to_analyze):
            try:
                seg_decode_epoch_prob = np.load(os.path.join(
                    epoch_decode_save_dir, 'segment_' + str(s_i) + '.npy'))
            except:
                print("\t\t\t\tSegment " + str(s_i) + " Never Decoded")
                pass

            seg_decode_save_dir = os.path.join(
                epoch_decode_save_dir, 'segment_' + str(s_i))
            if not os.path.isdir(seg_decode_save_dir):
                os.mkdir(seg_decode_save_dir)

            seg_start = segment_times[s_i]
            seg_end = segment_times[s_i+1]
            seg_len = seg_end - seg_start  # in dt = ms

            # Import raster plots for segment
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

            # Calculate maximally decoded taste
            decoded_taste_max = np.argmax(seg_decode_epoch_prob, 0)
            # Store binary decoding results
            decoded_taste_bin = np.zeros((num_tastes, len(decoded_taste_max)))
            for t_i in range(num_tastes):
                times_decoded_taste_max = np.where(decoded_taste_max == t_i)[0]
                #Now spread these times based on the full decoding bin
                for diff_i in np.arange(-1*half_bin_decode_dt,half_bin_decode_dt):
                    times_decoded_shifted = times_decoded_taste_max + diff_i
                    times_decoded_shifted = times_decoded_shifted[np.where((times_decoded_shifted>0)*(times_decoded_shifted<len(decoded_taste_max)))[0]]
                    decoded_taste_bin[t_i, times_decoded_shifted] = 1
            # To ensure starts and ends of bins align
            decoded_taste_bin[:, 0] = 0
            decoded_taste_bin[:, -1] = 0
            
            #Test for periods that are overlapping and remove from decoded_taste_bin
            summed_decode = np.sum(decoded_taste_bin,0)
            overlap_bin = (summed_decode > 1).astype('int')
            overlap_diff = np.diff(overlap_bin)
            overlap_starts = np.where(overlap_diff == 1)[0]+1
            overlap_ends = np.where(overlap_diff == -1)[0]+1
            for o_i in range(len(overlap_starts)):
                decoded_taste_bin[:,overlap_starts[o_i]:overlap_ends[o_i]] = 0
                decoded_taste_bin[-1,overlap_starts[o_i]:overlap_ends[o_i]] = 1

            # For each taste (except none) calculate start and end times of decoded intervals and plot
            all_taste_fr_vecs = []
            all_taste_fr_vecs_z = []
            all_taste_fr_vecs_mean = np.zeros((num_tastes, num_neur))
            all_taste_fr_vecs_mean_z = np.zeros((num_tastes, num_neur))
            # Grab taste firing rate vectors
            for t_i in range(num_tastes):
                # Import taste spike and cp times
                taste_spike_times = tastant_spike_times[t_i]
                taste_deliv_times = start_dig_in_times[t_i]
                max_num_deliv = len(taste_deliv_times)
                cp_times = cp_raster_inds[t_i]

                # If trial_start_frac > 0 use only trials after that threshold
                trial_start_ind = np.floor(
                    max_num_deliv*trial_start_frac).astype('int')
                new_max_num_deliv = (
                    max_num_deliv - trial_start_ind).astype('int')

                # Store as binary spike arrays
                taste_spike_times_bin = np.zeros(
                    (new_max_num_deliv, num_neur, post_taste_dt))  # Taste response spike times
                taste_cp_times = np.zeros(
                    (new_max_num_deliv, num_cp+1)).astype('int')
                taste_epoch_fr_vecs = np.zeros(
                    (new_max_num_deliv, num_neur))  # original firing rate vecs
                taste_epoch_fr_vecs_z = np.zeros(
                    (new_max_num_deliv, num_neur))  # z-scored firing rate vecs
                # store each delivery to binary spike matrix
                for d_i in range(len(taste_spike_times)):
                    if d_i >= trial_start_ind:
                        pre_taste_spike_times_bin = np.zeros(
                            (num_neur, pre_taste_dt))  # Pre-taste spike times
                        taste_deliv_i = taste_deliv_times[d_i]
                        for n_i in range(num_neur):
                            spikes_deliv_i = taste_spike_times[d_i][n_i]
                            if t_i == num_tastes-1:
                                if len(taste_spike_times[d_i-trial_start_ind][n_i]) > 0:
                                    d_i_spikes = np.array(
                                        spikes_deliv_i - (np.min(spikes_deliv_i)+pre_taste_dt)).astype('int')
                                else:
                                    d_i_spikes = np.empty(0)
                            else:
                                d_i_spikes = np.array(
                                    spikes_deliv_i - taste_deliv_i).astype('int')
                            d_i_spikes_posttaste = d_i_spikes[(
                                d_i_spikes < post_taste_dt)*(d_i_spikes >= 0)]
                            d_i_spikes_pretaste = d_i_spikes[d_i_spikes <
                                                             0] + pre_taste_dt
                            if len(d_i_spikes_posttaste) > 0:
                                taste_spike_times_bin[d_i-trial_start_ind,
                                                      n_i, d_i_spikes_posttaste] = 1
                            if len(d_i_spikes_pretaste) > 0:
                                pre_taste_spike_times_bin[n_i,
                                                          d_i_spikes_pretaste] = 1
                        taste_cp_times[d_i-trial_start_ind, :] = np.concatenate(
                            (np.zeros(1), np.cumsum(np.diff(cp_times[d_i, :])))).astype('int')
                        # Calculate the FR vectors by epoch for each taste response and the average FR vector
                        epoch_len_i = (
                            taste_cp_times[d_i, e_i+1]-taste_cp_times[d_i, e_i])/1000
                        if epoch_len_i == 0:
                            taste_epoch_fr_vecs[d_i-trial_start_ind,
                                                :] = np.zeros(num_neur)
                        else:
                            taste_epoch_fr_vecs[d_i-trial_start_ind, :] = np.sum(
                                taste_spike_times_bin[d_i-trial_start_ind, :, taste_cp_times[d_i, e_i]:taste_cp_times[d_i, e_i+1]], 1)/epoch_len_i  # FR in HZ
                        # Calculate z-scored FR vector
                        taste_epoch_fr_vecs_z[d_i-trial_start_ind, :] = (
                            taste_epoch_fr_vecs[d_i-trial_start_ind, :].flatten() - mean_fr_taste)/std_fr_taste

                all_taste_fr_vecs.append(taste_epoch_fr_vecs)
                all_taste_fr_vecs_z.append(taste_epoch_fr_vecs_z)
                # Calculate average taste fr vec
                taste_fr_vecs_mean = np.nanmean(taste_epoch_fr_vecs, 0)
                taste_fr_vecs_z_mean = np.nanmean(taste_epoch_fr_vecs_z, 0)
                all_taste_fr_vecs_mean[t_i, :] = taste_fr_vecs_mean
                all_taste_fr_vecs_mean_z[t_i, :] = taste_fr_vecs_z_mean
                #taste_fr_vecs_max_hz = np.max(taste_epoch_fr_vecs)

            # Now look at decoded events
            all_taste_event_fr_vecs = []
            all_taste_event_fr_vecs_z = []
            all_taste_event_fr_vecs_neur_cut = []
            all_taste_event_fr_vecs_z_neur_cut = []
            all_taste_event_fr_vecs_best = []
            all_taste_event_fr_vecs_z_best = []
            for t_i in range(num_tastes):
                taste_decode_save_dir = os.path.join(
                    seg_decode_save_dir, dig_in_names[t_i] + '_events')
                if not os.path.isdir(taste_decode_save_dir):
                    os.mkdir(taste_decode_save_dir)

                # Calculate where the taste is decoded and how many
                decoded_taste = decoded_taste_bin[t_i, :]
                decoded_taste[0] = 0
                decoded_taste[-1] = 0
                decoded_taste_prob = seg_decode_epoch_prob[t_i, :]
                decoded_taste[decoded_taste_prob < decode_prob_cutoff] = 0
                diff_decoded_taste = np.diff(decoded_taste)
                start_decoded = np.where(diff_decoded_taste == 1)[0] + 1 - half_bin_decode_dt
                end_decoded = np.where(diff_decoded_taste == -1)[0] + 1 + half_bin_decode_dt
                num_decoded = len(start_decoded)
                num_neur_decoded = np.zeros(num_decoded)
                prob_decoded = np.zeros(num_decoded)
                for nd_i in range(num_decoded):
                    d_start = start_decoded[nd_i]
                    if d_start < 0:
                        d_start = 0
                    d_end = end_decoded[nd_i]
                    if d_end > seg_len:
                        d_end = seg_len
                    d_len = d_end-d_start
                    if d_len > 0:
                        for n_i in range(num_neur):
                            if len(np.where(segment_spike_times_s_i_bin[n_i, d_start:d_end])[0]) > 0:
                                num_neur_decoded[nd_i] += 1
                        prob_decoded[nd_i] = np.mean(
                            seg_decode_epoch_prob[t_i, d_start:d_end])

                # Save the percent taste decoded matching threshold
                epoch_seg_taste_times[e_i, s_i, t_i] = np.sum(decoded_taste)
                epoch_seg_lengths[e_i, s_i, t_i] = len(decoded_taste)

                # Determine where the decoded data exceeds the neuron threshold
                decode_ind_neur = np.where(num_neur_decoded >= neur_cut)[0]
                decoded_bin_neur = np.zeros(np.shape(decoded_taste))
                for db in decode_ind_neur:
                    s_db = start_decoded[db]
                    e_db = end_decoded[db]
                    decoded_bin_neur[s_db:e_db] = 1
                # Grab overall percentages
                epoch_seg_taste_times_neur_cut[e_i, s_i, t_i] = np.sum(
                    decoded_bin_neur)

                # Calculate correlation data for each decode
                decoded_corr = np.zeros((num_decoded, num_tastes))
                decoded_z_corr = np.zeros((num_decoded, num_tastes))
                decoded_fr_vecs = []  # Store all decoded events firing rates
                decoded_z_fr_vecs = []  # Store all decoded events z-scored firing rates
                for nd_i in range(num_decoded):
                    # Grab decoded data
                    d_start = start_decoded[nd_i]
                    d_end = end_decoded[nd_i]
                    d_len = d_end-d_start
                    d_fr_vec = np.sum(
                        segment_spike_times_s_i_bin[:, d_start:d_end], 1)/(d_len/1000)
                    decoded_fr_vecs.append(d_fr_vec)
                    # Grab z-scored decoded data
                    d_fr_vec_z = (d_fr_vec-mean_fr)/std_fr
                    decoded_z_fr_vecs.append(d_fr_vec_z)
                    # Grab correlation data
                    corr_decode_event = np.array([pearsonr(all_taste_fr_vecs_mean[t_i, :], d_fr_vec)[
                                                 0] for t_i in range(num_tastes)])
                    decoded_corr[nd_i, :] = corr_decode_event
                    corr_decode_event_z = np.array([pearsonr(
                        all_taste_fr_vecs_mean_z[t_i, :], d_fr_vec_z)[0] for t_i in range(num_tastes)])
                    decoded_z_corr[nd_i] = corr_decode_event_z
                # Find where the correlation data is highest for the given taste
                decoded_corr_match = (
                    np.argmax(decoded_corr, 1) == t_i).astype('int')
                decoded_z_corr_match = (
                    np.argmax(decoded_z_corr, 1) == t_i).astype('int')
                decode_prob_avg = np.array([np.mean(
                    seg_decode_epoch_prob[t_i, start_decoded[i]:end_decoded[i]]) for i in range(len(start_decoded))])
                # Find where the decoding is higher than a cutoff
                decode_above_cutoff = (
                    decode_prob_avg >= decode_prob_cutoff).astype('int')
                decode_above_neur_cutoff = (
                    num_neur_decoded >= neur_cut).astype('int')
                best_across_metrics = np.where(
                    decode_above_cutoff*decoded_corr_match*decoded_z_corr_match*decode_above_neur_cutoff)[0]

                # Now only keep matching decoded intervals
                decoded_bin_best = np.zeros(np.shape(decoded_taste))
                for db in best_across_metrics:
                    s_db = start_decoded[db]
                    e_db = end_decoded[db]
                    decoded_bin_best[s_db:e_db] = 1
                # Grab overall percentages
                epoch_seg_taste_times_best[e_i, s_i,
                                           t_i] = np.sum(decoded_bin_best)

                # Store all the firing rate vectors for plotting
                all_taste_event_fr_vecs.append(np.array(decoded_fr_vecs))
                all_taste_event_fr_vecs_z.append(np.array(decoded_z_fr_vecs))
                all_taste_event_fr_vecs_neur_cut.append(
                    np.array(decoded_fr_vecs)[decode_ind_neur])
                all_taste_event_fr_vecs_z_neur_cut.append(
                    np.array(decoded_z_fr_vecs)[decode_ind_neur])
                all_taste_event_fr_vecs_best.append(
                    np.array(decoded_fr_vecs)[best_across_metrics])
                all_taste_event_fr_vecs_z_best.append(
                    np.array(decoded_z_fr_vecs)[best_across_metrics])

                # First calculate neurons decoded in all decoded intervals
                save_name = 'all_events'
                if not os.path.isfile(taste_decode_save_dir + 'epoch_' + str(e_i) + '_seg_' + str(s_i) + '_' + save_name + '.svg'):

                    # ____Create plots of decoded period statistics____
                    seg_dist_starts = np.arange(0, seg_len, seg_stat_bin)
                    seg_dist_midbin = seg_dist_starts[:-
                                                      1] + np.diff(seg_dist_starts)/2

                    # ________All Decoded________
                    num_decoded = len(start_decoded)
                    prob_decoded = prob_decoded
                    len_decoded = np.array(end_decoded-start_decoded)
                    iei_decoded = np.array(
                        start_decoded[1:] - end_decoded[:-1])

                    seg_distribution = np.zeros(len(seg_dist_starts)-1)
                    prob_distribution = np.zeros(len(seg_dist_starts)-1)
                    for sd_i in range(len(seg_dist_starts)-1):
                        bin_events = np.where(
                            (start_decoded < seg_dist_starts[sd_i+1])*(start_decoded >= seg_dist_starts[sd_i]))[0]
                        seg_distribution[sd_i] = len(bin_events)
                        prob_distribution[sd_i] = np.mean(
                            prob_decoded[bin_events])

                    plot_overall_decoded_stats(len_decoded, iei_decoded, num_neur_decoded,
                                               prob_decoded, prob_distribution, e_i, s_i,
                                               seg_dist_midbin, seg_distribution, seg_stat_bin,
                                               seg_len, save_name, taste_decode_save_dir)

                save_name = 'neur_cutoff_events'
                if not os.path.isfile(taste_decode_save_dir + 'epoch_' + str(e_i) + '_seg_' + str(s_i) + '_' + save_name + '.svg'):
                    # Re-calculate start and end times of the decoded intervals
                    start_decoded_neur_cut = start_decoded[decode_ind_neur]
                    end_decoded_neur_cut = end_decoded[decode_ind_neur]
                    # Re-calculate the decoded statistics
                    num_neur_decoded_neur_cut = num_neur_decoded[decode_ind_neur]
                    prob_decoded_neur_cut = prob_decoded[decode_ind_neur]
                    len_decoded_neur_cut = np.array(
                        end_decoded_neur_cut-start_decoded_neur_cut)
                    iei_decoded_neur_cut = np.array(
                        start_decoded_neur_cut[1:] - end_decoded_neur_cut[:-1])

                    seg_distribution_neur_cut = np.zeros(
                        len(seg_dist_starts)-1)
                    prob_distribution_neur_cut = np.zeros(
                        len(seg_dist_starts)-1)
                    for sd_i in range(len(seg_dist_starts)-1):
                        bin_events = np.where((start_decoded_neur_cut < seg_dist_starts[sd_i+1])*(
                            start_decoded_neur_cut >= seg_dist_starts[sd_i]))[0]
                        seg_distribution_neur_cut[sd_i] = len(bin_events)
                        prob_distribution_neur_cut[sd_i] = np.mean(
                            prob_decoded_neur_cut[bin_events])

                    # Plot the statistics for those events meeting the minimum neuron cutoff
                    plot_overall_decoded_stats(len_decoded_neur_cut, iei_decoded_neur_cut, num_neur_decoded_neur_cut,
                                               prob_decoded_neur_cut, prob_distribution_neur_cut, e_i, s_i,
                                               seg_dist_midbin, seg_distribution_neur_cut, seg_stat_bin,
                                               seg_len, save_name, taste_decode_save_dir)

                # ________Best Decoded________
                save_name = 'best_events'
                if not os.path.isfile(taste_decode_save_dir + 'epoch_' + str(e_i) + '_seg_' + str(s_i) + '_' + save_name + '.svg'):
                    # Re-calculate start and end times of the decoded intervals
                    start_decoded_best = start_decoded[best_across_metrics]
                    end_decoded_best = end_decoded[best_across_metrics]
                    # Re-calculate the decoded statistics
                    num_neur_decoded_best = num_neur_decoded[best_across_metrics]
                    prob_decoded_best = prob_decoded[best_across_metrics]
                    len_decoded_best = np.array(
                        end_decoded_best - start_decoded_best)
                    iei_decoded_best = np.array(
                        start_decoded_best[1:] - end_decoded_best[:-1])

                    seg_distribution_best = np.zeros(len(seg_dist_starts)-1)
                    prob_distribution_best = np.zeros(len(seg_dist_starts)-1)
                    for sd_i in range(len(seg_dist_starts)-1):
                        bin_events = np.where((start_decoded_best < seg_dist_starts[sd_i+1])*(
                            start_decoded_best >= seg_dist_starts[sd_i]))[0]
                        seg_distribution_best[sd_i] = len(bin_events)
                        prob_distribution_best[sd_i] = np.mean(
                            prob_decoded_best[bin_events])

                    # Plot the statistics for those decoded events that are best across metrics
                    plot_overall_decoded_stats(len_decoded_best, iei_decoded_best, num_neur_decoded_best,
                                               prob_decoded_best, prob_distribution_best, e_i, s_i,
                                               seg_dist_midbin, seg_distribution_best, seg_stat_bin,
                                               seg_len, save_name, taste_decode_save_dir)

                if num_decoded > max_decode:  # Reduce number if too many
                    # TODO: add flag to select which cutoff to use for plotting?
                    # Reduce to top decoding probability
                    #decode_plot_ind = sample(list(np.where(decode_above_cutoff)[0]),max_decode)

                    # Reduce to ones with both top decoding probability and highest correlation of both regular and z-scored
                    decode_plot_ind = sample(list(best_across_metrics), min(
                        max_decode, len(best_across_metrics)))
                else:
                    decode_plot_ind = np.arange(num_decoded)

                decode_plot_ind = np.array(decode_plot_ind)
                # Create plots of the decoded periods
                if len(decode_plot_ind) > 0:
                    for nd_i in decode_plot_ind:
                        if not os.path.isfile(os.path.join(taste_decode_save_dir, 'event_' + str(nd_i) + '.png')):
                            # Grab decode variables
                            d_start = start_decoded[nd_i]
                            d_end = end_decoded[nd_i]
                            d_len = d_end-d_start
                            d_plot_start = np.max(
                                (start_decoded[nd_i]-2*d_len, 0))
                            d_plot_end = np.min(
                                (end_decoded[nd_i]+2*d_len, seg_len))
                            d_plot_len = d_plot_end-d_plot_start
                            d_plot_x_vals = (np.linspace(
                                d_plot_start-d_start, d_plot_end-d_start, 10)).astype('int')
                            decode_plot_times = np.arange(
                                d_plot_start, d_plot_end)
                            event_spikes = segment_spike_times_s_i_bin[:,
                                                                       d_plot_start:d_plot_end]
                            decode_spike_times = []
                            for n_i in range(num_neur):
                                decode_spike_times.append(
                                    list(np.where(event_spikes[n_i, :])[0]))
                            event_spikes_expand = segment_spike_times_s_i_bin[:,
                                                                              d_plot_start-10:d_plot_end+10]
                            event_spikes_expand_count = np.sum(
                                event_spikes_expand, 0)
                            firing_rate_vec = np.zeros(d_plot_len)
                            for dpt_i in np.arange(10, d_plot_len+10):
                                firing_rate_vec[dpt_i-10] = np.sum(
                                    event_spikes_expand_count[dpt_i-10:dpt_i+10])/(20/1000)/num_neur
                            d_fr_vec = decoded_fr_vecs[nd_i]
                            # Grab z-scored data
                            d_fr_vec_z = decoded_z_fr_vecs[nd_i]
                            # Find max hz
                            #d_fr_vec_max_hz = np.max(d_fr_vec)
                            # Correlation of vector to avg taste vector
                            corr_decode_event = decoded_corr[nd_i]
                            corr_title_norm = [dig_in_names[t_i] + ' corr = ' + str(
                                np.round(corr_decode_event[t_i], 2)) for t_i in range(num_tastes)]
                            # Correlation of z-scored vector to z-scored avg taste vector
                            corr_decode_event_z = decoded_z_corr[nd_i]
                            corr_title_z = [dig_in_names[t_i] + ' z-corr = ' + str(
                                np.round(corr_decode_event_z[t_i], 2)) for t_i in range(num_tastes)]
                            corr_title = (', ').join(
                                corr_title_norm) + '\n' + (', ').join(corr_title_z)
                            # Start Figure
                            f, ax = plt.subplots(nrows=5, ncols=2, figsize=(
                                10, 10), gridspec_kw=dict(height_ratios=[1, 1, 1, 2, 2]))
                            gs = ax[0, 0].get_gridspec()
                            # Decoding probabilities
                            ax[0, 0].remove()
                            ax[1, 0].remove()
                            axbig = f.add_subplot(gs[0:2, 0])
                            decode_x_vals = decode_plot_times-d_start
                            leg_handles = []
                            for t_i_2 in range(num_tastes):
                                taste_decode_prob_y = seg_decode_epoch_prob[t_i_2,
                                                                            d_plot_start:d_plot_end]
                                p_h, = axbig.plot(
                                    decode_x_vals, taste_decode_prob_y, color=taste_colors[t_i_2, :])
                                leg_handles.append(p_h)
                                taste_decode_prob_y[0] = 0
                                taste_decode_prob_y[-1] = 0
                                high_prob_binary = np.zeros(len(decode_x_vals))
                                high_prob_times = np.where(
                                    taste_decode_prob_y >= decode_prob_cutoff)[0]
                                high_prob_binary[high_prob_times] = 1
                                high_prob_starts = np.where(
                                    np.diff(high_prob_binary) == 1)[0] + 1
                                high_prob_ends = np.where(
                                    np.diff(high_prob_binary) == -1)[0] + 1
                                if len(high_prob_starts) > 0:
                                    for hp_i in range(len(high_prob_starts)):
                                        axbig.fill_between(decode_x_vals[high_prob_starts[hp_i]:high_prob_ends[hp_i]], taste_decode_prob_y[
                                                           high_prob_starts[hp_i]:high_prob_ends[hp_i]], alpha=0.2, color=taste_colors[t_i_2, :])
                            axbig.axvline(0, color='k', alpha=0.5)
                            axbig.axvline(d_len, color='k', alpha=0.5)
                            axbig.legend(
                                leg_handles, dig_in_names, loc='right')
                            axbig.set_xticks(d_plot_x_vals)
                            axbig.set_xlim(
                                [decode_x_vals[0], decode_x_vals[-1]])
                            axbig.set_ylabel('Decoding Fraction')
                            axbig.set_xlabel('Time from Event Start (ms)')
                            axbig.set_title('Event ' + str(nd_i) + '\nStart Time = ' + str(round(
                                d_start/1000/60, 3)) + ' Minutes' + '\nEvent Length = ' + str(np.round(d_len, 2)))
                            # Decoded raster
                            ax[0, 1].eventplot(decode_spike_times)
                            ax[0, 1].set_xlim([0, d_plot_len])
                            x_ticks = np.linspace(
                                0, d_plot_len, 10).astype('int')
                            ax[0, 1].set_xticks(x_ticks, labels=d_plot_x_vals)
                            ax[0, 1].axvline(
                                d_start-d_plot_start, color='k', alpha=0.5)
                            ax[0, 1].axvline(
                                d_end-d_plot_start, color='k', alpha=0.5)
                            ax[0, 1].set_ylabel('Neuron Index')
                            ax[0, 1].set_title('Event Spike Raster')
                            ax[1, 0].axis('off')
                            # Plot population firing rates w 20ms smoothing
                            ax[1, 1].plot(decode_x_vals, firing_rate_vec)
                            ax[1, 1].axvline(0, color='k')
                            ax[1, 1].axvline(d_len, color='k')
                            ax[1, 1].set_xlim(
                                [decode_x_vals[0], decode_x_vals[-1]])
                            ax[1, 1].set_xticks(d_plot_x_vals)
                            ax[1, 1].set_title('Population Avg FR')
                            ax[1, 1].set_ylabel('FR (Hz)')
                            ax[1, 1].set_xlabel('Time from Event Start (ms)')
                            # Decoded Firing Rates
                            # vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
                            img = ax[2, 0].imshow(np.expand_dims(
                                d_fr_vec, 0), vmin=0, vmax=60)
                            ax[2, 0].set_xlabel('Neuron Index')
                            ax[2, 0].set_yticks(ticks=[])
                            #plt.colorbar(img, location='bottom',orientation='horizontal',label='Firing Rate (Hz)',panchor=(0.9,0.5),ax=ax[2,0])
                            ax[2, 0].set_title('Event FR')
                            # Decoded Firing Rates Z-Scored
                            img = ax[2, 1].imshow(np.expand_dims(
                                d_fr_vec_z, 0), vmin=-3, vmax=3, cmap='bwr')
                            ax[2, 1].set_xlabel('Neuron Index')
                            ax[2, 1].set_yticks(ticks=[])
                            #plt.colorbar(img, ax=ax[2,1], location='bottom',orientation='horizontal',label='Z-Scored Firing Rate (Hz)',panchor=(0.9,0.5))
                            ax[2, 1].set_title('Event FR Z-Scored')
                            # Taste Firing Rates
                            # vmax=np.max([taste_fr_vecs_max_hz,d_fr_vec_max_hz]))
                            img = ax[3, 0].imshow(np.expand_dims(
                                taste_fr_vecs_mean, 0), vmin=0, vmax=60)
                            ax[3, 0].set_xlabel('Neuron Index')
                            ax[3, 0].set_yticks(ticks=[])
                            plt.colorbar(
                                img, ax=ax[3, 0], location='bottom', orientation='horizontal', label='Firing Rate (Hz)', panchor=(0.9, 0.5))
                            ax[3, 0].set_title('Avg. Taste Resp. FR')
                            # Taste Firing Rates Z-Scored
                            img = ax[3, 1].imshow(np.expand_dims(
                                all_taste_fr_vecs_mean_z[t_i, :], 0), vmin=-3, vmax=3, cmap='bwr')
                            ax[3, 1].set_xlabel('Neuron Index')
                            ax[3, 1].set_yticks(ticks=[])
                            plt.colorbar(img, ax=ax[3, 1], location='bottom', orientation='horizontal',
                                         label='Z-Scored Firing Rate (Hz)', panchor=(0.9, 0.5))
                            ax[3, 1].set_title('Avg. Taste Resp. FR Z-Scored')
                            # Decoded Firing Rates x Average Firing Rates
                            max_lim = np.max(
                                [np.max(d_fr_vec_z), np.max(taste_fr_vecs_mean)])
                            ax[4, 0].plot([0, max_lim], [0, max_lim],
                                          alpha=0.5, linestyle='dashed')
                            ax[4, 0].scatter(taste_fr_vecs_mean, d_fr_vec)
                            ax[4, 0].set_xlabel('Average Taste FR')
                            ax[4, 0].set_ylabel('Decoded Taste FR')
                            ax[4, 0].set_title('Firing Rate Similarity')
                            # Z-Scored Decoded Firing Rates x Z-Scored Average Firing Rates
                            ax[4, 1].plot([-3, 3], [-3, 3], alpha=0.5,
                                          linestyle='dashed', color='k')
                            ax[4, 1].scatter(
                                all_taste_fr_vecs_mean_z[t_i, :], d_fr_vec_z)
                            ax[4, 1].set_xlabel(
                                'Average Taste Neuron FR Std > Mean')
                            ax[4, 1].set_ylabel('Event Neuron FR Std > Mean')
                            ax[4, 1].set_title(
                                'Z-Scored Firing Rate Similarity')
                            plt.suptitle(corr_title, wrap=True)
                            plt.tight_layout()
                            # Save Figure
                            f.savefig(os.path.join(
                                taste_decode_save_dir, 'event_' + str(nd_i) + '.png'))
                            f.savefig(os.path.join(
                                taste_decode_save_dir, 'event_' + str(nd_i) + '.svg'))
                            plt.close(f)

# 			#Taste event deviation plots
# 			save_name = 'all_events'
# 			title='Deviation Events x Individual Taste Response'
# 			if not os.path.isfile(seg_decode_save_dir + save_name + '_population_corr.png'):
# 				plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs,
# 										all_taste_fr_vecs,taste_colors,num_neur,save_name,title,
# 										seg_decode_save_dir)
#
# 			#Taste event deviation plots z-scored
# 			save_name = 'all_events_z'
# 			title='Z-Scored Deviation Events x Individual Z-Scored Taste Response'
# 			if not os.path.isfile(seg_decode_save_dir + save_name + '_population_corr.png'):
# 				plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs_z,
# 									all_taste_fr_vecs_z,taste_colors,num_neur,save_name,title,
# 									seg_decode_save_dir)
#
# 			#Taste event deviation plots
# 			save_name = 'neur_cutoff_events'
# 			title='Deviation Events x Individual Taste Response'
# 			if not os.path.isfile(seg_decode_save_dir + save_name + '_population_corr.png'):
# 				plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs_neur_cut,
# 									all_taste_fr_vecs,taste_colors,num_neur,save_name,title,
# 									seg_decode_save_dir)
#
# 			#Taste event deviation plots z-scored
# 			save_name = 'neur_cutoff_events_z'
# 			title='Z-Scored Deviation Events x Individual Z-Scored Taste Response'
# 			if not os.path.isfile(seg_decode_save_dir + save_name + '_population_corr.png'):
# 				plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs_z_neur_cut,
# 									all_taste_fr_vecs_z,taste_colors,num_neur,save_name,title,
# 									seg_decode_save_dir)
#
# 			#Taste event deviation plots
# 			save_name = 'best_events'
# 			title='Deviation Events x Individual Taste Response'
# 			if not os.path.isfile(seg_decode_save_dir + save_name + '_population_corr.png'):
# 				plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs_best,
# 									all_taste_fr_vecs,taste_colors,num_neur,save_name,title,
# 									seg_decode_save_dir)
#
            # Taste event deviation plots z-scored
# 			save_name = 'best_events_z'
# 			title='Z-Scored Deviation Events x Individual Z-Scored Taste Response'
# 			if not os.path.isfile(seg_decode_save_dir + save_name + '_population_corr.png'):
# 				plot_violin_fr_vecs_taste_all(num_tastes,dig_in_names,all_taste_event_fr_vecs_z_best,
# 									all_taste_fr_vecs_z,taste_colors,num_neur,save_name,title,
# 									seg_decode_save_dir)
#

    # Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments
    f, ax = plt.subplots(nrows=len(epochs_to_analyze), ncols=num_tastes, figsize=(
        num_tastes*4, len(epochs_to_analyze)*4))
    if len(epochs_to_analyze) > 1:
        for e_ind, e_i in enumerate(epochs_to_analyze):
            times = (epoch_seg_taste_times[e_i,
                     segments_to_analyze, :]).squeeze()
            lengths = (
                epoch_seg_lengths[e_i, segments_to_analyze, :]).squeeze()
            for t_i in range(num_tastes-1):
                ax[e_ind, t_i].plot(segments_to_analyze, np.round(
                    100*(times[:, t_i]/lengths[:, t_i]), 2))
                seg_labels = [segment_names[a] for a in segments_to_analyze]
                ax[e_ind, t_i].set_xticks(
                    segments_to_analyze, labels=seg_labels, rotation=-45)
                if t_i == 0:
                    ax[e_ind, t_i].set_ylabel('Epoch ' + str(e_i))
                ax[e_ind, t_i].set_title('Taste ' + dig_in_names[t_i])
            # Instead of no-taste, plot summed percents
            joint_percents = np.sum(times[:, :-1], -1)/lengths[:, 0]
            ax[e_ind, t_i].plot(segments_to_analyze,
                                np.round(100*joint_percents, 2))
            seg_labels = [segment_names[a] for a in segments_to_analyze]
            ax[e_ind, t_i].set_xticks(
                segments_to_analyze, labels=seg_labels, rotation=-45)
            ax[e_ind, t_i].set_title('Combined Tastes')

    else:
        times = (
            epoch_seg_taste_times[epochs_to_analyze[0], segments_to_analyze, :]).squeeze()
        lengths = (
            epoch_seg_lengths[epochs_to_analyze[0], segments_to_analyze, :]).squeeze()
        for t_i in range(num_tastes-1):
            ax[t_i].plot(segments_to_analyze, np.round(
                100*(times[:, t_i]/lengths[:, t_i]), 2))
            seg_labels = [segment_names[a] for a in segments_to_analyze]
            ax[t_i].set_xticks(segments_to_analyze,
                               labels=seg_labels, rotation=-45)
            if t_i == 0:
                ax[t_i].set_ylabel('Epoch ' + str(epochs_to_analyze[0]))
            ax[t_i].set_title('Taste ' + dig_in_names[t_i])
        t_i = num_tastes-1
        joint_percents = np.sum(times[:, :-1], -1)/lengths[:, 0]
        ax[e_ind, t_i].plot(segments_to_analyze,
                            np.round(100*(joint_percents), 2))
        seg_labels = [segment_names[a] for a in segments_to_analyze]
        ax[e_ind, t_i].set_xticks(
            segments_to_analyze, labels=seg_labels, rotation=-45)
        ax[e_ind, t_i].set_title('Combined Tastes')
    plt.tight_layout()
    f.savefig(os.path.join(save_dir, 'Decoding_Percents.png'))
    f.savefig(os.path.join(save_dir, 'Decoding_Percents.svg'))
    plt.close(f)

    # Summary Plot of Time of Each Taste Decoded Across Epochs and Segments
    f, ax = plt.subplots(nrows=len(epochs_to_analyze), ncols=num_tastes, figsize=(
        num_tastes*4, len(epochs_to_analyze)*4))
    if len(epochs_to_analyze) > 1:
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for t_i in range(num_tastes):
                times = (
                    epoch_seg_taste_times[e_i, segments_to_analyze, t_i]).flatten()
                ax[e_ind, t_i].plot(segments_to_analyze, times)
                seg_labels = [segment_names[a] for a in segments_to_analyze]
                ax[e_ind, t_i].set_xticks(
                    segments_to_analyze, labels=seg_labels, rotation=-45)
                if t_i == 0:
                    ax[e_ind, t_i].set_ylabel('Epoch ' + str(e_i))
                ax[e_ind, t_i].set_title('Taste ' + dig_in_names[t_i])
    else:
        for t_i in range(num_tastes):
            times = (
                epoch_seg_taste_times[epochs_to_analyze[0], segments_to_analyze, t_i]).flatten()
            ax[t_i].plot(segments_to_analyze, times)
            seg_labels = [segment_names[a] for a in segments_to_analyze]
            ax[t_i].set_xticks(segments_to_analyze,
                               labels=seg_labels, rotation=-45)
            if t_i == 0:
                ax[t_i].set_ylabel('Epoch ' + str(epochs_to_analyze[0]))
            ax[t_i].set_title('Taste ' + dig_in_names[t_i])
    plt.tight_layout()
    f.savefig(save_dir + 'Decoding_Times.png')
    f.savefig(save_dir + 'Decoding_Times.svg')
    plt.close(f)

    # Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments
    f, ax = plt.subplots(nrows=len(epochs_to_analyze), ncols=num_tastes, figsize=(
        num_tastes*4, len(epochs_to_analyze)*4))
    if len(epochs_to_analyze) > 1:
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for t_i in range(num_tastes-1):
                times = (
                    epoch_seg_taste_times_neur_cut[e_i, segments_to_analyze, t_i]).flatten()
                lengths = (
                    epoch_seg_lengths[e_i, segments_to_analyze, t_i]).flatten()
                ax[e_ind, t_i].plot(segments_to_analyze,
                                    np.round(100*(times/lengths), 2))
                seg_labels = [segment_names[a] for a in segments_to_analyze]
                ax[e_ind, t_i].set_xticks(
                    segments_to_analyze, labels=seg_labels, rotation=-45)
                if t_i == 0:
                    ax[e_ind, t_i].set_ylabel('Epoch ' + str(e_i))
                ax[e_ind, t_i].set_title('Taste ' + dig_in_names[t_i])
            # Instead of no-taste, plot summed percents
            t_i = num_tastes - 1
            times = np.sum(
                epoch_seg_taste_times_neur_cut[e_i, segments_to_analyze, :-1].flatten(), -1)
            lengths = (
                epoch_seg_lengths[e_i, segments_to_analyze, 0]).flatten()
            ax[e_ind, t_i].plot(segments_to_analyze,
                                np.round(100*(times/lengths), 2))
            seg_labels = [segment_names[a] for a in segments_to_analyze]
            ax[e_ind, t_i].set_xticks(
                segments_to_analyze, labels=seg_labels, rotation=-45)
            ax[e_ind, t_i].set_title('Combined Tastes')
    else:
        for t_i in range(num_tastes-1):
            times = (
                epoch_seg_taste_times_neur_cut[epochs_to_analyze[0], segments_to_analyze, t_i]).flatten()
            lengths = (
                epoch_seg_lengths[epochs_to_analyze[0], segments_to_analyze, t_i]).flatten()
            ax[t_i].plot(segments_to_analyze, np.round(100*(times/lengths), 2))
            seg_labels = [segment_names[a] for a in segments_to_analyze]
            ax[t_i].set_xticks(segments_to_analyze,
                               labels=seg_labels, rotation=-45)
            if t_i == 0:
                ax[t_i].set_ylabel('Epoch ' + str(epochs_to_analyze[0]))
            ax[t_i].set_title('Taste ' + dig_in_names[t_i])
        t_i = num_tastes-1
        times = np.sum(
            epoch_seg_taste_times_neur_cut[epochs_to_analyze[0], segments_to_analyze, :-1].flatten(), -1)
        lengths = (
            epoch_seg_lengths[epochs_to_analyze[0], segments_to_analyze, 0]).flatten()
        ax[e_ind, t_i].plot(segments_to_analyze,
                            np.round(100*(times/lengths), 2))
        seg_labels = [segment_names[a] for a in segments_to_analyze]
        ax[e_ind, t_i].set_xticks(
            segments_to_analyze, labels=seg_labels, rotation=-45)
        ax[e_ind, t_i].set_title('Combined Tastes')
    plt.tight_layout()
    f.savefig(os.path.join(save_dir, 'Decoding_Percents_Neuron_Cutoff.png'))
    f.savefig(os.path.join(save_dir, 'Decoding_Percents_Neuron_Cutoff.svg'))
    plt.close(f)

    # Summary Plot of Time of Each Taste Decoded Across Epochs and Segments
    f, ax = plt.subplots(nrows=len(epochs_to_analyze), ncols=num_tastes, figsize=(
        num_tastes*4, len(epochs_to_analyze)*4))
    if len(epochs_to_analyze) > 1:
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for t_i in range(num_tastes):
                times = (
                    epoch_seg_taste_times_neur_cut[e_i, segments_to_analyze, t_i]).flatten()
                ax[e_ind, t_i].plot(segments_to_analyze, times)
                seg_labels = [segment_names[a] for a in segments_to_analyze]
                ax[e_ind, t_i].set_xticks(
                    segments_to_analyze, labels=seg_labels, rotation=-45)
                if t_i == 0:
                    ax[e_ind, t_i].set_ylabel('Epoch ' + str(e_i))
                ax[e_ind, t_i].set_title('Taste ' + dig_in_names[t_i])
    else:
        for t_i in range(num_tastes):
            times = (
                epoch_seg_taste_times_neur_cut[epochs_to_analyze[0], segments_to_analyze, t_i]).flatten()
            ax[t_i].plot(segments_to_analyze, times)
            seg_labels = [segment_names[a] for a in segments_to_analyze]
            ax[t_i].set_xticks(segments_to_analyze,
                               labels=seg_labels, rotation=-45)
            if t_i == 0:
                ax[t_i].set_ylabel('Epoch ' + str(epochs_to_analyze[0]))
            ax[t_i].set_title('Taste ' + dig_in_names[t_i])
    plt.tight_layout()
    f.savefig(os.path.join(save_dir, 'Decoding_Times_Neuron_Cutoff.png'))
    f.savefig(os.path.join(save_dir, 'Decoding_Times_Neuron_Cutoff.svg'))
    plt.close(f)

    # Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments
    f, ax = plt.subplots(nrows=len(epochs_to_analyze), ncols=num_tastes, figsize=(
        num_tastes*4, len(epochs_to_analyze)*4))
    if len(epochs_to_analyze) > 1:
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for t_i in range(num_tastes-1):
                times = (
                    epoch_seg_taste_times_best[e_i, segments_to_analyze, t_i]).flatten()
                lengths = (
                    epoch_seg_lengths[e_i, segments_to_analyze, t_i]).flatten()
                ax[e_ind, t_i].plot(segments_to_analyze,
                                    np.round(100*(times/lengths), 2))
                seg_labels = [segment_names[a] for a in segments_to_analyze]
                ax[e_ind, t_i].set_xticks(
                    segments_to_analyze, labels=seg_labels, rotation=-45)
                if t_i == 0:
                    ax[e_ind, t_i].set_ylabel('Epoch ' + str(e_i))
                ax[e_ind, t_i].set_title('Taste ' + dig_in_names[t_i])
            # Instead of no-taste, plot summed percents
            t_i = num_tastes - 1
            times = np.sum(
                epoch_seg_taste_times_best[e_i, segments_to_analyze, :-1].flatten(), -1)
            lengths = (
                epoch_seg_lengths[e_i, segments_to_analyze, 0]).flatten()
            ax[e_ind, t_i].plot(segments_to_analyze,
                                np.round(100*(times/lengths), 2))
            seg_labels = [segment_names[a] for a in segments_to_analyze]
            ax[e_ind, t_i].set_xticks(
                segments_to_analyze, labels=seg_labels, rotation=-45)
            ax[e_ind, t_i].set_title('Combined Tastes')
    else:
        for t_i in range(num_tastes):
            times = (
                epoch_seg_taste_times_best[epochs_to_analyze[0], segments_to_analyze, t_i]).flatten()
            lengths = (
                epoch_seg_lengths[epochs_to_analyze[0], segments_to_analyze, t_i]).flatten()
            ax[t_i].plot(segments_to_analyze, np.round(100*(times/lengths), 2))
            seg_labels = [segment_names[a] for a in segments_to_analyze]
            ax[t_i].set_xticks(segments_to_analyze,
                               labels=seg_labels, rotation=-45)
            if t_i == 0:
                ax[t_i].set_ylabel('Epoch ' + str(epochs_to_analyze[0]))
            ax[t_i].set_title('Taste ' + dig_in_names[t_i])
        t_i = num_tastes-1
        times = np.sum(
            epoch_seg_taste_times_best[epochs_to_analyze[0], segments_to_analyze, :-1].flatten(), -1)
        lengths = (
            epoch_seg_lengths[epochs_to_analyze[0], segments_to_analyze, 0]).flatten()
        ax[e_ind, t_i].plot(segments_to_analyze,
                            np.round(100*(times/lengths), 2))
        seg_labels = [segment_names[a] for a in segments_to_analyze]
        ax[e_ind, t_i].set_xticks(
            segments_to_analyze, labels=seg_labels, rotation=-45)
        ax[e_ind, t_i].set_title('Combined Tastes')
    plt.tight_layout()
    f.savefig(os.path.join(save_dir, 'Decoding_Percents_Best.png'))
    f.savefig(os.path.join(save_dir, 'Decoding_Percents_Best.svg'))
    plt.close(f)

    # Summary Plot of Time of Each Taste Decoded Across Epochs and Segments
    f, ax = plt.subplots(nrows=len(epochs_to_analyze), ncols=num_tastes, figsize=(
        num_tastes*4, len(epochs_to_analyze)*4))
    if len(epochs_to_analyze) > 1:
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for t_i in range(num_tastes):
                times = (
                    epoch_seg_taste_times_best[e_i, segments_to_analyze, t_i]).flatten()
                ax[e_ind, t_i].plot(segments_to_analyze, times)
                seg_labels = [segment_names[a] for a in segments_to_analyze]
                ax[e_ind, t_i].set_xticks(
                    segments_to_analyze, labels=seg_labels, rotation=-45)
                if t_i == 0:
                    ax[e_ind, t_i].set_ylabel('Epoch ' + str(e_i))
                ax[e_ind, t_i].set_title('Taste ' + dig_in_names[t_i])
    else:
        for t_i in range(num_tastes):
            times = (
                epoch_seg_taste_times_best[epochs_to_analyze[0], segments_to_analyze, t_i]).flatten()
            ax[t_i].plot(segments_to_analyze, times)
            seg_labels = [segment_names[a] for a in segments_to_analyze]
            ax[t_i].set_xticks(segments_to_analyze,
                               labels=seg_labels, rotation=-45)
            if t_i == 0:
                ax[t_i].set_ylabel('Epoch ' + str(epochs_to_analyze[0]))
            ax[t_i].set_title('Taste ' + dig_in_names[t_i])
    plt.tight_layout()
    f.savefig(os.path.join(save_dir, 'Decoding_Times_Best.png'))
    f.savefig(os.path.join(save_dir, 'Decoding_Times_Best.svg'))
    plt.close(f)


def plot_combined_decoded(fr_dist, num_tastes, num_neur, segment_spike_times,
                          tastant_spike_times, start_dig_in_times, end_dig_in_times,
                          post_taste_dt, pre_taste_dt, cp_raster_inds, bin_dt, dig_in_names,
                          segment_times, segment_names, taste_num_deliv, taste_select_epoch,
                          save_dir, max_decode, max_hz, seg_stat_bin, neuron_count_thresh,
                          e_len_dt, trial_start_frac=0, epochs_to_analyze=[],
                          segments_to_analyze=[], decode_prob_cutoff=0.95):
    """Function to plot the periods when something other than no taste is 
    decoded"""
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    num_segments = len(segment_spike_times)
    half_bin_z_dt = np.floor(bin_dt/2).astype('int')
    plot_len = 500  # in ms (dt) --> 6 seconds
    half_bin = np.floor(e_len_dt/2).astype('int')  # Decoding half bin size
    hatch_types = ['//', '', 'o', '..', '', '**', '\\\\', '--', '', 'OO',
                   '||', '/', '', '\\', '++', '|', 'oo', '-', '+', 'O', '.', '*']
    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)

    for s_i in tqdm.tqdm(segments_to_analyze):
        print("\t\t\tPlotting Segment " + str(s_i) + '')

        seg_decode_save_dir = os.path.join(save_dir, 'segment_' + str(s_i))
        if not os.path.isdir(seg_decode_save_dir):
            os.mkdir(seg_decode_save_dir)

        # Load segment spike times
        seg_start = segment_times[s_i]
        seg_end = segment_times[s_i+1]
        seg_len = seg_end - seg_start  # in dt = ms

        # Import raster plots for segment
        # TODO: Build up taste selective neuron capacity
        segment_spike_times_s_i = segment_spike_times[s_i]
        segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
        for n_i in range(num_neur):
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
        
		# Collect across epochs the decoding of tastes for this segment
        all_epoch_decode_prob = dict()
        for t_i in range(num_tastes):
            all_epoch_decode_prob[dig_in_names[t_i]] = dict()
            all_epoch_decode_prob[dig_in_names[t_i]]['Probabilities'] = []
            all_epoch_decode_prob[dig_in_names[t_i]]['Labels'] = []
        for e_i in epochs_to_analyze:
            epoch_decode_save_dir = os.path.join(
                save_dir, 'decode_prob_epoch_' + str(e_i))
            if not os.path.isdir(epoch_decode_save_dir):
                pass
            try:
                seg_decode_epoch_prob = np.load(os.path.join(
                    epoch_decode_save_dir, 'segment_' + str(s_i) + '.npy'))
            except:
                pass

            decoded_taste_max = np.argmax(seg_decode_epoch_prob, 0)
            decoded_taste_bin = np.zeros((num_tastes, len(decoded_taste_max)))
            for t_i in range(num_tastes):
                decoded_taste_bin[t_i, np.where(
                    decoded_taste_max == t_i)[0]] = 1

            for t_i in range(num_tastes):
                decode_bins = np.where(decoded_taste_max == t_i)[0]
                new_decode_prob_vec = np.zeros(len(decoded_taste_max))
                new_decode_prob_vec[decode_bins] = seg_decode_epoch_prob[t_i, decode_bins]
                # Pull each taste's decode information and label and store
                all_epoch_decode_prob[dig_in_names[t_i]]['Probabilities'].append(
                    new_decode_prob_vec)
                all_epoch_decode_prob[dig_in_names[t_i]
                                      ]['Labels'].append('epoch_' + str(e_i))

        # Plot decodings in minute chunks for individual tastes - show epoch progression
        epoch_progression_plots(num_tastes, dig_in_names, all_epoch_decode_prob,
                                seg_len, plot_len, hatch_types, decode_prob_cutoff,
                                max_decode, segment_spike_times_s_i_bin, num_neur,
                                half_bin, seg_decode_save_dir)

        # Plot decodings of taste progressions - merge all epochs together to just be a taste
        taste_progression_plots(seg_len, plot_len, num_tastes, hatch_types, all_epoch_decode_prob,
                                dig_in_names, decode_prob_cutoff, max_decode, epochs_to_analyze,
                                segment_spike_times_s_i_bin, num_neur, half_bin,
                                seg_decode_save_dir)


def plot_decoded_func_p(fr_dist, num_tastes, num_neur, segment_spike_times, tastant_spike_times,
                        start_dig_in_times, end_dig_in_times, post_taste_dt, cp_raster_inds,
                        e_skip_dt, e_len_dt, dig_in_names, segment_times,
                        segment_names, taste_num_deliv, taste_select_epoch,
                        save_dir, max_decode, max_hz, seg_stat_bin,
                        epochs_to_analyze=[], segments_to_analyze=[]):
    """Function to plot the decoding statistics as a function of average decoding
    probability within the decoded interval."""
    warnings.filterwarnings('ignore')
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    num_segments = len(segment_spike_times)
    prob_cutoffs = np.arange(1/num_tastes, 1, 0.05)
    num_prob = len(prob_cutoffs)
    taste_colors = cm.viridis(np.linspace(0, 1, num_tastes))
    epoch_seg_taste_percents = np.zeros(
        (num_prob, num_cp, num_segments, num_tastes))

    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)

    for e_i in epochs_to_analyze:  # By epoch conduct decoding
        print('\t\t\tDecoding Epoch ' + str(e_i))
        taste_select_neur = np.where(taste_select_epoch[e_i, :] == 1)[0]
        epoch_decode_save_dir = os.path.join(
            save_dir, 'decode_prob_epoch_' + str(e_i) + '/')
        if not os.path.isdir(epoch_decode_save_dir):
            print("\t\t\t\tData not previously decoded, or passed directory incorrect.")
            pass

        for s_i in tqdm.tqdm(segments_to_analyze):
            try:
                seg_decode_epoch_prob = np.load(os.path.join(
                    epoch_decode_save_dir, 'segment_' + str(s_i) + '.npy'))
            except:
                print("\t\t\t\tSegment " + str(s_i) + " Never Decoded")
                pass

            seg_decode_save_dir = os.path.join(
                epoch_decode_save_dir, 'segment_' + str(s_i))
            if not os.path.isdir(seg_decode_save_dir):
                os.mkdir(seg_decode_save_dir)

            seg_start = segment_times[s_i]
            seg_end = segment_times[s_i+1]
            seg_len = seg_end - seg_start  # in dt = ms

            # Import raster plots for segment
            segment_spike_times_s_i = segment_spike_times[s_i]
            segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
            for n_i in taste_select_neur:
                n_i_spike_times = np.array(
                    segment_spike_times_s_i[n_i] - seg_start).astype('int')
                segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1

            decoded_taste_max = np.argmax(seg_decode_epoch_prob, 0)

            # Calculate decoded taste stats by probability cutoff
            num_neur_mean_p = np.zeros((num_tastes, num_prob))
            num_neur_std_p = np.zeros((num_tastes, num_prob))
            iei_mean_p = np.zeros((num_tastes, num_prob))
            iei_std_p = np.zeros((num_tastes, num_prob))
            len_mean_p = np.zeros((num_tastes, num_prob))
            len_std_p = np.zeros((num_tastes, num_prob))
            prob_mean_p = np.zeros((num_tastes, num_prob))
            prob_std_p = np.zeros((num_tastes, num_prob))
            for t_i in range(num_tastes):
                for prob_i, prob_val in enumerate(prob_cutoffs):
                    # Find where the decoding matches the probability cutoff for each taste
                    decode_prob_bin = seg_decode_epoch_prob[t_i, :] > prob_val
                    decode_max_bin = decoded_taste_max == t_i
                    decoded_taste = (
                        decode_prob_bin*decode_max_bin).astype('int')
                    decoded_taste[0] = 0
                    decoded_taste[-1] = 0

                    # Store the decoding percents
                    epoch_seg_taste_percents[prob_i, e_i, s_i, t_i] = (
                        np.sum(decoded_taste)/len(decoded_taste))*100

                    # Calculate statistics of num neur, IEI, event length, average decoding prob
                    start_decoded = np.where(
                        np.diff(decoded_taste) == 1)[0] + 1
                    end_decoded = np.where(np.diff(decoded_taste) == -1)[0] + 1
                    num_decoded = len(start_decoded)

                    # Create plots of decoded period statistics
                    # __Length
                    len_decoded = end_decoded-start_decoded
                    len_mean_p[t_i, prob_i] = np.nanmean(len_decoded)
                    len_std_p[t_i, prob_i] = np.nanstd(len_decoded)
                    # __IEI
                    iei_decoded = start_decoded[1:] - end_decoded[:-1]
                    iei_mean_p[t_i, prob_i] = np.nanmean(iei_decoded)
                    iei_std_p[t_i, prob_i] = np.nanstd(iei_decoded)
                    num_neur_decoded = np.zeros(num_decoded)
                    prob_decoded = np.zeros(num_decoded)
                    for nd_i in range(num_decoded):
                        d_start = start_decoded[nd_i]
                        d_end = end_decoded[nd_i]
                        for n_i in range(num_neur):
                            if len(np.where(segment_spike_times_s_i_bin[n_i, d_start:d_end])[0]) > 0:
                                num_neur_decoded[nd_i] += 1
                        prob_decoded[nd_i] = np.mean(
                            seg_decode_epoch_prob[t_i, d_start:d_end])
                    # __Num Neur
                    num_neur_mean_p[t_i, prob_i] = np.nanmean(num_neur_decoded)
                    num_neur_std_p[t_i, prob_i] = np.nanstd(num_neur_decoded)
                    # __Prob
                    prob_mean_p[t_i, prob_i] = np.nanmean(prob_decoded)
                    prob_std_p[t_i, prob_i] = np.nanstd(prob_decoded)

            # Plot statistics
            f, ax = plt.subplots(2, 3, figsize=(
                8, 8), width_ratios=[10, 10, 1])
            gs = ax[0, -1].get_gridspec()
            ax[0, 0].set_ylim([0, num_neur])
            ax[0, 1].set_ylim(
                [0, np.nanmax(len_mean_p) + np.nanmax(len_std_p) + 10])
            ax[1, 0].set_ylim(
                [0, np.nanmax(iei_mean_p) + np.nanmax(iei_std_p) + 10])
            ax[1, 1].set_ylim([0, 1.2])
            for t_i in range(num_tastes):
                # __Num Neur
                ax[0, 0].errorbar(prob_cutoffs, num_neur_mean_p[t_i, :], num_neur_std_p[t_i, :],
                                  linestyle='None', marker='o', color=taste_colors[t_i, :], alpha=0.8)
                # __Length
                ax[0, 1].errorbar(prob_cutoffs, len_mean_p[t_i, :], len_std_p[t_i, :],
                                  linestyle='None', marker='o', color=taste_colors[t_i, :], alpha=0.8)
                # __IEI
                ax[1, 0].errorbar(prob_cutoffs, iei_mean_p[t_i, :], iei_std_p[t_i, :],
                                  linestyle='None', marker='o', color=taste_colors[t_i, :], alpha=0.8)
                # __Prob
                ax[1, 1].errorbar(prob_cutoffs, prob_mean_p[t_i, :], prob_std_p[t_i, :],
                                  linestyle='None', marker='o', color=taste_colors[t_i, :], alpha=0.8)
            ax[0, 0].set_title('Number of Neurons')
            ax[0, 1].set_title('Length of Event')
            ax[1, 0].set_title('IEI (ms)')
            ax[1, 1].set_title('Average P(Decoding)')
            for ax_i in ax[:, -1]:
                ax_i.remove()  # remove the underlying axes
            axbig = f.add_subplot(gs[:, -1])
            cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'), cax=axbig, ticks=np.linspace(
                0, 1, num_tastes), orientation='vertical')
            cbar.ax.set_yticklabels(dig_in_names)
            #Edit and Save
            f.suptitle('Decoding Statistics by Probability Cutoff')
            plt.tight_layout()
            f.savefig(os.path.join(
                seg_decode_save_dir, 'prob_cutoff_stats.png'))
            f.savefig(os.path.join(
                seg_decode_save_dir, 'prob_cutoff_stats.svg'))
            plt.close(f)

    # Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments
    sum_width_ratios = np.concatenate(
        (10*np.ones(len(segments_to_analyze)), np.ones(1)))
    max_decoding_percent = np.max(epoch_seg_taste_percents)
    f, ax = plt.subplots(len(epochs_to_analyze), len(segments_to_analyze) + 1, figsize=(
        (len(segments_to_analyze) + 1)*4, len(epochs_to_analyze)*4), width_ratios=sum_width_ratios)
    if len(epochs_to_analyze) > 1:
        gs = ax[0, -1].get_gridspec()
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for s_ind, s_i in enumerate(segments_to_analyze):
                ax[e_ind, s_ind].set_ylim([0, max_decoding_percent+10])
                for t_i in range(num_tastes):
                    ax[e_ind, s_ind].plot(prob_cutoffs, (epoch_seg_taste_percents[:, e_i, s_i, t_i]).flatten(
                    ), color=taste_colors[t_i, :], alpha=0.8)
                if s_ind == 0:
                    ax[e_ind, s_ind].set_ylabel('Epoch ' + str(e_i))
                if e_ind == 0:
                    ax[e_ind, s_ind].set_title(segment_names[s_i])
                if e_ind == num_cp-1:
                    ax[e_ind, s_ind].set_xlabel('Probability Cutoff')
        for ax_i in ax[:, -1]:
            ax_i.remove()  # remove the underlying axes
    else:
        gs = ax[-1].get_gridspec()
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for s_ind, s_i in enumerate(segments_to_analyze):
                ax[s_ind].set_ylim([0, max_decoding_percent+10])
                for t_i in range(num_tastes):
                    ax[s_ind].plot(prob_cutoffs, (epoch_seg_taste_percents[:, e_i, s_i, t_i]).flatten(
                    ), color=taste_colors[t_i, :], alpha=0.8)
                if s_ind == 0:
                    ax[s_ind].set_ylabel('Epoch ' + str(e_i))
                if e_ind == 0:
                    ax[s_ind].set_title(segment_names[s_i])
                if e_ind == num_cp-1:
                    ax[s_ind].set_xlabel('Probability Cutoff')
        ax[-1].remove()  # remove the underlying axes
    axbig = f.add_subplot(gs[:, -1])
    cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'), cax=axbig,
                        ticks=np.linspace(0, 1, num_tastes), orientation='vertical')
    cbar.ax.set_yticklabels(dig_in_names)
    plt.tight_layout()
    f.savefig(os.path.join(save_dir, 'Decoding_Percents_By_Prob_Cutoff.png'))
    f.savefig(os.path.join(save_dir, 'Decoding_Percents_By_Prob_Cutoff.svg'))
    plt.close(f)


def plot_decoded_func_n(fr_dist, num_tastes, num_neur, segment_spike_times, tastant_spike_times,
                        start_dig_in_times, end_dig_in_times, post_taste_dt, cp_raster_inds,
                        e_skip_dt, e_len_dt, dig_in_names, segment_times,
                        segment_names, taste_num_deliv, taste_select_epoch,
                        save_dir, max_decode, max_hz, seg_stat_bin,
                        epochs_to_analyze=[], segments_to_analyze=[]):
    """Function to plot the decoding statistics as a function of number of 
    neurons firing within the decoded interval."""
    warnings.filterwarnings('ignore')
    num_cp = np.shape(cp_raster_inds[0])[-1] - 1
    num_segments = len(segment_spike_times)
    neur_cutoffs = np.arange(1, num_neur)
    num_cuts = len(neur_cutoffs)
    taste_colors = cm.viridis(np.linspace(0, 1, num_tastes))
    epoch_seg_taste_percents = np.zeros(
        (num_cuts, num_cp, num_segments, num_tastes))

    if len(epochs_to_analyze) == 0:
        epochs_to_analyze = np.arange(num_cp)
    if len(segments_to_analyze) == 0:
        segments_to_analyze = np.arange(num_segments)

    for e_i in epochs_to_analyze:  # By epoch conduct decoding
        print('\t\t\tDecoding Epoch ' + str(e_i))
        taste_select_neur = np.where(taste_select_epoch[e_i, :] == 1)[0]
        epoch_decode_save_dir = save_dir + \
            'decode_prob_epoch_' + str(e_i) + '/'
        if not os.path.isdir(epoch_decode_save_dir):
            print("\t\t\t\tData not previously decoded, or passed directory incorrect.")
            pass

        for s_i in tqdm.tqdm(segments_to_analyze):
            try:
                seg_decode_epoch_prob = np.load(
                    epoch_decode_save_dir + 'segment_' + str(s_i) + '.npy')
            except:
                print("\t\t\t\tSegment " + str(s_i) + " Never Decoded")
                pass

            seg_decode_save_dir = os.path.join(
                epoch_decode_save_dir, 'segment_' + str(s_i))
            if not os.path.isdir(seg_decode_save_dir):
                os.mkdir(seg_decode_save_dir)

            seg_start = segment_times[s_i]
            seg_end = segment_times[s_i+1]
            seg_len = seg_end - seg_start  # in dt = ms

            # Import raster plots for segment
            segment_spike_times_s_i = segment_spike_times[s_i]
            segment_spike_times_s_i_bin = np.zeros((num_neur, seg_len+1))
            for n_i in taste_select_neur:
                n_i_spike_times = np.array(
                    segment_spike_times_s_i[n_i] - seg_start).astype('int')
                segment_spike_times_s_i_bin[n_i, n_i_spike_times] = 1

            decoded_taste_max = np.argmax(seg_decode_epoch_prob, 0)

            # Calculate decoded taste stats by probability cutoff
            num_neur_mean_p = np.zeros((num_tastes, num_cuts))
            num_neur_std_p = np.zeros((num_tastes, num_cuts))
            iei_mean_p = np.zeros((num_tastes, num_cuts))
            iei_std_p = np.zeros((num_tastes, num_cuts))
            len_mean_p = np.zeros((num_tastes, num_cuts))
            len_std_p = np.zeros((num_tastes, num_cuts))
            prob_mean_p = np.zeros((num_tastes, num_cuts))
            prob_std_p = np.zeros((num_tastes, num_cuts))
            for t_i in range(num_tastes):
                # First calculate neurons decoded in all decoded intervals
                decoded_taste = (decoded_taste_max == t_i).astype('int')
                decoded_taste[0] = 0
                decoded_taste[-1] = 0
                diff_decoded_taste = np.diff(decoded_taste)
                start_decoded = np.where(diff_decoded_taste == 1)[0] + 1
                end_decoded = np.where(diff_decoded_taste == -1)[0] + 1
                num_decoded = len(start_decoded)
                num_neur_decoded = np.zeros(num_decoded)
                prob_decoded = np.zeros(num_decoded)
                for nd_i in range(num_decoded):
                    d_start = start_decoded[nd_i]
                    d_end = end_decoded[nd_i]
                    for n_i in range(num_neur):
                        if len(np.where(segment_spike_times_s_i_bin[n_i, d_start:d_end])[0]) > 0:
                            num_neur_decoded[nd_i] += 1
                    prob_decoded[nd_i] = np.mean(
                        seg_decode_epoch_prob[t_i, d_start:d_end])
                # Next look at stats as exclude by #neurons
                for cut_i, cut_val in enumerate(neur_cutoffs):
                    # Find where the decoding matches the neuron cutoff
                    decode_ind = np.where(num_neur_decoded > cut_val)[0]
                    decoded_bin = np.zeros(np.shape(decoded_taste))
                    for db in decode_ind:
                        s_db = start_decoded[db]
                        e_db = end_decoded[db]
                        decoded_bin[s_db:e_db] = 1
                    # Store the decoding percents
                    epoch_seg_taste_percents[cut_i, e_i, s_i, t_i] = (
                        np.sum(decoded_bin)/len(decoded_bin))*100
                    # Calculate statistics of num neur, IEI, event length, average decoding prob
                    num_decoded_i = num_neur_decoded[decode_ind]
                    prob_decoded_i = prob_decoded[decode_ind]
                    iei_i = start_decoded[decode_ind[1:]
                                          ] - end_decoded[decode_ind[:-1]]
                    len_i = end_decoded[decode_ind] - start_decoded[decode_ind]

                    # Create plots of decoded period statistics
                    # __Length
                    len_mean_p[t_i, cut_i] = np.nanmean(len_i)
                    len_std_p[t_i, cut_i] = np.nanstd(len_i)
                    # __IEI
                    iei_mean_p[t_i, cut_i] = np.nanmean(iei_i)
                    iei_std_p[t_i, cut_i] = np.nanstd(iei_i)
                    # __Num Neur
                    num_neur_mean_p[t_i, cut_i] = np.nanmean(num_decoded_i)
                    num_neur_std_p[t_i, cut_i] = np.nanstd(num_decoded_i)
                    # __Prob
                    prob_mean_p[t_i, cut_i] = np.nanmean(prob_decoded_i)
                    prob_std_p[t_i, cut_i] = np.nanstd(prob_decoded_i)

            # Plot statistics
            f, ax = plt.subplots(2, 3, figsize=(
                8, 8), width_ratios=[10, 10, 1])
            gs = ax[0, -1].get_gridspec()
            ax[0, 0].set_ylim([0, num_neur])
            ax[0, 1].set_ylim(
                [0, np.nanmax(len_mean_p) + np.nanmax(len_std_p) + 10])
            ax[1, 0].set_ylim(
                [0, np.nanmax(iei_mean_p) + np.nanmax(iei_std_p) + 10])
            ax[1, 1].set_ylim([0, 1.2])
            for t_i in range(num_tastes):
                # __Num Neur
                ax[0, 0].errorbar(neur_cutoffs, num_neur_mean_p[t_i, :], num_neur_std_p[t_i, :],
                                  linestyle='None', marker='o', color=taste_colors[t_i, :], alpha=0.8)
                # __Length
                ax[0, 1].errorbar(neur_cutoffs, len_mean_p[t_i, :], len_std_p[t_i, :],
                                  linestyle='None', marker='o', color=taste_colors[t_i, :], alpha=0.8)
                # __IEI
                ax[1, 0].errorbar(neur_cutoffs, iei_mean_p[t_i, :], iei_std_p[t_i, :],
                                  linestyle='None', marker='o', color=taste_colors[t_i, :], alpha=0.8)
                # __Prob
                ax[1, 1].errorbar(neur_cutoffs, prob_mean_p[t_i, :], prob_std_p[t_i, :],
                                  linestyle='None', marker='o', color=taste_colors[t_i, :], alpha=0.8)
            ax[0, 0].set_title('Number of Neurons')
            ax[0, 1].set_title('Length of Event')
            ax[1, 0].set_title('IEI (ms)')
            ax[1, 1].set_title('Average P(Decoding)')
            for ax_i in ax[:, -1]:
                ax_i.remove()  # remove the underlying axes
            axbig = f.add_subplot(gs[:, -1])
            cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'), cax=axbig, ticks=np.linspace(
                0, 1, num_tastes), orientation='vertical')
            cbar.ax.set_yticklabels(dig_in_names)
            #Edit and Save
            f.suptitle('Decoding Statistics by Neuron Cutoff')
            plt.tight_layout()
            f.savefig(os.path.join(
                seg_decode_save_dir, 'neur_cutoff_stats.png'))
            f.savefig(os.path.join(
                seg_decode_save_dir, 'neur_cutoff_stats.svg'))
            plt.close(f)

    # Summary Plot of Percent of Each Taste Decoded Across Epochs and Segments
    sum_width_ratios = np.concatenate(
        (10*np.ones(len(segments_to_analyze)), np.ones(1)))
    max_decoding_percent = np.max(epoch_seg_taste_percents)
    f, ax = plt.subplots(len(epochs_to_analyze), len(segments_to_analyze) + 1, figsize=(
        (len(segments_to_analyze) + 1)*4, len(epochs_to_analyze)*4), width_ratios=sum_width_ratios)
    if len(epochs_to_analyze) > 1:
        gs = ax[0, -1].get_gridspec()
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for s_ind, s_i in enumerate(segments_to_analyze):
                ax[e_ind, s_ind].set_ylim([0, max_decoding_percent+10])
                for t_i in range(num_tastes):
                    ax[e_ind, s_ind].plot(neur_cutoffs, (epoch_seg_taste_percents[:, e_i, s_i, t_i]).flatten(
                    ), color=taste_colors[t_i, :], alpha=0.8)
                if s_ind == 0:
                    ax[e_ind, s_ind].set_ylabel('Epoch ' + str(e_i))
                if e_ind == 0:
                    ax[e_ind, s_ind].set_title(segment_names[s_i])
                if e_ind == num_cp-1:
                    ax[e_ind, s_ind].set_xlabel('Neuron Cutoff')
        for ax_i in ax[:, -1]:
            ax_i.remove()  # remove the underlying axes
    else:
        gs = ax[-1].get_gridspec()
        for e_ind, e_i in enumerate(epochs_to_analyze):
            for s_ind, s_i in enumerate(segments_to_analyze):
                ax[s_ind].set_ylim([0, max_decoding_percent+10])
                for t_i in range(num_tastes):
                    ax[s_ind].plot(neur_cutoffs, (epoch_seg_taste_percents[:, e_i, s_i, t_i]).flatten(
                    ), color=taste_colors[t_i, :], alpha=0.8)
                if s_ind == 0:
                    ax[s_ind].set_ylabel('Epoch ' + str(e_i))
                if e_ind == 0:
                    ax[s_ind].set_title(segment_names[s_i])
                if e_ind == num_cp-1:
                    ax[s_ind].set_xlabel('Neuron Cutoff')
        ax[-1].remove()  # remove the underlying axes
    axbig = f.add_subplot(gs[:, -1])
    cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'), cax=axbig,
                        ticks=np.linspace(0, 1, num_tastes), orientation='vertical')
    cbar.ax.set_yticklabels(dig_in_names)
    plt.tight_layout()
    f.savefig(os.path.join(save_dir, 'Decoding_Percents_By_Neur_Cutoff.png'))
    f.savefig(os.path.join(save_dir, 'Decoding_Percents_By_Neur_Cutoff.svg'))
    plt.close(f)


def plot_overall_decoded_stats(len_decoded, iei_decoded, num_neur_decoded,
                               prob_decoded, prob_distribution, e_i, s_i,
                               seg_dist_midbin, seg_distribution, seg_stat_bin,
                               seg_len, save_name, taste_decode_save_dir):
    """For use by the plot_decoded function - plots decoded event statistics"""

    # Plot the statistics for those decoded events that are best across metrics
    f = plt.figure(figsize=(8, 8))
    plt.subplot(3, 2, 1)
    plt.hist(len_decoded)
    plt.xlabel('Length (ms)')
    plt.ylabel('Number of Occurrences')
    plt.title('Length of Decoded Event')
    plt.subplot(3, 2, 2)
    plt.hist(iei_decoded)
    plt.xlabel('IEI (ms)')
    plt.ylabel('Number of Occurrences')
    plt.title('Inter-Event-Interval (IEI)')
    plt.subplot(3, 2, 3)
    plt.hist(num_neur_decoded)
    plt.xlabel('# Neurons')
    plt.ylabel('Number of Occurrences')
    plt.title('Number of Neurons Active')
    plt.subplot(3, 2, 4)
    plt.bar(seg_dist_midbin, seg_distribution, width=seg_stat_bin)
    plt.xticks(np.linspace(0, seg_len, 8), labels=np.round(
        (np.linspace(0, seg_len, 8)/1000/60), 2), rotation=-45)
    plt.xlabel('Time in Segment (min)')
    plt.ylabel('# Events')
    plt.title('Number of Decoded Events')
    plt.subplot(3, 2, 5)
    plt.hist(prob_decoded)
    plt.xlabel('Event Avg P(Decoding)')
    plt.ylabel('Number of Occurrences')
    plt.title('Average Decoding Probability')
    plt.subplot(3, 2, 6)
    plt.bar(seg_dist_midbin, prob_distribution, width=seg_stat_bin)
    plt.xticks(np.linspace(0, seg_len, 8), labels=np.round(
        (np.linspace(0, seg_len, 8)/1000/60), 2), rotation=-45)
    plt.xlabel('Time in Segment (min)')
    plt.ylabel('Avg(Event Avg P(Decoding))')
    plt.title('Average Decoding Probability')
    plt.suptitle('')
    plt.tight_layout()
    f.savefig(os.path.join(taste_decode_save_dir, 'epoch_' +
              str(e_i) + '_seg_' + str(s_i) + '_' + save_name + '.png'))
    f.savefig(os.path.join(taste_decode_save_dir, 'epoch_' +
              str(e_i) + '_seg_' + str(s_i) + '_' + save_name + '.svg'))
    plt.close(f)


def plot_scatter_fr_vecs_taste_mean(num_tastes, dig_in_names, all_taste_event_fr_vecs,
                                    all_taste_fr_vecs_mean, taste_colors, save_name, title,
                                    seg_decode_save_dir):
    """Plot a scatter plot of the decoded event firing rate against the
    average taste response firing rate"""
    f, ax = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(
        num_tastes*2, num_tastes*2), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
    max_fr = 0
    max_fr_t_av = 0
    for t_i in range(num_tastes):  # Event Taste
        ax[t_i, 0].set_ylabel('Decoded ' + dig_in_names[t_i] + ' FR')
        taste_event_fr_vecs = all_taste_event_fr_vecs[t_i]
        if len(taste_event_fr_vecs) > 0:
            max_taste_fr = np.max(taste_event_fr_vecs)
            if max_taste_fr > max_fr:
                max_fr = max_taste_fr
            for t_i_c in range(num_tastes):  # Average Taste
                average_fr_vec_mat = all_taste_fr_vecs_mean[t_i_c, :]*np.ones(
                    np.shape(taste_event_fr_vecs))
                # Calculate max avg fr
                max_avg_fr = np.max(all_taste_fr_vecs_mean[t_i_c, :])
                if max_avg_fr > max_fr_t_av:
                    max_fr_t_av = max_avg_fr
                ax[t_i, t_i_c].set_xlabel(
                    'Average ' + dig_in_names[t_i_c] + ' FR')
                ax[t_i, t_i_c].scatter(
                    average_fr_vec_mat, taste_event_fr_vecs, color=taste_colors[t_i, :], alpha=0.3)
    for t_i in range(num_tastes):
        for t_i_c in range(num_tastes):
            ax[t_i, t_i_c].plot([0, max_fr], [0, max_fr],
                                alpha=0.5, color='k', linestyle='dashed')
            ax[t_i, t_i_c].set_ylim([0, max_fr])
            ax[t_i, t_i_c].set_xlim([0, max_fr_t_av])
            if t_i == t_i_c:
                for child in ax[t_i, t_i_c].get_children():
                    if isinstance(child, matplotlib.spines.Spine):
                        child.set_color('r')
    plt.suptitle(title)
    plt.tight_layout()
    f.savefig(os.path.join(seg_decode_save_dir, save_name + '.png'))
    f.savefig(os.path.join(seg_decode_save_dir, save_name + '.svg'))
    plt.close(f)


def plot_violin_fr_vecs_taste_all(num_tastes, dig_in_names, all_taste_event_fr_vecs,
                                  all_taste_fr_vecs, taste_colors, num_neur, save_name, title,
                                  seg_decode_save_dir):
    """Plot distances in fr space between the decoded event firing rate and the
    taste response firing rate and correlations between decoded event firing
    rates and taste firing rates"""
    f_dist, ax_dist = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(
        num_tastes*4, num_tastes*4), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
    f_corr, ax_corr = plt.subplots(nrows=num_tastes, ncols=num_tastes, figsize=(
        num_tastes*4, num_tastes*4), gridspec_kw=dict(width_ratios=list(6*np.ones(num_tastes))))
    max_x = 0
    all_diff_tastes = []
    all_corr_tastes = []
    max_density = 0
    for t_i in range(num_tastes):  # Event Taste
        all_diff_taste = []
        all_corr_taste = []
        taste_event_fr_vecs = all_taste_event_fr_vecs[t_i]
        ax_dist[t_i, 0].set_ylabel('Firing Rate Difference')
        ax_corr[t_i, 0].set_ylabel(dig_in_names[t_i])
        for t_i_c in range(num_tastes):  # Average Taste
            # Calculate max fr of taste response
            ax_dist[t_i, t_i_c].set_title(
                'Decoded ' + dig_in_names[t_i] + ' - Delivery ' + dig_in_names[t_i_c])
            ax_corr[t_i, t_i_c].set_title('Decodes x Deliveries')
            taste_fr_vecs = all_taste_fr_vecs[t_i_c]
            max_taste_resp_fr = np.max(taste_fr_vecs)
            x_vals = np.arange(num_neur)
            if max_taste_resp_fr > max_x:
                max_x = max_taste_resp_fr
            num_taste_deliv = np.shape(taste_fr_vecs)[0]
            num_events = np.shape(taste_event_fr_vecs)[0]
            if num_events > 0:
                all_diff = np.zeros((num_events*num_taste_deliv, num_neur))
                all_corr = []
                for td_i in range(num_taste_deliv):
                    diff = np.abs(taste_event_fr_vecs - taste_fr_vecs[td_i, :])
                    corr = corr_calculator(
                        taste_fr_vecs[td_i, :], taste_event_fr_vecs)
                    all_diff[td_i*num_events:(td_i+1)*num_events, :] = diff
                    all_corr.extend(corr)
                all_diff_taste.append(all_diff.flatten())
                all_corr_taste.append(all_corr)
                ax_dist[t_i, t_i_c].violinplot(
                    all_diff, x_vals, showmedians=True, showextrema=False)
                hist_results = ax_corr[t_i, t_i_c].hist(np.array(
                    all_corr), density=True, histtype='step', bins=np.arange(-1, 1, 0.025))
                if np.max(hist_results[0]) > max_density:
                    max_density = np.max(hist_results[0])
                ax_corr[t_i, t_i_c].axvline(np.nanmean(
                    all_corr), linestyle='dashed', color='k', alpha=0.7)
                ax_corr[t_i, t_i_c].text(np.nanmean(
                    all_corr)+0.1, 0, str(np.round(np.nanmean(all_corr), 2)), rotation=90)
                ax_dist[t_i, t_i_c].set_xlabel('Neuron Index')
                ax_corr[t_i, t_i_c].set_xlabel(dig_in_names[t_i_c])
            else:
                all_diff_taste.append([])
                all_corr_taste.append([])
        all_diff_tastes.append(all_diff_taste)
        all_corr_tastes.append(all_corr_taste)
    for t_i in range(num_tastes):
        for t_i_c in range(num_tastes):
            ax_dist[t_i, t_i_c].axhline(
                0, alpha=0.5, color='k', linestyle='dashed')
            ax_dist[t_i, t_i_c].set_ylim([-10, 100])
            ax_corr[t_i, t_i_c].set_ylim([-0.1, max_density])
            if t_i == t_i_c:
                for child in ax_dist[t_i, t_i_c].get_children():
                    if isinstance(child, matplotlib.spines.Spine):
                        child.set_color('r')
                for child in ax_corr[t_i, t_i_c].get_children():
                    if isinstance(child, matplotlib.spines.Spine):
                        child.set_color('r')
    f_dist.suptitle(title)
    f_dist.tight_layout()
    f_dist.savefig(os.path.join(seg_decode_save_dir,
                   save_name + '_distances.png'))
    f_dist.savefig(os.path.join(seg_decode_save_dir,
                   save_name + '_distances.svg'))
    plt.close(f_dist)
    f_corr.suptitle(title)
    f_corr.tight_layout()
    f_corr.savefig(os.path.join(seg_decode_save_dir,
                   save_name + '_correlations.png'))
    f_corr.savefig(os.path.join(seg_decode_save_dir,
                   save_name + '_correlations.svg'))
    plt.close(f_corr)

    f2_dist, ax2_dist = plt.subplots(
        nrows=1, ncols=num_tastes, figsize=(num_tastes*4, 4))
    f2_corr, ax2_corr = plt.subplots(
        nrows=1, ncols=num_tastes, figsize=(num_tastes*4, 4))
    for t_i in range(num_tastes):
        ax2_dist[t_i].hist(all_diff_tastes[t_i], bins=1000, histtype='step',
                           density=True, cumulative=True, label=dig_in_names)
        ax2_dist[t_i].legend()
        ax2_dist[t_i].set_ylim([-0.1, 1.1])
        ax2_dist[t_i].set_xlim([0, 100])
        ax2_dist[t_i].set_xlabel('|Distance|')
        ax2_dist[t_i].set_title('Decoded ' + dig_in_names[t_i])
        ax2_corr[t_i].hist(all_corr_tastes[t_i], bins=1000, histtype='step',
                           density=True, cumulative=True, label=dig_in_names)
        ax2_corr[t_i].legend()
        ax2_corr[t_i].set_ylim([-0.1, 1.1])
        ax2_corr[t_i].set_xlim([-0.1, 1.1])
        ax2_corr[t_i].set_xlabel('Correlation')
        ax2_corr[t_i].set_title('Decoded ' + dig_in_names[t_i])
    ax2_dist[0].set_ylabel('Cumulative Density')
    ax2_corr[0].set_ylabel('Cumulative Density')
    f2_dist.suptitle(title)
    f2_dist.tight_layout()
    f2_dist.savefig(os.path.join(seg_decode_save_dir,
                    save_name + '_population_dist.png'))
    f2_dist.savefig(os.path.join(seg_decode_save_dir,
                    save_name + '_population_dist.svg'))
    plt.close(f2_dist)
    f2_corr.suptitle(title)
    f2_corr.tight_layout()
    f2_corr.savefig(os.path.join(seg_decode_save_dir,
                    save_name + '_population_corr.png'))
    f2_corr.savefig(os.path.join(seg_decode_save_dir,
                    save_name + '_population_corr.svg'))
    plt.close(f2_corr)


def corr_calculator(deliv_fr_vec, decode_fr_mat):
    """
    This function calculates correlations between a matrix of firing rate vectors
    and a single firing rate vector. Note, this function assumes the length of
    the vectors is equivalent (i.e. the number of neurons).

    INPUTS:
            - deliv_fr_vec: single vector of taste delivery firing rate vector
            - decode_fr_mat: matrix with rows of individual decoded event firing 
                    rate vectors
    OUTPUTS:
            - list of correlations of each decoded event to the given delivery event
    """
    # First convert single vector into a matrix
    deliv_fr_mat = deliv_fr_vec*np.ones(np.shape(decode_fr_mat))

    # Calculate the mean-subtracted vectors
    deliv_mean_sub = deliv_fr_mat - \
        np.expand_dims(np.mean(deliv_fr_mat, 1), 1) * \
        np.ones(np.shape(deliv_fr_mat))
    decode_mean_sub = decode_fr_mat - \
        np.expand_dims(np.mean(decode_fr_mat, 1), 1) * \
        np.ones(np.shape(decode_fr_mat))

    # Calculate the squares of the mean-subtracted vectors
    deliv_mean_sub_squared = np.square(deliv_mean_sub)
    decode_mean_sub_squared = np.square(decode_mean_sub)

    # Calculate the numerators and denominators of the pearson's correlation calculation
    pearson_num = np.sum(np.multiply(deliv_mean_sub, decode_mean_sub), 1)
    pearson_denom = np.sqrt(np.sum(deliv_mean_sub_squared, 1)) * \
        np.sqrt(np.sum(decode_mean_sub_squared, 1))

    # Convert to list
    pearson_correlations = list(np.divide(pearson_num, pearson_denom))

    return pearson_correlations


def epoch_progression_plots(num_tastes, dig_in_names, all_epoch_decode_prob,
                            seg_len, plot_len, hatch_types, decode_prob_cutoff,
                            max_decode, segment_spike_times_s_i_bin, num_neur,
                            half_bin, seg_decode_save_dir):
    """This function plots rasters and firing rates with overlaid decoding probabilities
    for a single taste but all epochs. It allows the user to see the progression of
    decoded epochs
    INPUTS:
            - 
    OUTPUTS:
            - 
    """

    for t_i in range(num_tastes):

        taste_save_dir = os.path.join(seg_decode_save_dir, dig_in_names[t_i])
        if not os.path.isdir(taste_save_dir):
            os.mkdir(taste_save_dir)

        taste_name = dig_in_names[t_i]
        taste_probabilities = all_epoch_decode_prob[dig_in_names[t_i]
                                                    ]['Probabilities']
        taste_probabilities_array = np.array(taste_probabilities)
        taste_labels = all_epoch_decode_prob[dig_in_names[t_i]]['Labels']
        plot_starts = np.arange(0, seg_len, np.ceil(plot_len/2).astype('int'))
        num_decode_labels = len(taste_labels)
        decode_colors = cm.gist_rainbow(np.linspace(0, 1, num_decode_labels))
        decode_hatches = hatch_types[:num_decode_labels]

        # Pull decodings in chunks and plot those with the most numbers of labels in the window
        num_labels_per_bin = np.zeros(len(plot_starts))
        for ps_ind, ps in enumerate(plot_starts):
            start = ps
            end = min(ps+plot_len, seg_len)
            for dl in range(len(taste_labels)):
                probability_high = np.where(
                    taste_probabilities_array[dl, start:end] >= decode_prob_cutoff)[0]
                if len(probability_high) > 0:
                    num_labels_per_bin[ps_ind] += 1
        # Only plot the top max_decode number of bins with most labels
        plot_labels = np.argsort(num_labels_per_bin)[-max_decode:]
        for ps_ind in plot_labels:
            start = plot_starts[ps_ind]
            end = min(start+plot_len, seg_len)
            # Collect spike times in interval
            event_spikes = segment_spike_times_s_i_bin[:, start:end]
            decode_spike_times = []
            for n_i in range(num_neur):
                decode_spike_times.append(
                    list(np.where(event_spikes[n_i, :])[0]))
            # Collect firing rates in interval
            event_spikes_expand = segment_spike_times_s_i_bin[:,
                                                              start-10:end+10]
            event_spikes_expand_count = np.sum(event_spikes_expand, 0)
            firing_rate_vec = np.zeros(plot_len)
            for dpt_i in np.arange(10, plot_len+10):
                firing_rate_vec[dpt_i-10] = np.sum(
                    event_spikes_expand_count[dpt_i-10:dpt_i+10])/(20/1000)/num_neur
            # Prepare x ticks and labels
            x_ticks = np.linspace(0, plot_len, 5).astype('int')
            x_labels = np.round(np.linspace(start, start+plot_len, 5)/1000, 2)
            # Plot raster with decoder probabilities overlaid
            f, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
            ax[0].eventplot(decode_spike_times, color='k')
            ax[0].set_xlim([0, plot_len])
            ax[0].set_xticks(x_ticks, labels=x_labels)
            ax[0].set_ylabel('Neuron Index', color='k')
            # Now overlay the decoded probabilities
            ax2 = ax[0].twinx()
            ax2.set_ylabel('Decoding Probability', color='purple')
            ax2.set_ylim([0, 1])
            for dl in range(len(taste_labels)):
                plot_probability = taste_probabilities_array[dl, start:end]
                probability_high = np.where(
                    plot_probability >= decode_prob_cutoff)[0]
                plot_probability_fill = np.zeros(plot_len)
                for ph in probability_high:
                    min_ph = max(ph-half_bin, 0)
                    max_ph = min(ph+half_bin, seg_len)
                    plot_probability_fill[min_ph:max_ph] = taste_probabilities_array[dl, start+ph]
                # ax2.plot(np.arange(plot_len),plot_probability,color=decode_colors[dl,:],alpha=1/num_decode_labels,label='_')
                if decode_hatches[dl] == '':
                    face_color = decode_colors[dl, :]
                else:
                    face_color = 'none'
                ax2.fill_between(np.arange(plot_len), plot_probability_fill,
                                 edgecolor=decode_colors[dl,
                                                         :], color=decode_colors[dl, :],
                                 facecolor=face_color, alpha=1/num_decode_labels, hatch=decode_hatches[dl],
                                 linewidth=2, label=taste_labels[dl])
            ax2.legend()
            ax[0].set_xlabel('Time (s)')
            ax[0].set_title('Taste Decoding ' + taste_name)
            # Plot firing rates below
            ax[1].plot(np.arange(plot_len), firing_rate_vec,
                       color='k', linewidth=2)
            ax[1].set_xticks(x_ticks, labels=x_labels)
            ax[1].set_ylabel('Population Firing Rate (Hz)')
            ax[1].set_xlabel('Time (min)')
            ax[1].set_xlim([0, plot_len])
            # Overlay decodes on firing rates
            ax3 = ax[1].twinx()
            ax3.set_ylabel('Decoding Probability', color='purple')
            ax3.set_ylim([0, 1])
            for dl in range(len(taste_labels)):
                plot_probability = taste_probabilities_array[dl, start:end]
                probability_high = np.where(
                    plot_probability >= decode_prob_cutoff)[0]
                plot_probability_fill = np.zeros(plot_len)
                for ph in probability_high:
                    min_ph = max(ph-half_bin, 0)
                    max_ph = min(ph+half_bin, seg_len)
                    plot_probability_fill[min_ph:max_ph] = taste_probabilities_array[dl, start+ph]
                # ax3.plot(np.arange(plot_len),plot_probability,color=decode_colors[dl,:],alpha=1/num_decode_labels,label='_')
                if decode_hatches[dl] == '':
                    face_color = decode_colors[dl, :]
                else:
                    face_color = 'none'
                ax3.fill_between(np.arange(plot_len), plot_probability_fill,
                                 edgecolor=decode_colors[dl,
                                                         :], color=decode_colors[dl, :],
                                 facecolor=face_color, alpha=1/num_decode_labels, hatch=decode_hatches[dl],
                                 linewidth=2, label=taste_labels[dl])
            # Clean up and save
            plt.tight_layout()
            f.savefig(os.path.join(taste_save_dir,
                      'decodes_' + str(ps_ind) + '.png'))
            f.savefig(os.path.join(taste_save_dir,
                      'decodes_' + str(ps_ind) + '.svg'))
            plt.close(f)


def taste_progression_plots(seg_len, plot_len, num_tastes, hatch_types, all_epoch_decode_prob,
                            dig_in_names, decode_prob_cutoff, max_decode, epochs_to_analyze,
                            segment_spike_times_s_i_bin, num_neur, half_bin, seg_decode_save_dir):
    """This function plots rasters and firing rates with overlaid decoding 
    probabilities for all tastes. Epochs are collapsed and the highest decoding
    probability across epochs is kept as the decoding probability of the taste
    at that time. The highest decoded taste for each time bin is plotted.
    It allows the user to see the progression of tastes.
    INPUTS:
            - 
    OUTPUTS:
            - 

    """

    plot_starts = np.arange(0, seg_len, np.ceil(plot_len/2).astype('int'))
    decode_colors = cm.gist_rainbow(np.linspace(0, 1, num_tastes))
    decode_hatches = hatch_types[:num_tastes]
    # Collapse across epochs taste probabilities
    for e_i in epochs_to_analyze:

        epoch_save_dir = os.path.join(seg_decode_save_dir, 'epoch_' + str(e_i))
        if not os.path.isdir(epoch_save_dir):
            os.mkdir(epoch_save_dir)

        collapsed_taste_probabilities = []
        for t_i in range(num_tastes):
            taste_probabilities = np.array(
                all_epoch_decode_prob[dig_in_names[t_i]]['Probabilities'])[e_i, :]
            collapsed_taste_probabilities.append(taste_probabilities)
        collapsed_taste_probabilities = np.array(collapsed_taste_probabilities)
        # Clean up probabilities to keep maximal probability for each time bin only
        for b_i in tqdm.tqdm(range(seg_len)):
            max_taste = np.argmax(collapsed_taste_probabilities[:, b_i])
            other_taste = np.setdiff1d(
                np.arange(num_tastes), np.array([max_taste]))
            collapsed_taste_probabilities[other_taste, b_i] = 0
        # Pull decodings in chunks and plot those with the most numbers of labels in the window
        num_labels_per_bin = np.zeros(len(plot_starts))
        for ps_ind, ps in enumerate(plot_starts):
            start = ps
            end = min(ps+plot_len, seg_len)
            for t_i in range(num_tastes):
                probability_high = np.where(
                    collapsed_taste_probabilities[t_i, start:end] >= decode_prob_cutoff)[0]
                if len(probability_high) > 0:
                    num_labels_per_bin[ps_ind] += 1
        # Only plot the top max_decode number of bins with most labels
        plot_labels = np.argsort(num_labels_per_bin)[-max_decode:]
        for ps_ind in plot_labels:
            if ps_ind > 10:
                start = plot_starts[ps_ind]
                end = min(start+plot_len, seg_len)
                # Collect spike times in interval
                event_spikes = segment_spike_times_s_i_bin[:, start:end]
                decode_spike_times = []
                for n_i in range(num_neur):
                    decode_spike_times.append(
                        list(np.where(event_spikes[n_i, :])[0]))
                # Collect firing rates in interval
                event_spikes_expand = segment_spike_times_s_i_bin[:,
                                                                  start-10:end+10]
                event_spikes_expand_count = np.sum(event_spikes_expand, 0)
                firing_rate_vec = np.zeros(plot_len)
                for dpt_i in np.arange(10, plot_len+10):
                    firing_rate_vec[dpt_i-10] = np.sum(
                        event_spikes_expand_count[dpt_i-10:dpt_i+10])/(20/1000)/num_neur
                # Prepare x ticks and labels
                x_ticks = np.linspace(0, plot_len, 5).astype('int')
                x_labels = np.round(np.linspace(
                    start, start+plot_len, 5)/1000, 2)
                # Plot raster with decoder probabilities overlaid
                f, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
                ax[0].eventplot(decode_spike_times, color='k')
                ax[0].set_xlim([0, plot_len])
                ax[0].set_xticks(x_ticks, labels=x_labels)
                ax[0].set_ylabel('Neuron Index', color='k')
                # Now overlay the decoded probabilities
                ax2 = ax[0].twinx()
                ax2.set_ylabel('Decoding Probability', color='purple')
                ax2.set_ylim([0, 1])
                for t_i in range(num_tastes-1):
                    plot_probability = collapsed_taste_probabilities[t_i, start:end]
                    probability_high = np.where(
                        plot_probability >= decode_prob_cutoff)[0]
                    plot_probability_fill = np.zeros(plot_len)
                    for ph in probability_high:
                        min_ph = max(ph-half_bin, 0)
                        max_ph = min(ph+half_bin, seg_len)
                        plot_probability_fill[min_ph:max_ph] = collapsed_taste_probabilities[t_i, start+ph]
                    # ax2.plot(np.arange(plot_len),plot_probability,color=decode_colors[dl,:],alpha=1/num_decode_labels,label='_')
                    if decode_hatches[t_i] == '':
                        face_color = decode_colors[t_i, :]
                    else:
                        face_color = 'none'
                    ax2.fill_between(np.arange(plot_len), plot_probability_fill,
                                     edgecolor=decode_colors[t_i,
                                                             :], color=decode_colors[t_i, :],
                                     facecolor=face_color, alpha=1/num_tastes, hatch=decode_hatches[t_i],
                                     linewidth=2, label=dig_in_names[t_i])
                ax2.legend()
                ax[0].set_xlabel('Time (s)')
                ax[0].set_title('Taste Decoding')
                # Plot firing rates below
                ax[1].plot(np.arange(plot_len), firing_rate_vec,
                           color='k', linewidth=2)
                ax[1].set_xticks(x_ticks, labels=x_labels)
                ax[1].set_ylabel('Population Firing Rate (Hz)')
                ax[1].set_xlabel('Time (min)')
                ax[1].set_xlim([0, plot_len])
                # Now overlay the decoded probabilities
                ax3 = ax[1].twinx()
                ax3.set_ylabel('Decoding Probability', color='purple')
                ax3.set_ylim([0, 1])
                for t_i in range(num_tastes-1):
                    plot_probability = collapsed_taste_probabilities[t_i, start:end]
                    probability_high = np.where(
                        plot_probability >= decode_prob_cutoff)[0]
                    plot_probability_fill = np.zeros(plot_len)
                    for ph in probability_high:
                        min_ph = max(ph-half_bin, 0)
                        max_ph = min(ph+half_bin, seg_len)
                        plot_probability_fill[min_ph:max_ph] = collapsed_taste_probabilities[t_i, start+ph]
                    # ax2.plot(np.arange(plot_len),plot_probability,color=decode_colors[dl,:],alpha=1/num_decode_labels,label='_')
                    if decode_hatches[t_i] == '':
                        face_color = decode_colors[t_i, :]
                    else:
                        face_color = 'none'
                    ax3.fill_between(np.arange(plot_len), plot_probability_fill,
                                     edgecolor=decode_colors[t_i,
                                                             :], color=decode_colors[t_i, :],
                                     facecolor=face_color, alpha=1/num_tastes, hatch=decode_hatches[t_i],
                                     linewidth=2, label=dig_in_names[t_i])
                # Clean up and save
                plt.tight_layout()
                f.savefig(os.path.join(epoch_save_dir,
                          'all_tastes_' + str(ps_ind) + '.png'))
                f.savefig(os.path.join(epoch_save_dir,
                          'all_tastes_' + str(ps_ind) + '.svg'))
                plt.close(f)
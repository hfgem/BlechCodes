# Import stuff!
import sys
import os
import csv

current_path = os.path.realpath(__file__)
os.chdir(('/').join(current_path.split('/')[:-1]))

import tables
import itertools
import easygui
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind, ks_2samp
import functions.load_intan_rhd_format.load_intan_rhd_format as rhd
from functions.blech_held_units_funcs import *
    
# Time to use before and after a taste is delivered to compare response curves
pre_deliv_time = 500 #ms
post_deliv_time = 1000 #ms
binning = 100 #ms
bin_starts = np.arange(-1*pre_deliv_time,post_deliv_time,binning)
pre_deliv_inds = np.where(bin_starts<0)[0]

# Ask the user for the number of days to be compared
num_days = int_input("How many days-worth of data are you comparing for held units (integer)? ")

# Ask the user for the percentile criterion to use to determine held units
percent_criterion = int_input('What percentile of intra-J3 do you want to use to pull out held units (provide an integer)? ')
percent_criterion_fr = int_input('What percentile of FR distances do you want to use to pull out held units (provide an integer)? ')

# Ask the user for the waveform to use to determine held units
while_end = 0
while while_end == 0:
    wf_ind = int_input('Which types of waveforms should be used for held_unit analysis?' + \
                         '\n1: raw_CAR_waveform'
                         '\n2: norm_waveform' + '\nEnter the index: ')
    if wf_ind == 1:
        wf_type = 'raw_CAR_waveform'
        while_end = 1
    elif wf_ind == 2:
        wf_type = 'norm_waveform'
        while_end = 1
    else:
        print('Error: Incorrect entry, try again.')

# Ask the user for the analysis to use to determine held units
# while_end = 0
# while while_end == 0:
#     analysis_ind = int_input('Which type of analysis should be used?' + \
#                          '\n1: J3'
#                          '\n2: Euclidean' + '\nEnter the index: ')
#     if analysis_ind == 1:
#         analysis_type = 'J3'
#         while_end = 1
#     elif analysis_ind == 2:
#         analysis_type = 'Euclidean'
#         while_end = 1
#     else:
#         print('Error: Incorrect entry, try again.')

data_dict = dict() #Store all the different days' data in a dictionary
all_neur_inds = [] #Store all neuron indices to calculate cross-day combinations
all_intra_J3 = [] #Store all intra J3 data to calculate cutoff for inter-J3
all_intra_euc_dist = [] #Store all intra euclidean distances for firing rate curves
for n_i in range(num_days):
    # data_dict[n_i] = dict()
    # #Ask for directory of the dataset hdf5 file
    # print('Where is the hdf5 file from the ' + str(n_i + 1) + ' day?')
    # dir_name = askdirectory()
    # data_dict[n_i]['dir_name'] = dir_name
    
    dir_name = data_dict[n_i]['dir_name']
    #Find hdf5 in directory
    file_list = os.listdir(dir_name)
    hdf5_name = ''
    for files in file_list:
        if files[-2:] == 'h5':
            hdf5_name = files
    data_dict[n_i]['hdf5_name'] = hdf5_name
    #Open hdf5 file
    hf5 = tables.open_file(os.path.join(dir_name,hdf5_name), 'r')
    num_neur = len(hf5.root.unit_descriptor[:])
    all_neur_inds.append(list(np.arange(num_neur)))
    data_dict[n_i]['num_neur'] = num_neur
    #Check if dig in times saved as csv in directory or not
    start_dig_in_times_csv = os.path.join(dir_name,'start_dig_in_times.csv')
    if os.path.isfile(start_dig_in_times_csv):
        print("\t\tImporting previously saved digital input times")
        start_dig_in_times = []
        with open(start_dig_in_times_csv, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                start_dig_in_times.append(list(np.array(row).astype('int')))
        num_tastes = len(start_dig_in_times)
    else:
        dig_in_node = blech_clust_h5.list_nodes('/digital_in')
        dig_in_indices = np.array([d_i.name.split('_')[-1] for d_i in dig_in_node])
        dig_in_ind = []
        i = 0
        for d_i in dig_in_indices:
            try:
                int(d_i)
                dig_in_ind.extend([i])
            except:
                "not an input - do nothing"
            i += 1
        del dig_in_indices
        try:
            if len(dig_in_node[0][0]):
                dig_in_data = [list(dig_in_node[d_i][0][:]) for d_i in dig_in_ind]
        except:
            dig_in_data = [list(dig_in_node[d_i][:]) for d_i in dig_in_ind]
        num_tastes = len(dig_in_data)
        del dig_in_node
        #_____Convert dig_in_data to indices of dig_in start and end times_____
        print("\tConverting digital inputs to free memory")
        #Again, all are converted to ms timescale
        start_dig_in_times = [list(np.ceil((np.where(np.diff(np.array(dig_in_data[i])) == 1)[0] + 1)*self.ms_conversion).astype('int')) for i in range(num_tastes)]    
        #Store these into csv for import in future instead of full dig_in_data load which takes forever!
        with open(start_dig_in_times_csv, 'w') as f:
            write = csv.writer(f,delimiter=',')
            write.writerows(start_dig_in_times)
    flat_start_dig_in_times = []
    for d_i in range(len(start_dig_in_times)):
        flat_start_dig_in_times.extend(start_dig_in_times[d_i])
    start_dig_in_times = np.sort(np.array(flat_start_dig_in_times)) #All dig in times - regardless of taste
    data_dict[n_i]['start_dig_in_times'] = start_dig_in_times
    taste_start_time = np.max([np.min(start_dig_in_times) - 5000,0]).astype('int')
    taste_end_time = (np.max(start_dig_in_times) + 5000).astype('int')
    data_dict[n_i]['taste_interval'] = [taste_start_time,taste_end_time]
    #Grab sampling rate for time conversion
    try:
        sampling_rate = hf5.root.sampling_rate[0]
    except:
        #The old method doesn't currently store sampling_rate, so this picks it up
        rhd_dict = rhd.import_data(dir_name)
        sampling_rate = int(rhd_dict["frequency_parameters"]["amplifier_sample_rate"])
        atom = tables.IntAtom()
        hf5.create_earray('/','sampling_rate',atom,(0,))
        hf5.root.sampling_rate.append([sampling_rate])
    ms_conversion = (1/sampling_rate)*(1000/1) #ms/samples units
    
    #Calculate the Intra-J3/Euclidean Distance data for the units
    intra_J3 = []
    # euc_dist = []
    all_unit_waveforms = []
    all_unit_waveform_peaks = []
    all_unit_times = []
    all_unit_pca = []
    all_unit_fr_curves = []
    intra_euc_dist_fr_curves = []
    for unit in tqdm.tqdm(range(num_neur)):
        exec("wf_day1 = hf5.root.sorted_units.unit%03d.waveforms[:]" % (unit)) #num wav x 60
        exec("t_day1 = hf5.root.sorted_units.unit%03d.times[:]" % (unit)) #num wav
        wf_peak_ind = np.ceil(wf_day1.shape[1]/2).astype('int')
        if wf_type == 'norm_waveform':
            wf_day1 = wf_day1 / np.std(wf_day1)
        wf_peaks = wf_day1[:,wf_peak_ind].flatten()
        all_unit_waveform_peaks.append(wf_peaks)
        t_day1 = t_day1*ms_conversion
        #Pull out taste response interval waveforms only
        taste_wf_inds = np.where((t_day1 >= taste_start_time)*(t_day1 <= taste_end_time))[0]
        taste_t_day1 = t_day1[taste_wf_inds]
        all_unit_times.append(taste_t_day1)
        taste_wf_day1 = wf_day1[taste_wf_inds,:]
        all_unit_waveforms.append(taste_wf_day1)
        pca = PCA(n_components = 4)
        pca.fit(taste_wf_day1)
        pca_wf_day1 = pca.transform(taste_wf_day1) #num wav x 4
        all_unit_pca.append(pca_wf_day1)
        intra_J3.append(calculate_J3(pca_wf_day1[:int(taste_wf_day1.shape[0]*(1.0/3.0)), :], 
                                     pca_wf_day1[int(taste_wf_day1.shape[0]*(2.0/3.0)):, :]))
        #Calculate firing rates within given interval before and following taste delivery
        neur_fr = np.zeros((len(start_dig_in_times),len(bin_starts)))
        for st_ind, st_i in enumerate(start_dig_in_times):
            for bst_ind, bst_i in enumerate(bin_starts):
                spike_hz = len(np.where((taste_t_day1>=st_i+bst_i)*(taste_t_day1<=st_i+bst_i+binning))[0])/(binning/1000)
                neur_fr[st_ind,bst_ind] = spike_hz
        pre_deliv_mean = np.nanmean(neur_fr[:,pre_deliv_inds])
        neur_fr_rescale = (neur_fr - pre_deliv_mean*np.ones(np.shape(neur_fr)))/(pre_deliv_mean*np.ones(np.shape(neur_fr)))
        all_unit_fr_curves.append(np.nanmean(neur_fr_rescale,0))
        neur_fr_rescale_mean1 = np.nanmean(neur_fr_rescale[:int(len(start_dig_in_times)*(1.0/3.0)), :],0)
        neur_fr_rescale_mean2 = np.nanmean(neur_fr_rescale[int(len(start_dig_in_times)*(2.0/3.0)):, :],0)
        # pca_fr = PCA(n_components = 4)
        # pca_fr.fit(neur_fr)
        # pca_fr_day1 = pca_fr.transform(neur_fr) #num wav x 4
        intra_euc_dist_fr_curves.append(np.sqrt(np.sum(np.square(np.abs(neur_fr_rescale_mean1 - neur_fr_rescale_mean2)))))
        #Euclidean calcs
        # day_euc_dist = calculate_euc_dist(pca_wf_day1[:int(taste_wf_day1.shape[0]*(1.0/3.0)), :], 
        #                              pca_wf_day1[int(taste_wf_day1.shape[0]*(2.0/3.0)):, :])
        # euc_dist.append(calculate_euc_dist(pca_wf_day1[:int(taste_wf_day1.shape[0]*(1.0/3.0)), :], 
        #                              pca_wf_day1[int(taste_wf_day1.shape[0]*(2.0/3.0)):, :]))
    data_dict[n_i]['intra_J3'] = intra_J3
    # data_dict[n_i]['euc_dist'] = euc_dist
    data_dict[n_i]['all_unit_waveforms'] = all_unit_waveforms
    data_dict[n_i]['all_unit_waveform_peaks'] = all_unit_waveform_peaks
    data_dict[n_i]['all_unit_times'] = all_unit_times
    data_dict[n_i]['all_unit_pca'] = all_unit_pca
    data_dict[n_i]['all_unit_fr_curves'] = all_unit_fr_curves
    data_dict[n_i]['intra_euc_dist_fr_curves'] = intra_euc_dist_fr_curves
    all_intra_J3.extend(intra_J3)
    all_intra_euc_dist.extend(intra_euc_dist_fr_curves)
    #Pull unit info for all units
    all_unit_info = []
    for unit in range(num_neur):
        all_unit_info.append(get_unit_info(hf5.root.unit_descriptor[unit]))
    data_dict[n_i]['all_unit_info'] = all_unit_info
    #Close hdf5 file
    hf5.close()
    
# Ask the user for the output directory to save the held units and plots in
print('Where do you want to save the held units and plots?')
save_dir = askdirectory()

#Save the data dictionary just in case want in future
np.save(os.path.join(save_dir,'data_dict.npy'),data_dict,allow_pickle=True)
#%%

#Calculate the intra-J3 percentile cutoff
all_intra_J3_cutoff = np.percentile(all_intra_J3, percent_criterion)

#Calculate the euclidean distance cutoff
all_euc_dist_cutoff = np.percentile(all_intra_euc_dist, percent_criterion_fr)

held_unit_storage = [] #placeholder storage for held units across days

#Calculate all pairwise unit tests
all_neur_combos = list(itertools.product(*all_neur_inds))
all_day_combos = list(itertools.combinations(np.arange(num_days),2))

header = ''
for day in range(num_days-1):
    header += 'Day ' + str(day+1) + ','
header += 'Day ' + str(num_days)
    
# Make a file to save the numbers of the units that are deemed to have been held across days
with open(os.path.join(save_dir,f'held_units_{wf_type}_{percent_criterion}.csv'), 'w') as f:
    f.write(header)


all_inter_J3 = []
all_inter_euc_dist = []
held_index_counter = 0
shape = tuple([len(all_neur_inds[i]) for i in range(len(all_neur_inds))])
shape2 = tuple(np.concatenate((np.array(shape),len(all_day_combos)*np.ones(1))).astype('int'))
all_neur_combo_vals = np.zeros(shape2)
avg_neur_combo_val = np.zeros(shape)
viable_neur_combo = np.zeros(shape)
for nc in all_neur_combos:
    #Collect waveforms and pca data to be compared
    waveform_peaks = [] #list of numpy arrays
    waveforms_pca = []
    all_waveforms_pca = []
    fr_curves = []
    
    for day in range(len(nc)):
        wf_peaks = data_dict[day]['all_unit_waveform_peaks'][nc[day]]
        waveform_peaks.append(wf_peaks)
        wf_pca = data_dict[day]['all_unit_pca'][nc[day]]
        waveforms_pca.append(wf_pca)
        all_waveforms_pca.extend(wf_pca)
        fr_curves.append(data_dict[day]['all_unit_fr_curves'][nc[day]])
            
    #Do all inter_J3 match the cutoff?
    
    #Calculate the inter_J3 across days
    all_days_inter_J3 = []
    for dc in all_day_combos:
        all_days_inter_J3.extend([calculate_J3(waveforms_pca[dc[0]], waveforms_pca[dc[1]])])
    all_inter_J3.append(all_days_inter_J3) 
    avg_neur_combo_val[nc] = np.nanmean(all_days_inter_J3)
    all_neur_combo_vals[nc,:] = all_days_inter_J3
    
    if np.sum((np.array(all_days_inter_J3) <= all_intra_J3_cutoff).astype('int')) == len(all_day_combos):
        
        #Compare peak distributions via ttest
        ttest_res = np.ones(len(all_day_combos))
        for dp_i, dp in enumerate(all_day_combos):
            _, p_val = ttest_ind(waveform_peaks[dp[0]],waveform_peaks[dp[1]],nan_policy='omit')
            ttest_res[dp_i] = p_val
        
        ttest_check = np.sum((ttest_res >= .05).astype('int')) == len(all_day_combos)
        
        #Compare inter euclidean distance of rescaled firing rate curves to intra cutoff
        all_days_inter_euc_dist = []
        for dc in all_day_combos:
            #all_days_inter_J3.extend([calculate_J3(day_pca[dc[0]], day_pca[dc[1]])])
            day_1_mean = fr_curves[dc[0]]
            day_2_mean = fr_curves[dc[1]]
            all_days_inter_euc_dist.extend([np.sqrt(np.sum(np.square(np.abs(day_1_mean - day_2_mean))))])
          
        fr_check = np.sum((np.array(all_days_inter_euc_dist) <= all_euc_dist_cutoff).astype('int')) == len(all_day_combos)
            
        if ttest_check and fr_check:
        
            viable_neur_combo[nc] = 1
        
            #Save to csv the unit indices per day
            statement = '\n'
            for i in range(len(nc)-1):
                statement += str(nc[i]) + ','
            statement += str(nc[-1])
            with open(os.path.join(save_dir,f'held_units_{wf_type}_{percent_criterion}.csv'), 'a') as f:
                f.write(statement)
            
            #Create a plot of the matching waveforms
            fig, ax = plt.subplots(2, num_days, figsize=(12, 6))
            min_wf_val = 10000
            max_wf_val = -10000
            for d_i in range(num_days):
                #Plot waveforms
                wf_day_i = data_dict[d_i]['all_unit_waveforms'][nc[d_i]]
                num_wav = wf_day_i.shape[0]
                t = np.arange(wf_day_i.shape[1])
                mean_wfs = np.mean(wf_day_i, axis = 0)
                max_, min_ = np.max(mean_wfs), np.min(mean_wfs)
                if min_ < min_wf_val:
                    min_wf_val = min_
                if max_ > max_wf_val:
                    max_wf_val = max_
                ax[0,d_i].plot(t - 15, mean_wfs, linewidth = 5.0, color = 'black')
                ax[0,d_i].plot(t - 15, mean_wfs - np.std(wf_day_i, axis = 0), 
                        linewidth = 2.0, color = 'black', alpha = 0.5)
                ax[0,d_i].plot(t - 15, mean_wfs + np.std(wf_day_i, axis = 0), 
                        linewidth = 2.0, color = 'black', alpha = 0.5)
                ax[0,d_i].axhline(max_, color='r', ls='--')
                ax[0,d_i].axhline(min_, color='r', ls='--')
                ax[0,d_i].set_xlabel('Time (samples (30 per ms))', fontsize = 12)
                ax[0,d_i].set_ylabel('Voltage (microvolts)', fontsize = 12)
                ax[0,d_i].set_title('Unit ' + str(nc[d_i]) + ', total waveforms = ' + str(num_wav) + \
                                  '\nElectrode: ' + str(data_dict[d_i]['all_unit_info'][nc[d_i]][0]) , fontsize = 12)
                #Plot average firing rate curve rescaled
                fr_day_i = fr_curves[d_i]
                ax[1,d_i].plot(bin_starts,fr_day_i)
                ax[1,d_i].set_xlabel('Time from Delivery (ms)')
                ax[1,d_i].set_ylabel('Rescaled Binned Average Firing Rate')
            for d_i in range(num_days):
                ax[0,d_i].set_ylim([min_wf_val - 5, max_wf_val + 5])
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir,'held_index_' + str(held_index_counter) + '.png'), bbox_inches = 'tight')
            plt.close(fig)
            
            held_index_counter += 1
        
    # elif analysis_type == 'Euclidean':
    #     #Calculate the average inter_euc_dist across days
    #     all_days_inter_euc_dist = []
    #     for dc in all_day_combos:
    #         dc_euc_dist = calculate_euc_dist(waveforms_pca[dc[0]], waveforms_pca[dc[1]])
    #         all_days_inter_euc_dist.extend([dc_euc_dist])
    #     all_inter_euc_dist.append(all_days_inter_euc_dist)
    #     avg_neur_combo_val[nc] = np.nanmean(all_days_inter_euc_dist)
    #     all_neur_combo_vals[nc,:] = all_days_inter_euc_dist
        
    #     #Do all euc distances match the cutoff?
    #     if np.sum((np.array(all_days_inter_euc_dist) <= all_euc_dist_cutoff).astype('int')) == len(all_day_combos):
            
    #         #Compare peak distributions via ttest
    #         ttest_res = np.ones(len(all_day_combos))
    #         for dp_i, dp in enumerate(all_day_combos):
    #             _, p_val = ttest_ind(waveform_peaks[dp[0]],waveform_peaks[dp[1]],nan_policy='omit')
    #             ttest_res[dp_i] = p_val
                
    #         if np.sum((ttest_res >= .05).astype('int')) == len(all_day_combos):
    #             viable_neur_combo[nc] = 1
            
    #             #Save to csv the viable unit indices per day
    #             statement = '\n'
    #             for i in range(len(nc)-1):
    #                 statement += str(nc[i]) + ','
    #             statement += str(nc[-1])
    #             with open(os.path.join(save_dir,f'held_units_{wf_type}_{percent_criterion}.csv'), 'a') as f:
    #                 f.write(statement)
                
    #             #Create a plot of the matching waveforms
    #             fig, ax = plt.subplots(1, num_days, sharex=True, sharey=True, figsize=(12, 6))
    #             min_wf_val = 10000
    #             max_wf_val = -10000
    #             for d_i in range(num_days):
    #                 wf_day_i = data_dict[d_i]['all_unit_waveforms'][nc[d_i]]
    #                 num_wav = wf_day_i.shape[0]
    #                 t = np.arange(wf_day_i.shape[1])
    #                 mean_wfs = np.mean(wf_day_i, axis = 0)
    #                 max_, min_ = np.max(mean_wfs), np.min(mean_wfs)
    #                 if min_ < min_wf_val:
    #                     min_wf_val = min_
    #                 if max_ > max_wf_val:
    #                     max_wf_val = max_
    #                 ax[d_i].plot(t - 15, mean_wfs, linewidth = 5.0, color = 'black')
    #                 ax[d_i].plot(t - 15, mean_wfs - np.std(wf_day_i, axis = 0), 
    #                         linewidth = 2.0, color = 'black', alpha = 0.5)
    #                 ax[d_i].plot(t - 15, mean_wfs + np.std(wf_day_i, axis = 0), 
    #                         linewidth = 2.0, color = 'black', alpha = 0.5)
    #                 ax[d_i].axhline(max_, color='r', ls='--')
    #                 ax[d_i].axhline(min_, color='r', ls='--')
    #                 ax[d_i].set_xlabel('Time (samples (30 per ms))', fontsize = 12)
    #                 ax[d_i].set_ylabel('Voltage (microvolts)', fontsize = 12)
    #                 ax[d_i].set_title('Unit ' + str(nc[d_i]) + ', total waveforms = ' + str(num_wav) + \
    #                                   '\nElectrode: ' + str(data_dict[d_i]['all_unit_info'][nc[d_i]][0]) , fontsize = 12)
    #             for d_i in range(num_days):
    #                 ax[d_i].set_ylim([min_wf_val - 20, max_wf_val + 20])
    #             plt.tight_layout()
    #             fig.savefig(os.path.join(save_dir,'held_index_' + str(held_index_counter) + '.png'), bbox_inches = 'tight')
    #             plt.close(fig)
                
    #             held_index_counter += 1

# if analysis_type == 'J3':
    # Plot the intra and inter J3 in a different file
fig = plt.figure()
plt.hist(np.array(all_inter_J3).flatten(), bins = 20, alpha = 0.3, label = 'Across-session J3')
plt.hist(np.array(all_intra_J3).flatten(), bins = 20, alpha = 0.3, label = 'Within-session J3')
# Draw a vertical line at the percentile criterion used to choose held units
plt.axvline(all_intra_J3_cutoff, linewidth = 5.0, color = 'black', linestyle = 'dashed', label='J3 Cutoff')
plt.legend(loc='upper left')
plt.xlabel('J3', fontsize = 12)
plt.ylabel('Number of single unit pairs', fontsize = 12)
#plt.tick_params(axis='both', which='major', labelsize=32)
fig.savefig(os.path.join(save_dir,'J3_distributions_{wf_type}.png'), bbox_inches = 'tight')
plt.close(fig)  
# elif analysis_type == 'Euclidean':
#     # Plot the intra and inter euclidean distances in a different file
#     fig = plt.figure()
#     plt.hist(np.array(all_inter_euc_dist).flatten(), bins = 20, alpha = 0.3, label = 'Across-session Average Distances')
#     plt.hist(np.array(all_intra_euc_dist).flatten(), bins = 20, alpha = 0.3, label = 'Within-session Average Distances')
#     # Draw a vertical line at the percentile criterion used to choose held units
#     plt.axvline(all_euc_dist_cutoff, linewidth = 5.0, color = 'black', linestyle = 'dashed', label='J3 Cutoff')
#     plt.legend(loc='upper left')
#     plt.xlabel('J3', fontsize = 12)
#     plt.ylabel('Number of single unit pairs', fontsize = 12)
#     #plt.tick_params(axis='both', which='major', labelsize=32)
#     fig.savefig(os.path.join(save_dir,f'Euclidean_distributions_{wf_type}.png'), bbox_inches = 'tight')
#     plt.close(fig)
    
viable_where = np.where(viable_neur_combo)
viable_indices = np.array(viable_where)
avg_vals_viable = avg_neur_combo_val[viable_where]
day_1_neur = np.unique(viable_indices[0,:])
for n_1_i in day_1_neur:
    day_1_inds = np.where(viable_indices[0,:] == n_1_i)[0]
    other_inds = viable_indices[1:,day_1_inds]
    metric_avgs = avg_vals_viable[day_1_inds]
    

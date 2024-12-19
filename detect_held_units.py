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
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import ttest_ind, ks_2samp
import functions.load_intan_rhd_format.load_intan_rhd_format as rhd
from functions.blech_held_units_funcs import *
    
# Time to use before and after a taste is delivered to compare response curves
pca_components = 6
pre_deliv_time = 500 #ms
post_deliv_time = 1000 #ms
binning = 100 #ms
bin_starts = np.arange(-1*pre_deliv_time,post_deliv_time-binning)
pre_deliv_inds = np.where(bin_starts<0)[0]

num_days, use_electrode = user_held_unit_input()

file_exists = bool_input("Do you have a previously compiled held unit pickle file you'd like to use? ")

all_neur_inds = [] #Store all neuron indices to calculate cross-day combinations
all_electrode_inds = [] #Store the electrode indices for each neuron

if file_exists == 'y':
    print('Where did you save the held units pickle file from a prior run?')
    save_dir = askdirectory()
    
    stat_save_dir = os.path.join(save_dir,'statistic_plots')
    if not os.path.isdir(stat_save_dir):
        os.mkdir(stat_save_dir)
    
    #Load the data dictionary
    data_dict = np.load(os.path.join(save_dir,'data_dict.npy'),allow_pickle=True).item()
    
    #Check if previously saved intra calcs
    try:
        all_intra_J3 = np.load(os.path.join(save_dir,'all_intra_J3.npy'),allow_pickle=True)
        all_intra_J3_norm = np.load(os.path.join(save_dir,'all_intra_J3_norm.npy'),allow_pickle=True)
        all_intra_euc_dist = np.load(os.path.join(save_dir,'all_intra_euc_dist.npy'),allow_pickle=True)
        all_intra_euc_dist_norm = np.load(os.path.join(save_dir,'all_intra_euc_dist_norm.npy'),allow_pickle=True)
        num_days = len(data_dict)
        for d_i in range(num_days):
            num_neur = data_dict[d_i]['num_neur']
            all_neur_inds.append(list(np.arange(num_neur)))
            all_unit_info = data_dict[d_i]['all_unit_info']
            all_electrode_inds.append(list(np.squeeze(np.array(all_unit_info)[:,0])))
    except:
        try:
            all_intra_J3 = [] #Store all intra J3 data for original waveforms
            all_intra_J3_norm = [] #Store all intra J3 data for normalized waveforms
            all_intra_euc_dist = [] #Store all intra euclidean distances for firing rate curves
            all_intra_euc_dist_norm = [] #Store all intra euclidean distances for normalized firing rate curves
            num_days = len(data_dict)
            for d_i in range(num_days):
                num_neur = data_dict[d_i]['num_neur']
                all_neur_inds.append(list(np.arange(num_neur)))
                all_unit_info = data_dict[d_i]['all_unit_info']
                all_electrode_inds.append(list(np.squeeze(np.array(all_unit_info)[:,0])))
                start_dig_in_times = data_dict[d_i]['start_dig_in_times']
                [taste_start_time,taste_end_time] = data_dict[d_i]['taste_interval']
                intra_J3 = data_dict[d_i]['intra_J3']
                intra_J3_norm = data_dict[d_i]['intra_J3_norm']
                intra_euc_dist_fr_curves = data_dict[d_i]['intra_euc_dist_fr_curves']
                intra_euc_dist_fr_curves_norm = data_dict[d_i]['intra_euc_dist_fr_curves_norm']
                all_intra_J3.extend(intra_J3)
                all_intra_J3_norm.extend(intra_J3_norm)
                all_intra_euc_dist.extend(intra_euc_dist_fr_curves)
                all_intra_euc_dist_norm.extend(intra_euc_dist_fr_curves_norm)
            del d_i, num_neur, start_dig_in_times, taste_start_time, taste_end_time
            #Save the calculated intra datasets
            np.save(os.path.join(stat_save_dir,'all_intra_J3.npy'),np.array(all_intra_J3),allow_pickle=True)
            np.save(os.path.join(stat_save_dir,'all_intra_J3_norm.npy'),np.array(all_intra_J3_norm),allow_pickle=True)
            np.save(os.path.join(stat_save_dir,'all_intra_euc_dist.npy'),np.array(all_intra_euc_dist),allow_pickle=True)
            np.save(os.path.join(stat_save_dir,'all_intra_euc_dist_norm.npy'),np.array(all_intra_euc_dist_norm),allow_pickle=True) 
        except:
            print("ERROR: Start fresh and delete the old pickle file because it's not working.")
            quit()
    
else:
    all_intra_J3 = [] #Store all intra J3 data for original waveforms
    all_intra_J3_norm = [] #Store all intra J3 data for normalized waveforms
    all_intra_euc_dist = [] #Store all intra euclidean distances for firing rate curves
    all_intra_euc_dist_norm = [] #Store all intra euclidean distances for normalized firing rate curves
    data_dict = dict() #Store all the different days' data in a dictionary
    for d_i in range(num_days):
        data_dict[d_i] = dict()
        #Ask for directory of the dataset hdf5 file
        print('Where is the hdf5 file from the ' + str(d_i + 1) + ' day?')
        dir_name = askdirectory()
        data_dict[d_i]['dir_name'] = dir_name
        #Find hdf5 in directory
        file_list = os.listdir(dir_name)
        hdf5_name = ''
        for files in file_list:
            if files[-2:] == 'h5':
                hdf5_name = files
        data_dict[d_i]['hdf5_name'] = hdf5_name
        del file_list
        #Open hdf5 file
        hf5 = tables.open_file(os.path.join(dir_name,hdf5_name), 'r')
        num_neur = len(hf5.root.unit_descriptor[:])
        all_neur_inds.append(list(np.arange(num_neur).flatten()))
        data_dict[d_i]['num_neur'] = num_neur
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
            print("\t\tImporting digital input times")
            dig_in_node = blech_clust_h5.list_nodes('/digital_in')
            dig_in_indices = np.array([dig_i.name.split('_')[-1] for dig_i in dig_in_node])
            dig_in_ind = []
            i = 0
            for dig_i in dig_in_indices:
                try:
                    int(dig_i)
                    dig_in_ind.extend([i])
                except:
                    "not an input - do nothing"
                i += 1
            del dig_in_indices, i, dig_i
            try:
                if len(dig_in_node[0][0]):
                    dig_in_data = [list(dig_in_node[dig_i][0][:]) for dig_i in dig_in_ind]
            except:
                dig_in_data = [list(dig_in_node[dig_i][:]) for dig_i in dig_in_ind]
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
            del f, write
        #Request user input on which dig ins to use in the PSTH calculations
        print(str(num_tastes) + " tastes (Dig-Ins) have been identified.")
        num_taste_keep = int_input("How many tastes would you like to keep for PSTH comparisons? ")
        keep_inds = []
        for ntk_i in range(num_taste_keep):
            keep_inds.append(int_input("What is the index of taste  " + str(ntk_i + 1) + " (starting with a 0 index)? "))
        #Store digital input times and taste interval information
        flat_start_dig_in_times = []
        for st_i in keep_inds:
            flat_start_dig_in_times.extend(start_dig_in_times[st_i])
        start_dig_in_times = np.sort(np.array(flat_start_dig_in_times)) #All dig in times - regardless of taste
        data_dict[d_i]['start_dig_in_times'] = start_dig_in_times
        taste_start_time = np.max([np.min(start_dig_in_times) - 5000,0]).astype('int')
        taste_end_time = (np.max(start_dig_in_times) + 5000).astype('int')
        data_dict[d_i]['taste_interval'] = [taste_start_time,taste_end_time]
        del keep_inds, ntk_i, flat_start_dig_in_times
        #Grab sampling rate for time conversion of spike times to ms
        try:
            sampling_rate = hf5.root.sampling_rate[0]
        except:
            #The old method doesn't currently store sampling_rate, so this picks it up
            rhd_dict = rhd.import_data(dir_name)
            sampling_rate = int(rhd_dict["frequency_parameters"]["amplifier_sample_rate"])
            atom = tables.IntAtom()
            hf5.create_earray('/','sampling_rate',atom,(0,))
            hf5.root.sampling_rate.append([sampling_rate])
            del rhd_dict, atom
        ms_conversion = (1/sampling_rate)*(1000/1) #ms/samples units
        del sampling_rate
        #Calculate the Intra-J3/Euclidean Distance data for the units
        all_unit_waveforms = []
        all_unit_waveform_peaks = []
        all_unit_waveforms_norm = []
        all_unit_waveform_peaks_norm = []
        all_unit_times = []
        intra_J3 = []
        intra_J3_norm = []
        all_unit_fr_curves = [] #average PSTH for neuron
        all_unit_fr_curves_norm = [] #average normalized PSTH
        intra_euc_dist_fr_curves = []
        intra_euc_dist_fr_curves_norm = []
        for unit in tqdm.tqdm(range(num_neur)):
            #Collect and store waveform and time information
            exec("wf_day1 = hf5.root.sorted_units.unit%03d.waveforms[:]" % (unit)) #num wav x 60
            exec("t_day1 = hf5.root.sorted_units.unit%03d.times[:]" % (unit)) #num wav
            wf_peak_ind = np.ceil(wf_day1.shape[1]/2).astype('int')
            all_unit_waveforms.append(wf_day1)
            wf_day1_norm = wf_day1 / np.std(wf_day1)
            all_unit_waveforms_norm.append(wf_day1_norm)
            wf_peaks = wf_day1[:,wf_peak_ind].flatten()
            all_unit_waveform_peaks.append(wf_peaks)
            wf_peaks_norm = wf_day1_norm[:,wf_peak_ind].flatten()
            all_unit_waveform_peaks_norm.append(wf_peaks_norm)
            t_day1 = t_day1*ms_conversion #times converted to ms
            all_unit_times.append(t_day1)
            del wf_peak_ind, wf_peaks, wf_peaks_norm
            #PCA waveforms
            pca = PCA(n_components = pca_components)
            pca.fit(wf_day1)
            pca_wf_day1 = pca.transform(wf_day1) #num wav x 4
            pca_norm = PCA(n_components = pca_components)
            pca_norm.fit(wf_day1_norm)
            pca_wf_day1_norm = pca_norm.transform(wf_day1_norm) #num wav x 4
            #Calculate intra-J3 values
            intra_J3.append(calculate_J3(pca_wf_day1[:int(t_day1.shape[0]*(1.0/3.0)), :], 
                                         pca_wf_day1[int(t_day1.shape[0]*(2.0/3.0)):, :]))
            intra_J3_norm.append(calculate_J3(pca_wf_day1_norm[:int(t_day1.shape[0]*(1.0/3.0)), :], 
                                         pca_wf_day1_norm[int(t_day1.shape[0]*(2.0/3.0)):, :]))
            del pca, pca_wf_day1, pca_norm, pca_wf_day1_norm
            #Pull out taste response interval times only
            taste_wf_inds = np.where((t_day1 >= taste_start_time)*(t_day1 <= taste_end_time))[0]
            taste_t_day1 = t_day1[taste_wf_inds]
            #Calculate firing rates within given interval before and following taste delivery
            neur_fr = np.zeros((len(start_dig_in_times),len(bin_starts)))
            for st_ind, st_i in enumerate(start_dig_in_times):
                for bst_ind, bst_i in enumerate(bin_starts):
                    spike_hz = len(np.where((taste_t_day1>=st_i+bst_i)*(taste_t_day1<=st_i+bst_i+binning))[0])/(binning/1000)
                    neur_fr[st_ind,bst_ind] = spike_hz
            del st_ind, st_i, bst_ind, bst_i, spike_hz
            all_unit_fr_curves.append(np.nanmean(neur_fr,0))
            neur_fr_mean1 = np.nanmean(neur_fr[:int(len(start_dig_in_times)*(1.0/3.0)), :],0)
            neur_fr_mean2 = np.nanmean(neur_fr[int(len(start_dig_in_times)*(2.0/3.0)):, :],0)
            intra_euc_dist_fr_curves.append(euclidean(neur_fr_mean1,neur_fr_mean2))
            pre_deliv_mean = np.nanmean(neur_fr[:,pre_deliv_inds],1)
            neur_fr_rescale = neur_fr - np.expand_dims(pre_deliv_mean,1)*np.ones(np.shape(neur_fr)) #Subtract pre-delivery fr means
            rescale_mean = np.nanmean(neur_fr_rescale,0)
            all_unit_fr_curves_norm.append(rescale_mean/np.max(rescale_mean))
            neur_fr_rescale_mean1 = np.nanmean(neur_fr_rescale[:int(len(start_dig_in_times)*(1.0/3.0)), :],0)
            neur_fr_rescale_mean1 = neur_fr_rescale_mean1/np.max(neur_fr_rescale_mean1)
            neur_fr_rescale_mean2 = np.nanmean(neur_fr_rescale[int(len(start_dig_in_times)*(2.0/3.0)):, :],0)
            neur_fr_rescale_mean2 = neur_fr_rescale_mean2/np.max(neur_fr_rescale_mean2)
            intra_euc_dist_fr_curves_norm.append(euclidean(neur_fr_rescale_mean1,neur_fr_rescale_mean2))
            del neur_fr_mean1, neur_fr_mean2, pre_deliv_mean, neur_fr_rescale, rescale_mean, neur_fr_rescale_mean1, neur_fr_rescale_mean2
        del taste_start_time, taste_end_time, unit, wf_day1, t_day1 
            
        data_dict[d_i]['all_unit_waveforms'] = all_unit_waveforms
        data_dict[d_i]['all_unit_waveform_peaks'] = all_unit_waveform_peaks
        data_dict[d_i]['all_unit_waveforms_norm'] = all_unit_waveforms_norm
        data_dict[d_i]['all_unit_waveform_peaks_norm'] = all_unit_waveform_peaks_norm
        data_dict[d_i]['all_unit_times'] = all_unit_times
        data_dict[d_i]['intra_J3'] = intra_J3
        data_dict[d_i]['intra_J3_norm'] = intra_J3_norm
        data_dict[d_i]['all_unit_fr_curves'] = all_unit_fr_curves
        data_dict[d_i]['all_unit_fr_curves_norm'] = all_unit_fr_curves_norm
        data_dict[d_i]['intra_euc_dist_fr_curves'] = intra_euc_dist_fr_curves
        data_dict[d_i]['intra_euc_dist_fr_curves_norm'] = intra_euc_dist_fr_curves_norm
        all_intra_J3.extend(intra_J3)
        all_intra_J3_norm.extend(intra_J3_norm)
        all_intra_euc_dist.extend(intra_euc_dist_fr_curves)
        all_intra_euc_dist_norm.extend(intra_euc_dist_fr_curves_norm)
        #Pull unit info for all units
        all_unit_info = []
        for unit in range(num_neur):
            all_unit_info.append(get_unit_info(hf5.root.unit_descriptor[unit]))
        all_electrode_inds.append(list(np.squeeze(np.array(all_unit_info)[:,0])))
        data_dict[d_i]['all_unit_info'] = all_unit_info
        del all_unit_info
        #Close hdf5 file
        hf5.close()
        del hdf5_name, num_neur
    del d_i
        
    # Ask the user for the output directory to save the held units and plots in
    print('Where do you want to save the held units and plots?')
    save_dir = askdirectory()
    
    stat_save_dir = os.path.join(save_dir,'statistic_plots')
    if not os.path.isdir(stat_save_dir):
        os.mkdir(stat_save_dir)
    
    #Save the data dictionary just in case want in future
    np.save(os.path.join(save_dir,'data_dict.npy'),data_dict,allow_pickle=True)
    
    #Save the calculated intra datasets
    np.save(os.path.join(stat_save_dir,'all_intra_J3.npy'),np.array(all_intra_J3),allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_intra_J3_norm.npy'),np.array(all_intra_J3_norm),allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_intra_euc_dist.npy'),np.array(all_intra_euc_dist),allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_intra_euc_dist_norm.npy'),np.array(all_intra_euc_dist_norm),allow_pickle=True)

#%% Intra-unit joint score

# all_intra_unit_J3_score = (all_intra_J3 - np.min(all_intra_J3))/np.max(all_intra_J3 - np.min(all_intra_J3))
all_intra_unit_J3_norm_score = (all_intra_J3_norm - np.min(all_intra_J3_norm))/np.mean(all_intra_J3_norm - np.min(all_intra_J3_norm))
#all_intra_unit_PSTH_score = (all_intra_euc_dist - np.min(all_intra_euc_dist))/np.max(all_intra_euc_dist - np.min(all_intra_euc_dist))
all_intra_unit_PSTH_norm_score = (all_intra_euc_dist_norm - np.min(all_intra_euc_dist_norm))/np.mean(all_intra_euc_dist_norm - np.min(all_intra_euc_dist_norm))
#intra_unit_joint_score = (all_intra_unit_J3_score + all_intra_unit_J3_norm_score + all_intra_unit_PSTH_score + all_intra_unit_PSTH_norm_score)/4
if use_electrode == 'y':
    intra_unit_joint_norm_score = (all_intra_unit_J3_norm_score + all_intra_unit_PSTH_norm_score)/3 #/3 because the electrode distance for the same unit is 0 so don't need to add in here
else:
    intra_unit_joint_norm_score = (all_intra_unit_J3_norm_score + all_intra_unit_PSTH_norm_score)/2
intra_unit_joint_score_percentile = np.percentile(intra_unit_joint_norm_score,95)

#%% Plot units

#Create plots of unit information pulled above for manual review
indiv_unit_save_dir = os.path.join(save_dir,'unit_plots')
if not os.path.isdir(indiv_unit_save_dir):
    os.mkdir(indiv_unit_save_dir)
    
for d_i in range(num_days):
    day_unit_save_dir = os.path.join(indiv_unit_save_dir,'day_' + str(d_i+1))
    if not os.path.isdir(day_unit_save_dir):
        os.mkdir(day_unit_save_dir)
    num_neur = data_dict[d_i]['num_neur']
    for n_i in range(num_neur):
        #Pull waveform info
        unit_waveforms = data_dict[d_i]['all_unit_waveforms'][n_i]
        avg_waveform = np.nanmean(unit_waveforms,0)
        std_waveform = np.nanstd(unit_waveforms,0)
        unit_waveforms_norm = data_dict[d_i]['all_unit_waveforms_norm'][n_i]
        avg_waveform_norm = np.nanmean(unit_waveforms_norm,0)
        std_waveform_norm = np.nanstd(unit_waveforms_norm,0)
        unit_intra_J3 = data_dict[d_i]['intra_J3'][n_i]
        unit_intra_J3_norm = data_dict[d_i]['intra_J3_norm'][n_i]
        #Pull PSTH info
        avg_PSTH = data_dict[d_i]['all_unit_fr_curves'][n_i]
        avg_PSTH_norm = data_dict[d_i]['all_unit_fr_curves_norm'][n_i]
        euc_dist_PSTH = data_dict[d_i]['intra_euc_dist_fr_curves'][n_i]
        euc_dist_PSTH_norm = data_dict[d_i]['intra_euc_dist_fr_curves_norm'][n_i]
        #Create info title
        title = 'Neuron ' + str(n_i) + '\nIntra-J3: ' + str(np.round(unit_intra_J3,2)) + \
            '\nIntra-J3 Norm: ' + str(np.round(unit_intra_J3_norm,2)) + \
            '\nEuclidean Distance PSTH: ' + str(np.round(euc_dist_PSTH,2)) + \
            '\nEuclidean Distance Norm PSTH: ' + str(np.round(euc_dist_PSTH_norm,2))
        #Create figure
        f_unit, ax_unit = plt.subplots(nrows = 2, ncols = 2, figsize = (8,8))
        ax_unit[0,0].plot(avg_waveform,color='k')
        ax_unit[0,0].plot(avg_waveform + std_waveform,color='gray',alpha=0.5)
        ax_unit[0,0].plot(avg_waveform - std_waveform,color='gray',alpha=0.5)
        ax_unit[0,0].set_title('Average Waveform')
        ax_unit[0,1].plot(avg_waveform_norm,color='k')
        ax_unit[0,1].plot(avg_waveform_norm + std_waveform_norm,color='gray',alpha=0.5)
        ax_unit[0,1].plot(avg_waveform_norm - std_waveform_norm,color='gray',alpha=0.5)
        ax_unit[0,1].set_title('Average Normalized Waveform')
        ax_unit[1,0].plot(bin_starts,avg_PSTH)
        ax_unit[1,0].set_title('Avg PSTH')
        ax_unit[1,0].set_xlabel('Time from Taste Delivery (ms)')
        ax_unit[1,1].plot(bin_starts,avg_PSTH_norm)
        ax_unit[1,1].set_title('Avg Normalized PSTH')
        ax_unit[1,1].set_xlabel('Time from Taste Delivery (ms)')
        plt.suptitle(title)
        plt.tight_layout()
        f_unit.savefig(os.path.join(day_unit_save_dir,'Neuron_' + str(n_i) + '.png'))
        f_unit.savefig(os.path.join(day_unit_save_dir,'Neuron_' + str(n_i) + '.svg'))
        plt.close(f_unit)
        
        del unit_waveforms, avg_waveform, std_waveform, unit_waveforms_norm, avg_waveform_norm, \
            std_waveform_norm, unit_intra_J3, unit_intra_J3_norm, avg_PSTH, avg_PSTH_norm, \
                euc_dist_PSTH, euc_dist_PSTH_norm, title, f_unit, ax_unit
        
del d_i, day_unit_save_dir, num_neur, n_i

#%% Intra-Day Calcs

#For all pairs of neurons within a day calculate the inter-J3 and PSTH distance
#scores to use as "different neuron" cutoff information
    
try:
    all_intra_day_J3 = np.load(os.path.join(stat_save_dir,'all_intra_day_J3.npy'),allow_pickle=True)
    all_intra_day_J3_norm = np.load(os.path.join(stat_save_dir,'all_intra_day_J3_norm.npy'),allow_pickle=True)
    all_intra_day_euc_dist = np.load(os.path.join(stat_save_dir,'all_intra_day_euc_dist.npy'),allow_pickle=True)
    all_intra_day_euc_dist_norm = np.load(os.path.join(stat_save_dir,'all_intra_day_euc_dist_norm.npy'),allow_pickle=True)
    all_intra_day_electrode_dist = np.load(os.path.join(stat_save_dir,'all_intra_day_electrode_dist.npy'),allow_pickle=True)
except:
    all_intra_day_J3 = []
    all_intra_day_J3_norm = []
    all_intra_day_euc_dist = []
    all_intra_day_euc_dist_norm = []
    all_intra_day_electrode_dist = []
    for d_i in range(num_days):
        in_day_neur_combos = list(itertools.combinations(np.array(all_neur_inds[d_i]),2))
        in_day_electrode_indices = np.array(all_electrode_inds[d_i])
        in_day_electrode_dist = np.array([np.abs(in_day_electrode_indices[id_i[0]] - in_day_electrode_indices[id_i[1]]) for id_i in in_day_neur_combos])
        max_electrode_dist = np.max(in_day_electrode_dist)
        #distances are normalized by the maximal distance possible
        all_intra_day_electrode_dist.extend(list(in_day_electrode_dist/max_electrode_dist))
        del in_day_electrode_indices, in_day_electrode_dist, max_electrode_dist
        for nc_i, nc in tqdm.tqdm(enumerate(in_day_neur_combos)):
            #Waveforms
            unit_waveforms_1 = data_dict[d_i]['all_unit_waveforms'][nc[0]]
            unit_waveforms_2 = data_dict[d_i]['all_unit_waveforms'][nc[1]]
            combined_waveforms = np.concatenate((unit_waveforms_1,unit_waveforms_2),0)
            pca = PCA(n_components = 4)
            pca.fit(combined_waveforms)
            pca_wf_1 = pca.transform(unit_waveforms_1)
            pca_wf_2 = pca.transform(unit_waveforms_2)
            all_intra_day_J3.append(calculate_J3(pca_wf_1,pca_wf_2))
            #   Normalized waveforms
            unit_waveforms_1_norm = data_dict[d_i]['all_unit_waveforms_norm'][nc[0]]
            unit_waveforms_2_norm = data_dict[d_i]['all_unit_waveforms_norm'][nc[1]]
            combined_waveforms_norm = np.concatenate((unit_waveforms_1_norm,unit_waveforms_2_norm),0)
            pca_norm = PCA(n_components = 4)
            pca_norm.fit(combined_waveforms_norm)
            pca_wf_1_norm = pca_norm.transform(unit_waveforms_1_norm)
            pca_wf_2_norm = pca_norm.transform(unit_waveforms_2_norm)
            all_intra_day_J3_norm.append(calculate_J3(pca_wf_1_norm,pca_wf_2_norm))
            #Euclidean PSTH Calcs
            #   Regular PSTH
            psth_1 = data_dict[d_i]['all_unit_fr_curves'][nc[0]]
            psth_2 = data_dict[d_i]['all_unit_fr_curves'][nc[1]]
            all_intra_day_euc_dist.append(euclidean(psth_1,psth_2))
            #   Normalized PSTH
            psth_1_norm = data_dict[d_i]['all_unit_fr_curves_norm'][nc[0]]
            psth_2_norm = data_dict[d_i]['all_unit_fr_curves_norm'][nc[1]]
            all_intra_day_euc_dist_norm.append(euclidean(psth_1_norm,psth_2_norm))
            del unit_waveforms_1, unit_waveforms_2, combined_waveforms, pca, \
                pca_wf_1, pca_wf_2, unit_waveforms_1_norm, unit_waveforms_2_norm, \
                    pca_norm, pca_wf_1_norm, pca_wf_2_norm, psth_1, psth_2, \
                        psth_1_norm, psth_2_norm
        del nc_i, nc
    all_intra_day_J3 = np.array(all_intra_day_J3)
    all_intra_day_J3_norm = np.array(all_intra_day_J3_norm)
    all_intra_day_euc_dist = np.array(all_intra_day_euc_dist)
    all_intra_day_euc_dist_norm = np.array(all_intra_day_euc_dist_norm)
    all_intra_day_electrode_dist = np.array(all_intra_day_electrode_dist)
    
    #Save calcs
    np.save(os.path.join(stat_save_dir,'all_intra_day_J3.npy'),all_intra_day_J3,allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_intra_day_J3_norm.npy'),all_intra_day_J3_norm,allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_intra_day_euc_dist.npy'),all_intra_day_euc_dist,allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_intra_day_euc_dist_norm.npy'),all_intra_day_euc_dist_norm,allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_intra_day_electrode_dist.npy'),all_intra_day_electrode_dist,allow_pickle=True)
    
    del d_i
    
# all_intra_day_J3_score = (all_intra_day_J3 - np.min(all_intra_day_J3))/np.max(all_intra_day_J3 - np.min(all_intra_day_J3))
all_intra_day_J3_norm_score = (all_intra_day_J3_norm - np.min(all_intra_day_J3_norm))/np.mean(all_intra_day_J3_norm - np.min(all_intra_day_J3_norm))
# all_intra_day_PSTH_score = (all_intra_day_euc_dist - np.min(all_intra_day_euc_dist))/np.max(all_intra_day_euc_dist - np.min(all_intra_day_euc_dist))
all_intra_day_PSTH_norm_score = (all_intra_day_euc_dist_norm - np.min(all_intra_day_euc_dist_norm))/np.mean(all_intra_day_euc_dist_norm - np.min(all_intra_day_euc_dist_norm))
# intra_day_joint_score = (all_intra_day_J3_score + all_intra_day_J3_norm_score + all_intra_day_PSTH_score + all_intra_day_PSTH_norm_score)/4
all_intra_day_electrode_dist_score = (all_intra_day_electrode_dist - np.min(all_intra_day_electrode_dist))/np.mean(all_intra_day_electrode_dist - np.min(all_intra_day_electrode_dist))
if use_electrode == 'y':
    intra_day_joint_norm_score = (all_intra_day_J3_norm_score + all_intra_day_PSTH_norm_score + all_intra_day_electrode_dist_score)/3
else:
    intra_day_joint_norm_score = (all_intra_day_J3_norm_score + all_intra_day_PSTH_norm_score)/2
intra_day_joint_score_percentile = np.percentile(intra_day_joint_norm_score,5)

#%% Inter-Day Calcs
#For all pairs of units calculate the inter-J3 and euclidean distance 
#metrics and plot the results

all_neur_combos = list(itertools.product(*all_neur_inds))
all_day_combos = list(itertools.combinations(np.arange(num_days),2))

try:
    all_inter_J3 = np.load(os.path.join(stat_save_dir,'all_inter_J3.npy'),allow_pickle=True)
    all_inter_J3_norm = np.load(os.path.join(stat_save_dir,'all_inter_J3_norm.npy'),allow_pickle=True)
    all_inter_PSTH = np.load(os.path.join(stat_save_dir,'all_inter_PSTH.npy'),allow_pickle=True)
    all_inter_PSTH_norm = np.load(os.path.join(stat_save_dir,'all_inter_PSTH_norm.npy'),allow_pickle=True)
    all_inter_electrode_dist = np.load(os.path.join(stat_save_dir,'all_inter_electrode_dist.npy'),allow_pickle=True)
except:
    all_inter_J3 = np.nan*np.ones((len(all_neur_combos),len(all_day_combos)))
    all_inter_J3_norm = np.nan*np.ones((len(all_neur_combos),len(all_day_combos)))
    all_inter_PSTH = np.nan*np.ones((len(all_neur_combos),len(all_day_combos)))
    all_inter_PSTH_norm = np.nan*np.ones((len(all_neur_combos),len(all_day_combos)))
    all_inter_electrode_dist = np.nan*np.ones((len(all_neur_combos),len(all_day_combos)))
    for nc_i, nc in tqdm.tqdm(enumerate(all_neur_combos)):
        for dc_i, dc in enumerate(all_day_combos):
            #Electrode Calcs
            e_ind_1 = all_electrode_inds[dc[0]][nc[dc[0]]]
            e_ind_2 = all_electrode_inds[dc[1]][nc[dc[1]]]
            e_dist = np.abs(e_ind_1 - e_ind_2)
            all_inter_electrode_dist[nc_i,dc_i] = e_dist
            #J3 Calcs
            #   Regular waveforms
            unit_waveforms_1 = data_dict[dc[0]]['all_unit_waveforms'][nc[dc[0]]]
            unit_waveforms_2 = data_dict[dc[1]]['all_unit_waveforms'][nc[dc[1]]]
            combined_waveforms = np.concatenate((unit_waveforms_1,unit_waveforms_2),0)
            pca = PCA(n_components = 4)
            pca.fit(combined_waveforms)
            pca_wf_1 = pca.transform(unit_waveforms_1)
            pca_wf_2 = pca.transform(unit_waveforms_2)
            all_inter_J3[nc_i,dc_i] = calculate_J3(pca_wf_1,pca_wf_2)
            #   Normalized waveforms
            unit_waveforms_1_norm = data_dict[dc[0]]['all_unit_waveforms_norm'][nc[dc[0]]]
            unit_waveforms_2_norm = data_dict[dc[1]]['all_unit_waveforms_norm'][nc[dc[1]]]
            combined_waveforms_norm = np.concatenate((unit_waveforms_1_norm,unit_waveforms_2_norm),0)
            pca_norm = PCA(n_components = 4)
            pca_norm.fit(combined_waveforms_norm)
            pca_wf_1_norm = pca_norm.transform(unit_waveforms_1_norm)
            pca_wf_2_norm = pca_norm.transform(unit_waveforms_2_norm)
            all_inter_J3_norm[nc_i,dc_i] = calculate_J3(pca_wf_1_norm,pca_wf_2_norm)
            #Euclidean PSTH Calcs
            #   Regular PSTH
            psth_1 = data_dict[dc[0]]['all_unit_fr_curves'][nc[dc[0]]]
            psth_2 = data_dict[dc[1]]['all_unit_fr_curves'][nc[dc[1]]]
            all_inter_PSTH[nc_i,dc_i] = euclidean(psth_1,psth_2)
            #   Normalized PSTH
            psth_1_norm = data_dict[dc[0]]['all_unit_fr_curves_norm'][nc[dc[0]]]
            psth_2_norm = data_dict[dc[1]]['all_unit_fr_curves_norm'][nc[dc[1]]]
            all_inter_PSTH_norm[nc_i,dc_i] = euclidean(psth_1_norm,psth_2_norm)
            
    #Save calcs
    np.save(os.path.join(stat_save_dir,'all_inter_J3.npy'),all_inter_J3,allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_inter_J3_norm.npy'),all_inter_J3_norm,allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_inter_PSTH.npy'),all_inter_PSTH,allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_inter_PSTH_norm.npy'),all_inter_PSTH_norm,allow_pickle=True)
    np.save(os.path.join(stat_save_dir,'all_inter_electrode_dist.npy'),all_inter_electrode_dist,allow_pickle=True)

  
all_neur_ind_dist = np.zeros(len(all_neur_combos))
for dc_i, dc in enumerate(all_day_combos):
    all_neur_ind_dist += np.array([np.abs(all_neur_combos[anc_i][dc[0]] - all_neur_combos[anc_i][dc[1]]) for anc_i in range(len(all_neur_combos))])
all_neur_ind_dist = all_neur_ind_dist/len(all_day_combos)

f_stats, ax_stats = plt.subplots(nrows = 4, ncols = 2, figsize = (8,8))
#Histograms of all stats
ax_stats[0,0].hist(all_intra_J3, histtype='step', alpha=0.5, label='Intra')
ax_stats[0,0].hist(all_intra_day_J3, histtype='step', alpha=0.5, label='Intra-Day')
ax_stats[0,0].hist(all_inter_J3.flatten(), histtype='step', alpha=0.5, label='Inter')
ax_stats[0,0].legend(loc='upper left')
ax_stats[0,0].set_title('J3 Histograms')
ax_stats[0,1].hist(all_intra_J3_norm, histtype='step', alpha=0.5, label='Intra')
ax_stats[0,1].hist(all_intra_day_J3_norm, histtype='step', alpha=0.5, label='Intra-Day')
ax_stats[0,1].hist(all_inter_J3_norm.flatten(), histtype='step', alpha=0.5, label='Inter')
ax_stats[0,1].set_title('J3 Norm Histograms')
ax_stats[1,0].hist(all_intra_euc_dist, histtype='step', alpha=0.5, label='Intra')
ax_stats[1,0].hist(all_intra_day_euc_dist, histtype='step', alpha=0.5, label='Intra-Day')
ax_stats[1,0].hist(all_inter_PSTH.flatten(), histtype='step', alpha=0.5, label='Inter')
ax_stats[1,0].set_title('PSTH Distance Histograms')
ax_stats[1,1].hist(all_intra_euc_dist_norm, histtype='step', alpha=0.5, label='Intra')
ax_stats[1,1].hist(all_intra_day_euc_dist_norm, histtype='step', alpha=0.5, label='Intra-Day')
ax_stats[1,1].hist(all_inter_PSTH_norm.flatten(), histtype='step', alpha=0.5, label='Inter')
ax_stats[1,1].set_title('PSTH Norm Distance Histograms')
#Heatmaps of neuron pairs across days
ax = ax_stats[2,0].imshow(all_inter_J3.T,aspect='auto')
plt.colorbar(ax)
ax_stats[2,0].set_title('Inter J3')
ax = ax_stats[2,1].imshow(all_inter_J3_norm.T,aspect='auto')
plt.colorbar(ax)
ax_stats[2,1].set_title('Inter J3 Norm')
ax = ax_stats[3,0].imshow(all_inter_PSTH.T,aspect='auto')
plt.colorbar(ax)
ax_stats[3,0].set_title('PSTH Dist')
ax = ax_stats[3,1].imshow(all_inter_PSTH_norm.T,aspect='auto')
plt.colorbar(ax)
ax_stats[3,1].set_title('PSTH Dist Norm')
plt.suptitle('All Unit / Day Pair Statistics')
plt.tight_layout()
f_stats.savefig(os.path.join(stat_save_dir,'unit_day_pair_stats.png'))
f_stats.savefig(os.path.join(stat_save_dir,'unit_day_pair_stats.svg'))
plt.close(f_stats)

# all_inter_J3_score = (all_inter_J3 - np.min(all_inter_J3))/np.max(all_inter_J3 - np.min(all_inter_J3))
all_inter_J3_norm_score = (all_inter_J3_norm - np.min(all_inter_J3_norm))/np.mean(all_inter_J3_norm - np.min(all_inter_J3_norm))
# all_inter_PSTH_score = (all_inter_PSTH - np.min(all_inter_PSTH))/np.max(all_inter_PSTH - np.min(all_inter_PSTH))
all_inter_PSTH_norm_score = (all_inter_PSTH_norm - np.min(all_inter_PSTH_norm))/np.max(all_inter_PSTH_norm - np.min(all_inter_PSTH_norm))
mean_inter_electrode_dist = np.mean(all_inter_electrode_dist)
all_inter_electrode_dist = all_inter_electrode_dist/mean_inter_electrode_dist

if use_electrode == 'y':
    joint_norm_score = ((all_inter_J3_norm_score + all_inter_PSTH_norm_score + all_inter_electrode_dist)/3).squeeze()
else:
    joint_norm_score = ((all_inter_J3_norm_score + all_inter_PSTH_norm_score)/2).squeeze()

#%% Calc Held Units

#Calculate a joint cutoff based on intra-day and intra-neuron distributions
score_cutoff = np.min((intra_unit_joint_score_percentile,intra_day_joint_score_percentile))

#Just normalized data scores
neur_pairs_low_norm_score = np.array(all_neur_combos)[np.where(joint_norm_score <= score_cutoff)[0]]
neur_pair_norm_scores = joint_norm_score[np.where(joint_norm_score <= score_cutoff)[0]]
neur_pair_plus_score = np.concatenate((neur_pairs_low_norm_score,np.expand_dims(neur_pair_norm_scores,1)),1)

#First find where neurons are duplicated across any day
unique_inds_by_day = []
duplicate_inds_by_day = []
for d_i in range(num_days):
    neur_counts = np.unique(neur_pairs_low_norm_score[:, d_i], return_counts=True)
    unique_inds_by_day.append(neur_counts[0])
    duplicate_inds_by_day.append(neur_counts[0][np.where(neur_counts[1] > 1)[0]])

#Now progressively go by day to determine best matches
best_matches = []
#Pass 1 in order by day
for d_i in range(num_days):
    day_neur = unique_inds_by_day[d_i]
    for n_i in day_neur:
        pair_locs = np.where(neur_pairs_low_norm_score[:,d_i] == n_i)[0]
        if len(pair_locs) > 1:
            all_loc_scores = neur_pair_norm_scores[pair_locs]
            min_score = np.argmin(all_loc_scores)
            best_matches.append(np.squeeze(neur_pair_plus_score[pair_locs[min_score],:]))
        else:
            best_matches.append(np.squeeze(neur_pair_plus_score[pair_locs,:]))
#Pass 2 get rid of duplicates by day
best_matches = np.array(best_matches)
best_matches = np.sort(best_matches,0)
for d_i in np.sort(np.arange(num_days))[::-1]:
    neur_counts = np.unique(best_matches[:, d_i], return_counts=True)
    repeat_neurons = neur_counts[0][np.where(neur_counts[1] > 1)[0]]
    if len(repeat_neurons) > 0:
        loop_repeat = 1
    while loop_repeat == 1:
        pair_locs = np.where(best_matches[:,d_i] == repeat_neurons[0])[0]
        pair_loc_scores = best_matches[pair_locs,-1]
        min_score = np.argmin(pair_loc_scores)
        remove_locs = np.setdiff1d(pair_locs,pair_locs[min_score]*np.ones(1))
        keep_locs = np.setdiff1d(np.arange(np.shape(best_matches)[0]),remove_locs)
        best_matches = best_matches[keep_locs,:]
        #Recalculate the neuron counts and repeat neurons
        neur_counts = np.unique(best_matches[:, d_i], return_counts=True)
        repeat_neurons = neur_counts[0][np.where(neur_counts[1] > 1)[0]]
        if len(repeat_neurons) < 1:
            loop_repeat = 0
#Plot neuron pairs and print out .csv file with list
held_unit_csv = os.path.join(save_dir,'held_units.csv')
header = ''
for day in range(num_days-1):
    header += 'Day ' + str(day+1) + ','
header += 'Day ' + str(num_days)
    
# Make a file to save the numbers of the units that are deemed to have been held across days
with open(held_unit_csv, 'w') as f:
    f.write(header)
    for r_i in range(np.shape(best_matches)[0]):
        row_matches = best_matches[r_i,:-1]
        row_match_string = '\n'
        for d_i in range(num_days-1):
            row_match_string += str(best_matches[r_i,d_i].astype('int')) + ','
        row_match_string += str(best_matches[r_i,num_days-1].astype('int')) 
        f.write(row_match_string)

held_unit_save_dir = os.path.join(indiv_unit_save_dir,'held_units')
if not os.path.isdir(held_unit_save_dir):
    os.mkdir(held_unit_save_dir)

for bm_i, bm in enumerate(best_matches):
    f, ax = plt.subplots(nrows = 2, ncols = num_days, figsize = (8,8), sharey='row')
    for d_i in range(num_days):
        unit_waveforms_norm = data_dict[d_i]['all_unit_waveforms_norm'][int(bm[d_i])]
        avg_waveform_norm = np.nanmean(unit_waveforms_norm,0)
        std_waveform_norm = np.nanstd(unit_waveforms_norm,0)
        avg_psth = data_dict[d_i]['all_unit_fr_curves_norm'][int(bm[d_i])]
        #Plot waveform
        ax[0,d_i].plot(avg_waveform_norm,color='k')
        ax[0,d_i].plot(avg_waveform_norm + std_waveform_norm,color='gray',alpha=0.5)
        ax[0,d_i].plot(avg_waveform_norm - std_waveform_norm,color='gray',alpha=0.5)
        ax[0,d_i].set_title('Day ' + str(d_i) + '\nUnit ' + str(int(bm[d_i])) + '\nWaveform')
        #Plot PSTH
        ax[1,d_i].plot(bin_starts,avg_psth)
        ax[1,d_i].set_title('Avg PSTH')
    f.savefig(os.path.join(held_unit_save_dir,'held_pair_' + str(bm_i) + '.png'))
    f.savefig(os.path.join(held_unit_save_dir,'held_pair_' + str(bm_i) + '.svg'))
    plt.close(f)

#%% OLD CODE: Below is the more classic approach with percentile cutoffs

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
    

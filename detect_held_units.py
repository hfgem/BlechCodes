# Import stuff!
import sys
import os

current_path = os.path.realpath(__file__)
os.chdir(('/').join(current_path.split('/')[:-1]))

import numpy as np
import tables
import easygui
from tkinter.filedialog import askdirectory
import itertools
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from functions.blech_held_units_funcs import *
    
# Time to use after a taste is delivered to calculate an average firing rate for each neuron
post_deliv_time = 500

# Ask the user for the number of days to be compared
num_days = int_input("How many days-worth of data are you comparing for held units (integer)? ")

# Ask the user for the percentile criterion to use to determine held units
percent_criterion = easygui.multenterbox(msg = 'What percentile of the intra unit J3 distribution do you want to use to pull out held units?', fields = ['Percentile criterion (1-100) - lower is more conservative (e.g., 95)'])
percent_criterion = float(percent_criterion[0])

# Ask the user for the percentile criterion to use to determine held units
wf_type = easygui.multenterbox('Which types of waveforms to be used for held_unit analysis',
                               'Type either "raw_CAR_waveform" or "norm_waveform"',
                               ['waveform type'],
                               ['norm_waveform'])[0]

data_dict = dict() #Store all the different days' data in a dictionary
all_neur_inds = [] #Store all neuron indices to calculate cross-day combinations
all_intra_J3 = [] #Store all intra J3 data to calculate cutoff for inter-J3
for n_i in range(num_days):
    data_dict[n_i] = dict()
    #Ask for directory of the dataset hdf5 file
    print('Where is the hdf5 file from the ' + str(n_i + 1) + ' day?')
    dir_name = askdirectory()
    data_dict[n_i]['dir_name'] = dir_name
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
    for d_i in len(start_dig_in_times):
        flat_start_dig_in_times.extend(start_dig_in_times[d_i])
    start_dig_in_times = np.sort(np.array(flat_start_dig_in_times)) #All dig in times - regardless of taste
    data_dict[n_i]['start_dig_in_times'] = start_dig_in_times
    taste_start_time = np.max([np.min(start_dig_in_times) - 5000,0]).astype('int')
    taste_end_time = (np.max(start_dig_in_times) + 5000).astype('int')
    data_dict[n_i]['taste_interval'] = [taste_start_time,taste_end_time]
    #Calculate the Intra-J3 data for the units
    intra_J3 = []
    all_unit_waveforms = []
    all_unit_times = []
    for unit in range(num_neur):
        # Only go ahead if this is a single unit
        if hf5.root.unit_descriptor[unit]['single_unit'] == 1:
            exec("wf_day1 = hf5.root.sorted_units.unit%03d.waveforms[:]" % (unit)) #num wav x 60
            exec("t_day1 = hf5.root.sorted_units.unit%03d.times[:]" % (unit)) #num wav
            if wf_type == 'norm_waveform':
                wf_day1 = wf_day1 / np.std(wf_day1)
            #Pull out taste response interval waveforms only
            taste_wf_inds = np.where((t_day1 >= taste_start_time)*(t_day1 <= taste_end_time))[0]
            taste_t_day1 = t_day1[taste_wf_inds]
            all_unit_times.append(taste_t_day1)
            taste_wf_day1 = wf_day1[taste_wf_inds,:]
            all_unit_waveforms.append(taste_wf_day1)
            pca = PCA(n_components = 4)
            pca.fit(taste_wf_day1)
            pca_wf_day1 = pca.transform(taste_wf_day1) #num wav x 4
            #Calculate firing rates within given interval following taste delivery
            neur_fr = np.zeros(len(start_dig_in_times))
            for st_i in start_dig_in_times:
                spike_hz = len(np.where((taste_t_day1>st_i)*(taste_t_day<=st_i+post_deliv_time))[0])/(post_deliv_time/1000)
                neur_fr[st_i] = spike_hz
            mean_fr = np.nanmean(neur_fr)
            std_fr = np.nanstd(neur_fr)
            #Add to PCA of waveform the firing rate data in the form of mean and standard deviation
            pca_wf_day1 = np.concatenate((pca_wf_day1,mean_fr*np.ones((len(taste_t_day1),0)),std_fr*np.ones((len(taste_t_day1),0))),axis=1)
            intra_J3.append(calculate_J3(pca_wf_day1[:int(taste_wf_day1.shape[0]*(1.0/3.0)), :], 
                                         pca_wf_day1[int(taste_wf_day1.shape[0]*(2.0/3.0)):, :]))
    data_dict[n_i]['intra_J3'] = intra_J3
    data_dict[n_i]['all_unit_waveforms'] = all_unit_waveforms
    data_dict[n_i]['all_unit_times'] = all_unit_times
    all_intra_J3.extend(intra_J3)
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

#Calculate the intra-J3 percentile cutoff
all_intra_J3_cutoff = np.percentile(all_intra_J3, percent_criterion)

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
held_index_counter = 0
for nc in all_neur_combos:
    #Collect waveforms to be compared
    waveforms = [] #list of numpy arrays
    all_waveforms = []
    for day in range(len(nc)):
        wf = data_dict[day]['all_unit_waveforms'][nc[day]]
        if day == 0:
            wf_peak = np.ceil(wf.shape[1]/2).astype('int')
        if wf_type == 'norm_waveform':
            waveforms.append(wf/np.nanstd(wf))
            all_waveforms.extend(wf/np.nanstd(wf))
        else:
            waveforms.append(wf)
            all_waveforms.extend(wf)
            
    #Fit PCA to waveforms
    pca = PCA(n_components = 4)
    pca.fit(np.array(all_waveforms))
    day_pca = []
    for day in range(len(nc)):
        day_pca.append(pca.transform(np.array(waveforms[day])))
            
    #Calculate the inter_J3 across days
    all_days_inter_J3 = []
    for dc in all_day_combos:
        all_days_inter_J3.extend([calculate_J3(day_pca[dc[0]], day_pca[dc[1]])])
    all_inter_J3.append(all_days_inter_J3) 
    
    #Do all inter_J3 match the cutoff?
    if np.sum((np.array(all_days_inter_J3) <= all_intra_J3_cutoff).astype('int')) == len(all_day_combos):
        
        #Compare peak distributions via ttest
        ttest_res = np.ones(len(all_day_combos))
        for dp_i, dp in enumerate(all_day_combos):
            _, p_val = ttest_ind(waveforms[dp[0]][:,wf_peak],waveforms[dp[1]][:,wf_peak],nan_policy='omit')
            ttest_res[dp_i] = p_val
            
        if np.sum((ttest_res >= .05).astype('int')) == len(all_day_combos):
        
            #Save to csv the unit indices per day
            statement = '\n'
            for i in range(len(nc)-1):
                statement += str(nc[i]) + ','
            statement += str(nc[-1])
            with open(os.path.join(save_dir,f'held_units_{wf_type}_{percent_criterion}.csv'), 'a') as f:
                f.write(statement)
            
            #Create a plot of the matching waveforms
            fig, ax = plt.subplots(1, num_days, sharex=True, sharey=True, figsize=(12, 6))
            min_wf_val = 10000
            max_wf_val = -10000
            for d_i in range(num_days):
                wf_day_i = data_dict[d_i]['all_unit_waveforms'][nc[d_i]]
                num_wav = wf_day_i.shape[0]
                t = np.arange(wf_day_i.shape[1])
                mean_wfs = np.mean(wf_day_i, axis = 0)
                max_, min_ = np.max(mean_wfs), np.min(mean_wfs)
                if min_ < min_wf_val:
                    min_wf_val = min_
                if max_ > max_wf_val:
                    max_wf_val = max_
                ax[d_i].plot(t - 15, mean_wfs, linewidth = 5.0, color = 'black')
                ax[d_i].plot(t - 15, mean_wfs - np.std(wf_day_i, axis = 0), 
                       linewidth = 2.0, color = 'black', alpha = 0.5)
                ax[d_i].plot(t - 15, mean_wfs + np.std(wf_day_i, axis = 0), 
                       linewidth = 2.0, color = 'black', alpha = 0.5)
                ax[d_i].axhline(max_, color='r', ls='--')
                ax[d_i].axhline(min_, color='r', ls='--')
                ax[d_i].set_xlabel('Time (samples (30 per ms))', fontsize = 12)
                ax[d_i].set_ylabel('Voltage (microvolts)', fontsize = 12)
                ax[d_i].set_title('Unit ' + str(nc[d_i]) + ', total waveforms = ' + str(num_wav) + \
                                  '\nElectrode: ' + str(data_dict[d_i]['all_unit_info'][nc[d_i]][0]) , fontsize = 12)
            for d_i in range(num_days):
                ax[d_i].set_ylim([min_wf_val - 20, max_wf_val + 20])
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir,'held_index_' + str(held_index_counter) + '.png'), bbox_inches = 'tight')
            plt.close(fig)
            
            held_index_counter += 1

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


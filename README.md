# BlechCodes
 Repository for codes written for analysis of data generated in the Blech Lab at Brandeis University
 
 ## analyze_states.py
 This is a script to run through import and state-change analysis of sorted electrophysiology data

 ## bayes_replay_decoding.py
 This is a script to run Bayesian decoding of taste responses during the different segments of the experiment. It handles both full neuron populations and taste selective populations.

 ## compare_conditions.py
 This is a script to import results from running correlate_all.py, correlate_all_z_score.py, correlate_selective.py, and correlate_selective_z_score.py and run statistical significance tests and create all combinations of plots.

 ## compare_null.py 
 Script to generate null distributions and compare them against the true data across different statistics (neuron counts, spike counts, autocorrelation)

 ## correlate_[].py
 Each script with this naming convention runs correlations of population burst events with taste responses. Each one is titled based on what it correlates:
 - all: all neurons
 - all_z_score: all neurons with firing rates z-scored based on preceding interval firing rates
 - selective: taste selective neurons only
 - selective_z_score: taste selective neurons only but firing rates z-scored based on preceding interval firing rates

 ## find_deviations.py
 Script to calculate when population firing rate bursts / "deviations" occur in the dataset - for use in the correlate_[].py scripts.

 ## Functions

 ### archive
 This folder contains archived code.

 ### load_intan_rhd_format
 This folder contains IntanTech software provided by the company with minimal modification for use in data import and handling.

 ### analysis_funcs.py
 This script contains functions related to data analysis including: importing data, adding a "none" taste condition, calculating segment spike times, calculating tastant delivery spike times, storing/pulling data in/from the hdf5, calculating PSTHs, calculating rasters, calculating changepoint firing rates.
 
 ### blech_clust_tests.py
 This is a script to compare the results of process_data.py with the results of the blech_clust pipeline.

 ### changepoint_detection.py
 This is a set of functions to calculate changepoints in firing data both on the single neuron and population level.

 ### compare_conditions_funcs.py
 This is a set of functions relating to the statistical tests and plotting needed in compare_conditions.py

 ### corr_dist_calc_parallel[].py
 These 4 scripts contain functions related to parallelization of correlation distribution calculation used by correlate_[].py with the [] contents describing the changes (population analyses and zscored and combinations).
 
 ### data_cleaning.py
 A set of functions intended for cleaning electrode data - including (1) bandpass filtering, (2) signal averaging, and (3) signal clustering
 
 ### data_processing.py
 A set of functions intended for data processing - including (1) pulling filenames from directories, (2) splitting .dat files into electrodes, emg, and digital inputs, and (3) downsampling data

 ### decode_parallel.py
 Code to run parallelized decoding of tastes from rest interval time bins

 ### decoding_funcs.py
 A set of functions dedicated to decoding either taste selectivity or potential replay events.

 ### dev_calcs.py
 A set of functions used to find periods of time that have firing rate deviations and then analyze their statistics  [deprecated - to be moved to archive]

 ### dev_corr_calcs.py
 A set of functions used to calculate and plot correlations of deviation events with taste responses

 ### dev_funcs.py
 Functions used by find_deviations.py to pull/reformat/analyze deviations in true and null datasets

 ### dev_plot_funcs.py
 Functions used to plot deviation statistics

 ### dev_plots.py
 Functions used to plot deviation statistics [deprecated - to be moved to archive]
 
 ### hdf5_handling.py
 A set of functions dedicated to handling / creating hdf5 files.

 ### null_distributions.py
 Functions to calculate statistics and plot comparisons between null and true deviation datasets

 ### plot_funcs.py
 General plotting functions for spike data (raster, PSTH, etc...)
 
 ### postsort.py
 A set of functions to perform post-sorting tests/cleanup such as waveform similarity, spike time collisions, etc...

 ### seg_compare.py
 Code to compare cross-segment activity

 ### spike_clust.py
 A set of functions dedicated to clustering candidate spike waveforms.
 
 ### spike_nosort.py
 A set of functions dedicated to pulling candidate spike waveforms into "neurons" without using sorting techniques, as inspired by the 2019 Trautmann et al paper. IN PROGRESS.
 
 ### spike_sort.py
 A set of functions dedicated to sorting candidate spike waveforms into neurons.
 
 ### spike_template.py
 A set of functions dedicated to template-matching spike clustering techniques.

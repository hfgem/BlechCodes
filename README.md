# BlechCodes
 Repository for codes written for analysis of data generated in the Blech Lab at Brandeis University
 
 ## process_data.py
 This is a script to run through data import of .dat files, generated during electrophysiology recordings, and their subsequent processing and spike sorting.
 
 ## analyze_states.py
 This is a script to run through import and state-change analysis of sorted electrophysiology data
 
 ## Functions
 
 ### blech_clust_tests.py
 This is a script to compare the results of process_data.py with the results of the blech_clust pipeline.
 
 ### data_cleaning.py
 A set of functions intended for cleaning electrode data - including (1) bandpass filtering, (2) signal averaging, and (3) signal clustering
 
 ### data_processing.py
 A set of functions intended for data processing - including (1) pulling filenames from directories, (2) splitting .dat files into electrodes, emg, and digital inputs, and (3) downsampling data
 
 ### hdf5_handling.py
 A set of functions dedicated to handling / creating hdf5 files.
 
 ### postsort.py
 A set of functions to perform post-sorting tests/cleanup such as waveform similarity, spike time collisions, etc...

 ### spike_clust.py
 A set of functions dedicated to clustering candidate spike waveforms.
 
 ### spike_nosort.py
 A set of functions dedicated to pulling candidate spike waveforms into "neurons" without using sorting techniques, as inspired by the 2019 Trautmann et al paper. IN PROGRESS.
 
 ### spike_sort.py
 A set of functions dedicated to sorting candidate spike waveforms into neurons.
 
 ### spike_template.py
 A set of functions dedicated to template-matching spike clustering techniques.

 ### archive folder
 A collection of code endeavours that have been archived and are not to be used either for lack of completion or function.
 
 ### load_intan_rhd_format
 A set of functions for importing rhd data - courtesy of IntanTech
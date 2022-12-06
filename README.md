# BlechCodes
 Repository for codes written for analysis of data generated in the Blech Lab at Brandeis University
 
 ## analyze_data.py
 This is a script to run through data import of .dat files, generated during electrophysiology recordings, and their subsequent processing and analysis.
 
 ## Functions
 
 ### data_cleaning.py
 A set of functions intended for cleaning electrode data - including (1) bandpass filtering, (2) signal averaging, and (3) signal clustering
 
 ### data_processing.py
 A set of functions intended for data processing - including (1) pulling filenames from directories, (2) splitting .dat files into electrodes, emg, and digital inputs, and (3) downsampling data
 
 ### hdf5_handling.py
 A set of functions dedicated to handling / creating hdf5 files.

 ### spike_clust.py
 A set of functions dedicated to clustering candidate spike waveforms.
 
 ### spike_nosort.py
 A set of functions dedicated to pulling candidate spike waveforms into "neurons" without using sorting techniques, as inspired by the 2019 Trautmann et al paper.
 
 ### spike_sort.py
 A set of functions dedicated to sorting candidate spike waveforms into neurons.
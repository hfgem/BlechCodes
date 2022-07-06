# BlechCodes
 Repository for codes written for analysis of data generated in the Blech Lab at Brandeis University
 
 ## analyze_data.py
 This is a script to run through data import of .dat files, generated during electrophysiology recordings, and their subsequent processing and analysis.
 
 ## Functions
 
 ### data_processing.py
 This is a collection of functions intended for data processing - including (1) pulling filenames from directories, (2) splitting .dat files into electrodes, emg, and digital inputs, and (3) import into hdf5 file format.
 
 ### ICA_data_split.py
 This is a collection of functions dedicated to performing independent component analysis (ICA) on electrode data, as a spike-sorting step.

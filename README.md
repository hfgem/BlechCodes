# BlechCodes
 Repository for codes written for analysis of data generated in the Blech Lab at Brandeis University.
 
 ## analyze_replay.py
 Running this program will begin the full analysis pipeline. The pipeline is designed to progress through each stage of analysis and after each stage ask the user whether or not to continue. Subsequent launching of this program will continue where the particular dataset's analysis pipeline left off.
 
 ## compare_conditions.py
 This program allows the user to import multiple datasets (or just one) to analyze across epochs, segments, and tastes for multi-animal analyses. The expectation is that this is run after analyze_replay.py has been run on all individual animals to be combined into a joint analysis.
 
 ## functions
 This folder contains all collections of functions used by the pipeline. The first in the pipeline of which is 'run_analysis_handler.py', which handles the state checks and analysis pipeline progression. Below you'll find a diagram of the flow of analyses as well as information on which analysis step calls which other funcions.
 
 ## params
 This folder contains a template of the parameter file that analyze_replay.py uses for analyses. This file is copied by analyze_replay.py into the individual dataset folder, where the user can modify values to suit the needs of the analysis.
 
 ## utils
 This folder contains three utility files:
 - data_utils.py contains data import functions.
 - replay_utils.py contains metadata import and state tracking functions.
 - test_support.py is a script for testing code bits outside of the analyze_replay.py pipeline.
 
 ## archive
 A collection of archived scripts/functions to be deleted eventually.
 

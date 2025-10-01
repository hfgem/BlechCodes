# BlechCodes
 Repository for codes written for analysis of data generated in the Blech Lab at Brandeis University.
 
 ## Environment Setup
 To set up a conda environment with the proper dependencies, use the .yml file in the utils folder. Open your terminal or Anaconda Prompt, navigate to the directory containing the blechcodes.yml file, and execute the following command:
 
 ```conda env create -f blechcodes.yml```
 
 ## Running code
 To run the code in your terminal, first navigate to the folder where you've downloaded the code: ex. ```cd ~/Documents/GitHub/BlechCodes```. Once there, ensure that your conda environment is activated: ex. ```conda activate BlechCodes```. Run the program: ```python analyze_replay.py``` (replace analyze_replay.py with the desired pipeline below).

 ### analyze_replay.py
 Running this program will begin the full analysis pipeline. The pipeline is designed to progress through each stage of analysis and after each stage ask the user whether or not to continue. Subsequent launching of this program will continue where the particular dataset's analysis pipeline left off.

 ### detect_held_units.py
 This script will run through held unit detection across multiple days. It is written to have cells of code to run bit by bit and review the results. It should be run after analyze_replay.py.
 
 ### analyze_multiday_replay.py
 Running this program will begin the multiday replay analysis pipeline. The pipeline is designed to take day 1 replay events and taste responses across 2 days of recording and correlate/decode the events. This code relies on analyze_replay.py being run for each of the days first, and then detect_held_units.py being run on the days to be analyzed.
 
 ### compare_conditions.py
 This program allows the user to import multiple datasets (or just one) to analyze across epochs, segments, and tastes for multi-animal analyses. The expectation is that this is run after analyze_replay.py has been run on all individual animals to be combined into a joint analysis.

 ### compare_multiday.py
 This program allows the user to compare and combine the results of running analyze_multiday_replay.py across animals.
 
 ## Folders
 
 ### functions
 This folder contains all collections of functions used by the various above pipelines and scripts.
 
 ## params
 This folder contains a template of the parameter file that analyze_replay.py uses for analyses. This file is copied by analyze_replay.py into the individual dataset folder, where the user can modify values to suit the needs of the analysis.
 
 ## utils
 This folder contains utility files (for data import, state tracking, and more), the conda environment requirements .txt file, as well as support scripts used to test and run code during development.

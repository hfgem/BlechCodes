#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:55:57 2024

@author: Hannah Germaine

Pipeline master for analyzing replay in previously clustered data
"""

#Import necessary packages and functions
import os
from utils.replay_utils import import_metadata, state_tracker
from utils.data_utils import import_data
from functions.run_analysis_handler import run_analysis_steps

if __name__ == '__main__':
	
	# Grab current directory and data directory / metadata
	script_path = os.path.realpath(__file__)
	blechcodes_dir = os.path.dirname(script_path)
	
	metadata_handler = import_metadata([blechcodes_dir])
	try:
		dig_in_names = metadata_handler.info_dict['taste_params']['tastes']
	except:
		dig_in_names = []
	
	# import data from hdf5
	data_handler = import_data([metadata_handler.dir_name, metadata_handler.hdf5_dir, dig_in_names])
	
	# state tracker: determine where in the pipeline this dataset is at, and work from that point
	state_handler = state_tracker([metadata_handler.dir_name])
	
	# repackage data from all handlers
	metadata = dict()
	for var in vars(metadata_handler):
		metadata[var] = getattr(metadata_handler,var)
	del metadata_handler
	
	data_dict = dict()
	for var in vars(data_handler):
		data_dict[var] = getattr(data_handler,var)
	del data_handler
	
	state_dict = dict()
	for var in vars(state_handler):
		state_dict[var] = getattr(state_handler,var)
	del state_handler
	
	# run analysis!
	run_analysis_steps([metadata,data_dict,state_dict])
	
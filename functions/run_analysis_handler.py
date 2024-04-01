#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:36:46 2024

@author: Hannah Germaine

Pipeline run handler
"""

import os

current_path = os.path.realpath(__file__)
blech_codes_path = '/'.join(current_path.split('/')[:-1]) + '/'
os.chdir(blech_codes_path)
from utils.replay_utils import state_tracker
from functions.data_description_analysis import run_data_description_analysis
from functions.changepoint_analysis import run_changepoint_detection
from functions.deviation_analysis import run_find_deviations
from functions.deviation_null_analysis import run_deviation_null_analysis

class run_analysis_steps():
	
	def __init__(self,args):
		self.metadata = args[0]
		self.data_dict = args[1]
		self.state_dict = args[2]
		run_loop = 1
		while run_loop == 1:
			self.run_state()
			run_loop = self.check_continue()
		
	def run_state(self,):
		#Always run this step first to get the spike time datasets - if it's the first run, it will take longer to go through PSTH and raster plots
		run_data_description_analysis([self.metadata,self.data_dict])
		#Now proceed based on the state of the analysis
		state = self.state_dict['state']
		if state == 0:
			#Calculate changepoints in true data
			run_changepoint_detection([self.metadata,self.data_dict])
		elif state == 1:
			#Calculate population spike deviations in true data
			run_find_deviations([self.metadata,self.data_dict])
		elif state == 2:
			#Compare population spike deviations in true data to null distribution data
			run_deviation_null_analysis([self.metadata,self.data_dict])
		#elif state == 3:
			#
	
	def check_continue(self,):
		state_handler = state_tracker([self.metadata['dir_name'],1]) #added list value of 1 flags the state tracker to increment
		cont_choice = state_handler.cont_choice
		if cont_choice == 'y':
			run_loop = 1
		else:
			run_loop = 0
		state_dict = dict()
		for var in vars(state_handler):
			state_dict[var] = getattr(state_handler,var)
		self.state_dict = state_dict
		print("New State = " + str(self.state_dict['state']))
		
		return run_loop
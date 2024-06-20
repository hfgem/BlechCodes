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

from functions.dependent_bayes_analysis import run_dependent_bayes
from functions.deviation_correlations import run_deviation_correlations
from functions.data_null_analysis import run_data_null_analysis
from functions.deviation_null_analysis import run_deviation_null_analysis
from functions.deviation_analysis import run_find_deviations
from functions.changepoint_analysis import run_changepoint_detection
from functions.data_description_analysis import run_data_description_analysis
from utils.replay_utils import state_tracker



class run_analysis_steps():

    def __init__(self, args):
        self.metadata = args[0]
        self.data_dict = args[1]
        self.state_dict = args[2]
        self.state_exec_dict()
        run_loop = 1
        while run_loop == 1:
            self.run_state()
            run_loop = self.check_continue()

    def run_state(self,):
        # Always run this step first to get the spike time datasets - if it's the first run, it will take longer to go through PSTH and raster plots
        run_data_description_analysis([self.metadata, self.data_dict])
        # Now proceed based on the state of the analysis
        state = self.state_dict['state']
        state_name = self.state_exec_dict[state]['name']
        state_exec_line = self.state_exec_dict[state]['exec']
        print('Now beginning execution of analysis state: ' + state_name)
        exec(state_exec_line)

    def check_continue(self,):
        state = self.state_dict['state']
        state_name = self.state_exec_dict[state]['name']
        print('You just ran analysis state: ' + state_name)
        try:
            next_state_name = self.state_exec_dict[state+1]['name']
            print('The next analysis state is: ' + next_state_name)
            # added list value of 1 flags the state tracker to increment
            state_handler = state_tracker([self.metadata['dir_name'], 1])
            cont_choice = state_handler.cont_choice
            if cont_choice == 'y':
                run_loop = 1
            else:
                run_loop = 0
            state_dict = dict()
            for var in vars(state_handler):
                state_dict[var] = getattr(state_handler, var)
            self.state_dict = state_dict
            print("New State = " + str(self.state_dict['state']))
        except:
            print("There appear to be no more analysis states, you're done!")
            run_loop = 0

        return run_loop

    def state_exec_dict(self,):
        # Define a dictionary of state and execution line
        # TODO: maybe create a params dictionary that is imported so it's not hard-coded?
        #	 This would allow the user to also change ordering or skip certain analyses (or go back!)
        state_exec_dict = dict()
        state_exec_dict[0] = dict()
        state_exec_dict[0]['name'] = 'Changepoint Detection'
        state_exec_dict[0]['exec'] = 'run_changepoint_detection([self.metadata,self.data_dict])'
        state_exec_dict[1] = dict()
        state_exec_dict[1]['name'] = 'Calculate Deviations'
        state_exec_dict[1]['exec'] = 'run_find_deviations([self.metadata,self.data_dict])'
        state_exec_dict[2] = dict()
        state_exec_dict[2]['name'] = 'Deviation Null Analysis'
        state_exec_dict[2]['exec'] = 'run_deviation_null_analysis([self.metadata,self.data_dict])'
        state_exec_dict[3] = dict()
        state_exec_dict[3]['name'] = 'Null Deviation x Taste Correlations'
        state_exec_dict[3]['exec'] = 'run_dependent_bayes([self.metadata,self.data_dict])'
        state_exec_dict[4] = dict()
        state_exec_dict[4]['name'] = 'Deviation x Taste Correlations'
        state_exec_dict[4]['exec'] = 'run_deviation_correlations([self.metadata,self.data_dict])'
        state_exec_dict[5] = dict()
        state_exec_dict[5]['name'] = 'Bayesian Replay Decoding'
        state_exec_dict[5]['exec'] = 'run_dependent_bayes([self.metadata,self.data_dict])'
        state_exec_dict[6] = dict()
        state_exec_dict[6]['name'] = 'Bayesian Deviation Decoding'
        state_exec_dict[6]['exec'] = ''
        self.state_exec_dict = state_exec_dict

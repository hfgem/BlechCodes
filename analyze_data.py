#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:50:15 2022

@author: hannahgermaine
"""

#This file calls functions from data_processing.py to analyze data

import sys
sys.path.append('/Users/hannahgermaine/Documents/GitHub/BlechCodes/')
import functions.data_processing as dp


#Pull data names
dat_files_list, electrodes_list, emg_ind, dig_in_list, dig_in_names = dp.file_names()

#Import data by type


		
		
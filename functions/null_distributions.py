#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 10:33:10 2023

@author: Hannah Germaine
This set of code creates null distributions for a given dataset where the 
existing data is passed in as indices of spikes and length of dataset
"""

import os, random, json, gzip
os.environ["OMP_NUM_THREADS"] = "4"


def run_null_create_parallelized(inputs):
	null_ind = inputs[0]
	spikes = inputs[1]
	start_t = inputs[2]
	end_t = inputs[3]
	null_dir = inputs[4]
	fake_spike_times = [random.sample(range(start_t,end_t),len(spikes[n_i])) for n_i in range(len(spikes))]
	json_str = json.dumps(fake_spike_times)
	json_bytes = json_str.encode()
	filepath = null_dir + 'null_' + str(null_ind) + '.json'
	with gzip.GzipFile(filepath, mode="w") as f:
		f.write(json_bytes)
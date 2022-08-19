#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:29:22 2022
This code is used to generate spike waveform templates to use in template-matching.

The goal is to generate templates of:
	1. A regular spike
	2. A fast spike
	3. A positive spike

The positive/negative peaks will be placed as they would be expected in the 
extracted spikes from spike_sort.py (based on num_pts_left and num_pts_right).
The peak height will be normalized to 1, and non-peak time baselined at 0.
The regular spike will have a slower return to baseline, the fast spike will be
symmetrical in its deflection and return, and the positive spike will be
symmetrical in its deflection and return.

@author: hannahgermaine
"""

import numpy as np
import scipy.stats as ss
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def generate_templates(sampling_rate,num_pts_left,num_pts_right):
	"""This function generates 3 template vectors of neurons with a peak 
	centered between num_pts_left and num_pts_right."""
	
	x_points = np.arange(-num_pts_left,num_pts_right)
	templates = np.zeros((3,len(x_points)))
	
	fast_spike_width = sampling_rate*(1/1000)
	sd = fast_spike_width/12
	
	pos_spike = ss.norm.pdf(x_points, 0, sd)
	max_pos_spike = max(abs(pos_spike))
	pos_spike = pos_spike/max_pos_spike
	fast_spike = -1*pos_spike
	reg_spike_bit = ss.gamma.pdf(np.arange(fast_spike_width),5)
	peak_reg = find_peaks(reg_spike_bit)[0][0]
	reg_spike = np.concatenate((np.zeros(num_pts_left-peak_reg),-1*reg_spike_bit),axis=0)
	len_diff = len(x_points) - len(reg_spike)
	reg_spike = np.concatenate((reg_spike,np.zeros(len_diff)))
	max_reg_spike = max(abs(reg_spike))
	reg_spike = reg_spike/max_reg_spike
	
	templates[0,:] = pos_spike
	templates[1,:] = fast_spike
	templates[2,:] = reg_spike
	
# 	fig = plt.figure()
# 	plt.subplot(3,1,1)
# 	plt.plot(x_points,pos_spike)
# 	plt.title('Positive Spike')
# 	plt.subplot(3,1,2)
# 	plt.plot(x_points,fast_spike)
# 	plt.title('Fast Spike')
# 	plt.subplot(3,1,3)
# 	plt.plot(x_points,reg_spike)
# 	plt.title('Regular Spike')
# 	plt.tight_layout()
	
	return templates
	
	
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 11:38:22 2025

@author: Hannah Germaine

This file contains user input functions.
"""	

def select_analysis_groups(unique_list):
    """
    This function allows the user to select which aspects of the data to use 
    in the analysis.
    INPUTS:
        - unique_list: list of unique aspects available
    RETURNS:
        - unique_list: list of unique aspects to use
    NOTE:
        - This function will continue to prompt the user for an answer until the 
		answer given is a list of integers.
    """
    
    unique_prompt = ''
    for un_i, un in enumerate(unique_list):
        unique_prompt += str(un_i) + ': ' + un + '\n'
    unique_prompt += 'Please provide a comma-separate list of indices: '
    ind_to_keep = int_list_input(unique_prompt)
    unique_list = [unique_list[i] for i in ind_to_keep]
    
    return unique_list	

def int_input(prompt):
	"""
	This function asks a user for an integer input
	INPUTS:
		prompt = string containing boolean input prompt
	RETURNS:
		int_val = integer value
	NOTE:
		This function will continue to prompt the user for an answer until the 
		answer given is an integer.
	"""
	int_loop = 1	
	while int_loop == 1:
		response = input(prompt)
		try:
			int_val = int(response)
			int_loop = 0
		except:
			print("\tERROR: Incorrect data entry, please input an integer.")
	
	return int_val
	
def bool_input(prompt):
	"""
	This function asks a user for a boolean input of y/n.
	INPUTS:
		prompt = string containing boolean input prompt
	RETURNS:
		response = y / n
	NOTE:
		This function will continue to prompt the user for an answer until the 
		answer given is y or n.
	"""
	bool_loop = 1	
	while bool_loop == 1:
		response = input(prompt).lower()
		if (response == 'y') or (response == 'n'):
			bool_loop = 0
		else:
			print("\tERROR: Incorrect data entry, please try again with Y/y/N/n.")
	
	return response	
				
def int_list_input(prompt):
	"""
	This function asks a user for a list of integer inputs.
	INPUTS:
		prompt = string containing boolean input prompt
	RETURNS:
		int_list = list of integer values
	NOTE:
		This function will continue to prompt the user for an answer until the 
		answer given is a list of integers.
	"""
	int_loop = 1	
	while int_loop == 1:
		response = input(prompt)
		response_list = response.split(',')
		try:
			int_list = []
			for item in response_list:
				int_list.append(int(item))
			int_loop = 0
		except:
			print("\tERROR: Incorrect data entry, please input a list of integers.")
	
	return int_list


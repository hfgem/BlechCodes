#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:57:17 2024

@author: Hannah Germaine

This file contains utility functions to support replay analysis pipeline
"""
import os
import json
import csv
import easygui
import glob 
import shutil
from tkinter.filedialog import askdirectory

class import_metadata():
	
	def __init__(self,args): #args assumed to include only code directory
		self.dir_name = self.get_dir_name()
		self.get_hdf5_dir()
		self.get_params_path(args)
		self.load_params()
		self.get_info_path()
		self.load_info()
		
	def get_dir_name(self,):
		dir_name = askdirectory()
		if dir_name[-1] != '/':
			dir_name += '/'
		return dir_name
	
	def get_hdf5_dir(self,):
		file_list = glob.glob(os.path.join(self.dir_name,'**.h5'))
		if len(file_list) > 0:
			self.hdf5_dir = file_list[0]
		else:
			print('No HDF5 file found. Please run sorting or re-run analysis with correct folder selection.')
			quit()
		
	def get_params_path(self,args):
		#First check data folder
		self.params_file_path = os.path.join(self.dir_name,'analysis_params.json')
		if os.path.exists(self.params_file_path) == False:
			#Copy over from BlechCodes folder
			file_list = glob.glob(os.path.join(args[0],'params','**.json'))
			if len(file_list) > 0:
				if len(file_list) > 1:
					file_found = 0
					for f_i in range(len(file_list)):
						is_file = self.bool_input('Is ' + file_list[f_i] + ' the correct params file? ')
						if is_file == 'y':
							self.orig_params_file_path = file_list[f_i]
							file_found = 1
				else:
					self.orig_params_file_path = file_list[0]
					file_found = 1
				if file_found == 0:
					print('No PARAMS file found. Please ensure BlechCodes/params/ contains the params template json.')
					quit()
				else:
					print('Copying params file to data folder for future use and editing.')
					params_file_name = (self.orig_params_file_path).split('/')[-1]
					shutil.copy(self.orig_params_file_path,self.params_file_path)
					print('At this time, please edit the params as desired for analysis of your dataset.')
					print('You can find the params file at:')
					print(self.params_file_path)
					val = self.bool_input('When finished with the params file, type Y/y: ')
					if val == 'n':
						print('Why are you responding then?? Exiting program.')
						quit()
			else:
				print('No template PARAMS file found. Please ensure BlechCodes/params/ contains the params template json.')
				quit()
	
	def load_params(self,):
		with open(self.params_file_path, 'r') as params_file:
			self.params_dict = json.load(params_file)
	
	def get_info_path(self,):
		file_list = glob.glob(os.path.join(self.dir_name,'**.info'))
		if len(file_list) > 0:
			if len(file_list) > 1:
				file_found = 0
				for f_i in range(len(file_list)):
					is_file = self.bool_input('Is ' + file_list[f_i] + ' the correct info file? ')
					if is_file == 'y':
						self.info_file_path = file_list[f_i]
						file_found = 1
			else:
				self.info_file_path = file_list[0]
				file_found = 1
			if file_found == 0:
				print('No info file found. Please ensure you ran clustering / are in the correct folder!')
				quit()	
	
	def load_info(self,):
		with open(self.info_file_path, 'r') as info_file:
			self.info_dict = json.load(info_file)
			
	def bool_input(self,prompt):
		#This function asks a user for an integer input
		bool_loop = 1	
		while bool_loop == 1:
			print("Respond with Y/y/N/n:")
			response = input(prompt)
			if (response.lower() != 'y')*(response.lower() != 'n'):
				print("\tERROR: Incorrect data entry, only give Y/y/N/n.")
			else:
				bool_val = response.lower()
				bool_loop = 0
		
		return bool_val
		
class state_tracker():
	
	def __init__(self,args):
		self.data_path = args[0]
		self.get_state()
		if len(args) > 1:
			self.increase_state()
			self.cont_func()
		
	def get_state(self,):
		try:
			with open(os.path.join(self.data_path, 'analysis_state_tracker.csv'), newline='') as f:
				reader = csv.reader(f)
				state_list = list(reader)
			state_val = int(state_list[0][0])
			self.state = state_val
		except:
			state_val = 0
			with open(os.path.join(self.data_path, 'analysis_state_tracker.csv'), 'w') as f:
				# using csv.writer method from CSV package
				write = csv.writer(f)
				write.writerows([[state_val]])
			self.state = state_val
	
	def increase_state(self,):
		state_val = self.state + 1
		with open(os.path.join(self.data_path, 'analysis_state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		self.state = state_val
	
	def cont_func(self,):
		ask_loop = 1
		while ask_loop == 1:
			print('Analysis is at a good stopping point.')
			print('You can continue or quit here and resume at the same spot by running analyze_replay.py.')
			keep_going = input('Would you like to continue [y/n]? ')
			keep_going = keep_going.lower()
			try:
				if (keep_going == 'y') or (keep_going == 'n'):
					ask_loop = 0
				else:
					print("Please try again. Incorrect entry.")
			except:
				print("Please try again. Incorrect entry.")
		self.cont_choice = keep_going
	
	
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 14:17:33 2019
"""

import tensorboardX as tbx
import torch
import time
import csv
import warnings
from potion.common.misc_utils import maybe_make_dir

class Logger():
    def __init__(self, directory='./logs_t', name='auto', modes=['human', 'csv', 'tensorboard']):
        self.modes = modes
        self.directory = directory
        timestamp = str(int(time.time()))
        self.name = name + '_' + timestamp
        self.ready = False
        self.keys = []
        self.open_files = []
        
    def write_info(self, row):
        try:
            maybe_make_dir(self.directory)
            with open(self.directory + '/' + self.name + '_info.txt', 'w') as info_file:
                for key, val in row.items():
                    info_file.write(key + ':\t' + str(val) + '\n')
        except:
            warnings.warn('Could not write info file!')
        
    def open(self, keys):
        try:
            maybe_make_dir(self.directory)
            self.keys = keys
        except:
            warnings.warn('Could not create log directory!')
            return
            
        # csv
        if 'csv' in self.modes:
            try:
                self.csv_file = open(self.directory + '/' 
                                     + self.name + '.csv', 'w')
                self.open_files.append(self.csv_file)
                self.csv_writer = csv.DictWriter(self.csv_file, keys)
                self.csv_writer.writeheader()
                self.ready = True
            except:
                warnings.warn('Could not create csv file!')
        
            if 'tensorboard' in self.modes:
                try:
                    self.tb_writer = tbx.SummaryWriter(self.directory + '/'
                                                       + self.name)
                    self.ready = True
                except:
                    warnings.warn('Could not create TensorboardX files!')
            
            if 'human' in self.modes:
                self.ready = True
        
    def write_row(self, row, iteration):
        if not self.ready:
            warnings.warn('You must open the logger first!')
            return
        
        if 'human' in self.modes:
            for key, val in row.items():
                print(key, ':\t', val)
        
        #csv
        if 'csv' in self.modes:
            try:
                self.csv_writer.writerow(row)
            except:
                warnings.warn('Could not write data to csv!')
                
        if 'tensorboard' in self.modes:
            try:
                for key, val in row.items():   
                    self.tb_writer.add_scalar(key, val, iteration)
            except:
                warnings.warn('Could not write data to TensorboardX')

    def save_params(self, params, iteration):
        param_dir = self.directory + '/' + self.name + '_params'
        try:
            maybe_make_dir(param_dir)
        except:
            warnings.warn('Could not create log directory!')
            return
        
        try:
            torch.save(params, param_dir + '/generation_' + str(iteration) + '.pt')
        except:
            warnings.warn('Could not save parameters!')
        
    def close(self):
        if not self.ready:
            warnings.warn('You must open the logger first!')
            return
        
        try:
            for f in self.open_files:
                f.close()
            self.ready = False
        except:
            warnings.warn('Could not close logger!')
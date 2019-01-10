#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:44:22 2018

@author: Matteo Papini

Pytorch utilities
"""

import torch
import torch.nn as nn
import numpy as np

def flatten(params):
    """
    Turns a module's parameters (or gradients) into a flat numpy array
        
    params: the module's parameters (or gradients)
    """
    return np.concatenate([np.ravel(p.data.numpy()) for p in params])

def set_from_flat(params, values):
    """
    Sets a module's parameters from a flat array
    
    params: the module's parameters
    values: a flat array
    """
    k = 0
    for p in params:
        shape = tuple(list(p.shape))
        offset = sum(shape)
        val = values[k : k + offset]
        val = np.reshape(val, shape)
        with torch.no_grad():
            p.copy_(torch.tensor(val))
        k = k + offset
        
class FlatModule(nn.Module):
    """Module with flattened parameter management"""
    def num_params(self):
        """Number of parameters of the module"""
        return sum(p.numel() for p in self.parameters())
        
    def get_flat(self):
        """Module parameters as flat array"""
        return flatten(self.parameters())
    
    def set_from_flat(self, values):
        """Set module parameters from flat array"""
        set_from_flat(self.parameters(), values)
        
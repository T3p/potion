#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:44:22 2018

@author: Matteo Papini

Pytorch utilities
"""

import torch
import torch.nn as nn

def num_params(params):
    with torch.no_grad():
        return sum(p.numel() for p in params)

def flatten(params):
    """
    Turns a module's parameters (or gradients) into a flat numpy array
        
    params: the module's parameters (or gradients)
    """
    with torch.no_grad():
        return torch.cat([p.data.view(-1) for p in params])

def set_from_flat(params, values):
    """
    Sets a module's parameters from a flat array or tensor
    
    params: the module's parameters
    values: a flat array or tensor
    """
    with torch.no_grad():
        values = torch.tensor(values)
        k = 0
        for p in params:
            shape = tuple(list(p.shape))
            offset = sum(shape)
            val = values[k : k + offset]
            val = val.view(shape)
            with torch.no_grad():
                p.copy_(torch.tensor(val))
            k = k + offset
            
class FlatModule(nn.Module):
    """Module with flattened parameter management"""
    def num_params(self):
        """Number of parameters of the module"""
        return num_params(self.parameters())
        
    def get_flat(self):
        """Module parameters as flat array"""
        return flatten(self.parameters())
    
    def set_from_flat(self, values):
        """Set module parameters from flat array"""
        set_from_flat(self.parameters(), values)
        
def flat_gradients(module, loss, coeff=None):
    module.zero_grad()
    if coeff is None:
        coeff = torch.ones(loss.numel())
    loss.backward(coeff, retain_graph=True)
    return torch.cat([p.grad.view(-1) for p in module.parameters()])

def jacobian(module, loss):
    jac = torch.zeros((loss.numel(), module.num_params()))
    for i in range(loss.numel()):
        mask = torch.zeros(loss.numel(), dtype=torch.float)
        mask[i] = 1.
        jac[i, :] = flat_gradients(module, loss, mask)
    return jac
    

"""Testing"""
if __name__ == '__main__':
    pass
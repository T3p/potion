#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:44:22 2018

@author: Matteo Papini

Pytorch utilities
"""

import torch
import torch.nn as nn
import warnings

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
        if not torch.is_tensor(values):
            values = torch.tensor(values)
        k = 0
        for p in params:
            shape = tuple(list(p.shape))
            offset = torch.prod(torch.tensor(shape)).item()
            val = values[k : k + offset]
            val = val.view(shape)
            with torch.no_grad():
                p.copy_(val)
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
        
    def save_flat(self, path):
        try:
            torch.save(self.get_flat(), path)
        except:
            warnings.warn('Could not save parameters!')
    
    def load_from_flat(self, path):
        try:
            values = torch.load(path)
        except:
            warnings.warn('Could not load parameters!')
            return
        self.set_from_flat(values)
        
def flat_gradients(module, loss, coeff=None):
    module.zero_grad()
    loss.backward(coeff, retain_graph=True)
    return torch.cat([p.grad.view(-1) for p in module.parameters()])

def jacobian(module, loss, coeff=None):
    """Inefficient! Use jacobian-vector product whenever possible
    (still useful for nonlinear functions of gradients, such as 
    in Peter's baseline for REINFORCE)"""
    mask = torch.eye(loss.numel())

    jac = torch.stack([flat_gradients(module, loss, mask[i,:]) 
                      for i in range(loss.numel())],
                      dim = 0)
    return jac

def tensormat(a, b):
    """
    a: NxHxm
    b: NxH
    a*b: NxHxm 
    """
    return torch.einsum('ijk,ij->ijk', (a,b))

def complete_out(x, dim):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    while x.dim() < dim:
        x = x.unsqueeze(0)
    return x

def complete_in(x, dim):
    while x.dim() < dim:
        x = x.unsqueeze(-1)
    return x

def atanh(x):
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))

def maybe_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x

"""Testing"""
if __name__ == '__main__':
    from potion.common.mappings import LinearMapping
    F = LinearMapping(2,3)
    x = torch.ones(2, requires_grad=True)
    y = F(x)
    print(y)
    print(jacobian(F,y))
    print(atanh(torch.tanh(torch.tensor([0.5, -0.5]))))
    #y.backward(torch.ones(3))
    #print(x.grad)
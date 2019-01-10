#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:26:56 2019

@author: Matteo Papini
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import potion.common.torch_utils as tu
from potion.common.mappings import LinearMapping
from potion.common.densities import GaussianDensity

class ContinuousPolicy(tu.FlatModule):
    """
        To define a custom policy:
            define its self._pdf
            next: features and squashing
    """        
    def forward(self, s, a):
        return self._pdf(s, a)
    
    def log_pdf(self, s, a):
        return self._pdf.log_pdf(s,a)
    
    def sample(self, s):
        return self._pdf.sample(s)
    
    def get_loc_params(self):
        return self._pdf.get_loc_params()
    
    def get_scale_params(self):
        return self._pdf.get_scale_params()

class SimpleGaussianPolicy(ContinuousPolicy):
    def __init__(self, n_states, n_actions, mu_init=None, log_std_init=None, learn_std=False):
        super(SimpleGaussianPolicy, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        log_std_init = log_std_init if log_std_init is not None else torch.zeros(n_actions)
        
        self.mu = mu = LinearMapping(n_states, n_actions)
        if mu_init is not None:
            mu.set_from_flat(mu_init)
        
        if learn_std:
            self.log_std = log_std = nn.Parameter(log_std_init)
        else:
            self.log_std = log_std = autograd.Variable(log_std_init)
        
        self._pdf = GaussianDensity(mu, log_std)

"""
Testing
"""
if __name__ == '__main__':
    ds = 1
    da = 1
    s = torch.ones(ds)
    a = 99 + torch.zeros(da)
    mu_init = 100 + torch.zeros(ds)
    p = SimpleGaussianPolicy(ds, da, mu_init, learn_std=True)
    print(p.sample(s))
    print(p.num_params())
    print(p.get_flat())
    print(p.get_loc_params())
    print(p.get_scale_params())
    print(p(s, a))           
    print(p.log_pdf(s,a))
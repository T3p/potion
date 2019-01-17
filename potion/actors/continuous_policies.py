#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:26:56 2019

@author: Matteo Papini
"""

import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import potion.common.torch_utils as tu
from potion.common.mappings import LinearMapping
from torch.distributions import Normal

class ContinuousPolicy(tu.FlatModule):
    """Alias"""
    pass

class SimpleGaussianPolicy(ContinuousPolicy):
    """
    Factored
    \mu = \mu_{\theta}(x)
    \sigma = e^w (scalar)
    """
    def __init__(self, n_states, n_actions, feature_fun=None, squash_fun=None,
                 mu_init=None, logstd_init=None, 
                 learn_std=True):
        super(SimpleGaussianPolicy, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.feature_fun = feature_fun
        self.squash_fun = squash_fun
        self.learn_std = learn_std
        self.mu_init = mu_init
        self.logstd_init = logstd_init
        
        # Mean
        self.mu = LinearMapping(n_states, n_actions)
        if mu_init is not None:
            self.mu.set_from_flat(mu_init)
        
        # Log of standard deviation
        if logstd_init is None:
            logstd_init = torch.zeros(1)
        else:
            logstd_init = torch.tensor(logstd_init)
        if learn_std:
            self.logstd = nn.Parameter(logstd_init)
        else:
            self.logstd = autograd.Variable(logstd_init)
        
        # Normal(0,1)
        self._pdf = Normal(torch.zeros_like(self.logstd.data), 
                               torch.ones(n_actions))

    def log_pdf(self, s, a):
        log_sigma = self.logstd
        sigma = torch.exp(log_sigma)
        if self.feature_fun is not None:
            x = self.feature_fun(s)
        else:
            x = s
        return -((a - self.mu(x)) ** 2) / (2 * sigma ** 2) - \
            log_sigma  - .5 * math.log(2 * math.pi)
    
    def forward(self, s, a):
        return torch.exp(self.log_pdf(s, a))
    
    def act(self, s):
        with torch.no_grad():
            if self.feature_fun is not None:
                x = self.feature_fun(s)
            else:
                x = s
            sigma = torch.exp(self.logstd.data)
            a = self.mu(x) + self._pdf.sample() * sigma
            if self.squash_fun is not None:
                a = self.squash_fun(a)
            return a
    
    def num_loc_params(self):
        return self.n_states
    
    def num_scale_params(self):
        return self.n_actions
    
    def get_loc_params(self):
        return self.mu.get_flat()
    
    def get_scale_params(self):
        with torch.no_grad():
            return self.logstd.data
    
    def set_loc_params(self, val):
        self.mu.set_from_flat(val)
        
    def set_scale_params(self, val):
        with torch.no_grad():
            self.logstd.data = torch.tensor(val)
    
    def exploration(self):
        return torch.exp(self.logstd).item()
            
    def theta_score(self, s, a):
        sigma = torch.exp(self.logstd)
        if self.feature_fun is not None:
            x = self.feature_fun(s)
        else:
            x = s
        return x * (a - self.mu(x)) / sigma ** 2
    
    def omega_score(self, s, a):
        sigma = torch.exp(self.logstd)
        if self.feature_fun is not None:
            x = self.feature_fun(s)
        else:
            x = s
        return (((a - self.mu(x)) / sigma) ** 2 - 1)
            
    def info(self):
        return {'PolicyClass': 'SimpleGaussianPolicy',
                'LearnStd': self.learn_std,
                'nStates': self.n_states,
                'nActions': self.n_actions,
                'muInit': self.mu_init,
                'logstdInit': self.logstd_init,
                'featureFun': self.feature_fun,
                'squashFun': self.squash_fun}

"""
Testing
"""
if __name__ == '__main__':
    ds = 2
    da = 1
    s = torch.ones(ds)
    a = 100 + torch.zeros(da)
    mu_init = 50 + torch.zeros(ds)
    for flag in [False, True]:
        p = SimpleGaussianPolicy(ds, da, None, None, mu_init, learn_std=flag, logstd_init=0.)
        print(p.act(s))
        print(p.num_params())
        print(p.get_flat())
        print(p.get_loc_params())
        print(p.get_scale_params())  
        print(p(s,a))
        print('...')
        p.set_loc_params((-50.,-20.))
        p.set_scale_params((2.,-1.))
        print(p.act(s))
        print(p.get_flat())
        print(p.log_pdf(s,a))
        print()
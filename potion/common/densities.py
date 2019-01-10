#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:26:29 2019

@author: Matteo Papini

Parametric pdf modules with sample method 
"""

import math
import torch
import potion.common.torch_utils as tu 
from torch.distributions import Normal


class GaussianDensity(tu.FlatModule):
    """
        Factorized Gaussian from features
        
        mu: module or functional
        sigma: nn.Parameter or autograd.Variable
    """
    def __init__(self, mu, log_std):
        super(GaussianDensity, self).__init__()
        self.mu = mu
        self.log_std = log_std
        
        self._standard = Normal(torch.zeros_like(log_std.data), 
                               torch.ones(mu.d_out))
    def log_pdf(self, x, a):
        log_sigma = self.log_std.data
        sigma = torch.exp(log_sigma)
        return -((a - self.mu(x)) ** 2) / (2 * sigma ** 2) - \
            log_sigma  - .5 * math.log(2 * math.pi)
    
    def forward(self, x, a):
        return torch.exp(self.log_pdf(x, a))
            
    def sample(self, x):
        with torch.no_grad():
            sigma = torch.exp(self.log_std.data)
            return self.mu(x) + self._standard.sample() * sigma
    
    def get_loc_params(self):
        return self.mu.get_flat()
    
    def get_scale_params(self):
        return tu.flatten(self.log_std.data)

"""
Testing
"""
if __name__ == '__main__':
    from potion.common.mappings import LinearMapping
    from torch.autograd import Variable
    import torch.nn as nn
    
    x = torch.tensor([1., 2.])
    mu = LinearMapping(2, 2)
    mu.set_from_flat([0. , 1., 10., 0.])
    sigma = nn.Parameter(torch.zeros(2))
    #sigma = Variable(torch.zeros(2))
    p = GaussianDensity(mu, sigma)
    print(p.sample(x))
    print(p.num_params())
    print(p.get_flat())
    print(p(x, 10.+torch.zeros_like(x)))
    
    print(p.get_loc_params())
    print(p.get_scale_params())
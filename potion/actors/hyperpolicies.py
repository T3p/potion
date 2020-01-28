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
from torch.distributions import Normal
import math

class Hyperpolicy(tu.FlatModule):
    def __init__(self, lower_policy):
        super(Hyperpolicy, self).__init__()
        
        #Wrap lower policy to keep separated parameters 
        class _PolicyWrapper:
            def __init__(self, lower_policy):
                self.policy_module = lower_policy
            
            def act(self, s):
                return self.policy_module.act(s)
            
            def num_params(self):
                return self.policy_module.num_params()
    
            def get_params(self):
                return self.policy_module.get_params()

            def set_params(self, val):
                self.policy_module.set_params(val)
            
            def info(self):
                return self.policy_module.info()
        
        with torch.no_grad():
            self.lower_policy = _PolicyWrapper(lower_policy)
            self.num_lower_params = self.lower_policy.num_params()

class GaussianHyperpolicy(Hyperpolicy):
    """Gaussian hyperpolicy (factored)
        Initialization of base policy is irrelevant
    """
    def __init__(self, lower_policy, 
                 mu_init=None, 
                 logstd_init=None, 
                 bias=False, 
                 learn_std=True):
        super(GaussianHyperpolicy, self).__init__(lower_policy)
        self.mu_init = torch.zeros(self.num_lower_params) if mu_init is None else tu.maybe_tensor(mu_init)
        self.logstd_init = torch.zeros(self.num_lower_params) if logstd_init is None else tu.maybe_tensor(logstd_init)
        self.learn_std = learn_std
        
        if self.learn_std:
            self.logstd = nn.Parameter(self.logstd_init)
        else:
            self.logstd = autograd.Variable(self.logstd_init)
        self.mu = nn.Parameter(self.mu_init)
            
        # Normal(0,1)
        self._pdf = Normal(torch.zeros_like(self.mu), 
                               torch.ones_like(self.logstd))
    
    def log_pdf(self, lower_params):
        log_sigma = self.logstd
        sigma = torch.exp(log_sigma)
            
        logp = -((lower_params - self.mu) ** 2) / (2 * sigma ** 2) - \
            log_sigma  - .5 * math.log(2 * math.pi)
        return torch.sum(logp, -1)
    
    def forward(self, lower_params):
        return torch.exp(self.log_pdf(lower_params))
        
    def get_params(self):
        with torch.no_grad():
            return self.get_flat()
    
    def get_lower_params(self):
        with torch.no_grad():
            return self.lower_policy.get_params()
    
    def resample(self, update=True, deterministic=False):
        with torch.no_grad():
            if deterministic:
                lower_params = self.mu
            else:
                lower_params = self.mu + self._pdf.sample() * torch.exp(self.logstd)
            
            if update:
                self.lower_policy.set_params(lower_params)
            
            return lower_params
    
    def act(self, s):
        with torch.no_grad():
            return self.lower_policy.act(s)
    
    def loc_score(self, lower_params):
        with torch.no_grad():
            sigma = torch.exp(self.logstd)
            score = (lower_params - self.mu) / sigma ** 2
            return score

    def scale_score(self, lower_params):
        with torch.no_grad():
            sigma = torch.exp(self.logstd)
            return ((lower_params - self.mu) / sigma) ** 2 - 1
    
    def score(self, lower_params):
        with torch.no_grad():
            lower_params = tu.complete_out(lower_params, 2)
            if self.learn_std:
                return torch.cat((self.scale_score(lower_params),
                               self.loc_score(lower_params)), 
                               1)
            else:
                return self.loc_score(lower_params)
    
    def fisher(self):
        with torch.no_grad():
            fisher_mu = torch.exp(-2*self.logstd)
            if not self.learn_std:
                return fisher_mu
            else:
                fisher_logstd = torch.zeros_like(fisher_mu) + 2
                return torch.cat((fisher_logstd, fisher_mu))
    def info(self):
        hyperpolicy_info = {'HyperPolicyClass': self.__class__.__name__,
                'LearnStd': self.learn_std,
                'MuInit': self.mu_init,
                'LogstdInit': self.logstd_init}
        return {**hyperpolicy_info, **self.lower_policy.info()}

"""
Testing
"""
if __name__ == '__main__':
    from potion.common.misc_utils import seed_all_agent
    from potion.actors.continuous_deterministic_policies import ShallowDeterministicPolicy, DeepDeterministicPolicy
    
    ds = 2
    da = 2
    s = 0.1 * torch.ones(ds)
    a = 0.1 + torch.zeros(da)
    theta = torch.zeros(ds*da)
    
    #Testing Gaussian hyperpolicy with linear deterministic lower policy
    pol = ShallowDeterministicPolicy(ds, da)
    hpol = GaussianHyperpolicy(pol, 
                               learn_std=True)
    print(hpol.num_params())
    print(hpol.get_params())
    print()
    
    print(hpol.lower_policy.get_params())
    print(hpol.resample())
    print(hpol.lower_policy.get_params())
    print(hpol.act(s))
    print(hpol.act(s))
    print()
    
    print(hpol.log_pdf(theta))
    print(hpol(theta))
    print()
    
    print(hpol.loc_score(torch.stack((theta, theta+0.1),0)))
    print(hpol.scale_score(torch.stack((theta, theta+0.1),0)))
    print(hpol.score(theta))
    print()
    
    print(hpol.fisher())
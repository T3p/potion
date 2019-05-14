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
from torch.distributions import Normal, uniform

class ContinuousPolicy(tu.FlatModule):
    """Alias"""
    pass

class ShallowGaussianPolicy(ContinuousPolicy):
    """
    Factored
    linear mean \mu_{\theta}(x)
    diagonal, state-independent std \sigma = e^{\omega}
    """
    def __init__(self, n_states, n_actions, feature_fun=None, squash_fun=None,
                 mu_init=None, logstd_init=None, 
                 learn_std=True):
        super(ShallowGaussianPolicy, self).__init__()
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
            logstd_init = torch.zeros(self.n_actions)
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
        logp = -((a - self.mu(x)) ** 2) / (2 * sigma ** 2) - \
            log_sigma  - .5 * math.log(2 * math.pi)
        return torch.sum(logp, 2)
    
    def forward(self, s, a):
        return torch.exp(self.log_pdf(s, a))
    
    def act(self, s, deterministic=False):
        with torch.no_grad():
            if self.feature_fun is not None:
                x = self.feature_fun(s)
            else:
                x = s
            
            if deterministic:
                return self.mu(x)
            
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
        return self.logstd.data
    
    def set_loc_params(self, val):
        self.mu.set_from_flat(val)
        
    def set_scale_params(self, val):
        with torch.no_grad():
            self.logstd.data = torch.tensor(val)
    
    def exploration(self):
        return torch.exp(self.logstd).data
            
    def loc_score(self, s, a):
        sigma = torch.exp(self.logstd)
        if self.feature_fun is not None:
            x = self.feature_fun(s)
        else:
            x = s
        score = torch.einsum('ijk,ijh->ijkh', (x, (a - self.mu(x)) / sigma ** 2))
        score = score.reshape((score.shape[0], score.shape[1], score.shape[2]*score.shape[3]))
        return score
    
    def scale_score(self, s, a):
        sigma = torch.exp(self.logstd)
        if self.feature_fun is not None:
            x = self.feature_fun(s)
        else:
            x = s
        return ((a - self.mu(x)) / sigma) ** 2 - 1
    
    def score(self, s, a):
        s = tu.complete_out(s, 3)
        a = tu.complete_out(a, 3)
        if self.learn_std:
            return torch.cat((self.scale_score(s, a),
                           self.loc_score(s, a)), 
                           2)
        else:
            return self.loc_score(s,a)
            
    def info(self):
        return {'PolicyDlass': self.__class__.__name__,
                'LearnStd': self.learn_std,
                'StateDim': self.n_states,
                'ActionDim': self.n_actions,
                'MuInit': self.mu_init,
                'LogstdInit': self.logstd_init,
                'FeatureFun': self.feature_fun,
                'SquashFun': self.squash_fun}
        
        
class UniformPolicy(ContinuousPolicy):
    def __init__(self, n_actions, min_action=None, max_action=None):
        super(UniformPolicy, self).__init__()
        self.n_actions = n_actions
        min_action = torch.Tensor(min_action) if min_action is not None else -torch.ones(n_actions)
        max_action = torch.Tensor(max_action) if max_action is not None else torch.ones(n_actions)
        assert min_action.shape == max_action.shape and max_action.shape == torch.Size([n_actions])
        self.distr = uniform.Uniform(min_action, max_action)
            
    def log_pdf(self, s, a):
        return torch.sum(self.distr.log_prob(a))
    
    def forward(self, s, a):
        return torch.exp(self.log_pdf(s, a))
    
    def act(self, s=None, deterministic=False):
        if not deterministic:
            return self.distr.sample()
        else:
            return self.distr.mean


"""
Testing
"""
if __name__ == '__main__':
    ds = 2
    da = 2
    s = torch.ones(ds)
    a = 100 + torch.zeros(da)
    mu_init = 50 + torch.zeros(ds*da)
    for flag in [False, True]:
        p = ShallowGaussianPolicy(ds, da, None, None, mu_init, learn_std=flag, logstd_init=torch.zeros(da))
        print(p.act(s))
        print(p.num_params())
        print(p.get_flat())
        print(p.get_loc_params())
        print(p.get_scale_params())  
        print(p(s,a))
        print('...')
        p.set_loc_params((-50.,-20., 1., 1.))
        p.set_scale_params((2.,-1.))
        print(p.act(s))
        print(p.get_flat())
        print(p.log_pdf(s,a))
        print()
        ####
        q = UniformPolicy(2, [-1., 0.], [2., 4.])
        print(q.act(0))
        print(q.act(0, True))
        print(q.forward(0, torch.Tensor([-.4, 2.5])))
        print()
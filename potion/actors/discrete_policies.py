#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:26:56 2019

@author: Matteo Papini
"""

import torch
import potion.common.torch_utils as tu
from potion.common.mappings import LinearMapping
from random import randint
from torch.distributions.categorical import Categorical
from potion.actors.feature_functions import one_hot_fun, stack_fun
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box

class DiscretePolicy(tu.FlatModule):
    """Alias"""
    pass

class UniformPolicy(DiscretePolicy):
    def __init__(self, n_actions):
        super(UniformPolicy, self).__init__()
        self.n_actions = n_actions
        self.sample = lambda: randint(0, n_actions - 1)
        
    def log_pdf(self, s, a):
        return  - torch.log(torch.tensor(self.n_actions, dtype=torch.float))
    
    def forward(self, s, a):
        return torch.exp(self.log_pdf(s, a))
    
    def act(self, s=None, deterministic=False):
        if not deterministic:
            return torch.tensor(self.sample(), dtype=torch.int)
        else:
            return 0

class ShallowGibbsPolicy(DiscretePolicy):
    """
    Gibbs policy
    linear mean \mu_{\theta}(s, a)
    Fixed temperature \tau
    """
    def __init__(self, env, feature_fun=None, squash_fun=None,
                 pref_init=None, temp=1.):
        super(ShallowGibbsPolicy, self).__init__()
        if type(env.action_space) == Discrete:
            self.n_actions = env.action_space.n
        else:
            raise NotImplementedError
        if type(env.observation_space) == Discrete:
            self.n_states = env.observation_space.n
            self.feat = feature_fun if feature_fun is not None else one_hot_fun(self.n_states, self.n_actions)
            input_dim = self.feat(0, 0).shape[0]
        elif type(env.observation_space) == Box:
            self.n_states = sum(env.observation_space.shape)
            self.feat = feature_fun if feature_fun is not None else stack_fun(self.n_actions)
            input_dim = self.n_states * self.n_actions
        self.squash_fun = squash_fun
        self.temp = temp
        self.pref_init = pref_init
        # Mean
        self.pref = LinearMapping(input_dim, 1)
        self.n_params = self.pref.num_params()
        if pref_init is not None:
            self.pref.set_from_flat(pref_init)

    def log_pdf(self, s, a):
        a = tu.complete_in(a, 1)
        actions = torch.arange(0, self.n_actions) #|s| x n_a
        actions = tu.complete_in(actions, len(s.shape))
        actions = actions.repeat((1,) + s.shape[:-1])        
        states = tu.complete_out(s, 1 + len(s.shape))
        states = states.repeat((self.n_actions,) + (1,)*len(s.shape)) #|s| x n_a
        prefs = self.pref(self.feat(states, actions)).squeeze(-1) #|s| x n_a
        stab = torch.max(prefs, 0)[0]
        prefs -= stab.unsqueeze(0)
        norm = torch.log(torch.sum(torch.exp(prefs / self.temp), 0)) #|s|
        return 1. / self.temp * (self.pref(self.feat(s,a)).squeeze(-1) - stab) - norm #|s|
    
    def forward(self, s, a):
        return torch.exp(self.log_pdf(s, a))
    
    def act(self, s, deterministic=False):
        with torch.no_grad():
            states = tu.complete_out(s, 2)
            actions = torch.arange(0, self.n_actions)
            states = states.repeat((self.n_actions, 1)) # #s x n_a
            probs = self(states, actions) #|s| x n_a
            if deterministic:
                return torch.max(probs, -1)[1]
     
            a = Categorical(probs).sample()
        
            if self.squash_fun is not None:
                a = self.squash_fun(a)
            return a
     
    def score(self, s, a):
        with torch.no_grad():
            actions = torch.arange(0, self.n_actions) #|s| x n_a
            actions = tu.complete_in(actions, len(s.shape))
            actions = actions.repeat((1,) + s.shape[:-1])   #|s| x n_a
            states = tu.complete_out(s, 1 + len(s.shape)) #|s| x n_a
            states = states.repeat((self.n_actions,) + (1,)*len(s.shape))
            probs = self(states, actions) #|s| x n_a
            expect = torch.sum(self.feat(states, actions) * probs.unsqueeze(-1), 0)
            return  1. / self.temp * (self.feat(s, a.squeeze(-1)) - expect)
            
    def exploration(self):
        return torch.tensor(self.temp)
        
    def info(self):
        return {'PolicyClass': self.__class__.__name__,
                'Temperature': self.temp,
                'NStates': self.n_states,
                'NActions': self.n_actions,
                'PrefInit': self.pref_init,
                'FeatureFun': self.feat,
                'SquashFun': self.squash_fun}

"""
Testing
"""
if __name__ == '__main__':
    s = 0
    a = 1
    #s = torch.tensor([0,1,2])
    #a = torch.tensor([1,1,1])
    #pref_init = torch.zeros(6)
    #pref_init = torch.tensor([0.0, 0.1, 1.0, 1.1, 2.0, 2.1])
    pref_init = torch.tensor([0., 1., 1., 0., -1., 100.])
    pol = ShallowGibbsPolicy(3,2, pref_init=pref_init)
    o = pol(s,a)
    print(o)
    o = pol.score(s,a)
    print(o)
    print()
    lpdf = pol.log_pdf(s,a)
    print(tu.flat_gradients(pol, lpdf))
    """
    na = 3
    s = torch.zeros(1)
    a = torch.zeros(1)
    pol = UniformPolicy(na)
    for i in range(na):
        print(pol(s, a+i))
        print(pol.log_pdf(s, a+i))
    print()
    
    print(pol.act(s))
    #"""
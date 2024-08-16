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
from potion.policies.feature_functions import one_hot_fun, stack_fun
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.box import Box
from potion.common.mappings import LinearMapping, MLPMapping
from torch.distributions.gumbel import Gumbel
from torch.nn.functional import one_hot

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
    
    #STUB!!
    def entropy(self, s):
        return torch.ones(1)
        
    def info(self):
        return {'PolicyClass': self.__class__.__name__,
                'Temperature': self.temp,
                'NStates': self.n_states,
                'NActions': self.n_actions,
                'PrefInit': self.pref_init,
                'FeatureFun': self.feat,
                'SquashFun': self.squash_fun}


class DeepGibbsPolicy(DiscretePolicy):
    def __init__(self, state_dim, n_actions, 
                 hidden_neurons=[], 
                 state_preproc=None, 
                 pref_init=None, 
                 temp=1.,
                 bias=False,
                 activation=torch.tanh,
                 init=torch.nn.init.xavier_uniform_):
        super(DeepGibbsPolicy, self).__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.preproc = state_preproc
        self.pref_init = pref_init
        self.temp = temp
        
        # Preference scores
        self.pref = MLPMapping(state_dim, n_actions, 
                             hidden_neurons, 
                             bias, 
                             activation, 
                             init,
                             output_range=None)
        if pref_init is not None:
            self.pref.set_from_flat(pref_init)
            
        #Gumbel noise
        self.gumbel = Gumbel(torch.zeros(n_actions), torch.ones(n_actions))
        
        #Softmax layer
        self.softmax = torch.nn.Softmax(dim=0)
    
    def _unn_log_probs(self, s):
        with torch.no_grad():
            if self.preproc is not None:
                x = self.preproc(s)
            else:
                x = s
        return self.pref(x) / self.temp
    
    def _select(self, values, a):
        with torch.no_grad():
            assert values.shape[-1] == self.n_actions
            a = a.squeeze(-1)
            assert values.shape[:-1] == a.shape
            indexer = one_hot(a.to(torch.long))
            assert values.shape == indexer.shape
        return torch.sum(values * indexer, dim=-1)
    
    def log_pdf(self, s, a):
        unn_logps = self._unn_log_probs(s)
        assert unn_logps.shape[-1] == self.n_actions
        return self._select(unn_logps, a) - torch.log(
            torch.sum(torch.exp(unn_logps), axis=-1))
    
    def forward(self, s, a):
        exp_weights = self.softmax(self._unn_log_probs(s))
        return self._select(exp_weights, a)
    
    def act(self, s, deterministic=False):
            unn_logps = self._unn_log_probs(s)
            assert unn_logps.shape[-1] == self.n_actions
        
            if deterministic:
                action = torch.argmax(unn_logps)
            else:
                noise = self.gumbel.sample()
                action = torch.argmax(unn_logps + noise)
            
            return action
    
    def num_loc_params(self):
        return self.pref.num_params()
    
    def num_scale_params(self):
        return 1
    
    def get_loc_params(self):
        return self.pref.get_flat()
    
    def get_scale_params(self):
        return torch.tensor(self.temp)
    
    def set_loc_params(self, val):
        self.pref.set_from_flat(val)
        
    def set_scale_params(self, val):
        assert val >= 0
        self.temp = val.item()
    
    def exploration(self):
        return torch.tensor(self.temp)
    
    def entropy(self, s):
        if type(s) == float:
            s = torch.zeros(self.state_dim) + s
        probs = self.softmax(self._unn_log_probs(s))
        return - torch.dot(probs, torch.log(probs))
    
    def entropy_grad(self, s):
        return tu.flat_gradients(self, self.entropy(s))
    
    def info(self):
        return {'PolicyClass': self.__class__.__name__,
                'StateDim': self.state_dim,
                'NActions': self.n_actions,
                'PrefInit': self.pref_init,
                'Temp': self.temp,
                'StatePreprocFun': self.preproc}

"""
Testing
"""
if __name__ == '__main__':    
    dgp = DeepGibbsPolicy(4, 2, pref_init=torch.zeros(8))
    assert torch.allclose(dgp(torch.ones(4), 0), torch.tensor(0.5))
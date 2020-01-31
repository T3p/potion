#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:26:56 2019

@author: Matteo Papini
"""

import torch
import potion.common.torch_utils as tu
from potion.common.mappings import LinearMapping, MLPMapping

class ContinuousDeterministicPolicy(tu.FlatModule):
    """Alias"""
    pass

class ShallowDeterministicPolicy(ContinuousDeterministicPolicy):
    """
    Linear mapping from states to actions
    """
    def __init__(self, n_states, n_actions, 
                 feature_fun=None, 
                 squash_fun=None,
                 param_init=None,
                 bias=False, 
                 squash_grads=True):
        super(ShallowDeterministicPolicy, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.feature_fun = feature_fun
        self.squash_fun = squash_fun
        self.param_init = param_init
        self.squash_grads = squash_grads
        
        # Mean
        if feature_fun is None:
            self.linear = LinearMapping(n_states, n_actions, bias)
        else:
            self.linear = LinearMapping(len(feature_fun(torch.ones(n_states))), n_actions, bias)

        if param_init is not None:
            self.linear.set_from_flat(param_init)
          
    def forward(self, s):
        if self.feature_fun is not None:
            x = self.feature_fun(s)
        else:
            x = s
        a = self.linear(x)
        
        if self.squash_fun is not None and self.squash_grads:
            a = self.squash_fun(a)
        return a
    
    def act(self, s, noise = 0):
        with torch.no_grad():
            if self.feature_fun is not None:
                x = self.feature_fun(s)
            else:
                x = s
            
            a = self.linear(x) + noise
            if self.squash_fun is not None:
                a = self.squash_fun(a)
            return a
        
    def num_params(self):
        return self.linear.num_params()
    
    def get_params(self):
        return self.linear.get_flat()

    def set_params(self, val):
        self.linear.set_from_flat(val)
        
    def info(self):
        return {'PolicyClass': self.__class__.__name__,
                'StateDim': self.n_states,
                'ActionDim': self.n_actions,
                'ParamInit': self.param_init,
                'FeatureFun': self.feature_fun,
                'SquashFun': self.squash_fun}


class DeepDeterministicPolicy(ContinuousDeterministicPolicy):
    """
    MLP mapping from states to actions
    """
    def __init__(self, n_states, n_actions, 
                 hidden_neurons=[], 
                 feature_fun=None, 
                 squash_fun=None,
                 param_init=None,
                 bias=False,
                 activation=torch.tanh,
                 init=torch.nn.init.xavier_uniform_,
                 squash_grads=True):
        super(DeepDeterministicPolicy, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.feature_fun = feature_fun
        self.squash_fun = squash_fun
        self.param_init = param_init
        self.squash_grads = squash_grads
        
        # Mean
        self.mlp = MLPMapping(n_states, n_actions, 
                             hidden_neurons, 
                             bias, 
                             activation, 
                             init)
        if param_init is not None:
            self.mlp.set_from_flat(param_init)
        
    def forward(self, s):
        if self.feature_fun is not None:
            x = self.feature_fun(s)
        else:
            x = s
        
        a = self.mlp(x)
        if self.squash_fun is not None and self.squash_grads:
            a = self.squash_fun(a)
        return a
    
    def act(self, s, noise = 0):
        with torch.no_grad():
            if self.feature_fun is not None:
                x = self.feature_fun(s)
            else:
                x = s
            
            a = self.mlp(x) + noise
            if self.squash_fun is not None:
                a = self.squash_fun(a)
            return a
        
    def num_params(self):
        return self.mlp.num_params()
    
    def get_params(self):
        return self.mlp.get_flat()
    
    def set_params(self, val):
        self.mlp.set_from_flat(val)
        
    def info(self):
        return {'PolicyClass': self.__class__.__name__,
                'StateDim': self.n_states,
                'ActionDim': self.n_actions,
                'ParamInit': self.param_init,
                'FeatureFun': self.feature_fun,
                'SquashFun': self.squash_fun}
        

"""
Testing
"""
if __name__ == '__main__':
    from potion.common.misc_utils import seed_all_agent
    
    ds = 2
    da = 2
    s = torch.ones(ds)
    a = 1 + torch.zeros(da)
    feat = lambda s : s / 2
    squash = lambda u : torch.tanh(u)
    use_bias = False
    squash_grads = True
    seed_all_agent(0)
    
    #test linear policy
    param_init = 1 + torch.zeros((ds + use_bias) * da)
    p = ShallowDeterministicPolicy(ds, da, feat, squash, param_init, use_bias, squash_grads)
    print(p.act(s))
    print(tu.flat_gradients(p, torch.sum(p(a))))
    print(p.get_params())
    p.set_params(torch.tensor([0.1] * p.num_params()))
    print(p.get_params())
    print()
    
    #test deep policy
    dp = DeepDeterministicPolicy(ds, da, [4, 2], feat, squash, 
                                 param_init=None, 
                                 bias=use_bias, 
                                 squash_grads=squash_grads)
    print(dp.act(s))
    print(tu.flat_gradients(dp, torch.sum(dp(a))))
    print(dp.get_params())
    dp.set_params(torch.tensor([0.1] * dp.num_params()))
    print(dp.get_params())
    print()
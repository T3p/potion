#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:44:00 2018

@author: Matteo Papini

Mappings from states (or state features) to an output of interest (e.g. mean action)
"""

import torch
import torch.nn as nn
import potion.common.torch_utils as tu

class LinearMapping(tu.FlatModule):
    def __init__(self, d_in, d_out, bias=False):
        """
        Linear mapping (single fully-connected layer with linear activations)
        
        d_in: input dimension
        d_out: output dimension
        bias: whether to use a bias parameter (default: false)
        """
        super(LinearMapping, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.linear = nn.Linear(d_in, d_out, bias)
        
    def forward(self, x):
        return self.linear(x)

class MLPMapping(tu.FlatModule):
    def __init__(self, d_in, d_out, hidden_neurons, 
                 bias=False, 
                 activation=torch.tanh, 
                 init=nn.init.xavier_uniform_):
        """
        Multi-layer perceptron
        
        d_in: input dimension
        d_out: output dimension
        hidden_neurons: list with number of hidden neurons per layer
        bias: whether to use a bias parameter (default: false)
        """
        super(MLPMapping, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.activation = activation
        
        self.hidden_layers = []
        input_size = self.d_in
        for i, depth in enumerate(hidden_neurons):
            layer = nn.Linear(input_size, depth, bias)
            init(layer.weight)
            self.add_module("hidden"+str(i), layer)
            self.hidden_layers.append(layer)
            input_size = depth
        self.last = nn.Linear(input_size, self.d_out, bias)
        init(self.last.weight)
        self.add_module("last", self.last)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = self.activation(linear(x))
        return self.last(x) #final linear mapping
    
"""
Testing
"""
if __name__ == '__main__':
    import torch
    m = LinearMapping(2,2, bias=False)
    m.set_from_flat(torch.tensor([1,2,3,4]))    
    for p in m.parameters():
        print(p)
    print(m.num_params())
    print(m.get_flat())
    m.save_flat('../../logs/y.pt')
    m.load_from_flat('../../logs/x.pt')
    print(m.get_flat())
    m.load_from_flat('../../logs/y.pt')
    print(m.get_flat())
    
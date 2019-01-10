#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:44:00 2018

@author: Matteo Papini

Mappings from states (or state features) to an output of interest (e.g. mean action)
"""

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
        
    
"""
Testing
"""
if __name__ == '__main__':
    m = LinearMapping(2,2, bias=False)
    m.set_from_flat([1,2,3,4])
    for p in m.parameters():
        print(p)
    print(m.num_params())
    print(m.get_flat())
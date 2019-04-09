#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:08:26 2019

@author: matteo
"""

import torch
import potion.common.torch_utils as tu

def clip_fun(min_a, max_a):
    def clip(a):
        return torch.clamp(a, min_a, max_a)
    
    return clip

def one_hot_fun(n_s, n_a):
    def one_hot(s, a):
        s = torch.clamp(torch.tensor(s), 0, n_s - 1)
        a = torch.clamp(torch.tensor(a), 0, n_a - 1)
        assert s.shape == a.shape
        feat = torch.zeros(s.shape + (n_s * n_a,))
        indexes = (s * n_a + a)
        indexes = tu.complete_in(indexes, len(feat.shape))
        feat.scatter_(-1, indexes, 1)
        return feat
    
    return one_hot
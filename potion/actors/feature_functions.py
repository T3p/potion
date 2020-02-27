#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:08:26 2019

@author: matteo
"""

import torch
import potion.common.torch_utils as tu

def one_hot_fun(n_s, n_a):
    def one_hot(s, a):
        s = torch.clamp(torch.tensor(s, dtype=torch.int64), 0, n_s - 1)
        a = torch.clamp(torch.tensor(a, dtype=torch.int64), 0, n_a - 1)
        assert s.shape[:-1] == a.shape
        feat = torch.zeros(s.shape[:-1] + (n_s * n_a,))
        indexes = (s * n_a + a.unsqueeze(-1))
        indexes = tu.complete_in(indexes, len(feat.shape))
        feat.scatter_(-1, indexes, 1)
        return feat
    
    return one_hot

def stack_fun(n_a):
    def stack(s, a):
        a = torch.clamp(torch.tensor(a, dtype=torch.int64), 0, n_a - 1)
        s_dim = s.shape[-1]
        assert s.shape[:-1] == a.squeeze().shape
        
        s = s.unsqueeze(-1)
        s = s.repeat((1,)*(len(s.shape) - 1) + (n_a,))
        
        indexes = tu.complete_in(a, s.dim())
        indexes = indexes.repeat((1,)*(len(indexes.shape)-2) + (s_dim,) + (1,))
        mask = torch.zeros_like(s)
        mask.scatter_(-1, indexes, 1)
        feat = s * mask
        feat = feat.view(feat.shape[:-2] + (feat.shape[-2] * feat.shape[-1],))
        return feat
    
    return stack

def gauss(x, c, sigma):
    """
    isotropic
    """
    return torch.exp(-0.5 * torch.norm(x - c, dim=-1)**2 / sigma**2).unsqueeze(-1)

def rbf_fun(centers, sigmas):
    def rbf(s):
        return torch.cat([gauss(s, c, sigma) for (c, sigma) in zip(centers, sigmas)], dim=-1)
    
    return rbf

def poly_fun(order, bias=True, normalization=1.):
    powers = range(order + 1)
    if not bias:
        powers = powers[1:]
    
    def poly(s):
        if s.shape[-1] == 1:
            return torch.cat([(s / normalization)**i for i in powers], dim=-1)
        else:
            raise NotImplementedError
    
    return poly
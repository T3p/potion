#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 23:12:09 2019

@author: Matteo Papini
"""

import random
import numpy as np
import torch
import os

def clip(env):
    def action_filter(a):
        return np.clip(a, env.action_space.low, env.action_space.high)
    return lambda a : action_filter(a)

def seed_all_agent(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def unpack(batch):
    "Unpacks list of tuples of tensors into one tuple of stacked arrays"
    return (torch.stack(x) for x in zip(*batch))

def discount(rewards, disc):
    """rewards: array or tensor"""
    i = 0 if rewards.dim() < 2 else 1
    discounts = torch.tensor(disc**np.indices(rewards.shape)[i], dtype=torch.float)
    return rewards * discounts
        
def returns(batch, gamma):
    return [torch.sum(discount(rewards,gamma)).item() 
                                    for (_, _, rewards, _, _) in batch]

def mean_sum_info(batch):
    return torch.mean(torch.tensor([torch.sum(inf).item() 
                for (_, _, _, _, inf) in batch]))

def performance(batch, disc):
    return torch.mean(torch.tensor(returns(batch, disc))).item()

def avg_horizon(batch):
    return torch.mean(torch.tensor([torch.sum(mask)
                       for (_, _, _, mask, _) in batch], dtype=torch.float)).item()
        
def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
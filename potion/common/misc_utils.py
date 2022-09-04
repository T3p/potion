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
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

separator = '---------------------------------------------------------'

def clip(env):
    if type(env.action_space) is Box:
        low = env.action_space.low
        high = env.action_space.high
    elif type(env.action_space) is Discrete:
        low = 0
        high = env.action_space.n
    
    def action_filter(a):
        return np.clip(a, low, high)
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

def unpack_multi(batches):
    n_policies = len(batches) #J
    sizes = [len(batch) for batch in batches]
    max_size = max(sizes) #N

    states, actions, rewards, mask, info  = unpack((t if t.shape[0] == max_size else 
                       torch.nn.functional.pad(t, 
                                               (0, 0) * (t.dim() - 1) + (0, max_size - t.shape[0]), #pad sequence
                                               "constant", 
                                               0.
                       ) for t in unpack(batch)
                  ) for batch in batches
                 ) #JxNxHxds, JxNxHxda, JxNxH, JxNxH, JxNxH
    
    policy_mask = torch.stack(tuple(torch.cumsum(torch.ones(max_size), 0) <= sizes[i]
                              for i in range(n_policies))).unsqueeze(-1) #JxNx1
    
    return states, actions, rewards, policy_mask * mask, info #JxNxHxds, JxNxHxda, JxNxH, JxNxH, JxNxH
    

def discount(rewards, disc):
    """rewards: array or tensor"""
    discounts = torch.tensor(disc**np.indices((rewards.shape[-1],))[0], dtype=torch.float)
    return torch.einsum("...i,i->...i", (rewards, discounts))
        
def returns(batch, gamma):
    return [torch.sum(discount(rewards,gamma)).item() 
                                    for (_, _, rewards, _, _) in batch]
def max_reward(batch):
    return max(torch.max(torch.abs(rewards)).item() for (_, _, rewards, _, _) in batch)

def max_feature(batch):
    return max(torch.max(torch.abs(states)).item() for (states, _, _, _, _) in batch)

def mean_sum_info(batch):
    return torch.mean(torch.tensor([torch.sum(inf).item() 
                for (_, _, _, _, inf) in batch]))

def performance(batch, disc):
    return torch.mean(torch.tensor(returns(batch, disc))).item()

def performance_lcb(batch, disc, max_rew, delta, horizon=None):
    n = len(batch)
    if horizon is not None:
        time_factor = (1 - disc**horizon) / (1 - disc)
    else:
        time_factor = 1. / (1 - disc)
    std, mean = torch.std_mean(torch.tensor(returns(batch, disc)), 
                               unbiased=True)
    lcb = mean.item() - std.item() * np.sqrt(2 * np.log(2 / delta) / n) - (7 * max_rew * 
            time_factor * np.log(2 / delta) / (3 * (n - 1)))
    return lcb

def performance_ucb(batch, disc, max_rew, delta, horizon=None):
    n = len(batch)
    if horizon is not None:
        time_factor = (1 - disc**horizon) / (1 - disc)
    else:
        time_factor = 1. / (1 - disc)
    std, mean = torch.std_mean(torch.tensor(returns(batch, disc)), 
                               unbiased=True)
    ucb = mean.item() + std.item() * np.sqrt(2 * np.log(2 / delta) / n) + (7 * max_rew * 
            time_factor * np.log(2 / delta) / (3 * (n - 1)))
    return ucb

def avg_horizon(batch):
    return torch.mean(torch.tensor([torch.sum(mask)
                       for (_, _, _, mask, _) in batch], dtype=torch.float)).item()
        
def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
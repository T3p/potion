#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:05:00 2019

@author: Matteo Papini
"""

import torch
import potion.common.torch_utils as tu
from potion.common.misc_utils import unpack, discount

def gpomdp_sample(states, actions, values, mask, policy):
    logps = policy.log_pdf(states, actions)
    cum_logps = torch.cumsum(logps, 0).view(-1)
    return tu.flat_gradients(policy, cum_logps, values*mask)

def gpomdp_estimator(batch, gamma, policy, baseline_kind='basic'):
    N = len(batch)
    
    states, actions, rewards, mask = unpack(batch)
    disc_rewards = discount(rewards, gamma)
    
    if baseline_kind == 'basic':
        baseline = torch.mean(disc_rewards, 0)
    else:
        baseline = 0
    values = disc_rewards - baseline
    
    return torch.mean(torch.stack([gpomdp_sample(states[i,:,:], actions[i,:,:], 
                                                 values[i,:], mask[i,:], 
                                                 policy) 
                                   for i in range(N)], 0), 0)
    

"""Testing"""
if __name__ == '__main__':
    from potion.actors.continuous_policies import SimpleGaussianPolicy as Gauss
    from potion.simulation.trajectory_generators import generate_batch
    import gym.spaces
    env = gym.make('MountainCarContinuous-v0')
    N = 4
    H = 3
    gamma = 0.99
    pol = Gauss(2,1, mu_init=[1.,1.])
    
    batch = generate_batch(env, pol, H, N)
    o = gpomdp_estimator(batch, gamma, pol)
    print(o)
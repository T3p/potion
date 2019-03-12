#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:05:00 2019
"""

import torch
import potion.common.torch_utils as tu
from potion.common.misc_utils import unpack, discount
from potion.common.torch_utils import tensormat

def gpomdp_sample(states, actions, values, mask, policy):
    """
    s: Hxm
    a: Hxd
    values, mask: H
    """
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

def simple_gpomdp_estimator(batch, gamma, policy, baseline_kind='peters', result='mean'):
    with torch.no_grad():        
        states, actions, rewards, mask = unpack(batch) # NxHxm, NxHxd, NxH, NxH
        
        scores = torch.cat((policy.omega_score(states, actions),
                           policy.theta_score(states, actions)), 
                           2) # NxHx(m+1)
        G = torch.cumsum(scores, 1)
        
        disc_rewards = discount(rewards, gamma)# NxH
        if baseline_kind == 'basic':
            baseline = torch.mean(disc_rewards, 0)
        elif baseline_kind == 'peters':
            baseline = torch.mean(tensormat(G ** 2, disc_rewards), 0) / torch.mean(G ** 2, 0)
        else:
            baseline = 0
        baseline[baseline != baseline] = 0
        values = disc_rewards.unsqueeze(2) - baseline.unsqueeze(0)
        
        G = tensormat(G, mask)
        _samples = torch.sum(G * values, 1)
        if result == 'samples':
            return _samples #Nxm
        elif result == 'moments':
            _mean = torch.mean(_samples, 0)
            _variance = torch.var(_samples, 0, unbiased=True)
            return _mean, _variance
        else:
            _mean = torch.mean(_samples, 0)
            return _mean
            
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
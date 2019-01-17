#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:57:23 2019

@author: matteo
"""

import torch
import math
from potion.estimation.gradients import simple_gpomdp_estimator
from potion.common.misc_utils import unpack, discount
from potion.common.torch_utils import tensormat



def h_estimator(states, actions, disc_rewards, mask, policy, baseline_kind):
    """
    states: NxHxm
    actions: NxHx1
    disc_rewards, mask: NxH
    """
    sigma = math.exp(policy.get_scale_params().item())
    theta_scores = policy.theta_score(states, actions) # NxHxm
    G = torch.cumsum(theta_scores, 1) # NxHxm
    sigma_scores = policy.omega_score(states, actions).squeeze() / sigma #NxH
    H = torch.cumsum(sigma_scores, 1) # NxH
    
    if baseline_kind == 'basic':
        baseline = torch.mean(disc_rewards, 0)
    elif baseline_kind == 'peters':
        baseline = torch.mean(tensormat(G, H * disc_rewards), 0) / torch.mean(tensormat(G, H), 0)
    else:
        baseline = 0
    baseline[baseline != baseline] = 0
    values = disc_rewards.unsqueeze(2) - baseline.unsqueeze(0)
    
    G = tensormat(G, mask)
    terms = tensormat(G * values, H) # NxHxm
    samples = torch.sum(terms, 1) # Nxm
    return torch.mean(samples, 0) # m
    
def mixed_estimator(batch, gamma, policy, baseline_kind='peters'):
    grad = simple_gpomdp_estimator(batch, gamma, policy, baseline_kind)
    theta_grad = grad[1:]
    with torch.no_grad():
        sigma = math.exp(policy.get_scale_params().item())
        
        states, actions, rewards, mask = unpack(batch)
        
        disc_rewards = discount(rewards, gamma)
            
        h = h_estimator(states, actions, disc_rewards, mask, policy, baseline_kind)
        mixed_sigma = h - 2 * theta_grad / sigma
        return mixed_sigma * sigma, theta_grad

def omega_metagradient_estimator(batch, gamma, policy, alpha, baseline_kind='peters'):                
    mixed, theta_grad = mixed_estimator(batch, gamma, policy, baseline_kind)
    return 2 * mixed.dot(alpha * theta_grad)
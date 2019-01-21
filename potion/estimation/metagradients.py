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



def h_estimator(states, actions, disc_rewards, mask, policy, baseline_kind, result='mean'):
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
    if result == 'samples':
        return samples #Nxm
    elif result == 'moments':
        return torch.mean(samples, 0), torch.var(samples, 0, unbiased=True)
    else:
        return torch.mean(samples, 0) # m
        
    
def mixed_estimator(batch, gamma, policy, baseline_kind='peters', theta_grad=None):
    if theta_grad is None:
        grad = simple_gpomdp_estimator(batch, gamma, policy, baseline_kind)
        theta_grad = grad[1:]
    with torch.no_grad():
        sigma = math.exp(policy.get_scale_params().item())
        
        states, actions, rewards, mask = unpack(batch)
        
        disc_rewards = discount(rewards, gamma)
            
        h = h_estimator(states, actions, disc_rewards, mask, policy, baseline_kind)
        mixed_sigma = h - 2 * theta_grad / sigma
        return mixed_sigma, theta_grad

def metagrad(batch, gamma, policy, alpha, clip_at, baseline_kind='peters', result='mean'):                
    grad = simple_gpomdp_estimator(batch, gamma, policy, baseline_kind, result='samples') #Nx(m+1)
    with torch.no_grad():
        sigma = math.exp(policy.get_scale_params().item()) #float
        
        theta_grad = grad[:, 1:] #Nxm
        omega_grad = grad[:, 0] #N
        
        states, actions, rewards, mask = unpack(batch)
        disc_rewards = discount(rewards, gamma)
        h = h_estimator(states, actions, disc_rewards, mask, policy, baseline_kind, result='samples') #Nxm
        
        mixed = h - 2 * theta_grad / sigma #Nxm
        norm_grad = 2 * torch.bmm(theta_grad.unsqueeze(1), mixed.unsqueeze(2)).view(-1) #N
        A = omega_grad #N
        B = 2 * alpha * torch.bmm(theta_grad.unsqueeze(1), theta_grad.unsqueeze(2)).view(-1) #N
        C = sigma * alpha * norm_grad #N
        C = torch.clamp(C, min=-clip_at, max=clip_at) #N
        samples = A + B + C #N
        if result == 'samples':
            return samples
        elif result == 'moments':
            return torch.mean(samples, 0), torch.var(samples, 0, unbiased=True)
        else:
            return torch.mean(samples, 0)
    
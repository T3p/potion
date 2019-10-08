#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:57:23 2019

@author: matteo
"""

import torch
from potion.estimation.gradients import gpomdp_estimator
from potion.common.misc_utils import unpack, discount
from potion.common.torch_utils import tensormat



def mix_estimator(states, actions, disc_rewards, mask, policy, result='mean'):
    """
    states: NxHxm
    actions: NxHx1
    disc_rewards, mask: NxH
    """
    upsilon_scores = policy.loc_score(states, actions) #NxHxm
    G = torch.cumsum(upsilon_scores, 1) #NxHxm
    sigma_scores = policy.scale_score(states, actions).squeeze() #NxH
    H = torch.cumsum(sigma_scores, 1) #NxH
    
    baseline = torch.mean(tensormat(G, H * disc_rewards), 0) / torch.mean(tensormat(G, H), 0)
    baseline[baseline != baseline] = 0
    
    values = disc_rewards.unsqueeze(2) - baseline.unsqueeze(0)
    
    G = tensormat(G, mask)
    terms = tensormat(G * values, H) #NxHxm
    samples = torch.sum(terms, 1) #Nxm
    if result == 'samples':
        return samples #Nxm
    else:
        return torch.mean(samples, 0) #m
        
def metagrad(batch, disc, policy, alpha, result='mean', grad_samples=None,
                 no_first=False, no_second=False, no_third=False):
    sigma = torch.exp(policy.get_scale_params())
    
    if grad_samples is None:                
        grad_samples = gpomdp_estimator(batch, disc, policy, 
                                    baselinekind='peters', 
                                    shallow=True,
                                    result='samples') #Nx(m+1)
    with torch.no_grad():
        upsilon_grad = grad_samples[:, 1:] #Nxm
        omega_grad = grad_samples[:, 0] #N
        
        states, actions, rewards, mask, _ = unpack(batch)
        disc_rewards = discount(rewards, disc)
        
        mix = mix_estimator(states, actions, disc_rewards, mask, policy, result='samples') #Nxm
        mixed_der = mix - 2 * upsilon_grad #Nxm
        grad_norm = torch.sqrt(torch.bmm(upsilon_grad.unsqueeze(1), upsilon_grad.unsqueeze(2)).view(-1))
        norm_grad = torch.bmm(upsilon_grad.unsqueeze(1), mixed_der.unsqueeze(2)).view(-1) / grad_norm #N
        A = omega_grad #N
        B = 2 * alpha * sigma**2 * grad_norm #N
        C = alpha * sigma**2 * norm_grad #N
        samples = A * (1 - no_first) + B * (1 - no_second) + C * (1 - no_third) #N
        if result == 'samples':
            return samples
        else:
            return torch.mean(samples, 0)
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:55:39 2019

@author: matteo
"""
import torch
from potion.common.misc_utils import unpack

def importance_weights(batch, policy, target_params, normalize=False):
    #Samples
    states, actions, _, mask, _ = unpack(batch) #NxHx*
    
    #Proposal log-probability
    proposal = policy.log_pdf(states, actions) #NxH
        
    #Target log-probability
    params = policy.get_flat()
    policy.set_from_flat(target_params)
    target = policy.log_pdf(states, actions) #NxH
        
    #Restore proposal
    policy.set_from_flat(params)
    
    #Importance weights
    iws = torch.exp(torch.sum((target - proposal) * mask, 1)) #N
    
    #Self-normalization
    if normalize:
        iws /= torch.mean(iws)
    
    return iws

"""Testing"""
if __name__ == '__main__':
    from potion.actors.continuous_policies import ShallowGaussianPolicy as Gauss
    from potion.simulation.trajectory_generators import generate_batch
    from potion.common.misc_utils import seed_all_agent
    import potion.envs
    import gym.spaces
    env = gym.make('ContCartPole-v0')
    env.seed(0)
    seed_all_agent(0)
    N = 100
    H = 100
    disc = 0.99
    pol = Gauss(4,1, mu_init=[0.,0.,0.,0.], learn_std=True)
    
    batch = generate_batch(env, pol, H, N)
    print(importance_weights(batch, pol, pol.get_flat()))
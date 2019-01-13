#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 18:11:15 2019

@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch, performance, avg_horizon
from potion.estimation.gradients import gpomdp_estimator

def gpomdp(env, policy, horizon, 
           batch_size = 100, 
           iterations = 1000,
           gamma = 0.99,
           step_size = 1e-4,
           annealing = None,
           baseline = 'basic',
           action_filter = None,
           parallel_sim = False,
           parallel_comp = False,
           verbose = 1):
    it = 0
    while(it < iterations):
        it += 1
        if verbose:
            print('\nIteration ', it)
        
        if verbose > 1:
            print('Params: ', policy.get_flat())
    
        # Simulation
        batch = generate_batch(env, policy, horizon, batch_size, action_filter, parallel_sim)
        J = performance(batch, gamma)
        if verbose:
            print('Performance J: ', J)
            print('Horizon: ', avg_horizon(batch))
        
        # Estimation
        grad = gpomdp_estimator(batch, gamma, policy, baseline)
        if verbose > 1:
            print('Gradients: ', grad)
        
        # Update
        if annealing is not None:
            alpha = step_size * annealing(it)
        else:
            alpha = step_size
        params = policy.get_flat()
        new_params = params + alpha * grad
        policy.set_from_flat(new_params)
        

"""Testing"""
if __name__ == '__main__':
    import torch
    import gym
    import potion.envs.lqg1d
    import potion.envs.cartpole
    from potion.actors.continuous_policies import SimpleGaussianPolicy as Gauss
    from potion.common.gym_utils import clip
    import math
    #""""
    env = gym.make('LQG1D-v0')
    policy = Gauss(1, 1, mu_init=[0.], learn_std=True)
    H = 10
    step_size = 1e-3
    annealing = None
    """
    env = gym.make('ContCartPole-v0')
    policy = Gauss(4,1, mu_init=torch.zeros(4), learn_std=False)
    H = 500
    step_size = 1.
    annealing = lambda t: 1./math.sqrt(t)
    #"""
    
    gpomdp(env, policy, H, verbose=2, action_filter=clip(env), 
           step_size=step_size,
           annealing=annealing)
        
        
        


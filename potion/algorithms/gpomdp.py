#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 18:11:15 2019

@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon
from potion.estimation.gradients import gpomdp_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent

def gpomdp(env, policy, horizon,
           batch_size = 100, 
           iterations = 1000,
           gamma = 0.99,
           alpha = 1e-4,
           seed = None,
           annealing = None,
           baseline = 'basic',
           action_filter = None,
           logger = Logger(name='gpomdp'),
           save_params = 1000,
           log_params = True,
           parallel_sim = False,
           parallel_comp = False,
           verbose = True):
    """
        G(PO)MDP algorithm
    """
    # Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    # Seeding agent
    if seed is not None:
        seed_all_agent(seed)
    
    # Preparing logger
    algo_info = {'Algorithm': 'gpomdp', 'Env': str(env), 
                       'BatchSize': batch_size, 'alpha': alpha, 
                       'gamma': gamma, 'Annealing': annealing, 'seed': seed,
                       'actionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 'StepSize', 'BatchSize']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    log_row = dict.fromkeys(log_keys)

    logger.open(log_row.keys())
    
    # Learning
    it = 0
    while(it < iterations):
        # Begin iteration
        if verbose:
            print('\nIteration ', it)
        if verbose:
            print('Params: ', policy.get_flat())
    
        # Simulation
        batch = generate_batch(env, policy, horizon, batch_size, action_filter, parallel_sim)
        log_row['Perf'] = performance(batch, gamma)
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
    
        # Estimation
        grad = gpomdp_estimator(batch, gamma, policy, baseline)
        if verbose > 1:
            print('Gradients: ', grad)
        
        # Meta-parameters
        if annealing is not None:
            step_size = alpha * annealing(it)
        else:
            step_size = alpha
        log_row['StepSize'] = step_size
        log_row['BatchSize'] = batch_size
        
        # Update policy parameters
        params = policy.get_flat()
        new_params = params + step_size * grad
        policy.set_from_flat(new_params)
        
        # Log
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i]
        logger.write_row(log_row, it)
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        # Next iteration
        it += 1
    
    # Final policy
    if save_params:
        logger.save_params(params, it)
    
    # Cleanup
    logger.close()
        

"""Testing"""
if __name__ == '__main__':
    import torch
    import gym
    import potion.envs.lqg1d
    import potion.envs.cartpole
    from potion.actors.continuous_policies import SimpleGaussianPolicy as Gauss
    import math
    """"
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
    annealing = lambda t: 1./math.sqrt(t+1)
    #"""
    
    logger = Logger(directory='../../logs', name='test_gpomdp')
    seed = 0
    env.seed(seed)
    gpomdp(env, policy, H, verbose=True, 
           seed=seed,
           alpha=step_size,
           annealing=annealing,
           logger=logger,
           iterations=100)
        
        
        


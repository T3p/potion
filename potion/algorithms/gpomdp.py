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
           batchsize = 100, 
           iterations = 1000,
           gamma = 0.99,
           alpha = 1e-4,
           seed = None,
           decay = None,
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
                       'BatchSize': batchsize, 'alpha': alpha, 
                       'gamma': gamma, 'Decay': decay, 'seed': seed,
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
        batch = generate_batch(env, policy, horizon, batchsize, action_filter, parallel_sim)
        log_row['Perf'] = performance(batch, gamma)
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
    
        # Estimation
        grad = gpomdp_estimator(batch, gamma, policy, baseline)
        if verbose > 1:
            print('Gradients: ', grad)
        
        # Meta-parameters
        if decay is not None:
            stepsize = alpha * decay(it)
        else:
            stepsize = alpha
        log_row['StepSize'] = stepsize
        log_row['BatchSize'] = batchsize
        
        # Update policy parameters
        params = policy.get_flat()
        new_params = params + stepsize * grad
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


def gpomdp_adaptive(env, policy, horizon,
                    batchsize = 100, 
                    iterations = 1000,
                    gamma = 0.99,
                    stepsize_rule = 'RMSprop',
                    seed = None,
                    baseline = 'basic',
                    action_filter = None,
                    logger = Logger(name='gpomdp_adaptive'),
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
                       'BatchSize': batchsize, 
                       'gamma': gamma, 'StepSizeCriterion': stepsize_rule, 'seed': seed,
                       'actionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 'BatchSize']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
        log_keys += ['stepsize%d' % i for i in range(policy.num_params())]
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
        batch = generate_batch(env, policy, horizon, batchsize, action_filter, parallel_sim)
        log_row['Perf'] = performance(batch, gamma)
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
    
        # Estimation
        grad = gpomdp_estimator(batch, gamma, policy, baseline)
        if verbose > 1:
            print('Gradients: ', grad)
        
        # Meta-parameters
        
        stepsize = stepsize_rule.next(grad)
        
        log_row['BatchSize'] = batchsize
        
        # Update policy parameters
        params = policy.get_flat()
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
        
        # Log
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i]
                log_row['stepsize%d' % i] = stepsize[i]
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
    from potion.meta.step_sizes import RMSprop
    """"
    env = gym.make('LQG1D-v0')
    policy = Gauss(1, 1, mu_init=[0.], learn_std=True)
    H = 10
    stepsize = 1e-3
    decay = None
    """
    env = gym.make('ContCartPole-v0')
    policy = Gauss(4,1, mu_init=torch.zeros(4), learn_std=True)
    H = 500
    stepsize_rule = RMSprop(alpha=0.5)
    #"""
    
    logger = Logger(directory='../../logs', name='test_gpomdp_adaptive')
    seed = 0
    env.seed(seed)
    gpomdp_adaptive(env, policy, H, verbose=True, 
           seed=seed,
           stepsize_rule=stepsize_rule,
           logger=logger,
           iterations=100)
        
        
        


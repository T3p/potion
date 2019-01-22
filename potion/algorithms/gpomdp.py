#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 18:11:15 2019

@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon
from potion.estimation.gradients import gpomdp_estimator, simple_gpomdp_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
from potion.meta.steppers import ConstantStepper
import torch

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
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 'StepSize', 'BatchSize', 'Exploration']
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
        log_row['Exploration'] = policy.exploration()
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
                    stepper = ConstantStepper(1e-1),
                    seed = None,
                    baseline = 'peters',
                    simple = True,
                    action_filter = None,
                    test_det = True,
                    logger = Logger(name='gpomdpadaptive'),
                    save_params = 1000,
                    log_params = True,
                    parallel = False,
                    n_jobs = 4,
                    render = False,
                    verbose = True):
    """
        G(PO)MDP algorithm + adaptive step size (e.g., RMSprop)
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
                       'gamma': gamma, 'StepSizeCriterion': stepper, 'seed': seed,
                       'actionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 'StepSize', 'BatchSize', 'Exploration', 'DetPerf', 'ThetaGradNorm']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if policy.learn_std:
        log_keys += ['MetaStepSize', 'OmegaGrad']
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
            
        
        # Test
        if test_det:
            omega = policy.get_scale_params()
            policy.set_scale_params(-100.)
            batch = generate_batch(env, policy, horizon, 1, action_filter)
            policy.set_scale_params(omega)
            log_row['DetPerf'] = performance(batch, gamma)
        if render:
            batch = generate_batch(env, policy, horizon, 1, action_filter, render=True)
    
        # Simulation
        batch = generate_batch(env, policy, horizon, batchsize, action_filter, seed=seed, parallel=parallel, n_jobs=n_jobs)
        log_row['Perf'] = performance(batch, gamma)
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
    
        # Estimation
        if simple:
            grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
        else:
            grad = gpomdp_estimator(batch, gamma, policy, baseline)
        if verbose > 1:
            print('Gradients: ', grad)
        log_row['ThetaGradNorm'] = torch.norm(grad[1:]).item() if policy.learn_std else torch.norm(grad).item()
        
        # Meta-parameters
        
        stepsize = stepper.next(grad)
        
        log_row['BatchSize'] = batchsize
        log_row['Exploration'] = policy.exploration()
        if policy.learn_std:
            log_row['StepSize'] = torch.norm(torch.tensor(stepsize)).item()
            log_row['MetaStepSize'] = torch.tensor(stepsize).item()
            log_row['OmegaGrad'] = grad[0].item()
        else:
            grad = grad[1:]
            log_row['StepSize'] = torch.norm(torch.tensor(stepsize)).item()
        
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

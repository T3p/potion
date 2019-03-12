#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:18:45 2019
"""
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon
from potion.estimation.gradients import simple_gpomdp_estimator
from potion.estimation.metagradients import mixed_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
import torch

def mepg(env, policy, 
            horizon,
            batchsize = 100, 
            iterations = 1000,
            gamma = 0.99,
            alpha = 1e-2,
            eta = 1e-3,
            clip_at = 100,
            test_det = True,
            render = False,
            seed = None,
            baseline = 'peters',
            action_filter = None,
            parallel = False,
            n_jobs = 4,
            logger = Logger(name='mepgtest'),
            save_params = 1000,
            log_params = True,
            verbose = True):
    """
        Only for SIMPLE Gaussian policy w/ scalar variance
        Policy must have learn_std = False, as std is META-learned
    """
        
    # Defaults
    assert policy.learn_std
    if action_filter is None:
        action_filter = clip(env)
    
    # Seeding agent
    if seed is not None:
        seed_all_agent(seed)
    
    # Preparing logger
    algo_info = {'Algorithm': 'SUNDAY', 
                 'Environment': str(env), 
                 'BatchSize': batchsize, 
                 'Max horizon': horizon,
                 'Iterations': iterations,
                 'gamma': gamma, 
                 'alpha': alpha,
                 'eta': eta, 'seed': seed,
                 'actionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 
                'Alpha', 'Eta', 'BatchSize', 'Exploration', 
                'OmegaGrad', 'OmegaMetagrad', 'ThetaGradNorm',
                'IterationKind', 'A', 'B', 'C'] #0: theta, 1: omega
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_det:
        log_keys.append('DetPerf')
    log_row = dict.fromkeys(log_keys)

    logger.open(log_row.keys())
    
    # Learning
    omega_metagrad = torch.tensor(float('nan'))
    A  = torch.tensor(float('nan'))
    B  = torch.tensor(float('nan'))
    C  = torch.tensor(float('nan'))
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
            generate_batch(env, policy, horizon, 1, action_filter, render=True)


        omega = policy.get_scale_params()
        sigma = torch.exp(omega)
        log_row['IterationKind'] = 3
        log_row['Exploration'] = policy.exploration()
        log_row['Alpha'] = (alpha * sigma**2).item()
        log_row['Eta'] = eta
        batch = generate_batch(env, policy, horizon, batchsize, action_filter, parallel=parallel, n_jobs=n_jobs, seed=seed)
        grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
        theta_grad = grad[1:]
        omega_grad = grad[0]
        mixed, _ = mixed_estimator(batch, gamma, policy, baseline, theta_grad)
        norm_grad = 2 * theta_grad.dot(mixed)
        A = omega_grad
        B = 2 * sigma**2 * alpha * torch.norm(theta_grad)**2
        C = sigma**3 * alpha * norm_grad
        C = torch.clamp(C, min=-clip_at, max=clip_at)
        omega_metagrad = A + B + C
        
        theta = policy.get_loc_params()
        new_theta = theta + alpha * sigma**2 * theta_grad
        policy.set_loc_params(new_theta)
        
        new_omega = omega + eta * omega_metagrad
        policy.set_scale_params(new_omega)

        log_row['OmegaGrad'] = omega_grad.item()
        log_row['OmegaMetagrad'] = omega_metagrad.item()
        log_row['A'] = A.item()
        log_row['B'] = B.item()
        log_row['C'] = C.item()

        # Log
        log_row['ThetaGradNorm'] = torch.norm(theta_grad).item()
        log_row['BatchSize'] = batchsize
        log_row['Perf'] = performance(batch, gamma)
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        params = policy.get_flat()
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        logger.write_row(log_row, it)
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        # Next iteration
        it += 1
    
    # Final policy
    if save_params:
        logger.save_params(params, it)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:18:45 2019

@author: matteo
"""
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon
from potion.estimation.gradients import simple_gpomdp_estimator
from potion.meta.steppers import ConstantStepper, RMSprop
from potion.estimation.metagradients import omega_metagradient_estimator, mixed_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
import torch
import math

def metaexplore(env, policy, 
                    horizon,
                    batchsize = 100, 
                    iterations = 1000,
                    gamma = 0.99,
                    stepper = ConstantStepper(1e-3),
                    metastepper = ConstantStepper(1e-3),
                    explore = 'balanced',
                    test_det = True,
                    render = False,
                    seed = None,
                    baseline = 'peters',
                    action_filter = None,
                    logger = Logger(name='gpomdp_metasigma'),
                    save_params = 1000,
                    log_params = True,
                    parallel_sim = False,
                    parallel_comp = False,
                    verbose = True):
    """
        Only for SIMPLE Gaussian policy w/ scalar variance
        Policy must have learn_std = False, as std is META-learned
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
                       'gamma': gamma, 'Stepper': stepper,
                       'MetaStepper': metastepper, 'seed': seed,
                       'actionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 'StepSize', 'MetaStepSize', 'BatchSize', 'Exploration', 
                'OmegaGrad', 'OmegaMetagrad', 'ThetaGradNorm']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_det:
        log_keys.append('DetPerf')
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
            batch = generate_batch(env, policy, horizon, 1, action_filter, parallel_sim)
            policy.set_scale_params(omega)
            log_row['DetPerf'] = performance(batch, gamma)
        if render:
            batch = generate_batch(env, policy, horizon, 1, action_filter, parallel_sim, render=True)

        # Simulation
        batch = generate_batch(env, policy, horizon, batchsize, action_filter, parallel_sim)
        log_row['BatchSize'] = batchsize
        log_row['Exploration'] = policy.exploration()
        log_row['Perf'] = performance(batch, gamma)
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        params = policy.get_flat()
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i]
    
        # Gradient estimation
        grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
        theta_grad = grad[1:]
        omega_grad = grad[0]
        log_row['ThetaGradNorm'] = torch.norm(theta_grad).item()
        log_row['OmegaGrad'] = omega_grad.item()

        # Step size
        alpha = stepper.next(theta_grad)
        log_row['StepSize'] = torch.norm(alpha).item()
        
        # Metagradient estimation
        omega_metagrad = omega_metagradient_estimator(batch, gamma, policy, alpha)
        log_row['OmegaMetagrad'] = omega_metagrad.item()
        log_row['BalancedMetagrad'] = omega_metagrad.item() + omega_grad.item()
                
        # Meta step size
        eta = metastepper.next(omega_grad)
        log_row['MetaStepSize'] = torch.norm(eta).item()
        
        # Update policy mean parameters
        theta = policy.get_loc_params()
        new_theta = theta + alpha * theta_grad
        policy.set_loc_params(new_theta)
        
        # Update policy variance parameter
        omega = policy.get_scale_params()
        if explore == 'balanced':
            new_omega = omega + eta * (omega_grad + omega_metagrad)
        elif explore == 'greedy':
            new_omega = omega + eta * omega_grad
        elif explore == 'meta':
            new_omega = omega + eta * omega_metagrad
        elif explore == 'none':
            new_omega = omega
        else:
            raise ValueError
        policy.set_scale_params(new_omega)
                
        # Log
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
    
def metaexplore2(env, policy, 
                    horizon,
                    batchsize = 100, 
                    iterations = 1000,
                    gamma = 0.99,
                    stepper = ConstantStepper(1e-3),
                    metastepper = ConstantStepper(1e-3),
                    explore = 'balanced',
                    test_det = True,
                    render = False,
                    seed = None,
                    baseline = 'peters',
                    action_filter = None,
                    logger = Logger(name='gpomdp_metasigma'),
                    save_params = 1000,
                    log_params = True,
                    parallel_sim = False,
                    parallel_comp = False,
                    verbose = True):
    """
        Only for SIMPLE Gaussian policy w/ scalar variance
        Policy must have learn_std = False, as std is META-learned
    """
        
    # Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    # Seeding agent
    if seed is not None:
        seed_all_agent(seed)
    
    # Preparing logger
    algo_info = {'Algorithm': 'metaExplore2', 'Env': str(env), 
                       'BatchSize': batchsize, 
                       'gamma': gamma, 'Stepper': stepper,
                       'MetaStepper': metastepper, 'seed': seed,
                       'actionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 'StepSize', 'MetaStepSize', 'BatchSize', 'Exploration', 
                'OmegaGrad', 'OmegaMetagrad', 'ThetaGradNorm']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_det:
        log_keys.append('DetPerf')
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
            batch = generate_batch(env, policy, horizon, 1, action_filter, parallel_sim)
            policy.set_scale_params(omega)
            log_row['DetPerf'] = performance(batch, gamma)
        if render:
            batch = generate_batch(env, policy, horizon, 1, action_filter, parallel_sim, render=True)

        # Simulation
        batch = generate_batch(env, policy, horizon, batchsize, action_filter, parallel_sim)
        log_row['BatchSize'] = batchsize
        log_row['Exploration'] = policy.exploration()
        log_row['Perf'] = performance(batch, gamma)
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        params = policy.get_flat()
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i]
    
        # Gradient estimation
        grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
        theta_grad = grad[1:]
        omega_grad = grad[0]
        log_row['ThetaGradNorm'] = torch.norm(theta_grad).item()
        log_row['OmegaGrad'] = omega_grad.item()

        # Step size
        sigma = torch.exp(policy.get_scale_params())
        alpha_0 = stepper.next(theta_grad)
        alpha = alpha_0 * sigma ** 2
        log_row['StepSize'] = torch.norm(alpha).item()
        
        # Metagradient estimation
        mixed = mixed_estimator(batch, gamma, policy, baseline)[0]
        sigma_metagrad = 2 * sigma * alpha_0 * torch.norm(theta_grad) ** 2 + \
            sigma ** 2 * alpha_0 * theta_grad.dot(mixed)
        omega_metagrad = sigma_metagrad * sigma
        log_row['OmegaMetagrad'] = omega_metagrad.item()
        log_row['BalancedMetagrad'] = omega_metagrad.item() + omega_grad.item()
                
        # Meta step size
        eta = metastepper.next(omega_grad)
        log_row['MetaStepSize'] = torch.norm(eta).item()
        
        # Update policy mean parameters
        theta = policy.get_loc_params()
        new_theta = theta + alpha * theta_grad
        policy.set_loc_params(new_theta)
        
        # Update policy variance parameter
        omega = policy.get_scale_params()
        if explore == 'balanced':
            new_omega = omega + eta * (omega_grad + omega_metagrad)
        elif explore == 'greedy':
            new_omega = omega + eta * omega_grad
        elif explore == 'meta':
            new_omega = omega + eta * omega_metagrad
        elif explore == 'none':
            new_omega = omega
        else:
            raise ValueError
        policy.set_scale_params(new_omega)
                
        # Log
        logger.write_row(log_row, it)
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        # Next iteration
        it += 1
    
    # Final policy
    if save_params:
        logger.save_params(params, it)
        
def sigmaprop(env, policy, 
                    horizon,
                    batchsize = 100, 
                    iterations = 1000,
                    gamma = 0.99,
                    alpha = 1e-2,
                    stepper = RMSprop(1e-3),
                    test_det = True,
                    render = False,
                    seed = None,
                    baseline = 'peters',
                    action_filter = None,
                    logger = Logger(name='gpomdp_metasigma'),
                    save_params = 1000,
                    log_params = True,
                    parallel_sim = False,
                    parallel_comp = False,
                    verbose = True):
    """
        Only for SIMPLE Gaussian policy w/ scalar variance
        Policy must have learn_std = False, as std is META-learned
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
                       'gamma': gamma, 'Stepper': stepper,
                       'seed': seed,
                       'actionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 'StepSize', 'BatchSize', 'Exploration', 
                'OmegaGrad', 'OmegaMetagrad', 'ThetaGradNorm']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_det:
        log_keys.append('DetPerf')
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
            batch = generate_batch(env, policy, horizon, 1, action_filter, parallel_sim)
            policy.set_scale_params(omega)
            log_row['DetPerf'] = performance(batch, gamma)
        if render:
            batch = generate_batch(env, policy, horizon, 1, action_filter, parallel_sim, render=True)

        # Simulation
        batch = generate_batch(env, policy, horizon, batchsize, action_filter, parallel_sim)
        log_row['BatchSize'] = batchsize
        log_row['Exploration'] = policy.exploration()
        log_row['Perf'] = performance(batch, gamma)
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        params = policy.get_flat()
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i]
    
        # Gradient estimation
        grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
        theta_grad = grad[1:]
        log_row['ThetaGradNorm'] = torch.norm(theta_grad).item()

        # Step size
        beta = stepper.next(theta_grad)
        new_sigma = torch.sqrt(torch.norm(beta) / alpha)
        # Update policy mean parameters
        theta = policy.get_loc_params()
        new_theta = theta + beta * theta_grad
        policy.set_loc_params(new_theta)
        
        # Update policy variance parameter
        new_omega = torch.log(new_sigma)
        policy.set_scale_params(new_omega)
                
        # Log
        logger.write_row(log_row, it)
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        # Next iteration
        it += 1
    
    # Final policy
    if save_params:
        logger.save_params(params, it)
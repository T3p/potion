#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:18:45 2019

@author: matteo
"""
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon
from potion.estimation.gradients import simple_gpomdp_estimator
from potion.meta.steppers import ConstantStepper
from potion.estimation.metagradients import omega_metagradient_estimator, mixed_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
import torch

def sunday(env, policy, 
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
            logger = Logger(name='test_sunday'),
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


        log_row['IterationKind'] = 3
        omega = policy.get_scale_params()
        sigma = torch.exp(omega)
        batch = generate_batch(env, policy, horizon, batchsize, action_filter)
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
        log_row['Exploration'] = policy.exploration()
        log_row['Alpha'] = (alpha * sigma**2).item()
        log_row['Eta'] = eta
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

def saturday(env, policy, 
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
                    logger = Logger(name='test_saturday'),
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
    algo_info = {'Algorithm': 'SATURDAY', 
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

        if it % 2 == 0:
            # theta iteration
            log_row['IterationKind'] = 0
            omega = policy.get_scale_params()
            sigma = torch.exp(omega)
            batch = generate_batch(env, policy, horizon, batchsize, action_filter)
            grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
            theta_grad = grad[1:]
            omega_grad = grad[0]
            theta = policy.get_loc_params()
            new_theta = theta + alpha * sigma**2 * theta_grad
            policy.set_loc_params(new_theta)
        else:
            # omega iteration
            log_row['IterationKind'] = 1
            omega = policy.get_scale_params()
            sigma = torch.exp(omega)
            batch = generate_batch(env, policy, horizon, batchsize, action_filter)
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
        log_row['Exploration'] = policy.exploration()
        log_row['Alpha'] = (alpha * sigma**2).item()
        log_row['Eta'] = eta
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

def friday(env, policy, 
                    horizon,
                    batchsize = 100, 
                    iterations = 1000,
                    gamma = 0.99,
                    alpha_0 = 1e-2,
                    eta = 1e-3,
                    clip_at = 100,
                    test_det = True,
                    render = False,
                    seed = None,
                    baseline = 'peters',
                    action_filter = None,
                    logger = Logger(name='test_friday'),
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
    algo_info = {'Algorithm': 'FRIDAY', 
                 'Environment': str(env), 
                 'BatchSize': batchsize, 
                 'Max horizon': horizon,
                 'Iterations': iterations,
                 'gamma': gamma, 
                 'alpha_0': alpha_0,
                 'eta': eta, 'seed': seed,
                 'actionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 
                'Alpha', 'Eta', 'BatchSize', 'Exploration', 
                'OmegaGrad', 'OmegaMetagrad', 'ThetaGradNorm',
                'IterationKind'] #0: theta, 1: omega
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_det:
        log_keys.append('DetPerf')
    log_row = dict.fromkeys(log_keys)

    logger.open(log_row.keys())
    
    # Learning
    omega_metagrad = torch.tensor(float('nan'))
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

        if it % 2 == 0:
            # theta iteration
            log_row['IterationKind'] = 0
            omega = policy.get_scale_params()
            sigma = torch.exp(omega)
            alpha = torch.tensor(alpha_0) * sigma**2
            batch = generate_batch(env, policy, horizon, batchsize, action_filter)
            grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
            theta_grad = grad[1:]
            omega_grad = grad[0]
            theta = policy.get_loc_params()
            new_theta = theta + alpha * theta_grad
            policy.set_loc_params(new_theta)
        else:
            # omega iteration
            log_row['IterationKind'] = 1
            omega = policy.get_scale_params()
            sigma = torch.exp(omega)
            batch = generate_batch(env, policy, horizon, batchsize, action_filter)
            grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
            theta_grad = grad[1:]
            omega_grad = grad[0]
            sigma_grad = omega_grad / sigma
            mixed, _ = mixed_estimator(batch, gamma, policy, baseline, theta_grad)
            A = sigma_grad + alpha_0 * sigma**2 * theta_grad.dot(mixed)
            B = 1 + torch.sum(2 * alpha_0 * sigma * theta_grad + alpha_0 * sigma**2 * mixed)
            sigma_metagrad = A * B
            omega_metagrad = sigma_metagrad * sigma
            omega_metagrad = torch.clamp(omega_metagrad, -clip_at, clip_at)
            new_omega = omega + eta * omega_metagrad
            policy.set_scale_params(new_omega)

        log_row['OmegaGrad'] = omega_grad.item()
        log_row['OmegaMetagrad'] = omega_metagrad.item()

        # Log
        log_row['ThetaGradNorm'] = torch.norm(theta_grad).item()
        log_row['BatchSize'] = batchsize
        log_row['Exploration'] = policy.exploration()
        log_row['Alpha'] = alpha.item()
        log_row['Eta'] = eta
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

def subaltern(env, policy, 
                    horizon,
                    batchsize = 100, 
                    iterations = 1000,
                    gamma = 0.99,
                    alpha_0 = 1e-2,
                    eta = 1e-3,
                    threshold = 1e-4,
                    test_det = True,
                    render = False,
                    seed = None,
                    baseline = 'peters',
                    action_filter = None,
                    logger = Logger(name='test_subaltern'),
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
    algo_info = {'Algorithm': 'SUBALTERN', 
                 'Environment': str(env), 
                 'BatchSize': batchsize, 
                 'Max horizon': horizon,
                 'Iterations': iterations,
                 'gamma': gamma, 
                 'alpha_0': alpha_0,
                 'eta': eta, 'seed': seed,
                 'actionFilter': action_filter,
                 'threshold': threshold}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 
                'Alpha', 'Eta', 'BatchSize', 'Exploration', 
                'OmegaGrad', 'OmegaMetagrad', 'ThetaGradNorm',
                'IterationKind'] #0: theta, 1: meta-omega, 2: omega
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_det:
        log_keys.append('DetPerf')
    log_row = dict.fromkeys(log_keys)

    logger.open(log_row.keys())
    
    # Learning
    omega_grad = torch.tensor(float('nan'))
    omega_metagrad = torch.tensor(float('nan'))
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

        if it % 2 == 0:
            # theta iteration
            log_row['IterationKind'] = 0
            omega = policy.get_scale_params()
            sigma = torch.exp(omega)
            alpha = torch.tensor(alpha_0) * sigma**2
            batch = generate_batch(env, policy, horizon, batchsize, action_filter)
            grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
            theta_grad = grad[1:]
            theta = policy.get_loc_params()
            new_theta = theta + alpha * theta_grad
            policy.set_loc_params(new_theta)
        else:
            # [meta-]omega iteration
            omega = policy.get_scale_params()
            sigma = torch.exp(omega)
            batch = generate_batch(env, policy, horizon, batchsize, action_filter)
            grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
            theta_grad = grad[1:]
            omega_grad = grad[0]
            mixed, _ = mixed_estimator(batch, gamma, policy, baseline, theta_grad)
            sigma_metagrad = 2 * sigma * alpha_0 * torch.norm(theta_grad)**2 + \
                sigma ** 2 * alpha_0 * theta_grad.dot(mixed)
            omega_metagrad = sigma_metagrad * sigma
            if torch.norm(omega_metagrad) > threshold:
                # meta-omega iteration
                log_row['IterationKind'] = 1
                new_omega = omega + eta * omega_metagrad
            else:
                # omega iteration
                log_row['IterationKind'] = 2
                new_omega = omega + eta * omega_grad
            policy.set_scale_params(new_omega)
        log_row['OmegaGrad'] = omega_grad.item()
        log_row['OmegaMetagrad'] = omega_metagrad.item()

        # Log
        log_row['ThetaGradNorm'] = torch.norm(theta_grad).item()
        log_row['BatchSize'] = batchsize
        log_row['Exploration'] = policy.exploration()
        log_row['Alpha'] = alpha.item()
        log_row['Eta'] = eta
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

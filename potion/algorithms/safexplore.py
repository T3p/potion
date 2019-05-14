#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:18:45 2019

@author: matteo
"""
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon
from potion.estimation.gradients import gpomdp_estimator
from potion.estimation.metagradients import mixed_estimator, metagrad
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
import torch
import math
from potion.meta.safety_requirements import MonotonicImprovement, Budget, FixedThreshold
import scipy.stats as sts


def sepg(env, policy, 
            horizon,
            batchsize = 100, 
            iterations = 1000,
            gamma = 0.99,
            rmax = 1.,
            phimax = 1.,
            safety_requirement = 'mi',
            delta = 1.,
            confidence_schedule = None,
            clip_at = 100,
            test_batchsize = False,
            render = False,
            seed = None,
            baseline = 'peters',
            shallow = True,
            action_filter = None,
            parallel = False,
            logger = Logger(name='SEPG'),
            save_params = 1000,
            log_params = True,
            verbose = True):
    """
        Only for SIMPLE Gaussian policy w/ scalar variance
        Policy must have learn_std = False, as std is META-learned
    """
        
    #Defaults
    assert policy.learn_std
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'SEPG', 
                 'Environment': str(env), 
                 'BatchSize': batchsize, 
                 'Max horizon': horizon,
                 'Iterations': iterations,
                 'gamma': gamma, 
                 'actionFilter': action_filter,
                 'rmax': rmax,
                 'phimax': phimax}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 
                'Alpha', 'BatchSize', 'Exploration', 'Eta', 
                'ThetaGradNorm', 'OmegaGrad', 'OmegaMetagrad',
                'Penalty', 'MetaPenalty',
                'IterationKind',
                'ThetaGradNorm', 'Eps', 'Up', 'Down', 'C', 'Cmax', 'Delta'] #0: theta, 1: omega
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys.append('DetPerf')
    log_row = dict.fromkeys(log_keys)

    logger.open(log_row.keys())
    
    #Safety requirements
    if safety_requirement == 'mi':
        thresholder = MonotonicImprovement()
    elif safety_requirement == 'budget':
        batch = generate_batch(env, policy, horizon, batchsize, action_filter)
        thresholder = Budget(performance(batch, gamma))
    else:
        thresholder = FixedThreshold(float(safety_requirement))
    
    #Learning loop
    omega_grad = float('nan')
    omega_metagrad  = float('nan')
    metapenalty  = float('nan')
    eta = float('nan')
    it = 0
    while(it < iterations):
        #Begin iteration
        if verbose:
            print('\nIteration ', it)
        if verbose:
            print('Params: ', policy.get_flat())
    
        #Test mean parameters on deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, test_batchsize, 
                                        action_filter=action_filter,
                                        seed=seed,
                                        njobs=parallel,
                                        deterministic=True)
            log_row['DetPerf'] = performance(test_batch, gamma)
        #Render behavior
        if render:
            generate_batch(env, policy, horizon, 1, action_filter, render=True)

        #
        if it % 2 == 0:
            #Std update
            omega = policy.get_scale_params()
            sigma = torch.exp(omega).item()
            batch = generate_batch(env, policy, horizon, batchsize, 
                                   action_filter=action_filter, 
                                   njobs=parallel,
                                   seed=seed)
            if confidence_schedule is not None:
                delta = confidence_schedule.next(it)
            log_row['Delta'] = delta
            if delta <1:
                grad, grad_var = simple_gpomdp_estimator(batch, gamma, policy, baseline, result='moments')
                omega_grad = grad[0]
                omega_grad_var = grad_var[0]
                omega_metagrad, omega_metagrad_var = metagrad(batch, gamma, policy, alpha, clip_at, baseline, result='moments')
                quant = 2 * sts.t.interval(1 - delta, batchsize-1,loc=0.,scale=1.)[1]
                eps = torch.tensor(quant * torch.sqrt(omega_grad_var / batchsize), dtype=torch.float)
                log_row['Eps'] = torch.norm(eps).item()
                metaeps = torch.tensor(quant * torch.sqrt(omega_metagrad_var / batchsize), dtype=torch.float)
                if torch.sign(omega_grad).item() >= 0 and torch.sign(omega_metagrad).item() >= 0:
                    up = torch.clamp(torch.abs(omega_grad - eps), min=0.) * torch.clamp(torch.abs(omega_metagrad - metaeps), min=0.)
                elif torch.sign(omega_grad).item() >= 0 and torch.sign(omega_metagrad).item() < 0:
                    up = (omega_grad + eps) * (omega_metagrad - metaeps)
                elif torch.sign(omega_grad).item() < 0 and torch.sign(omega_metagrad).item() >=0:
                    up = (omega_grad - eps) * (omega_metagrad + eps)
                else:
                    up = torch.abs(omega_grad + eps) * torch.abs(omega_metagrad + metaeps)
                down = omega_metagrad + metaeps * torch.sign(omega_metagrad)
                log_row['Up'] = up.item()
                log_row['Down'] = down.item()
                metapenalty = rmax /  (1 - gamma)**2 * (0.53 * avol / (2 * sigma) + gamma / (1 - gamma))
                eta_star = (up / (2 * metapenalty * down**2 + 1e-12)).item()
                Cmax = up**2 / (4 * metapenalty * down**2).item()
            else:
                log_row['Eps'] = 0
                grad = gpomdp_estimator(batch, gamma, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow)
                theta_grad = grad[1:]
                omega_grad = grad[0]
                #->
                mixed, _ = mixed_estimator(batch, gamma, policy, baseline, theta_grad)
                norm_grad = 2 * theta_grad.dot(mixed)
                A = omega_grad
                B = 2 * alpha * torch.norm(theta_grad)**2
                C = sigma * alpha * norm_grad
                C = torch.clamp(C, min=-clip_at, max=clip_at)
                omega_metagrad = A + B + C
                metapenalty = rmax /  (1 - gamma)**2 * (0.53 * avol / (2 * sigma) + gamma / (1 - gamma))
                eta_star = (omega_grad / (2 * metapenalty * omega_metagrad) + 1e-12).item()
                Cmax = (omega_grad ** 2 / (4 * metapenalty)).item()
                log_row['Up'] = torch.tensor(omega_grad).item()
                log_row['Down'] = torch.tensor(omega_metagrad).item()
        
            perf = performance(batch, gamma)
            Co = thresholder.next(perf)
            Co = min(Co, Cmax)
            log_row['C'] = Co
            log_row['Cmax'] = Cmax
            eta = eta_star + abs(eta_star) * math.sqrt(1 - Co / (Cmax + 1e-12) + 1e-12)
            new_omega = omega + eta * omega_metagrad
            policy.set_scale_params(new_omega)
            ###
        else:
            #Mean update
            omega = policy.get_scale_params()
            sigma = torch.exp(omega).item()
            batch = generate_batch(env, policy, horizon, batchsize, 
                                   action_filter=action_filter, 
                                   n_jobs=parallel, 
                                   seed=seed)
            if confidence_schedule is not None:
                delta = confidence_schedule.next(it)
            log_row['Delta'] = delta
            if delta < 1:
                grad, grad_var = simple_gpomdp_estimator(batch, gamma, policy, baseline, result='moments')
                theta_grad = grad[1:]
                theta_grad_var = grad_var[1:]
                quant = 2*sts.t.interval(1 - delta, batchsize-1,loc=0.,scale=1.)[1]
                eps = quant * torch.sqrt(theta_grad_var / batchsize)
                log_row['Eps'] = torch.norm(eps).item()
                norm2 = torch.norm(torch.clamp(torch.abs(theta_grad) - eps, min=0.))
                norm1 = torch.sum(torch.abs(theta_grad) + eps)
                log_row['Up'] = norm1.item()
                log_row['Down'] = norm2.item()
            else:
                log_row['Eps'] = 0
                grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
                theta_grad = grad[1:]
                norm2 = torch.norm(theta_grad)
                norm1 = torch.sum(torch.abs(theta_grad))
                log_row['Up'] = norm1.item()
                log_row['Down'] = norm2.item()
            penalty = rmax * phimax**2 / (1-gamma)**2 * (avol / (sigma * math.sqrt(2*math.pi)) + gamma / (2*(1-gamma)))
            alpha_star = sigma ** 2 * norm2 ** 2 / (2 * penalty * norm1 ** 2 + 1e-12)
            Cmax = (alpha_star * norm2**2 / 2).item()
            perf = performance(batch, gamma)
            Co = thresholder.next(perf)
            Co = min(Co, Cmax)
            log_row['C'] = Co
            log_row['Cmax'] = Cmax
            alpha = alpha_star * (1 + math.sqrt(1 - Co / (Cmax + 1e-12) + 1e-12))
            theta = policy.get_loc_params()
            new_theta = theta + alpha * theta_grad
            policy.set_loc_params(new_theta)
            ###


        # Log
        log_row['IterationKind'] = it % 2
        log_row['ThetaGradNorm'] = torch.norm(theta_grad).item()
        log_row['Alpha'] = alpha
        log_row['Eta'] = eta
        log_row['Penalty'] = penalty
        log_row['MetaPenalty'] = metapenalty
        log_row['OmegaGrad'] = torch.tensor(omega_grad).item()
        log_row['OmegaMetagrad'] = torch.tensor(omega_metagrad).item()
        log_row['ThetaGradNorm'] = torch.norm(theta_grad).item()
        log_row['BatchSize'] = batchsize
        log_row['Exploration'] = policy.exploration()
        log_row['Alpha'] = alpha.item()
        log_row['Perf'] = perf
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


def adastep(env, policy, 
            horizon,
            batchsize = 100, 
            iterations = 1000,
            gamma = 0.99,
            rmax = 1.,
            phimax = 1.,
            greedy = True,
            delta = 1.,
            test_det = True,
            render = False,
            seed = None,
            baseline = 'peters',
            action_filter = None,
            parallel = False,
            n_jobs = 4,
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
    algo_info = {'Algorithm': 'ADASTEP', 
                 'Environment': str(env), 
                 'BatchSize': batchsize, 
                 'Max horizon': horizon,
                 'Iterations': iterations,
                 'gamma': gamma, 
                 'actionFilter': action_filter,
                 'rmax': rmax,
                 'phimax': phimax,
                 'greedy': greedy}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 
                'Alpha', 'BatchSize', 'Exploration', 
                'ThetaGradNorm',
                'Penalty']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_det:
        log_keys.append('DetPerf')
    log_row = dict.fromkeys(log_keys)

    logger.open(log_row.keys())
    
    # Learning
    avol = torch.tensor(env.action_space.high - env.action_space.low).item()
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
        sigma = torch.exp(omega).item()
        batch = generate_batch(env, policy, horizon, batchsize, action_filter, parallel=parallel, n_jobs=n_jobs, seed=seed)        
        if delta < 1:
            grad, grad_var = simple_gpomdp_estimator(batch, gamma, policy, baseline, result='moments')
            theta_grad = grad[1:]
            theta_grad_var = grad_var[1:]
            quant = 2*sts.t.interval(1 - delta, batchsize-1,loc=0.,scale=1.)[1]
            eps = quant * torch.sqrt(theta_grad_var / batchsize + 1e-12)
            norm2 = torch.norm(torch.clamp(torch.abs(theta_grad) - eps, min=0.))
            norm1 = torch.sum(torch.abs(theta_grad) + eps)
        else:
            grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
            theta_grad = grad[1:]
            norm2 = torch.norm(theta_grad)
            norm1 = torch.sum(torch.abs(theta_grad))
        penalty = rmax * phimax**2 / (1-gamma)**2 * (avol / (sigma * math.sqrt(2*math.pi)) + gamma / (2*(1-gamma)))
        alpha_star = sigma ** 2 * norm2 ** 2 / (2 * penalty * norm1 ** 2 + 1e-12)
        Cmax = alpha_star * norm2**2 / 2
            
        if greedy:
            C = Cmax
        else:
            C = 0
        alpha = alpha_star * (1 + math.sqrt(1 - C / (Cmax + 1e-12) + 1e-12))
        theta = policy.get_loc_params()
        new_theta = theta + alpha * theta_grad
        policy.set_loc_params(new_theta)

        # Log
        log_row['Alpha'] = alpha
        log_row['Penalty'] = penalty
        log_row['ThetaGradNorm'] = torch.norm(theta_grad).item()
        log_row['BatchSize'] = batchsize
        log_row['Exploration'] = policy.exploration()
        log_row['Alpha'] = alpha.item()
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
        
def adabatch(env, policy, 
            horizon,
            batchsize = 100, 
            iterations = 1000,
            gamma = 0.99,
            rmax = 1.,
            phimax = 1.,
            safety_requirement = MonotonicImprovement(),
            test_det = True,
            render = False,
            seed = None,
            baseline = 'peters',
            action_filter = None,
            parallel = False,
            n_jobs = 4,
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
    algo_info = {'Algorithm': 'ADASTEP', 
                 'Environment': str(env), 
                 'BatchSize': batchsize, 
                 'Max horizon': horizon,
                 'Iterations': iterations,
                 'gamma': gamma, 
                 'actionFilter': action_filter,
                 'rmax': rmax,
                 'phimax': phimax}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 
                'Alpha', 'BatchSize', 'Exploration', 
                'ThetaGradNorm',
                'Penalty', 'Coordinate']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_det:
        log_keys.append('DetPerf')
    log_row = dict.fromkeys(log_keys)

    logger.open(log_row.keys())
    
    # Learning
    avol = torch.tensor(env.action_space.high - env.action_space.low).item()
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
        sigma = torch.exp(omega).item()
        batch = generate_batch(env, policy, horizon, batchsize, action_filter, parallel=parallel, n_jobs=n_jobs, seed=seed)
        grad = simple_gpomdp_estimator(batch, gamma, policy, baseline)
        theta_grad = grad[1:]
        norminf = torch.max(torch.abs(theta_grad))
        k = torch.argmax(torch.abs(theta_grad))
        penalty = rmax * phimax**2 / (1-gamma)**2 * (avol / (sigma * math.sqrt(2*math.pi)) + gamma / (2*(1-gamma)))
        alpha_star = sigma ** 2/ (2 * penalty)
        Cmax = alpha_star * norminf*2 / 2
        C = safety_requirement.next()
        alpha = alpha_star * (1 + math.sqrt(1 - C / (Cmax + 1e-12) + 1e-12))
        theta = policy.get_loc_params()
        new_theta = theta
        new_theta[k] += alpha * theta_grad[k]
        policy.set_loc_params(new_theta)

        # Log
        log_row['Coordinate'] = k.item()
        log_row['Alpha'] = alpha
        log_row['Penalty'] = penalty
        log_row['ThetaGradNorm'] = torch.norm(theta_grad).item()
        log_row['BatchSize'] = batchsize
        log_row['Exploration'] = policy.exploration()
        log_row['Alpha'] = alpha
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


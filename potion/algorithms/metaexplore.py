#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:18:45 2019

@author: matteo
"""
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon
from potion.estimation.gradients import gpomdp_estimator
from potion.estimation.metagradients import metagrad
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
from potion.policies.gaussian_policies import LinearGaussianPolicy
import torch

def mepg(env, policy, 
            horizon,
            batchsize = 500, 
            iterations = 200,
            disc = 0.99,
            alpha = 1e-1,
            eta = 1e-3,
            clip_at = 100,
            test_batchsize = False,
            render = False,
            seed = None,
            action_filter = None,
            parallel = False,
            logger = Logger(name='MEPG'),
            save_params = 50,
            log_params = True,
            verbose = True):
    """
        MEPG algorithm
        Only for shallow Gaussian policy w/ scalar variance
    """
        
    #Defaults
    assert type(policy) == LinearGaussianPolicy
    assert policy._learn_std
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'MEPG', 
                 'Environment': str(env), 
                 'BatchSize': batchsize, 
                 'Horizon': horizon,
                 'Iterations': iterations,
                 'Disc': disc, 
                 'Alpha': alpha,
                 'Eta': eta, 
                 'Seed': seed,
                 'ActionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 
                'StepSize', 'MetaStepSize', 'BatchSize', 'Exploration', 
                'OmegaGrad', 'OmegaMetagrad', 'upsilonGradNorm']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys.append('DetPerf')
    log_row = dict.fromkeys(log_keys)

    logger.open(log_row.keys())
    
    #Learning loop
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
            log_row['DetPerf'] = performance(test_batch, disc)
        #Render behavior
        if render:
            generate_batch(env, policy, horizon, 1, action_filter, render=True)

        #Set metaparameters
        omega = policy.get_scale_params()
        sigma = torch.exp(omega)
        stepsize = alpha * sigma**2

        #Collect trajectories
        batch = generate_batch(env, policy, horizon, batchsize, 
                               action_filter=action_filter, 
                               seed=seed, 
                               n_jobs=parallel)
        
        #Estimate policy gradient
        grad = gpomdp_estimator(batch, disc, policy, 
                                    baselinekind='peters', 
                                    shallow=True)
        upsilon_grad = grad[1:]
        omega_grad = grad[0]
        
        omega_metagrad = metagrad(batch, disc, policy, alpha, clip_at, 
                                  grad=grad)
        
        upsilon = policy.get_loc_params()
        new_upsilon = upsilon + stepsize * upsilon_grad
        policy.set_loc_params(new_upsilon)
        
        new_omega = omega + eta * omega_metagrad
        policy.set_scale_params(new_omega)

        # Log
        log_row['Exploration'] = policy.exploration()
        log_row['StepSize'] = stepsize.item()
        log_row['MetaStepSize'] = eta
        log_row['OmegaGrad'] = omega_grad.item()
        log_row['OmegaMetagrad'] = omega_metagrad.item()
        log_row['UpsilonGradNorm'] = torch.norm(upsilon_grad).item()
        log_row['BatchSize'] = batchsize
        log_row['Perf'] = performance(batch, disc)
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

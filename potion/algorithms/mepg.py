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
from potion.common.misc_utils import clip, seed_all_agent, mean_sum_info
from potion.actors.continuous_policies import ShallowGaussianPolicy
import torch

def mepg(env, policy, 
            horizon,
            batchsize = 500, 
            iterations = 200,
            disc = 0.99,
            alpha = 1e-1,
            eta = 1e-3,
            test_batchsize = False,
            render = False,
            seed = None,
            action_filter = None,
            parallel = False,
            logger = Logger(name='MEPG'),
            info_key = 'danger',
            save_params = 50,
            log_params = True,
            verbose = True,
            ablation = False):
    """
        MEPG algorithm
        Only for shallow Gaussian policy w/ scalar variance
    """
        
    #Defaults
    assert type(policy) == ShallowGaussianPolicy
    assert policy.learn_std
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
                'OmegaGrad', 'OmegaMetagrad', 'UpsilonGradNorm', 'Info']
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
                                        deterministic=True,
                                        key=info_key)
            log_row['DetPerf'] = performance(test_batch, disc)
        #Render behavior
        if render:
            generate_batch(env, policy, horizon, 1, action_filter, render=True, key=info_key)

        #Set metaparameters
        omega = policy.get_scale_params()
        sigma = torch.exp(omega)
        
        #Collect trajectories
        batch = generate_batch(env, policy, horizon, batchsize, 
                               action_filter=action_filter, 
                               seed=seed, 
                               n_jobs=parallel,
                               key=info_key)
        
        #Estimate policy gradient
        grad_samples = gpomdp_estimator(batch, disc, policy, 
                                    baselinekind='peters', 
                                    shallow=True,
                                    result='samples')
        grad = torch.mean(grad_samples, 0)
        upsilon_grad = grad[1:]
        omega_grad = grad[0]
        
        #Estimate meta gradient
        omega_metagrad = metagrad(batch, disc, policy, alpha,
                                  grad_samples=grad_samples, 
                                  no_first=(ablation==1),
                                  no_second=(ablation==2), 
                                  no_third=(ablation==3))
        
        #Update mean parameters
        upsilon = policy.get_loc_params()
        new_upsilon = upsilon + alpha * sigma**2 * upsilon_grad / torch.norm(upsilon_grad)
        policy.set_loc_params(new_upsilon)
        
        #Update variance parameters
        new_omega = omega + eta * omega_metagrad / torch.norm(omega_metagrad)
        policy.set_scale_params(new_omega)

        # Log
        log_row['Exploration'] = sigma.item()
        log_row['StepSize'] = (alpha * sigma**2 / torch.norm(upsilon_grad)).item()
        log_row['MetaStepSize'] = (eta / torch.norm(omega_metagrad)).item()
        log_row['OmegaGrad'] = omega_grad.item()
        log_row['OmegaMetagrad'] = omega_metagrad.item()
        log_row['UpsilonGradNorm'] = torch.norm(upsilon_grad).item()
        log_row['BatchSize'] = batchsize
        log_row['Perf'] = performance(batch, disc)
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['Info'] = mean_sum_info(batch).item()
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

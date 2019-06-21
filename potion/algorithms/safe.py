#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe Policy Gradient (SPG)
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon, mean_sum_info, clip, seed_all_agent, returns
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator
from potion.common.logger import Logger
import scipy.stats as sts
import torch
import time
import math

def safepg(env, policy, horizon, lip_const, var_bound, *,
                    conf = 0.2,
                    min_batchsize = 32,
                    max_batchsize = 5000,
                    iterations = float('inf'),
                    max_samples = 1e6,
                    disc = 0.9,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='SPG'),
                    shallow = True,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'danger',
                    save_params = 100,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1,
                    meta_conf = 0.05):
    """
    SafePG algorithm, incremental version
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    lip_const: Lipschitz constant of the gradient (upper bound)
    var_bound: upper bound on the variance of the PG estimator
    conf: probability of failure
    max_batchsize: maximum number of trajectories to estimate policy gradient
    iterations: number of policy updates
    max_samples: maximum number of total trajectories
    disc: discount factor
    action_filter: function to apply to the agent's action before feeding it to 
        the environment, not considered in gradient estimation. By default,
        the action is clipped to satisfy evironmental boundaries
    estimator: either 'reinforce' or 'gpomdp' (default). The latter typically
        suffers from less variance
    baseline: control variate to be used in the gradient estimator. Either
        'avg' (average reward, default), 'peters' (variance-minimizing) or
        'zero' (no baseline)
    logger: for human-readable logs (standard output, csv, tensorboard...)
    shallow: whether to employ pre-computed score functions (only available for
        shallow policies)
    seed: random seed (None for random behavior)
    test_batchsize: number of test trajectories used to evaluate the 
        corresponding deterministic policy at each iteration. If 0 or False, no 
        test is performed
    info_key: name of the environment info to log
    save_params: how often (every x iterations) to save the policy 
        parameters to disk. Final parameters are always saved for 
        x>0. If False, they are never saved.
    log_params: whether to include policy parameters in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    parallel: number of parallel jobs for simulation. If 0 or False, 
        sequential simulation is performed.
    render: how often (every x iterations) to render the agent's behavior
        on a sample trajectory. If False, no rendering happens
    verbose: level of verbosity
    """
    #Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'SPG',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Discount': disc,
                   'Confidence': conf,
                   'ConfidenceParam': conf,
                   'Seed': seed,
                   'MinBatchSize': min_batchsize,
                   'MaxBatchSize': max_batchsize,
                   'LipschitzConstant': lip_const,
                   'VarianceBound': var_bound
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'GradNorm', 
                'Time',
                'StepSize',
                'BatchSize',
                'Info',
                'TotSamples',
                'Safety']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Initializations
    it = 0
    tot_samples = 0
    safe_it = 0
    safety = 1.
    optimal_batchsize = min_batchsize
    min_safe_batchsize = min_batchsize
    _estimator = reinforce_estimator if estimator=='reinforce' else gpomdp_estimator
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\nIteration ', it)
        params = policy.get_flat()
        
        #Test the corresponding deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, 
                                        episodes=test_batchsize, 
                                        action_filter=action_filter,
                                        n_jobs=parallel,
                                        deterministic=True,
                                        key=info_key)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['UTestPerf'] = performance(test_batch, 1)
            log_row['TestInfo'] = mean_sum_info(test_batch).item()
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon,
                           episodes=1,
                           action_filter=action_filter, 
                           render=True)
    
    
        #Experience loop
        _conf = conf
        #Collect trajectories according to previous optimal batch size
        batch = generate_batch(env, policy, horizon, 
                                episodes=max(min_batchsize, min(max_batchsize, optimal_batchsize)), 
                                action_filter=action_filter,
                                n_jobs=parallel,
                                key=info_key)
        batchsize = len(batch)
        
        do = True
        while do or batchsize < min_safe_batchsize:
            do = False
            #Collect more trajectories to match minimum safe batch size
            batch += generate_batch(env, policy, horizon, 
                        episodes=min(max_batchsize, min_safe_batchsize) - batchsize, 
                        action_filter=action_filter,
                        n_jobs=parallel,
                        key=info_key)
            batchsize = len(batch)
            
            #Estimate policy gradient
            grad_samples = _estimator(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='samples')
            grad = torch.mean(grad_samples, 0)
            
            #Optimal batch size
            optimal_batchsize = torch.ceil(4 * var_bound / 
                                 (conf * torch.norm(grad)**2)).item()
            min_safe_batchsize = optimal_batchsize / 4.
            if verbose and optimal_batchsize < max_batchsize:
                print('Collected %d / %d trajectories' % (batchsize, optimal_batchsize))
            elif verbose:
                print('Collected %d / %d trajectories' % (batchsize, min(max_batchsize, min_safe_batchsize)))
            
            #Adjust confidence before collecting more data for the same update
            _conf /= 2
            if batchsize >= max_batchsize:
                break
        
        if verbose:
            print('Optimal batch size: %d' % optimal_batchsize if optimal_batchsize < float('inf') else -1)
            print('Minimum safe batch size: %d' % min_safe_batchsize if min_safe_batchsize < float('inf') else -1)
            if batchsize >= min_safe_batchsize and batchsize < optimal_batchsize:
                print('Low sample regime')

        #Update long-term quantities
        tot_samples += batchsize
        
        #Update safety measure
        if it == 0:
            old_rets= returns(batch, disc)
        else:
            new_rets = returns(batch, disc)
            _, pval = sts.ttest_ind(new_rets, old_rets)
            if pval / 2 < meta_conf:
                safe_it += 1
            safety = safe_it / it
        
        #Log
        log_row['Safety'] = safety
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['GradNorm'] = torch.norm(grad).item()
        log_row['BatchSize'] = batchsize
        log_row['TotSamples'] = tot_samples
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()
                
        #Check if number of samples is sufficient to perform update
        if batchsize < min_safe_batchsize:
            if verbose:
                print('Unsafe, skipping')
            #Log
            log_row['StepSize'] = 0.
            log_row['Time'] = time.time() - start
            logger.write_row(log_row, it)
            
            #Adjust confidence before collecting new data for the same update
            _conf /= 2
            
            #Skip to next iteration (current trajectories are discarded)
            it += 1
            continue
        
        #Select step size
        stepsize = 1. / lip_const * (1 - math.sqrt(var_bound) / (torch.norm(grad) * math.sqrt(batchsize * conf)).item())
        log_row['StepSize'] = stepsize
                
        #Update policy parameters
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        log_row['Time'] = time.time() - start
        logger.write_row(log_row, it)
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()

    
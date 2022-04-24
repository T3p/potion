#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Gradient algorithms with monotonic improvement guarantees
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import (performance, avg_horizon, mean_sum_info, 
                                      clip, seed_all_agent, returns, separator,
                                      performance_lcb, performance_ucb)
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator
from potion.common.logger import Logger
from potion.estimation.jackknife import jackknife
import scipy.stats as sts
import torch
import time
import math
import numpy as np

def spg(env, policy, horizon, lip_const, err_bound, *,
                    fail_prob = 0.05,
                    mini_batchsize = 10,
                    max_batchsize = 10000,
                    iterations = float('inf'),
                    max_samples = 1e6,
                    disc = 0.9,
                    fast = False,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='SPG'),
                    shallow = True,
                    seed = None,
                    save_params = 1000,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    oracle = None,
                    verbose = 1):
    """
    Safe PG algorithm with adaptive batch size from 
    "Smoothing Policies and Safe Policy Gradients", Papini et al., 2019
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    lip_const: Lipschitz constant of the gradient (upper bound)
    err_bound: statistical upper bound on the PG estimation error (function of
                probability)
    conf: probability of failure
    min_batchsize: minimum number of trajectories to estimate policy gradient
    max_batchsize: maximum number of trajectories to estimate policy gradient
    iterations: maximum number of learning iterations
    max_samples: maximum number of total collected trajectories
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
    meta_conf: confidence level of safe-update test (for evaluation only)
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
    verbose: level of verbosity on standard output
    """
    #Defaults
    """
    if action_filter is None:
        action_filter = clip(env)
    """
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
                   'Confidence': 1 - fail_prob,
                   'Seed': seed,
                   'MiniBatchSize': mini_batchsize,
                   'MaxBatchSize': max_batchsize,
                   'LipschitzConstant': lip_const,
                   'ErrorBound': err_bound,
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
                'TotSamples']
    if oracle is not None:
        log_keys += ['Oracle']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Initializations
    it = 1
    tot_samples = 0
    _estimator = (reinforce_estimator if estimator=='reinforce' 
                  else gpomdp_estimator)
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
        params = policy.get_flat()
                
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon,
                           episodes=1,
                           action_filter=action_filter, 
                           render=True)
        
        #Collect trajectories to match optimal safe batch size
        batch = []
        batchsize = 0
        delta = fail_prob / (it * (it + 1))
        do = True
        i = 0
        while do or batchsize + mini_batchsize <= max_batchsize:
            do = False
            i = i + 1
            batch += generate_batch(env, policy, horizon, 
                        episodes=mini_batchsize, 
                        action_filter=action_filter,
                        n_jobs=parallel)
            batchsize = len(batch)
            
            #Estimate policy gradient
            grad = _estimator(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='mean')
            
            #Optimal batch size
            delta_i = delta / (i * (i + 1))
            min_safe_batchsize = torch.ceil(err_bound(delta_i, batchsize) / 
                                 torch.norm(grad)**2).item()
            if not fast:
                optimal_batchsize = torch.ceil(4 * err_bound(delta_i, batchsize) / 
                     torch.norm(grad)**2).item()
            else:
                optimal_batchsize = min_safe_batchsize
            if verbose:
                print('Collected %d / %d trajectories' % (batchsize, 
                                                          optimal_batchsize))
            
            #Collecting more data for the same update?
            if batchsize >= optimal_batchsize:
                break
        
        if verbose:
            print('Optimal batch size: %d' % optimal_batchsize)

        #Update long-term quantities
        tot_samples += batchsize
        
        #Log
        log_row['Perf'] = performance(batch, disc)
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['GradNorm'] = torch.norm(grad).item()
        log_row['BatchSize'] = batchsize
        log_row['TotSamples'] = tot_samples
        if oracle is not None:
            log_row['Oracle'] = oracle(params.numpy())
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()

            #Log
            log_row['StepSize'] = 0.
            log_row['Time'] = time.time() - start
            if verbose:
                print(separator)
            logger.write_row(log_row, it)
            if verbose:
                print(separator)

            #Skip to next iteration (current trajectories are discarded)
            it += 1
            continue
        
        #Select step size
        if batchsize >= min_safe_batchsize:
            if not fast:
                stepsize = 1. / (2 * lip_const)
            else:
                stepsize = 1. / lip_const
        else:
            if verbose:
                print('Safe update would require more samples than maximum allowed')
            stepsize = 0.
        log_row['StepSize'] = stepsize
                
        #Update policy parameters
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        log_row['Time'] = time.time() - start
        if verbose:
            print(separator)
        logger.write_row(log_row, it)
        if verbose:
            print(separator)
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()

def relaxed_spg(env, policy, horizon, lip_const, err_bound, max_rew, *,
                    empirical = False,    
                    degradation = 0.,
                    fail_prob = 0.05,
                    mini_batchsize = 10,
                    max_batchsize = 10000,
                    iterations = float('inf'),
                    max_samples = 1e6,
                    disc = 0.9,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    warm_start = False,
                    logger = Logger(name='RSPG'),
                    shallow = True,
                    seed = None,
                    save_params = 1000,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    oracle = None,
                    verbose = 1):
    """
    Safe PG algorithm with adaptive batch size from 
    "Smoothing Policies and Safe Policy Gradients", Papini et al., 2019
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    lip_const: Lipschitz constant of the gradient (upper bound)
    err_bound: statistical upper bound on the PG estimation error (function of
                probability)
    conf: probability of failure
    min_batchsize: minimum number of trajectories to estimate policy gradient
    max_batchsize: maximum number of trajectories to estimate policy gradient
    iterations: maximum number of learning iterations
    max_samples: maximum number of total collected trajectories
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
    meta_conf: confidence level of safe-update test (for evaluation only)
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
    verbose: level of verbosity on standard output
    """
    #Defaults
    """
    if action_filter is None:
        action_filter = clip(env)
    """
    
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
                   'Confidence': 1 - fail_prob,
                   'Seed': seed,
                   'MiniBatchSize': mini_batchsize,
                   'MaxBatchSize': max_batchsize,
                   'LipschitzConstant': lip_const,
                   'ErrorBound': err_bound,
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
                'TotSamples',
                'Target']
    if oracle is not None:
        log_keys += ['Oracle']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Initializations
    it = 1
    tot_samples = 0
    _opt_perf = -np.infty
    _estimator = (reinforce_estimator if estimator=='reinforce' 
                  else gpomdp_estimator)
    
    #Learning loop
    batchsize = 0
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
        params = policy.get_flat()
                
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon,
                           episodes=1,
                           action_filter=action_filter, 
                           render=True)
        
        #Collect trajectories to match optimal safe batch size
        batch = []
        if not warm_start:
            batchsize = 0
        delta = fail_prob / (it * (it + 1))
        do = True
        i = 0
        safe_flag = False
        
        if warm_start:
            batch += generate_batch(env, policy, horizon, 
                        episodes=int(3/2*batchsize), 
                        action_filter=action_filter,
                        n_jobs=parallel)
            batchsize = len(batch) 
        
        while do or batchsize + mini_batchsize <= max_batchsize:
            do = False
            i = i + 1
            batch += generate_batch(env, policy, horizon, 
                        episodes=mini_batchsize, 
                        action_filter=action_filter,
                        n_jobs=parallel)
            batchsize = len(batch)
            
            #Estimate policy gradient
            grad_samples = _estimator(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='samples')
            grad = torch.mean(grad_samples, 0)            
            #Optimal batch size
            delta_i = delta / (3 * i * (i + 1))
            
            #(Over) estimate best performance so far
            opt_perf = max(_opt_perf, performance_ucb(batch, disc, max_rew, 
                                                     delta_i, horizon))
            
            #(Under) estimate current performance
            pess_perf = performance_lcb(batch, disc, max_rew, delta_i, horizon)
            threshold = pess_perf - (1-degradation)*opt_perf
            if threshold < 0:
                threshold = 0
            
            #Collecting more data for the same update?
            if empirical:
                sample_var = torch.sum(torch.norm(grad_samples - grad, dim=-1)**2) / (batchsize - 1)
                eps = err_bound(delta_i, sample_var, batchsize)
            else:
                eps = err_bound(delta_i, batchsize)
            gnorm = torch.norm(grad).item()
            #unsafety = eps - 3. * gnorm / 4. - 2. * threshold * lip_const / gnorm
            unsafety = eps - gnorm / 2. - threshold * lip_const / gnorm
            if verbose:
                print("Samples: %d, PessPerf: %f, TargetPerf: %f, Threshold: %f, Unsafety: %f, SampleVar: % f, GradNorm: %f, Eps: %f" 
                      % (batchsize,pess_perf,(1-degradation)*opt_perf,threshold,unsafety,sample_var,gnorm,eps))
            if unsafety <= 0:
                safe_flag = True
                break
        
        #Update long-term quantities
        _opt_perf = max(_opt_perf, opt_perf)
        tot_samples += batchsize
        
        #Log
        log_row['Perf'] = performance(batch, disc)
        log_row['Target'] = (1-degradation)*opt_perf
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['GradNorm'] = torch.norm(grad).item()
        log_row['BatchSize'] = batchsize
        log_row['TotSamples'] = tot_samples
        if oracle is not None:
            log_row['Oracle'] = oracle(params.numpy())
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()

            #Log
            log_row['StepSize'] = 0.
            log_row['Time'] = time.time() - start
            if verbose:
                print(separator)
            logger.write_row(log_row, it)
            if verbose:
                print(separator)

            #Skip to next iteration (current trajectories are discarded)
            it += 1
            continue
        
        #Select step size
        if safe_flag==True:
            #stepsize = 1. / (2 * lip_const)
            stepsize = 1. / lip_const
        else:
            if verbose:
                print('Safe update would require more samples than maximum allowed')
            stepsize = 0.
        log_row['StepSize'] = stepsize
                
        #Update policy parameters
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        log_row['Time'] = time.time() - start
        if verbose:
            print(separator)
        logger.write_row(log_row, it)
        if verbose:
            print(separator)
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()
    
    
def safe_step_strict(env, policy, disc, horizon, lip_const, var_bound,
                    batchsize = 100,
                    conf = 0.2,
                    iterations = float('inf'),
                    max_samples = 1e7,
                    action_filter = None,
                    logger = Logger(name='SafeStepStrict'),
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    shallow = True,
                    seed = None,
                    save_params = 10000,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    info_key = None,
                    verbose = 1):
    """
    Strict version of Safe PG algorithm with adaptive step size from 
    "Smoothing Policies and Safe Policy Gradients" (Algorithm 2)
    """
    #Defaults
    """
    if action_filter is None:
        action_filter = clip(env)
    """
    if baseline != 'zero':
        assert batchsize > 2
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'SafeStepStrict',
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Discount': disc,
                   'Seed': seed,
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize',
                'BatchSize',
                'GradNorm', 
                'Time',
                'TotSamples',
                'Threshold',
                'VarBound']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if info_key is not None:
        log_keys.append(info_key)
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Initializations
    _estimator = (reinforce_estimator if estimator=='reinforce' 
                  else gpomdp_estimator)
    it = 0
    tot_samples = 0
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
        params = policy.get_flat()
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon,
                           episodes=1,
                           action_filter=action_filter, 
                           render=True)

        #Collect trajectories
        batch = generate_batch(env, policy, horizon, 
                                episodes=batchsize, 
                                action_filter=action_filter,
                                n_jobs=parallel,
                                key=info_key)

        #Estimate policy gradient
        grad_samples = _estimator(batch, disc, policy, 
                                    baselinekind=baseline, 
                                    shallow=shallow,
                                    result='samples')
        grad = torch.mean(grad_samples, 0)
        grad_norm = torch.norm(grad)
        
        #Update long-term quantities
        tot_samples += batchsize
        
        #Safety test
        threshold = torch.ceil(var_bound / (conf * grad_norm**2))
                       
        #Log
        log_row['BatchSize'] = batchsize
        log_row['VarBound'] = var_bound
        log_row['LipConst'] = lip_const
        log_row['Threshold'] = threshold.item()
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['GradNorm'] = grad_norm.item()
        log_row['TotSamples'] = tot_samples
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()
                
        #Safety test
        if batchsize < threshold:
            #Log
            log_row['StepSize'] = 0.
            log_row['Time'] = time.time() - start
            if verbose:
                print(separator)
            logger.write_row(log_row, it)
            if verbose:
                print(separator)
            #Terminate
            if verbose:
                print ('Not safe! Need at least %d samples' 
                       % int(threshold.item()))
                break
        
        #Select step size
        stepsize = (1. - math.sqrt(var_bound / (conf * batchsize)) 
                    / grad_norm) / lip_const
        log_row['StepSize'] = stepsize.item()
                
        #Update policy parameters
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Log
        log_row['Time'] = time.time() - start
        if verbose:
            print(separator)
        logger.write_row(log_row, it)
        if verbose:
            print(separator)
        
        #Prepare next iteration
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()
    
def safe_step(env, policy, disc, horizon, lip_const,
                    batchsize = 100,
                    conf = 0.2,
                    iterations = float('inf'),
                    max_samples = 1e7,
                    action_filter = None,
                    logger = Logger(name='SafeStep'),
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    shallow = True,
                    seed = None,
                    save_params = 10000,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    info_key = None,
                    verbose = 1):
    """
    Strict version of Safe PG algorithm with adaptive step size from 
    "Smoothing Policies and Safe Policy Gradients"
    """
    #Defaults
    """
    if action_filter is None:
        action_filter = clip(env)
    """
    if baseline != 'zero':
        assert batchsize > 2
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'SafeStep',
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Discount': disc,
                   'Seed': seed,
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize',
                'BatchSize',
                'GradNorm', 
                'Time',
                'TotSamples',
                'Threshold',
                'VarBound']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if info_key is not None:
        log_keys.append(info_key)
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Initializations
    _estimator = (reinforce_estimator if estimator=='reinforce' 
                  else gpomdp_estimator)
    it = 0
    tot_samples = 0
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
        params = policy.get_flat()
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon,
                           episodes=1,
                           action_filter=action_filter, 
                           render=True)

        #Collect trajectories
        batch = generate_batch(env, policy, horizon, 
                                episodes=batchsize, 
                                action_filter=action_filter,
                                n_jobs=parallel,
                                key=info_key)

        #Estimate policy gradient
        grad_samples = _estimator(batch, disc, policy, 
                                    baselinekind=baseline, 
                                    shallow=shallow,
                                    result='samples')
        grad = torch.mean(grad_samples, dim=0)
        grad_norm = torch.norm(grad)
        
        #Estimate variance
        var_estimator = lambda samples: torch.std(samples, dim=0, 
                                                  unbiased=True)**2
        jack_mean, jack_var = jackknife(var_estimator, grad_samples)
        quantile = sts.t.ppf(1 - conf/2, batchsize)
        var_bound = jack_mean + torch.sqrt(jack_var) * quantile
        
        #Update long-term quantities
        tot_samples += batchsize
        
    
        #Safety test
        threshold = torch.ceil(2 * var_bound / (conf * grad_norm**2))
                       
        #Log
        log_row['BatchSize'] = batchsize
        log_row['VarBound'] = var_bound.item()
        log_row['LipConst'] = lip_const
        log_row['Threshold'] = threshold.item()
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['GradNorm'] = grad_norm.item()
        log_row['TotSamples'] = tot_samples
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()
                
        #Safety test
        if batchsize < threshold:
            #Log
            log_row['StepSize'] = 0.
            log_row['Time'] = time.time() - start
            if verbose:
                print(separator)
            logger.write_row(log_row, it)
            if verbose:
                print(separator)
            #Terminate
            if verbose:
                print ('Not safe! Need at least %d samples' 
                       % int(threshold.item()))
                break
        
        #Select step size
        stepsize = (1. - math.sqrt(2 * var_bound / (conf * batchsize)) 
                    / grad_norm) / lip_const
        log_row['StepSize'] = stepsize.item()
                
        #Update policy parameters
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Log
        log_row['Time'] = time.time() - start
        if verbose:
            print(separator)
        logger.write_row(log_row, it)
        if verbose:
            print(separator)
        
        #Prepare next iteration
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()

def legacy_adastep(env, policy, horizon, pen_coeff, var_bound, *,
                    conf = 0.2,
                    batchsize = 5000,
                    iterations = float('inf'),
                    max_samples = 1e6,
                    disc = 0.9,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='AdaStep'),
                    shallow = True,
                    meta_conf = 0.05,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'danger',
                    save_params = 10000,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    Safe PG algorithm from "Adaptive Step Size for Policy Gradient Methods", 
                        Pirotta et al., 2013.
    Only for Gaussian policies.
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    pen_coeff: penalty coefficient for policy update
    var_bound: upper bound on the variance of the PG estimator
    conf: probability of failure
    batchsize: number of trajectories to estimate policy gradient
    iterations: maximum number of learning iterations
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
    meta_conf: confidence level of safe-update test (for evaluation only)
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
    verbose: level of verbosity on standard output
    """
    #Defaults
    """
    if action_filter is None:
        action_filter = clip(env)
    """
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'AdaStep',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Discount': disc,
                   'Confidence': conf,
                   'ConfidenceParam': conf,
                   'Seed': seed,
                   'BatchSize': batchsize,
                   'PenalizationCoefficient': pen_coeff,
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
    safety = 1.
    _estimator = (reinforce_estimator if estimator=='reinforce' 
                  else gpomdp_estimator)
    updated = False
    updates = 0
    unsafe_updates = 0
    eps = math.sqrt(var_bound / conf)
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
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
    
        #Collect trajectories according to fixed batch size
        batch = generate_batch(env, policy, horizon, 
                                episodes=batchsize, 
                                action_filter=action_filter,
                                n_jobs=parallel,
                                key=info_key)
            
        #Estimate policy gradient
        grad_samples = _estimator(batch, disc, policy, 
                                    baselinekind=baseline, 
                                    shallow=shallow,
                                    result='samples')
        grad = torch.mean(grad_samples, 0)
        
        lower = torch.clamp(torch.abs(grad) - eps / math.sqrt(batchsize), 0, 
                            float('inf'))
        upper = torch.abs(grad) + eps / math.sqrt(batchsize)

        #Update long-term quantities
        tot_samples += batchsize
        
        #Update safety measure
        if updates == 0:
            old_rets= returns(batch, disc)
        elif updated:
            new_rets = returns(batch, disc)
            tscore, pval = sts.ttest_ind(old_rets, new_rets)
            if pval / 2 < meta_conf and tscore > 0:
                unsafe_updates += 1
                if verbose:
                    print('The previous update was unsafe! (p-value = %f)' 
                          % (pval / 2))
            old_rets = new_rets
            safety = 1 - unsafe_updates / updates
        
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
        if torch.norm(lower) == 0:
            updated = False
            if verbose:
                print('No update, would require more samples')
        
        #Select step size
        stepsize = (torch.norm(lower)**2 /
                    (2 * pen_coeff * torch.sum(upper)**2)).item()
        log_row['StepSize'] = stepsize
                
        #Update policy parameters
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
        updated = True
        updates += 1
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        log_row['Time'] = time.time() - start
        if verbose:
            print(separator)
        logger.write_row(log_row, it)
        if verbose:
            print(separator)

        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()


def legacy_adabatch(env, policy, horizon, pen_coeff, *,
                    bound = 'bernstein',
                    var_bound = None,
                    grad_range = None,
                    fail_prob = 0.05,
                    min_batchsize = 32,
                    max_batchsize = 10000,
                    iterations = float('inf'),
                    max_samples = 1e6,
                    disc = 0.9,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='AdaBatch'),
                    shallow = True,
                    meta_conf = 0.05,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'danger',
                    oracle = None,
                    save_params = 10000,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    
    """
    Safe PG algorithm from "Adaptive Batch Size for Safe Policy Gradients",
                        Papini et al., 2017.
    Only for Gaussian policies.
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    pen_coeff: penalty coefficient for policy update
    bound: statistical inequality used to determine optimal batchsize 
        (chebyshev/student/hoeffding/bernstein)
    var_bound: upper bound on the variance of the PG estimator. Must not be 
        None if Chebyshev's bound is employed
    grad_range: theoretical range of gradient estimate. If none, it is 
        estimated from data (in a biased way)
    conf: probability of failure
    min_batchsize: minimum number of trajectories to estimate policy gradient
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
    meta_conf: confidence level of safe-update test (for evaluation only)
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
    """
    if action_filter is None:
        action_filter = clip(env)
    """
    if bound == 'chebyshev' and var_bound is None:
        raise NotImplementedError
    empirical_range = (grad_range is None)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'AdaBatch',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Discount': disc,
                   'FailProb': fail_prob,
                   'Seed': seed,
                   'MinBatchSize': min_batchsize,
                   'MaxBatchSize': max_batchsize,
                   'PenalizationCoefficient': pen_coeff,
                   'VarianceBound': var_bound,
                   'Bound': bound
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
                'GradVar',
                'GradRange',
                'Safety',
                'Err',
                'GradInfNorm']
    if oracle is not None:
        log_keys += ['Oracle']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Initializations
    it = 1
    tot_samples = 0
    safety = 1.
    optimal_batchsize = min_batchsize
    _estimator = (reinforce_estimator if estimator=='reinforce' 
                  else gpomdp_estimator)
    updated = False
    updates = 0
    unsafe_updates = 0
    params = policy.get_flat()
    max_grad = torch.zeros_like(params) - float('inf')
    min_grad = torch.zeros_like(params) + float('inf')
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
        params = policy.get_flat()
        delta = fail_prob / (it * (it + 1))
        
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
    
        #Collect trajectories according to previous optimal batch size
        batch = generate_batch(env, policy, horizon, 
                                episodes=max(min_batchsize, 
                                             min(max_batchsize, 
                                                 optimal_batchsize)), 
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
        grad_infnorm = torch.max(torch.abs(grad))
        coordinate = torch.min(torch.argmax(torch.abs(grad))).item()
        
        #Compute statistics for estimation error
        if bound in ['bernstein', 'student']:
            grad_var = torch.var(grad_samples, 0, unbiased = True)
            grad_var = torch.max(grad_var).item()
            log_row['GradVar'] = grad_var
        else:
            log_row['GradVar'] = var_bound
        if bound in ['bernstein', 'hoeffding'] and empirical_range:
            max_grad = torch.max(grad, max_grad)
            min_grad = torch.min(min_grad, grad)
            grad_range = torch.max(max_grad - min_grad).item()
            if grad_range <= 0:
                grad_range = torch.max(2 * abs(max_grad)).item()
        log_row['GradRange'] = grad_range
          
        #Compute estimation error
        if bound == 'chebyshev':
            eps = math.sqrt(var_bound / delta)
        elif bound == 'student':
            quant = sts.t.ppf(1 - delta, batchsize) 
            eps = quant * math.sqrt(grad_var)
        elif bound == 'hoeffding':
            eps = grad_range * math.sqrt(math.log(2. / delta) / 2)
        elif bound == 'bernstein':
            eps = math.sqrt(2 * grad_var * math.log(3. / delta))
            eps2 = 3 * grad_range * math.log(3. / delta)
        
        #Compute optimal batch size
        if bound in ['chebyshev', 'student', 'hoeffding']:
            optimal_batchsize = math.ceil(((13 + 3 * math.sqrt(17)) * eps**2 / 
                                           (2 * grad_infnorm**2)).item())
            min_safe_batchsize = math.ceil((eps**2 / grad_infnorm**2).item())
        else:
            min_safe_batchsize = math.ceil(((eps + math.sqrt(eps**2 
                                                            + 4 * eps2 
                                                            * grad_infnorm)) 
                                            / (2 * grad_infnorm))**2)
            optimal_batchsize = min_safe_batchsize
            _stepsize = ((grad_infnorm - eps / math.sqrt(optimal_batchsize)
                            - eps2 / optimal_batchsize)**2
                         / (2 * pen_coeff * (grad_infnorm + eps 
                            / math.sqrt(optimal_batchsize)
                            + eps2 / optimal_batchsize)**2)).item()
            ups = (grad_infnorm**2  * _stepsize * (1 - pen_coeff * _stepsize)
                    / optimal_batchsize)
            old_ups = -float('inf')
            while ups > old_ups:
                optimal_batchsize += 1
                old_ups = ups
                _stepsize = ((grad_infnorm - eps / math.sqrt(optimal_batchsize)
                            - eps2 / optimal_batchsize)**2
                         / (2 * pen_coeff * (grad_infnorm + eps 
                            / math.sqrt(optimal_batchsize)
                            + eps2 / optimal_batchsize)**2)).item()
                ups = (grad_infnorm**2  * _stepsize 
                       * (1 - pen_coeff * _stepsize)
                       / optimal_batchsize)
            optimal_batchsize -= 1
        
        if verbose:
            print('Optimal batch size: %d' % optimal_batchsize)

        #Update long-term quantities
        tot_samples += batchsize
        
        #Update safety measure
        if updates == 0:
            old_rets= returns(batch, disc)
        elif updated:
            new_rets = returns(batch, disc)
            tscore, pval = sts.ttest_ind(old_rets, new_rets)
            if pval / 2 < meta_conf and tscore > 0:
                unsafe_updates += 1
                if verbose:
                    print('The previous update was unsafe! (p-value = %f)' 
                          % (pval / 2))
            old_rets = new_rets
            safety = 1 - unsafe_updates / updates
        
        #Log
        log_row['Err'] = eps
        log_row['Safety'] = safety
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['GradNorm'] = torch.norm(grad).item()
        log_row['GradInfNorm'] = grad_infnorm.item()
        log_row['BatchSize'] = batchsize
        log_row['TotSamples'] = tot_samples
        if oracle is not None:
            log_row['Oracle'] = oracle(params.numpy())
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()
                
        #Check if number of samples is sufficient to perform update
        if grad_infnorm < eps / math.sqrt(batchsize):
            updated = False
            if verbose:
                print('No update, need more samples')
                
            #Log
            log_row['StepSize'] = 0.
            log_row['Time'] = time.time() - start
            if verbose:
                print(separator)
            logger.write_row(log_row, it)
            if verbose:
                print(separator)
            
            #Skip to next iteration (current trajectories are discarded)
            it += 1
            continue
        
        #Select step size
        if bound == 'bernstein':
            stepsize = ((grad_infnorm - eps / math.sqrt(batchsize)
                            - eps2 / batchsize)**2
                         / (2 * pen_coeff * (grad_infnorm + eps 
                            / math.sqrt(batchsize)
                            + eps2 / batchsize)**2)).item()
        else:
            stepsize = (13 - 3 * math.sqrt(17)) / (4 * pen_coeff)
        log_row['StepSize'] = stepsize
                
        #Update policy parameters
        new_params = params
        new_params[coordinate] = (params[coordinate] 
                                    + stepsize * grad[coordinate])
        policy.set_from_flat(new_params)
        updated = True
        updates += 1
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        log_row['Time'] = time.time() - start
        if verbose:
            print(separator)
        logger.write_row(log_row, it)
        if verbose:
            print(separator)
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe Policy Gradient algorithms
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import (performance, avg_horizon, mean_sum_info, 
                                      clip, seed_all_agent, returns, separator)
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator
from potion.common.logger import Logger
import scipy.stats as sts
import torch
import time
import math

def safepg(env, policy, horizon, lip_const, var_bound, *,
                    curv_oracle = False,
                    conf = 0.2,
                    min_batchsize = 32,
                    max_batchsize = 10000,
                    iterations = float('inf'),
                    max_samples = 1e6,
                    disc = 0.9,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='SPG'),
                    shallow = True,
                    meta_conf = 0.05,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'danger',
                    save_params = 100,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    Safe PG algorithm from "Smoothing Policies and Safe Policy Gradients",
                                Papini et al., 2019
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    lip_const: Lipschitz constant of the gradient (upper bound)
    var_bound: upper bound on the variance of the PG estimator
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
                   'Seed': seed,
                   'MinBatchSize': min_batchsize,
                   'MaxBatchSize': max_batchsize,
                   'LipschitzConstant': lip_const,
                   'VarianceBound': var_bound,
                   'CurvatureOracle' : curv_oracle
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
                'Safety',
                'LipConst']
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
    optimal_batchsize = min_batchsize
    min_safe_batchsize = min_batchsize
    _conf = conf
    _estimator = (reinforce_estimator if estimator=='reinforce' 
                  else gpomdp_estimator)
    updated = False
    updates = 0
    unsafe_updates = 0
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
        params = policy.get_flat()
        if curv_oracle:
            std = torch.exp(policy.get_scale_params())
        
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
        
        #Collect more trajectories to match minimum safe batch size
        do = True
        while do or batchsize < min_safe_batchsize:
            do = False
            batch += generate_batch(env, policy, horizon, 
                        episodes=(min(max_batchsize, min_safe_batchsize) 
                                    - batchsize), 
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
            min_safe_batchsize = torch.ceil(var_bound / 
                                 (conf * torch.norm(grad)**2)).item()
            if verbose and optimal_batchsize < max_batchsize:
                print('Collected %d / %d trajectories' % (batchsize, 
                                                          optimal_batchsize))
            elif verbose:
                print('Collected %d / %d trajectories' % 
                      (batchsize, min(max_batchsize, min_safe_batchsize)))
            
            #Adjust confidence before collecting more data for the same update
            if batchsize >= max_batchsize:
                break
            _conf /= 2
        
        if verbose:
            print('Optimal batch size: %d' % optimal_batchsize 
                  if optimal_batchsize < float('inf') else -1)
            print('Minimum safe batch size: %d' % min_safe_batchsize 
                  if min_safe_batchsize < float('inf') else -1)

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
        if batchsize < min_safe_batchsize:
            updated = False
            if verbose:
                print('No update, would require more samples than allowed')
            
            #Log
            log_row['StepSize'] = 0.
            log_row['Time'] = time.time() - start
            if verbose:
                print(separator)
            logger.write_row(log_row, it)
            if verbose:
                print(separator)
            
            #Adjust confidence before collecting new data for the same update
            _conf /= 2
            
            #Skip to next iteration (current trajectories are discarded)
            it += 1
            continue
        
        #Reset confidence for next update
        _conf = conf
        
        #Select step size
        if curv_oracle:
            lip_const = abs(env._hess(params, std, disc)).item()
        stepsize = 1. / lip_const * (1 - math.sqrt(var_bound) 
                    / (torch.norm(grad) * math.sqrt(batchsize * conf))).item()
        log_row['LipConst'] = lip_const
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
    
    
def safepg_exact(env, policy, horizon,
                    iterations = float('inf'),
                    max_samples = 1e6,
                    disc = 0.9,
                    action_filter = None,
                    logger = Logger(name='ESPG'),
                    shallow = True,
                    meta_conf = 0.05,
                    seed = None,
                    test_batchsize = 100,
                    info_key = 'danger',
                    save_params = 100,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    Exact version of Safe PG algorithm from "Smoothing Policies and Safe Policy Gradients",
                                Papini et al., 2019
    """
    #Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'SPG',
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Discount': disc,
                   'Seed': seed,
                   'TestBatchSize': test_batchsize,
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'GradNorm', 
                'Time',
                'Info',
                'TotSamples',
                'Safety',
                'LipConst']
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
    updates = 0
    unsafe_updates = 0
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
        params = policy.get_flat()
        std = torch.exp(policy.get_scale_params())
        
        #Test the corresponding deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, 
                                        episodes=test_batchsize, 
                                        action_filter=action_filter,
                                        n_jobs=parallel,
                                        deterministic=False,
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
    
        
        #Compute policy gradient
        grad = torch.tensor(env._grad(params, std, disc))
        
        #Update long-term quantities
        tot_samples += test_batchsize
        
        #Update safety measure
        if updates == 0:
            old_rets= returns(test_batch, disc)
        else:
            new_rets = returns(test_batch, disc)
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
        log_row['Perf'] = performance(test_batch, disc)
        log_row['ExPerf'] = env._performance(params, std, disc, horizon=horizon).item()
        log_row['Info'] = mean_sum_info(test_batch).item()
        log_row['UPerf'] = performance(test_batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(test_batch)
        log_row['GradNorm'] = torch.norm(grad).item()
        log_row['TotSamples'] = tot_samples
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
        
        #Select step size
        lip_const = abs(env._hess(params, std, disc))
        stepsize = (1. / lip_const).item()
        log_row['StepSize'] = stepsize
        log_row['LipConst'] = lip_const.item()
                
        #Update policy parameters
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
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
    

def adastep(env, policy, horizon, pen_coeff, var_bound, *,
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
                    save_params = 100,
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
    if action_filter is None:
        action_filter = clip(env)
    
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


def adabatch(env, policy, horizon, pen_coeff, *,
                    bound = 'chebyshev',
                    var_bound = None,
                    grad_range = None,
                    conf = 0.2,
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
                    save_params = 100,
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
    if action_filter is None:
        action_filter = clip(env)
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
                   'Confidence': conf,
                   'ConfidenceParam': conf,
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
            eps = math.sqrt(var_bound / conf)
        elif bound == 'student':
            quant = sts.t.ppf(1 - conf, batchsize) 
            eps = quant * math.sqrt(grad_var)
        elif bound == 'hoeffding':
            eps = grad_range * math.sqrt(math.log(2. / conf) / 2)
        elif bound == 'bernstein':
            eps = math.sqrt(2 * grad_var * math.log(3. / conf))
            eps2 = 3 * grad_range * math.log(3. / conf)
        
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
    
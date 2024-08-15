#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smooth Policy Gradient (SmoPG)
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import (performance, avg_horizon, mean_sum_info, 
                                      clip, seed_all_agent, returns, separator)
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator
from potion.common.logger import Logger
import torch
import time
import scipy.stats as sts
import math

def smoothpg(env, policy, horizon, lip_const1, lip_const2, *,
                    min_batchsize = 100,
                    max_batchsize = 1000,
                    iterations = float('inf'),
                    max_samples = 1e6,
                    disc = 0.9,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='SmoPG'),
                    shallow = True,
                    meta_conf = 0.05,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'danger',
                    save_params = 100,
                    log_params = True,
                    log_grad = True,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    Second-order semi-safe PG algorithm from "Smoothing Policies and Safe Policy Gradients,
                                    Papini et al., 2019"
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    conf: probability of unsafety (per update)
    min_batchsize: minimum number of trajectories used to estimate policy 
        gradient
    max_batchsize: maximum number of trajectories used to estimate policy 
        gradient
    iterations: maximum number of learning iterations
    max_samples: maximum number of total trajectories 
    disc: discount factor
    forget: decay of the (estimated) global gradient Lipscthiz constant
    action_filter: function to apply to the agent's action before feeding it to 
        the environment, not considered in gradient estimation. By default,
        the action is clipped to satisfy evironmental boundaries
    estimator: either 'reinforce' or 'gpomdp' (default). The latter typically
        suffers from less variance
    baseline: control variate to be used in the gradient estimator. Either
        'avg' (average reward, default), 'peters' (variance-minimizing) or
        'zero' (no baseline)
    logger: for human-readable logs (standard output, csv, tensorboard)
    shallow: whether to employ pre-computed score functions (only available for
        shallow policies)
    fast: whether to pursue maximum convergence speed 
        (under safety constraints)
    meta_conf: confidence level of safe update test (for evaluation)
    seed: random seed (None for random behavior)
    test_batchsize: number of test trajectories used to evaluate the 
        corresponding deterministic policy at each iteration. If False, no 
        test is performed
    info_key: name of the environment info to log
    save_params: how often (every x iterations) to save the policy 
        parameters to disk. Final parameters are always saved for 
        x>0. If False, they are never saved.
    log_params: whether to include policy parameters in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    parallel: number of parallel jobs for simulation. If False, 
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
    algo_info = {'Algorithm': 'SmoPG',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Discount': disc,
                   'Seed': seed,
                   'MinBatchSize': min_batchsize,
                   'MaxBatchSize': max_batchsize,
                   'LipConst2': lip_const2
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
                'Curv', 
                'StepSize1', 
                'StepSize2', 
                'Improv1', 
                'Improv2', 
                'Order']
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
    old_stepsize = 0
    old_grad = 0
    old_grad_var = 0
    unsafe_updates = 0
    safety = 1.
    tot_samples = 0
    batchsize = min_batchsize
    _estimator = (reinforce_estimator 
                  if estimator=='reinforce' else gpomdp_estimator)    
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
        params = policy.get_flat()
        
        #Test the corresponding deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon,
                                        n_episodes=test_batchsize,
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
                           n_episodes=1,
                           action_filter=action_filter,
                           render=True)
    
        #Collect trajectories
        batch = generate_batch(env, policy, horizon,
                               n_episodes=batchsize,
                               action_filter=action_filter,
                               n_jobs=parallel,
                               key=info_key)
                
        #Estimate policy gradient
        grad_samples = _estimator(batch, disc, policy, 
                                    baselinekind=baseline, 
                                    shallow=shallow,
                                    result='samples')
        grad = torch.mean(grad_samples, 0)
        grad_var = torch.var(grad_samples, 0)
    
        grad_norm = torch.norm(grad).item()
            
        #Update safety measure
        if it == 0:
            old_rets= returns(batch, disc)
        else:
            new_rets = returns(batch, disc)
            tscore, pval = sts.ttest_ind(old_rets, new_rets)
            if pval / 2 < meta_conf and tscore > 0:
                unsafe_updates += 1
                if verbose:
                    print('The previous update was unsafe! (p-value = %f)' 
                          % (pval / 2))
            old_rets = new_rets
            safety = 1 - unsafe_updates / it

        #Update long-term quantities
        tot_samples += batchsize
        
        #Log
        log_row['Safety'] = safety
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['GradNorm'] = grad_norm
        log_row['BatchSize'] = batchsize
        log_row['TotSamples'] = tot_samples
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()
                
        
        #Select step size
        stepsize1 = 1. / lip_const1
        improv1 = grad_norm**2 / (2 * lip_const1)
        max_curv = lip_const1 * grad_norm**2
        if it > 0:    
            #Estimate curvature
            curv = torch.dot(grad, grad - old_grad) / old_stepsize
            curv = torch.clamp(curv, min=-max_curv, max=max_curv).item()
            stepsize2 = 1. / (lip_const2 * grad_norm**3) * (curv + math.sqrt(curv**2 + 2 * lip_const2 * grad_norm**5))
            improv2 = stepsize2 * grad_norm**2 + stepsize2**2 / 2 * curv - stepsize2**3 / 6 * lip_const2 * grad_norm**3
        else:
            curv = -max_curv
            stepsize2 = 0
            improv2 = 0
        
        log_row['Curv'] = curv
        log_row['StepSize1'] = stepsize1
        log_row['StepSize2'] = stepsize2
        log_row['Improv1'] = improv1
        log_row['Improv2'] = improv2
        log_row['Order'] = 2 if improv2 > improv1 else 1
        
        #Select batch size
        if it > 0:
            batchsize *= torch.norm(old_grad)**2 * torch.norm(grad_var) / (grad_norm**2 * old_grad_var)
            batchsize = min(max_batchsize, max(min_batchsize, int(batchsize.item())))
        
        stepsize = stepsize2 if improv2 > improv1 else stepsize1
        log_row['StepSize'] = stepsize
        
        #Store old info for curvature estimation
        old_stepsize = stepsize
        old_grad = grad
        old_grad_var = torch.norm(grad_var)
        
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

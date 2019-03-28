#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe Policy Gradient (actor-only), practical version
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon, mean_sum_info
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator
from potion.estimation.importance_sampling import importance_weights
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
import torch
import math
import time
import scipy.stats as sts

def incr_semisafepg(env, policy, horizon, *,
                    conf = 0.05,
                    minibatch_size = 10,
                    max_batchsize = 5000,
                    iterations = 1000,
                    disc = 0.99,
                    dec = 0.9,
                    forget = 0.1,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='semisafepg'),
                    shallow = True,
                    pow_alpha = 0.1,
                    max_pow_it = 100,
                    err_tol = 0.1,
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
    Semi-Safe PG algorithm (incremental version)
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    batchsize: number of trajectories used to estimate policy gradient
    iterations: number of policy updates
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
    save_params: how often (every x iterations) to save the policy 
        parameters to disk. Final parameters are always saved for 
        x>0. If False, they are never saved.
    log_params: whether to include policy parameters in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    parallel: number of parallel jobs for simulation. If 0 or False, 
        sequential simulation is performed.
    render: how often (every x iterations) to render the agent's behavior
        on a sample trajectory. If False, no rendering happens
    verbose: level of verbosity (0: only logs; 1: normal; 2: maximum)
    """
    #Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'REINFORCE',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Disc': disc,
                   'ConfidenceParam': conf,
                   'Seed': seed,
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
                'LipConst',
                'SampleVar',
                'Info']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Learning loop
    it = 0
    old_lip_const = 0.
    old_eps = 0.
    low_samples = True
    unsafe = False
    while(it < iterations and not unsafe):
        #Begin iteration
        start = time.time()
        if verbose:
            print('\nIteration ', it)
        params = policy.get_flat()
        if verbose > 1:
            print('Parameters:', params)
        
        #Test the corresponding deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, test_batchsize, 
                                        action_filter=action_filter,
                                        seed=seed,
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
                           render=True,
                           key=info_key)
    
        #Collect trajectories
        batch = []
        while True:
            batch += generate_batch(env, policy, horizon, minibatch_size, 
                                   action_filter=action_filter, 
                                   seed=seed,
                                   n_jobs=parallel,
                                   key=info_key)
            batchsize = len(batch)
        
            #Estimate policy gradient
            _estimator = reinforce_estimator if estimator=='reinforce' else gpomdp_estimator
            grad_samples = _estimator(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='samples')
            grad = torch.mean(grad_samples, 0)
            
            #Compute estimation error 
            grad_var = torch.sum(torch.var(grad_samples, 0, unbiased=True))
            quant = 2 * sts.t.interval(1 - conf, batchsize-1,loc=0.,scale=1.)[1]
            eps = quant * torch.sqrt(grad_var).item()
            if it > 0:
                eps = (1 - forget) * old_eps + forget * eps
            
            #Optimal batch size
            opt_batchsize = torch.ceil(4 * eps**2 / 
                                   (torch.norm(grad)**2)).item() #next
            if batchsize >= opt_batchsize or batchsize > max_batchsize:
                break
        
        old_eps = eps
        batchsize = min(batchsize, max_batchsize)
        low_samples = batchsize > max_batchsize 
        log_row['SampleVar'] = grad_var.item()
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['GradNorm'] = torch.norm(grad).item()
        log_row['BatchSize'] = batchsize #current
        
        #Power method
        if verbose:
            print('Estimating the Lipschitz Constant')
        
 
        err = 999
        attempts = 0       
        _pow_alpha = pow_alpha
        while err > err_tol:
            pow_it = 0
            psi = torch.rand_like(grad)
            _lip_const = torch.norm(psi).item()
            attempts += 1
            while err > err_tol and pow_it < max_pow_it:
                params_2 = params + _pow_alpha * psi / torch.norm(psi)
                policy.set_from_flat(params_2)
                grad_2_samples = _estimator(batch, disc, policy, 
                                          baselinekind=baseline,
                                          shallow=shallow,
                                          result='samples')
                policy.set_from_flat(params)
                iws = importance_weights(batch, policy, params_2, normalize=True)
                grad_2 = torch.sum(grad_2_samples * iws.unsqueeze(1))
                psi = 1. / _pow_alpha * (grad_2 - grad)
                lip_const = torch.norm(psi).item()
                if math.isnan(lip_const):
                    err = 999
                    break
                err = abs(lip_const - _lip_const)
                _lip_const = lip_const
                pow_it += 1
                #print('lip: %f, err: %f' % (lip_const, err))
            policy.set_from_flat(params)
            _pow_alpha /= 10
        
        if verbose:
            print('Converged (err = %f) after %d iterations, %d-th attempt' % (err, pow_it, attempts))
        if it > 0:
            lip_const = (1 - forget) * old_lip_const + forget * lip_const
        old_lip_const = lip_const
        log_row['LipConst'] = lip_const
        
        #Select step size        
        stepsize = 1. /(2 * lip_const)
        if low_samples:    
            stepsize *= (1 - eps / (torch.norm(grad) * batchsize).item())
        log_row['StepSize'] = stepsize
                
        #Update policy parameters
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
        
        #Log
        log_row['Time'] = time.time() - start
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()
        logger.write_row(log_row, it)
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()
    

def semisafepg(env, policy, horizon, *,
                    conf = 0.2,
                    init_batchsize = 100,
                    max_batchsize = 5000,
                    iterations = 1000,
                    disc = 0.99,
                    dec = 0.9,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='semisafepg'),
                    shallow = True,
                    pow_alpha = 0.01,
                    max_pow_it = 100,
                    err_tol = 0.1,
                    info_key = 'danger',
                    seed = None,
                    test_batchsize = False,
                    save_params = 100,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    SafePG algorithm
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    batchsize: number of trajectories used to estimate policy gradient
    iterations: number of policy updates
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
    save_params: how often (every x iterations) to save the policy 
        parameters to disk. Final parameters are always saved for 
        x>0. If False, they are never saved.
    log_params: whether to include policy parameters in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    parallel: number of parallel jobs for simulation. If 0 or False, 
        sequential simulation is performed.
    render: how often (every x iterations) to render the agent's behavior
        on a sample trajectory. If False, no rendering happens
    verbose: level of verbosity (0: only logs; 1: normal; 2: maximum)
    """
    #Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'REINFORCE',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Disc': disc,
                   'ConfidenceParam': conf,
                   'InitialBatchSize': init_batchsize,
                   'Seed': seed,
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
                'LipConst',
                'SampleVar']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Learning loop
    it = 0
    batchsize = init_batchsize
    low_samples = True
    unsafe = False
    while(it < iterations and not unsafe):
        #Begin iteration
        start = time.time()
        if verbose:
            print('\nIteration ', it)
        params = policy.get_flat()
        if verbose > 1:
            print('Parameters:', params)
        
        #Test the corresponding deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, test_batchsize, 
                                        action_filter=action_filter,
                                        seed=seed,
                                        parallel=parallel,
                                        deterministic=True,
                                        key=info_key)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['UTestPerf'] = performance(test_batch, 1)
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon, 
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True,
                           key=info_key)
    
        #Collect trajectories
        if verbose:
            print('Sampling %d trajectories' % batchsize)
        batch = generate_batch(env, policy, horizon, batchsize, 
                               action_filter=action_filter, 
                               seed=seed,
                               n_jobs=parallel,
                               key=info_key)
        log_row['Perf'] = performance(batch, disc)
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
    
        #Estimate policy gradient
        _estimator = reinforce_estimator if estimator=='reinforce' else gpomdp_estimator
        grad_samples = _estimator(batch, disc, policy, 
                                    baselinekind=baseline, 
                                    shallow=shallow,
                                    result='samples')
        grad = torch.mean(grad_samples, 0)
        if verbose > 1:
            print('Gradients: ', grad)
        log_row['GradNorm'] = torch.norm(grad).item()
        
        #Power method
        if verbose:
            print('Estimating the Lipschitz Constant')
        psi = torch.rand_like(grad)
        old_lip_const = torch.norm(psi).item()
        err = 999
        pow_it = 0
        while err > err_tol and pow_it < max_pow_it:
            params_2 = params + pow_alpha * psi / torch.norm(psi)
            policy.set_from_flat(params_2)
            grad_2_samples = _estimator(batch, disc, policy, 
                                      baselinekind=baseline,
                                      shallow=shallow,
                                      result='samples')
            policy.set_from_flat(params)
            iws = importance_weights(batch, policy, params_2, normalize=True)
            grad_2 = torch.sum(grad_2_samples * iws.unsqueeze(1))
            psi = 1. / pow_alpha * (grad_2 - grad)
            lip_const = torch.norm(psi).item()
            err = abs(lip_const - old_lip_const)
            old_lip_const = lip_const
            pow_it += 1
            #print('lip: %f, err: %f' % (lip_const, err))
        policy.set_from_flat(params)
        if verbose:
            if err <= err_tol:
                print('Converged (err = %f) after %d iterations' % (err, pow_it))
            else:
                print('Still err = %f after %d iterations' % (err, pow_it))
        log_row['LipConst'] = lip_const
        
        #Compute estimation error 
        grad_var = torch.sum(torch.var(grad_samples, 0, unbiased=True))
        log_row['SampleVar'] = grad_var.item()
        quant = 2 * sts.t.interval(1 - conf, batchsize-1,loc=0.,scale=1.)[1]
        eps = quant * torch.sqrt(grad_var).item()
        print('PG estimation error:', eps)

        #Select step size        
        stepsize = 1. /(2 * lip_const)
        if low_samples:    
            stepsize *= (1 - eps / (torch.norm(grad) * batchsize).item())
        log_row['StepSize'] = stepsize
        
        #Select batch size
        log_row['BatchSize'] = batchsize #current
        batchsize = torch.ceil(4 * eps**2 / 
                               (torch.norm(grad)**2)).item() #next
        low_samples = batchsize > max_batchsize
        batchsize = min(batchsize, max_batchsize)
        
        #Update policy parameters
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
        
        #Log
        log_row['Time'] = time.time() - start
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()
        logger.write_row(log_row, it)
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()
    
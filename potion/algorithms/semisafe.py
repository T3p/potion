#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semi-Safe Policy Gradient (SSPG)
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon, mean_sum_info
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
import torch
from potion.estimation.eigenvalues import power
import time
import scipy.stats as sts

def semisafepg(env, policy, horizon, *,
                    conf = 0.05,
                    min_batchsize = 10,
                    max_batchsize = 10000,
                    iterations = float('inf'),
                    max_samples = 2e6,
                    disc = 0.99,
                    forget = 0.1,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='SSPG'),
                    shallow = True,
                    pow_alpha = 0.01,
                    max_pow_it = 100,
                    pow_err_tol = 0.1,
                    pow_clip = 0.2,
                    max_pow_attempts = 3,
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
    conf: probability of failure
    min_batchsize: minimum number of trajectories used to estimate policy gradient
    max_batchsize: maximum number of trajectories used to estimate policy gradient
    iterations: maximum number of policy updates
    max_samples: maximum number of total trajectories 
    disc: discount factor
    forget: decay of the estimated global constants
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
    pow_alpha: initial step size of the power method
    max_pow_it: maximum number of per-attempt iterations of the power method
    err_tol: error tolerance of the power method
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
    algo_info = {'Algorithm': 'SSPG',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Discount': disc,
                   'ConfidenceParam': conf,
                   'Seed': seed,
                   'MinBatchSize': min_batchsize,
                   'MaxBatchSize': max_batchsize,
                   'ForgetParam': forget,
                   'PowerStep': pow_alpha,
                   'PowerIters': max_pow_it,
                   'PowerTolerance': pow_err_tol
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
                'ErrBound',
                'SampleVar',
                'Info',
                'TotSamples']
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
    tot_samples = 0
    _estimator = reinforce_estimator if estimator=='reinforce' else gpomdp_estimator
    while(it < iterations and tot_samples < max_samples):
        #Begin iteration
        start = time.time()
        if verbose:
            print('\nIteration ', it)
        params = policy.get_flat()
        
        #Test the corresponding deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, 
                                        episodes=test_batchsize, 
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
            batch += generate_batch(env, policy, horizon, 
                                   episodes=min_batchsize, 
                                   action_filter=action_filter, 
                                   seed=seed,
                                   n_jobs=parallel,
                                   key=info_key)
            batchsize = len(batch)
        
            #Estimate policy gradient
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
                                   (torch.norm(grad)**2)).item()
            if batchsize >= opt_batchsize:
                low_samples = False
                break
            elif batchsize >= max_batchsize:
                low_samples = True
                break
        tot_samples += batchsize
        old_eps = eps
        
        #Log
        log_row['SampleVar'] = grad_var.item()
        log_row['ErrBound'] = eps
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
                
        #Check if number of samples is sufficient
        min_safe_batchsize = eps**2 / torch.norm(grad).item()**2
        if batchsize < min_safe_batchsize:
            print('Unsafe, stopping. Would require %d samples' % min_safe_batchsize)
            log_row['LipConst'] = old_lip_const
            log_row['StepSize'] = 0.
            log_row['Time'] = time.time() - start
            logger.write_row(log_row, it)
            break
        
        #Power method
        if verbose:
            print('Estimating the Lipschitz Constant')
            start_pow = time.time()
        lip_const = power(policy, batch, grad, disc, 
              pow_alpha=pow_alpha, 
              err_tol=pow_err_tol, 
              max_it=max_pow_it, 
              max_attempts=max_pow_attempts, 
              estimator=_estimator, 
              baseline=baseline, 
              shallow=shallow, 
              clip=pow_clip,
              verbose=verbose)
        if verbose:
            print('Spectral radius found in %f seconds' % (time.time() - start_pow))
        if it > 0:
            lip_const = (1 - forget) * old_lip_const + forget * lip_const
        old_lip_const = lip_const
        log_row['LipConst'] = lip_const
        
        #Select step size        
        stepsize = 1. /(2 * lip_const)
        if low_samples:
            if verbose:
                print('Low sample regire: reducing step size')
            stepsize *= 2. * (1 - eps / (torch.norm(grad) * batchsize).item())
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
    
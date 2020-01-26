#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE family of algorithms (actor-only policy gradient)
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon, mean_sum_info
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator, egpomdp_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
from potion.meta.steppers import ConstantStepper
import torch
import time


def reinforce(env, policy, horizon, *,
                    batchsize = 100, 
                    iterations = 1000,
                    disc = 0.99,
                    stepper = ConstantStepper(1e-2),
                    entropy_coeff = 0.,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'avg',
                    logger = Logger(name='gpomdp'),
                    shallow = False,
                    seed = None,
                    estimate_var = False,
                    test_batchsize = False,
                    info_key = 'danger',
                    save_params = 100,
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    REINFORCE/G(PO)MDP algorithmn
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    batchsize: number of trajectories used to estimate policy gradient
    iterations: number of policy updates
    disc: discount factor
    stepper: step size criterion. A constant step size is used by default
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
                   'BatchSize': batchsize, 
                   'Disc': disc, 
                   'StepSizeCriterion': str(stepper), 
                   'Seed': seed,
                   'EntropyCoefficient': entropy_coeff
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'GradNorm', 
                'Time',
                'StepSize',
                'Exploration',
                'Entropy',
                'Info']
    if estimate_var:
        log_keys.append('SampleVar')
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
    while(it < iterations):
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
                                        njobs=parallel,
                                        deterministic=True,
                                        key=info_key)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['TestInfo'] = mean_sum_info(test_batch).item()
            log_row['UTestPerf'] = performance(test_batch, 1)
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon, 
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True,
                           key=info_key)
    
        #Collect trajectories
        batch = generate_batch(env, policy, horizon, batchsize, 
                               action_filter=action_filter, 
                               seed=seed, 
                               n_jobs=parallel,
                               key=info_key)
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['Exploration'] = policy.exploration().item()
        log_row['Entropy'] = policy.entropy(0.).item()
    
        #Estimate policy gradient
        result = 'samples' if estimate_var else 'mean'
        if estimator == 'gpomdp' and entropy_coeff == 0:
            grad_samples = gpomdp_estimator(batch, disc, policy, 
                                    baselinekind=baseline, 
                                    shallow=shallow,
                                    result=result)
        elif estimator == 'gpomdp':
            grad_samples = egpomdp_estimator(batch, disc, policy, entropy_coeff,
                                     baselinekind=baseline,
                                     shallow=shallow,
                                    result=result)
        elif estimator == 'reinforce':
            grad_samples = reinforce_estimator(batch, disc, policy, 
                                       baselinekind=baseline, 
                                       shallow=shallow,
                                    result=result)
        else:
            raise ValueError('Invalid policy gradient estimator')
        
        if estimate_var:
            grad = torch.mean(grad_samples, 0)
            centered = grad_samples - grad.unsqueeze(0)
            grad_cov = (batchsize/(batchsize - 1) * 
                        torch.mean(torch.bmm(centered.unsqueeze(2), 
                                             centered.unsqueeze(1)),0))
            grad_var = torch.sum(torch.diag(grad_cov)).item() #for humans
        else:
            grad = grad_samples
        
        if verbose > 1:
            print('Gradients: ', grad)
        log_row['GradNorm'] = torch.norm(grad).item()
        if estimate_var:
            log_row['SampleVar'] = grad_var
        
        #Select meta-parameters
        stepsize = stepper.next(grad)
        if not torch.is_tensor(stepsize):
            stepsize = torch.tensor(stepsize)
        log_row['StepSize'] = torch.norm(stepsize).item()
        
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
    
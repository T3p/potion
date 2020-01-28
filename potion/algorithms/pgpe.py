#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE family of algorithms (actor-only policy gradient)
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import light_episode_generator
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
from potion.meta.steppers import ConstantStepper
import potion.common.torch_utils as tu
import torch
import time

def evaluate_hyperpolicy(env, hyperpolicy, horizon, disc, batchsize, action_filter, info_key, deterministic=False, parallel=False):
    if not parallel:
        policy_params = []
        rets = []
        urets = []
        ep_lens = []
        info_sums = []
        for _ in range(batchsize):
            #Sample policy parameters
            policy_params.append(hyperpolicy.resample(deterministic=deterministic))
            #Generate trajectory
            ret, uret, ep_len, info_sum = light_episode_generator(env, hyperpolicy.lower_policy, horizon, disc,
                               action_filter=action_filter,
                               key=info_key)
            rets.append(ret)
            urets.append(uret)
            ep_lens.append(ep_len)
            info_sums.append(info_sum)
    else:
        raise NotImplementedError
        
    return policy_params, rets, urets, ep_lens, info_sums


def pgpe(env, hyperpolicy, horizon, *,
                    batchsize = 100, 
                    iterations = 1000,
                    disc = 0.99,
                    stepper = ConstantStepper(1e-2),
                    natural = True,
                    action_filter = None,
                    baseline = 'peters',
                    logger = Logger(name='pgpe'),
                    seed = None,
                    test_batchsize = False,
                    info_key = 'danger',
                    save_params = 100,
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    (N)PGPE algorithmn
        
    env: environment
    hyperpolicy: the one to improve
    horizon: maximum task horizon
    batchsize: number of trajectories used to estimate policy gradient
    iterations: number of policy updates
    disc: discount factor
    stepper: step size criterion. A constant step size is used by default
    natural: whether to use natural gradient
    action_filter: function to apply to the agent's action before feeding it to 
        the environment, not considered in gradient estimation. By default,
        the action is clipped to satisfy evironmental boundaries
    baseline: control variate to be used in the gradient estimator. Either
        'avg' (average reward, default), 'sugiyama' (variance-minimizing) or
        'zero' (no baseline)
    logger: for human-readable logs (standard output, csv, tensorboard...)
    seed: random seed (None for random behavior)
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
    algo_info = {'Algorithm': 'PGPE',
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'BatchSize': batchsize, 
                   'Disc': disc, 
                   'StepSizeCriterion': str(stepper), 
                   'Seed': seed,
                   'Natural': natural
                   }
    logger.write_info({**algo_info, **hyperpolicy.info()})
    log_keys = ['Perf',
                'UPerf',
                'AvgHorizon',
                'StepSize', 
                'GradNorm', 
                'Time',
                'StepSize',
                'Info']
    if test_batchsize:
        log_keys += ['TestPerf']
    if log_params:
        log_keys += ['param%d' % i for i in range(hyperpolicy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(hyperpolicy.num_params())]
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Learning loop
    it = 0
    while(it < iterations):
        #Begin iteration
        start = time.time()
        if verbose:
            print('\nIteration ', it)
        params = hyperpolicy.get_flat()
        if verbose > 1:
            print('Parameters:', params)
                
        #Render the agent's behavior
        if render and it % render==0:
            hyperpolicy.resample()
            light_episode_generator(env, hyperpolicy.lower_policy, horizon, disc,
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True,
                           key=info_key)
    
        if test_batchsize:
            _, test_rets, _, _, _ = evaluate_hyperpolicy(env, hyperpolicy, horizon, disc, test_batchsize, action_filter, info_key, 
                                                                                  deterministic=True,
                                                                                  parallel=parallel)
            log_row['TestPerf'] = torch.mean(torch.tensor(test_rets, dtype=torch.float)).item()

    
        #Collect trajectories
        policy_params, rets, urets, ep_lens, info_sums = evaluate_hyperpolicy(env, hyperpolicy, horizon, disc, batchsize, action_filter, info_key, parallel)
        
        policy_params = torch.stack(policy_params, 0)
        rets = torch.tensor(rets, dtype=torch.float)
        log_row['Perf'] = torch.mean(rets).item()
        log_row['UPerf'] = torch.mean(torch.tensor(urets, dtype=torch.float)).item()
        log_row['Info'] = torch.mean(torch.tensor(info_sums, dtype=torch.float)).item()
        log_row['AvgHorizon'] = torch.mean(torch.tensor(ep_lens, dtype=torch.float)).item()
    
        #Baseline
        if baseline == 'avg':
            b = torch.mean(rets)
        elif baseline == 'sugiyama':
            b = torch.mean(torch.norm(hyperpolicy.score(policy_params), 1)**2 * rets.unsqueeze(-1), 0) \
                / torch.mean(torch.norm(hyperpolicy.score(policy_params), 1)**2, 0)
        elif baseline == 'peters':
            b = torch.mean(hyperpolicy.score(policy_params)**2 * rets.unsqueeze(-1), 0) \
                / torch.mean(hyperpolicy.score(policy_params)**2, 0)
        else:
            b = torch.tensor(0.)
    
        #Estimate gradient
        grad = torch.mean(hyperpolicy.score(policy_params) * (rets.unsqueeze(-1) - b), 0)
        if natural:
            grad = grad / hyperpolicy.fisher()
        
        if verbose > 1:
            print('Gradients: ', grad)
        log_row['GradNorm'] = torch.norm(grad).item()
        
        #Select meta-parameters
        stepsize = stepper.next(grad)
        stepsize = tu.maybe_tensor(stepsize)
        log_row['StepSize'] = torch.norm(stepsize).item()
        
        #Update policy parameters
        new_params = params + stepsize * grad
        hyperpolicy.set_from_flat(new_params)
        
        #Log
        log_row['Time'] = time.time() - start
        if log_params:
            for i in range(hyperpolicy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(hyperpolicy.num_params()):
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
    
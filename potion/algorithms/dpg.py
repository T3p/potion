#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:12:52 2020

@author: matteo
"""
from potion.simulation.trajectory_generators import light_episode_generator, generate_batch
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent, performance, mean_sum_info
import torch
import time
import numpy as np


def dpg(env, 
        policy,
        horizon, 
        batchsize=100, 
        iterations=1000, 
        disc=0.99, 
        noise=0.1,
        actor_step=1e-3, 
        critic_step=1e-2,
        v_step=1e-2,
        u_step=1e-2,
        natural=True,
        action_filter = None,
        logger = Logger(name='dpg'),
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
    DPG algorithm (COPDAC-GQ version)
    Step-based, but with episodic interface
    """
    
    #Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
        
    #Prepare logger
    algo_info = {'Algorithm': 'DPG',
                   'Env': str(env), 
                   'Horizon': horizon,
                   'BatchSize': batchsize, 
                   'Disc': disc, 
                   'ActorStep': actor_step,
                   'CriticStep': critic_step,
                   'Seed': seed,
                   'Natural': natural
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf',
                'UPerf',
                'AvgHorizon',
                'GradNorm', 
                'Time',
                'Info']
    if test_batchsize:
        log_keys += ['TestPerf', 'TestInfo', 'UTestPerf']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    

    #Learning loop
    it = 0
    s = env.reset()
    s = np.array(s, dtype=np.float)
    s = torch.tensor(s, dtype=torch.float).view(-1)
    a = policy.act(s)
    if action_filter is not None:
        a = action_filter(a)
    critic_params = torch.zeros_like(policy.score(s, a))
    v = torch.zeros_like(policy.feat(s))
    u = torch.zeros_like(critic_params)
    while(it < iterations):
        #Begin iteration
        start = time.time()
        if verbose:
            print('\nIteration ', it)
        params = policy.get_flat()
        grads = critic_params
        if verbose > 1:
            print('Parameters:', params)
                
        #Render the agent's behavior
        if render and it % render==0:
            light_episode_generator(env, policy, horizon, disc,
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True,
                           key=info_key)
    
        #Test the (deterministic) policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, test_batchsize, 
                                        action_filter=action_filter,
                                        seed=seed,
                                        n_jobs=parallel,
                                        key=info_key)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['TestInfo'] = mean_sum_info(test_batch).item()
            log_row['UTestPerf'] = performance(test_batch, 1)
        
        rets = []
        urets = []
        ep_lens = []
        info_sums = []
        while len(rets) < batchsize:
            ret = 0
            uret = 0
            info_sum = 0
            t = 0
            done = False
            s = env.reset()
            s = np.array(s, dtype=np.float)
            s = torch.tensor(s, dtype=torch.float).view(-1)
            while not done and t < horizon:
                actor_params = policy.get_flat()
                a = policy.act(s, noise)
                if action_filter is not None:
                    a = action_filter(a)
                next_s, r, done, info = env.step(a.numpy())
                next_s = np.array(next_s, dtype=np.float)
                next_s = torch.tensor(next_s, dtype=torch.float).view(-1)
                                    
                #Here the magic happens
                next_det_a = policy.act(next_s)
                if not done:
                    td_err = r + torch.dot(critic_params, disc * policy.score(next_s, next_det_a) - policy.score(s, a))
                else:
                    td_err = r - torch.dot(critic_params, policy.score(s,a)) #Q(T) = 0
                new_actor_params = actor_params + actor_step * critic_params
                new_critic_params = critic_params + critic_step * td_err * policy.score(s,a) -\
                    critic_step * disc * policy.score(next_s, next_det_a) * torch.dot(policy.score(s,a), u)
                new_v = v + v_step * td_err * policy.feat(s) - \
                    v_step * disc * policy.feat(next_s) * torch.dot(policy.score(s,a), u)
                new_u = u + u_step * (td_err - torch.dot(policy.score(s,a), u)) * policy.score(s,a)
                
                ret += disc**t * r
                uret += r
                if info_key is not None and info_key in info:
                    info_sum += info[info_key]
                policy.set_params(new_actor_params)
                critic_params = new_critic_params
                v = new_v
                u = new_u
                s = next_s
                t += 1
            rets.append(ret)
            urets.append(uret)
            ep_lens.append(t)
            info_sums.append(info_sum)
            
        log_row['Perf'] = torch.mean(torch.tensor(rets)).item()
        log_row['UPerf'] = torch.mean(torch.tensor(urets, dtype=torch.float)).item()
        log_row['Info'] = torch.mean(torch.tensor(info_sums, dtype=torch.float)).item()
        log_row['AvgHorizon'] = torch.mean(torch.tensor(ep_lens, dtype=torch.float)).item()
        log_row['GradNorm'] = torch.norm(grads).item()
    
        #Log
        log_row['Time'] = time.time() - start
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grads[i].item()
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
        
                    
                    
                
        
        
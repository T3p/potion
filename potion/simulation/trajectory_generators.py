#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 01:17:17 2019

@author: Matteo Papini
"""

import torch
import numpy as np
from joblib import Parallel, delayed
from potion.common.misc_utils import seed_all_agent
import random
import math

def sequential_episode_generator(env, policy, horizon=float('inf'), max_episodes=float('inf'),
                                 action_filter=None, render=False, deterministic=False, key=None):
    
    ds = sum(env.observation_space.shape)
    ds = max(ds, 1)
    da = sum(env.action_space.shape)
    da = max(da, 1)
    
    n = 0
    while n < max_episodes:
        # Episode
        states = torch.zeros((horizon, ds),
                             dtype=torch.float)
        actions = torch.zeros((horizon, da),
                              dtype=torch.float)
        rewards = torch.zeros(horizon, dtype=torch.float)
        mask = torch.zeros(horizon, dtype=torch.float)
        infos = torch.zeros(horizon, dtype=torch.float)
        s = env.reset()
        done = False
        t = 0
        if render:
            try:    
                env.render()
            except:
                pass
        while not done and t < horizon:
            s = np.array(s, dtype=np.float)
            s = torch.tensor(s, dtype=torch.float).view(-1)
            a = policy.act(s, deterministic)
            if not torch.is_tensor(a):
                a = torch.tensor(a)
            a = a.view(-1)
            if action_filter is not None:
                a = action_filter(a)

            next_s, r, done, info = env.step(a)
            if render:
                try:
                    env.render()
                except:
                    print(s, a, r, done, info)
                    print()
            states[t] = s
            actions[t] = a
            rewards[t] = r
            if key is not None and key in info:
                infos[t] = info[key]
            mask[t] = 1
            
            s = next_s
            t += 1
        
        yield states, actions, rewards, mask, infos
        n += 1


def parallel_episode_generator(env, policy, horizon=float('inf'), action_filter=None, seed=None, deterministic=False, key=None):
        ds = sum(env.observation_space.shape)
        ds = max(ds, 1)
        da = sum(env.action_space.shape)
        da = max(da, 1)    
    
        env.seed(seed)
        seed_all_agent(seed)
        states = torch.zeros((horizon, ds),
                             dtype=torch.float)
        actions = torch.zeros((horizon, da),
                              dtype=torch.float)
        rewards = torch.zeros(horizon, dtype=torch.float)
        mask = torch.zeros(horizon, dtype=torch.float)
        infos = torch.zeros(horizon, dtype=torch.float)
        s = env.reset()
        done = False
        t = 0
        while not done and t < horizon:
            s = np.array(s, dtype=np.float)
            s = torch.tensor(s, dtype=torch.float).view(-1)
            a = policy.act(s, deterministic)
            a = torch.tensor(a, dtype=torch.float).view(-1)
            if action_filter is not None:
                a = action_filter(a)
            next_s, r, done, info = env.step(a.numpy())
            
            states[t] = s
            actions[t] = a
            rewards[t] = r
            mask[t] = 1
            if key is not None and key in info:
                infos[t] = info[key]
            
            s = next_s
            t += 1
        return states, actions, rewards, mask, infos
    
def light_episode_generator(env, policy, horizon=float('inf'), disc=1., action_filter=None, key=None):
        s = env.reset()
        done = False
        t = 0
        ret = 0.
        uret = 0.
        info_sum = 0.
        while not done and t < horizon:
            s = np.array(s, dtype=np.float)
            s = torch.tensor(s, dtype=torch.float).view(-1)
            a = policy.act(s)
            a = torch.tensor(a, dtype=torch.float).view(-1)
            if action_filter is not None:
                a = action_filter(a)
            next_s, r, done, info = env.step(a.numpy())
            ret += disc**t * r
            uret += r
            if key is not None and key in info:
                info_sum += info[key]
            
            s = next_s
            t += 1
        return ret, uret, t, info_sum
    
def generate_batch(env, policy, horizon, episodes, action_filter=None, render=False, n_jobs=False, seed=None, deterministic=False, key=None):
    """Batch: list of (features, actions, rewards, mask) tuples"""
    if not n_jobs:
        gen = sequential_episode_generator(env, policy, horizon, episodes, action_filter, render, deterministic, key)
        batch = [ep for ep in gen]
    else:
        if seed is None:
            seed = random.randint(0,999999)
        batch = Parallel(n_jobs=n_jobs)(delayed(parallel_episode_generator)(env, policy, horizon, action_filter, seed=seed*10000+i, deterministic=deterministic, key=key) for i in range(episodes))
    return batch


def evaluate_hyperpolicy(env, hyperpolicy, horizon, disc, episodes, resample=True, action_filter=None, render=False, n_jobs=False, seed=None, key=None):
    if not n_jobs:
        gen = sequential_hyperpolicy_evaluator(env, hyperpolicy, horizon, disc, episodes, resample,
                                 action_filter, render, key)
        batch = [ep for ep in gen]
    else:
        if seed is None:
            seed = random.randint(0,999999)
        batch = Parallel(n_jobs=n_jobs)(delayed(parallel_hyperpolicy_evaluator)(env, hyperpolicy, horizon, disc, episodes, resample,
                                 action_filter, render, seed*10000+i, key) for i in range(episodes))
    return batch

    
"""Testing"""
if __name__ == '__main__':
    from potion.actors.continuous_policies import SimpleGaussianPolicy as Gaussian
    import gym.spaces
    env = gym.make('MountainCarContinuous-v0')
    policy = Gaussian(2, 1)
    
    batch = generate_batch(env, policy, 3, 4)
    print(batch)

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

def sequential_episode_generator(env, policy, horizon=float('inf'), max_episodes=float('inf'),
                                 action_filter=None, render=False, deterministic=False):
    n = 0
    while n < max_episodes:
        # Episode
        states = torch.zeros((horizon, sum(env.observation_space.shape)),
                             dtype=torch.float)
        actions = torch.zeros((horizon, sum(env.action_space.shape)),
                              dtype=torch.float)
        rewards = torch.zeros(horizon, dtype=torch.float)
        mask = torch.zeros(horizon, dtype=torch.float)
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
            next_s, r, done, _ = env.step(a)
            if render:
                env.render()
            
            states[t] = s
            actions[t] = a
            rewards[t] = r
            mask[t] = 1
            
            s = next_s
            t += 1
            
        yield states, actions, rewards, mask
        n += 1

def parallel_episode_generator(env, policy, horizon=float('inf'), action_filter=None, seed=None, deterministic=False):
        env.seed(seed)
        seed_all_agent(seed)
        states = torch.zeros((horizon, sum(env.observation_space.shape)),
                             dtype=torch.float)
        actions = torch.zeros((horizon, sum(env.action_space.shape)),
                              dtype=torch.float)
        rewards = torch.zeros(horizon, dtype=torch.float)
        mask = torch.zeros(horizon, dtype=torch.float)
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
            next_s, r, done, _ = env.step(a)
            
            states[t] = s
            actions[t] = a
            rewards[t] = r
            mask[t] = 1
            
            s = next_s
            t += 1
        return states, actions, rewards, mask

def generate_batch(env, policy, horizon, episodes, action_filter=None, parallel=False, render=False, n_jobs=4, seed=None, deterministic=False):
    """Batch: list of (features, actions, rewards, mask) tuples"""
    if not parallel:
        gen = sequential_episode_generator(env, policy, horizon, episodes, action_filter, render, deterministic)
        batch = [ep for ep in gen]
    else:
        batch = Parallel(n_jobs=n_jobs)(delayed(parallel_episode_generator)(env, policy, horizon, action_filter, seed=seed*1000+i, deterministic=deterministic) for i in range(episodes))
    return batch

"""Testing"""
if __name__ == '__main__':
    from potion.actors.continuous_policies import SimpleGaussianPolicy as Gaussian
    import gym.spaces
    env = gym.make('MountainCarContinuous-v0')
    policy = Gaussian(2, 1)
    
    batch = generate_batch(env, policy, 3, 4)
    print(batch)
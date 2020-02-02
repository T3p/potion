#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:51:13 2020

@author: matteo
"""

import safety_envs
import gym
from potion.actors.continuous_deterministic_policies import ShallowDeterministicPolicy
import torch
from potion.simulation.play import play
from potion.common.misc_utils import returns
import torch

env = gym.make('BasicReach-v0')
m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
param_init=torch.tensor([0.11200468987226486, 0.038586314767599106, 
                         4.222198009490967, 0.10747773945331573, 
                         0.006883372087031603, 0.12164688855409622, 
                         0.011850409209728241, -0.1272290199995041, 
                         -0.046683888882398605, 0.01381702534854412, 
                         0.2929554879665375, 0.11795885115861893, 
                         -14.66159439086914, 1.1246546506881714, 
                         -0.08598880469799042, -0.17306293547153473, 
                         -0.0005653838161379099, -0.3166707754135132])
policy = policy = ShallowDeterministicPolicy(m, d, 
                                             squash_fun=torch.tanh,
                                             param_init=param_init)

batch = play(env, policy, horizon = 100, episodes = 1, render=True)
print()
print(returns(batch))
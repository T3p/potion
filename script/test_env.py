#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:36:25 2019

@author: matteo
"""

import gym
import gym.spaces
import potion.envs
from potion.simulation.play import play
from potion.actors.continuous_policies import ShallowGaussianPolicy
import torch
from potion.common.misc_utils import performance

env = gym.make('Drone-v0')
ds = 3
da = 1
mu_init = torch.tensor([-1.76882298, -0.66070184,  0.95622158])
logstd_init = torch.log(torch.tensor(1.))
policy = ShallowGaussianPolicy(ds, da, 
                               mu_init=mu_init, 
                               logstd_init=logstd_init, 
                               learn_std=False)

batch = play(env, policy, horizon=20, episodes=100, render=False, action_filter=None)
print(performance(batch, disc=0.9))
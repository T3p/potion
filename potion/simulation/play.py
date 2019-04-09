#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matteo Papini
"""

from potion.actors.continuous_policies import ShallowGaussianPolicy
import potion.envs
import gym    
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance
import math

horizon = 100
episodes = 100
disc = 0.9
std = 1.
env = gym.make('lqr1d-v0')
render = False

param = env._optimal_param(disc)
print('Optimal param:', param)
policy = ShallowGaussianPolicy(1, 1, mu_init=[param], logstd_init=math.log(std))
batch = generate_batch(env, policy, horizon, episodes, render=render)
print('Expected:', env._performance(param, std, disc))
print('Actual:', performance(batch, disc))
env.close()

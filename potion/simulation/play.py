#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matteo Papini
"""

from potion.actors.continuous_policies import ShallowGaussianPolicy
import potion.envs
import gym    
from potion.simulation.trajectory_generators import generate_batch
import math

horizon = 100
episodes = 100
env = gym.make('Mass-v0')
policy = ShallowGaussianPolicy(2, 1, mu_init=[-0.20764519, -1.13575787], logstd_init=math.log(0.1))

generate_batch(env, policy, horizon, episodes, render=True)
env.close()

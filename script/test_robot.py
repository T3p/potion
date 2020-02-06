#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:13:33 2020

@author: matteo
"""

from potion.simulation.play import play
import gym
import safety_envs
from potion.actors.continuous_deterministic_policies import ShallowDeterministicPolicy
import torch

env =  gym.make('BasicReach-v0')
m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)

policy = ShallowDeterministicPolicy(m, d, 
                                    param_init=torch.tensor([0.10787985473871231, 0.02179303579032421, 4.300711154937744, 0.10839951038360596,
                  0.017089104279875755, 0.1119314506649971, 0.018646063283085823, -0.17877089977264404,
                  -0.03759196400642395, -0.004248579498380423, 0.48613205552101135, 0.10498402267694473,
                  -12.068914413452148, 1.0702580213546753, -0.04661020636558533, -0.22232159972190857,
                  0.0361342579126358, -0.39843615889549255]))

ret = play(env, policy, horizon=200, episodes=1000, render=False)
print(ret)
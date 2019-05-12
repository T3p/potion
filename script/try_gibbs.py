#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:35:08 2019

@author: matteo
"""
import potion.envs
from potion.actors.discrete_policies import ShallowGibbsPolicy as Gibbs, UniformPolicy as Uniform
import gym
import time

env = gym.make('GridWorld-v0')
pol = Gibbs(env.n_states, env.n_actions)

done = False
s = env.reset()
while not done:
    a = pol.act(s)
    print(a)
    time.sleep(1)
    s, r, done, _ = env.step(a)
    env.render()
    time.sleep(1)
    print('r =', r)
    time.sleep(1)
    print()
    time.sleep(1)


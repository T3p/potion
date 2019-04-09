#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:22:11 2019

@author: matteo
"""

from potion.estimation.eigenvalues import power, power_fo, power_op
import gym
import potion.envs
from potion.actors.continuous_policies import ShallowGaussianPolicy
import numpy as np
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import seed_all_agent
from potion.meta.smoothing_constants import gauss_lip_const
import matplotlib.pyplot as plt
from potion.estimation.gradients import gpomdp_estimator
from potion.common.misc_utils import performance

env = gym.make('lqr1d-v0')
std = 0.1
disc = 0.99
horizon = 20
batchsize = 100
points = 20
max_feat = env.max_pos
max_rew = env.Q * env.max_pos**2 + env.R * env.max_action**2
seed = 0
env.seed(seed)
seed_all_agent(seed)
pol = ShallowGaussianPolicy(1, 1, learn_std=False, logstd_init=np.log(std))

perf = []
perf_hat = []
params = np.linspace(-1., 0., points)
it = 0
for param in params:
    pol.set_from_flat([param])
    it+=1
    print(it)
    batch = generate_batch(env, pol, horizon, batchsize)
    perf.append(env._performance(param, std, disc, horizon=horizon))
    perf_hat.append(performance(batch, disc))
    
    
plt.plot(params, perf, label='True')
plt.plot(params, perf_hat, label='Estimated')
plt.legend()
plt.show()
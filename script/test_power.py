#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:22:11 2019

@author: matteo
"""

from potion.estimation.eigenvalues import power
import gym
import potion.envs
from potion.actors.continuous_policies import ShallowGaussianPolicy
import numpy as np
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import seed_all_agent
from potion.meta.smoothing_constants import gauss_lip_const
import matplotlib.pyplot as plt
from potion.estimation.gradients import gpomdp_estimator

env = gym.make('lqr1d-v0')
std = 0.1
disc = 0.9
horizon = 20
batchsize = 100
points = 50
max_feat = env.max_pos
max_rew = env.Q * env.max_pos**2 + env.R * env.max_action**2
seed = 0
env.seed(seed)
seed_all_agent(seed)
pol = ShallowGaussianPolicy(1, 1, learn_std=False, logstd_init=np.log(std))

estimated = []
bound = []
real = []
fo = []
op = []
params = np.linspace(-1., 0., points)
it = 0
for param in params:
    plt.close()
    it+=1
    print(it)
    pol.set_from_flat([param])
    batch = generate_batch(env, pol, horizon, batchsize)
    grad = gpomdp_estimator(batch, disc, pol, shallow=True)
    
    estimated.append(power(pol, batch, grad, disc,alpha=0.01, gamma=0.1, clip=0.2))
    real.append(abs(env._hess(param, std, disc, horizon=horizon)))
    #bound.append(gauss_lip_const(max_feat, max_rew, disc, std))
    

plt.plot(params, estimated, label='Estimated')
plt.plot(params, real, label='True')
#plt.plot(params, bound, label='Bound')
plt.legend()
plt.show()
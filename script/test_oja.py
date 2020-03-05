#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:22:11 2019

@author: matteo
"""

from potion.estimation.spectrum import ojapg
import gym
import potion.envs
from potion.actors.continuous_policies import ShallowGaussianPolicy
import numpy as np
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import seed_all_agent
from potion.meta.smoothing_constants import gauss_lip_const
import matplotlib.pyplot as plt
from potion.estimation.gradients import gpomdp_estimator
import argparse

# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--seed', help='Random seed', type=int, default=0)
parser.add_argument('--iterations', help='Oja iterations', type=int, default=1000)
parser.add_argument('--step', help='Oja step size', type=float, default=1e-2)
parser.add_argument('--pert', help='Finite difference perturbation', type=float, default=1e-2)
parser.add_argument('--points', help='Number of policies to evaluate', type=int, default=100)
parser.add_argument('--batchsize', help='Number of trajectories per policy', type=int, default=100)

args = parser.parse_args()

env = gym.make('lqr1d-v0')
std = 0.1
disc = 0.95
horizon = 20
batchsize = args.batchsize
points = args.points
max_feat = env.max_pos
max_rew = env.Q * env.max_pos**2 + env.R * env.max_action**2
seed = args.seed
env.seed(seed)
seed_all_agent(seed)
pol = ShallowGaussianPolicy(1, 1, learn_std=False, logstd_init=np.log(std))

oja_norm = np.zeros(points)
real_norm = []
params = np.linspace(-1., 0., points)
for (j, param) in enumerate(params):
    plt.close()
    print('%d/%d' % (j+1, points))
    pol.set_from_flat([param])
    batch = generate_batch(env, pol, horizon, batchsize)
    

    oja_norm[j] = ojapg(pol, batch, disc, args.iterations, args.step, args.pert)
    real_norm.append(abs(env._hess(param, std, disc, horizon=horizon)))

plt.plot(params, oja_norm, label='oja')
plt.plot(params, real_norm, label='True')
plt.legend()
plt.show()
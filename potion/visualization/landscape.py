#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:59:43 2019

@author: matteo
"""
import numpy as np
import matplotlib.pyplot as plt
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance

def interpolate_landscape(env, policy, param_1, param_2, horizon, disc = 0.99, episodes = 100, resolution = 0.01):
    perfs = []
    lams = np.arange(0., 1. + resolution, resolution)
    for lam in lams:
        param = (1 - lam) * param_1 + lam * param_2
        policy.set_from_flat(param)
        batch = generate_batch(env, policy, horizon, episodes)
        perfs.append(performance(batch, disc))
    
    plt.plot(lams, perfs)
    plt.show()
    
if __name__ == '__main__':
    from potion.actors.continuous_policies import ShallowGaussianPolicy
    import potion.envs
    import gym
    import torch
    
    env = gym.make('Hole-v0')
    #env = gym.make('LQGX-v0')
    pol = ShallowGaussianPolicy(1, 1, learn_std=False, logstd_init=np.log(0.1))
    p1 = torch.tensor([0.])
    p2 = torch.tensor([-1.])
    interpolate_landscape(env, pol, p1, p2, horizon=20, disc=0.9)
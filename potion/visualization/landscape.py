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

def interpolate_landscape(env, policy, param_1, param_2, horizon, disc = 0.99, episodes = 100, resolution = 0.01, fun=None):
    perfs = []
    lams = np.arange(0., 1. + resolution, resolution)
    for lam in lams:
        param = (1 - lam) * param_1 + lam * param_2
        policy.set_from_flat(param)
        if fun is None:
            batch = generate_batch(env, policy, horizon, episodes)
            perfs.append(performance(batch, disc))
        else:
            perfs.append(fun(param))
    
    plt.plot(lams, perfs)
    plt.show()
    
if __name__ == '__main__':
    from potion.actors.continuous_policies import ShallowGaussianPolicy
    import potion.envs
    import gym
    import torch
    
    env = gym.make('lqr1d-v0')
    std = 1.
    disc = 0.9
    #env = gym.make('LQGX-v0')
    pol = ShallowGaussianPolicy(1, 1, learn_std=False, logstd_init=np.log(1.))
    p1 = torch.tensor([-1.])
    p2 = torch.tensor([1.])
    fun = None#lambda x: env._grad(x, std, disc)
    interpolate_landscape(env, pol, p1, p2, horizon=20, disc=disc, resolution=0.1, fun=fun)
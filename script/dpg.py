#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
from potion.actors.continuous_deterministic_policies import ShallowDeterministicPolicy
from potion.actors.hyperpolicies import GaussianHyperpolicy
from potion.common.logger import Logger
from potion.algorithms.dpg import dpg
import argparse
import re
from potion.meta.steppers import ConstantStepper, RMSprop, Adam
from gym.spaces.discrete import Discrete
import safety_envs
import numpy as np

# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='DPG')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='ContCartPole-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=200)
parser.add_argument('--batchsize', help='Initial batch size', type=int, default=100)
parser.add_argument('--iterations', help='Iterations', type=int, default=500)
parser.add_argument('--disc', help='Discount factor', type=float, default=0.99)
parser.add_argument('--noise', help='Initial policy std', type=float, default=1.)
parser.add_argument('--actor_step', help='Step size for actor', type=float, default=1e-4)
parser.add_argument('--critic_step', help='Step size for critic', type=float, default=1e-5)
parser.add_argument('--test_batchsize', help='Test batchsize for deterministic policy', type=float, default=10)

parser.add_argument("--render", help="Render an episode",
                    action="store_true")
parser.add_argument("--no-render", help="Do not render any episode",
                    action="store_false")
parser.add_argument("--temp", help="Save logs in temp folder",
                    action="store_true")
parser.add_argument("--no-temp", help="Save logs in logs folder",
                    action="store_false")
parser.add_argument("--natural", help="Use natural gradient",
                    action="store_true")
parser.add_argument("--no-natural", help="Use vanilla gradient",
                    action="store_false")
parser.add_argument("--bias", help="Use bias parameter",
                    action="store_true")
parser.add_argument("--no-bias", help="Use bias parameter",
                    action="store_false")
parser.set_defaults(render=False, temp=False, learnstd=True, natural=False, bias=False) 

args = parser.parse_args()

# Prepare

env = gym.make(args.env)
env.seed(args.seed)

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)

feat = None
if 'Minigolf' in args.env:
    def feat(s):
        sigma = 4.
        centers = [4., 8., 12., 16.]
        res = [np.exp(-1 / (2 * sigma ** 2) * (s - c) ** 2) for c in centers]
        cat_dim = len(s.shape)
        res = torch.cat(res, cat_dim - 1)
        return res
    
mu_init = torch.zeros(m*d)
if 'Minigolf' in args.env:
    mu_init = torch.ones(4)
elif 'DoubleIntegrator' in args.env:
    mu_init = torch.ones(2) * -0.3
policy = ShallowDeterministicPolicy(m, d, feature_fun=feat, param_init=mu_init)


envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)

# Run
dpg(env, policy,
            horizon = args.horizon,
            actor_step = args.actor_step,
            critic_step = args.critic_step,
            v_step = args.critic_step,
            u_step = args.critic_step,
            batchsize = args.batchsize,
            iterations = args.iterations,
            disc = args.disc,
            natural = args.natural,
            seed = args.seed,
            logger = logger,
            render = args.render,
            test_batchsize = args.test_batchsize,
            log_params=True)
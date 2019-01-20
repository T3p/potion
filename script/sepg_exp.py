#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
from potion.actors.continuous_policies import SimpleGaussianPolicy as Gauss
from potion.common.logger import Logger
from potion.algorithms.safe import sepg
from potion.common.misc_utils import clip
import argparse
import re
from potion.common.rllab_utils import rllab_env_from_name, Rllab2GymWrapper

# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='sepgtest')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='LQG1D-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=20)
parser.add_argument('--batchsize', help='Batch size', type=int, default=100)
parser.add_argument('--iterations', help='Iterations', type=int, default=2000)
parser.add_argument('--gamma', help='Discount factor', type=float, default=0.9)
parser.add_argument('--saveon', help='How often to save parameters', type=int, default=100)
parser.add_argument('--sigmainit', help='Initial policy std', type=float, default=1.)
parser.add_argument('--njobs', help='Number of workers', type=int, default=4)
parser.add_argument('--rmax', help='Discount factor', type=float, default=28.8)
parser.add_argument('--phimax', help='Discount factor', type=float, default=4.)
parser.add_argument("--render", help="Render an episode",
                    action="store_true")
parser.add_argument("--no-render", help="Do not render any episode",
                    action="store_false")
parser.add_argument("--trial", help="Save logs in temp folder",
                    action="store_true")
parser.add_argument("--no-trial", help="Save logs in logs folder",
                    action="store_false")
parser.add_argument("--parallel", help="Use parallel simulation",
                    action="store_true")
parser.add_argument("--no-parallel", help="Do not use parallel simulation",
                    action="store_false")
parser.set_defaults(render=False, trial=False, parallel=False) 

args = parser.parse_args()

# Prepare
if args.env.startswith('rllab'):
    env_rllab_class = rllab_env_from_name(args.env)
    env_rllab = env_rllab_class()
    env = Rllab2GymWrapper(env_rllab)
    af = lambda a: clip(env)(a).item()
else:
    env = gym.make(args.env)
    af = clip(env)
env.seed(args.seed)

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
mu_init = torch.zeros(m)
logstd_init = torch.log(torch.zeros(1) + args.sigmainit)
policy = Gauss(m, d, mu_init=mu_init, logstd_init=logstd_init, learn_std=True)

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.trial:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)
    
# Run
sepg(env,
            policy,
            horizon = args.horizon,
            batchsize = args.batchsize,
            iterations = args.iterations,
            gamma = args.gamma,
            rmax = args.rmax,
            phimax = args.phimax,
            seed = args.seed,
            action_filter = af,
            logger = logger,
            save_params = args.saveon,
            render = args.render,
            parallel = args.parallel,
            n_jobs = args.njobs)
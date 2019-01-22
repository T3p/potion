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
from potion.algorithms.metaexplore import mepg
from potion.common.misc_utils import clip
import argparse
import re
from potion.common.rllab_utils import rllab_env_from_name, Rllab2GymWrapper
from dm_control import suite

# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='mepgtest')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='ContCartPole-v0')
parser.add_argument('--alpha', help='Step size', type=float, default=1e-1)
parser.add_argument('--eta', help='Meta step size', type=float, default=1e-3)
parser.add_argument('--horizon', help='Task horizon', type=int, default=1000)
parser.add_argument('--batchsize', help='Batch size', type=int, default=500)
parser.add_argument('--iterations', help='Iterations', type=int, default=200)
parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
parser.add_argument('--saveon', help='How often to save parameters', type=int, default=10)
parser.add_argument('--sigmainit', help='Initial policy std', type=float, default=1.)
parser.add_argument('--njobs', help='Number of workers', type=int, default=4)
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
elif args.env.startswith('dm'):
    domain_name, task_name = str.split(args.env[2:], '-')
    env = suite.load(domain_name=domain_name, task_name=task_name)
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
mepg(env,
            policy,
            horizon = args.horizon,
            batchsize = args.batchsize,
            iterations = args.iterations,
            gamma = args.gamma,
            alpha = args.alpha,
            eta = args.eta,
            seed = args.seed,
            action_filter = af,
            logger = logger,
            save_params = args.saveon,
            render = args.render,
            parallel = args.parallel,
            n_jobs = args.njobs)
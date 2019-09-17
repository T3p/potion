#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.common.logger import Logger
from potion.algorithms.safe import safepg_exact
import argparse
import re
from potion.common.rllab_utils import rllab_env_from_name, Rllab2GymWrapper


# Command line arguments
parser = argparse.ArgumentParser(formatter_class
                                 =argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='ESPG')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, 
                    default='lqr1d-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=20)
parser.add_argument('--max_samples', help='Maximum total samples', type=int, 
                    default=1e6)
parser.add_argument('--min_batchsize', help='(Minimum) batch size', type=int, 
                    default=100)
parser.add_argument('--batchsize', help='Maximum batch size', type=int, 
                    default=100)
parser.add_argument('--disc', help='Discount factor', type=float, default=0.9)
parser.add_argument('--std_init', help='Initial policy std', type=float, 
                    default=1.)
parser.add_argument("--render", help="Render an episode",
                    action="store_true")
parser.add_argument("--no-render", help="Do not render any episode",
                    action="store_false")
parser.add_argument("--temp", help="Save logs in temp folder",
                    action="store_true")
parser.add_argument("--no-temp", help="Save logs in logs folder",
                    action="store_false")

args = parser.parse_args()

# Prepare
if args.env.startswith('rllab'):
    env_rllab_class = rllab_env_from_name(args.env)
    env_rllab = env_rllab_class()
    env = Rllab2GymWrapper(env_rllab)
else:
    env = gym.make(args.env)
env.seed(args.seed)

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
mu_init = torch.zeros(m)
logstd_init = torch.log(torch.zeros(1) + args.std_init)
policy = ShallowGaussianPolicy(m, d, 
                               mu_init=mu_init, 
                               logstd_init=logstd_init, 
                               learn_std=False)

test_batchsize = args.batchsize

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)

# Run
safepg_exact(env, policy,
            horizon = args.horizon,
            max_samples = args.max_samples,
            disc = args.disc,
            seed = args.seed,
            logger = logger,
            render = args.render,
            shallow = True,
            test_batchsize=test_batchsize)
